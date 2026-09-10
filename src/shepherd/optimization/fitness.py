"""Fitness oracle abstractions for the GA optimizer."""
from __future__ import annotations
import logging
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal

from rdkit import Chem
import numpy as np

if TYPE_CHECKING:
    from shepherd.optimization.individual import Individual

logger = logging.getLogger(__name__)


class FitnessOracle(ABC):
    """Abstract scalar oracle: maps a batch of Individuals to scalar scores."""

    def __init__(self) -> None:
        self.score_cache: Dict[str, float] = {}

    @abstractmethod
    def evaluate(self, individuals: List["Individual"]) -> List[float]:
        """Score *individuals* and return one float per entry."""

    def evaluate_with_cache(self, individuals: List["Individual"]) -> List[float]:
        """Like :meth:`evaluate` but caches results by SMILES.

        Individuals whose SMILES is already cached are returned immediately;
        only uncached individuals are forwarded to :meth:`evaluate`.
        Individuals with ``smiles=None`` receive ``float('inf')``
        immediately without going to the oracle.

        Returns
        -------
        list[float]
        """
        results: List[float | None] = [None] * len(individuals)
        to_eval_idx: List[int] = []
        to_eval_inds: List["Individual"] = []

        for i, ind in enumerate(individuals):
            smi = ind.smiles
            if smi is None:
                results[i] = float("inf")
            elif smi in self.score_cache:
                results[i] = self.score_cache[smi]
            else:
                to_eval_idx.append(i)
                to_eval_inds.append(ind)

        if to_eval_inds:
            new_scores = self.evaluate(to_eval_inds)
            for idx, ind, score in zip(to_eval_idx, to_eval_inds, new_scores):
                s = float(score) if np.isfinite(score) else float("inf")
                if ind.smiles is not None:
                    self.score_cache[ind.smiles] = s
                results[idx] = s

        return [r if r is not None else float("inf") for r in results]

    @property
    def n_evaluated(self) -> int:
        """Number of unique SMILES evaluated so far (via cache)."""
        return len(self.score_cache)


class DockingOracle(FitnessOracle):
    """Fitness oracle that scores molecules by Vina docking.

    Wraps either a ``DockingEvalPipeline`` object (which exposes an
    ``.evaluate()`` method) or any plain callable that maps
    ``List[str] -> List[float]``.

    Lower Vina scores (more negative kcal/mol) indicate better binding.

    Arguments
    ---------
    docking_pipeline
        A ``DockingEvalPipeline`` instance *or* a callable
        ``(smiles: list[str]) -> list[float]``.
    exhaustiveness : int
        Vina exhaustiveness parameter (higher = more thorough, slower).
    n_poses : int
        Number of docking poses requested per molecule.
    protonate : bool
        Whether to protonate molecules before docking.
    verbose : bool
        Pass verbosity flag to the underlying pipeline.
    """

    def __init__(
        self,
        docking_pipeline: Any,
        exhaustiveness: int = 32,
        n_poses: int = 1,
        protonate: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._pipeline = docking_pipeline
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.protonate = protonate
        self.verbose = verbose

    def evaluate(self, individuals: List["Individual"]) -> List[float]:
        if not individuals:
            return []

        smiles = [ind.smiles for ind in individuals]

        if hasattr(self._pipeline, "evaluate"):
            energies = self._pipeline.evaluate(
                smiles,
                exhaustiveness=self.exhaustiveness,
                n_poses=self.n_poses,
                protonate=self.protonate,
                verbose=self.verbose,
            )
        elif callable(self._pipeline):
            energies = self._pipeline(smiles)
        else:
            raise TypeError(
                "docking_pipeline must be a DockingEvalPipeline or a callable "
                "mapping List[str] -> List[float]."
            )

        return [float(e) if np.isfinite(e) else float("inf") for e in energies]

    @property
    def docked_pose_buffer(self) -> Dict | None:
        """Return the pose buffer if the underlying pipeline exposes one."""
        return getattr(self._pipeline, "buffer", None)


class MultiObjectiveOracle(FitnessOracle):
    """Bundle multiple FitnessOracle instances for multi-objective (Pareto-front) optimization.

    Each sub-oracle contributes one objective dimension.  The GA detects this
    class and activates NSGA-II Pareto selection automatically.

    Arguments
    ---------
    oracles : list[FitnessOracle]
        One oracle per objective.  All are evaluated independently.
    weights : list[float], optional
        Per-objective weights used **only** by :meth:`evaluate` (the scalar
        fallback).  Defaults to equal weights.  The Pareto selection path in
        the GA does *not* use weights — it uses the raw per-objective scores.

    Attributes
    ----------
    multi_score_cache : dict[str, list[float]]
        Cache keyed by SMILES mapping to the full list of per-oracle scores.
    """

    def __init__(
        self,
        oracles: List[FitnessOracle],
        weights: List[float] | None = None,
    ) -> None:
        super().__init__()
        if len(oracles) < 2:
            raise ValueError(
                "MultiObjectiveOracle requires at least 2 oracles; "
                "use a plain FitnessOracle for single-objective runs."
            )
        self.sub_oracles = oracles
        self.weights: List[float] = weights if weights is not None else [1.0] * len(oracles)
        if len(self.weights) != len(oracles):
            raise ValueError("len(weights) must equal len(oracles).")
        self.multi_score_cache: Dict[str, List[float]] = {}

    # scalar interface
    def evaluate(self, individuals: List["Individual"]) -> List[float]:
        """Return a weighted scalar score per individual.

        Uses :meth:`evaluate_multi` internally and applies ``self.weights``.
        This path is used by :meth:`evaluate_with_cache` in the base class
        for the scalar cache (``score_cache``).  The GA's Pareto path calls
        :meth:`evaluate_multi_with_cache` instead.
        """
        multi = self.evaluate_multi(individuals)
        return [
            sum(w * s for w, s in zip(self.weights, scores))
            for scores in multi
        ]

    # multi-objective interface
    def evaluate_multi(
        self, individuals: List["Individual"]
    ) -> List[List[float]]:
        """Return ``[[score_obj0, score_obj1, ...], ...]`` for each individual.

        Each sub-oracle is called independently; results are transposed from
        ``[oracle][individual]`` to ``[individual][oracle]`` layout.
        """
        if not individuals:
            return []
        per_oracle = [oracle.evaluate(individuals) for oracle in self.sub_oracles]
        n_ind = len(individuals)
        n_obj = len(self.sub_oracles)
        return [
            [float(per_oracle[o][i]) if np.isfinite(per_oracle[o][i]) else float("inf")
             for o in range(n_obj)]
            for i in range(n_ind)
        ]

    def evaluate_multi_with_cache(
        self, individuals: List["Individual"]
    ) -> List[List[float]]:
        """Cache-aware version of :meth:`evaluate_multi`.

        Populates both ``multi_score_cache`` (per-objective list) and the
        parent class's scalar ``score_cache`` (weighted sum).
        """
        n_obj = len(self.sub_oracles)
        inf_scores = [float("inf")] * n_obj
        results: List[List[float] | None] = [None] * len(individuals)
        to_eval_idx: List[int] = []
        to_eval_inds: List["Individual"] = []

        for i, ind in enumerate(individuals):
            smi = ind.smiles
            if smi is None:
                results[i] = list(inf_scores)
            elif smi in self.multi_score_cache:
                results[i] = self.multi_score_cache[smi]
            else:
                to_eval_idx.append(i)
                to_eval_inds.append(ind)

        if to_eval_inds:
            new_multi = self.evaluate_multi(to_eval_inds)
            for idx, ind, scores in zip(to_eval_idx, to_eval_inds, new_multi):
                if ind.smiles is not None:
                    self.multi_score_cache[ind.smiles] = scores
                    self.score_cache[ind.smiles] = sum(
                        w * s for w, s in zip(self.weights, scores)
                    )
                results[idx] = scores

        return [r if r is not None else list(inf_scores) for r in results]

    @property
    def n_evaluated(self) -> int:
        return len(self.multi_score_cache)


class CallableOracle(FitnessOracle):
    """Wrap any ``(List[Individual]) -> List[float]`` function as a FitnessOracle.

    Arguments
    ---------
    fn : callable
        A function (or lambda) mapping a list of Individuals to a list of floats.
    name : str, optional
        Human-readable label for logging.
    """

    def __init__(
        self, fn: Callable[["List[Individual]"], List[float]], name: str = "custom"
    ) -> None:
        super().__init__()
        self._fn = fn
        self.name = name

    def evaluate(self, individuals: List["Individual"]) -> List[float]:
        return list(self._fn(individuals))


class InteractionFingerprintOracle(FitnessOracle):
    """Score molecules by ProLIF interaction fingerprint (IFP) similarity to a reference ligand.

    Returns ``1 - similarity`` so that the GA minimises the score (lower =
    more similar to the reference = better).

    This oracle requires that each :class:`Individual` has a docked 3D mol
    stored in ``ind.mol`` with explicit hydrogens and 3D coordinates.  It is
    therefore most useful **after** a docking step and works naturally as the
    second objective in a :class:`MultiObjectiveOracle` alongside
    :class:`DockingOracle`.

    Arguments
    ---------
    protein_pdb_path : str
        Path to the protonated receptor PDB file.
    ref_ligand : Chem.Mol
        Reference ligand with explicit Hs and 3D coordinates (e.g. a crystal
        or docked pose).  Monomer info should have residue name ``"UNL"``
        (required by ProLIF); see :func:`prepare_ref_ligand` for a helper.
    residues : list[str], optional
        Restrict the IFP to specific residues, e.g. ``["THR163.B", "VAL143.B"]``.
        ``None`` (default) uses all binding-site residues.
    scoring : {"tanimoto", "recovery"}
        ``"tanimoto"`` – symmetric Tanimoto similarity.  Penalises both
        missing reference interactions *and* extra interactions not in the
        reference.

        ``"recovery"`` – asymmetric interaction recovery (fraction of the
        reference interactions that are recapitulated).  Extra interactions
        in the pose do *not* reduce the score.
    interaction_count : bool
        If ``True`` (default), count repeated interactions rather than using
        binary presence/absence.

    Notes
    -----
    The underlying :class:`Interactions` object is stateful: each call to
    :meth:`evaluate` overwrites the internal ProLIF fingerprint state.  The
    oracle is therefore **not** thread-safe for concurrent batch calls.
    """

    def __init__(
        self,
        protein_pdb_path: str,
        ref_ligand: Chem.Mol,
        residues: List[str] | None = None,
        scoring: Literal["tanimoto", "recovery"] = "tanimoto",
        interaction_count: bool = True,
    ) -> None:
        super().__init__()
        try:
            from shepherd_score.evaluations.docking.interactions import Interactions
        except ImportError as exc:
            raise ImportError(
                "shepherd_score is required for InteractionFingerprintOracle. "
                "Install it or check your PYTHONPATH."
            ) from exc

        if scoring not in ("tanimoto", "recovery"):
            raise ValueError(f"scoring must be 'tanimoto' or 'recovery', got {scoring!r}")

        self._interactions = Interactions(
            protein_pdb_path=protein_pdb_path,
            ref_ligand=ref_ligand,
            residues=residues,
            interaction_count=interaction_count,
        )
        self.scoring = scoring
        self.interaction_count = interaction_count


    def evaluate(self, individuals: List["Individual"]) -> List[float]:
        """Compute ``1 - IFP_similarity`` for each individual.

        Individuals whose ``mol`` is ``None`` receive ``float('inf')``.

        Arguments
        ---------
        individuals : list[Individual]
            Must have ``ind.mol`` populated with a 3D docked conformer.

        Returns
        -------
        list[float]
            One score per individual; range ``[0, 1]`` for valid molecules,
            ``float('inf')`` when no 3D mol is available.
        """
        results: List[float] = [float("inf")] * len(individuals)
        valid_idx: List[int] = []
        valid_mols: List[Chem.Mol] = []

        for i, ind in enumerate(individuals):
            mol = ind.mol
            if mol is None:
                continue
            # Ensure explicit Hs with coords are present (no-op if already there).
            mol_h = Chem.AddHs(mol, addCoords=True)
            # Translate mol_h so its center of mass matches the profile's stored value.
            if ind.profile.com_before_centering is not None:
                try:
                    conf = mol_h.GetConformer()
                    positions = conf.GetPositions()
                    new_positions = positions + ind.profile.com_before_centering
                    for j, pos in enumerate(new_positions):
                        conf.SetAtomPosition(j, pos.tolist())
                except Exception as exc:
                    logger.warning("Failed to apply center of mass to mol_h: %s", exc)
            valid_idx.append(i)
            valid_mols.append(mol_h)

        if not valid_mols:
            return results

        # Batch fingerprint computation overwrites internal ProLIF state.
        try:
            self._interactions.get_fingerprints(valid_mols, include_ref=False)
        except Exception as exc:
            logger.warning("IFP fingerprint computation failed: %s", exc)
            return results

        # Retrieve similarity scores (length == len(valid_mols)).
        try:
            if self.scoring == "tanimoto":
                sims = self._interactions.get_fingerprint_similarity(
                    interaction_count=self.interaction_count,
                    fixed_length=False,
                )
            else:  # recovery
                sims = self._interactions.get_interaction_recovery(
                    interaction_count=self.interaction_count,
                )
        except Exception as exc:
            logger.warning("IFP similarity computation failed: %s", exc)
            return results

        sims = np.asarray(sims, dtype=float)

        # defensive alignment: ProLIF occasionally returns an extra leading entry for the ref pose
        if len(sims) == len(valid_idx) + 1:
            warnings.warn(
                "IFP similarity array has one extra entry; dropping first entry "
                "to align with the evaluated molecules.",
                stacklevel=2,
            )
            sims = sims[1:]

        n_aligned = min(len(valid_idx), len(sims))
        if n_aligned < len(valid_idx):
            logger.warning(
                "IFP result length (%d) < valid molecules (%d); "
                "unaligned entries will score as inf.",
                len(sims), len(valid_idx),
            )

        for arr_i, pop_i in enumerate(valid_idx[:n_aligned]):
            sim = float(sims[arr_i]) if np.isfinite(sims[arr_i]) else 0.0
            # Clamp to [0, 1] in case of floating-point noise.
            sim = max(0.0, min(1.0, sim))
            results[pop_i] = 1.0 - sim  # lower = more similar = better

        return results


def prepare_ref_ligand(mol_path: str) -> Chem.Mol:
    """Prepare a reference ligand for use with :class:`InteractionFingerprintOracle`.

    ProLIF requires that all atoms in the ligand carry monomer info with a
    consistent residue name (``"UNL"``).  This function adds explicit
    hydrogens (with 3D coordinates) and patches the monomer info on every
    atom.

    Arguments
    ---------
    mol_path : str
        Path to the input molecule file (SDF, PDB, etc.).
        Hydrogens will be added if missing.

    Returns
    -------
    Chem.Mol
        Copy of *mol* with explicit Hs and ``"UNL"`` residue labels on all
        atoms, ready to pass as ``ref_ligand`` to
        :class:`InteractionFingerprintOracle`.
    """

    if mol_path.endswith('.sdf'):
        mol_supplier = Chem.SDMolSupplier(mol_path, removeHs=False)
        for mol in mol_supplier:
            ref_mol = Chem.AddHs(mol, addCoords=True)
            break
    elif mol_path.endswith('.mol2'):
        ref_mol = Chem.MolFromMol2File(mol_path, removeHs=False)
    elif mol_path.endswith('.pdb'):
        ref_mol = Chem.MolFromPDBFile(mol_path, removeHs=False)
    else:
        raise ValueError(f"Unsupported file format: {mol_path}")

    mol = Chem.AddHs(ref_mol, addCoords=True)

    if mol_path.endswith('.pdb'):
        for atom in mol.GetAtoms():
            info = atom.GetMonomerInfo()
            if info is None:
                info = Chem.AtomPDBResidueInfo()
                atom.SetMonomerInfo(info)
            info.SetResidueName("UNL")
            info.SetResidueNumber(1)
            info.SetChainId("")

    return mol
