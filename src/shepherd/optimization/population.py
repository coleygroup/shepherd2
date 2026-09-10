"""Population initialization abstractions for the GA optimizer."""
from __future__ import annotations

import os
import re
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from shepherd.optimization.individual import Individual


class PopulationInitializer(ABC):
    """Abstract base class for population initialization strategies.

    Subclasses must implement :meth:`initialize`, which returns a list of
    :class:`Individual` objects that seed the first generation of the GA.
    """

    @abstractmethod
    def initialize(self, **kwargs) -> List["Individual"]:
        """Build and return the initial population.

        Returns
        -------
        list[Individual]
            Fully constructed individuals (profiles extracted, SMILES set,
            ``generation=0``, ``origin='seed'``).
        """


class SeedMoleculeInitializer(PopulationInitializer):
    """Initialize the population from a list of RDKit molecules with 3D conformers.

    This mirrors the original ``InteractionSpaceGA.initialise_population``
    logic: each molecule is centered, an :class:`InteractionProfile` is
    extracted via the ShEPhERD pipeline, and the result is wrapped in an
    :class:`Individual`.

    Arguments
    ---------
    params : dict
        ShEPhERD model parameter dict (``model_pl.params``).
    """

    def __init__(self, params: dict) -> None:
        self.params = params

    def initialize(
        self,
        mols: List[Chem.Mol],
        xtb_optimize: bool = False,  # default False: seed mols are typically crystal/docked poses whose geometry should be kept as-is
        partial_charges: List[np.ndarray] | None = None,
    ) -> List["Individual"]:
        """Extract interaction profiles from mols and return seed individuals.

        Arguments
        ---------
        mols : list[Chem.Mol]
            Starting molecules; each must have a 3D conformer.
        xtb_optimize: bool (default=False)
            Whether to optimize the molecules with xtb. Defaults to False since seed
            molecules are typically extracted from a crystal structure or docked pose,
            whose geometry should not be perturbed by relaxation.
        partial_charges : Optional[list[np.ndarray]] (default=None)
            Pre-computed partial charges (one array per molecule). Will compute with xTB if None.

        Returns
        -------
        list[Individual]
        """
        from shepherd.optimization.individual import Individual
        from shepherd.interaction_profile import extract_interaction_profile

        population: List[Individual] = []
        for i, mol in enumerate(mols):
            charges = partial_charges[i] if partial_charges is not None else None
            profile = extract_interaction_profile(mol, partial_charges=charges, xtb_optimize=xtb_optimize)
            if profile is None:
                continue
            try:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                smi = None
            population.append(Individual(
                profile=profile,
                smiles=smi,
                mol=mol,
                generation=0,
                origin="seed",
            ))
        return population


class LibrarySamplingInitializer(PopulationInitializer):
    """Initialize the population by sampling molecules from a SMILES library.

    Molecules are sampled from *library* (a list of SMILES strings or a path
    to a newline-delimited ``.smi`` / ``.txt`` file), embedded into 3D using
    RDKit's ETKDG conformer generator, and then forwarded to
    :class:`SeedMoleculeInitializer` for profile extraction.

    Arguments
    ---------
    params : dict
        ShEPhERD model parameter dict (``model_pl.params``).
    library : list[str] or str
        Either a list of SMILES strings **or** a path to a plain-text file
        with one SMILES per line (leading/trailing whitespace stripped;
        blank lines and lines starting with ``#`` are ignored).
    n_samples : int, optional
        Number of molecules to draw from the library.  When ``None`` all
        library entries are used.
    seed : int, optional
        Random seed for reproducible sampling (default ``42``).
    max_embed_attempts : int, optional
        Number of ETKDG random seeds tried per molecule before giving up
        (default ``5``).
    """

    def __init__(
        self,
        params: dict,
        library,
        n_samples: int | None = None,
        seed: int = 42,
        max_embed_attempts: int = 5,
    ) -> None:
        self.params = params
        self.n_samples = n_samples
        self.seed = seed
        self.max_embed_attempts = max_embed_attempts
        self._seed_init = SeedMoleculeInitializer(params)

        # Resolve library to a list of SMILES strings.
        if isinstance(library, str):
            self._smiles_library = self._load_smiles_file(library)
        else:
            self._smiles_library = list(library)

    def initialize(self, **kwargs) -> List["Individual"]:
        """Sample from the library, embed molecules, and return seed individuals.

        Any additional keyword arguments are ignored (kept to keep API consistent).

        Returns
        -------
        list[Individual]
        """
        rng = np.random.default_rng(self.seed)

        pool = self._smiles_library
        if self.n_samples is not None and self.n_samples < len(pool):
            indices = rng.choice(len(pool), size=self.n_samples, replace=False)
            pool = [pool[i] for i in indices]

        mols = []
        for smi in pool:
            mol = self._embed_smiles(smi)
            if mol is not None:
                mols.append(mol)

        if not mols:
            warnings.warn(
                "LibrarySamplingInitializer: no molecules could be embedded; "
                "returning empty population."
            )
            return []

        return self._seed_init.initialize(mols=mols)


    @staticmethod
    def _load_smiles_file(path: str) -> List[str]:
        """Read SMILES from a plain-text file (one per line)."""
        smiles = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Accept "SMILES name" format — take only the first token.
                smiles.append(line.split()[0])
        return smiles

    def _embed_smiles(self, smiles: str) -> Chem.Mol | None:
        """Convert a SMILES string to an RDKit Mol with a 3D conformer."""
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            warnings.warn(f"Could not parse SMILES: {smiles!r}")
            return None

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        for attempt in range(self.max_embed_attempts):
            params.randomSeed = self.seed + attempt
            result = AllChem.EmbedMolecule(mol, params)
            if result == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass
                return mol

        warnings.warn(
            f"ETKDG embedding failed for SMILES after "
            f"{self.max_embed_attempts} attempts: {smiles!r}"
        )
        return None


class DockedPoseInitializer(PopulationInitializer):
    """Initialize the population from docked PDBQT files in a directory.

    Molecules can be selected by:

    - **indices**: zero-based integer positions in the sorted file list.
    - **ids**: integer IDs embedded in the filename (e.g. ``mol_42.pdbqt``
      matches id ``42``).  The first integer found via regex is used.
    - **poses**: pre-loaded :class:`meeko.PDBQTMolecule` objects passed
      directly (``pdbqt_dir`` is not required in this case).
    - **None**: all ``.pdbqt`` files in *pdbqt_dir* are used.

    After selection the PDBQT entries are converted to RDKit molecules via
    meeko and then forwarded to :class:`SeedMoleculeInitializer`.

    Arguments
    ---------
    params : dict
        ShEPhERD model parameter dict (``model_pl.params``).
    pdbqt_dir : str, optional
        Directory containing ``.pdbqt`` files.  Required unless *poses* are
        provided directly to :meth:`initialize`.
    """

    def __init__(self, params: dict, pdbqt_dir: str | None = None) -> None:
        self.params = params
        self.pdbqt_dir = pdbqt_dir
        self._seed_init = SeedMoleculeInitializer(params)

    def initialize(
        self,
        indices: List[int] | None = None,
        ids: List[int] | None = None,
        poses=None,
        partial_charges: List[np.ndarray] | None = None,
    ) -> List["Individual"]:
        """Build initial population from docked poses.

        Exactly one of *indices*, *ids*, *poses*, or ``None`` (use all files)
        should be supplied.

        Arguments
        ---------
        indices : list[int], optional
            Zero-based positions into the sorted list of ``.pdbqt`` files.
        ids : list[int], optional
            Integer IDs to match against filenames (first integer in filename).
        poses : list[meeko.PDBQTMolecule], optional
            Pre-loaded PDBQT molecule objects.
        partial_charges : list[np.ndarray], optional
            Pre-computed partial charges forwarded to
            :class:`SeedMoleculeInitializer`.

        Returns
        -------
        list[Individual]
        """
        if poses is not None:
            selected_poses = list(poses)
        else:
            if self.pdbqt_dir is None:
                raise ValueError("pdbqt_dir must be set when poses are not provided directly.")
            selected_poses = self._load_poses(indices=indices, ids=ids)

        mols = self._poses_to_rdkit(selected_poses)
        return self._seed_init.initialize(mols=mols, partial_charges=partial_charges)

    def _load_poses(
        self,
        indices: List[int] | None = None,
        ids: List[int] | None = None,
    ):
        """Load and filter PDBQTMolecule objects from *self.pdbqt_dir*."""
        from meeko import PDBQTMolecule

        pdbqt_files = sorted(
            f for f in os.listdir(self.pdbqt_dir) if f.endswith(".pdbqt")
        )

        if ids is not None:
            id_set = set(ids)
            selected_files = [
                f for f in pdbqt_files
                if (m := re.search(r"(\d+)\.pdbqt$", f)) and int(m.group(1)) in id_set
            ]
        elif indices is not None:
            selected_files = [pdbqt_files[i] for i in indices if i < len(pdbqt_files)]
        else:
            selected_files = pdbqt_files

        poses = []
        for filename in selected_files:
            path = os.path.join(self.pdbqt_dir, filename)
            try:
                poses.append(PDBQTMolecule.from_file(path))
            except Exception as exc:
                warnings.warn(f"Skipping {filename}: failed to parse PDBQT ({exc})")
        return poses

    @staticmethod
    def _poses_to_rdkit(poses) -> List[Chem.Mol]:
        """Convert a list of PDBQTMolecule objects to RDKit Mol objects."""
        from meeko import RDKitMolCreate

        mols: List[Chem.Mol] = []
        for pose in poses:
            try:
                candidates = RDKitMolCreate.from_pdbqt_mol(pose)
            except RuntimeError as exc:
                warnings.warn(f"Skipping pose: meeko conversion error ({exc})")
                continue
            except Exception as exc:
                warnings.warn(f"Skipping pose: unexpected conversion error ({exc})")
                continue

            rdkit_mol = next((c for c in candidates if c is not None), None)
            if rdkit_mol is None:
                warnings.warn("Skipping pose: no valid RDKit molecule returned by meeko")
                continue

            mols.append(Chem.AddHs(rdkit_mol, addCoords=True))
        return mols
