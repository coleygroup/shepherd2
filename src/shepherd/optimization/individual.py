"""Individual dataclass for the abstract GA optimizer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from rdkit import Chem

from shepherd.interaction_profile import InteractionProfile


@dataclass
class Individual:
    """A single member of the GA population.

    Arguments
    ---------
    profile : InteractionProfile
        Interaction profile of the molecule.
    smiles : str, optional
        Canonical SMILES (``None`` if molecule reconstruction failed).
    mol : Chem.Mol, optional
        RDKit molecule with 3D conformer.
    fitness_score : float
        Scalar score from the :class:`~shepherd.optimization.fitness.FitnessOracle`
        (lower = better; ``float('inf')`` for unevaluated individuals).
        In multi-objective mode this is set to the Pareto rank to be compatible with
        scalar-based code paths.
    fitness_scores : list[float]
        Per-oracle raw scores in multi-objective mode (empty for
        single-objective runs).  Index *i* corresponds to oracle *i*.
    pareto_rank : int
        NSGA-II non-domination rank (0 = Pareto front, higher = dominated).
        Only meaningful in multi-objective mode.
    crowding_distance : float
        NSGA-II crowding distance within the individual's Pareto front.
        Higher values indicate a less crowded region of objective space.
    generation : int
        GA generation in which this individual was created.
    parent_indices : tuple[int, ...]
        Indices into the parent population (empty for seeds).
    parent_smiles : tuple[str, ...]
        SMILES of parent(s) — length 0 for seeds, 1 for mutation, 2 for crossover.
    origin : str
        One of ``'seed'``, ``'crossover'``, ``'mutate_conditional'``,
        ``'mutate_composed'``, ``'unknown'``.
    """

    profile: InteractionProfile
    smiles: str | None = None
    mol: Chem.Mol | None = None
    fitness_score: float = float("inf")
    fitness_scores: List[float] = field(default_factory=list)
    pareto_rank: int = 0
    crowding_distance: float = 0.0
    generation: int = 0
    parent_indices: Tuple[int, ...] = field(default_factory=tuple)
    parent_smiles: Tuple[str, ...] = field(default_factory=tuple)
    origin: str = "unknown"

    @property
    def is_valid(self) -> bool:
        """``True`` iff a SMILES string is present (i.e. molecule was reconstructed)."""
        return self.smiles is not None

    def to_dict(self) -> dict:
        """Serialize to a plain dict for checkpointing."""
        return {
            "profile": self.profile.to_dict(),
            "smiles": self.smiles,
            "fitness_score": self.fitness_score,
            "fitness_scores": list(self.fitness_scores),
            "pareto_rank": self.pareto_rank,
            "crowding_distance": self.crowding_distance,
            "generation": self.generation,
            "parent_indices": list(self.parent_indices),
            "parent_smiles": list(self.parent_smiles),
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Individual":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        profile = InteractionProfile.from_dict(d["profile"])
        score = d.get("fitness_score", float("inf"))
        return cls(
            profile=profile,
            smiles=d.get("smiles"),
            mol=profile.mol,
            fitness_score=score,
            fitness_scores=list(d.get("fitness_scores", [])),
            pareto_rank=d.get("pareto_rank", 0),
            crowding_distance=d.get("crowding_distance", 0.0),
            generation=d.get("generation", 0),
            parent_indices=tuple(d.get("parent_indices", ())),
            parent_smiles=tuple(d.get("parent_smiles", ())),
            origin=d.get("origin", "unknown"),
        )
