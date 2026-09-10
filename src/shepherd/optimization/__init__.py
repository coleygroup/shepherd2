"""Abstract, pluggable GA optimization pipeline."""
from shepherd.optimization.individual import Individual
from shepherd.optimization.population import (
    PopulationInitializer,
    SeedMoleculeInitializer,
    LibrarySamplingInitializer,
    DockedPoseInitializer,
)
from shepherd.optimization.fitness import (
    FitnessOracle,
    DockingOracle,
    MultiObjectiveOracle,
    CallableOracle,
    InteractionFingerprintOracle,
    prepare_ref_ligand,
)
from shepherd.optimization.validity import (
    ValidityChecker,
    DrugLikenessChecker,
    SubstructureFilter,
    CompositeValidityChecker,
)
from shepherd.optimization.ga import InteractionSpaceGA, GAConfig

__all__ = [
    # Individual
    "Individual",
    # Population
    "PopulationInitializer",
    "SeedMoleculeInitializer",
    "LibrarySamplingInitializer",
    "DockedPoseInitializer",
    # Fitness
    "FitnessOracle",
    "DockingOracle",
    "MultiObjectiveOracle",
    "CallableOracle",
    "InteractionFingerprintOracle",
    "prepare_ref_ligand",
    # Validity
    "ValidityChecker",
    "DrugLikenessChecker",
    "SubstructureFilter",
    "CompositeValidityChecker",
    # GA
    "InteractionSpaceGA",
    "GAConfig",
]
