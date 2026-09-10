"""
Shepherd Compositional Inference submodule.

Provides the main compositional sampling function.
"""
from shepherd.comp_inference.sampler import generate_composition
from shepherd.interaction_profile import (
    ConditionAtoms,
    InpaintAdvancedOptions,
    InteractionProfile,
    extract_interaction_profile,
)

__all__ = [
    'ConditionAtoms',
    'InpaintAdvancedOptions',
    'InteractionProfile',
    'extract_interaction_profile',
    'generate_composition',
]
