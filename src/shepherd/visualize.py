import py3Dmol
from shepherd.interaction_profile import InteractionProfile
from shepherd_score.visualize import (
    draw,
    draw_mol,
    draw_molecule,
    draw_sample,
    draw_atom_sample,
    draw_pharm,
    draw_pharmacophores,
    draw_surface,
    draw_2d_highlight,
    draw_2d_valid,
    view_sample_trajectory,
)


def draw_profile(
    profile: InteractionProfile,
    probe_radius: float = 0.6,
    view: py3Dmol.view | None = None,
    removeHs: bool = False,
    color_scheme: str | None = None,
    custom_carbon_color: str | None = None,
    opacity: float = 1,
    opacity_features: float = 1,
    no_surface_points: bool = False,
    width: int = 800,
    height: int = 400
):
    num_surf_points = None
    if profile.surface is not None:
        num_surf_points = len(profile.surface)
    return draw_molecule(
        profile.to_molecule(
            num_surf_points=num_surf_points, probe_radius=probe_radius),
            view=view,
            removeHs=removeHs,
            color_scheme=color_scheme,
            custom_carbon_color=custom_carbon_color,
            opacity=opacity,
            opacity_features=opacity_features,
            no_surface_points=no_surface_points,
            width=width,
            height=height,
        )

__all__ = [
    'draw',
    'draw_profile',
    'draw_mol',
    'draw_molecule',
    'draw_sample',
    'draw_atom_sample',
    'draw_pharm',
    'draw_pharmacophores',
    'draw_surface',
    'draw_2d_highlight',
    'draw_2d_valid',
    'view_sample_trajectory',
]