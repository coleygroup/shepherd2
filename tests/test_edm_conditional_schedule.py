from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shepherd.inference.edm_sampler import get_edm_sigma_schedule
from shepherd.inference.sampler import generate
from shepherd.interaction_profile import InteractionProfile


class _RecordingModel:
    def __init__(self):
        self.recorded_sigmas: list[float] = []

    def forward(self, input_dict: dict):
        sigma_val = float(input_dict['x1']['decoder']['sigma'][0].item())
        self.recorded_sigmas.append(sigma_val)

        x1_dec = input_dict['x1']['decoder']
        x2_dec = input_dict['x2']['decoder']
        x3_dec = input_dict['x3']['decoder']
        x4_dec = input_dict['x4']['decoder']

        output_dict = {
            'x1': {
                'decoder': {
                    'denoiser': {
                        'pos_out': torch.zeros_like(x1_dec['pos']),
                        'x_out': torch.zeros_like(x1_dec['x']),
                        'bond_edge_x_out': torch.zeros_like(x1_dec['bond_edge_x']),
                    }
                }
            },
            'x2': {
                'decoder': {
                    'denoiser': {
                        'pos_out': torch.zeros_like(x2_dec['pos']),
                    }
                }
            },
            'x3': {
                'decoder': {
                    'denoiser': {
                        'pos_out': torch.zeros_like(x3_dec['pos']),
                        'x_out': torch.zeros_like(x3_dec['x']),
                    }
                }
            },
            'x4': {
                'decoder': {
                    'denoiser': {
                        'pos_out': torch.zeros_like(x4_dec['pos']),
                        'direction_out': torch.zeros_like(x4_dec['direction']),
                        'x_out': torch.zeros_like(x4_dec['x']),
                    }
                }
            },
        }
        return None, output_dict


def _make_dummy_model_pl(edm_params: dict):
    params = {
        'edm': dict(edm_params),
        'scaffold_conditioning': True,
        'dataset': {
            'scale_point_cloud': 1.0,
            'x1': {
                'atom_types': [None, 'H', 'C', 'N', 'O'],
                'charge_types': [0, 1, -1],
                'bond_types': [None, 'SINGLE', 'DOUBLE'],
                'scale_atom_features': 0.25,
                'scale_bond_features': 1.0,
            },
            'x2': {'num_points': 4},
            'x3': {'num_points': 4, 'scale_node_features': 1.0},
            'x4': {
                'max_node_types': 5,
                'scale_node_features': 1.0,
                'scale_vector_features': 1.0,
            },
        },
    }
    return SimpleNamespace(
        params=params,
        device=torch.device('cpu'),
        model=_RecordingModel(),
    )


def _make_dummy_profile() -> InteractionProfile:
    return InteractionProfile(
        surface=np.zeros((4, 3), dtype=np.float32),
        electrostatics=np.zeros(4, dtype=np.float32),
        pharm_types=np.array([0, 1], dtype=int),
        pharm_positions=np.zeros((2, 3), dtype=np.float32),
        pharm_directions=np.zeros((2, 3), dtype=np.float32),
        n_atoms=3,
    )


def test_conditional_generate_uses_sigma_schedule_overrides():
    # Matches training/parameters/base.yaml where checkpoint stores sigma_max=2.0
    checkpoint_edm = {
        'sigma_data_pos': 3.0,
        'sigma_max': 2.0,
        'sigma_min': 1e-3,
        'rho': 7.0,
    }
    model_pl = _make_dummy_model_pl(checkpoint_edm)
    profile = _make_dummy_profile()

    generate(
        model_pl,
        batch_size=1,
        N_x1=3,
        N_x4=2,
        condition=profile,
        num_steps=5,
        early_stop_edm=-1,
        sigma_max=5.0,
        sigma_min=0.05,
        rho=3.0,
        verbose=False,
    )

    expected = get_edm_sigma_schedule(
        5, {'sigma_data_pos': 3.0, 'sigma_max': 5.0, 'sigma_min': 0.05, 'rho': 3.0}
    )[:-1]
    np.testing.assert_allclose(model_pl.model.recorded_sigmas, expected, rtol=1e-6)
    # Caller overrides must not mutate model_pl.params['edm']
    assert model_pl.params['edm'] == checkpoint_edm


def test_conditional_generate_default_sigma_max_overrides_checkpoint_param():
    # generate() defaults to sigma_max=3.0 while base.yaml sets sigma_max=2.0
    model_pl = _make_dummy_model_pl({'sigma_data_pos': 3.0, 'sigma_max': 2.0})
    profile = _make_dummy_profile()

    generate(
        model_pl,
        batch_size=1,
        N_x1=3,
        N_x4=2,
        condition=profile,
        num_steps=4,
        early_stop_edm=-1,
        verbose=False,
    )

    expected = get_edm_sigma_schedule(
        4, {'sigma_data_pos': 3.0, 'sigma_max': 3.0, 'sigma_min': 0.001, 'rho': 7.0}
    )[:-1]
    np.testing.assert_allclose(model_pl.model.recorded_sigmas, expected, rtol=1e-6)


def test_conditional_generate_none_overrides_fall_back_to_model_params():
    checkpoint_edm = {
        'sigma_data_pos': 3.0,
        'sigma_max': 2.0,
        'sigma_min': 0.02,
        'rho': 5.0,
    }
    model_pl = _make_dummy_model_pl(checkpoint_edm)
    profile = _make_dummy_profile()

    generate(
        model_pl,
        batch_size=1,
        N_x1=3,
        N_x4=2,
        condition=profile,
        num_steps=4,
        early_stop_edm=-1,
        sigma_max=None,
        sigma_min=None,
        rho=None,
        verbose=False,
    )

    expected = get_edm_sigma_schedule(4, checkpoint_edm)[:-1]
    np.testing.assert_allclose(model_pl.model.recorded_sigmas, expected, rtol=1e-6)
