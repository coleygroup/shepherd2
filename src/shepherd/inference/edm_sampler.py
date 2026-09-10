"""ShEPhERD-2 EDM-based sampler."""

from __future__ import annotations

from tqdm import tqdm
import numpy as np
import torch

import torch_scatter

from shepherd.lightning_module import LightningModule
from shepherd.diffusion.edm import EDMNoiseSchedule
from shepherd.inference.initialization import (
    _initialize_x1_state,
    _initialize_x2_state,
    _initialize_x3_state,
    _initialize_x4_state,
)
from shepherd.inference.steps import _extract_generated_samples
from shepherd.inference.utils import (
    _add_trajectories_to_generated_structures,
    resolve_early_stop_steps,
)
from shepherd.generated_sample import GeneratedSample

def get_edm_sigma_schedule(num_steps: int, edm_params: dict) -> np.ndarray:
    """
    Build EDM noise schedule.

    Arguments
    ---------
    num_steps : int
        Number of denoising steps.
    edm_params : dict
        EDM config from params['edm']. Uses sigma_data (or sigma_data_pos as
        fallback), sigma_min (default 1e-3), sigma_max (default 3.0), rho (default 7).

    Returns
    -------
    np.ndarray
        Shape (num_steps + 1,). Sigma values from high to low.
    """
    sigma_data = edm_params.get('sigma_data', edm_params.get('sigma_data_pos', 3.0))
    sigma_min = edm_params.get('sigma_min', 1e-3)
    sigma_max = edm_params.get('sigma_max', 3.0)
    rho = edm_params.get('rho', 7)

    schedule = EDMNoiseSchedule(
        sigma_data=sigma_data,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
    )
    t_steps = np.linspace(0.0, 1.0, num_steps + 1)
    sigma_steps = schedule.get_sigma(t_steps)
    return sigma_steps


def _prepare_edm_model_input(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    sigma: float | torch.Tensor,
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_batch: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    x1_bond_edge_index: torch.Tensor,
    x1_virtual_node_mask: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_x_t: torch.Tensor,
    x2_batch: torch.Tensor,
    x2_virtual_node_mask: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    x3_batch: torch.Tensor,
    x3_virtual_node_mask: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    x4_batch: torch.Tensor,
    x4_virtual_node_mask: torch.Tensor,
    x1_is_diffused_atom: torch.Tensor | None = None,
    x4_is_diffused_pharm: torch.Tensor | None = None,
    scaffold_task: torch.Tensor | None = None,
) -> dict:
    """Build the model input_dict for ShEPhERD--2."""
    if isinstance(sigma, (int, float)):
        sigma_batch = torch.full((batch_size,), float(sigma), dtype=dtype, device=device)
    else:
        sigma_batch = sigma.to(device=device, dtype=dtype)
        if sigma_batch.dim() == 0:
            sigma_batch = sigma_batch.expand(batch_size)

    if x1_is_diffused_atom is None:
        x1_is_diffused_atom = ~x1_virtual_node_mask
    if x4_is_diffused_pharm is None:
        x4_is_diffused_pharm = ~x4_virtual_node_mask
    if scaffold_task is None:
        scaffold_task = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    input_dict = {
        'device': device,
        'dtype': dtype,
        'x1': {
            'decoder': {
                'pos': x1_pos_t.to(device),
                'x': x1_x_t.to(device),
                'batch': x1_batch.to(device),
                'bond_edge_x': x1_bond_edge_x_t.to(device),
                'bond_edge_index': x1_bond_edge_index.to(device),
                'virtual_node_mask': x1_virtual_node_mask.to(device),
                'is_diffused_atom': x1_is_diffused_atom.to(device),
                'scaffold_task': scaffold_task.to(device),
                'sigma': sigma_batch,
            },
        },
        'x2': {
            'decoder': {
                'pos': x2_pos_t.to(device),
                'x': x2_x_t.to(device),
                'batch': x2_batch.to(device),
                'virtual_node_mask': x2_virtual_node_mask.to(device),
                'sigma': sigma_batch,
            },
        },
        'x3': {
            'decoder': {
                'pos': x3_pos_t.to(device),
                'x': x3_x_t.to(device),
                'batch': x3_batch.to(device),
                'virtual_node_mask': x3_virtual_node_mask.to(device),
                'sigma': sigma_batch,
            },
        },
        'x4': {
            'decoder': {
                'x': x4_x_t.to(device),
                'pos': x4_pos_t.to(device),
                'direction': x4_direction_t.to(device),
                'batch': x4_batch.to(device),
                'virtual_node_mask': x4_virtual_node_mask.to(device),
                'is_diffused_pharm': x4_is_diffused_pharm.to(device),
                'sigma': sigma_batch,
            },
        },
    }
    return input_dict


_SIGMA_EPS = 1e-9  # guard against division by zero when sigma is very small


def _edm_ode_step(
    model_pl: LightningModule,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    sigma_cur: float | torch.Tensor,
    sigma_next: float | torch.Tensor,
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_batch: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    x1_bond_edge_index: torch.Tensor,
    x1_virtual_node_mask: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_x_t: torch.Tensor,
    x2_batch: torch.Tensor,
    x2_virtual_node_mask: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    x3_batch: torch.Tensor,
    x3_virtual_node_mask: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    x4_batch: torch.Tensor,
    x4_virtual_node_mask: torch.Tensor,
    shepherd_pred: bool = True,
    x1_is_diffused_atom: torch.Tensor | None = None,
    x4_is_diffused_pharm: torch.Tensor | None = None,
    scaffold_task: torch.Tensor | None = None,
    anchor_frame_active: bool = False,
    include_x0_pred: bool = False,
) -> tuple[dict, dict]:
    """Single EDM step.

    Predict-renoise step
    -------------------
    Default is to use predict-renoise (shepherd_pred=True) mode.
    x_next = D_theta(x_curr) + noise*sigma_next

    When ``anchor_frame_active`` is True (scaffold fixing and/or fixed pharmacophores),
    COM removal on x1 position updates is skipped to preserve the anchored coordinate frame.

    Traditional EDM ODE step
    ------------------------
    One EDM ODE step: d = (x - D_theta)/sigma_cur, x_next = x + (sigma_next - sigma_cur) * d.
    Returns (next_state_dict, d_dict) for use by 2nd-order correction. Virtual nodes zeroed on x_next.
    """
    sigma_cur_f = float(sigma_cur) if isinstance(sigma_cur, (int, float)) else float(sigma_cur.item())
    sigma_next_f = float(sigma_next) if isinstance(sigma_next, (int, float)) else float(sigma_next.item())
    sigma_safe = max(sigma_cur_f, _SIGMA_EPS)
    d_sigma = (sigma_next_f - sigma_cur_f)

    input_dict = _prepare_edm_model_input(
        device, dtype, batch_size, sigma_cur,
        x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, x1_bond_edge_index, x1_virtual_node_mask,
        x2_pos_t, x2_x_t, x2_batch, x2_virtual_node_mask,
        x3_pos_t, x3_x_t, x3_batch, x3_virtual_node_mask,
        x4_pos_t, x4_direction_t, x4_x_t, x4_batch, x4_virtual_node_mask,
        x1_is_diffused_atom, x4_is_diffused_pharm, scaffold_task,
    )
    with torch.no_grad():
        _, output_dict = model_pl.model.forward(input_dict)

    # D_theta (predicted clean); move to CPU and zero virtual nodes for masking consistency
    x1_pos_D = output_dict['x1']['decoder']['denoiser']['pos_out'].detach().cpu()
    x1_x_D = output_dict['x1']['decoder']['denoiser']['x_out'].detach().cpu()
    x1_bond_edge_x_D = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'].detach().cpu()
    # Need to check if this is necessary, qualitatively results look a little worse?
    # x1_pos_D = x1_pos_D - torch_scatter.scatter_mean(x1_pos_D[~x1_virtual_node_mask], x1_batch[~x1_virtual_node_mask], dim=0)[x1_batch]
    x1_pos_D[x1_virtual_node_mask, :] = 0.0
    x1_x_D[x1_virtual_node_mask, :] = 0.0

    # Scaffold/pharmacophore conditioning: clamp D_theta for fixed atoms to their current
    # (clean) values. For the ODE branch this makes d=0 so x_next=x_t. For shepherd_pred
    # this sets the mean of the distribution to the target; noise is zeroed below.
    skip_x1_com_removal = anchor_frame_active or (
        scaffold_task is not None and scaffold_task.any()
    )
    if x1_is_diffused_atom is not None:
        x1_pos_D[~x1_is_diffused_atom] = x1_pos_t[~x1_is_diffused_atom]
        x1_x_D[~x1_is_diffused_atom] = x1_x_t[~x1_is_diffused_atom]

    if shepherd_pred:
        x1_pos_epsilon = torch.randn_like(x1_pos_D)
        # Skip COM subtraction on epsilon when the anchored frame must be preserved.
        if not skip_x1_com_removal:
            x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(x1_pos_epsilon[~x1_virtual_node_mask], x1_batch[~x1_virtual_node_mask], dim=0)[x1_batch]
        x1_x_epsilon = torch.randn_like(x1_x_D)
        x1_bond_edge_x_epsilon = torch.randn_like(x1_bond_edge_x_D)
        # Zero noise for scaffold atoms so they receive no stochastic displacement.
        if x1_is_diffused_atom is not None:
            x1_pos_epsilon[~x1_is_diffused_atom] = 0.0
            x1_x_epsilon[~x1_is_diffused_atom] = 0.0
        x1_pos_next = x1_pos_D + sigma_next * x1_pos_epsilon
        x1_x_next = x1_x_D + sigma_next * x1_x_epsilon
        x1_bond_edge_x_next = x1_bond_edge_x_D + sigma_next * x1_bond_edge_x_epsilon
        if not skip_x1_com_removal:
            x1_pos_next = x1_pos_next - torch_scatter.scatter_mean(x1_pos_next[~x1_virtual_node_mask], x1_batch[~x1_virtual_node_mask], dim=0)[x1_batch]
        x1_pos_next[x1_virtual_node_mask, :] = 0.0
        x1_x_next[x1_virtual_node_mask, :] = 0.0
    else:
        # EDM ODE step
        # d = (x - D)/sigma_safe; x_next = x + d_sigma * d
        x1_pos_d = (x1_pos_t - x1_pos_D) / sigma_safe
        x1_x_d = (x1_x_t - x1_x_D) / sigma_safe
        x1_bond_edge_x_d = (x1_bond_edge_x_t - x1_bond_edge_x_D) / sigma_safe
        x1_pos_next = x1_pos_t + d_sigma * x1_pos_d
        if not skip_x1_com_removal:
            x1_pos_next = x1_pos_next - torch_scatter.scatter_mean(x1_pos_next[~x1_virtual_node_mask], x1_batch[~x1_virtual_node_mask], dim=0)[x1_batch]
        x1_x_next = x1_x_t + d_sigma * x1_x_d
        x1_bond_edge_x_next = x1_bond_edge_x_t + d_sigma * x1_bond_edge_x_d
        x1_pos_next[x1_virtual_node_mask, :] = 0.0
        x1_x_next[x1_virtual_node_mask, :] = 0.0

    # x2: pos diffused; x2_x not diffused, keep unchanged
    x2_pos_out = output_dict['x2']['decoder']['denoiser']['pos_out']
    if x2_pos_out is not None:
        x2_pos_D = x2_pos_out.detach().cpu()
        x2_pos_D[x2_virtual_node_mask, :] = 0.0
        if shepherd_pred:
            x2_pos_next = x2_pos_D + sigma_next * torch.randn_like(x2_pos_D)
        else:
            x2_pos_d = (x2_pos_t - x2_pos_D) / sigma_safe
            x2_pos_next = x2_pos_t + d_sigma * x2_pos_d
        x2_pos_next[x2_virtual_node_mask, :] = 0.0
    else:
        x2_pos_d = torch.zeros_like(x2_pos_t)
        x2_pos_next = x2_pos_t

    # x3
    x3_pos_out = output_dict['x3']['decoder']['denoiser']['pos_out']
    x3_x_out = output_dict['x3']['decoder']['denoiser']['x_out']
    if x3_pos_out is not None:
        x3_pos_D = x3_pos_out.detach().cpu()
        x3_pos_D[x3_virtual_node_mask, :] = 0.0
        x3_x_D = x3_x_out.detach().cpu().squeeze()
        x3_x_D[x3_virtual_node_mask] = 0.0
        if shepherd_pred:
            x3_pos_next = x3_pos_D + sigma_next * torch.randn_like(x3_pos_D)
            x3_x_next = x3_x_D + sigma_next * torch.randn_like(x3_x_D)
        else:
            x3_pos_d = (x3_pos_t - x3_pos_D) / sigma_safe
            x3_x_d = (x3_x_t - x3_x_D) / sigma_safe
            x3_pos_next = x3_pos_t + d_sigma * x3_pos_d
            x3_x_next = x3_x_t + d_sigma * x3_x_d
        x3_pos_next[x3_virtual_node_mask, :] = 0.0
        x3_x_next[x3_virtual_node_mask] = 0.0
    else:
        x3_pos_d = torch.zeros_like(x3_pos_t)
        x3_x_d = torch.zeros_like(x3_x_t)
        x3_pos_next = x3_pos_t
        x3_x_next = x3_x_t

    # x4
    x4_pos_out = output_dict['x4']['decoder']['denoiser']['pos_out']
    x4_direction_out = output_dict['x4']['decoder']['denoiser']['direction_out']
    x4_x_out = output_dict['x4']['decoder']['denoiser']['x_out']
    if x4_x_out is not None:
        x4_pos_D = x4_pos_out.detach().cpu()
        x4_pos_D[x4_virtual_node_mask, :] = 0.0
        x4_direction_D = x4_direction_out.detach().cpu()
        x4_direction_D[x4_virtual_node_mask, :] = 0.0
        x4_x_D = x4_x_out.detach().cpu().squeeze()
        x4_x_D[x4_virtual_node_mask] = 0.0
        # Clamp D_theta for conditioned pharmacophores to their current clean values.
        if x4_is_diffused_pharm is not None:
            x4_pos_D[~x4_is_diffused_pharm] = x4_pos_t[~x4_is_diffused_pharm]
            x4_direction_D[~x4_is_diffused_pharm] = x4_direction_t[~x4_is_diffused_pharm]
            x4_x_D[~x4_is_diffused_pharm] = x4_x_t[~x4_is_diffused_pharm]
        if shepherd_pred:
            x4_pos_epsilon = torch.randn_like(x4_pos_D)
            x4_direction_epsilon = torch.randn_like(x4_direction_D)
            x4_x_epsilon = torch.randn_like(x4_x_D)
            # Zero noise for conditioned pharmacophores.
            if x4_is_diffused_pharm is not None:
                x4_pos_epsilon[~x4_is_diffused_pharm] = 0.0
                x4_direction_epsilon[~x4_is_diffused_pharm] = 0.0
                x4_x_epsilon[~x4_is_diffused_pharm] = 0.0
            x4_pos_next = x4_pos_D + sigma_next * x4_pos_epsilon
            x4_direction_next = x4_direction_D + sigma_next * x4_direction_epsilon
            x4_x_next = x4_x_D + sigma_next * x4_x_epsilon
        else:
            x4_pos_d = (x4_pos_t - x4_pos_D) / sigma_safe
            x4_direction_d = (x4_direction_t - x4_direction_D) / sigma_safe
            x4_x_d = (x4_x_t - x4_x_D) / sigma_safe
            x4_pos_next = x4_pos_t + d_sigma * x4_pos_d
            x4_direction_next = x4_direction_t + d_sigma * x4_direction_d
            x4_x_next = x4_x_t + d_sigma * x4_x_d
        x4_pos_next[x4_virtual_node_mask, :] = 0.0
        x4_direction_next[x4_virtual_node_mask, :] = 0.0
        x4_x_next[x4_virtual_node_mask] = 0.0
    else:
        x4_pos_d = torch.zeros_like(x4_pos_t)
        x4_direction_d = torch.zeros_like(x4_direction_t)
        x4_x_d = torch.zeros_like(x4_x_t)
        x4_pos_next = x4_pos_t
        x4_direction_next = x4_direction_t
        x4_x_next = x4_x_t

    next_state = {
        'x1_pos_t_1': x1_pos_next,
        'x1_x_t_1': x1_x_next,
        'x1_bond_edge_x_t_1': x1_bond_edge_x_next,
        'x2_pos_t_1': x2_pos_next,
        'x2_x_t_1': x2_x_t,
        'x3_pos_t_1': x3_pos_next,
        'x3_x_t_1': x3_x_next,
        'x4_pos_t_1': x4_pos_next,
        'x4_direction_t_1': x4_direction_next,
        'x4_x_t_1': x4_x_next,
    }
    if include_x0_pred:
        # for logging x0 predictions
        x2_pos_0 = x2_pos_D if x2_pos_out is not None else x2_pos_t
        if x3_pos_out is not None:
            x3_pos_0 = x3_pos_D
            x3_x_0 = x3_x_D
        else:
            x3_pos_0 = x3_pos_t
            x3_x_0 = x3_x_t
        if x4_x_out is not None:
            x4_pos_0 = x4_pos_D
            x4_direction_0 = x4_direction_D
            x4_x_0 = x4_x_D
        else:
            x4_pos_0 = x4_pos_t
            x4_direction_0 = x4_direction_t
            x4_x_0 = x4_x_t
        next_state['x0_pred'] = {
            'x1_pos_0': x1_pos_D, 'x1_x_0': x1_x_D, 'x1_bond_edge_x_0': x1_bond_edge_x_D,
            'x2_pos_0': x2_pos_0, 'x2_x_0': x2_x_t,
            'x3_pos_0': x3_pos_0, 'x3_x_0': x3_x_0,
            'x4_pos_0': x4_pos_0, 'x4_direction_0': x4_direction_0, 'x4_x_0': x4_x_0,
        }
    if not shepherd_pred:
        d_dict = {
            'x1_pos_d': x1_pos_d,
            'x1_x_d': x1_x_d,
            'x1_bond_edge_x_d': x1_bond_edge_x_d,
            'x2_pos_d': x2_pos_d,
            'x3_pos_d': x3_pos_d,
            'x3_x_d': x3_x_d,
            'x4_pos_d': x4_pos_d,
            'x4_direction_d': x4_direction_d,
            'x4_x_d': x4_x_d,
        }
    else:
        d_dict = None
    return next_state, d_dict


def _edm_denoising_step(
    model_pl: LightningModule,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    sigma_cur: float | torch.Tensor,
    sigma_next: float | torch.Tensor,
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_batch: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    x1_bond_edge_index: torch.Tensor,
    x1_virtual_node_mask: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_x_t: torch.Tensor,
    x2_batch: torch.Tensor,
    x2_virtual_node_mask: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    x3_batch: torch.Tensor,
    x3_virtual_node_mask: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    x4_batch: torch.Tensor,
    x4_virtual_node_mask: torch.Tensor,
    shepherd_pred: bool = True,
    x1_is_diffused_atom: torch.Tensor | None = None,
    x4_is_diffused_pharm: torch.Tensor | None = None,
    scaffold_task: torch.Tensor | None = None,
    anchor_frame_active: bool = False,
    include_x0_pred: bool = False,
) -> dict:
    """
    One EDM denoising step (Euler ODE): thin wrapper around _edm_ode_step; returns only next_state.
    """
    next_state, _ = _edm_ode_step(
        model_pl, device, dtype, batch_size, sigma_cur, sigma_next,
        x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, x1_bond_edge_index, x1_virtual_node_mask,
        x2_pos_t, x2_x_t, x2_batch, x2_virtual_node_mask,
        x3_pos_t, x3_x_t, x3_batch, x3_virtual_node_mask,
        x4_pos_t, x4_direction_t, x4_x_t, x4_batch, x4_virtual_node_mask,
        shepherd_pred,
        x1_is_diffused_atom,
        x4_is_diffused_pharm,
        scaffold_task,
        anchor_frame_active=anchor_frame_active,
        include_x0_pred=include_x0_pred,
    )
    return next_state


def _edm_add_churn_noise(
    sigma_cur: float,
    gamma: float,
    S_noise: float,
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    x1_virtual_node_mask: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_virtual_node_mask: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    x3_virtual_node_mask: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    x4_virtual_node_mask: torch.Tensor,
) -> tuple[dict, float]:
    """
    Add churn noise per EDM Algorithm 2: hat_sigma = sigma_cur + gamma*sigma_cur,
    scale = sqrt(hat_sigma^2 - sigma_cur^2), add scale * S_noise * epsilon to each diffused
    tensor, then zero virtual nodes. Only call when gamma > 0.
    Returns (noised_state_dict, hat_sigma).
    """
    hat_sigma = sigma_cur + gamma * sigma_cur
    var_diff = max(0.0, hat_sigma ** 2 - sigma_cur ** 2)
    scale = (var_diff ** 0.5) * S_noise
    if scale <= 0:
        # No noise to add; return state unchanged and hat_sigma
        state_dict = {
            'x1_pos_t_1': x1_pos_t.clone(),
            'x1_x_t_1': x1_x_t.clone(),
            'x1_bond_edge_x_t_1': x1_bond_edge_x_t.clone(),
            'x2_pos_t_1': x2_pos_t.clone(),
            'x3_pos_t_1': x3_pos_t.clone(),
            'x3_x_t_1': x3_x_t.clone(),
            'x4_pos_t_1': x4_pos_t.clone(),
            'x4_direction_t_1': x4_direction_t.clone(),
            'x4_x_t_1': x4_x_t.clone(),
        }
        return state_dict, hat_sigma

    def _add_noise(t: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = t + scale * torch.randn_like(t)
        if mask is not None:
            out = out.clone()
            if out.dim() == 2:
                out[mask, :] = 0.0
            else:
                out[mask] = 0.0
        return out

    x1_pos_n = _add_noise(x1_pos_t, x1_virtual_node_mask)
    x1_x_n = _add_noise(x1_x_t, x1_virtual_node_mask)
    x1_bond_edge_x_n = _add_noise(x1_bond_edge_x_t, None)  # bond edges: no virtual mask per node; keep as-is or zero by index if needed
    x2_pos_n = _add_noise(x2_pos_t, x2_virtual_node_mask)
    x3_pos_n = _add_noise(x3_pos_t, x3_virtual_node_mask)
    x3_x_n = _add_noise(x3_x_t, x3_virtual_node_mask)
    x4_pos_n = _add_noise(x4_pos_t, x4_virtual_node_mask)
    x4_direction_n = _add_noise(x4_direction_t, x4_virtual_node_mask)
    x4_x_n = _add_noise(x4_x_t, x4_virtual_node_mask)

    state_dict = {
        'x1_pos_t_1': x1_pos_n,
        'x1_x_t_1': x1_x_n,
        'x1_bond_edge_x_t_1': x1_bond_edge_x_n,
        'x2_pos_t_1': x2_pos_n,
        'x3_pos_t_1': x3_pos_n,
        'x3_x_t_1': x3_x_n,
        'x4_pos_t_1': x4_pos_n,
        'x4_direction_t_1': x4_direction_n,
        'x4_x_t_1': x4_x_n,
    }
    return state_dict, hat_sigma


def generate_edm_unconditional(
    model_pl: LightningModule,
    batch_size: int,
    N_x1: int,
    N_x4: int,
    num_steps: int,
    early_stop_edm: int | float = -1,
    sigma_max: float | None = 3.0,
    sigma_min: float | None = 1e-3,
    rho: float | None = 7.0,
    shepherd_pred: bool = True,
    verbose: bool = True,
    store_trajectories: bool = False,
    store_trajectories_x0: bool = False,
    use_stochastic: bool = False,
    S_churn: float = 40.0,
    S_noise: float = 1.0,
    use_2nd_order_correction: bool = False,
) -> list[GeneratedSample]:
    """
    EDM unconditional generation. Default is to use predict-renoise (shepherd_pred=True) mode.
    This leads to better results than the traditional EDM Markov sampler.
    use_stochastic should be False when using shepherd_pred=True.

    When use_stochastic is True (default), uses EDM Algorithm 2: churn noise in
    [sigma_min, sigma_max], then ODE step, then optional 2nd-order (Heun) correction.
    """
    params = model_pl.params
    N_x2 = params['dataset']['x2']['num_points']
    N_x3 = params['dataset']['x3']['num_points']
    edm_params = dict(params.get('edm', {}))
    if sigma_max is not None:
        edm_params['sigma_max'] = sigma_max
    if sigma_min is not None:
        edm_params['sigma_min'] = sigma_min
    if rho is not None:
        edm_params['rho'] = rho
    sigma_steps = get_edm_sigma_schedule(num_steps, edm_params)
    prior_noise_scale_edm = float(sigma_steps[0])

    include_virtual_node = True
    (x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, virtual_node_mask_x1, bond_edge_index_x1) = _initialize_x1_state(
        batch_size, N_x1, params, prior_noise_scale_edm, include_virtual_node, scaffold_conditioning=False
    )
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2 = _initialize_x2_state(
        batch_size, N_x2, params, prior_noise_scale_edm, include_virtual_node
    )
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3 = _initialize_x3_state(
        batch_size, N_x3, params, prior_noise_scale_edm, include_virtual_node
    )
    (x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4) = _initialize_x4_state(
        batch_size, N_x4, params, prior_noise_scale_edm, include_virtual_node
    )
    x1_is_diffused_atom = ~virtual_node_mask_x1
    x4_is_diffused_pharm = ~virtual_node_mask_x4
    scaffold_task = torch.zeros((batch_size,), dtype=torch.bool)

    device = model_pl.device
    dtype = torch.float32
    trajectories: list[list[dict]] = [] if store_trajectories else []
    trajectories_x0: list[list[dict]] = [] if store_trajectories_x0 else []

    _gamma_max = (2.0 ** 0.5) - 1.0  # sqrt(2) - 1

    _num_steps = resolve_early_stop_steps(early_stop_edm, num_steps)
    for step_idx in tqdm(range(_num_steps), desc="EDM Steps", disable=not verbose):
        sigma_cur = float(sigma_steps[step_idx])
        sigma_next = float(sigma_steps[step_idx + 1])

        if not use_stochastic:
            next_state = _edm_denoising_step(
                model_pl, device, dtype, batch_size, sigma_cur, sigma_next,
                x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, bond_edge_index_x1, virtual_node_mask_x1,
                x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2,
                x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3,
                x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4,
                shepherd_pred,
                x1_is_diffused_atom,
                x4_is_diffused_pharm,
                scaffold_task,
                include_x0_pred=store_trajectories_x0,
            )
        else:
            # Stochastic path: churn (optional) + ODE step + optional 2nd-order correction
            gamma_i = 0.0
            if sigma_min <= sigma_cur and sigma_cur <= sigma_max:
                gamma_i = min(S_churn / num_steps, _gamma_max)
            if gamma_i > 0:
                noised_state, hat_sigma = _edm_add_churn_noise(
                    sigma_cur, gamma_i, S_noise,
                    x1_pos_t, x1_x_t, x1_bond_edge_x_t, virtual_node_mask_x1,
                    x2_pos_t, virtual_node_mask_x2,
                    x3_pos_t, x3_x_t, virtual_node_mask_x3,
                    x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
                )
                x1_pos_t = noised_state['x1_pos_t_1']
                x1_x_t = noised_state['x1_x_t_1']
                x1_bond_edge_x_t = noised_state['x1_bond_edge_x_t_1']
                x2_pos_t = noised_state['x2_pos_t_1']
                x3_pos_t = noised_state['x3_pos_t_1']
                x3_x_t = noised_state['x3_x_t_1']
                x4_pos_t = noised_state['x4_pos_t_1']
                x4_direction_t = noised_state['x4_direction_t_1']
                x4_x_t = noised_state['x4_x_t_1']
                sigma_cur = hat_sigma
            # x_hat for 2nd-order: state at sigma_cur (after churn if any)
            x_hat_pos = x1_pos_t
            x_hat_x = x1_x_t
            x_hat_bond = x1_bond_edge_x_t
            x_hat_x2_pos = x2_pos_t
            x_hat_x3_pos = x3_pos_t
            x_hat_x3_x = x3_x_t
            x_hat_x4_pos = x4_pos_t
            x_hat_x4_dir = x4_direction_t
            x_hat_x4_x = x4_x_t
            sigma_cur_step = sigma_cur

            next_state, d_dict = _edm_ode_step(
                model_pl, device, dtype, batch_size, sigma_cur, sigma_next,
                x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, bond_edge_index_x1, virtual_node_mask_x1,
                x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2,
                x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3,
                x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4,
                shepherd_pred,
                x1_is_diffused_atom,
                x4_is_diffused_pharm,
                scaffold_task,
                include_x0_pred=store_trajectories_x0,
            )

            if use_2nd_order_correction and sigma_next > 0:
                # Second forward at (next_state, sigma_next) to get d'; then Heun: x = x_hat + (sigma_next - sigma_cur) * (0.5*d + 0.5*d')
                _, d_prime_dict = _edm_ode_step(
                    model_pl, device, dtype, batch_size, sigma_next, sigma_next,
                    next_state['x1_pos_t_1'], next_state['x1_x_t_1'], x1_batch, next_state['x1_bond_edge_x_t_1'],
                    bond_edge_index_x1, virtual_node_mask_x1,
                    next_state['x2_pos_t_1'], next_state['x2_x_t_1'], x2_batch, virtual_node_mask_x2,
                    next_state['x3_pos_t_1'], next_state['x3_x_t_1'], x3_batch, virtual_node_mask_x3,
                    next_state['x4_pos_t_1'], next_state['x4_direction_t_1'], next_state['x4_x_t_1'], x4_batch, virtual_node_mask_x4,
                    shepherd_pred,
                    x1_is_diffused_atom,
                    x4_is_diffused_pharm,
                    scaffold_task,
                    include_x0_pred=False,
                )
                d_sigma_2 = sigma_next - sigma_cur_step
                half = 0.5 * d_sigma_2
                next_state['x1_pos_t_1'] = x_hat_pos + half * (d_dict['x1_pos_d'] + d_prime_dict['x1_pos_d'])
                next_state['x1_x_t_1'] = x_hat_x + half * (d_dict['x1_x_d'] + d_prime_dict['x1_x_d'])
                next_state['x1_bond_edge_x_t_1'] = x_hat_bond + half * (d_dict['x1_bond_edge_x_d'] + d_prime_dict['x1_bond_edge_x_d'])
                next_state['x2_pos_t_1'] = x_hat_x2_pos + half * (d_dict['x2_pos_d'] + d_prime_dict['x2_pos_d'])
                next_state['x3_pos_t_1'] = x_hat_x3_pos + half * (d_dict['x3_pos_d'] + d_prime_dict['x3_pos_d'])
                next_state['x3_x_t_1'] = x_hat_x3_x + half * (d_dict['x3_x_d'] + d_prime_dict['x3_x_d'])
                next_state['x4_pos_t_1'] = x_hat_x4_pos + half * (d_dict['x4_pos_d'] + d_prime_dict['x4_pos_d'])
                next_state['x4_direction_t_1'] = x_hat_x4_dir + half * (d_dict['x4_direction_d'] + d_prime_dict['x4_direction_d'])
                next_state['x4_x_t_1'] = x_hat_x4_x + half * (d_dict['x4_x_d'] + d_prime_dict['x4_x_d'])
                next_state['x1_pos_t_1'][virtual_node_mask_x1, :] = 0.0
                next_state['x1_x_t_1'][virtual_node_mask_x1, :] = 0.0
                next_state['x2_pos_t_1'][virtual_node_mask_x2, :] = 0.0
                next_state['x3_pos_t_1'][virtual_node_mask_x3, :] = 0.0
                next_state['x3_x_t_1'][virtual_node_mask_x3] = 0.0
                next_state['x4_pos_t_1'][virtual_node_mask_x4, :] = 0.0
                next_state['x4_direction_t_1'][virtual_node_mask_x4, :] = 0.0
                next_state['x4_x_t_1'][virtual_node_mask_x4] = 0.0

        if store_trajectories:
            trajectories.append(_extract_generated_samples(
                x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
                x2_pos_t, virtual_node_mask_x2,
                x3_pos_t, x3_x_t, virtual_node_mask_x3,
                x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
                params, batch_size,
                x4_batch=x4_batch,
            ))
        if store_trajectories_x0 and 'x0_pred' in next_state:
            trajectories_x0.append(_extract_generated_samples(
                next_state['x0_pred']['x1_x_0'], next_state['x0_pred']['x1_pos_0'],
                next_state['x0_pred']['x1_bond_edge_x_0'], virtual_node_mask_x1,
                next_state['x0_pred']['x2_pos_0'], virtual_node_mask_x2,
                next_state['x0_pred']['x3_pos_0'], next_state['x0_pred']['x3_x_0'], virtual_node_mask_x3,
                next_state['x0_pred']['x4_pos_0'], next_state['x0_pred']['x4_direction_0'],
                next_state['x0_pred']['x4_x_0'], virtual_node_mask_x4,
                params, batch_size,
                x4_batch=x4_batch,
            ))
        x1_pos_t = next_state['x1_pos_t_1']
        x1_x_t = next_state['x1_x_t_1']
        x1_bond_edge_x_t = next_state['x1_bond_edge_x_t_1']
        x2_pos_t = next_state['x2_pos_t_1']
        x2_x_t = next_state['x2_x_t_1']
        x3_pos_t = next_state['x3_pos_t_1']
        x3_x_t = next_state['x3_x_t_1']
        x4_pos_t = next_state['x4_pos_t_1']
        x4_direction_t = next_state['x4_direction_t_1']
        x4_x_t = next_state['x4_x_t_1']

    generated_structures = _extract_generated_samples(
        x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
        x2_pos_t, virtual_node_mask_x2,
        x3_pos_t, x3_x_t, virtual_node_mask_x3,
        x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
        params, batch_size,
        x4_batch=x4_batch,
    )
    if store_trajectories or store_trajectories_x0:
        if store_trajectories and trajectories:
            _add_trajectories_to_generated_structures(generated_structures, batch_size, trajectories, is_x0=False)
        if store_trajectories_x0 and trajectories_x0:
            _add_trajectories_to_generated_structures(generated_structures, batch_size, trajectories_x0, is_x0=True)
    return generated_structures


def _edm_shift_system_to_reference_com(
    x1_pos_t: torch.Tensor,
    x1_batch: torch.Tensor,
    virtual_node_mask_x1: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_batch: torch.Tensor,
    virtual_node_mask_x2: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_batch: torch.Tensor,
    virtual_node_mask_x3: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_batch: torch.Tensor,
    virtual_node_mask_x4: torch.Tensor,
    reference_com: torch.Tensor,
    batch_size: int,
    target_inpaint_x1_pos: torch.Tensor | None = None,
    target_inpaint_x4_pos: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor | None, torch.Tensor | None,
]:
    """Rigidly shift all position modalities so x1 real-atom COM equals reference_com."""
    real = ~virtual_node_mask_x1
    current_com = torch_scatter.scatter_mean(x1_pos_t[real], x1_batch[real], dim=0)
    ref = reference_com.to(device=current_com.device, dtype=current_com.dtype)
    if ref.dim() == 1:
        ref = ref.view(1, 3).expand(batch_size, -1)
    shift = current_com - ref

    x1_pos_t = x1_pos_t - shift[x1_batch]
    x1_pos_t[virtual_node_mask_x1, :] = 0.0
    x2_pos_t = x2_pos_t - shift[x2_batch]
    x2_pos_t[virtual_node_mask_x2, :] = 0.0
    x3_pos_t = x3_pos_t - shift[x3_batch]
    x3_pos_t[virtual_node_mask_x3, :] = 0.0
    x4_pos_t = x4_pos_t - shift[x4_batch]
    x4_pos_t[virtual_node_mask_x4, :] = 0.0

    shift0 = shift[0].detach().cpu()
    if target_inpaint_x1_pos is not None:
        target_inpaint_x1_pos = target_inpaint_x1_pos.clone()
        target_inpaint_x1_pos.sub_(shift0)
        target_inpaint_x1_pos[0, :] = 0.0
    if target_inpaint_x4_pos is not None:
        target_inpaint_x4_pos = target_inpaint_x4_pos.clone()
        target_inpaint_x4_pos.sub_(shift0)
        target_inpaint_x4_pos[0, :] = 0.0

    return x1_pos_t, x2_pos_t, x3_pos_t, x4_pos_t, target_inpaint_x1_pos, target_inpaint_x4_pos


def _edm_maybe_release_scaffold(
    step_idx: int,
    edm_inp: dict,
    scaffold_pos_active: bool,
    scaffold_com_shifted: bool,
    x1_is_diffused_atom: torch.Tensor,
    virtual_node_mask_x1: torch.Tensor,
    scaffold_task: torch.Tensor,
    batch_size: int,
    x1_pos_t: torch.Tensor,
    x1_batch: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_batch: torch.Tensor,
    virtual_node_mask_x2: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_batch: torch.Tensor,
    virtual_node_mask_x3: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_batch: torch.Tensor,
    virtual_node_mask_x4: torch.Tensor,
    target_inpaint_x1_pos: torch.Tensor | None,
    target_inpaint_x4_pos: torch.Tensor | None,
) -> tuple[
    bool, bool, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor | None, torch.Tensor | None,
]:
    """Release scaffold fixing at stop_scaffold_at_step; optional COM shift, then scaffold_task off."""
    if not scaffold_pos_active or step_idx < edm_inp.get('stop_scaffold_at_step', 0):
        return (
            scaffold_pos_active, scaffold_com_shifted, x1_is_diffused_atom, scaffold_task,
            x1_pos_t, x2_pos_t, x3_pos_t, x4_pos_t, target_inpaint_x1_pos, target_inpaint_x4_pos,
        )

    reference_com_np = edm_inp.get('reference_com_internal')
    if reference_com_np is not None and not scaffold_com_shifted:
        reference_com = torch.as_tensor(reference_com_np, dtype=x1_pos_t.dtype, device=x1_pos_t.device)
        (
            x1_pos_t, x2_pos_t, x3_pos_t, x4_pos_t,
            target_inpaint_x1_pos, target_inpaint_x4_pos,
        ) = _edm_shift_system_to_reference_com(
            x1_pos_t, x1_batch, virtual_node_mask_x1,
            x2_pos_t, x2_batch, virtual_node_mask_x2,
            x3_pos_t, x3_batch, virtual_node_mask_x3,
            x4_pos_t, x4_batch, virtual_node_mask_x4,
            reference_com, batch_size,
            target_inpaint_x1_pos, target_inpaint_x4_pos,
        )
        if target_inpaint_x1_pos is not None:
            target_inpaint_x1_pos = target_inpaint_x1_pos.clone()
        if target_inpaint_x4_pos is not None:
            target_inpaint_x4_pos = target_inpaint_x4_pos.clone()
        scaffold_com_shifted = True

    x1_is_diffused_atom = x1_is_diffused_atom.clone()
    x1_is_diffused_atom[~virtual_node_mask_x1] = True
    scaffold_pos_active = False
    scaffold_task = torch.zeros((batch_size,), dtype=torch.bool, device=scaffold_task.device)

    return (
        scaffold_pos_active, scaffold_com_shifted, x1_is_diffused_atom, scaffold_task,
        x1_pos_t, x2_pos_t, x3_pos_t, x4_pos_t, target_inpaint_x1_pos, target_inpaint_x4_pos,
    )


def _molecule_block_indices(
    batch: torch.Tensor,
    batch_size: int,
    local_offset: int,
    block_width: int,
    *,
    block_name: str = "node",
) -> torch.Tensor:
    """Flat indices for the same molecule-local block in a ragged batch."""
    counts = torch.bincount(batch, minlength=batch_size)
    if bool((counts < local_offset + block_width).any()):
        raise ValueError(
            f"{block_name} block does not fit every sample: "
            f"offset={local_offset}, width={block_width}, counts={counts.tolist()}"
        )
    starts = torch.cumsum(
        torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0
    )
    offsets = torch.arange(block_width, device=starts.device)
    return (starts[:, None] + local_offset + offsets[None, :]).reshape(-1)


def _replace_molecule_block(
    state: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
    block_width: int,
    *,
    local_offset: int = 0,
    batch: torch.Tensor | None = None,
    values_are_batched: bool = False,
    block_name: str = "node",
) -> torch.Tensor:
    """Assign one shared or per-sample block without reshaping a ragged state."""
    state = state.clone()

    if values_are_batched:
        expected = batch_size * block_width
        if values.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} batched {block_name} values, "
                f"got {values.shape[0]}."
            )
        batch_values = values.reshape(batch_size, block_width, *values.shape[1:])
    else:
        if values.shape[0] != block_width:
            raise ValueError(
                f"Expected {block_width} shared {block_name} values, got {values.shape[0]}."
            )
        batch_values = values.unsqueeze(0).expand(
            batch_size, block_width, *values.shape[1:]
        )

    if batch is None:
        # Uniform fallback retained for compositional inference
        rectangular = state.reshape(batch_size, -1, *state.shape[1:])
        rectangular[:, local_offset:local_offset + block_width] = batch_values.to(
            state.device
        )
        return rectangular.reshape_as(state)

    indices = _molecule_block_indices(
        batch,
        batch_size,
        local_offset,
        block_width,
        block_name=block_name,
    ).to(state.device)
    state[indices] = batch_values.reshape(-1, *values.shape[1:]).to(state.device)
    return state


def _replace_scaffold_block(
    x1_pos: torch.Tensor,
    x1_x: torch.Tensor,
    target_pos: torch.Tensor,
    target_x: torch.Tensor,
    batch_size: int,
    x1_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pin one shared scaffold prefix in every molecule of an x1 batch"""
    block_width = target_pos.shape[0]
    if target_x.shape[0] != block_width:
        raise ValueError(
            "Scaffold position and feature targets must have the same number of nodes, "
            f"got {block_width} and {target_x.shape[0]}."
        )
    x1_pos = _replace_molecule_block(
        x1_pos, target_pos, batch_size,
        block_width, batch=x1_batch, block_name="x1 scaffold",
    )
    x1_x = _replace_molecule_block(
        x1_x, target_x, batch_size,
        block_width, batch=x1_batch, block_name="x1 scaffold",
    )
    return x1_pos, x1_x


def _x4_trajectory_block_layout(
    edm_inp: dict,
    trajectory: torch.Tensor,
    batch_size: int,
) -> tuple[int, int, bool]:
    """Return local offset, width, and batching mode for an x4 trajectory."""
    values_are_batched = edm_inp.get(
        'batch_x3_x4_inpainting_trajectories', False
    )
    divisor = batch_size if values_are_batched else 1
    if trajectory.shape[0] % divisor != 0:
        raise ValueError(
            f"x4 trajectory length {trajectory.shape[0]} is not divisible by "
            f"batch size {divisor}."
        )
    block_width = trajectory.shape[0] // divisor
    local_offset = (
        edm_inp.get('num_pharm_cond', 0) + 1
        if edm_inp.get('is_mixed_pharm_mode', False)
        else 0
    )
    return local_offset, block_width, values_are_batched


def _apply_edm_inpainting_replacement(
    step_idx: int,
    edm_inp: dict,
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    batch_size: int,
    virtual_node_mask_x2: torch.Tensor,
    virtual_node_mask_x3: torch.Tensor,
    virtual_node_mask_x4: torch.Tensor,
    num_atom_types: int,
    num_x_types: int,
    num_pharm_types: int,
    MAX_BOND_TYPES: int,
    x4_batch: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Inpainting: replace predicted structures with simulated structure."""
    if edm_inp.get('inpaint_x1_pos') and step_idx < edm_inp['stop_inpainting_at_step_x1_pos']:
        traj = edm_inp['x1_pos_inpainting_trajectory_edm'][step_idx]
        if edm_inp.get('do_partial_atom_inpainting'):
            x1_pos_t_inpaint = traj.reshape(batch_size, -1, 3)
            x1_pos_t = x1_pos_t.reshape(batch_size, -1, 3)
            x1_pos_t[:, :x1_pos_t_inpaint.shape[1]] = x1_pos_t_inpaint
            x1_pos_t = x1_pos_t.reshape(-1, 3)
        else:
            x1_pos_t = traj
    else:
        x1_pos_t = x1_pos_t

    if (edm_inp.get('inpaint_x1_x') and step_idx < edm_inp['stop_inpainting_at_step_x1_x']):
        traj = edm_inp['x1_x_inpainting_trajectory_edm'][step_idx]
        if edm_inp.get('do_partial_atom_inpainting'):
            num_inpainted_atoms = edm_inp['num_inpainted_atoms']
            x1_x_t_inpaint = traj.reshape(batch_size, -1, num_x_types)
            x1_x_t = x1_x_t.reshape(batch_size, -1, num_x_types)
            atom_slice = slice(1, num_inpainted_atoms + 1)  # skip virtual node
            x1_x_t[:, atom_slice, :num_atom_types] = x1_x_t_inpaint[:, atom_slice, :num_atom_types]
            x1_x_t = x1_x_t.reshape(-1, num_x_types)
        else:
            x1_x_t[:, :num_atom_types] = traj[:, :num_atom_types]

    if edm_inp.get('inpaint_x1_formal_charge') and step_idx < edm_inp['stop_inpainting_at_step_x1_x']:
        traj = edm_inp['x1_x_inpainting_trajectory_edm'][step_idx]
        num_inpainted_formal_charges = edm_inp['num_inpainted_formal_charges']
        x1_x_t_inpaint = traj.reshape(batch_size, -1, num_x_types)
        x1_x_t = x1_x_t.reshape(batch_size, -1, num_x_types)
        x1_x_t[:, 1:num_inpainted_formal_charges + 1, num_atom_types:] = x1_x_t_inpaint[:, 1:num_inpainted_formal_charges + 1, num_atom_types:]
        x1_x_t = x1_x_t.reshape(-1, num_x_types)

    if edm_inp.get('inpaint_x1_bonds') and step_idx < edm_inp['stop_inpainting_at_step_x1_bonds']:
        bond_inpaint_mask = edm_inp['bond_inpaint_mask']
        traj = edm_inp['x1_bond_edge_x_inpainting_trajectory_edm'][step_idx]
        x1_bond_edge_x_t = x1_bond_edge_x_t.reshape(batch_size, -1, MAX_BOND_TYPES)
        x1_bond_edge_x_t[:, bond_inpaint_mask] = traj
        x1_bond_edge_x_t = x1_bond_edge_x_t.reshape(-1, MAX_BOND_TYPES)
    else:
        x1_bond_edge_x_t = x1_bond_edge_x_t

    if edm_inp.get('inpaint_x2_pos') and step_idx < edm_inp['stop_inpainting_at_step_x2']:
        traj = edm_inp['x2_pos_inpainting_trajectory_edm'][step_idx]
        x2_pos_t = torch.cat([traj for _ in range(batch_size)], dim=0)
        noise = torch.randn_like(x2_pos_t)
        noise[virtual_node_mask_x2] = 0.0
        x2_pos_t = x2_pos_t + edm_inp.get('add_noise_to_inpainted_x2_pos', 0.0) * noise
    else:
        x2_pos_t = x2_pos_t

    if step_idx < edm_inp.get('stop_inpainting_at_step_x3', 0):
        batch_trajectories = edm_inp.get('batch_x3_x4_inpainting_trajectories', False)
        if edm_inp.get('inpaint_x3_pos'):
            traj = edm_inp['x3_pos_inpainting_trajectory_edm'][step_idx]
            x3_pos_t = traj if batch_trajectories else torch.cat(
                [traj for _ in range(batch_size)], dim=0
            )
            noise = torch.randn_like(x3_pos_t)
            noise[virtual_node_mask_x3] = 0.0
            x3_pos_t = x3_pos_t + edm_inp.get('add_noise_to_inpainted_x3_pos', 0.0) * noise
        if edm_inp.get('inpaint_x3_x'):
            traj = edm_inp['x3_x_inpainting_trajectory_edm'][step_idx]
            x3_x_t = traj if batch_trajectories else torch.cat(
                [traj for _ in range(batch_size)], dim=0
            )
            noise = torch.randn_like(x3_x_t)
            noise[virtual_node_mask_x3] = 0.0
            x3_x_t = x3_x_t + edm_inp.get('add_noise_to_inpainted_x3_x', 0.0) * noise
    else:
        pass  # x3_pos_t, x3_x_t unchanged

    if step_idx < edm_inp.get('stop_inpainting_at_step_x4', 0):
        def replace_x4_trajectory(state: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
            local_offset, block_width, values_are_batched = (
                _x4_trajectory_block_layout(edm_inp, traj, batch_size)
            )
            return _replace_molecule_block(
                state, traj, batch_size, block_width, local_offset=local_offset,
                batch=x4_batch, values_are_batched=values_are_batched, block_name="x4",
            )

        if edm_inp.get('inpaint_x4_pos'):
            traj = edm_inp['x4_pos_inpainting_trajectory_edm'][step_idx]
            x4_pos_t = replace_x4_trajectory(x4_pos_t, traj)
            # noise = torch.randn_like(x4_pos_t)
            # noise[virtual_node_mask_x4] = 0.0
            # x4_pos_t = x4_pos_t + edm_inp.get('add_noise_to_inpainted_x4_pos', 0.0) * noise
        if edm_inp.get('inpaint_x4_direction'):
            traj = edm_inp['x4_direction_inpainting_trajectory_edm'][step_idx]
            x4_direction_t = replace_x4_trajectory(x4_direction_t, traj)
            # noise = torch.randn_like(x4_direction_t)
            # noise[virtual_node_mask_x4] = 0.0
            # x4_direction_t = x4_direction_t + edm_inp.get('add_noise_to_inpainted_x4_direction', 0.0) * noise
        if edm_inp.get('inpaint_x4_type'):
            traj = edm_inp['x4_x_inpainting_trajectory_edm'][step_idx]
            x4_x_t = replace_x4_trajectory(x4_x_t, traj)
            # noise = torch.randn_like(x4_x_t)
            # noise[virtual_node_mask_x4] = 0.0
            # x4_x_t = x4_x_t + edm_inp.get('add_noise_to_inpainted_x4_type', 0.0) * noise

    return (
        x1_pos_t, x1_x_t, x1_bond_edge_x_t,
        x2_pos_t, x3_pos_t, x3_x_t,
        x4_pos_t, x4_direction_t, x4_x_t,
    )


def generate_edm_conditional(
    model_pl: LightningModule,
    batch_size: int,
    N_x1: int | list[int],
    N_x4: int | list[int],
    num_steps: int,
    edm_inpainting_dict: dict,
    early_stop_edm: int | float = -1,
    shepherd_pred: bool = True,
    verbose: bool = True,
    store_trajectories: bool = False,
    store_trajectories_x0: bool = False,
    use_stochastic: bool = False,
    S_churn: float = 40.0,
    S_noise: float = 1.0,
    use_2nd_order_correction: bool = False,
    recenter_offset: 'np.ndarray | None' = None,
) -> list[GeneratedSample]:
    """
    EDM conditional generation.

    Default is to use predict-renoise (shepherd_pred=True) mode and uses inpainting
    for conditional generation (replacement guidance).
    x_next = D_theta(x_curr) + noise*sigma_next

    When ``property_cfg`` is provided (same structure as ``generate`` builds),
    is True and ``cfg_weight`` is non-zero, classifier-free guidance uses two forward
    passes per step (masked-off vs masked-on properties), matching ``_inference_step``.
    """
    params = model_pl.params
    N_x2 = params['dataset']['x2']['num_points']
    N_x3 = params['dataset']['x3']['num_points']
    # These are modified in `generate`
    edm_params = params.get('edm', {})
    sigma_max = edm_params.get('sigma_max', 3.0)
    sigma_min = edm_params.get('sigma_min', 1e-3)
    sigma_steps = get_edm_sigma_schedule(num_steps, edm_params)
    prior_noise_scale_edm = float(sigma_steps[0])

    edm_inp = edm_inpainting_dict
    scaffold_conditioning = edm_inp.get('scaffold_conditioning', False)
    pharmacophore_conditioning = edm_inp.get('pharmacophore_conditioning', False)
    is_mixed_pharm_mode = edm_inp.get('is_mixed_pharm_mode', False)
    num_pharm_cond = edm_inp.get('num_pharm_cond', 0)
    n_cond_with_vn = num_pharm_cond + 1  # VN at index 0 + conditional pharms

    include_virtual_node = True
    (x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, virtual_node_mask_x1, bond_edge_index_x1) = _initialize_x1_state(
        batch_size, N_x1, params, prior_noise_scale_edm, include_virtual_node, scaffold_conditioning
    )
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2 = _initialize_x2_state(
        batch_size, N_x2, params, prior_noise_scale_edm, include_virtual_node
    )
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3 = _initialize_x3_state(
        batch_size, N_x3, params, prior_noise_scale_edm, include_virtual_node
    )
    (x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4) = _initialize_x4_state(
        batch_size, N_x4, params, prior_noise_scale_edm, include_virtual_node
    )
    x1_is_diffused_atom = ~virtual_node_mask_x1
    if scaffold_conditioning:
        num_fixed_atoms = edm_inp.get('num_inpainted_atoms')
        if num_fixed_atoms is None:
            num_fixed_atoms = edm_inp['target_inpaint_x1_pos'].shape[0] - 1
        x1_is_diffused_atom = x1_is_diffused_atom.clone()
        fixed_indices = _molecule_block_indices(
            x1_batch,
            batch_size,
            local_offset=0,
            block_width=num_fixed_atoms + 1,
            block_name="x1 scaffold",
        )
        x1_is_diffused_atom[fixed_indices] = False

    x4_is_diffused_pharm = ~virtual_node_mask_x4
    if pharmacophore_conditioning:
        num_fixed_pharms = edm_inp.get('num_pharm_cond')
        if num_fixed_pharms is None:
            num_fixed_pharms = edm_inp['target_inpaint_x4_pos'].shape[0] - 1
        x4_is_diffused_pharm = x4_is_diffused_pharm.clone()
        fixed_indices = _molecule_block_indices(
            x4_batch,
            batch_size,
            local_offset=0,
            block_width=num_fixed_pharms + 1,
            block_name="x4",
        )
        x4_is_diffused_pharm[fixed_indices] = False

    scaffold_task = torch.zeros((batch_size,), dtype=torch.bool)
    if params.get('scaffold_conditioning', False) and (scaffold_conditioning or pharmacophore_conditioning):
        scaffold_task = torch.ones((batch_size,), dtype=torch.bool)

    scaffold_pos_active = scaffold_conditioning
    scaffold_com_shifted = False

    num_atom_types = len(params['dataset']['x1']['atom_types'])
    num_x_types = num_atom_types + len(params['dataset']['x1']['charge_types'])
    num_pharm_types = params['dataset']['x4']['max_node_types']
    MAX_BOND_TYPES = len(params['dataset']['x1']['bond_types'])

    target_inpaint_x1_pos = edm_inp.get('target_inpaint_x1_pos')
    target_inpaint_x1_x = edm_inp.get('target_inpaint_x1_x')
    target_inpaint_x4_pos = edm_inp.get('target_inpaint_x4_pos')
    target_inpaint_x4_direction = edm_inp.get('target_inpaint_x4_direction')
    target_inpaint_x4_x = edm_inp.get('target_inpaint_x4_x')
    if target_inpaint_x1_pos is not None:
        target_inpaint_x1_pos = target_inpaint_x1_pos.clone()
    if target_inpaint_x1_x is not None:
        target_inpaint_x1_x = target_inpaint_x1_x.clone()
    if target_inpaint_x4_pos is not None:
        target_inpaint_x4_pos = target_inpaint_x4_pos.clone()
    if target_inpaint_x4_direction is not None:
        target_inpaint_x4_direction = target_inpaint_x4_direction.clone()
    if target_inpaint_x4_x is not None:
        target_inpaint_x4_x = target_inpaint_x4_x.clone()

    # Initial state replacement: scaffold/pharmacophore (clean targets) then inpainting trajectory[0]
    if scaffold_conditioning:
        x1_pos_t, x1_x_t = _replace_scaffold_block(
            x1_pos_t,
            x1_x_t,
            target_inpaint_x1_pos,
            target_inpaint_x1_x,
            batch_size,
            x1_batch,
        )

    if pharmacophore_conditioning:
        # In mixed mode only seed the conditional slots (VN + priority-1 pharms);
        # the inpainting slots stay as random noise and are seeded by the inpaint trajectory.
        _n_cond = n_cond_with_vn if is_mixed_pharm_mode else target_inpaint_x4_pos.shape[0]
        x4_pos_t = _replace_molecule_block(
            x4_pos_t,
            target_inpaint_x4_pos[:_n_cond],
            batch_size,
            _n_cond,
            batch=x4_batch,
            block_name="x4",
        )
        x4_direction_t = _replace_molecule_block(
            x4_direction_t,
            target_inpaint_x4_direction[:_n_cond],
            batch_size,
            _n_cond,
            batch=x4_batch,
            block_name="x4",
        )
        x4_x_t = _replace_molecule_block(
            x4_x_t,
            target_inpaint_x4_x[:_n_cond],
            batch_size,
            _n_cond,
            batch=x4_batch,
            block_name="x4",
        )

    # Overwrite with trajectory[0] for each inpainted modality (inpainting at sigma_0)
    (x1_pos_t, x1_x_t, x1_bond_edge_x_t, x2_pos_t, x3_pos_t, x3_x_t, x4_pos_t, x4_direction_t, x4_x_t) = _apply_edm_inpainting_replacement(
        0, edm_inp,
        x1_pos_t, x1_x_t, x1_bond_edge_x_t, x2_pos_t, x3_pos_t, x3_x_t, x4_pos_t, x4_direction_t, x4_x_t,
        batch_size, virtual_node_mask_x2, virtual_node_mask_x3, virtual_node_mask_x4,
        num_atom_types, num_x_types, num_pharm_types, MAX_BOND_TYPES,
        x4_batch=x4_batch,
    )

    device = model_pl.device
    dtype = torch.float32
    _gamma_max = (2.0 ** 0.5) - 1.0

    trajectories: list[list[dict]] = [] if store_trajectories else []
    trajectories_x0: list[list[dict]] = [] if store_trajectories_x0 else []

    _num_steps = resolve_early_stop_steps(early_stop_edm, num_steps)
    for step_idx in tqdm(range(_num_steps), desc="EDM Steps (conditional)", disable=not verbose):
        sigma_cur = float(sigma_steps[step_idx])
        sigma_next = float(sigma_steps[step_idx + 1])

        # Replacement at start of step (step_idx is current; replace before denoising)
        (x1_pos_t, x1_x_t, x1_bond_edge_x_t, x2_pos_t, x3_pos_t, x3_x_t, x4_pos_t, x4_direction_t, x4_x_t) = _apply_edm_inpainting_replacement(
            step_idx, edm_inp,
            x1_pos_t, x1_x_t, x1_bond_edge_x_t, x2_pos_t, x3_pos_t, x3_x_t, x4_pos_t, x4_direction_t, x4_x_t,
            batch_size, virtual_node_mask_x2, virtual_node_mask_x3, virtual_node_mask_x4,
            num_atom_types, num_x_types, num_pharm_types, MAX_BOND_TYPES,
            x4_batch=x4_batch,
        )

        (
            scaffold_pos_active, scaffold_com_shifted, x1_is_diffused_atom, scaffold_task,
            x1_pos_t, x2_pos_t, x3_pos_t, x4_pos_t,
            target_inpaint_x1_pos, target_inpaint_x4_pos,
        ) = _edm_maybe_release_scaffold(
            step_idx, edm_inp,
            scaffold_pos_active, scaffold_com_shifted,
            x1_is_diffused_atom, virtual_node_mask_x1, scaffold_task, batch_size,
            x1_pos_t, x1_batch, x2_pos_t, x2_batch, virtual_node_mask_x2,
            x3_pos_t, x3_batch, virtual_node_mask_x3,
            x4_pos_t, x4_batch, virtual_node_mask_x4,
            target_inpaint_x1_pos if scaffold_conditioning else None,
            target_inpaint_x4_pos if pharmacophore_conditioning else None,
        )

        anchor_frame_active = scaffold_pos_active or (
            pharmacophore_conditioning
            and x4_is_diffused_pharm is not None
            and (~x4_is_diffused_pharm).any()
        )

        if not use_stochastic:
            next_state = _edm_denoising_step(
                model_pl, device, dtype, batch_size, sigma_cur, sigma_next,
                x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, bond_edge_index_x1, virtual_node_mask_x1,
                x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2,
                x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3,
                x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4,
                shepherd_pred,
                x1_is_diffused_atom,
                x4_is_diffused_pharm,
                scaffold_task,
                anchor_frame_active=anchor_frame_active,
                include_x0_pred=store_trajectories_x0,
            )
        else:
            gamma_i = 0.0
            if sigma_min <= sigma_cur and sigma_cur <= sigma_max:
                gamma_i = min(S_churn / num_steps, _gamma_max)
            if gamma_i > 0:
                noised_state, hat_sigma = _edm_add_churn_noise(
                    sigma_cur, gamma_i, S_noise,
                    x1_pos_t, x1_x_t, x1_bond_edge_x_t, virtual_node_mask_x1,
                    x2_pos_t, virtual_node_mask_x2,
                    x3_pos_t, x3_x_t, virtual_node_mask_x3,
                    x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
                )
                if not edm_inp.get('inpaint_x1_pos', False) and scaffold_pos_active:
                    x1_pos_t = noised_state['x1_pos_t_1']
                if not edm_inp.get('inpaint_x1_x', False) and scaffold_pos_active:
                    x1_x_t = noised_state['x1_x_t_1']
                if not edm_inp.get('inpaint_x1_bonds', False):
                    x1_bond_edge_x_t = noised_state['x1_bond_edge_x_t_1']
                if not edm_inp.get('inpaint_x2_pos', False):
                    x2_pos_t = noised_state['x2_pos_t_1']
                if not edm_inp.get('inpaint_x3_pos', False):
                    x3_pos_t = noised_state['x3_pos_t_1']
                if not edm_inp.get('inpaint_x3_x', False):
                    x3_x_t = noised_state['x3_x_t_1']

                def merge_x4_churn(
                    current: torch.Tensor,
                    noised: torch.Tensor,
                    inpaint_flag: str,
                    trajectory_key: str,
                ) -> torch.Tensor:
                    inpaint_active = (
                        edm_inp.get(inpaint_flag, False)
                        and step_idx < edm_inp.get('stop_inpainting_at_step_x4', 0)
                    )
                    if not pharmacophore_conditioning and not inpaint_active:
                        return noised

                    free_mask = x4_is_diffused_pharm.clone()
                    if inpaint_active:
                        trajectory = edm_inp[trajectory_key][step_idx]
                        offset, width, _ = _x4_trajectory_block_layout(edm_inp, trajectory, batch_size)
                        inpaint_indices = _molecule_block_indices(
                            x4_batch, batch_size, offset, width, block_name="x4",
                        ).to(free_mask.device)
                        free_mask[inpaint_indices] = False

                    merged = current.clone()
                    merged[free_mask] = noised[free_mask]
                    return merged

                x4_pos_t = merge_x4_churn(
                    x4_pos_t, noised_state['x4_pos_t_1'], 'inpaint_x4_pos', 'x4_pos_inpainting_trajectory_edm'
                )
                x4_direction_t = merge_x4_churn(
                    x4_direction_t, noised_state['x4_direction_t_1'], 'inpaint_x4_direction', 'x4_direction_inpainting_trajectory_edm'
                )
                x4_x_t = merge_x4_churn(x4_x_t, noised_state['x4_x_t_1'], 'inpaint_x4_type', 'x4_x_inpainting_trajectory_edm')
                sigma_cur = hat_sigma
            x_hat_pos = x1_pos_t
            x_hat_x = x1_x_t
            x_hat_bond = x1_bond_edge_x_t
            x_hat_x2_pos = x2_pos_t
            x_hat_x3_pos = x3_pos_t
            x_hat_x3_x = x3_x_t
            x_hat_x4_pos = x4_pos_t
            x_hat_x4_dir = x4_direction_t
            x_hat_x4_x = x4_x_t
            sigma_cur_step = sigma_cur

            next_state, d_dict = _edm_ode_step(
                model_pl, device, dtype, batch_size, sigma_cur, sigma_next,
                x1_pos_t, x1_x_t, x1_batch, x1_bond_edge_x_t, bond_edge_index_x1, virtual_node_mask_x1,
                x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2,
                x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3,
                x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4,
                shepherd_pred,
                x1_is_diffused_atom,
                x4_is_diffused_pharm,
                scaffold_task,
                anchor_frame_active=anchor_frame_active,
                include_x0_pred=store_trajectories_x0,
            )

            if use_2nd_order_correction and sigma_next > 0 and not shepherd_pred:
                _, d_prime_dict = _edm_ode_step(
                    model_pl, device, dtype, batch_size, sigma_next, sigma_next,
                    next_state['x1_pos_t_1'], next_state['x1_x_t_1'], x1_batch, next_state['x1_bond_edge_x_t_1'],
                    bond_edge_index_x1, virtual_node_mask_x1,
                    next_state['x2_pos_t_1'], next_state['x2_x_t_1'], x2_batch, virtual_node_mask_x2,
                    next_state['x3_pos_t_1'], next_state['x3_x_t_1'], x3_batch, virtual_node_mask_x3,
                    next_state['x4_pos_t_1'], next_state['x4_direction_t_1'], next_state['x4_x_t_1'], x4_batch, virtual_node_mask_x4,
                    shepherd_pred,
                    x1_is_diffused_atom,
                    x4_is_diffused_pharm,
                    scaffold_task,
                    anchor_frame_active=anchor_frame_active,
                    include_x0_pred=False,
                )
                d_sigma_2 = sigma_next - sigma_cur_step
                half = 0.5 * d_sigma_2
                next_state['x1_pos_t_1'] = x_hat_pos + half * (d_dict['x1_pos_d'] + d_prime_dict['x1_pos_d'])
                next_state['x1_x_t_1'] = x_hat_x + half * (d_dict['x1_x_d'] + d_prime_dict['x1_x_d'])
                next_state['x1_bond_edge_x_t_1'] = x_hat_bond + half * (d_dict['x1_bond_edge_x_d'] + d_prime_dict['x1_bond_edge_x_d'])
                next_state['x2_pos_t_1'] = x_hat_x2_pos + half * (d_dict['x2_pos_d'] + d_prime_dict['x2_pos_d'])
                next_state['x3_pos_t_1'] = x_hat_x3_pos + half * (d_dict['x3_pos_d'] + d_prime_dict['x3_pos_d'])
                next_state['x3_x_t_1'] = x_hat_x3_x + half * (d_dict['x3_x_d'] + d_prime_dict['x3_x_d'])
                next_state['x4_pos_t_1'] = x_hat_x4_pos + half * (d_dict['x4_pos_d'] + d_prime_dict['x4_pos_d'])
                next_state['x4_direction_t_1'] = x_hat_x4_dir + half * (d_dict['x4_direction_d'] + d_prime_dict['x4_direction_d'])
                next_state['x4_x_t_1'] = x_hat_x4_x + half * (d_dict['x4_x_d'] + d_prime_dict['x4_x_d'])
                next_state['x1_pos_t_1'][virtual_node_mask_x1, :] = 0.0
                next_state['x1_x_t_1'][virtual_node_mask_x1, :] = 0.0
                next_state['x2_pos_t_1'][virtual_node_mask_x2, :] = 0.0
                next_state['x3_pos_t_1'][virtual_node_mask_x3, :] = 0.0
                next_state['x3_x_t_1'][virtual_node_mask_x3] = 0.0
                next_state['x4_pos_t_1'][virtual_node_mask_x4, :] = 0.0
                next_state['x4_direction_t_1'][virtual_node_mask_x4, :] = 0.0
                next_state['x4_x_t_1'][virtual_node_mask_x4] = 0.0
                # Restore scaffold/pharmacophore positions after Heun correction.
                if scaffold_pos_active:
                    (
                        next_state['x1_pos_t_1'],
                        next_state['x1_x_t_1'],
                    ) = _replace_scaffold_block(
                        next_state['x1_pos_t_1'], next_state['x1_x_t_1'], target_inpaint_x1_pos,
                        target_inpaint_x1_x, batch_size, x1_batch,
                    )
                if pharmacophore_conditioning:
                    _n_cond = n_cond_with_vn if is_mixed_pharm_mode else target_inpaint_x4_pos.shape[0]
                    next_state['x4_pos_t_1'] = _replace_molecule_block(
                        next_state['x4_pos_t_1'], target_inpaint_x4_pos[:_n_cond],
                        batch_size, _n_cond, batch=x4_batch, block_name="x4",
                    )
                    next_state['x4_direction_t_1'] = _replace_molecule_block(
                        next_state['x4_direction_t_1'], target_inpaint_x4_direction[:_n_cond],
                        batch_size, _n_cond, batch=x4_batch, block_name="x4",
                    )
                    next_state['x4_x_t_1'] = _replace_molecule_block(
                        next_state['x4_x_t_1'], target_inpaint_x4_x[:_n_cond],
                        batch_size, _n_cond, batch=x4_batch, block_name="x4",
                    )

        if shepherd_pred and use_2nd_order_correction and sigma_next > 0: # and step_idx < _num_steps * 0.6:
            next_state = _edm_denoising_step(
                model_pl, device, dtype, batch_size, sigma_next, sigma_next,
                next_state['x1_pos_t_1'], next_state['x1_x_t_1'], x1_batch, next_state['x1_bond_edge_x_t_1'],
                bond_edge_index_x1, virtual_node_mask_x1,
                next_state['x2_pos_t_1'], next_state['x2_x_t_1'], x2_batch, virtual_node_mask_x2,
                next_state['x3_pos_t_1'], next_state['x3_x_t_1'], x3_batch, virtual_node_mask_x3,
                next_state['x4_pos_t_1'], next_state['x4_direction_t_1'], next_state['x4_x_t_1'], x4_batch, virtual_node_mask_x4,
                shepherd_pred,
                x1_is_diffused_atom,
                x4_is_diffused_pharm,
                scaffold_task,
                anchor_frame_active=anchor_frame_active,
                include_x0_pred=store_trajectories_x0,
            )

        if store_trajectories:
            trajectories.append(_extract_generated_samples(
                x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
                x2_pos_t, virtual_node_mask_x2,
                x3_pos_t, x3_x_t, virtual_node_mask_x3,
                x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
                params, batch_size,
                x1_batch=x1_batch,
                x4_batch=x4_batch,
                recenter_offset=recenter_offset,
            ))
        if store_trajectories_x0 and 'x0_pred' in next_state:
            trajectories_x0.append(_extract_generated_samples(
                next_state['x0_pred']['x1_x_0'], next_state['x0_pred']['x1_pos_0'],
                next_state['x0_pred']['x1_bond_edge_x_0'], virtual_node_mask_x1,
                next_state['x0_pred']['x2_pos_0'], virtual_node_mask_x2,
                next_state['x0_pred']['x3_pos_0'], next_state['x0_pred']['x3_x_0'], virtual_node_mask_x3,
                next_state['x0_pred']['x4_pos_0'], next_state['x0_pred']['x4_direction_0'],
                next_state['x0_pred']['x4_x_0'], virtual_node_mask_x4,
                params, batch_size,
                x1_batch=x1_batch,
                x4_batch=x4_batch,
                recenter_offset=recenter_offset,
            ))

        x1_pos_t = next_state['x1_pos_t_1']
        x1_x_t = next_state['x1_x_t_1']
        x1_bond_edge_x_t = next_state['x1_bond_edge_x_t_1']
        x2_pos_t = next_state['x2_pos_t_1']
        x2_x_t = next_state['x2_x_t_1']
        x3_pos_t = next_state['x3_pos_t_1']
        x3_x_t = next_state['x3_x_t_1']
        x4_pos_t = next_state['x4_pos_t_1']
        x4_direction_t = next_state['x4_direction_t_1']
        x4_x_t = next_state['x4_x_t_1']

        # Restore scaffold/pharmacophore positions to their clean target values mainly as a safety net
        if scaffold_pos_active:
            x1_pos_t, x1_x_t = _replace_scaffold_block(
                x1_pos_t,
                x1_x_t,
                target_inpaint_x1_pos,
                target_inpaint_x1_x,
                batch_size,
                x1_batch,
            )
        if pharmacophore_conditioning:
            _n_cond = n_cond_with_vn if is_mixed_pharm_mode else target_inpaint_x4_pos.shape[0]
            x4_pos_t = _replace_molecule_block(
                x4_pos_t, target_inpaint_x4_pos[:_n_cond],
                batch_size, _n_cond, batch=x4_batch, block_name="x4",
            )
            x4_direction_t = _replace_molecule_block(
                x4_direction_t, target_inpaint_x4_direction[:_n_cond],
                batch_size, _n_cond, batch=x4_batch, block_name="x4",
            )
            x4_x_t = _replace_molecule_block(
                x4_x_t, target_inpaint_x4_x[:_n_cond],
                batch_size, _n_cond, batch=x4_batch, block_name="x4",
            )

    generated_structures = _extract_generated_samples(
        x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
        x2_pos_t, virtual_node_mask_x2,
        x3_pos_t, x3_x_t, virtual_node_mask_x3,
        x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
        params, batch_size,
        x1_batch=x1_batch,
        x4_batch=x4_batch,
        recenter_offset=recenter_offset,
    )

    if store_trajectories:
        _add_trajectories_to_generated_structures(
            generated_structures, batch_size, trajectories, is_x0=False
        )

    if store_trajectories_x0:
        _add_trajectories_to_generated_structures(
            generated_structures, batch_size, trajectories_x0, is_x0=True
        )

    return generated_structures
