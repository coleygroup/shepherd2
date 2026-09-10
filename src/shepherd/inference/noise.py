"""
This module contains the noise functions for the inference sampler.
"""
import numpy as np
import torch
import torch_scatter


def forward_trajectory_edm(
    x: torch.Tensor,
    sigma_steps: np.ndarray,
    remove_COM_from_noise: bool = False,
    mask: torch.Tensor | None = None,
    deterministic: bool = False,
    batch: torch.Tensor | None = None,
    batch_size: int | None = None,
) -> dict[int, torch.Tensor]:
    """
    Simulate an EDM forward noising trajectory: for each step index,
    trajectory[step_idx] = x + sigma_steps[step_idx] * noise.

    sigma_steps has shape (num_steps + 1,) with high to low sigma.
    Returns dict keyed by step index 0..num_steps for use in EDM inpainting.

    Args
    ----
        x: Clean target tensor.
        sigma_steps: Sigma at each step, shape (num_steps + 1,), high to low.
        remove_COM_from_noise: Whether to remove center of mass from noise.
        mask: True for positions to noise; non-masked are left as x (or caller uses slice).
        deterministic: If True, noise = 0 (clean trajectory).
        batch: Batch indices for COM removal when mask is used.
        batch_size: If > 1, repeat x (and mask/batch) for batch_size independent trajectories.

    Returns
    -------
        dict[int, torch.Tensor]: trajectory[step_idx] for step_idx in 0..len(sigma_steps)-1.
    """
    num_steps_plus_1 = len(sigma_steps)
    if mask is None:
        mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    if isinstance(x, torch.Tensor):
        x = x.clone()
    else:
        x = torch.as_tensor(x).clone()

    if batch_size is not None and batch_size > 1:
        orig_num_nodes = x.shape[0]
        x = x.unsqueeze(0).repeat(batch_size, *([1] * x.dim())).reshape(-1, *x.shape[1:])
        if mask is not None:
            mask = mask.repeat(batch_size, *([1] * (mask.dim() - 1)))
        if batch is not None:
            batch = batch.repeat(batch_size)
        else:
            batch = torch.repeat_interleave(torch.arange(batch_size, device=x.device), repeats=orig_num_nodes)

    if remove_COM_from_noise:
        assert x.shape[1] == 3

    trajectory = {}
    for step_idx in range(num_steps_plus_1):
        sigma = float(sigma_steps[step_idx])
        noise = torch.randn_like(x)
        if remove_COM_from_noise:
            if batch is not None:
                noise = noise - torch_scatter.scatter_mean(noise[mask], index=batch[mask], dim=0)[batch]
            else:
                noise = noise - torch.mean(noise[mask], dim=0)
        noise[~mask, ...] = 0.0
        if deterministic:
            noise = 0.0
        x_sigma = x + sigma * noise
        x_sigma[~mask, ...] = x[~mask, ...]
        trajectory[step_idx] = x_sigma

    return trajectory
