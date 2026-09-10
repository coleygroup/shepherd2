"""
Contains the process to extract generated samples from denoised states.
"""
from typing import Iterable

import numpy as np
import torch
from rdkit import Chem

from shepherd.generated_sample import GeneratedSample

VIRTUAL_ATOMIC_NUMBER = 0  # None (virtual node) and Dummy


def _atomic_number_for_symbol(symbol, ptable) -> int:
    if symbol is None or symbol == "Dummy":
        return VIRTUAL_ATOMIC_NUMBER
    if not isinstance(symbol, str):
        raise TypeError(
            f"atom type must be None, 'Dummy', or an element symbol, got {symbol!r}"
        )
    return int(ptable.GetAtomicNumber(symbol))


def atomic_numbers_for_atom_types(atom_types: Iterable) -> torch.Tensor:
    """Map a model's ``atom_types`` list to the atomic-number for decoding.

    ``None`` and ``'Dummy'`` both become 0.
    """
    ptable = Chem.GetPeriodicTable()
    return torch.tensor(
        [_atomic_number_for_symbol(symbol, ptable) for symbol in atom_types],
        dtype=torch.long,
    )


def atomic_number_to_symbol_map(atom_types: Iterable) -> dict[int, str]:
    """Map atomic number -> symbol using the vocab defined by
    params['dataset']['x1']['atom_types'].

    Virtual-node ``None`` and ``'Dummy'`` are omitted (they both use 0).
    """
    ptable = Chem.GetPeriodicTable()
    return {
        ptable.GetAtomicNumber(symbol): symbol
        for symbol in atom_types
        if isinstance(symbol, str) and symbol != "Dummy"
    }


def _extract_generated_samples(
        x1_x_t: torch.Tensor, x1_pos_t: torch.Tensor, x1_bond_edge_x_t: torch.Tensor, virtual_node_mask_x1: torch.Tensor,
        x2_pos_t: torch.Tensor, virtual_node_mask_x2: torch.Tensor,
        x3_pos_t: torch.Tensor, x3_x_t: torch.Tensor, virtual_node_mask_x3: torch.Tensor,
        x4_pos_t: torch.Tensor, x4_direction_t: torch.Tensor, x4_x_t: torch.Tensor, virtual_node_mask_x4: torch.Tensor,
        params: dict, batch_size: int,
        x1_batch: 'torch.Tensor | None' = None,
        x4_batch: 'torch.Tensor | None' = None,
        recenter_offset: 'np.ndarray | None' = None,
        ):
    """
    Extract final structures, and re-scale.

    Arguments
    ---------
    x1_x_t: The x1 feature tensor.
    x1_pos_t: The x1 position tensor.
    x1_bond_edge_x_t: The x1 bond edge feature tensor.
    virtual_node_mask_x1: The x1 virtual node mask tensor.
    x2_pos_t: The x2 position tensor.
    virtual_node_mask_x2: The x2 virtual node mask tensor.
    x3_pos_t: The x3 position tensor.
    x3_x_t: The x3 feature tensor.
    virtual_node_mask_x3: The x3 virtual node mask tensor.
    x4_pos_t: The x4 position tensor.
    x4_direction_t: The x4 direction tensor.
    x4_x_t: The x4 feature tensor.
    virtual_node_mask_x4: The x4 virtual node mask tensor.
    params: The parameters.
    batch_size: The batch size.
    x1_batch: Optional x1 batch vector used to split variable atom counts.
    x4_batch: Optional x4 batch vector used to split variable pharmacophore counts.
    recenter_offset: Optional (3,) array to add back to all position arrays after
        un-scaling. Used to restore the user's input reference frame when the
        inputs were auto-centered (center_of_mass=None with scaffold/pharmacophore
        conditioning). Defaults to None (no shift).

    Returns
    -------
    list: The generated structures, as GeneratedSample instances (dict subclass).
        dict:
            x1: dict
                    atoms: torch.Tensor
                    bonds: torch.Tensor
                    positions: torch.Tensor
                    formal_charges: torch.Tensor
            x2: dict
                positions: torch.Tensor
            x3: dict
                charges: torch.Tensor
                positions: torch.Tensor
            x4: dict
                types: torch.Tensor
                positions: torch.Tensor
                directions: torch.Tensor
    """
    x2_pos_final = x2_pos_t[~virtual_node_mask_x2].numpy().copy() / params['dataset'].get('scale_point_cloud', 1.0)

    x3_pos_final = x3_pos_t[~virtual_node_mask_x3].numpy().copy() / params['dataset'].get('scale_point_cloud', 1.0)
    x3_x_final = x3_x_t[~virtual_node_mask_x3].numpy().copy()
    x3_x_final = x3_x_final / params['dataset']['x3']['scale_node_features']

    x4_x_final = torch.argmin(torch.abs(x4_x_t[~virtual_node_mask_x4] - params['dataset']['x4']['scale_node_features']), dim = -1)
    x4_x_final = x4_x_final - 1 # readjusting for the previous addition of the virtual node pharmacophore type
    x4_pos_final = x4_pos_t[~virtual_node_mask_x4].numpy().copy() / params['dataset'].get('scale_point_cloud', 1.0)

    x4_direction_final = x4_direction_t[~virtual_node_mask_x4].numpy().copy() / params['dataset']['x4']['scale_vector_features']
    x4_direction_final_norm = np.linalg.norm(x4_direction_final, axis = 1)
    x4_direction_final[x4_direction_final_norm < 0.5] = 0.0
    x4_direction_final[x4_direction_final_norm >= 0.5] = x4_direction_final[x4_direction_final_norm >= 0.5] / x4_direction_final_norm[x4_direction_final_norm >= 0.5][..., None]

    # Work on a copy to avoid modifying the original tensor in-place
    x1_x_t_copy = x1_x_t.clone()
    x1_x_t_copy[~virtual_node_mask_x1, 0] = -np.inf # this masks out remaining probability assigned to virtual nodes
    x1_pos_final = x1_pos_t[~virtual_node_mask_x1].numpy().copy() / params['dataset'].get('scale_point_cloud', 1.0)
    x1_bond_edge_x_final = torch.argmin(torch.abs(x1_bond_edge_x_t - params['dataset']['x1']['scale_bond_features']), dim = -1)

    # Restore the user's input reference frame when inputs were auto-centered.
    if recenter_offset is not None:
        x1_pos_final = x1_pos_final + recenter_offset
        x2_pos_final = x2_pos_final + recenter_offset
        x3_pos_final = x3_pos_final + recenter_offset
        x4_pos_final = x4_pos_final + recenter_offset

    atomic_number_remapping = atomic_numbers_for_atom_types(params['dataset']['x1']['atom_types'])

    x1_x_final = torch.argmin(torch.abs(x1_x_t_copy[~virtual_node_mask_x1, 0:-len(params['dataset']['x1']['charge_types'])] - params['dataset']['x1']['scale_atom_features']), dim = -1)
    x1_formal_charge_final = torch.argmin(torch.abs(x1_x_t_copy[~virtual_node_mask_x1, -len(params['dataset']['x1']['charge_types']):] - params['dataset']['x1']['scale_atom_features']), dim = -1)
    x1_x_final = atomic_number_remapping[x1_x_final]
    formal_charge_remapping = torch.tensor(params['dataset']['x1']['charge_types'])
    x1_formal_charge_final = formal_charge_remapping[x1_formal_charge_final]

    # Compute per-molecule x1 split points.
    # When x1_batch is provided (variable N_x1 per molecule), derive exact per-molecule
    # node and edge counts; otherwise fall back to equal division (uniform N_x1 case).
    if x1_batch is not None:
        # real-atom counts per molecule, in order 0..batch_size-1
        real_atom_counts = torch.bincount(x1_batch[~virtual_node_mask_x1], minlength=batch_size).numpy()
        real_atom_splits = np.cumsum(real_atom_counts)[:-1]
        bond_edge_counts = (real_atom_counts * (real_atom_counts - 1) // 2).astype(int)
        bond_edge_splits = np.cumsum(bond_edge_counts)[:-1]
        x1_atoms_split = np.split(x1_x_final.numpy(), real_atom_splits)
        x1_charges_split = np.split(x1_formal_charge_final.numpy(), real_atom_splits)
        x1_bonds_split = np.split(x1_bond_edge_x_final.numpy(), bond_edge_splits)
        x1_pos_split = np.split(x1_pos_final, real_atom_splits)
    else:
        x1_atoms_split = np.split(x1_x_final.numpy(), batch_size)
        x1_charges_split = np.split(x1_formal_charge_final.numpy(), batch_size)
        x1_bonds_split = np.split(x1_bond_edge_x_final.numpy(), batch_size)
        x1_pos_split = np.split(x1_pos_final, batch_size)

    if x4_batch is not None:
        real_pharm_counts = torch.bincount(
            x4_batch[~virtual_node_mask_x4], minlength=batch_size
        ).numpy()
        real_pharm_splits = np.cumsum(real_pharm_counts)[:-1]
        x4_types_split = np.split(x4_x_final.numpy(), real_pharm_splits)
        x4_pos_split = np.split(x4_pos_final, real_pharm_splits)
        x4_direction_split = np.split(x4_direction_final, real_pharm_splits)
    else:
        x4_types_split = np.split(x4_x_final.numpy(), batch_size)
        x4_pos_split = np.split(x4_pos_final, batch_size)
        x4_direction_split = np.split(x4_direction_final, batch_size)

    # return generated structures
    generated_structures = []
    for b in range(batch_size):
        generated_dict = {
            'x1': {
                'atoms': x1_atoms_split[b],
                'formal_charges': x1_charges_split[b],
                'bonds': x1_bonds_split[b],
                'positions': x1_pos_split[b],
            },
            'x2': {
                'positions': np.split(x2_pos_final, batch_size)[b],
            },
            'x3': {
                'charges': np.split(x3_x_final, batch_size)[b], # electrostatic potential
                'positions': np.split(x3_pos_final, batch_size)[b],
            },
            'x4': {
                'types': x4_types_split[b],
                'positions': x4_pos_split[b],
                'directions': x4_direction_split[b],
            },
        }
        generated_structures.append(GeneratedSample(generated_dict))
    return generated_structures
