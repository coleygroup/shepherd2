"""
Compositional inference sampler with ShEPhERD-2.
"""
from typing import Literal

from rdkit import Chem
import numpy as np
import torch
from tqdm import tqdm

from shepherd.lightning_module import LightningModule
from shepherd.generated_sample import GeneratedSample
from shepherd.interaction_profile import ConditionAtoms, InpaintAdvancedOptions, InteractionProfile
from shepherd.inference.initialization import (
    _initialize_x1_state,
    _initialize_x2_state,
    _initialize_x3_state,
    _initialize_x4_state,
)
from shepherd.inference.noise import forward_trajectory_edm
from shepherd.inference.edm_sampler import (
    get_edm_sigma_schedule,
    _edm_add_churn_noise,
    _apply_edm_inpainting_replacement,
)
from shepherd.inference.steps import (
    _extract_generated_samples,
    atomic_number_to_symbol_map,
)
from shepherd.inference.utils import (
    _add_trajectories_to_generated_structures,
    resolve_early_stop_steps,
)
from shepherd.comp_inference.steps import (
    _inference_step_comp_edm,
    _edm_ode_step_from_D_theta,
)

import shepherd.comp_inference.utils as utils


def gen_edm_inpainting_dict(
    model_pl: LightningModule,
    N_x1: int,
    N_x4: int,
    batch_size: int,
    sigma_steps: np.ndarray,

    # toggle inpainting per modality
    inpaint_x2_pos: bool = False,
    inpaint_x3_pos: bool = False,
    inpaint_x3_x: bool = False,
    pharmacophore_conditioning: bool = False,
    inpaint_x4_pos: bool = False,
    inpaint_x4_direction: bool = False,
    inpaint_x4_type: bool = False,

    stop_inpainting_at_frac_x2: float = 1.0,
    add_noise_to_inpainted_x2_pos: float = 0.0,
    stop_inpainting_at_frac_x3: float = 1.0,
    add_noise_to_inpainted_x3_pos: float = 0.0,
    add_noise_to_inpainted_x3_x: float = 0.0,
    stop_inpainting_at_frac_x4: float = 1.0,
    add_noise_to_inpainted_x4_pos: float = 0.0,
    add_noise_to_inpainted_x4_direction: float = 0.0,
    add_noise_to_inpainted_x4_type: float = 0.0,

    # inpainting targets
    surface: np.ndarray = np.zeros((75, 3)),
    electrostatics: np.ndarray = np.zeros(75),
    pharm_types: np.ndarray = np.zeros(5, dtype=int),
    pharm_pos: np.ndarray = np.zeros((5, 3)),
    pharm_direction: np.ndarray = np.zeros((5, 3)),
    center_of_mass: np.ndarray = np.zeros(3),

    **kwargs,
) -> dict:
    """
    Build an EDM inpainting dict for surface / ESP / pharmacophore conditioning.

    Trajectories are step-indexed (0..num_steps) rather than time-indexed,
    and the stop threshold is expressed as a fraction of num_steps.

    Arguments
    ---------
    model_pl : LightningModule
    N_x1, N_x4 : int
        Number of atoms / pharmacophores to generate.
    batch_size : int
    sigma_steps : np.ndarray
        EDM sigma schedule (length num_steps + 1) used to step-index trajectories.

    inpaint_x2_pos : bool (default=False) Toggle inpainting.
        Note that x2 is implicitly modeled via x3.
    inpaint_x3_pos : bool (default=False)
    inpaint_x3_x : bool (default=False)
    pharmacophore_conditioning : bool (default=False)
    inpaint_x4_pos : bool (default=False)
    inpaint_x4_direction : bool (default=False)
    inpaint_x4_type : bool (default=False)

    stop_inpainting_at_frac_x2 : float (default=1.0)
        Progress fraction at which to stop inpainting (1.0 = data / end of
        denoising; 0.0 = prior / start). Same convention as ``early_stop_edm``.
    add_noise_to_inpainted_x2_pos : float (default=0.0)
        Scale of noise to add to inpainted values.
    stop_inpainting_at_frac_x3 : float (default=1.0)
    add_noise_to_inpainted_x3_pos : float (default=0.0)
    add_noise_to_inpainted_x3_x : float (default=0.0)
    stop_inpainting_at_frac_x4 : float (default=1.0)
    add_noise_to_inpainted_x4_pos : float (default=0.0)
    add_noise_to_inpainted_x4_direction : float (default=0.0)
    add_noise_to_inpainted_x4_type : float (default=0.0)

    *these are the inpainting targets*
    surface : np.ndarray (75, 3) (default=np.zeros((75, 3))) Surface point coordinates.
    electrostatics : np.ndarray (75,) (default=np.zeros(75)) Electrostatics at each surface point.
    pharm_types : np.ndarray (<=N_x4,) (default=np.zeros(5, dtype=int)) Pharmacophore types.
    pharm_pos : np.ndarray (<=N_x4, 3) (default=np.zeros((5, 3))) Pharmacophore positions as
        coordinates.
    pharm_direction : np.ndarray (<=N_x4, 3) (default=np.zeros((5, 3))) Pharmacophore directions
        as unit vectors.
    center_of_mass : np.ndarray (3,) (default=np.zeros(3)) Must be supplied if target molecule is
        not already centered.

    Returns
    -------
    dict : EDM inpainting dict consumed by generate_composition / _inference_step_comp_edm.
    """
    params = model_pl.params
    num_steps = len(sigma_steps) - 1

    # assertion checks for valid types
    assert len(pharm_direction) == len(pharm_pos) == len(pharm_types)
    assert N_x4 >= len(pharm_pos)

    do_partial_pharm_inpainting = N_x4 > len(pharm_pos)

    # centering about provided center of mass (of x1)
    surface = surface - center_of_mass
    pharm_pos = pharm_pos - center_of_mass

    # small noise on pharm_pos avoids overlapping points (breaks clean-structure encoding)
    pharm_pos = pharm_pos + np.random.randn(*pharm_pos.shape) * 0.01

    # accounting for virtual nodes
    surface = np.concatenate([np.zeros((1, 3)), surface], axis=0)
    electrostatics = np.concatenate([[0.0], electrostatics])
    pharm_types = pharm_types + 1  # shift for virtual node
    pharm_types = np.concatenate([[0], pharm_types])
    pharm_pos = np.concatenate([np.zeros((1, 3)), pharm_pos], axis=0)
    pharm_direction = np.concatenate([np.zeros((1, 3)), pharm_direction], axis=0)

    # one-hot encodings
    pharm_types_one_hot = np.zeros((len(pharm_types), params['dataset']['x4']['max_node_types']))
    pharm_types_one_hot[np.arange(len(pharm_types)), pharm_types] = 1

    # scaling features
    electrostatics = electrostatics * params['dataset']['x3']['scale_node_features']
    pharm_types_scaled = pharm_types_one_hot * params['dataset']['x4']['scale_node_features']
    pharm_direction_scaled = pharm_direction * params['dataset']['x4']['scale_vector_features']

    # virtual-node masks (True = real node, False = virtual)
    target_x2_mask = torch.zeros(len(surface), dtype=torch.bool)
    target_x2_mask[0] = True  # virtual node
    target_x2_mask = ~target_x2_mask  # True for real nodes

    target_x3_mask = torch.zeros(len(electrostatics), dtype=torch.bool)
    target_x3_mask[0] = True
    target_x3_mask = ~target_x3_mask

    target_x4_mask = torch.zeros(len(pharm_types), dtype=torch.bool)
    target_x4_mask[0] = True
    target_x4_mask = ~target_x4_mask

    # defining inpainting targets

    target_x2_pos = torch.as_tensor(surface, dtype=torch.float)
    target_x3_pos = torch.as_tensor(surface, dtype=torch.float)
    target_x3_x = torch.as_tensor(electrostatics, dtype=torch.float)
    target_x4_pos = torch.as_tensor(pharm_pos, dtype=torch.float)
    target_x4_direction = torch.as_tensor(pharm_direction_scaled, dtype=torch.float)
    target_x4_x = torch.as_tensor(pharm_types_scaled, dtype=torch.float)

    x2_pos_traj = None
    x3_pos_traj = None
    x3_x_traj = None
    x4_pos_traj = None
    x4_direction_traj = None
    x4_x_traj = None

    ############## generating inpainting trajectories ##############
    if inpaint_x2_pos:
        x2_pos_traj = forward_trajectory_edm(
            target_x2_pos, sigma_steps,
            remove_COM_from_noise=False, mask=target_x2_mask,
            deterministic=False, batch_size=None,
        )
    if inpaint_x3_pos:
        x3_pos_traj = forward_trajectory_edm(
            target_x3_pos, sigma_steps,
            remove_COM_from_noise=False, mask=target_x3_mask,
            deterministic=False, batch_size=None,
        )
    if inpaint_x3_x:
        x3_x_traj = forward_trajectory_edm(
            target_x3_x, sigma_steps,
            remove_COM_from_noise=False, mask=target_x3_mask,
            deterministic=False, batch_size=None,
        )
    if inpaint_x4_type:
        x4_x_traj = forward_trajectory_edm(
            target_x4_x, sigma_steps,
            remove_COM_from_noise=False, mask=target_x4_mask,
            deterministic=False, batch_size=None,
        )
    if inpaint_x4_pos:
        x4_pos_traj = forward_trajectory_edm(
            target_x4_pos, sigma_steps,
            remove_COM_from_noise=False, mask=target_x4_mask,
            deterministic=False, batch_size=None,
        )
    if inpaint_x4_direction:
        x4_direction_traj = forward_trajectory_edm(
            target_x4_direction, sigma_steps,
            remove_COM_from_noise=False, mask=target_x4_mask,
            deterministic=False, batch_size=None,
        )

    ####################################

    # stop thresholds in step indices (inpaint while step_idx < stop_at).
    # Fractions use the same convention as early_stop_edm: 1.0 = data, 0.0 = prior.
    stop_x2 = int(num_steps * stop_inpainting_at_frac_x2)
    stop_x3 = int(num_steps * stop_inpainting_at_frac_x3)
    stop_x4 = int(num_steps * stop_inpainting_at_frac_x4)

    inpainting_dict = {
        'inpaint_x2_pos': inpaint_x2_pos,
        'inpaint_x3_pos': inpaint_x3_pos,
        'inpaint_x3_x': inpaint_x3_x,
        'inpaint_x4_pos': inpaint_x4_pos,
        'inpaint_x4_direction': inpaint_x4_direction,
        'inpaint_x4_type': inpaint_x4_type,
        'pharmacophore_conditioning': pharmacophore_conditioning,
        'x2_pos_inpainting_trajectory_edm': x2_pos_traj,
        'x3_pos_inpainting_trajectory_edm': x3_pos_traj,
        'x3_x_inpainting_trajectory_edm': x3_x_traj,
        'x4_pos_inpainting_trajectory_edm': x4_pos_traj,
        'x4_direction_inpainting_trajectory_edm': x4_direction_traj,
        'x4_x_inpainting_trajectory_edm': x4_x_traj,
        'stop_inpainting_at_step_x2': stop_x2,
        'stop_inpainting_at_step_x3': stop_x3,
        'stop_inpainting_at_step_x4': stop_x4,
        'add_noise_to_inpainted_x2_pos': add_noise_to_inpainted_x2_pos,
        'add_noise_to_inpainted_x3_pos': add_noise_to_inpainted_x3_pos,
        'add_noise_to_inpainted_x3_x': add_noise_to_inpainted_x3_x,
        'add_noise_to_inpainted_x4_pos': add_noise_to_inpainted_x4_pos,
        'add_noise_to_inpainted_x4_direction': add_noise_to_inpainted_x4_direction,
        'add_noise_to_inpainted_x4_type': add_noise_to_inpainted_x4_type,
        'do_partial_pharm_inpainting': do_partial_pharm_inpainting,
        'target_inpaint_x4_pos': target_x4_pos,
        'target_inpaint_x4_direction': target_x4_direction,
        'target_inpaint_x4_x': target_x4_x,
        'target_x3_pos': target_x3_pos,
        'target_x3_x': target_x3_x,
    }

    return inpainting_dict


def gen_x1_edm_inpainting_dict(
    model_pl: LightningModule,
    N_x1: int,
    N_x4: int,
    batch_size: int,
    sigma_steps: np.ndarray,

    inpaint_x1_pos: bool = False,
    inpaint_x1_x: bool = False,
    inpaint_x1_bonds: bool = False,
    inpaint_x1_formal_charge: bool = False,
    scaffold_conditioning: bool = False,

    stop_inpainting_at_frac_x1_pos: float = 1.0,
    stop_inpainting_at_frac_x1_x: float = 1.0,
    stop_inpainting_at_frac_x1_bonds: float = 1.0,

    mol: Chem.Mol | None = None,
    atom_inds_to_inpaint: list | None = None,
    atom_types: list | None = None,
    atom_pos: np.ndarray | None = None,
    atom_formal_charges: np.ndarray | None = None,
    center_of_mass: np.ndarray | None = np.zeros(3),

    **kwargs,
) -> dict:
    """
    Build an EDM inpainting dict for atom (x1) conditioning.

    Equivalent to gen_x1_inpainting_dict in comp_inference/sampler.py but
    uses forward_trajectory_edm (step-indexed) instead of forward_trajectory.

    Arguments
    ---------
    model_pl : LightningModule
    N_x1 : int
        The number of atoms to diffuse. Must be >= the number of atoms in the `mol`.
    N_x4 : int
        The number of pharmacophores to diffuse.
    batch_size : int
    sigma_steps : np.ndarray
        EDM sigma schedule (length num_steps + 1) used to step-index trajectories.

    inpaint_x1_pos : bool (default=False)
    inpaint_x1_x : bool (default=False)
    inpaint_x1_bonds : bool (default=False)
    inpaint_x1_formal_charge : bool (default=False)

    scaffold_conditioning : bool (default=False)
        Enable scaffold-conditioned generation (distinct from inpainting).
        When True:
        - Scaffold atoms have step=0 and use learned scaffold time embeddings
        - Scaffold positions are kept fixed (no noise added)
        - COM is NOT removed from predicted noise (scaffold is at origin)
        Requires `atom_pos` and `atom_inds_to_inpaint` to specify scaffold atoms.
        Model must have been trained with `scaffold_conditioning=True`.

    stop_inpainting_at_frac_x1_pos : float (default=1.0)
        Progress fraction at which to stop inpainting atom positions
        (1.0 = data / end of denoising; 0.0 = prior / start).
    stop_inpainting_at_frac_x1_x : float (default=1.0)
        Progress fraction at which to stop inpainting atom types.
    stop_inpainting_at_frac_x1_bonds : float (default=1.0)
        Progress fraction at which to stop inpainting bond types.

    *these are the inpainting targets*
    mol : Optional[Chem.Mol] (default=None)
        Target molecule specifically for *atom*-inpainting.
        If provided, `atom_inds_to_inpaint` *must* also be provided.
        This is required for *bond*-inpainting.
        If `atom_pos` and `atom_types` are also provided,
        they will override the atom types and positions extracted from `mol`.
    atom_inds_to_inpaint : Optional[list[int]] (default=None)
        Indices of atoms to inpaint.
        This is required for *bond*-inpainting.
    atom_types : Optional[list[int]] (default=None)
        Atom elements expected as a list of atomic numbers.
        If provided alongside `mol`, `atom_inds_to_inpaint`,
        it will override the atom types extracted from `mol`.
    atom_pos : Optional[np.ndarray] (default=None) Atom positions as coordinates.
        If provided alongside `mol`, `atom_inds_to_inpaint`,
        it will override the atom positions extracted from `mol`.
    atom_formal_charges : Optional[np.ndarray] (default=None) Atom formal charges.
        If provided alongside `mol`, `atom_inds_to_inpaint`,
        it will override the atom formal charges extracted from `mol`.
    center_of_mass : np.ndarray (3,) | None (default=np.zeros(3))
        Coordinate-frame handling, mirroring `condition_center_of_mass` on
        `shepherd.inference.sampler.generate`. `np.zeros(3)` ('origin') uses
        `atom_pos` as given. `None` ('auto') is only meaningful with
        `scaffold_conditioning=True`: the scaffold COM is derived from
        `atom_pos` and used to center it (recorded in the returned dict's
        `recenter_offset` so the caller can shift outputs back); with no
        scaffold to center on, it falls back to `np.zeros(3)`. An explicit
        length-3 array subtracts that offset instead.

    Returns
    -------
    dict : EDM inpainting dict consumed by generate_composition / _inference_step_comp_edm.
        Includes `recenter_offset` (`np.ndarray (3,) | None`): the scaffold-COM
        offset applied when `center_of_mass=None` ('auto'), or `None` if no
        auto-centering occurred (add this back to generated positions to
        restore the input reference frame).
    """
    params = model_pl.params
    num_steps = len(sigma_steps) - 1
    atom_inds_to_inpaint = atom_inds_to_inpaint or []

    if scaffold_conditioning and not params.get('scaffold_conditioning', False):
        raise ValueError(
            "scaffold_conditioning=True requires a model trained with scaffold_conditioning=True."
        )

    if not scaffold_conditioning:
        params['scaffold_conditioning'] = False
        params['dataset']['x1'].pop('fixed_substructure', None)
        params['dataset']['x4'].pop('fixed_substructure', None)

    # bond types
    BOND_TYPES = params['dataset']['x1']['bond_types']
    BOND_TYPES_DICT = {b: BOND_TYPES.index(b) for b in BOND_TYPES}
    MAX_BOND_TYPES = len(BOND_TYPES_DICT)

    bond_inpaint_mask = None
    inpaint_x1_bond_edge_x = None

    if mol is not None and atom_inds_to_inpaint:
        if (inpaint_x1_pos or scaffold_conditioning) and atom_pos is None:
            atom_pos = mol.GetConformer().GetPositions()[np.array(atom_inds_to_inpaint)]
        if (inpaint_x1_x or scaffold_conditioning) and atom_types is None:
            atom_types = list(np.array(
                [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in atom_inds_to_inpaint]
            ))
        if inpaint_x1_formal_charge and atom_formal_charges is None:
            atom_formal_charges = np.array(
                [mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in atom_inds_to_inpaint]
            )
        if inpaint_x1_bonds:
            _bond_adj = np.triu(1 - np.diag(np.ones(N_x1, dtype=int)))
            _bond_edge_index = np.stack(_bond_adj.nonzero(), axis=0)
            _atom_mask = np.isin(_bond_edge_index, np.arange(len(atom_inds_to_inpaint)))
            bond_inpaint_inds = np.where(_atom_mask.all(axis=0))[0]
            bond_inpaint_mask = _atom_mask.all(axis=0)

            bond_types = []
            for idx_1, idx_2 in _bond_edge_index[:, bond_inpaint_mask].T:
                max_inpaint_idx = len(atom_inds_to_inpaint) - 1
                if int(idx_1) > max_inpaint_idx or int(idx_2) > max_inpaint_idx:
                    bond_types.append(BOND_TYPES_DICT[None])
                    continue
                idx_1m = atom_inds_to_inpaint[int(idx_1)]
                idx_2m = atom_inds_to_inpaint[int(idx_2)]
                bond = mol.GetBondBetweenAtoms(idx_1m, idx_2m)
                if bond is None:
                    bond_types.append(BOND_TYPES_DICT[None])  # non-bonded edge type; == 0
                else:
                    bond_types.append(BOND_TYPES_DICT[str(bond.GetBondType())])

            # one-hot encoding of bond types
            inpaint_x1_bond_edge_x = np.zeros((_bond_edge_index.shape[1], MAX_BOND_TYPES))
            # just select the bonds that we care about
            inpaint_x1_bond_edge_x[bond_inpaint_inds, bond_types] = 1
            inpaint_x1_bond_edge_x = (
                inpaint_x1_bond_edge_x * params['dataset']['x1']['scale_bond_features']
            )
            inpaint_x1_bond_edge_x = torch.from_numpy(inpaint_x1_bond_edge_x.copy()).float()
            bond_inpaint_mask = torch.from_numpy(bond_inpaint_mask.copy()).long()

    do_partial_atom_inpainting = False
    num_inpainted_atoms = None
    if atom_pos is not None:
        if N_x1 < len(atom_pos):
            raise ValueError(
                f"N_x1 ({N_x1}) must be >= number of inpainted atom positions ({len(atom_pos)})."
            )
        do_partial_atom_inpainting = N_x1 > len(atom_pos)
        num_inpainted_atoms = len(atom_pos)

    if atom_types is not None and atom_pos is not None:
        assert len(atom_types) == len(atom_pos)
        atomic_number_to_symbol = atomic_number_to_symbol_map(
            params['dataset']['x1']['atom_types']
        )
        atom_types = [atomic_number_to_symbol[z] for z in atom_types]

    # set scaffold atom positions/types to ground truth without any noise
    scaffold_com = None
    if scaffold_conditioning and num_inpainted_atoms is not None and atom_pos is not None:
        if center_of_mass is None:
            # 'auto': center x1 about the scaffold COM when scaffold conditioning is active
            scaffold_com = np.mean(atom_pos, axis=0)

        # disable inpainting when using scaffold conditioning
        inpaint_x1_pos = False
        inpaint_x1_x = False
        inpaint_x1_formal_charge = False
        inpaint_x1_bonds = False
        do_partial_atom_inpainting = False

    if atom_pos is None:
        # initialize dummy atom positions
        atom_pos = np.zeros((N_x1, 3))
    if atom_types is None:
        # initialize dummy atom types
        atom_types = ['C'] * N_x1

    if scaffold_com is not None:
        center_of_mass = scaffold_com
    elif center_of_mass is None:
        # 'auto' was requested but there's no scaffold to center on
        center_of_mass = np.zeros(3)
    assert center_of_mass is not None

    # centering about provided center of mass (of x1)
    atom_pos = atom_pos - center_of_mass

    # accounting for virtual nodes
    atom_pos = np.concatenate([np.zeros((1, 3)), atom_pos], axis=0)
    # atom types are converted to one-hot encoding
    atom_type_map = {sym: i for i, sym in enumerate(params['dataset']['x1']['atom_types'])}
    # virtual node type is 0 (from param file) so don't need to add +1 like in the pharm_types case
    atom_type_indices = np.array([atom_type_map[z] for z in atom_types])
    atom_type_indices = np.concatenate([[0], atom_type_indices])

    # formal charges
    if atom_formal_charges is not None and (inpaint_x1_formal_charge or scaffold_conditioning):
        charge_type_map = {
            charge_type: i
            for i, charge_type in enumerate(params['dataset']['x1']['charge_types'])
        }
        charge_type_indices = (
            np.array([charge_type_map[z] for z in atom_formal_charges])
            + len(params['dataset']['x1']['atom_types'])
        )

    num_x_types = (len(params['dataset']['x1']['atom_types'])
                   + len(params['dataset']['x1']['charge_types']))
    atom_types_one_hot = np.zeros((len(atom_type_indices), num_x_types))
    atom_types_one_hot[np.arange(len(atom_type_indices)), atom_type_indices] = 1
    if atom_formal_charges is not None and inpaint_x1_formal_charge:
        # +1 to skip the virtual node
        atom_types_one_hot[np.arange(charge_type_indices.size) + 1, charge_type_indices] = 1

    # scaling features
    atom_types_scaled = atom_types_one_hot * params['dataset']['x1']['scale_atom_features']

    # defining inpainting targets

    target_x1_pos = torch.as_tensor(atom_pos, dtype=torch.float)
    target_x1_x = torch.as_tensor(atom_types_scaled, dtype=torch.float)
    target_x1_mask = torch.ones(atom_pos.shape[0], dtype=torch.bool)
    target_x1_mask[0] = False  # virtual node excluded from noising

    x1_pos_traj = None
    x1_x_traj = None
    x1_bond_edge_x_traj = None

    if inpaint_x1_pos:
        x1_pos_traj = forward_trajectory_edm(
            target_x1_pos, sigma_steps,
            # only removes COM from noise, not the x1_pos
            remove_COM_from_noise=True, mask=target_x1_mask,
            deterministic=False, batch_size=batch_size,
        )

    if inpaint_x1_x or inpaint_x1_formal_charge:
        x1_x_traj = forward_trajectory_edm(
            target_x1_x, sigma_steps,
            remove_COM_from_noise=False, mask=target_x1_mask,
            deterministic=False, batch_size=batch_size,
        )

    num_inpainted_formal_charges = None
    if inpaint_x1_formal_charge and atom_formal_charges is not None:
        num_inpainted_formal_charges = len(atom_formal_charges)

    if inpaint_x1_bonds and inpaint_x1_bond_edge_x is not None:
        # trajectory for bonds we care about; shape (N_bonds_inpaint, MAX_BOND_TYPES), no batching
        x1_bond_edge_x_traj = forward_trajectory_edm(
            inpaint_x1_bond_edge_x[bond_inpaint_mask.bool()], sigma_steps,
        )

    ####################################

    stop_x1_pos = int(num_steps * stop_inpainting_at_frac_x1_pos)
    stop_x1_x = int(num_steps * stop_inpainting_at_frac_x1_x)
    stop_x1_bonds = int(num_steps * stop_inpainting_at_frac_x1_bonds)

    inpainting_dict = {
        'inpaint_x1_pos': inpaint_x1_pos,
        'inpaint_x1_x': inpaint_x1_x,
        'inpaint_x1_bonds': inpaint_x1_bonds,
        'inpaint_x1_formal_charge': inpaint_x1_formal_charge,
        'scaffold_conditioning': scaffold_conditioning,
        'target_inpaint_x1_pos': target_x1_pos,
        'target_inpaint_x1_x': target_x1_x,
        'x1_pos_inpainting_trajectory_edm': x1_pos_traj,
        'x1_x_inpainting_trajectory_edm': x1_x_traj,
        'x1_bond_edge_x_inpainting_trajectory_edm': x1_bond_edge_x_traj,
        'bond_inpaint_mask': bond_inpaint_mask,
        'stop_inpainting_at_step_x1_pos': stop_x1_pos,
        'stop_inpainting_at_step_x1_x': stop_x1_x,
        'stop_inpainting_at_step_x1_bonds': stop_x1_bonds,
        'do_partial_atom_inpainting': do_partial_atom_inpainting,
        'num_inpainted_atoms': num_inpainted_atoms,
        'num_inpainted_formal_charges': num_inpainted_formal_charges,
        'center_of_mass': center_of_mass,
        'recenter_offset': scaffold_com,
    }
    return inpainting_dict


def generate_composition(
    model_pl: LightningModule,
    N_x1: int,
    N_x4: int,
    batch_size: int,
    *,

    profiles: list[InteractionProfile],
    weights_conditions: np.ndarray,
    composition_mode: str = "default",

    atom_condition: InteractionProfile | ConditionAtoms | None = None,
    condition_modalities: str | set[str] = 'all',
    pharmacophore_conditioning: bool = False,
    scaffold_conditioning: bool = False,
    condition_center_of_mass: 'Literal["origin", "auto"] | np.ndarray' = 'origin',
    inpaint: InpaintAdvancedOptions | None = None,

    num_steps: int = 400,

    # EDM stochastic sampler options
    use_stochastic: bool = False,
    S_churn: float = 40.0,
    S_noise: float = 1.0,
    shepherd_pred: bool = True,
    # if != -1, overrides num_steps and stops EDM sampling at this step count
    # int: absolute steps; float in [0, 1]: fraction of num_steps
    early_stop_edm: int | float = -1,

    # override EDM schedule params (falls back to params['edm'])
    sigma_max: float | None = 3.0,
    sigma_min: float | None = 1e-3,
    rho: float | None = 7.0,

    store_trajectories: bool = False,
    verbose: bool = True,
    # ESP alignment: fraction of steps (from the end) during which alignment is applied
    alignment_start_frac: float = 0.0,
    alignment_interval: int = 10,
    alignment_mode: str = 'so3',
    alignment_ema_alpha: float = 1.0,  # 1.0 = no smoothing, <1.0 = EMA smoothing
) -> list[GeneratedSample]:
    """
    Compositional EDM generation mirroring gen_composition3 from comp_inference.

    Multiple interaction-profile conditions are composed via weighted combination
    of EDM D_theta predictions, then denoised with an EDM ODE step.

    Arguments
    ---------
    model_pl : LightningModule
    N_x1, N_x4 : int
        Number of atoms / pharmacophores to generate.
    batch_size : int
    profiles : list[InteractionProfile]
        One condition per entry, composed via weighted combination of EDM
        D_theta predictions, translated via
        ``comp_inference.utils.condition_kwargs_from_profile`` (same objects
        used by ``shepherd.inference.sampler.generate``).
    weights_conditions : np.ndarray
        Per-condition weights (excluding the implicit unconditional component).
    composition_mode : str
        "default" prepends an unconditional condition (weight = 1 - sum(weights)).
        "conditional" uses only the supplied conditions.
    atom_condition : InteractionProfile | ConditionAtoms, optional
        x1 conditioning target (scaffold and/or x1 inpainting), translated via
        ``comp_inference.utils.atom_condition_kwargs_from_profile``. ``None``
        means no atom conditioning.
    condition_modalities : see InteractionProfile.to_generate_kwargs.
        Applied to every entry of ``profiles``.
    pharmacophore_conditioning, scaffold_conditioning : bool (default=False)
        As in ``generate()`` — not derived from ``profiles``/``atom_condition``
        themselves.
    condition_center_of_mass : {'origin', 'auto'} | np.ndarray (3,), default='origin'
        Coordinate-frame handling for ``atom_condition``, mirroring
        ``generate()``'s argument of the same name. ``'origin'`` uses
        ``atom_condition``'s coordinates as given. ``'auto'`` -- only
        meaningful with ``scaffold_conditioning=True`` -- centers the scaffold
        on its own COM for denoising and shifts the output back to the input
        frame afterward. A length-3 array subtracts that explicit offset
        instead. Ignored if ``atom_condition`` is None. The single shared
        offset this resolves to is also applied to every entry of
        ``profiles`` (surface/ESP/pharmacophore conditioning share one frame
        with the atom condition; see ``gen_x1_edm_inpainting_dict``).
        We suggest using 'origin' for most cases.
    inpaint : InpaintAdvancedOptions, optional
        Overrides applied to every profile/atom_condition, as in ``generate()``.
    num_steps : int
        Number of EDM denoising steps.
    early_stop_edm : int | float
        Truncate the sampling loop. ``-1`` runs all steps; an ``int`` is an
        absolute step count; a ``float`` in ``[0, 1]`` is a fraction of
        ``num_steps``.
    use_stochastic : bool
        If True, adds churn noise per EDM Algorithm 2.
    S_churn, S_noise : float
        Churn parameters for stochastic sampling.
    sigma_max, sigma_min, rho : float, optional
        Override the EDM schedule from params['edm'].
    store_trajectories : bool

    Returns
    -------
    list[dict] : generated structures, one per batch element.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if isinstance(condition_center_of_mass, str):
        if condition_center_of_mass == 'origin':
            resolved_center_of_mass = np.zeros(3)
        elif condition_center_of_mass == 'auto':
            resolved_center_of_mass = None
        else:
            raise ValueError(
                "condition_center_of_mass must be 'origin', 'auto', or a "
                "length-3 array."
            )
    else:
        resolved_center_of_mass = np.asarray(condition_center_of_mass, dtype=float)
        if resolved_center_of_mass.shape != (3,):
            raise ValueError(
                "condition_center_of_mass must be 'origin', 'auto', or a "
                f"length-3 array; got shape {resolved_center_of_mass.shape}."
            )

    conditions = [
        utils.condition_kwargs_from_profile(
            profile,
            condition_modalities=condition_modalities,
            pharmacophore_conditioning=pharmacophore_conditioning,
            inpaint=inpaint,
        )
        for profile in profiles
    ]

    if atom_condition is None:
        atom_conditions = {}
    else:
        atom_conditions = utils.atom_condition_kwargs_from_profile(
            atom_condition,
            scaffold_conditioning=scaffold_conditioning,
            inpaint=inpaint,
            center_of_mass=resolved_center_of_mass,
        )

    params = model_pl.params
    edm_params = dict(params.get('edm', {}))
    if sigma_max is not None:
        edm_params['sigma_max'] = sigma_max
    if sigma_min is not None:
        edm_params['sigma_min'] = sigma_min
    if rho is not None:
        edm_params['rho'] = rho

    sigma_steps = get_edm_sigma_schedule(num_steps, edm_params)
    sigma_max_val = float(sigma_steps[0])
    sigma_min_val = float(edm_params.get('sigma_min', 1e-3))
    sigma_max_sched = float(edm_params.get('sigma_max', 80.0))

    N_x2 = params['dataset']['x2']['num_points']
    N_x3 = params['dataset']['x3']['num_points']
    num_atom_types = len(params['dataset']['x1']['atom_types'])
    num_x_types = num_atom_types + len(params['dataset']['x1']['charge_types'])
    num_pharm_types = params['dataset']['x4']['max_node_types']
    MAX_BOND_TYPES = len(params['dataset']['x1']['bond_types'])

    scaffold_conditioning = atom_conditions.get('scaffold_conditioning', False)

    # ---- build inpainting dicts ----
    atom_edm_ipt_dict = gen_x1_edm_inpainting_dict(
        model_pl=model_pl,
        N_x1=N_x1,
        N_x4=N_x4,
        batch_size=batch_size,
        sigma_steps=sigma_steps,
        **atom_conditions,
    )
    center_of_mass = atom_edm_ipt_dict.get('center_of_mass', np.zeros(3))
    recenter_offset = atom_edm_ipt_dict.get('recenter_offset')

    if composition_mode == 'default':
        all_edm_ipt_dicts = [None]  # first entry = unconditional
    else:
        all_edm_ipt_dicts = []

    for cdtn_i in conditions:
        cdtn_i = {k: v for k, v in cdtn_i.items() if k != 'center_of_mass'}
        ipt_dict_i = gen_edm_inpainting_dict(
            model_pl=model_pl,
            N_x1=N_x1,
            N_x4=N_x4,
            batch_size=batch_size,
            sigma_steps=sigma_steps,
            center_of_mass=center_of_mass,
            **cdtn_i,
        )
        all_edm_ipt_dicts.append(ipt_dict_i)

    # ---- build composition weights ----
    weights = torch.tensor(weights_conditions, dtype=torch.float32)
    unconditional_weight = 1.0 - weights.sum().unsqueeze(-1)
    if composition_mode == 'default':
        weights = torch.cat([unconditional_weight, weights])
    elif composition_mode == 'conditional':
        if verbose:
            print("Using conditional composition (no unconditional component).")
        if torch.all(weights >= 0):
            pass  # use supplied weights as-is
        elif weights_conditions[0] > 0 and weights_conditions[1] < 0:
            # normalize to sum 1: anchor cond1, repulsion |w[1]| (D = (1+|w1|)*D1 - |w1|*D2)
            weights = torch.tensor(
                [weights_conditions[0]+1, weights_conditions[1]], dtype=torch.float32
            )
        else:
            raise NotImplementedError(
                "Invalid two conditional weight configuration -- "
                "possibly valid but yet to be implemented"
            )

    # ---- initialise states at sigma_max ----
    include_virtual_node = True
    (x1_pos_t, x1_x_t, x1_bond_edge_x_t,
     x1_batch, virtual_node_mask_x1, bond_edge_index_x1) = _initialize_x1_state(
        batch_size, N_x1, params, sigma_max_val, include_virtual_node, scaffold_conditioning
    )

    target_inpaint_x1_pos = atom_edm_ipt_dict.get('target_inpaint_x1_pos')
    target_inpaint_x1_x = atom_edm_ipt_dict.get('target_inpaint_x1_x')
    num_inpainted_atoms = atom_edm_ipt_dict.get('num_inpainted_atoms')

    if scaffold_conditioning and target_inpaint_x1_pos is not None:
        x1_pos_t = x1_pos_t.reshape(batch_size, -1, 3)
        x1_pos_t[:, :target_inpaint_x1_pos.shape[0]] = target_inpaint_x1_pos
        x1_pos_t = x1_pos_t.reshape(-1, 3)
        x1_x_t = x1_x_t.reshape(batch_size, -1, num_x_types)
        x1_x_t[:, :target_inpaint_x1_x.shape[0]] = target_inpaint_x1_x
        x1_x_t = x1_x_t.reshape(-1, num_x_types)

    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2 = _initialize_x2_state(
        batch_size, N_x2, params, sigma_max_val, include_virtual_node
    )
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3 = _initialize_x3_state(
        batch_size, N_x3, params, sigma_max_val, include_virtual_node
    )
    (x4_pos_t, x4_direction_t, x4_x_t,
     x4_batch, virtual_node_mask_x4) = _initialize_x4_state(
        batch_size, N_x4, params, sigma_max_val, include_virtual_node
    )

    # ---- scaffold / pharmacophore diffusion masks ----
    x1_is_diffused_atom = ~virtual_node_mask_x1
    x4_is_diffused_pharm = ~virtual_node_mask_x4

    if scaffold_conditioning and num_inpainted_atoms is not None:
        x1_is_diffused_atom = x1_is_diffused_atom.clone().reshape(batch_size, -1)
        x1_is_diffused_atom[:, :num_inpainted_atoms + 1] = False  # +1 for virtual node
        x1_is_diffused_atom = x1_is_diffused_atom.reshape(-1)

    scaffold_task = torch.zeros((batch_size,), dtype=torch.bool)
    if params.get('scaffold_conditioning', False) and scaffold_conditioning:
        scaffold_task = torch.ones((batch_size,), dtype=torch.bool)

    # ---- apply trajectory[0] replacement for inpainted modalities ----
    if atom_edm_ipt_dict is not None:
        (x1_pos_t, x1_x_t, x1_bond_edge_x_t,
         x2_pos_t, x3_pos_t, x3_x_t,
         x4_pos_t, x4_direction_t, x4_x_t) = _apply_edm_inpainting_replacement(
            0, atom_edm_ipt_dict,
            x1_pos_t, x1_x_t, x1_bond_edge_x_t,
            x2_pos_t, x3_pos_t, x3_x_t,
            x4_pos_t, x4_direction_t, x4_x_t,
            batch_size,
            virtual_node_mask_x2, virtual_node_mask_x3, virtual_node_mask_x4,
            num_atom_types, num_x_types, num_pharm_types, MAX_BOND_TYPES,
        )

    _gamma_max = (2.0 ** 0.5) - 1.0

    trajectories = []

    if store_trajectories:
        trajectories.append(_extract_generated_samples(
            x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
            x2_pos_t, virtual_node_mask_x2,
            x3_pos_t, x3_x_t, virtual_node_mask_x3,
            x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
            params, batch_size,
            recenter_offset=recenter_offset,
        ))

    # ESP alignment state: updated each step from D_theta, applied next step
    alignments = None  # list of per-condition alignments, one entry per ipt_dict
    alignments_ema = None  # EMA-smoothed version of alignments (same shape)
    x3_pos_0_prev = None
    x3_x_0_prev = None
    alignment_start_step = int((1.0 - alignment_start_frac) * num_steps)

    _any_esp_condition = any(
        d is not None and d.get('inpaint_x3_pos', False) and d.get('inpaint_x3_x', False)
        for d in all_edm_ipt_dicts
    )

    # ---- main denoising loop ----
    _num_steps = resolve_early_stop_steps(early_stop_edm, num_steps)
    pbar = tqdm(range(_num_steps), desc="EDM compositional steps", disable=not verbose)
    for step_idx in pbar:
        sigma_cur = float(sigma_steps[step_idx])
        sigma_next = float(sigma_steps[step_idx + 1])
        pbar.set_postfix(sigma=f"{sigma_cur:.3f}")

        if use_stochastic:
            gamma_i = 0.0
            if sigma_min_val <= sigma_cur <= sigma_max_sched:
                gamma_i = min(S_churn / _num_steps, _gamma_max)
            if gamma_i > 0:
                noised_state, hat_sigma = _edm_add_churn_noise(
                    sigma_cur, gamma_i, S_noise,
                    x1_pos_t, x1_x_t, x1_bond_edge_x_t, virtual_node_mask_x1,
                    x2_pos_t, virtual_node_mask_x2,
                    x3_pos_t, x3_x_t, virtual_node_mask_x3,
                    x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
                )
                # don't overwrite inpainted modalities with churn noise
                if not atom_edm_ipt_dict.get('inpaint_x1_x', False) and not scaffold_conditioning:
                    x1_x_t = noised_state['x1_x_t_1']
                if not atom_edm_ipt_dict.get('inpaint_x1_bonds', False):
                    x1_bond_edge_x_t = noised_state['x1_bond_edge_x_t_1']
                any_x2 = any(d is not None and d.get('inpaint_x2_pos', False)
                             for d in all_edm_ipt_dicts)
                any_x3 = any(d is not None and (d.get('inpaint_x3_pos', False)
                             or d.get('inpaint_x3_x', False)) for d in all_edm_ipt_dicts)
                any_x4 = any(d is not None and (d.get('inpaint_x4_pos', False)
                             or d.get('inpaint_x4_direction', False)
                             or d.get('inpaint_x4_type', False)) for d in all_edm_ipt_dicts)
                if not any_x2:
                    x2_pos_t = noised_state['x2_pos_t_1']
                if not any_x3:
                    x3_pos_t = noised_state['x3_pos_t_1']
                    x3_x_t = noised_state['x3_x_t_1']
                if not any_x4:
                    x4_pos_t = noised_state['x4_pos_t_1']
                    x4_direction_t = noised_state['x4_direction_t_1']
                    x4_x_t = noised_state['x4_x_t_1']

                _no_inpaint_x1_pos = not atom_edm_ipt_dict.get('inpaint_x1_pos', False)
                if _no_inpaint_x1_pos and not scaffold_conditioning:
                    x1_pos_t = noised_state['x1_pos_t_1']

                sigma_cur = hat_sigma

        # recompute per-condition ESP alignment every alignment_interval steps (else reuse previous)
        if step_idx < alignment_start_step or not _any_esp_condition:
            alignments = None
            alignments_ema = None
        elif (x3_pos_0_prev is not None and x3_x_0_prev is not None
                and (step_idx - alignment_start_step) % alignment_interval == 0):
            if verbose:
                print(f"Computing alignments at step {step_idx} (sigma={sigma_cur:.4f})")
            x3_scale = params['dataset']['x3']['scale_node_features']
            raw_alignments = []
            for ipt_dict in all_edm_ipt_dicts:
                if (ipt_dict is not None
                        and ipt_dict.get('inpaint_x3_pos', False)
                        and ipt_dict.get('inpaint_x3_x', False)):
                    cond_alignment = []
                    for i in range(batch_size):
                        _, se3 = utils.align_model_states(
                            model_outputs={},
                            D_theta={'x3_pos': x3_pos_0_prev[i], 'x3_x': x3_x_0_prev[i]},
                            cdt_dict={
                                'surface': ipt_dict['target_x3_pos'][1:],
                                'electrostatics': ipt_dict['target_x3_x'][1:] / x3_scale,
                            },
                            params=params,
                            mode=alignment_mode,
                            verbose=False,
                        )
                        cond_alignment.append(se3)
                    raw_alignments.append(cond_alignment)
                else:
                    raw_alignments.append(None)

            if alignment_ema_alpha >= 1.0 or alignments_ema is None:
                alignments_ema = raw_alignments
            else:
                new_ema = []
                for raw_cond, ema_cond in zip(raw_alignments, alignments_ema):
                    if raw_cond is None or ema_cond is None:
                        new_ema.append(raw_cond)
                        continue
                    blended_cond = []
                    for se3_raw, se3_ema in zip(raw_cond, ema_cond):
                        blended = torch.eye(4)
                        R_blend = (
                            (1.0 - alignment_ema_alpha) * se3_ema[:3, :3]
                            + alignment_ema_alpha * se3_raw[:3, :3]
                        )
                        U, _, Vt = torch.linalg.svd(R_blend)
                        blended[:3, :3] = U @ Vt
                        if alignment_mode == 'se3':
                            blended[:3, 3] = (
                                (1.0 - alignment_ema_alpha) * se3_ema[:3, 3]
                                + alignment_ema_alpha * se3_raw[:3, 3]
                            )
                        blended_cond.append(blended)
                    new_ema.append(blended_cond)
                alignments_ema = new_ema

            alignments = alignments_ema
        # else: inside alignment window but not a recompute step - keep previous alignments

        step_result = _inference_step_comp_edm(
            model_pl=model_pl,
            params=params,
            step_idx=step_idx,
            sigma_cur=sigma_cur,
            batch_size=batch_size,
            x1_pos_t=x1_pos_t, x1_x_t=x1_x_t, x1_bond_edge_x_t=x1_bond_edge_x_t,
            x1_batch=x1_batch, bond_edge_index_x1=bond_edge_index_x1,
            virtual_node_mask_x1=virtual_node_mask_x1,
            x2_pos_t=x2_pos_t, x2_x_t=x2_x_t, x2_batch=x2_batch,
            virtual_node_mask_x2=virtual_node_mask_x2,
            x3_pos_t=x3_pos_t, x3_x_t=x3_x_t, x3_batch=x3_batch,
            virtual_node_mask_x3=virtual_node_mask_x3,
            x4_pos_t=x4_pos_t, x4_direction_t=x4_direction_t, x4_x_t=x4_x_t,
            x4_batch=x4_batch, virtual_node_mask_x4=virtual_node_mask_x4,
            edm_inpainting_dict_arr=all_edm_ipt_dicts,
            atom_edm_inpainting_dict=atom_edm_ipt_dict,
            x1_is_diffused_atom=x1_is_diffused_atom,
            x4_is_diffused_pharm=x4_is_diffused_pharm,
            scaffold_task=scaffold_task,
            composition_mode=composition_mode,
            weights_conditions=weights,
            include_x0_pred=False,
            alignments=alignments,
            mode=alignment_mode,
        )

        D_theta = step_result['D_theta']

        # store D_theta x3 predictions for next step's alignment computation
        if _any_esp_condition:
            x3_pos_0_prev = D_theta['x3_pos'].reshape(batch_size, -1, 3)
            x3_x_raw = D_theta['x3_x']
            x3_x_0_prev = x3_x_raw.reshape(batch_size, -1)

        scaffold_task_any = bool(scaffold_task.any())

        next_state = _edm_ode_step_from_D_theta(
            sigma_cur=sigma_cur,
            sigma_next=sigma_next,
            x1_pos_t=x1_pos_t, x1_x_t=x1_x_t, x1_bond_edge_x_t=x1_bond_edge_x_t,
            virtual_node_mask_x1=virtual_node_mask_x1, x1_batch=x1_batch,
            x2_pos_t=x2_pos_t, x2_x_t=x2_x_t, virtual_node_mask_x2=virtual_node_mask_x2,
            x3_pos_t=x3_pos_t, x3_x_t=x3_x_t, virtual_node_mask_x3=virtual_node_mask_x3,
            x4_pos_t=x4_pos_t, x4_direction_t=x4_direction_t, x4_x_t=x4_x_t,
            virtual_node_mask_x4=virtual_node_mask_x4,
            D_theta=D_theta,
            x1_is_diffused_atom=x1_is_diffused_atom,
            x4_is_diffused_pharm=x4_is_diffused_pharm,
            scaffold_task_any=scaffold_task_any,
            shepherd_pred=shepherd_pred,
        )
        if scaffold_task_any:
            next_state['x1_pos_t_1'][~x1_is_diffused_atom] = x1_pos_t[~x1_is_diffused_atom]
            next_state['x1_x_t_1'][~x1_is_diffused_atom] = x1_x_t[~x1_is_diffused_atom]
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

        if store_trajectories and step_idx < num_steps - 1:
            trajectories.append(_extract_generated_samples(
                x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
                x2_pos_t, virtual_node_mask_x2,
                x3_pos_t, x3_x_t, virtual_node_mask_x3,
                x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
                params, batch_size,
                recenter_offset=recenter_offset,
            ))

    generated_structures = _extract_generated_samples(
        x1_x_t, x1_pos_t, x1_bond_edge_x_t, virtual_node_mask_x1,
        x2_pos_t, virtual_node_mask_x2,
        x3_pos_t, x3_x_t, virtual_node_mask_x3,
        x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4,
        params, batch_size,
        recenter_offset=recenter_offset,
    )

    if store_trajectories:
        _add_trajectories_to_generated_structures(
            generated_structures, batch_size, trajectories, is_x0=False
        )

    return generated_structures
