import itertools
import time
from typing import Tuple

import numpy as np
import rdkit
import torch
from rdkit import Chem

from shepherd_score.container import Molecule, MoleculePair, update_mol_coordinates
from shepherd_score.score.constants import ALPHA
from shepherd_score.score.electrostatic_scoring import get_overlap_esp
from shepherd_score.alignment.utils.se3 import (
    apply_SE3_transform,
    apply_SO3_transform,
    get_SE3_transform,
)
from shepherd_score.alignment import (
    optimize_ROCS_esp_overlay,
    _initialize_se3_params,
    objective_ROCS_esp_overlay,
)

from shepherd.interaction_profile import ConditionAtoms, InpaintAdvancedOptions, InteractionProfile


def get_random_conditions(n_cond=2, n_samples=100, max_len=100, seed=42):
    """
    Sample random combinations of indices, e.g. for picking random condition pairs.

    Arguments
    ---------
    n_cond : int (default=2)
        Number of indices per combination.
    n_samples : int (default=100)
        Number of combinations to sample.
    max_len : int (default=100)
        Indices are sampled from range(max_len).
    seed : int (default=42)

    Returns
    -------
    np.ndarray : n_samples combinations, each of length n_cond.
    """
    points = list(range(max_len))  # or list of coordinates, etc.
    pairs = list(itertools.combinations(points, n_cond))

    rng = np.random.default_rng(seed)  # use new Generator API
    arr = rng.choice(pairs, size=n_samples, replace=False)

    return arr


def optimize_SO3_esp_overlay(
    ref_points: torch.Tensor,
    fit_points: torch.Tensor,
    ref_charges: torch.Tensor,
    fit_charges: torch.Tensor,
    alpha: float,
    lam: float,
    num_repeats: int = 50,
    lr: float = 0.1,
    max_num_steps: int = 200,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same interface as optimize_ROCS_esp_overlay but restricts optimization to SO3 (rotation only).
    Translation is frozen at the CoM-aligning value computed during initialization - gradients are
    never propagated through the translation indices of se3_params.

    The trick: split the (num_repeats, 7) initialization into
      quat_params  (num_repeats, 4)  requires_grad=True   <- optimized
      fixed_trans  (num_repeats, 3)  requires_grad=False  <- frozen
    then rebuild se3_params = cat([quat_params, fixed_trans]) each step so autograd only
    touches the rotation leaf. No changes to any underlying objective or transform functions.

    Arguments
    ---------
    ref_points, fit_points : torch.Tensor
        Reference and fit point clouds to align.
    ref_charges, fit_charges : torch.Tensor
        Per-point charges/features used by the ESP similarity objective.
    alpha, lam : float
        ESP similarity scoring hyperparameters.
    num_repeats : int (default=50)
        Number of random restarts.
    lr : float (default=0.1)
    max_num_steps : int (default=200)
    verbose : bool (default=False)

    Returns
    -------
    aligned_points : torch.Tensor (M, 3)
    SE3_transform  : torch.Tensor (4, 4)
    score          : torch.Tensor (1,)
    """
    se3_init = _initialize_se3_params(
        ref_points=ref_points, fit_points=fit_points, num_repeats=num_repeats
    )
    is_batched = se3_init.dim() == 2

    if is_batched:
        current_num_repeats = se3_init.shape[0]
        quat_params = se3_init[:, :4].detach().clone().requires_grad_(True)
        fixed_trans = se3_init[:, 4:].detach().clone()
        fit_points_batched = fit_points.repeat((current_num_repeats, 1, 1))
        fit_charges_batched = fit_charges.repeat((current_num_repeats, 1))
    else:
        current_num_repeats = 1
        quat_params = se3_init[:4].detach().clone().requires_grad_(True)
        fixed_trans = se3_init[4:].detach().clone()
        fit_points_batched = fit_points
        fit_charges_batched = fit_charges

    optimizer = torch.optim.Adam([quat_params], lr=lr)

    if verbose:
        initial_score = get_overlap_esp(
            ref_points, fit_points, ref_charges, fit_charges, alpha, lam
        )
        print(f'Initial ESP similarity score: {initial_score:.3f}')

    last_loss = torch.tensor(float('inf'), device=ref_points.device)
    counter = 0

    for step in range(max_num_steps):
        se3_params = torch.cat([quat_params, fixed_trans], dim=-1)
        loss = objective_ROCS_esp_overlay(
            se3_params=se3_params,
            ref_points=ref_points,
            fit_points=fit_points_batched,
            ref_charges=ref_charges,
            fit_charges=fit_charges_batched,
            alpha=alpha,
            lam=lam,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 100 == 0:
            print(f'Step {step}, Score: {1 - loss.item():.3f}')

        if torch.abs(loss - last_loss) > 1e-5:
            counter = 0
        else:
            counter += 1
        last_loss = loss
        if counter > 10:
            break

    with torch.no_grad():
        se3_params_final = torch.cat([quat_params, fixed_trans], dim=-1).detach()
        SE3_transform = get_SE3_transform(se3_params_final)
        aligned_points = apply_SE3_transform(fit_points_batched, SE3_transform)
        scores = get_overlap_esp(
            centers_1=ref_points,
            charges_1=ref_charges,
            centers_2=aligned_points,
            charges_2=fit_charges_batched,
            alpha=alpha,
            lam=lam,
        )

    if current_num_repeats == 1:
        if verbose:
            print(f'Optimized ESP similarity score: {scores.item():.3f}')
        return aligned_points.cpu(), SE3_transform.cpu(), scores.cpu()

    best_idx = torch.argmax(scores.detach().cpu())
    if verbose:
        print(
            f'Optimized ESP similarity score -- max: {scores[best_idx].item():.3f} '
            f'| mean: {scores.mean().item():.3f} | min: {scores.min().item():.3f}'
        )
    return aligned_points.cpu()[best_idx], SE3_transform.cpu()[best_idx], scores.cpu()[best_idx]


def align_two_ref_mol(
    molblocks_and_charges,
    mol_to_align_i: int,
    ref_mol_i: int,
    condition: str,
    num_surf_points: int | None = 400,
    pharm_multi_vector: bool | None = False,
    probe_radius: float | None = 0.6,
    do_center: bool | None = True
    ):
    """
    Aligns two molecules from mol_block

    Arguments
    ---------
    molblocks_and_charges: List[Tuple(Mol Block, xtb charges))]
        Data structure of mol blocks and partial charges calculated with xtb
    mol_to_align_i : int
        Index on molblocks for molecule to align
    ref_mol_i: int
        Index on molblock for reference molecule for alignment
    num_surf_points : Optional[int] Number of surface points to sample.
        If None, the surface point cloud is not generated. More efficient if only doing volumentric.
    probe_radius : Optional[float] the radius of a probe atom to act as a "solvent accessible
        surface". Default is 1.2 if `None` is passed.
    partial_charges : Optional[np.ndarray] (N,) Partial charges for each atom.
        If `None` is passed and ESP surface is generated, it will default to MMFF94 partial charges.
    do_center : bool (default = True)
        THIS IS CRUCIAL
        Whether to initially align molecule centers together. For global optimizations, set to
        True. For scoring of current alignment or local alignment set to False.

    Returns
    -------
        mol_aligned: aligned rdkit.Chem.Mol
    """

    #Optimal parameters
    alpha = ALPHA(num_surf_points) # Fitted to probe_radius=1.2
    lam = 0.3 # Optimal lambda for probe_radius=1.2 -> ONLY TO BE USED FOR ESP ALIGNMENT

    # target natural product
    mol_to_align = rdkit.Chem.MolFromMolBlock(
        molblocks_and_charges[mol_to_align_i][0], removeHs=False
    )
    ref_mol = rdkit.Chem.MolFromMolBlock(molblocks_and_charges[ref_mol_i][0], removeHs=False)


    mol_to_align = Molecule(
        mol_to_align,
        num_surf_points = num_surf_points,
        partial_charges = np.array(molblocks_and_charges[mol_to_align_i][1]),
        probe_radius = probe_radius,
        pharm_multi_vector=True
    )

    ref_mol = Molecule(
        ref_mol,
        num_surf_points = num_surf_points,
        partial_charges = np.array(molblocks_and_charges[ref_mol_i][1]),
        probe_radius = probe_radius,
        pharm_multi_vector=True
    )

    fit_and_ref = MoleculePair(
        ref_mol,
        mol_to_align,
        do_center = do_center,
        num_surf_points = num_surf_points
    )

    if (condition == 'surface'):
        fit_and_ref.align_with_surf(
            alpha,
            num_repeats=5,
            trans_init=False,
            use_jax=False
        )
        mol_aligned = fit_and_ref.get_transformed_molecule(fit_and_ref.transform_surf)

    if (condition == 'esp' or condition == 'all'):
        start = time.time()
        fit_and_ref.align_with_esp(
            alpha,
            lam=lam,
            num_repeats=5,
            trans_init=False,
            use_jax=False,
            verbose=True,
            max_num_steps = 100,
        )

        mol_aligned = fit_and_ref.get_transformed_molecule(fit_and_ref.transform_esp)

        print(fit_and_ref.transform_esp)
        end = time.time()
        print("time")
        print(end-start)

    if (condition == 'pharm') and isinstance(pharm_multi_vector, bool):

        fit_and_ref.align_with_pharm(
            similarity='tanimoto',
            extended_points=False,
            only_extended=False,
            num_repeats=5,
            verbose=True,
            trans_init=False,
            use_jax=False
        )

        mol_aligned = fit_and_ref.get_transformed_molecule(fit_and_ref.transform_pharm)

    return mol_aligned.mol


def align_two_ref_mol_from_mol_charge(
    mol_to_align_mol: rdkit.Chem.Mol,
    # mol_to_align_charges: np.ndarray,
    ref_mol: rdkit.Chem.Mol,
    # ref_mol_charges: np.ndarray,
    condition: str,
    num_surf_points: int | None = 400,
    pharm_multi_vector: bool | None = False,
    probe_radius: float | None = 0.6,
    do_center: bool | None = False
    ):
    """
    Aligns two molecules given directly as rdkit Mol objects with charges already set.

    Arguments
    ---------
    mol_to_align_mol : rdkit.Chem.Mol
        Molecule to align; must already have partial charges set as an atom property.
    ref_mol : rdkit.Chem.Mol
        Reference molecule for alignment; must already have partial charges set.
    condition : str
        'surface', 'esp', 'pharm', or 'all'; which alignment(s) to compute.
    num_surf_points : Optional[int] (default=400)
        Number of surface points to sample. If None, the surface point cloud is not generated.
    pharm_multi_vector : Optional[bool] (default=False)
    probe_radius : Optional[float] (default=0.6)
        Radius of a probe atom to act as a "solvent accessible surface".
    do_center : Optional[bool] (default=False)
        Whether to initially align molecule centers together.

    Returns
    -------
    mol_aligned : aligned rdkit.Chem.Mol
    """

    #Optimal parameters
    alpha = ALPHA(num_surf_points) # Fitted to probe_radius=1.2
    lam = 0.3 # Optimal lambda for probe_radius=1.2 -> ONLY TO BE USED FOR ESP ALIGNMENT

    mol_to_align = Molecule(
        mol_to_align_mol,
        num_surf_points = num_surf_points,
        # partial_charges = mol_to_align_charges,
        probe_radius = probe_radius,
        pharm_multi_vector=True
    )

    ref_mol = Molecule(
        ref_mol,
        num_surf_points = num_surf_points,
        # partial_charges = ref_mol_charges,
        probe_radius = probe_radius,
        pharm_multi_vector=True
    )

    fit_and_ref = MoleculePair(
        ref_mol,
        mol_to_align,
        do_center = do_center,
        num_surf_points = num_surf_points
    )

    if (condition == 'surface'):
        fit_and_ref.align_with_surf(
            alpha,
            num_repeats=5,
            trans_init=False,
            use_jax=False
        )
        mol_aligned = fit_and_ref.get_transformed_molecule(fit_and_ref.transform_surf)
        se3_transform = fit_and_ref.transform_surf

    if (condition == 'esp' or condition == 'all'):
        start = time.time()
        fit_and_ref.align_with_esp(
            alpha,
            lam=lam,
            num_repeats=5,
            trans_init=False,
            use_jax=False,
            verbose=True,
            max_num_steps = 500,
        )

        mol_aligned = fit_and_ref.get_transformed_molecule(fit_and_ref.transform_esp)
        se3_transform = fit_and_ref.transform_esp
        print(se3_transform)
        end = time.time()
        print("time")
        print(end-start)

    if (condition == 'pharm') and isinstance(pharm_multi_vector, bool):

        fit_and_ref.align_with_pharm(
            similarity='tanimoto',
            extended_points=False,
            only_extended=False,
            num_repeats=5,
            verbose=True,
            trans_init=False,
            use_jax=False
        )

        mol_aligned = fit_and_ref.get_transformed_molecule(fit_and_ref.transform_pharm)
        se3_transform = fit_and_ref.transform_pharm

    return se3_transform, mol_aligned.mol

def align_model_states(
    model_outputs,
    D_theta,
    cdt_dict,
    params,
    num_repeats: int = 5,
    trans_init: bool = False,
    lr: float = 0.1,
    max_steps: int = 10,
    mode: str = 'so3',
    verbose: bool = True,
):
    """
    EDM version of align_model_states.

    Identical to the DDPM version except D_theta (the EDM denoiser output) is
    passed instead of x0_pred.  For EDM, D_theta IS the clean x0 prediction,
    so the surface point positions and ESP charges are read from:
        D_theta['x3_pos']   (vs DDPM x0_pred['x3_pos_0'])
        D_theta['x3_x']     (vs DDPM x0_pred['x3_x_0'])

    Arguments
    ---------
    model_outputs : dict
        All positional outputs keyed by modality; every 'pos' key is transformed
        under the recovered SE3/SO3 alignment.
    D_theta : dict
        EDM denoiser output from _inference_step_comp_edm. Keys include
        'x3_pos' (surface point positions) and 'x3_x' (ESP features).
    cdt_dict : dict
        Condition dict with 'surface' (ref surface points) and
        'electrostatics' (ref ESP values).
    params : dict
        Model params dict (used for x3 feature scaling).
    num_repeats : int
        Number of SE3 init restarts for the overlay optimiser.
    lr : float
        Learning rate for the overlay optimiser.
    max_steps : int
        Max optimisation steps.
    mode : str
        'so3'  - apply rotation only (strip translation after fitting).
        'se3'  - apply full SE3 transform.
    verbose : bool

    Returns
    -------
    model_outputs : dict  (positions transformed in-place)
    se3_transform : torch.Tensor (4, 4)
    """
    ref_mol_pos = cdt_dict['surface']
    ref_charges = cdt_dict['electrostatics']

    # For EDM, D_theta['x3_pos'] and D_theta['x3_x'] are the clean x0 predictions.
    fit_mol_pos = D_theta['x3_pos'].squeeze()
    fit_charges = D_theta['x3_x'].squeeze() / params['dataset']['x3']['scale_node_features']

    ref_points = torch.tensor(ref_mol_pos).to(torch.float32).to('cuda:0')
    ref_charges_t = torch.tensor(ref_charges).to(torch.float32).to('cuda:0')

    # slice off virtual node (index 0)
    fit_points = fit_mol_pos[1:].to('cuda:0').squeeze()
    fit_charges_t = fit_charges[1:].to('cuda:0').squeeze()

    lam = 0.3

    alpha = ALPHA(ref_points.size(0))

    _, se3_transform, score = optimize_ROCS_esp_overlay(
        ref_points=fit_points,
        fit_points=ref_points,
        ref_charges=fit_charges_t,
        fit_charges=ref_charges_t,
        alpha=alpha,
        lam=lam,
        num_repeats=num_repeats,
        trans_centers=None,
        lr=lr,
        max_num_steps=max_steps,
        verbose=verbose
    )

    print(f"Final ESP similarity score after alignment: {score.item():.3f}")

    if mode == 'so3':
        fit_points = apply_SO3_transform(
            points=fit_points.cpu(),
            SE3_transform=se3_transform,
        )
    else:
        fit_points = apply_SE3_transform(
            points=fit_points.cpu(),
            SE3_transform=se3_transform,
        )

    for key in model_outputs:
        for j in range(len(model_outputs[key])):
            if 'pos' in key:
                if mode == 'so3':
                    model_outputs[key][j] = apply_SO3_transform(
                        points=model_outputs[key][j],
                        SE3_transform=se3_transform,
                    )
                elif mode == 'se3':
                    model_outputs[key][j] = apply_SE3_transform(
                        points=model_outputs[key][j],
                        SE3_transform=se3_transform,
                    )

    return model_outputs, se3_transform

def _align_model_states(
                ref_mol_pos,
                ref_charges,
                fit_mol_pos,
                fit_charges,
                num_repeats: int = 5,
                trans_init: bool = False,
                lr: float = 0.1,
                max_steps: int = 10,
                mode='so3',
                verbose: bool = True,
                ):


    """
    Aligns a single fit point cloud to a reference point cloud via ESP overlay.

    Arguments
    ---------
    ref_mol_pos, fit_mol_pos : torch.Tensor
        Reference and fit surface point positions (virtual node at index 0, sliced off below).
    ref_charges, fit_charges : torch.Tensor
        Reference and fit ESP values, aligned with ref_mol_pos / fit_mol_pos.
    num_repeats : int (default=5)
        Number of random SE3 initializations for the overlay optimizer.
    trans_init : bool (default=False)
    lr : float (default=0.1)
        Learning rate for the overlay optimizer.
    max_steps : int (default=10)
        Maximum number of optimization steps.
    mode : str (default='so3')
        'so3' applies rotation only; anything else applies the full SE3 transform.
    verbose : bool (default=True)

    Returns
    -------
    torch.Tensor (4, 4) : the recovered SE3 transform (identity if the overlay score is too low).
    """
    ref_points = ref_mol_pos[1:].to('cuda:0')
    ref_charges = ref_charges[1:].to('cuda:0')

    fit_points = fit_mol_pos.squeeze()[1:].to('cuda:0')
    fit_charges = fit_charges.squeeze().squeeze()[1:].to('cuda:0')

    lam = 0.3

    _, se3_transform, score = optimize_ROCS_esp_overlay(
            ref_points=fit_points,
            fit_points=ref_points,
            ref_charges=fit_charges,
            fit_charges=ref_charges,
            alpha=ALPHA(fit_points.size(0)),
            lam=lam,
            num_repeats=num_repeats,
            trans_centers = None,
            lr=lr,
            max_num_steps=max_steps,
            verbose=False
        )

    if score <= 0.01:
        se3_transform = torch.eye(4)


    if mode == 'so3':
        ref_points = apply_SO3_transform(
                                        points = ref_points.cpu(),
                                        SE3_transform = se3_transform
                                        )
    else:
        ref_points = apply_SE3_transform(
                                        points = ref_points.cpu(),
                                        SE3_transform = se3_transform
                            )

    return se3_transform


def align_batchwise(pos, se3_transforms, mode='so3'):
    """
    Apply a different SE3/SO3 transform to each batch element's slice of a stacked tensor.

    Arguments
    ---------
    pos : torch.Tensor (len(se3_transforms) * w, ...)
        Positions stacked batchwise; split into len(se3_transforms) equal-sized slices.
    se3_transforms : list[torch.Tensor (4, 4)]
        One SE3 transform per batch element.
    mode : str (default='so3')
        'so3' applies rotation only; 'se3' applies the full SE3 transform.

    Returns
    -------
    torch.Tensor : pos with each batch slice transformed in-place.
    """
    w = pos.size(0) // len(se3_transforms)

    for i in range(len(se3_transforms)):
        if mode == 'se3':
            pos[i*w:(i+1)*w] = apply_SE3_transform(
                    points = pos[i*w:(i+1)*w],
                    SE3_transform = se3_transforms[i]
                    )
        elif mode == 'so3':
            pos[i*w:(i+1)*w] = apply_SO3_transform(
                        points = pos[i*w:(i+1)*w],
                        SE3_transform = se3_transforms[i]
                        )
    return pos

def unflatten_dict(d, cdtn_size, B,):
    """
    Unflatten a condition-batched dict into a list of per-batch-element dicts.

    Arguments
    ---------
    d : dict
        Tensors of shape (cdtn_size * B * A, ...).
    cdtn_size : int
        Number of conditions batched together.
    B : int
        Batch size.

    Returns
    -------
    list[dict] : length B, each dict holding tensors of shape (cdtn_size, A, ...).
    """
    out = []
    for b in range(B):
        out.append({})

    for k, v in d.items():
        # reshape from (C, B*A, ...) -> (C, B, A, ...)
        v = v.view(cdtn_size, int(v.size(0) / cdtn_size), -1)
        v = v.reshape(v.shape[0], B, int(v.shape[1] / B), -1)

        # split into list of length B
        vs = [v[:, b] for b in range(B)]

        # assign into dicts
        for b in range(B):
            out[b][k] = vs[b]

    return out


def unflatten_dict_values(d, cdtn_size, B,):
    """
    Same as unflatten_dict, but keeps one dict with a per-batch-element list per key
    instead of a list of per-batch-element dicts.

    Arguments
    ---------
    d : dict
        Tensors of shape (cdtn_size * B * A, ...).
    cdtn_size : int
        Number of conditions batched together.
    B : int
        Batch size.

    Returns
    -------
    dict : each key maps to a list of length B of tensors shaped (cdtn_size, A, ...).
    """
    out = {}
    for k, v in d.items():
        # reshape from (C, B*A, ...) -> (C, B, A, ...)
        v = v.view(cdtn_size, int(v.size(0) / cdtn_size), -1)
        v = v.reshape(v.shape[0], B, int(v.shape[1] / B), -1)

        # split into list of length B
        vs = [v[:, b] for b in range(B)]

        # assign into dicts
        out[k] = vs

    return out

def unflatten_tensor(v, cdtn_size, B,):
    """
    Same as unflatten_dict, but for a single tensor instead of a dict.

    Arguments
    ---------
    v : torch.Tensor
        Shape (cdtn_size * B * A, ...).
    cdtn_size : int
        Number of conditions batched together.
    B : int
        Batch size.

    Returns
    -------
    list[torch.Tensor] : length B, each shaped (cdtn_size, A, ...).
    """
    v = v.view(cdtn_size, int(v.size(0) / cdtn_size), -1)
    v = v.reshape(v.shape[0], B, int(v.shape[1] / B), -1)

    # split into list of length B
    vs = [v[:, b] for b in range(B)]

    return vs

def combine_list_of_dicts(xs):
    """
    Inverse of unflatten_dict: stack a list of per-batch-element dicts back into one dict.

    Arguments
    ---------
    xs : list[dict]
        Length B, each dict holding tensors of shape (C, A, ...).

    Returns
    -------
    dict : each key maps to a tensor of shape (C, B*A, ...).
    """
    B = len(xs)

    out = {}
    for k in xs[0]:
        # stack along new dim 1 -> shape (C, B, A, ...)
        stacked = torch.stack([x[k] for x in xs], dim=1)

        C, B, A = stacked.shape[:3]
        rest = stacked.shape[3:]

        # reshape to (C, B*A, ...)
        out[k] = stacked.reshape(C, B * A, *rest)

    return out


def center(mol,
           ref_center = None):
    """
    Recenter a molecule's conformer coordinates.

    Arguments
    ---------
    mol : rdkit.Chem.Mol
        Molecule with a conformer to recenter.
    ref_center : Optional[np.ndarray] (3,) (default=None)
        Center to subtract. If None, centers on the molecule's own coordinate mean.

    Returns
    -------
    rdkit.Chem.Mol : mol with recentered conformer coordinates.
    """
    mol_coordinates = np.array(mol.GetConformer().GetPositions())
    if ref_center is None:
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)
    else:
        mol_coordinates = mol_coordinates - ref_center

    mol = update_mol_coordinates(mol, mol_coordinates)

    return mol


def return_atom_condition_dict(
    inpaint_x1_pos: bool = False,
    inpaint_x1_x: bool = False,
    inpaint_x1_bonds: bool = False,
    inpaint_x1_formal_charge: bool = False,

    scaffold_conditioning: bool = False,

    stop_inpainting_at_time_x1_pos: float = 1.0,
    stop_inpainting_at_time_x1_x: float = 1.0,
    stop_inpainting_at_time_x1_bonds: float = 1.0,

    # these are the inpainting targets
    mol: Chem.Mol | None = None,
    ref_center: np.ndarray | None = None,
    atom_inds_to_inpaint: list[int] | None = None,
    atom_types: list[int] | None = None,
    atom_pos: np.ndarray | None = None,
    atom_formal_charges: np.ndarray | None = None,
    center_of_mass: np.ndarray = np.zeros(3),
):
    """
    Build an atom (x1) condition kwargs dict, centering mol on ref_center if provided.

    Arguments
    ---------
    inpaint_x1_pos, inpaint_x1_x, inpaint_x1_bonds, inpaint_x1_formal_charge : bool (default=False)
    scaffold_conditioning : bool (default=False)
    stop_inpainting_at_time_x1_pos, stop_inpainting_at_time_x1_x : float (default=1.0)
        Progress fraction at which to stop inpainting (1.0 = data; 0.0 = prior).
    stop_inpainting_at_time_x1_bonds : float (default=1.0)
    mol : Optional[Chem.Mol] (default=None)
        Target molecule; recentered on ref_center (or its own mean) if provided.
    ref_center : Optional[np.ndarray] (3,) (default=None)
    atom_inds_to_inpaint : Optional[list[int]] (default=None)
    atom_types : Optional[list[int]] (default=None)
    atom_pos : Optional[np.ndarray] (default=None)
    atom_formal_charges : Optional[np.ndarray] (default=None)
    center_of_mass : np.ndarray (3,) (default=np.zeros(3))

    Returns
    -------
    dict : kwargs suitable for gen_x1_edm_inpainting_dict / gen_x1_inpainting_dict.
    """
    if mol is not None:
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        if ref_center is None:
            mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)
        else:
            mol_coordinates = mol_coordinates - ref_center

        mol = update_mol_coordinates(mol, mol_coordinates)

    atom_conditions = {
        'inpaint_x1_pos': inpaint_x1_pos,
        'inpaint_x1_x': inpaint_x1_x,
        'inpaint_x1_bonds': inpaint_x1_bonds,
        'inpaint_x1_formal_charge': inpaint_x1_formal_charge,
        'scaffold_conditioning': scaffold_conditioning,
        'stop_inpainting_at_frac_x1_pos': stop_inpainting_at_time_x1_pos,
        'stop_inpainting_at_frac_x1_x': stop_inpainting_at_time_x1_x,
        'stop_inpainting_at_frac_x1_bonds': stop_inpainting_at_time_x1_bonds,
        'mol': mol,
        'atom_inds_to_inpaint': atom_inds_to_inpaint,
        'atom_types': atom_types,
        'atom_pos': atom_pos,
        'atom_formal_charges': atom_formal_charges,
        'center_of_mass': center_of_mass,
    }

    return atom_conditions


def return_condition_dict(mol,
                          charges,
                          params,
                          mode='all',
                          inds = None,
                          ref_center = None):
    """
    Build a surface/ESP/pharmacophore condition dict (plus a drawing-friendly variant)
    from a molecule and its partial charges.

    Arguments
    ---------
    mol : rdkit.Chem.Mol
        Molecule with a conformer; recentered on ref_center (or its own mean) before scoring.
    charges : np.ndarray
        Partial charges per atom, passed to Molecule for ESP surface generation.
    params : dict
        Model params dict (surface point count, probe radius, pharmacophore settings).
    mode : str (default='all')
        Which modalities to mark for inpainting: 'x2', 'x3', 'x4', 'x2_x4', or 'all'.
    inds : optional (default=None)
        If provided, subset the pharmacophore arrays to these indices.
    ref_center : Optional[np.ndarray] (3,) (default=None)

    Returns
    -------
    condition_dict : dict of surface/ESP/pharmacophore targets and inpaint_* toggles,
        suitable for gen_edm_inpainting_dict / gen_inpainting_dict.
    condition_dict_draw : dict
        Same underlying data, laid out for visualization.
    """
    mol_coordinates = np.array(mol.GetConformer().GetPositions())
    if ref_center is None:
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)
    else:
        mol_coordinates = mol_coordinates - ref_center

    mol = update_mol_coordinates(mol, mol_coordinates)
    molec = Molecule(
        mol,
        params['dataset']['x3']['num_points'],
        probe_radius=params['dataset']['probe_radius'],
        partial_charges=charges,
        pharm_multi_vector=params['dataset']['x4']['multivectors'],
    )

    # conditional targets

    surface = molec.surf_pos
    pharm_types, pharm_pos, pharm_direction = molec.pharm_types, molec.pharm_ancs, molec.pharm_vecs

    print(f"Number of pharmacophores: {len(pharm_types)}, Molecule atoms: {mol.GetNumAtoms()}")

    if inds is not None:
        pharm_types = pharm_types[inds]
        pharm_pos = pharm_pos[inds]
        pharm_direction = pharm_direction[inds]

    electrostatics = molec.surf_esp

    inpaint_x2_pos = False
    inpaint_x3_pos = False
    inpaint_x3_x = False
    inpaint_x4_pos = False
    inpaint_x4_direction = False
    inpaint_x4_type = False

    if mode == 'x2':
        inpaint_x2_pos = True
        inpaint_x3_pos = True
    elif mode == 'x3':
        inpaint_x3_pos = True
        inpaint_x3_x = True
    elif mode == 'x4':
        inpaint_x4_pos = True
        inpaint_x4_direction = True
        inpaint_x4_type = True
    elif mode == 'all':
        inpaint_x3_pos = True
        inpaint_x3_x = True
        inpaint_x4_pos = True
        inpaint_x4_direction = True
        inpaint_x4_type = True
    elif mode == 'x2_x4':
        inpaint_x2_pos = True
        inpaint_x3_pos = True
        inpaint_x4_pos = True
        inpaint_x4_direction = True
        inpaint_x4_type = True
    else:
        raise ValueError(f"{mode} is not a valid mode: either x2, x3, x4, x2_x4, or all")


    condition_dict = {
        'surface': surface,
        'pharm_types': pharm_types,
        'pharm_pos': pharm_pos,
        'pharm_direction': pharm_direction,
        'electrostatics': electrostatics,
        'inpaint_x2_pos': inpaint_x2_pos,
        'inpaint_x3_pos': inpaint_x3_pos,
        'inpaint_x3_x': inpaint_x3_x,
        'inpaint_x4_pos':inpaint_x4_pos,
        'inpaint_x4_direction':inpaint_x4_direction,
        'inpaint_x4_type': inpaint_x4_type,
    }

    condition_dict_draw = {
        'mol': mol,
        'pharm_types': pharm_types,
        'pharm_ancs': pharm_pos,
        'pharm_vecs': pharm_direction,
        'point_cloud': surface,
        'esp': electrostatics,
    }

    return condition_dict, condition_dict_draw


def condition_kwargs_from_profile(
    profile: InteractionProfile,
    condition_modalities='all',
    pharmacophore_conditioning: bool = False,
    inpaint: InpaintAdvancedOptions | None = None,
) -> dict:
    """
    Translate an InteractionProfile (+ optional InpaintAdvancedOptions) into the
    kwargs dict expected by gen_edm_inpainting_dict — the same surface/ESP/
    pharmacophore condition dict shape produced by return_condition_dict.

    This is a pure interface adapter: it does not change what gen_edm_inpainting_dict
    or generate_composition compute.

    Arguments
    ---------
    profile : InteractionProfile
    condition_modalities : see InteractionProfile.to_generate_kwargs.
    pharmacophore_conditioning : bool (default=False)
        Not derived from the profile (mirrors the `generate()` signature, where
        pharmacophore_conditioning is a separate argument from `condition`).
    inpaint : Optional[InpaintAdvancedOptions] (default=None)
        Overrides for inpaint flags / stop-time / add-noise knobs, merged the same
        way `generate()` merges them (non-default fields win, then resolved to
        hard defaults).

    Returns
    -------
    dict : kwargs suitable for gen_edm_inpainting_dict.
    """
    cond_kwargs = profile.to_generate_kwargs(condition_modalities)

    opts = InpaintAdvancedOptions(
        inpaint_x3_pos=cond_kwargs['inpaint_x3_pos'],
        inpaint_x3_x=cond_kwargs['inpaint_x3_x'],
        inpaint_x4_pos=cond_kwargs['inpaint_x4_pos'],
        inpaint_x4_direction=cond_kwargs['inpaint_x4_direction'],
        inpaint_x4_type=cond_kwargs['inpaint_x4_type'],
    )
    if inpaint is not None:
        opts = opts.merge_overrides(inpaint)
    opts = opts.resolved()

    surface = cond_kwargs['surface']
    electrostatics = cond_kwargs['electrostatics']
    pharm_types = cond_kwargs['pharm_types']
    pharm_pos = cond_kwargs['pharm_positions']
    pharm_direction = cond_kwargs['pharm_directions']

    return {
        'surface': np.zeros((0, 3)) if surface is None else surface,
        'electrostatics': np.zeros(0) if electrostatics is None else electrostatics,
        'pharm_types': np.zeros(0, dtype=int) if pharm_types is None else pharm_types,
        'pharm_pos': np.zeros((0, 3)) if pharm_pos is None else pharm_pos,
        'pharm_direction': np.zeros((0, 3)) if pharm_direction is None else pharm_direction,
        'center_of_mass': cond_kwargs['center_of_mass'],
        'pharmacophore_conditioning': pharmacophore_conditioning,
        'inpaint_x3_pos': opts.inpaint_x3_pos,
        'inpaint_x3_x': opts.inpaint_x3_x,
        'inpaint_x4_pos': opts.inpaint_x4_pos,
        'inpaint_x4_direction': opts.inpaint_x4_direction,
        'inpaint_x4_type': opts.inpaint_x4_type,
        'stop_inpainting_at_frac_x3': opts.stop_inpainting_at_time_x3,
        'add_noise_to_inpainted_x3_pos': opts.add_noise_to_inpainted_x3_pos,
        'add_noise_to_inpainted_x3_x': opts.add_noise_to_inpainted_x3_x,
        'stop_inpainting_at_frac_x4': opts.stop_inpainting_at_time_x4,
        'add_noise_to_inpainted_x4_pos': opts.add_noise_to_inpainted_x4_pos,
        'add_noise_to_inpainted_x4_direction': opts.add_noise_to_inpainted_x4_direction,
        'add_noise_to_inpainted_x4_type': opts.add_noise_to_inpainted_x4_type,
    }


def atom_condition_kwargs_from_profile(
    profile: InteractionProfile | ConditionAtoms,
    scaffold_conditioning: bool = False,
    inpaint: InpaintAdvancedOptions | None = None,
    center_of_mass: np.ndarray | None = np.zeros(3),
) -> dict:
    """
    Translate an InteractionProfile/ConditionAtoms (+ optional InpaintAdvancedOptions)
    into the kwargs dict expected by gen_x1_edm_inpainting_dict — the same shape
    produced by return_atom_condition_dict.

    Pure interface adapter: does not change what gen_x1_edm_inpainting_dict or
    generate_composition compute. Note gen_x1_edm_inpainting_dict has no exit-vector-
    aware bond inpainting (unlike the newer inference.sampler.generate), so
    ConditionAtoms.exit_vector_inds is not forwarded — carrying it through would not
    change behavior, since it isn't consumed downstream.

    Arguments
    ---------
    profile : InteractionProfile | ConditionAtoms
        An InteractionProfile with `condition_atoms` set (e.g. via
        `with_condition_atoms`/`from_condition_atoms`), or a bare ConditionAtoms.
    scaffold_conditioning : bool (default=False)
    inpaint : Optional[InpaintAdvancedOptions] (default=None)
        Overrides for x1 inpaint flags / stop-time knobs, merged and resolved the
        same way `generate()` does.
    center_of_mass : np.ndarray (3,) | None (default=np.zeros(3))
        Resolved coordinate-frame offset to center atom positions about, as
        produced by generate_composition's ``condition_center_of_mass``
        handling (``np.zeros(3)`` for ``'origin'``, ``None`` for ``'auto'`` --
        auto-derive from the scaffold COM inside gen_x1_edm_inpainting_dict --
        or an explicit length-3 array). Not derived from ``profile`` itself:
        ``InteractionProfile.to_generate_kwargs()['center_of_mass']`` is always
        ``np.zeros(3)`` and is intentionally ignored here.

    Returns
    -------
    dict : kwargs suitable for gen_x1_edm_inpainting_dict.
    """
    if isinstance(profile, ConditionAtoms):
        condition_atoms = profile
        mol = None
    else:
        cond_kwargs = profile.to_generate_kwargs()
        mol = cond_kwargs['mol']
        condition_atoms = profile.condition_atoms

    if condition_atoms is None:
        atom_inds_to_inpaint = None
        atom_types = None
        atom_pos = None
        atom_formal_charges = None
    else:
        atom_inds_to_inpaint = condition_atoms.inds
        atom_types = condition_atoms.types
        atom_pos = condition_atoms.pos
        atom_formal_charges = condition_atoms.formal_charges

    opts = InpaintAdvancedOptions()
    if inpaint is not None:
        opts = opts.merge_overrides(inpaint)
    opts = opts.resolved()

    return {
        'inpaint_x1_pos': opts.inpaint_x1_pos,
        'inpaint_x1_x': opts.inpaint_x1_x,
        'inpaint_x1_bonds': opts.inpaint_x1_bonds,
        'inpaint_x1_formal_charge': opts.inpaint_x1_formal_charge,
        'scaffold_conditioning': scaffold_conditioning,
        'stop_inpainting_at_frac_x1_pos': opts.stop_inpainting_at_time_x1_pos,
        'stop_inpainting_at_frac_x1_x': opts.stop_inpainting_at_time_x1_x,
        'stop_inpainting_at_frac_x1_bonds': opts.stop_inpainting_at_time_x1_bonds,
        'mol': mol,
        'atom_inds_to_inpaint': atom_inds_to_inpaint,
        'atom_types': atom_types,
        'atom_pos': atom_pos,
        'atom_formal_charges': atom_formal_charges,
        'center_of_mass': center_of_mass,
    }


def mol_to_gen_mol(mol):
    """
    Extract atomic numbers and coordinates from a molecule's conformer.

    Arguments
    ---------
    mol : rdkit.Chem.Mol
        Molecule with a conformer.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] : (atomic numbers, coordinates).
    """
    mol_coordinates = np.array(mol.GetConformer().GetPositions())
    mol_atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])

    return (mol_atomic_numbers,mol_coordinates)
