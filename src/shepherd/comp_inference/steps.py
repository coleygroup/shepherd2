"""Compositional inference step with ShEPhERD-2."""
import torch
import torch_scatter

from shepherd.inference.edm_sampler import (
    _prepare_edm_model_input,
    _apply_edm_inpainting_replacement,
    _SIGMA_EPS,
)
import shepherd.comp_inference.utils as utils


def _get_model_input_states_edm(
    step_idx: int,
    batch_size: int,
    # current states
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    x1_batch: torch.Tensor,
    bond_edge_index_x1: torch.Tensor,
    virtual_node_mask_x1: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_x_t: torch.Tensor,
    x2_batch: torch.Tensor,
    virtual_node_mask_x2: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    x3_batch: torch.Tensor,
    virtual_node_mask_x3: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    x4_batch: torch.Tensor,
    virtual_node_mask_x4: torch.Tensor,
    # EDM inpainting dicts
    edm_inpainting_dict: dict | None = None,
    atom_edm_inpainting_dict: dict | None = None,
    # feature dims for _apply_edm_inpainting_replacement
    num_atom_types: int = 0,
    num_x_types: int = 0,
    num_pharm_types: int = 0,
    MAX_BOND_TYPES: int = 0,
    # ESP alignment
    alignment=None,
    mode: str = 'so3',
) -> dict:
    """
    Apply EDM step-indexed inpainting replacement for one condition and return
    the resulting state dict.

    Arguments
    ---------
    step_idx : int
        Current EDM denoising step index (0-based).
    batch_size : int
    x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, bond_edge_index_x1 : torch.Tensor
    virtual_node_mask_x1 : torch.Tensor
        Current x1 (atom) state.
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2 : torch.Tensor
        Current x2 (surface) state.
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3 : torch.Tensor
        Current x3 (ESP) state.
    x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4 : torch.Tensor
        Current x4 (pharmacophore) state.
    edm_inpainting_dict : Optional[dict] (default=None)
        Surface/ESP/pharmacophore inpainting dict, as built by gen_edm_inpainting_dict.
    atom_edm_inpainting_dict : Optional[dict] (default=None)
        Atom (x1) inpainting dict, as built by gen_x1_edm_inpainting_dict.
    num_atom_types, num_x_types, num_pharm_types, MAX_BOND_TYPES : int (default=0)
        Feature dims forwarded to _apply_edm_inpainting_replacement.
    alignment : optional (default=None)
        SE3 alignment applied to inpainted spatial coordinates, if provided.
    mode : str (default='so3')
        Alignment mode, forwarded to utils.align_batchwise.

    Returns
    -------
    dict : the state dict (x1/x2/x3/x4 tensors) after inpainting replacement and alignment.
    """
    # apply atom inpainting (x1 modality) if provided
    if atom_edm_inpainting_dict is not None:
        (x1_pos_t, x1_x_t, x1_bond_edge_x_t,
         x2_pos_t, x3_pos_t, x3_x_t,
         x4_pos_t, x4_direction_t, x4_x_t) = _apply_edm_inpainting_replacement(
            step_idx, atom_edm_inpainting_dict,
            x1_pos_t, x1_x_t, x1_bond_edge_x_t,
            x2_pos_t, x3_pos_t, x3_x_t,
            x4_pos_t, x4_direction_t, x4_x_t,
            batch_size,
            virtual_node_mask_x2, virtual_node_mask_x3, virtual_node_mask_x4,
            num_atom_types, num_x_types, num_pharm_types, MAX_BOND_TYPES,
        )

    # apply surface/pharm inpainting (x2/x3/x4 modalities) if provided
    if edm_inpainting_dict is not None:
        (x1_pos_t, x1_x_t, x1_bond_edge_x_t,
         x2_pos_t, x3_pos_t, x3_x_t,
         x4_pos_t, x4_direction_t, x4_x_t) = _apply_edm_inpainting_replacement(
            step_idx, edm_inpainting_dict,
            x1_pos_t, x1_x_t, x1_bond_edge_x_t,
            x2_pos_t, x3_pos_t, x3_x_t,
            x4_pos_t, x4_direction_t, x4_x_t,
            batch_size,
            virtual_node_mask_x2, virtual_node_mask_x3, virtual_node_mask_x4,
            num_atom_types, num_x_types, num_pharm_types, MAX_BOND_TYPES,
        )

    # apply ESP alignment to inpainted spatial coordinates if available
    if alignment is not None and edm_inpainting_dict is not None:
        stop_x2 = edm_inpainting_dict.get('stop_inpainting_at_step_x2', 0)
        stop_x3 = edm_inpainting_dict.get('stop_inpainting_at_step_x3', 0)
        stop_x4 = edm_inpainting_dict.get('stop_inpainting_at_step_x4', 0)
        if edm_inpainting_dict.get('inpaint_x2_pos', False) and step_idx < stop_x2:
            x2_pos_t = utils.align_batchwise(x2_pos_t, alignment, mode=mode)
        if edm_inpainting_dict.get('inpaint_x3_pos', False) and step_idx < stop_x3:
            x3_pos_t = utils.align_batchwise(x3_pos_t, alignment, mode=mode)
        if edm_inpainting_dict.get('inpaint_x4_pos', False) and step_idx < stop_x4:
            x4_pos_t = utils.align_batchwise(x4_pos_t, alignment, mode=mode)

    return {
        'x1_pos_t': x1_pos_t,
        'x1_x_t': x1_x_t,
        'x1_bond_edge_x_t': x1_bond_edge_x_t,
        'x1_batch': x1_batch,
        'bond_edge_index_x1': bond_edge_index_x1,
        'virtual_node_mask_x1': virtual_node_mask_x1,
        'x2_pos_t': x2_pos_t,
        'x2_x_t': x2_x_t,
        'x2_batch': x2_batch,
        'virtual_node_mask_x2': virtual_node_mask_x2,
        'x3_pos_t': x3_pos_t,
        'x3_x_t': x3_x_t,
        'x3_batch': x3_batch,
        'virtual_node_mask_x3': virtual_node_mask_x3,
        'x4_pos_t': x4_pos_t,
        'x4_direction_t': x4_direction_t,
        'x4_x_t': x4_x_t,
        'x4_batch': x4_batch,
        'virtual_node_mask_x4': virtual_node_mask_x4,
    }


def _inference_step_comp_edm(
    model_pl,
    params: dict,
    step_idx: int,
    sigma_cur: float,
    batch_size: int,
    # current states
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    x1_batch: torch.Tensor,
    bond_edge_index_x1: torch.Tensor,
    virtual_node_mask_x1: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_x_t: torch.Tensor,
    x2_batch: torch.Tensor,
    virtual_node_mask_x2: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    x3_batch: torch.Tensor,
    virtual_node_mask_x3: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    x4_batch: torch.Tensor,
    virtual_node_mask_x4: torch.Tensor,
    composition_mode: str,
    # per-condition EDM inpainting dicts (first may be None = unconditional)
    edm_inpainting_dict_arr: list,
    # atom EDM inpainting dict (applied to every condition)
    atom_edm_inpainting_dict: dict | None = None,
    # scaffold / pharmacophore masks for EDM (optional)
    x1_is_diffused_atom: torch.Tensor | None = None,
    x4_is_diffused_pharm: torch.Tensor | None = None,
    scaffold_task: torch.Tensor | None = None,
    # composition
    weights_conditions: torch.Tensor | None = None,
    include_x0_pred: bool = False,
    # per-condition ESP alignments (list of per-batch SE3 lists, one per ipt_dict, or None)
    alignments=None,
    mode: str = 'so3',
) -> dict:
    """
    One EDM compositional inference step.

    For each entry in edm_inpainting_dict_arr, applies the corresponding
    step-indexed inpainting replacement to get the per-condition noisy state,
    then batches all conditions together and runs the EDM model once.  The
    D_theta (denoiser) outputs are weight-combined across conditions and
    returned for use in the caller's ODE step.

    Arguments
    ---------
    model_pl : LightningModule
    params : dict
    step_idx : int
    sigma_cur : float
        Current EDM noise level.
    batch_size : int
    x1_pos_t, x1_x_t, x1_bond_edge_x_t, x1_batch, bond_edge_index_x1 : torch.Tensor
    virtual_node_mask_x1 : torch.Tensor
    x2_pos_t, x2_x_t, x2_batch, virtual_node_mask_x2 : torch.Tensor
    x3_pos_t, x3_x_t, x3_batch, virtual_node_mask_x3 : torch.Tensor
    x4_pos_t, x4_direction_t, x4_x_t, x4_batch, virtual_node_mask_x4 : torch.Tensor
        Current, shared (not yet per-condition) state.
    composition_mode : str
        "default" or "conditional"; see generate_composition.
    edm_inpainting_dict_arr : list[Optional[dict]]
        Per-condition EDM inpainting dicts (first entry may be None = unconditional).
    atom_edm_inpainting_dict : Optional[dict] (default=None)
        Atom inpainting dict applied to every condition.
    x1_is_diffused_atom, x4_is_diffused_pharm, scaffold_task : Optional[torch.Tensor] (default=None)
        Scaffold / pharmacophore diffusion masks.
    weights_conditions : Optional[torch.Tensor] (default=None)
        Per-condition composition weights.
    include_x0_pred : bool (default=False)
    alignments : optional (default=None)
        Per-condition ESP alignments (list of per-batch SE3 lists, one per ipt_dict, or None).
    mode : str (default='so3')

    Returns
    -------
    dict with keys:
        'D_theta': dict of weight-combined denoiser outputs (x1_pos, x1_x, ...),
        'x0_pred' (only when include_x0_pred=True): same dict as D_theta
            (for EDM, D_theta IS the x0 prediction).
        'states_unconditional': state dict for the first (unconditional) condition.
    """
    num_atom_types = len(params['dataset']['x1']['atom_types'])
    num_x_types = num_atom_types + len(params['dataset']['x1']['charge_types'])
    num_pharm_types = params['dataset']['x4']['max_node_types']
    MAX_BOND_TYPES = len(params['dataset']['x1']['bond_types'])

    cdtn_size = len(edm_inpainting_dict_arr)

    states_list = []
    N_x1_plus_virtual = int(x1_pos_t.size(0) / batch_size)

    for ind, ipt_dict in enumerate(edm_inpainting_dict_arr):
        states_i = _get_model_input_states_edm(
            step_idx=step_idx,
            batch_size=batch_size,
            x1_pos_t=x1_pos_t,
            x1_x_t=x1_x_t,
            x1_bond_edge_x_t=x1_bond_edge_x_t,
            x1_batch=x1_batch,
            bond_edge_index_x1=bond_edge_index_x1,
            virtual_node_mask_x1=virtual_node_mask_x1,
            x2_pos_t=x2_pos_t,
            x2_x_t=x2_x_t,
            x2_batch=x2_batch,
            virtual_node_mask_x2=virtual_node_mask_x2,
            x3_pos_t=x3_pos_t,
            x3_x_t=x3_x_t,
            x3_batch=x3_batch,
            virtual_node_mask_x3=virtual_node_mask_x3,
            x4_pos_t=x4_pos_t,
            x4_direction_t=x4_direction_t,
            x4_x_t=x4_x_t,
            x4_batch=x4_batch,
            virtual_node_mask_x4=virtual_node_mask_x4,
            edm_inpainting_dict=ipt_dict,
            atom_edm_inpainting_dict=atom_edm_inpainting_dict,
            num_atom_types=num_atom_types,
            num_x_types=num_x_types,
            num_pharm_types=num_pharm_types,
            MAX_BOND_TYPES=MAX_BOND_TYPES,
            alignment=alignments[ind] if alignments is not None else None,
            mode=mode,
        )

        # offset batch indices for this condition
        for key in ['x1_batch', 'x2_batch', 'x3_batch', 'x4_batch']:
            states_i[key] = states_i[key].clone() + ind * batch_size
        states_i['bond_edge_index_x1'] = (
            states_i['bond_edge_index_x1'].clone() + ind * batch_size * N_x1_plus_virtual
        )

        if ind == 0:
            states_unconditional = {k: v.clone() for k, v in states_i.items()}

        states_list.append(states_i)

    # batch all conditions
    batched = {}
    for k in states_list[0]:
        dim = 1 if k == 'bond_edge_index_x1' else 0
        batched[k] = torch.cat([d[k] for d in states_list], dim=dim)

    virtual_node_mask_x1_batched = batched['virtual_node_mask_x1']
    virtual_node_mask_x2_batched = batched['virtual_node_mask_x2']
    virtual_node_mask_x3_batched = batched['virtual_node_mask_x3']
    virtual_node_mask_x4_batched = batched['virtual_node_mask_x4']

    # build per-condition scaffold / diffused masks if needed
    if x1_is_diffused_atom is not None:
        x1_is_diffused_batched = x1_is_diffused_atom.repeat(cdtn_size)
    else:
        x1_is_diffused_batched = ~virtual_node_mask_x1_batched

    if x4_is_diffused_pharm is not None:
        x4_is_diffused_batched = x4_is_diffused_pharm.repeat(cdtn_size)
    else:
        x4_is_diffused_batched = ~virtual_node_mask_x4_batched

    if scaffold_task is not None:
        scaffold_task_batched = scaffold_task.repeat(cdtn_size)
    else:
        scaffold_task_batched = torch.zeros((batch_size * cdtn_size,), dtype=torch.bool)

    # build EDM model input dict (sigma replaces timestep)
    input_dict = _prepare_edm_model_input(
        device=model_pl.device,
        dtype=torch.float32,
        batch_size=batch_size * cdtn_size,
        sigma=sigma_cur,
        x1_pos_t=batched['x1_pos_t'],
        x1_x_t=batched['x1_x_t'],
        x1_batch=batched['x1_batch'],
        x1_bond_edge_x_t=batched['x1_bond_edge_x_t'],
        x1_bond_edge_index=batched['bond_edge_index_x1'],
        x1_virtual_node_mask=virtual_node_mask_x1_batched,
        x2_pos_t=batched['x2_pos_t'],
        x2_x_t=batched['x2_x_t'],
        x2_batch=batched['x2_batch'],
        x2_virtual_node_mask=virtual_node_mask_x2_batched,
        x3_pos_t=batched['x3_pos_t'],
        x3_x_t=batched['x3_x_t'],
        x3_batch=batched['x3_batch'],
        x3_virtual_node_mask=virtual_node_mask_x3_batched,
        x4_pos_t=batched['x4_pos_t'],
        x4_direction_t=batched['x4_direction_t'],
        x4_x_t=batched['x4_x_t'],
        x4_batch=batched['x4_batch'],
        x4_virtual_node_mask=virtual_node_mask_x4_batched,
        x1_is_diffused_atom=x1_is_diffused_batched,
        x4_is_diffused_pharm=x4_is_diffused_batched,
        scaffold_task=scaffold_task_batched,
    )

    with torch.no_grad():
        _, output_dict = model_pl.model.forward(input_dict)

    # extract D_theta outputs (predicted clean x0) for each modality
    x1_pos_D = output_dict['x1']['decoder']['denoiser']['pos_out'].detach().cpu()
    x1_x_D = output_dict['x1']['decoder']['denoiser']['x_out'].detach().cpu()
    x1_bond_edge_x_D = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'].detach().cpu()

    x1_pos_D[virtual_node_mask_x1_batched, :] = 0.0
    x1_x_D[virtual_node_mask_x1_batched, :] = 0.0

    # zero scaffold atoms in D_theta
    if x1_is_diffused_atom is not None:
        x1_is_diffused_batched_cpu = x1_is_diffused_batched.cpu()
        x1_pos_D[~x1_is_diffused_batched_cpu] = batched['x1_pos_t'][~x1_is_diffused_batched_cpu]
        x1_x_D[~x1_is_diffused_batched_cpu] = batched['x1_x_t'][~x1_is_diffused_batched_cpu]

    x2_pos_out = output_dict['x2']['decoder']['denoiser']['pos_out']
    if x2_pos_out is not None:
        x2_pos_D = x2_pos_out.detach().cpu()
        x2_pos_D[virtual_node_mask_x2_batched, :] = 0.0
    else:
        x2_pos_D = batched['x2_pos_t']

    x3_pos_out = output_dict['x3']['decoder']['denoiser']['pos_out']
    x3_x_out = output_dict['x3']['decoder']['denoiser']['x_out']
    if x3_pos_out is not None:
        x3_pos_D = x3_pos_out.detach().cpu()
        x3_pos_D[virtual_node_mask_x3_batched, :] = 0.0
        x3_x_D = x3_x_out.detach().cpu().squeeze()
        x3_x_D[virtual_node_mask_x3_batched] = 0.0
    else:
        x3_pos_D = batched['x3_pos_t']
        x3_x_D = batched['x3_x_t']

    x4_pos_out = output_dict['x4']['decoder']['denoiser']['pos_out']
    x4_direction_out = output_dict['x4']['decoder']['denoiser']['direction_out']
    x4_x_out = output_dict['x4']['decoder']['denoiser']['x_out']
    if x4_x_out is not None:
        x4_pos_D = x4_pos_out.detach().cpu()
        x4_pos_D[virtual_node_mask_x4_batched, :] = 0.0
        x4_direction_D = x4_direction_out.detach().cpu()
        x4_direction_D[virtual_node_mask_x4_batched, :] = 0.0
        x4_x_D = x4_x_out.detach().cpu().squeeze()
        x4_x_D[virtual_node_mask_x4_batched] = 0.0
        # clamp D_theta for conditioned pharmacophores
        if x4_is_diffused_pharm is not None:
            x4_is_diffused_batched_cpu = x4_is_diffused_batched.cpu()
            x4_pos_D[~x4_is_diffused_batched_cpu] = batched['x4_pos_t'][~x4_is_diffused_batched_cpu]
            x4_direction_D[~x4_is_diffused_batched_cpu] = (
                batched['x4_direction_t'][~x4_is_diffused_batched_cpu]
            )
            x4_x_D[~x4_is_diffused_batched_cpu] = batched['x4_x_t'][~x4_is_diffused_batched_cpu]
    else:
        x4_pos_D = batched['x4_pos_t']
        x4_direction_D = batched['x4_direction_t']
        x4_x_D = batched['x4_x_t']

    # weight-combine D_theta outputs across conditions
    D_theta_raw = {
        'x1_pos': x1_pos_D,
        'x1_x': x1_x_D,
        'x1_bond_edge_x': x1_bond_edge_x_D,
        'x2_pos': x2_pos_D,
        'x3_pos': x3_pos_D,
        'x3_x': x3_x_D,
        'x4_pos': x4_pos_D,
        'x4_direction': x4_direction_D,
        'x4_x': x4_x_D,
    }

    # derive per-condition pharmacophore counts from the inpainting dicts
    n_pharms_per_cond = []
    for ipt_dict in edm_inpainting_dict_arr:
        if (ipt_dict is not None
                and ipt_dict.get('do_partial_pharm_inpainting', False)
                and ipt_dict.get('target_inpaint_x4_pos') is not None):
            # target_inpaint_x4_pos has virtual node prepended → shape[0] - 1 = real pharm count
            n_pharms_per_cond.append(ipt_dict['target_inpaint_x4_pos'].shape[0] - 1)
        else:
            n_pharms_per_cond.append(None)

    _X4_KEYS = {'x4_pos', 'x4_direction', 'x4_x'}
    _need_x4_fix_inds = [
        n is None and weights_conditions[c] < 0
        for c, n in enumerate(n_pharms_per_cond)
    ]
    _need_x4_fix = any(_need_x4_fix_inds)

    D_theta_combined = {}
    for key, tensor in D_theta_raw.items():
        # reshape: (cdtn_size * nodes_per_condition, ...) -> (cdtn_size, nodes_per_condition, ...)
        n_total = tensor.size(0)
        n_per_cond = n_total // cdtn_size
        out_reshaped = tensor.view(cdtn_size, n_per_cond, -1)
        if key in _X4_KEYS and _need_x4_fix and composition_mode == 'conditional':
            eff_w = torch.ones_like(weights_conditions)
            eff_w[torch.tensor(_need_x4_fix_inds)] = 0.0
            D_theta_combined[key] = (out_reshaped * eff_w[:, None, None]).sum(dim=0).squeeze(-1)
        else:
            D_theta_combined[key] = (
                (out_reshaped * weights_conditions[:, None, None]).sum(dim=0).squeeze(-1)
            )

    del output_dict
    del input_dict

    result = {
        'D_theta': D_theta_combined,
        'states_unconditional': states_unconditional,
    }
    if include_x0_pred:
        # for EDM, D_theta is the x0 prediction
        result['x0_pred'] = D_theta_combined

    return result


def _edm_ode_step_from_D_theta(
    sigma_cur: float,
    sigma_next: float,
    # current noisy states (single-condition, not batched)
    x1_pos_t: torch.Tensor,
    x1_x_t: torch.Tensor,
    x1_bond_edge_x_t: torch.Tensor,
    virtual_node_mask_x1: torch.Tensor,
    x1_batch: torch.Tensor,
    x2_pos_t: torch.Tensor,
    x2_x_t: torch.Tensor,
    virtual_node_mask_x2: torch.Tensor,
    x3_pos_t: torch.Tensor,
    x3_x_t: torch.Tensor,
    virtual_node_mask_x3: torch.Tensor,
    x4_pos_t: torch.Tensor,
    x4_direction_t: torch.Tensor,
    x4_x_t: torch.Tensor,
    virtual_node_mask_x4: torch.Tensor,
    # weight-combined D_theta from _inference_step_comp_edm
    D_theta: dict,
    x1_is_diffused_atom: torch.Tensor | None = None,
    x4_is_diffused_pharm: torch.Tensor | None = None,
    scaffold_task_any: bool = False,
    shepherd_pred: bool = False,
) -> dict:
    """
    EDM step given pre-computed weight-combined D_theta.

    ODE mode (shepherd_pred=False):
        d = (x_t - D_theta) / sigma_cur
        x_next = x_t + (sigma_next - sigma_cur) * d

    Stochastic mode (shepherd_pred=True):
        x_next = D_theta + sigma_next * noise

    Arguments
    ---------
    sigma_cur, sigma_next : float
    x1_pos_t, x1_x_t, x1_bond_edge_x_t, virtual_node_mask_x1, x1_batch : torch.Tensor
    x2_pos_t, x2_x_t, virtual_node_mask_x2 : torch.Tensor
    x3_pos_t, x3_x_t, virtual_node_mask_x3 : torch.Tensor
    x4_pos_t, x4_direction_t, x4_x_t, virtual_node_mask_x4 : torch.Tensor
        Current noisy state (single-condition, not batched).
    D_theta : dict
        Weight-combined denoiser outputs from _inference_step_comp_edm.
    x1_is_diffused_atom, x4_is_diffused_pharm : Optional[torch.Tensor] (default=None)
    scaffold_task_any : bool (default=False)
    shepherd_pred : bool (default=False)
        If True, use the stochastic ShEPhERD-style update; else the standard ODE step.

    Returns
    -------
    dict : the next state (x1/x2/x3/x4 tensors) after one EDM step.
    """
    sigma_safe = max(float(sigma_cur), _SIGMA_EPS)
    d_sigma = float(sigma_next) - float(sigma_cur)
    sigma_next_f = float(sigma_next)

    x1_pos_D = D_theta['x1_pos']
    x1_x_D = D_theta['x1_x']
    x1_bond_edge_x_D = D_theta['x1_bond_edge_x']

    if shepherd_pred:
        x1_pos_epsilon = torch.randn_like(x1_pos_D)
        if not scaffold_task_any:
            x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(
                x1_pos_epsilon[~virtual_node_mask_x1],
                x1_batch[~virtual_node_mask_x1],
                dim=0,
            )[x1_batch]
        x1_x_epsilon = torch.randn_like(x1_x_D)
        x1_bond_edge_x_epsilon = torch.randn_like(x1_bond_edge_x_D)
        if x1_is_diffused_atom is not None:
            x1_pos_epsilon[~x1_is_diffused_atom] = 0.0
            x1_x_epsilon[~x1_is_diffused_atom] = 0.0
        x1_pos_next = x1_pos_D + sigma_next_f * x1_pos_epsilon
        x1_x_next = x1_x_D + sigma_next_f * x1_x_epsilon
        x1_bond_edge_x_next = x1_bond_edge_x_D + sigma_next_f * x1_bond_edge_x_epsilon
        if not scaffold_task_any:
            x1_pos_next = x1_pos_next - torch_scatter.scatter_mean(
                x1_pos_next[~virtual_node_mask_x1],
                x1_batch[~virtual_node_mask_x1],
                dim=0,
            )[x1_batch]
    else:
        x1_pos_d = (x1_pos_t - x1_pos_D) / sigma_safe
        x1_x_d = (x1_x_t - x1_x_D) / sigma_safe
        x1_bond_edge_x_d = (x1_bond_edge_x_t - x1_bond_edge_x_D) / sigma_safe
        x1_pos_next = x1_pos_t + d_sigma * x1_pos_d
        if not scaffold_task_any:
            x1_pos_next = x1_pos_next - torch_scatter.scatter_mean(
                x1_pos_next[~virtual_node_mask_x1],
                x1_batch[~virtual_node_mask_x1],
                dim=0,
            )[x1_batch]
        x1_x_next = x1_x_t + d_sigma * x1_x_d
        x1_bond_edge_x_next = x1_bond_edge_x_t + d_sigma * x1_bond_edge_x_d

    x1_pos_next[virtual_node_mask_x1, :] = 0.0
    x1_x_next[virtual_node_mask_x1, :] = 0.0

    x2_pos_D = D_theta['x2_pos']
    if shepherd_pred:
        x2_pos_next = x2_pos_D + sigma_next_f * torch.randn_like(x2_pos_D)
    else:
        x2_pos_d = (x2_pos_t - x2_pos_D) / sigma_safe
        x2_pos_next = x2_pos_t + d_sigma * x2_pos_d
    x2_pos_next[virtual_node_mask_x2, :] = 0.0

    x3_pos_D = D_theta['x3_pos']
    x3_x_D = D_theta['x3_x']
    if shepherd_pred:
        x3_pos_next = x3_pos_D + sigma_next_f * torch.randn_like(x3_pos_D)
        x3_x_next = x3_x_D + sigma_next_f * torch.randn_like(x3_x_D)
    else:
        x3_pos_d = (x3_pos_t - x3_pos_D) / sigma_safe
        x3_x_d = (x3_x_t - x3_x_D) / sigma_safe
        x3_pos_next = x3_pos_t + d_sigma * x3_pos_d
        x3_x_next = x3_x_t + d_sigma * x3_x_d
    x3_pos_next[virtual_node_mask_x3, :] = 0.0
    x3_x_next[virtual_node_mask_x3] = 0.0

    x4_pos_D = D_theta['x4_pos']
    x4_direction_D = D_theta['x4_direction']
    x4_x_D = D_theta['x4_x']
    if shepherd_pred:
        x4_pos_epsilon = torch.randn_like(x4_pos_D)
        x4_direction_epsilon = torch.randn_like(x4_direction_D)
        x4_x_epsilon = torch.randn_like(x4_x_D)
        if x4_is_diffused_pharm is not None:
            x4_pos_epsilon[~x4_is_diffused_pharm] = 0.0
            x4_direction_epsilon[~x4_is_diffused_pharm] = 0.0
            x4_x_epsilon[~x4_is_diffused_pharm] = 0.0
        x4_pos_next = x4_pos_D + sigma_next_f * x4_pos_epsilon
        x4_direction_next = x4_direction_D + sigma_next_f * x4_direction_epsilon
        x4_x_next = x4_x_D + sigma_next_f * x4_x_epsilon
    else:
        x4_pos_d = (x4_pos_t - x4_pos_D) / sigma_safe
        x4_direction_d = (x4_direction_t - x4_direction_D) / sigma_safe
        x4_x_d = (x4_x_t - x4_x_D) / sigma_safe
        x4_pos_next = x4_pos_t + d_sigma * x4_pos_d
        x4_direction_next = x4_direction_t + d_sigma * x4_direction_d
        x4_x_next = x4_x_t + d_sigma * x4_x_d
    x4_pos_next = x4_pos_next.clamp(-50.0, 50.0)
    x4_pos_next[virtual_node_mask_x4, :] = 0.0
    x4_direction_next[virtual_node_mask_x4, :] = 0.0
    x4_x_next[virtual_node_mask_x4] = 0.0

    return {
        'x1_pos_t_1': x1_pos_next,
        'x1_x_t_1': x1_x_next,
        'x1_bond_edge_x_t_1': x1_bond_edge_x_next,
        'x2_pos_t_1': x2_pos_next,
        'x2_x_t_1': x2_x_t,  # x2_x not diffused
        'x3_pos_t_1': x3_pos_next,
        'x3_x_t_1': x3_x_next,
        'x4_pos_t_1': x4_pos_next,
        'x4_direction_t_1': x4_direction_next,
        'x4_x_t_1': x4_x_next,
    }
