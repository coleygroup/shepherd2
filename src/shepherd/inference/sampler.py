"""Inference with ShEPhERD-2."""
from typing import Literal

import numpy as np
import torch
import torch_geometric  # noqa: F401
import torch_scatter  # noqa: F401

import pytorch_lightning as pl  # noqa: F401
from shepherd.lightning_module import LightningModule
from shepherd.interaction_profile import InteractionProfile, InpaintAdvancedOptions
from shepherd.generated_sample import GeneratedSample

from shepherd.inference.noise import (
    forward_trajectory_edm,
)
from shepherd.inference.edm_sampler import generate_edm_unconditional, generate_edm_conditional, get_edm_sigma_schedule
from shepherd.inference.steps import atomic_number_to_symbol_map


def _target_tensor_from_numpy(arr) -> torch.Tensor:
    """Copy numpy data into an owned float tensor (never shares caller memory)."""
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32).clone()


def _normalize_node_counts(
    name: str,
    value: int | list[int],
    batch_size: int,
    *,
    unconditional: bool,
) -> list[int]:
    """Validate a scalar/list node-count specification and return per-sample counts."""
    if isinstance(value, list):
        if len(value) != batch_size:
            raise ValueError(
                f"{name} as a list must have length == batch_size, got "
                f"len({name})={len(value)} and batch_size={batch_size}."
            )
        if not all(isinstance(n, int) and n > 0 for n in value):
            raise ValueError(f"All elements of {name} must be positive integers.")
        if unconditional:
            raise ValueError(
                f"{name} as a list[int] is not supported for unconditional generation."
            )
        return value
    return [value] * batch_size


def generate(
    model_pl: LightningModule,
    batch_size: int,

    N_x1: int | list[int],
    N_x4: int | list[int],

    condition: InteractionProfile | None = None,
    condition_modalities: str | set[str] = 'all',
    condition_center_of_mass: Literal['origin', 'auto'] | np.ndarray = 'origin',
    inpaint: InpaintAdvancedOptions | None = None,

    scaffold_conditioning: bool = False,
    pharmacophore_conditioning: bool = False,
    pharmacophore_prioritization: list[int] | None = None,

    num_steps: int = 400,
    early_stop_edm: int | float = 0.9,
    shepherd_pred: bool = True,

    # EDM sampler
    sigma_max: float | None = 3.0,
    sigma_min: float | None = 0.001,
    rho: float | None = 7.0,

    # parameters for EDM unconditional generation
    use_stochastic: bool = False,
    S_churn: float = 40.0,
    S_noise: float = 1.0,

    # not applicable for shepherd_pred=True
    use_2nd_order_correction: bool = False,

    # Other options for inference
    verbose: bool = True,
    store_trajectories: bool = False,
    store_trajectories_x0: bool = False,
    ) -> list[GeneratedSample]:
    """
    Runs inference of ShEPhERD to sample `batch_size` number of molecules.

    Mode is inferred from ``condition``: omit it for unconditional generation,
    or pass an :class:`~shepherd.interaction_profile.InteractionProfile` for
    conditional generation. Traditional conditional generation from a reference
    molecule::

        from shepherd.interaction_profile import extract_interaction_profile

        profile = extract_interaction_profile(mol)
        samples = generate(
            model_pl, batch_size=8,
            N_x1=profile.n_atoms, N_x4=profile.n_pharms,
            condition=profile,  # condition_modalities defaults to 'all'
        )

    Arguments
    ---------
    model_pl : PyTorch Lightning module.

    batch_size : int Number of molecules to sample in a single batch.

    N_x1 : int | list[int]
        Number of atoms to diffuse. Pass a list[int] of length batch_size.
        Only available during conditional generation (not unconditional).
        With scaffold conditioning, every count must be at least the number of shared scaffold atoms;
        extra atoms are freely diffused. Variable counts do not support x1 inpainting.
    N_x4 : int | list[int]
        Number of pharmacophores to diffuse. Pass a list[int] of length batch_size to
        use a different pharmacophore count per molecule in a conditional batch. If
        inpainting/conditioning, every count must be greater than or equal to len(pharm_types).
        Extra pharmacophores will be freely diffused by the model.

    condition : InteractionProfile (default = None)
        Interaction-profile target for conditional generation.
        - ``None`` (default): unconditional generation.
        - ``InteractionProfile``: conditional generation on the given profile.
            Use flags on ``condition_modalities`` to set which interactions to condition on.
            By default, we inpaint shape, ESP, and pharmacophores if they are present in the
            provided InteractionProfile. You may additionally condition on atoms if
            ``InteractionProfile.condition_atoms`` is set and ``scaffold_conditioning=True``
            is passed below.
    condition_modalities : str | set[str] (default = 'all')
        - ``'all'`` (default): inpaint-condition on all interactions.
        - ``'shape'``: inpaint-condition on shape.
        - ``'esp'``: inpaint-condition on ESP and shape.
        - ``'pharm'``: inpaint-condition on pharmacophores.
        - ``{'shape', 'pharm'}``: inpaint-condition on shape and pharmacophores but not ESP.
        Ignored if ``condition`` is None.
    condition_center_of_mass : {'origin', 'auto'} | np.ndarray (3,), default='origin'
        Coordinate-frame handling for the condition. ``'origin'`` uses the profile's
        coordinates as supplied. ``'auto'`` centers scaffold/pharmacophore conditioning
        on the corresponding condition coordinates and restores that offset on output.
        A length-3 array subtracts that explicit offset from all condition positions.
        Ignored if ``condition`` is None.
        We suggest using 'origin' for most cases.
    inpaint : InpaintAdvancedOptions (default = None)
        Optional overrides for
        - which modalities to inpaint (including atom pos, bonds, etc.)
        - granular control over when to stop inpainting for each modality.
          ``stop_inpainting_at_time_*`` uses the same convention as
          ``early_stop_edm``: ``1.0`` (default) is the data; ``0.0`` is the prior.

    scaffold_conditioning : bool (default = False)
        Enable scaffold-conditioned generation (distinct from inpainting).
        Requires ``condition.condition_atoms`` (types/positions, or ``mol`` plus atom indices)
        on the profile.
        When True:
        - Scaffold positions are fixed throughout denoising (unless released; see below)
        - COM is NOT removed from predicted noise
        - scaffold frame is controlled by ``condition_center_of_mass``
        Release scaffold atoms ``InpaintAdvancedOptions.stop_inpainting_at_time_x1_*`` < 1.0.
        Use ``condition_center_of_mass='auto'`` to center on the scaffold COM.
        We suggest using 'origin' for most cases.
    pharmacophore_conditioning : bool (default = False)
        Enable fixed pharmacophore-conditioned generation. Requires ``condition``.
        When True:
        - Pharmacophore positions are fixed throughout denoising
        - pharmacophore frame is controlled by ``condition_center_of_mass``
    pharmacophore_prioritization : Optional[list[int]] (default = None)
        Prioritizes pharmacophores between high and low priority via balance of fixed
        and inpainting. If None, all pharmacophores are treated equally.
        Ignored if `pharmacophore_conditioning` is False.
        1: high priority (conditional), 0: low priority (inpainting): ex) [1, 1, 1, 0, 0]
        The length of the list must be equal to the number of pharmacophores in the profile.

    num_steps : int (default = 400)
        Number of steps used for constructing the denoising noise schedule and sampling.
    early_stop_edm : int | float (default = 0.9)
        Number of EDM denoising steps to actually run. The full ``num_steps``
        schedule is still constructed; this only truncates the sampling loop.
        By default, we stop at 90% of the total number of steps for our default EDM parameters.
        - ``-1``: no early stop (run all ``num_steps``)
        - ``int``: run this many steps
        - ``float`` in ``[0, 1]``: run ``int(early_stop_edm * num_steps)`` steps

    sigma_max: float | None (default = 3.0)
        Controls the prior standard deviation.
    sigma_min: float | None (default = 0.001)
        Controls the minimum standard deviation for the nonise schedule.
    rho: float | None (default = 7.0)
        Controls the shape of the noise schedule.
        >1 denoises quickly with more timesteps concentrated close to sigma_min.

    # parameters for EDM unconditional generation
    use_stochastic: bool (default = False)
        Whether to use stochastic sampling for unconditional generation. Not necessary
        for the shepherd_pred=True (predict-renoise) sampler.
    S_churn: float (default = 40.0)
        Controls the strength of the churn noise for unconditional generation.
    S_noise: float (default = 1.0)
        Controls the strength of the noise for unconditional generation.

    use_2nd_order_correction: bool (default = False)
        Whether to use 2nd order correction for the EDM denoising process.
        This is not applicable to the shepherd_pred=True (predict-renoise) sampler.

    verbose : bool (default = True) Whether to print progress bar.
    store_trajectories : bool (default = False) Whether to store the trajectories.
    store_trajectories_x0 : bool (default = False) Whether to store the trajectories of the x0 predictions.

    Returns
    -------
    generated_structures : list[GeneratedSample]
        One GeneratedSample (a dict subclass with typed accessors and
        `.to_rdkit_mol()`/`.to_smiles()`/`.to_interaction_profile()` helpers) per
        molecule. The underlying dict is structured as:
        {
        'x1': {
            'atoms': np.ndarray (N_x1,) of ints for atomic numbers.
            'bonds': np.ndarray of bond types between every atom pair.
            'positions': np.ndarray (N_x1, 3) Coordinates of atoms.
        },
        'x3': {
            'charges': np.ndarray (75, 3) ESP at surface points.
            'positions': np.ndarray (75, 3) Coordinates of surface points.
        },
        'x4': {
            'types': np.ndarray (N_x4,) of ints for pharmacophore types.
            'positions': np.ndarray (N_x4, 3) Coordinates of pharmacophores.
            'directions': np.ndarray (N_x4, 3) Unit vectors of pharmacophores.
        },
        }
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    params = model_pl.params

    unconditional = condition is None

    if unconditional:
        for flag, name in (
            (scaffold_conditioning, "scaffold_conditioning"),
            (pharmacophore_conditioning, "pharmacophore_conditioning"),
        ):
            if flag:
                raise ValueError(
                    f"{name} requires `condition` (an InteractionProfile). "
                    "Omit condition only for unconditional generation."
                )
    else:
        cond_kwargs = condition.to_generate_kwargs(condition_modalities)
        surface = cond_kwargs.get('surface')
        electrostatics = cond_kwargs.get('electrostatics')
        pharm_types = cond_kwargs.get('pharm_types')
        pharm_pos = cond_kwargs.get('pharm_positions')
        pharm_direction = cond_kwargs.get('pharm_directions')

        if pharmacophore_conditioning and (pharm_pos is None or pharm_direction is None or pharm_types is None):
            raise ValueError(
                "pharm_positions, pharm_directions, and pharm_types must be provided on "
                "`condition` when using pharmacophore conditioning."
            )

        if isinstance(condition_center_of_mass, str):
            if condition_center_of_mass == 'origin':
                center_of_mass = np.zeros(3)
            elif condition_center_of_mass == 'auto':
                center_of_mass = None
            else:
                raise ValueError(
                    "condition_center_of_mass must be 'origin', 'auto', or a "
                    "length-3 array."
                )
        else:
            center_of_mass = np.asarray(condition_center_of_mass, dtype=float)
            if center_of_mass.shape != (3,):
                raise ValueError(
                    "condition_center_of_mass must be 'origin', 'auto', or a "
                    f"length-3 array; got shape {center_of_mass.shape}."
                )
        mol = cond_kwargs['mol']
        atom_inds_to_inpaint = cond_kwargs['atom_inds_to_inpaint']
        atom_types = cond_kwargs['atom_types']
        atom_pos = cond_kwargs['atom_pos']
        atom_formal_charges = cond_kwargs['atom_formal_charges']
        exit_vector_atom_inds = cond_kwargs['exit_vector_atom_inds']

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

        inpaint_x1_pos = opts.inpaint_x1_pos
        inpaint_x1_x = opts.inpaint_x1_x
        inpaint_x1_bonds = opts.inpaint_x1_bonds
        inpaint_x1_formal_charge = opts.inpaint_x1_formal_charge
        inpaint_x3_pos = opts.inpaint_x3_pos
        inpaint_x3_x = opts.inpaint_x3_x
        inpaint_x4_pos = opts.inpaint_x4_pos
        inpaint_x4_direction = opts.inpaint_x4_direction
        inpaint_x4_type = opts.inpaint_x4_type
        stop_inpainting_at_time_x1_pos = opts.stop_inpainting_at_time_x1_pos
        stop_inpainting_at_time_x1_x = opts.stop_inpainting_at_time_x1_x
        stop_inpainting_at_time_x1_bonds = opts.stop_inpainting_at_time_x1_bonds
        stop_inpainting_at_time_x3 = opts.stop_inpainting_at_time_x3
        add_noise_to_inpainted_x3_pos = opts.add_noise_to_inpainted_x3_pos
        add_noise_to_inpainted_x3_x = opts.add_noise_to_inpainted_x3_x
        stop_inpainting_at_time_x4 = opts.stop_inpainting_at_time_x4
        add_noise_to_inpainted_x4_pos = opts.add_noise_to_inpainted_x4_pos
        add_noise_to_inpainted_x4_direction = opts.add_noise_to_inpainted_x4_direction
        add_noise_to_inpainted_x4_type = opts.add_noise_to_inpainted_x4_type

        # use empty tensors since we always concat virtual-node concat despite not using them
        if surface is None:
            inpaint_x3_pos = False
            surface = np.zeros((0, 3))
        if electrostatics is None:
            inpaint_x3_x = False
            electrostatics = np.zeros(0)
        if pharm_pos is None:
            inpaint_x4_pos = False
            pharm_pos = np.zeros((0, 3))
        if pharm_direction is None:
            inpaint_x4_direction = False
            pharm_direction = np.zeros((0, 3))
        if pharm_types is None:
            inpaint_x4_type = False
            pharm_types = np.zeros(0, dtype=int)

        # Copy caller-owned index lists so inference never mutates them across reruns
        atom_inds_to_inpaint = [] if atom_inds_to_inpaint is None else list(atom_inds_to_inpaint)
        exit_vector_atom_inds = [] if exit_vector_atom_inds is None else list(exit_vector_atom_inds)

        # Record whether the caller supplied an explicit COM before it may be auto-derived below
        user_supplied_com = center_of_mass is not None

        if scaffold_conditioning and not params.get('scaffold_conditioning', False):
            raise ValueError(
                "scaffold_conditioning=True was specified, but the model was not trained "
                "with scaffold_conditioning=True. Either use a model trained with "
                "scaffold_conditioning=True, or use inpainting instead."
            )
        if scaffold_conditioning and ((atom_pos is None and atom_types is None) and (mol is None or not atom_inds_to_inpaint)):
            raise ValueError(
                "scaffold_conditioning requires `condition.condition_atoms` "
                "(types and positions, or mol plus atom indices)."
            )

    if isinstance(N_x1, np.ndarray):
        N_x1 = N_x1.tolist()

    N_x1_counts = _normalize_node_counts("N_x1", N_x1, batch_size, unconditional=unconditional)
    if isinstance(N_x1, list):
        if inpaint_x1_pos or inpaint_x1_x or inpaint_x1_bonds or inpaint_x1_formal_charge:
            raise ValueError(
                "N_x1 as a list[int] cannot be combined with x1 inpainting "
                "(inpaint_x1_pos, inpaint_x1_x, inpaint_x1_bonds, inpaint_x1_formal_charge)."
            )
        if not scaffold_conditioning and (
            (mol is not None and atom_inds_to_inpaint)
            or atom_pos is not None
            or atom_types is not None
            or atom_formal_charges is not None
        ):
            raise ValueError(
                "N_x1 as a list[int] cannot be combined with x1 conditioned atoms "
                "unless scaffold_conditioning=True."
            )

    if isinstance(N_x4, np.ndarray):
        N_x4 = N_x4.tolist()

    N_x4_counts = _normalize_node_counts("N_x4", N_x4, batch_size, unconditional=unconditional)

    if unconditional:
        return generate_edm_unconditional(
            model_pl, batch_size, N_x1, N_x4, num_steps,
            verbose=verbose,
            store_trajectories=store_trajectories,
            store_trajectories_x0=store_trajectories_x0,
            use_stochastic=use_stochastic,
            S_churn=S_churn,
            S_noise=S_noise,
            use_2nd_order_correction=use_2nd_order_correction,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
            shepherd_pred=shepherd_pred,
            early_stop_edm=early_stop_edm,
        )

    ####### Defining inpainting targets ########

    do_partial_pharm_inpainting = False
    assert len(pharm_direction) == len(pharm_pos) and len(pharm_pos) == len(pharm_types), \
        f"pharm_direction, pharm_pos, and pharm_types must have the same length, got {len(pharm_direction)}, {len(pharm_pos)}, and {len(pharm_types)}"
    x4_target_active = (pharmacophore_conditioning or inpaint_x4_pos or inpaint_x4_direction or inpaint_x4_type)

    required_x4_count = len(pharm_pos) if x4_target_active else 0
    n4_too_small = [n for n in N_x4_counts if n < required_x4_count]
    if n4_too_small:
        raise ValueError(
            "Every N_x4 count must be >= number of pharmacophores in the shared profile, "
            f"got N_x4={N_x4} and {required_x4_count} active pharmacophores."
        )
    do_partial_pharm_inpainting = (
        x4_target_active
        and any(n > required_x4_count for n in N_x4_counts)
    )

    expected_x3_count = params['dataset']['x3']['num_points']
    if inpaint_x3_pos and len(surface) != expected_x3_count:
        raise ValueError(
            "Surface point count must match the model's x3 node count for inpainting, "
            f"got {len(surface)} and expected {expected_x3_count}."
        )
    if inpaint_x3_x and len(electrostatics) != expected_x3_count:
        raise ValueError(
            "Electrostatic point count must match the model's x3 node count for inpainting, "
            f"got {len(electrostatics)} and expected {expected_x3_count}."
        )

    # bond types
    BOND_TYPES = params['dataset']['x1']['bond_types']
    BOND_TYPES_DICT = {b:BOND_TYPES.index(b) for b in BOND_TYPES}
    MAX_BOND_TYPES = len(BOND_TYPES_DICT)

    bond_inpaint_mask = None
    inpaint_x1_bond_edge_x = None
    if mol is not None and atom_inds_to_inpaint:
        if (inpaint_x1_pos or scaffold_conditioning) and atom_pos is None:
            atom_pos = mol.GetConformer().GetPositions()[np.array(atom_inds_to_inpaint)]
        if (inpaint_x1_x or scaffold_conditioning) and atom_types is None:
            atom_types = list(np.array([mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in atom_inds_to_inpaint]))

        if inpaint_x1_formal_charge and atom_formal_charges is None:
            atom_formal_charges = np.array([mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in atom_inds_to_inpaint])

        if inpaint_x1_bonds:
            _bond_adj = np.triu(1-np.diag(np.ones(N_x1, dtype = int)))
            _bond_edge_index = np.stack(_bond_adj.nonzero(), axis = 0)
            # _bond_mask = np.isin(_bond_edge_index, atom_inds_to_inpaint)
            if exit_vector_atom_inds:
                # This will inpaint any "bond" between atoms-to-inpaint with all other atoms,
                # but does not specify bonds with exit vector atoms and non-inpainted atoms.
                assert all(ind in atom_inds_to_inpaint for ind in exit_vector_atom_inds), \
                    "exit_vector_atom_inds must be a subset of atom_inds_to_inpaint"
                _index_inpaint_no_exit_vector = np.array([atom_inds_to_inpaint.index(idx) for idx in atom_inds_to_inpaint
                                    if idx not in exit_vector_atom_inds])
                _atom_mask_no_exit_vector = np.isin(_bond_edge_index,
                    np.arange(len(atom_inds_to_inpaint))[_index_inpaint_no_exit_vector]
                )
                bond_inpaint_inds = np.where(_atom_mask_no_exit_vector.any(axis=0))[0]
                bond_inpaint_mask = np.where(_atom_mask_no_exit_vector.any(axis=0), True, False)
            else:
                _atom_mask = np.isin(_bond_edge_index, np.arange(len(atom_inds_to_inpaint)))
                bond_inpaint_inds = np.where(_atom_mask.all(axis=0))[0]
                bond_inpaint_mask = np.where(_atom_mask.all(axis=0), True, False)

            bond_types = []
            for idx_1, idx_2 in _bond_edge_index[:,bond_inpaint_mask].T:
                if int(idx_1) > len(atom_inds_to_inpaint) - 1 or int(idx_2) > len(atom_inds_to_inpaint) - 1:
                    bond_types.append(BOND_TYPES_DICT[None])
                    continue
                idx_1_mapped = atom_inds_to_inpaint[int(idx_1)]
                idx_2_mapped = atom_inds_to_inpaint[int(idx_2)]
                bond = mol.GetBondBetweenAtoms(idx_1_mapped, idx_2_mapped)
                if bond is None:
                    bond_types.append(BOND_TYPES_DICT[None]) # non-bonded edge type; == 0
                else:
                    bond_type = BOND_TYPES_DICT[str(bond.GetBondType())]
                    bond_types.append(bond_type)

            # one-hot encoding of bond types
            inpaint_x1_bond_edge_x = np.zeros((_bond_edge_index.shape[1], MAX_BOND_TYPES))
            # just select the bonds that we care about
            inpaint_x1_bond_edge_x[bond_inpaint_inds, bond_types] = 1
            inpaint_x1_bond_edge_x = inpaint_x1_bond_edge_x * params['dataset']['x1']['scale_bond_features']
            inpaint_x1_bond_edge_x = torch.from_numpy(inpaint_x1_bond_edge_x.copy()).float()
            bond_inpaint_mask = torch.from_numpy(bond_inpaint_mask.copy()).bool()

    do_partial_atom_inpainting = False
    num_inpainted_atoms = None
    if atom_pos is not None:
        if any(n < len(atom_pos) for n in N_x1_counts):
            raise ValueError(
                "Every N_x1 count must be >= number of atoms in the shared target, "
                f"got N_x1={N_x1} and {len(atom_pos)} target atoms."
            )
        do_partial_atom_inpainting = any(n > len(atom_pos) for n in N_x1_counts)
        num_inpainted_atoms = len(atom_pos)

    if atom_types is not None:
        assert len(atom_types) == len(atom_pos), \
            f"Number of atom types in the target molecule ({len(atom_types)}) must be equal to the number of atom positions to inpaint ({len(atom_pos)})."
        atomic_number_to_symbol = atomic_number_to_symbol_map(params['dataset']['x1']['atom_types'])
        atom_types = [atomic_number_to_symbol[z] for z in atom_types]

    if atom_formal_charges is not None:
        if any(n < len(atom_formal_charges) for n in N_x1_counts):
            raise ValueError(
                "Every N_x1 count must be >= number of formal charges in the shared "
                f"target, got N_x1={N_x1} and {len(atom_formal_charges)} charges."
            )

    # num_pharm_cond: number of pharmacophores to fix via scaffold conditioning (priority=1)
    # num_pharm_inpaint: number of pharmacophores to use inpainting (priority=0)
    # is_mixed_pharm_mode: True when both conditioning and inpainting are active simultaneously
    num_pharm_cond = len(pharm_pos)  # default: all provided pharms are conditional
    num_pharm_inpaint = 0
    is_mixed_pharm_mode = False

    if pharmacophore_prioritization is None and pharmacophore_conditioning:
        pharmacophore_prioritization = np.ones(len(pharm_pos), dtype=int)
    elif pharmacophore_prioritization is not None and pharmacophore_conditioning:
        assert len(pharmacophore_prioritization) == len(pharm_pos), \
            f"pharmacophore_prioritization must be the same length as the number of conditional pharmacophores, got {len(pharmacophore_prioritization)} and {len(pharm_pos)}"
        assert all(p in [0, 1] for p in pharmacophore_prioritization), \
            "pharmacophore_prioritization must be a list of 0s and 1s"
        pharmacophore_prioritization = np.array(pharmacophore_prioritization)
        if all(pharmacophore_prioritization == 0):
            # All pharmacophores use inpainting; disable conditioning entirely
            pharmacophore_conditioning = False
            inpaint_x4_pos = True
            inpaint_x4_direction = True
            inpaint_x4_type = True
            num_pharm_cond = 0
            num_pharm_inpaint = len(pharm_pos)
        elif all(pharmacophore_prioritization == 1):
            # All pharmacophores are fixed conditional; disable inpainting
            pharmacophore_conditioning = True
            inpaint_x4_pos = False
            inpaint_x4_direction = False
            inpaint_x4_type = False
            num_pharm_cond = len(pharm_pos)
            num_pharm_inpaint = 0
        else:
            # Mixed: some pharmacophores are fixed (priority=1), others use inpainting (priority=0)
            # Reorder so conditional pharmacophores come first, inpainting ones after
            pharm_cond_inds = np.where(pharmacophore_prioritization == 1)[0]
            pharm_inpaint_inds = np.where(pharmacophore_prioritization == 0)[0]
            num_pharm_cond = len(pharm_cond_inds)
            num_pharm_inpaint = len(pharm_inpaint_inds)
            pharm_pos = np.concatenate([pharm_pos[pharm_cond_inds], pharm_pos[pharm_inpaint_inds]])
            pharm_direction = np.concatenate([pharm_direction[pharm_cond_inds], pharm_direction[pharm_inpaint_inds]])
            pharm_types = np.concatenate([pharm_types[pharm_cond_inds], pharm_types[pharm_inpaint_inds]])
            pharmacophore_conditioning = True
            inpaint_x4_pos = True
            inpaint_x4_direction = True
            inpaint_x4_type = True
            is_mixed_pharm_mode = True


    # Set scaffold atom positions/types to ground truth without any noise
    scaffold_com = None
    if (
        center_of_mass is None and
        scaffold_conditioning and
        num_inpainted_atoms is not None and
        atom_pos is not None and
        atom_types is not None
        ):
        scaffold_com = np.mean(atom_pos, axis = 0)

        # Disable inpainting when using scaffold conditioning
        inpaint_x1_pos = False
        inpaint_x1_x = False
        inpaint_x1_formal_charge = False
        inpaint_x1_bonds = False
        do_partial_atom_inpainting = False

    # If pharmacophore conditioning, we adjust the com
    if (
        pharmacophore_conditioning and
        pharm_pos is not None and
        pharm_types is not None and
        pharm_direction is not None
    ):
        # In mixed mode, inpainting flags stay active for the inpainting subset
        if not is_mixed_pharm_mode:
            inpaint_x4_pos = False
            inpaint_x4_direction = False
            inpaint_x4_type = False
        scaffold_com = pharm_pos.mean(axis = 0)

    # if pharmacophore and scaffold conditioning, we adjust the com to be the overall COM.
    if (
        pharmacophore_conditioning and
        scaffold_conditioning
    ):
        scaffold_com = np.concatenate([atom_pos, pharm_pos]).mean(axis = 0)

    if atom_pos is None:
        # initialize dummy atom positions (unused when N_x1 is a list, since x1 inpainting
        # and scaffold conditioning are asserted away in that case)
        _n_x1_scalar = N_x1[0] if isinstance(N_x1, list) else N_x1
        atom_pos = np.zeros((_n_x1_scalar, 3))
    if atom_types is None:
        # initialize dummy atom types
        _n_x1_scalar = N_x1[0] if isinstance(N_x1, list) else N_x1
        atom_types = ['C'] * _n_x1_scalar

    # centering about provided center of mass (of x1)
    if scaffold_com is not None and center_of_mass is None:
        # if we are using scaffold conditioning, then we need to center the x1 positions about the scaffold COM
        center_of_mass = scaffold_com
    elif scaffold_com is None and center_of_mass is None:
        center_of_mass = np.zeros(3)
    surface = surface - center_of_mass
    pharm_pos = pharm_pos - center_of_mass
    atom_pos = atom_pos - center_of_mass

    # When center_of_mass was auto-derived (user passed None), record the offset so we
    # can shift all output positions back to the user's input frame after sampling.
    recenter_offset = scaffold_com if (not user_supplied_com and scaffold_com is not None) else None

    # adding small noise to pharm_pos to avoid overlapping points (causes error when encoding clean structure)
    pharm_pos = pharm_pos + np.random.randn(*pharm_pos.shape) * 0.01

    if params['dataset'].get('scale_point_cloud', 1.0) != 1.0:
        atom_pos = atom_pos * params['dataset'].get('scale_point_cloud', 1.0)
        pharm_pos = pharm_pos * params['dataset'].get('scale_point_cloud', 1.0)
        surface = surface * params['dataset'].get('scale_point_cloud', 1.0)

    # accounting for virtual nodes
    surface = np.concatenate([np.array([[0.0, 0.0, 0.0]]), surface], axis = 0) # virtual node
    electrostatics = np.concatenate([np.array([0.0]), electrostatics], axis = 0) # virtual node
    pharm_types = pharm_types + 1 # accounting for virtual node as the zeroeth type
    pharm_types = np.concatenate([np.array([0]), pharm_types], axis = 0) # virtual node
    pharm_pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_pos], axis = 0) # virtual node
    pharm_direction = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_direction], axis = 0) # virtual node

    atom_pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), atom_pos], axis = 0)
    # atom types are converted to one-hot encoding
    atom_type_map = {atomic_symbol: i for i, atomic_symbol in enumerate(params['dataset']['x1']['atom_types'])}
    # virtual node type is 0 (from param file) so don't need to add +1 like in the pharm_types case
    atom_type_indices = np.array([atom_type_map[z] for z in atom_types])
    atom_type_indices = np.concatenate([np.array([0]), atom_type_indices], axis = 0) # virtual node

    # formal charges
    if atom_formal_charges is not None and (inpaint_x1_formal_charge or scaffold_conditioning):
        charge_type_map = {charge_type: i for i, charge_type in enumerate(params['dataset']['x1']['charge_types'])}
        charge_type_indices = np.array([charge_type_map[z] for z in atom_formal_charges]) + len(params['dataset']['x1']['atom_types'])

    num_x_types = len(params['dataset']['x1']['atom_types']) + len(params['dataset']['x1']['charge_types'])
    atom_types_one_hot = np.zeros((atom_type_indices.size, num_x_types))
    atom_types_one_hot[np.arange(atom_type_indices.size), atom_type_indices] = 1
    if atom_formal_charges is not None and inpaint_x1_formal_charge:
        atom_types_one_hot[np.arange(charge_type_indices.size) + 1, charge_type_indices] = 1 # +1 to skip the virtual node
    atom_types = atom_types_one_hot

    # one-hot encodings
    pharm_types_one_hot = np.zeros((pharm_types.size, params['dataset']['x4']['max_node_types']))
    pharm_types_one_hot[np.arange(pharm_types.size), pharm_types] = 1
    pharm_types = pharm_types_one_hot

    # scaling features
    electrostatics = electrostatics * params['dataset']['x3']['scale_node_features']
    pharm_types = pharm_types * params['dataset']['x4']['scale_node_features']
    pharm_direction = pharm_direction * params['dataset']['x4']['scale_vector_features']
    atom_types = atom_types * params['dataset']['x1']['scale_atom_features']

    # defining inpainting targets (owned copies; sampling must not mutate caller buffers)
    target_inpaint_x1_x = _target_tensor_from_numpy(atom_types)
    target_inpaint_x1_pos = _target_tensor_from_numpy(atom_pos)
    target_inpaint_x1_mask = torch.zeros(atom_pos.shape[0], dtype=torch.long)
    target_inpaint_x1_mask[0] = 1
    target_inpaint_x1_mask = target_inpaint_x1_mask == 0

    target_inpaint_x3_x = _target_tensor_from_numpy(electrostatics)
    target_inpaint_x3_pos = _target_tensor_from_numpy(surface)
    target_inpaint_x3_mask = torch.zeros(electrostatics.shape[0], dtype=torch.long)
    target_inpaint_x3_mask[0] = 1
    target_inpaint_x3_mask = target_inpaint_x3_mask == 0

    target_inpaint_x4_x = _target_tensor_from_numpy(pharm_types)
    target_inpaint_x4_pos = _target_tensor_from_numpy(pharm_pos)  # include VN
    target_inpaint_x4_direction = _target_tensor_from_numpy(pharm_direction)
    target_inpaint_x4_mask = torch.zeros(pharm_types.shape[0], dtype = torch.long)
    target_inpaint_x4_mask[0] = 1
    target_inpaint_x4_mask = target_inpaint_x4_mask == 0

    # Centroid of fixed (conditional) pharmacophores in the internal centered frame.
    fixed_pharm_com_internal = None
    if pharmacophore_conditioning and num_pharm_cond > 0:
        fixed_pharm_com_internal = (
            target_inpaint_x4_pos[1:num_pharm_cond + 1].mean(dim=0).cpu().numpy()
        )

    # In mixed mode, the inpainting trajectory covers only the inpainting-subset pharmacophores
    # (indices num_pharm_cond+1 due to virtual node at index 0).
    # In pure inpainting mode, the trajectory covers all pharmacophores (including VN at index 0).
    if is_mixed_pharm_mode:
        _x4_traj_pos = target_inpaint_x4_pos[num_pharm_cond + 1:]
        _x4_traj_direction = target_inpaint_x4_direction[num_pharm_cond + 1:]
        _x4_traj_x = target_inpaint_x4_x[num_pharm_cond + 1:]
        _x4_traj_mask = torch.ones(num_pharm_inpaint, dtype=torch.bool)
    else:
        _x4_traj_pos = target_inpaint_x4_pos
        _x4_traj_direction = target_inpaint_x4_direction
        _x4_traj_x = target_inpaint_x4_x
        _x4_traj_mask = target_inpaint_x4_mask

    ####################################

    # Progress fractions (1.0 = data / end of denoising; 0.0 = prior / start).
    _stop_frac_x1_pos = stop_inpainting_at_time_x1_pos
    _stop_frac_x1_x = stop_inpainting_at_time_x1_x
    _stop_frac_x1_bonds = stop_inpainting_at_time_x1_bonds
    _stop_frac_x3 = stop_inpainting_at_time_x3
    _stop_frac_x4 = stop_inpainting_at_time_x4

    # EDM conditional/inpainting path: build EDM trajectories and call generate_edm_conditional
    edm_params = dict(params.get('edm', {}))
    if sigma_max is not None:
        edm_params['sigma_max'] = sigma_max
    if sigma_min is not None:
        edm_params['sigma_min'] = sigma_min
    if rho is not None:
        edm_params['rho'] = rho
    sigma_steps = get_edm_sigma_schedule(num_steps, edm_params)

    stop_inpainting_at_step_x1_pos = int(num_steps * _stop_frac_x1_pos)
    stop_inpainting_at_step_x1_x = int(num_steps * _stop_frac_x1_x)
    stop_inpainting_at_step_x1_bonds = int(num_steps * _stop_frac_x1_bonds)
    stop_inpainting_at_step_x3 = int(num_steps * _stop_frac_x3)
    stop_inpainting_at_step_x4 = int(num_steps * _stop_frac_x4)

    scaffold_release_enabled = (
        scaffold_conditioning
        and (_stop_frac_x1_pos < 1.0 or _stop_frac_x1_x < 1.0 or _stop_frac_x1_bonds < 1.0)
    )
    reference_com_internal = None
    stop_scaffold_at_step = num_steps
    if scaffold_release_enabled:
        stop_scaffold_at_step = int(num_steps * min(
            _stop_frac_x1_pos, _stop_frac_x1_x, _stop_frac_x1_bonds,
        ))
        if user_supplied_com:
            # Input frame was fixed by center_of_mass at setup; release scaffold only.
            reference_com_internal = None
        elif scaffold_conditioning and pharmacophore_conditioning:
            if fixed_pharm_com_internal is None:
                raise ValueError(
                    "Combined scaffold and pharmacophore conditioning requires at least "
                    "one fixed pharmacophore to release scaffold conditioning via "
                    "stop_inpainting_at_time_x1_*."
                )
            reference_com_internal = fixed_pharm_com_internal
        else:
            raise ValueError(
                "center_of_mass must be supplied when stop_inpainting_at_time_x1_* < 1.0 "
                "with scaffold_conditioning alone (without pharmacophore conditioning)."
            )

    x1_pos_inpainting_trajectory_edm = None
    x1_x_inpainting_trajectory_edm = None
    x1_bond_edge_x_inpainting_trajectory_edm = None
    x3_pos_inpainting_trajectory_edm = None
    x3_x_inpainting_trajectory_edm = None
    x4_pos_inpainting_trajectory_edm = None
    x4_direction_inpainting_trajectory_edm = None
    x4_x_inpainting_trajectory_edm = None

    if inpaint_x1_pos:
        x1_pos_inpainting_trajectory_edm = forward_trajectory_edm(
            target_inpaint_x1_pos, sigma_steps,
            remove_COM_from_noise=True, mask=target_inpaint_x1_mask,
            deterministic=False, batch_size=batch_size,
        )
    if inpaint_x1_x or inpaint_x1_formal_charge:
        x1_x_inpainting_trajectory_edm = forward_trajectory_edm(
            target_inpaint_x1_x, sigma_steps,
            remove_COM_from_noise=False, mask=target_inpaint_x1_mask,
            deterministic=False, batch_size=batch_size,
        )
    if inpaint_x1_bonds:
        x1_bond_edge_x_inpainting_trajectory_edm = forward_trajectory_edm(
            inpaint_x1_bond_edge_x[bond_inpaint_mask], sigma_steps,
        )
    if inpaint_x3_pos:
        x3_pos_inpainting_trajectory_edm = forward_trajectory_edm(
            target_inpaint_x3_pos, sigma_steps,
            remove_COM_from_noise=False, mask=target_inpaint_x3_mask,
            deterministic=False, batch_size=batch_size,
        )
    if inpaint_x3_x:
        x3_x_inpainting_trajectory_edm = forward_trajectory_edm(
            target_inpaint_x3_x, sigma_steps,
            remove_COM_from_noise=False, mask=target_inpaint_x3_mask,
            deterministic=False, batch_size=batch_size,
        )
    if inpaint_x4_type:
        x4_x_inpainting_trajectory_edm = forward_trajectory_edm(
            _x4_traj_x, sigma_steps,
            remove_COM_from_noise=False, mask=_x4_traj_mask,
            deterministic=False, batch_size=batch_size,
        )
    if inpaint_x4_pos:
        x4_pos_inpainting_trajectory_edm = forward_trajectory_edm(
            _x4_traj_pos, sigma_steps,
            remove_COM_from_noise=False, mask=_x4_traj_mask,
            deterministic=False, batch_size=batch_size,
        )
    if inpaint_x4_direction:
        x4_direction_inpainting_trajectory_edm = forward_trajectory_edm(
            _x4_traj_direction, sigma_steps,
            remove_COM_from_noise=False, mask=_x4_traj_mask,
            deterministic=False, batch_size=batch_size,
        )

    num_inpainted_formal_charges = None
    if inpaint_x1_formal_charge:
        num_inpainted_formal_charges = len(atom_formal_charges)

    edm_inpainting_dict = {
        'inpaint_x1_pos': inpaint_x1_pos,
        'inpaint_x1_x': inpaint_x1_x,
        'inpaint_x1_bonds': inpaint_x1_bonds,
        'inpaint_x1_formal_charge': inpaint_x1_formal_charge,
        'inpaint_x3_pos': inpaint_x3_pos,
        'inpaint_x3_x': inpaint_x3_x,
        'inpaint_x4_pos': inpaint_x4_pos,
        'inpaint_x4_direction': inpaint_x4_direction,
        'inpaint_x4_type': inpaint_x4_type,
        'x1_pos_inpainting_trajectory_edm': x1_pos_inpainting_trajectory_edm,
        'x1_x_inpainting_trajectory_edm': x1_x_inpainting_trajectory_edm,
        'x1_bond_edge_x_inpainting_trajectory_edm': x1_bond_edge_x_inpainting_trajectory_edm,
        'x3_pos_inpainting_trajectory_edm': x3_pos_inpainting_trajectory_edm,
        'x3_x_inpainting_trajectory_edm': x3_x_inpainting_trajectory_edm,
        'x4_pos_inpainting_trajectory_edm': x4_pos_inpainting_trajectory_edm,
        'x4_direction_inpainting_trajectory_edm': x4_direction_inpainting_trajectory_edm,
        'x4_x_inpainting_trajectory_edm': x4_x_inpainting_trajectory_edm,
        'batch_x3_x4_inpainting_trajectories': True,
        'stop_inpainting_at_step_x1_pos': stop_inpainting_at_step_x1_pos,
        'stop_inpainting_at_step_x1_x': stop_inpainting_at_step_x1_x,
        'stop_inpainting_at_step_x1_bonds': stop_inpainting_at_step_x1_bonds,
        'stop_inpainting_at_step_x3': stop_inpainting_at_step_x3,
        'stop_inpainting_at_step_x4': stop_inpainting_at_step_x4,
        'add_noise_to_inpainted_x3_pos': add_noise_to_inpainted_x3_pos,
        'add_noise_to_inpainted_x3_x': add_noise_to_inpainted_x3_x,
        'add_noise_to_inpainted_x4_pos': add_noise_to_inpainted_x4_pos,
        'add_noise_to_inpainted_x4_direction': add_noise_to_inpainted_x4_direction,
        'add_noise_to_inpainted_x4_type': add_noise_to_inpainted_x4_type,
        'do_partial_pharm_inpainting': do_partial_pharm_inpainting,
        'do_partial_atom_inpainting': do_partial_atom_inpainting,
        'num_inpainted_atoms': num_inpainted_atoms,
        'num_inpainted_formal_charges': num_inpainted_formal_charges,
        'is_mixed_pharm_mode': is_mixed_pharm_mode,
        'bond_inpaint_mask': bond_inpaint_mask,
        'scaffold_conditioning': scaffold_conditioning,
        'pharmacophore_conditioning': pharmacophore_conditioning,
        'num_pharm_cond': num_pharm_cond,
        'num_pharm_inpaint': num_pharm_inpaint,
        'target_inpaint_x1_pos': target_inpaint_x1_pos,
        'target_inpaint_x1_x': target_inpaint_x1_x,
        'target_inpaint_x4_pos': target_inpaint_x4_pos,
        'target_inpaint_x4_direction': target_inpaint_x4_direction,
        'target_inpaint_x4_x': target_inpaint_x4_x,
        'scaffold_release_enabled': scaffold_release_enabled,
        'stop_scaffold_at_step': stop_scaffold_at_step,
        'reference_com_internal': reference_com_internal,
    }
    return generate_edm_conditional(
        model_pl, batch_size, N_x1, N_x4, num_steps, edm_inpainting_dict,
        verbose=verbose,
        store_trajectories=store_trajectories,
        store_trajectories_x0=store_trajectories_x0,
        use_stochastic=use_stochastic,
        S_churn=S_churn,
        S_noise=S_noise,
        use_2nd_order_correction=use_2nd_order_correction,
        shepherd_pred=shepherd_pred,
        early_stop_edm=early_stop_edm,
        recenter_offset=recenter_offset,
    )
