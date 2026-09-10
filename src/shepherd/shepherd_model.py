"""
Thin wrapper for the lightning module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    from shepherd.interaction_profile import (
        ConditionAtoms,
        InpaintAdvancedOptions,
        InteractionProfile,
    )
    from shepherd_score.container import Molecule
    from shepherd.generated_sample import GeneratedSample


class ShepherdModel:
    """Wraps a loaded `LightningModule` so that users can call
        - `model.generate(...)`
        - `model.generate_composition(...)`

    All LightningModule attributes are available (`.params`, `.model`, `.device`, `.eval()`, etc.)
    """

    def __init__(self, lightning_module, *, checkpoint_path: str | None = None):
        self.lightning_module = lightning_module
        self._checkpoint_path = checkpoint_path

    def __getattr__(self, name):
        return getattr(self.lightning_module, name)

    def generate(
        self,
        batch_size: int,
        N_x1: int | list[int],
        N_x4: int | list[int],
        condition: InteractionProfile | None = None,
        condition_modalities: str | set[str] = "all",
        condition_center_of_mass: Literal["origin", "auto"] | np.ndarray = "origin",
        inpaint: InpaintAdvancedOptions | None = None,
        scaffold_conditioning: bool = False,
        pharmacophore_conditioning: bool = False,
        pharmacophore_prioritization: list[int] | None = None,
        num_steps: int = 400,
        early_stop_edm: int | float = 0.9,
        shepherd_pred: bool = True,
        sigma_max: float | None = 3.0,
        sigma_min: float | None = 0.001,
        rho: float | None = 7.0,
        use_stochastic: bool = False,
        S_churn: float = 40.0,
        S_noise: float = 1.0,
        use_2nd_order_correction: bool = False,
        verbose: bool = True,
        store_trajectories: bool = False,
        store_trajectories_x0: bool = False,
    ) -> list[GeneratedSample]:
        """
        Sample molecules. See :func:`shepherd.inference.sampler.generate`.

        Omit ``condition`` for unconditional generation. For traditional
        conditional generation from a reference molecule::

            from shepherd.interaction_profile import extract_interaction_profile

            profile = extract_interaction_profile(mol)
            samples = model.generate(
                batch_size=48, N_x1=profile.n_atoms, N_x4=profile.n_pharms,
                condition=profile,
            )

        To scaffold-condition on a subset of atoms *together with* the
        surface/ESP/pharmacophore conditioning above, extract with
        ``condition_atom_inds=...`` and pass ``scaffold_conditioning=True``::

            profile = extract_interaction_profile(
                mol, condition_atom_inds=scaffold_atom_inds,
            )
            samples = model.generate(
                batch_size=48, N_x1=profile.n_atoms, N_x4=profile.n_pharms,
                condition=profile, scaffold_conditioning=True,
            )

        ``InteractionProfile.from_condition_atoms(mol, inds=...)`` builds a
        profile with *only* the atom subset (no surface/ESP/pharmacophores) —
        use it only when there is no other conditioning target.

        Advanced inpainting (x1 atom/bond inpainting, stop times, add-noise)
        is controlled by ``inpaint=InpaintAdvancedOptions(...)``.

        Arguments
        ---------
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
            Coordinate-frame handling for the condition. ``'origin'`` uses profile
            coordinates as supplied; ``'auto'`` centers scaffold/pharmacophore
            conditioning and restores the offset on output; a length-3 array subtracts
            that explicit offset from all condition positions.
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
        pharmacophore_prioritization : list[int] | None (default = None)
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
            Whether to use stochastic sampling for unconditional generation.
        S_churn: float (default = 40.0)
            Controls the strength of the churn noise for unconditional generation.
        S_noise: float (default = 1.0)
            Controls the strength of the noise for unconditional generation.

        use_2nd_order_correction: bool (default = False)
            Whether to use 2nd order correction for the EDM denoising process.
            This is not applicable to the shepherd_pred=True (predict-renoise) sampler.

        # property conditioning
        do_property_cfg : bool (default = False) Whether to use property conditioning.
        cfg_weight : float (default = 0.0) Classifier-free guidance scale ``w`` for property conditioning.
            Sampling blends denoiser outputs as ``(1 + w) * f_cond - w * f_uncond`` per step.
            Set ``w = 0`` for no guidance.
        property_dict : dict | None (default = None) Dictionary containing the property values.
            Keys are the property names and values are the property values.
            If None, ``global_props`` is all zeros and ``global_props_mask`` is all False (no property conditioning).
            {
                'single_point_property_name': property_value,
                'range_property_name': (property_value_min, property_value_max),
            }
            If a single-point property is not provided, it will be set to None.
            If a range property is not provided, it will be set to (None, None).
            Only properties whose names appear in this dict (before normalization) get
            ``global_props_mask`` True; others stay unconditional with neutral stored values.

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
        from shepherd.inference.sampler import generate
        self.lightning_module.freeze()
        try:
            generated_structures = generate(
                self.lightning_module,
                batch_size=batch_size,
                N_x1=N_x1,
                N_x4=N_x4,
                condition=condition,
                condition_modalities=condition_modalities,
                condition_center_of_mass=condition_center_of_mass,
                inpaint=inpaint,
                scaffold_conditioning=scaffold_conditioning,
                pharmacophore_conditioning=pharmacophore_conditioning,
                pharmacophore_prioritization=pharmacophore_prioritization,
                num_steps=num_steps,
                early_stop_edm=early_stop_edm,
                shepherd_pred=shepherd_pred,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                rho=rho,
                use_stochastic=use_stochastic,
                S_churn=S_churn,
                S_noise=S_noise,
                use_2nd_order_correction=use_2nd_order_correction,
                verbose=verbose,
                store_trajectories=store_trajectories,
                store_trajectories_x0=store_trajectories_x0,
            )
        finally:
            self.lightning_module.unfreeze()
        return generated_structures

    def generate_distributed(
        self,
        batch_size: int,
        N_x1: int | list[int],
        N_x4: int | list[int],
        condition: InteractionProfile | None = None,
        condition_modalities: str | set[str] = "all",
        condition_center_of_mass: Literal["origin", "auto"] | np.ndarray = "origin",
        inpaint: InpaintAdvancedOptions | None = None,
        scaffold_conditioning: bool = False,
        pharmacophore_conditioning: bool = False,
        pharmacophore_prioritization: list[int] | None = None,
        num_steps: int = 400,
        early_stop_edm: int | float = 0.9,
        shepherd_pred: bool = True,
        sigma_max: float | None = 3.0,
        sigma_min: float | None = 0.001,
        rho: float | None = 7.0,
        use_stochastic: bool = False,
        S_churn: float = 40.0,
        S_noise: float = 1.0,
        use_2nd_order_correction: bool = False,
        verbose: bool = True,
        *,
        devices: int | Sequence[int] | None = None,
        seed: int | None = None,
        checkpoint_path: str | None = None,
    ) -> list[GeneratedSample]:
        """Generate independent sample shards concurrently on multiple GPUs.

        ``batch_size`` is the total number of samples across all selected GPUs. A
        sequence-valued ``N_x1`` or ``N_x4`` is sliced consistently with the sample
        shards; all other generation arguments are copied to every worker.

        Each worker loads its own checkpoint replica. Models returned by
        :func:`shepherd.load_model` remember that checkpoint automatically. Pass
        ``checkpoint_path`` only for a manually constructed ``ShepherdModel`` or to
        intentionally generate from a different checkpoint than the loaded instance.

        Parameters
        ----------
        devices : int | sequence[int] | None
            Number of visible CUDA devices to use, explicit local CUDA indices, or
            ``None`` to use every visible GPU.
        seed : int | None
            Base seed for worker random streams. Defaults to ``torch.initial_seed()``.
        checkpoint_path : str | None
            Checkpoint each worker reloads. Defaults to the path recorded by
            :func:`shepherd.load_model`.
        Generation parameters mirror :meth:`generate`; ``batch_size`` denotes the
        total distributed batch size.

        Notes
        -----
        This method uses spawned CUDA processes, so call it from a Python script under
        an ``if __name__ == '__main__':`` guard. It increases sample throughput; it
        does not split one molecule's denoising trajectory across GPUs.

        ``store_trajectories`` or ``store_trajectories_x0`` are deliberately unavailable.
        """
        generate_kwargs = {
            "batch_size": batch_size,
            "N_x1": N_x1,
            "N_x4": N_x4,
            "condition": condition,
            "condition_modalities": condition_modalities,
            "condition_center_of_mass": condition_center_of_mass,
            "inpaint": inpaint,
            "scaffold_conditioning": scaffold_conditioning,
            "pharmacophore_conditioning": pharmacophore_conditioning,
            "pharmacophore_prioritization": pharmacophore_prioritization,
            "num_steps": num_steps,
            "early_stop_edm": early_stop_edm,
            "shepherd_pred": shepherd_pred,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "rho": rho,
            "use_stochastic": use_stochastic,
            "S_churn": S_churn,
            "S_noise": S_noise,
            "use_2nd_order_correction": use_2nd_order_correction,
            "verbose": verbose,
        }
        return self._run_distributed(
            "generate",
            generate_kwargs,
            devices=devices,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )

    def generate_composition(
        self,
        N_x1: int,
        N_x4: int,
        batch_size: int,
        *,
        profiles: list[InteractionProfile],
        weights_conditions: np.ndarray,
        composition_mode: str = "default",
        atom_condition: InteractionProfile | ConditionAtoms | None = None,
        condition_modalities: str | set[str] = "all",
        pharmacophore_conditioning: bool = False,
        scaffold_conditioning: bool = False,
        condition_center_of_mass: Literal["origin", "auto"] | np.ndarray = "origin",
        inpaint: InpaintAdvancedOptions | None = None,
        num_steps: int = 400,
        use_stochastic: bool = False,
        S_churn: float = 40.0,
        S_noise: float = 1.0,
        shepherd_pred: bool = True,
        early_stop_edm: int | float = -1,
        sigma_max: float | None = 3.0,
        sigma_min: float | None = 1e-3,
        rho: float | None = 7.0,
        store_trajectories: bool = False,
        verbose: bool = True,
        alignment_start_frac: float = 0.0,
        alignment_interval: int = 10,
        alignment_mode: str = "so3",
        alignment_ema_alpha: float = 1.0,
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
        conditions : list[dict]
            Each dict is passed to gen_edm_inpainting_dict as keyword arguments.
        atom_conditions : dict
            Passed to gen_x1_edm_inpainting_dict as keyword arguments.
        weights_conditions : np.ndarray
            Per-condition weights (excluding the implicit unconditional component).
        composition_mode : str
            "default" prepends an unconditional condition (weight = 1 - sum(weights)).
            "conditional" uses only the supplied conditions.
        num_steps : int
            Number of EDM denoising steps.
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
        from shepherd.comp_inference.sampler import generate_composition
        self.lightning_module.freeze()
        try:
            generated_structures = generate_composition(
                self.lightning_module,
                N_x1=N_x1,
                N_x4=N_x4,
                batch_size=batch_size,
                profiles=profiles,
                weights_conditions=weights_conditions,
                composition_mode=composition_mode,
                atom_condition=atom_condition,
                condition_modalities=condition_modalities,
                pharmacophore_conditioning=pharmacophore_conditioning,
                scaffold_conditioning=scaffold_conditioning,
                condition_center_of_mass=condition_center_of_mass,
                inpaint=inpaint,
                num_steps=num_steps,
                use_stochastic=use_stochastic,
                S_churn=S_churn,
                S_noise=S_noise,
                shepherd_pred=shepherd_pred,
                early_stop_edm=early_stop_edm,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                rho=rho,
                store_trajectories=store_trajectories,
                verbose=verbose,
                alignment_start_frac=alignment_start_frac,
                alignment_interval=alignment_interval,
                alignment_mode=alignment_mode,
                alignment_ema_alpha=alignment_ema_alpha,
            )
        finally:
            self.lightning_module.unfreeze()
        return generated_structures

    def generate_composition_distributed(
        self,
        N_x1: int,
        N_x4: int,
        batch_size: int,
        *,
        profiles: list[InteractionProfile],
        weights_conditions: np.ndarray,
        composition_mode: str = "default",
        atom_condition: InteractionProfile | ConditionAtoms | None = None,
        condition_modalities: str | set[str] = "all",
        pharmacophore_conditioning: bool = False,
        scaffold_conditioning: bool = False,
        condition_center_of_mass: Literal["origin", "auto"] | np.ndarray = "origin",
        inpaint: InpaintAdvancedOptions | None = None,
        num_steps: int = 400,
        use_stochastic: bool = False,
        S_churn: float = 40.0,
        S_noise: float = 1.0,
        shepherd_pred: bool = True,
        early_stop_edm: int | float = -1,
        sigma_max: float | None = 3.0,
        sigma_min: float | None = 1e-3,
        rho: float | None = 7.0,
        verbose: bool = True,
        alignment_start_frac: float = 0.0,
        alignment_interval: int = 10,
        alignment_mode: str = "so3",
        alignment_ema_alpha: float = 1.0,
        devices: int | Sequence[int] | None = None,
        seed: int | None = None,
        checkpoint_path: str | None = None,
    ) -> list[GeneratedSample]:
        """Run :meth:`generate_composition` as independent shards across GPUs.

        ``batch_size`` is the total sample count across the selected devices. See
        :meth:`generate_distributed` for process-launching and checkpoint behavior.
        """
        generate_kwargs = {
            "N_x1": N_x1,
            "N_x4": N_x4,
            "batch_size": batch_size,
            "profiles": profiles,
            "weights_conditions": weights_conditions,
            "composition_mode": composition_mode,
            "atom_condition": atom_condition,
            "condition_modalities": condition_modalities,
            "pharmacophore_conditioning": pharmacophore_conditioning,
            "scaffold_conditioning": scaffold_conditioning,
            "condition_center_of_mass": condition_center_of_mass,
            "inpaint": inpaint,
            "num_steps": num_steps,
            "use_stochastic": use_stochastic,
            "S_churn": S_churn,
            "S_noise": S_noise,
            "shepherd_pred": shepherd_pred,
            "early_stop_edm": early_stop_edm,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "rho": rho,
            "verbose": verbose,
            "alignment_start_frac": alignment_start_frac,
            "alignment_interval": alignment_interval,
            "alignment_mode": alignment_mode,
            "alignment_ema_alpha": alignment_ema_alpha,
        }
        return self._run_distributed(
            "generate_composition",
            generate_kwargs,
            devices=devices,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )

    def _run_distributed(
        self,
        generation_method: Literal["generate", "generate_composition"],
        generate_kwargs: dict[str, object],
        *,
        devices: int | Sequence[int] | None,
        seed: int | None,
        checkpoint_path: str | None,
    ) -> list[GeneratedSample]:
        """Launch and collect process-per-GPU generation workers."""
        from shepherd.utils.distributed_generation import run_distributed

        return run_distributed(
            self.lightning_module,
            generation_method,
            generate_kwargs,
            devices=devices,
            seed=seed,
            checkpoint_path=checkpoint_path or self._checkpoint_path,
        )

    def to_shepherd_score_inputs(
        self,
        samples: list[dict],
        condition: InteractionProfile | Molecule | None = None,
        *,
        condition_modalities: str | set[str] = 'all',
        num_surf_points: int = 400,
        probe_radius: float = 1.2,
        pharm_multi_vector: bool | None = False,
        pharmacophore_prioritization: list[int] | None = None,
        priority_pharm_indices: list[int] | None = None,
        partial_charges=None,
    ) -> dict:
        """Convert generated samples into a ``shepherd_score`` input batch.

        Supply an InteractionProfile or Molecule to ``condition`` to return the inputs necessary
        for the ConditionalEvalPipeline. Note that the reference molecule
        (InteractionProfile.mol) is rebuilt for evaluation (400 surface points and probe radius
        1.2) which differs from the generation defaults (75 / 0.6).

        ``pharmacophore_prioritization`` accepts the same 0/1 mask as :meth:`generate` and
        converts it to the indices of priority pharmacophores expected by
        ``ConditionalEvalPipeline``. An all-zero or all-one mask produces ``None``, because
        shepherd-score's priority comparison requires both priority and non-priority
        pharmacophores.

        ``priority_pharm_indices`` are the indices of priority pharmacophores and are
        mutually exclusive with ``pharmacophore_prioritization``.
        """
        import numpy as np

        if (
            pharmacophore_prioritization is not None
            and priority_pharm_indices is not None
        ):
            raise ValueError(
                'pharmacophore_prioritization and priority_pharm_indices are mutually exclusive'
            )

        generated_mols = []
        surf_points = []
        surf_esp = []
        pharm_feats = []
        modalities = None

        x4_params = self.params['dataset'].get('x4', {})
        remove_dummy_pharms = x4_params.get('include_dummy_pharm', False)
        dummy_pharm_type = (
            x4_params['max_node_types'] - 2
            if remove_dummy_pharms
            else None
        )

        for sample in samples:
            atoms = np.asarray(sample['x1']['atoms'])
            atom_mask = atoms != 0
            generated_mols.append((
                atoms[atom_mask],
                np.asarray(sample['x1']['positions'])[atom_mask],
            ))

            if condition is not None:
                # The conditional path below only needs `generated_mols`.
                continue

            has_x3 = 'x3' in sample
            has_x2 = 'x2' in sample
            has_x4 = 'x4' in sample
            sample_modalities = (has_x3, has_x2 and not has_x3, has_x4)
            if modalities is None:
                modalities = sample_modalities
            elif sample_modalities != modalities:
                raise ValueError(
                    'All samples must contain the same x2/x3/x4 modalities'
                )

            if has_x3:
                surf_points.append(np.asarray(sample['x3']['positions']))
                surf_esp.append(np.asarray(sample['x3']['charges']).reshape(-1))
            elif has_x2:
                surf_points.append(np.asarray(sample['x2']['positions']))

            if has_x4:
                pharm_types = np.asarray(sample['x4']['types'])
                pharm_pos = np.asarray(sample['x4']['positions'])
                pharm_direction = np.asarray(sample['x4']['directions'])
                if dummy_pharm_type is not None:
                    pharm_mask = pharm_types != dummy_pharm_type
                    pharm_types = pharm_types[pharm_mask]
                    pharm_pos = pharm_pos[pharm_mask]
                    pharm_direction = pharm_direction[pharm_mask]
                pharm_feats.append((pharm_types, pharm_pos, pharm_direction))

        # Conditional evaluation
        if condition is not None:
            from shepherd.interaction_profile import InteractionProfile
            from shepherd_score.container import Molecule

            if isinstance(condition, Molecule):
                ref_molec = condition
            elif isinstance(condition, InteractionProfile):
                ref_molec = condition.to_molecule(
                    num_surf_points=num_surf_points,
                    probe_radius=probe_radius,
                    pharm_multi_vector=pharm_multi_vector,
                    partial_charges=partial_charges,
                )
            else:
                raise TypeError(
                    'condition must be an InteractionProfile or '
                    f'shepherd_score Molecule, got {type(condition)!r}'
                )

            if pharmacophore_prioritization is not None:
                priority_mask = np.asarray(pharmacophore_prioritization)
                n_pharms = 0 if ref_molec.pharm_types is None else len(ref_molec.pharm_types)

                if priority_mask.ndim != 1 or len(priority_mask) != n_pharms:
                    raise ValueError(
                        'pharmacophore_prioritization must be a one-dimensional '
                        f'mask of length {n_pharms}, got shape {priority_mask.shape}'
                    )
                if not np.isin(priority_mask, (0, 1)).all():
                    raise ValueError('pharmacophore_prioritization must contain only 0s and 1s')

                # Convert the priority mask to the indices of priority pharmacophores
                priority_pharm_indices = np.flatnonzero(priority_mask == 1).tolist()
                # don't do priority evals if all pharmacophores are priority or none are priority
                if not priority_pharm_indices or len(priority_pharm_indices) == n_pharms:
                    priority_pharm_indices = None

            return {
                'ref_molec': ref_molec,
                'generated_mols': generated_mols,
                'condition': _eval_condition_name(condition_modalities),
                'num_surf_points': ref_molec.num_surf_points,
                'pharm_multi_vector': (
                    ref_molec.pharm_multi_vector
                    if pharm_multi_vector is None
                    else pharm_multi_vector
                ),
                'priority_pharm_indices': priority_pharm_indices,
            }

        # Unconditional/consistency evaluation
        has_surface = bool(modalities and (modalities[0] or modalities[1]))
        has_esp = bool(modalities and modalities[0])
        has_pharm = bool(modalities and modalities[2])
        return {
            'generated_mols': generated_mols,
            'generated_surf_points': surf_points if has_surface else None,
            'generated_surf_esp': surf_esp if has_esp else None,
            'generated_pharm_feats': pharm_feats if has_pharm else None,
        }


def _eval_condition_name(condition_modalities: str | set[str]) -> str:
    """Map ``generate()`` modality names to a shepherd-score's ConditionalEvalPipeline condition."""
    from shepherd.interaction_profile import _resolve_modalities
    resolved = _resolve_modalities(condition_modalities)

    has_surf = 'surface' in resolved
    has_esp = 'electrostatics' in resolved
    has_pharm = bool(resolved & {'pharm_positions', 'pharm_directions', 'pharm_type'})
    if has_surf and has_pharm:
        return 'all'
    if has_esp:
        return 'esp'
    if has_pharm:
        return 'pharm'
    if has_surf:
        return 'surface'
    return 'all'
