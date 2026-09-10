#!/usr/bin/env python
"""Generate and evaluate two-condition composition samples on MOSES-aq."""

from __future__ import annotations

import os
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import torch
from tqdm import tqdm

from shepherd import load_model
from shepherd.comp_inference.utils import align_two_ref_mol, get_random_conditions
from shepherd.inference.sampler import GeneratedSample
from shepherd.interaction_profile import InteractionProfile, extract_interaction_profile
from shepherd_score.evaluations.evaluate.pipelines import ConditionalEvalPipeline

os.environ.setdefault("TMPDIR", tempfile.gettempdir())

torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "conformers" / "moses_aq"
TEST_PATH = DATA_DIR / "moses_test_scaffold_molblock_charges.pkl"
# "default" prepends an unconditional component weighted 1 - sum(weights)
COMPOSITION_MODES = ("default", "conditional")
# Which similarity drives the rigid alignment of condition 1 onto condition 2
ALIGN_CONDITIONS = ("esp", "surface", "pharm", "all")
CONDITION_MODALITIES = ("all", "x2", "x3", "x4", "x2_x4")
# Profile extraction settings baked into the released checkpoints
DEFAULT_NUM_SURF_POINTS = 75
DEFAULT_PROBE_RADIUS = 0.6


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, value: Any) -> None:
    # Replace completed files atomically so a preempted job never leaves a
    # half-written checkpoint that a resume would trust
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def select_pair(
    molblocks_and_charges: list,
    sample_id: int,
    given_pairs: list | None,
    n_samples: int,
    seed: int,
) -> tuple[list, int, int]:
    """Resolve one sample id to a (mol_data, i, j) condition pair.

    Three input layouts are supported, matching the ShEPhERD-1x script:
    a pre-paired file of ((molblock, charges), (molblock, charges)) tuples, a
    flat molblock/charges list plus a file of (i, j) index pairs, or a flat
    list from which index pairs are drawn at random.
    """
    if isinstance(molblocks_and_charges[0][0], tuple):
        pair = molblocks_and_charges[sample_id]
        return [pair[0], pair[1]], 0, 1

    if given_pairs is not None:
        # Draw which of the supplied pairs this sample id uses
        random_conditions = get_random_conditions(
            n_cond=1, n_samples=n_samples, seed=seed, max_len=len(given_pairs)
        )
        i, j = given_pairs[random_conditions[sample_id][0]]
        return molblocks_and_charges, int(i), int(j)

    random_conditions = get_random_conditions(
        n_cond=2, n_samples=n_samples, seed=seed, max_len=len(molblocks_and_charges)
    )
    i, j = random_conditions[sample_id]
    return molblocks_and_charges, int(i), int(j)


def resolve_atom_counts(
    n_atoms_1: int,
    n_atoms_2: int,
    weights_conditions: list[float],
    atom_range: list[int],
) -> list[int]:
    """Build the ragged per-batch atom counts for one sample.

    A composition whose second weight is zero is really condition 1 alone, so
    it sizes off condition 1 rather than the larger of the two. The count list
    is doubled, as in 1x, so each atom count is sampled by two batches.
    """
    if weights_conditions[1] <= 0:
        n_atoms = n_atoms_1
    else:
        n_atoms = max(n_atoms_1, n_atoms_2)

    # Upper bound is exclusive, matching 1x: --atom_range -5 5 spans N-5..N+4
    return list(range(n_atoms + atom_range[0], n_atoms + atom_range[1])) * 2


def calculate_summary_statistics(combined_df: pd.DataFrame) -> dict:
    """Summary statistics."""
    total_samples = int(len(combined_df))
    valid_post = (
        int((~combined_df["molblocks_post_opt"].isna()).sum()) if total_samples else 0
    )
    validity_post_rate = float(valid_post / total_samples) if total_samples > 0 else 0.0

    non_nan_df = combined_df[combined_df.notna()]
    filtered_df = non_nan_df[(non_nan_df["graph_similarities_post_opt"] <= 0.3)]

    return {
        "total_samples": total_samples,
        "validity_rate": validity_post_rate,
        "graph_similarity": np.nanmedian(combined_df["graph_similarities_post_opt"]),
        "surface_similarity": np.nanmedian(filtered_df["sims_surf_target_relax_optimal"]),
        "esp_similarity": np.nanmedian(filtered_df["sims_esp_target_relax_optimal"]),
        "pharm_similarity": np.nanmedian(filtered_df["sims_pharm_target_relax_optimal"]),
        "strain_energy": np.nanmedian(combined_df["strain_energies"]),
        "SA_score": np.nanmedian(combined_df["SA_scores_post_opt"]),
        "QED": np.nanmedian(combined_df["QEDs_post_opt"]),
    }


def evaluate_single_condition(work_item):
    """Evaluate one sample's generated molecules against one of its conditions."""
    key, pipeline_kwargs, rowwise_path, global_path, metadata = work_item
    try:
        pipeline = ConditionalEvalPipeline(**pipeline_kwargs)
        blocker = rdBase.BlockLogs()
        try:
            pipeline.evaluate(
                num_workers=1, num_processes=1, verbose=False, timeout_minutes=15
            )
        finally:
            del blocker

        series_global, df_rowwise = pipeline.to_pandas()
        df_rowwise = df_rowwise.assign(**metadata)
        global_results = series_global.to_dict()
        global_results.update(metadata)
        write_pickle(rowwise_path, df_rowwise)
        write_pickle(global_path, global_results)
        return key, len(df_rowwise), None
    except Exception as error:
        return key, None, f"{type(error).__name__}: {error}\n{traceback.format_exc()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)

    # Model
    parser.add_argument("--model_path", default=None,
        help="ShEPhERD-2 checkpoint (.ckpt); omit to use the packaged default",
    )
    parser.add_argument("--ema_path", default=None,
        help="Optional EMA checkpoint (.ckpt)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Conditions
    parser.add_argument("--path", type=Path, default=TEST_PATH,
        help="Pickle of ((molblock, charges), (molblock, charges)) pairs, or a "
             "flat (molblock, charges) list to pair up",
    )
    parser.add_argument("--given_pairs_path", type=Path, default=None,
        help="Pickle of (i, j) index pairs into a flat --path list; ignored when "
             "--path is already paired",
    )
    parser.add_argument("--save_dir", type=Path, required=True,
        help="Output directory for checkpoints, per-sample results, and summaries",
    )
    parser.add_argument("--sample_id", type=int, default=None,
        help="Run this one sample id, for SLURM array jobs",
    )
    parser.add_argument("--indices", type=int, nargs="+", default=None,
        help="Run these sample ids; defaults to all --n_samples ids",
    )
    parser.add_argument("--n_samples", type=int, default=100,
        help="Size of the sample-id space, i.e. how many pairs are drawn",
    )
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for drawing condition pairs",
    )
    parser.add_argument("--align_condition", default="esp", choices=ALIGN_CONDITIONS,
        help="Similarity used to rigidly align condition 1 onto condition 2",
    )

    # Sampling
    parser.add_argument("--composition_mode", default="default",
        choices=COMPOSITION_MODES,
        help="'default' prepends an unconditional component weighted "
             "1 - sum(--weights_conditions); 'conditional' uses only the two "
             "supplied conditions",
    )
    parser.add_argument("--weights_conditions", type=float, nargs=2,
        default=[0.5, 0.5], metavar=("W_1", "W_2"),
        help="Per-condition composition weights",
    )
    parser.add_argument("--batch_size_per_sample", type=int, default=20,
        help="Molecules per atom count; each atom count is run as two batches "
             "of half this size",
    )
    parser.add_argument("--atom_range", type=int, nargs=2, default=[-5, 5],
        metavar=("LO", "HI"),
        help="Atom counts swept around the reference size, upper bound exclusive",
    )
    parser.add_argument("--condition_modalities", default="all",
        choices=CONDITION_MODALITIES,
        help="Modalities to condition on; x2/x3/x4 are the ShEPhERD-1x names",
    )
    parser.add_argument("--pharmacophore_conditioning", action="store_true",
        help="Inpaint only the reference pharmacophores rather than all of x4",
    )
    parser.add_argument("--neutralize_esp", action="store_true",
        help="Smear net charge over atoms so the reference ESP sums to zero",
    )
    parser.add_argument("--profile_xtb_optimize", action="store_true",
        help="Relax the reference geometries with xTB before extracting "
             "profiles. Off by default because the inputs are already posed "
             "and carry precomputed charges.",
    )
    parser.add_argument("--num_surf_points", type=int, default=DEFAULT_NUM_SURF_POINTS,
        help="Surface points sampled for the conditioning profiles",
    )
    parser.add_argument("--probe_radius", type=float, default=DEFAULT_PROBE_RADIUS,
        help="Probe radius for the conditioning surface",
    )

    # EDM sampler
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--shepherd_pred", action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic shepherd prediction instead of an ODE step",
    )
    parser.add_argument("--use_stochastic", action="store_true",
        help="Add churn noise to EDM sampling",
    )
    parser.add_argument("--early_stop_edm", type=float, default=0.9,
        help="Truncate EDM sampling. -1 runs all --num_steps; a value in [0, 1] "
             "is a fraction of num_steps; a value above 1 is an absolute step "
             "count. 0.9 reproduces the ShEPhERD-1x default.",
    )
    parser.add_argument("--sigma_max", type=float, default=3.0)
    parser.add_argument("--sigma_min", type=float, default=None,
        help="Override the EDM schedule sigma_min",
    )
    parser.add_argument("--rho", type=float, default=None,
        help="Override the EDM schedule rho",
    )
    parser.add_argument("--alignment_start_frac", type=float, default=0.0,
        help="Fraction of steps, counted from the end, over which ESP alignment "
             "is applied",
    )
    parser.add_argument("--alignment_interval", type=int, default=30,
        help="Recompute the ESP alignment every N steps",
    )
    parser.add_argument("--alignment_mode", default="so3", choices=("so3", "se3"),
        help="ESP alignment mode: so3 (rotation only) or se3 (full rigid)",
    )
    parser.add_argument("--alignment_ema_alpha", type=float, default=0.3,
        help="EMA smoothing factor for alignment (1.0 = no smoothing)",
    )

    # Evaluation and output
    parser.add_argument("--skip_eval", action="store_true",
        help="Generate and checkpoint only, skipping the shepherd-score "
             "evaluation and the combined summary",
    )
    parser.add_argument("--eval_workers", type=int, default=20,
        help="Parallel workers for evaluation",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    if args.sample_id is not None and args.indices is not None:
        parser.error("--sample_id and --indices are mutually exclusive")
    if args.atom_range[0] >= args.atom_range[1]:
        parser.error("--atom_range LO must be less than HI")
    if args.batch_size_per_sample < 2:
        parser.error("--batch_size_per_sample must be at least 2")
    if min(args.weights_conditions) < 0:
        parser.error("--weights_conditions must be non-negative")
    if args.composition_mode == "default" and sum(args.weights_conditions) > 1:
        parser.error(
            "--weights_conditions must sum to at most 1 when --composition_mode "
            "is 'default', since the remainder weights the unconditional component"
        )
    if args.eval_workers < 1:
        parser.error("--eval_workers must be positive")
    return args


def _check_profile_settings(args: argparse.Namespace, params: dict) -> None:
    """Warn when profile extraction disagrees with the checkpoint's own params."""
    x3_params = params.get("dataset", {}).get("x3", {})
    for name, value in (
        ("num_surf_points", args.num_surf_points),
        ("probe_radius", args.probe_radius),
    ):
        expected = x3_params.get(name)
        if expected is not None and expected != value:
            print(
                f"WARNING: --{name} is {value} but the checkpoint was trained "
                f"with {expected}; conditioning may be out of distribution",
                file=sys.stderr,
            )


def _prepare_profiles(
    args: argparse.Namespace,
    mol_data: list,
    i: int,
    j: int,
) -> tuple[InteractionProfile, InteractionProfile]:
    """Align condition 1 onto condition 2 and extract both profiles."""
    aligned_mol = align_two_ref_mol(
        mol_data,
        mol_to_align_i=i,
        ref_mol_i=j,
        condition=args.align_condition,
    )
    ref_mol = Chem.MolFromMolBlock(mol_data[j][0], removeHs=False)
    if ref_mol is None:
        raise ValueError(f"condition index {j} has an invalid molblock")

    profiles = []
    for mol, charges in ((aligned_mol, mol_data[i][1]), (ref_mol, mol_data[j][1])):
        charges = np.asarray(charges)
        if charges.shape != (mol.GetNumAtoms(),):
            raise ValueError(
                f"{len(charges)} charges for {mol.GetNumAtoms()} atoms"
            )
        profile = extract_interaction_profile(
            mol,
            partial_charges=charges,
            xtb_optimize=args.profile_xtb_optimize,
            num_surf_points=args.num_surf_points,
            probe_radius=args.probe_radius,
            neutralize_esp=args.neutralize_esp,
        )
        if profile is None:
            raise RuntimeError("interaction-profile extraction failed")
        profiles.append(profile)
    return profiles[0], profiles[1]


def _generate_sample(
    args: argparse.Namespace,
    model,
    profiles: tuple[InteractionProfile, InteractionProfile],
    n_atoms_list: list[int],
    n_x4: int,
) -> list[GeneratedSample]:
    """Sweep the atom counts, composing both conditions at each one."""
    batch_size = args.batch_size_per_sample // 2
    weights = np.array(args.weights_conditions)
    generated_samples: list[GeneratedSample] = []

    for n_atoms in n_atoms_list:
        batch = model.generate_composition(
            N_x1=int(n_atoms),
            N_x4=n_x4,
            batch_size=batch_size,
            profiles=list(profiles),
            weights_conditions=weights,
            composition_mode=args.composition_mode,
            condition_modalities=args.condition_modalities,
            pharmacophore_conditioning=args.pharmacophore_conditioning,
            num_steps=args.num_steps,
            shepherd_pred=args.shepherd_pred,
            use_stochastic=args.use_stochastic,
            early_stop_edm=args.early_stop_edm,
            sigma_max=args.sigma_max,
            sigma_min=args.sigma_min,
            rho=args.rho,
            alignment_start_frac=args.alignment_start_frac,
            alignment_interval=args.alignment_interval,
            alignment_mode=args.alignment_mode,
            alignment_ema_alpha=args.alignment_ema_alpha,
            store_trajectories=False,
            verbose=False,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        generated_samples += batch

    return generated_samples


def _run_config(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _check_existing_config(path: Path, config: dict) -> None:
    """Prevent incompatible runs from sharing an output directory."""
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    immutable_keys = (
        "path",
        "given_pairs_path",
        "n_samples",
        "seed",
        "composition_mode",
        "weights_conditions",
        "align_condition",
        "atom_range",
        "batch_size_per_sample",
        "condition_modalities",
    )
    differences = [
        key for key in immutable_keys if previous.get(key) != config.get(key)
    ]
    if differences:
        raise ValueError(
            "--save_dir holds an incompatible run_config.json (different "
            f"{', '.join(differences)}); use a new directory"
        )


def main() -> None:
    args = parse_args()

    print(f"Composition mode: {args.composition_mode} | "
          f"weights {args.weights_conditions}")
    print("Loading condition molblocks and charges...")
    molblocks_and_charges = read_pickle(args.path)
    if not isinstance(molblocks_and_charges, (list, tuple)) or not molblocks_and_charges:
        raise ValueError("--path must contain a non-empty sequence")

    is_paired = isinstance(molblocks_and_charges[0][0], tuple)
    given_pairs = None
    if is_paired:
        # A pre-paired file indexes samples directly, so it caps the id space
        id_space = len(molblocks_and_charges)
        if args.given_pairs_path is not None:
            print(
                "Ignoring --given_pairs_path because --path is already paired",
                file=sys.stderr,
            )
    else:
        id_space = args.n_samples
        if args.given_pairs_path is not None:
            given_pairs = read_pickle(args.given_pairs_path)
            print(f"Loaded {len(given_pairs)} given pairs")

    if args.sample_id is not None:
        selected = [args.sample_id]
    elif args.indices is not None:
        selected = args.indices
    else:
        selected = list(range(id_space))
    if len(selected) != len(set(selected)):
        raise ValueError("sample ids must be unique")
    invalid = [index for index in selected if index < 0 or index >= id_space]
    if invalid:
        raise ValueError(f"sample ids out of range [0, {id_space}): {invalid}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.save_dir / "run_config.json"
    config = _run_config(args)
    _check_existing_config(config_path, config)
    write_json(config_path, config)

    print("Loading model...")
    model = load_model(
        local_checkpoint_path=str(args.model_path) if args.model_path else None,
        device=args.device,
    )
    if args.ema_path is not None:
        model.load_ema_weights_for_inference(ema_checkpoint_path=str(args.ema_path))
    _check_profile_settings(args, model.params)

    failed = {}

    def record_failure(key: str, error: Exception | str) -> None:
        failed[key] = (
            error
            if isinstance(error, str)
            else f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        )
        write_json(args.save_dir / "failed_samples.json", failed)

    evaluation_work = []
    progress = tqdm(
        selected, total=len(selected), desc="Generating", disable=not args.verbose
    )
    for sample_id in progress:
        progress.set_description(f"Generating sample {sample_id:04d}")
        samples_path = checkpoint_dir / f"sample_{sample_id:04d}_samples.pkl"
        profiles_path = checkpoint_dir / f"sample_{sample_id:04d}_profiles.pkl"

        try:
            mol_data, i, j = select_pair(
                molblocks_and_charges, sample_id, given_pairs, args.n_samples, args.seed
            )

            if profiles_path.exists() and not args.overwrite:
                profile_1, profile_2 = read_pickle(profiles_path)
            else:
                profile_1, profile_2 = _prepare_profiles(args, mol_data, i, j)
                write_pickle(profiles_path, (profile_1, profile_2))

            n_atoms_list = resolve_atom_counts(
                profile_1.n_atoms,
                profile_2.n_atoms,
                args.weights_conditions,
                args.atom_range,
            )
            n_x4 = max(profile_1.n_pharms, profile_2.n_pharms)
            tqdm.write(
                f"Sample {sample_id:04d}: pair ({i}, {j}) | "
                f"N_x1 {profile_1.n_atoms}, {profile_2.n_atoms} -> "
                f"{n_atoms_list[0]}..{n_atoms_list[len(n_atoms_list) // 2 - 1]} | "
                f"N_x4 {profile_1.n_pharms}, {profile_2.n_pharms} -> {n_x4}"
            )

            if samples_path.exists() and not args.overwrite:
                generated_samples = read_pickle(samples_path)
            else:
                generated_samples = _generate_sample(
                    args, model, (profile_1, profile_2), n_atoms_list, n_x4
                )
                write_pickle(samples_path, generated_samples)

            if args.skip_eval:
                continue

            # Composition answers to both parents, so score the same molecules
            # against each condition separately
            for condition_index, profile in enumerate((profile_1, profile_2)):
                key = f"{sample_id:04d}_cond{condition_index}"
                rowwise_path = args.save_dir / f"sample_{key}_rowwise.pkl"
                global_path = args.save_dir / f"sample_{key}_global.pkl"
                if (
                    rowwise_path.exists()
                    and global_path.exists()
                    and not args.overwrite
                ):
                    continue
                pipeline_kwargs = model.to_shepherd_score_inputs(
                    generated_samples,
                    condition=profile,
                    condition_modalities=args.condition_modalities,
                    partial_charges=profile.partial_charges,
                )
                evaluation_work.append(
                    (
                        key,
                        pipeline_kwargs,
                        rowwise_path,
                        global_path,
                        {
                            "sample_id": sample_id,
                            "condition_index": condition_index,
                            "condition_mol_index": i if condition_index == 0 else j,
                            "composition_mode": args.composition_mode,
                            "weights_conditions": str(args.weights_conditions),
                        },
                    )
                )

        except Exception as error:
            record_failure(str(sample_id), error)
            tqdm.write(f"Sample {sample_id} failed: {error}", file=sys.stderr)

    # Release the GPU model before starting CPU evaluation workers
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skip_eval:
        print(f"Generation complete; samples saved under {checkpoint_dir}")
        print("Done!")
        return

    print("Starting evaluation...")
    if evaluation_work:
        worker_count = min(args.eval_workers, len(evaluation_work))
        spawn_context = mp.get_context("spawn")
        with spawn_context.Pool(worker_count) as pool:
            results = pool.imap_unordered(
                evaluate_single_condition, evaluation_work, chunksize=1
            )
            for key, _num_samples, error_message in tqdm(
                results,
                total=len(evaluation_work),
                desc="Evaluating",
                disable=not args.verbose,
            ):
                if error_message is not None:
                    record_failure(key, error_message)
                    print(
                        f"Sample {key} failed: {error_message.splitlines()[0]}",
                        file=sys.stderr,
                    )

    print("Combining results...")
    frames, spent_paths = [], []
    for sample_id in selected:
        for condition_index in (0, 1):
            key = f"{sample_id:04d}_cond{condition_index}"
            rowwise_path = args.save_dir / f"sample_{key}_rowwise.pkl"
            if rowwise_path.exists():
                frames.append(read_pickle(rowwise_path))
                spent_paths.append(rowwise_path)
            global_path = args.save_dir / f"sample_{key}_global.pkl"
            if global_path.exists():
                spent_paths.append(global_path)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path = args.save_dir / "combined_results.pkl"
    write_pickle(combined_path, combined)

    stats = calculate_summary_statistics(combined)
    stats_path = args.save_dir / "combined_results_stats.json"
    write_json(stats_path, stats)

    for path in spent_paths:
        path.unlink()

    if failed:
        print(f"Completed with {len(failed)} failures; see failed_samples.json")
    else:
        failed_path = args.save_dir / "failed_samples.json"
        if failed_path.exists():
            failed_path.unlink()
    print(f"Saved combined results to {combined_path}")
    print(f"Saved summary statistics to {stats_path}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Validity rate (post-opt): {stats['validity_rate']:.3%}")
    print("Done!")


if __name__ == "__main__":
    main()
