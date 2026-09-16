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


def select_pair(
    molblocks_and_charges: list,
    sample_id: int,
    given_pairs: list | None,
    n_samples: int,
    seed: int,
) -> tuple[list, int, int]:
    """Resolve one sample id to a (mol_data, i, j) condition pair.

    Three input layouts are supported:
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

    # Upper bound is exclusive, matching 1x: --atom-range -5 5 spans N-5..N+4
    return list(range(n_atoms + atom_range[0], n_atoms + atom_range[1])) * 2


def _json_median(values) -> float:
    return float(np.nanmedian(values))


def _summary_block(combined_df: pd.DataFrame) -> dict:
    """Medians for one set of shepherd-score rowwise results."""
    total_samples = int(len(combined_df))
    if total_samples == 0 or "molblocks_post_opt" not in combined_df.columns:
        return {
            "total_samples": 0,
            "validity_rate": 0.0,
            "graph_similarity": float("nan"),
            "surface_similarity": float("nan"),
            "esp_similarity": float("nan"),
            "pharm_similarity": float("nan"),
            "strain_energy": float("nan"),
            "SA_score": float("nan"),
            "QED": float("nan"),
        }

    valid_post = int((~combined_df["molblocks_post_opt"].isna()).sum())
    validity_post_rate = float(valid_post / total_samples)

    non_nan_df = combined_df.dropna(subset=["molblocks_post_opt"])
    filtered_df = non_nan_df[(non_nan_df["graph_similarities_post_opt"] <= 0.3)]

    return {
        "total_samples": total_samples,
        "validity_rate": validity_post_rate,
        "graph_similarity": _json_median(combined_df["graph_similarities_post_opt"]),
        "surface_similarity": _json_median(filtered_df["sims_surf_target_relax_optimal"]),
        "esp_similarity": _json_median(filtered_df["sims_esp_target_relax_optimal"]),
        "pharm_similarity": _json_median(filtered_df["sims_pharm_target_relax_optimal"]),
        "strain_energy": _json_median(combined_df["strain_energies"]),
        "SA_score": _json_median(combined_df["SA_scores_post_opt"]),
        "QED": _json_median(combined_df["QEDs_post_opt"]),
    }


def calculate_summary_statistics(combined_df: pd.DataFrame) -> dict:
    """Separate summaries for each composition parent (no pooling or averaging)."""
    stats: dict = {"total_eval_rows": int(len(combined_df))}
    for condition_index in (0, 1):
        if (
            combined_df.empty
            or "condition_index" not in combined_df.columns
        ):
            subset = pd.DataFrame()
        else:
            subset = combined_df[combined_df["condition_index"] == condition_index]
        stats[f"condition_{condition_index}"] = _summary_block(subset)
    return stats


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
        with rowwise_path.open("wb") as handle:
            pickle.dump(df_rowwise, handle)
        with global_path.open("wb") as handle:
            pickle.dump(global_results, handle)
        return key, len(df_rowwise), None
    except Exception as error:
        return key, None, f"{type(error).__name__}: {error}\n{traceback.format_exc()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Indexing and condition pairs
    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_PATH,
        help="Pickle of ((molblock, charges), (molblock, charges)) pairs, or a "
        "flat (molblock, charges) list to pair up",
    )
    parser.add_argument("--given-pairs-path", type=Path, default=None,
        help="Pickle of (i, j) index pairs into a flat --test-path list; ignored "
        "when --test-path is already paired",
    )
    parser.add_argument("--sample-id", type=int, default=None,
        help="Run this one sample id, for SLURM array jobs",
    )
    parser.add_argument("--indices", type=int, nargs="+", default=None,
        help="Run these sample ids; defaults to all --n-samples ids",
    )
    parser.add_argument("--n-samples", type=int, default=100,
        help="Size of the sample-id space, i.e. how many pairs are drawn",
    )
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for drawing condition pairs",
    )
    parser.add_argument("--align-condition", default="esp", choices=ALIGN_CONDITIONS,
        help="Similarity used to rigidly align condition 1 onto condition 2",
    )

    # Sampling
    parser.add_argument("--composition-mode", default="default",
        choices=COMPOSITION_MODES,
        help="'default' prepends an unconditional component weighted "
        "1 - sum(--weights-conditions); 'conditional' uses only the two "
        "supplied conditions",
    )
    parser.add_argument("--weights-conditions", type=float, nargs=2,
        default=[0.5, 0.5], metavar=("W_1", "W_2"),
        help="Per-condition composition weights",
    )
    parser.add_argument("--batch-size-per-sample", type=int, default=20,
        help="Molecules per atom count; each atom count is run as two batches "
        "of half this size",
    )
    parser.add_argument("--atom-range", type=int, nargs=2, default=[-5, 5],
        metavar=("LO", "HI"),
        help="Atom counts swept around the reference size, upper bound exclusive",
    )
    parser.add_argument("--condition-modalities", default="all",
        choices=CONDITION_MODALITIES,
        help="Modalities to condition on; x2/x3/x4 are the ShEPhERD-1x names",
    )
    parser.add_argument("--pharmacophore-conditioning", action="store_true",
        help="Inpaint only the reference pharmacophores rather than all of x4",
    )
    parser.add_argument("--neutralize-esp", action="store_true",
        help="Smear net charge over atoms so the reference ESP sums to zero",
    )
    parser.add_argument("--profile-xtb-optimize", action="store_true",
        help="Relax the reference geometries with xTB before extracting "
        "profiles. Off by default because the inputs are already posed "
        "and carry precomputed charges.",
    )
    parser.add_argument("--probe-radius", type=float, default=DEFAULT_PROBE_RADIUS,
        help="Probe radius for the conditioning surface",
    )

    # EDM sampler
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--early-stop-edm", type=float, default=0.9,
        help="Truncate EDM sampling. -1 runs all --num-steps; a value in [0, 1] "
             "is a fraction of num_steps; a value above 1 is an absolute step "
             "count. 0.9 reproduces the ShEPhERD-1x default.",
    )
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--sigma-min", type=float, default=None,
        help="Override the EDM schedule sigma_min",
    )
    parser.add_argument("--rho", type=float, default=None,
        help="Override the EDM schedule rho",
    )
    parser.add_argument("--alignment-start-frac", type=float, default=0.0,
        help="Fraction of steps, counted from the end, over which ESP alignment "
        "is applied",
    )
    parser.add_argument("--alignment-interval", type=int, default=30,
        help="Recompute the ESP alignment every N steps",
    )
    parser.add_argument("--alignment-mode", default="so3", choices=("so3", "se3"),
        help="ESP alignment mode: so3 (rotation only) or se3 (full rigid)",
    )
    parser.add_argument("--alignment-ema-alpha", type=float, default=0.3,
        help="EMA smoothing factor for alignment (1.0 = no smoothing)",
    )

    # Evaluation and output
    parser.add_argument("--skip-eval", action="store_true",
        help="Generate and checkpoint only, skipping the shepherd-score "
        "evaluation and the combined summary",
    )
    parser.add_argument("--eval-workers", type=int, default=20,
        help="Parallel workers for evaluation",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    if args.sample_id is not None and args.indices is not None:
        parser.error("--sample-id and --indices are mutually exclusive")
    if args.atom_range[0] >= args.atom_range[1]:
        parser.error("--atom-range LO must be less than HI")
    if args.batch_size_per_sample < 2:
        parser.error("--batch-size-per-sample must be at least 2")
    if args.composition_mode == "default":
        if min(args.weights_conditions) < 0:
            parser.error(
                "--weights-conditions must be non-negative when --composition-mode "
                "is 'default'; negative weights are only supported with "
                "--composition-mode conditional"
            )
        if sum(args.weights_conditions) > 1:
            parser.error(
                "--weights-conditions must sum to at most 1 when --composition-mode "
                "is 'default', since the remainder weights the unconditional component"
            )
    if args.eval_workers < 1:
        parser.error("--eval-workers must be positive")
    return args


def _check_profile_settings(args: argparse.Namespace, params: dict) -> None:
    """Warn when profile extraction disagrees with the checkpoint's own params."""
    x3_params = params.get("dataset", {}).get("x3", {})
    for name, value in (
        ("num_surf_points", DEFAULT_NUM_SURF_POINTS),
        ("probe_radius", args.probe_radius),
    ):
        expected = x3_params.get(name)
        if expected is not None and expected != value:
            print(
                f"WARNING: --{name.replace('_', '-')} is {value} but the checkpoint "
                f"was trained with {expected}; conditioning may be out of distribution",
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
            num_surf_points=DEFAULT_NUM_SURF_POINTS,
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


def _check_existing_config(path: Path, config: dict) -> None:
    """Prevent incompatible runs from sharing an output directory."""
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    immutable_keys = (
        "test_path",
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
            "--output-dir holds an incompatible run_config.json (different "
            f"{', '.join(differences)}); use a new directory"
        )


def _write_failed_samples(path: Path, failed: dict) -> None:
    path.write_text(
        json.dumps({str(key): value for key, value in failed.items()}, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()

    print(f"Composition mode: {args.composition_mode} | "
          f"weights {args.weights_conditions}")
    print("Loading condition molblocks and charges...")
    with args.test_path.open("rb") as handle:
        molblocks_and_charges = pickle.load(handle)
    if not isinstance(molblocks_and_charges, (list, tuple)) or not molblocks_and_charges:
        raise ValueError("--test-path must contain a non-empty sequence")

    is_paired = isinstance(molblocks_and_charges[0][0], tuple)
    given_pairs = None
    if is_paired:
        # A pre-paired file indexes samples directly, so it caps the id space
        id_space = len(molblocks_and_charges)
        if args.given_pairs_path is not None:
            print(
                "Ignoring --given-pairs-path because --test-path is already paired",
                file=sys.stderr,
            )
    else:
        id_space = args.n_samples
        if args.given_pairs_path is not None:
            with args.given_pairs_path.open("rb") as handle:
                given_pairs = pickle.load(handle)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_path = args.output_dir / "run_config.json"
    _check_existing_config(config_path, config)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    print("Loading model...")
    model = load_model(
        local_checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )
    _check_profile_settings(args, model.params)

    failed = {}
    failed_path = args.output_dir / "failed_samples.json"

    def record_failure(key: str, error: Exception | str) -> None:
        failed[key] = (
            error
            if isinstance(error, str)
            else f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        )
        _write_failed_samples(failed_path, failed)

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
                with profiles_path.open("rb") as handle:
                    profile_1, profile_2 = pickle.load(handle)
            else:
                profile_1, profile_2 = _prepare_profiles(args, mol_data, i, j)
                with profiles_path.open("wb") as handle:
                    pickle.dump((profile_1, profile_2), handle)

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
                with samples_path.open("rb") as handle:
                    generated_samples = pickle.load(handle)
            else:
                generated_samples = _generate_sample(
                    args, model, (profile_1, profile_2), n_atoms_list, n_x4
                )
                with samples_path.open("wb") as handle:
                    pickle.dump(generated_samples, handle)

            if args.skip_eval:
                continue

            # Composition answers to both parents, so score the same molecules
            # against each condition separately
            for condition_index, profile in enumerate((profile_1, profile_2)):
                key = f"{sample_id:04d}_cond{condition_index}"
                rowwise_path = args.output_dir / f"sample_{key}_rowwise.pkl"
                global_path = args.output_dir / f"sample_{key}_global.pkl"
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
            rowwise_path = args.output_dir / f"sample_{key}_rowwise.pkl"
            if rowwise_path.exists():
                with rowwise_path.open("rb") as handle:
                    frames.append(pickle.load(handle))
                spent_paths.append(rowwise_path)
            global_path = args.output_dir / f"sample_{key}_global.pkl"
            if global_path.exists():
                spent_paths.append(global_path)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path = args.output_dir / "combined_results.pkl"
    with combined_path.open("wb") as handle:
        pickle.dump(combined, handle)

    stats = calculate_summary_statistics(combined)
    stats_path = args.output_dir / "combined_results_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    for path in spent_paths:
        path.unlink()

    if failed:
        print(f"Completed with {len(failed)} failures; see failed_samples.json")
    else:
        if failed_path.exists():
            failed_path.unlink()
    print(f"Saved combined results to {combined_path}")
    print(f"Saved summary statistics to {stats_path}")
    print(f"  Total eval rows: {stats['total_eval_rows']}")
    for condition_index in (0, 1):
        block = stats[f"condition_{condition_index}"]
        print(
            f"  vs condition {condition_index}: "
            f"{block['total_samples']} rows, "
            f"validity {block['validity_rate']:.3%}, "
            f"ESP median {block['esp_similarity']}"
        )
    print("Done!")


if __name__ == "__main__":
    main()
