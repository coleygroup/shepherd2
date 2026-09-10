#!/usr/bin/env python
"""Generate and evaluate the 100-reference MOSES conditional benchmark."""

from __future__ import annotations

import os
import tempfile
import argparse
import json
import multiprocessing as mp
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import torch
from tqdm import tqdm

from shepherd import load_model
from shepherd.data_utils.subgraph import (
    select_brics_scaffold_indices,
    select_hetero_scaffold_indices,
)
from shepherd.interaction_profile import extract_interaction_profile, InteractionProfile
from shepherd_score.evaluations.evaluate.pipelines import ConditionalEvalPipeline
from shepherd.inference.sampler import GeneratedSample

os.environ.setdefault("TMPDIR", tempfile.gettempdir())

torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "conformers" / "moses_aq"
TEST_PATH = DATA_DIR / "moses_test_scaffold_molblock_charges.pkl"
PRIORITY_PATH = DATA_DIR / "interaction_pharm_prioritization.json"
NUM_REFERENCES = 100
MODES = (
    "conditional",
    "scaffold-random-hetero",
    "scaffold-brics",
    "pharm-full",
    "pharm-priority",
    "pharm-priority-only",
)
# mode -> the selection method its scaffolds come from
SCAFFOLD_METHODS = {
    "scaffold-brics": "brics",
    "scaffold-random-hetero": "random-hetero",
}
SCAFFOLD_SELECTORS = {
    "brics": select_brics_scaffold_indices,
    "random-hetero": select_hetero_scaffold_indices,
}
# Flat {index: [atoms]} files written by split_moses_scaffold_indices.py
SCAFFOLD_PATHS = {
    "brics": DATA_DIR / "scaffold_atom_indices_brics.json",
    "random-hetero": DATA_DIR / "scaffold_atom_indices_random_hetero.json",
}
MOLECULE_IDX_COLUMN = "reference_index"


def read_indexed_json(path: Path, name: str) -> dict[int, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object mapping reference indices to values")
    try:
        return {int(key): item for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} keys must be integer reference indices") from error


def read_flat_index_lists(path: Path, name: str) -> dict[int, list[int]]:
    """Read a generate.py-style ``{reference index: [ints]}`` JSON object."""
    raw = read_indexed_json(path, name)
    try:
        return {index: [int(item) for item in items] for index, items in raw.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be lists of integers") from error


def calculate_summary_statistics(combined_df: pd.DataFrame) -> dict:
    """Summary statistics."""
    total_samples = int(len(combined_df))
    valid_post = (
        int((~combined_df["molblocks_post_opt"].isna()).sum()) if total_samples else 0
    )
    validity_post_rate = float(valid_post / total_samples) if total_samples > 0 else 0.0

    non_nan_df = combined_df.dropna(subset=["molblocks_post_opt"])
    filtered_df = non_nan_df[(non_nan_df['graph_similarities_post_opt'] <= 0.3)]

    stats = {
        "total_samples": total_samples,
        "validity_rate": validity_post_rate,
        "graph_similarity": np.nanmedian(combined_df['graph_similarities_post_opt']),
        "surface_similarity": np.nanmedian(filtered_df['sims_surf_target_relax_optimal']),
        "esp_similarity": np.nanmedian(filtered_df['sims_esp_target_relax_optimal']),
        "pharm_similarity": np.nanmedian(filtered_df['sims_pharm_target_relax_optimal']),
        "strain_energy": np.nanmedian(combined_df['strain_energies']),
        "SA_score": np.nanmedian(combined_df['SA_scores_post_opt']),
        "QED": np.nanmedian(combined_df['QEDs_post_opt']),
    }
    return stats


def evaluate_single_reference(work_item):
    """Evaluate all generated samples for one reference."""
    (
        index,
        pipeline_kwargs,
        rowwise_path,
        global_path,
        metadata,
    ) = work_item
    try:
        pipeline = ConditionalEvalPipeline(**pipeline_kwargs)
        blocker = rdBase.BlockLogs()
        try:
            pipeline.evaluate(num_workers=1, num_processes=1, verbose=False, timeout_minutes=15)
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
        return index, len(df_rowwise), None
    except Exception as error:
        return (
            index,
            None,
            f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--add-atoms", type=int, default=0)
    parser.add_argument("--add-pharms", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--eval-workers", type=int, default=20, help="Parallel workers for evaluation over each reference")
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument(
        "--scaffold-indices-path",
        type=Path,
        help="Flat {index: [atoms]} JSON; defaults to the committed file "
             "for the mode's selection method",
    )
    parser.add_argument("--pharm-prioritization-path", type=Path, default=PRIORITY_PATH)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--select-scaffold-on-the-fly",
        action="store_true",
        help="Reselect scaffold atoms from the MOSES molblocks instead of reading "
             "the committed spec",
    )

    args = parser.parse_args()
    if args.add_atoms < 0 or args.add_pharms < 0:
        parser.error("--add-atoms and --add-pharms must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.select_scaffold_on_the_fly and args.mode not in SCAFFOLD_METHODS:
        parser.error("--select-scaffold-on-the-fly only applies to a scaffold-* mode")
    if args.scaffold_indices_path is not None and args.mode not in SCAFFOLD_METHODS:
        parser.error("--scaffold-indices-path only applies to a scaffold-* mode")
    return args


def main() -> None:
    args = parse_args()
    selected = args.indices if args.indices is not None else list(range(NUM_REFERENCES))
    if len(selected) != len(set(selected)):
        raise ValueError("--indices must be unique")

    print('Running in mode: ', args.mode)

    print('Loading moses test references...')
    # Load moses test references
    # Expects a [(molblock, charges), ...]
    with open(args.test_path, "rb") as f:
        test_molblock_charges = pickle.load(f)
    test_molblock_charges = {i: v for i, v in enumerate(test_molblock_charges)}

    # Load pharmacophore prioritizations
    # Expects a {int: list[int]}
    pharm_prioritizations = read_indexed_json(
        args.pharm_prioritization_path, "pharmacophore prioritizations"
    )

    # Load scaffold indices
    # Expects a flat {int: list[int]} written by split_moses_scaffold_indices.py
    scaffold_indices: dict[int, list[int]] = {}
    scaffold_method = SCAFFOLD_METHODS.get(args.mode)
    if scaffold_method is not None and not args.select_scaffold_on_the_fly:
        if args.scaffold_indices_path is None:
            args.scaffold_indices_path = SCAFFOLD_PATHS[scaffold_method]
        scaffold_indices = read_flat_index_lists(
            args.scaffold_indices_path, "scaffold indices"
        )

    missing = [i for i in selected if i not in test_molblock_charges]
    if missing:
        raise ValueError(f"missing molblock/charges for indices {missing}")
    missing_scaffolds = [i for i in selected if i not in scaffold_indices]
    if scaffold_indices and missing_scaffolds:
        raise ValueError(f"missing scaffold indices for indices {missing_scaffolds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    print('Extracting interaction profiles...')
    # Extract profiles first
    prepared = {}
    for index in selected:
        molblock, charges = test_molblock_charges[index]
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            raise ValueError(f"reference {index} has an invalid molblock")
        charges = np.asarray(charges)
        if charges.shape != (mol.GetNumAtoms(),):
            raise ValueError(
                f"reference {index} has {len(charges)} charges for {mol.GetNumAtoms()} atoms"
            )

        # Extract interaction profile
        profile: InteractionProfile = extract_interaction_profile(
            mol,
            partial_charges=charges,
            xtb_optimize=False,
            condition_atom_inds=None,
        )

        if profile is None:
            raise RuntimeError(f"reference {index} interaction-profile extraction failed")

        pp_mask = [int(value) for value in pharm_prioritizations[index]]
        if len(pp_mask) != profile.n_pharms or any(value not in (0, 1) for value in pp_mask):
            raise ValueError(
                f"reference {index} priority vector must be 0/1 of length {profile.n_pharms}"
            )
        prepared[index] = (mol, charges, profile, pp_mask)

    model = load_model(
        local_checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )

    failed = {}
    evaluation_work = []
    pbar = tqdm(selected, total=len(selected), desc="Generating", disable=not args.verbose)
    for index in pbar:
        pbar.set_description(f"Generating reference {index:03d}")
        rowwise_path = args.output_dir / f"reference_{index:03d}_rowwise.pkl"
        global_path = args.output_dir / f"reference_{index:03d}_global.pkl"
        raw_samples = args.output_dir / f"checkpoints/reference_{index:03d}_samples.pkl"
        if (
            rowwise_path.exists()
            and global_path.exists()
            and raw_samples.exists()
            and not args.overwrite
        ):
            print(f"Skipping completed reference {index}")
            continue

        try:
            mol, charges, profile, pharm_prioritization_mask = prepared[index]
            condition = profile
            scaffold_conditioning = False
            pharmacophore_conditioning = False
            pharmacophore_prioritization = None

            if args.mode in SCAFFOLD_METHODS:
                method = SCAFFOLD_METHODS[args.mode]
                if args.select_scaffold_on_the_fly:
                    atom_indices = SCAFFOLD_SELECTORS[method](mol, index)
                else:
                    atom_indices = list(scaffold_indices[index])

                # add scaffold conditioning atom inds to the condition profile
                condition = profile.with_condition_atoms(atom_indices)
                scaffold_conditioning = True

            elif args.mode == "pharm-full":
                pharmacophore_conditioning = True

            elif args.mode == "pharm-priority":
                pharmacophore_conditioning = True
                pharmacophore_prioritization = pharm_prioritization_mask

            elif args.mode == "pharm-priority-only":
                # Drop low-priority pharmacophores instead of inpainting them.
                condition: InteractionProfile = profile.subselection_pharm(
                    pharm_prioritization_labels=pharm_prioritization_mask
                )
                pharmacophore_conditioning = True

            if raw_samples.exists() and not args.overwrite:
                with raw_samples.open("rb") as handle:
                    generated_samples: list[GeneratedSample] = pickle.load(handle)
            else:
                generated_samples = model.generate(
                    batch_size=args.batch_size,
                    N_x1=profile.n_atoms + args.add_atoms,
                    N_x4=profile.n_pharms + args.add_pharms,
                    condition=condition,
                    condition_modalities="all",
                    scaffold_conditioning=scaffold_conditioning,
                    pharmacophore_conditioning=pharmacophore_conditioning,
                    pharmacophore_prioritization=pharmacophore_prioritization,
                    verbose=False,
                )
                with raw_samples.open("wb") as handle:
                    pickle.dump(generated_samples, handle)

            # Convert the generated samples to shepherd-score inputs.
            # Evaluate against the full profile to score pharm similarity for each subset
            eval_pharmacophore_prioritization = (
                pharm_prioritization_mask
                if args.mode in ("pharm-priority", "pharm-priority-only")
                else None
            )
            pipeline_kwargs = model.to_shepherd_score_inputs(
                generated_samples,
                condition=profile,
                condition_modalities="all",
                pharmacophore_prioritization=eval_pharmacophore_prioritization,
                partial_charges=charges,
            )
            evaluation_work.append(
                (
                    index, pipeline_kwargs, rowwise_path, global_path,
                    {
                        "reference_index": index,
                        "mode": args.mode,
                        "add_atoms": args.add_atoms,
                        "add_pharms": args.add_pharms,
                    },
                )
            )

        except Exception as error:
            failed[index] = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
            print(f"Reference {index} failed: {error}", file=sys.stderr)

        finally:
            if failed:
                (args.output_dir / "failed_references.json").write_text(
                    json.dumps({str(key): value for key, value in failed.items()}, indent=2)
                    + "\n"
                )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('Starting evaluation...')
    if evaluation_work:
        worker_count = min(args.eval_workers, len(evaluation_work))
        spawn_context = mp.get_context("spawn")
        with spawn_context.Pool(worker_count) as pool:
            results = pool.imap_unordered(
                evaluate_single_reference,
                evaluation_work,
                chunksize=1,
            )
            for index, num_samples, error_message in tqdm(
                results,
                total=len(evaluation_work),
                desc="Evaluating references",
                disable=not args.verbose,
            ):
                if error_message is not None:
                    failed[index] = error_message
                    print(
                        f"Reference {index} failed: {error_message.splitlines()[0]}",
                        file=sys.stderr,
                    )
                if failed:
                    (args.output_dir / "failed_references.json").write_text(
                    json.dumps(
                        {str(key): value for key, value in failed.items()},
                        indent=2,
                    )
                    + "\n"
                )

    print('Combining results...')
    frames = []
    del_paths = []
    for index in selected:
        path = args.output_dir / f"reference_{index:03d}_rowwise.pkl"
        if path.exists():
            with path.open("rb") as handle:
                frames.append(pickle.load(handle))
            del_paths.append(path)
        path = args.output_dir / f"reference_{index:03d}_global.pkl"
        if path.exists():
            del_paths.append(path)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_path = args.output_dir / "combined_results.pkl"
    with combined_path.open("wb") as handle:
        pickle.dump(combined, handle)

    stats = calculate_summary_statistics(combined)
    stats_path = args.output_dir / "combined_results_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    for path in del_paths:
        path.unlink()

    print(f"Saved combined results to {combined_path}")
    print(f"Saved summary statistics to {stats_path}")
    print(f"  References evaluated: {stats.get('num_test_molecules', 0)}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Validity rate (post-opt): {stats['validity_rate']:.3%}")
    print("Done!")

if __name__ == "__main__":
    main()
