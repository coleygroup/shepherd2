#!/usr/bin/env python
"""Generate and evaluate the 100-reference MOSES conditional benchmark.

    python examples/experiments/moses_aq/run_moses_conditional_evals_via_generate.py \
        --mode pharm-priority --output-dir out/moses_pharm_priority
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

# Module level, not inside main(): --parallel-over "reference" pickles generate.py
# functions and the spawned children re-import this module to resolve them.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")) # src first is important
sys.path.insert(0, str(ROOT))

import generate  # noqa: E402
from shepherd.data_utils.subgraph import (  # noqa: E402
    select_brics_scaffold_indices,
    select_hetero_scaffold_indices,
)

DATA_DIR = ROOT / "data" / "conformers" / "moses_aq"
TEST_PATH = DATA_DIR / "moses_test_scaffold_molblock_charges.pkl"
PRIORITY_PATH = DATA_DIR / "interaction_pharm_prioritization.json"

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
SCAFFOLD_PATHS = {
    "brics": DATA_DIR / "scaffold_atom_indices_brics.json",
    "random-hetero": DATA_DIR / "scaffold_atom_indices_random_hetero.json",
}
MOLECULE_IDX_COLUMN = "reference_index"


def calculate_summary_statistics(combined_df: pd.DataFrame) -> dict:
    """Summary statistics."""
    total_samples = int(len(combined_df))
    valid_post = (
        int((~combined_df["molblocks_post_opt"].isna()).sum()) if total_samples else 0
    )
    validity_post_rate = float(valid_post / total_samples) if total_samples > 0 else 0.0

    non_nan_df = combined_df.dropna(subset=["molblocks_post_opt"])
    filtered_df = non_nan_df[(non_nan_df['graph_similarities_post_opt'] < 0.3)]

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


def select_scaffold_atoms(test_path: Path, method: str, indices: list[int] | None) -> dict:
    """Reselect scaffold atoms from the molblock pickle, seeded by reference index.

    Indexes the same RDKit mol generate.py will parse: the same molblock, the same
    ``removeHs=False``. generate.py's check_atom_indices confirms the atom count
    still matches when it applies these.
    """
    with test_path.open("rb") as handle:
        records = pickle.load(handle)
    selector = SCAFFOLD_SELECTORS[method]
    wanted = range(len(records)) if indices is None else indices

    resolved = {}
    for index in wanted:
        if index >= len(records):
            raise ValueError(f"--indices entry {index} is past the end of {test_path}")
        molblock, _ = records[index]
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None:
            raise ValueError(f"reference {index} has an invalid molblock")
        resolved[str(index)] = [int(atom) for atom in selector(mol, index)]
    return resolved


def annotate_combined(args: argparse.Namespace) -> None:
    """Reshape generate.py's combined_rowwise.pkl into combined results and stats."""
    rowwise_path = args.output_dir / "combined_rowwise.pkl"
    if not rowwise_path.exists():
        print(f"No {rowwise_path}; nothing to combine", file=sys.stderr)
        return

    with rowwise_path.open("rb") as handle:
        combined = pickle.load(handle)
    combined = combined.rename(columns={"ref_ind": MOLECULE_IDX_COLUMN}).assign(
        mode=args.mode, add_atoms=args.add_atoms, add_pharms=args.add_pharms
    )

    combined_path = args.output_dir / "combined_results.pkl"
    with combined_path.open("wb") as handle:
        pickle.dump(combined, handle)

    stats = calculate_summary_statistics(combined)
    stats_path = args.output_dir / "combined_results_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Saved combined results to {combined_path}")
    print(f"Saved summary statistics to {stats_path}")
    print(f"  References evaluated: {combined[MOLECULE_IDX_COLUMN].nunique()}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Validity rate (post-opt): {stats['validity_rate']:.3%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--add-atoms", type=int, default=0)
    parser.add_argument("--add-pharms", type=int, default=0)
    parser.add_argument("--condition-com", choices=generate.CONDITION_COM_CHOICES,
                        default="origin",
                        help="Frame the condition is generated in. 'origin' (default, "
                             "recommended) keeps the reference's own frame so samples overlay it directly; "
                             "'auto' centers the scaffold/pharmacophore condition for "
                             "denoising and shifts the samples back afterwards")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Samples per model call")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-workers", type=int, default=20,
                        help="Parallel workers for evaluation")
    parser.add_argument("--eval-parallelism", choices=("reference", "sample"),
                        default="reference",
                        help="Spend --eval-workers on whole references (default) "
                             "or on the samples of one reference at a time")
    parser.add_argument("--xtb-timeout", type=int, default=900,
                        help="Per-molecule xTB timeout in seconds (default: 15 min)")
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--scaffold-indices-path", type=Path,
                        help="Flat {index: [atoms]} JSON; defaults to the committed file "
                             "for the mode's selection method")
    parser.add_argument("--pharm-prioritization-path", type=Path, default=PRIORITY_PATH)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--select-scaffold-on-the-fly", action="store_true",
                        help="Reselect scaffold atoms from --test-path instead of reading "
                             "the committed spec")
    args = parser.parse_args()

    if args.add_atoms < 0 or args.add_pharms < 0:
        parser.error("--add-atoms and --add-pharms must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.eval_workers < 1:
        parser.error("--eval-workers must be positive")
    if args.select_scaffold_on_the_fly and args.mode not in SCAFFOLD_METHODS:
        parser.error("--select-scaffold-on-the-fly only applies to a scaffold-* mode")
    if args.scaffold_indices_path is not None and args.mode not in SCAFFOLD_METHODS:
        parser.error("--scaffold-indices-path only applies to a scaffold-* mode")
    return args


def resolve_scaffold_path(args: argparse.Namespace) -> Path | None:
    """The flat --scaffold-atoms file for this mode, writing it first if reselecting."""
    method = SCAFFOLD_METHODS.get(args.mode)
    if method is None:
        return None
    if not args.select_scaffold_on_the_fly:
        return args.scaffold_indices_path or SCAFFOLD_PATHS[method]

    resolved = select_scaffold_atoms(args.test_path, method, args.indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "scaffold_atoms_resolved.json"
    path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(f"Selected {method} scaffolds for {len(resolved)} reference(s) -> {path}")
    return path


def build_argv(args: argparse.Namespace, scaffold_path: Path | None) -> list[str]:
    """The generate.py command line this run is equivalent to."""
    argv = [
        "--molblock-charges", str(args.test_path),
        "--output-dir", str(args.output_dir),
        "--n-samples", "20",
        "--batch-size", str(args.batch_size),
        "--add-atoms", str(args.add_atoms),
        "--add-pharms", str(args.add_pharms),
        "--condition-com", args.condition_com,
        "--device", args.device,
        "--seed", str(args.seed),
        "--num-workers", str(args.eval_workers),
        "--parallel-over", args.eval_parallelism,
        "--xtb-timeout", str(args.xtb_timeout),
        "--evaluate",
    ]
    if args.checkpoint is not None:
        argv += ["--checkpoint", str(args.checkpoint)]
    if scaffold_path is not None:
        argv += ["--scaffold-atoms", str(scaffold_path)]
    if args.mode.startswith("pharm-"):
        argv += ["--pharm-mode", args.mode]
        # pharm-full fixes every extracted pharmacophore and takes no selection
        if args.mode != "pharm-full":
            argv += ["--pharm-masks", str(args.pharm_prioritization_path)]
    if args.indices is not None:
        argv += ["--indices", *(str(index) for index in args.indices)]
    if args.overwrite:
        argv.append("--overwrite")
    argv.append("--verbose" if args.verbose else "--no-verbose")
    return argv


def main() -> int:
    args = parse_args()
    scaffold_path = resolve_scaffold_path(args)
    argv = build_argv(args, scaffold_path)
    print(f"Running in mode: {args.mode}")
    print("generate.py " + " ".join(argv))

    status = generate.run(
        generate.parse_args(argv),
        config_extra={
            "benchmark": "moses_aq",
            "mode": args.mode,
            "test_path": str(args.test_path),
            "scaffold_indices_path": None if scaffold_path is None else str(scaffold_path),
            "pharm_prioritization_path": str(args.pharm_prioritization_path),
            "select_scaffold_on_the_fly": args.select_scaffold_on_the_fly,
        },
    )
    annotate_combined(args)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
