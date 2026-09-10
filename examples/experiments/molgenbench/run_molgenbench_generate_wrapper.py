#!/usr/bin/env python
"""Generate molecules for the 200-sample MolGenBench conditional benchmark.

    python examples/experiments/molgenbench/run_molgenbench_generation_via_generate.py \
        --mode pharm-priority-scaffold --output-dir out/molgenbench_pp_scaffold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Module level, not inside main(): --parallel-over reference pickles generate.py functions
# by reference, and the spawned children re-import this module to resolve them.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))  # src FIRST -- see CLAUDE.md's PYTHONPATH gotcha
sys.path.insert(0, str(ROOT))

import generate  # noqa: E402

DATA_DIR = ROOT / "data" / "conformers" / "molgenbench"
TEST_PATH = DATA_DIR / "molgenbench_inputs_20260609.pkl"
SCAFFOLD_PATH = DATA_DIR / "interaction_atom_indices.json"
PRIORITY_PATH = DATA_DIR / "interaction_pharm_prioritization.json"

MODES = ("conditional", "pharm-priority", "pharm-priority-scaffold")

# The ten (added atoms, added pharmacophores) combinations the benchmark samples over
PATTERN_TOKENS = ("a0p0", "a1p0", "a2p0", "a2p1", "a2p2",
                  "a3p0", "a3p1", "a3p2", "a4p1", "a4p2")
PATTERNS = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (4, 1), (4, 2))


def pattern_plan(n_samples: int) -> list[tuple[int, int]]:
    """The per-sample pattern order: whole cycles of all ten patterns, back to back.

    Pattern-major rather than pattern-blocked is what keeps generate.py's chunks balanced
    for free -- with --batch-size a multiple of ten, every chunk is a whole number of
    complete cycles, which is what balanced_batch_repeats used to arrange by hand.
    """
    return list(PATTERNS) * (n_samples // len(PATTERNS))


def refine_conditioning(ref, cond, *, n_samples: int):
    """Replace the flat node counts with this reference's aXpY grid."""
    plan = pattern_plan(n_samples)
    return cond.with_node_counts(
        n_x1=[cond.n_x1_base + add_atoms for add_atoms, _ in plan],
        n_x4=[cond.n_x4_base + add_pharms for _, add_pharms in plan],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Samples per reference, divided equally among the ten aXpY "
                             "patterns (a positive multiple of 10, at most 200)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Samples per model call; a multiple of 10, "
                             "so every chunk holds an entire add atom/pharm pattern.")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-workers", type=int, default=10,
                        help="Worker processes converting samples to RDKit molecules")
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--scaffold-indices-path", type=Path, default=SCAFFOLD_PATH)
    parser.add_argument("--pharm-prioritization-path", type=Path, default=PRIORITY_PATH)
    parser.add_argument("--indices", type=int, nargs="+",
                        help="Reference positions in --test-path to run (default: all)")
    parser.add_argument("--start-index", type=int,
                        help="Alternative to --indices: run range(start, end)")
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    num_patterns = len(PATTERNS)
    if args.num_samples < num_patterns or args.num_samples % num_patterns:
        parser.error(f"--num-samples must be a positive multiple of {num_patterns} so "
                     "every aXpY pattern is equally represented")
    if args.num_samples > 200:
        parser.error("--num-samples cannot exceed 200")
    if args.batch_size < num_patterns or args.batch_size % num_patterns:
        parser.error(f"--batch-size must be a multiple of {num_patterns} (at least "
                     f"{num_patterns}) so every chunk holds whole pattern cycles")
    if args.eval_workers < 1:
        parser.error("--eval-workers must be positive")

    if (args.start_index is None) != (args.end_index is None):
        parser.error("--start-index and --end-index must be given together")
    if args.start_index is not None:
        if args.indices is not None:
            parser.error("--indices and --start-index/--end-index are mutually exclusive")
        if args.start_index < 0 or args.end_index <= args.start_index:
            parser.error("--end-index must be greater than a non-negative --start-index")
        args.indices = list(range(args.start_index, args.end_index))
    return args


def build_argv(args: argparse.Namespace) -> list[str]:
    """The generate.py command line this run is equivalent to."""
    argv = [
        "--molblock-charges", str(args.test_path),
        "--output-dir", str(args.output_dir),
        "--n-samples", str(args.num_samples),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--seed", str(args.seed),
        "--num-workers", str(args.eval_workers),
        # Fixed by the benchmark: charged references are neutralized, samples are converted
        # without xTB, and whole references are converted in parallel.
        "--neutralize-esp",
        "--sample-xtb", "none",
        "--parallel-over", "reference",
    ]
    if args.checkpoint is not None:
        argv += ["--checkpoint", str(args.checkpoint)]
    if args.mode != "conditional":
        # Priority pharmacophores are fixed and the rest are inpainted, equivalent to
        # pharmacophore_conditioning=True plus a 0/1 mask.
        argv += ["--pharm-mode", "pharm-priority",
                 "--pharm-masks", str(args.pharm_prioritization_path)]
    if args.mode == "pharm-priority-scaffold":
        argv += ["--scaffold-atoms", str(args.scaffold_indices_path)]
    if args.indices is not None:
        argv += ["--indices", *(str(index) for index in args.indices)]
    if args.overwrite:
        argv.append("--overwrite")
    argv.append("--verbose" if args.verbose else "--no-verbose")
    return argv


def main() -> int:
    args = parse_args()
    argv = build_argv(args)
    print(f"Running in mode: {args.mode}")
    print("generate.py " + " ".join(argv))
    print(f"Node counts per reference: {len(PATTERNS)} aXpY patterns x "
          f"{args.num_samples // len(PATTERNS)} repeats = {args.num_samples} samples")

    return generate.run(
        generate.parse_args(argv),
        refine_conditioning=lambda ref, cond: refine_conditioning(
            ref, cond, n_samples=args.num_samples
        ),
        config_extra={
            "benchmark": "molgenbench",
            "mode": args.mode,
            "patterns": list(PATTERN_TOKENS),
            "test_path": str(args.test_path),
            "scaffold_indices_path": str(args.scaffold_indices_path),
            "pharm_prioritization_path": str(args.pharm_prioritization_path),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
