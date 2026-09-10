#!/usr/bin/env python
"""Generate the 200-sample MolGenBench conditional benchmark."""

from __future__ import annotations

import os
import argparse
import tempfile
import json
import math
import multiprocessing as mp
import pickle
import re
import sys
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

os.environ.setdefault("TMPDIR", tempfile.gettempdir())

import numpy as np
from rdkit import Chem
import torch
from tqdm import tqdm

from shepherd import load_model
from shepherd.extract import create_rdkit_molecule
from shepherd.interaction_profile import InteractionProfile, extract_interaction_profile
from shepherd.utils.generation import restore_frame

if TYPE_CHECKING:
    from shepherd import ShepherdModel

torch.set_float32_matmul_precision("high")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "conformers" / "molgenbench"
TEST_PATH = DATA_DIR / "molgenbench_inputs_20260609.pkl"
SCAFFOLD_PATH = DATA_DIR / "interaction_atom_indices.json"
PRIORITY_PATH = DATA_DIR / "interaction_pharm_prioritization.json"

# Experimented conditioning modes
MODES = (
    "conditional",
    "pharm-priority",
    "pharm-priority-scaffold",
)
# used for different combinations of added atoms and pharmacophores
PATTERN_TOKENS = (
    "a0p0",
    "a1p0",
    "a2p0",
    "a2p1",
    "a2p2",
    "a3p0",
    "a3p1",
    "a3p2",
    "a4p1",
    "a4p2",
)
_PATTERN_RE = re.compile(r"^a(\d+)p(\d+)$")
PATTERNS = tuple(
    tuple(int(value) for value in _PATTERN_RE.fullmatch(token).groups())
    for token in PATTERN_TOKENS
)


# Serialization helpers
def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_indexed_json(path: Path, name: str) -> dict[int, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object mapping reference indices to values")
    try:
        return {int(key): item for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} keys must be integer reference indices") from error


def write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_pickle(path: Path, value: Any) -> None:
    # Replace completed files atomically
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def write_reference_sdf(
    path: Path,
    molecules: list[Chem.Mol | None],
    reference_name: str,
) -> int:
    """Write valid converted molecules using generate.py's record names."""
    # Write to a temporary file so partial SDFs are never treated as complete
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    writer = Chem.SDWriter(str(temporary_path))
    num_valid = 0
    try:
        for sample_index, mol in enumerate(molecules):
            if mol is None:
                continue
            output_mol = Chem.Mol(mol)
            output_mol.SetProp(
                "_Name",
                f"{reference_name}_sample_{sample_index:03d}",
            )
            writer.write(output_mol)
            num_valid += 1
    finally:
        writer.close()
    temporary_path.replace(path)
    return num_valid


def reference_name(mol: Chem.Mol, index: int) -> str:
    """Match generate.py's input-name handling."""
    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    return name or f"reference_{index:03d}"


def reference_paths(results_dir: Path, index: int) -> tuple[Path, Path, Path]:
    """Return the generate.py-style reference, checkpoint, and SDF paths."""
    reference_dir = results_dir / f"reference_{index:03d}"
    return (
        reference_dir,
        reference_dir / "samples.pkl",
        reference_dir / "molecules.sdf",
    )


def convert_samples_to_mols(
    samples: list,
    eval_pool,
    eval_workers: int,
) -> list[Chem.Mol | None]:
    """Convert generated samples to RDKit molecules in parallel"""
    chunksize = max(1, math.ceil(len(samples) / (eval_workers * 4)))
    return list(
        eval_pool.imap(create_rdkit_molecule, samples, chunksize=chunksize))


def balanced_batch_repeats(num_samples: int, batch_size: int) -> list[int]:
    """Split equal copies of all ten patterns across balanced generation calls."""
    # Keep every pattern equally represented in every call
    num_patterns = len(PATTERNS)
    if num_samples % num_patterns:
        raise ValueError(
            f"--num-samples must be divisible by {num_patterns} so every "
            "aXpY pattern has equal representation"
        )
    if batch_size < num_patterns:
        raise ValueError(
            f"--batch-size must be at least {num_patterns} so every generation "
            "call includes all aXpY patterns"
        )

    repeats_per_pattern = num_samples // num_patterns
    max_repeats_per_call = batch_size // num_patterns
    num_calls = math.ceil(repeats_per_pattern / max_repeats_per_call)
    base, remainder = divmod(repeats_per_pattern, num_calls)
    return [base + (call_index < remainder) for call_index in range(num_calls)]


def validate_priority_mask(value: Any, n_pharms: int, index: int) -> list[int]:
    """Masks must align with the extracted pharmacophores"""
    vector = list(value)
    if len(vector) != n_pharms:
        raise ValueError(
            f"reference {index} priority vector has length {len(vector)}; expected {n_pharms}"
        )
    if any(item not in (0, 1, False, True) for item in vector):
        raise ValueError(f"reference {index} priority vector must contain only 0 and 1")
    return [int(item) for item in vector]


def validate_scaffold_indices(value: Any, n_atoms: int, index: int) -> list[int]:
    """Scaffold indices refer to atoms in the input molblock"""
    try:
        indices = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"reference {index} scaffold indices must be integers"
        ) from error
    if len(indices) != len(set(indices)):
        raise ValueError(f"reference {index} scaffold indices contain duplicates")
    if any(item < 0 or item >= n_atoms for item in indices):
        raise ValueError(f"reference {index} scaffold indices are out of bounds")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=200,
       help="Samples per reference, divided equally among the ten aXpY patterns",
    )
    parser.add_argument("--batch-size", type=int, default=50,
        help="Maximum generation batch size (actual balanced batches may be smaller)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=min(10, mp.cpu_count()),
        help="Worker processes used to convert generated samples to RDKit molecules",
    )
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--scaffold-indices-path", type=Path, default=SCAFFOLD_PATH)
    parser.add_argument("--pharm-prioritization-path", type=Path, default=PRIORITY_PATH)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.num_samples < len(PATTERNS):
        parser.error(f"--num-samples must be at least {len(PATTERNS)}")
    if args.num_samples > 200:
        parser.error("--num-samples cannot exceed 200")
    if args.eval_workers < 1:
        parser.error("--eval-workers must be positive")
    try:
        balanced_batch_repeats(args.num_samples, args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    return args


def _generation_config(args: argparse.Namespace, batch_repeats: list[int]) -> dict:
    """Record both user arguments and the resolved batch plan"""
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "patterns": list(PATTERN_TOKENS),
            "batch_sizes": [
                repeats * len(PATTERNS) for repeats in batch_repeats
            ],
            "neutralize_esp": True,
            "output_format": "generate_results_per_reference_molecules_sdf",
        }
    )
    return config


def _check_existing_config(path: Path, config: dict) -> None:
    """Prevent incompatible runs from sharing checkpoints"""
    if not path.exists():
        return
    with path.open() as handle:
        previous = json.load(handle)
    immutable_keys = (
        "mode",
        "num_samples",
        "test_path",
        "scaffold_indices_path",
        "pharm_prioritization_path",
        "patterns",
        "neutralize_esp",
        "output_format",
    )
    differences = [
        key for key in immutable_keys if previous.get(key) != config.get(key)
    ]
    if differences:
        raise ValueError(
            "output directory contains an incompatible run_config.json "
            f"(different {', '.join(differences)}); use a new output directory"
        )


def _prepare_profile(record: Any, index: int) -> tuple[InteractionProfile, np.ndarray]:
    """Prepare the interaction profile for a given record."""
    if not isinstance(record, (list, tuple)) or len(record) != 2:
        raise ValueError(f"reference {index} must be a (molblock, partial_charges) pair")
    molblock, partial_charges = record
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    if mol is None:
        raise ValueError(f"reference {index} has an invalid molblock")
    partial_charges = np.asarray(partial_charges)
    if partial_charges.shape != (mol.GetNumAtoms(),):
        raise ValueError(
            f"reference {index} has {len(partial_charges)} charges for "
            f"{mol.GetNumAtoms()} atoms"
        )

    # Center the profile and neutralize charged reference ESPs
    profile = extract_interaction_profile(
        mol,
        partial_charges=partial_charges,
        xtb_optimize=False,
        neutralize_esp=True,
    )
    if profile is None:
        raise RuntimeError(f"reference {index} interaction-profile extraction failed")
    return profile, partial_charges


def _generate_reference(
    model: ShepherdModel,
    profile: InteractionProfile,
    *,
    mode: str,
    priority_mask: list[int] | None,
    scaffold_indices: list[int] | None,
    batch_repeats: list[int],
) -> list[dict]:
    """Generate molecules for a given reference profile."""
    condition = profile
    scaffold_conditioning = False
    pharmacophore_conditioning = mode != "conditional"

    # An empty scaffold falls back to normal interaction-profile conditioning
    if mode == "pharm-priority-scaffold" and scaffold_indices:
        condition = profile.with_condition_atoms(scaffold_indices)
        scaffold_conditioning = True

    generated_molecules = []
    for repeats in batch_repeats:
        # Build ragged node counts from complete pattern sets
        batch_patterns = [
            pattern
            for _ in range(repeats)
            for pattern in PATTERNS
        ]
        n_x1 = [profile.n_atoms + add_atoms for add_atoms, _ in batch_patterns]
        n_x4 = [profile.n_pharms + add_pharms for _, add_pharms in batch_patterns]
        samples = model.generate(
            batch_size=len(batch_patterns),
            N_x1=n_x1,
            N_x4=n_x4,
            condition=condition,
            condition_modalities="all",
            scaffold_conditioning=scaffold_conditioning,
            pharmacophore_conditioning=pharmacophore_conditioning,
            # An all-zero mask falls back to normal pharmacophore inpainting
            pharmacophore_prioritization=priority_mask,
            verbose=False,
        )
        if len(samples) != len(batch_patterns):
            raise RuntimeError(
                f"generation returned {len(samples)} samples for a batch of "
                f"{len(batch_patterns)}"
            )
        # Keep only the atom data needed for later RDKit conversion
        generated_molecules.extend(
            {
                "x1": {
                    "atoms": np.asarray(sample["x1"]["atoms"]),
                    "positions": np.asarray(sample["x1"]["positions"]),
                }
            }
            for sample in samples
        )
    return generated_molecules


def main() -> None:
    args = parse_args()
    batch_repeats = balanced_batch_repeats(args.num_samples, args.batch_size)

    # Load references and select the requested subset
    records = read_pickle(args.test_path)
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("--test-path must contain a non-empty sequence of records")
    selected = args.indices if args.indices is not None else list(range(len(records)))
    if len(selected) != len(set(selected)):
        raise ValueError("--indices must be unique")
    invalid = [index for index in selected if index < 0 or index >= len(records)]
    if invalid:
        raise ValueError(f"--indices contains out-of-range values: {invalid}")

    # Load only the metadata needed by the selected mode
    priorities = None
    if args.mode != "conditional":
        priorities = read_indexed_json(
            args.pharm_prioritization_path, "pharmacophore prioritizations"
        )
        missing = [index for index in selected if index not in priorities]
        if missing:
            raise ValueError(f"pharmacophore prioritizations missing indices: {missing}")

    scaffolds = None
    if args.mode == "pharm-priority-scaffold":
        scaffolds = read_indexed_json(args.scaffold_indices_path, "scaffold indices")
        missing = [index for index in selected if index not in scaffolds]
        if missing:
            raise ValueError(f"scaffold indices missing indices: {missing}")

    # Use generate.py's per-reference output layout. samples.pkl is temporary and is
    # removed once molecules.sdf has been written successfully.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    config = _generation_config(args, batch_repeats)
    _check_existing_config(config_path, config)
    write_json(config_path, config)

    print(f"Running in mode: {args.mode}")
    print(
        "Balanced generation calls per reference: "
        + " + ".join(str(repeats * len(PATTERNS)) for repeats in batch_repeats)
        + f" = {args.num_samples}"
    )
    print("Automatically neutralizing charged reference ESPs")
    print("Loading model...")
    model = load_model(device=args.device)

    # Generate each reference independently so completed work can be reused
    failed = {}
    progress = tqdm(
        selected,
        total=len(selected),
        desc="Generating",
        disable=not args.verbose,
    )
    with torch.inference_mode():
        for index in progress:
            progress.set_description(f"Generating reference {index:03d}")
            reference_dir, checkpoint_path, sdf_path = reference_paths(
                results_dir, index
            )
            reference_dir.mkdir(parents=True, exist_ok=True)
            if sdf_path.exists() and not args.overwrite:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                continue
            if checkpoint_path.exists() and not args.overwrite:
                generated = read_pickle(checkpoint_path)
                if len(generated) != args.num_samples:
                    raise ValueError(
                        f"{checkpoint_path} contains {len(generated)} samples; "
                        f"expected {args.num_samples}. Use --overwrite to replace it."
                    )
                continue
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if sdf_path.exists():
                sdf_path.unlink()

            try:
                # Validate condition metadata against the extracted profile
                profile, _ = _prepare_profile(records[index], index)
                priority_mask = (
                    None
                    if priorities is None
                    else validate_priority_mask(
                        priorities[index], profile.n_pharms, index
                    )
                )
                scaffold_indices = (
                    None
                    if scaffolds is None
                    else validate_scaffold_indices(
                        scaffolds[index], profile.n_atoms, index
                    )
                )
                if (
                    args.mode == "pharm-priority-scaffold"
                    and not scaffold_indices
                ):
                    tqdm.write(
                        f"Reference {index}: empty scaffold; using pharmacophore "
                        "prioritization without scaffold conditioning"
                    )

                generated = _generate_reference(
                    model,
                    profile,
                    mode=args.mode,
                    priority_mask=priority_mask,
                    scaffold_indices=scaffold_indices,
                    batch_repeats=batch_repeats,
                )
                if len(generated) != args.num_samples:
                    raise RuntimeError(
                        f"reference {index} produced {len(generated)} samples; "
                        f"expected {args.num_samples}"
                    )
                write_pickle(checkpoint_path, generated)
            except Exception as error:
                failed[index] = (
                    f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                )
                tqdm.write(f"Reference {index} failed: {error}", file=sys.stderr)
                write_json(
                    args.output_dir / "failed_references.json",
                    {str(key): value for key, value in failed.items()},
                )

    # Release the GPU model before starting CPU conversion workers
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert generated samples after all GPU generation is complete
    conversion_indices = [
        index
        for index in selected
        if reference_paths(results_dir, index)[1].exists()
        and not reference_paths(results_dir, index)[2].exists()
    ]
    if conversion_indices:
        spawn_context = mp.get_context("spawn")
        with spawn_context.Pool(args.eval_workers) as eval_pool:
            conversion_progress = tqdm(
                conversion_indices,
                desc="Converting",
                disable=not args.verbose,
            )
            for index in conversion_progress:
                conversion_progress.set_description(
                    f"Converting reference {index:03d}"
                )
                _, checkpoint_path, sdf_path = reference_paths(results_dir, index)
                try:
                    generated = read_pickle(checkpoint_path)
                    molecules = convert_samples_to_mols(
                        generated,
                        eval_pool,
                        args.eval_workers,
                    )
                    molblock, _ = records[index]
                    reference_mol = Chem.MolFromMolBlock(molblock, removeHs=False)
                    if reference_mol is None:
                        raise ValueError(f"reference {index} has an invalid molblock")
                    # Sampling is in the centered profile frame; shift back so
                    # the SDF overlays the original reference coordinates.
                    ref_com = np.mean(reference_mol.GetConformer().GetPositions(), axis=0)
                    valid = [i for i, mol in enumerate(molecules) if mol is not None]
                    if valid:
                        restored = restore_frame([molecules[i] for i in valid], ref_com)
                        for i, mol in zip(valid, restored):
                            molecules[i] = mol
                    num_valid = write_reference_sdf(
                        sdf_path,
                        molecules,
                        reference_name(reference_mol, index),
                    )
                    checkpoint_path.unlink()
                    tqdm.write(
                        f"Reference {index}: saved {num_valid} valid molecules "
                        f"from {len(generated)} samples"
                    )
                except Exception as error:
                    failed[index] = (
                        f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                    )
                    tqdm.write(
                        f"Reference {index} conversion failed: {error}",
                        file=sys.stderr,
                    )
                    write_json(
                        args.output_dir / "failed_references.json",
                        {str(key): value for key, value in failed.items()},
                    )

    num_sdf_files = sum(
        reference_paths(results_dir, index)[2].exists()
        for index in selected
    )
    if failed:
        print(f"Generation completed with {len(failed)} failed references.")
    else:
        # Remove failures left over from an earlier successful retry
        failed_path = args.output_dir / "failed_references.json"
        if failed_path.exists():
            failed_path.unlink()
        print("Generation complete.")
    print(
        f"Saved {num_sdf_files} per-reference SDF files to {results_dir}"
    )


if __name__ == "__main__":
    main()
