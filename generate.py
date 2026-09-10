#!/usr/bin/env python
"""Interaction-conditioned generation with ShEPhERD-2.

Takes one or more reference molecules (inline SMILES, a SMILES file, a 3D SDF, or a
pickle of already-prepared molblocks with their xTB partial charges), extracts an
interaction profile from each, conditionally generates samples, and writes the valid
molecules to one SDF per reference. Optionally scores the samples against the
conditioning profile with shepherd-score's ConditionalEvalPipeline.

Examples
--------
    python generate.py --smiles CCO CCN --output-dir out/
    python generate.py --sdf refs.sdf --condition shape --condition pharm \
        --evaluate --output-dir out/
    python generate.py --sdf refs.sdf --scaffold-atoms scaffold.json \
        --pharm-mode pharm-priority --pharm-atoms pharm.json --output-dir out/
    python generate.py --sdf refs.sdf --add-atoms-range -5 10 --output-dir out/
    python generate.py --molblock-charges refs.pkl --indices 0 1 2 --output-dir out/
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import tempfile
import traceback
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from tqdm import tqdm

from shepherd import load_model
from shepherd.extract import create_rdkit_molecule, mol_charges_from_samples
from shepherd.interaction_profile import InteractionProfile, extract_interaction_profile
from shepherd.utils.generation import (
    Conditioning,
    ConditioningSpecs,
    PendingReference,
    Reference,
    apply_node_floor,
    canonical_smiles,
    chunk_node_counts,
    chunk_sizes,
    comparable_setting,
    has_molecules,
    load_chunk,
    modalities_for_json,
    read_json_dict,
    read_previous_config,
    read_index_spec,
    read_reference_entries,
    read_reference_sources,
    read_smiles_file,
    reference_complete,
    reference_dir,
    reference_fingerprint,
    reference_name,
    resolve_modalities,
    resolve_node_range,
    resolve_workers,
    restore_frame,
    trim_sample,
    write_json,
    write_sdf,
)
from shepherd_score.conformer_generation import (
    embed_conformer_from_smiles,
)
from shepherd_score.evaluations.evaluate.pipelines import ConditionalEvalPipeline

os.environ.setdefault("TMPDIR", tempfile.gettempdir())

torch.set_float32_matmul_precision("high")

CONDITION_CHOICES = ("all", "shape", "esp", "pharm")
PHARM_MODES = ("pharm-full", "pharm-priority", "pharm-priority-only")
SAMPLE_XTB_MODES = ("relax", "charges", "none")
CONDITION_COM_CHOICES = ("origin", "auto")

SMILES_WARNING = (
    "WARNING: SMILES input -- conditioning on one arbitrary RDKit ETKDG/MMFF conformer "
    "(seed={seed}), not a bioactive pose; xTB-relaxed before profile extraction. "
    "Use --sdf to supply your own 3D structure."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input sources
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", nargs="+", help="One or more reference SMILES")
    source.add_argument("--smiles-file", type=Path,
        help="File with one SMILES per line; '#' comments and blank lines are skipped, "
        "an optional second whitespace-separated column is used as the name",
    )
    source.add_argument("--sdf", type=Path, help="SDF with one or more 3D references")
    source.add_argument("--molblock-charges", type=Path,
        help="Pickle of [(molblock, partial charges), ...] -- already-prepared 3D references "
        "whose xTB partial charges are supplied, so no xTB call is made for them. Same format "
        "as the references.pkl this script writes.",
    )

    # Output and idexing of the input references
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int,
        nargs="+",
        help="Positions in the input to run (default: all). Output is laid out by input "
        "position, so several --indices shards can share one --output-dir.",
    )

    # Reference preparation
    parser.add_argument("--relax-references", action=argparse.BooleanOptionalAction, default=None,
        help="Relax SDF references with xTB before extracting the profile (default: off, "
        "keeping the supplied pose). SMILES references are always relaxed.",
    )
    parser.add_argument("--neutralize-esp", action="store_true",
        help="Neutralize the conditioning ESP (recommended for charged references for MOSES model)",
    )
    parser.add_argument("--seed", type=int, default=1, help="RDKit embedding and torch/numpy seed")
    parser.add_argument("--solvent", default="water",
        help="Implicit solvent for xTB. Models were trained with water solvent."
    )

    # Conditioning
    parser.add_argument("--condition", choices=CONDITION_CHOICES,
        action="append",
        help="Modality to condition on; repeat to combine (default: all). "
        "Note 'esp' implies 'shape'.",
    )
    parser.add_argument("--add-atoms", type=int, help="Extra x1 nodes beyond the reference (default: 0)")
    parser.add_argument("--add-pharms", type=int, help="Extra x4 nodes beyond the reference (default: 0)")
    parser.add_argument("--add-atoms-range", type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Draw each sample's N_x1 uniformly from [n_atoms+MIN, n_atoms+MAX] instead of the "
        "fixed --add-atoms offset. Both bounds are offsets relative to the reference, are "
        "inclusive, and may be negative.",
    )
    parser.add_argument("--add-pharms-range", type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Draw each sample's N_x4 uniformly from [n_pharms+MIN, n_pharms+MAX] instead of the "
        "fixed --add-pharms offset. Same convention as --add-atoms-range.",
    )

    parser.add_argument("--scaffold-atoms", type=Path,
        help="JSON atom indices to hold fixed during generation (--sdf or --molblock-charges "
        "only). Either a list of lists applied in input order, or an object keyed by the "
        "reference's input position; a record with no entry is generated without scaffold "
        "conditioning.",
    )
    parser.add_argument("--pharm-mode", choices=PHARM_MODES,
        help="Pharmacophore conditioning (--sdf or --molblock-charges only). 'pharm-full' fixes every extracted "
        "pharmacophore; 'pharm-priority' fixes the selected ones and inpaints the rest; "
        "'pharm-priority-only' drops the rest instead of inpainting them.",
    )
    parser.add_argument("--pharm-atoms", type=Path,
        help="JSON atom indices whose pharmacophores are high priority, in the same two forms "
        "as --scaffold-atoms. Required by --pharm-mode pharm-priority and pharm-priority-only "
        "unless --pharm-masks is given.",
    )
    parser.add_argument("--pharm-masks", type=Path,
        help="JSON 0/1 masks, one per reference, each exactly as long as that reference's "
        "extracted pharmacophores. Alternative to --pharm-atoms.",
    )
    parser.add_argument("--min-ring-atoms", type=int,
        help="With --pharm-atoms, the minimum number of selected ring atoms before a whole "
        "aromatic or hydrophobe ring counts as high priority (default: 3)",
    )
    parser.add_argument("--condition-com", choices=CONDITION_COM_CHOICES, default="origin",
        help="Frame the scaffold/pharmacophore condition is generated in. 'origin' "
        "(default, recommended) generates in the reference's own frame, so samples overlay "
        "it directly. 'auto' shifts the condition to the COM of the scaffold/pharmacophore "
        "for denoising and shifts the samples back afterwards.",
    )

    # Generation
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per reference")
    parser.add_argument("--batch-size", type=int,
        help="Samples per model call (default: min(50, --n-samples)). Tune to your GPU; "
        "see the batch-size table in the README.",
    )

    # post-generation work/evals
    parser.add_argument("--sample-xtb", choices=SAMPLE_XTB_MODES, default="relax",
        help="xTB-based post-processing of generatd samples: 'relax' (default) geometry "
        "optimization, 'charges' computes single-point partial charges of the conformer, "
        "'none' just converts the geometry to a molecule. Ignored with --evaluate, which "
        "uses post-relaxed molecules from the shepherd-score pipeline's output.",
    )
    parser.add_argument("--xtb-timeout", type=int, default=300,
        help="Per-molecule xTB timeout in seconds, for both --sample-xtb and --evaluate",
    )
    parser.add_argument("--num-workers", type=int, default=-1,
        help="Processes for the post-generation conversion/evaluation of generated molecules "
        "(-1 selects automatically)",
    )
    parser.add_argument("--parallel-over", default="reference",
        choices=("reference", "sample"),
        help="Spend --num-workers on whole references in parallel (default) or on the samples "
        "of one reference at a time.",
    )

    parser.add_argument("--evaluate", action="store_true", help="Score samples with shepherd-score")

    parser.add_argument("--checkpoint", type=Path, help="Local checkpoint; omit to download the default")
    parser.add_argument("--cache-dir", type=Path, help="HuggingFace cache directory")
    parser.add_argument( "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu",
        help="CPU inference is very slow (especially compilation) and should be avoided.",
    )
    parser.add_argument("--save-samples", action="store_true", help="Also pickle the raw samples")
    parser.add_argument("--overwrite", action="store_true", help="Redo references that already have output")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args(argv)

    # arg validation
    if (args.add_atoms or 0) < 0 or (args.add_pharms or 0) < 0:
        parser.error("--add-atoms and --add-pharms must be non-negative; use --add-atoms-range / "
                     "--add-pharms-range for offsets that may go below the reference")
    if args.n_samples < 1:
        parser.error("--n-samples must be positive")
    if args.batch_size is None:
        args.batch_size = min(50, args.n_samples)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.seed < 1:
        parser.error("--seed must be positive (RDKit treats -1 as unseeded)")
    if args.indices is not None:
        if len(set(args.indices)) != len(args.indices):
            parser.error("--indices must be unique")
        if any(index < 0 for index in args.indices):
            parser.error("--indices must be non-negative input positions")
    # None means the user never asked either way, so this note stays quiet then
    if args.relax_references is False and (args.smiles or args.smiles_file):
        print("NOTE: --no-relax-references does not apply to SMILES input; "
              "xTB relaxation is always used.")
    args.relax_references = bool(args.relax_references)

    validate_conditioning_args(parser, args)
    return args


def validate_conditioning_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject unusable scaffold / pharmacophore combinations before any work is done."""
    supplied = [
        name
        for name, value in (
            ("--scaffold-atoms", args.scaffold_atoms),
            ("--pharm-mode", args.pharm_mode),
            ("--pharm-atoms", args.pharm_atoms),
            ("--pharm-masks", args.pharm_masks),
            ("--min-ring-atoms", args.min_ring_atoms),
        )
        if value is not None
    ]
    # Both controls select atoms of a supplied 3D structure; a SMILES reference is embedded
    # here, so its atom numbering is not something the user can have indexed in advance.
    if supplied and args.sdf is None and args.molblock_charges is None:
        parser.error(
            f"{', '.join(supplied)} index into the supplied structures and require "
            "--sdf or --molblock-charges"
        )

    if args.pharm_atoms is not None and args.pharm_masks is not None:
        parser.error("--pharm-atoms and --pharm-masks are mutually exclusive")

    if args.pharm_mode is None:
        stray = [name for name in supplied if name != "--scaffold-atoms"]
        if stray:
            parser.error(f"{', '.join(stray)} has no effect without --pharm-mode")
    elif args.pharm_mode == "pharm-full":
        if args.pharm_atoms is not None or args.pharm_masks is not None:
            parser.error(
                "--pharm-mode pharm-full fixes every extracted pharmacophore and takes no "
                "selection; use pharm-priority or pharm-priority-only to select a subset"
            )
    elif args.pharm_atoms is None and args.pharm_masks is None:
        parser.error(f"--pharm-mode {args.pharm_mode} requires --pharm-atoms or --pharm-masks")

    if args.min_ring_atoms is not None and args.min_ring_atoms < 1:
        parser.error("--min-ring-atoms must be at least 1")
    if args.min_ring_atoms is None:
        args.min_ring_atoms = 3

    for range_flag, offset_flag in (("add_atoms_range", "add_atoms"), ("add_pharms_range", "add_pharms")):
        bounds = getattr(args, range_flag)
        if bounds is not None and bounds[0] > bounds[1]:
            parser.error(f"--{range_flag.replace('_', '-')} needs MIN <= MAX, got {bounds}")
        if bounds is not None and getattr(args, offset_flag) is not None:
            print(
                f"NOTE: --{range_flag.replace('_', '-')} supersedes "
                f"--{offset_flag.replace('_', '-')}, which is ignored."
            )
    args.add_atoms = args.add_atoms or 0
    args.add_pharms = args.add_pharms or 0

    modalities = resolve_modalities(args.condition)
    if args.pharm_mode is not None and modalities != "all" and "pharm" not in modalities:
        parser.error(
            "--pharm-mode conditions on pharmacophores, so --condition must include "
            "'pharm' (or be left at the default 'all')"
        )


def require_xtb() -> None:
    if shutil.which("xtb") is None:
        raise RuntimeError(
            "xtb was not found on PATH. It is required for partial charges and relaxation; "
            "install it with `conda install conda-forge::xtb` or from source."
        )


def xtb_required(args: argparse.Namespace, references: list["Reference"],
                 cached: dict[int, tuple]) -> bool:
    """Whether this run needs xTB at all.

    Only if a reference is missing partial charges, or a pose or sample is being relaxed.
    """
    def has_charges(ref: "Reference") -> bool:
        return ref.charges is not None or cached.get(ref.index, (None, None))[1] is not None

    return (
        args.evaluate
        or args.sample_xtb != "none"
        or args.relax_references
        or not all(has_charges(ref) for ref in references)
    )


def record_failure(failed: dict[str, str], index: int, error: Exception) -> None:
    """Record one reference's failure with its traceback and report it on stderr."""
    failed[str(index)] = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
    print(f"Reference {index} failed: {error}", file=sys.stderr)


def read_molblock_charges(path: Path) -> list:
    """Read a ``[(molblock, partial charges), ...]`` pickle, as written by write_references."""
    try:
        with path.open("rb") as handle:
            records = pickle.load(handle)
    except Exception as error:
        raise ValueError(f"--molblock-charges: could not read {path} ({error})") from error
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError(
            f"--molblock-charges: {path} must hold a non-empty sequence of "
            "(molblock, partial charges) pairs"
        )
    return list(records)


def load_references(args: argparse.Namespace) -> tuple[list[Reference], dict[str, str]]:
    """Build the reference list from --smiles, --smiles-file, --sdf, or --molblock-charges.

    Reference indices track the position in the input, so a reference that cannot be
    loaded -- or that ``--indices`` left out -- leaves a gap. Returns the usable
    references plus a failure record for the rest.
    """
    references: list[Reference] = []
    failed: dict[str, str] = {}

    def reject(index: int, message: str) -> None:
        failed[str(index)] = message
        print(f"Reference {index} failed: {message}", file=sys.stderr)

    # --indices is applied before each record is parsed
    selected = None if args.indices is None else set(args.indices)
    matched: set[int] = set()

    def wanted(index: int) -> bool:
        if selected is None:
            return True
        if index not in selected:
            return False
        matched.add(index)
        return True

    if args.sdf is not None:
        supplier = Chem.SDMolSupplier(str(args.sdf), removeHs=False)
        for index in range(len(supplier)):
            if not wanted(index):
                continue
            mol = supplier[index]
            if mol is None:
                reject(index, f"RDKit could not parse SDF record {index}")
                continue
            if mol.GetNumConformers() == 0:
                reject(index, f"SDF record {index} has no 3D conformer")
                continue
            # GetTotalNumHs counts the hydrogens that are *not* in the graph, so a zero
            # total means the record is already all-explicit and is used as supplied
            missing_h = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
            if missing_h:
                print(
                    f"Reference {index}: adding {missing_h} implicit hydrogen(s) with "
                    "RDKit-placed coordinates; supply an all-atom SDF to control their geometry."
                )
                mol = Chem.AddHs(mol, addCoords=True)
            references.append(
                Reference(index, reference_name(mol, index), mol, from_smiles=False)
            )
    elif args.molblock_charges is not None:
        records = read_molblock_charges(args.molblock_charges)
        if args.relax_references:
            print(
                "NOTE: --relax-references discards the charges in "
                f"{args.molblock_charges}; the pose is xTB-relaxed and its charges recomputed."
            )
        for index, record in enumerate(records):
            if not wanted(index):
                continue
            if record is None:
                reject(index, f"record {index} is None")
                continue
            if not isinstance(record, (list, tuple)) or len(record) != 2:
                reject(index, f"record {index} is not a (molblock, partial charges) pair")
                continue
            molblock, charges = record
            mol = Chem.MolFromMolBlock(molblock, removeHs=False)
            if mol is None:
                reject(index, f"RDKit could not parse the molblock of record {index}")
                continue
            if mol.GetNumConformers() == 0:
                reject(index, f"record {index} has no 3D conformer")
                continue
            charges = np.asarray(charges).ravel()
            if charges.shape != (mol.GetNumAtoms(),):
                reject(index, f"record {index} has {charges.size} charges for "
                              f"{mol.GetNumAtoms()} atoms")
                continue
            # AddHs would append atoms after the ones the charges describe, so this path
            # refuses a record with implicit hydrogens rather than desynchronizing them.
            missing_h = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
            if missing_h:
                reject(index, f"record {index} has {missing_h} implicit hydrogen(s); "
                              "--molblock-charges needs an all-atom molblock so the supplied "
                              "charges stay aligned with the atoms")
                continue
            references.append(
                Reference(index, reference_name(mol, index), mol,
                          from_smiles=False, charges=charges)
            )
    else:
        if args.smiles is not None:
            entries = [(smiles, None) for smiles in args.smiles]
        else:
            entries = read_smiles_file(args.smiles_file)
        print(SMILES_WARNING.format(seed=args.seed))
        for index, (smiles, name) in enumerate(entries):
            if not wanted(index):
                continue
            mol = embed_conformer_from_smiles(smiles, MMFF_optimize=True, random_seed=args.seed)
            if mol is None:
                reject(index, f"RDKit could not embed a conformer for {smiles!r}")
                continue
            references.append(Reference(index, name or smiles, mol, from_smiles=True))

    if selected is not None:
        missing = sorted(selected - matched)
        if missing:
            raise RuntimeError(f"--indices {missing} are past the end of the input")

    if not references:
        raise RuntimeError("No usable reference molecules were loaded")

    charged = [ref.index for ref in references if Chem.GetFormalCharge(ref.mol) != 0]
    if charged and not args.neutralize_esp:
        print(
            f"NOTE: {len(charged)} reference(s) carry a non-zero formal charge "
            f"(indices {charged}). The default MOSES-aq checkpoint was trained on neutral "
            "molecules; consider --neutralize-esp so the conditioning ESP is comparable."
        )
    return references, failed


# settings that decide cached references
PROFILE_SETTINGS = ("seed", "solvent", "relax_references", "neutralize_esp")

# settings for generation
GENERATION_SETTINGS = PROFILE_SETTINGS + (
    "n_samples", "checkpoint", "sample_xtb", "condition_com",
)


RESUME_HINT = "Pass --overwrite to regenerate them or choose a fresh --output-dir."


def reference_cache_compatible(args: argparse.Namespace, previous_config: dict) -> bool:
    """Whether an earlier run's references.pkl was built under this run's settings."""
    return all(previous_config.get(key) == getattr(args, key) for key in PROFILE_SETTINGS)


def write_references(output_dir: Path, references, profiles, previous_entries=None) -> None:
    """Save the prepared references for inspection and for reuse by a later run.

    references.sdf is the viewable copy; references.pkl is a list of
    ``(molblock, partial charges)``, ``None`` where preparation failed. Both keep each
    reference's own frame, so re-extracting recovers the same ``com_before_centering``.
    ``previous_entries`` are merged in so an ``--indices`` shard extends the shared file
    instead of truncating it; references_source.json records each entry's input.
    """
    by_index = {ref.index: ref for ref in references}
    entries = list(previous_entries or [])
    n_inputs = max([*by_index, len(entries) - 1], default=-1) + 1
    entries.extend([None] * (n_inputs - len(entries)))

    fresh: dict[int, Chem.Mol] = {}
    for index in range(n_inputs):
        ref, profile = by_index.get(index), profiles.get(index)
        # A reference this run did not prepare keeps whatever an earlier run recorded
        if ref is None or profile is None:
            continue
        mol = Chem.Mol(restore_frame([profile.mol], profile.com_before_centering)[0])
        # so the pickle is self-describing when it is fed back as --molblock-charges
        mol.SetProp("_Name", ref.name)
        charges = None
        if profile.partial_charges is not None:
            charges = np.asarray(profile.partial_charges).ravel()
        fresh[index] = mol
        entries[index] = (Chem.MolToMolBlock(mol), charges)

    mols = []
    for index, entry in enumerate(entries):
        if entry is None:
            continue
        mol = fresh.get(index) or Chem.MolFromMolBlock(entry[0], removeHs=False)
        if mol is not None:
            mols.append(mol)
    if not mols:
        return
    write_sdf(output_dir / "references.sdf", mols)
    with (output_dir / "references.pkl").open("wb") as handle:
        pickle.dump(entries, handle)

    # Merged, not replaced, for the same --indices shard reason as ``entries``
    sources = read_reference_sources(output_dir)
    for index, ref in by_index.items():
        if index not in fresh:
            continue
        fingerprint = reference_fingerprint(ref)
        if fingerprint is None:
            sources.pop(str(index), None)
        else:
            sources[str(index)] = fingerprint
    write_json(output_dir / "references_source.json", sources)


def read_references(output_dir: Path, references, args, previous_config: dict) -> dict[int, tuple]:
    """Load reusable prepared references from an earlier run.

    An entry is reused only if the settings that shaped it are unchanged, it is still the
    same molecule as the reference now at that position, and that reference is still the
    same input -- same pose, atom ordering and charges -- the entry was built from.
    """
    path = output_dir / "references.pkl"
    if not path.exists():
        return {}
    if not reference_cache_compatible(args, previous_config):
        return {}
    try:
        with path.open("rb") as handle:
            entries = pickle.load(handle)
    except Exception as _:
        print("Ignoring unreadable references.pkl; rebuilding profiles", file=sys.stderr)
        return {}

    sources = read_reference_sources(output_dir)
    cached: dict[int, tuple] = {}
    replaced, unfingerprinted = [], []
    for ref in references:
        if ref.index >= len(entries) or entries[ref.index] is None:
            continue
        molblock, charges = entries[ref.index]
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        if canonical_smiles(mol) != canonical_smiles(ref.mol):
            continue
        recorded = sources.get(str(ref.index))
        if recorded is None:
            unfingerprinted.append(ref.index)
            continue
        if recorded != reference_fingerprint(ref):
            replaced.append(ref.index)
            continue
        if charges is not None and len(charges) != mol.GetNumAtoms():
            charges = None
        cached[ref.index] = (mol, charges)

    if replaced:
        print(
            f"NOTE: the input for reference(s) {replaced} is not the one references.pkl "
            "was built from (the pose, atom ordering or supplied charges differ); "
            "re-preparing them from the input rather than reusing the cache."
        )
    if unfingerprinted:
        print(
            f"NOTE: references.pkl records no input fingerprint for reference(s) "
            f"{unfingerprinted}; re-preparing them rather than assume the input is "
            "unchanged."
        )
    return cached


def build_profile(ref: Reference, args: argparse.Namespace, cached=None) -> InteractionProfile:
    """Extract the (centered) conditioning profile for one reference."""
    if cached is None and ref.charges is not None and not args.relax_references:
        # Already prepared, so it takes the same no-xTB path as a cached reference. Under
        # --relax-references the supplied charges are dropped: they describe the old pose.
        cached = (ref.mol, ref.charges)
    if cached is not None:
        mol, partial_charges = cached
        profile = extract_interaction_profile(
            mol,
            xtb_optimize=False,
            partial_charges=partial_charges,
            solvent=args.solvent,
            neutralize_esp=args.neutralize_esp,
        )
    else:
        profile = extract_interaction_profile(
            ref.mol,
            xtb_optimize=True if ref.from_smiles else args.relax_references,
            solvent=args.solvent,
            neutralize_esp=args.neutralize_esp,
        )
    if profile is None:
        raise RuntimeError(f"Interaction-profile extraction failed for reference {ref.index}")
    return profile


def check_atom_indices(inds: list[int], profile: InteractionProfile, ref: "Reference", flag: str) -> None:
    """Validate user-supplied atom indices against the profile's molecule.

    RDKit's own errors here (IndexError, a bare ``RuntimeError: Range Error idx``,
    OverflowError) never say which reference or which flag was at fault.
    """
    if profile.mol.GetNumAtoms() != ref.mol.GetNumAtoms():
        raise ValueError(
            f"{flag}: reference {ref.index} went from {ref.mol.GetNumAtoms()} to "
            f"{profile.mol.GetNumAtoms()} atoms during profile extraction, so the supplied "
            "indices no longer identify the intended atoms"
        )
    if not inds:
        raise ValueError(f"{flag}: reference {ref.index} has an empty atom index list")
    if len(set(inds)) != len(inds):
        raise ValueError(f"{flag}: reference {ref.index} repeats an atom index")
    n_atoms = profile.mol.GetNumAtoms()
    outside = sorted({i for i in inds if not 0 <= i < n_atoms})
    if outside:
        raise ValueError(
            f"{flag}: reference {ref.index} has atom indices {outside} outside 0..{n_atoms - 1} "
            "(indices count explicit hydrogens)"
        )


def check_mask(mask: list[int], n_pharms: int, ref_index: int, flag: str) -> None:
    """Validate a pharmacophore prioritization mask.

    subselection_pharm() silently truncates a short mask, and the sampler's own check is a
    bare assert, which ``python -O`` strips.
    """
    if len(mask) != n_pharms:
        raise ValueError(
            f"{flag}: reference {ref_index} has {n_pharms} extracted pharmacophore(s) but the "
            f"mask is {len(mask)} long"
        )
    if any(value not in (0, 1) for value in mask):
        raise ValueError(f"{flag}: reference {ref_index} mask must contain only 0s and 1s")
    # An all-zero mask is not an error: it asks for nothing to be held fixed, which
    # resolve_conditioning turns into plain pharmacophore inpainting.


def resolve_pharm_mask(profile: InteractionProfile, ref: "Reference",
                       specs: ConditioningSpecs,
                       args: argparse.Namespace) -> tuple[list[int] | None, str, dict]:
    """One reference's 0/1 priority mask, from --pharm-masks or --pharm-atoms.

    Returns the mask (None if neither spec covers this reference), the flag it came from
    for error messages, and the fields to record so resume notices a changed selection.
    """
    mask = specs.pharm_masks.get(ref.index)
    if mask is not None:
        return [int(value) for value in mask], "--pharm-masks", {}

    atoms = specs.pharm_atoms.get(ref.index)
    if atoms is None:
        return None, "--pharm-masks", {}
    record = {"pharm_atoms": list(atoms), "min_ring_atoms": args.min_ring_atoms}
    if not atoms:
        # an empty selection selects no pharmacophores, i.e. an all-zero mask
        return [0] * profile.n_pharms, "--pharm-atoms", record

    check_atom_indices(atoms, profile, ref, "--pharm-atoms")
    # returns an ndarray despite the list[int] annotation, and re-extracts the
    # pharmacophores from profile.mol, so the caller still checks its length
    labels = profile.pharm_prioritization_labels(
        atoms, min_ring_priority_atoms=args.min_ring_atoms
    )
    return [int(value) for value in labels], "--pharm-atoms", record


def resolve_conditioning(profile: InteractionProfile, ref: "Reference",
                         specs: ConditioningSpecs, args: argparse.Namespace) -> Conditioning:
    """Apply this reference's scaffold and pharmacophore selections to its profile.

    A reference with no entry in a spec keeps the plain interaction profile, so one run can
    scaffold-condition part of an SDF and generate the rest unconstrained. The two combine:
    subselection_pharm() replaces only the pharmacophore fields, leaving the scaffold fixed.

    An empty scaffold list, --pharm-mode on a reference with no pharmacophores, and an
    all-zero mask all fix nothing, so each falls back to plain conditioning with a NOTE
    rather than handing the sampler an empty condition.
    """
    condition = profile
    record: dict = {}

    scaffold_atoms = specs.scaffold.get(ref.index)
    scaffold = False
    if scaffold_atoms is not None:
        # recorded even when empty, so a later non-empty list still invalidates resume
        record["scaffold_atoms"] = list(scaffold_atoms)
        if scaffold_atoms:
            check_atom_indices(scaffold_atoms, profile, ref, "--scaffold-atoms")
            condition = condition.with_condition_atoms(scaffold_atoms)
            scaffold = True
        else:
            print(f"NOTE: reference {ref.index} has an empty --scaffold-atoms entry; "
                  "generating it with plain interaction conditioning")

    pharm = False
    prioritization = None
    eval_prioritization = None
    if args.pharm_mode == "pharm-full":
        record["pharm_mode"] = args.pharm_mode
        if profile.n_pharms == 0:
            print(f"NOTE: reference {ref.index} has no pharmacophores; ignoring --pharm-mode")
        else:
            pharm = True
    elif args.pharm_mode is not None:
        mask, flag, mask_record = resolve_pharm_mask(profile, ref, specs, args)
        record.update(mask_record)
        if mask is not None:
            check_mask(mask, profile.n_pharms, ref.index, flag)
            record["pharm_mode"] = args.pharm_mode
            record["pharm_mask"] = mask
            # Evaluate against the full pharm profile to get pharm similarity subset scores
            eval_prioritization = mask
            if not any(mask):
                print(f"NOTE: reference {ref.index} has an all-zero {flag} mask; generating "
                      "it with plain pharmacophore inpainting")
            elif args.pharm_mode == "pharm-priority-only":
                # low-priority pharmacophores are dropped outright, not inpainted, so no
                # mask goes to generate(): it would now be the wrong length
                condition = condition.subselection_pharm(pharm_prioritization_labels=mask)
                pharm = True
            else:
                prioritization = mask
                pharm = True

    # The sampler rejects any count below the amount we inpaint/condition on.
    modalities = resolve_modalities(args.condition)
    pharm_active = pharm or modalities == "all" or "pharm" in modalities
    n_x1_floor = max(1, len(scaffold_atoms) if scaffold else 0)
    n_x4_floor = max(1, condition.n_pharms if pharm_active else 0)
    n_x1 = resolve_node_range(
        "N_x1", profile.n_atoms, args.add_atoms, args.add_atoms_range,
        floor=n_x1_floor, ref_index=ref.index,
    )
    n_x4 = resolve_node_range(
        "N_x4", profile.n_pharms, args.add_pharms, args.add_pharms_range,
        floor=n_x4_floor, ref_index=ref.index,
    )
    record["n_x1"] = list(n_x1)
    record["n_x4"] = list(n_x4)

    return Conditioning(
        profile=condition,
        scaffold=scaffold,
        pharm=pharm,
        prioritization=prioritization,
        eval_prioritization=eval_prioritization,
        n_x1=n_x1,
        n_x4=n_x4,
        ref_index=ref.index,
        n_x1_base=profile.n_atoms,
        n_x4_base=profile.n_pharms,
        n_x1_floor=n_x1_floor,
        n_x4_floor=n_x4_floor,
        record=record,
    )


def report_unconditioned(references, resolved, specs: ConditioningSpecs) -> None:
    """Name the references a dict-form spec left out, which generate unconstrained."""
    indices = [ref.index for ref in references if ref.index in resolved]
    for flag, spec in (("--scaffold-atoms", specs.scaffold),
                       ("--pharm-atoms", specs.pharm_atoms),
                       ("--pharm-masks", specs.pharm_masks)):
        if not spec:
            continue
        missing = [index for index in indices if index not in spec]
        if missing:
            print(f"NOTE: no {flag} entry for reference(s) {missing}; generating them without that conditioning")


def check_generation_settings_change(args: argparse.Namespace, results_dir: Path,
                                     references, previous_config: dict,
                                     modalities) -> None:
    """Refuse to resume into results that were generated under different settings.

    Resume keeps any reference that already has a molecules.sdf, so a rerun under a new
    GENERATION_SETTING or modality set would report the old molecules under the new
    run_config.json with nothing to show for it. Raised before run_config.json is written,
    so a refused run leaves the directory describing the run that produced it.
    """
    if args.overwrite or not previous_config:
        return
    finished = [ref.index for ref in references if has_molecules(results_dir, ref.index)]
    if not finished:
        return

    current = {key: comparable_setting(getattr(args, key)) for key in GENERATION_SETTINGS}
    current["condition_modalities"] = modalities_for_json(modalities)
    changed = {
        key: (previous_config[key], value) for key, value in current.items()
        if key in previous_config and previous_config[key] != value
    }
    if not changed:
        return
    detail = "; ".join(
        f"{key} {before!r} -> {after!r}" for key, (before, after) in sorted(changed.items())
    )
    raise RuntimeError(
        f"{args.output_dir / 'run_config.json'} records different generation settings "
        f"({detail}), and reference(s) {finished} already have results in this directory. "
        "Resume would keep those results unchanged and report them under the new "
        f"settings. {RESUME_HINT}"
    )


def check_reference_input_change(args: argparse.Namespace, results_dir: Path,
                                 references) -> None:
    """Refuse to resume a finished reference whose input has since been replaced.

    read_references handles the other half, declining to reuse a cached entry whose input
    changed. This one covers molecules already written from a pose, atom ordering or set
    of charges the input no longer supplies.
    """
    if args.overwrite:
        return
    sources = read_reference_sources(args.output_dir)
    if not sources:
        return
    changed = [
        ref.index for ref in references
        if str(ref.index) in sources
        and sources[str(ref.index)] != reference_fingerprint(ref)
        and has_molecules(results_dir, ref.index)
    ]
    if changed:
        raise RuntimeError(
            f"the input for reference(s) {changed} differs from the one their existing "
            f"results in {args.output_dir} were generated from (the pose, atom ordering "
            f"or supplied charges changed). {RESUME_HINT}"
        )


def check_conditioning_change(args: argparse.Namespace, results_dir: Path,
                              conditionings: dict[int, Conditioning]) -> None:
    """Record the resolved conditioning, and refuse to resume across a change to it.

    Comparing the resolved conditioning rather than the command line also catches a spec
    file edited between runs.
    """
    path = args.output_dir / "conditioning.json"
    record = {str(index): cond.record for index, cond in conditionings.items()}

    previous = {} if args.overwrite else read_json_dict(path)
    changed = [
        index for index, cond in conditionings.items()
        if str(index) in previous
        and previous[str(index)] != cond.record
        and has_molecules(results_dir, index)
    ]
    if changed:
        raise RuntimeError(
            f"{path} records different conditioning for reference(s) {sorted(changed)}, "
            f"which already have results in this directory. {RESUME_HINT}"
        )

    # References this run skipped keep whatever the earlier run recorded for them
    write_json(path, {**previous, **record})


def generate_for_reference(model, cond: Conditioning, modalities, args,
                           reference_dir: Path, ref_index: int) -> list:
    """Generate --n-samples samples in --batch-size chunks, resuming finished chunks.

    Each chunk is seeded independently so the samples do not depend on how many
    chunks a previous run completed -- a resumed run reproduces a clean one.
    """
    chunks_dir = reference_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "modalities": modalities_for_json(modalities),
        # a chunk trimmed to x1 cannot serve a resumed run that now wants full samples
        "full_samples": bool(args.save_samples),
        **cond.record,
    }

    samples = []
    # The sample index the next chunk starts at, which an explicit per-sample node-count
    # list is sliced by -- so it has to advance across cached chunks too.
    start = 0
    sizes = chunk_sizes(args.n_samples, args.batch_size)
    for i, size in enumerate(sizes):
        chunk_path = chunks_dir / f"chunk_{i:03d}.pkl"
        cached = load_chunk(chunk_path, size, spec)
        if cached is not None:
            samples.extend(cached)
            start += size
            continue

        seed = args.seed + 1_000_000 * ref_index + i
        torch.manual_seed(seed)
        # Seeded per chunk, like the torch seed, so a resumed run reproduces a clean one
        rng = np.random.default_rng(seed)
        chunk = model.generate(
            batch_size=size,
            N_x1=chunk_node_counts(rng, cond.n_x1, start, size),
            N_x4=chunk_node_counts(rng, cond.n_x4, start, size),
            condition=cond.profile,
            condition_modalities=modalities,
            scaffold_conditioning=cond.scaffold,
            pharmacophore_conditioning=cond.pharm,
            pharmacophore_prioritization=cond.prioritization,
            condition_center_of_mass=args.condition_com,
            verbose=False,
        )
        if not args.save_samples:
            chunk = [trim_sample(sample) for sample in chunk]
        with chunk_path.open("wb") as handle:
            pickle.dump({"size": size, "spec": spec, "samples": chunk}, handle)
        samples.extend(chunk)
        start += size
    return samples


def convert_samples(samples, num_workers: int):
    """Convert samples to RDKit molecules with no xTB at all.

    Deliberately not mol_charges_from_samples' fork pool: this also runs inside the
    per-reference workers of --parallel-over reference, where a nested daemonic pool is
    forbidden, so it spawns one only when it is the one doing the parallelism.
    """
    if num_workers <= 1:
        return [create_rdkit_molecule(sample) for sample in samples]
    chunksize = max(1, math.ceil(len(samples) / (num_workers * 4)))
    context = mp.get_context("spawn")
    with context.Pool(num_workers) as pool:
        return list(pool.imap(create_rdkit_molecule, samples, chunksize=chunksize))


def mols_from_samples(samples, args: argparse.Namespace, num_workers: int):
    """Convert samples to molecules in the centered frame, per --sample-xtb.

    The returned list is aligned one-to-one with ``samples``, holding ``None`` where the
    conversion failed, so a molecule can still be named by the sample it came from.
    """
    if args.sample_xtb == "none":
        return convert_samples(samples, num_workers)
    results = mol_charges_from_samples(
        samples,
        num_workers=num_workers,
        solvent=args.solvent,
        xtb_optimize=args.sample_xtb == "relax",
        xtb_timeout=args.xtb_timeout,
        verbose=args.verbose,
    )
    return [mol for mol, _ in results]


def evaluate_samples(pipeline_kwargs: dict, args: argparse.Namespace, num_workers: int):
    """Score one reference's samples, returning ``(mols, series_global, df_rowwise)``.

    ``molblocks_post_opt`` is pre-filled to the generated-sample count, so the molecules
    stay aligned one-to-one with the samples, ``None`` where the sample was unusable.
    """
    pipeline = ConditionalEvalPipeline(**pipeline_kwargs)
    blocker = rdBase.BlockLogs()
    try:
        pipeline.evaluate(
            num_workers=num_workers,
            num_processes=1,
            verbose=args.verbose,
            timeout_minutes=args.xtb_timeout / 60,
        )
    finally:
        del blocker
    mols = [
        Chem.MolFromMolBlock(molblock, removeHs=False) if molblock else None
        for molblock in pipeline.molblocks_post_opt
    ]
    series_global, df_rowwise = pipeline.to_pandas()
    return mols, series_global, df_rowwise


def finish_reference(task: PendingReference, args: argparse.Namespace,
                     num_workers: int) -> tuple[int, int, str | None]:
    """Convert or evaluate one reference and write its molecules.sdf and metrics.

    Returns ``(index, molecules written, error message)`` rather than raising: an
    exception crossing a pool boundary loses the reference it belonged to.
    """
    try:
        if task.pipeline_kwargs is not None:
            mols, series_global, df_rowwise = evaluate_samples(
                task.pipeline_kwargs, args, num_workers
            )
            with (task.directory / "metrics_rowwise.pkl").open("wb") as handle:
                pickle.dump(df_rowwise, handle)
            with (task.directory / "metrics_global.pkl").open("wb") as handle:
                pickle.dump(series_global, handle)
        else:
            mols = mols_from_samples(task.samples, args, num_workers)

        # keep the sample index
        kept = [(index, mol) for index, mol in enumerate(mols) if mol is not None]
        molecules = [mol for _, mol in kept]
        if task.offset is not None:
            molecules = restore_frame(molecules, task.offset)
        write_sdf(
            task.directory / "molecules.sdf",
            molecules,
            [f"{task.name}_sample_{index:03d}" for index, _ in kept],
        )
        # only on success, so a failed finish still resumes generation for free
        shutil.rmtree(task.directory / "chunks", ignore_errors=True)
        return task.index, len(molecules), None
    except Exception as error:  # noqa: BLE001 - reported to the parent, not raised across it
        return task.index, 0, f"{type(error).__name__}: {error}\n{traceback.format_exc()}"


def load_generation_model(args: argparse.Namespace, conditionings):
    """Load the checkpoint and confirm it supports the conditioning this run asks for."""
    model = load_model(
        device=args.device,
        local_checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    # sampler.py raises this per reference; catching it here names the checkpoint instead
    if any(cond.scaffold for cond in conditionings) and not model.params.get(
        "scaffold_conditioning", False
    ):
        raise RuntimeError(
            "--scaffold-atoms needs a checkpoint trained with scaffold_conditioning=True; "
            f"{args.checkpoint or 'the default model'} was not"
        )
    return model


def run(
    args: argparse.Namespace,
    *,
    refine_conditioning: Callable[[Reference, Conditioning], Conditioning] | None = None,
    config_extra: dict | None = None,
) -> int:
    """Generate for every reference, then convert or evaluate what was generated.

    ``refine_conditioning`` is called once per reference, after resolve_conditioning and
    before check_conditioning_change, so what it returns drives both conditioning.json and
    the chunk cache; it is how a caller supplies explicit per-sample node counts.
    ``config_extra`` is merged into run_config.json as the caller's own record.
    """
    torch.manual_seed(args.seed)

    modalities = resolve_modalities(args.condition)
    # Parsed before anything expensive so a malformed spec fails immediately
    specs = ConditioningSpecs.load(args)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failures.json").unlink(missing_ok=True)

    # Read before it is overwritten: record the settings the cached references were built with
    previous_config = read_previous_config(args.output_dir)

    config = {key: comparable_setting(value) for key, value in vars(args).items()}
    config["condition_modalities"] = modalities_for_json(modalities)
    config.update(config_extra or {})

    if args.evaluate and args.sample_xtb != "relax":
        print(f"NOTE: --sample-xtb {args.sample_xtb} is ignored with --evaluate; molecules "
              "come from the pipeline's xTB-relaxed output.")

    references, failed = load_references(args)

    def flush_failures() -> None:
        if failed:
            write_json(args.output_dir / "failures.json", failed)

    flush_failures()

    # validate then write the run config
    check_generation_settings_change(args, results_dir, references, previous_config, modalities)
    check_reference_input_change(args, results_dir, references)
    write_json(args.output_dir / "run_config.json", config)

    # load the cached references
    cached_references = (
        {} if args.overwrite
        else read_references(args.output_dir, references, args, previous_config)
    )
    if cached_references:
        print(f"Reusing {len(cached_references)} relaxed reference(s) from references.pkl")

    # Whether xTB is needed depends on which references arrived with charges, then checks xTB is installed
    if xtb_required(args, references, cached_references):
        require_xtb()

    print("Extracting interaction profiles for input references...")
    profiles: dict[int, InteractionProfile] = {}
    for ref in references:
        try:
            profiles[ref.index] = build_profile(ref, args, cached_references.get(ref.index))
        except Exception as error:
            record_failure(failed, ref.index, error)
    flush_failures()

    # Merge --indices shard that shared this output directory
    write_references(
        args.output_dir,
        references,
        profiles,
        previous_entries=(
            read_reference_entries(args.output_dir)
            if reference_cache_compatible(args, previous_config)
            else None
        ),
    )

    # Get conditioning for each reference
    conditionings: dict[int, Conditioning] = {}
    for ref in references:
        if ref.index not in profiles:
            continue
        try:
            cond = resolve_conditioning(profiles[ref.index], ref, specs, args)
            if refine_conditioning is not None:
                cond = refine_conditioning(ref, cond)
            for name in ("n_x1", "n_x4"):
                counts = getattr(cond, name)
                if isinstance(counts, list) and len(counts) != args.n_samples:
                    raise ValueError(
                        f"{name} for reference {ref.index} lists {len(counts)} counts for "
                        f"{args.n_samples} sample(s)"
                    )
            conditionings[ref.index] = cond
        except Exception as error:
            record_failure(failed, ref.index, error)
    flush_failures()
    report_unconditioned(references, conditionings, specs)
    check_conditioning_change(args, results_dir, conditionings)

    # Decide which references to generate, then load the model
    to_generate = []
    for ref in references:
        if ref.index not in conditionings:
            continue
        if not args.overwrite and reference_complete(results_dir, ref.index, args.evaluate):
            print(f"Skipping completed reference {ref.index}")
            continue
        to_generate.append(ref)

    model = None
    if to_generate:
        # load the model and confirm it supports the conditioning this run asks for
        model = load_generation_model(
            args, [conditionings[ref.index] for ref in to_generate]
        )

    pending: list[PendingReference] = []
    progress = tqdm(to_generate, desc="Generating", disable=not args.verbose)
    for ref in progress:
        progress.set_description(f"Generating reference {ref.index:03d}")
        ref_dir = reference_dir(results_dir, ref.index)
        ref_dir.mkdir(parents=True, exist_ok=True)

        try:
            profile = profiles[ref.index]
            cond = conditionings[ref.index]
            samples = generate_for_reference(
                model, cond, modalities, args, ref_dir, ref.index
            )
            if args.save_samples:
                with (ref_dir / "samples.pkl").open("wb") as handle:
                    pickle.dump(samples, handle)

            # Build shepherd-score pipeline kwargs since requires model.params. Evaluated
            # against the full profile (not cond.profile) for pharm subset scoring.
            pipeline_kwargs = None
            if args.evaluate:
                pipeline_kwargs = model.to_shepherd_score_inputs(
                    samples,
                    condition=profile,
                    condition_modalities=modalities,
                    pharmacophore_prioritization=cond.eval_prioritization,
                )
            pending.append(
                PendingReference(
                    index=ref.index,
                    name=ref.name,
                    directory=ref_dir,
                    samples=None if args.evaluate else samples,
                    pipeline_kwargs=pipeline_kwargs,
                    offset=(
                        None if ref.from_smiles
                        else np.asarray(profile.com_before_centering)
                    ),
                    n_samples=len(samples),
                )
            )
        except Exception as error:
            record_failure(failed, ref.index, error)
        finally:
            flush_failures()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finish_pending(pending, args, failed, flush_failures)

    if args.evaluate:
        combine_metrics(results_dir, args.output_dir, references)

    print(f"Done. Results in {args.output_dir}")
    return 1 if failed else 0


def finish_pending(pending: list[PendingReference], args: argparse.Namespace,
                   failed: dict[str, str], flush_failures) -> None:
    """Convert or evaluate every generated reference, the GPU-free second phase.

    Under the default --parallel-over reference, whole references run in a spawn pool;
    ``num_workers=1`` inside them is required, not stylistic, since pool workers are
    daemonic and mol_charges_from_samples would build a pool of its own. --parallel-over
    sample instead spends the workers within one reference at a time.
    """
    if not pending:
        return

    def record(index: int, count: int, error: str | None) -> None:
        if error is not None:
            failed[str(index)] = error
            print(f"Reference {index} failed: {error.splitlines()[0]}", file=sys.stderr)
        else:
            print(f"Reference {index:03d}: {count} valid molecule(s)")
        flush_failures()

    if args.parallel_over == "reference" and len(pending) > 1:
        workers = resolve_workers(args.num_workers, len(pending))
        context = mp.get_context("spawn")
        with context.Pool(workers) as pool:
            results = pool.imap_unordered(
                partial(finish_reference, args=args, num_workers=1), pending, chunksize=1
            )
            for index, count, error in tqdm(
                results, total=len(pending), desc="Finishing", disable=not args.verbose
            ):
                record(index, count, error)
        return

    for task in tqdm(pending, desc="Finishing", disable=not args.verbose):
        workers = resolve_workers(args.num_workers, max(1, task.n_samples))
        record(*finish_reference(task, args, workers))


def combine_metrics(results_dir: Path, output_dir: Path, references: list[Reference]) -> None:
    """Concatenate every reference's metrics into one frame with a ref_ind column.

    Re-read from disk, so references a resumed run skipped are included alongside the
    ones it produced.
    """
    rowwise, globals_ = [], []
    for ref in references:
        ref_dir = reference_dir(results_dir, ref.index)
        rowwise_path = ref_dir / "metrics_rowwise.pkl"
        if rowwise_path.exists():
            with rowwise_path.open("rb") as handle:
                frame = pickle.load(handle)
            # sample_ind is the row's position within its own reference, which is the
            # sample index only because the pipeline pre-fills a row per generated sample
            rowwise.append(frame.assign(ref_ind=ref.index, sample_ind=range(len(frame))))
        global_path = ref_dir / "metrics_global.pkl"
        if global_path.exists():
            with global_path.open("rb") as handle:
                row = pickle.load(handle).to_dict()
            row["ref_ind"] = ref.index
            globals_.append(row)

    if rowwise:
        combined = pd.concat(rowwise, ignore_index=True)
        with (output_dir / "combined_rowwise.pkl").open("wb") as handle:
            pickle.dump(combined, handle)
        print(f"Combined {len(combined)} rows across {len(rowwise)} references")
    if globals_:
        with (output_dir / "combined_global.pkl").open("wb") as handle:
            pickle.dump(pd.DataFrame(globals_), handle)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
