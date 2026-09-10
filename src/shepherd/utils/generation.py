"""Small data and file helpers for the generation CLI."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import pickle
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from rdkit import Chem

from shepherd.interaction_profile import InteractionProfile
from shepherd_score.conformer_generation import update_mol_coordinates


def resolve_modalities(condition: list[str] | None) -> str | set[str]:
    if not condition or "all" in condition:
        return "all"
    return set(condition)


def modalities_for_json(modalities: str | set[str]) -> str | list[str]:
    return sorted(modalities) if isinstance(modalities, set) else modalities


def resolve_workers(num_workers: int, num_items: int) -> int:
    if num_workers > 0:
        return num_workers
    return max(1, min(num_items, mp.cpu_count()))


@dataclass
class Reference:
    index: int
    name: str
    mol: Chem.Mol
    from_smiles: bool
    charges: np.ndarray | None = None


def reference_name(mol: Chem.Mol, index: int) -> str:
    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    return name or f"reference_{index:03d}"


def read_smiles_file(path: Path) -> list[tuple[str, str | None]]:
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        entries.append((fields[0], fields[1] if len(fields) > 1 else None))
    return entries


def reference_dir(results_dir: Path, index: int) -> Path:
    return results_dir / f"reference_{index:03d}"


def has_molecules(results_dir: Path, index: int) -> bool:
    return (reference_dir(results_dir, index) / "molecules.sdf").exists()


def reference_complete(results_dir: Path, index: int, evaluate: bool) -> bool:
    ref_dir = reference_dir(results_dir, index)
    return (ref_dir / "molecules.sdf").exists() and (
        not evaluate or (ref_dir / "metrics_rowwise.pkl").exists()
    )


def canonical_smiles(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:
        return None


def reference_fingerprint(ref: Reference) -> str | None:
    try:
        molblock = Chem.MolToMolBlock(ref.mol)
    except Exception:
        return None
    digest = hashlib.sha256(molblock.encode())
    if ref.charges is not None:
        digest.update(np.ascontiguousarray(ref.charges, dtype=np.float64).ravel().tobytes())
    return digest.hexdigest()


def read_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_reference_sources(output_dir: Path) -> dict[str, str]:
    return read_json_dict(output_dir / "references_source.json")


def read_previous_config(output_dir: Path) -> dict:
    return read_json_dict(output_dir / "run_config.json")


def read_reference_entries(output_dir: Path) -> list:
    path = output_dir / "references.pkl"
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            entries = pickle.load(handle)
    except Exception:
        return []
    return list(entries) if isinstance(entries, (list, tuple)) else []


def comparable_setting(value):
    return str(value) if isinstance(value, Path) else value


def read_index_spec(path: Path, flag: str) -> dict[int, list[int]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{flag}: could not read {path} ({error})") from error

    if isinstance(value, list):
        items = list(enumerate(value))
    elif isinstance(value, dict):
        try:
            items = [(int(key), entry) for key, entry in value.items()]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{flag}: object keys must be integer reference indices (position in the SDF)"
            ) from error
    else:
        raise ValueError(
            f"{flag}: expected a list of lists or an object keyed by reference index, "
            f"got {type(value).__name__}"
        )

    spec = {}
    for index, entry in items:
        if not isinstance(entry, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in entry
        ):
            raise ValueError(f"{flag}: entry for reference {index} must be a list of integers")
        spec[index] = list(entry)
    return spec


@dataclass
class ConditioningSpecs:
    scaffold: dict[int, list[int]]
    pharm_atoms: dict[int, list[int]]
    pharm_masks: dict[int, list[int]]

    @classmethod
    def load(cls, args) -> "ConditioningSpecs":
        def read(path: Path | None, flag: str) -> dict[int, list[int]]:
            return {} if path is None else read_index_spec(path, flag)

        return cls(
            scaffold=read(args.scaffold_atoms, "--scaffold-atoms"),
            pharm_atoms=read(args.pharm_atoms, "--pharm-atoms"),
            pharm_masks=read(args.pharm_masks, "--pharm-masks"),
        )

    @property
    def empty(self) -> bool:
        return not (self.scaffold or self.pharm_atoms or self.pharm_masks)


def resolve_node_range(
    name: str,
    base: int,
    offset: int,
    bounds: list[int] | None,
    floor: int,
    ref_index: int,
) -> tuple[int, int]:
    lo, hi = (base + bounds[0], base + bounds[1]) if bounds else (base + offset, base + offset)
    if hi < floor:
        raise ValueError(
            f"{name} range [{lo}, {hi}] for reference {ref_index} is entirely below the "
            f"{floor} node(s) the conditioning fixes; nothing in it can be generated"
        )
    if lo < floor:
        print(
            f"NOTE: raising reference {ref_index}'s {name} range from [{lo}, {hi}] to "
            f"[{floor}, {hi}], the floor set by its conditioning"
        )
        lo = floor
    return lo, hi


def apply_node_floor(name: str, counts, floor: int, ref_index: int) -> list[int]:
    counts = [int(value) for value in counts]
    invalid = sorted({value for value in counts if value < 1})
    if invalid:
        raise ValueError(f"{name} for reference {ref_index} must be positive, got {invalid}")
    raised = sum(1 for value in counts if value < floor)
    if raised:
        print(
            f"NOTE: raising {raised} of reference {ref_index}'s {name} entries to {floor}, "
            "the floor set by its conditioning"
        )
    return [max(value, floor) for value in counts]


def chunk_node_counts(
    rng: np.random.Generator, counts, start: int, size: int
) -> int | list[int]:
    if isinstance(counts, list):
        return list(counts[start : start + size])
    lo, hi = counts
    if lo == hi:
        return lo
    return rng.integers(lo, hi + 1, size=size).tolist()


@dataclass
class Conditioning:
    profile: InteractionProfile
    scaffold: bool = False
    pharm: bool = False
    prioritization: list[int] | None = None
    n_x1: tuple[int, int] | list[int] = (0, 0)
    n_x4: tuple[int, int] | list[int] = (0, 0)
    ref_index: int = -1
    n_x1_base: int = 0
    n_x4_base: int = 0
    n_x1_floor: int = 1
    n_x4_floor: int = 1
    record: dict = field(default_factory=dict)

    def with_node_counts(self, *, n_x1=None, n_x4=None) -> "Conditioning":
        updates, record = {}, dict(self.record)
        for name, key, value, floor in (
            ("N_x1", "n_x1", n_x1, self.n_x1_floor),
            ("N_x4", "n_x4", n_x4, self.n_x4_floor),
        ):
            if value is None:
                continue
            counts = apply_node_floor(name, value, floor, self.ref_index)
            updates[key] = counts
            record[key] = list(counts)
        return replace(self, record=record, **updates)


def chunk_sizes(n_samples: int, batch_size: int) -> list[int]:
    sizes = []
    remaining = n_samples
    while remaining > 0:
        sizes.append(min(batch_size, remaining))
        remaining -= sizes[-1]
    return sizes


def load_chunk(path: Path, expected_size: int, spec: dict) -> list | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        samples = payload["samples"]
    except Exception:
        print(f"Ignoring unreadable chunk {path.name}; regenerating it", file=sys.stderr)
        return None
    if payload.get("size") != expected_size or len(samples) != expected_size:
        return None
    if payload.get("spec") != spec:
        return None
    return samples


def trim_sample(sample) -> dict:
    return {
        "x1": {
            "atoms": np.asarray(sample["x1"]["atoms"]),
            "positions": np.asarray(sample["x1"]["positions"]),
        }
    }


def write_sdf(path: Path, mols, names=None) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        for i, mol in enumerate(mols):
            if names is not None:
                mol.SetProp("_Name", names[i])
            writer.write(mol)
    finally:
        writer.close()


def restore_frame(mols, offset):
    offset = np.asarray(offset)
    return [
        update_mol_coordinates(mol, mol.GetConformer().GetPositions() + offset)
        for mol in mols
    ]


@dataclass
class PendingReference:
    index: int
    name: str
    directory: Path
    samples: list | None
    pipeline_kwargs: dict | None
    offset: np.ndarray | None
    n_samples: int
