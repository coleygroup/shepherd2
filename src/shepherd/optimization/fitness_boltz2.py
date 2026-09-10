"""Boltz2 affinity FitnessOracle."""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from shepherd.optimization.fitness import FitnessOracle

if TYPE_CHECKING:
    from shepherd.optimization.individual import Individual

logger = logging.getLogger(__name__)


class Boltz2AffinityOracle(FitnessOracle):
    """Fitness oracle that scores molecules using Boltz2 predicted affinity.

    Each call to :meth:`evaluate` submits one ``boltz predict`` subprocess per
    molecule and reads the resulting affinity JSON.  Molecules are scored
    sequentially; use :meth:`evaluate_with_cache` (inherited) to skip
    previously seen SMILES.

    Arguments
    ---------
    protein_sequences : str or list[str]
        Amino-acid sequence(s) of the target protein(s) in single-letter codes
        (no gaps, no spaces).  A plain string is treated as a single chain.
    out_dir : str or Path
        Root directory for all Boltz2 prediction outputs.  Created
        automatically if it does not exist.
    protein_msa : str or Path or list[str or Path or None], optional
        Path(s) to pre-computed ``.a3m`` MSA file(s) for the target(s).  A
        single value is broadcast to all chains.  When an entry is ``None``
        and ``use_msa_server=False``, ``msa: empty`` is written for that chain
        (single-sequence mode, lower accuracy).
    protein_template : str or Path, optional
        Path to a ``.cif`` (or ``.pdb``) structural template applied globally
        to the complex.  When provided, a ``templates:`` block is appended to
        enable Boltz2 template conditioning.
    use_msa_server : bool
        Pass ``--use_msa_server`` to ``boltz predict`` so that it fetches the
        MSA from the ColabFold mmseqs2 server at prediction time.  Requires
        an active internet connection.
    protein_chain_ids : str or list[str]
        Chain identifier(s) for the protein chain(s) in the YAML input.  A
        single string ``"A"`` with multiple sequences auto-assigns ``A, B,
        C, …``.  Explicit lists must have the same length as
        ``protein_sequences``.
    ligand_chain_id : str
        Chain identifier used for the ligand in the YAML input (default
        ``"L"``).
    score_field : {"affinity_pred_value", "affinity_probability_binary"}
        Which field from the Boltz2 affinity JSON to use as the fitness score.
    recycling_steps : int
        ``--recycling_steps`` passed to ``boltz predict``.
    diffusion_samples : int
        ``--diffusion_samples`` passed to ``boltz predict``.
    sampling_steps : int
        ``--sampling_steps`` passed to ``boltz predict``.
    device : str, optional
        ``--accelerator`` value passed to ``boltz predict`` (e.g. ``"gpu"``.)
    timeout : int
        Per-molecule subprocess timeout in seconds (default 600 = 10 min).
        Molecules that exceed this limit are scored as ``float('inf')``.
    boltz_executable : str
        Path to the ``boltz`` executable.  Defaults to ``"boltz"`` (resolved
        via ``PATH``).  Set to an absolute path to use boltz from a different
        conda environment, e.g.
        ``"/home/user/.conda/envs/boltz_env/bin/boltz"``.
    num_workers : int
        ``--num_workers`` passed to ``boltz predict`` (dataloader workers).
    preprocessing_threads : int
        ``--preprocessing-threads`` passed to ``boltz predict``.  Parallelises
        the CPU-side preprocessing (MSA parsing, featurisation) when a batch of
        molecules is scored in a single invocation.
    batch_timeout : int, optional
        Subprocess timeout in seconds for a whole batched invocation.  When
        ``None`` the per-molecule ``timeout`` is scaled by the batch size.
    """

    def __init__(
        self,
        protein_sequences: str | List[str],
        out_dir: str | Path = "./boltz_oracle_out",
        protein_msa: str | Path | List[str | Path | None] | None = None,
        protein_template: str | Path | None = None,
        protein_template_force: bool = None,
        protein_template_threshold: float = 0.0,
        use_msa_server: bool = False,
        use_steer_potential: bool = False,
        protein_chain_ids: str | List[str] = "A",
        ligand_chain_id: str = "L",
        score_field: Literal[
            "affinity_pred_value", "affinity_probability_binary"
        ] = "affinity_pred_value",
        recycling_steps: int = 3,
        diffusion_samples: int = 1,
        sampling_steps: int = 200,
        device: str | None = None,
        timeout: int = 600,
        boltz_executable: str = "boltz",
        num_workers: int = 2,
        preprocessing_threads: int = 8,
        batch_timeout: int | None = None,
    ) -> None:
        super().__init__()

        # Normalize sequences to list
        if isinstance(protein_sequences, str):
            protein_sequences = [protein_sequences]
        if not protein_sequences or not all(protein_sequences):
            raise ValueError("protein_sequences must be a non-empty list of amino-acid strings.")
        self.protein_sequences = protein_sequences
        n = len(protein_sequences)

        if score_field not in ("affinity_pred_value", "affinity_probability_binary"):
            raise ValueError(
                f"score_field must be 'affinity_pred_value' or "
                f"'affinity_probability_binary', got {score_field!r}"
            )

        # Normalize chain IDs
        if isinstance(protein_chain_ids, str):
            if n == 1:
                protein_chain_ids = [protein_chain_ids]
            else:
                protein_chain_ids = [chr(ord(protein_chain_ids[0]) + i) for i in range(n)]
        if len(protein_chain_ids) != n:
            raise ValueError(
                f"protein_chain_ids length ({len(protein_chain_ids)}) must match "
                f"protein_sequences length ({n})."
            )
        self.protein_chain_ids = protein_chain_ids

        # Normalize MSAs to per-chain list
        if protein_msa is None or isinstance(protein_msa, (str, Path)):
            protein_msa = [protein_msa] * n
        if len(protein_msa) != n:
            raise ValueError(
                f"protein_msa length ({len(protein_msa)}) must match "
                f"protein_sequences length ({n})."
            )
        self.protein_msa = [Path(m) if m is not None else None for m in protein_msa]

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.protein_template = Path(protein_template) if protein_template is not None else None
        self.protein_template_force = protein_template_force
        self.protein_template_threshold = protein_template_threshold
        self.use_steer_potential = use_steer_potential
        self.use_msa_server = use_msa_server
        self.ligand_chain_id = ligand_chain_id
        self.score_field = score_field
        self.recycling_steps = recycling_steps
        self.diffusion_samples = diffusion_samples
        self.sampling_steps = sampling_steps
        self.device = device
        self.timeout = timeout
        self.boltz_executable = boltz_executable
        self.num_workers = num_workers
        self.preprocessing_threads = preprocessing_threads
        # per-batch subprocess timeout; defaults to timeout scaled by batch size if None
        self.batch_timeout = batch_timeout

        # thread-safe counter for unique batch directory names
        self._batch_counter = 0
        self._batch_counter_lock = threading.Lock()

        # thread-safe counter for unique prediction names
        self._counter = 0
        self._counter_lock = threading.Lock()

        # pose buffer (smiles -> {"docked_mol": Chem.Mol}); mol is in the first success's frame
        self.docked_pose_buffer: Dict[str, dict] = {}

        # reference Cα coords (N×3); later predictions are aligned to this frame
        self._ref_ca_coords: np.ndarray | None = None
        self._ref_lock = threading.Lock()

    # ---- FitnessOracle interface ----

    def evaluate(self, individuals: List["Individual"]) -> List[float]:
        """Score *smiles_list* with Boltz2 affinity prediction.

        All valid SMILES are scored in a **single** batched ``boltz predict``
        invocation (Boltz's directory-input mode), so the model weights and CUDA
        context are loaded once per call rather than once per molecule.  A
        single-element list is handled identically (one invocation).  Any
        molecule whose output is missing after the batch run — e.g. a CUDA OOM
        that aborted the invocation — is retried individually via the
        single-molecule path so one bad molecule cannot poison the batch.

        Arguments
        ---------
        individuals : list[Individual]
            Individual objects containing SMILES strings to score.  ``None`` entries are scored as
            ``float('inf')``.

        Returns
        -------
        list[float]
            One affinity score per SMILES (lower = better binder).
            Failed predictions return ``float('inf')``.
        """
        smiles_list = [ind.smiles for ind in individuals]
        results: List[float] = [float("inf")] * len(smiles_list)

        # assign a unique name per valid SMILES, keeping the index so results land in the right slot
        to_run: List[tuple[int, str, str]] = []  # (index, smiles, name)
        for i, smi in enumerate(smiles_list):
            if smi is None:
                continue
            to_run.append((i, smi, self._unique_name(smi)))

        if not to_run:
            return results

        batch_dir = self._new_batch_dir()
        results_dir = self.out_dir / f"boltz_results_{batch_dir.name}"
        try:
            for _, smi, name in to_run:
                self._write_yaml(smi, name, batch_dir / f"{name}.yaml")

            timeout = self.batch_timeout or self.timeout * len(to_run)
            # a non-zero exit (e.g. one OOM) still leaves completed predictions; parse & fall back
            try:
                self._run_boltz_batch(batch_dir, timeout)
            except Exception as exc:
                logger.warning(
                    "Batched boltz predict failed for %s (%d molecules); "
                    "falling back to per-molecule for any missing outputs: %s",
                    batch_dir.name, len(to_run), exc,
                )

            for idx, smi, name in to_run:
                score = self._collect_result(smi, name, results_dir)
                if score is None:
                    # Output missing — isolate this molecule with its own run.
                    score = self._score_smiles(smi, self._unique_name(smi))
                results[idx] = score
        finally:
            self._cleanup_batch_dir(batch_dir)

        return results

    def _collect_result(
        self, smiles: str, name: str, results_dir: Path
    ) -> float | None:
        """Parse a single molecule's affinity from a batched run.

        Returns the score, or ``None`` if the output is missing (so the caller
        can retry the molecule individually).  Pose storage is best-effort.
        """
        try:
            score = self._parse_output(name, results_dir=results_dir)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("Boltz2 parse failed for %s (%s): %s", name, smiles, exc)
            return float("inf")

        try:
            self._store_aligned_pose(smiles, name, results_dir=results_dir)
        except Exception as exc:
            logger.warning("Failed to store pose for %s: %s", name, exc)
        return score

    def _new_batch_dir(self) -> Path:
        """Create and return a fresh, uniquely named batch input directory."""
        with self._batch_counter_lock:
            idx = self._batch_counter
            self._batch_counter += 1
        batch_dir = self.out_dir / f"_batch_{idx:06d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    @staticmethod
    def _cleanup_batch_dir(batch_dir: Path) -> None:
        """Remove batch input YAMLs (and the dir); keep prediction outputs."""
        if not batch_dir.exists():
            return
        for yaml_file in batch_dir.glob("*.yaml"):
            yaml_file.unlink()
        try:
            batch_dir.rmdir()
        except OSError:
            pass  # non-empty (shouldn't happen); leave for debugging

    # ---- internal helpers ----

    def _unique_name(self, smiles: str) -> str:
        """Return a filesystem-safe unique name for a prediction run."""
        with self._counter_lock:
            idx = self._counter
            self._counter += 1
        # short SMILES hash for readability; counter ensures uniqueness
        smi_hash = hashlib.md5(smiles.encode()).hexdigest()[:8]
        return f"mol_{idx:06d}_{smi_hash}"

    def _score_smiles(self, smiles: str, name: str) -> float:
        """Run the full Boltz2 predict pipeline for one SMILES."""
        yaml_path = self.out_dir / f"{name}.yaml"
        try:
            self._write_yaml(smiles, name, yaml_path)
            # invalidate the cached record only if the YAML changed, to avoid redundant server calls
            self._invalidate_stale_record(name, yaml_path)
            self._run_boltz(yaml_path)
            score = self._parse_output(name)
        except MemoryError as exc:
            logger.warning("Boltz2 OOM for %s — scored as inf. (%s)", name, exc)
            return float("inf")
        except Exception as exc:
            logger.warning(
                "Boltz2 scoring failed for %s (%s): %s", name, smiles, exc
            )
            return float("inf")
        finally:
            # Remove YAML input; keep prediction dirs for debugging.
            if yaml_path.exists():
                yaml_path.unlink()
        # Pose storage is best-effort — failures must not discard a valid score.
        try:
            self._store_aligned_pose(smiles, name)
        except Exception as exc:
            logger.warning("Failed to store pose for %s: %s", name, exc)
        return score

    def _invalidate_stale_record(self, name: str, yaml_path: Path) -> None:
        """Delete boltz's cached processed record only when the YAML has changed.

        Boltz skips reprocessing (including template parsing and MSA fetching)
        when processed/records/{name}.json already exists.  We hash the current
        YAML and compare it against a sidecar file; if they differ we delete the
        record so boltz re-parses from scratch, but leave the cached MSA intact
        so the server is not re-queried unnecessarily.
        """
        record = self.out_dir / f"boltz_results_{name}" / "processed" / "records" / f"{name}.json"
        hash_sidecar = record.parent / f"{name}.yaml.md5"

        current_hash = hashlib.md5(yaml_path.read_bytes()).hexdigest()

        if record.exists():
            cached_hash = hash_sidecar.read_text().strip() if hash_sidecar.exists() else ""
            if current_hash != cached_hash:
                record.unlink()
                logger.debug("YAML changed for %s — deleted stale processed record.", name)

        hash_sidecar.parent.mkdir(parents=True, exist_ok=True)
        hash_sidecar.write_text(current_hash)

    def _write_yaml(self, smiles: str, name: str, path: Path) -> None:
        """Write the Boltz2 YAML input for a protein-ligand complex."""
        protein_blocks = ""
        protein_zip = zip(self.protein_sequences, self.protein_chain_ids, self.protein_msa)
        for seq, chain_id, msa in protein_zip:
            if msa is not None:
                msa_line = f"\n      msa: {msa}"
            elif self.use_msa_server:
                msa_line = ""  # omit; server fetches MSA at runtime
            else:
                msa_line = "\n      msa: empty"
            protein_blocks += (
                f"  - protein:\n"
                f"      id: {chain_id}\n"
                f"      sequence: {seq}{msa_line}\n"
            )

        if self.protein_template is not None:
            fmt = "pdb" if self.protein_template.suffix.lower() == ".pdb" else "cif"
            template_section = f"templates:\n  - {fmt}: {self.protein_template}\n"
            if self.protein_template_force:
                template_section += (
                    f"    force: true\n"
                    f"    threshold: {self.protein_template_threshold}\n"
                )
            # omit chain_id/template_id; boltz auto-detects via sequence alignment (PDB-dependent)
        else:
            template_section = ""

        yaml_content = (
            f"version: 1\n"
            f"sequences:\n"
            f"{protein_blocks}"
            f"  - ligand:\n"
            f"      id: {self.ligand_chain_id}\n"
            f"      smiles: '{smiles}'\n"
        )
        if template_section:
            yaml_content += template_section
        yaml_content += (
            f"properties:\n"
            f"  - affinity:\n"
            f"      binder: {self.ligand_chain_id}\n"
        )

        path.write_text(yaml_content)

    def _base_boltz_cmd(self, data_path: Path) -> List[str]:
        """Build the shared ``boltz predict`` command for *data_path*.

        *data_path* may be a single YAML (single-molecule mode) or a directory
        of YAMLs (batched mode).
        """
        cmd = [
            self.boltz_executable, "predict", str(data_path),
            "--out_dir", str(self.out_dir),
            "--recycling_steps", str(self.recycling_steps),
            "--diffusion_samples", str(self.diffusion_samples),
            "--sampling_steps", str(self.sampling_steps),
            "--output_format", "pdb",
            "--override",
        ]
        if self.use_msa_server:
            cmd.append("--use_msa_server")
        if self.use_steer_potential:
            cmd.append("--use_potentials")
        if self.device is not None:
            cmd += ["--accelerator", self.device]
        return cmd

    def _boltz_env(self) -> dict:
        """Return an environment dict with Boltz's CUDA libs on LD_LIBRARY_PATH."""
        env = os.environ.copy()
        boltz_prefix = Path(self.boltz_executable).resolve().parent.parent
        cuda_lib_dirs = glob.glob(
            str(boltz_prefix / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib")
        )
        if cuda_lib_dirs:
            extra = ":".join(sorted(cuda_lib_dirs))
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{extra}:{existing}" if existing else extra
        return env

    def _run_boltz(self, yaml_path: Path) -> None:
        """Invoke ``boltz predict`` on a single YAML and raise on failure."""
        cmd = self._base_boltz_cmd(yaml_path)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=self.timeout,
            env=self._boltz_env(),
        )

        torch.cuda.empty_cache()
        # check GPU OOM first: it cascades into a FileNotFoundError in the affinity step (rc != 0)
        if "ran out of memory" in result.stdout or "out of memory" in result.stdout.lower():
            raise MemoryError(
                f"boltz predict ran out of GPU memory (rc={result.returncode}); "
                "molecule scored as inf"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"boltz predict failed (rc={result.returncode}):\n"
                f"  stdout: {result.stdout[-500:]}\n"
                f"  stderr: {result.stderr[-500:]}"
            )

    def _run_boltz_batch(self, batch_dir: Path, timeout: int) -> None:
        """Invoke ``boltz predict`` once over a directory of YAMLs.

        Boltz loads the model weights and initialises CUDA a single time for the
        whole directory, then runs every complex through one dataloader — the
        core speedup over one subprocess per molecule.

        Unlike :meth:`_run_boltz` this does **not** raise on a non-zero return
        code: Boltz writes each prediction as it completes, so a mid-run failure
        (e.g. one molecule OOMs) still leaves the earlier predictions on disk.
        The caller parses whatever succeeded and retries missing molecules
        individually, so a non-zero exit is logged as a warning rather than
        raised.
        """
        cmd = self._base_boltz_cmd(batch_dir)
        cmd += [
            "--num_workers", str(self.num_workers),
            "--preprocessing-threads", str(self.preprocessing_threads),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout,
            env=self._boltz_env(),
        )

        torch.cuda.empty_cache()
        if "ran out of memory" in result.stdout.lower():
            logger.warning(
                "boltz predict hit GPU OOM during batch %s (rc=%d); "
                "partial results kept, missing molecules retried individually.",
                batch_dir.name, result.returncode,
            )
        elif result.returncode != 0:
            logger.warning(
                "boltz predict batch %s exited rc=%d; parsing partial results.\n"
                "  stdout: %s\n  stderr: %s",
                batch_dir.name, result.returncode,
                result.stdout[-500:], result.stderr[-500:],
            )
    @staticmethod
    def _apply_se3(mol: Chem.Mol, R: np.ndarray, t: np.ndarray, conf_id: int = 0) -> Chem.Mol:
        """Apply x' = R @ x + t to all atoms in-place.

        R: (3, 3) rotation matrix
        t: (3,) translation vector
        """
        conf = mol.GetConformer(conf_id)
        coords = conf.GetPositions()            # (N, 3) numpy array
        new_coords = coords @ R.T + t           # (N, 3)

        for i, (x, y, z) in enumerate(new_coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

        return mol
    def _store_aligned_pose(
        self, smiles: str, name: str, results_dir: Path | None = None
    ) -> None:
        """Parse, align, and extract the ligand from a Boltz PDB into docked_pose_buffer.

        On the first successful call the protein Cα coordinates from that
        prediction become the reference; all later predictions are rotated and
        translated to match it using the Kabsch algorithm.

        *results_dir* is the ``boltz_results_*`` directory holding the
        prediction.  It defaults to the single-molecule layout
        (``boltz_results_{name}``); batched runs pass the batch's results dir.
        """
        if results_dir is None:
            results_dir = self.out_dir / f"boltz_results_{name}"
        pdb_path = (
            results_dir
            / "predictions"
            / name
            / f"{name}_model_0.pdb"
        )
        if not pdb_path.exists():
            logger.warning("PDB not found for %s, skipping pose storage.", name)
            return

        try:
            ca_coords, pdb_mol = self._parse_pdb(pdb_path)
        except Exception as exc:
            logger.warning("PDB parse failed for %s: %s", name, exc)
            return

        if ca_coords is None or len(ca_coords) == 0:
            logger.warning("No Cα atoms found in PDB for %s.", name)
            return

        with self._ref_lock:
            if self._ref_ca_coords is None:
                self._ref_ca_coords = ca_coords.copy()
                R = np.eye(3)
                t = np.zeros(3)
                print(f"  [Boltz2 alignment] {name}: established reference frame "
                      f"({len(ca_coords)} Cα atoms)")
            else:
                if len(ca_coords) != len(self._ref_ca_coords):
                    logger.warning(
                        "Cα count mismatch for %s (%d vs ref %d); "
                        "skipping alignment.",
                        name, len(ca_coords), len(self._ref_ca_coords),
                    )
                    R = np.eye(3)
                    t = np.zeros(3)
                else:
                    R, t = self._kabsch(ca_coords, self._ref_ca_coords)
                    aligned_ca = ca_coords @ R.T + t
                    rmsd = float(np.sqrt(np.mean(
                        np.sum((aligned_ca - self._ref_ca_coords) ** 2, axis=1)
                    )))
                    print(f"  [Boltz2 alignment] {name}: Cα RMSD to reference = "
                          f"{rmsd:.3f} Å (n={len(ca_coords)})")
                    if rmsd > 2.0:
                        logger.warning(
                            "Kabsch alignment for %s has high Cα RMSD (%.3f Å) — "
                            "predicted backbone may have shifted from the reference "
                            "frame; downstream interaction-profile conditioning for "
                            "this molecule may not be comparable to the rest of the "
                            "population.",
                            name, rmsd,
                        )

        if pdb_mol is None:
            logger.warning("No ligand found in PDB for %s.", name)
            return

        # Apply the alignment rotation/translation to the ligand coordinates.

        pdb_mol = self._apply_se3(pdb_mol, R, t)

        mol = self._build_rdkit_mol(smiles, pdb_mol)
        if mol is not None:
            self.docked_pose_buffer[smiles] = {"docked_mol": mol}

    def _parse_pdb(
        self,
        pdb_path: Path,
    ):
        """Extract Cα coordinates and the parsed molecule from a Boltz PDB.

        Returns
        -------
        ca_coords : np.ndarray (N, 3) or None
            Cα coordinates for the protein backbone.  None if no Cα atoms are
            found.
        pdb_mol : Chem.Mol or None
        """
        mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=True)
        if mol is None:
            return None, None

        conf = mol.GetConformer()
        ca_coords_list: List[List[float]] = []

        for atom in mol.GetAtoms():
            info = atom.GetMonomerInfo()
            if info is None:
                continue
            if info.GetName().strip().upper() == "CA":
                pos = conf.GetAtomPosition(atom.GetIdx())
                ca_coords_list.append([pos.x, pos.y, pos.z])

        ca_arr = np.array(ca_coords_list, dtype=float) if ca_coords_list else None
        return ca_arr, mol

    @staticmethod
    def _kabsch(
        mobile: np.ndarray,
        ref: np.ndarray,
    ):
        """Kabsch algorithm: return (R, t) that minimises RMSD of mobile→ref.

        R is (3×3), t is (3,).  Apply as:  aligned = mobile @ R.T + t
        """
        mob_center = mobile.mean(axis=0)
        ref_center = ref.mean(axis=0)
        mob_c = mobile - mob_center
        ref_c = ref - ref_center

        H = mob_c.T @ ref_c
        U, _, Vt = np.linalg.svd(H)
        # Correct for reflection
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = ref_center - mob_center @ R.T
        return R, t

    def _build_rdkit_mol(
        self,
        smiles: str,
        pdb_mol: Chem.Mol,
    ) -> Chem.Mol | None:
        """Extract the ligand from a Boltz PDB mol and return it with correct chemistry.

        Splits ``pdb_mol`` by PDB chain ID, isolates the ligand chain
        (``self.ligand_chain_id``), and optionally corrects bond orders using
        ``smiles`` as a template via :func:`AssignBondOrdersFromTemplate`.
        The Boltz-predicted 3-D coordinates are preserved throughout.
        Hydrogens are added with :func:`AddHs` before returning.
        """
        mol_split = Chem.rdmolops.SplitMolByPDBChainId(pdb_mol)

        ligand = mol_split[self.ligand_chain_id.strip().upper()]
        ligand = Chem.RemoveHs(ligand)

        # assign bond orders from reference smiles if provided, else keep RDKit's best guess
        if smiles is not None:
            reference_mol = Chem.MolFromSmiles(smiles)
            prepared_ligand = AllChem.AssignBondOrdersFromTemplate(reference_mol, ligand)
            prepared_ligand.AddConformer(ligand.GetConformer(0))

        else:
            prepared_ligand = ligand

        # protonate ligand
        prepared_ligand = Chem.rdmolops.AddHs(prepared_ligand, addCoords=True)
        prepared_ligand = Chem.MolFromMolBlock(Chem.MolToMolBlock(prepared_ligand), removeHs=False)

        # return ligand
        return prepared_ligand


    def save_docked_poses(self, output_path: str | Path) -> int:
        """Write all buffered docked poses to an SDF file.

        Arguments
        ---------
        output_path : str or Path
            Destination SDF file.  Parent directories are created if needed.

        Returns
        -------
        int
            Number of molecules written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = Chem.SDWriter(str(output_path))
        n_written = 0
        for smiles, entry in self.docked_pose_buffer.items():
            mol = entry.get("docked_mol")
            if mol is None:
                continue
            mol.SetProp("SMILES", smiles)
            writer.write(mol)
            n_written += 1
        writer.close()

        logger.info("Wrote %d docked poses to %s", n_written, output_path)
        return n_written

    def _parse_output(self, name: str, results_dir: Path | None = None) -> float:
        """Read Boltz2's affinity JSON and return the configured score field.

        *results_dir* is the ``boltz_results_*`` directory (defaults to the
        single-molecule ``boltz_results_{name}`` layout; batched runs pass the
        batch's results dir).
        """
        # boltz predict writes outputs to {results_dir}/predictions/{name}/
        if results_dir is None:
            results_dir = self.out_dir / f"boltz_results_{name}"
        affinity_json = (
            results_dir / "predictions" / name / f"affinity_{name}.json"
        )
        if not affinity_json.exists():
            raise FileNotFoundError(
                f"Affinity output not found: {affinity_json}"
            )

        with open(affinity_json) as fh:
            data = json.load(fh)

        if self.score_field not in data:
            raise KeyError(
                f"Field '{self.score_field}' not in {affinity_json}: "
                f"{list(data.keys())}"
            )

        raw = float(data[self.score_field])

        # Negate probability so "lower is better" still holds.
        if self.score_field == "affinity_probability_binary":
            return -raw

        # affinity predicted value is in log10(uM) units; convert to kcal/mol
        return 1.3646 * (raw - 6)