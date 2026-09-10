#!/usr/bin/env python
"""Merge posed fragments with a GA over docking score and ligand efficiency."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from rdkit import Chem

from shepherd import load_model
from shepherd.optimization import (
    GAConfig,
    InteractionSpaceGA,
    SeedMoleculeInitializer,
    DockingOracle,
    SubstructureFilter,
    DrugLikenessChecker,
    CompositeValidityChecker,
)
from shepherd.optimization.fitness import FitnessOracle

try:
    from shepherd_score.evaluations.docking import DockingEvalPipeline
except ImportError:
    DockingEvalPipeline = None

# Fragment file formats loaded from --frag_dir
SUPPORTED_EXTENSIONS = {".mol", ".sdf", ".mol2", ".pdb", ".pdbqt"}
VALIDITY_MODES = ("drug_likeness", "composite")
MUTATION_MODES = ("conditional", "composed", "ph4_conditioned")
CONDITION_MODES = ("x2", "x3", "x4", "all", "x2_x4")
FRAGMENT_SELECTION_MODES = ("lineage", "use_count")
# Vina search box used when --box_size is not given
DEFAULT_BOX_SIZE = (20.0, 20.0, 20.0)
GAS_PHASE_ALIASES = ("none", "null", "gas")


class LigandEfficiencyOracle(FitnessOracle):
    """Score docking_score / n_heavy_atoms"""

    def __init__(self, docking_oracle: DockingOracle) -> None:
        super().__init__()
        self.docking_oracle = docking_oracle

    def evaluate(self, individuals: list) -> list[float]:
        docking_scores = self.docking_oracle.evaluate_with_cache(individuals)
        results: list[float] = []
        for individual, docking_score in zip(individuals, docking_scores):
            if docking_score == float("inf") or individual.smiles is None:
                results.append(float("inf"))
                continue
            mol = Chem.MolFromSmiles(individual.smiles)
            if mol is None:
                results.append(float("inf"))
                continue
            n_heavy_atoms = mol.GetNumHeavyAtoms()
            if n_heavy_atoms == 0:
                results.append(float("inf"))
                continue
            results.append(docking_score / n_heavy_atoms)
        return results


def _load_pdb_file(path: Path) -> Chem.Mol | None:
    """Parse a .pdb file, reconstructing a single model if RDKit refuses it."""
    mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=True)
    if mol is not None:
        return mol

    with open(path) as handle:
        lines = handle.readlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    if not atom_lines:
        return None
    conect_lines = [line for line in lines if line.startswith("CONECT")]
    block = "".join(atom_lines + conect_lines) + "END\n"
    return Chem.MolFromPDBBlock(block, removeHs=False, sanitize=True)


def _load_mol_file(path: Path) -> list[Chem.Mol]:
    """Load one or more molecules from a single file, preserving 3-D coords."""
    extension = path.suffix.lower()
    mols: list[Chem.Mol] = []

    if extension == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        for mol in suppl:
            if mol is not None and mol.GetNumConformers() > 0:
                mols.append(mol)

    elif extension == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
        if mol is not None and mol.GetNumConformers() > 0:
            mols.append(mol)

    elif extension == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
        if mol is not None and mol.GetNumConformers() > 0:
            mols.append(mol)

    elif extension == ".pdb":
        mol = _load_pdb_file(path)
        if mol is not None and mol.GetNumConformers() > 0:
            mols.append(mol)

    elif extension == ".pdbqt":
        try:
            from meeko import PDBQTMolecule, RDKitMolCreate
            pdbqt_mol = PDBQTMolecule.from_file(str(path))
            candidates = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            rdkit_mol = next(
                (candidate for candidate in candidates if candidate is not None), None
            )
            if rdkit_mol is not None:
                rdkit_mol = Chem.AddHs(rdkit_mol, addCoords=True)
                if rdkit_mol.GetNumConformers() > 0:
                    mols.append(rdkit_mol)
        except ImportError:
            warnings.warn(
                f"meeko is required to load .pdbqt files; skipping {path.name}"
            )
        except Exception as error:
            warnings.warn(f"Failed to load {path.name}: {error}")

    else:
        warnings.warn(
            f"Unsupported file extension '{extension}'; skipping {path.name}"
        )

    return mols


def load_fragments_from_dir(
    directory: str | Path,
    file_types: list[str] | None = None,
    max_mols: int | None = None,
) -> list[Chem.Mol]:
    """Load fragment molecules with 3-D conformers from a directory."""
    if file_types is None:
        allowed = SUPPORTED_EXTENSIONS
    else:
        allowed = {
            extension if extension.startswith(".") else f".{extension}"
            for extension in file_types
        }

    mols: list[Chem.Mol] = []
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Fragment directory not found: {directory}")

    for path in sorted(dir_path.iterdir()):
        if path.suffix.lower() not in allowed:
            continue
        loaded = _load_mol_file(path)
        if not loaded:
            warnings.warn(f"No valid 3-D molecules found in {path.name}; skipping")
            continue
        for mol in loaded:
            mols.append(mol)
            if max_mols is not None and len(mols) >= max_mols:
                return mols

    return mols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model
    parser.add_argument("--checkpoint", required=True,
        help="ShEPhERD model checkpoint (.ckpt)",
    )
    parser.add_argument("--ema_checkpoint", default=None,
        help="Optional EMA checkpoint (.ckpt)",
    )
    parser.add_argument("--device", default=None, help="Device: cuda / cpu / auto")

    # Fragment input
    parser.add_argument("--frag_dir", default=None,
        help="Directory of posed fragment files (mol / sdf / mol2 / pdb / pdbqt); "
             "not needed when resuming",
    )
    parser.add_argument("--file_types", nargs="+", default=None, metavar="EXT",
        help="File extensions to load from --frag_dir (default: all supported "
             "types). Example: --file_types .mol .sdf",
    )
    parser.add_argument("--max_initial_mols", type=int, default=None,
        help="Cap on the number of seed fragments to load",
    )
    parser.add_argument("--seed_xtb_optimize", action="store_true", default=False,
        help="Relax the seed fragment geometries and charges with xTB before "
             "extracting their interaction profiles. Off by default",
    )

    # Docking target
    parser.add_argument("--pdb_id", default="1iep",
        help="PDB ID of a built-in shepherd_score docking target",
    )
    parser.add_argument("--receptor_pdbqt", default=None,
        help="Custom receptor .pdbqt, overriding --pdb_id",
    )
    parser.add_argument("--center", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Pocket center; required with --receptor_pdbqt",
    )
    parser.add_argument("--box_size", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help=f"Search box size (default {' '.join(str(edge) for edge in DEFAULT_BOX_SIZE)})",
    )
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--docking_cpus", type=int, default=32)

    # Validity checker
    parser.add_argument("--validity", default="drug_likeness", choices=VALIDITY_MODES,
        help="drug_likeness = Lipinski Ro5 filter (default); "
             "composite = Ro5 plus PAINS/Brenk substructure filtering",
    )
    parser.add_argument("--mw_limit", type=float, default=500.0,
        help="DrugLikenessChecker: max molecular weight (Da)",
    )
    parser.add_argument("--logp_limit", type=float, default=5.0,
        help="DrugLikenessChecker: max Wildman-Crippen LogP",
    )
    parser.add_argument("--hbd_limit", type=int, default=5,
        help="DrugLikenessChecker: max H-bond donors",
    )
    parser.add_argument("--hba_limit", type=int, default=10,
        help="DrugLikenessChecker: max H-bond acceptors",
    )
    parser.add_argument("--max_violations", type=int, default=1,
        help="DrugLikenessChecker: max Ro5 rule violations (0 = strict)",
    )
    parser.add_argument("--tpsa_limit", type=float, default=None,
        help="DrugLikenessChecker: optional TPSA upper bound (square angstroms)",
    )
    parser.add_argument("--rotatable_bonds_limit", type=int, default=None,
        help="DrugLikenessChecker: optional max rotatable bonds",
    )
    parser.add_argument("--sa_score_limit", type=float, default=4.5,
        help="DrugLikenessChecker: max SA score (1-10)",
    )
    parser.add_argument("--qed_min", type=float, default=0.2,
        help="DrugLikenessChecker: min QED (0-1)",
    )
    parser.add_argument("--validity_verbose", action="store_true", default=False,
        help="Print per-molecule Ro5 and filter details as they are checked",
    )

    # GA hyperparameters
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--max_iterations_fraction", type=float, default=4.0,
        help="Max iterations as a fraction of the total search space size",
    )
    parser.add_argument("--num_generations", type=int, default=50)
    parser.add_argument("--mutation_mode", default="conditional", choices=MUTATION_MODES)
    parser.add_argument("--crossover_w", type=float, nargs=2, default=[0.3, 0.3],
        metavar=("W_A", "W_B"),
    )
    parser.add_argument("--crossover_prob", type=float, default=0.1)
    parser.add_argument("--selection", default="tournament", choices=("tournament",),
        help="Base selection method, automatically promoted to pareto_tournament "
             "in multi-objective mode",
    )
    parser.add_argument("--tournament_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=None,
        help="Restrict the parent pool to the top k individuals before tournament "
             "selection. None (default) uses the whole population.",
    )
    parser.add_argument("--elite_fraction", type=float, default=0.1)
    parser.add_argument("--N_x1_range", type=int, nargs=2, default=[0, 2],
        metavar=("LO", "HI"),
    )
    parser.add_argument("--N_x4_range", type=int, nargs=2, default=[0, 2],
        metavar=("LO", "HI"),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mutate_batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--condition_mode", default="all", choices=CONDITION_MODES)
    parser.add_argument("--seed", type=int, default=42)

    # Fragment merging
    parser.add_argument("--no_fragment_merge_mode", action="store_true",
        help="Disable fragment-merge crossover, using the standard MAX reference "
             "instead of SUM. Off by default, so fragment merging is enabled.",
    )
    parser.add_argument("--fragment_atom_threshold", type=int, default=20,
        help="Max heavy-atom count for a parent to qualify as a fragment in "
             "fragment_merge_mode (default: 20)",
    )
    parser.add_argument("--fragment_inclusion_prob", type=float, default=0.5,
        help="Probability of including a fragment parent in crossover when "
             "fragment_merge_mode is enabled (default: 0.5)",
    )
    parser.add_argument("--fragment_merge_alpha", type=float, default=0.5,
        help="Fraction of the smaller parent's atoms added to the larger parent's "
             "atom count in a mixed merge (default: 0.5)",
    )
    parser.add_argument("--fragment_selection_mode", default="lineage",
        choices=FRAGMENT_SELECTION_MODES,
        help="How parent_b is picked. 'lineage' avoids fragments already in "
             "either parent's ancestry; 'use_count' round-robins by least-used "
             "fragment.",
    )

    # Molecule conversion
    parser.add_argument("--xtb_optimize", action="store_true", default=True,
        help="Relax GA offspring geometries and charges with xTB before scoring "
             "(default: on)",
    )
    parser.add_argument("--no_xtb_optimize", dest="xtb_optimize", action="store_false",
        help="Skip xTB relaxation of GA offspring (faster, lower-fidelity charges)",
    )
    parser.add_argument("--profile_conversion", default="fixed",
        choices=("fixed", "inferred"),
        help="How the molecular charge is obtained when converting a generated "
             "sample. 'fixed' pins it to 0 (default); 'inferred' recovers it "
             "from the geometry by trying 0, +/-1, +/-2.",
    )
    parser.add_argument("--profile_solvent", default="water",
        help="Implicit solvent for the xTB calls behind profile extraction "
             "(default: water). Pass 'none' to run them in the gas phase for "
             "ESP charges.",
    )

    # EDM sampler
    parser.add_argument("--use_stochastic", action="store_true", default=False,
        help="Enable churn noise in EDM sampling",
    )
    parser.add_argument("--no_stochastic", dest="use_stochastic", action="store_false",
        help="Disable EDM churn noise (pure ODE)",
    )
    parser.add_argument("--shepherd_pred", action="store_true",
        help="Use stochastic shepherd prediction instead of an ODE step in EDM",
    )
    parser.add_argument("--early_stop_edm", type=float, default=-1,
        help="Truncate EDM sampling. -1 runs all --num_steps (default); a value "
             "in [0, 1] is a fraction of num_steps; a value above 1 is an "
             "absolute step count.",
    )
    parser.add_argument("--sigma_max", type=float, default=3.0,
        help="Override the EDM schedule sigma_max",
    )
    parser.add_argument("--sigma_min", type=float, default=None,
        help="Override the EDM schedule sigma_min",
    )
    parser.add_argument("--rho", type=float, default=None,
        help="Override the EDM schedule rho",
    )
    parser.add_argument("--alignment_start_frac", type=float, default=0.0,
        help="Fraction of steps, counted from the end, over which ESP alignment "
             "is applied during EDM sampling",
    )
    parser.add_argument("--alignment_interval", type=int, default=10,
        help="Recompute the ESP alignment every N EDM steps",
    )
    parser.add_argument("--alignment_mode", default="so3", choices=("so3", "se3"),
        help="EDM ESP alignment mode: so3 (rotation only) or se3 (full rigid)",
    )
    parser.add_argument("--alignment_ema_alpha", type=float, default=1.0,
        help="EMA smoothing factor for EDM alignment (1.0 = no smoothing)",
    )

    # Checkpointing and resume
    parser.add_argument("--checkpoint_dir", default=None,
        help="Directory for per-generation GA checkpoints",
    )
    parser.add_argument("--checkpoint_every", type=int, default=1)
    parser.add_argument("--resume", default=None,
        help="GA checkpoint .pkl file to resume from",
    )

    # Output
    parser.add_argument("--output", default="ga_frag_merge_le_results.pkl")

    args = parser.parse_args()
    if args.resume is None and args.frag_dir is None:
        parser.error("--frag_dir is required when not resuming")
    if args.receptor_pdbqt is not None and args.center is None:
        parser.error("--center is required with --receptor_pdbqt")
    return args


def _build_docking_oracle(args: argparse.Namespace) -> DockingOracle:
    """Build the docking oracle that the LE oracle also reads its cache from."""
    if DockingEvalPipeline is None:
        raise ImportError(
            "shepherd_score.evaluations.docking is required. "
            "Install vina/meeko and shepherd_score."
        )

    if args.receptor_pdbqt is not None:
        docking_target_info = {
            "custom": {
                "center": tuple(args.center),
                "size": tuple(args.box_size) if args.box_size else DEFAULT_BOX_SIZE,
                "pdbqt": args.receptor_pdbqt,
                "pH": 7.4,
            }
        }
        docking_pipeline = DockingEvalPipeline(
            pdb_id="custom",
            num_processes=args.docking_cpus,
            docking_target_info_dict=docking_target_info,
            verbose=0,
        )
    else:
        docking_pipeline = DockingEvalPipeline(
            pdb_id=args.pdb_id,
            num_processes=args.docking_cpus,
            verbose=0,
        )

    return DockingOracle(
        docking_pipeline=docking_pipeline,
        exhaustiveness=args.exhaustiveness,
        n_poses=1,
        protonate=False,
        verbose=True,
    )


def _build_validity_checker(args: argparse.Namespace):
    """Build the validity checker selected by --validity."""
    ro5 = DrugLikenessChecker(
        mw_limit=args.mw_limit,
        logp_limit=args.logp_limit,
        hbd_limit=args.hbd_limit,
        hba_limit=args.hba_limit,
        max_violations=args.max_violations,
        tpsa_limit=args.tpsa_limit,
        rotatable_bonds_limit=args.rotatable_bonds_limit,
        sa_score_limit=args.sa_score_limit,
        qed_min=args.qed_min,
        verbose=args.validity_verbose,
    )
    if args.validity == "drug_likeness":
        return ro5
    if args.validity == "composite":
        return CompositeValidityChecker([
            SubstructureFilter(filter_pains=True, filter_brenk=True),
            ro5,
        ])
    raise ValueError(f"Unknown validity mode: {args.validity}")


def _ga_config(args: argparse.Namespace, fragment_merge_mode: bool) -> GAConfig:
    """Translate the parsed arguments into a GAConfig."""
    return GAConfig(
        population_size=args.population_size,
        max_iterations_fraction=args.max_iterations_fraction,
        num_generations=args.num_generations,
        # Seed fragments arrive already posed, so the initial population needs
        # no new docking pass
        skip_initial_eval=True,
        elite_fraction=args.elite_fraction,
        crossover_weight_a=args.crossover_w[0],
        crossover_weight_b=args.crossover_w[1],
        crossover_prob=args.crossover_prob,
        mutation_mode=args.mutation_mode,
        selection_method=args.selection,
        tournament_size=args.tournament_size,
        top_k=args.top_k,
        N_x1_range=tuple(args.N_x1_range),
        N_x4_range=tuple(args.N_x4_range),
        batch_size=args.batch_size,
        mutate_batch_size=args.mutate_batch_size,
        num_steps=args.num_steps,
        condition_mode=args.condition_mode,
        fragment_merge_mode=fragment_merge_mode,
        fragment_atom_threshold=args.fragment_atom_threshold,
        fragment_inclusion_prob=args.fragment_inclusion_prob,
        fragment_merge_alpha=args.fragment_merge_alpha,
        fragment_selection_mode=args.fragment_selection_mode,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        verbose=True,
        use_stochastic=args.use_stochastic,
        shepherd_pred=args.shepherd_pred,
        xtb_optimize=args.xtb_optimize,
        profile_conversion=args.profile_conversion,
        profile_solvent=(
            None
            if str(args.profile_solvent).lower() in GAS_PHASE_ALIASES
            else args.profile_solvent
        ),
        early_stop_edm=args.early_stop_edm,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        rho=args.rho,
        alignment_start_frac=args.alignment_start_frac,
        alignment_interval=args.alignment_interval,
        alignment_mode=args.alignment_mode,
        alignment_ema_alpha=args.alignment_ema_alpha,
    )


def _print_pareto_front(ga: InteractionSpaceGA, num_shown: int = 10) -> None:
    """Print the rank-0 front, best docking score first."""
    front_pool = ga.get_all_time_best() if ga.cfg.top_k is not None else ga.population
    pareto_front = [
        individual
        for individual in front_pool
        if getattr(individual, "pareto_rank", 0) == 0
    ]
    if not pareto_front:
        pareto_front = front_pool[:]

    def first_objective(individual):
        scores = getattr(individual, "fitness_scores", [])
        return scores[0] if scores else individual.fitness_score

    pareto_front.sort(key=first_objective)

    print("Pareto-front molecules (rank 0), sorted by docking score:")
    print(f"  {'Dock (kcal/mol)':>16}  {'LE':>8}  {'n_heavy':>7}  SMILES")
    for individual in pareto_front[:num_shown]:
        scores = getattr(individual, "fitness_scores", [])
        docking_score = scores[0] if len(scores) > 0 else float("nan")
        ligand_efficiency = scores[1] if len(scores) > 1 else float("nan")
        mol = Chem.MolFromSmiles(individual.smiles) if individual.smiles else None
        n_heavy_atoms = mol.GetNumHeavyAtoms() if mol else -1
        print(
            f"  {docking_score:>16.4f}  {ligand_efficiency:>8.4f}  "
            f"{n_heavy_atoms:>7d}  {individual.smiles}"
        )


def main() -> None:
    args = parse_args()

    print("Loading ShEPhERD-2 model...")
    model = load_model(
        local_checkpoint_path=args.checkpoint,
        device=args.device,
    )
    if args.ema_checkpoint is not None:
        model.load_ema_weights_for_inference(ema_checkpoint_path=args.ema_checkpoint)

    # A resumed run restores its population from the checkpoint instead
    mols: list[Chem.Mol] | None = None
    if args.resume is None:
        print(f"Loading fragments from {args.frag_dir}...")
        mols = load_fragments_from_dir(
            args.frag_dir,
            file_types=args.file_types,
            max_mols=args.max_initial_mols,
        )
        if not mols:
            raise RuntimeError(
                f"No valid 3-D molecules found in {args.frag_dir}. Ensure files "
                "have 3-D conformers and use a supported format."
            )
        print(f"  Loaded {len(mols)} fragment(s).")

    docking_oracle = _build_docking_oracle(args)
    # The LE oracle reuses cached docking scores, so it adds no docking calls
    le_oracle = LigandEfficiencyOracle(docking_oracle)
    validity_checker = _build_validity_checker(args)
    initializer = SeedMoleculeInitializer(params=model.params)
    fragment_merge_mode = not args.no_fragment_merge_mode
    config = _ga_config(args, fragment_merge_mode)

    print("Objectives: [0] docking score (kcal/mol)  [1] LE = score / n_heavy_atoms")
    print("Selection: NSGA-II pareto_tournament (auto-activated for 2+ oracles)")
    print(f"Validity checker: {validity_checker.__class__.__name__}")
    print(
        f"Fragment merge mode: {'ENABLED' if fragment_merge_mode else 'DISABLED'}"
        + (
            f" (threshold <= {args.fragment_atom_threshold} heavy atoms)"
            if fragment_merge_mode
            else ""
        )
    )
    print(
        f"Profile conversion: {args.profile_conversion} | "
        f"xTB solvent for profile extraction: {args.profile_solvent}"
    )

    ga = InteractionSpaceGA(
        model_pl=model.lightning_module,
        oracles=[docking_oracle, le_oracle],
        config=config,
        validity_checker=validity_checker,
        population_initializer=initializer,
    )

    if args.resume is not None:
        ga.resume(args.resume)
        print(
            f"\nResumed from {args.resume}; continuing to generation "
            f"{config.num_generations}...\n"
        )
    else:
        if args.checkpoint_dir:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ga.initialize_population(mols=mols, xtb_optimize=args.seed_xtb_optimize)
        # More seeds than the population size are dropped rather than ranked
        if len(ga.population) > config.population_size:
            ga.population = ga.population[:config.population_size]
        print(
            f"\nStarting GA with {len(ga.population)} seed fragment(s) over "
            f"{config.num_generations} generations...\n"
        )

    ga.run()
    ga.save_results(path=args.output)

    _print_pareto_front(ga)
    print(f"Saved results to {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
