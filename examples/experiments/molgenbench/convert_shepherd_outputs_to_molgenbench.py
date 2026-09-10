"""
Prepare generate.py outputs for MolGenBench benchmarking.

1. Load reference index -> uniprot/series mapping.
2. Read per-reference molecules.sdf from generate.py output.
3. Save valid molecules to .sdf files in the MolGenBench directory layout.
"""
import argparse
import pickle
from pathlib import Path

from rdkit import Chem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_PATH = ROOT / "data" / "conformers" / "molgenbench" / "data_summary.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare generate.py outputs for MolGenBench benchmarking"
    )
    parser.add_argument(
        "--molgenbench-data-dir",
        type=str,
        required=True,
        help="Path to directory containing molgenbench data directory.",
    )
    parser.add_argument(
        "--shepherd-output-dir",
        type=str,
        required=True,
        help="Path to generate.py output directory (contains results/reference_XXX/molecules.sdf).",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to summary file about the reference molecules.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=3,
        help="Round number for the benchmarking run (default: 3).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name for output paths (default: --shepherd-output-dir basename).",
    )
    args = parser.parse_args()

    input_dir = Path(args.shepherd_output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    model_name = args.model_name or input_dir.name

    molgenbench_data_dir = Path(args.molgenbench_data_dir)
    if not molgenbench_data_dir.exists():
        raise FileNotFoundError(f"Molgenbench data directory {molgenbench_data_dir} does not exist")

    summary_file = Path(args.summary_file)
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file {summary_file} does not exist")

    results_dir = input_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist")

    round_num = args.round

    with open(summary_file, "rb") as f:
        summary = pickle.load(f)
    uniprot_ids = summary["uniprot_ids"]
    series_ids = summary["series_ids"]

    assert len(uniprot_ids) == len(series_ids)

    for i, (uniprot_id, series_id) in tqdm(
        enumerate(zip(uniprot_ids, series_ids)), total=len(uniprot_ids)
    ):
        source_path = results_dir / f"reference_{i:03d}" / "molecules.sdf"
        if not source_path.exists():
            continue

        save_dir = (
            molgenbench_data_dir
            / uniprot_id
            / f"Round{round_num}"
            / "Hit_to_Lead_Results"
            / f"Sries{series_id}"
            / model_name
        )

        supplier = Chem.SDMolSupplier(str(source_path), removeHs=False)
        sampled_mols = [mol for mol in supplier if mol is not None]

        save_dir.mkdir(parents=True, exist_ok=True)
        with Chem.SDWriter(save_dir / f"{uniprot_id}_Sries{series_id}_{model_name}.sdf") as writer:
            for j, mol in enumerate(sampled_mols):
                mol.SetProp("_Name", f"{uniprot_id}_Sries{series_id}_{model_name}_{j}")
                writer.write(mol)

        print(
            f"Saved {len(sampled_mols)} molecules to "
            f"{save_dir / f'{uniprot_id}_Sries{series_id}_{model_name}.sdf'}"
        )


if __name__ == "__main__":
    main()
