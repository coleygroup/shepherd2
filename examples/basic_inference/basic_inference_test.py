"""Basic end-to-end conditional inference example.
This tests to ensure that shepherd and xTB works as expected.

Embed a SMILES string, extract all interaction modalities, generate five
conditioned samples, convert them to xTB-relaxed RDKit molecules, and evaluate
the generated tensors with shepherd-score.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch
from rdkit import Chem
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.evaluations.evaluate.pipelines import ConditionalEvalPipeline

from shepherd import load_model
from shepherd.extract import mol_charges_from_samples
from shepherd.interaction_profile import extract_interaction_profile

torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", default="CN1C=NC2=C1C(=O)N(C(=O)N2C)C", help="Conditioning molecule")
    parser.add_argument("--checkpoint", type=Path, help="Optional local checkpoint; omit to download the default model")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
        help="Default: CUDA when available, otherwise CPU. Warning: CPU inference is very slow and may not work due to compilation.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory for samples.pkl, SDF, and smiles.txt")
    parser.add_argument("--batch-size", type=int, default=5, help="Kept small for quick testing",)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional directory for model cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.output_dir is not None:
        args.output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load ShEPhERD-2
    model = load_model(
        device=device,
        local_checkpoint_path=(str(args.checkpoint) if args.checkpoint else None),
        cache_dir=(str(args.cache_dir) if args.cache_dir else None),
    )

    # 2. SMILES -> explicit H 3D conformer -> interaction profile. xTB relaxes
    # the input and supplies the partial charges used for ESP.
    reference_mol = embed_conformer_from_smiles(
        args.smiles, MMFF_optimize=True, random_seed=1
    )
    if reference_mol is None:
        raise RuntimeError(f"Could not embed SMILES: {args.smiles!r}")

    profile = extract_interaction_profile(reference_mol, xtb_optimize=True)
    if profile is None:
        raise RuntimeError("Could not extract the conditioning interaction profile")

    # 3. Generate by conditioning on shape, ESP, and pharmacophores
    start = time.perf_counter()
    samples = model.generate(
        batch_size=args.batch_size,
        N_x1=profile.n_atoms,
        N_x4=profile.n_pharms,
        condition=profile,
        condition_modalities="all",
    )
    elapsed = time.perf_counter() - start
    print(f"Generated {len(samples)} samples on {device} in {elapsed:.1f} s")

    if args.output_dir is not None:
        with (args.output_dir / "samples.pkl").open("wb") as handle:
            pickle.dump(samples, handle)

    # 4a. Convert samples to RDKit molecules with xTB relaxation
    relaxed_mols = []
    mol_charges = mol_charges_from_samples(samples, xtb_optimize=True, xtb_timeout=180, verbose=True)
    relaxed_mols = [mol for mol, _ in mol_charges if mol is not None]

    if args.output_dir is not None:
        writer = Chem.SDWriter(str(args.output_dir / "sampled_molecules.sdf"))
        try:
            for mol in relaxed_mols:
                if mol is not None:
                    writer.write(mol)
        finally:
            writer.close()

    smiles = [
        Chem.MolToSmiles(Chem.RemoveHs(mol)) if mol is not None else None
        for mol in relaxed_mols
    ]

    print(f'Sampled smiles: {smiles}')
    if args.output_dir is not None:
        (args.output_dir / "smiles.txt").write_text(
            "\n".join(smile or "INVALID" for smile in smiles) + "\n"
        )

    # 4b. Use shepherd-score to evaluate the generated molecules against the conditioning profile
    pipeline = ConditionalEvalPipeline(
        **model.to_shepherd_score_inputs(
            samples,
            condition=profile,
            condition_modalities="all",
        )
    )
    pipeline.evaluate(num_workers=1, num_processes=1, verbose=True)
    print(
        "shepherd-score: "
        f"valid={pipeline.num_valid}/{args.batch_size}, "
        f"valid_post_opt={pipeline.num_valid_post_opt}/{args.batch_size}"
    )


if __name__ == "__main__":
    main()
