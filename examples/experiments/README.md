# Experiment scripts

Scripts used for preprint generation, evaluation, and optimization. Run any script with `--help` for all options.

While the basic operations for interaction-conditioned generation and composition can be done using this repository alone, to run the exact experiments you will need to download the `shepherd2-moses-input_data.zip` (e.g., reference molecules, PDBQTs for docking, etc.) from [Zenodo](https://doi.org/10.5281/zenodo.22798054).

## Evaluation


### Interaction conditioned generation

#### MOSES
[run_moses_benchmark.py](run_moses_benchmark.py): 20 samples for each of 100 MOSES-aq references.

```bash
# Interaction-conditioned generation
python examples/experiments/run_moses_benchmark.py \
    --mode conditional --output-dir out/moses_conditional

python examples/experiments/run_moses_benchmark.py \
    --mode conditional --output-dir out/moses_conditional \
    --add-atoms 4 --add-pharms 2

python examples/experiments/run_moses_benchmark.py \
    --mode conditional --output-dir out/moses_conditional \
    --add-atoms 8 --add-pharms 4

# Pharmacophore conditioned generation
python examples/experiments/run_moses_benchmark.py \
    --mode pharm-full --output-dir out/moses_pharm_full

python examples/experiments/run_moses_benchmark.py \
    --mode pharm-priority --output-dir out/moses_pharm_priority

python examples/experiments/run_moses_benchmark.py \
    --mode pharm-priority-only --output-dir out/moses_pharm_priority_only

# Scaffold conditioned generation
python examples/experiments/run_moses_benchmark.py \
    --mode scaffold-random-hetero --output-dir out/moses_scaffold_random_hetero

python examples/experiments/run_moses_benchmark.py \
    --mode scaffold-brics --output-dir out/moses_scaffold_brics
```

#### MolGenBench
[run_molgenbench_generate.py](run_molgenbench_generate.py): 200 samples for each of 600 MolGenBench references.

Automatically loads data from [molgenbench/](/data/conformers/molgenbench/).

```bash
# Ligand-based: Sh+E+Ph
python examples/experiments/run_molgenbench_generate.py \
  --mode conditional --output-dir out/molgenbench_conditional

# Structure-based: Sh+E+IPh
python examples/experiments/run_molgenbench_generate.py \
  --mode pharm-priority --output-dir out/molgenbench_pharm_priority

# Structure-based: Sh+E+IPh+IA
python examples/experiments/run_molgenbench_generate.py \
  --mode pharm-priority-scaffold --output-dir out/molgenbench_pharm_priority_scaffold
```

### Interaction conditioned composition
[run_moses_composition_evals.py](run_moses_composition_evals.py): compose pairs of MOSES-aq conditions and evaluate against both references.

While this constructs the pairs on-the-fly with a fixed seed, we may also provide the exact pairs from the Zenodo in `shepherd2-moses/input_data/interaction_conditioned_composition/moses_test_scaffold_molblock_charges_pairs_n100_seed42.pkl` that can be directly supplied to `--test-path`. Note that `--composition-mode conditional` is used here because we do not include an unconditional component.

```bash
# AND
python examples/experiments/run_moses_composition_evals.py \
  --composition-mode conditional \
  --weights-conditions 0.5 0.5 \
  --output-dir out/moses_composition

# AND NOT
python examples/experiments/run_moses_composition_evals.py \
  --composition-mode conditional \
  --weights-conditions 0.5 -0.5 \
  --output-dir out/moses_composition
```


## Docking optimization

- [run_ga_docking_exploration.py](run_ga_docking_exploration.py): merge fragments while optimizing docking score and ligand efficiency.
- [run_ga_docking_exploitation.py](run_ga_docking_exploitation.py): optimize docking score from seed molecules.

### Bioisosteric fragment merging

For the case study, you will need to download the `shepherd2-moses-input_data.zip`. The relevant files are in `shepherd2-moses/input_data/bioisosteric_fragment_merging/`. In particular, the PDBQT for PDB docking is `8cnx_rec.pdbqt`, and the fragments are `fragments/*.mol`. Note that the fragments subfolder is also in this repo: [fragments/](/data/conformers/fragment_merging/fragments/).

```bash
# Exploration strategy to merge fragments
python examples/experiments/run_ga_docking_exploration.py \
  --frag-dir data/conformers/fragment_merging/fragments \
  --receptor-pdbqt path/to/8cnx_rec.pdbqt \
  --center -6.87 -5.28 -3.86 \
  --output ga_frag_merge_exploration_results.pkl

# Stage 2: single-objective refinement from exploration results
python examples/experiments/run_ga_docking_exploitation.py \
  --input-mols ga_frag_merge_exploration_results.pkl \
  --input-format past_ga \
  --receptor-pdbqt path/to/8cnx_rec.pdbqt \
  --center -6.87 -5.28 -3.86 \
  --output ga_frag_merge_exploitation_results.pkl
```



## Utilities

- [molgenbench/convert_shepherd_outputs_to_molgenbench.py](molgenbench/convert_shepherd_outputs_to_molgenbench.py): convert generated SDFs to MolGenBench's Hit-to-Lead layout. Requires a downloaded MolGenBench dataset from [Zenodo](https://zenodo.org/records/18183463).
- [molgenbench/run_molgenbench_generate_wrapper.py](molgenbench/run_molgenbench_generate_wrapper.py) and [moses_aq/run_moses_benchmark_wrapper.py](moses_aq/run_moses_benchmark_wrapper.py): examples using [generate.py](/generate.py); only for demonstrations.

