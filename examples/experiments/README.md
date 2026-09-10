# Experiment scripts

Scripts used for preprint generation, evaluation, and optimization. Run any script with `--help` for all options.

## Evaluation

`[run_moses_benchmark.py](run_moses_benchmark.py)`: 20 samples for each of 100 MOSES-aq references.

```bash
# Interaction-conditioned generation
python examples/experiments/run_moses_benchmark.py \
    --mode conditional --output-dir out/moses_conditional

python examples/experiments/run_moses_benchmark.py \
    --mode conditional --output-dir out/moses_conditional
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

`[run_molgenbench_generate.py](run_molgenbench_generate.py)`: 200 samples for each of 600 MolGenBench references.

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

`[run_moses_composition_evals.py](run_moses_composition_evals.py)`: compose pairs of MOSES-aq conditions and evaluate against both references.

```bash
python examples/experiments/run_moses_composition_evals.py \
  --save_dir out/moses_composition
```



## Docking optimization

- `[run_ga_docking_exploitation.py](run_ga_docking_exploitation.py)`: optimize docking score from seed molecules.
- `[run_ga_docking_exploration.py](run_ga_docking_exploration.py)`: merge posed fragments while optimizing docking score and ligand efficiency.

```bash
python examples/experiments/run_ga_docking_exploitation.py \
  --checkpoint model.ckpt --input_mols seeds.sdf --output ga_results.pkl

python examples/experiments/run_ga_docking_exploration.py \
  --checkpoint model.ckpt --frag_dir fragments --output ga_results.pkl
```



## Utilities

- `[molgenbench/convert_shepherd_outputs_to_molgenbench.py](molgenbench/convert_shepherd_outputs_to_molgenbench.py)`: convert generated SDFs to MolGenBench's Hit-to-Lead layout. Requires a downloaded MolGenBench dataset from [Zenodo](https://zenodo.org/records/18183463).
- `[molgenbench/run_molgenbench_generate_wrapper.py](molgenbench/run_molgenbench_generate_wrapper.py)` and `[moses_aq/run_moses_benchmark_wrapper.py](moses_aq/run_moses_benchmark_wrapper.py)`: examples using `[generate.py](../../generate.py)`; only for demonstrations.

