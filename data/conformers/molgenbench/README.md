# MolGenBench Hit-to-Lead inputs

Prepared inputs for the [MolGenBench](https://github.com/CAODH/MolGenBench) Hit-to-Lead conditional benchmark containing 600 reference ligands and their conditioning indices.

## Files

| File | Format | Contents |
|------|--------|----------|
| `molgenbench_inputs_20260609.pkl` | `list[tuple[str, ndarray]]` | `(molblock, xTB partial charges)` per reference. Passed to `generate.py` as `--molblock-charges`. |
| `data_summary.pkl` | `dict` | Metadata aligned with the input pickle: `uniprot_ids`, `series_ids`, `formal_charges`, `charge_stats`. Used when copying outputs into the MolGenBench directory tree. |
| `interaction_pharm_prioritization.json` | `{index: [0\|1, ...]}` | Per-pharmacophore priority mask for `--pharm-masks` in `pharm-priority` / `pharm-priority-scaffold` modes. Length matches the number of extracted pharmacophores for that reference. |
| `interaction_atom_indices.json` | `{index: [atom_idx, ...]}` | Interaction-site atoms for `--scaffold-atoms` in `pharm-priority-scaffold` mode. |
