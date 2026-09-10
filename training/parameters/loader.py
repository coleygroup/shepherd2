"""Loads training hyperparameters from YAML + OmegaConf.

Replaces the old `importlib.import_module(f'training.parameters.2025.{model_name}').params`
pattern. `load_params` merges an experiment override YAML on top of `base.yaml` and returns
a plain nested dict with the exact shape LightningModule/Model/HeteroDataset expect
(no DictConfig/OmegaConf types), so it remains safe to pickle into checkpoint hyper_parameters
via `LightningModule.save_hyperparameters()`.
"""

from pathlib import Path

from omegaconf import OmegaConf

PARAMETERS_ROOT = Path(__file__).resolve().parent
BASE_YAML = PARAMETERS_ROOT / "base.yaml"
EXPERIMENTS_DIR = PARAMETERS_ROOT / "experiments"

# (dotted path into params, formula computed from the resolved `constants` block)
_DERIVED_INJECTIONS = (
    (("x1", "decoder", "input_node_channels"), lambda c: c["num_atom_types"] + c["num_charge_types"]),
    (("x1", "decoder", "denoiser", "output_node_channels"), lambda c: c["num_atom_types"] + c["num_charge_types"]),
    (("x1", "decoder", "encoder", "input_bond_channels"), lambda c: c["num_bond_types"]),
    (("x1", "decoder", "denoiser", "output_bond_channels"), lambda c: c["num_bond_types"]),
    (("x4", "decoder", "input_node_channels"), lambda c: c["num_pharmacophore_types"]),
    (("x4", "decoder", "denoiser", "output_node_channels"), lambda c: c["num_pharmacophore_types"]),
    (("dataset", "x4", "max_node_types"), lambda c: c["num_pharmacophore_types"]),
)


def _get_path(d, path):
    for key in path:
        d = d[key]
    return d


def _set_path(d, path, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def _resolve_experiment_path(experiment_yaml):
    path = Path(experiment_yaml)
    if path.is_absolute():
        return path
    for candidate in (PARAMETERS_ROOT / path, EXPERIMENTS_DIR / path):
        if candidate.exists():
            return candidate
    return path


def _compute_constants(raw_constants):
    constants = dict(raw_constants)
    constants["num_atom_types"] = len(constants["atom_types"])
    constants["num_charge_types"] = int(constants["diffuse_formal_charges"]) * len(constants["charge_types"])
    constants["num_bond_types"] = len(constants["bond_types"])
    return constants


def load_params(experiment_yaml) -> dict:
    """Load `base.yaml`, merge the given experiment override on top of it, and fill in
    the channel-count fields derived from `constants` (kept `null` in the YAML).

    `experiment_yaml` may be an absolute path, a path relative to the working directory,
    a path relative to `training/parameters/` (e.g. "experiments/foo.yaml"), or a bare
    filename resolved under `training/parameters/experiments/`.
    """
    override_cfg = OmegaConf.load(_resolve_experiment_path(experiment_yaml))
    merged = OmegaConf.merge(OmegaConf.load(BASE_YAML), override_cfg)
    params = OmegaConf.to_container(merged, resolve=True)

    constants = _compute_constants(params.pop("constants"))
    for path, formula in _DERIVED_INJECTIONS:
        current = _get_path(params, path)
        if current is not None:
            raise ValueError(
                f"Expected derived field '{'.'.join(path)}' to be null in YAML "
                f"(it is computed by load_params from `constants`), but got {current!r}."
            )
        _set_path(params, path, formula(constants))

    return params
