"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copied from fairchem.core.common.utils.py
- most crucial functions for eqV3 encoder are: conditional_grad and compute_neighbors
- otherwise kept (
    add_edge_distance_to_graph,
    dict_set_recursively,
    parse_value,
    create_dict_from_args,
    find_relative_file_in_paths,
    load_config,
    build_config,
    sum_partitions,
    get_counts,
    get_max_neighbors_mask,
    merge_dicts,
    setup_env_vars,
    tensor_stats,
    get_weight_table,
)
- deleted pbc-related code
"""

from __future__ import annotations

import ast
import copy
import logging
import os
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import yaml


DEFAULT_ENV_VARS = {
    # Expandable segments is a new cuda feature that helps with memory fragmentation during frequent allocations (ie: in the case of variable batch sizes).
    # see https://pytorch.org/docs/stable/notes/cuda.html.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


# copied from https://stackoverflow.com/questions/33490870/parsing-yaml-in-python-detect-duplicated-keys
# prevents loading YAMLS where keys have been overwritten
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            each_key = self.construct_object(key_node, deep=deep)
            if each_key in mapping:
                raise ValueError(
                    f"Duplicate Key: {each_key!r} is found in YAML File.\n"
                    f"Error File location: {key_node.end_mark}"
                )
            mapping.add(each_key)
        return super().construct_mapping(node, deep)


def print_cuda_usage() -> None:
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print(
        "Max Memory Allocated:",
        torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
    print("Memory Cached:", torch.cuda.memory_reserved() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_reserved() / (1024 * 1024))


def conditional_grad(dec):
    "Decorator to enable/disable grad depending on whether force/energy predictions are being made"

    # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


def add_edge_distance_to_graph(
    batch,
    device="cpu",
    dmin: float = 0.0,
    dmax: float = 6.0,
    num_gaussians: int = 50,
):
    # Make sure x has positions.
    if not all(batch.pos[0][:] == batch.x[0][-3:]):
        batch.x = torch.cat([batch.x, batch.pos.float()], dim=1)
    # First set computations to be tracked for positions.
    batch.x = batch.x.requires_grad_(True)
    # Then compute Euclidean distance between edge endpoints.
    pdist = torch.nn.PairwiseDistance(p=2.0)
    distances = pdist(
        batch.x[batch.edge_index[0]][:, -3:],
        batch.x[batch.edge_index[1]][:, -3:],
    )
    # Expand it using a gaussian basis filter.
    gdf_filter = torch.linspace(dmin, dmax, num_gaussians)
    var = gdf_filter[1] - gdf_filter[0]
    gdf_filter, var = gdf_filter.to(device), var.to(device)
    gdf_distances = torch.exp(-((distances.view(-1, 1) - gdf_filter) ** 2) / var**2)
    # Reassign edge attributes.
    batch.edge_weight = distances
    batch.edge_attr = gdf_distances.float()
    return batch


def dict_set_recursively(dictionary, key_sequence, val) -> None:
    top_key = key_sequence.pop(0)
    if len(key_sequence) == 0:
        dictionary[top_key] = val
    else:
        if top_key not in dictionary:
            dictionary[top_key] = {}
        dict_set_recursively(dictionary[top_key], key_sequence, val)


def parse_value(value):
    """
    Parse string as Python literal if possible and fallback to string.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Use as string if nothing else worked
        return value


def create_dict_from_args(args: list, sep: str = "."):
    """
    Create a (nested) dictionary from console arguments.
    Keys in different dictionary levels are separated by sep.
    """
    return_dict = {}
    for arg in args:
        keys_concat, val = arg.removeprefix("--").split("=")
        val = parse_value(val)
        key_sequence = keys_concat.split(sep)
        dict_set_recursively(return_dict, key_sequence, val)
    return return_dict


# given a filename and set of paths , return the full file path
def find_relative_file_in_paths(filename, include_paths):
    if os.path.exists(filename):
        return filename
    for path in include_paths:
        include_filename = os.path.join(path, filename)
        if os.path.exists(include_filename):
            return include_filename
    raise ValueError(f"Cannot find include YML {filename}")


def load_config(
    path: str,
    files_previously_included: list | None = None,
    include_paths: list | None = None,
):
    """
    Load a given config with any defined imports

    When imports are present this is a recursive function called on imports.
    To prevent any cyclic imports we keep track of already imported yml files
    using files_previously_included
    """
    if include_paths is None:
        include_paths = []
    if files_previously_included is None:
        files_previously_included = []
    path = Path(path)
    if path in files_previously_included:
        raise ValueError(
            f"Cyclic config include detected. {path} included in sequence {files_previously_included}."
        )
    files_previously_included = [*files_previously_included, path]

    with open(path) as fp:
        current_config = yaml.load(fp, Loader=UniqueKeyLoader)

    # Load config from included files.
    includes_listed_in_config = (
        current_config.pop("includes") if "includes" in current_config else []
    )
    if not isinstance(includes_listed_in_config, list):
        raise AttributeError(
            f"Includes must be a list, '{type(includes_listed_in_config)}' provided"
        )

    config_from_includes = {}
    duplicates_warning = []
    duplicates_error = []
    for include in includes_listed_in_config:
        include_filename = find_relative_file_in_paths(
            include, [os.path.dirname(path), *include_paths]
        )
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include_filename, files_previously_included
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config_from_includes, merge_dup_error = merge_dicts(
            config_from_includes, include_config
        )
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config_from_includes, merge_dup_warning = merge_dicts(
        config_from_includes, current_config
    )
    duplicates_warning += merge_dup_warning
    return config_from_includes, duplicates_warning, duplicates_error


def build_config(args, args_override, include_paths=None):
    config, duplicates_warning, duplicates_error = load_config(
        args.config_yml, include_paths=include_paths
    )
    if len(duplicates_warning) > 0:
        logging.warning(
            f"Overwritten config parameters from included configs "
            f"(non-included parameters take precedence): {duplicates_warning}"
        )
    if len(duplicates_error) > 0:
        raise ValueError(
            f"Conflicting (duplicate) parameters in simultaneously "
            f"included configs: {duplicates_error}"
        )

    # Some other flags.
    config["mode"] = args.mode
    config["identifier"] = args.identifier
    config["timestamp_id"] = args.timestamp_id
    config["seed"] = args.seed
    config["is_debug"] = args.debug
    config["run_dir"] = args.run_dir
    config["print_every"] = args.print_every
    config["amp"] = args.amp
    config["checkpoint"] = args.checkpoint
    config["cpu"] = args.cpu
    # Submit
    config["submit"] = args.submit
    config["summit"] = args.summit
    # Distributed
    config["world_size"] = args.num_nodes * args.num_gpus
    config["distributed_backend"] = "gloo" if args.cpu else "nccl"
    config["gp_gpus"] = args.gp_gpus

    # Check for overridden parameters.
    if args_override != []:
        overrides = create_dict_from_args(args_override)
        config, _ = merge_dicts(config, overrides)

    return config


def sum_partitions(x: torch.Tensor, partition_idxs: torch.Tensor) -> torch.Tensor:
    sums = torch.zeros(partition_idxs.shape[0] - 1, device=x.device, dtype=x.dtype)
    for idx in range(partition_idxs.shape[0] - 1):
        sums[idx] = x[partition_idxs[idx] : partition_idxs[idx + 1]].sum()
    return sums


def get_counts(x: torch.Tensor, length: int):
    dtype = x.dtype
    device = x.device
    return torch.zeros(length, device=device, dtype=dtype).scatter_reduce(
        dim=0,
        index=x,
        src=torch.ones(x.shape[0], device=device, dtype=dtype),
        reduce="sum",
    )


def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    num_neighbors = get_counts(index, num_atoms)

    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = sum_partitions(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold] + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(max=num_included_per_atom)

        num_neighbors_image = sum_partitions(num_neighbors_thresholded, image_indptr)

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)
    return mask_num_neighbors, num_neighbors_image


def get_pruned_edge_idx(
    edge_index, num_atoms: int, max_neigh: float = 1e9
) -> torch.Tensor:
    assert num_atoms is not None  # TODO: Shouldn't be necessary

    # removes neighbors > max_neigh
    # assumes neighbors are sorted in increasing distance
    _nonmax_idx_list = []
    for i in range(num_atoms):
        idx_i = torch.arange(len(edge_index[1]))[(edge_index[1] == i)][:max_neigh]
        _nonmax_idx_list.append(idx_i)
    return torch.cat(_nonmax_idx_list)


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def compute_neighbors(data, edge_index):
    # Get number of neighbors
    num_neighbors = get_counts(edge_index[1], data.natoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    return sum_partitions(num_neighbors, image_indptr)


def setup_env_vars() -> None:
    for k, v in DEFAULT_ENV_VARS.items():
        os.environ[k] = v
        logging.info(f"Setting env {k}={v}")


@torch.no_grad()
def tensor_stats(name: str, x: torch.Tensor) -> dict:
    return {
        f"{name}.max": x.max().item(),
        f"{name}.min": x.min().item(),
        f"{name}.std": x.std().item(),
        f"{name}.mean": x.mean().item(),
        f"{name}.norm": torch.norm(x, p=2).item(),
        f"{name}.nonzero_fraction": torch.nonzero(x).shape[0] / float(x.numel()),
    }


def get_weight_table(model: torch.nn.Module) -> tuple[list, list]:
    stat_names = list(tensor_stats("weight", torch.Tensor([1])).keys())
    columns = ["ParamName", "shape"] + stat_names + ["grad." + n for n in stat_names]
    data = []
    for param_name, params in model.named_parameters():
        row_weight = list(tensor_stats(f"weights/{param_name}", params).values())
        if params.grad is not None:
            row_grad = list(tensor_stats(f"grad/{param_name}", params.grad).values())
        else:
            row_grad = [None] * len(row_weight)
        data.append([param_name] + [params.shape] + row_weight + row_grad)  # noqa
    return columns, data
