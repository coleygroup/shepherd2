"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copied from fairchem.core.models.base.py
- only kept GraphData and GraphModelMixin classes
- deleted pbc-related code; removed cell_offsets and offset_distances
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch_cluster import radius_graph

from shepherd.model.equiformer_v3.models.equiformer_v3.fairchem_utils import compute_neighbors


@dataclass
class GraphData:
    """Class to keep graph attributes nicely packaged."""

    edge_index: torch.Tensor
    edge_distance: torch.Tensor
    edge_distance_vec: torch.Tensor
    neighbors: torch.Tensor
    batch_full: torch.Tensor  # used for GP functionality
    atomic_numbers_full: torch.Tensor  # used for GP functionality
    node_offset: int = 0  # used for GP functionality


class GraphModelMixin:
    """Mixin Model class implementing some general convenience properties and methods.

    Edited to remove pbc-related code. Removed cell_offsets and offset_distances.
    """

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is None:
            enforce_max_neighbors_strictly = getattr(
                self, "enforce_max_neighbors_strictly", True
            )

        if not otf_graph:
            try:
                edge_index = data.edge_index

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        else:
            edge_index = radius_graph(
                data.pos,
                r=cutoff,
                batch=data.batch,
                max_num_neighbors=max_neighbors,
            )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        neighbors = compute_neighbors(data, edge_index)

        return GraphData(
            edge_index=edge_index,
            edge_distance=edge_dist,
            edge_distance_vec=distance_vec,
            neighbors=neighbors,
            node_offset=0,
            batch_full=data.batch,
            atomic_numbers_full=data.atomic_numbers.long(),
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns a list of parameters with no weight decay."""
        no_wd_list = []
        for name, _ in self.named_parameters():
            if "embedding" in name or "frequencies" in name or "bias" in name:
                no_wd_list.append(name)
        return no_wd_list
