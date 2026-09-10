"""
Copied from equiformer_v3/models/equiformer_v3/equiformer_v3.py
and modified with the following changes:
- removed pbc-related code
- removed cell_offsets and offset_distances
- unlike ShEPhERDv1, we edit EdgeDegreeEmbedding directly in equiformer_v3.input_block.py
- adjusted TransformerBlock directly in equiformer_v3/models/equiformer_v3/transformer_block.py
- changed avg_num_nodes and avg_degree to dataset parameters (directly from ShEPhERD v1)
- added final_block_channels, input_sphere_channels, edge_attr_input_channels args
- removed regress_forces and regress_stress args
- deleted forces and stress blocks
- deleted _forward_gradient method
"""

import math
import torch

# from shepherd.model.equiformer_v3.models.equiformer_v3.fairchem_utils import conditional_grad
from shepherd.model.equiformer_v3.models.equiformer_v3.fairchem_core_models_base import GraphModelMixin
# from shepherd.model.equiformer_v3.models.equiformer_v3.utils import reduce_edge

from shepherd.model.equiformer_v3.models.equiformer_v3.edge_rot_mat import init_edge_rot_mat
from shepherd.model.equiformer_v3.models.equiformer_v3.envelope import PolynomialEnvelope
from shepherd.model.equiformer_v3.models.equiformer_v3.so3 import (
    SO3Rotation,
    SO3Linear
)
from shepherd.model.equiformer_v3.models.equiformer_v3.radial_function import (
    GaussianSmearing,
    RadialFunction
)
from shepherd.model.equiformer_v3.models.equiformer_v3.input_block import EdgeDegreeEmbedding
from shepherd.model.equiformer_v3.models.equiformer_v3.layer_norm import (
    EquivariantLayerNorm,
    EquivariantSeparableLayerNorm,
    EquivariantMergeLayerNorm,
    RMSNorm,
    get_normalization_layer
)
from shepherd.model.equiformer_v3.models.equiformer_v3.transformer_block import (
    EquivariantGraphAttention, # noqa: F401
    FeedForwardNetwork,
    TransBlockV3,
)
# We redefine EdgeDegreeEmbedding in this module instead of importing it from the original module to allow for easy modifications
# from shepherd.model.equiformer_v3.models.equiformer_v3.input_block import EdgeDegreeEmbedding
# from shepherd.model.equiformer_v3.models.equiformer_v3.output_block import (
#     ScalarFeedForwardNetwork, # noqa: F401
# )


# Statistics of IS2RE 100K
# _AVG_NUM_NODES = 77.81317
# _AVG_DEGREE = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100

# _NORM_SCALE_NODES = math.sqrt(_AVG_NUM_NODES)   # 8.82117735906041
# _NORM_SCALE_DEGREE = math.sqrt(_AVG_DEGREE)     # 4.836862503353054

# set to dataset parameters
# Directly from ShEPhERD v1
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666

_NORM_SCALE_NODES = math.sqrt(_AVG_NUM_NODES)   # 4.246255699115012
_NORM_SCALE_DEGREE = math.sqrt(_AVG_DEGREE)     # 3.946925229301434


class EquiformerV3(torch.nn.Module, GraphModelMixin):
    """
    Edits for ShEPhERDv2
    - removed pbc-related code (cell_offsets and offset_distances)
    - removed max_num_elements arg (not used since we use linear instead of embedding)
    - set otf_graph to False since we build our own graphs
    - set num_radial_basis to 100
    - deleted forces and stress blocks
    - deleted _forward_gradient method

    Args for ShEPhERDv2: (added)
        final_block_channels (int): Number of spherical channels in final x embedding; if 0, there is no final block.
        input_sphere_channels (int): Number of spherical channels in input x embedding, used for edge embedding.
        edge_attr_input_channels (int): Number of channels in edge attributes, used for edge embedding.
            optionally used for bond type features

    Args:
        otf_graph (bool):       Compute graph On The Fly (OTF) (ShEPhERDv2: set to False since we build our own graphs)

        [deleted] regress_forces (bool):  Compute forces (ShEPhERDv2: set to False)
        [deleted] regress_stress (bool):  Compute stress
        direct_prediction (bool):   Whether to use direct methods to predict forces and stress
            (ShEPhERDv2: kept to True to stay on the same path, but don't regress forces or stress)

        [deleted] max_neighbors (int):    Maximum number of neighbors per atom
        cutoff (float):     Maximum distance between nieghboring atoms in Angstroms
            [renamed] from max_radius to match ShEPhERDv1
        num_radial_basis (int): Number of radial basis functions
        [Deleted] max_num_elements (int): Maximum atomic number

        num_layers (int):           Number of layers in the GNN
        num_channels (int):         Number of channels in node embeddings
        attn_hidden_channels (int): Number of hidden channels in equivariant graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_channels (int):  Number of channels for alpha vector in each attention head
        attn_value_channels (int):  Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels in feedforward network
        norm_type (str):            Type of normalization layer
                                    (['sep_layer_norm', 'merge_layer_norm',
                                    'merge_layer_norm_attn_rms_norm', 'merge_rms_norm'])

        lmax (int):                 Maximum degrees (l)
        mmax (int):                 Maximum order (m)
        attn_grid_resolution_list (list:int):
                                    Grid resolution list in class `SO3Grid` in attention
        ffn_grid_resolution_list (list:int):
                                    Grid resolution list in class `SO3Grid` in feedforward network

        edge_channels (int):                Number of channels for edge-wise invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        use_envelope (bool):        Whether to apply an envelope function to attention

        attn_activation (str):      Type of activation function in equivariant graph attention
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_add_merge (bool):       Default: False
                                    If True, use addition to merge the source/target node features instead of concat,
                                    which can save 2x compute when rotating with Wigner-D matrices.
        use_rad_l_parametrization (bool):
                                    Default: True
                                    If True, all the m components within the same type-L vector will share the same
                                    weight from the radial function.
        softcap (float):            Default: None
                                    If not None, use soft capping to limit the range of attention logits to
                                    [- `softcap`, + `softcap`].
        attn_eps (float):           Default: 1e-16
                                    Epsilon value used in the softmax operation of attention
        ffn_activation (str):       Type of activation function for feedforward network
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.

        [deleted] use_gate_force_head (bool): If `True`, use `GateActivation` in the equivariant attention of the force prediction head.

        alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
        attn_mask_rate (float):     Mask rate for neighbors considered in attention
        attn_weights_drop (float):  Dropout rate for attention weights
        value_drop (float):         Dropout rate for the hidden features in non-linear value vectors
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks
        ffn_drop (float):           Dropout rate for the hidden features in FFN
        use_head_reg (bool):        Whether to apply regularization to output head (dummy argument for backend compatibility)

        gradient_checkpointing_block_list (list):
                                    A list indicating which block we apply gradient/activation checkpointing to save memory.

        avg_num_nodes (float):   Normalization factor for sum aggregation over nodes
        avg_degree (float):      Normalization factor for sum aggregation over edges

        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
    """
    def __init__(
        self,

        # Args for ShEPhERDv2
        final_block_channels=0,
        input_sphere_channels=128,
        edge_attr_input_channels=0,

        # original args
        otf_graph=False, # set to False since we build our own graphs

        # regress_forces=False, # set to False since we don't regress forces
        # regress_stress=False,
        direct_prediction=True,

        # max_neighbors=20,
        # max_radius=12.0,
        cutoff=5.0,  # ShEPhERDv1: used 5.0
        num_radial_basis=100, # ShEPhERDv1: used 100, down from 600 of EqV2/3

        num_layers=12,
        num_channels=128, # used to be num_sphere_channels
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type='merge_layer_norm',

        lmax=6,
        mmax=2,
        attn_grid_resolution_list=[20, 8],
        ffn_grid_resolution_list=[20, 20],

        edge_channels=128,
        use_atom_edge_embedding=True,
        use_envelope=True,

        attn_activation='sep-merge_gates2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        attn_eps=1e-16,
        ffn_activation='sep-merge_gates2_swiglu',
        use_grid_mlp=True,

        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.1,
        value_drop=0.0,
        drop_path_rate=0.05,
        proj_drop=0.0,
        ffn_drop=0.0,
        use_head_reg=False,

        gradient_checkpointing_block_list=None,

        avg_num_nodes=_AVG_NUM_NODES,
        avg_degree=_AVG_DEGREE,

        enforce_max_neighbors_strictly=True,
    ):
        super().__init__()

        # Args for ShEPhERDv2
        self.final_block_channels = final_block_channels
        self.input_sphere_channels = input_sphere_channels
        self.edge_attr_input_channels = edge_attr_input_channels

        self.otf_graph = otf_graph

        # self.regress_forces = regress_forces
        # self.regress_stress = regress_stress
        self.direct_prediction = direct_prediction

        # self.max_neighbors = max_neighbors
        self.cutoff = cutoff
        self.num_radial_basis = num_radial_basis

        self.num_layers = num_layers
        self.num_channels = num_channels

        # from ShEPhERDv1:
        # for now, we must enforce input_sphere_channels == num_channels.
        # This could be made more flexible by including a feed-forward SO3 network as a node-embedding step
        assert self.input_sphere_channels == self.num_channels

        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax = lmax
        self.mmax = mmax
        self.attn_grid_resolution_list = attn_grid_resolution_list
        self.ffn_grid_resolution_list = ffn_grid_resolution_list

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_envelope = use_envelope

        self.attn_activation = attn_activation
        self.use_attn_renorm = use_attn_renorm
        self.use_add_merge = use_add_merge
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.softcap = softcap
        self.attn_eps = attn_eps
        self.ffn_activation = ffn_activation
        self.use_grid_mlp = use_grid_mlp

        self.alpha_drop = alpha_drop
        self.attn_mask_rate = attn_mask_rate
        self.attn_weights_drop = attn_weights_drop
        self.value_drop = value_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.ffn_drop = ffn_drop
        self.use_head_reg = use_head_reg

        self.gradient_checkpointing_block_list = gradient_checkpointing_block_list
        if self.gradient_checkpointing_block_list is not None:
            assert len(self.gradient_checkpointing_block_list) == self.num_layers
        else:
            self.gradient_checkpointing_block_list = [0] * self.num_layers

        self.avg_num_nodes = avg_num_nodes
        self.avg_degree = avg_degree

        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly

        # Atom-type embedding
        # ShEPhERDv2: replaced with linear layers
        # self.sphere_embedding = torch.nn.Embedding(self.max_num_elements, self.num_channels)

        # Radial basis function
        self.distance_expansion = GaussianSmearing(
            0.0,
            self.cutoff,
            self.num_radial_basis,
            2.0,
        )
        edge_input_channels = int(self.distance_expansion.num_output)

        self.edge_attr_embedding = None
        if self.edge_attr_input_channels > 0:
            self.edge_attr_embedding = torch.nn.Linear(self.edge_attr_input_channels, edge_input_channels)

        # The sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [edge_input_channels] + [self.edge_channels] * 2

        # Envelope function
        self.envelope_func = PolynomialEnvelope(
            cutoff=self.cutoff,
            exponent=5
        ) if self.use_envelope else None

        # Computing Wigner-D matrices
        self.so3_rotation = SO3Rotation(self.lmax, self.mmax, use_rotation_mask=(not self.direct_prediction))

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            input_sphere_channels=self.input_sphere_channels,
            num_channels=self.num_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            so3_rotation=self.so3_rotation,
            # max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            use_atom_edge_embedding=self.use_atom_edge_embedding,
            rescale_factor=self.avg_degree
        )

        # Transformer block
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if self.gradient_checkpointing_block_list[i] == 1:
                attn_activation = self.attn_activation.replace('_mem', '')
                ffn_activation  = self.ffn_activation.replace('_mem', '')
            else:
                attn_activation = self.attn_activation
                ffn_activation  = self.ffn_activation
            block_config_dict = dict(
                input_sphere_channels=self.num_channels,
                num_in_channels=self.num_channels,
                attn_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                ffn_hidden_channels=self.ffn_hidden_channels,
                num_out_channels=self.num_channels,
                lmax=self.lmax,
                mmax=self.mmax,
                so3_rotation=self.so3_rotation,
                attn_grid_resolution_list=self.attn_grid_resolution_list,
                ffn_grid_resolution_list=self.ffn_grid_resolution_list,
                # max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.use_atom_edge_embedding,
                attn_activation=attn_activation,
                use_attn_renorm=self.use_attn_renorm,
                use_add_merge=self.use_add_merge,
                use_rad_l_parametrization=self.use_rad_l_parametrization,
                softcap=self.softcap,
                attn_eps=self.attn_eps,
                ffn_activation=ffn_activation,
                use_grid_mlp=self.use_grid_mlp,
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                attn_mask_rate=self.attn_mask_rate,
                attn_weights_drop=attn_weights_drop,
                value_drop=self.value_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop,
                ffn_drop=self.ffn_drop
            )
            block_class = TransBlockV3
            self.blocks.append(block_class(**block_config_dict))

        # self.energy_block = ScalarFeedForwardNetwork(
        #     num_in_channels=self.num_channels,
        #     num_hidden_channels=self.ffn_hidden_channels,
        #     num_out_channels=1,
        #     dropout=0.0
        # )

        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=self.lmax,
            num_channels=self.num_channels
        )

        # ShEPhERDv2: Add final block for output
        if self.final_block_channels > 0:
            self.final_block = FeedForwardNetwork(
                num_in_channels=self.num_channels,
                num_hidden_channels=self.ffn_hidden_channels,
                num_out_channels=self.final_block_channels,
                lmax=self.lmax,
                mmax=self.mmax,
                grid_resolution_list=self.ffn_grid_resolution_list,
                activation=self.ffn_activation,
                use_grid_mlp=self.use_grid_mlp,
                dropout=self.ffn_drop,
            )
        else:
            self.final_block = None

        # initialize weights
        self.apply(self._init_weights)


    def _forward_edge(
        self,
        edge_distance,
        edge_distance_vec
    ):
        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)

        # Compute Wigner-D matrices
        self.so3_rotation.set_wigner(edge_rot_mat)

        # Envelope function
        edge_envelope_weight = self.envelope_func(edge_distance) if self.envelope_func is not None else None

        # Radial basis function
        edge_distance_rbf = self.distance_expansion(edge_distance)

        return edge_distance_rbf, edge_envelope_weight


    def _forward_embedding(
        self,
        x_input,
        edge_distance_rbf,
        edge_index,
        edge_envelope_weight,
        edge_attr = None,
    ):
        """
        Adjusted for ShEPhERDv2 to use x_input instead of atomic_numbers

        x_input is a tensor with shape (num_atoms, (lmax+1)^2, input_sphere_channels)
        - identical layout to V2's SO3_Embedding.embedding tensor
        - x_input[:, 0, :] holds the linearly-projected continuous atomic features
        - x_input[:, 1:4, :] may optionally hold l=1 direction features (x4 case)
        - added edge_attr argument to allow for bond type features
        - change edge_distance to edge_distance_rbf since we already expand it in _forward_edge
        """

        # num_atoms = len(atomic_numbers)

        # # Initialize node embedding
        # x = torch.zeros(
        #     (
        #         num_atoms,
        #         ((self.lmax + 1) ** 2),
        #         self.num_channels
        #     ),
        #     device=self.device,
        #     dtype=self.dtype
        # )

        # # Atom-type embedding
        # atom_embedding = self.sphere_embedding(atomic_numbers)
        # x[:, 0, :] = atom_embedding

        # From ShEPhERDv1:
        # interpret x_input l=0 channel as a minimal embedding of one-hot atomic numbers
        # do we want to include an extra FeedForwardNetwork mapping x (input_sphere_channels) to (sphere_channels) ?

        # Edge encoding (distance and atom edge)
        if edge_attr is not None:
            assert self.edge_attr_embedding is not None
            edge_distance_rbf = edge_distance_rbf + self.edge_attr_embedding(edge_attr)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            x_input,
            edge_distance_rbf,
            edge_index,
            edge_envelope_weight
        )
        x = x_input + edge_degree # MUST stay out of place -> do NOT do x+= edge_degree since we don't .copy

        return x


    def _forward_blocks(
        self,
        x: torch.Tensor,
        edge_distance,
        edge_index,
        edge_envelope_weight,
        batch,
    ):
        # Transformer blocks
        for i in range(self.num_layers):
            if self.gradient_checkpointing_block_list[i] == 0:
                x = self.blocks[i](
                    x,
                    edge_distance,
                    edge_index,
                    edge_envelope_weight,
                    batch,     # for GraphDropPath
                )
            elif self.gradient_checkpointing_block_list[i] == 1:
                x = torch.utils.checkpoint.checkpoint(
                    self.blocks[i],
                    x,
                    edge_distance,
                    edge_index,
                    edge_envelope_weight,
                    batch,     # for GraphDropPath
                    use_reentrant=False
                )
            else:
                raise ValueError

        # Final layer norm
        x = self.norm(x)

        if self.final_block is not None:
            x_final = self.final_block(x)
        else:
            x_final = x

        # x_scalar = x.narrow(1, 0, 1)
        # x_scalar = x_scalar.view(x_scalar.shape[0], self.num_channels)
        return x, x_final


    def _forward_direct(self, x_input, pos, edge_index, edge_distance, edge_distance_vec, batch, edge_attr = None):
        """
        Adjusted for ShEPhERDv2
        - removed forces and stress blocks
        - changed arg from data to x_input, pos, edge_index, edge_distance, edge_distance_vec, batch, edge_attr
        - removed generate_graph method since we build our own graphs
        """
        edge_distance_rbf, edge_envelope_weight = self._forward_edge(
            edge_distance=edge_distance,
            edge_distance_vec=edge_distance_vec
        )
        x = self._forward_embedding(
            x_input=x_input,
            edge_distance_rbf=edge_distance_rbf,
            edge_index=edge_index,
            edge_envelope_weight=edge_envelope_weight,
            edge_attr=edge_attr,
        )
        x_post_norm, x_final = self._forward_blocks(
            x,
            edge_distance_rbf,
            edge_index,
            edge_envelope_weight,
            batch,
        )

        # outputs = {}

        # # Energy prediction
        # node_energy = self.energy_block(x_scalar)
        # energy = torch.zeros(self.batch_size, device=node_energy.device, dtype=node_energy.dtype)
        # energy.index_add_(0, batch, node_energy.view(-1))
        # energy = energy / self.avg_num_nodes
        # outputs['energy'] = energy

        # deleted forces and stress blocks

        return x_post_norm, x_final # ShEPhERD v2: returns (x, x_final) but x_final is always ignored since final_blocks = 0


    def forward(self, x_input, pos, edge_index, edge_distance, edge_distance_vec, batch, edge_attr = None):
        """
        Adjusted for ShEPhERDv2: removed _forward_gradient method
        """
        output = self._forward_direct(
            x_input=x_input,
            pos=pos,
            edge_index=edge_index,
            edge_distance=edge_distance,
            edge_distance_vec=edge_distance_vec,
            batch=batch,
            edge_attr=edge_attr,
        )
        return output

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec, use_rotation_mask=(not self.direct_prediction))


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3Linear)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)


    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Embedding)
                or isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3Linear)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, RMSNorm)
                or isinstance(module, EquivariantLayerNorm)
                or isinstance(module, EquivariantSeparableLayerNorm)
                or isinstance(module, EquivariantMergeLayerNorm)
            ):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3Linear)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


    # @torch._dynamo.disable
    # def generate_graph(
    #     self,
    #     data,
    #     cutoff=None,
    #     max_neighbors=None,
    #     otf_graph=None,
    #     enforce_max_neighbors_strictly=None,
    # ):
    #     graph_data = super().generate_graph(
    #         data,
    #         cutoff,
    #         max_neighbors,
    #         otf_graph,
    #         enforce_max_neighbors_strictly,
    #     )

    #     edge_index   = graph_data.edge_index
    #     edge_dist    = graph_data.edge_distance
    #     distance_vec = graph_data.edge_distance_vec
    #     neighbors    = graph_data.neighbors

    #     return (
    #         edge_index,
    #         edge_dist,
    #         distance_vec,
    #         neighbors,
    #     )
