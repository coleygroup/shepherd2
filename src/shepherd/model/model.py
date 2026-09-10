from copy import deepcopy

import torch
import torch.nn as nn
import torch_scatter
import torch_geometric
from torch_cluster import radius_graph

from shepherd.model.egnn.egnn import EGNN, MultiLayerPerceptron, GaussianSmearing
from shepherd.model.equiformer_v3_encoder import EquiformerV3
from shepherd.model.equiformer_v3.models.equiformer_v3.transformer_block import FeedForwardNetwork
from shepherd.model.equiformer_v3.models.equiformer_v3.layer_norm import get_normalization_layer

from shepherd.model.utils.add_virtual_edges_to_edge_index import add_virtual_edges_to_edge_index
from shepherd.model.utils.add_virtual_edges_to_edge_index import bond_weighting
from shepherd.model.utils.positional_encoding import fourier_embedding
from shepherd.diffusion.edm import EDMPreconditioner

import e3nn
from shepherd.model.equiformer_operations import FeedForwardNetwork_equiformer, convert_e3nn_to_equiformerv3, convert_equiformerv3_to_e3nn


def remap_values(remapping_tuple, input_tensor):
    """
    # credit to: https://discuss.pytorch.org/t/cv2-remap-in-pytorch/99354/8

    Maps integer values in input_tensor to new integer values specified by the map remapping_tuple[0]:remapping_tuple[1]

    Args:
        remapping_tuple (Tuple(torch.LongTensor, torch.LongTensor))
        input_tensor (torch.LongTensor)
    Returns:
        (torch.LongTensor) with new values
    """
    index = torch.bucketize(input_tensor.ravel(), remapping_tuple[0])
    return remapping_tuple[1][index].reshape(input_tensor.shape)


# useful for debugging
def display_dict(d, indent=''):
    for key in d:
        print(indent + key)
        value = d[key]
        if isinstance(value, dict):
            display_dict(value, indent = indent + '    ')


class Model(torch.nn.Module):

    _ENCODER_CONFIG_ONLY_KEYS = {
        'use', 'fully_connected', 'max_neighbors', 'input_bond_channels', 'prune_joint_edges',
        'l0_res_conditioning',
    }

    @classmethod
    def _equiformer_kwargs(cls, config):
        return {
            key: value for key, value in config.items()
            if key not in cls._ENCODER_CONFIG_ONLY_KEYS
        }

    @staticmethod
    def _freeze_egnn_unused_node_update(egnn):
        """EGNN denoisers use only coordinate updates; keep ignored feature heads non-trainable."""
        egnn.node_mlp.requires_grad_(False)
        egnn.node_output_embedding.requires_grad_(False)

    def _init_embedding(self, num_nodes, lmax, num_channels):
        return torch.empty(
            num_nodes,
            (lmax + 1) ** 2,
            num_channels,
            device=self.device,
            dtype=self.dtype,
        )

    def _set_encoder_outputs(self, output_dict, x_str, node_embedding, batch, lmax_list):
        global_embedding = torch_scatter.scatter_sum(
            node_embedding,
            batch,
            dim=0,
        )

        output_dict[x_str]['decoder']['encoder']['node_embedding'] = node_embedding
        output_dict[x_str]['decoder']['encoder']['global_embedding'] = global_embedding
        return output_dict

    def __init__(self, params):
        super(Model, self).__init__()

        self.params = params
        self.device = 'cpu'

        self._ffn_config = params['feedforward_network']
        self._ffn_options = {
            key: value for key, value in self._ffn_config.items()
            if key != 'num_hidden_channels'
        }

        self.x1_bond_diffusion = params['x1_bond_diffusion']
        self.scale_point_cloud = params['dataset'].get('scale_point_cloud', 1.0)

        # EDM-style sigma projection in prep for fourier embedding: linear map from sigma (scalar) to node_channels
        if 'x1' in self.params['explicit_diffusion_variables']:
            self.x1_edm_sigma_projection = nn.Linear(1, self.params['x1']['decoder']['time_embedding_size'])
        if 'x2' in self.params['explicit_diffusion_variables']:
            self.x2_edm_sigma_projection = nn.Linear(1, self.params['x2']['decoder']['time_embedding_size'])
        if 'x3' in self.params['explicit_diffusion_variables']:
            self.x3_edm_sigma_projection = nn.Linear(1, self.params['x3']['decoder']['time_embedding_size'])
        if 'x4' in self.params['explicit_diffusion_variables']:
            self.x4_edm_sigma_projection = nn.Linear(1, self.params['x4']['decoder']['time_embedding_size'])

        # EDM noise schedules
        self.edm_noise_schedule_pos = EDMPreconditioner(
            sigma_data=self.params['edm'].get('sigma_data_pos', 3.0),
            clip_loss_weighting_max=self.params['edm'].get('clip_loss_weighting_max', None),
        )
        self.edm_noise_schedule_atom_one_hot = EDMPreconditioner(
            sigma_data=self.params['edm'].get('sigma_data_atom_one_hot', 0.3),
            clip_loss_weighting_max=self.params['edm'].get('clip_loss_weighting_max', None),
        )
        self.edm_noise_schedule_bond = EDMPreconditioner(
            sigma_data=self.params['edm'].get('sigma_data_bond', 1.0),
            clip_loss_weighting_max=self.params['edm'].get('clip_loss_weighting_max', None),
        )
        if 'x3' in self.params['explicit_diffusion_variables']:
            self.edm_noise_schedule_esp = EDMPreconditioner(
                sigma_data=self.params['edm'].get('sigma_data_esp', 0.46),
                clip_loss_weighting_max=self.params['edm'].get('clip_loss_weighting_max', None),
            )
        if 'x4' in self.params['explicit_diffusion_variables']:
            self.edm_noise_schedule_pharm_one_hot = EDMPreconditioner(
                sigma_data=self.params['edm'].get('sigma_data_pharm_one_hot', 0.27),
                clip_loss_weighting_max=self.params['edm'].get('clip_loss_weighting_max', None),
            )
            self.edm_noise_schedule_direction = EDMPreconditioner(
                sigma_data=self.params['edm'].get('sigma_data_direction', 0.45),
                clip_loss_weighting_max=self.params['edm'].get('clip_loss_weighting_max', None),
            )

        self.joint_mixer_lmax = params.get('joint_mixer_lmax', 1)
        self._joint_mixer_K = (self.joint_mixer_lmax + 1) ** 2

        if params.get('scaffold_conditioning', False):
            # different scaffold task time embeddings for each explicit diffusion variable per-node
            self.scaffold_task_time_embedding = torch.nn.ParameterDict()
            for x_str in params['explicit_diffusion_variables']:
                if params['dataset'][x_str].get('fixed_substructure', {}):
                    self.scaffold_task_time_embedding[x_str] = torch.nn.Parameter(
                        torch.randn(1, 1, params[x_str]['decoder']['node_channels']), requires_grad = True,
                    )
            self.scaffold_task_global_embedding = torch.nn.Parameter(
                    torch.randn(1, 1, params['x1']['decoder']['node_channels']), requires_grad = True,
                )

        # Joint Module

        self.explicit_diffusion_variables = params['explicit_diffusion_variables']
        self.exclude_variables_from_decoder_heterogeneous_graph = params['exclude_variables_from_decoder_heterogeneous_graph']
        decoder_heterogeneous_variables = deepcopy([x_ for x_ in self.explicit_diffusion_variables if x_ not in self.exclude_variables_from_decoder_heterogeneous_graph])


        # heterogeneous graph encoding (in decoder, before joint global code processing)
        self.decoder_joint_heterogeneous_graph_encoder = None
        if 'decoder_heterogeneous_graph_encoder' in params:
            if params['decoder_heterogeneous_graph_encoder']['use']:
                assert len(decoder_heterogeneous_variables) > 1

                for x_ in decoder_heterogeneous_variables:
                    assert params[x_]['decoder']['encoder']['num_channels'] == params['decoder_heterogeneous_graph_encoder']['num_channels']

                hetero_params = params['decoder_heterogeneous_graph_encoder']
                hetero_kwargs = {key: value for key, value in hetero_params.items() if key not in {
                    'use', 'fully_connected', 'max_neighbors', 'input_bond_channels', 'prune_joint_edges'
                }}
                self.decoder_joint_heterogeneous_graph_encoder = EquiformerV3(
                    **{
                        **self._equiformer_kwargs(hetero_kwargs),
                        "cutoff": hetero_params['cutoff'] * self.scale_point_cloud,
                    },
                )

        # Joint processing of global latent representations

        # these COULD share parameters, but for now, they are initialized separately for each explicit diffusion variable
        if 'x1' in self.explicit_diffusion_variables:
            self.x1_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]), # in ['x1', 'x2', ...]
                params['x1']['decoder']['node_channels'],
            )
        if 'x2' in self.explicit_diffusion_variables:
            self.x2_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]),
                params['x2']['decoder']['node_channels'],
            )
        if 'x3' in self.explicit_diffusion_variables:
            self.x3_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]),
                params['x3']['decoder']['node_channels'],
            )
        if 'x4' in self.explicit_diffusion_variables:
            self.x4_decoder_global_timestep_embedding = torch.nn.Linear(
                sum([params[x_]['decoder']['time_embedding_size'] for x_ in self.explicit_diffusion_variables]),
                params['x4']['decoder']['node_channels'],
            )

        # these could also all share parameters, but for now, they are initialized separately for each explicit diffusion variable
        _joint_ffn_shared = dict(
            num_hidden_channels=self._ffn_config['num_hidden_channels'],
            lmax=params['lmax'],
            mmax=params['mmax'],
            activation=self._ffn_config['activation'],
            grid_resolution_list=self._ffn_config['grid_resolution_list'],
            use_grid_mlp=self._ffn_config['use_grid_mlp'],
            dropout=self._ffn_config['dropout'],
        )
        if 'x1' in self.explicit_diffusion_variables:
            self.x1_decoder_global_l1_embedding = FeedForwardNetwork(
                num_in_channels=sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                num_out_channels=params['x1']['decoder']['node_channels'],
                **_joint_ffn_shared,
            )
        if 'x2' in self.explicit_diffusion_variables:
            # self.x2_decoder_global_l1_embedding = self.x1_decoder_global_l1_embedding # share parameters ?
            self.x2_decoder_global_l1_embedding = FeedForwardNetwork(
                num_in_channels=sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                num_out_channels=params['x2']['decoder']['node_channels'],
                **_joint_ffn_shared,
            )
        if 'x3' in self.explicit_diffusion_variables:
            # self.x3_decoder_global_l1_embedding = self.x1_decoder_global_l1_embedding # share parameters ?
            self.x3_decoder_global_l1_embedding = FeedForwardNetwork(
                num_in_channels=sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                num_out_channels=params['x3']['decoder']['node_channels'],
                **_joint_ffn_shared,
            )
        if 'x4' in self.explicit_diffusion_variables:
            # self.x4_decoder_global_l1_embedding = self.x1_decoder_global_l1_embedding # share parameters ?
            self.x4_decoder_global_l1_embedding = FeedForwardNetwork(
                num_in_channels=sum([params[x_]['decoder']['node_channels'] for x_ in self.explicit_diffusion_variables]),
                num_out_channels=params['x4']['decoder']['node_channels'],
                **_joint_ffn_shared,
            )

        # for mixing l=0 and l=1 channels of the joint embeddings prior to denoising
                # these could also all share parameters, but for now, they are initialized separately for each explicit diffusion variable
        if 'x1' in self.explicit_diffusion_variables:
            lmax = self.joint_mixer_lmax
            num_channels = params['x1']['decoder']['node_channels']
            self.x1_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )
        if 'x2' in self.explicit_diffusion_variables:
            lmax = self.joint_mixer_lmax
            num_channels = params['x2']['decoder']['node_channels']
            self.x2_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )
        if 'x3' in self.explicit_diffusion_variables:
            lmax = self.joint_mixer_lmax
            num_channels = params['x3']['decoder']['node_channels']
            self.x3_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )
        if 'x4' in self.explicit_diffusion_variables:
            lmax = self.joint_mixer_lmax
            num_channels = params['x4']['decoder']['node_channels']
            self.x4_decoder_equiformer_tensor_product = FeedForwardNetwork_equiformer(
                irreps_node_input = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_attr = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_node_output = e3nn.o3.Irreps(''.join([f'{num_channels}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                irreps_mlp_mid = e3nn.o3.Irreps(''.join([f'{num_channels//(2*(i+1))}x{i}e + ' for i in range(lmax +1)])[0:-3]),
                proj_drop=0.0,
            )



        # Denoising Modules

        if 'x1' in self.explicit_diffusion_variables:

            self.x1_decoder_encoder_embedding = torch.nn.Linear(
                params['x1']['decoder']['input_node_channels'],
                params['x1']['decoder']['node_channels'], # linear embedding
            )

            self.x1_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x1']['decoder']['time_embedding_size'],
                params['x1']['decoder']['node_channels'],
            )

            x1_decoder_encoder_params = params['x1']['decoder']['encoder']
            assert params['x1']['decoder']['node_channels'] == x1_decoder_encoder_params['input_sphere_channels']
            assert x1_decoder_encoder_params['input_sphere_channels'] == x1_decoder_encoder_params['num_channels']

            self.x1_decoder_encoder_bond_edge_embedding = None
            if self.x1_bond_diffusion:
                self.x1_decoder_encoder_bond_edge_embedding = torch.nn.Linear(
                    x1_decoder_encoder_params['input_bond_channels'], # (noised) one hot bond type embedding
                    x1_decoder_encoder_params['edge_attr_input_channels'], # linear embedding
                )

            x1_decoder_encoder_kwargs = dict(x1_decoder_encoder_params)
            x1_decoder_encoder_kwargs["cutoff"] = x1_decoder_encoder_params['cutoff'] * self.scale_point_cloud
             # EXTREMELY IMPORTANT: this is the only place where we set edge_attr_input_channels
            x1_decoder_encoder_kwargs["edge_attr_input_channels"] = (
                x1_decoder_encoder_params['edge_attr_input_channels'] if self.x1_bond_diffusion else 0
            )
            x1_decoder_encoder_kwargs.pop("edge_attr_channels", None)
            self.x1_decoder_encoder = EquiformerV3(
                    **self._equiformer_kwargs(x1_decoder_encoder_kwargs),
            )


            assert params['x1']['decoder']['node_channels'] == params['x1']['decoder']['encoder']['num_channels']

            self.x1_decoder_denoiser_MLP = MultiLayerPerceptron(
                input_dim = params['x1']['decoder']['node_channels'],
                hidden_dim = params['x1']['decoder']['denoiser']['MLP_hidden_dim'],
                output_dim = params['x1']['decoder']['denoiser']['output_node_channels'],
                num_hidden_layers = params['x1']['decoder']['denoiser']['num_MLP_hidden_layers'],
                activation=torch.nn.LeakyReLU(0.2),
                include_final_activation = False,
            )

            self.x1_decoder_denoiser_bond_MLP = None
            if self.x1_bond_diffusion:
                bond_distance_expansion_dim = 32
                self.x1_decoder_denoiser_bond_MLP =  MultiLayerPerceptron(
                    input_dim = 2 * params['x1']['decoder']['node_channels'] + params['x1']['decoder']['encoder']['input_bond_channels'] + bond_distance_expansion_dim,
                    hidden_dim = params['x1']['decoder']['denoiser']['MLP_hidden_dim'],
                    output_dim = params['x1']['decoder']['denoiser']['output_bond_channels'],
                    num_hidden_layers = params['x1']['decoder']['denoiser']['num_MLP_hidden_layers'],
                    activation=torch.nn.LeakyReLU(0.2),
                    include_final_activation = False,
                )
                self.x1_decoder_denoiser_bond_distance_scalar_expansion = GaussianSmearing(
                    start = 0.0,
                    stop = 5.0 * self.scale_point_cloud,
                    num_gaussians = bond_distance_expansion_dim,
                )


            if params['x1']['decoder']['denoiser']['use_e3nn']:

                self.x1_decoder_denoiser_E3NN = FeedForwardNetwork(
                    num_in_channels=params['x1']['decoder']['node_channels'],
                    num_hidden_channels=params['x1']['decoder']['denoiser']['e3nn']['num_hidden_channels'],
                    num_out_channels=1,
                    lmax=params['x1']['decoder']['denoiser']['e3nn']['lmax'],
                    mmax=params['x1']['decoder']['denoiser']['e3nn']['mmax'],
                    **self._ffn_options,
                )

            if params['x1']['decoder']['denoiser']['use_egnn_positions_update']:
                self.x1_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x1']['decoder']['node_channels'],
                    node_output_embedding_dim = params['x1']['decoder']['denoiser']['output_node_channels'], # ignored
                    edge_attr_dim = 0,
                    distance_expansion_dim = params['x1']['decoder']['denoiser']['egnn']['distance_expansion_dim'],
                    normalize_distance_vectors = params['x1']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'],
                    num_MLP_hidden_layers = params['x1']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x1']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )
                self._freeze_egnn_unused_node_update(self.x1_decoder_denoiser_EGNN)




        if 'x2' in self.explicit_diffusion_variables:

            self.x2_decoder_encoder_embedding = torch.nn.Linear(
                params['x2']['decoder']['input_node_channels'], # one hot embedding of real node vs virtual node
                params['x2']['decoder']['node_channels'], # linear embedding
            )

            self.x2_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x2']['decoder']['time_embedding_size'],
                params['x2']['decoder']['node_channels'],
            )

            x2_decoder_encoder_params = params['x2']['decoder']['encoder']
            assert params['x2']['decoder']['node_channels'] == x2_decoder_encoder_params['input_sphere_channels']
            assert x2_decoder_encoder_params['input_sphere_channels'] == x2_decoder_encoder_params['num_channels']
            x2_decoder_encoder_kwargs = dict(x2_decoder_encoder_params)
            x2_decoder_encoder_kwargs["final_block_channels"] = (
                params['x2']['decoder']['encoder']['num_channels'] + params['x3']['decoder']['encoder']['num_channels']
                if self.combine_x2_x3_convolution_decoder else 0
            )
            x2_decoder_encoder_kwargs["cutoff"] = x2_decoder_encoder_params['cutoff'] * self.scale_point_cloud
            self.x2_decoder_encoder = EquiformerV3(
                    **self._equiformer_kwargs(x2_decoder_encoder_kwargs),
            )


            assert params['x2']['decoder']['node_channels'] == params['x2']['decoder']['encoder']['num_channels']

            if params['x2']['decoder']['denoiser']['use_e3nn']:

                self.x2_decoder_denoiser_E3NN = FeedForwardNetwork(
                    num_in_channels=params['x2']['decoder']['node_channels'],
                    num_hidden_channels=params['x2']['decoder']['denoiser']['e3nn']['num_hidden_channels'],
                    num_out_channels=1,
                    lmax=params['x2']['decoder']['denoiser']['e3nn']['lmax'],
                    mmax=params['x2']['decoder']['denoiser']['e3nn']['mmax'],
                    **self._ffn_options,
                )

            if self.params['x2']['decoder']['denoiser']['use_egnn_positions_update']:
                self.x2_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x2']['decoder']['node_channels'],
                    node_output_embedding_dim = params['x2']['decoder']['denoiser']['output_node_channels'], # node output embeddings are ignored
                    edge_attr_dim = 0,
                    distance_expansion_dim = params['x2']['decoder']['denoiser']['egnn']['distance_expansion_dim'],
                    normalize_distance_vectors = params['x2']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'],
                    num_MLP_hidden_layers = params['x2']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x2']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )
                self._freeze_egnn_unused_node_update(self.x2_decoder_denoiser_EGNN)




        if 'x3' in self.explicit_diffusion_variables:

            self.x3_decoder_scalar_expansion = GaussianSmearing(
                start = params['x3']['decoder']['scalar_expansion_min'],
                stop = params['x3']['decoder']['scalar_expansion_max'],
                num_gaussians = params['x3']['decoder']['input_node_channels'],
            )
            self.x3_decoder_encoder_embedding = torch.nn.Linear(
                params['x3']['decoder']['input_node_channels'], # RBF expansion
                params['x3']['decoder']['node_channels'], # linear embedding
            )
            self.x3_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x3']['decoder']['time_embedding_size'],
                params['x3']['decoder']['node_channels'],
            )


            x3_decoder_encoder_params = params['x3']['decoder']['encoder']
            assert params['x3']['decoder']['node_channels'] == x3_decoder_encoder_params['input_sphere_channels']
            assert x3_decoder_encoder_params['input_sphere_channels'] == x3_decoder_encoder_params['num_channels']
            x3_decoder_encoder_kwargs = dict(x3_decoder_encoder_params)
            x3_decoder_encoder_kwargs["cutoff"] = x3_decoder_encoder_params['cutoff'] * self.scale_point_cloud
            self.x3_decoder_encoder = EquiformerV3(
                    **self._equiformer_kwargs(x3_decoder_encoder_kwargs),
            )


            assert params['x3']['decoder']['node_channels'] == params['x3']['decoder']['encoder']['num_channels']
            self.x3_decoder_denoiser_MLP = MultiLayerPerceptron(
                input_dim = params['x3']['decoder']['node_channels'], # see above assertion
                hidden_dim = params['x3']['decoder']['denoiser']['MLP_hidden_dim'],
                output_dim = params['x3']['decoder']['denoiser']['output_node_channels'], # should be 1 for a scalar potential
                num_hidden_layers = params['x3']['decoder']['denoiser']['num_MLP_hidden_layers'],
                activation=torch.nn.LeakyReLU(0.2),
                include_final_activation = False,
            )


            if params['x3']['decoder']['denoiser']['use_e3nn']:
                self.x3_decoder_denoiser_E3NN = FeedForwardNetwork(
                    num_in_channels=params['x3']['decoder']['node_channels'],
                    num_hidden_channels=params['x3']['decoder']['denoiser']['e3nn']['num_hidden_channels'],
                    num_out_channels=1,
                    lmax=params['x3']['decoder']['denoiser']['e3nn']['lmax'],
                    mmax=params['x3']['decoder']['denoiser']['e3nn']['mmax'],
                    **self._ffn_options,
                )

            if params['x3']['decoder']['denoiser']['use_egnn_positions_update']:
                self.x3_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x3']['decoder']['node_channels'],
                    node_output_embedding_dim = params['x3']['decoder']['denoiser']['output_node_channels'], # ignored; x3_decoder_denoiser_MLP is used for feature denoising
                    edge_attr_dim = 0,
                    distance_expansion_dim = params['x3']['decoder']['denoiser']['egnn']['distance_expansion_dim'],
                    normalize_distance_vectors = params['x3']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'],
                    num_MLP_hidden_layers = params['x3']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x3']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )
                self._freeze_egnn_unused_node_update(self.x3_decoder_denoiser_EGNN)




        if 'x4' in self.explicit_diffusion_variables:
            self.x4_decoder_encoder_embedding = torch.nn.Linear(
                params['x4']['decoder']['input_node_channels'], # pharmacophore oh
                params['x4']['decoder']['node_channels'], # linear embedding
            )

            # embedding l=1 directions, conditioned on pharmacophore type linear embedding.
            self.x4_decoder_encoder_embedding_l1 = FeedForwardNetwork(
                num_in_channels=params['x4']['decoder']['node_channels'],
                num_hidden_channels=self._ffn_config['num_hidden_channels'],
                num_out_channels=params['x4']['decoder']['node_channels'],
                lmax=params['lmax'],
                mmax=params['mmax'],
                **self._ffn_options,
            )

            self.x4_decoder_local_timestep_embedding = torch.nn.Linear(
                params['x4']['decoder']['time_embedding_size'],
                params['x4']['decoder']['node_channels'],
            )


            x4_decoder_encoder_params = params['x4']['decoder']['encoder']
            assert params['x4']['decoder']['node_channels'] == x4_decoder_encoder_params['input_sphere_channels']
            assert x4_decoder_encoder_params['input_sphere_channels'] == x4_decoder_encoder_params['num_channels']
            x4_decoder_encoder_kwargs = dict(x4_decoder_encoder_params)
            x4_decoder_encoder_kwargs["cutoff"] = x4_decoder_encoder_params['cutoff'] * self.scale_point_cloud
            self.x4_decoder_encoder = EquiformerV3(
                    **self._equiformer_kwargs(x4_decoder_encoder_kwargs),
            )

            assert params['x4']['decoder']['node_channels'] == params['x4']['decoder']['encoder']['num_channels']

            self.x4_decoder_denoiser_MLP = MultiLayerPerceptron(
                input_dim = params['x4']['decoder']['node_channels'],
                hidden_dim = params['x4']['decoder']['denoiser']['MLP_hidden_dim'],
                output_dim = params['x4']['decoder']['denoiser']['output_node_channels'],
                num_hidden_layers = params['x4']['decoder']['denoiser']['num_MLP_hidden_layers'],
                activation=torch.nn.LeakyReLU(0.2),
                include_final_activation = False,
            )

            if params['x4']['decoder']['denoiser']['use_e3nn']:
                self.x4_decoder_denoiser_E3NN = FeedForwardNetwork(
                    num_in_channels=params['x4']['decoder']['node_channels'],
                    num_hidden_channels=params['x4']['decoder']['denoiser']['e3nn']['num_hidden_channels'],
                    num_out_channels=1,
                    lmax=params['x4']['decoder']['denoiser']['e3nn']['lmax'],
                    mmax=params['x4']['decoder']['denoiser']['e3nn']['mmax'],
                    **self._ffn_options,
                )

            if params['x4']['decoder']['denoiser']['use_egnn_positions_update']:
                self.x4_decoder_denoiser_EGNN = EGNN(
                    node_embedding_dim = params['x4']['decoder']['node_channels'],
                    node_output_embedding_dim = params['x4']['decoder']['denoiser']['output_node_channels'], # ignored
                    edge_attr_dim = 0,
                    distance_expansion_dim = params['x4']['decoder']['denoiser']['egnn']['distance_expansion_dim'],
                    normalize_distance_vectors = params['x4']['decoder']['denoiser']['egnn']['normalize_egnn_vectors'],
                    num_MLP_hidden_layers = params['x4']['decoder']['denoiser']['egnn']['num_MLP_hidden_layers'],
                    MLP_hidden_dim = params['x4']['decoder']['denoiser']['egnn']['MLP_hidden_dim'],
                )
                self._freeze_egnn_unused_node_update(self.x4_decoder_denoiser_EGNN)

            self.x4_decoder_denoiser_E3NN_direction = FeedForwardNetwork(
                num_in_channels=params['x4']['decoder']['node_channels'],
                num_hidden_channels=params['x4']['decoder']['denoiser']['e3nn']['num_hidden_channels'],
                num_out_channels=1,
                lmax=params['x4']['decoder']['denoiser']['e3nn']['lmax'],
                mmax=params['x4']['decoder']['denoiser']['e3nn']['mmax'],
                **self._ffn_options,
            )



    ######## Forward Pass ########

    def forward_x1_decoder_encoder(self, input_dict, output_dict):

        # initial node embeddings for the graph (for discrete or continuous atom features)

        x = self._init_embedding(
            input_dict['x1']['decoder']['pos'].shape[0],
            self.params['x1']['decoder']['encoder']['lmax'],
            self.params['x1']['decoder']['encoder']['input_sphere_channels'],
        )
        x_embedding = x
        x_in = input_dict['x1']['decoder']['x']
        x_embedding[:, 0, :] = self.x1_decoder_encoder_embedding(x_in)


        # Adding time step encoding to l=0 node features
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well.

        # this is sigma that is already scaled: 1/4 * log(sigma / sigma_data) in EDM
        x1_timestep = input_dict['x1']['decoder']['c_noise']
        x1_timestep_embedding = self.x1_edm_sigma_projection(x1_timestep)
        x1_timestep_embedding = fourier_embedding(x1_timestep_embedding)
        x1_timestep_embedding = self.x1_decoder_local_timestep_embedding(x1_timestep_embedding)
        x1_timestep_embedding_pernode = x1_timestep_embedding[input_dict['x1']['decoder']['batch']]
        if self.params.get('scaffold_conditioning', False) and self.params['dataset']['x1'].get('fixed_substructure', {}):
            # shape (N, 1) so torch.where broadcasts correctly against scaffold_task_time_embedding of shape (1, 1, C)
            is_scaffold = (~input_dict['x1']['decoder']['is_diffused_atom']).view(-1, 1)
            x1_timestep_embedding_pernode = torch.where(is_scaffold, self.scaffold_task_time_embedding['x1'], x1_timestep_embedding_pernode)
        x_embedding[:, 0, :] = x_embedding[:, 0, :] + x1_timestep_embedding_pernode
        x_embedding[:, 1:, :].zero_()


        # 3D graph convolution (with Equiformer)

        edge_index = radius_graph(
            input_dict['x1']['decoder']['pos'],
            r = 1000000 if self.params['x1']['decoder']['encoder']['fully_connected'] else self.params['x1']['decoder']['encoder']['cutoff'],
            batch = input_dict['x1']['decoder']['batch'],
            # should never hit 2048, but 4096 doesn't affect speed much if necessary
            max_num_neighbors = min(self.params['x1']['decoder']['encoder'].get('max_neighbors', 2048), 2048)
        )

        # True if VN, False otherwise
        virtual_node_mask = input_dict['x1']['decoder']['virtual_node_mask']

        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x1']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x1']['decoder']['batch'])


        j, i = edge_index
        edge_distance_vec = input_dict['x1']['decoder']['pos'][j] - input_dict['x1']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)


        # embedding bond types into edge_attr
        edge_attr = None
        if self.x1_bond_diffusion:
            # fully connected, with both directed edges. Edges to virtual node will have all-zero features.
            undirected_bond_edge_index, undirected_bond_edge_x = torch_geometric.utils.to_undirected(
                input_dict['x1']['decoder']['bond_edge_index'],
                input_dict['x1']['decoder']['bond_edge_x'],
                num_nodes = input_dict['x1']['decoder']['batch'].shape[0],
                reduce = 'mean',
            )
            dense_bond_edge_attr = torch_geometric.utils.to_dense_adj(
                undirected_bond_edge_index,
                edge_attr = undirected_bond_edge_x,
                max_num_nodes = input_dict['x1']['decoder']['batch'].shape[0],
            )[0] # (N,N,channels)

            edge_attr = dense_bond_edge_attr[edge_index[0], edge_index[1]] # (N_edges, channels)
            edge_attr = self.x1_decoder_encoder_bond_edge_embedding(edge_attr)

        x = x_embedding
        x1_decoder_encoder_nodes, _ = self.x1_decoder_encoder(
            x,
            input_dict['x1']['decoder']['pos'],
            edge_index,
            edge_distance,
            edge_distance_vec,
            input_dict['x1']['decoder']['batch'],
            edge_attr = edge_attr,
        )

        x1_decoder_encoder_nodes_embedding = x1_decoder_encoder_nodes
        # Don't add a scaffold task embedding to the global embedding to condition on the scaffold task
        # because this is overidden in the heterogenous graph endoer of the decoder.

        # store results in output_dict
        output_dict['x1']['decoder']['encoder']['edge_index'] = edge_index
        output_dict = self._set_encoder_outputs(
            output_dict,
            'x1',
            x1_decoder_encoder_nodes_embedding,
            input_dict['x1']['decoder']['batch'],
            self.params['x1']['decoder']['encoder']['lmax'],
        )

        return output_dict




    def forward_x2_decoder_encoder(self, input_dict, output_dict):

        # initial node embeddings for the surface cloud

        x = self._init_embedding(
            input_dict['x2']['decoder']['pos'].shape[0],
            self.params['x2']['decoder']['encoder']['lmax'],
            self.params['x2']['decoder']['encoder']['input_sphere_channels'],
        )
        x2_embedding = self.x2_decoder_encoder_embedding(input_dict['x2']['decoder']['x']) # this embeds one-hot representations of virtual vs real nodes

        x_embedding = x
        x_embedding[:, 0, :] = x2_embedding


        # Adding time step encoding to l=0 node features
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well.

        x2_timestep = input_dict['x2']['decoder']['c_noise']
        x2_timestep_embedding = self.x2_edm_sigma_projection(x2_timestep)
        x2_timestep_embedding = fourier_embedding(x2_timestep_embedding)
        x2_timestep_embedding = self.x2_decoder_local_timestep_embedding(x2_timestep_embedding)
        if self.params.get('scaffold_conditioning', False) and self.params['dataset']['x2'].get('fixed_substructure', {}):
            raise NotImplementedError("Scaffold conditioning for x2 is not implemented yet")
        else:
            x2_timestep_embedding_pernode = x2_timestep_embedding[input_dict['x2']['decoder']['batch']]
        x_embedding[:, 0, :] = x_embedding[:, 0, :] + x2_timestep_embedding_pernode
        x_embedding[:, 1:, :].zero_()

        # 3D surface cloud convolution  (with Equiformer)

        edge_index = radius_graph(
            input_dict['x2']['decoder']['pos'],
            r = self.params['x2']['decoder']['encoder']['cutoff'] * self.scale_point_cloud,
            batch = input_dict['x2']['decoder']['batch'],
            max_num_neighbors = min(self.params['x2']['decoder']['encoder'].get('max_neighbors', 2048), 2048),
        )

        # True if VN, False otherwise
        virtual_node_mask = input_dict['x2']['decoder']['virtual_node_mask']

        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x2']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x2']['decoder']['batch'])


        j, i = edge_index
        edge_distance_vec = input_dict['x2']['decoder']['pos'][j] - input_dict['x2']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)

        x = x_embedding
        _, x2_decoder_encoder_nodes = self.x2_decoder_encoder(
            x,
            input_dict['x2']['decoder']['pos'],
            edge_index,
            edge_distance,
            edge_distance_vec,
            input_dict['x2']['decoder']['batch'],
        )

        x2_decoder_encoder_nodes_embedding = x2_decoder_encoder_nodes


        # store results in output_dict
        output_dict['x2']['decoder']['encoder']['edge_index'] = edge_index
        output_dict = self._set_encoder_outputs(
            output_dict,
            'x2',
            x2_decoder_encoder_nodes_embedding,
            input_dict['x2']['decoder']['batch'],
            self.params['x2']['decoder']['encoder']['lmax'],
        )

        return output_dict




    def forward_x3_decoder_encoder(self, input_dict, output_dict):

        # initial node embeddings for the surface cloud

        x = self._init_embedding(
            input_dict['x3']['decoder']['pos'].shape[0],
            self.params['x3']['decoder']['encoder']['lmax'],
            self.params['x3']['decoder']['encoder']['input_sphere_channels'],
        )
        x3_in = self.x3_decoder_scalar_expansion(input_dict['x3']['decoder']['x'])
        x3_embedding = self.x3_decoder_encoder_embedding(x3_in)
        virtual_node_mask = input_dict['x3']['decoder']['virtual_node_mask']
        num_nodes = input_dict['x3']['decoder']['pos'].shape[0]  # canonical node count (matches virtual_node_mask)
        if virtual_node_mask is not None:
            # zeroing-out x3_embedding for virtual nodes (which have no electrostatic potential)
            assert x3_embedding.shape[0] == num_nodes, (
                f"x3_embedding.shape[0] ({x3_embedding.shape[0]}) != num_nodes ({num_nodes}); "
                "sigma/c_noise indexing may have produced wrong x shape in EDM preconditioner."
            )
            mask = torch.ones(num_nodes, device=self.device)
            mask[virtual_node_mask] = 0.0
            x3_embedding = x3_embedding * mask[:, None]
        x_embedding = x
        x_embedding[:, 0, :] = x3_embedding


        # Adding time step encoding to l=0 node features
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well.

        x3_timestep = input_dict['x3']['decoder']['c_noise'] # (B, 1)
        x3_timestep_embedding = self.x3_edm_sigma_projection(x3_timestep)
        x3_timestep_embedding = fourier_embedding(x3_timestep_embedding)
        x3_timestep_embedding = self.x3_decoder_local_timestep_embedding(x3_timestep_embedding)
        if self.params.get('scaffold_conditioning', False) and self.params['dataset']['x3'].get('fixed_substructure', {}):
            raise NotImplementedError("Scaffold conditioning for x3 is not implemented yet")
        else:
            x3_timestep_embedding_pernode = x3_timestep_embedding[input_dict['x3']['decoder']['batch']]
        x_embedding[:, 0, :] = x_embedding[:, 0, :] + x3_timestep_embedding_pernode
        x_embedding[:, 1:, :].zero_()


        # 3D surface cloud convolution (with Equiformer)

        edge_index = radius_graph(
            input_dict['x3']['decoder']['pos'],
            r = self.params['x3']['decoder']['encoder']['cutoff'] * self.scale_point_cloud,
            batch = input_dict['x3']['decoder']['batch'],
            max_num_neighbors = min(self.params['x3']['decoder']['encoder'].get('max_neighbors', 2048), 2048),
        )

        # True if VN, False otherwise
        virtual_node_mask = input_dict['x3']['decoder']['virtual_node_mask']

        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x3']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x3']['decoder']['batch'])


        j, i = edge_index
        edge_distance_vec = input_dict['x3']['decoder']['pos'][j] - input_dict['x3']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)

        x = x_embedding
        x3_decoder_encoder_nodes, _ = self.x3_decoder_encoder(
            x,
            input_dict['x3']['decoder']['pos'],
            edge_index,
            edge_distance,
            edge_distance_vec,
            input_dict['x3']['decoder']['batch'],
        )

        x3_decoder_encoder_nodes_embedding = x3_decoder_encoder_nodes


        # store results in output_dict
        output_dict['x3']['decoder']['encoder']['edge_index'] = edge_index
        output_dict = self._set_encoder_outputs(
            output_dict,
            'x3',
            x3_decoder_encoder_nodes_embedding,
            input_dict['x3']['decoder']['batch'],
            self.params['x3']['decoder']['encoder']['lmax'],
        )


        return output_dict



    def forward_x4_decoder_encoder(self, input_dict, output_dict):
        # initial node embeddings for the graph

        x = self._init_embedding(
            input_dict['x4']['decoder']['pos'].shape[0],
            self.params['x4']['decoder']['encoder']['lmax'],
            self.params['x4']['decoder']['encoder']['input_sphere_channels'],
        )
        x_embedding = x
        x4_in = input_dict['x4']['decoder']['x']
        x_embedding[:, 0, :] = self.x4_decoder_encoder_embedding(x4_in)

        # insert vector directions as l=1 features
        x_embedding[:, 1:4, :] = input_dict['x4']['decoder']['direction'][..., None]
        x_embedding[:, 4:, :].zero_()

        # further embedding of l=0, l=1 input features
        x = self.x4_decoder_encoder_embedding_l1(x) # FeedForward


        # Adding time step encoding to l=0 node features
            # we could also concatenate these as extra channels, but then we'd need to expand the l=1 channels as well.

        x4_timestep = input_dict['x4']['decoder']['c_noise']
        x4_timestep_embedding = self.x4_edm_sigma_projection(x4_timestep)
        x4_timestep_embedding = fourier_embedding(x4_timestep_embedding)
        x4_timestep_embedding = self.x4_decoder_local_timestep_embedding(x4_timestep_embedding)
        x4_timestep_embedding_pernode = x4_timestep_embedding[input_dict['x4']['decoder']['batch']]
        if self.params.get('scaffold_conditioning', False) and self.params['dataset']['x4'].get('fixed_substructure', {}):
            # shape (N, 1) so torch.where broadcasts correctly against scaffold_task_time_embedding of shape (1, 1, C)
            is_scaffold = (~input_dict['x4']['decoder']['is_diffused_pharm']).view(-1, 1)
            x4_timestep_embedding_pernode = torch.where(is_scaffold, self.scaffold_task_time_embedding['x4'], x4_timestep_embedding_pernode)
        x_embedding = x
        x_embedding[:, 0, :] = x_embedding[:, 0, :] + x4_timestep_embedding_pernode


        # 3D graph convolution (with Equiformer)

        edge_index = radius_graph(
            input_dict['x4']['decoder']['pos'],
            r = self.params['x4']['decoder']['encoder']['cutoff'] * self.scale_point_cloud,
            batch = input_dict['x4']['decoder']['batch'],
            max_num_neighbors = min(self.params['x4']['decoder']['encoder'].get('max_neighbors', 2048), 2048),
        )

        # True if VN, False otherwise
        virtual_node_mask = input_dict['x4']['decoder']['virtual_node_mask']

        if virtual_node_mask is not None:
            force_edges_to_virtual_nodes = self.params['x4']['decoder']['force_edges_to_virtual_nodes']
            if force_edges_to_virtual_nodes and (virtual_node_mask.any()):
                # if a graph instance has multiple VNs, this will introduce edges between those VNs
                    # this will remove self-loops on individual VNs
                edge_index = add_virtual_edges_to_edge_index(edge_index, virtual_node_mask, input_dict['x4']['decoder']['batch'])


        j, i = edge_index
        edge_distance_vec = input_dict['x4']['decoder']['pos'][j] - input_dict['x4']['decoder']['pos'][i]
        edge_distance = edge_distance_vec.norm(dim=-1)

        x = x_embedding
        x4_decoder_encoder_nodes, _ = self.x4_decoder_encoder(
            x,
            input_dict['x4']['decoder']['pos'],
            edge_index,
            edge_distance,
            edge_distance_vec,
            input_dict['x4']['decoder']['batch'],
        )

        x4_decoder_encoder_nodes_embedding = x4_decoder_encoder_nodes

        # store results in output_dict
        output_dict['x4']['decoder']['encoder']['edge_index'] = edge_index
        output_dict = self._set_encoder_outputs(
            output_dict,
            'x4',
            x4_decoder_encoder_nodes_embedding,
            input_dict['x4']['decoder']['batch'],
            self.params['x4']['decoder']['encoder']['lmax'],
        )

        return output_dict



    def forward_decoder_joint_heterogeneous_graph_encoder(self, input_dict, output_dict):

        heterogeneous_variables = deepcopy([x_ for x_ in self.explicit_diffusion_variables if x_ not in self.exclude_variables_from_decoder_heterogeneous_graph])

        hetero_pos = torch.cat(
            [input_dict[x_]['decoder']['pos'] for x_ in heterogeneous_variables]
        , dim = 0)
        hetero_virtual_node_mask = torch.cat(
            [input_dict[x_]['decoder']['virtual_node_mask'] for x_ in heterogeneous_variables],
            dim = 0)
        hetero_batch = torch.cat(
            [input_dict[x_]['decoder']['batch'] for x_ in heterogeneous_variables]
            , dim = 0)
        hetero_x_identifier = torch.cat(
            [torch.ones_like(input_dict[x_]['decoder']['batch']) * i for i, x_ in enumerate(heterogeneous_variables)]
            , dim = 0)


        hetero_node_embeddings = torch.cat(
                [output_dict[x_]['decoder']['encoder']['node_embedding'] for x_ in heterogeneous_variables],
                dim = 0
            )


        # creating new edge index for heteregeneous graph, adding new edges for heterogeneous nodes within cut-off radius
        # radius_graph also assumes a sorted `batch` and silently returns wrong edges otherwise, and
        # hetero_batch concatenates each variable's batch requiring the argsort and remap.
        argsorted_batch = torch.argsort(hetero_batch)
        hetero_edge_index = radius_graph(
            hetero_pos[argsorted_batch],
            r = self.params['decoder_heterogeneous_graph_encoder']['cutoff'] * self.scale_point_cloud,
            batch = hetero_batch[argsorted_batch],
            max_num_neighbors = min(self.params['decoder_heterogeneous_graph_encoder'].get('max_neighbors', 2048), 2048),
        )
        hetero_edge_index = remap_values(
            (torch.arange(len(hetero_batch), device = argsorted_batch.device), argsorted_batch),
            hetero_edge_index,
        )
        hetero_edge_index = torch_geometric.utils.sort_edge_index(hetero_edge_index)

        # removing intra-x edges (except for intra-x1 edges), mainly to increase speed
        # only applied when prune_joint_edges is set; the legacy mask kept every edge (it is a no-op),
        # so leaving the flag off reproduces the behavior of previously trained checkpoints
        assert 'x1' in heterogeneous_variables # we'll have to change this code if we don't want to explicitly diffuse over x1
        if self.params['decoder_heterogeneous_graph_encoder'].get('prune_joint_edges', False):
            edge_x_identifier = hetero_x_identifier[hetero_edge_index]
            x1_edges = edge_x_identifier == heterogeneous_variables.index('x1')
            edge_index_mask = (edge_x_identifier[0] != edge_x_identifier[1]) | (x1_edges[0] & x1_edges[1])
            hetero_edge_index = hetero_edge_index[:, edge_index_mask]

        # removing any edges to or from a virtual node
        edge_index_mask = hetero_virtual_node_mask[hetero_edge_index]
        edge_index_mask = edge_index_mask[0] | edge_index_mask[1]
        hetero_edge_index = hetero_edge_index[:, ~edge_index_mask]

        j, i = hetero_edge_index
        hetero_edge_distance_vec = hetero_pos[j] - hetero_pos[i]
        hetero_edge_distance = hetero_edge_distance_vec.norm(dim=-1) + 1e-6


        hetero_node_embeddings, _ = self.decoder_joint_heterogeneous_graph_encoder(
            hetero_node_embeddings,
            hetero_pos,
            hetero_edge_index,
            hetero_edge_distance,
            hetero_edge_distance_vec,
            hetero_batch,
        )


        for i, x_ in enumerate(heterogeneous_variables):
            # residual connection to heterogeneous node embeddings

            x_node_embedding = output_dict[x_]['decoder']['encoder']['node_embedding']
            x_node_embedding_tensor = x_node_embedding + hetero_node_embeddings[hetero_x_identifier == i, ...]

            output_dict[x_]['decoder']['encoder']['node_embedding'] = x_node_embedding_tensor

            # also updating the global embeddings
            x_global_embedding = torch_scatter.scatter_sum(
                x_node_embedding_tensor,
                input_dict[x_]['decoder']['batch'],
                dim = 0,
            )
            output_dict[x_]['decoder']['encoder']['global_embedding'] = x_global_embedding

            # Add scaffold task embedding to the global embedding for molecules in the scaffold task.
            # Use masked addition (same idea as forward_decoder_joint_processing): purely indexed assignment
            # with an empty mask disconnects scaffold_task_global_embedding from autograd and breaks DDP
            # when a rank gets a batch with no scaffold-task molecules.
            if self.params.get('scaffold_conditioning', False):
                x_global_embedding = output_dict[x_]['decoder']['encoder']['global_embedding']
                scaffold_mask = input_dict['x1']['decoder']['scaffold_task'].to(x_global_embedding.dtype)
                scaffold_mask = scaffold_mask[:, None, None]
                l0_mask = torch.zeros(
                    (1, x_global_embedding.shape[1], 1),
                    dtype=x_global_embedding.dtype,
                    device=x_global_embedding.device,
                )
                l0_mask[:, 0:1, :] = 1.0
                scaffold_l0_mask = scaffold_mask * l0_mask
                # In-place add keeps writes on the underlying node-embedding tensor; reassignment would drop updates.
                x_global_embedding.add_(scaffold_l0_mask * self.scaffold_task_global_embedding)

        return output_dict




    def forward_decoder_joint_processing(self, x_str, input_dict, output_dict):

        assert x_str in self.explicit_diffusion_variables

        x = output_dict[x_str]['decoder']['encoder']['node_embedding']
        x_embedding = x
        x_global_embedding = output_dict[x_str]['decoder']['encoder']['global_embedding']
        batch_size = x_global_embedding.shape[0]


        # Obtain joint l>0 global embeddings from all explicit decoders. Optionally
        # retain l=0 so the equivariant FFN can learn scalar-conditioned l>0 mixing.

        _K = self._joint_mixer_K
        joint_ffn_sources = [
            output_dict[source_x]['decoder']['encoder']['global_embedding'][:, 0:_K, :]
            for source_x in self.explicit_diffusion_variables
        ]
        joint_ffn_input = torch.cat(joint_ffn_sources, dim=-1)
        joint_ffn_input = torch.cat(
            (torch.zeros_like(joint_ffn_input[:, 0:1, :]), joint_ffn_input[:, 1:_K, :]),
            dim=1,
        )

        joint_l1_embedding = self._init_embedding(
            batch_size,
            self.params['lmax'],
            joint_ffn_input.shape[-1],
        )
        joint_l1_embedding_tensor = joint_l1_embedding
        joint_l1_embedding_tensor[:, 0:_K, :] = joint_ffn_input
        joint_l1_embedding_tensor[:, _K:, :].zero_()
        joint_l1_embedding = joint_l1_embedding_tensor


        # Only the l>0 FFN outputs are carried into joint_embedding_update below;
        # its l=0 slot remains reserved for the timestep/scaffold embeddings.
        if x_str == 'x1':
            joint_l1_embedding = self.x1_decoder_global_l1_embedding(joint_l1_embedding)
        if x_str == 'x2':
            joint_l1_embedding = self.x2_decoder_global_l1_embedding(joint_l1_embedding)
        if x_str == 'x3':
            joint_l1_embedding = self.x3_decoder_global_l1_embedding(joint_l1_embedding)
        if x_str == 'x4':
            joint_l1_embedding = self.x4_decoder_global_l1_embedding(joint_l1_embedding)



        # Obtaining l=0 time-step embeddings
        timesteps = [input_dict[x_]['decoder']['c_noise'] for x_ in self.explicit_diffusion_variables]

        concat_timestep_embedding = []
        for i, x_ in enumerate(self.explicit_diffusion_variables):
            if x_ == 'x1':
                concat_timestep_embedding.append(fourier_embedding(self.x1_edm_sigma_projection(timesteps[i])))
            if x_ == 'x2':
                concat_timestep_embedding.append(fourier_embedding(self.x2_edm_sigma_projection(timesteps[i])))
            if x_ == 'x3':
                concat_timestep_embedding.append(fourier_embedding(self.x3_edm_sigma_projection(timesteps[i])))
            if x_ == 'x4':
                concat_timestep_embedding.append(fourier_embedding(self.x4_edm_sigma_projection(timesteps[i])))
        concat_timestep_embedding = torch.cat(concat_timestep_embedding, dim = -1)

        if x_str == 'x1':
            global_timestep_embedding = self.x1_decoder_global_timestep_embedding(concat_timestep_embedding)
        if x_str == 'x2':
            global_timestep_embedding = self.x2_decoder_global_timestep_embedding(concat_timestep_embedding)
        if x_str == 'x3':
            global_timestep_embedding = self.x3_decoder_global_timestep_embedding(concat_timestep_embedding)
        if x_str == 'x4':
            global_timestep_embedding = self.x4_decoder_global_timestep_embedding(concat_timestep_embedding)


        # Aggregating all global features, mixing their l-channels, and applying them at once in a residual update to the node embeddings

        # aggregating updates
        joint_embedding_update = self._init_embedding(
            batch_size,
            self.params['lmax'],
            x_global_embedding.shape[-1],
        )
        joint_embedding_update_tensor = joint_embedding_update
        joint_embedding_update_tensor[:, 1:_K, :] = joint_l1_embedding[:, 1:_K, :]
        joint_embedding_update_tensor[:, 0, :] = global_timestep_embedding
        joint_embedding_update_tensor[:, _K:, :].zero_()

        # Add scaffold task embedding to the global embedding for the molecules in the scaffold task to l=0 channel
        # before mixing with l=0 and l=1 channels
        # Add this to all explicit diffusion variables.
        if self.params.get('scaffold_conditioning', False):
            scaffold_mask = input_dict['x1']['decoder']['scaffold_task'].to(joint_embedding_update_tensor.dtype)
            scaffold_mask = scaffold_mask[:, None, None]
            l0_mask = torch.zeros(
                (1, joint_embedding_update_tensor.shape[1], 1),
                dtype=joint_embedding_update_tensor.dtype,
                device=joint_embedding_update_tensor.device,
            )
            l0_mask[:, 0:1, :] = 1.0
            scaffold_l0_mask = scaffold_mask * l0_mask
            joint_embedding_update_tensor = (
                joint_embedding_update_tensor
                + scaffold_l0_mask * self.scaffold_task_global_embedding
            )
            if x_str in self.scaffold_task_time_embedding:
                joint_embedding_update_tensor = (
                    joint_embedding_update_tensor
                    + scaffold_l0_mask * self.scaffold_task_time_embedding[x_str]
                )


        # (learnable) mixing of l=0..joint_mixer_lmax channels of joint_embedding_update with tensor products
        joint_embedding_update_e3nn = convert_equiformerv3_to_e3nn(joint_embedding_update_tensor[:, 0:_K, :], lmax=self.joint_mixer_lmax)
        if x_str == 'x1':
            joint_embedding_update_e3nn = self.x1_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)
        if x_str == 'x2':
            joint_embedding_update_e3nn = self.x2_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)
        if x_str == 'x3':
            joint_embedding_update_e3nn = self.x3_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)
        if x_str == 'x4':
            joint_embedding_update_e3nn = self.x4_decoder_equiformer_tensor_product(joint_embedding_update_e3nn, joint_embedding_update_e3nn)

        joint_embedding_update_tensor[:, 0:_K, :] = convert_e3nn_to_equiformerv3(
            joint_embedding_update_e3nn,
            lmax=self.joint_mixer_lmax,
            num_channels=joint_embedding_update_tensor.shape[-1],
        )

        # residually updating node embeddings with mixed global joint embeddings
        x_embedding[:, 0:_K, :] = x_embedding[:, 0:_K, :] + joint_embedding_update_tensor[input_dict[x_str]['decoder']['batch'], 0:_K, :]


        output_dict[x_str]['decoder']['node_joint_embedding'] = x_embedding

        return output_dict




    def forward_x1_decoder_denoiser(self, input_dict, output_dict):

        x1 = output_dict['x1']['decoder']['node_joint_embedding']
        x1_positions = input_dict['x1']['decoder']['pos']

        virtual_node_mask = input_dict['x1']['decoder']['virtual_node_mask']

        # atom type update
        x1_features = x1[:,0,:]  # only l=0 features
        x1_features_update = self.x1_decoder_denoiser_MLP(x1_features)


        # bond type update
        x1_bond_features_update = None
        if self.x1_bond_diffusion:
            x1_features = x1[:,0,:]  # only l=0 features
            # this bond_edge_index includes only 1 directed edge per bond
            bond_edge_index = input_dict['x1']['decoder']['bond_edge_index']
            x1_bond_features = input_dict['x1']['decoder']['bond_edge_x']

            # get distance expansion of pairwise distances between nodes
            x1_bond_distance_expansion = x1_positions[bond_edge_index[0]] - x1_positions[bond_edge_index[1]]
            x1_bond_distance_expansion = x1_bond_distance_expansion.norm(dim = -1, keepdim = True)
            x1_bond_distance_expansion = self.x1_decoder_denoiser_bond_distance_scalar_expansion(x1_bond_distance_expansion)

            x1_bond_features_update_01 = self.x1_decoder_denoiser_bond_MLP(
                torch.cat([x1_bond_features, x1_bond_distance_expansion, x1_features[bond_edge_index[0]], x1_features[bond_edge_index[1]]], dim = 1)
            )
            x1_bond_features_update_10 = self.x1_decoder_denoiser_bond_MLP(
                torch.cat([x1_bond_features, x1_bond_distance_expansion, x1_features[bond_edge_index[1]], x1_features[bond_edge_index[0]]], dim = 1)
            )
            x1_bond_features_update = (x1_bond_features_update_01 + x1_bond_features_update_10) / 2.0 # symmetrical update



        # denoising steps for node coordinates

        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x1']['decoder']['encoder']['edge_index']


        x1_positions_update = torch.zeros_like(x1_positions)
        if self.params['x1']['decoder']['denoiser']['use_e3nn']:
            x1_e3nn_update = self.x1_decoder_denoiser_E3NN(x1)
            x1_positions_update_e3nn = x1_e3nn_update[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x1_positions_update_e3nn[virtual_node_mask] = 0.0

            x1_positions = x1_positions + x1_positions_update_e3nn
            x1_positions_update = x1_positions_update + x1_positions_update_e3nn

        if self.params['x1']['decoder']['denoiser']['use_egnn_positions_update']:
            _, x1_positions_update_egnn = self.x1_decoder_denoiser_EGNN(
                x = x1_features,
                pos = x1_positions,
                edge_index = edge_index,
                batch = input_dict['x1']['decoder']['batch'],
                edge_attr = None,
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x1_positions_update_egnn[virtual_node_mask] = 0.0

            x1_positions = x1_positions + x1_positions_update_egnn
            x1_positions_update = x1_positions_update + x1_positions_update_egnn

        # can we use an bond distance/angle validity loss on top of x1_positions?

        # store results in output_dict
        output_dict['x1']['decoder']['denoiser']['x_out'] = x1_features_update # continuous for now
        output_dict['x1']['decoder']['denoiser']['pos_out'] = x1_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)
        output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'] = x1_bond_features_update # continuous for now

        return output_dict



    def forward_x2_decoder_denoiser(self, input_dict, output_dict):

        x2 = output_dict['x2']['decoder']['node_joint_embedding']
        x2_positions = input_dict['x2']['decoder']['pos']

        virtual_node_mask = input_dict['x2']['decoder']['virtual_node_mask']


        # denoising steps for coordinates (e.g., with E3NN/EGNN)

        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x2']['decoder']['encoder']['edge_index']

        x2_positions_update = torch.zeros_like(x2_positions)
        if self.params['x2']['decoder']['denoiser']['use_e3nn']:
            x2_e3nn_update = self.x2_decoder_denoiser_E3NN(x2)
            x2_positions_update_e3nn = x2_e3nn_update[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x2_positions_update_e3nn[virtual_node_mask] = 0.0

            x2_positions = x2_positions + x2_positions_update_e3nn
            x2_positions_update = x2_positions_update + x2_positions_update_e3nn

        if self.params['x2']['decoder']['denoiser']['use_egnn_positions_update']:
            x2_features = x2[:,0,:]

            _, x2_positions_update_egnn = self.x2_decoder_denoiser_EGNN(
                x = x2_features,  # only l=0 features
                pos = x2_positions,
                edge_index = edge_index,
                batch = input_dict['x2']['decoder']['batch'],
                edge_attr = None,
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x2_positions_update_egnn[virtual_node_mask] = 0.0

            x2_positions = x2_positions + x2_positions_update_egnn
            x2_positions_update = x2_positions_update + x2_positions_update_egnn

        # store results in output_dict
        output_dict['x2']['decoder']['denoiser']['pos_out'] = x2_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)

        return output_dict



    def forward_x3_decoder_denoiser(self, input_dict, output_dict):

        x3 = output_dict['x3']['decoder']['node_joint_embedding']
        virtual_node_mask = input_dict['x3']['decoder']['virtual_node_mask']

        x3_features_update = self.x3_decoder_denoiser_MLP(x3[:,0,:])
        if virtual_node_mask is not None:
            x3_features_update[virtual_node_mask] = 0.0
        output_dict['x3']['decoder']['denoiser']['x_out'] = x3_features_update.squeeze() # (M,1) -> (M,)


        x3_positions = input_dict['x3']['decoder']['pos']

        # denoising steps for coordinates (e.g., with E3NN/EGNN)

        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x3']['decoder']['encoder']['edge_index']

        x3_positions_update = torch.zeros_like(x3_positions)
        if self.params['x3']['decoder']['denoiser']['use_e3nn']:
            x3_e3nn_update = self.x3_decoder_denoiser_E3NN(x3)
            x3_positions_update_e3nn = x3_e3nn_update[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x3_positions_update_e3nn[virtual_node_mask] = 0.0

            x3_positions = x3_positions + x3_positions_update_e3nn
            x3_positions_update = x3_positions_update + x3_positions_update_e3nn

        if self.params['x3']['decoder']['denoiser']['use_egnn_positions_update']:
            x3_features = x3[:,0,:]

            _, x3_positions_update_egnn = self.x3_decoder_denoiser_EGNN(
                x = x3_features,  # only l=0 features
                pos = x3_positions,
                edge_index = edge_index,
                batch = input_dict['x3']['decoder']['batch'],
                edge_attr = None,
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x3_positions_update_egnn[virtual_node_mask] = 0.0

            x3_positions = x3_positions + x3_positions_update_egnn
            x3_positions_update = x3_positions_update + x3_positions_update_egnn

        # store results in output_dict
        output_dict['x3']['decoder']['denoiser']['pos_out'] = x3_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)

        return output_dict



    def forward_x4_decoder_denoiser(self, input_dict, output_dict):

        x4 = output_dict['x4']['decoder']['node_joint_embedding']
        x4_positions = input_dict['x4']['decoder']['pos']
        # x4_directions = input_dict['x4']['decoder']['direction']

        virtual_node_mask = input_dict['x4']['decoder']['virtual_node_mask']


        x4_features = x4[:,0,:]  # only l=0 features
        x4_features_update = self.x4_decoder_denoiser_MLP(x4_features)


        # denoising steps for node directions and coordinates

        x4_e3nn_direction_update = self.x4_decoder_denoiser_E3NN_direction(x4)[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs
        if virtual_node_mask is not None:
            x4_e3nn_direction_update[virtual_node_mask] = 0.0


        # re-using edge_index from radius graph of the structure encoder
            # already forces edges between every node and the virtual nodes
        edge_index = output_dict['x4']['decoder']['encoder']['edge_index']

        x4_positions_update = torch.zeros_like(x4_positions)
        if self.params['x4']['decoder']['denoiser']['use_e3nn']:
            x4_e3nn_update = self.x4_decoder_denoiser_E3NN(x4)
            x4_positions_update_e3nn = x4_e3nn_update[:, 1:4, :].squeeze(dim=2) # (B,3) l=1 outputs

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x4_positions_update_e3nn[virtual_node_mask] = 0.0

            x4_positions = x4_positions + x4_positions_update_e3nn
            x4_positions_update = x4_positions_update + x4_positions_update_e3nn

        if self.params['x4']['decoder']['denoiser']['use_egnn_positions_update']:
            _, x4_positions_update_egnn = self.x4_decoder_denoiser_EGNN(
                x = x4_features,
                pos = x4_positions,
                edge_index = edge_index,
                batch = input_dict['x4']['decoder']['batch'],
                edge_attr = None,
                pos_update_mask = None, # mask applied separately below
                residual_pos_update = False,
                residual_x_update = False,
            )

            # need to apply VN mask here
            if virtual_node_mask is not None:
                x4_positions_update_egnn[virtual_node_mask] = 0.0

            x4_positions = x4_positions + x4_positions_update_egnn
            x4_positions_update = x4_positions_update + x4_positions_update_egnn


        # store results in output_dict
        output_dict['x4']['decoder']['denoiser']['x_out'] = x4_features_update # continuous for now
        output_dict['x4']['decoder']['denoiser']['pos_out'] = x4_positions_update # these are "delta" positions (e.g., predicted noise, not a predicted structure)
        output_dict['x4']['decoder']['denoiser']['direction_out'] = x4_e3nn_direction_update # these are "delta" directions (e.g., predicted noise, not a predicted structure)

        return output_dict


    def edm_preconditioner(self, input_dict):
        """
        Scales input features by c_in and noise levels by c_noise. Returns a context dict
        containing original inputs and scaling factors for the skip connection.
        """
        edm_context = {}

        # Helper to process a single field
        # scheduler: instance of EDMPreconditioner or similar
        def process_variable(var_name, field_name, preconditioner: EDMPreconditioner, scale_by_sigma_data: bool = False):
            decoder_dict = input_dict.get(var_name, {}).get('decoder', {})
            if field_name not in decoder_dict:
                return

            x_original = decoder_dict[field_name]   # Shape: (N, D)
            sigma_batch = decoder_dict['sigma']     # Shape: (B,)
            batch_idx = decoder_dict['batch']       # Shape: (N,)
            if field_name == 'bond_edge_x':
                sigma_node = bond_weighting(
                    input_tensor=sigma_batch,
                    graph_batch_index=batch_idx,
                    virtual_node_x1=self.params['dataset'][var_name]['add_virtual_node']
                )
                factors = preconditioner.get_scaling_factors(sigma_node)

            else:
                sigma_node = sigma_batch[batch_idx]
                factors = preconditioner.get_scaling_factors(
                    sigma_batch, unsqueeze=False if (var_name == 'x3' and field_name == 'x') else True
                )
                # do not expand c_noise since we expect batch size
                factors.c_in = factors.c_in[batch_idx]
                factors.c_out = factors.c_out[batch_idx]
                factors.c_skip = factors.c_skip[batch_idx]

            # Scale the input for the network (c_in * x) to unit variance
            x_scaled = x_original * factors.c_in
            if scale_by_sigma_data:
                # scale positions back up by sigma_data for Equiformer RBFs
                x_scaled = x_scaled * preconditioner.sigma_data.to(device=x_original.device, dtype=x_original.dtype)
            input_dict[var_name]['decoder'][field_name] = x_scaled

            # Noise level scaling (modules expect 1/4*ln(sigma/sigma_data)). Single c_noise slot per task
            # is harmless: fields share one sigma, so the offset is a constant absorbed by xN_edm_sigma_projection.
            if field_name != 'bond_edge_x':
                input_dict[var_name]['decoder']['c_noise'] = factors.c_noise.unsqueeze(-1) if (var_name == 'x3' and field_name == 'x') else factors.c_noise

            # print(f"{var_name}: {field_name}")
            # print(f"\tc_in: {factors.c_in.shape}, virtual_node_mask: {decoder_dict['virtual_node_mask'].shape}")
            # print(f"\tc_out: {factors.c_out.shape}, c_noise: {factors.c_noise.shape}")
            # print(f"\tx_original: {x_original.shape}, x_scaled: {x_scaled.shape}")

            edm_context[f"{var_name}_{field_name}"] = {
                'original_x': x_original, # raw input for skip connection
                'c_skip': factors.c_skip,
                'c_out': factors.c_out,
                'output_key': f"{field_name}_out" # Map 'pos' -> 'pos_out'
            }

        if 'x1' in self.explicit_diffusion_variables:
            # use `not self.params['edm']['scale_by_sigma_data']` because the default is `True` (`not False`):
            # - DO scale by sigma_data HERE, and don't scale NOISE (in datasets.py) by sigma_data --> DEFAULT
            # - DO NOT scale the sigma_data here, because we DO scale the noise in datasets.py so the noise is at the correct scale
            process_variable('x1', 'pos', self.edm_noise_schedule_pos, scale_by_sigma_data=not self.params['edm'].get('scale_by_sigma_data', False))
            process_variable('x1', 'x', self.edm_noise_schedule_atom_one_hot)
            process_variable('x1', 'bond_edge_x', self.edm_noise_schedule_bond)

        if 'x2' in self.explicit_diffusion_variables:
            process_variable('x2', 'pos', self.edm_noise_schedule_pos, scale_by_sigma_data=not self.params['edm'].get('scale_by_sigma_data', False))

        if 'x3' in self.explicit_diffusion_variables:
            process_variable('x3', 'pos', self.edm_noise_schedule_pos, scale_by_sigma_data=not self.params['edm'].get('scale_by_sigma_data', False))
            process_variable('x3', 'x', self.edm_noise_schedule_esp)

        if 'x4' in self.explicit_diffusion_variables:
            process_variable('x4', 'pos', self.edm_noise_schedule_pos, scale_by_sigma_data=not self.params['edm'].get('scale_by_sigma_data', False))
            process_variable('x4', 'x', self.edm_noise_schedule_pharm_one_hot)
            process_variable('x4', 'direction', self.edm_noise_schedule_direction)

        return input_dict, edm_context

    def edm_postprocess(self, output_dict, edm_context):
        """
        Applies EDM skip connection to the output_dict:
        Output = c_skip * original_x + c_out * NetworkOutput

        This is a prediction of the clean structure, not the noise.
        """
        if not edm_context:
            return output_dict

        for key, ctx in edm_context.items():
            var_name, field_name = key.split('_', 1) # e.g., "x1_pos"
            output_key = ctx['output_key'] # e.g., "pos_out"

            # network output
            denoiser_dict = output_dict[var_name]['decoder']['denoiser']

            if denoiser_dict.get(output_key) is not None:
                F_x = denoiser_dict[output_key]

                # D(x, sigma) = c_skip * x + c_out * F(x_scaled, sigma)
                D_x = ctx['c_skip'] * ctx['original_x'] + ctx['c_out'] * F_x

                denoiser_dict[output_key] = D_x
        return output_dict


    # forward function for training
        # this training function could also be split into separate diffusion processes
            # nothing REQUIRES us to train on all diffusion branches in the same batch ...
    def forward(self, input_dict):

        self.device = input_dict['device']
        self.dtype = input_dict['dtype']

        # placeholder to define the organization of this dictionary
        output_dict = {
            'x1': {

                'decoder': {

                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },

                    'node_joint_embedding': None,

                    'denoiser': {
                        'x_out': None,
                        'pos_out': None,
                    },

                },
            },



            'x2': {

                'decoder': {
                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },

                    'node_joint_embedding': None,

                    'denoiser': {
                        'pos_out': None,
                    },

                },
            },



            'x3': {

                'decoder': {
                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },

                    'node_joint_embedding': None,

                    'denoiser': {
                        'x_out': None,
                        'pos_out': None,
                    },

                },
            },


            'x4': {

                'decoder': {
                    'encoder': {
                        'node_embedding': None,
                        'global_embedding': None,
                        'edge_index': None,
                    },

                    'node_joint_embedding': None,

                    'denoiser': {
                        'x_out': None,
                        'pos_out': None,
                        'direction_out': None,
                    },

                },
            },

        }

        # EDM scale network input
        input_dict, edm_context = self.edm_preconditioner(input_dict)

        # Embedding Modules
        if 'x1' in self.explicit_diffusion_variables:
            output_dict = self.forward_x1_decoder_encoder(input_dict, output_dict)
        if 'x2' in self.explicit_diffusion_variables:
            output_dict = self.forward_x2_decoder_encoder(input_dict, output_dict)
        if 'x3' in self.explicit_diffusion_variables:
            output_dict = self.forward_x3_decoder_encoder(input_dict, output_dict)
        if 'x4' in self.explicit_diffusion_variables:
            output_dict = self.forward_x4_decoder_encoder(input_dict, output_dict)

        # Joint Module
            # - pass local messages within the heterogeneous graph of the explicit diffusion (decoder) variables
            # - jointly process global codes
        if self.decoder_joint_heterogeneous_graph_encoder is not None:
            output_dict = self.forward_decoder_joint_heterogeneous_graph_encoder(input_dict, output_dict)

        if 'x1' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x1', input_dict, output_dict)
        if 'x2' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x2', input_dict, output_dict)
        if 'x3' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x3', input_dict, output_dict)
        if 'x4' in self.explicit_diffusion_variables:
            output_dict = self.forward_decoder_joint_processing('x4', input_dict, output_dict)

        # Denoising Modules
        if 'x1' in self.explicit_diffusion_variables:
            output_dict = self.forward_x1_decoder_denoiser(input_dict, output_dict)
        if 'x2' in self.explicit_diffusion_variables:
            output_dict = self.forward_x2_decoder_denoiser(input_dict, output_dict)
        if 'x3' in self.explicit_diffusion_variables:
            output_dict = self.forward_x3_decoder_denoiser(input_dict, output_dict)
        if 'x4' in self.explicit_diffusion_variables:
            output_dict = self.forward_x4_decoder_denoiser(input_dict, output_dict)

        # Predicts the clean structure
        output_dict = self.edm_postprocess(output_dict, edm_context)

        return input_dict, output_dict
