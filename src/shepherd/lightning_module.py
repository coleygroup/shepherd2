import math
from pathlib import Path
import traceback

import numpy as np

import torch

import pytorch_lightning as pl

from shepherd.model.model import Model
from shepherd.model.utils.add_virtual_edges_to_edge_index import bond_weighting
from shepherd_score.evaluations.evaluate.pipelines import ConsistencyEvalPipeline


class ExponentialLRScheduleFunc:
    """Picklable learning rate schedule function for exponential decay."""
    def __init__(self, gamma, min_lr_ratio):
        self.gamma = gamma
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step):
        return max(self.gamma ** step, self.min_lr_ratio)


class ChainedExponentialLRScheduleFunc:
    """Exponential decay to min_lr, then jump to chain_exp_lr and decay again.

    LambdaLR multiplies the optimizer's initial lr (phase-1 peak). Phase 2 uses
    absolute chain_exp_lr via lr2_ratio = chain_exp_lr / lr.
    """
    def __init__(self, gamma1, min_ratio1, milestone, lr2_ratio, gamma2, min_ratio2):
        self.gamma1 = gamma1
        self.min_ratio1 = min_ratio1
        self.milestone = milestone
        self.lr2_ratio = lr2_ratio
        self.gamma2 = gamma2
        self.min_ratio2 = min_ratio2

    def __call__(self, step):
        if step < self.milestone:
            return max(self.gamma1 ** step, self.min_ratio1)
        k = step - self.milestone
        return self.lr2_ratio * max(self.gamma2 ** k, self.min_ratio2)


class MultiStageExponentialLRScheduleFunc:
    """Picklable multi-stage exponential LR schedule.

    Generalizes ChainedExponentialLRScheduleFunc to an arbitrary number of decay
    stages. Stage i begins at absolute step ``starts[i]``, starts at peak lr
    ``peaks[i]``, and decays exponentially with rate ``gammas[i]`` toward floor
    ``floors[i]`` (reaching the floor after that stage's decay_steps). The active
    stage at a given step is the last stage whose start <= step. Two stages
    reproduce ChainedExponentialLRScheduleFunc exactly.

    LambdaLR multiplies the optimizer's initial lr (base_lr == peaks[0]), so all
    returned values are absolute-lr / base_lr ratios.
    """
    def __init__(self, base_lr, starts, peaks, gammas, floors):
        self.base_lr = base_lr
        self.starts = starts
        self.peaks = peaks
        self.gammas = gammas
        self.floors = floors

    def __call__(self, step):
        idx = 0
        for i, start in enumerate(self.starts):
            if step >= start:
                idx = i
            else:
                break
        k = step - self.starts[idx]
        lr = max(self.peaks[idx] * (self.gammas[idx] ** k), self.floors[idx])
        return lr / self.base_lr


class CosineWarmupLRScheduleFunc:
    """Picklable learning rate schedule function for cosine warmup."""
    def __init__(self, warmup_steps, min_lr_ratio):
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step):
        return 0.5 * (1 + math.cos(math.pi * (step / self.warmup_steps + 1)))


class LightningModule(pl.LightningModule):

    def __init__(self, params):
        super().__init__()

        self.save_hyperparameters()

        self.params = params

        self.model = Model(params)

        def _compile_model_attr(attr):
            if hasattr(self.model, attr) and getattr(self.model, attr) is not None:
                setattr(self.model, attr, torch.compile(getattr(self.model, attr), dynamic=True))

        # Compile the encoder if compile_encoder is True
        if params.get('compile_encoder', False):
            # DDPOptimizer splits compiled graphs at DDP bucket boundaries. For large EquiformerV3
            # models (≥128 channels) this split lands inside transformer blocks, causing integer
            # module attributes (norm.lmax, so2_linear.num_channels_m0_list, …) to cross submod
            # boundaries as plain Python ints. aot_autograd then fails because it expects FX nodes
            # with .meta["val"]. Disabling DDPOptimizer prevents the split; the minor gradient
            # comm/compute overlap benefit is negligible at ≤4 GPUs.
            import torch._dynamo
            torch._dynamo.config.optimize_ddp = False
            if 'x1' in self.model.explicit_diffusion_variables:
                self.model.x1_decoder_encoder = torch.compile(self.model.x1_decoder_encoder, dynamic=True)
            if 'x2' in self.model.explicit_diffusion_variables:
                self.model.x2_decoder_encoder = torch.compile(self.model.x2_decoder_encoder, dynamic=True)
            if 'x3' in self.model.explicit_diffusion_variables:
                self.model.x3_decoder_encoder = torch.compile(self.model.x3_decoder_encoder, dynamic=True)
            if 'x4' in self.model.explicit_diffusion_variables:
                self.model.x4_decoder_encoder = torch.compile(self.model.x4_decoder_encoder, dynamic=True)
            if hasattr(self.model, 'decoder_joint_heterogeneous_graph_encoder'):
                self.model.decoder_joint_heterogeneous_graph_encoder = torch.compile(
                    self.model.decoder_joint_heterogeneous_graph_encoder, dynamic=True)

        if params.get('compile_joint_processing', False):
            for x_ in self.model.explicit_diffusion_variables:
                _compile_model_attr(f'{x_}_decoder_global_l1_embedding')
                _compile_model_attr(f'{x_}_decoder_equiformer_tensor_product')
                _compile_model_attr(f'{x_}_decoder_global_timestep_embedding')

        if params.get('compile_denoisers', False):
            for x_ in self.model.explicit_diffusion_variables:
                _compile_model_attr(f'{x_}_decoder_denoiser_MLP')
                _compile_model_attr(f'{x_}_decoder_denoiser_E3NN')
            _compile_model_attr('x1_decoder_denoiser_bond_MLP')
            _compile_model_attr('x4_decoder_denoiser_E3NN_direction')

        self.train_x1_denoising = params['training']['train_x1_denoising']
        self.train_x2_denoising = params['training']['train_x2_denoising']
        self.train_x3_denoising = params['training']['train_x3_denoising']
        self.train_x4_denoising = params['training']['train_x4_denoising']

        self.lr = params['training']['lr']
        self.min_lr = params['training']['min_lr']
        self.lr_steps = params['training']['lr_steps']
        self.warmup_steps = params['training'].get('warmup_steps', None)

        # Track OOM-induced batch skips
        self._oom_skip_count = 0
        self._oom_skip_count_consecutive = 0

        # Track NaN-loss batch skips (e.g. from torch.compile numerical issues)
        self._nan_skip_count = 0


    def configure_optimizers(self):
        if self.params['training'].get('optimizer', 'adam').lower() == 'adamw' and self.params['training'].get('weight_decay', 0.) > 0:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr = self.lr,
                betas=(self.params['training'].get('adam_betas', (0.9, 0.999))),
                weight_decay=self.params['training'].get('weight_decay', 0.)
            )
        else:
            optimizer = torch.optim.Adam(
            self.parameters(),
            lr = self.lr,
            betas=(self.params['training'].get('adam_betas', (0.9, 0.999)))
        )

        # exponential lr decay from self.lr to self.min_lr in self.lr_steps steps
        gamma = (self.min_lr / self.lr) ** (1.0 / self.lr_steps)
        min_lr_ratio = self.min_lr / self.lr

        lr_schedule_func = ExponentialLRScheduleFunc(gamma, min_lr_ratio) # pickleable

        if self.warmup_steps is not None:
            warmup_func = CosineWarmupLRScheduleFunc(self.warmup_steps, self.min_lr / self.lr)
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_func)
            scheduler_exp_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_schedule_func)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler_exp_decay],
                milestones=[self.warmup_steps],
            )
        elif self.params['training'].get('cosine_annealing_T_max', None) is not None and self.params['training'].get('cosine_annealing_lr_min', None) is not None:
            # This is usually not touched since T_max is usually ~500k steps
            # but it does not work as expected if you want to use a smaller T_max, so avoid this branch.
            scheduler_1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_schedule_func)
            scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = self.params['training']['cosine_annealing_T_max'],
                eta_min = self.params['training']['cosine_annealing_lr_min'],
            )
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_1, scheduler_2])
        elif self.params['training'].get('chain_exp_lr', None) is not None:
            t = self.params['training']
            lr2 = float(t.get('chain_exp_lr', 3e-4))
            min_lr2 = float(t.get('chain_exp_min_lr', 3e-10))
            steps2 = int(t.get('chain_exp_lr_steps', 1_000_000))
            gamma2 = (min_lr2 / lr2) ** (1.0 / steps2)
            min_ratio2 = min_lr2 / lr2
            chained_lr_schedule_func = ChainedExponentialLRScheduleFunc(
                gamma,
                min_lr_ratio,
                self.lr_steps,
                lr2 / self.lr,
                gamma2,
                min_ratio2,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=chained_lr_schedule_func)
        elif self.params['training'].get('lr_stages', None) is not None:
            # Arbitrary-N chained exponential decays (generalizes chain_exp_lr).
            # Each stage: {'start_step', 'lr', 'min_lr', 'decay_steps'}, ordered by start_step.
            stages = self.params['training']['lr_stages']
            assert stages[0]['start_step'] == 0, "lr_stages: first stage must start at step 0"
            assert float(stages[0]['lr']) == float(self.lr), (
                "lr_stages: first stage 'lr' must equal training['lr'] (optimizer init lr)"
            )
            starts = [int(s['start_step']) for s in stages]
            assert starts == sorted(starts), "lr_stages: stages must be ordered by ascending start_step"
            peaks  = [float(s['lr']) for s in stages]
            floors = [float(s['min_lr']) for s in stages]
            gammas = [(float(s['min_lr']) / float(s['lr'])) ** (1.0 / int(s['decay_steps']))
                      for s in stages]
            multi_stage_func = MultiStageExponentialLRScheduleFunc(
                self.lr, starts, peaks, gammas, floors,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multi_stage_func)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_schedule_func)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
            "name": None,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

    def get_training_input_dict(self, data):

        input_dict = {}

        if self.params['dataset']['compute_x1']:
            x1_data = data['x1']
            try:
                # New data structure for PyG > 2.0.4
                edge_store = data['x1', 'bond', 'x1']
                bond_edge_mask = edge_store.mask
                bond_edge_index = edge_store.edge_index
                bond_edge_x = edge_store.x_forward_noised
                bond_edge_x_noise = edge_store.x_noise
                bond_edge_x_clean = edge_store.x
            except (AttributeError, KeyError):
                # Fallback to old data structure for backward compatibility
                bond_edge_mask = x1_data.bond_edge_mask
                bond_edge_x = x1_data.bond_edge_x_forward_noised
                bond_edge_x_noise = x1_data.bond_edge_x_noise
                bond_edge_x_clean = x1_data.bond_edge_x
                try:
                    bond_edge_index = data['x1', 'x1'].bond_edge_index
                except (AttributeError, KeyError):
                    bond_edge_index = x1_data.bond_edge_index

            input_dict['x1'] = {
                # NOTE: 'pos' and 'x' are the forward-noised structures NOT clean ones
                'decoder': {
                    'pos': x1_data.pos_forward_noised,
                    'x': x1_data.x_forward_noised,
                    'batch': x1_data.batch,
                    'bond_edge_mask': bond_edge_mask,
                    'bond_edge_index': bond_edge_index,
                    'bond_edge_x': bond_edge_x,
                    'timestep': None,
                    'virtual_node_mask': x1_data.virtual_node_mask,
                    'pos_noise': x1_data.pos_noise,
                    'x_noise': x1_data.x_noise,
                    'bond_edge_x_noise': bond_edge_x_noise,
                },
            }

            if self.params.get('scaffold_conditioning', False) and self.params['dataset']['x1'].get('fixed_substructure', {}):
                input_dict['x1']['decoder']['scaffold_task'] = data.scaffold_task # notably not in data['x1']
                input_dict['x1']['decoder']['is_diffused_atom'] = x1_data.is_diffused_atom

            input_dict['x1']['decoder']['sigma'] = x1_data.sigma
            input_dict['x1']['decoder']['x_clean'] = x1_data.x
            input_dict['x1']['decoder']['pos_clean'] = x1_data.pos
            input_dict['x1']['decoder']['bond_edge_x_clean'] = bond_edge_x_clean

        if self.params['dataset']['compute_x2']:
            input_dict['x2'] =  {

                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'pos': data['x2'].pos_forward_noised, # this is the structure after forward-noising
                    'x': data['x2'].x_forward_noised, # currently, this is just one-hot embedding of virtual / real node (equal to data['x2'].x)
                    'batch': data['x2'].batch,

                    'timestep': None,

                    'virtual_node_mask': data['x2'].virtual_node_mask,

                    'pos_noise': data['x2'].pos_noise, # this is the added (gaussian) noise

                },
            }

            input_dict['x2']['decoder']['sigma'] = data['x2'].sigma
            input_dict['x2']['decoder']['pos_clean'] = data['x2'].pos


        if self.params['dataset']['compute_x3']:
            input_dict['x3'] = {

                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'pos': data['x3'].pos_forward_noised, # this is the structure after forward-noising
                    'x': data['x3'].x_forward_noised, # this is the structure after forward-noising
                    'batch': data['x3'].batch,

                    'timestep': None,

                    'virtual_node_mask': data['x3'].virtual_node_mask,

                    'pos_noise': data['x3'].pos_noise, # this is the added (gaussian) noise
                    'x_noise': data['x3'].x_noise, # this is the added (gaussian) noise

                },
            }

            input_dict['x3']['decoder']['sigma'] = data['x3'].sigma
            input_dict['x3']['decoder']['pos_clean'] = data['x3'].pos
            input_dict['x3']['decoder']['x_clean'] = data['x3'].x


        if self.params['dataset']['compute_x4']:
            input_dict['x4'] = {

                # the decoder/denoiser uses the forward-noised structures
                'decoder': {
                    'x': data['x4'].x_forward_noised, # this is the structure after forward-noising
                    'pos': data['x4'].pos_forward_noised, # this is the structure after forward-noising
                    'direction': data['x4'].direction_forward_noised, # this is the structure after forward-noising
                    'batch': data['x4'].batch,

                    'timestep': None,

                    'virtual_node_mask': data['x4'].virtual_node_mask,

                    'direction_noise': data['x4'].direction_noise, # this is the added (gaussian) noise
                    'pos_noise': data['x4'].pos_noise, # this is the added (gaussian) noise
                    'x_noise': data['x4'].x_noise, # this is the added (gaussian) noise

                },
            }

            if self.params.get('scaffold_conditioning', False) and self.params['dataset']['x4'].get('fixed_substructure', {}):
                input_dict['x4']['decoder']['is_diffused_pharm'] = data['x4'].is_diffused_pharm

            input_dict['x4']['decoder']['sigma'] = data['x4'].sigma
            input_dict['x4']['decoder']['x_clean'] = data['x4'].x
            input_dict['x4']['decoder']['pos_clean'] = data['x4'].pos
            input_dict['x4']['decoder']['direction_clean'] = data['x4'].direction

        input_dict['device'] = self.device
        input_dict['dtype'] = torch.float32
        return input_dict


    def forward_training(self, input_dict):
        _, output_dict = self.model.forward(input_dict)
        return output_dict


    def training_step(self, train_batch, batch_idx):
        try:
            data = train_batch
            batch_size = data.molecule_id.shape[0]

            input_dict = self.get_training_input_dict(data)

            output_dict = self.forward_training(input_dict)

            loss = 0.0
            #loss = torch.tensor(0.0, requires_grad=True)
            if self.train_x1_denoising:
                loss_x1, feature_loss_x1, pos_loss_x1, bond_loss_x1 = self.x1_denoising_loss(input_dict, output_dict)
                loss = loss + loss_x1

                batch_size_nodes = (~input_dict['x1']['decoder']['virtual_node_mask']).sum().item()
                batch_size_edges = input_dict['x1']['decoder']['bond_edge_x_noise'].shape[0]

                self.log('train_loss_x1', loss_x1, batch_size = batch_size_nodes)
                self.log('train_pos_loss_x1', pos_loss_x1, batch_size = batch_size_nodes)
                self.log('train_feature_loss_x1', feature_loss_x1, batch_size = batch_size_nodes)
                self.log('train_bond_loss_x1', bond_loss_x1, batch_size = batch_size_edges)

            if self.train_x2_denoising:
                loss_x2 = self.x2_denoising_loss(input_dict, output_dict)
                loss = loss + loss_x2

                batch_size_nodes = (~input_dict['x2']['decoder']['virtual_node_mask']).sum().item()

                self.log('train_loss_x2', loss_x2, batch_size = batch_size_nodes)

            if self.train_x3_denoising:
                loss_x3, feature_loss_x3, pos_loss_x3 = self.x3_denoising_loss(input_dict, output_dict)
                loss = loss + loss_x3

                batch_size_nodes = (~input_dict['x3']['decoder']['virtual_node_mask']).sum().item()

                self.log('train_loss_x3', loss_x3, batch_size = batch_size_nodes)
                self.log('train_pos_loss_x3', pos_loss_x3, batch_size = batch_size_nodes)
                self.log('train_feature_loss_x3', feature_loss_x3, batch_size = batch_size_nodes)

            if self.train_x4_denoising:
                loss_x4, feature_loss_x4, pos_loss_x4, direction_loss_x4 = self.x4_denoising_loss(input_dict, output_dict)
                loss = loss + loss_x4

                batch_size_nodes = (~input_dict['x4']['decoder']['virtual_node_mask']).sum().item()

                self.log('train_loss_x4', loss_x4, batch_size = batch_size_nodes)
                self.log('train_pos_loss_x4', pos_loss_x4, batch_size = batch_size_nodes)
                self.log('train_direction_loss_x4', direction_loss_x4, batch_size = batch_size_nodes)
                self.log('train_feature_loss_x4', feature_loss_x4, batch_size = batch_size_nodes)

            if not torch.isfinite(loss):
                rank = self.trainer.global_rank if self.trainer is not None else 0
                self._nan_skip_count += 1
                print(
                    f"[rank{rank}] Non-finite loss={loss.item():.4g} at batch {batch_idx}, "
                    f"step {self.global_step} (nan_skips={self._nan_skip_count}). Skipping batch.",
                    flush=True,
                )
                self.log('train_nan_skips', float(self._nan_skip_count), on_step=True)
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            self.log('train_loss', loss, batch_size = batch_size)
            self._oom_skip_count_consecutive = 0
            return loss

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise

            rank = self.trainer.global_rank if self.trainer is not None else 0
            if self.trainer is not None and self.trainer.world_size > 1:
                print(
                    f"[rank{rank}] GPU OOM at batch {batch_idx}, step {self.global_step}. "
                    "Failing this DDP attempt so all ranks restart together.",
                    flush=True,
                )
                print(f"[rank{rank}] Error: {str(e)}", flush=True)
                try:
                    Path(".shepherd_recoverable_error").touch()
                except Exception:
                    pass
                raise

            if self.trainer.is_global_zero:
                print(f"GPU OOM Error at batch {batch_idx}, step {self.global_step}. Skipping batch.")
                print(f"Error: {str(e)}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._oom_skip_count += 1
            self._oom_skip_count_consecutive += 1
            self.log('train_oom_skips', float(self._oom_skip_count), on_step=True, sync_dist=True)

            if self._oom_skip_count_consecutive > 25:
                raise RuntimeError("Too many consecutive OOM errors. Stopping training.")

            return torch.tensor(0.0, device=self.device, requires_grad=True) # Return zero loss to avoid breaking the training loop


    def x1_denoising_loss(self, input_dict, output_dict):

        mask = ~input_dict['x1']['decoder']['virtual_node_mask'] # kept for backwards compatibility
        if self.params.get('scaffold_conditioning', False):
            mask = input_dict['x1']['decoder']['is_diffused_atom'] # False for scaffolding atoms and virtual nodes

        if mask.sum() == 0:
            denoiser_outputs = output_dict['x1']['decoder']['denoiser']
            zero = 0.0 * (denoiser_outputs['x_out'].sum() + denoiser_outputs['pos_out'].sum())
            if self.model.x1_bond_diffusion:
                zero = zero + 0.0 * denoiser_outputs['bond_edge_x_out'].sum()
            return zero, zero, zero, zero

        loss_weight_pos = self.model.edm_noise_schedule_pos.get_loss_weight(input_dict['x1']['decoder']['sigma'])
        loss_weight_x = self.model.edm_noise_schedule_atom_one_hot.get_loss_weight(input_dict['x1']['decoder']['sigma'])
        # sigma is per-batch in EDM and masked out per-node below so need to expand to per-node
        loss_weight_pos = loss_weight_pos[input_dict['x1']['decoder']['batch']]
        loss_weight_x = loss_weight_x[input_dict['x1']['decoder']['batch']]

        # Loss is w.r.t. clean structure not noise
        pos_loss = torch.mean(
            loss_weight_pos[mask] * (
                (input_dict['x1']['decoder']['pos_clean'] - output_dict['x1']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
            )
        )

        feature_loss = torch.mean(
            loss_weight_x[mask] * (
                (input_dict['x1']['decoder']['x_clean'] - output_dict['x1']['decoder']['denoiser']['x_out'])[mask] ** 2.0
            )
        )

        bond_loss = torch.zeros_like(feature_loss)
        if self.model.x1_bond_diffusion:

            input_dict['x1']['decoder']['bond_edge_index']

            pred_noise = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out']
            bond_mask = input_dict['x1']['decoder']['bond_edge_mask'] # indicates real-bond (True) or non-bond (False)

            # weighting contributions from real-bonds and non-bonds equally
                # otherwise, the loss from the non-bonds will overwhelm the loss from the real-bonds
            loss_weight_bond_batch = self.model.edm_noise_schedule_bond.get_loss_weight(input_dict['x1']['decoder']['sigma'])
            loss_weight_bond_perbond = bond_weighting(
                input_tensor=loss_weight_bond_batch,
                graph_batch_index=input_dict['x1']['decoder']['batch'],
                virtual_node_x1=self.params['dataset']['x1']['add_virtual_node']
            )
            diff_clean_pred = input_dict['x1']['decoder']['bond_edge_x_clean'] - pred_noise
            bond_loss = ( # labeled "pred_noise" for legacy reasons, but in edm framework, we predict the clean structure
                torch.mean(loss_weight_bond_perbond[bond_mask].unsqueeze(-1) * (diff_clean_pred[bond_mask] ** 2.0)) +
                torch.mean(loss_weight_bond_perbond[~bond_mask].unsqueeze(-1) * (diff_clean_pred[~bond_mask] ** 2.0))
            )*0.5

        if self.params['training'].get('loss', {}).get('x1', {}):
            pos_loss = pos_loss * self.params['training']['loss']['x1'].get('pos_weight', 1.0)
            feature_loss = feature_loss * self.params['training']['loss']['x1'].get('feature_weight', 1.0)
            bond_loss = bond_loss * self.params['training']['loss']['x1'].get('bond_weight', 1.0)

        loss = pos_loss + feature_loss + bond_loss

        return loss, feature_loss, pos_loss, bond_loss

    def x2_denoising_loss(self, input_dict, output_dict):

        mask = ~input_dict['x2']['decoder']['virtual_node_mask']
        loss_weight_pos = self.model.edm_noise_schedule_pos.get_loss_weight(input_dict['x2']['decoder']['sigma'])
        pos_loss = torch.mean(
            loss_weight_pos[mask] * (
                (input_dict['x2']['decoder']['pos_clean'] - output_dict['x2']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
            )
        )

        if self.params['training'].get('loss', {}).get('x2', {}):
            pos_loss = pos_loss * self.params['training']['loss']['x2'].get('pos_weight', 1.0)

        loss = pos_loss

        return loss


    def x3_denoising_loss(self, input_dict, output_dict):

        mask = ~input_dict['x3']['decoder']['virtual_node_mask']
        loss_weight_pos = self.model.edm_noise_schedule_pos.get_loss_weight(input_dict['x3']['decoder']['sigma'])
        loss_weight_pos = loss_weight_pos[input_dict['x3']['decoder']['batch']] # expand to per node
        loss_weight_x = self.model.edm_noise_schedule_esp.get_loss_weight(input_dict['x3']['decoder']['sigma'])
        loss_weight_x = loss_weight_x[input_dict['x3']['decoder']['batch']]
        pos_loss = torch.mean(
            loss_weight_pos[mask] * (
                (input_dict['x3']['decoder']['pos_clean'] - output_dict['x3']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
            )
        )
        # important: must squeeze the loss weight to (N,) before multiplying. since the x3 feature outputs are (N,)
        # Not doing so leads to mean over broadcassted (N,N) matrix which is nonsense
        feature_loss = torch.mean(
            loss_weight_x[mask].squeeze(-1) * (
                (input_dict['x3']['decoder']['x_clean'] - output_dict['x3']['decoder']['denoiser']['x_out'])[mask] ** 2.0
            )
        )

        if self.params['training'].get('loss', {}).get('x3', {}):
            pos_loss = pos_loss * self.params['training']['loss']['x3'].get('pos_weight', 1.0)
            feature_loss = feature_loss * self.params['training']['loss']['x3'].get('feature_weight', 1.0)

        loss = feature_loss + pos_loss

        return loss, feature_loss, pos_loss


    def x4_denoising_loss(self, input_dict, output_dict):

        mask = ~input_dict['x4']['decoder']['virtual_node_mask']
        if self.params.get('scaffold_conditioning', False) and self.params['dataset']['x4'].get('fixed_substructure', {}):
            mask = input_dict['x4']['decoder']['is_diffused_pharm']
        if sum(mask) == 0:
            denoiser_outputs = output_dict['x4']['decoder']['denoiser']
            # Keep a zero-valued autograd path through x4 outputs so DDP sees the
            # same used parameters on every rank, even when this rank has no
            # diffused pharmacophores in the batch.
            zero = 0.0 * (
                denoiser_outputs['x_out'].sum()
                + denoiser_outputs['pos_out'].sum()
                + denoiser_outputs['direction_out'].sum()
            )
            return zero, zero, zero, zero

        loss_weight_pos = self.model.edm_noise_schedule_pos.get_loss_weight(input_dict['x4']['decoder']['sigma'])
        loss_weight_x = self.model.edm_noise_schedule_atom_one_hot.get_loss_weight(input_dict['x4']['decoder']['sigma'])
        loss_weight_direction = self.model.edm_noise_schedule_direction.get_loss_weight(input_dict['x4']['decoder']['sigma'])
        # sigma is per-batch in EDM and masked out per-node below so need to expand to per-node
        loss_weight_pos = loss_weight_pos[input_dict['x4']['decoder']['batch']]
        loss_weight_x = loss_weight_x[input_dict['x4']['decoder']['batch']]
        loss_weight_direction = loss_weight_direction[input_dict['x4']['decoder']['batch']]

        pos_loss = torch.mean(
            loss_weight_pos[mask] * (
                (input_dict['x4']['decoder']['pos_clean'] - output_dict['x4']['decoder']['denoiser']['pos_out'])[mask] ** 2.0
            )
        )
        feature_loss = torch.mean(
            loss_weight_x[mask] * (
                (input_dict['x4']['decoder']['x_clean'] - output_dict['x4']['decoder']['denoiser']['x_out'].squeeze())[mask] ** 2.0
            )
        )
        direction_loss = torch.mean(
            loss_weight_direction[mask] * (
                (input_dict['x4']['decoder']['direction_clean'] - output_dict['x4']['decoder']['denoiser']['direction_out'])[mask] ** 2.0
            )
        )

        if self.params['training'].get('loss', {}).get('x4', {}):
            pos_loss = pos_loss * self.params['training']['loss']['x4'].get('pos_weight', 1.0)
            feature_loss = feature_loss * self.params['training']['loss']['x4'].get('feature_weight', 1.0)
            direction_loss = direction_loss * self.params['training']['loss']['x4'].get('direction_weight', 1.0)

        loss = feature_loss + pos_loss + direction_loss

        return loss, feature_loss, pos_loss, direction_loss


    def _log_failed_validation_batch(self, batch_size: int, is_distributed: bool):
        """
        Helper function to log zeros for all validation metrics when a batch fails.

        Args:
            batch_size: Size of the validation batch
            is_distributed: Whether training is distributed across multiple GPUs
        """
        self.log("valid/num_valid", 0.0, sync_dist=True, reduce_fx="sum", batch_size=batch_size)
        self.log("valid/num_valid_post_opt", 0.0, sync_dist=True, reduce_fx="sum", batch_size=batch_size)
        self.log("valid/num_total_gen", 0.0, sync_dist=True, reduce_fx="sum", batch_size=batch_size)

        if is_distributed:
            self.log("valid/frac_unique", 0.0, sync_dist=True, reduce_fx="mean", batch_size=batch_size)
            self.log("valid/frac_unique_post_opt", 0.0, sync_dist=True, reduce_fx="mean", batch_size=batch_size)
        else:
            self.log("valid/frac_unique", 0.0, sync_dist=True, batch_size=batch_size)
            self.log("valid/frac_unique_post_opt", 0.0, sync_dist=True, batch_size=batch_size)
            self.log("valid/frac_valid", 0.0, sync_dist=True, batch_size=batch_size)

        self.log("valid/sims_surf_consistent", 0.0, sync_dist=True, batch_size=batch_size)
        self.log("valid/sims_esp_consistent", 0.0, sync_dist=True, batch_size=batch_size)
        self.log("valid/sims_pharm_consistent", 0.0, sync_dist=True, batch_size=batch_size)
        self.log('valid/rmsd', 0.0, sync_dist=True, batch_size=batch_size)


    def _clean_samples(self, gen_structs, skip_condition: bool = False):
        generated_mols = []
        for gen_struct in gen_structs:
            atoms = gen_struct['x1']['atoms']
            dummy_atom_inds = np.where(atoms == 0)[0]
            atoms = np.delete(atoms, dummy_atom_inds)
            positions = np.delete(gen_struct['x1']['positions'], dummy_atom_inds, axis = 0)
            generated_mols.append((atoms, positions))

        surf_points = None
        surf_esp = None
        pharm_feats = None

        if not skip_condition:
            if self.params['training']['train_x2_denoising']:
                surf_points = [gen_struct['x2']['positions'] for gen_struct in gen_structs]

            if self.params['training']['train_x3_denoising']:
                surf_points = [gen_struct['x3']['positions'] for gen_struct in gen_structs]
                surf_esp = [gen_struct['x3']['charges'].reshape(-1) for gen_struct in gen_structs]

            if self.params['training']['train_x4_denoising']:
                pharm_feats = []
                for gen_struct in gen_structs:
                    pharm_types = gen_struct['x4']['types']
                    pharm_pos = gen_struct['x4']['positions']
                    pharm_direction = gen_struct['x4']['directions']
                    if self.params['dataset']['x4'].get('include_dummy_pharm', False):
                        # -2 since virtual node is already subtracted, and need to account for 0 indexing
                        dummy_pharm_inds = np.where(pharm_types == self.params['dataset']['x4']['max_node_types'] - 2)[0]
                        pharm_types = np.delete(pharm_types, dummy_pharm_inds)
                        pharm_pos = np.delete(pharm_pos, dummy_pharm_inds, axis = 0)
                        pharm_direction = np.delete(pharm_direction, dummy_pharm_inds, axis = 0)
                    pharm_feats.append((pharm_types, pharm_pos, pharm_direction))

        return generated_mols, surf_points, surf_esp, pharm_feats


    def _unconditional_validation(self, val_batch, is_distributed):

        from shepherd.inference import generate

        val_batch, distributions = val_batch
        batch_size = val_batch.size()[0]
        prob = distributions.sum(axis=1) / distributions.sum()
        n_atoms = int(np.random.multinomial(1, pvals=prob).argmax())

        prob = distributions[n_atoms,:]
        if prob.sum() > 0.0:
            prob = prob / prob.sum()
            N_x4 = int(np.random.multinomial(1, pvals=prob).argmax())
        else:
            prob = distributions.sum(axis = 0)
            prob = prob / prob.sum()
            N_x4 = int(np.random.multinomial(1, pvals=prob).argmax())

        try:
            gen_structs = generate(
                self,
                batch_size=batch_size,
                N_x1=n_atoms,
                N_x4=N_x4,
                verbose=False,
                num_steps = 400,
                use_stochastic=False,
                sigma_max=3.0,
                S_churn=20.,
                rho=7.0,
                use_2nd_order_correction=False,
                shepherd_pred=True,
                early_stop_edm=0.9,
            )
        except Exception:
            print(traceback.format_exc())
            self._log_failed_validation_batch(batch_size, is_distributed)
            return

        generated_mols, surf_points, surf_esp, pharm_feats = self._clean_samples(gen_structs, skip_condition=False)

        # Clear cache before heavy computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        consis_eval_pipe = ConsistencyEvalPipeline(
            generated_mols=generated_mols,
            generated_surf_points=surf_points,
            generated_surf_esp=surf_esp,
            generated_pharm_feats=pharm_feats,
            pharm_multi_vector=False,
            solvent=self.params['dataset']['solvent'] if 'solvent' in self.params['dataset'] else None,
            probe_radius=1.2,
            random_molblock_charges=None
            # num_processes=self.model.params['training']['num_workers']
        )

        try:
            consis_eval_pipe.evaluate(num_workers=self.model.params['training']['num_workers'], num_processes=1, verbose=False)
        except RuntimeError as e:
            if "CUDA" in str(e) and ("out of memory" in str(e) or "CUBLAS" in str(e)):
                print(f"CUDA OOM during validation evaluation: {e}")
                # Clear cache and try to recover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Log zeros and continue
                self._log_failed_validation_batch(batch_size, is_distributed)
                return
            else:
                raise

        # can contain nan
        sims_surf_consistent = consis_eval_pipe.sims_surf_consistent_relax_optimal
        sims_esp_consistent = consis_eval_pipe.sims_esp_consistent_relax_optimal
        sims_pharm_consistent = consis_eval_pipe.sims_pharm_consistent_relax_optimal

        num_unique = len(set([smi for smi in consis_eval_pipe.smiles if smi is not None]))
        num_unique_post_opt = len(set([smi for smi in consis_eval_pipe.smiles_post_opt if smi is not None]))
        num_valid = consis_eval_pipe.num_valid
        num_valid_post_opt = consis_eval_pipe.num_valid_post_opt
        num_total_gen = consis_eval_pipe.num_generated_mols
        rmsds = consis_eval_pipe.rmsds

        self.log('valid/num_valid', num_valid, sync_dist=True, reduce_fx="sum", batch_size=batch_size)
        self.log('valid/num_valid_post_opt', num_valid_post_opt, sync_dist=True, reduce_fx="sum", batch_size=batch_size)
        self.log('valid/num_total_gen', num_total_gen, sync_dist=True, reduce_fx="sum", batch_size=batch_size)
        if num_valid_post_opt > 0:
            self.log("valid/sims_surf_consistent", np.nansum(sims_surf_consistent)/num_valid_post_opt, sync_dist=True, batch_size=batch_size)
            self.log("valid/sims_esp_consistent", np.nansum(sims_esp_consistent)/num_valid_post_opt, sync_dist=True, batch_size=batch_size)
            self.log("valid/sims_pharm_consistent", np.nansum(sims_pharm_consistent)/num_valid_post_opt, sync_dist=True, batch_size=batch_size)
            self.log('valid/rmsd', np.nansum(rmsds)/num_valid_post_opt, sync_dist=True, batch_size=batch_size)
        else:
            self.log("valid/sims_surf_consistent", 0.0, sync_dist=True, batch_size=batch_size)
            self.log("valid/sims_esp_consistent", 0.0, sync_dist=True, batch_size=batch_size)
            self.log("valid/sims_pharm_consistent", 0.0, sync_dist=True, batch_size=batch_size)
            self.log('valid/rmsd', np.nan, sync_dist=True, batch_size=batch_size)

        # Compute local fraction unique on each GPU
        if num_valid > 0:
            fraction_unique = num_unique / num_valid
        else:
            fraction_unique = 0.0

        if num_valid_post_opt > 0:
            fraction_unique_post_opt = num_unique_post_opt / num_valid_post_opt
        else:
            fraction_unique_post_opt = 0.0

        # In distributed setting, average the fraction unique across GPUs
        if is_distributed:
            self.log("valid/frac_unique", fraction_unique, sync_dist=True, reduce_fx="mean", batch_size=batch_size)
            self.log("valid/frac_unique_post_opt", fraction_unique_post_opt, sync_dist=True, reduce_fx="mean", batch_size=batch_size)
        else:
            self.log("valid/frac_valid", num_valid/num_total_gen, sync_dist=True, batch_size=batch_size)
            self.log("valid/frac_unique", fraction_unique, sync_dist=True, batch_size=batch_size)
            self.log("valid/frac_unique_post_opt", fraction_unique_post_opt, sync_dist=True, batch_size=batch_size)

    def validation_step(self, val_batch, batch_idx):
        """Check the consistency of generated molecules."""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        is_distributed = self.trainer.world_size > 1

        self._unconditional_validation(val_batch, is_distributed)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        is_distributed = self.trainer.world_size > 1

        global_valid_post_opt = self.trainer.callback_metrics.get("valid/num_valid_post_opt", 0.0)
        global_valid_post_opt = float(global_valid_post_opt)

        if is_distributed:
            global_valid_count = self.trainer.callback_metrics.get("valid/num_valid", 0.0)
            global_total_count = self.trainer.callback_metrics.get("valid/num_total_gen", 0.0)
            global_valid_count_post_opt = self.trainer.callback_metrics.get("valid/num_valid_post_opt", 0.0)

            if global_total_count > 0:
                self.log("valid/frac_valid", global_valid_count / global_total_count, sync_dist=True)
                fraction_valid_post_opt = global_valid_count_post_opt / global_total_count
                self.log("valid/frac_valid_post_opt", fraction_valid_post_opt, sync_dist=True)

        if is_distributed and self.params['training'].get('validation_conditional', False):
            global_valid_count_cond = self.trainer.callback_metrics.get("valid-cond/num_valid", 0.0)
            global_total_count_cond = self.trainer.callback_metrics.get("valid-cond/num_total_gen", 0.0)
            global_valid_count_cond_post_opt = self.trainer.callback_metrics.get("valid-cond/num_valid_post_opt", 0.0)

            if global_total_count_cond > 0:
                self.log("valid-cond/frac_valid", global_valid_count_cond / global_total_count_cond, sync_dist=True)
                fraction_valid_cond_post_opt = global_valid_count_cond_post_opt / global_total_count_cond
                self.log("valid-cond/frac_valid_post_opt", fraction_valid_cond_post_opt, sync_dist=True)
