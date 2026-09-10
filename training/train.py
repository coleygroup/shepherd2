"""Train a model from an experiment config and random seed."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from shepherd.lightning_module import LightningModule
from shepherd.ema import ShepherdOnlineEMA
from shepherd.data_utils.hdf5_dataset import HeteroDatasetHDF5
from shepherd.data_utils.datamodule import ResumableDataModule
from training.parameters.loader import load_params


VALIDATION_DISTRIBUTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "conformers"
    / "distributions"
    / "atom_pharm_count.npz"
)


class SaveWandbRunIDCallback(pl.Callback):
    """
    Callback to save the wandb run ID after initialization.
    """
    def __init__(self, wandb_run_id_file, initial_wandb_run_id):
        self.wandb_run_id_file = wandb_run_id_file
        self.initial_wandb_run_id = initial_wandb_run_id

    def on_train_start(self, trainer, pl_module):
        # Only save if we started a new run (not resuming)
        if self.initial_wandb_run_id is None:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    # Use version property which is the run ID
                    run_id = logger.version
                    if run_id is not None:
                        with open(self.wandb_run_id_file, "w") as f:
                            f.write(run_id)
                        print(f"Starting new wandb run with ID {run_id} saved to {self.wandb_run_id_file}")
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to an experiment YAML under training/parameters/ (e.g. experiments/<name>.yaml)")
    parser.add_argument("seed", type=int)
    parser.add_argument("--data-dir", type=Path, help="Directory containing the training .h5 files; overrides `data_dir` in the config")
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")

    seed_everything(seed=args.seed, workers=True)

    params = load_params(args.config)
    for compile_key in (
        "compile_encoder",
        "compile_joint_processing",
        "compile_denoisers",
    ):
        if compile_key in params.get("training", {}):
            params[compile_key] = params["training"][compile_key]

    dataset_params = {
        "edm_schedule_dict": params.get("edm", {}),
        "explicit_hydrogens": params["dataset"]["explicit_hydrogens"],
        "formal_charge_diffusion": params["x1_formal_charge_diffusion"],
        "x1": params["dataset"]["compute_x1"],
        "x2": params["dataset"]["compute_x2"],
        "x3": params["dataset"]["compute_x3"],
        "x4": params["dataset"]["compute_x4"],
        "recenter_x1": params["dataset"]["x1"]["recenter"],
        "add_virtual_node_x1": params["dataset"]["x1"]["add_virtual_node"],
        "remove_noise_COM_x1": params["dataset"]["x1"]["remove_noise_COM"],
        "atom_types_x1": params["dataset"]["x1"]["atom_types"],
        "charge_types_x1": params["dataset"]["x1"]["charge_types"],
        "bond_types_x1": params["dataset"]["x1"]["bond_types"],
        "scale_atom_features_x1": params["dataset"]["x1"]["scale_atom_features"],
        "scale_bond_features_x1": params["dataset"]["x1"]["scale_bond_features"],
        "recenter_x2": params["dataset"]["x2"]["recenter"],
        "add_virtual_node_x2": params["dataset"]["x2"]["add_virtual_node"],
        "remove_noise_COM_x2": params["dataset"]["x2"]["remove_noise_COM"],
        "num_points_x2": params["dataset"]["x2"]["num_points"],
        "recenter_x3": params["dataset"]["x3"]["recenter"],
        "add_virtual_node_x3": params["dataset"]["x3"]["add_virtual_node"],
        "remove_noise_COM_x3": params["dataset"]["x3"]["remove_noise_COM"],
        "num_points_x3": params["dataset"]["x3"]["num_points"],
        "scale_node_features_x3": params["dataset"]["x3"]["scale_node_features"],
        "recenter_x4": params["dataset"]["x4"]["recenter"],
        "add_virtual_node_x4": params["dataset"]["x4"]["add_virtual_node"],
        "remove_noise_COM_x4": params["dataset"]["x4"]["remove_noise_COM"],
        "max_node_types_x4": params["dataset"]["x4"]["max_node_types"],
        "scale_node_features_x4": params["dataset"]["x4"]["scale_node_features"],
        "scale_vector_features_x4": params["dataset"]["x4"]["scale_vector_features"],
        "multivectors": params["dataset"]["x4"]["multivectors"],
        "check_accessibility": params["dataset"]["x4"]["check_accessibility"],
        "probe_radius": params["dataset"]["probe_radius"],  # for x2 and x3
        "include_dummy_atoms": params["dataset"]["x1"].get("include_dummy_atoms", False),
        "include_dummy_pharm": params["dataset"]["x4"].get("include_dummy_pharm", False),
        "scale_point_cloud": params["dataset"].get("scale_point_cloud", 1.0),
        "scaffold_conditioning": params.get("scaffold_conditioning", False),
        "fixed_substructure_params_x1": params["dataset"]["x1"].get("fixed_substructure", {}),
        "fixed_substructure_params_x4": params["dataset"]["x4"].get("fixed_substructure", {}),
    }

    data_dir = args.data_dir or params.get("data_dir")
    if data_dir is None:
        parser.error("Set `data_dir` in the config or pass --data-dir")

    if params["data"] == "MOSES_aq_HDF5":
        data_dir = Path(data_dir).expanduser()
        distribution_key = "moses_aq"
    else:
        raise ValueError(f"Invalid data type: {params['data']}")

    with np.load(VALIDATION_DISTRIBUTIONS_PATH) as distribution_archive:
        distributions = distribution_archive[distribution_key]

    train_dataset = HeteroDatasetHDF5(
        hdf5_data_dir=data_dir,
        **dataset_params,
    )

    data_module = ResumableDataModule(
        train_dataset=train_dataset,
        val_distributions=distributions,
        batch_size=params["training"]["batch_size"],
        num_workers=params["training"]["num_workers"],
        seed=args.seed,
        multiprocessing_spawn=params["training"]["multiprocessing_spawn"],
        persistent_workers=params["training"].get("persistent_workers", True),
        prefetch_factor=params["training"].get("prefetch_factor", 2),
        pin_memory=True,  # Speed up GPU transfers
    )

    output_dir = Path("jobs") / params["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = output_dir / "last.ckpt"
    if resume_checkpoint.is_file():
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f%z")
        backup_path = output_dir / f"last.backup-{timestamp}.ckpt"
        shutil.copy2(resume_checkpoint, backup_path)
        print(f"Backed up {resume_checkpoint} to {backup_path}")
        print(f"Resuming training from {resume_checkpoint}")
    else:
        resume_checkpoint = None

    gradient_clip_val = params["training"]["gradient_clip_val"]
    accumulate_grad_batches = params["training"]["accumulate_grad_batches"]
    cuda_available = torch.cuda.is_available()

    tqdm_callback = TQDMProgressBar(refresh_rate=50)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        dirpath=output_dir,
        every_n_train_steps=params["training"]["log_every_n_steps"],
    )
    callbacks_list = [checkpoint_callback, tqdm_callback]
    if params["training"].get("save_checkpoint_every_epoch", False):
        epoch_checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,  # save all epochs
            save_last=False,
            dirpath=output_dir,
            filename="epoch_{epoch:04d}",
            every_n_epochs=1,
        )
        callbacks_list.append(epoch_checkpoint_callback)
        print(f"Epoch checkpoint saving enabled: will save weights every epoch to {output_dir}")
    if params["training"].get("save_checkpoint_every_n_steps", 20_000) is not None:
        step_checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,
            save_last=False,
            dirpath=output_dir,
            filename="step_{step:09d}",
            every_n_train_steps=params["training"].get("save_checkpoint_every_n_steps", 20_000),
        )
        callbacks_list.append(step_checkpoint_callback)
        print(
            f"Step checkpoint saving enabled: will save weights every {params['training'].get('save_checkpoint_every_n_steps')} steps to {output_dir}"
        )

    loggers = []
    if params["training"].get("wandb_project", None):
        wandb_run_id_file = output_dir / "wandb_run_id.txt"
        if wandb_run_id_file.exists():
            with open(wandb_run_id_file, "r") as f:
                wandb_run_id = f.read().strip()
            print(f"Wandb run ID: {wandb_run_id}")
        else:
            wandb_run_id = None

        wandb_logger = WandbLogger(
            project=params["training"].get("wandb_project", None),
            save_dir=output_dir,
            name=params["training"].get("wandb_name", None),
            id=wandb_run_id,
            resume="allow",
        )
        loggers.append(wandb_logger)

        if wandb_run_id is None:
            callbacks_list.append(SaveWandbRunIDCallback(wandb_run_id_file, wandb_run_id))

    ema_cfg = params["training"].get("ema", {})
    if ema_cfg.get("enabled", False) and ema_cfg.get("online", False):
        callbacks_list.append(
            ShepherdOnlineEMA(
                decay=float(ema_cfg.get("online_decay", 0.999)),
                use_for_validation=ema_cfg.get("use_for_validation", False),
                update_starting_at_step=ema_cfg.get("update_starting_at_step"),
            )
        )

    # DDP: `find_unused_parameters=True` is slower; prefer fixing the graph (see scaffold / masked-loss paths).
    # Override with training['ddp_find_unused_parameters'] (defaults to False).
    _ddp_find_unused = params["training"].get("ddp_find_unused_parameters", False)
    _ddp_strategy = (
        DDPStrategy(find_unused_parameters=_ddp_find_unused)
        if (params["training"]["num_gpus"] > 1 and cuda_available)
        else "auto"
    )

    trainer = pl.Trainer(
        callbacks=callbacks_list,
        logger=loggers or None,
        default_root_dir=output_dir,
        accelerator="gpu" if (params["training"]["num_gpus"] >= 1 and cuda_available) else "cpu",
        max_epochs=50,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=params["training"]["log_every_n_steps"],
        devices=params["training"]["num_gpus"] if cuda_available else "auto",
        strategy=_ddp_strategy,
        precision=32,
        limit_val_batches=1,  # 1 batch per validation loop
        val_check_interval=params["training"].get("val_check_interval", 1.0),  # float : epochs; int : batches
        num_sanity_val_steps=0,
    )

    model_pl = LightningModule(params)
    print(sum(p.numel() for p in model_pl.parameters() if p.requires_grad))

    trainer.fit(
        model_pl,
        datamodule=data_module,
        ckpt_path=resume_checkpoint,
        weights_only=False,
    )


if __name__ == "__main__":
    main()
