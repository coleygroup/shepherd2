import torch
import torch_geometric
import pytorch_lightning as pl

class SingleBatchDataset(torch.utils.data.Dataset):
    """
    This is a dummy dataset for validation.
    """
    def __init__(self, total_batch_size, distributions):
        super().__init__()
        self.data = torch.empty((total_batch_size, 1)).to(torch.int8)  # Fake sample batch
        self.distributions = distributions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.distributions


def collate_fn(batch):
    data_items = [item[0] for item in batch]  # Extract data tensors
    data_batch = torch.stack(data_items)  # Stack individual data samples
    distributions = batch[0][1]  # Get distributions from batch
    return data_batch, distributions


class ConditionalSingleBatchDataset(torch.utils.data.Dataset):
    """
    This is a dummy dataset for conditional validation.
    """
    def __init__(self, total_batch_size, distributions):
        super().__init__()
        self.data = torch.empty((total_batch_size, 1)).to(torch.int8)  # Fake sample batch
        self.distributions = distributions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.distributions


class StatefulPyGDataLoader(torch_geometric.loader.DataLoader):
    """
    A simple wrapper for torch_geometric.loader.DataLoader
    that explicitly implements state_dict and load_state_dict.

    This is necessary to pass PyTorch Lightning's `_is_resumable`
    check in single-GPU (non-DDP) mode. The Trainer will call
    the DataModule's hooks, but it *checks* the loader object.
    """
    def __init__(self, *args, **kwargs):
        # We must have a generator for this to be stateful
        assert 'generator' in kwargs, "StatefulPyGDataLoader requires a generator."
        self._stateful_generator = kwargs['generator']
        super().__init__(*args, **kwargs)

    def state_dict(self):
        """
        Advertise the generator's state as our own.
        """
        return {
            'generator_state': self._stateful_generator.get_state()
        }

    def load_state_dict(self, state_dict):
        """
        This will not actually be called by the Trainer (it calls
        the DataModule's hook), but we implement it to
        pass the `is_overridden` check.
        """
        self._stateful_generator.set_state(state_dict['generator_state'])


class ResumableDataModule(pl.LightningDataModule):
    """
    A simple, robust, and resumable DataModule.

    It leverages the Trainer's state and hooks, avoiding
    redundant state tracking.

    1. `state_dict` / `load_state_dict` save/load the generator for perfect mid-epoch resumption.
    2. `on_train_epoch_start` re-seeds the generator using `self.trainer.current_epoch` for new,
        reproducible shuffles on new epochs.
    """

    def __init__(
        self,
        train_dataset,
        val_distributions = None,
        batch_size: int = 24,
        num_workers: int = 4,
        seed: int = 42,
        multiprocessing_spawn: bool = True,
        worker_init_fn=None,
        **dataloader_kwargs
    ):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_distributions = val_distributions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_seed = seed
        self.multiprocessing_spawn = multiprocessing_spawn
        self.worker_init_fn_custom = worker_init_fn
        self.dataloader_kwargs = dataloader_kwargs

        self.generator = torch.Generator()

        self.worker_seed = 0 # Will be set in hooks

    def on_train_epoch_start(self):
        """
        Called by the Trainer at the start of a *new* epoch.
        (NOT called when resuming mid-epoch).

        This is the perfect place to re-seed our generator
        for a new, reproducible shuffle.
        """
        # Use the Trainer's epoch to create a deterministic seed
        new_seed = self.base_seed + self.trainer.current_epoch
        self.generator.manual_seed(new_seed)
        self.worker_seed = self.generator.initial_seed()

    def train_dataloader(self):
        """Create the training DataLoader."""

        mp_context = torch.multiprocessing.get_context("spawn") if self.multiprocessing_spawn else None

        loader = StatefulPyGDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            multiprocessing_context=mp_context,
            worker_init_fn=self._worker_init_fn,
            generator=self.generator,  # Pass the managed generator
            **self.dataloader_kwargs
        )

        return loader

    def _worker_init_fn(self, worker_id: int):
        """Initialize workers with deterministic seeds."""
        # Dataloader workers inherit OMP/MKL thread env vars from the parent rank.
        # Keep each worker single-threaded so worker pools do not oversubscribe CPUs.
        torch.set_num_threads(1)

        worker_seed = (self.worker_seed + worker_id) % (2**32)
        torch.manual_seed(worker_seed)

        if self.worker_init_fn_custom is not None:
            self.worker_init_fn_custom(worker_id)

    def state_dict(self) -> dict:
        """
        Save the *only* state we care about: the generator.
        The Trainer handles epoch and batch_idx.
        """
        return {'generator_state': self.generator.get_state()}

    def load_state_dict(self, state_dict: dict):
        """
        Restore the generator state for seamless resumption.
        """
        self.generator.set_state(state_dict['generator_state'])

    def val_dataloader(self):
        if self.val_distributions is None:
            return None

        val_dataset = SingleBatchDataset(self.batch_size*self.trainer.world_size, self.val_distributions)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=0
        )  # Must be 0 since collate_fn references outer scope variable
        return val_dataloader
