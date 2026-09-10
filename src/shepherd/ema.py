"""Online EMA via PyTorch Lightning WeightAveraging (torch.optim.swa_utils AveragedModel)."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

_USE_LIGHTNING_BUILTIN_EMA = False
try:
    from pytorch_lightning.callbacks import EMAWeightAveraging as _OnlineEMABase
    _USE_LIGHTNING_BUILTIN_EMA = True
except ImportError:
    class _OnlineEMABase(WeightAveraging):  # type: ignore[misc]
        """Fallback when ``EMAWeightAveraging`` is not in this Lightning version."""

        def __init__(
            self,
            *,
            decay: float = 0.999,
            device: Any | None = None,
            use_buffers: bool = True,
            update_starting_at_step: int | None = None,
            update_every_n_steps: int = 1,
        ) -> None:
            super().__init__(
                device=device,
                use_buffers=use_buffers,
                avg_fn=get_ema_avg_fn(decay=decay),
            )
            self._update_starting_at_step = update_starting_at_step
            self._update_every_n_steps = update_every_n_steps

        def should_update(self, step_idx=None, epoch_idx=None):
            if step_idx is None:
                return False
            if self._update_starting_at_step is not None and step_idx < self._update_starting_at_step:
                return False
            if self._update_every_n_steps <= 0:
                return False
            return step_idx % self._update_every_n_steps == 0


class ShepherdOnlineEMA(_OnlineEMABase):
    """Exponential moving average of the LightningModule weights.

    When ``use_for_validation`` is false, validation uses the training weights.
    """

    def __init__(
        self,
        *,
        decay: float = 0.999,
        use_for_validation: bool = False,
        device: Any | None = None,
        use_buffers: bool = True,
        update_starting_at_step: int | None = None,
    ) -> None:
        self._use_ema_for_validation = use_for_validation
        if _USE_LIGHTNING_BUILTIN_EMA:
            super().__init__(
                device=device,
                use_buffers=use_buffers,
                decay=decay,
                update_every_n_steps=1,
                update_starting_at_step=update_starting_at_step,
                update_starting_at_epoch=None,
            )
        else:
            super().__init__(
                decay=decay,
                device=device,
                use_buffers=use_buffers,
                update_starting_at_step=update_starting_at_step,
                update_every_n_steps=1,
            )

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._use_ema_for_validation:
            super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self._use_ema_for_validation:
            super().on_validation_epoch_end(trainer, pl_module)
