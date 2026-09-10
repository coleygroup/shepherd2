"""Model and checkpoint loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .checkpoint_manager import (
    MODEL_CHECKPOINTS,
    CheckpointManager,
    get_checkpoint_filename,
    get_checkpoint_path,
)

if TYPE_CHECKING:
    from .model_loader import clear_model_cache, get_model_info, load_model
else:

    def load_model(*args, **kwargs):
        from .model_loader import load_model as _load_model

        return _load_model(*args, **kwargs)

    def get_model_info():
        return CheckpointManager().get_available_models()

    def clear_model_cache(model_type=None, cache_dir=None):
        CheckpointManager(cache_dir=cache_dir).clear_cache(model_type=model_type)

__all__ = [
    "load_model",
    "get_model_info",
    "clear_model_cache",
    "CheckpointManager",
    "get_checkpoint_path",
    "get_checkpoint_filename",
    "MODEL_CHECKPOINTS",
]
