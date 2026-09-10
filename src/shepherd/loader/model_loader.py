"""
Model loader for ShEPhERD-2.
"""
from pathlib import Path
from typing import Literal

import torch

from shepherd.lightning_module import LightningModule
from shepherd.shepherd_model import ShepherdModel

from .checkpoint_manager import get_checkpoint_path


def load_model(
    model_type: Literal['mosesaq'] = 'mosesaq',
    local_checkpoint_path: str | None = None,
    cache_dir: str | None = None,
    force_download: bool = False,
    inference_only: bool = True,
    device: str | None = None,
) -> ShepherdModel:
    """
    Load a ShEPhERD-2 model from Huggingface or a local checkpoint.

    Arguments
    ---------
    model_type: Type of model to load
        - 'mosesaq': trained on MOSES-aq dataset
    local_checkpoint_path: Path to a local checkpoint file (bypasses HF download)
    cache_dir: Directory to cache downloaded checkpoints (None uses default HF cache)
    force_download: Whether to force download even if cached checkpoint exists
    inference_only: If True (default), download the stripped inference checkpoint
        from Huggingface; if False, download the full training-resume
        checkpoint with optimizer state
    device: Device to load model on ('cuda', 'cpu', or None for auto-detection)

    Returns
    -------
    ShepherdModel wrapping the loaded LightningModule, ready for inference
    (call `model.generate(...)`; all LightningModule attributes/methods remain
    accessible directly on the wrapper, e.g. `model.params`, `model.device`).

    Example
    -------
    >>> # Load default MOSES-aq inference checkpoint
    >>> model = load_model()
    >>> from shepherd.interaction_profile import extract_interaction_profile
    >>> ref_profile = extract_interaction_profile(mol)  # mol: RDKit Mol with 3D coords + explicit H
    >>> samples = model.generate(
    ...     batch_size=10, N_x1=profile.n_atoms, N_x4=profile.n_pharms, condition=ref_profile,
    ... )

    >>> # Load training-resume checkpoint
    >>> model = load_model(inference_only=False)

    >>> # Force download latest version
    >>> model = load_model('mosesaq', force_download=True)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if local_checkpoint_path is not None:
        try:
            local_checkpoint_path = str(Path(local_checkpoint_path).expanduser().resolve())
            device_obj = torch.device(device)
            model_pl = LightningModule.load_from_checkpoint(
                local_checkpoint_path,
                weights_only=False,
                map_location=device_obj
            )

            model_pl.eval()
            model_pl.model.device = device_obj

            print(f"Successfully loaded {model_type} model from local checkpoint.")
            return ShepherdModel(model_pl, checkpoint_path=local_checkpoint_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from local checkpoint: {str(e)}") from e

    try:
        # Get checkpoint path with automatic downloading
        model_path = get_checkpoint_path(
            model_type=model_type,
            cache_dir=cache_dir,
            force_download=force_download,
            inference_only=inference_only,
        )

        variant = "inference" if inference_only else "resume"
        print(f"Loading {model_type} ({variant}) model from: {model_path}")
        print(f"Using device: {device}")

        device_obj = torch.device(device)
        model_pl = LightningModule.load_from_checkpoint(
            model_path,
            weights_only=False,
            map_location=device_obj
        )

        model_pl.eval()
        model_pl.model.device = device_obj

        print(f"Successfully loaded {model_type} model")
        resolved_model_path = str(Path(model_path).expanduser().resolve())
        return ShepherdModel(model_pl, checkpoint_path=resolved_model_path)

    except Exception as e:
        raise RuntimeError(f"Failed to load {model_type} model: {str(e)}") from e


def get_model_info() -> dict:
    """
    Get information about available ShEPhERD models.

    Returns
    -------
    Dictionary mapping model types to their descriptions
    """
    from .checkpoint_manager import CheckpointManager

    manager = CheckpointManager()
    return manager.get_available_models()


def clear_model_cache(model_type: str | None = None, cache_dir: str | None = None):
    """
    Clear cached model checkpoints.

    Arguments
    ---------
    model_type: Specific model type to clear, or None to clear all
    cache_dir: Cache directory to clear from (None uses default HF cache)
    """
    from .checkpoint_manager import CheckpointManager

    manager = CheckpointManager(cache_dir=cache_dir)
    manager.clear_cache(model_type=model_type)
