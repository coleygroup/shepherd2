"""
Checkpoint manager for downloading and caching ShEPhERD model weights from Hugging Face.
"""
import os
from typing import Literal
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

HF_REPO_ID = 'kabeywar/shepherd2'

# Model checkpoint mappings
MODEL_CHECKPOINTS = {
    'mosesaq': {
        'repo_id': HF_REPO_ID,
        'inference_filename': 'shepherd2/shepherd2-mosesaq-inference.ckpt',
        'resume_filename': 'shepherd2/shepherd2-mosesaq-resume.ckpt',
        'description': 'MOSES-aq trained model with shape, electrostatics, and pharmacophores',
    },
}


def get_checkpoint_filename(model_type: str, inference_only: bool = True) -> str:
    """Return the Hugging Face filename for a model type and checkpoint variant."""
    if model_type not in MODEL_CHECKPOINTS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(MODEL_CHECKPOINTS.keys())}"
        )
    checkpoint_info = MODEL_CHECKPOINTS[model_type]
    if inference_only:
        return checkpoint_info['inference_filename']
    return checkpoint_info['resume_filename']


class CheckpointManager:
    """Manages downloading and caching of ShEPhERD model checkpoints."""

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize checkpoint manager.

        Arguments
        ---------
        cache_dir: Directory to cache checkpoints. If None, uses default HF cache.
            Typically this is: ~/.cache/huggingface/
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def get_checkpoint_path(
        self,
        model_type: Literal['mosesaq'],
        force_download: bool = False,
        inference_only: bool = True,
    ) -> str:
        """
        Download or retrieve cached checkpoint for the specified model type.

        Arguments
        ---------
        model_type: Type of model checkpoint to retrieve
        force_download: Whether to force re-download even if cached
        inference_only: If True, download the stripped inference checkpoint;
            if False, download the full training-resume checkpoint

        Returns
        -------
        Path to the downloaded checkpoint file
        """
        checkpoint_info = MODEL_CHECKPOINTS[model_type]
        filename = get_checkpoint_filename(model_type, inference_only=inference_only)

        try:
            # Download from Hugging Face Hub with caching
            checkpoint_path = hf_hub_download(
                repo_id=checkpoint_info['repo_id'],
                filename=filename,
                repo_type='model',
                cache_dir=self.cache_dir,
                force_download=force_download
            )

            # Verify the file exists and is readable
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Downloaded checkpoint not found at {checkpoint_path}")

            return checkpoint_path

        except HfHubHTTPError as e:
            if "401" in str(e):
                raise HfHubHTTPError(
                    f"Access denied to {checkpoint_info['repo_id']}. "
                    "The repository might be private or require authentication. "
                    "Please check the repository permissions or provide a valid token.",
                    response=e.response,
                ) from e
            elif "404" in str(e):
                raise HfHubHTTPError(
                    f"Checkpoint not found: {filename} "
                    f"in repository {checkpoint_info['repo_id']}. "
                    "Please verify the repository and file names are correct.",
                    response=e.response,
                ) from e
            else:
                raise HfHubHTTPError(
                    f"Failed to download checkpoint: {e}. "
                    "Please check your internet connection and try again.",
                    response=e.response,
                ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error downloading checkpoint: {e}") from e

    def get_available_models(self) -> dict:
        """
        Get information about available model checkpoints.

        Returns
        -------
        Dictionary with model types and their descriptions
        """
        return {
            model_type: info['description']
            for model_type, info in MODEL_CHECKPOINTS.items()
        }

    def clear_cache(self, model_type: str | None = None):
        """
        Clear cached checkpoints.

        Arguments
        ---------
        model_type: Specific model type to clear, or None to clear all
        """
        if not self.cache_dir:
            print("Using default HF cache directory. Use huggingface-cli to manage cache.")
            return

        if model_type:
            if model_type not in MODEL_CHECKPOINTS:
                raise ValueError(f"Unknown model type: {model_type}")
            # Clear both inference and resume variants for this model type
            for inference_only in (True, False):
                filename = get_checkpoint_filename(model_type, inference_only=inference_only)
                cache_path = os.path.join(self.cache_dir, filename)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            print(f"Cleared cache for {model_type}")
        else:
            # Clear all cached checkpoints
            for model_type_key in MODEL_CHECKPOINTS:
                for inference_only in (True, False):
                    filename = get_checkpoint_filename(model_type_key, inference_only=inference_only)
                    cache_path = os.path.join(self.cache_dir, filename)
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
            print("Cleared all cached checkpoints")


def get_checkpoint_path(
    model_type: Literal['mosesaq'],
    cache_dir: str | None = None,
    force_download: bool = False,
    inference_only: bool = True,
) -> str:
    """
    Download or retrieve a cached checkpoint from Hugging Face.

    Arguments
    ---------
    model_type: Type of model checkpoint to retrieve
    cache_dir: Directory to cache downloaded checkpoints (optional)
    force_download: Whether to force download even if cached
    inference_only: If True, prefer the stripped inference checkpoint;
        if False, prefer the full training-resume checkpoint

    Returns
    -------
    Path to the checkpoint file
    """
    manager = CheckpointManager(cache_dir=cache_dir)
    return manager.get_checkpoint_path(
        model_type,
        force_download=force_download,
        inference_only=inference_only,
    )
