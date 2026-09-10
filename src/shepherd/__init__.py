"""
ShEPhERD-2: Diffusing Shape, Electrostatics, and Pharmacophores for Drug Design.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:  # noqa: SIM105
    __version__ = version("shepherd")
except PackageNotFoundError:
    pass

from .shepherd_model import ShepherdModel

if TYPE_CHECKING:
    from .loader import clear_model_cache, get_model_info, load_model
else:
    def load_model(*args, **kwargs):
        from .loader import load_model as _load_model

        return _load_model(*args, **kwargs)

    def get_model_info(*args, **kwargs):
        from .loader import get_model_info as _get_model_info

        return _get_model_info(*args, **kwargs)

    def clear_model_cache(*args, **kwargs):
        from .loader import clear_model_cache as _clear_model_cache

        _clear_model_cache(*args, **kwargs)

__all__ = [
    "ShepherdModel",
    "load_model",
    "get_model_info",
    "clear_model_cache",
]
