from __future__ import annotations

"""Dataset loaders, preprocessors and collators (stub)."""

from typing import Protocol, Any
from .module import DataModule

__all__ = [
    "DatasetLoader",
    "DataModule",
    "processors",
    "collators",
]


class DatasetLoader(Protocol):
    """Protocol for dataset loader implementations."""

    def load(self, split: str = "train", **kwargs: Any):  # noqa: D401
        ... 