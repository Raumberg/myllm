from __future__ import annotations

"""Abstract metric interface used by :pymod:`myllm.metrics`."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""

    name: str  # Every metric must expose a human-readable name

    # ------------------------------------------------------------------
    def __init__(self) -> None:  # noqa: D401
        self.reset()

    # ------------------------------------------------------------------
    @abstractmethod
    def update(self, **kwargs: Any) -> None:  # noqa: D401
        """Update internal state with data from a single batch.

        Concrete implementations decide which kwargs they require – for example
        *Perplexity* expects ``loss`` (float) and ``tokens`` (int).
        """

    # ------------------------------------------------------------------
    @abstractmethod
    def compute(self) -> float | Dict[str, float]:  # noqa: D401
        """Return final metric value(s)."""

    # ------------------------------------------------------------------
    @abstractmethod
    def reset(self) -> None:  # noqa: D401
        """Clear accumulated state so the metric can be reused.""" 