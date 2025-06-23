from __future__ import annotations

"""Perplexity metric implementation."""

import math
from typing import Any

from .base import BaseMetric

__all__ = ["Perplexity"]


class Perplexity(BaseMetric):
    """Compute token-level perplexity from average negative log-likelihood (NLL).

    Expected update parameters:
    loss: float – mean cross-entropy loss over valid tokens in the batch (as
      returned by HuggingFace models).
    tokens: int – number of *valid* tokens that contributed to the loss (i.e.
      labels != ignore_index).
    """

    name = "perplexity"

    # ------------------------------------------------------------------
    def reset(self) -> None:  # noqa: D401
        self.nll_total: float = 0.0
        self.token_count: int = 0

    # ------------------------------------------------------------------
    def update(self, *, loss: float, tokens: int, **_: Any) -> None:  # noqa: D401
        self.nll_total += loss * tokens  # bring back summed nll
        self.token_count += tokens

    # ------------------------------------------------------------------
    def compute(self) -> float:  # noqa: D401
        if self.token_count == 0:
            raise RuntimeError("PerplexityMetric.compute() called with no data.")
        avg_nll = self.nll_total / self.token_count
        return float(math.exp(avg_nll)) 