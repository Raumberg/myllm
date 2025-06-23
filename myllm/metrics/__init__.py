from __future__ import annotations

"""Metrics and evaluation utilities.

Currently supports only *perplexity* but the architecture is extensible:

Examples
--------
>>> from myllm.metrics import Evaluator, Perplexity
>>> ev = Evaluator(model, metrics=[Perplexity()])
>>> scores = ev.evaluate(dataloader)
"""

from importlib import import_module
from typing import Any, List, Sequence

MetricLike = "BaseMetric | type[BaseMetric]"

# ---------------------------------------------------------------------
# Re-export core symbols
# ---------------------------------------------------------------------
from .base import BaseMetric  # noqa: F401
from .perplexity import Perplexity  # noqa: F401
from .evaluator import Evaluator  # noqa: F401

__all__: Sequence[str] = [
    "BaseMetric",
    "Perplexity",
    "Evaluator",
] 