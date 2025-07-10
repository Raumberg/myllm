"""FP8 training helpers.

Thin wrappers around NVIDIA/AMD Transformer Engine to keep the rest of the
codebase independent from the actual backend. If Transformer Engine is missing
(e.g. CI on CPU), the API degrades gracefully:

* When cfg.quant.use_fp8 == False → functions fall back to no-op so that the
  caller can still execute.
* When cfg.quant.use_fp8 == True → `ImportError` is raised, making the problem
  explicit.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any

from myllm.quant.backend import FP8Backend


__all__ = ["fp8_autocast", "FP8RecipeBuilder"]


class AmaxComputeAlgo(Enum):
    MAX = "max"
    MOST_RECENT = "most_recent"

class FP8Format(Enum):
    E4M3 = "E4M3"
    E5M2 = "E5M2"
    HYBRID = "HYBRID"


@dataclass
class FP8RecipeBuilder:
    """Builds a DelayedScaling recipe for Transformer Engine."""

    margin: int = 0
    fp8_format: FP8Format = FP8Format.HYBRID
    amax_history_len: int = 1
    amax_compute_algo: AmaxComputeAlgo = AmaxComputeAlgo.MOST_RECENT

    def to_recipe(self) -> Any:
        """Builds a DelayedScaling recipe for `fp8_autocast`."""
        if not FP8Backend.is_available:
            raise ImportError(
                "FP8 support requires TransformerEngine to be installed. "
                "Please `pip install transformer-engine`."
            )

        fp8_format_enum = getattr(FP8Backend.Format, self.fp8_format.value)

        return FP8Backend.DelayedScaling(
            margin=self.margin,
            fp8_format=fp8_format_enum,
            amax_history_len=self.amax_history_len,
            amax_compute_algo=self.amax_compute_algo.value,
        )


def fp8_autocast(enabled: bool = False, fp8_recipe: Any | None = None) -> Any:
    """FP8 autocast context manager.

    This is a wrapper around `transformer_engine.pytorch.fp8_autocast`.
    It gracefully handles cases where Transformer Engine is not installed
    or the hardware does not support FP8.
    """
    if not enabled:
        return nullcontext()

    if not FP8Backend.is_fp8_supported:
        if not FP8Backend.is_available:
            raise ModuleNotFoundError(
                "Transformer Engine is not available. "
                "Please install it to use FP8 precision: `pip install transformer-engine`"
            )
        else:
            raise RuntimeError(
                "FP8 training is enabled, but the current GPU does not support it. "
                "FP8 is only available on NVIDIA Hopper architecture (H100) and newer."
            )

    return FP8Backend.te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)