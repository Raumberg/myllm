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

from dataclasses import dataclass
from typing import Literal, Any, Callable

__all__ = ["fp8_autocast", "FP8Config", "get_fp8_recipe"]

# --- Optional dependency: Transformer Engine ---
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    _TE_AVAILABLE = True
except (ImportError, FileNotFoundError):
    _TE_AVAILABLE = False
    # Define dummy types for mypy
    te = None
    DelayedScaling = object
    Format = object


AmaxComputeAlgo = Literal["max", "most_recent"]


@dataclass
class FP8Config:
    """Configuration for FP8 training using the DelayedScaling recipe."""

    margin: int = 0
    fp8_format: Literal["E4M3", "HYBRID"] = "HYBRID"
    amax_history_len: int = 1024
    amax_compute_algo: AmaxComputeAlgo = "max"
    scaling_factor_compute_algo: Callable | None = None

    def to_recipe(self) -> DelayedScaling:
        """Builds a DelayedScaling recipe for `fp8_autocast`."""
        if not _TE_AVAILABLE:
            raise ImportError(
                "FP8 support requires TransformerEngine to be installed. "
                "Please `pip install transformer-engine`."
            )

        fp8_format_enum = getattr(Format, self.fp8_format)

        return DelayedScaling(
            margin=self.margin,
            fp8_format=fp8_format_enum,
            amax_history_len=self.amax_history_len,
            amax_compute_algo=self.amax_compute_algo,
            scaling_factor_compute_algo=self.scaling_factor_compute_algo,
        )


def get_fp8_recipe(config: FP8Config) -> DelayedScaling:
    """Builds a DelayedScaling recipe from the provided configuration."""
    return config.to_recipe()


def fp8_autocast(
    enabled: bool = False, fp8_recipe: DelayedScaling | None = None, **kwargs: Any
) -> Any:
    """FP8 autocast context manager.

    This is a wrapper around `transformer_engine.pytorch.fp8_autocast`.
    """
    if not _TE_AVAILABLE:
        # Only raise error if user is actually trying to use FP8
        if enabled:
            raise ModuleNotFoundError(
                "Transformer Engine is not available. "
                "Please install it to use FP8 precision: `pip install transformer-engine`"
            )
        # Otherwise return a null context
        from contextlib import nullcontext
        return nullcontext()
        
    return te.fp8_autocast(enabled=enabled, fp8_recipe=fp8_recipe, **kwargs)