"""
Low-precision sub-package.
"""

from __future__ import annotations

# flake8: noqa: F401 -- re-exports

from .fp8 import fp8_autocast, FP8RecipeBuilder

__all__ = [
    "fp8_autocast",
    "FP8RecipeBuilder",
] 