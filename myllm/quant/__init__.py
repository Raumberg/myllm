"""
Low-precision sub-package.
"""

# flake8: noqa: F401 -- re-exports

from .fp8 import fp8_autocast, get_fp8_recipe

__all__ = [
    "get_fp8_recipe",
    "fp8_autocast",
] 