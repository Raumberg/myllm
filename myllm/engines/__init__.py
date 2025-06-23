from __future__ import annotations

"""Engine back-ends (DeepSpeed, Accelerate, FSDP, etc.)."""

from importlib import import_module
from typing import TYPE_CHECKING, Any


def get_engine(name: str) -> Any:
    """Factory that resolves engine by name.

    Parameters
    ----------
    name: str
        Name of the engine. Currently supports ``deepspeed`` and ``accelerate``.
    """
    name = name.lower()
    if name == "deepspeed":
        return import_module("myllm.engines.deepspeed")
    if name == "accelerate":
        return import_module("myllm.engines.accelerate")

    raise ValueError(f"Unsupported engine: {name}")


__all__ = ["get_engine"] 