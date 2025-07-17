from __future__ import annotations

"""Engine back-ends (DeepSpeed, Accelerate, FSDP, etc.)."""

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, Type

from myllm.enums import EngineType

if TYPE_CHECKING:
    from types import ModuleType

# Registry mapping enum to module names
_ENGINE_REGISTRY: Dict[EngineType, str] = {
    EngineType.DEEPSPEED: "myllm.engines.deepspeed",
    EngineType.ACCELERATE: "myllm.engines.accelerate",
    EngineType.DEFAULT: "myllm.engines.accelerate",
}


def get_engine(name: EngineType) -> ModuleType:
    """Factory that resolves engine by name.

    Parameters
    ----------
    name: EngineType
        The engine to resolve.
    """
    if name not in _ENGINE_REGISTRY:
        raise ValueError(f"Unsupported engine: {name}")

    module_path = _ENGINE_REGISTRY[name]
    return import_module(module_path)


__all__ = ["get_engine"] 