from __future__ import annotations

"""Training algorithms (SFT, PPO, DPO, GRPO, etc.)."""

from importlib import import_module
from typing import Any, Dict, Type
from myllm.algorithms.base import BaseTrainer
from myllm.enums import AlgorithmType

# Registry mapping enum to module names
_ALGO_REGISTRY: Dict[AlgorithmType, str] = {
    AlgorithmType.SFT: "myllm.algorithms.sft",
    AlgorithmType.PPO: "myllm.algorithms.ppo",
    AlgorithmType.DPO: "myllm.algorithms.distill",
    AlgorithmType.DISTILL: "myllm.algorithms.distill",
    AlgorithmType.GRPO: "myllm.algorithms.grpo",
}

__all__ = ["get_algorithm", "get_trainer_class"]


def get_algorithm(name: AlgorithmType) -> Any:
    """Factory that resolves algorithm module by name."""
    if name not in _ALGO_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}")
    return import_module(_ALGO_REGISTRY[name])


def get_trainer_class(algo_mod: Any) -> type[BaseTrainer]:
    """Return *Trainer* class from an algorithm module.

    Priority:
    1. Explicit attribute ``Trainer`` inside module.
    2. Fallback to ``<NAME_UPPER>Trainer`` where NAME = module's basename.
    """

    # 1. Explicit export
    if hasattr(algo_mod, "Trainer"):
        return getattr(algo_mod, "Trainer")

    # 2. Derive camel name from module path
    mod_name = algo_mod.__name__.split(".")[-1]  # e.g. 'sft'
    camel = f"{mod_name.upper()}Trainer"          # → 'SFTTrainer'
    trainer_cls = getattr(algo_mod, camel, None)

    if trainer_cls is None:
        raise RuntimeError(f"Algorithm module {algo_mod.__name__} does not expose a Trainer class")

    return trainer_cls