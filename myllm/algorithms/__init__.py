from __future__ import annotations

"""Training algorithms (SFT, PPO, DPO, GRPO, etc.)."""

from importlib import import_module
from typing import Any
from myllm.algorithms.base import BaseTrainer

__all__ = ["get_algorithm", "get_trainer_class"]


def get_algorithm(name: str) -> Any:
    name = name.lower()
    match name:
        case "sft": return import_module(
            "myllm.algorithms.sft"
            )
        case "grpo": return import_module(
            "myllm.algorithms.grpo"
            )
        case "ppo": return import_module(
            "myllm.algorithms.ppo"
            )
        case "dpo" | "distill": return import_module(
            "myllm.algorithms.distill"
            )
        case _: raise ValueError(f"Unknown algorithm: {name}")

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