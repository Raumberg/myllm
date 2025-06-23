from __future__ import annotations

"""Training algorithms (SFT, PPO, DPO, GRPO, etc.)."""

from importlib import import_module
from typing import Any

__all__ = ["get_algorithm"]


def get_algorithm(name: str) -> Any:
    name = name.lower()
    if name == "sft":
        return import_module("myllm.algorithms.sft")
    if name == "grpo":
        return import_module("myllm.algorithms.grpo")
    if name == "ppo":
        return import_module("myllm.algorithms.ppo")
    if name in {"dpo", "distill"}:
        return import_module("myllm.algorithms.distill")
    # TODO: map others
    raise ValueError(f"Unknown algorithm: {name}") 