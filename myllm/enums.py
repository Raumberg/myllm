from __future__ import annotations

"""Core data types and enumerations."""

from enum import Enum


class EngineType(str, Enum):
    """Enumeration of supported training engines."""

    DEEPSPEED = "deepspeed"
    ACCELERATE = "accelerate"


class AlgorithmType(str, Enum):
    """Enumeration of supported training algorithms."""

    SFT = "sft"
    PPO = "ppo"
    DPO = "dpo"
    DISTILL = "distill"
    GRPO = "grpo" 