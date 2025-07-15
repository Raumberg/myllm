from __future__ import annotations

"""FSDP engine wrapper.
NOTE: For a future use! Can be rewritten to better match H100 GPU setup.
For now, to use FSDP, you need to access it from accelerate:
```bash
accelerate launch --config_file /path/to/accelerate-cfg-with-fsdp.yaml myllm train ...
```
Module: `myllm.engines.fsdp`
Provides a thin abstraction around `accelerate.Accelerator` with FSDP enabled.
"""

import logging
from typing import Any, Tuple

import torch
from torch import optim

try:
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullStateDictConfig,
        ShardingStrategy,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "accelerate is not installed. Run `pip install accelerate` or choose another engine."
    ) from exc


logger = logging.getLogger(__name__)
__all__ = ["prepare"]


def _infer_precision(dtype: str) -> str:
    """Map the string data type from the config to an Accelerator-compatible precision flag."""
    dtype_low = dtype.lower()
    if "bf16" in dtype_low:
        return "bf16"
    if dtype_low in {"fp16", "float16", "16"}:
        return "fp16"
    return "no"


def _fallback_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Create a simple AdamW optimizer as a fallback."""
    train_cfg = getattr(cfg, "training", None)
    lr = getattr(train_cfg, "lr", 2e-5) if train_cfg else 2e-5
    return optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)


def prepare(
    cfg: Any,
    model: torch.nn.Module,
    dataloader_len: int | None = None,
) -> Tuple[Accelerator, torch.nn.Module, torch.optim.Optimizer | None, Any]:
    """
    Initializes the Accelerator with FSDP, using the main YAML as the source of truth.
    """
    train_cfg = cfg.training
    grad_acc = getattr(train_cfg, "gradient_accumulation_steps", 1)
    engine_cfg = getattr(cfg, "engine", {})

    # FSDP settings can be provided under `engine.config` in the YAML
    fsdp_config = engine_cfg.config or {}

    # Default to FULL_SHARD if not specified
    sharding_strategy = fsdp_config.get("sharding_strategy", "FULL_SHARD")
    
    # Map string from config to the ShardingStrategy enum
    try:
        sharding_strategy_enum = ShardingStrategy[sharding_strategy.upper()]
    except KeyError:
        raise ValueError(f"Invalid FSDP sharding_strategy: {sharding_strategy}") from None

    # For saving sharded models
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=sharding_strategy_enum,
        auto_wrap_policy=None,  # We can add auto-wrap policies later
        state_dict_config=full_state_dict_config,
        **fsdp_config.get("plugin_kwargs", {}),  # For other FSDP plugin settings
    )

    accelerator_kwargs = {
        "mixed_precision": _infer_precision(cfg.model.dtype),
        "gradient_accumulation_steps": grad_acc,
        "fsdp_plugin": fsdp_plugin,
    }

    accelerator = Accelerator(**accelerator_kwargs)

    # With FSDP, we must prepare the model before creating the optimizer
    model = accelerator.prepare(model)

    # Create a default optimizer if one isn't configured elsewhere
    optimizer = _fallback_optimizer(model, cfg)
    optimizer = accelerator.prepare(optimizer)

    # Note: Dataloader preparation will be handled by the trainer
    return accelerator, model, optimizer, None 