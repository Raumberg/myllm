from __future__ import annotations

"""DeepSpeed engine integration.
Minimal stub; will be extended with optimizer, scheduler and zero-offload support.
"""

from pathlib import Path
from typing import Any, Tuple

import json
import logging
import os

import torch
import torch.optim as optim  # local import to avoid cost when DS not used

try:
    import deepspeed  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover — optional dependency
    raise RuntimeError(
        "DeepSpeed is not installed. Add `deepspeed` in your environment to use this engine."
    ) from exc

logger = logging.getLogger(__name__)


def _load_config(config_path: str | os.PathLike | None) -> dict[str, Any]:
    if config_path is None:
        logger.warning("No DeepSpeed config provided – falling back to default fp16 config.")
        return {
            "train_batch_size": 1,
            "fp16": {
                "enabled": True,
            },
        }

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    logger.debug("Loaded DeepSpeed config from %s", config_path)
    return cfg


def _tune_ds_config(ds_cfg: dict[str, Any], cfg: Any, model: torch.nn.Module) -> None:  # noqa: D401, C901
    """Mutate ``ds_cfg`` in-place to better match current run.

    1. Auto-set ``train_batch_size`` if missing.
    2. Tweak ZeRO-3 buckets based on hidden size (mirrors logic from v1).
    3. Ensure gradient accumulation matches our ``TrainingCfg``.
    4. Fill in sensible defaults for fp16/bf16 mixed precision.
    """

    # ------------------------------------------------------------------
    # Auto batch size
    # ------------------------------------------------------------------
    train_cfg = getattr(cfg, "training", None)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if train_cfg and "train_batch_size" not in ds_cfg:
        micro = getattr(train_cfg, "micro_batch_size", 1)
        grad_acc = getattr(train_cfg, "gradient_accumulation_steps", 1)
        ds_cfg["train_batch_size"] = micro * grad_acc * world_size

    # ------------------------------------------------------------------
    # Make sure gradient accumulation matches config (if user forgot)
    # ------------------------------------------------------------------
    if train_cfg:
        ds_cfg.setdefault("gradient_accumulation_steps", getattr(train_cfg, "gradient_accumulation_steps", 1))

    # ------------------------------------------------------------------
    # ZeRO-3 bucket tuning (helps with OOM)
    # ------------------------------------------------------------------
    zero_opt = ds_cfg.get("zero_optimization", {})
    stage = zero_opt.get("stage", 0)
    if stage == 3:
        hidden_size: int | None = None
        if hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", None)
            if hidden_size is None and getattr(model.config, "hidden_sizes", None):
                hidden_size = max(model.config.hidden_sizes)  # type: ignore

        if hidden_size:
            factor = hidden_size * hidden_size
            zero_opt.setdefault("reduce_bucket_size", factor)
            zero_opt.setdefault("stage3_param_persistence_threshold", 10 * hidden_size)
            zero_opt.setdefault("stage3_prefetch_bucket_size", int(0.9 * factor))
    ds_cfg["zero_optimization"] = zero_opt

    # ------------------------------------------------------------------
    # Mixed precision default if user omitted
    # ------------------------------------------------------------------
    model_dtype = getattr(getattr(cfg, "model", None), "dtype", "bf16").lower()
    if model_dtype in {"fp16", "float16", "16"} and "fp16" not in ds_cfg:
        ds_cfg["fp16"] = {"enabled": True}
    elif model_dtype == "bf16" and "bf16" not in ds_cfg:
        ds_cfg["bf16"] = {"enabled": True}


def _fallback_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:  # noqa: D401
    """Create Adam optimizer for trainable params with LR from cfg."""

    default_lr = getattr(getattr(cfg, "training", None), "lr", 2e-5)
    return optim.Adam([p for p in model.parameters() if p.requires_grad], lr=default_lr)


def prepare(
    cfg: Any,
    model: torch.nn.Module,
    dataloader_len: int | None = None,
) -> Tuple["deepspeed.DeepSpeedEngine", torch.nn.Module, torch.optim.Optimizer, Any]:
    """Initialize DeepSpeed engine.

    Parameters
    ----------
    cfg: Any
        Arbitrary training config that must contain ``deepspeed_config`` attribute or key.
    model: torch.nn.Module
        Model instance ready for training (could be LoRA-wrapped, quantized, etc.).
    dataloader_len: int | None
        Length of the training dataloader (needed for scheduler in some cases).
    """

    # Look for config path in new field names first, fall back to legacy ones for backward compat.
    ds_conf_path: str | os.PathLike | None = getattr(cfg, "config", None)
    if ds_conf_path is None and hasattr(cfg, "engine"):
        ds_conf_path = getattr(cfg.engine, "config", None)
    if ds_conf_path is None:
        # Legacy fallback
        ds_conf_path = getattr(cfg, "deepspeed_config", None) or getattr(getattr(cfg, "engine", object()), "deepspeed_config", None)

    ds_cfg = _load_config(ds_conf_path)

    # Dynamically adjust config based on current run
    _tune_ds_config(ds_cfg, cfg, model)

    if all(v not in os.environ for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        logger.debug("No distributed env vars – configuring fake single-process ranks for DeepSpeed")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        dist_init_required = True  # DeepSpeed will init torch.distributed internally
    else:
        dist_init_required = True

    # Decide whether we need to create the optimizer manually or let DeepSpeed build it.
    user_defined_opt = ds_cfg.get("optimizer", {}).get("type") not in (None, "none", "None")

    engine, optimizer, _, scheduler = deepspeed.initialize(  # type: ignore
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        optimizer=None if user_defined_opt else _fallback_optimizer(model, cfg),
        config_params=ds_cfg,
        dist_init_required=dist_init_required,
    )

    if dataloader_len and scheduler is None and "scheduler" in ds_cfg:
        logger.warning(
            "Scheduler requested in DeepSpeed config but engine returned None. "
            "You may need to create it manually."
        )

    logger.info("DeepSpeed engine initialized (%s GPUs)", torch.cuda.device_count())
    return engine, engine.module, optimizer, scheduler


__all__ = ["prepare"] 