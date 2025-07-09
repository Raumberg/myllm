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
__all__ = ["prepare"] 

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _is_auto(val: Any) -> bool:  # noqa: D401
    """Return True if value is the special DeepSpeed placeholder 'auto'."""

    return isinstance(val, str) and val.lower() == "auto"


def _replace_auto(val: Any, replacement: Any):  # noqa: D401
    """Replace the string 'auto' (case-insensitive) with *replacement*."""

    return replacement if _is_auto(val) else val


def _assert_no_auto(cfg: dict[str, Any], path: str = "root") -> None:  # noqa: D401
    """Recursively walk *cfg* and raise if any 'auto' placeholders remain."""

    if isinstance(cfg, dict):
        for k, v in cfg.items():
            _assert_no_auto(v, f"{path}.{k}")
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            _assert_no_auto(v, f"{path}[{i}]")
    elif _is_auto(cfg):
        raise ValueError(f"DeepSpeed config still contains 'auto' value at {path}")


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


def _tune_ds_config(
    ds_cfg: dict[str, Any],
    cfg: Any,
    model: torch.nn.Module,
    dataloader_len: int | None = None,
) -> None:  # noqa: D401, C901
    """Mutate ``ds_cfg`` in-place to better match current run.

    1. Auto-set ``train_batch_size`` if missing.
    2. Tweak ZeRO-3 buckets based on hidden size (mirrors logic from v1).
    3. Ensure gradient accumulation matches our ``TrainingCfg``.
    4. Fill in sensible defaults for fp16/bf16 mixed precision.
    """

    # ------------------------------------------------------------------
    # Auto batch size
    # ------------------------------------------------------------------
    train_cfg = cfg.training
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if train_cfg:
        micro = getattr(train_cfg, "micro_batch_size", 1)
        grad_acc = getattr(train_cfg, "gradient_accumulation_steps", 1)

        ds_cfg["train_micro_batch_size_per_gpu"] = _replace_auto(
            ds_cfg.get("train_micro_batch_size_per_gpu", "auto"), micro
        )

        ds_cfg["gradient_accumulation_steps"] = _replace_auto(
            ds_cfg.get("gradient_accumulation_steps", "auto"), grad_acc
        )

        ds_cfg["train_batch_size"] = _replace_auto(
            ds_cfg.get("train_batch_size", "auto"), micro * grad_acc * world_size
        )

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

            zero_opt["reduce_bucket_size"] = _replace_auto(
                zero_opt.get("reduce_bucket_size", "auto"), factor
            )

            zero_opt["stage3_param_persistence_threshold"] = _replace_auto(
                zero_opt.get("stage3_param_persistence_threshold", "auto"), 10 * hidden_size
            )

            zero_opt["stage3_prefetch_bucket_size"] = _replace_auto(
                zero_opt.get("stage3_prefetch_bucket_size", "auto"), int(0.9 * factor)
            )
    ds_cfg["zero_optimization"] = zero_opt

    # ------------------------------------------------------------------
    # Optimizer & scheduler – replace remaining 'auto' placeholders
    # ------------------------------------------------------------------

    opt_cfg = ds_cfg.get("optimizer", {})
    opt_params = opt_cfg.get("params", {})

    # LR comes from training cfg if provided, else fallback
    default_lr = getattr(getattr(cfg, "training", None), "lr", 2e-5)

    if opt_params:
        opt_params["lr"] = _replace_auto(opt_params.get("lr", "auto"), default_lr)
        opt_params["betas"] = _replace_auto(opt_params.get("betas", "auto"), (0.9, 0.999))
        opt_params["eps"] = _replace_auto(opt_params.get("eps", "auto"), 1e-8)
        opt_params["weight_decay"] = _replace_auto(
            opt_params.get("weight_decay", "auto"), getattr(getattr(cfg, "training", None), "weight_decay", 0.0)
        )
        opt_cfg["params"] = opt_params
        ds_cfg["optimizer"] = opt_cfg

    sch_cfg = ds_cfg.get("scheduler", {})
    sch_params = sch_cfg.get("params", {})

    if sch_params:
        # derive sensible numbers
        warmup_steps = getattr(getattr(cfg, "training", None), "warmup_steps", 0)
        epochs = getattr(getattr(cfg, "training", None), "epochs", 1)

        total_steps_estimate: int | None = None
        if dataloader_len:
            total_steps_estimate = max(1, dataloader_len * epochs)

        sch_params["warmup_min_lr"] = _replace_auto(sch_params.get("warmup_min_lr", "auto"), 0.0)
        sch_params["warmup_max_lr"] = _replace_auto(sch_params.get("warmup_max_lr", "auto"), default_lr)
        sch_params["warmup_num_steps"] = _replace_auto(sch_params.get("warmup_num_steps", "auto"), warmup_steps)
        if total_steps_estimate is not None:
            sch_params["total_num_steps"] = _replace_auto(
                sch_params.get("total_num_steps", "auto"), total_steps_estimate
            )
        elif isinstance(sch_params.get("total_num_steps"), str) and sch_params["total_num_steps"] == "auto":
            # fallback when we cannot compute – remove scheduler to avoid DS error
            logger.warning(
                "Removing scheduler section from DeepSpeed config because total_num_steps could not be inferred"
            )
            ds_cfg.pop("scheduler", None)
        else:
            sch_cfg["params"] = sch_params
            ds_cfg["scheduler"] = sch_cfg

    # gradient clipping
    ds_cfg["gradient_clipping"] = _replace_auto(
        ds_cfg.get("gradient_clipping", "auto"), getattr(getattr(cfg, "training", None), "gradient_clipping", 1.0)
    )

    # ------------------------------------------------------------------
    # Mixed precision default if user omitted
    # ------------------------------------------------------------------
    model_dtype = getattr(cfg.model, "dtype", "bf16").lower()

    # Helper to patch 'enabled': 'auto' placeholders
    def _fix_enabled(section: str, should_enable: bool) -> None:
        if section in ds_cfg:
            enabled_val = ds_cfg[section].get("enabled", False)
            if enabled_val == "auto":
                ds_cfg[section]["enabled"] = should_enable
        else:
            if should_enable:
                ds_cfg[section] = {"enabled": True}

    if model_dtype in {"fp16", "float16", "16"}:
        _fix_enabled("fp16", True)
        _fix_enabled("bf16", False)
    elif model_dtype == "bf16":
        _fix_enabled("bf16", True)
        _fix_enabled("fp16", False)
    else:  # full precision
        _fix_enabled("fp16", False)
        _fix_enabled("bf16", False)

    # ------------------------------------------------------------------
    # Final validation: ensure no 'auto' remains unless user allowed it
    # ------------------------------------------------------------------

    auto_fill_flag = True
    if hasattr(cfg, "engine") and getattr(cfg.engine, "auto_fill", None) is not None:
        auto_fill_flag = cfg.engine.auto_fill

    if auto_fill_flag:
        _assert_no_auto(ds_cfg)


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
    _tune_ds_config(ds_cfg, cfg, model, dataloader_len)

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

