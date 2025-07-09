from __future__ import annotations

"""Accelerate engine wrapper.

Provides a thin abstraction around `accelerate.Accelerator` to keep the rest of
the codebase engine-agnostic.
"""

import collections.abc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import optim


try:
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate.utils import DummyOptim, HfDeepSpeedConfig
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "accelerate is not installed. Run `pip install accelerate` or choose another engine."
    ) from exc

# Reuse tuning logic from the dedicated DeepSpeed engine — proven and battle-tested.
from myllm.engines.deepspeed import _tune_ds_config  # type: ignore

logger = logging.getLogger(__name__)
__all__ = ["prepare"]


def _deepspeed_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and key in source:
            source[key] = _deepspeed_update(source.get(key, {}), value)
        else:
            source[key] = value
    return source


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
    Initializes the Accelerator, using the main YAML as the source of truth.

    This function implements the "YAML is king" philosophy. The `deepspeed.json`
    is treated as a template, which is then recursively updated by the `engine.override`
    section from the main YAML configuration. This gives the user full control from
    a single file.

    A single, necessary patch for `train_micro_batch_size_per_gpu` is still
    applied manually to satisfy an `accelerate.prepare()` pre-flight check.
    """
    train_cfg = cfg.training
    grad_acc = getattr(train_cfg, "gradient_accumulation_steps", 1)
    micro_batch = getattr(train_cfg, "micro_batch_size", 1)
    engine_cfg = getattr(cfg, "engine", None)

    accelerator_kwargs = {
        "mixed_precision": _infer_precision(cfg.model.dtype),
        "gradient_accumulation_steps": grad_acc,
    }

    use_ds_optimizer = False

    if engine_cfg and engine_cfg.config:
        ds_conf_path = Path(engine_cfg.config)
        if not ds_conf_path.exists(): raise FileNotFoundError(f"DeepSpeed config not found: {ds_conf_path}")

        with ds_conf_path.open("r", encoding="utf-8") as f:
            ds_json_config = json.load(f)

        # --- Apply Overrides from YAML ---
        overrides = engine_cfg.override
        if overrides:
            logger.info("Applying DeepSpeed overrides from YAML config.")
            _deepspeed_update(ds_json_config, overrides)

        # --- Auto-tune DeepSpeed config using the same logic as the standalone engine ---
        _tune_ds_config(ds_json_config, cfg, model, dataloader_len)

        # Check for optimizer *after* overrides and auto-tuning have been applied.
        if "optimizer" in ds_json_config and ds_json_config["optimizer"].get("type"):
            use_ds_optimizer = True

        hf_ds_config = HfDeepSpeedConfig(ds_json_config)
        accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(hf_ds_config=hf_ds_config)

    accelerator = Accelerator(**accelerator_kwargs)

    if use_ds_optimizer:
        optimizer = DummyOptim(model.parameters())
    else:
        optimizer = _fallback_optimizer(model, cfg)

    model, optimizer, *_ = accelerator.prepare(model, optimizer)

    return accelerator, model, optimizer, None 