from __future__ import annotations

"""A cleaner, object-oriented DeepSpeed engine integration."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from transformers import AutoModelForCausalLM

import torch

try:
    import deepspeed
except ModuleNotFoundError:
    deepspeed = None


logger = logging.getLogger(__name__)
__all__ = ["prepare", "prepare_inference"]


class DeepSpeedConfigTuner:
    """A collection of helpers to tune a DeepSpeed config dictionary."""

    def __init__(self, cfg: Any, model: torch.nn.Module | AutoModelForCausalLM, dataloader_len: Optional[int] = None):
        self.cfg = cfg
        self.model = model
        self.dataloader_len = dataloader_len
        self.ds_cfg: Dict[str, Any] = {}

    def build(self) -> Dict[str, Any]:
        """Build the final, tuned DeepSpeed configuration."""
        self._load_base_config()

        # Chain all the tuning steps
        self._tune_batch_size()
        self._tune_precision()
        self._tune_zero3()
        self._tune_optimizer()
        self._tune_scheduler()
        self._tune_gradient_clipping()
        self._tune_transformer_engine()

        self._validate()
        return self.ds_cfg

    def _load_base_config(self):
        """Load the user-provided base JSON config."""
        config_path = self._get_config_path()
        if config_path is None:
            logger.warning("No DeepSpeed config provided. Using a minimal default.")
            self.ds_cfg = {
                "train_batch_size": "auto",
                "gradient_accumulation_steps": "auto",
            }
            return

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as fp:
            self.ds_cfg = json.load(fp)
        logger.debug("Loaded DeepSpeed config from %s", config_path)

    def _get_config_path(self) -> Optional[str | os.PathLike]:
        """Find the path to the DeepSpeed config file within the main config."""
        if hasattr(self.cfg, "engine") and getattr(self.cfg.engine, "config", None):
            return self.cfg.engine.config
        # Legacy fallbacks for backward compatibility
        if getattr(self.cfg, "deepspeed_config", None):
            return self.cfg.deepspeed_config
        if hasattr(self.cfg, "engine") and getattr(self.cfg.engine, "deepspeed_config", None):
            return self.cfg.engine.deepspeed_config
        return None

    @staticmethod
    def _is_auto(val: Any) -> bool:
        """Return True if value is the special DeepSpeed placeholder 'auto'."""
        return isinstance(val, str) and val.lower() == "auto"

    def _replace_auto(self, val: Any, replacement: Any):
        """Replace the string 'auto' (case-insensitive) with *replacement*."""
        return replacement if self._is_auto(val) else val

    def _tune_batch_size(self):
        """Tune batch size and gradient accumulation settings from myllm config."""
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        cfg = self.cfg.training

        # Automatic BS calculation
        # ---------------------------------
        # train_batch_size = micro_batch_size * gradient_accumulation * world_size
        train_bs = self.ds_cfg["train_batch_size"]
        micro_bs = self.ds_cfg.get("train_micro_batch_size_per_gpu", train_bs)
        grad_accum = self.ds_cfg["gradient_accumulation_steps"]

        # All three are auto -> get from myllm config
        if self._is_auto(train_bs) and self._is_auto(micro_bs) and self._is_auto(grad_accum):
            micro_bs = cfg.micro_batch_size
            grad_accum = cfg.gradient_accumulation_steps
            train_bs = micro_bs * grad_accum * world_size

        # Two are auto -> infer from the third
        elif self._is_auto(train_bs) and self._is_auto(micro_bs):
            grad_accum = int(grad_accum)
            train_bs = micro_bs * grad_accum * world_size
        elif self._is_auto(train_bs) and self._is_auto(grad_accum):
            micro_bs = int(micro_bs)
            train_bs = micro_bs * cfg.gradient_accumulation_steps * world_size
        elif self._is_auto(micro_bs) and self._is_auto(grad_accum):
            train_bs = int(train_bs)
            micro_bs = train_bs // (cfg.gradient_accumulation_steps * world_size)
            grad_accum = cfg.gradient_accumulation_steps

        # One is auto -> infer from the other two
        elif self._is_auto(train_bs):
            train_bs = int(micro_bs) * int(grad_accum) * world_size
        elif self._is_auto(micro_bs):
            micro_bs = int(train_bs) // (int(grad_accum) * world_size)
        elif self._is_auto(grad_accum):
            grad_accum = int(train_bs) // (int(micro_bs) * world_size)

        self.ds_cfg["train_batch_size"] = train_bs
        self.ds_cfg["train_micro_batch_size_per_gpu"] = micro_bs
        self.ds_cfg["gradient_accumulation_steps"] = grad_accum
        logger.debug("Auto-tuned batch size: train_bs=%d, micro_bs=%d, grad_accum=%d", train_bs, micro_bs, grad_accum)

    def _tune_precision(self):
        """Tune fp16/bf16 settings from myllm config."""
        dtype_map = {"bf16": "bf16", "fp16": "fp16"}
        precision = dtype_map.get(self.cfg.model.dtype, "bf16")

        # Ensure sections exist
        if "bf16" not in self.ds_cfg:
            self.ds_cfg["bf16"] = {"enabled": "auto"}
        if "fp16" not in self.ds_cfg:
            self.ds_cfg["fp16"] = {"enabled": "auto"}

        if precision == "bf16":
            self.ds_cfg["bf16"]["enabled"] = True
            self.ds_cfg["fp16"]["enabled"] = False
            logger.debug("Auto-tuned precision: bf16 enabled, fp16 disabled")
        elif precision == "fp16":
            self.ds_cfg["bf16"]["enabled"] = False
            self.ds_cfg["fp16"]["enabled"] = True
            logger.debug("Auto-tuned precision: fp16 enabled, bf16 disabled")

    def _tune_zero3(self):
        """Tune ZeRO Stage 3 specific parameters from myllm config."""
        if self.ds_cfg.get("zero_optimization", {}).get("stage", 0) != 3:
            return
            
        zero3_cfg = self.ds_cfg["zero_optimization"]
        hidden_size = self.model.config.hidden_size
        hidden_layers = self.model.config.num_hidden_layers

        # Reduce bucket size
        if self._is_auto(zero3_cfg.get("reduce_bucket_size")):
            zero3_cfg["reduce_bucket_size"] = hidden_size * hidden_layers
        
        # All-gather bucket size
        if self._is_auto(zero3_cfg.get("stage3_allgather_bucket_size")):
             zero3_cfg["stage3_allgather_bucket_size"] = hidden_size * hidden_layers
        
        if self._is_auto(zero3_cfg.get("stage3_prefetch_bucket_size")):
            zero3_cfg["stage3_prefetch_bucket_size"] = hidden_size * hidden_layers

        # Enable async partitioning for better performance
        if self._is_auto(zero3_cfg.get("overlap_comm")):
            zero3_cfg["overlap_comm"] = True
            logger.debug("Auto-tuned overlap_comm: enabled")

        # Thresholds and limits
        if self._is_auto(zero3_cfg.get("stage3_param_persistence_threshold")):
            zero3_cfg["stage3_param_persistence_threshold"] = 10 * hidden_size
        if self._is_auto(zero3_cfg.get("stage3_max_live_parameters")):
            zero3_cfg["stage3_max_live_parameters"] = 1e9

        logger.debug("Auto-tuned ZeRO-3 parameters")


    def _tune_optimizer(self):
        """Tune optimizer parameters like learning rate and weight decay from myllm config."""
        if "optimizer" not in self.ds_cfg:
            return

        optimizer_cfg = self.ds_cfg["optimizer"]
        training_cfg = self.cfg.training

        # Set AdamW as default. DeepSpeed will use FusedAdam implementation if available.
        if self._is_auto(optimizer_cfg.get("type")):
            optimizer_cfg["type"] = "AdamW"
            logger.debug("Auto-tuned optimizer type: AdamW (will use FusedAdam if available)")

        # Ensure 'params' sub-dictionary exists
        if "params" not in optimizer_cfg:
            optimizer_cfg["params"] = {}
        
        optimizer_params = optimizer_cfg["params"]

        if self._is_auto(optimizer_params.get("lr")):
            optimizer_params["lr"] = training_cfg.lr
        if self._is_auto(optimizer_params.get("weight_decay")):
            optimizer_params["weight_decay"] = getattr(training_cfg, "weight_decay", 0.0)
        if self._is_auto(optimizer_params.get("betas")):
            optimizer_params["betas"] = (0.9, 0.999)
        if self._is_auto(optimizer_params.get("eps")):
            optimizer_params["eps"] = 1e-8
        
        logger.debug("Auto-tuned optimizer parameters")


    def _tune_scheduler(self):
        """Tune scheduler parameters from myllm config."""
        if "scheduler" not in self.ds_cfg:
            return

        scheduler_cfg = self.ds_cfg["scheduler"]
        training_cfg = self.cfg.training

        if "params" not in scheduler_cfg:
            scheduler_cfg["params"] = {}
        
        scheduler_params = scheduler_cfg["params"]

        if self._is_auto(scheduler_params.get("warmup_num_steps")):
            scheduler_params["warmup_num_steps"] = getattr(training_cfg, "warmup_steps", 0)
        
        if self._is_auto(scheduler_params.get("warmup_min_lr")):
            scheduler_params["warmup_min_lr"] = 0
        
        if self._is_auto(scheduler_params.get("warmup_max_lr")):
            scheduler_params["warmup_max_lr"] = training_cfg.lr

        # Special case: total steps for cosine scheduler
        if (
            self.dataloader_len is not None 
            and scheduler_cfg.get("type") in ["WarmupLR", "WarmupDecayLR"]
            and self._is_auto(scheduler_params.get("total_num_steps"))
        ):
            scheduler_params["total_num_steps"] = self.dataloader_len * training_cfg.epochs

        logger.debug("Auto-tuned scheduler parameters")


    def _tune_gradient_clipping(self):
        """Set gradient clipping from myllm config if not set in deepspeed config."""
        if self._is_auto(self.ds_cfg.get("gradient_clipping")):
            self.ds_cfg["gradient_clipping"] = self.cfg.training.gradient_clipping
            logger.debug("Auto-tuned gradient clipping")

    def _tune_transformer_engine(self):
        """Enable Flash Attention if available and set in config."""
        if "transformer_engine" not in self.ds_cfg:
            self.ds_cfg["transformer_engine"] = {"enabled": "auto"}

        if self._is_auto(self.ds_cfg["transformer_engine"].get("enabled")):
            # Enable by default as it's a good practice for modern GPUs
            self.ds_cfg["transformer_engine"]["enabled"] = True
            logger.debug("Auto-tuned transformer_engine: enabled")

        if self.ds_cfg["transformer_engine"]["enabled"]:
            if "transformer_kernel" not in self.ds_cfg:
                self.ds_cfg["transformer_kernel"] = {}
            if "flash_attn" not in self.ds_cfg["transformer_kernel"]:
                self.ds_cfg["transformer_kernel"]["flash_attn"] = True
                logger.debug("Auto-tuned transformer_kernel: flash_attn enabled")

    def _validate(self) -> None:
        """Recursively walk config and raise if any 'auto' placeholders remain."""
        if not (hasattr(self.cfg, "engine") and getattr(self.cfg.engine, "auto_fill", True)):
            return  # User has disabled auto-filling, so we don't validate

        self._assert_no_auto_recursive(self.ds_cfg)

    def _assert_no_auto_recursive(self, cfg: Any, path: str = "root") -> None:
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                self._assert_no_auto_recursive(v, f"{path}.{k}")
        elif isinstance(cfg, list):
            for i, v in enumerate(cfg):
                self._assert_no_auto_recursive(v, f"{path}[{i}]")
        elif self._is_auto(cfg):
            raise ValueError(f"DeepSpeed config still contains 'auto' value at {path}")


def prepare(
    cfg: Any,
    model: torch.nn.Module,
    dataloader_len: int | None = None,
) -> Tuple[Any, torch.nn.Module, torch.optim.Optimizer, Any]:
    """
    Initialize the DeepSpeed engine using the new object-oriented configurator.
    """
    if deepspeed is None:
        raise RuntimeError("DeepSpeed is not installed.")

    # 1. Build the config
    config_builder = DeepSpeedConfigTuner(cfg, model, dataloader_len)
    ds_config = config_builder.build()

    # 2. Prepare environment for single-GPU runs if needed
    if int(os.environ.get("WORLD_SIZE", 1)) <= 1:
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        # If no optimizer is specified, use Adam from training config
        if "optimizer" not in ds_config:
            logger.info("No optimizer specified in DeepSpeed config, creating Adam from main config")
            optim_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                optim_params,
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                fused=True,
            )
        else:
            optimizer = None
    else:
        optimizer = None
    
    # 3. Initialize DeepSpeed
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    logger.info("DeepSpeed engine initialized with new configurator.")
    return engine, model, optimizer, scheduler


def prepare_inference(
    cfg: Any,
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Initialize a model with DeepSpeed for inference.

    This function uses the same core ConfigBuilder but disables the optimizer
    and scheduler to prepare a model for evaluation or as a reference model.
    """
    if deepspeed is None:
        raise RuntimeError("DeepSpeed is not installed.")

    # 1. Build the config, but force optimizer to None
    config_builder = DeepSpeedConfigTuner(cfg, model)
    ds_config = config_builder.build()
    ds_config["optimizer"] = {"type": None}
    ds_config["scheduler"] = {"type": None}

    # If not ZeRO-3, disable ZeRO altogether for inference to be safe
    if ds_config.get("zero_optimization", {}).get("stage", 0) != 3:
        ds_config["zero_optimization"] = {"stage": 0}

    # 2. Initialize DeepSpeed for inference
    engine, *_ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    engine.eval()
    logger.info("DeepSpeed engine initialized for inference.")
    return engine 