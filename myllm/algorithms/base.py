from __future__ import annotations

"""Common training loop skeleton (engine-agnostic)."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import os

from transformers import TrainingArguments

from myllm.utils.lazy import peft
from myllm.utils.std import infer_dtype

__all__ = ["BaseTrainer"]


class BaseTrainer(ABC):
    """Common facilities shared by all algorithm-specific trainers."""

    def __init__(self, model: torch.nn.Module, engine: Any, cfg: Any):  # noqa: D401
        self.model = model
        self.engine = engine  # DeepSpeedEngine or Accelerator wrapper (can be None)
        self.cfg = cfg

        # Resolve output dir early so all helpers can reuse it
        self.output_dir: str = getattr(cfg.training, "output_dir", "experiments")

        # Shared utilities -------------------------------------------------
        self._setup_wandb_env()
        self._peft_cfg = self._build_peft_cfg()

    # ------------------------------------------------------------------
    # Public helper – build transformers.TrainingArguments in one call
    # ------------------------------------------------------------------
    def build_training_args(self, **extra: Dict[str, Any]) -> TrainingArguments:  # noqa: D401
        """Create `TrainingArguments` from config with optional *extra* overrides."""

        # get all configs
        train_cfg = self.cfg.training
        model_cfg = self.cfg.model
        engine_cfg = self.cfg.engine

        # get dtype flags
        fp16, bf16 = self._dtype_flags(model_cfg.dtype)

        # Ensure numeric LR
        lr_val = self._ensure_numeric_lr(train_cfg.lr)

        # Determine local_rank for distributed runs so that TrainingArguments.device maps correctly.
        local_rank_env = int(os.environ.get("LOCAL_RANK", -1))

        # derive base kwargs
        base_kwargs: Dict[str, Any] = dict(
            output_dir=self.output_dir,
            per_device_train_batch_size=train_cfg.micro_batch_size,
            per_device_eval_batch_size=train_cfg.micro_batch_size,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            num_train_epochs=train_cfg.epochs,
            learning_rate=lr_val,
            fp16=fp16,
            bf16=bf16,
            seed=train_cfg.seed,
            report_to=(["wandb"] if self.cfg.wandb.enable else ["none"]),
            logging_steps=train_cfg.logging_steps,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
            local_rank=local_rank_env,
            disable_tqdm=self.cfg.logging.disable_tqdm,
            use_liger_kernel=train_cfg.use_liger_kernel,
        )

        # add deepspeed config if deepspeed engine is used
        if engine_cfg.name == "deepspeed" and getattr(engine_cfg, "config", None):
            base_kwargs["deepspeed"] = str(engine_cfg.config)

        # add extra kwargs if any 
        base_kwargs.update(extra)

        return TrainingArguments(**base_kwargs)

    # ------------------------------------------------------------------
    # Abstract method to be implemented by subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def train(self, dataloader: DataLoader, *, resume_from: str | None = None):  # noqa: D401
        """Launch training loop – must be implemented by subclasses."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dtype_flags(dtype: str):  # noqa: D401
        return (infer_dtype(dtype) == torch.float16, infer_dtype(dtype) == torch.bfloat16)

    def _setup_wandb_env(self):  # noqa: D401
        wb = self.cfg.wandb
        if not wb.enable:
            return

        os.environ.setdefault("WANDB_PROJECT", wb.project)
        if wb.entity:
            os.environ.setdefault("WANDB_ENTITY", wb.entity)
        if wb.name:
            os.environ.setdefault("WANDB_NAME", wb.name)
        if wb.resume_id:
            os.environ.setdefault("WANDB_RUN_ID", wb.resume_id)

    def _build_peft_cfg(self):  # noqa: D401
        mc = self.cfg.model
        if not getattr(mc, "use_peft", False):
            return None

        task_type = getattr(peft.TaskType, mc.lora_task_type, peft.TaskType.CAUSAL_LM)

        return peft.LoraConfig(
            r=mc.lora_r,
            lora_alpha=mc.lora_alpha,
            lora_dropout=mc.lora_dropout,
            bias="none",
            target_modules=mc.lora_target_modules,
            task_type=task_type,
        )

    # ------------------------------------------------------------------
    # Abstract method to be used by custom trainers (no huggingface)
    # ------------------------------------------------------------------
    def _iteration(self, batch: Any):  # noqa: D401
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Default callbacks helper
    # ------------------------------------------------------------------

    def default_callbacks(self, *, collate_fn=None):  # noqa: D401
        from myllm.callbacks.progress import RichProgressCallback
        from myllm.callbacks.wandb import WandBCallback
        from myllm.callbacks.gpu_stats import GpuStatsCallback

        cbs = []
        if getattr(self.cfg.logging, "level", "info").lower() != "debug":
            cbs.append(RichProgressCallback(collator=collate_fn))
        if self.cfg.wandb.enable:
            cbs.append(WandBCallback())
        cbs.append(GpuStatsCallback())
        return cbs 

    def _ensure_numeric_lr(self, lr: Any) -> float:
        if isinstance(lr, str):
            try:
                lr = float(lr)
            except ValueError:
                raise ValueError(f"training.lr must be float, got '{lr}'")
        return lr