from __future__ import annotations

"""Common training loop skeleton (engine-agnostic)."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
import os

from transformers import TrainingArguments, AutoTokenizer

from myllm.utils.lazy import peft, bitsandbytes
from myllm.utils.std import infer_dtype

__all__ = ["BaseTrainer"]


class BaseTrainer(ABC):
    """Common facilities shared by all algorithm-specific trainers."""

    def __init__(self, model: torch.nn.Module, engine: Any, cfg: Any, tokenizer: AutoTokenizer):  # noqa: D401
        self.model = model
        self.engine = engine  # DeepSpeedEngine or Accelerator wrapper (can be None)
        self.cfg = cfg

        self.model_cfg = self.cfg.model
        self.train_cfg = self.cfg.training
        self.engine_cfg = self.cfg.engine

        self.tokenizer = tokenizer

        # Shared utilities -------------------------------------------------
        self._setup_wandb_env()
        self._peft_cfg = self._build_peft_cfg()

    # ------------------------------------------------------------------
    # Public helper – build transformers.TrainingArguments in one call
    # ------------------------------------------------------------------
    def build_training_args(self, **extra: Dict[str, Any]) -> TrainingArguments:  # noqa: D401
        """Create `TrainingArguments` from config with optional *extra* overrides."""

        # get dtype flags
        fp16, bf16 = self._dtype_flags(self.model_cfg.dtype)

        # Ensure numeric LR
        lr_val = self._ensure_numeric_lr(self.train_cfg.lr)

        # Determine local_rank for distributed runs so that TrainingArguments.device maps correctly.
        local_rank_env = int(os.environ.get("LOCAL_RANK", -1))

        # Using AdamW 8bit optimizer (Does not work for now)
        # if self.train_cfg.optimizer_type == "AdamW8bit": 

            # --- GPU VRAM USAGE --- # 
            # Tests for 2xH100:
            # Full finetuning of 8B Model
            # Original fused AdamW:       |      68.785GB   VRAM each
            # BNB 8bit AdamW:             |      ???        VRAM each
            # --- ************** --- #

            # TODO: check how it affects embeddings, bcs bnb.nn.StableEmbedding is recommended
            # optimizer = bitsandbytes.optim.Adam(
            #     self.model.parameters(),
            #     lr=lr_val,
            #     betas=(0.9, 0.995),
            #     eps=1e-8,
            # )
            # TODO: probably should add Manager.override_config for different layers to use precise adamw instead of 8bit
            # TODO: probably also dig towards paged_optim, but is it even supported in deepspeed?.. fuck me
        # else:
        #     optimizer = self.train_cfg.optimizer_type or "adamw_torch_fused"

        # derive base kwargs
        base_kwargs: Dict[str, Any] = dict(
            output_dir=self.train_cfg.output_dir,
            per_device_train_batch_size=self.train_cfg.micro_batch_size,
            per_device_eval_batch_size=self.train_cfg.micro_batch_size,
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            num_train_epochs=self.train_cfg.epochs,
            learning_rate=lr_val,
            fp16=fp16,
            bf16=bf16,
            seed=self.train_cfg.seed,
            report_to=(["wandb"] if self.cfg.wandb.enable else [self.cfg.data.report_to]),
            logging_steps=self.train_cfg.logging_steps,
            logging_dir=self.train_cfg.run_dir,
            gradient_checkpointing=self.train_cfg.gradient_checkpointing,
            local_rank=local_rank_env,
            disable_tqdm=self.cfg.logging.disable_tqdm,
            use_liger_kernel=self.train_cfg.use_liger_kernel,
            save_strategy=self.train_cfg.save_strategy,
            save_steps=self.train_cfg.save_steps
            # optim=self.train_cfg.optimizer_type # this shit no more accepts pure optim class, or maybe I'm dumb and didn't figure it out yet
        )

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
        if not self.model_cfg.use_peft:
            return None

        return peft.LoraConfig(
            r=self.model_cfg.lora_r,
            lora_alpha=self.model_cfg.lora_alpha,
            lora_dropout=self.model_cfg.lora_dropout,
            bias="none",
            target_modules=self.model_cfg.lora_target_modules,
            task_type=self.model_cfg.lora_task_type,
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