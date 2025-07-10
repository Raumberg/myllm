from __future__ import annotations

"""Supervised Fine-Tuning (SFT) algorithm stub."""

from typing import Any, Optional

from transformers import TrainingArguments
from trl import SFTTrainer as HF_SFTTrainer
from trl import SFTConfig

from myllm.algorithms.base import BaseTrainer
from myllm.callbacks.progress import RichProgressCallback
from myllm.callbacks.wandb import WandBCallback
from myllm.utils.lazy import peft

__all__ = ["SFTTrainer"]


class SFTTrainer(BaseTrainer):
    """Thin wrapper around HuggingFace TRL ``SFTTrainer``.

    It adapts our internal ``Config`` dataclass into ``transformers.TrainingArguments``
    and delegates all heavy lifting to TRL. The ``engine`` argument is kept for
    API compatibility but is not used – DeepSpeed/Accelerate are handled
    internally by HF Trainer when the appropriate fields are passed to
    ``TrainingArguments`` (e.g. *deepspeed* json config path).
    """

    def __init__(self, model: Any, engine: Optional[Any], cfg: Any):  # noqa: D401
        super().__init__(model, engine, cfg)

        if cfg.model.cast_to_fp8 and (cfg.model.use_4bit or cfg.model.use_8bit):
            raise ValueError(
                "FP8 training (cast_to_fp8) is mutually exclusive with "
                "bitsandbytes quantization (use_4bit or use_8bit)."
            )

        # build base TrainingArguments via helper
        self.ta = self.build_training_args()

        # get extra kwargs
        raw_extra:  dict[str, Any]   = getattr(cfg.training, "sft", {})
        sft_kwargs: dict[str, Any]  = {**self.ta.to_dict(), **raw_extra}

        # Add max_length for SFTConfig (not supported in TrainingArguments)
        if hasattr(cfg.training, 'max_seq_length'):
            sft_kwargs['max_length'] = cfg.training.max_seq_length
            sft_kwargs['max_seq_length'] = cfg.training.max_seq_length

        # populate SFTConfig with kwargs
        self.sft_args = SFTConfig(**sft_kwargs)  # type: ignore[arg-type]

        # set up TRL trainer, will be initialized on first call to train()
        self.trl_trainer: Optional[HF_SFTTrainer] = None

    # ------------------------------------------------------------------
    def train(self, dataloader, *, resume_from: str | None = None):  # noqa: D401
        """Create underlying TRL trainer on first call and launch training."""

        # Infer tokenizer if possible (not strictly needed for HF Trainer but useful later)
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None and hasattr(dataloader, "dataset"):
            tokenizer = getattr(dataloader.dataset, "tokenizer", None)

        # get callbacks
        callbacks = self.default_callbacks(
            collate_fn=dataloader.collate_fn if hasattr(dataloader, "collate_fn") else None
            )

        # initialize TRL trainer
        self.trl_trainer = HF_SFTTrainer(
            model=self.model,
            args=self.sft_args,
            train_dataset=dataloader.dataset,
            data_collator=dataloader.collate_fn if callable(dataloader.collate_fn) else None,
            processing_class=tokenizer,
            peft_config=self._peft_cfg,
            callbacks=callbacks,
        )

        # ------------------------------------------------------------------
        # Enable global FP8 autocast if requested in config.
        # ------------------------------------------------------------------
        self.trl_trainer.train(resume_from_checkpoint=resume_from)

    # ------------------------------------------------------------------
    def _iteration(self, batch: Any):  # noqa: D401
        raise RuntimeError("_iteration should not be called when using TRL SFTTrainer") 