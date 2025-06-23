from __future__ import annotations

"""Classic Knowledge Distillation (KL student ↔ teacher) trainer."""

from typing import Any, Optional

import torch
from torch import nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from myllm.algorithms.base import BaseTrainer

__all__ = ["KDTrainer", "Trainer"]


class _DistillHFTrainer(Trainer):
    """HF Trainer with custom KD loss (KL + optional CE)."""

    def __init__(
        self,
        *args,
        teacher_model: nn.Module,
        temperature: float = 2.0,
        alpha_ce: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False):  # noqa: D401
        labels = inputs.get("labels")

        outputs_student = model(**inputs)
        student_logits = outputs_student.logits  # [B, T, V]

        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
            teacher_logits = outputs_teacher.logits.detach()

        T = self.temperature
        kl_loss = self.kl_loss(
            nn.functional.log_softmax(student_logits / T, dim=-1),
            nn.functional.softmax(teacher_logits / T, dim=-1),
        ) * (T * T)

        if labels is not None:
            ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        else:
            ce_loss = 0.0

        loss = (1 - self.alpha_ce) * kl_loss + self.alpha_ce * ce_loss

        return (loss, outputs_student) if return_outputs else loss


class KDTrainer(BaseTrainer):
    """Knowledge distillation with KL divergence to teacher model."""

    def __init__(self, model, engine, cfg):  # noqa: D401
        super().__init__(model, engine, cfg)

        train_args = self.build_training_args()

        kd_cfg: dict[str, Any] = getattr(cfg.training, "distill", {})
        teacher_path: str | None = kd_cfg.get("teacher_model")
        if teacher_path is None:
            raise ValueError("training.distill.teacher_model must be set for distillation.")

        temperature: float = kd_cfg.get("temperature", 2.0)
        alpha_ce: float = kd_cfg.get("alpha_ce", 0.5)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path)
        teacher_model.requires_grad_(False)

        self.trainer = _DistillHFTrainer(
            model=self.model,
            args=TrainingArguments(**train_args.to_dict()),
            train_dataset=None,  # filled in train()
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            temperature=temperature,
            alpha_ce=alpha_ce,
            data_collator=lambda features: features,
            callbacks=self.default_callbacks(),
        )

    # ------------------------------------------------------------------
    def train(self, dataloader, *, resume_from: Optional[str] = None):  # noqa: D401
        self.trainer.train_dataset = dataloader.dataset
        self.trainer.train(resume_from_checkpoint=resume_from)


# Alias
Trainer = KDTrainer 