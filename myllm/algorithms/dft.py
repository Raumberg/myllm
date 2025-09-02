from __future__ import annotations

"""Dynamic Fine-Tuning (DFT) algorithm built on top of TRL SFTTrainer.

This trainer applies a lightweight, probability-weighted loss modification:

    loss = loss * torch.softmax(shift_logits, dim=-1)\
        .gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()

Implemented by overriding ``compute_loss`` to re-weigh the per-token CE loss.
"""

from typing import Any, Optional

import torch
from transformers import TrainingArguments, AutoTokenizer
from trl.trainer.sft_trainer import SFTTrainer as TRL_SFTTrainer
from trl import SFTConfig

from myllm.algorithms.base import BaseTrainer

__all__ = ["DFTTrainer"]


class HF_DFTTrainer(TRL_SFTTrainer):
    """Thin override of TRL's SFTTrainer to apply DFT loss weighting."""

    def __init__(self, *args, dft_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dft_alpha = dft_alpha

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):  # type: ignore[override]
        # First, delegate to parent to run the forward pass, gather metrics, etc.
        parent = super()
        parent_result = parent.compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if isinstance(parent_result, tuple) and len(parent_result) == 2:
            base_loss, outputs = parent_result
        else:
            # Should not happen with return_outputs=True, but be safe
            base_loss, outputs = parent_result, None

        # If logits/labels are unavailable (e.g., Liger kernel), fall back to the base loss
        if outputs is None or self.args.use_liger_kernel or "labels" not in inputs:
            return parent_result if return_outputs else base_loss

        logits = outputs.logits
        labels = inputs["labels"]

        if logits is None or labels is None:
            return parent_result if return_outputs else base_loss

        # ------------- DFT loss computation -------------
        # Compute per-token cross-entropy without reduction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Build mask for valid tokens (ignore_index = -100)
        valid_mask = shift_labels != -100

        # Flatten for CE computation
        vocab_size = shift_logits.size(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_ce = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        token_ce = token_ce.view(shift_labels.shape)

        probs = torch.softmax(shift_logits, dim=-1)
        
        # Replace -100 (ignore_index) with 0 to avoid gather() out-of-bounds error
        gather_labels = shift_labels.clone()
        gather_labels[~valid_mask] = 0
        
        gold_probs = probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1).detach()

        # DFT weighting: mix original CE with probability-based weight controlled by alpha
        weight = gold_probs * self.dft_alpha + (1.0 - self.dft_alpha)

        # Apply weighting only to valid tokens
        weighted_token_loss = token_ce * weight
        weighted_token_loss = weighted_token_loss.masked_fill(~valid_mask, 0.0)

        # Normalize by number of valid tokens to get a scalar loss
        num_valid = valid_mask.sum().clamp_min(1)
        dft_loss = weighted_token_loss.sum() / num_valid

        if model.training and self.state is not None:
            # Average probability of correct token across valid positions on this device
            self.__log_metrics(valid_mask, gold_probs)

        return (dft_loss, outputs) if return_outputs else dft_loss

    def __log_metrics(self, valid_mask: torch.Tensor, gold_probs: torch.Tensor) -> None:
        with torch.no_grad():
            if valid_mask.any():
                avg_p = (gold_probs[valid_mask]).mean()
                # gather across processes for global avg
                try:
                    avg_p_all = self.accelerator.gather_for_metrics(avg_p)
                    avg_p_val = avg_p_all.mean().item()
                except Exception:
                    avg_p_val = avg_p.item()
                
                if (self.state.global_step % self.args.logging_steps) == 0:
                    self.log({"avgP": avg_p_val})


class DFTTrainer(BaseTrainer):
    """DFT algorithm trainer wrapping TRL SFT with custom loss weighting."""

    def __init__(self, model: Any, engine: Optional[Any], cfg: Any, tokenizer: AutoTokenizer):  # noqa: D401
        super().__init__(model, engine, cfg, tokenizer)

        if cfg.model.cast_to_fp8 and (cfg.model.use_4bit or cfg.model.use_8bit):
            raise ValueError(
                "FP8 training (cast_to_fp8) is mutually exclusive with "
                "bitsandbytes quantization (use_4bit or use_8bit)."
            )

        # Read DFT alpha hyperparameter (0..1)
        self.dft_alpha: float = getattr(cfg.training, "dft_alpha", 1.0)

        # Build base TrainingArguments via helper
        self.ta = self.build_training_args()

        # Get extra kwargs for SFT
        raw_extra: dict[str, Any] = getattr(cfg.training, "sft", {})
        sft_kwargs: dict[str, Any] = {**self.ta.to_dict(), **raw_extra}

        # Add max_length for SFTConfig (not supported in TrainingArguments)
        if hasattr(cfg.training, "max_seq_length"):
            sft_kwargs["max_length"] = cfg.training.max_seq_length

        # Populate SFTConfig with kwargs
        self.sft_args = SFTConfig(**sft_kwargs)  # type: ignore[arg-type]

        # Underlying TRL trainer will be created on first call to train()
        self.trl_trainer: Optional[HF_DFTTrainer] = None

    def train(self, dataloader, *, resume_from: str | None = None):  # noqa: D401
        """Create underlying TRL trainer on first call and launch training."""

        callbacks = self.default_callbacks(
            collate_fn=dataloader.collate_fn if hasattr(dataloader, "collate_fn") else None
        )

        # Initialize TRL trainer with our DFT override
        self.trl_trainer = HF_DFTTrainer(
            model=self.model,
            args=self.sft_args,
            train_dataset=dataloader.dataset,
            data_collator=dataloader.collate_fn if callable(dataloader.collate_fn) else None,
            processing_class=self.tokenizer,
            peft_config=self._peft_cfg,
            dft_alpha=self.dft_alpha,
            callbacks=callbacks,
        )

        self.trl_trainer.train(resume_from_checkpoint=resume_from)

    def _iteration(self, batch: Any):  # noqa: D401
        raise RuntimeError("_iteration should not be called when using TRL SFTTrainer")


