from __future__ import annotations

"""Lightweight WandB callback that logs loss/learning-rate each log step.

We rely on transformers integrated WandB but add model size & run summary at end.
"""

from typing import Any, Dict
import wandb
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer

__all__ = ["WandBCallback"]


class WandBCallback(TrainerCallback):
    """Extra logging on top of HF built-in WandB integration."""

    def on_train_begin(self, 
                       args: TrainingArguments, 
                       state: TrainerState, 
                       control: TrainerControl, 
                       **kwargs
                       ):  # noqa: D401
        # Ensure wandb run exists – transformers may create it later, so we fall back gracefully.
        if not wandb.run:
            return

        model = kwargs.get("model")
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.run.summary["trainable_params"] = total_params

    def on_log(self, 
               args: TrainingArguments, 
               state: TrainerState, 
               control: TrainerControl, 
               logs: Dict[str, float], 
               **kwargs):  # noqa: D401
        if not wandb.run:
            return
        wandb.log(logs, step=state.global_step)

    def on_train_end(self, 
                     args: TrainingArguments, 
                     state: TrainerState, 
                     control: TrainerControl, 
                     **kwargs
                     ):  # noqa: D401
        wandb.finish() 