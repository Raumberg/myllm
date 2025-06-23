from __future__ import annotations

"""Generative Reward Preference Optimization (GRPO) training algorithm.

Thin wrapper around `trl.GRPOTrainer` to fit myllm's BaseTrainer API. The
implementation mirrors `algorithms/sft.py` structure so we have consistent
behaviour across algorithms.
"""

from typing import Any, Optional, Callable, List

from transformers import TrainingArguments
from trl import GRPOTrainer as HF_GRPOTrainer, GRPOConfig

from myllm.algorithms.base import BaseTrainer
from myllm.callbacks.progress import RichProgressCallback
from myllm.callbacks.wandb import WandBCallback
from myllm.rewards import build_rewards

# Optional PEFT imports
try:
    from peft import LoraConfig, TaskType
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore

__all__ = ["GRPOTrainer", "Trainer"]


class GRPOTrainer(BaseTrainer):
    """Wrapper around TRL ``GRPOTrainer``.

    We adapt internal ``Config`` dataclass → ``GRPOConfig`` (which itself is a
    superset of ``TrainingArguments``).
    """

    def __init__(self, model: Any, engine: Optional[Any], cfg: Any):  # noqa: D401
        super().__init__(model, engine, cfg)

        # Build base TrainingArguments via helper
        self.ta = self.build_training_args()

        # Build GRPOConfig – we will just pass through the dict of TrainingArguments
        # but *exclude* keys that GRPOConfig does not accept, e.g. our custom ``reward_funcs`` list.
        raw_extra: dict[str, Any] = getattr(cfg.training, "grpo", {})

        # Pop reward_funcs (handled separately) – keep the rest intact
        extra_args = {k: v for k, v in raw_extra.items() if k != "reward_funcs"}

        # ------------------------------------------------------------------
        # Smart defaults / auto-fixes for GRPOConfig parameters
        # ------------------------------------------------------------------
        # 1. If user specified ``num_generations`` but omitted an explicit
        #    ``generation_batch_size`` (or incompatible effective batch-size),
        #    we adaptively set it so that constraints inside GRPOConfig pass.
        num_gen = extra_args.get("num_generations")
        gen_bs = extra_args.get("generation_batch_size")

        if num_gen is not None and gen_bs is None:
            eff_bs = self.ta.per_device_train_batch_size * self.ta.gradient_accumulation_steps
            # world_size unknown here (handled by Accelerator later) – assume 1
            if eff_bs % num_gen != 0:
                # set to the next multiple
                extra_args["generation_batch_size"] = eff_bs * num_gen

        # 2. Pass-through all other kwargs as-is
        # Disable removal of unused columns so that fields like 'answer' are preserved for reward functions.
        self.ta.remove_unused_columns = False
        extra_args.setdefault("remove_unused_columns", False)

        grpo_kwargs = {**self.ta.to_dict(), **extra_args}
        self.grpo_args = GRPOConfig(**grpo_kwargs)  # type: ignore[arg-type]

        # Build reward functions from config (list or None)
        grpo_cfg = getattr(cfg.training, "grpo", {})
        self.reward_funcs = build_rewards(grpo_cfg.get("reward_funcs"))

        self.trl_trainer: Optional[HF_GRPOTrainer] = None

    def train(self, dataloader, *, resume_from: str | None = None):  # noqa: D401
        # Prepare callbacks
        callbacks = self.default_callbacks(collate_fn=dataloader.collate_fn if hasattr(dataloader, "collate_fn") else None)

        self.trl_trainer = HF_GRPOTrainer(
            model=self.model,
            args=self.grpo_args,
            train_dataset=dataloader.dataset,
            reward_funcs=self.reward_funcs,
            peft_config=self._peft_cfg,
            processing_class=getattr(self.model, "tokenizer", None),
            callbacks=callbacks,
        )

        self.trl_trainer.train(resume_from_checkpoint=resume_from)


# For CLI generic access
Trainer = GRPOTrainer 