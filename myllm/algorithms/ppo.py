from __future__ import annotations

"""PPO fine-tuning (RLHF) using HuggingFace TRL `PPOTrainer`."""

from typing import Any, Optional

from trl import PPOTrainer as HF_PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl.models import AutoModelForCausalLMWithValueHead

try:
    from peft import LoraConfig, TaskType  # type: ignore
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore

from myllm.algorithms.base import BaseTrainer

__all__ = ["PPOTrainer", "Trainer"]

class PPOTrainer(BaseTrainer):
    """Wrap TRL ``PPOTrainer`` into our generic interface."""

    def __init__(self, model, engine, cfg):  # noqa: D401
        super().__init__(model, engine, cfg)

        # ------------------------------------------------------------------
        # Build TrainingArguments base via helper
        # ------------------------------------------------------------------
        self.ta = self.build_training_args()
        # Ensure unused columns kept – we might need extra fields for rewards
        self.ta.remove_unused_columns = False

        # ------------------------------------------------------------------
        # Merge user-supplied PPO-specific kwargs
        # ------------------------------------------------------------------
        raw_extra: dict[str, Any] = getattr(cfg.training, "ppo", {})
        ppo_kwargs = {**self.ta.to_dict(), **raw_extra}

        self.ppo_args = PPOConfig(**ppo_kwargs)  # type: ignore[arg-type]

        # Keep tokenizer handy for later
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

        # Stash extra paths for reward / reference models (loaded lazily in train())
        self._reward_model_path: str | None = raw_extra.get("reward_model_path")
        self._ref_model_path: str | None = raw_extra.get("ref_model_path")

    # ------------------------------------------------------------------
    def _load_reward_model(self) -> Any:  # noqa: D401
        if self._reward_model_path is None:
            raise ValueError("training.ppo.reward_model_path must be set – PPO needs a reward model.")
        return AutoModelForSequenceClassification.from_pretrained(self._reward_model_path)

    def _load_ref_model(self) -> Optional[Any]:  # noqa: D401
        if self._ref_model_path is None:
            return None
        try:
            return AutoModelForCausalLMWithValueHead.from_pretrained(self._ref_model_path)
        except Exception:
            # Fallback to plain CausalLM if value-head variant unavailable
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(self._ref_model_path)

    def _build_value_model(self) -> Any:  # noqa: D401
        # Start from the same base model weights and add value head
        return AutoModelForCausalLMWithValueHead.from_pretrained(self.cfg.model.name)

    # ------------------------------------------------------------------
    def train(self, dataloader, *, resume_from: str | None = None):  # noqa: D401
        callbacks = self.default_callbacks(collate_fn=dataloader.collate_fn if hasattr(dataloader, "collate_fn") else None)

        # Lazily load heavy models right before trainer init to save memory during CLI startup
        reward_model = self._load_reward_model()
        ref_model = self._load_ref_model()
        value_model = self._build_value_model()

        trl_trainer = HF_PPOTrainer(
            model=self.model,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            args=self.ppo_args,
            train_dataset=dataloader.dataset,
            tokenizer=self.tokenizer,
            peft_config=self._peft_cfg,
            callbacks=callbacks,
        )

        trl_trainer.train(resume_from_checkpoint=resume_from)

# ------------------------------------------------------------------
# Expose in module exports
# ------------------------------------------------------------------


# Alias for CLI discovery
Trainer = PPOTrainer 