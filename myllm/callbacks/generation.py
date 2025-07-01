from __future__ import annotations

"""Periodically generate samples during training (stub)."""

from typing import List

from transformers import PreTrainedTokenizerBase

__all__ = ["GenerationCallback"]


class GenerationCallback:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompts: List[str], interval_steps: int = 500):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.interval_steps = interval_steps

    def __call__(self, step: int, model, engine):  # noqa: D401
        if step % self.interval_steps != 0:
            return

        model.eval()
        device = next(model.parameters()).device
        for p in self.prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=64)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            print("\n[GEN]", text)
        model.train() 