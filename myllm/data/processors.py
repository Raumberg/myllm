from __future__ import annotations

"""Row processors for various training paradigms (ported from myllm v1).

Each processor takes a row (dict), config objects and tokenizer, and returns
a tokenized dict (input_ids, attention_mask, labels).
"""

from typing import Any, Dict, Protocol, Callable, List
import logging
import random

from transformers import AutoTokenizer

from myllm.config.schema import DataCfg, TrainingCfg

__all__ = [
    "BaseProcessor",
    "DefaultProcessor",
    "HistoryProcessor",
    "GRPOProcessor",
    "PairProcessor",
    "get_processor",
]

logger = logging.getLogger(__name__)


class BaseProcessor(Protocol):
    """Processor protocol – any callable(str) returning tokenized dict."""

    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
        ...


class _ProcessorImpl:
    def __init__(self, data_cfg: DataCfg, training_cfg: TrainingCfg, tokenizer: AutoTokenizer):
        self.data_cfg = data_cfg
        self.processor_cfg = data_cfg.processor
        self.training_cfg = training_cfg
        self.tokenizer = tokenizer

        # Prepare system message list once
        self._system_message = (
            [{"role": "system", "content": data_cfg.system_prompt}] if data_cfg.system_prompt else []
        )

    # Helper -------------------------------------------------------------
    def _merge_system(self, messages: List[Dict[str, str]]):
        """Insert/merge system prompt depending on model capability."""
        if not self.data_cfg.system_prompt:
            return messages

        if self.data_cfg.model_support_system_role:
            return self._system_message + messages

        # Otherwise merge into first user message
        messages = list(messages)  # copy
        if messages and messages[0]["role"] == "user":
            messages[0]["content"] = f"{self.data_cfg.system_prompt}\n" + messages[0]["content"]
        else:
            # If first message is assistant or something else, convert system to user
            messages.insert(0, {"role": "user", "content": self.data_cfg.system_prompt})
        return messages

    # --------------------------------------------------------------------
    def _apply(self, messages: List[Dict[str, str]], *, add_generation_prompt: bool = False, thinking: bool = False):
        """Apply tokenizer chat template and return tokenized dict."""
        constructed = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=thinking,
        )

        # Remove duplicated BOS token if present
        if self.tokenizer.bos_token and constructed.startswith(self.tokenizer.bos_token):
            constructed = constructed[len(self.tokenizer.bos_token) :]

        return self.tokenizer(
            constructed,
            truncation=True,
            padding=False,
            max_length=self.data_cfg.max_length,
        )


class DefaultProcessor(_ProcessorImpl):
    """Single message processor (user prompt, optional system)."""

    def __call__(self, row: Dict[str, Any]):  # noqa: D401
        message = row[self.processor_cfg.text_field]
        if isinstance(message, list):
            message = message[-1]  # Use last message in conversation

        messages = self._merge_system([message])
        return self._apply(messages, add_generation_prompt=False)


class HistoryProcessor(_ProcessorImpl):
    """Conversation history processor: keeps entire list, optional truncation."""

    def __init__(self, *args, history_turns: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_turns = history_turns

    def __call__(self, row: Dict[str, Any]):  # noqa: D401
        history: List[Dict[str, str]] = row[self.processor_cfg.text_field]

        # Optionally truncate to last k turns
        if self.history_turns is not None and len(history) > self.history_turns:
            history = history[-self.history_turns :]

        messages = self._merge_system(history)
        return self._apply(messages, add_generation_prompt=False)


class GRPOProcessor(_ProcessorImpl):
    """Processor for GRPO tasks that adds reflection and extracts answers."""

    def __call__(self, row: Dict[str, Any]):  # noqa: D401
        reflection_chance = getattr(self.processor_cfg, "reflection_chance", 0.0)
        reflection_prompt = getattr(self.processor_cfg, "reflection_prompt", None)

        system_prompt = self.data_cfg.system_prompt or ""

        # Maybe add reflection prompt
        if reflection_chance > 0 and reflection_prompt:
            if random.random() < reflection_chance:
                system_prompt = f"{system_prompt}\n\n{reflection_prompt}".strip()

        result = {
            "prompt": [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': row[self.processor_cfg.problem_field]} 
            ],
            "answer": row[self.processor_cfg.answer_field]
        }

        return result


class PairProcessor(_ProcessorImpl):
    """Simple prompt-answer conversation builder for SFT.

    Expects two textual fields in each row: ``problem_field`` (prompt) and
    ``answer_field`` (assistant response). Constructs a minimal two-turn chat
    and tokenizes via ``apply_chat_template``.
    """

    def __call__(self, row: Dict[str, Any]):  # noqa: D401
        prompt = row[self.processor_cfg.problem_field]
        answer = row[self.processor_cfg.answer_field]

        messages = self._merge_system([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])

        return self._apply(messages, add_generation_prompt=False)


def get_processor(name: str, data_cfg: DataCfg, training_cfg: TrainingCfg, tokenizer: AutoTokenizer):
    name = name.lower()
    if name == "default":
        return DefaultProcessor(data_cfg, training_cfg, tokenizer)
    if name == "history":
        return HistoryProcessor(data_cfg, training_cfg, tokenizer)
    if name == "grpo":
        return GRPOProcessor(data_cfg, training_cfg, tokenizer)
    if name in "pair":
        return PairProcessor(data_cfg, training_cfg, tokenizer)

    raise ValueError(f"Unknown processor_type '{name}'") 