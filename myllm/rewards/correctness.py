from __future__ import annotations

"""Correctness reward – rewritten from legacy function into class form."""

from typing import List, Any

from myllm.rewards import BaseReward, register_reward
from myllm.rewards.utils import extract_boxed_answer


@register_reward
class CorrectnessReward(BaseReward):
    """Reward = 2 if extracted answer matches ground truth else 0."""

    name = "correctness_reward"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ------------------------------------------------------------------
    def __call__(self, *, prompts: List[Any], completions: List[Any], answer: List[str], **_):  # noqa: D401
        """Return 2.0 if model answer matches *exactly* the reference one.

        TRL может передавать как «chat»-формат (list[list[dict]]) так и plain strings
        – зависит от того, сочла ли библиотека пример «conversational».  Считаем оба
        варианта.
        """

        # ------------------------------------------------------------------
        # Extract assistant responses
        # ------------------------------------------------------------------
        if completions and isinstance(completions[0], str):
            # Non-chat – each completion already string
            responses = completions  # type: ignore[assignment]
        else:
            # Conversational: list of messages -> take first assistant message
            responses = [c[0]["content"] for c in completions]  # type: ignore[index]

        # ------------------------------------------------------------------
        # (Optional) debug – original question, if available
        # ------------------------------------------------------------------
        if prompts and isinstance(prompts[0], list):
            _question = prompts[0][-1]["content"]  # noqa: F841  # may be useful when verbose

        # ------------------------------------------------------------------
        # Compare extracted boxed answers
        # ------------------------------------------------------------------
        extracted = [extract_boxed_answer(r) for r in responses]
        rewards = [2.0 if e == a else 0.0 for e, a in zip(extracted, answer)]

        return rewards 