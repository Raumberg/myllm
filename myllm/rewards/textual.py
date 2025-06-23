from __future__ import annotations

"""Textual / formatting & language quality rewards.

Fully rewritten from legacy functions into class-based rewards.
"""

import re
from typing import Any, List

from myllm.rewards import BaseReward, register_reward
from myllm.rewards.utils import (
    check_xml_structure,
    get_xml_content,
    get_content,
    count_unique_ngrams,
    detect_reflection_marks,
)

# -----------------------------------------------------------------------------
# Helper utilities within this module
# -----------------------------------------------------------------------------

def _extract_text(comp: Any) -> str:  # noqa: D401
    """Return text content regardless of legacy structure variations."""
    if isinstance(comp, list):
        if comp and isinstance(comp[0], dict):
            return comp[0].get("content", "")
        return str(comp[0]) if comp else ""
    if isinstance(comp, dict):
        return comp.get("content", "")
    return str(comp)


# -----------------------------------------------------------------------------
# 1. FormatReward
# -----------------------------------------------------------------------------

@register_reward
class FormatReward(BaseReward):
    """Reward 1.0 if <think> block present & multi-line, 0.5 if single line, else 0."""

    name = "format_reward"

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        rewards: List[float] = []
        for comp in completions:
            content = _extract_text(comp)
            if check_xml_structure(content, ("think",)):
                rewards.append(1.0 if "\n" in get_xml_content(content, "think") else 0.5)
            else:
                rewards.append(0.0)
        return rewards

# -----------------------------------------------------------------------------
# 2. MultilingualCoherencePenalty
# -----------------------------------------------------------------------------

@register_reward
class MultilingualCoherencePenalty(BaseReward):
    """Negative reward for foreign characters / mixed words / chinese chars."""

    name = "multilingual_coherence_penalty"

    # Compile regex once
    _CHINESE_CHAR = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]")
    _MIXED_ALPHABET = re.compile(
        r"\b(?:[a-zа-яё]*[a-z][a-zа-яё]*[а-яё][a-zа-яё]*|[а-яёa-z]*[а-яё][a-zа-яё]*[a-z][a-zа-яё]*)\b",
        re.IGNORECASE | re.UNICODE,
    )
    _TECH_BLOCKS = re.compile(r"`.*?`|\\[a-z]+|\$.*?\$|http\S+", re.IGNORECASE)

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        penalties: List[float] = []
        for comp in completions:
            text = _extract_text(comp)
            # Remove code/latex/url blocks
            clean = self._TECH_BLOCKS.sub(" ", text)
            penalty = 0.0
            penalty += min(len(self._CHINESE_CHAR.findall(clean)) * 0.1, 0.5)
            mixed = [w for w in self._MIXED_ALPHABET.findall(clean) if not (w.isascii() or w.isalpha())]
            penalty += min(len(mixed) * 0.3, 1.0)
            foreign_seq = re.findall(r"[^\u0400-\u04FF\s]{8,}", clean)
            penalty += min(len(foreign_seq) * 0.2, 0.5)
            penalties.append(round(-min(penalty, 2.0), 2))
        return penalties


# -----------------------------------------------------------------------------
# 3. RussianPurityReward
# -----------------------------------------------------------------------------

@register_reward
class RussianPurityReward(BaseReward):
    """Reward based on proportion of Russian chars (ignores code / formulas)."""

    name = "russian_purity_reward"

    _ALLOWED_PATTERNS = [
        r"<code>.*?</code>",
        r"`.*?`",
        r"```.*?```",
        r"\$.*?\$",
        r"\\[a-zA-Z]+",
        r"\b[a-zA-Z]{1,4}\d*\b",
    ]
    _RUS_CHARS = re.compile(r"[а-яёА-ЯЁ]")

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        rewards: List[float] = []
        for comp in completions:
            text = _extract_text(comp)
            clean = text
            for pat in self._ALLOWED_PATTERNS:
                clean = re.sub(pat, "", clean, flags=re.DOTALL)
            chars = re.findall(r"\S", clean)
            if not chars:
                rewards.append(0.0)
                continue
            rus = self._RUS_CHARS.findall(clean)
            ratio_non = 1 - (len(rus) / len(chars))
            if ratio_non < 0.33:
                r = 1.0
            elif ratio_non < 0.5:
                r = 0.8
            elif ratio_non < 0.75:
                r = 0.5
            else:
                r = 0.0
            rewards.append(r)
        return rewards


# -----------------------------------------------------------------------------
# 4. RedundancyPenaltyReward (native implementation, no legacy)
# -----------------------------------------------------------------------------

@register_reward
class RedundancyPenaltyReward(BaseReward):
    """Penalty based on repetition of 4-grams (closer to 0 unique/total)."""

    name = "redundancy_penalty"

    def __init__(self, n: int = 4, scale: float = 0.5):
        self.n = n
        self.scale = scale

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        penalties: List[float] = []
        for comp in completions:
            text = _extract_text(comp)
            uniq, total = count_unique_ngrams(text, self.n)
            penalty = 0.0 if total == 0 else -(1 - uniq / total) * self.scale
            penalties.append(round(penalty, 3))
        return penalties


# -----------------------------------------------------------------------------
# 5. NgramPenaltyReward / NgramReward
# -----------------------------------------------------------------------------

class _BaseNgram(BaseReward):
    ngram_size: int = 4
    min_safe_ngrams: int = 5

    def _count_penalty(self, text: str):  # noqa: D401
        uniq, total = count_unique_ngrams(text, self.ngram_size)
        safe = uniq >= self.min_safe_ngrams
        return uniq, total, safe


@register_reward
class NgramPenaltyReward(_BaseNgram):
    name = "ngram_penalty"

    def __init__(self, ngram_size: int = 4, max_penalty: float = -1.0, min_safe_ngrams: int = 5):
        self.ngram_size = ngram_size
        self.max_penalty = max_penalty
        self.min_safe_ngrams = min_safe_ngrams

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        outs: List[float] = []
        for comp in completions:
            text = _extract_text(comp)
            uniq, total, _ = self._count_penalty(text)
            if total == 0:
                outs.append(0.0)
                continue
            redundancy = 1 - uniq / total
            outs.append(max(self.max_penalty, -redundancy))
        return outs


@register_reward
class NgramReward(_BaseNgram):
    name = "ngram_reward"

    def __init__(self, ngram_size: int = 4, max_reward: float = 1.0, min_safe_ngrams: int = 5):
        self.ngram_size = ngram_size
        self.max_reward = max_reward
        self.min_safe_ngrams = min_safe_ngrams

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        outs: List[float] = []
        for comp in completions:
            text = _extract_text(comp)
            uniq, total, safe = self._count_penalty(text)
            if not safe or total == 0:
                outs.append(0.0)
                continue
            ratio = uniq / total
            outs.append(min(self.max_reward, ratio))
        return outs


# -----------------------------------------------------------------------------
# 6. ReflectionReward
# -----------------------------------------------------------------------------

@register_reward
class ReflectionReward(BaseReward):
    name = "reflection_reward"

    def __init__(self):
        pass

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        outs: List[float] = []
        for comp in completions:
            text = _extract_text(comp)
            outs.append(round(detect_reflection_marks(text), 3))
        return outs


# -----------------------------------------------------------------------------
# 7. RussianCoherenceReward
# -----------------------------------------------------------------------------

@register_reward
class RussianCoherenceReward(BaseReward):
    name = "russian_coherence_reward"

    _RUS_WORD = re.compile(r"\b[а-яёА-ЯЁ]+\b")
    _NON_RUS_CHAR = re.compile(r"[^а-яёА-ЯЁ]")

    def __call__(self, *, completions: List[Any], **_) -> List[float]:  # noqa: D401
        rewards: List[float] = []
        for comp in completions:
            text = _extract_text(comp)
            words = self._RUS_WORD.findall(text)
            if not words:
                rewards.append(0.0)
                continue
            incorrect = sum(1 for w in words if self._NON_RUS_CHAR.search(w))
            ratio = 1 - incorrect / len(words)
            if ratio > 0.8:
                r = 1.0
            elif ratio > 0.6:
                r = 0.5
            else:
                r = 0.0
            rewards.append(r)
        return rewards 