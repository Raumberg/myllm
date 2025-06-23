from __future__ import annotations

"""Math & QA related reward implementations."""

from typing import List, Any
import re

from myllm.rewards import BaseReward, register_reward

# External deps for LaTeX parsing & verification
from math_verify import LatexExtractionConfig, parse, verify  # type: ignore
from latex2sympy2_extended import NormalizationConfig  # type: ignore

# ------------------------------------------------------------------
@register_reward
class AccuracyReward(BaseReward):
    name = "accuracy_reward"

    def __init__(self):
        # Build extraction config once
        self.extraction_config = LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=True,
                malformed_operators=False,
                basic_latex=True,
                equations=True,
                boxed="all",
                units=True,
            ),
            boxed_match_priority=1,
            try_extract_without_anchor=False,
        )

    def __call__(self, *, prompts: List[Any], completions: List[Any], answer: List[str], **_):  # noqa: D401
        contents = [c[0]["content"] for c in completions]  # type: ignore[index]
        rewards: List[float] = []

        for content, sol in zip(contents, answer):
            gold_parsed = parse(sol, extraction_mode="any_match", extraction_config=[self.extraction_config])
            if not gold_parsed:
                rewards.append(0.0)
                continue
            ans_parsed = parse(content, extraction_mode="any_match", extraction_config=[self.extraction_config])
            ok = bool(ans_parsed and verify(ans_parsed, gold_parsed))
            rewards.append(1.0 if ok else 0.0)
        return rewards


# ------------------------------------------------------------------
@register_reward
class EquationStructureReward(BaseReward):
    name = "equation_structure_reward"

    _BOXED_RE = re.compile(r"\\boxed{\s*([+-]?\d+\.?\d*)\s*}")

    def __call__(self, *, completions: List[Any], **_):  # noqa: D401
        rewards: List[float] = []
        for comp in completions:
            content = comp[0]["content"] if isinstance(comp, list) else comp.get("content", "")  # type: ignore[index]
            rewards.append(0.2 if self._BOXED_RE.search(content) else 0.0)
        return rewards 