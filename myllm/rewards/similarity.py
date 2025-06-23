from __future__ import annotations

"""Similarity-based rewards (TF-IDF, DRAMA embeddings, etc.)."""

from typing import List, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from myllm.rewards import BaseReward, register_reward


# -----------------------------------------------------------------------------
# Simple TF-IDF similarity reward
# -----------------------------------------------------------------------------


@register_reward
class SimilarityReward(BaseReward):
    """Discrete reward tiers based on TF-IDF cosine similarity."""

    name = "similarity_reward"

    def __init__(self, thresholds: Optional[List[float]] = None, values: Optional[List[float]] = None, verbose: bool = False):
        # thresholds should be ascending list of similarity cut-offs
        self.thresholds = thresholds or [0.8, 0.6, 0.3]
        self.values = values or [1.0, 0.5, 0.1, 0.0]
        self.verbose = verbose

    def __call__(self, *, prompts: List[Any], completions: List[Any], reference_answers: List[str], **_) -> List[float]:  # noqa: D401
        texts = [c[0]["content"] for c in completions]  # type: ignore[index]
        all_texts = texts + reference_answers
        tfidf = TfidfVectorizer().fit_transform(all_texts)
        sim_matrix = cosine_similarity(tfidf)

        rewards: List[float] = []
        n = len(texts)
        for i, sim in enumerate(sim_matrix[:n, n:][:, 0]):
            for t, v in zip(self.thresholds, self.values):
                if sim >= t:
                    reward = v
                    break
            else:
                reward = self.values[-1]
            rewards.append(reward)

        return rewards


# -----------------------------------------------------------------------------
# Semantic similarity via DRAMA embeddings (if available)
# -----------------------------------------------------------------------------


@register_reward
class SemanticSimilarityReward(BaseReward):
    name = "semantic_similarity_reward"

    def __init__(self, tiers: Optional[List[float]] = None, rewards: Optional[List[float]] = None, verbose: bool = False):
        self.tiers = tiers or [0.33, 0.6, 0.8]
        self.rewards = rewards or [0.0, 1.0, 1.5, 2.0]
        self.verbose = verbose

        # Lazy import DRAMA; if missing we fallback to zeros
        try:
            from src.reward_models.drama import DRAMAModel  # type: ignore

            drama = DRAMAModel.get_instance()
            self.model = drama["model"]
            self.tokenizer = drama["tokenizer"]
            self.device = drama["device"]
            self.available = True
        except Exception as e:  # pragma: no cover
            print("[SemanticSimilarityReward] DRAMA model unavailable:", e)
            self.available = False

    def _score(self, generated: List[str], reference: List[str]) -> np.ndarray:  # noqa: D401
        import torch

        with torch.no_grad():
            gen_emb = self.model.encode_documents(self.tokenizer, generated)  # type: ignore[attr-defined]
            ref_emb = self.model.encode_documents(self.tokenizer, reference)  # type: ignore[attr-defined]
            sims = (gen_emb @ ref_emb.T).diagonal().cpu().numpy()
        return sims

    def __call__(self, *, prompts: List[Any], completions: List[Any], reference_answers: List[str] | None = None, **_) -> List[float]:  # noqa: D401
        if reference_answers is None or len(reference_answers) != len(completions):
            return [0.0] * len(completions)

        if not self.available:
            return [0.0] * len(completions)

        generated = [c[0]["content"] for c in completions]  # type: ignore[index]
        sims = self._score(generated, reference_answers)

        out: List[float] = []
        for i, sim in enumerate(sims):
            if sim < self.tiers[0]:
                r = self.rewards[0]
            elif sim < self.tiers[1]:
                r = self.rewards[1]
            elif sim < self.tiers[2]:
                r = self.rewards[2]
            else:
                r = self.rewards[3]
            out.append(r)

        return out 