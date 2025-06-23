from __future__ import annotations

"""Debug / utility rewards.

Contains helper rewards that don't modify optimisation objective but are
useful for inspecting behaviour during training (e.g. printing
completions).
"""

import json
import os
import random
import logging
from pathlib import Path
from typing import Any, List

# Rely on env vars to identify the primary rank – avoids creating an Accelerator instance
# which would require full Accelerate initialisation at import-time.

from myllm.rewards import BaseReward, register_reward

__all__ = ["LogCompletionsReward"]

logger = logging.getLogger(__name__)


@register_reward
class LogCompletionsReward(BaseReward):
    """Periodically dumps completions to stdout / a file for manual inspection.

    It **always returns 0.0** (no influence on optimisation). Add it to the
    ``reward_funcs`` list with weight ``0`` if you want to be explicit.

    Parameters
    ----------
    every_n_steps: int
        How often (in trainer *calls*, roughly one per gradient step) to log.
    max_samples: int
        How many samples from the current batch to dump.
    output_dir: str | None
        If provided, write a JSONL file ``<output_dir>/completions_log.jsonl``
        containing {step,prompt,completion} objects.
    """

    name = "log_completions_reward"

    def __init__(
        self,
        every_n_steps: int = 50,
        max_samples: int = 3,
        output_dir: str | None = None,
    ) -> None:
        self.every_n_steps = every_n_steps
        self.max_samples = max_samples
        self.output_dir = Path(output_dir) if output_dir else None
        self._step = 0

        # Determine if current process is the main one (rank 0).  We cannot
        # import/instantiate ``accelerate.Accelerator`` here because the
        # library expects a global singleton to be created first in the
        # training script.  Instead we use environment variables that are
        # always set by `accelerate launch` / torchrun / deepspeed.
        self._is_main = int(os.environ.get("RANK", "0")) == 0

        if self.output_dir and self._is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self.output_dir / "completions_log.jsonl", "a", encoding="utf-8")
        else:
            self._file = None

    # ------------------------------------------------------------------
    def __call__(self, *, prompts: List[Any], completions: List[Any], **_):  # noqa: D401
        if self._is_main and (self._step % self.every_n_steps == 0):
            # Flatten conversation prompts if needed
            formatted_prompts: List[str] = []
            if prompts and isinstance(prompts[0], str):
                formatted_prompts = prompts  # type: ignore[assignment]
            else:
                for msg_list in prompts:
                    # Each msg is {role,content}; concatenate for quick view.
                    txt = "\n".join(f"{m['role']}: {m['content']}" for m in msg_list)
                    formatted_prompts.append(txt)

            # Completions may be chat or plain
            formatted_completions: List[str] = []
            if completions and isinstance(completions[0], str):
                formatted_completions = completions  # type: ignore[assignment]
            else:
                for comp in completions:
                    formatted_completions.append(comp[0]["content"])  # type: ignore[index]

            # Pick subset
            idxs = random.sample(range(len(formatted_prompts)), k=min(self.max_samples, len(formatted_prompts)))
            for i in idxs:
                logger.info("\n--- GRPO SAMPLE step=%s idx=%s ---\nPROMPT:\n%s\n--- COMPLETION ---\n%s\n", self._step, i, formatted_prompts[i], formatted_completions[i])

                if self._file:
                    rec = {
                        "step": self._step,
                        "idx": i,
                        "prompt": formatted_prompts[i],
                        "completion": formatted_completions[i],
                    }
                    self._file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._file.flush()

        self._step += 1
        return [0.0] * len(completions)

    # ------------------------------------------------------------------
    def __del__(self):
        if hasattr(self, "_file") and self._file:
            try:
                self._file.close()
            except Exception:  # pragma: no cover
                pass 