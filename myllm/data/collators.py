from __future__ import annotations

"""Custom data collators (ported from myllm v1)."""

from typing import Union, List, Any, Dict
import logging

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

__all__ = [
    "DataCollatorForCompletionOnlyLM",
]

logger = logging.getLogger(__name__)


def _filter_indices(a: List[int], b: List[int]) -> List[int]:
    """Helper copied from v1 `array_utils.filter_indices`."""
    filtered_b = []
    a_len = len(a)
    b_len = len(b)

    j = 0  # Pointer for list b

    for i in range(a_len):
        while j < b_len and b[j] <= a[i]:
            j += 1
        if j < b_len:
            filtered_b.append(b[j])
            j += 1

    return filtered_b


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Collator that masks loss outside assistant replies for chat completion tasks."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_prompt_template: Union[str, List[int]],
        mlm: bool = False,
        ignore_index: int = -100,
        strict: bool = False,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)

        self.response_prompt_template = response_prompt_template
        if isinstance(response_prompt_template, str):
            self.response_token_ids = self.tokenizer.encode(response_prompt_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_prompt_template

        self.eos_token_id = self.tokenizer.eos_token_id
        self.ignore_index = ignore_index
        self.strict = strict

        # Stats
        self.hits: int = 0
        self.misses: int = 0

        if not self.mlm and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logger.warning(
                "pad_token_id = eos_token_id – might cause model to loop on generation; consider different pad_token."
                )

    # transformers >=4.38 uses __call__, earlier uses torch_call; we override both for safety
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]):  # type: ignore
        return self._process(super().torch_call(examples))

    def __call__(self, examples, *args, **kwargs):  # type: ignore
        return self.torch_call(examples)

    # ------------------------------------------------------------------
    def _process(self, batch: Dict[str, Any]):
        """Mask labels to ignore_index outside assistant completion span."""
        for i in range(len(batch["labels"])):
            response_starts = []
            eos_indexes = []

            # find beginnings of response token template
            for idx in torch.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                if self.response_token_ids == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist():
                    response_starts.append(idx.item())

            # find eos indices
            for idx in torch.where(batch["labels"][i] == self.eos_token_id)[0]:
                eos_indexes.append(idx.item())

            eos_indexes = _filter_indices(response_starts, eos_indexes)

            if not response_starts or not eos_indexes:
                msg = f"Could not find response key `{self.response_prompt_template}` in instance; ignoring loss for example."
                if self.strict:
                    raise ValueError(msg)
                logger.warning(msg)
                batch["labels"][i, :] = self.ignore_index
            else:
                new_labels = torch.full_like(batch["labels"][i], self.ignore_index).to(batch["labels"][i].device)
                for start, end in zip(response_starts, eos_indexes):
                    new_labels[start : end + 1] = batch["labels"][i, start : end + 1]
                batch["labels"][i] = new_labels
                self.hits += 1
            if not response_starts or not eos_indexes:
                self.misses += 1
        return batch

    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict[str, int]:  # noqa: D401
        return {"hits": self.hits, "misses": self.misses} 