from __future__ import annotations

"""Custom data collators (ported from myllm v1)."""

from typing import Union, List, Any, Dict, Optional
import logging
import warnings

import torch
import numpy as np
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

__all__ = [
    "DataCollatorForCompletionOnlyLM",
    "DataCollatorForCompletionOnlyLMV2",
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
    """
    Data collator that does language modeling prediction only on the completion part.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        response_template: str | list[int],
        mlm: bool = False,
        ignore_index: int = -100,
        strict: bool = False,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)

        self.response_template = response_template
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template

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
                msg = f"Could not find response key `{self.response_template}` in instance; ignoring loss for example."
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


# ---------------------------------------------------------------------------
# V2 – smarter completion-only collator (based on TRL's version)
# ---------------------------------------------------------------------------

class DataCollatorForCompletionOnlyLMV2(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only calculated on the completion made by
    the assistant.

    Args:
        response_template (`Union[str, list[int]]`):
            the template form that indicates the start of the response, typically something like '### Response:\n'. It
            can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, list[int]]`):
            the template form that indicates the start of the human instruction, typically something like '###
            Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, list[int]],
        instruction_template: Optional[Union[str, list[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        strict: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        warnings.warn(
            "This class is deprecated and will be removed in version 0.20.0. To train on completion only, please use "
            "the parameter `completion_only_loss` of `SFTConfig` instead.",
            DeprecationWarning,
        )

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free
        self.strict = strict
        self.verbose = verbose

        # ------------------------------------------------------------------
        # Runtime statistics
        # ------------------------------------------------------------------
        self.hits: int = 0      # examples where at least one response span kept
        self.misses: int = 0    # examples where no assistant span found (labels ignored)

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    msg = (
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    if self.verbose:
                        warnings.warn(msg, UserWarning)
                    batch["labels"][i, :] = self.ignore_index
                    self.misses += 1
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
                    self.hits += 1

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    msg = (
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    if self.verbose:
                        logger.warning(msg)
                    batch["labels"][i, :] = self.ignore_index
                    self.misses += 1

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    msg = (
                        f"Could not find instruction key `{self.instruction_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    if self.verbose:
                        logger.warning(msg)
                    batch["labels"][i, :] = self.ignore_index
                    self.misses += 1

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

                # If at least one response span kept, count as hit
                if len(response_token_ids_idxs) > 0:
                    self.hits += 1
                else:
                    self.misses += 1

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and keys to prevent graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0), device=flattened_position_ids.device, dtype=torch.int32
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(), device=flattened_position_ids.device, dtype=torch.int32
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to prevent graph breaks during further computations.
            batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
            batch["max_length_q"] = batch["max_length_k"]

        return batch

    # ------------------------------------------------------------------
    # Utility: public stats
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict[str, int]:  # noqa: D401
        return {"hits": self.hits, "misses": self.misses}