from __future__ import annotations

"""DataModule – loads HF dataset, tokenizes and creates PyTorch dataloaders."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
# from jinja2 import Template  # Template may be used in future processors; keep commented

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from myllm.config.schema import DataCfg, TrainingCfg
from myllm.data.processors import get_processor
from myllm.models.tokenizer_wrapper import TokenizerWrapper

import os

__all__ = ["DataModule"]

logger = logging.getLogger(__name__)


class DataModule:
    def __init__(self, data_cfg: DataCfg, training_cfg: TrainingCfg, tokenizer: AutoTokenizer):
        self.data_cfg = data_cfg
        self.training_cfg = training_cfg
        self.tokenizer = tokenizer

        # Activate HF offline mode if requested BEFORE any HF import that triggers network
        if self.data_cfg.offline:
            os.environ["HF_DATASETS_OFFLINE"] = "1"

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def setup(self) -> "DataModule":
        ds_dict = self._load_dataset()

        processor = get_processor(
            self.data_cfg.processor_type, self.data_cfg, self.training_cfg, self.tokenizer
        )

        processed_splits: Dict[str, Any] = {}
        for split_name, split_set in ds_dict.items():
            processed_splits[split_name] = split_set.map(
                processor,  # type: ignore[arg-type]
                num_proc=None,
                remove_columns=split_set.column_names,
            )

        self.train_ds = processed_splits["train"]
        self.eval_ds: Optional[Any] = processed_splits.get("eval")
        self.test_ds: Optional[Any] = processed_splits.get("test")

        # ------------------------------------------------------------------
        # Select and build data collator
        # ------------------------------------------------------------------
        collator_type = self.data_cfg.collator.type.lower()

        # ------------------------------------------------------------------
        # Completion-only
        # ------------------------------------------------------------------
        if collator_type in {"completion_only", "completion"}:
            from myllm.data.collators import DataCollatorForCompletionOnlyLM

            self.collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                response_template=self.data_cfg.collator.template,  # can be None
                ignore_index=self.data_cfg.collator.ignore_index,
                strict=self.data_cfg.collator.strict,
                verbose=getattr(self.data_cfg.collator, "verbose", False),
            )
        # ------------------------------------------------------------------
        # Auto collator will be handled by the trainer itself in the 0.19 trl release
        # https://github.com/huggingface/trl/pull/3586
        # NOTE: Check if this still works and has no conflicts
        # ------------------------------------------------------------------
        elif collator_type in {"auto"}:
            self.collator = None

        # ------------------------------------------------------------------
        # Standard causal-LM collator, return tensors as pt
        # ------------------------------------------------------------------
        else:
            from transformers import DataCollatorForLanguageModeling

            self.collator = DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False, return_tensors="pt"
            )

        # attach chat template if provided
        if self.data_cfg.chat_template and not self.tokenizer.chat_template:
            self.tokenizer.chat_template = self.data_cfg.chat_template

        return self

    # def sync_with_model(self, model: Any) -> "DataModule":
    #     """Convenience chaining method to sync tokenizer with a model."""
    #     self.tokenizer_wrapper.sync_with_model(model)
    #     return self

    def get_train_dataloader(self, shuffle: bool = False, num_workers: int = 2) -> DataLoader:  # noqa: D401
        return DataLoader(
            self.train_ds,
            batch_size=self.training_cfg.micro_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def get_eval_dataloader(self, shuffle: bool = False, num_workers: int = 2) -> Optional[DataLoader]:  # noqa: D401
        if self.eval_ds is None: return None
        return DataLoader(
            self.eval_ds,
            batch_size=self.training_cfg.micro_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def get_test_dataloader(self, shuffle: bool = False, num_workers: int = 2) -> Optional[DataLoader]:  # noqa: D401
        if self.test_ds is None: return None
        return DataLoader(
            self.test_ds,
            batch_size=self.training_cfg.micro_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    def _load_dataset(self) -> Dict[str, Any]:  # noqa: D401
        """Return processed DatasetDict with at least a 'train' split."""

        name = self.data_cfg.name
        p = Path(name)

        try:
            # -----------------------------------------------------
            # Local paths (directory or file)
            # -----------------------------------------------------
            if p.exists():
                if p.is_dir() or self.data_cfg.from_disk:
                    ds = load_from_disk(str(p))
                    ds = self._ensure_dict(ds)
                else:
                    # Heuristic based on extension – json/jsonl/csv
                    ext = p.suffix.lower()
                    if ext in {".jsonl", ".json"}:
                        ds = load_dataset("json", data_files=str(p))
                    elif ext == ".csv":
                        ds = load_dataset("csv", data_files=str(p))
                    else:
                        raise ValueError(f"Unsupported file extension '{ext}' for dataset path '{name}'.")
                    ds = self._ensure_dict(ds)
            # -----------------------------------------------------
            # Remote HF repo – can specify custom split strings
            # -----------------------------------------------------
            else:
                split_spec = self.data_cfg.split or "train"

                # When explicit eval/test splits provided, we can fetch them in one go
                if self.data_cfg.eval_split or self.data_cfg.test_split:
                    split_dict: Dict[str, str] = {"train": split_spec}
                    if self.data_cfg.eval_split:
                        split_dict["eval"] = self.data_cfg.eval_split
                    if self.data_cfg.test_split:
                        split_dict["test"] = self.data_cfg.test_split
                    ds = load_dataset(name, split=split_dict)
                else:
                    ds = load_dataset(name, split=split_spec)
                    ds = self._ensure_dict(ds)

            # ---------------------------------------------
            # Handle automatic split via test_size if needed
            # ---------------------------------------------
            if self.data_cfg.test_size and "eval" not in ds:
                from datasets import DatasetDict  # lazy import

                if not isinstance(ds, dict):
                    raise RuntimeError("Unexpected dataset format after loading – expected dict-like with 'train' key")

                # If dataset is already a DatasetDict we cast to object else wrap
                if isinstance(ds, DatasetDict):
                    train_split = ds["train"]
                else:
                    train_split = ds["train"]

                split = train_split.train_test_split(test_size=self.data_cfg.test_size, seed=42, load_from_cache_file=True)
                ds["train"] = split["train"]
                ds["eval"] = split["test"]

            return ds

        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to load dataset '{name}': {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dict(ds):  # noqa: D401
        """Wrap single `Dataset` into a dict with key 'train' to standardise interface."""
        from datasets import Dataset  # local import to avoid heavy import if not needed

        if isinstance(ds, Dataset):
            return {"train": ds}
        return ds

    # ------------------------------------------------------------------
    # Row-level processing helpers
    # ------------------------------------------------------------------

    # The old generic normalizer/tokenizer helpers are not needed anymore –
    # processor classes encapsulate that logic. Keeping them commented for reference.

    # def _normalize_text(self, raw: Any) -> str:  # noqa: D401
    #     """Convert a raw dataset field value to plain text suitable for tokenization."""
    #
    #     # Plain string
    #     if isinstance(raw, str):
    #         return raw
    #
    #     # Conversation history – list[dict]
    #     if isinstance(raw, list):
    #         try:
    #             return self.tokenizer.apply_chat_template(raw, tokenize=False)
    #         except Exception:
    #             logging.warning("Failed to apply chat template, falling back to naive join")
    #             return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in raw)
    #
    #     # Single message dict
    #     if isinstance(raw, dict):
    #         role = raw.get("role", "user")
    #         return f"{role}: {raw.get('content', '')}"
    #
    #     # Any other type – string representation
    #     return str(raw)
    #
    # def _process_example(self, example: Dict[str, Any]):  # noqa: D401
    #     """Map-style processing used by `datasets.Dataset.map`. Returns tokenized record."""
    #     text = self._normalize_text(example.get(self.data_cfg.text_field, ""))
    #     return self.tokenizer(text, truncation=True, max_length=self.data_cfg.max_length) 