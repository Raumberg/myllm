from __future__ import annotations

"""TokenizerWrapper - proper tokenizer setup and configuration."""

import logging
from typing import Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

__all__ = ["TokenizerWrapper"]

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """High-level wrapper for tokenizer with proper special tokens setup."""

    def __init__(
        self,
        tokenizer_name: str,
        *,
        pad_token: Optional[str] = None,
        add_special_tokens: Optional[dict] = None,
        chat_template: Optional[str] = None,
        model_max_length: Optional[int] = None,
        **hf_kwargs: Any,
    ):
        """Load and properly configure tokenizer.
        
        Args:
            tokenizer_name: HF model name or local path
            pad_token: Explicit pad token to add, e.g. '<|pad|>'
            add_special_tokens: Dict of special tokens to add
            chat_template: Chat template to set
            model_max_length: Max sequence length
            **hf_kwargs: Additional kwargs for AutoTokenizer.from_pretrained
        """
        self.tokenizer_name = tokenizer_name
        self.custom_pad_token = pad_token
        self._tokens_added = 0
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **hf_kwargs)
        
        # Set model_max_length if provided
        if model_max_length is not None:
            self.tokenizer.model_max_length = model_max_length
            
        # Set chat template if provided
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
            
        # Add any custom special tokens first
        if add_special_tokens:
            num_added = self.tokenizer.add_special_tokens(add_special_tokens)
            self._tokens_added += num_added
            if num_added > 0:
                logger.info(f"Added {num_added} special tokens: {add_special_tokens}")
        
        # Setup pad token properly
        self._setup_pad_token()
        
        # Log final configuration
        self._log_tokenizer_info()

    def _setup_pad_token(self):
        """Setup pad_token according to user preference and safety checks."""
        # 1. If user provides a custom pad token, add it and set it.
        if self.custom_pad_token:
            logger.info(f"User provided custom pad token '{self.custom_pad_token}'.")
            num_added = self.tokenizer.add_special_tokens({'pad_token': self.custom_pad_token})
            self._tokens_added += num_added
            return

        # 2. If no custom token, check if one already exists.
        if self.tokenizer.pad_token is not None:
            logger.info(f"Using existing pad_token: '{self.tokenizer.pad_token}'")
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                logger.warning(
                    "Existing pad_token is the same as eos_token. This is common for GPT-like models "
                    "but can cause issues with generation if not handled carefully."
                )
            return

        # 3. If no existing token, try to infer a SAFE one (not EOS).
        logger.info("No pad_token found. Attempting to infer a safe one.")
        if self.tokenizer.unk_token is not None and self.tokenizer.unk_token_id != self.tokenizer.eos_token_id:
            logger.info(f"Setting pad_token to unk_token: '{self.tokenizer.unk_token}'")
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            raise ValueError(
                "Could not automatically set a pad_token that is different from eos_token. "
                "Please provide an explicit pad_token in your config (e.g., data.pad_token='<|pad|>')."
            )

    def _log_tokenizer_info(self):
        """Log tokenizer configuration for debugging."""
        pad_token = self.tokenizer.pad_token
        pad_id = self.tokenizer.pad_token_id
        eos_token = self.tokenizer.eos_token
        eos_id = self.tokenizer.eos_token_id
        
        logger.info("Tokenizer configured:")
        logger.info(f"  - vocab_size: {len(self.tokenizer)}")
        logger.info(f"  - model_max_length: {self.tokenizer.model_max_length}")
        logger.info(f"  - pad_token: '{pad_token}' (id: {pad_id})")
        logger.info(f"  - eos_token: '{eos_token}' (id: {eos_id})")
        
        if pad_id == eos_id:
            logger.warning("⚠️ pad_token_id is the same as eos_token_id. This may be intended for some models, but can cause issues.")

    def sync_with_model(self, model: AutoModelForCausalLM):
        """Sync tokenizer special tokens with model config and resize embeddings if needed."""
        # Sync model config
        if (
            hasattr(model.config, 'pad_token_id') 
            and self.tokenizer.pad_token_id != model.config.pad_token_id
        ):
            model.config.pad_token_id = self.tokenizer.pad_token_id
            logger.debug(f"Synced model.config.pad_token_id = {self.tokenizer.pad_token_id}")
            
        # Sync generation config if it exists
        if hasattr(model, 'generation_config'):
            if (
                hasattr(model.generation_config, 'pad_token_id')
                and model.generation_config.pad_token_id != self.tokenizer.pad_token_id
            ):
                model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            if (
                hasattr(model.generation_config, 'eos_token_id')
                and model.generation_config.eos_token_id != self.tokenizer.eos_token_id
            ):
                model.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
        # Resize embeddings if we added new tokens
        if self._tokens_added > 0:
            model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized model embeddings by {self._tokens_added} tokens.")

        logger.info("Tokenizer synced with model config")

    @property
    def vocab_size_changed(self) -> bool:
        """Check if vocabulary size was changed (need to resize model embeddings)."""
        return self._tokens_added > 0

    def __getattr__(self, name):
        """Delegate unknown attributes to underlying tokenizer."""
        return getattr(self.tokenizer, name) 