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
        pad_token_strategy: str = "unk_fallback_eos",
        add_special_tokens: Optional[dict] = None,
        chat_template: Optional[str] = None,
        model_max_length: Optional[int] = None,
        **hf_kwargs: Any,
    ):
        """Load and properly configure tokenizer.
        
        Args:
            tokenizer_name: HF model name or local path
            add_special_tokens: Dict of special tokens to add
            chat_template: Chat template to set
            model_max_length: Max sequence length
            **hf_kwargs: Additional kwargs for AutoTokenizer.from_pretrained
        """
        self.tokenizer_name = tokenizer_name
        self.pad_token_strategy = pad_token_strategy
        
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
            self._tokens_added = num_added
            if num_added > 0:
                logger.info(f"Added {num_added} special tokens: {add_special_tokens}")
        
        # Setup pad token properly
        self._setup_pad_token()
        
        # Log final configuration
        self._log_tokenizer_info()

    def _setup_pad_token(self):
        """Setup pad_token according to strategy."""
        if self.tokenizer.pad_token is not None:
            logger.info(f"Pad token already exists: {self.tokenizer.pad_token}")
            return
            
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logger.warning("⚠️  CRITICAL: pad_token_id == eos_token_id! This WILL cause training issues!")
            raise ValueError("pad_token_id == eos_token_id! Watch out for training process.")

    def _log_tokenizer_info(self):
        """Log tokenizer configuration for debugging."""
        logger.info(f"Tokenizer configured:")
        logger.info(f"  - vocab_size: {len(self.tokenizer)}")
        logger.info(f"  - model_max_length: {self.tokenizer.model_max_length}")
        logger.info(f"  - pad_token: '{self.tokenizer.pad_token}' (id: {self.tokenizer.pad_token_id})")
        logger.info(f"  - eos_token: '{self.tokenizer.eos_token}' (id: {self.tokenizer.eos_token_id})")
        logger.info(f"  - bos_token: '{self.tokenizer.bos_token}' (id: {self.tokenizer.bos_token_id})")
        logger.info(f"  - unk_token: '{self.tokenizer.unk_token}' (id: {self.tokenizer.unk_token_id})")
        
        # Critical warning
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logger.warning("⚠️  CRITICAL: pad_token_id == eos_token_id! This WILL cause training issues!")

    def sync_with_model(self, model: AutoModelForCausalLM):
        """Sync tokenizer special tokens with model config."""
        # Sync model config
        if (
            hasattr(model.config, 'pad_token_id') 
            and self.tokenizer.pad_token_id != model.config.pad_token_id
        ):
            model.config.pad_token_id = self.tokenizer.pad_token_id
            logger.debug(f"Synced model.config.pad_token_id = {self.tokenizer.pad_token_id}")
            
        if (
            hasattr(model.config, 'eos_token_id')
            and model.config.eos_token_id != self.tokenizer.eos_token_id
        ):
            model.config.eos_token_id = self.tokenizer.eos_token_id
            
        if (
            hasattr(model.config, 'bos_token_id')
            and model.config.bos_token_id != self.tokenizer.bos_token_id
        ):
            model.config.bos_token_id = self.tokenizer.bos_token_id
        
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
            if (
                hasattr(model.generation_config, 'bos_token_id')
                and model.generation_config.bos_token_id != self.tokenizer.bos_token_id
            ):
                model.generation_config.bos_token_id = self.tokenizer.bos_token_id
            logger.debug("Synced generation_config with tokenizer")

        if (
            hasattr(self, '_tokens_added')
            and self._tokens_added > 0
        ):
            model.resize_token_embeddings(len(self.tokenizer))
            logger.info("Resized model embeddings due to added tokens")

        logger.info("Tokenizer synced with model config")

    @property
    def vocab_size_changed(self) -> bool:
        """Check if vocabulary size was changed (need to resize model embeddings)."""
        # Simple heuristic: if we added tokens, vocab size changed
        return hasattr(self, '_tokens_added') and self._tokens_added > 0

    def __getattr__(self, name):
        """Delegate unknown attributes to underlying tokenizer."""
        return getattr(self.tokenizer, name) 