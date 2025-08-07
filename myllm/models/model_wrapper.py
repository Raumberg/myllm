from __future__ import annotations

"""Simple HuggingFace causal LM wrapper (LoRA/quant aware)."""

from typing import Any
import warnings
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from myllm.utils.lazy import bitsandbytes
from myllm.config.schema import ModelCfg

__all__ = ["ModelWrapper"]
logger = logging.getLogger(__name__)


class ModelWrapper:
    """High-level convenience wrapper for causal language models."""

    def __init__(self, model_cfg: ModelCfg, **hf_kwargs: Any):
        """Load tokenizer & model with optional 4-/8-bit quantisation (bitsandbytes)."""

        from myllm.utils.std import infer_dtype

        self.model_name            = model_cfg.name
        self.__use_4bit            = getattr(model_cfg, "use_4bit", False)
        self.__use_8bit            = getattr(model_cfg, "use_8bit", False)
        self.__bnb_compute_dtype   = getattr(model_cfg, "bnb_compute_dtype", "fp16")
        self.dtype                 = infer_dtype(model_cfg.dtype)

        self.quant_kwargs: dict[str, Any] = {}

        self.update_quant_kwargs()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            **self.quant_kwargs,
            **hf_kwargs,
        )

        if not model_cfg.use_peft:
            self.check_grad_parameters()

    def check_grad_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                warnings.warn(
                    f"Detected frozen parameter while attempting to full finetune! Parameter: {name}"
                )
                logger.info(f"Parameter {name} does not require grad!")
            if param.numel() == 0:
                warnings.warn(
                    f"Detected empty parameter! Parameter: {name}, shape: {param.shape}"
                )
                logger.info(f"Parameter {name} is empty!")
        logger.info("Ready for full finetuning")

    def update_quant_kwargs(self):
        if self.__use_4bit and self.__use_8bit:
            raise ValueError("use_4bit and use_8bit are mutually exclusive – choose only one.")
        if self.__use_4bit:
            self.update_4bit()
        if self.__use_8bit:
            self.update_8bit()

    def update_4bit(self):
        compute_dtype = (
            torch.float16
            if self.__bnb_compute_dtype.lower() == "fp16"
            else torch.bfloat16
        )
        self.quant_kwargs.update(
            {
                "device_map": "auto",
                "load_in_4bit": True,
                "quantization_config": bitsandbytes.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    
    def update_8bit(self):
        self.quant_kwargs.update(
            {
                "device_map": "auto", 
                "load_in_8bit": True
            }
        )


