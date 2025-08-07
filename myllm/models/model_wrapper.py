from __future__ import annotations

"""Simple HuggingFace causal LM wrapper (LoRA/quant aware)."""

from typing import Any
import warnings
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from myllm.utils.lazy import bitsandbytes

__all__ = ["ModelWrapper"]
logger = logging.getLogger(__name__)


class ModelWrapper:
    """High-level convenience wrapper for causal language models."""

    def __init__(self, model_cfg: Any, **hf_kwargs: Any):
        """Load tokenizer & model with optional 4-/8-bit quantisation (bitsandbytes)."""

        from myllm.utils.std import infer_dtype

        model_name = model_cfg.name
        use_4bit = getattr(model_cfg, "use_4bit", False)
        use_8bit = getattr(model_cfg, "use_8bit", False)
        bnb_compute_dtype = getattr(model_cfg, "bnb_compute_dtype", "fp16")
        dtype = infer_dtype(model_cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_kwargs: dict[str, Any] = {}
        if use_4bit and use_8bit:
            raise ValueError("use_4bit and use_8bit are mutually exclusive – choose only one.")

        if use_4bit:
            compute_dtype = (
                torch.float16
                if bnb_compute_dtype.lower() == "fp16"
                else torch.bfloat16
            )
            quant_kwargs.update(
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

        elif use_8bit:
            quant_kwargs.update({"device_map": "auto", "load_in_8bit": True})

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **quant_kwargs,
            **hf_kwargs,
        )

        # if torch.cuda.is_available() and not (use_4bit or use_8bit):
            # When quantised with device_map=auto model is already on GPU
            # self.model.cuda()

        if not model_cfg.use_peft:
            self.check_grad_parameters()
            

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:  # noqa: D401
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True) 

    def check_grad_parameters(self):
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