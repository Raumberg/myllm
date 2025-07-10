from __future__ import annotations

"""Simple HuggingFace causal LM wrapper (LoRA/quant aware)."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from myllm.utils.lazy import bitsandbytes

__all__ = ["ModelWrapper"]


class ModelWrapper:
    """High-level convenience wrapper for causal language models."""

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.float16,
        use_4bit: bool = False,
        use_8bit: bool = False,
        bnb_compute_dtype: str = "fp16",
        **hf_kwargs: Any,
    ):
        """Load tokenizer & model with optional 4-/8-bit quantisation (bitsandbytes)."""

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_kwargs: dict[str, Any] = {}
        if use_4bit and use_8bit:
            raise ValueError("use_4bit and use_8bit are mutually exclusive – choose only one.")

        if use_4bit:
            compute_dtype = torch.float16 if bnb_compute_dtype.lower() == "fp16" else torch.bfloat16
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

        if torch.cuda.is_available() and not (use_4bit or use_8bit):
            # When quantised with device_map=auto model is already on GPU
            self.model.cuda()

    def generate(self, prompt: str, **gen_kwargs: Any) -> str:  # noqa: D401
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(out[0], skip_special_tokens=True) 