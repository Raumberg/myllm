from __future__ import annotations

"""Config schema using Pydantic for validation and type-safety."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from myllm.quant.fp8 import FP8RecipeBuilder as QuantCfg

__all__ = [
    "ModelCfg",
    "TrainingCfg",
    "DataCfg",
    "EngineCfg",
    "LoggingCfg",
    "Config",
    "WandBCfg",
    "QuantCfg",
    "CollatorCfg",
]


class ModelCfg(BaseModel):
    name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    dtype: str = "bf16"
    attn_implementation: Optional[str] = None
    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None
    lora_task_type: str = "CAUSAL_LM"
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_compute_dtype: str = "fp16"
    cast_to_fp8: bool = False


class TrainingCfg(BaseModel):
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    seed: int = 42
    output_dir: str = "experiments"
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    gradient_clipping: Optional[float] = 1.0
    grpo: dict[str, Any] = Field(default_factory=dict)
    sft: dict[str, Any] = Field(default_factory=dict)
    ppo: dict[str, Any] = Field(default_factory=dict)
    distill: dict[str, Any] = Field(default_factory=dict)
    resume_from_checkpoint: Optional[str] = None
    use_liger_kernel: bool = True
    max_seq_length: int = 2048


class CollatorCfg(BaseModel):
    type: str = Field("standard", alias="collator_type")
    template: str | None = Field(None, alias="response_template")
    ignore_index: int = -100
    strict: bool = Field(False, alias="collator_strict")
    verbose: bool = False


class DataCfg(BaseModel):
    name: str = "/path/to/dataset"
    split: str = "train[:1%]"
    test_size: Optional[float] = None
    from_disk: bool = False
    text_field: str = "conversation"
    problem_field: str = "problem"
    answer_field: str = "answer"
    max_length: int = 512
    chat_template: str | None = None
    system_prompt: str | None = None
    model_support_system_role: bool = True
    processor_type: str = "default"
    offline: bool = False
    eval_split: str | None = None
    test_split: str | None = None
    collator: CollatorCfg = Field(default_factory=CollatorCfg)
    pad_token: str | None = None


class EngineCfg(BaseModel):
    name: str = "accelerate"
    config: Optional[Path] = None
    auto_fill: bool = True
    # Arbitrary DeepSpeed settings that override values from the JSON template.
    # This enables the "YAML is king" philosophy, letting users control every
    # DeepSpeed parameter from a single configuration file.
    override: Dict[str, Any] = Field(default_factory=dict)


class LoggingCfg(BaseModel):
    level: str = "info"
    suppress: list[str] = Field(
        default_factory=lambda: [
            "transformers",
            "accelerate",
            "datasets",
            "urllib3",
            "deepspeed",
            "torch.distributed",
            "huggingface_hub",
        ]
    )
    warnings_ignore: list[str] = Field(default_factory=list)
    disable_tqdm: bool = True


class WandBCfg(BaseModel):
    enable: bool = False
    project: str = "myllm"
    entity: Optional[str] = None
    name: Optional[str] = None
    resume_id: Optional[str] = None


class Config(BaseModel):
    model: ModelCfg = Field(default_factory=ModelCfg)
    training: TrainingCfg = Field(default_factory=TrainingCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    engine: EngineCfg | str = Field(default_factory=EngineCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    wandb: WandBCfg = Field(default_factory=WandBCfg)
    quant: QuantCfg = Field(default_factory=QuantCfg)

    raw: Dict[str, Any] | None = None

    @field_validator("data", mode="before")
    def build_data_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Merge legacy `dataset` and `collator` keys into the new `data` key."""
        if "collator" in v:
            return v  # Already in new format

        collator_keys = {"collator_type", "response_template", "ignore_index", "collator_strict"}
        data_collator_conf = {k: v.pop(k) for k in collator_keys if k in v}

        # Handle aliasing from yaml (e.g. `type` -> `collator_type`)
        if "type" in v:
            data_collator_conf["collator_type"] = v.pop("type")
        if "template" in v:
            data_collator_conf["response_template"] = v.pop("template")
        if "strict" in v:
            data_collator_conf["collator_strict"] = v.pop("strict")

        v["collator"] = data_collator_conf
        return v
    
    @field_validator("engine", mode="before")
    def build_engine_config(cls, v: Any) -> EngineCfg:
        """Allow engine to be specified as a simple string."""
        if isinstance(v, str):
            return EngineCfg(name=v)
        if isinstance(v, dict):
            return EngineCfg(**v)
        return v


    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        # Remap legacy keys
        if "dataset" in d:
            d["data"] = d.pop("dataset")
        if "collator" in d:
            # move collator settings into data block
            d.setdefault("data", {}).update(d.pop("collator"))

        return cls(**d, raw=d) 