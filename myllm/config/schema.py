from __future__ import annotations

"""Lightweight config schema using dataclasses.

Все поля сделаны максимально simple, чтобы быстро запуститься. По мере роста
будем дополнять.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "ModelCfg",
    "TrainingCfg",
    "DataCfg",
    "EngineCfg",
    "LoggingCfg",
    "Config",
    "WandBCfg",
]


@dataclass
class ModelCfg:
    name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    dtype: str = "bf16"  # bf16|fp16|fp32
    attn_implementation: Optional[str] = None  # xformers|flash_attention_2

    # PEFT / LoRA options
    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None  # None = all linear layers
    lora_task_type: str = "CAUSAL_LM"  # passed to TaskType enum

    # Quantisation (bitsandbytes)
    use_4bit: bool = False  # enable bnb 4-bit quantisation
    use_8bit: bool = False  # enable 8-bit quantisation
    bnb_compute_dtype: str = "fp16"  # compute dtype inside 4-bit (fp16|bf16)


@dataclass
class TrainingCfg:
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    lr: float = 2e-5
    seed: int = 42
    output_dir: str = "experiments"
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    grpo: dict[str, Any] = field(default_factory=dict)  # GRPO-specific params
    sft: dict[str, Any] = field(default_factory=dict)   # SFT-specific params
    ppo: dict[str, Any] = field(default_factory=dict)   # PPO-specific params
    distill: dict[str, Any] = field(default_factory=dict)  # DPO/Distill params
    resume_from_checkpoint: Optional[str] = None


@dataclass
class DataCfg:
    name: str = "/path/to/dataset"
    split: str = "train[:1%]"
    test_size: Optional[float] = None
    from_disk: bool = False
    text_field: str = "conversation"
    problem_field: str = "problem"  # used by GRPO
    answer_field: str = "answer"
    max_length: int = 512
    chat_template: str | None = None

    # Advanced processor options (ported from v1)
    system_prompt: str | None = None
    model_support_system_role: bool = True
    processor_type: str = "default"  # default|history|grpo

    # Collator options
    collator_type: str = "standard"  # standard|completion_only
    response_template: str | None = None  # used when collator_type=completion_only
    ignore_index: int = -100
    collator_strict: bool = False  # if true, raise error when template not found

    offline: bool = False  # If True, enforce HF offline mode (no internet fetch)
    eval_split: str | None = None  # explicit split name for evaluation, overrides test_size if provided
    test_split: str | None = None  # explicit split name for test set


@dataclass
class EngineCfg:
    name: str = "deepspeed"  # deepspeed|accelerate
    config: Optional[Path] = None


@dataclass
class LoggingCfg:
    level: str = "info"  # debug|info|warning|error
    suppress: list[str] = field(default_factory=lambda: [
        "transformers",
        "accelerate",
        "datasets",
        "urllib3",
        "deepspeed",
        "torch.distributed",
        "huggingface_hub",
    ])  # modules to silence completely
    warnings_ignore: list[str] = field(default_factory=list)
    disable_tqdm: bool = True


@dataclass
class WandBCfg:
    enable: bool = False
    project: str = "myllm"
    entity: Optional[str] = None
    name: Optional[str] = None  # run name
    resume_id: Optional[str] = None  # wandb resume run id


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    data: DataCfg = field(default_factory=DataCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    wandb: WandBCfg = field(default_factory=WandBCfg)

    raw: Dict[str, Any] | None = None  # keep original dict for debugging

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        # Build merged dataset+collator dict
        _ds = d.get("dataset", d.get("data", {}))
        _coll_raw = d.get("collator", {})

        _coll: Dict[str, Any] = {}
        if "type" in _coll_raw:
            _coll["collator_type"] = _coll_raw["type"]
        if "template" in _coll_raw:
            _coll["response_template"] = _coll_raw["template"]
        if "ignore_index" in _coll_raw:
            _coll["ignore_index"] = _coll_raw["ignore_index"]
        if "strict" in _coll_raw:
            _coll["collator_strict"] = _coll_raw["strict"]

        data_dict = {**_ds, **_coll}

        engine_raw = d.get("engine", {})
        engine_name: str
        engine_conf_path: Path | None = None
        if isinstance(engine_raw, dict):
            engine_name = engine_raw.get("name", "deepspeed")
            if engine_raw.get("config") is not None:
                engine_conf_path = Path(engine_raw["config"])
        else:
            engine_name = engine_raw  # if user passed plain string

        return cls(
            model=ModelCfg(**d.get("model", {})),
            training=TrainingCfg(**d.get("training", {})),
            data=DataCfg(**data_dict),
            engine=EngineCfg(name=engine_name, config=engine_conf_path),
            logging=LoggingCfg(**d.get("logging", {})),
            wandb=WandBCfg(**d.get("wandb", {})),
            raw=d,
        ) 