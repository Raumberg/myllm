from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    train_only_on_completions: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to train only on completions."}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Number of training samples per batch on a single GPU."}
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={"help": "Learning rate for the optimizer."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "Type of learning rate scheduler to use."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=8,
        metadata={"help": "Number of steps to accumulate gradients before updating."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory."}
    )
    warmup_steps: Optional[int] = field(
        default=20,
        metadata={'help': 'Whether to use warmup steps and how many steps to warm up'}
    )
    warmup_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of total steps to use for warmup."}
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 precision."}
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use float16 precision."}
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for initialization."}
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use Parameter-Efficient Fine-Tuning (PEFT)."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Implementation of attention mechanism to use."}
    )
    use_liger: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use liger kernels"}
    )