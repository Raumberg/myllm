from dataclasses import dataclass, field
from src.utils.configurators import MetaArguments

@dataclass
class GRPOScriptArguments(MetaArguments):
    use_vllm: bool | None = field(
        default=True,
        metadata={'help': 'Use vLLM for a teacher'}
    )

    vllm_gpu_memory_utilization: float | None = field(
        default=0.5,
        metadata={'help': 'How much of gpu % to use by vLLM'}
    )

    learning_rate: float = field(
        default=5e-6,
        metadata={'help': 'Learning rate for the optimizer'}
    )

    adam_beta1: float = field(
        default=0.9,
        metadata={'help': 'Beta1 parameter for Adam optimizer'}
    )

    adam_beta2: float = field(
        default=0.99,
        metadata={'help': 'Beta2 parameter for Adam optimizer'}
    )

    weight_decay: float = field(
        default=0.1,
        metadata={'help': 'Weight decay for optimizer'}
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={'help': 'Warmup ratio for learning rate scheduling'}
    )

    lr_scheduler_type: str = field(
        default="cosine",
        metadata={'help': 'Type of learning rate scheduler'}
    )

    optim: str = field(
        default="adamw_8bit",
        metadata={'help': 'Optimizer type'}
    )

    logging_steps: int = field(
        default=1,
        metadata={'help': 'Log every X updates steps'}
    )

    bf16: bool = field(
        default=False,  # Set to False by default; adjust as needed
        metadata={'help': 'Use bfloat16 precision'}
    )

    fp16: bool = field(
        default=True,  # Set to True by default; adjust as needed
        metadata={'help': 'Use float16 precision'}
    )

    per_device_train_batch_size: int = field(
        default=1,
        metadata={'help': 'Batch size per device during training'}
    )

    gradient_accumulation_steps: int = field(
        default=8,
        metadata={'help': 'Number of updates steps to accumulate before performing a backward/update pass'}
    )

    num_generations: int = field(
        default=8,
        metadata={'help': 'Number of responses to generate for each task'}
    )

    max_prompt_length: int = field(
        default=512,
        metadata={'help': 'Maximum length of the input prompt in tokens'}
    )

    max_completion_length: int = field(
        default=2048,
        metadata={'help': 'Maximum length of the model response'}
    )

    num_train_epochs: int = field(
        default=1,
        metadata={'help': 'Number of times to iterate over the training dataset'}
    )

    save_steps: int = field(
        default=100,
        metadata={'help': 'Save checkpoint every X updates steps'}
    )

    max_grad_norm: float = field(
        default=0.1,
        metadata={'help': 'Max norm for gradient clipping'}
    )

    report_to: str = field(
        default="wandb",
        metadata={'help': 'Reporting tool for logging'}
    )

    eval_strategy: str = field(
        default="steps",
        metadata={'help': 'Evaluation strategy to adopt during training'}
    )
    eval_steps: int = field(
        default=10,
        metadata={'help': 'Evaluation steps'}
    )