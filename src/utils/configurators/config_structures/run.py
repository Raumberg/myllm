from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RunConfig:
    save_strategy: Optional[str] = field(
        default='steps',
        metadata={"help": "Strategy to use for saving checkpoints. Options: 'no', 'steps', 'epoch'."}
    )
    save_steps: Optional[int] = field(
        default=300,
        metadata={"help": "Number of steps between saving checkpoints. Required if save_strategy is 'steps'."}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep. Older checkpoints will be deleted."}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the current run. Useful for tracking experiments."}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory where the model and checkpoints will be saved."}
    )
    report_to: Optional[str] = field(
        default='wandb',
        metadata={"help": "Reporting tool to use for logging. Options: 'tensorboard', 'wandb', etc."}
    )
    logging_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log the first step of training."}
    )
    logging_steps: Optional[int] = field(
        default=1,
        metadata={'help': 'How many steps befor log writing'}
    )