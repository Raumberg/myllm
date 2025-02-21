from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class DatasetsConfig:
    dataset: str | List[str] = field(
        default="/path/to/dataset",
        metadata={"help": "The name on Hugging Face Hub or path to a JSONL file of the dataset to use. Can be a list of paths."},
    )
    dataset_ratio: Optional[float] = field(
        default=1,
        metadata={"help": "Proportion of the dataset to use. Each ratio should be between 0 and 1."}
    )
    test_size: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of the dataset to use for the test set (e.g., 0.05). If the dataset already contains a test split, leave this empty."},
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Whether to remove columns not required by the model."}
    )
    dataloader_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading. Default is 0, which means data loading will be done in the main process."}
    )
    generate_eval_examples: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to generate evaluation examples."}
    )
    num_gen_examples: Optional[int] = field(
        default=20,
        metadata={"help": "Number of examples to generate for evaluation."}
    )
    evaluation_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "Evaluation strategy to adopt during training. Options: 'no', 'steps', 'epoch'."}
    )
    eval_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Number of steps between evaluations."}
    )
    conversation_field: Optional[str] = field(
        default="conversation",
        metadata={"help": "Field in the dataset that contains the conversation data."}
    )