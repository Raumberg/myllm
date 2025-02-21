from dataclasses import dataclass, field

from src.utils.configurators import MetaArguments

@dataclass
class DPOScriptArguments(MetaArguments):
    generate_eval_examples: bool | None = field(
        default=True,
        metadata={"help": "Do generate examples on eval"}
    )
    num_gen_examples: int | None = field(
        default=50,
        metadata={"help": "Number of examples to generate on eval phase"}
    )

    def __post_init__(self):
        self.project_name = "dpo-tuning" if self.project_name == "default-project" else self.project_name