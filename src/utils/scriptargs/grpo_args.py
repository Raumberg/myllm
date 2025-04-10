from dataclasses import dataclass, field
from src.utils.configurators import MetaArguments

from dataclasses import dataclass, field

from src.utils.configurators import MetaArguments

@dataclass
class GRPOScriptArguments(MetaArguments):
    problem_field: str | None = field(
        default="problem",
        metadata={"help": "Problem field in the dataset"}
    )
    solution_field: str | None = field(
        default="solution",
        metadata={"help": "Solution field in the dataset"}
    )
    extract_hash: bool | None = field(
        default=False,
        metadata={"help": "Whether to extract answer after ### tags in solution_field"}
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Will use system prompt if there is no one in dialogue, set to None to disable"}
    )
    preload_rm: bool | None = field(
        default=False,
        metadata={"help": "Whether to preload reward models"}
    )
    rm: str | None = field(
        default=None,
        metadata={"help": "List of reward models to use"}
    )
    generate_eval_examples: bool | None = field(
        default=True,
        metadata={"help": "Do generate examples on eval"}
    )
    assistant_message_template: str | None = field(
        default="<|start_header_id|>assistant<|end_header_id|>\n\n",
        metadata={"help": "Assistant message template for the training only on completions"}
    )
    num_gen_examples: int | None = field(
        default=50,
        metadata={"help": "Number of examples to generate on eval phase"}
    )
    model_support_system_role: bool | None = field(
        default=True,
        metadata={"help": "Flag that indicates if model have support for system prompt. If not, will use user for setting system prompt"}
    )
    resume_from: str | None = field(
        default=None,
        metadata={"help": "Whether to resume from checkpoint. Path is expected"}
    )
    max_seq_length: int | None = field(
        default=8192,
        metadata={"help": "Max sequence length property."}
    )
    use_liger: bool | None = field(
        default=False,
        metadata={"help": "Whether to use Liger Kernel for the model"}
    )

    def __post_init__(self):
        self.project_name = "grpo-rl" if self.project_name == "default-project" else self.project_name
