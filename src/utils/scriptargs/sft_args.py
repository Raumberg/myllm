from dataclasses import dataclass, field

from src.utils.configurators import MetaArguments

@dataclass
class SFTScriptArguments(MetaArguments):
    conversation_field: str | None = field(
        default="prompt",
        metadata={"help": "Field in dataset with conversations (in list of dicts format)"}
    )
    construct_history: bool | None = field(
        default=False,
        metadata={"help": "Whether to construct history from conversation"}
    )
    system_prompt: str | None = field(
        default=None,
        metadata={"help": "Will use system prompt if there is no one in dialogue, set to None to disable"}
    )
    train_only_on_completions: bool | None = field(
        default=True,
        metadata={"help": "Do train only on completions or not"}
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
    use_dyntanh: bool | None = field(
        default=False,
        metadata={"help": "Whether to use Fused Dynamic Tanh instead of LayerNorm"}
    )
    patch_dyntanh: bool | None = field(
        default=False,
        metadata={"help": "Whether to use Fused Dynamic Tanh instead of LayerNorm"}
    )

    def __post_init__(self):
        self.project_name = "sft-tuning" if self.project_name == "default-project" else self.project_name