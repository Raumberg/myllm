from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TokenizerConfig:
    assistant_message_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template for the assistant's message in the conversation."}
    )
    pad_token: Optional[str] = field(
        default="<pad>",
        metadata={"help": "Token used for padding sequences."}
    )
    eos_token: Optional[str] = field(
        default="<eos>",
        metadata={"help": "End-of-sequence token."}
    )
    chat_template: Optional[str] = field(
        default="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|im_start|>' + message['role'] + '<|im_sep|>'+ message['content'] | trim + '<|im_end|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}",
        metadata={"help": "Default chat template for model usage"}
    )
    force_chat_template: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to force the use of the chat template."}
    )
    model_support_system_role: Optional[bool] = field(
        default=False,
        metadata={"help": "Indicates if the model supports a system role in the conversation."}
    )
    added_special_tokens: Optional[str] = field(
        default='',
        metadata={"help": "New added tokens for tokenizer extension"}
    )