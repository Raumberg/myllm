from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class LoRAConfig:
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules for LoRA. If None, LoRA will be applied to all modules."}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "Rank of the LoRA layers. Higher values allow for more expressiveness."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Scaling factor for the LoRA layers. Adjusting this can help with training stability."}
    )