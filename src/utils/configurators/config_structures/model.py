from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    model_name_or_path: str = field(
        default='attn-signs/Watari-7b-v1',
        metadata={'help': 'Model repo or local path'}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "Max sequence length of the model"}
    )