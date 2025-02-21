from dataclasses import dataclass

from .config_structures import *

@dataclass
class MetaConfig:
    dataset: DatasetsConfig
    lora: LoRAConfig
    model: ModelConfig
    run: RunConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig