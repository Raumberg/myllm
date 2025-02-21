from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

from typing import Any, NewType

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DataClassType = NewType("DataClassType", Any)