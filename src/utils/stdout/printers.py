from src.utils.configurators import tabula
from transformers import AutoModelForCausalLM, AutoTokenizer

from time import sleep


def inspect_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, secs: int = 10) -> str:
    """Inspect key model parameters and architecture details"""
    inspection_data = {
        "Model Architecture": model.__class__.__name__,
        "Total Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        "Trainable Parameters": f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        "Dtype": str(model.dtype),
        "Device": str(next(model.parameters()).device),
        "Tokenizer Vocab Size": len(tokenizer),
        "Model Embedding Size": model.get_input_embeddings().weight.shape[0],
        "Padding Token": f"{tokenizer.pad_token} (ID: {tokenizer.pad_token_id})",
        "EOS Token": f"{tokenizer.eos_token} (ID: {tokenizer.eos_token_id})",
        "Max Sequence Length": getattr(model.config, "max_position_embeddings", "N/A")
    }
    
    if hasattr(model.config, "architectures"):
        inspection_data["Architecture"] = model.config.architectures[0]
        
    if hasattr(model.config, "hidden_size"):
        inspection_data["Hidden Size"] = model.config.hidden_size
        
    if hasattr(model.config, "num_attention_heads"):
        inspection_data["Attention Heads"] = model.config.num_attention_heads

    sleep(secs)
    
    return tabula(inspection_data)

def print_configs(ScriptArguments: dict, TrainingArguments: dict, ModelConfig: dict, secs: int = 10) -> None:
    print(f'\n\nScript arguments:\n{tabula(ScriptArguments.__dict__)}\n\n')
    sleep(secs)
    print(f'\n\nTraining config:\n{tabula(TrainingArguments.to_dict())}\n\n')
    sleep(secs)
    print(f'\n\nModel config:\n{tabula(ModelConfig.__dict__)}\n\n')
    sleep(secs)