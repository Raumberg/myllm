import click
import torch
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
import os

@click.group()
def cli():
    """CLI for merging LoRA adapters with original weights."""
    pass

@cli.command('merge')
@click.option('--source', '-s', required=True, help='Source model path')
# @click.option('--adapters', '-a', required=True, help='Adapters configuration path')
@click.option('--output', '-o', required=True, help='Output path')
@click.option('--is-clf', '-c', is_flag=True, help='Is model type AutoPeftModelForSequenceClassification or AutoPeftModelForCausalLM')
@click.option('--dtype', '-t', type=click.Choice(['f32', 'f16', 'bf16'], case_sensitive=False), default='bf16', help='Torch data type to use for model loading')
def merge(source: str, output: str, is_clf: bool, dtype: str):
    """Merge LoRA adapters with original weights."""
    
    dtype_mapping = {
        'f32': torch.float32,
        'f16': torch.float16,
        'bf16': torch.bfloat16
    }
    
    selected_dtype = dtype_mapping[dtype.lower()]

    tokenizer = AutoTokenizer.from_pretrained(source)

    if not is_clf:
        adapter_model = AutoPeftModelForCausalLM.from_pretrained(source, torch_dtype=selected_dtype)
    else:
        adapter_model = AutoPeftModelForSequenceClassification.from_pretrained(source, torch_dtype=selected_dtype, num_labels=1)

    adapter_save_path = os.path.join(output, 'original_adapter')
    os.makedirs(adapter_save_path, exist_ok=True)
    adapter_model.save_pretrained(adapter_save_path)

    merged_model = adapter_model.merge_and_unload()

    merged_model.save_pretrained(output)
    tokenizer.save_pretrained(output)

if __name__ == "__main__":
    cli()