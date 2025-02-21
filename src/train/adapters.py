import argparse

import torch
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Merging LoRA adapters with original weights")
    parser.add_argument('--source', type=str, required=True,
                        help="Adapter and model configuration path")
    parser.add_argument('--output', type=str, required=True,
                        help="Output path")
    parser.add_argument('--is_clf', action='store_true',
                        help="Is model type AutoPeftModelForSequenceClassification or AutoPeftModelForCausalLM")
    return parser.parse_args()


def merge(source_path: str, output_path: str, is_clf: str):
    tokenizer = AutoTokenizer.from_pretrained(source_path)

    if not is_clf:
        adapter_model = AutoPeftModelForCausalLM.from_pretrained(source_path, torch_dtype=torch.bfloat16)
    else:
        adapter_model = AutoPeftModelForSequenceClassification.from_pretrained(source_path, torch_dtype=torch.bfloat16, num_labels=1)

    adapter_save_path = os.path.join(output_path, 'original_adapter')
    os.makedirs(adapter_save_path, exist_ok=True)
    adapter_model.save_pretrained(adapter_save_path)

    merged_model = adapter_model.merge_and_unload()

    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    args = parse_args()
    merge(args.source, args.output, args.is_clf)
