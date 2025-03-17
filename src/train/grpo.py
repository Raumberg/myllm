import multiprocessing
import os
import random
import uuid
import warnings

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOTrainer, GRPOConfig, ModelConfig, get_peft_config

from src.utils.configurators import ArgParser
from src.utils.scriptargs import GRPOScriptArguments

from src.utils.data import load_datasets
from src.utils.logs import setup_logging
from src.utils.model import setup_model_and_tokenizer

from src.rewards import *

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME

def main():
    parser = ArgParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    args, grpo_config, model_config = parser.parse()
    print(f'\n\n\nARGS : {args}')
    print(f'\n\n\nGRPO : {grpo_config}')
    print(f'\n\n\nMODEL : {model_config}')

    setup_logging(logger, grpo_config)
    set_seed(grpo_config.seed)

    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16 if grpo_config.bf16 else torch.float16,
        attn_implementation=model_config.attn_implementation
    )

    total_grad_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: total_grad_params += 1
        parameter.requires_grad = not model_config.use_peft

    print(f'Total parameters with grads: {total_grad_params}')

    peft_config = get_peft_config(model_config)

    if model_config.lora_task_type != "CAUSAL_LM":
        warnings.warn(
            "You are using a `task_type` that is different than `CAUSAL_LM` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type CAUSAL_LM when using this script."
        )

    setup_model_and_tokenizer(args, model, tokenizer, args.max_seq_length)

    if PartialState().is_main_process:
        print(f'Tokenizer: {tokenizer}')
        print(f'Model config: {model.config}')
        print(f"Pad Token: {tokenizer.pad_token}, EOS Token: {tokenizer.eos_token}, BOS Token: {tokenizer.bos_token}")

    ################
    # Dataset
    ################
    if not args.system_prompt: warnings.warn("System Prompt is not set in your configuration file, this can cause reasoning biases.")

    ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)

    signature_columns = ["input_ids", "labels", "attention_mask"]
    extra_columns = list(set(ds['train'].column_names) - set(signature_columns))

    def process_row(row):
        return {
            "prompt": [
                {'role': 'system', 'content': args.system_prompt},
                {'role': 'user', 'content': row[args.problem_field]} 
                ],
            "answer": row[args.problem_field]
        }

    with PartialState().local_main_process_first():
        ds = ds.map(
            process_row,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True,
            remove_columns=extra_columns
        )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    if PartialState().is_main_process:
        print('Example from train dataset:')
        print(train_dataset[0])
        print('Example from test dataset:')
        print(eval_dataset[0])

    PartialState().wait_for_everyone()

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config
    )

    trainer.train() if not args.resume_from else trainer.train(args.resume_from)
    trainer.save_model(grpo_config.output_dir)


if __name__ == '__main__':
    main()