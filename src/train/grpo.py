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

from src.utils.data import load_datasets, grpo_row_processor
from src.utils.logs import setup_logging
from src.utils.model import setup_model_and_tokenizer
from src.utils.stdout import print_configs, print_table

from functools import partial
from time import sleep

from src.rewards import *

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME

def main():
    # ================== #
    # Parsers
    # ================== #
    parser = ArgParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    args, grpo_config, model_config = parser.parse()

    # ================== #
    # stdout information
    # ================== #
    if PartialState().is_main_process:
        print_configs(args, grpo_config, model_config, 10)

    # ================== #
    # Logging
    # ================== #
    setup_logging(logger, grpo_config)
    set_seed(grpo_config.seed)

    # ================== #
    # Environment
    # ================== #
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    # ================== #
    # Model & Tokenizer
    # ================== #
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        torch_dtype=torch.bfloat16 if grpo_config.bf16 else torch.float16,
        attn_implementation=model_config.attn_implementation
    )

    # ================== #
    # Grad params
    # ================== #
    total_grad_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: total_grad_params += 1
        parameter.requires_grad = not model_config.use_peft

    if PartialState().is_main_process:
        k, v = ["Total Grad. params:"], [total_grad_params]
        print_table(k, v)
        sleep(10)

    # ================== #
    # PEFT config
    # ================== #    
    peft_config = get_peft_config(model_config)

    if model_config.lora_task_type != "CAUSAL_LM":
        warnings.warn(
            "You are using a `task_type` that is different than `CAUSAL_LM` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type CAUSAL_LM when using this script."
        )

    setup_model_and_tokenizer(args, model, tokenizer, args.max_seq_length)

    if PartialState().is_main_process:
        k, v = [
                'Pad token:',
                'EOS token:',
                'BOS token:'
            ], [
                tokenizer.pad_token,
                tokenizer.eos_token,
                tokenizer.bos_token,
            ]
        print_table(k, v)
        sleep(5)

    # ================== #
    # Dataset
    # ================== #
    if not args.system_prompt: warnings.warn("System Prompt is not set in your configuration file, this can cause reasoning biases.")

    ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)

    signature_columns = ["input_ids", "labels", "attention_mask"]
    extra_columns = list(set(ds['train'].column_names) - set(signature_columns))
    
    row_processor = partial(
        grpo_row_processor,
        args=args
    )

    with PartialState().local_main_process_first():
        ds = ds.map(
            row_processor,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True,
            remove_columns=extra_columns
        )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    if PartialState().is_main_process:
        print(f'Example from [TRAIN]: {train_dataset[0]}')
        print(f'Example from [TEST]: {eval_dataset[0]}')
        sleep(5)

    PartialState().wait_for_everyone()

    # ================== #
    # Trainer
    # ================== #
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