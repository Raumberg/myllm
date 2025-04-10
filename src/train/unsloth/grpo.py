# | CORE |
import multiprocessing
import os
import random
import uuid
import warnings
import sys

sys.path.append('/home/nshestopalov/projects/myllm') # insert your myllm path here

# | TORCH | 
import torch

# | UNSLOTH | 
from unsloth import FastLanguageModel, is_bf16_supported
from trl import GRPOTrainer, GRPOConfig, ModelConfig

# | PARSERS |
from src.utils.configurators import ArgParser
from src.utils.scriptargs import GRPOScriptArguments

# | DATA |
from src.utils.data.utils import load_datasets
from src.utils.data.processors import grpo_row_processor

# | MODEL |
from src.utils.logs import setup_logging
from src.utils.model import setup_model_and_tokenizer

# | STDOUT | 
from src.utils.stdout import print_configs, print_table

# | FUSION |
from src.utils.kernels import get_liger_kernel

# | MISC |
from functools import partial
from time import sleep

# | GRPO REWARDS | 
import src.rewards as rwd

# | REWARD MODELS | 
from src.reward_models.drama import DRAMAModel

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
    print_configs(args, grpo_config, model_config, 10)

    # ================== #
    # Environment
    # ================== #
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    # ================== #
    # Model & Tokenizer
    # ================== #
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = model_config.load_in_4bit,
        fast_inference = args.fast_inference,
        max_lora_rank = model_config.lora_r,
        gpu_memory_utilization = grpo_config.vllm_gpu_memory_utilization,
        float8_kv_cache = args.fast_kv_cache
        )

    model = FastLanguageModel.get_peft_model(
        model = model,
        r = model_config.lora_r,
        target_modules = model_config.lora_target_modules,
        lora_alpha = model_config.lora_alpha,
        use_gradient_checkpointing = args.grad_checkpointing,
        random_state = 42
        )

    # ================== #
    # Grad params
    # ================== #
    total_grad_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: total_grad_params += 1
        parameter.requires_grad = not model_config.use_peft

    k, v = ["Total Grad. params:"], [total_grad_params]
    print_table(k, v)
    sleep(10)

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
    
    row_processor = partial(
        grpo_row_processor,
        args=args
    )

    ds = ds.map(
        row_processor,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
    )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    print(f'Example from [TRAIN]: {train_dataset[0]}')
    print(f'Example from [TEST]: {eval_dataset[0]}')
    sleep(5)

    # ================== #
    # Trainer
    # ================== #
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            rwd.equation_structure_reward,
            # rwd.redundancy_penalty,
            rwd.correctness_reward,
            rwd.multilingual_coherence_reward,
            rwd.strict_chinese_penalty,
            rwd.bormann_format_reward,
            rwd.russian_purity_reward
        ],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train() if not args.resume_from else trainer.train(args.resume_from)
    trainer.save_model(grpo_config.output_dir)


if __name__ == '__main__':
    main()