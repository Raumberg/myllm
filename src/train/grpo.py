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

# | TRANSFORMERS / DATAPARALLEL | 
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOTrainer, GRPOConfig, ModelConfig, get_peft_config

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
    # Fusion / Kernels
    # ================== #
    if args.use_liger:
        get_liger_kernel()

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
    # Reward Models
    # ================== #
    if PartialState().is_main_process and args.preload_rm:
        print('Initializing reward models..')
        # DRAMAModel.get_instance('cuda')
        # sleep(5)

    # ================== #
    # Dataset
    # ================== #
    if not args.system_prompt: warnings.warn("System Prompt is not set in your configuration file, this can cause reasoning biases.")

    ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)
    
    row_processor = partial(
        grpo_row_processor,
        args=args
    )

    with PartialState().local_main_process_first():
        ds = ds.map(
            row_processor,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=True,
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
        peft_config=peft_config
    )

    trainer.train() if not args.resume_from else trainer.train(args.resume_from)
    trainer.save_model(grpo_config.output_dir)


if __name__ == '__main__':
    main()