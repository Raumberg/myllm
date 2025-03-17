# import multiprocessing
import os
import random
import uuid
import warnings

from multiprocessing import cpu_count
from functools import partial

import torch
from torchsummary import summary

from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import SFTTrainer, SFTConfig, ModelConfig, get_peft_config

from src.utils.callbacks import GenerateExamplesCallback
from src.utils.collators import DataCollatorForCompletionOnlyLM
from src.utils.configurators import ArgParser, tabula
from src.utils.scriptargs import SFTScriptArguments

from src.utils.data import load_datasets, default_row_processor, history_row_processor
from src.utils.logs import setup_logging
from src.utils.model import setup_model_and_tokenizer
from src.utils.kernels import get_liger_kernel

from src.utils.stdout import print_configs, inspect_model

from time import sleep

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME


def main():
    # ================== #
    # Parsers
    # ================== #
    parser = ArgParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, sft_config, model_config = parser.parse()

    # ================== #
    # stdout information
    # ================== #
    if PartialState().is_main_process:
        print_configs(args, sft_config, model_config, 10)

    # ================== #
    # Logging
    # ================== #
    setup_logging(logger, sft_config)
    set_seed(sft_config.seed)

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
        torch_dtype=torch.bfloat16 if sft_config.bf16 else torch.float16,
        attn_implementation=model_config.attn_implementation
    )

    if PartialState().is_main_process:
        print("Model Inspection:\n" + inspect_model(model, tokenizer))
        sleep(10)

    # ================== #
    # Fusion / Kernels
    # ================== #
    if sft_config.use_liger:
        get_liger_kernel()

    # ================== #
    # Parameters
    # ================== #
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not model_config.use_peft

    peft_config = get_peft_config(model_config)

    if model_config.lora_task_type != "CAUSAL_LM":
        warnings.warn(
            "You are using a `task_type` that is different than `CAUSAL_LM` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type CAUSAL_LM when using this script."
        )

    setup_model_and_tokenizer(args, model, tokenizer, sft_config.max_seq_length)

    if PartialState().is_main_process:
        print(f'Tokenizer: {tokenizer}')
        print(f'Model config: {model.config}')
        print(f"Pad Token: {tokenizer.pad_token}, EOS Token: {tokenizer.eos_token}, BOS Token: {tokenizer.bos_token}")
        sleep(5)

    # ================== #
    # Dataset
    # ================== #
    row_processor = partial(
        history_row_processor if args.construct_history else default_row_processor,
        args=args,
        training_config=sft_config,
        tokenizer=tokenizer,
        add_gen_prompt=False
    )

    ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)

    generate_dataset = ds['test']

    signature_columns = ["input_ids", "labels", "attention_mask"]
    extra_columns = list(set(ds['train'].column_names) - set(signature_columns))

    with PartialState().local_main_process_first():
        ds = ds.map(
            row_processor,
            num_proc=cpu_count(),
            load_from_cache_file=True,
            remove_columns=extra_columns
        )
        generate_dataset = generate_dataset.map(
            lambda row: row_processor(row, add_gen_prompt=True),
            num_proc=cpu_count(),
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
        print('Example from gen dataset:')
        print(generate_dataset[0])
        sleep(5)

    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=args.assistant_message_template,
        tokenizer=tokenizer
    ) if args.train_only_on_completions else None

    generate_callback = GenerateExamplesCallback(
        preprocessed_dataset=generate_dataset,
        tokenizer=tokenizer,
        num_examples=args.num_gen_examples,
        is_deepspeed_zero3=is_deepspeed_zero3_enabled(),
        logger_backend=sft_config.report_to[0]
    )

    PartialState().wait_for_everyone()

    sft_config.dataset_kwargs = {
        "skip_prepare_dataset": True
    }

    # ================== #
    # Trainer
    # ================== #
    trainer = SFTTrainer(
        model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=collator,
        callbacks=[generate_callback] if args.generate_eval_examples else []
    )

    trainer.train() if not args.resume_from else trainer.train(sft_config.model_name_or_path)
    trainer.save_model(sft_config.output_dir)


if __name__ == '__main__':
    main()
