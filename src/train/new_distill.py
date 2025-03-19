import os
import random
import uuid
import warnings
from functools import partial
from multiprocessing import cpu_count

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainingArguments, DataCollatorForLanguageModeling
from trl import ModelConfig, get_peft_config

from src.utils.callbacks import GenerateExamplesCallback
from src.utils.collators import DataCollatorForCompletionOnlyLM
from src.utils.configurators import ArgParser, tabula
from src.utils.scriptargs import DistillScriptArguments
from src.utils.data import load_datasets, default_row_processor, history_row_processor
from src.utils.logs import setup_logging
from src.utils.model import setup_model_and_tokenizer
from src.utils.kernels import get_liger_kernel
from src.utils.trainers import DistillationTrainer
from src.utils.stdout import print_configs

logger = get_logger(__name__)

LOGGING_TASK_NAME = str(uuid.uuid4())

os.environ['WANDB_RUN_ID'] = str(random.randint(100000, 999999))
os.environ['WANDB_NAME'] = LOGGING_TASK_NAME
os.environ['CLEARML_TASK'] = LOGGING_TASK_NAME


def main():
    # ================== #
    # Parsers
    # ================== #
    parser = ArgParser((DistillScriptArguments, TrainingArguments, ModelConfig))
    args, training_config, model_config = parser.parse()

    # ================== #
    # stdout information
    # ================== #
    if PartialState().is_main_process:
        print_configs(args, training_config, model_config, 10)

    # ================== #
    # Logging
    # ================== #
    setup_logging(logger, training_config)
    set_seed(training_config.seed)

    # ================== #
    # Environment
    # ================== #
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ['CLEARML_PROJECT'] = args.project_name

    # ================== #
    # Model & Tokenizer
    # ================== #
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)

    # Load the teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config.attn_implementation
    )

    # Load the student model
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config.attn_implementation
    )

    # ================== #
    # Fusion / Kernels
    # ================== #
    if training_config.use_liger:
        get_liger_kernel()

    # ================== #
    # Parameters
    # ================== #
    peft_config = get_peft_config(model_config)

    # ================== #
    # Dataset
    # ================== #
    row_processor = partial(
        history_row_processor if args.construct_history else default_row_processor,
        args=args,
        training_config=training_config,
        tokenizer=tokenizer,
        add_gen_prompt=False
    )

    ds = load_datasets(args.dataset_train_path, args.dataset_test_path)

    train_dataset = ds['train']
    eval_dataset = ds['test']

    # Process datasets
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.map(
            row_processor,
            num_proc=cpu_count(),
            load_from_cache_file=True
        )
        eval_dataset = eval_dataset.map(
            row_processor,
            num_proc=cpu_count(),
            load_from_cache_file=True
        )

    # ================== #
    # Data Collator
    # ================== #
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template=args.assistant_message_template,
        tokenizer=tokenizer
    ) if args.train_only_on_completions else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ================== #
    # Trainer
    # ================== #
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        arguments=training_config,
        distill_loss=args.distillation_loss,
        temperature=args.temperature,
        callbacks=[GenerateExamplesCallback(preprocessed_dataset=eval_dataset, tokenizer=tokenizer)]
    )

    # Start training
    trainer.train()
    trainer.save_model(training_config.output_dir)


if __name__ == '__main__':
    main()