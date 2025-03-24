import multiprocessing
import os
import random
import uuid
import warnings
import sys

sys.path.append('/home/nshestopalov/projects/myllm')

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

# from src.rewards import *
import re

logger = get_logger(__name__)

# Список функций наград.
# Данные функции проверяют ответы модели на соблюдение определенный условий и
# числовое вознаграждение.
def extract_xml_answer(text: str) -> str:
    """Вытаскивает ответ из тегов <answer>ответ</answer>"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
        Данная функция проверяет есть ли правильное решение на поставленную задачу.
        Извлекает ответ из каждого варианта.
        Если извлечённый ответ совпадает с правильным, возвращается награда 2, иначе - 0.
    """
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """
        Для каждого извлечённого ответа проверяет, состоит ли он исключительно из цифр, используя метод isdigit().
        Если условие выполнено (то есть ответ — число), возвращается награда 0.5, иначе — 0.0
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
        Функция награды которая проверяет, что модель соблюдает указанный формат.
        Явно задаёт начало (^) и конец ($) строки. То есть вся строка должна полностью соответствовать шаблону.
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
        Еще одная функция награды которая проверяет, что модель соблюдает указанный формат.
        Не требует полного соответствия всей строки.
        Позволяет между тегами иметь произвольное количество пробельных символов.
        Не накладывает строгих требований к разбиению на строки.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

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

    # ds = load_datasets(args.dataset, args.test_size, args.dataset_ratio)
    
    # row_processor = partial(
    #     grpo_row_processor,
    #     args=args
    # )

    # with PartialState().local_main_process_first():
    #     ds = ds.map(
    #         row_processor,
    #         num_proc=multiprocessing.cpu_count(),
    #         load_from_cache_file=True,
    #     )

    # train_dataset = ds["train"]
    # eval_dataset = ds["test"]

    SYSTEM_PROMPT = """
    Отвечай в формате:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

    from datasets import load_dataset

    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    def get_gsm8k_questions(split = "train"):
        data = load_dataset('d0rj/gsm8k-ru')[split] # type: ignore
        data = data.map(lambda x: { # type: ignore
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        }) # type: ignore
        return data # type: ignore

    train_dataset = get_gsm8k_questions(split="train")
    eval_dataset = get_gsm8k_questions(split="test")

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