# train_grpo.py
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

dataset = load_dataset("d0rj/gsm8k-ru", split="train")

SYSTEM_PROMPT = """
Отвечай в формате:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Вытаскивает ответ из тегов <answer>ответ</answer>"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('d0rj/gsm8k-ru')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

# Список функций наград.
# Данные функции проверяют ответы модели на соблюдение определенный условий и
# числовое вознаграждение.
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

repo = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(repo)
tokenizer = AutoTokenizer.from_pretrained(repo)

train_dataset = get_gsm8k_questions(split = "train")
# print(train_dataset[0])
eval_dataset = get_gsm8k_questions(split = "test")


training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10, bf16=True, warmup_steps=0, report_to="wandb")
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    processing_class=tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()