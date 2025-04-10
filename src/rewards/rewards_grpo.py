from typing import List, Optional, Tuple
import torch
import logging

from src.utils.data.extraction import extract_tagged_answer, extract_after_thinking, extract_boxed_answer
from src.reward_models.drama import DRAMAModel
from src.utils.stdout.model_answer import print_section, format_box, format_reward_bar
from src.validators.russian import RussianWordValidator

import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

# <UTILITIES>

def get_content(completion: List[str] | None) -> str:
    if isinstance(completion, list) and len(completion) > 0:
        return completion[0].get('content', '')
    elif isinstance(completion, dict):
        return completion.get('content', '')
    return str(completion)

def get_xml_content(text: str, tag: str) -> str:
    """
    Universal XML tags extractor
    """
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""

def check_xml_structure(text: str, required_tags: Tuple[str]) -> bool:
    """
    XML structure compliance
    """
    return all(
        text.count(f"<{tag}>") == 1 and text.count(f"</{tag}>") == 1
        for tag in required_tags
    )

def is_valid_xml_block(content: str, tag: str) -> bool:
    """Проверяет наличие полной структуры тега и непустого содержимого"""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    return (content.count(start_tag) == 1 and 
            content.count(end_tag) == 1 and
            content.find(start_tag) < content.find(end_tag) and
            len(content.split(start_tag)[1].split(end_tag)[0].strip()) > 0)

# /<UTILITIES>

# <REWARDS>

def format_compliance_with_answer_reward(text: str) -> float:
    num_reason_open  = text.count("<think>")
    num_reason_close = text.count("</think>")
    num_ans_open     = text.count("<answer>")
    num_ans_close    = text.count("</answer>")
    if (num_reason_open == 1 and 
        num_reason_close == 1 and
        num_ans_open == 1 and
        num_ans_close == 1):
        return 0.2
    else:
        return 0.0

def format_compliance_without_answer_reward(text: str) -> float:
    num_reason_open  = text.count("<think>")
    num_reason_close = text.count("</think>")
    if (num_reason_open == 1 and 
        num_reason_close == 1):
        return 0.2
    else:
        return 0.0

def format_compliance_bormann_reward(text: str) -> float:
    num_reason_open  = text.count("<think>")
    num_reason_close = text.count("</think>")
    num_diag_open     = text.count("<diagram>")
    num_diag_close    = text.count("</diagram>")
    if (num_reason_open == 1 and 
        num_reason_close == 1 and
        num_diag_open == 1 and
        num_diag_close == 1):
        return 0.2
    else:
        return 0.0

def format_reward(completions: list, **kwargs) -> list[float]:
    """
    Universal format reward function
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"]
        if check_xml_structure(content, ('think', 'diagram')):
            rewards.append(1.0 if '\n' in get_xml_content(content, 'think') else 0.5)
        else:
            rewards.append(0.0)
    return rewards

def bormann_format_reward(completions: list, **kwargs) -> list[float]:
    """
    Bormann format reward function (with diagrams)
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"]
        reward = 0.0
        
        # Основные проверки
        reason_valid = is_valid_xml_block(content, "think")
        diagram_valid = is_valid_xml_block(content, "diagram")

        if "<think>" in content and "</think>" in content:
            reason_content = content.split("<think>")[1].split("</think>")[0]
            has_multiline = any(line.strip() for line in reason_content.splitlines())
        else:
            has_multiline = False

        # Начисление наград
        if reason_valid:
            reward += 0.6                           # Базовый reward за структуру
            reward += 0.1 if has_multiline else 0   # Бонус за многострочность
            if diagram_valid:
                reward += 0.3                       # Максимальный бонус за диаграмму
        
        # Ограничение и добавление в результаты
        rewards.append(min(reward, 1.0))
    
    return rewards

def russian_vocabulary_reward(completions: list, **kwargs) -> List[float]:
    """
    Награда за лексическую корректность русского языка:
    - 1.0 если >95% слов существуют
    - 0.8 если 85-95%
    - 0.6 если 75-85%
    - 0.3 если 50-75%
    - 0.0 если <50%
    """
    validator = RussianWordValidator()
    rewards = []
    
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion['content']
        
        words = validator.word_pattern.findall(text.lower())
        if not words:
            rewards.append(0.0)
            continue
            
        valid_count = 0
        for word in words:
            if validator.is_valid_word(word):
                valid_count += 1
                
        valid_ratio = valid_count / len(words)
        
        if valid_ratio >= 0.95:
            reward = 1.0
        elif valid_ratio >= 0.85:
            reward = 0.8
        elif valid_ratio >= 0.75:
            reward = 0.6
        elif valid_ratio >= 0.5:
            reward = 0.3
        else:
            reward = 0.0
            
        rewards.append(round(reward, 2))
    
    return rewards

def multilingual_coherence_reward(completions: list, **kwargs) -> List[float]:
    """
    Комплексная проверка с жёстким штрафом за:
    - Смешанные алфавиты
    - Китайские иероглифы
    - Несоответствие языку
    """
    # Регулярные выражения
    WORD_PATTERN = re.compile(r'''
        \b
        (?![\d_]+(?:\b|_))      # Исключаем чисто числовые
        [^\s\u4E00-\u9FFF]+     # Любые символы кроме пробелов и китайских иероглифов
        |                       # ИЛИ
        [\u4E00-\u9FFF]+        # Отдельные китайские иероглифы
        \b
    ''', re.IGNORECASE | re.VERBOSE | re.UNICODE)
    
    CHINESE_CHAR = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900–\uFAFF]')  # Все основные китайские символы
    MIXED_ALPHABET = re.compile(r'''
        (?:[a-z].*[а-яё]|      # Смесь латиницы и кириллицы
        [а-яё].*[a-z])         # В любом порядке
    ''', re.IGNORECASE | re.VERBOSE | re.UNICODE)

    rewards = []
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion['content']
        
        # Находим все слова и отдельные китайские символы
        words = WORD_PATTERN.findall(text)
        if not words:
            rewards.append(1.0)
            continue
            
        # Подсчёт ошибок
        error_score = 0
        for word in words:
            penalty = 0
            # Жёсткий штраф за китайские символы (3x вес)
            if CHINESE_CHAR.search(word):
                penalty += 5 * len(CHINESE_CHAR.findall(word))
                
            # Штраф за смешанные алфавиты
            if MIXED_ALPHABET.search(word):
                penalty += 5  # Фиксированный штраф за слово
                
            error_score += min(penalty, 10) # 10 штрафных баллов за слово
        
        # Нормализация ошибок
        max_possible_errors = 5 * len(words)
        error_ratio = error_score / max_possible_errors if max_possible_errors > 0 else 0
        
        # Экспоненциальный штраф
        reward = max(0.0, 1.0 - (error_ratio ** 1.5))
        rewards.append(round(reward, 2))
    
    return rewards

def strict_chinese_penalty(completions: list, **kwargs) -> List[float]:
    rewards = multilingual_coherence_reward(completions, **kwargs)
    return [
        0.0 if any(0x4E00 <= ord(c) <= 0x9FFF for c in (
            comp[0]['content'] if isinstance(comp, list) else comp['content']
        )) 
        else r 
        for r, comp in zip(rewards, completions)
    ]

def russian_purity_reward(completions: list, **kwargs) -> List[float]:
    """
    Вычисляет вознаграждение на основе процента не-русских символов в тексте,
    игнорируя специальные паттерны (код, формулы, технические обозначения).
    
    Штрафная система:
    - <1/3% не-русских символов: нет штрафа (reward = 1.0)
    - 33-50%: reward = 0.8
    - 50-75%: reward = 0.5
    - >75%: reward = 0.0
    
    Args:
        completions: Список сгенерированных ответов
        
    Returns:
        Список значений вознаграждения от 0 (много не-русских символов) до 1 (преимущественно русский)
    """
    # Паттерны для исключений (код, формулы, тех. обозначения)
    ALLOWED_PATTERNS = [
        r'<code>.*?</code>',       # код в тегах
        r'`.*?`',                  # код в ``
        r'```.*?```',              # код в ``` ```
        r'\$.*?\$',                # математические формулы
        r'\\[a-zA-Z]+',            # LaTeX команды
        r'\b[a-zA-Z]{1,4}\d*\b'    # короткие обозначения (X1, QR и т.д.)
    ]
    
    RUSSIAN_CHARS = re.compile(r'[а-яёА-ЯЁ]')
    
    rewards = []
    for completion in completions:
        text = completion[0]['content']
        
        clean_text = text
        for pattern in ALLOWED_PATTERNS:
            clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL)
        
        all_chars = re.findall(r'\S', clean_text)
        if not all_chars:
            rewards.append(0)
            continue
            
        russian_chars = RUSSIAN_CHARS.findall(clean_text)
        
        non_russian_ratio = 1 - (len(russian_chars) / len(all_chars))
        
        if non_russian_ratio < 0.33:
            reward = 1.0
        elif non_russian_ratio < 0.5:
            reward = 0.8
        elif non_russian_ratio < 0.75:
            reward = 0.5
        else:
            reward = 0.0
            
        rewards.append(reward)
    
    return rewards

def russian_coherence_reward(completions: list, **kwargs) -> List[float]:
    """
    Проверяет когерентность русского текста, анализируя, чтобы русские слова
    содержали только русские символы (без смешанных написаний типа 'звiзда').
    
    Возвращает:
    - 1.0 если >80% русских слов корректны
    - 0.5 если 60-80% русских слов корректны
    - 0.0 если <60% русских слов корректны
    
    Args:
        completions: Список сгенерированных ответов
        
    Returns:
        Список значений вознаграждения
    """
    # Регулярные выражения
    RUSSIAN_WORD = re.compile(r'\b[а-яёА-ЯЁ]+\b')  # Только русские буквы
    NON_RUSSIAN_CHAR = re.compile(r'[^а-яёА-ЯЁ]')  # Любые не-русские символы
    
    rewards = []
    for completion in completions:
        text = completion[0]['content'] if isinstance(completion, list) else completion['content']
        
        # Находим все русские слова
        russian_words = RUSSIAN_WORD.findall(text)
        if not russian_words:
            rewards.append(0)  # Нет русских слов - считаем некорректным
            continue
            
        # Считаем слова с некорректными символами
        incorrect_count = 0
        for word in russian_words:
            if NON_RUSSIAN_CHAR.search(word):
                incorrect_count += 1
                
        correct_ratio = 1 - (incorrect_count / len(russian_words))
        
        if correct_ratio > 0.8:
            reward = 1.0
        elif correct_ratio > 0.6:
            reward = 0.5
        else:
            reward = 0.0
            
        rewards.append(reward)
    
    return rewards

def correctness_reward(
        prompts: list, 
        completions: list, 
        answer: list, 
        verbose: bool = True, 
        **kwargs) -> list[float]:
    """
    Evaluate the correctness of the extracted answers against the expected answer.

    This function extracts answers from each completion and compares them to the expected answer.
    If the extracted answer matches the expected answer, a reward of 2 is returned; otherwise, 0 is returned.

    Args:
        prompts (list): A list of prompts used for generating completions.
        completions (list): A list of generated completions.
        answer (list): The expected correct answer.

    Returns:
        list[float]: A list of rewards for each completion based on correctness.
    """
    responses = [completion[0]['content'] for completion in completions]
    question = prompts[0][-1]['content']
    extracted_responses = [extract_boxed_answer(r) for r in responses]
    
    if verbose:
        # Clear screen between evaluations for better readability
        print('\033c', end='')
        
        # Print formatted evaluation
        print(format_box(
            "Evaluation Result", 
            f"Question: {question[:70]}..." if len(question) > 70 else question,
            width=140
            ))
        print(format_box(
            "Expected Answer", 
            answer[0],
            width=140
            ))
        print(format_box(
            "Model Response", 
            responses[0],
            width=140
            ))

        similarity = 1.0 if extracted_responses[0] == answer[0] else 0.0
        rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
        reward_bar = format_reward_bar(
            similarity=similarity,
            reward=rewards[0],
            width=60
        )
        
        # Detailed view in terminal-friendly format
        print_section("Detailed Comparison", 
                    f"Extracted Answer: {extracted_responses[0]}\n"
                    f"Matches Expected: {extracted_responses[0] == answer[0]}\n"
                    f"{reward_bar}")

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def accuracy_reward(completions: list, solution: list, **kwargs) -> list[float]:
    """
    Reward function that evaluates the accuracy of model completions against the ground truth.

    This function checks if the completions generated by the model match the expected solutions.
    It first attempts to parse the ground truth solution using LaTeX parsing. If successful, it then
    parses the model's completion and compares the two. A reward of 1.0 is given if the parsed completion
    matches the parsed ground truth; otherwise, a reward of 0.0 is given. If the ground truth cannot be parsed,
    a reward of 1.0 is assigned to skip this example.

    Args:
        completions (list): A list of model-generated completions, where each completion is expected to be a 
                            dictionary containing a "content" key with the generated text.
        solution (list): A list of ground truth solutions corresponding to the completions.
        **kwargs: Additional keyword arguments that may be used for further customization (not utilized in this function).

    Returns:
        list[float]: A list of rewards for each completion, where each reward is either 1.0 or 0.0 based on the 
                     accuracy of the completion compared to the ground truth.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        
        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            reward = 1.0
        rewards.append(reward)
    return rewards

def latex_solution_reward(completions, **kwargs) -> list[float]:
    """Evaluates completions by checking if they match the correct solution."""
    solutions = kwargs.get("solution", [])
    completion_contents = [c[0]["content"] for c in completions]
    rewards = []

    for content, solution in zip(completion_contents, solutions):
        parsed_solution = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        parsed_content = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])

        if not parsed_solution:
            rewards.append(1.0)  # Default reward when no valid solution exists
            continue
        try:
            rewards.append(float(verify(parsed_content, parsed_solution)))
        except Exception:
            rewards.append(0.0)  # Return 0 if verification fails

    return rewards

def redundancy_penalty(completions, **kwargs):
    penalties = []
    for comp in completions:
        content = comp[0]['content']
        
        # 4-gram counts penalty
        ngrams = [content[i:i+4] for i in range(len(content) - 3)]
        unique_ngrams = len(set(ngrams))
        penalty = 1.0 - (unique_ngrams / len(ngrams)) if ngrams else 0.0
        
        penalties.append(-penalty * 0.5)
    return penalties

def equation_structure_reward(completions, **kwargs):
    rewards = []
    for comp in completions:
        content = comp[0]['content']
        pattern = r'\\boxed{\s*([+-]?\d+\.?\d*)\s*}'
        rewards.append(1.0 if re.search(pattern, content) else 0.0)
    return rewards

def semantic_similarity_reward(
    prompts: List[str],
    completions: List[str],
    reference_answers: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Computes discrete rewards based on cosine similarity between generated answers and reference answers using DRAMA model.
    
    Reward tiers:
    - similarity < 0.33: reward 0 (poor match)
    - 0.33 ≤ similarity < 0.60: reward 1 (partial match)
    - 0.60 ≤ similarity < 0.80: reward 1.5 (good match)
    - similarity ≥ 0.80: reward 2 (excellent match)
    
    Args:
        prompts: List of input prompts (unused in this function but required by GRPOTrainer)
        completions: List of model-generated completions (containing <answer> tags)
        reference_answers: List of ground truth answers
        **kwargs: Additional arguments
        
    Returns:
        List of discrete rewards (0, 1, 1.5, or 2) for each completion
    """
    if reference_answers is None or len(reference_answers) != len(completions):
        return [0.0] * len(completions)
    
    try:
        # Clear screen between evaluations
        print('\033c', end='')
        
        # Initialize DRAMA model
        drama = DRAMAModel.get_instance()
        model = drama["model"]
        tokenizer = drama["tokenizer"]
        device = drama["device"]
        
        # Extract answers
        extracted_answers = [extract_after_thinking(c[0]["content"]) for c in completions]
        question = prompts[0][-1]['content'] if prompts and len(prompts[0]) > 0 else "No prompt provided"
        
        # Print evaluation header
        print(format_box("Semantic Evaluation", f"Question: {question[:70]}..." if len(question) > 70 else question))
        
        # Encode and compute similarities
        with torch.no_grad():
            gen_embs = model.encode_documents(tokenizer, extracted_answers)
            ref_embs = model.encode_documents(tokenizer, reference_answers)
            similarities = (gen_embs @ ref_embs.T).diagonal().cpu().numpy()
        
        rewards = []
        for i, (sim, ans, ref) in enumerate(zip(similarities, extracted_answers, reference_answers)):
            # Determine reward tier
            if sim < 0.33:
                reward = 0.0
            elif 0.33 <= sim < 0.60:
                reward = 1.0
            elif 0.60 <= sim < 0.80:
                reward = 1.5
            else:
                reward = 2.0
            rewards.append(reward)
            
            # Print comparison for first completion (or all if debugging)
            if i == 0 or kwargs.get('verbose', False):
                print(format_box("Reference Answer", ref))
                print(format_box("Model Response", ans))
                print(f"\n\033[1mREWARD ANALYSIS\033[0m")
                print(format_reward_bar(sim, reward))
                print(f"\n\033[90mSimilarity breakdown:\033[0m")
                print(f"- Raw similarity score: {sim:.4f}")
                print(f"- Reward tier: {reward}")
                print("-" * 50 + "\n")
        
        return rewards
    
    except Exception as e:
        logging.warning(f"Semantic similarity reward failed: {str(e)}")
        return [0.0] * len(completions)

# </REWARDS>