from typing import List, Optional, Tuple
import torch
import logging

from src.utils.data.extraction import (extract_tagged_answer, 
                                    extract_after_thinking, 
                                    extract_boxed_answer,
                                    extract_thinking
                                    )
from src.reward_models.drama import DRAMAModel
from src.utils.stdout.model_answer import print_section, format_box, format_reward_bar
from src.validators.russian import RussianWordValidator

import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    
def detect_reflection_marks(text: str) -> float:
    reflection_indicators = {
        'questioning': {
            'patterns': ["?", "возможно", "может быть", "стоит ли", "верно ли"],
            'weight': 0.4
        },
        'uncertainty': {
            'patterns': ["сомневаюсь", "не уверен", "неоднозначно", "спорно", "однако"],
            'weight': 0.3
        },
        'analysis': {
            'patterns': ["с одной стороны", "с другой стороны", "альтернатива", "компромисс", "проверка"],
            'weight': 0.2
        },
        'metacognition': {
            'patterns': ["думаю", "думая", "рассуждая", "анализируя", "оценивая"],
            'weight': 0.1
        },
        'cognitivity': {
            'patterns': ["следовательно", "таким образом", "делать вывод", "утверждая"],
            'weight': 0.2
        }
    }

    # Предобработка текста
    text_lower = text.lower()
    # Удаляем пунктуацию, кроме специальных символов (например, "?")
    text_clean = re.sub(r'[^\w\s?]', ' ', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    score = 0.0
    for category in reflection_indicators.values():
        category_count = 0
        for pattern in category['patterns']:
            # Экранируем спецсимволы и учитываем границы слов
            if pattern == "?":
                # Ищем "?" как отдельный символ
                regex = r'(^| )\?( |$)'  # проверяем пробелы вокруг
                matches = re.findall(regex, text_lower)
                count = len(matches)
            else:
                # Для текста без лишней пунктуации ищем целые слова/фразы
                regex = r'\b' + re.escape(pattern) + r'\b'
                count = len(re.findall(regex, text_clean, flags=re.IGNORECASE))
            category_count += count
        score += category_count * category['weight']
        print(f"Category: {category}, Count: {category_count}, Score Contribution: {category_count * category['weight']}")

    # Нормализация
    normalized_score = 1 / (1 + math.exp(-score / 2))
    print(f"Total Score: {score}, Normalized Score: {normalized_score}")
    return normalized_score

# /<UTILITIES>

# <REWARDS>

def format_reward(completions: list, **kwargs) -> list[float]:
    """
    Universal format reward function
    """
    rewards = []
    for comp in completions:
        content = comp[0]["content"]
        if check_xml_structure(content, ('think')):
            rewards.append(1.0 if '\n' in get_xml_content(content, 'think') else 0.5)
        else:
            rewards.append(0.0)
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
        
        if valid_ratio >= 0.95:     reward = 1.0
        elif valid_ratio >= 0.85:   reward = 0.8
        elif valid_ratio >= 0.75:   reward = 0.6
        elif valid_ratio >= 0.5:    reward = 0.3
        else:                       reward = 0.0
            
        rewards.append(round(reward, 2))
    
    return rewards

def multilingual_coherence_penalty(completions: list, **kwargs) -> List[float]:
    """
    Комплексная проверка с рациональными штрафами за:
    - Смешанные алфавиты внутри слов
    - Китайские иероглифы
    - Длинные иностранные последовательности
    
    Возвращает:
    - 0.0 если нет нарушений
    - Прогрессивные штрафы от -0.1 до -2.0
    """
    # Расширенные регулярные выражения
    CHINESE_CHAR = re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]')
    MIXED_ALPHABET = re.compile(
        r'\b(?:[a-zа-яё]*[a-z][a-zа-яё]*[а-яё][a-zа-яё]*|[а-яёa-z]*[а-яё][a-zа-яё]*[a-z][a-zа-яё]*)\b', 
        re.IGNORECASE | re.UNICODE
    )
    TECHNICAL_BLOCKS = re.compile(r'`.*?`|\\[a-z]+|\$.*?\$|http\S+', re.IGNORECASE)

    penalties = []
    for completion in completions:
        try:
            text = completion[0].get('content', '') if isinstance(completion, list) else completion.get('content', '')
        except (AttributeError, KeyError, IndexError):
            text = ''

        # Игнорируем технические блоки
        clean_text = TECHNICAL_BLOCKS.sub(' ', text)
        total_penalty = 0.0

        # 1. Китайские иероглифы (только явные случаи)
        chinese_chars = CHINESE_CHAR.findall(clean_text)
        total_penalty += min(len(chinese_chars) * 0.1, 0.5)  # Макс -0.5

        # 2. Смешанные слова (только реальные смеси внутри слов)
        mixed_words = [
            word for word in MIXED_ALPHABET.findall(clean_text) 
            if not (word.isascii() or word.isalpha())
        ]
        total_penalty += min(len(mixed_words) * 0.3, 1.0)  # Макс -1.0

        # 3. Длинные иностранные последовательности (только вне технических блоков)
        foreign_sequences = re.findall(r'[^\u0400-\u04FF\s]{8,}', clean_text)
        total_penalty += min(len(foreign_sequences) * 0.2, 0.5)  # Макс -0.5

        # Нормализация и ограничение штрафа
        normalized_penalty = min(total_penalty, 2.0)
        penalties.append(round(-normalized_penalty, 2))

    return penalties

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
        
        if non_russian_ratio < 0.33:    reward = 1.0
        elif non_russian_ratio < 0.5:   reward = 0.8
        elif non_russian_ratio < 0.75:  reward = 0.5
        else:                           reward = 0.0
            
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
        
        if correct_ratio > 0.8:     reward = 1.0
        elif correct_ratio > 0.6:   reward = 0.5
        else:                       reward = 0.0
            
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
        print('\033c', end='')
        
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
        
        print_section("Detailed Comparison", 
                    f"Extracted Answer: {extracted_responses[0]}\n"
                    f"Matches Expected: {extracted_responses[0] == answer[0]}\n"
                    f"{reward_bar}")

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def accuracy_reward(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
    """
    Reward function that evaluates the accuracy of model completions against the ground truth.

    This function checks if the completions generated by the model match the expected solutions.
    Both the solution and the completion are parsed using the same LaTeX parsing configuration
    to ensure consistent comparison. If the solution cannot be parsed, a reward of 0.0 is assigned.

    Args:
        completions (list): A list of model-generated completions, where each completion is a list of
                            dictionaries containing a "content" key with the generated text.
        solution (list): A list of ground truth solutions corresponding to the completions.
        **kwargs: Additional keyword arguments (e.g., for logging or custom extraction configs).

    Returns:
        list[float]: A list of rewards (1.0 or 0.0) based on the accuracy of each completion.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    extraction_config = LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=True,
            malformed_operators=False,
            basic_latex=True,
            equations=True,
            boxed="all",
            units=True,
        ),
        boxed_match_priority=1,
        try_extract_without_anchor=False,
    )
    
    for content, sol in zip(contents, answer):
        gold_parsed = parse(
            sol,
            extraction_mode="any_match",
            extraction_config=[extraction_config],
        )
        
        if not gold_parsed:
            rewards.append(0.0)
            continue
        
        answer_parsed = parse(
            content,
            extraction_mode="any_match",
            extraction_config=[extraction_config],
        )
        
        reward = 1.0 if (
            answer_parsed and verify(
                answer_parsed, gold_parsed
                )
            ) else 0.0
        rewards.append(reward)
    
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
        rewards.append(0.2 if re.search(pattern, content) else 0.0)
    return rewards

def similarity_reward(prompts: list, completions: list, reference_answers: list, **kwargs) -> List[float]:
    """
    Оценка сходства между сгенерированными ответами и эталонными ответами.
    
    Args:
        completions: Список сгенерированных ответов.
        reference_answers: Список эталонных ответов.
        
    Returns:
        Список значений награды на основе сходства.
    """
    texts = [comp[0]['content'] for comp in completions] + reference_answers
    vectorizer = TfidfVectorizer().fit(texts)
    tfidf_matrix = vectorizer.transform(texts)
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    rewards = []
    num_completions = len(completions)
    
    for i in range(num_completions):
        sim_score = similarity_matrix[i][num_completions]

        if sim_score > 0.8:     reward = 1.0
        elif sim_score > 0.6:   reward = 0.5
        elif sim_score > 0.3:   reward = 0.1
        else:                   reward = 0

        print(f"\n\033[1mREWARD ANALYSIS\033[0m")
        print(format_reward_bar(sim_score, reward))
        print(f"\n\033[90mSimilarity breakdown:\033[0m")
        print(f"- Raw similarity score: {sim_score:.4f}")
        print(f"- Reward tier: {sim_score}")
        print("-" * 50 + "\n")
    
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

def ngram_penalty(
    completions: list, 
    ngram_size: int = 4, 
    max_penalty: float = -1.0, 
    min_safe_ngrams: int = 5,
    **kwargs
) -> List[float]:
    """
    Штраф за повторяющиеся фразы с защитой от ложных срабатываний.
    
    Особенности:
    - Игнорирует стоп-слова в N-граммах
    - Защита от коротких текстов
    - Прогрессивная шкала штрафов
    - Фильтрация служебных конструкций
    """
    # Стоп-слова для русского и английского
    STOP_WORDS = {
        'и', 'в', 'на', 'с', 'по', 'для', 'не', 'из', 'от', 'это',
        'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'it'
    }
    
    penalties = []
    for comp in completions:
        content = comp[0]['content'].lower()
        
        # Удаляем технические конструкции перед анализом
        clean_content = re.sub(r'<.*?>|\$.*?\$|\\\w+', ' ', content)
        words = [w for w in re.findall(r'\b\w+\b', clean_content) 
                if w not in STOP_WORDS and len(w) > 2]
        
        # Формируем N-граммы
        ngrams = [' '.join(words[i:i+ngram_size]) 
                 for i in range(len(words) - ngram_size + 1)]
        
        if len(ngrams) < min_safe_ngrams:
            penalties.append(0.0)
            continue
            
        # Считаем уникальность
        total = len(ngrams)
        unique = len(set(ngrams))
        uniqueness = unique / total
        
        # Квадратичная прогрессия штрафа
        penalty = -( (1 - uniqueness) ** 1.5 ) * max_penalty
        
        # Ограничиваем минимальный штраф
        penalties.append(max(penalty, max_penalty))
    
    return [round(p, 2) for p in penalties]

def ngram_reward(
    completions: list,
    ngram_size: int = 4,
    max_reward: float = 1.0,
    min_safe_ngrams: int = 5,
    **kwargs
) -> List[float]:
    """
    Бонус за уникальные фразы с защитой от ложных срабатываний.
    
    Особенности:
    - Игнорирует стоп-слова в N-граммах
    - Защита от коротких текстов
    - Прогрессивная шкала бонусов
    - Фильтрация служебных конструкций
    """
    # Стоп-слова для русского и английского
    STOP_WORDS = {
        'и', 'в', 'на', 'с', 'по', 'для', 'не', 'из', 'от', 'это',
        'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'it'
    }
    
    rewards = []
    for comp in completions:
        content = comp[0]['content'].lower()
        
        # Удаляем технические конструкции перед анализом
        clean_content = re.sub(r'<.*?>|\$.*?\$|\\\w+', ' ', content)
        words = [w for w in re.findall(r'\b\w+\b', clean_content) 
                if w not in STOP_WORDS and len(w) > 2]
        
        # Формируем N-граммы
        ngrams = [' '.join(words[i:i+ngram_size]) 
                 for i in range(len(words) - ngram_size + 1)]
        
        if len(ngrams) < min_safe_ngrams:
            rewards.append(0.0)
            continue
            
        # Считаем уникальность
        total = len(ngrams)
        unique = len(set(ngrams))
        uniqueness = unique / total
        
        # Квадратичная прогрессия бонуса
        reward = (uniqueness ** 1.5) * max_reward
        
        # Ограничиваем максимальный бонус
        rewards.append(min(reward, max_reward))
    
    return [round(b, 2) for b in rewards]

def reflection_reward(completions, **kwargs):
    rewards = []
    for comp in completions:
        text = comp[0]['content']
        
        reflection_content = extract_thinking(text)
        
        # Глубина рефлексии
        cognitive_score = detect_reflection_marks(reflection_content) * 3.0
        
        rewards.append(max(cognitive_score, 0.0))
    
    return [round(r, 2) for r in rewards]

# </REWARDS>