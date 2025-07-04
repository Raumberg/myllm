from typing import Tuple, Any
import re

def extract_tagged_answer(text: str) -> str:
    """Extracts content between <answer> and </answer> tags in a given text.

    Args:
        text: Input string containing XML-style answer tags.

    Returns:
        Extracted answer content as a stripped string. If tags are malformed,
        returns everything after the last <answer> tag or before the first </answer>.

    Example:
        >>> extract_tagged_answer("Some text <answer>42</answer>")
        '42'
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_after_thinking(text: str) -> str | int:
    """Extracts all text after the closing </think> tag.

    Args:
        text: Input string containing XML-style think tags.

    Returns:
        All content after </think> as a stripped string. Returns empty string
        if </think> tag is not found.

    Example:
        >>> extract_after_thinking("<think>Consider...</think> The answer is 42")
        'The answer is 42'
    """
    parts = text.split("</think>")
    
    if len(parts) < 2:
        return ""
    
    full_answer = parts[1].strip()
    
    return full_answer

def extract_thinking_and_answer(text: str) -> Tuple[str]:
    """Extracts both reasoning (think) and answer content from XML-tagged text.

    Parses a string containing <think> and <answer> XML-style tags to extract:
    1. The reasoning process between <think> and </think> tags
    2. The final answer between <answer> and </answer> tags

    Args:
        text: Input string containing XML-style think/answer tags. Expected format:
              "<think>reasoning text</think> <answer>final answer</answer>"
              (order can vary, tags are searched independently)

    Returns:
        A tuple of two strings: (reasoning_text, answer_text). 
        Each string will be:
        - The stripped content between tags if found
        - Empty string ("") if corresponding tags are missing
        - Content from first occurrence if multiple tags exist

    Examples:
        >>> extract_thinking_and_answer(
        ...     "<think>Calculate 40+2</think> <answer>42</answer>")
        ('Calculate 40+2', '42')

        >>> extract_thinking_and_answer("No tags here")
        ('', '')

        >>> extract_thinking_and_answer(
        ...     "<answer>42</answer> Text <think>Reasoning</think>")
        ('Reasoning', '42')

    Note:
        - Uses DOTALL flag in regex, so tags can span multiple lines
        - Returns first match if multiple think/answer blocks exist
        - Maintains original whitespace within tags but strips outer whitespace
    """
    reasoning_pat = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    answer_pat    = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    reasoning_match = reasoning_pat.search(text)
    answer_match    = answer_pat.search(text)

    reasoning_str = reasoning_match.group(1).strip() if reasoning_match else ""
    answer_str    = answer_match.group(1).strip() if answer_match else ""

    return (reasoning_str, answer_str)

def extract_thinking(text: str) -> str | None:
    reasoning_pat = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    reasoning_match = reasoning_pat.search(text)
    reasoning_str = reasoning_match.group(1).strip() if reasoning_match else ""
    return reasoning_str

def extract_boxed_answer(text: str) -> str | None:
    """Extracts content within \boxed{} in a given text.

    Args:
        text: Input string potentially containing \boxed{}.

    Returns:
        Content inside \boxed{} as a stripped string if marker exists, otherwise None.
    """
    start = text.find(r'\boxed{')
    if start == -1:
        return None
    
    end = start + len(r'\boxed{')
    brace_count = 1 

    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1

    if brace_count == 0:
        return text[start + len(r'\boxed{'):end - 1].strip()
    
    return None

def extract_hash_answer(text: str) -> str | None:
    """Extracts content following #### marker in a given text.

    Args:
        text: Input string potentially containing #### delimiter.

    Returns:
        Content after #### as a stripped string if marker exists, otherwise None.

    Example:
        >>> extract_hash_answer("The result is #### 42")
        '42'
        >>> extract_hash_answer("No marker here")
        None
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Additional utility functions migrated from legacy

def detect_reflection_marks(text: str) -> float:  # noqa: D401
    """Ported logic to score reflection markers in Russian text."""

    import math, re

    reflection_indicators = {
        "questioning": {
            "patterns": ["?", "возможно", "может быть", "стоит ли", "верно ли"],
            "weight": 0.4,
        },
        "uncertainty": {
            "patterns": ["сомневаюсь", "не уверен", "неоднозначно", "спорно", "однако"],
            "weight": 0.3,
        },
        "analysis": {
            "patterns": ["с одной стороны", "с другой стороны", "альтернатива", "компромисс", "проверка"],
            "weight": 0.2,
        },
        "metacognition": {
            "patterns": ["думаю", "думая", "рассуждая", "анализируя", "оценивая"],
            "weight": 0.1,
        },
        "cognitivity": {
            "patterns": ["следовательно", "таким образом", "делать вывод", "утверждая"],
            "weight": 0.2,
        },
    }

    text_lower = text.lower()
    text_clean = re.sub(r"[^\w\s?]", " ", text_lower)
    text_clean = re.sub(r"\s+", " ", text_clean).strip()

    score = 0.0
    for cat in reflection_indicators.values():
        cnt = 0
        for pat in cat["patterns"]:
            if pat == "?":
                cnt += text_lower.count("?")
            else:
                cnt += len(re.findall(rf"\b{re.escape(pat)}\b", text_clean))
        score += cnt * cat["weight"]

    return 1 / (1 + math.exp(-score / 2))

def count_unique_ngrams(text: str, n: int = 4):  # noqa: D401
    """Return unique ngram count and total ngrams for text."""
    if len(text) < n:
        return 0, 0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)), len(grams)

# --- Extraction helpers -------------------------------------------------------

def get_content(completion: Any) -> str:  # noqa: D401
    if isinstance(completion, list) and completion:

        if isinstance(completion[0], dict):

            return completion[0].get("content", "")
        
        return str(completion[0])
    
    if isinstance(completion, dict):

        return completion.get("content", "")
    
    return str(completion)

def get_xml_content(text: str, tag: str) -> str:  # noqa: D401
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""

def check_xml_structure(text: str, required_tags: Tuple[str]) -> bool:  # noqa: D401
    return all(text.count(f"<{t}>") == 1 and text.count(f"</{t}>") == 1 for t in required_tags)


__all__ = [
    "extract_tagged_answer",
    "extract_after_thinking",
    "extract_thinking_and_answer",
    "extract_thinking",
    "extract_boxed_answer",
    "extract_hash_answer",
    "detect_reflection_marks",
    "count_unique_ngrams",
    "get_content",
    "get_xml_content",
    "check_xml_structure"
]