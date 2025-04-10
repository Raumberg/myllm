from typing import Tuple
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

def extract_boxed_answer(text: str) -> str | None:
    """Extracts content within \boxed{} in a given text.

    Args:
        text: Input string potentially containing \boxed{}.

    Returns:
        Content inside \boxed{} as a stripped string if marker exists, otherwise None.

    Example:
        >>> extract_boxed_answer("The result is \\boxed{42}")
        '42'
        >>> extract_boxed_answer("No boxed answer here")
        None
    """
    match = re.search(r'\\boxed{(.*?)}', text)
    
    if match:
        return match.group(1).strip()
    
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