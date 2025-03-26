from src.utils.data.extraction import extract_xml_answer

import re
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

# <UTILITIES>

def count_xml(text: str) -> float:
    """
    Count specific XML tags in the given text and return a score based on their presence.

    The function checks for the presence of specific XML tags and assigns a score based on the following criteria:
    - Presence of a single <thought> tag increases the score by 0.125.
    - Presence of a single </thought> tag increases the score by 0.125.
    - Presence of a single <answer> tag increases the score by 0.125, but decreases the score based on the length of the text after the last </answer> tag.
    - Presence of a single </answer> tag increases the score by 0.125, but decreases the score based on the length of the text after the last </answer> tag.

    Args:
        text (str): The input text containing XML content.

    Returns:
        float: The calculated score based on the presence of XML tags.
    """
    count = 0.0
    if text.count("<thought>\n") == 1:
        count += 0.125
    if text.count("\n</thought>\n") == 1:
        count += 0.125
    if text.count("\n<diagram>\n") == 1:
        count += 0.125
    if text.count("\n</diagram>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

# /<UTILITIES>

# <REWARDS>

def correctness_reward_func(prompts: list, completions: list, answer: list, **kwargs) -> list[float]:
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
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('=' * 40, '\n')
    print(f'TASK / QUESTION:\n{question}')
    print('-' * 40, '\n')
    print(f'SOLUTION / ANSWER: \n{answer[0]}')
    print('-' * 40, '\n')
    print(f'RESPONSE:\n{responses[0]}')
    print('=' * 40, '\n')
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Evaluate whether the extracted answers consist solely of digits.

    For each extracted answer, this function checks if it is entirely numeric using the isdigit() method.
    If the condition is met (i.e., the answer is a number), a reward of 0.5 is returned; otherwise, 0.0.

    Args:
        completions (list): A list of generated completions.

    Returns:
        list[float]: A list of rewards for each completion based on whether the answer is numeric.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Evaluate whether the model adheres to a strict output format.

    This reward function checks that the model's output matches a specified format exactly.
    It uses regular expressions to ensure that the entire string conforms to the pattern.

    Args:
        completions (list): A list of generated completions.

    Returns:
        list[float]: A list of rewards for each completion based on strict format adherence.
    """
    pattern = r"^<thought>\n.*?\n<diagram>\n.*?\n</diagram>\n*?</thought>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Evaluate whether the model's output loosely adheres to a specified format.

    This reward function checks that the model's output follows a specified format without requiring
    exact matches for the entire string. It allows for arbitrary whitespace between tags and does not
    impose strict line breaks.

    Args:
        completions (list): A list of generated completions.

    Returns:
        list[float]: A list of rewards for each completion based on loose format adherence.
    """
    pattern = r"<thought>.*?</thought>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Calculate a score based on the presence of specific XML tags in the completions.

    This function counts the specific XML tags in each completion and returns a score based on their presence.

    Args:
        completions (list): A list of generated completions.

    Returns:
        list[float]: A list of scores for each completion based on the presence of XML tags.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

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

# </REWARDS>