# def format_box(title: str, content: str, title_color: str = '\033[91m', width: int = 100) -> str:
#     """Helper to create consistent bordered boxes with colored titles"""
#     reset = '\033[0m'
#     lines = []
    
#     top_bottom = f'╔{"═" * (width - 2)}╗'
#     lines.append(top_bottom)
    
#     title_line = f'║ {title_color}{title.upper():<{width-3}}{reset}║'
#     lines.append(title_line)
    
#     separator = f'╠{"═" * (width - 2)}╣'
#     lines.append(separator)
    
#     max_lines = 10
#     content_lines = content.split('\n')
#     truncated = len(content_lines) > max_lines
    
#     for line in content_lines[:max_lines]:
#         visible_line = line.rstrip('\n')
#         if len(visible_line) > width - 4:
#             clean_line = visible_line[:width-7].strip() + '...'
#             lines.append(f'║ {clean_line.ljust(width-4)} ║')
#         else:
#             lines.append(f'║ {visible_line.ljust(width-4)} ║')
    
#     if truncated:
#         lines.append(f'║ {"[...]".ljust(width-4)} ║')
    
#     bottom = f'╚{"═" * (width - 2)}╝'
#     lines.append(bottom)
    
#     return '\n'.join(lines)

import re

def format_box(title: str, content: str, title_color: str = '\033[96m', width: int = 100) -> str:
    """Helper to create consistent bordered boxes with colored titles"""
    reset = '\033[0m'
    lines = []
    
    # Удаляем ANSI-коды для расчёта длины
    clean_title = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', title)
    visible_width = len(clean_title)
    
    # Верхняя граница
    top_bottom = f'╔{"═" * (width - 2)}╗'
    lines.append(top_bottom)
    
    # Заголовок с цветом и правильным выравниванием
    title_padding = width - 3 - visible_width
    title_line = f'║ {title_color}{title.upper()}{reset}{" " * title_padding}║'
    lines.append(title_line)
    
    # Разделитель
    separator = f'╠{"═" * (width - 2)}╣'
    lines.append(separator)
    
    # Обработка контента
    max_lines = 10
    content_lines = content.split('\n')
    truncated = len(content_lines) > max_lines
    
    for line in content_lines[:max_lines]:
        # Удаляем ANSI-коды для расчёта длины
        clean_line = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', line)
        visible_line = clean_line.rstrip('\n')
        
        if len(visible_line) > width - 4:
            truncated_line = visible_line[:width-7].strip() + '...'
            padded_line = f'║ {truncated_line.ljust(width - 4)} ║'
        else:
            padded_line = f'║ {line.ljust(width - 4)} ║'  # Оригинальная строка с цветами
        
        lines.append(padded_line)
    
    if truncated:
        lines.append(f'║ {"[...]".ljust(width - 4)} ║')
    
    # Нижняя граница
    bottom = f'╚{"═" * (width - 2)}╝'
    lines.append(bottom)
    
    return '\n'.join(lines)

def print_section(title: str, content: str):
    """Print a section with colored formatting"""
    color_map = {
        "EVALUATION RESULT": '\033[91m',    # R
        "EXPECTED ANSWER": '\033[92m',      # G
        "MODEL RESPONSE": '\033[96m'        # B
    }
    
    color = color_map.get(title.upper(), '\033[0m')
    reset = '\033[0m'
    
    print(f'\n{color}{title.upper()}{reset}')
    print(f'{color}{"-" * len(title)}{reset}')
    print(content if len(content) < 1000 else content[:997] + '...')
    print()

def format_reward_bar(similarity: float, reward: float, width: int = 50) -> str:
    """Create visual reward meter with color coding"""
    colors = [
        (0.0, '\033[91m'),    # Red
        (0.5, '\033[93m'),    # Yellow
        (1.0, '\033[92m')     # Green
    ]
    reset = '\033[0m'
    
    # Find appropriate color
    reward_color = colors[-1][1]
    for threshold, color in colors:
        if reward <= threshold:
            reward_color = color
            break
    
    filled = int(similarity * width)
    bar = f"{reward_color}▶{'─' * filled}{reset}{'─' * (width - filled)}▶"
    return f"Similarity: {similarity:.2f} {bar} Reward: {reward_color}{reward:.2f}{reset}"


