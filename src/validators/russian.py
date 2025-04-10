import re
from typing import List
from pymorphy2 import MorphAnalyzer
from functools import lru_cache

class RussianWordValidator:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.word_pattern = re.compile(r'\b[а-яё]+\b', re.IGNORECASE)
        
        # Словарь исключений для специальных случаев
        self.exceptions = {
            ''
        }
    
    @lru_cache(maxsize=10000)
    def is_valid_word(self, word: str) -> bool:
        """Проверяет слово с учётом морфологии и исключений"""
        lower_word = word.lower()
        
        # Проверка исключений
        if lower_word in self.exceptions:
            return True
            
        # Проверка через морфологический анализатор
        parsed = self.morph.parse(lower_word)
        if any( p.score >= 0.1 and 
                'UNKN' not in p.tag and
                'LATN' not in p.tag and
                'Erro' not in p.methods_stack
                for p in parsed ):
            return True
            
        return False

