"""
Word Suggestion System for ISL
Provides dictionary-based word suggestions with English and Hindi support
"""

import os
import sys
from typing import List, Optional, Set
from difflib import SequenceMatcher
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import setup_logging


class WordSuggester:
    """
    Provides intelligent word suggestions based on partial input.
    Supports English and optionally Hindi dictionaries.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None,
                 max_suggestions: int = 4):
        """
        Initialize word suggester.
        
        Args:
            dictionary_path: Path to custom dictionary file (optional)
            max_suggestions: Maximum number of suggestions to return
        """
        self.logger = setup_logging()
        self.max_suggestions = max_suggestions
        
        # Load dictionaries
        self.english_words = self._load_english_dictionary()
        self.custom_words = self._load_custom_dictionary(dictionary_path)
        
        # Combine dictionaries
        self.all_words = self.english_words | self.custom_words
        
        self.logger.info(
            f"WordSuggester initialized with {len(self.all_words)} words"
        )
    
    def _load_english_dictionary(self) -> Set[str]:
        """
        Load basic English dictionary.
        
        Returns:
            Set of English words
        """
        # Common English words (top ~1000 most used)
        common_words = {
            'THE', 'BE', 'TO', 'OF', 'AND', 'A', 'IN', 'THAT', 'HAVE', 'I',
            'IT', 'FOR', 'NOT', 'ON', 'WITH', 'HE', 'AS', 'YOU', 'DO', 'AT',
            'THIS', 'BUT', 'HIS', 'BY', 'FROM', 'THEY', 'WE', 'SAY', 'HER', 'SHE',
            'OR', 'AN', 'WILL', 'MY', 'ONE', 'ALL', 'WOULD', 'THERE', 'THEIR', 'WHAT',
            'SO', 'UP', 'OUT', 'IF', 'ABOUT', 'WHO', 'GET', 'WHICH', 'GO', 'ME',
            'WHEN', 'MAKE', 'CAN', 'LIKE', 'TIME', 'NO', 'JUST', 'HIM', 'KNOW', 'TAKE',
            'PEOPLE', 'INTO', 'YEAR', 'YOUR', 'GOOD', 'SOME', 'COULD', 'THEM', 'SEE', 'OTHER',
            'THAN', 'THEN', 'NOW', 'LOOK', 'ONLY', 'COME', 'ITS', 'OVER', 'THINK', 'ALSO',
            'BACK', 'AFTER', 'USE', 'TWO', 'HOW', 'OUR', 'WORK', 'FIRST', 'WELL', 'WAY',
            'EVEN', 'NEW', 'WANT', 'BECAUSE', 'ANY', 'THESE', 'GIVE', 'DAY', 'MOST', 'US',
            'IS', 'WAS', 'ARE', 'BEEN', 'HAS', 'HAD', 'WERE', 'SAID', 'DID', 'HAVING',
            'TELL', 'HELP', 'LOVE', 'NEED', 'HERE', 'FEEL', 'HOME', 'SCHOOL', 'HAND', 'FOOD',
            'WATER', 'FAMILY', 'FRIEND', 'PLEASE', 'THANK', 'YES', 'NO', 'HELLO', 'GOODBYE', 'SORRY',
            # ISL/Sign language specific
            'SIGN', 'LANGUAGE', 'DEAF', 'HEAR', 'UNDERSTAND', 'LEARN', 'TEACH', 'SPEAK',
            'LETTER', 'WORD', 'SENTENCE', 'GESTURE', 'COMMUNICATE', 'EXPRESSION',
            # Common verbs
            'EAT', 'DRINK', 'SLEEP', 'WALK', 'RUN', 'SIT', 'STAND', 'READ', 'WRITE',
            'PLAY', 'WORK', 'STUDY', 'TALK', 'LISTEN', 'WATCH', 'WAIT', 'STOP', 'START',
            # Common nouns
            'PERSON', 'MAN', 'WOMAN', 'CHILD', 'BOY', 'GIRL', 'MOTHER', 'FATHER', 'BROTHER', 'SISTER',
            'HOUSE', 'ROOM', 'DOOR', 'WINDOW', 'TABLE', 'CHAIR', 'BED', 'PHONE', 'BOOK', 'CAR',
            'MORNING', 'AFTERNOON', 'EVENING', 'NIGHT', 'TODAY', 'TOMORROW', 'YESTERDAY',
            # Common adjectives
            'GOOD', 'BAD', 'BIG', 'SMALL', 'HOT', 'COLD', 'FAST', 'SLOW', 'HAPPY', 'SAD',
            'EASY', 'HARD', 'OLD', 'NEW', 'LONG', 'SHORT', 'HIGH', 'LOW', 'NEAR', 'FAR',
            # Numbers
            'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'TEN',
            # Colors
            'RED', 'BLUE', 'GREEN', 'YELLOW', 'BLACK', 'WHITE', 'ORANGE', 'PURPLE', 'PINK', 'BROWN',
            # Question words
            'WHO', 'WHAT', 'WHEN', 'WHERE', 'WHY', 'WHICH', 'HOW',
            # Pronouns
            'I', 'YOU', 'HE', 'SHE', 'IT', 'WE', 'THEY', 'ME', 'HIM', 'HER', 'US', 'THEM',
            'MY', 'YOUR', 'HIS', 'HER', 'ITS', 'OUR', 'THEIR',
            # Prepositions
            'IN', 'ON', 'AT', 'TO', 'FROM', 'WITH', 'ABOUT', 'AFTER', 'BEFORE', 'DURING',
            # Conjunctions
            'AND', 'BUT', 'OR', 'SO', 'BECAUSE', 'IF', 'WHEN', 'WHILE', 'UNTIL',
        }
        
        return common_words
    
    def _load_custom_dictionary(self, path: Optional[str]) -> Set[str]:
        """
        Load custom dictionary from file.
        
        Args:
            path: Path to dictionary file (JSON or text)
            
        Returns:
            Set of custom words
        """
        if not path or not os.path.exists(path):
            return set()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.json'):
                    data = json.load(f)
                    words = set(data) if isinstance(data, list) else set(data.keys())
                else:
                    # Plain text file, one word per line
                    words = {line.strip().upper() for line in f if line.strip()}
            
            self.logger.info(f"Loaded {len(words)} custom words from {path}")
            return words
        
        except Exception as e:
            self.logger.error(f"Error loading custom dictionary: {e}")
            return set()
    
    def add_word(self, word: str):
        """
        Add a word to the dictionary.
        
        Args:
            word: Word to add
        """
        if word:
            self.all_words.add(word.upper())
    
    def add_words(self, words: List[str]):
        """
        Add multiple words to the dictionary.
        
        Args:
            words: List of words to add
        """
        for word in words:
            self.add_word(word)
    
    def get_suggestions(self, partial_word: str, 
                       min_similarity: float = 0.6) -> List[str]:
        """
        Get word suggestions for partial input.
        
        Args:
            partial_word: Partial or complete word input
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of suggested words
        """
        if not partial_word:
            return []
        
        partial_word = partial_word.upper().strip()
        
        # If exact match exists, return it first
        if partial_word in self.all_words:
            exact_matches = [partial_word]
        else:
            exact_matches = []
        
        # Find words that start with the partial word
        prefix_matches = [
            word for word in self.all_words
            if word.startswith(partial_word) and word != partial_word
        ]
        
        # Sort prefix matches by length (shorter first)
        prefix_matches.sort(key=len)
        
        # If we have enough prefix matches, return them
        if len(exact_matches) + len(prefix_matches) >= self.max_suggestions:
            suggestions = exact_matches + prefix_matches
            return suggestions[:self.max_suggestions]
        
        # Find similar words using fuzzy matching
        similar_words = []
        for word in self.all_words:
            if word in exact_matches or word in prefix_matches:
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(partial_word, word)
            
            if similarity >= min_similarity:
                similar_words.append((word, similarity))
        
        # Sort by similarity (descending)
        similar_words.sort(key=lambda x: x[1], reverse=True)
        
        # Combine results
        suggestions = (
            exact_matches + 
            prefix_matches + 
            [word for word, _ in similar_words]
        )
        
        return suggestions[:self.max_suggestions]
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        return SequenceMatcher(None, word1, word2).ratio()
    
    def check_spelling(self, word: str) -> bool:
        """
        Check if word is in dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word exists in dictionary
        """
        return word.upper() in self.all_words
    
    def get_corrections(self, word: str, max_corrections: int = 3) -> List[str]:
        """
        Get spelling corrections for a word.
        
        Args:
            word: Word to correct
            max_corrections: Maximum number of corrections
            
        Returns:
            List of possible corrections
        """
        if self.check_spelling(word):
            return [word.upper()]
        
        # Find most similar words
        word_upper = word.upper()
        similarities = []
        
        for dict_word in self.all_words:
            similarity = self._calculate_similarity(word_upper, dict_word)
            similarities.append((dict_word, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top corrections
        return [word for word, _ in similarities[:max_corrections]]


class ISLWordSuggester(WordSuggester):
    """
    Extended word suggester specifically for ISL context.
    Includes common ISL phrases and patterns.
    """
    
    def __init__(self, dictionary_path: Optional[str] = None):
        super().__init__(dictionary_path, max_suggestions=4)
        
        # Add ISL-specific phrases and common patterns
        isl_phrases = {
            # Greetings
            'HELLO', 'HI', 'GOOD MORNING', 'GOOD AFTERNOON', 'GOOD EVENING',
            'GOODBYE', 'BYE', 'SEE YOU', 'WELCOME', 'NAMASTE',
            # Courtesy
            'PLEASE', 'THANK YOU', 'SORRY', 'EXCUSE ME', 'YOU ARE WELCOME',
            # Questions
            'HOW ARE YOU', 'WHAT IS YOUR NAME', 'WHERE ARE YOU FROM',
            'HOW OLD ARE YOU', 'WHAT TIME', 'HOW MUCH',
            # Responses
            'I AM FINE', 'MY NAME IS', 'I AM FROM', 'I DON\'T KNOW',
            'I UNDERSTAND', 'I DON\'T UNDERSTAND',
            # Common needs
            'I NEED HELP', 'CAN YOU HELP ME', 'WHERE IS', 'HOW TO',
        }
        
        # Split phrases and add individual words
        for phrase in isl_phrases:
            self.add_word(phrase)
            for word in phrase.split():
                self.add_word(word)


def demo_word_suggester():
    """Demo function to test word suggestion system."""
    print("=== ISL Word Suggester Demo ===\n")
    
    # Create suggester
    suggester = ISLWordSuggester()
    
    # Test suggestions
    test_cases = [
        'HEL',
        'THA',
        'SOR',
        'GRE',
        'FAM',
        'WOR',
        'SIG',
        'LEA',
    ]
    
    print("Testing word suggestions:")
    print("-" * 60)
    
    for partial in test_cases:
        suggestions = suggester.get_suggestions(partial)
        print(f"\n'{partial}' -> {suggestions}")
    
    # Test spelling check
    print("\n\nTesting spelling check:")
    print("-" * 60)
    
    test_words = ['HELLO', 'WRLD', 'THANK', 'FAMILEY']
    
    for word in test_words:
        is_correct = suggester.check_spelling(word)
        print(f"'{word}' - {'✓ Correct' if is_correct else '✗ Incorrect'}")
        
        if not is_correct:
            corrections = suggester.get_corrections(word)
            print(f"  Suggestions: {corrections}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_word_suggester()
