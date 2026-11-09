"""
Letter-to-Word Conversion Module for ISL Hand Gesture Recognition
Converts detected letter sequences into words with spell-checking
"""

import time
from collections import deque
from typing import List, Optional, Tuple
import sys
import os

try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False
    print("Warning: PyEnchant not available. Spell-checking will be disabled.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import setup_logging


class LetterToWordConverter:
    """
    Converts detected letter sequences into words.
    Handles word boundary detection, duplicate removal, and spell-checking.
    """
    
    def __init__(
        self,
        pause_threshold: float = 2.0,
        min_word_length: int = 2,
        max_word_length: int = 20,
        use_spell_check: bool = True
    ):
        """
        Initialize the letter-to-word converter.
        
        Args:
            pause_threshold: Time threshold (seconds) to detect word boundaries
            min_word_length: Minimum word length to consider valid
            max_word_length: Maximum word length to consider valid
            use_spell_check: Whether to use spell-checking
        """
        self.pause_threshold = pause_threshold
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.use_spell_check = use_spell_check and ENCHANT_AVAILABLE
        
        # Initialize spell checker
        if self.use_spell_check:
            try:
                self.spell_checker = enchant.Dict("en_US")
            except:
                print("Warning: Could not initialize spell checker. Disabling spell-check.")
                self.use_spell_check = False
        
        # Letter buffer
        self.letter_buffer = []
        self.letter_timestamps = []
        
        # Word storage
        self.completed_words = []
        self.current_word_in_progress = ""
        
        # Common ISL words (frequently used)
        self.common_isl_words = self.load_common_isl_words()
        
        # Statistics
        self.total_letters_detected = 0
        self.total_words_formed = 0
        
        # Logging
        self.logger = setup_logging()
        self.logger.info("LetterToWordConverter initialized")
    
    def load_common_isl_words(self) -> set:
        """
        Load common ISL words.
        
        Returns:
            Set of common words
        """
        common_words = {
            'HELLO', 'HI', 'THANKS', 'THANK', 'YOU', 'PLEASE', 'YES', 'NO',
            'GOOD', 'BAD', 'HELP', 'SORRY', 'WELCOME', 'BYE', 'OKAY', 'OK',
            'MORNING', 'EVENING', 'NIGHT', 'DAY', 'NAME', 'MY', 'YOUR',
            'HOW', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'WHO', 'WHICH',
            'WATER', 'FOOD', 'EAT', 'DRINK', 'WANT', 'NEED', 'HAVE',
            'LIKE', 'LOVE', 'HATE', 'HAPPY', 'SAD', 'ANGRY', 'TIRED',
            'HOME', 'SCHOOL', 'WORK', 'FAMILY', 'FRIEND', 'MOTHER', 'FATHER',
            'SISTER', 'BROTHER', 'SON', 'DAUGHTER', 'CHILD', 'BABY'
        }
        return common_words
    
    def add_letter(self, letter: str, timestamp: Optional[float] = None) -> None:
        """
        Add a detected letter to the buffer.
        
        Args:
            letter: Detected letter
            timestamp: Detection timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Check for word boundary before adding
        if self.check_word_boundary(timestamp):
            self.form_word_from_buffer()
        
        # Add letter and timestamp
        self.letter_buffer.append(letter)
        self.letter_timestamps.append(timestamp)
        
        self.total_letters_detected += 1
        
        # Update word in progress
        self.current_word_in_progress = ''.join(self.letter_buffer)
    
    def check_word_boundary(self, current_timestamp: Optional[float] = None) -> bool:
        """
        Check if a word boundary should be inserted based on pause.
        
        Args:
            current_timestamp: Current timestamp
        
        Returns:
            True if word boundary detected
        """
        if not self.letter_buffer or not self.letter_timestamps:
            return False
        
        if current_timestamp is None:
            current_timestamp = time.time()
        
        # Check time since last letter
        time_since_last_letter = current_timestamp - self.letter_timestamps[-1]
        
        if time_since_last_letter >= self.pause_threshold:
            return True
        
        return False
    
    def remove_duplicate_letters(self, letters: List[str]) -> List[str]:
        """
        Remove consecutive duplicate letters.
        
        Args:
            letters: List of letters
        
        Returns:
            List with duplicates removed
        """
        if not letters:
            return letters
        
        cleaned = [letters[0]]
        
        for letter in letters[1:]:
            if letter != cleaned[-1]:
                cleaned.append(letter)
        
        return cleaned
    
    def form_word_from_buffer(self) -> Optional[str]:
        """
        Form word from current letter buffer.
        
        Returns:
            Formed word or None if buffer is empty/invalid
        """
        if not self.letter_buffer:
            return None
        
        # Remove consecutive duplicates
        cleaned_letters = self.remove_duplicate_letters(self.letter_buffer)
        
        # Form word
        word = ''.join(cleaned_letters)
        
        # Check word length
        if len(word) < self.min_word_length:
            self.logger.info(f"Word too short: '{word}' (length {len(word)})")
            self.reset_letter_buffer()
            return None
        
        if len(word) > self.max_word_length:
            self.logger.info(f"Word too long: '{word}' (length {len(word)})")
            self.reset_letter_buffer()
            return None
        
        # Spell check and get suggestions if needed
        if self.use_spell_check:
            if word.upper() in self.common_isl_words:
                # Common ISL word - accept without spell check
                final_word = word
            elif self.spell_checker.check(word):
                # Valid English word
                final_word = word
            else:
                # Invalid word - get suggestions
                suggestions = self.get_suggestions(word)
                
                if suggestions:
                    # Use first suggestion
                    final_word = f"{word}[?→{suggestions[0]}]"
                    self.logger.info(f"Spell check: '{word}' → suggested: {suggestions[0]}")
                else:
                    final_word = f"{word}[?]"
                    self.logger.info(f"Spell check: no suggestions for '{word}'")
        else:
            final_word = word
        
        # Add to completed words
        self.completed_words.append({
            'word': final_word,
            'original': word,
            'timestamp': time.time(),
            'letters_used': len(self.letter_buffer)
        })
        
        self.total_words_formed += 1
        self.logger.info(f"Word formed: '{final_word}'")
        
        # Reset buffer
        self.reset_letter_buffer()
        
        return final_word
    
    def spell_check(self, word: str) -> bool:
        """
        Check if word is spelled correctly.
        
        Args:
            word: Word to check
        
        Returns:
            True if word is valid
        """
        if not self.use_spell_check:
            return True
        
        # Check common ISL words
        if word.upper() in self.common_isl_words:
            return True
        
        # Check with spell checker
        return self.spell_checker.check(word)
    
    def get_suggestions(self, word: str, max_suggestions: int = 3) -> List[str]:
        """
        Get spelling suggestions for word.
        
        Args:
            word: Word to get suggestions for
            max_suggestions: Maximum number of suggestions
        
        Returns:
            List of suggested words
        """
        if not self.use_spell_check:
            return []
        
        try:
            suggestions = self.spell_checker.suggest(word)
            return suggestions[:max_suggestions]
        except:
            return []
    
    def get_current_word_in_progress(self) -> str:
        """
        Get the current word being formed.
        
        Returns:
            Current word in progress
        """
        return self.current_word_in_progress
    
    def get_completed_words(self, limit: Optional[int] = None) -> List[dict]:
        """
        Get completed words.
        
        Args:
            limit: Maximum number of words to return (most recent)
        
        Returns:
            List of completed words with metadata
        """
        if limit is None:
            return self.completed_words
        else:
            return self.completed_words[-limit:]
    
    def get_last_word(self) -> Optional[str]:
        """
        Get the last completed word.
        
        Returns:
            Last word or None if no words formed
        """
        if self.completed_words:
            return self.completed_words[-1]['word']
        return None
    
    def reset_letter_buffer(self) -> None:
        """Reset letter buffer."""
        self.letter_buffer.clear()
        self.letter_timestamps.clear()
        self.current_word_in_progress = ""
    
    def reset_all(self) -> None:
        """Reset all buffers and storage."""
        self.reset_letter_buffer()
        self.completed_words.clear()
        self.total_letters_detected = 0
        self.total_words_formed = 0
        self.logger.info("All buffers reset")
    
    def manual_word_boundary(self) -> Optional[str]:
        """
        Manually insert word boundary and form word.
        
        Returns:
            Formed word or None
        """
        return self.form_word_from_buffer()
    
    def get_statistics(self) -> dict:
        """
        Get conversion statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_letters_detected': self.total_letters_detected,
            'total_words_formed': self.total_words_formed,
            'letters_in_buffer': len(self.letter_buffer),
            'current_word_length': len(self.current_word_in_progress),
            'completed_words_count': len(self.completed_words)
        }


def main():
    """Main function for testing."""
    print("=" * 50)
    print("Letter-to-Word Converter Test")
    print("=" * 50)
    
    # Initialize converter
    converter = LetterToWordConverter(
        pause_threshold=2.0,
        min_word_length=2,
        use_spell_check=True
    )
    
    print("\nEnter letters one by one (or 'pause' to simulate pause)")
    print("Type 'word' to manually form word")
    print("Type 'stats' to see statistics")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Enter letter: ").strip().upper()
        
        if user_input == 'QUIT':
            break
        
        elif user_input == 'PAUSE':
            print("Simulating pause...")
            time.sleep(2.5)
            if converter.check_word_boundary():
                word = converter.form_word_from_buffer()
                if word:
                    print(f"✓ Word formed: {word}")
        
        elif user_input == 'WORD':
            word = converter.manual_word_boundary()
            if word:
                print(f"✓ Word formed: {word}")
            else:
                print("✗ No word formed (buffer empty or invalid)")
        
        elif user_input == 'STATS':
            stats = converter.get_statistics()
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print(f"\nCompleted words: {[w['word'] for w in converter.get_completed_words()]}")
            print()
        
        elif len(user_input) == 1 and user_input.isalpha():
            converter.add_letter(user_input)
            print(f"Current word: {converter.get_current_word_in_progress()}")
        
        else:
            print("Invalid input. Enter a single letter.")
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("Final Statistics")
    print("=" * 50)
    stats = converter.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if converter.completed_words:
        print("\nCompleted Words:")
        for i, word_info in enumerate(converter.get_completed_words(), 1):
            print(f"  {i}. {word_info['word']} (from {word_info['letters_used']} letters)")


if __name__ == "__main__":
    main()
