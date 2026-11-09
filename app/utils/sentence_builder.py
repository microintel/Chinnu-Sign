"""
Word-to-Sentence Formation Module for ISL Hand Gesture Recognition
Converts word sequences into grammatically correct sentences using ISL grammar rules
"""

import time
from typing import List, Optional, Tuple
import sys
import os

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
    
    # Download required NLTK data (only once)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Basic sentence formation will be used.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import setup_logging


class WordToSentenceConverter:
    """
    Converts word sequences into grammatically correct sentences.
    Applies ISL grammar rules (Subject-Object-Verb structure).
    """
    
    def __init__(
        self,
        sentence_pause: float = 5.0,
        min_sentence_length: int = 2,
        max_sentence_length: int = 15,
        use_isl_grammar: bool = True
    ):
        """
        Initialize the word-to-sentence converter.
        
        Args:
            sentence_pause: Time threshold (seconds) to detect sentence boundaries
            min_sentence_length: Minimum number of words in sentence
            max_sentence_length: Maximum number of words in sentence
            use_isl_grammar: Whether to apply ISL grammar rules
        """
        self.sentence_pause = sentence_pause
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.use_isl_grammar = use_isl_grammar
        
        # Word buffer
        self.word_buffer = []
        self.word_timestamps = []
        
        # Sentence storage
        self.completed_sentences = []
        self.current_sentence_in_progress = ""
        
        # ISL grammar rules
        self.isl_grammar_rules = self.load_isl_grammar_rules()
        
        # Statistics
        self.total_words_received = 0
        self.total_sentences_formed = 0
        
        # Logging
        self.logger = setup_logging()
        self.logger.info("WordToSentenceConverter initialized")
    
    def load_isl_grammar_rules(self) -> dict:
        """
        Load ISL grammar rules.
        ISL typically follows Subject-Object-Verb (SOV) order.
        
        Returns:
            Dictionary of grammar rules
        """
        rules = {
            'word_order': 'SOV',  # Subject-Object-Verb
            'question_markers': ['WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW', 'WHICH'],
            'common_patterns': {
                'greeting': ['HELLO', 'HI', 'GOOD MORNING', 'GOOD EVENING'],
                'question': ['WHAT YOUR NAME', 'WHERE YOU GO', 'HOW YOU'],
                'statement': ['I WANT', 'YOU NEED', 'HE LIKE']
            }
        }
        return rules
    
    def add_word(self, word: str, timestamp: Optional[float] = None) -> None:
        """
        Add a word to the buffer.
        
        Args:
            word: Word to add
            timestamp: Word timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Check for sentence boundary before adding
        if self.check_sentence_boundary(timestamp):
            self.form_sentence_from_buffer()
        
        # Add word and timestamp
        self.word_buffer.append(word)
        self.word_timestamps.append(timestamp)
        
        self.total_words_received += 1
        
        # Update sentence in progress
        self.current_sentence_in_progress = ' '.join(self.word_buffer)
    
    def check_sentence_boundary(self, current_timestamp: Optional[float] = None) -> bool:
        """
        Check if a sentence boundary should be inserted based on pause.
        
        Args:
            current_timestamp: Current timestamp
        
        Returns:
            True if sentence boundary detected
        """
        if not self.word_buffer or not self.word_timestamps:
            return False
        
        if current_timestamp is None:
            current_timestamp = time.time()
        
        # Check time since last word
        time_since_last_word = current_timestamp - self.word_timestamps[-1]
        
        if time_since_last_word >= self.sentence_pause:
            return True
        
        # Check if max sentence length reached
        if len(self.word_buffer) >= self.max_sentence_length:
            return True
        
        return False
    
    def pos_tag_words(self, words: List[str]) -> List[Tuple[str, str]]:
        """
        POS tag words using NLTK.
        
        Args:
            words: List of words
        
        Returns:
            List of (word, pos_tag) tuples
        """
        if not NLTK_AVAILABLE or not words:
            return [(word, 'UNKNOWN') for word in words]
        
        try:
            # Clean words (remove spell-check markers)
            cleaned_words = []
            for word in words:
                if '[?' in word:
                    # Extract original word
                    word = word.split('[')[0]
                cleaned_words.append(word)
            
            # POS tag
            tagged = pos_tag(cleaned_words)
            return tagged
            
        except Exception as e:
            self.logger.error(f"Error in POS tagging: {e}")
            return [(word, 'UNKNOWN') for word in words]
    
    def apply_isl_grammar(self, words: List[str]) -> List[str]:
        """
        Apply ISL grammar rules to reorder words.
        ISL follows Subject-Object-Verb (SOV) structure.
        
        Args:
            words: List of words
        
        Returns:
            Reordered list of words
        """
        if not self.use_isl_grammar or len(words) < 3:
            return words
        
        try:
            # POS tag words
            tagged = self.pos_tag_words(words)
            
            # Extract components
            subject = []
            object_words = []
            verb = []
            other = []
            
            for word, tag in tagged:
                if tag.startswith('PRP') or tag.startswith('NN'):
                    # Pronoun or Noun - likely subject or object
                    if not subject:
                        subject.append(word)
                    else:
                        object_words.append(word)
                elif tag.startswith('VB'):
                    # Verb
                    verb.append(word)
                else:
                    other.append(word)
            
            # Reconstruct in SOV order
            reordered = subject + object_words + verb + other
            
            self.logger.info(f"Grammar applied: {words} → {reordered}")
            return reordered
            
        except Exception as e:
            self.logger.error(f"Error applying ISL grammar: {e}")
            return words
    
    def capitalize_and_punctuate(self, words: List[str]) -> str:
        """
        Capitalize first word and add punctuation.
        
        Args:
            words: List of words
        
        Returns:
            Formatted sentence string
        """
        if not words:
            return ""
        
        # Clean words (remove spell-check markers for display)
        cleaned_words = []
        for word in words:
            if '[?' in word:
                # Keep only the original word
                word = word.split('[')[0]
            cleaned_words.append(word)
        
        # Capitalize first word
        sentence = ' '.join(cleaned_words)
        sentence = sentence[0].upper() + sentence[1:] if sentence else ""
        
        # Check if it's a question
        first_word = cleaned_words[0].upper()
        if first_word in self.isl_grammar_rules['question_markers']:
            sentence += '?'
        else:
            sentence += '.'
        
        return sentence
    
    def form_sentence_from_buffer(self) -> Optional[str]:
        """
        Form sentence from current word buffer.
        
        Returns:
            Formed sentence or None if buffer is empty/invalid
        """
        if not self.word_buffer:
            return None
        
        # Check sentence length
        if len(self.word_buffer) < self.min_sentence_length:
            self.logger.info(f"Sentence too short: {len(self.word_buffer)} words")
            self.reset_word_buffer()
            return None
        
        # Apply ISL grammar
        if self.use_isl_grammar and NLTK_AVAILABLE:
            reordered_words = self.apply_isl_grammar(self.word_buffer)
        else:
            reordered_words = self.word_buffer
        
        # Capitalize and punctuate
        sentence = self.capitalize_and_punctuate(reordered_words)
        
        # Add to completed sentences
        self.completed_sentences.append({
            'sentence': sentence,
            'original_words': self.word_buffer.copy(),
            'reordered_words': reordered_words,
            'timestamp': time.time(),
            'word_count': len(self.word_buffer)
        })
        
        self.total_sentences_formed += 1
        self.logger.info(f"Sentence formed: '{sentence}'")
        
        # Reset buffer
        self.reset_word_buffer()
        
        return sentence
    
    def get_current_sentence_in_progress(self) -> str:
        """
        Get the current sentence being formed.
        
        Returns:
            Current sentence in progress
        """
        return self.current_sentence_in_progress
    
    def get_completed_sentences(self, limit: Optional[int] = None) -> List[dict]:
        """
        Get completed sentences.
        
        Args:
            limit: Maximum number of sentences to return (most recent)
        
        Returns:
            List of completed sentences with metadata
        """
        if limit is None:
            return self.completed_sentences
        else:
            return self.completed_sentences[-limit:]
    
    def get_last_sentence(self) -> Optional[str]:
        """
        Get the last completed sentence.
        
        Returns:
            Last sentence or None if no sentences formed
        """
        if self.completed_sentences:
            return self.completed_sentences[-1]['sentence']
        return None
    
    def reset_word_buffer(self) -> None:
        """Reset word buffer."""
        self.word_buffer.clear()
        self.word_timestamps.clear()
        self.current_sentence_in_progress = ""
    
    def reset_all(self) -> None:
        """Reset all buffers and storage."""
        self.reset_word_buffer()
        self.completed_sentences.clear()
        self.total_words_received = 0
        self.total_sentences_formed = 0
        self.logger.info("All buffers reset")
    
    def manual_sentence_boundary(self) -> Optional[str]:
        """
        Manually insert sentence boundary and form sentence.
        
        Returns:
            Formed sentence or None
        """
        return self.form_sentence_from_buffer()
    
    def get_statistics(self) -> dict:
        """
        Get conversion statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_words_received': self.total_words_received,
            'total_sentences_formed': self.total_sentences_formed,
            'words_in_buffer': len(self.word_buffer),
            'current_sentence_length': len(self.word_buffer),
            'completed_sentences_count': len(self.completed_sentences)
        }


def main():
    """Main function for testing."""
    print("=" * 50)
    print("Word-to-Sentence Converter Test")
    print("=" * 50)
    
    # Initialize converter
    converter = WordToSentenceConverter(
        sentence_pause=5.0,
        min_sentence_length=2,
        use_isl_grammar=True
    )
    
    print("\nEnter words one by one (or 'pause' to simulate pause)")
    print("Type 'sentence' to manually form sentence")
    print("Type 'stats' to see statistics")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Enter word: ").strip()
        
        if user_input.upper() == 'QUIT':
            break
        
        elif user_input.upper() == 'PAUSE':
            print("Simulating pause...")
            time.sleep(5.5)
            if converter.check_sentence_boundary():
                sentence = converter.form_sentence_from_buffer()
                if sentence:
                    print(f"✓ Sentence formed: {sentence}")
        
        elif user_input.upper() == 'SENTENCE':
            sentence = converter.manual_sentence_boundary()
            if sentence:
                print(f"✓ Sentence formed: {sentence}")
            else:
                print("✗ No sentence formed (buffer empty or invalid)")
        
        elif user_input.upper() == 'STATS':
            stats = converter.get_statistics()
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print(f"\nCompleted sentences: {[s['sentence'] for s in converter.get_completed_sentences()]}")
            print()
        
        elif user_input:
            converter.add_word(user_input)
            print(f"Current sentence: {converter.get_current_sentence_in_progress()}")
        
        else:
            print("Invalid input. Enter a word.")
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("Final Statistics")
    print("=" * 50)
    stats = converter.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if converter.completed_sentences:
        print("\nCompleted Sentences:")
        for i, sent_info in enumerate(converter.get_completed_sentences(), 1):
            print(f"  {i}. {sent_info['sentence']}")
            print(f"      (from {sent_info['word_count']} words: {' '.join(sent_info['original_words'])})")


if __name__ == "__main__":
    main()
