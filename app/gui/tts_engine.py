"""
Text-to-Speech Module for ISL System
Supports English and Hindi speech output
"""

import pyttsx3
import sys
import os
from typing import Optional, List
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import setup_logging


class Language(Enum):
    """Supported languages for TTS."""
    ENGLISH = "english"
    HINDI = "hindi"


class TextToSpeech:
    """
    Text-to-Speech engine for converting text to speech.
    Supports multiple languages and voice customization.
    """
    
    def __init__(self, language: Language = Language.ENGLISH,
                 rate: int = 150, volume: float = 1.0):
        """
        Initialize TTS engine.
        
        Args:
            language: Language for speech output
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.logger = setup_logging()
        self.language = language
        
        try:
            # Initialize pyttsx3 engine
            self.engine = pyttsx3.init()
            
            # Set properties
            self.set_rate(rate)
            self.set_volume(volume)
            
            # Get available voices
            self.voices = self.engine.getProperty('voices')
            
            # Set voice based on language
            self._set_language_voice()
            
            self.logger.info(f"TTS engine initialized with {language.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def _set_language_voice(self):
        """Set voice based on selected language."""
        if not self.engine or not self.voices:
            return
        
        try:
            if self.language == Language.HINDI:
                # Try to find Hindi voice
                hindi_voice = None
                for voice in self.voices:
                    # Check for Hindi indicators in voice name/languages
                    voice_info = voice.name.lower()
                    if 'hindi' in voice_info or 'hi-in' in voice_info:
                        hindi_voice = voice.id
                        break
                
                if hindi_voice:
                    self.engine.setProperty('voice', hindi_voice)
                    self.logger.info("Hindi voice selected")
                else:
                    self.logger.warning("Hindi voice not found, using default")
            else:
                # Use first English voice (usually default)
                if self.voices:
                    self.engine.setProperty('voice', self.voices[0].id)
                    self.logger.info("English voice selected")
        
        except Exception as e:
            self.logger.error(f"Error setting voice: {e}")
    
    def set_rate(self, rate: int):
        """
        Set speech rate.
        
        Args:
            rate: Words per minute (50-300, typical 150)
        """
        if self.engine:
            # Clamp rate to reasonable range
            rate = max(50, min(300, rate))
            self.engine.setProperty('rate', rate)
            self.logger.debug(f"Speech rate set to {rate}")
    
    def set_volume(self, volume: float):
        """
        Set volume level.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.engine:
            # Clamp volume to valid range
            volume = max(0.0, min(1.0, volume))
            self.engine.setProperty('volume', volume)
            self.logger.debug(f"Volume set to {volume}")
    
    def set_language(self, language: Language):
        """
        Change language.
        
        Args:
            language: New language to use
        """
        self.language = language
        self._set_language_voice()
    
    def speak(self, text: str, block: bool = True):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
            block: Whether to block until speech completes
        """
        if not self.engine:
            self.logger.error("TTS engine not available")
            print(f"[TTS] Would speak: {text}")
            return
        
        if not text or not text.strip():
            self.logger.warning("Empty text provided for speech")
            return
        
        try:
            self.logger.info(f"Speaking: {text}")
            
            if block:
                # Blocking speech - wait until complete
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Non-blocking - speak in background
                self.engine.say(text)
                self.engine.startLoop(False)
                self.engine.iterate()
                self.engine.endLoop()
        
        except Exception as e:
            self.logger.error(f"Error during speech: {e}")
    
    def speak_async(self, text: str):
        """
        Speak text asynchronously (non-blocking).
        
        Args:
            text: Text to speak
        """
        import threading
        
        def _speak():
            self.speak(text, block=True)
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop current speech."""
        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                self.logger.error(f"Error stopping speech: {e}")
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voice names.
        
        Returns:
            List of voice names
        """
        if not self.voices:
            return []
        
        return [voice.name for voice in self.voices]
    
    def list_voices(self):
        """Print all available voices."""
        if not self.voices:
            print("No voices available")
            return
        
        print("\nAvailable Voices:")
        print("-" * 60)
        for idx, voice in enumerate(self.voices):
            print(f"{idx}: {voice.name}")
            print(f"   ID: {voice.id}")
            if hasattr(voice, 'languages'):
                print(f"   Languages: {voice.languages}")
            print()


class SentenceSpeaker:
    """
    Higher-level interface for speaking ISL-formed sentences.
    Handles sentence formatting and language detection.
    """
    
    def __init__(self):
        """Initialize sentence speaker with both language engines."""
        self.logger = setup_logging()
        
        # Initialize TTS engines
        self.english_tts = TextToSpeech(Language.ENGLISH, rate=150)
        self.hindi_tts = TextToSpeech(Language.HINDI, rate=140)
        
        self.current_language = Language.ENGLISH
        self.logger.info("SentenceSpeaker initialized")
    
    def speak_sentence(self, sentence: str, language: Optional[Language] = None):
        """
        Speak a sentence in specified or detected language.
        
        Args:
            sentence: Sentence to speak
            language: Language to use (auto-detect if None)
        """
        if not sentence or not sentence.strip():
            self.logger.warning("Empty sentence provided")
            return
        
        # Clean sentence
        sentence = sentence.strip()
        
        # Use specified language or current
        lang = language if language else self.current_language
        
        # Select appropriate TTS engine
        if lang == Language.HINDI:
            tts = self.hindi_tts
        else:
            tts = self.english_tts
        
        # Speak
        tts.speak(sentence)
    
    def speak_character(self, character: str, language: Optional[Language] = None):
        """
        Speak a single character.
        
        Args:
            character: Character to speak
            language: Language to use
        """
        if not character:
            return
        
        # Format character for speech
        text = f"{character}"
        
        # Use specified language or current
        lang = language if language else self.current_language
        
        # Select appropriate TTS engine
        if lang == Language.HINDI:
            tts = self.hindi_tts
        else:
            tts = self.english_tts
        
        tts.speak(text)
    
    def speak_word(self, word: str, language: Optional[Language] = None):
        """
        Speak a word.
        
        Args:
            word: Word to speak
            language: Language to use
        """
        self.speak_sentence(word, language)
    
    def set_language(self, language: Language):
        """
        Set default language.
        
        Args:
            language: Language to use
        """
        self.current_language = language
        self.logger.info(f"Language set to {language.value}")
    
    def toggle_language(self):
        """Toggle between English and Hindi."""
        if self.current_language == Language.ENGLISH:
            self.current_language = Language.HINDI
        else:
            self.current_language = Language.ENGLISH
        
        self.logger.info(f"Language toggled to {self.current_language.value}")
    
    def set_rate(self, rate: int):
        """Set speech rate for both engines."""
        self.english_tts.set_rate(rate)
        self.hindi_tts.set_rate(rate)
    
    def set_volume(self, volume: float):
        """Set volume for both engines."""
        self.english_tts.set_volume(volume)
        self.hindi_tts.set_volume(volume)


def demo_tts():
    """Demo function to test TTS functionality."""
    print("=== ISL Text-to-Speech Demo ===\n")
    
    # Create TTS engine
    tts = TextToSpeech(Language.ENGLISH, rate=150)
    
    # List available voices
    print("Available voices:")
    tts.list_voices()
    
    # Test English speech
    print("\n1. Testing English speech...")
    tts.speak("Hello! This is the Indian Sign Language detection system.")
    
    # Test with ISL-formed sentence
    print("\n2. Testing ISL-formed sentence...")
    isl_sentence = "I AM LEARNING SIGN LANGUAGE"
    tts.speak(f"ISL Sentence: {isl_sentence}")
    
    # Test SentenceSpeaker
    print("\n3. Testing SentenceSpeaker...")
    speaker = SentenceSpeaker()
    
    speaker.speak_sentence("Welcome to the ISL system")
    speaker.speak_character("A")
    speaker.speak_word("HELLO")
    
    # Try Hindi if available
    print("\n4. Testing Hindi (if available)...")
    speaker.set_language(Language.HINDI)
    speaker.speak_sentence("नमस्ते")  # Namaste
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_tts()
