"""
Enhanced ISL Real-Time Detection Demo
Combines skeleton visualization, GUI, TTS, and word suggestions
Based on ASL project improvements
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import sys
import os
from typing import Optional
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.gui.skeleton_visualizer import HandSkeletonVisualizer
from app.gui.tts_engine import SentenceSpeaker, Language
from app.gui.word_suggester import ISLWordSuggester
from app.utils import setup_logging


class EnhancedISLDetector:
    """
    Enhanced real-time ISL detector with all features integrated.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the enhanced detector."""
        self.logger = setup_logging()
        
        # Video capture
        self.video_capture = cv2.VideoCapture(0)
        
        # Initialize components
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.visualizer = HandSkeletonVisualizer(canvas_size=400)
        self.speaker = SentenceSpeaker()
        self.suggester = ISLWordSuggester()
        
        # Load model
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.logger.info(f"Model loaded: {model_path}")
            except Exception as e:
                self.logger.error(f"Model loading failed: {e}")
        
        # State variables
        self.sentence = ""
        self.current_word = ""
        self.current_char = ""
        self.prev_char = ""
        self.char_count = {}
        self.stability_frames = 15
        
        # GUI
        self.root = tk.Tk()
        self.root.title("🤟 Enhanced ISL Detection System 🤟")
        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        self.root.geometry("1500x900")
        self.root.configure(bg='#1a1a2e')
        
        self._setup_enhanced_gui()
        
        self.is_running = True
        self.video_loop()
        
        self.logger.info("EnhancedISLDetector initialized")
    
    def _setup_enhanced_gui(self):
        """Setup enhanced GUI with all components."""
        
        # Main title with gradient effect
        title_frame = tk.Frame(self.root, bg='#16213e', height=80)
        title_frame.pack(fill=tk.X)
        
        title = tk.Label(
            title_frame,
            text="🤟 ISL REAL-TIME DETECTION SYSTEM 🤟",
            font=("Helvetica", 32, "bold"),
            bg='#16213e',
            fg='#00d9ff'
        )
        title.pack(pady=15)
        
        subtitle = tk.Label(
            title_frame,
            text="Enhanced with Skeleton Visualization • TTS • Smart Suggestions",
            font=("Helvetica", 12),
            bg='#16213e',
            fg='#8e9aaf'
        )
        subtitle.pack()
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#1a1a2e')
        content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # === Left Panel: Video Feeds ===
        video_panel = tk.Frame(content_frame, bg='#1a1a2e')
        video_panel.grid(row=0, column=0, padx=10, sticky='nsew')
        
        # Camera feed
        tk.Label(
            video_panel,
            text="📹 CAMERA FEED",
            font=("Helvetica", 14, "bold"),
            bg='#1a1a2e',
            fg='#00d9ff'
        ).pack()
        
        self.panel_camera = tk.Label(video_panel, bg='#0f3460', relief=tk.SUNKEN, bd=3)
        self.panel_camera.pack(pady=5)
        
        # Skeleton view
        tk.Label(
            video_panel,
            text="🦴 SKELETON VIEW",
            font=("Helvetica", 14, "bold"),
            bg='#1a1a2e',
            fg='#00d9ff'
        ).pack(pady=(15, 0))
        
        self.panel_skeleton = tk.Label(video_panel, bg='#0f3460', relief=tk.SUNKEN, bd=3)
        self.panel_skeleton.pack(pady=5)
        
        # === Right Panel: Detection & Text ===
        right_panel = tk.Frame(content_frame, bg='#1a1a2e')
        right_panel.grid(row=0, column=1, padx=10, sticky='nsew')
        
        # Current character detection
        char_container = tk.Frame(right_panel, bg='#16213e', relief=tk.RAISED, bd=5)
        char_container.pack(pady=10, fill=tk.X)
        
        tk.Label(
            char_container,
            text="DETECTED CHARACTER",
            font=("Helvetica", 16, "bold"),
            bg='#16213e',
            fg='#00d9ff'
        ).pack(pady=5)
        
        self.char_display = tk.Label(
            char_container,
            text="--",
            font=("Helvetica", 72, "bold"),
            bg='#0f3460',
            fg='#00ff00',
            width=3,
            relief=tk.SUNKEN,
            bd=5
        )
        self.char_display.pack(pady=10)
        
        # Sentence formation
        sentence_container = tk.Frame(right_panel, bg='#16213e', relief=tk.RAISED, bd=5)
        sentence_container.pack(pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(
            sentence_container,
            text="📝 FORMED SENTENCE",
            font=("Helvetica", 16, "bold"),
            bg='#16213e',
            fg='#00d9ff'
        ).pack(pady=5)
        
        sentence_frame = tk.Frame(sentence_container, bg='#16213e')
        sentence_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.sentence_text = tk.Text(
            sentence_frame,
            font=("Courier New", 14, "bold"),
            bg='#0f3460',
            fg='#ffffff',
            height=4,
            wrap=tk.WORD,
            relief=tk.SUNKEN,
            bd=3
        )
        self.sentence_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(sentence_frame, command=self.sentence_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sentence_text.config(yscrollcommand=scrollbar.set)
        
        # Control buttons
        btn_container = tk.Frame(right_panel, bg='#1a1a2e')
        btn_container.pack(pady=10, fill=tk.X)
        
        buttons = [
            ("➕ Add Char", self.add_current_char, '#27ae60'),
            ("SPACE", self.add_space, '#3498db'),
            ("⌫ Back", self.backspace, '#e74c3c'),
            ("🗑️ Clear", self.clear_all, '#e67e22'),
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            btn = tk.Button(
                btn_container,
                text=text,
                font=("Helvetica", 11, "bold"),
                bg=color,
                fg='white',
                width=10,
                height=2,
                command=command,
                relief=tk.RAISED,
                bd=3,
                cursor='hand2'
            )
            btn.grid(row=0, column=i, padx=5)
        
        # TTS Controls
        tts_container = tk.Frame(right_panel, bg='#16213e', relief=tk.RAISED, bd=5)
        tts_container.pack(pady=10, fill=tk.X)
        
        tk.Label(
            tts_container,
            text="🔊 TEXT-TO-SPEECH",
            font=("Helvetica", 14, "bold"),
            bg='#16213e',
            fg='#00d9ff'
        ).pack(pady=5)
        
        tts_btn_frame = tk.Frame(tts_container, bg='#16213e')
        tts_btn_frame.pack(pady=5)
        
        tk.Button(
            tts_btn_frame,
            text="🔊 SPEAK (EN)",
            font=("Helvetica", 12, "bold"),
            bg='#9b59b6',
            fg='white',
            width=15,
            height=2,
            command=lambda: self.speak(Language.ENGLISH),
            relief=tk.RAISED,
            bd=3
        ).grid(row=0, column=0, padx=5)
        
        tk.Button(
            tts_btn_frame,
            text="🔊 SPEAK (HI)",
            font=("Helvetica", 12, "bold"),
            bg='#8e44ad',
            fg='white',
            width=15,
            height=2,
            command=lambda: self.speak(Language.HINDI),
            relief=tk.RAISED,
            bd=3
        ).grid(row=0, column=1, padx=5)
        
        # Word suggestions
        suggest_container = tk.Frame(right_panel, bg='#16213e', relief=tk.RAISED, bd=5)
        suggest_container.pack(pady=10, fill=tk.X)
        
        tk.Label(
            suggest_container,
            text="💡 WORD SUGGESTIONS",
            font=("Helvetica", 14, "bold"),
            bg='#16213e',
            fg='#00d9ff'
        ).pack(pady=5)
        
        self.suggestion_btns = []
        suggest_grid = tk.Frame(suggest_container, bg='#16213e')
        suggest_grid.pack(pady=5)
        
        for i in range(4):
            btn = tk.Button(
                suggest_grid,
                text="---",
                font=("Helvetica", 11),
                bg='#0f3460',
                fg='#ffffff',
                width=18,
                height=1,
                command=lambda idx=i: self.apply_suggestion(idx),
                relief=tk.RAISED,
                bd=2,
                cursor='hand2'
            )
            btn.grid(row=i//2, column=i%2, padx=5, pady=3)
            self.suggestion_btns.append(btn)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="⚡ System Ready",
            font=("Helvetica", 11),
            bg='#0f3460',
            fg='#00ff00',
            anchor=tk.W,
            relief=tk.SUNKEN,
            bd=2
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure grid weights
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)
    
    def video_loop(self):
        """Main video processing loop."""
        if not self.is_running:
            return
        
        ret, frame = self.video_capture.read()
        
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Process frame
            skeleton_img, landmarks = self.visualizer.process_frame(frame, self.hands)
            
            if skeleton_img is not None and landmarks is not None:
                # Predict character
                if self.model:
                    pred_char = self.predict_character(landmarks)
                    self.update_character(pred_char)
                    
                    # Annotate frame
                    cv2.putText(frame, f"Detected: {pred_char}",
                              (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              1.5, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Model not loaded",
                              (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 0, 255), 2)
                
                # Update displays
                self.update_panel(self.panel_skeleton, skeleton_img)
                self.status_bar.config(text=f"✓ Hand Detected | Current: {self.current_char}",
                                      fg='#00ff00')
            else:
                cv2.putText(frame, "Show your hand",
                          (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (255, 255, 0), 2)
                self.status_bar.config(text="⚠ No hand detected", fg='#ffaa00')
            
            self.update_panel(self.panel_camera, frame)
        
        self.root.after(10, self.video_loop)
    
    def update_panel(self, panel, image):
        """Update image panel."""
        display_size = (480, 360)
        image = cv2.resize(image, display_size)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=pil_img)
        
        panel.configure(image=photo)
        panel.image = photo
    
    def predict_character(self, landmarks: np.ndarray) -> str:
        """Predict character from landmarks."""
        try:
            landmarks_input = landmarks.reshape(1, -1)
            prediction = self.model.predict(landmarks_input, verbose=0)
            pred_class = np.argmax(prediction[0])
            return chr(65 + pred_class)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return "?"
    
    def update_character(self, char: str):
        """Update current character with stability."""
        if char == self.prev_char:
            self.char_count[char] = self.char_count.get(char, 0) + 1
        else:
            self.char_count = {char: 1}
            self.prev_char = char
        
        if self.char_count.get(char, 0) >= self.stability_frames:
            self.current_char = char
            self.char_display.config(text=char)
    
    def add_current_char(self):
        """Add current character to sentence."""
        if self.current_char and self.current_char != "?":
            self.current_word += self.current_char
            self.sentence += self.current_char
            self.update_sentence_display()
            self.update_suggestions()
    
    def add_space(self):
        """Add space."""
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
            self.current_word = ""
            self.update_sentence_display()
    
    def backspace(self):
        """Remove last character."""
        if self.sentence:
            self.sentence = self.sentence[:-1]
            words = self.sentence.split()
            self.current_word = words[-1] if words else ""
            self.update_sentence_display()
            self.update_suggestions()
    
    def clear_all(self):
        """Clear everything."""
        self.sentence = ""
        self.current_word = ""
        self.update_sentence_display()
        for btn in self.suggestion_btns:
            btn.config(text="---")
    
    def update_sentence_display(self):
        """Update sentence display."""
        self.sentence_text.delete('1.0', tk.END)
        self.sentence_text.insert('1.0', self.sentence)
    
    def update_suggestions(self):
        """Update word suggestions."""
        if self.current_word:
            suggestions = self.suggester.get_suggestions(self.current_word)
            for i, btn in enumerate(self.suggestion_btns):
                text = suggestions[i] if i < len(suggestions) else "---"
                btn.config(text=text)
        else:
            for btn in self.suggestion_btns:
                btn.config(text="---")
    
    def apply_suggestion(self, idx: int):
        """Apply selected suggestion."""
        text = self.suggestion_btns[idx].cget('text')
        if text != "---" and self.current_word:
            # Replace current word
            if " " in self.sentence:
                parts = self.sentence.rsplit(" ", 1)
                self.sentence = parts[0] + " " + text
            else:
                self.sentence = text
            self.current_word = text
            self.update_sentence_display()
    
    def speak(self, language: Language):
        """Speak sentence."""
        if self.sentence.strip():
            self.status_bar.config(text=f"🔊 Speaking: {self.sentence[:50]}...",
                                  fg='#ff00ff')
            threading.Thread(
                target=self.speaker.speak_sentence,
                args=(self.sentence, language),
                daemon=True
            ).start()
        else:
            self.status_bar.config(text="⚠ Nothing to speak!", fg='#ffaa00')
    
    def on_closing(self):
        """Handle window close."""
        self.is_running = False
        self.video_capture.release()
        self.hands.close()
        self.root.destroy()
    
    def run(self):
        """Start the application."""
        self.logger.info("Starting Enhanced ISL Detector...")
        self.root.mainloop()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ISL Detection System")
    parser.add_argument('--model', type=str, help='Path to trained model',
                       default=None)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🤟 ENHANCED ISL DETECTION SYSTEM 🤟")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✓ Skeleton Visualization (eliminates background/lighting issues)")
    print("  ✓ Real-Time Detection with GUI")
    print("  ✓ Text-to-Speech (English & Hindi)")
    print("  ✓ Smart Word Suggestions")
    print("  ✓ Sentence Formation")
    print("\nInspired by ASL project improvements")
    print("=" * 70)
    print()
    
    app = EnhancedISLDetector(model_path=args.model)
    app.run()


if __name__ == "__main__":
    main()
