"""
Real-Time ISL Detection GUI with Text Formation and Speech
Inspired by ASL project approach with Tkinter GUI
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import sys
import os
from typing import Optional, List
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.gui.skeleton_visualizer import HandSkeletonVisualizer
from app.utils import setup_logging


class ISLRealtimeGUI:
    """
    Real-time ISL detection GUI with sentence formation and text-to-speech.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the real-time ISL detection GUI.
        
        Args:
            model_path: Path to trained model file
        """
        self.logger = setup_logging()
        
        # Initialize video capture
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = None
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize skeleton visualizer
        self.visualizer = HandSkeletonVisualizer(canvas_size=400)
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
        
        # Text formation variables
        self.sentence = ""
        self.current_character = ""
        self.word = ""
        self.prev_character = ""
        self.character_count = {}
        self.stability_threshold = 10  # frames to confirm character
        
        # Word suggestions
        self.word_suggestions = ["", "", "", ""]
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("ISL Real-Time Detection System")
        self.root.protocol('WM_DELETE_WINDOW', self.on_closing)
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')
        
        self._setup_gui()
        
        # Start video loop
        self.is_running = True
        self.video_loop()
        
        self.logger.info("ISLRealtimeGUI initialized")
    
    def _setup_gui(self):
        """Setup the GUI components."""
        
        # Title
        title_label = tk.Label(
            self.root,
            text="🤟 Indian Sign Language Detection System 🤟",
            font=("Arial", 28, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=10)
        
        # Main frame for video displays
        video_frame = tk.Frame(self.root, bg='#2c3e50')
        video_frame.pack(pady=10)
        
        # Original video panel
        video_label = tk.Label(video_frame, text="Camera Feed", 
                              font=("Arial", 14, "bold"),
                              bg='#2c3e50', fg='#ecf0f1')
        video_label.grid(row=0, column=0, padx=10)
        
        self.panel_original = tk.Label(video_frame, bg='#34495e')
        self.panel_original.grid(row=1, column=0, padx=10)
        
        # Skeleton video panel
        skeleton_label = tk.Label(video_frame, text="Skeleton View", 
                                 font=("Arial", 14, "bold"),
                                 bg='#2c3e50', fg='#ecf0f1')
        skeleton_label.grid(row=0, column=1, padx=10)
        
        self.panel_skeleton = tk.Label(video_frame, bg='#34495e')
        self.panel_skeleton.grid(row=1, column=1, padx=10)
        
        # Info frame
        info_frame = tk.Frame(self.root, bg='#2c3e50')
        info_frame.pack(pady=20)
        
        # Current character display
        char_frame = tk.Frame(info_frame, bg='#2c3e50')
        char_frame.grid(row=0, column=0, padx=30)
        
        tk.Label(char_frame, text="Current Character:",
                font=("Arial", 18, "bold"),
                bg='#2c3e50', fg='#ecf0f1').pack()
        
        self.char_display = tk.Label(
            char_frame,
            text="--",
            font=("Arial", 48, "bold"),
            bg='#3498db',
            fg='white',
            width=5,
            height=1,
            relief=tk.RAISED,
            bd=5
        )
        self.char_display.pack(pady=10)
        
        # Sentence display
        sentence_frame = tk.Frame(info_frame, bg='#2c3e50')
        sentence_frame.grid(row=0, column=1, padx=30)
        
        tk.Label(sentence_frame, text="Formed Sentence:",
                font=("Arial", 18, "bold"),
                bg='#2c3e50', fg='#ecf0f1').pack()
        
        self.sentence_display = tk.Text(
            sentence_frame,
            font=("Arial", 16),
            width=40,
            height=3,
            bg='#ecf0f1',
            fg='#2c3e50',
            relief=tk.SUNKEN,
            bd=3,
            wrap=tk.WORD
        )
        self.sentence_display.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        # Add space button
        self.space_btn = tk.Button(
            button_frame,
            text="Add Space",
            font=("Arial", 14, "bold"),
            bg='#27ae60',
            fg='white',
            width=12,
            height=2,
            command=self.add_space,
            relief=tk.RAISED,
            bd=3
        )
        self.space_btn.grid(row=0, column=0, padx=10)
        
        # Backspace button
        self.backspace_btn = tk.Button(
            button_frame,
            text="Backspace",
            font=("Arial", 14, "bold"),
            bg='#e74c3c',
            fg='white',
            width=12,
            height=2,
            command=self.backspace,
            relief=tk.RAISED,
            bd=3
        )
        self.backspace_btn.grid(row=0, column=1, padx=10)
        
        # Clear button
        self.clear_btn = tk.Button(
            button_frame,
            text="Clear All",
            font=("Arial", 14, "bold"),
            bg='#e67e22',
            fg='white',
            width=12,
            height=2,
            command=self.clear_sentence,
            relief=tk.RAISED,
            bd=3
        )
        self.clear_btn.grid(row=0, column=2, padx=10)
        
        # Speak button (will be implemented with TTS)
        self.speak_btn = tk.Button(
            button_frame,
            text="🔊 Speak",
            font=("Arial", 14, "bold"),
            bg='#9b59b6',
            fg='white',
            width=12,
            height=2,
            command=self.speak_sentence,
            relief=tk.RAISED,
            bd=3
        )
        self.speak_btn.grid(row=0, column=3, padx=10)
        
        # Word suggestions frame
        suggestions_frame = tk.Frame(self.root, bg='#2c3e50')
        suggestions_frame.pack(pady=10)
        
        tk.Label(suggestions_frame, text="Word Suggestions:",
                font=("Arial", 16, "bold"),
                bg='#2c3e50', fg='#ecf0f1').pack()
        
        self.suggestion_buttons = []
        btn_frame = tk.Frame(suggestions_frame, bg='#2c3e50')
        btn_frame.pack()
        
        for i in range(4):
            btn = tk.Button(
                btn_frame,
                text="--",
                font=("Arial", 12),
                bg='#34495e',
                fg='#ecf0f1',
                width=15,
                height=1,
                command=lambda idx=i: self.apply_suggestion(idx),
                relief=tk.RAISED,
                bd=2
            )
            btn.grid(row=0, column=i, padx=5)
            self.suggestion_buttons.append(btn)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Status: Initializing...",
            font=("Arial", 12),
            bg='#34495e',
            fg='#ecf0f1',
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def video_loop(self):
        """Main video loop for capturing and processing frames."""
        if not self.is_running:
            return
        
        ret, frame = self.video_capture.read()
        
        if ret:
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Process frame
            skeleton_img, landmarks = self.visualizer.process_frame(
                frame, self.hands
            )
            
            if skeleton_img is not None and landmarks is not None:
                # Predict if model is available
                if self.model is not None:
                    prediction = self.predict_character(landmarks)
                    self.update_character(prediction)
                    
                    # Draw prediction on frame
                    cv2.putText(frame, f"Detected: {prediction}",
                              (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                              1.2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Model not loaded",
                              (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 0, 255), 2)
                
                # Update skeleton panel
                self.update_image_panel(self.panel_skeleton, skeleton_img)
                
                self.status_label.config(text="Status: Hand detected ✓")
            else:
                cv2.putText(frame, "No hand detected",
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 0, 255), 2)
                self.status_label.config(text="Status: No hand detected")
            
            # Update original video panel
            self.update_image_panel(self.panel_original, frame)
        
        # Schedule next frame
        self.root.after(10, self.video_loop)
    
    def update_image_panel(self, panel, image):
        """Update image in a panel."""
        # Resize for display
        display_size = (480, 360)
        image = cv2.resize(image, display_size)
        
        # Convert to RGB for PIL
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update panel
        panel.configure(image=photo)
        panel.image = photo  # Keep a reference
    
    def predict_character(self, landmarks: np.ndarray) -> str:
        """
        Predict character from landmarks.
        
        Args:
            landmarks: Flattened landmark array (63,)
            
        Returns:
            Predicted character
        """
        try:
            # Reshape for model input
            landmarks_input = landmarks.reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(landmarks_input, verbose=0)
            predicted_class = np.argmax(prediction[0])
            
            # Convert to letter (A-Z)
            predicted_char = chr(65 + predicted_class)
            
            return predicted_char
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return "?"
    
    def update_character(self, character: str):
        """Update current character with stability check."""
        # Count consecutive detections
        if character == self.prev_character:
            self.character_count[character] = \
                self.character_count.get(character, 0) + 1
        else:
            self.character_count = {character: 1}
            self.prev_character = character
        
        # Update display if stable
        if self.character_count.get(character, 0) >= self.stability_threshold:
            self.current_character = character
            self.char_display.config(text=character)
            
            # Auto-add to sentence after stability (optional)
            # Uncomment to enable auto-addition:
            # if character != self.prev_character:
            #     self.add_character_to_sentence(character)
    
    def add_character_to_sentence(self, character: str):
        """Add character to current word/sentence."""
        if character and character != "?":
            self.word += character
            self.sentence += character
            self.update_sentence_display()
            self.update_suggestions()
    
    def add_space(self):
        """Add space to sentence."""
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
            self.word = ""
            self.update_sentence_display()
    
    def backspace(self):
        """Remove last character from sentence."""
        if self.sentence:
            self.sentence = self.sentence[:-1]
            # Update word (last word after space)
            words = self.sentence.split()
            self.word = words[-1] if words else ""
            self.update_sentence_display()
            self.update_suggestions()
    
    def clear_sentence(self):
        """Clear entire sentence."""
        self.sentence = ""
        self.word = ""
        self.word_suggestions = ["", "", "", ""]
        self.update_sentence_display()
        self.update_suggestions_display()
    
    def update_sentence_display(self):
        """Update sentence display in GUI."""
        self.sentence_display.delete('1.0', tk.END)
        self.sentence_display.insert('1.0', self.sentence)
    
    def update_suggestions(self):
        """Update word suggestions (placeholder for now)."""
        # TODO: Implement dictionary-based suggestions
        # For now, just show the current word variations
        if self.word:
            self.word_suggestions = [
                self.word.upper(),
                self.word.lower(),
                self.word.capitalize(),
                ""
            ]
        else:
            self.word_suggestions = ["", "", "", ""]
        
        self.update_suggestions_display()
    
    def update_suggestions_display(self):
        """Update suggestion buttons."""
        for i, btn in enumerate(self.suggestion_buttons):
            text = self.word_suggestions[i] if i < len(self.word_suggestions) else ""
            btn.config(text=text if text else "--")
    
    def apply_suggestion(self, index: int):
        """Apply a word suggestion."""
        suggestion = self.word_suggestions[index]
        if suggestion and self.word:
            # Replace last word with suggestion
            if " " in self.sentence:
                parts = self.sentence.rsplit(" ", 1)
                self.sentence = parts[0] + " " + suggestion
            else:
                self.sentence = suggestion
            
            self.word = suggestion
            self.update_sentence_display()
    
    def speak_sentence(self):
        """Speak the formed sentence using TTS."""
        if self.sentence.strip():
            # Placeholder - TTS will be implemented separately
            print(f"Speaking: {self.sentence}")
            self.status_label.config(text=f"Speaking: {self.sentence}")
            # TODO: Implement actual TTS
        else:
            self.status_label.config(text="Nothing to speak!")
    
    def on_closing(self):
        """Handle window closing."""
        self.is_running = False
        self.video_capture.release()
        self.hands.close()
        self.root.destroy()
        self.logger.info("GUI closed")
    
    def run(self):
        """Start the GUI main loop."""
        self.logger.info("Starting GUI...")
        self.status_label.config(text="Status: Ready")
        self.root.mainloop()


def main():
    """Main function to run the GUI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="ISL Real-Time Detection GUI"
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model file',
        default=None
    )
    
    args = parser.parse_args()
    
    # Initialize and run GUI
    app = ISLRealtimeGUI(model_path=args.model)
    app.run()


if __name__ == "__main__":
    main()
