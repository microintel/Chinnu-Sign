"""
Advanced Real-Time ISL Hand Gesture Recognition System
Professional GUI with webcam feed, predictions, and statistics
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import queue
from datetime import datetime
import json
import os


class AdvancedISLDetector:
    def __init__(self, model_path='models/trained/hybrid_model_20260117_203608.h5'):
        """Initialize the advanced ISL detector"""
        self.model_path = model_path
        self.model = None
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = None
            self.mp_draw = mp.solutions.drawing_utils
        except (ImportError, AttributeError) as e:
            print(f"Warning: MediaPipe unavailable ({e})")
            self.mp_hands = None
            self.hands = None
            self.mp_draw = None
        
        # Letter labels
        self.letters = [chr(65+i) for i in range(26)]  # A-Z
        
        # Detection state
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.prediction_history = []
        self.confidence_threshold = 0.7
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'confident_predictions': 0,
            'session_start': None,
            'predicted_letters': {}
        }
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"✓ Model loaded from {self.model_path}")
            else:
                print(f"⚠️ Model not found at {self.model_path}")
                print("Please train the model first using model_training.ipynb")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks as feature vector"""
        if hand_landmarks is None:
            return None
        
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def predict_gesture(self, landmarks):
        """Predict gesture from landmarks"""
        if self.model is None or landmarks is None:
            return None, 0.0
        
        try:
            # Reshape for model input
            landmarks = landmarks.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(landmarks, verbose=0)
            confidence = np.max(prediction)
            letter_idx = np.argmax(prediction)
            letter = self.letters[letter_idx]
            
            return letter, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def update_stats(self, letter, confidence):
        """Update detection statistics"""
        self.stats['total_predictions'] += 1
        
        if confidence >= self.confidence_threshold:
            self.stats['confident_predictions'] += 1
            if letter in self.stats['predicted_letters']:
                self.stats['predicted_letters'][letter] += 1
            else:
                self.stats['predicted_letters'][letter] = 1


class ISLDetectorGUI:
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("🤟 ISL Hand Gesture Recognition - Professional Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize detector
        self.detector = AdvancedISLDetector()
        
        # Video capture
        self.cap = None
        self.is_camera_on = False
        
        # UI Elements
        self.setup_ui()
        
        # Start stats update loop
        self.update_stats_display()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2d2d30', height=80)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🤟 Indian Sign Language Recognition System",
            font=('Arial', 24, 'bold'),
            bg='#2d2d30',
            fg='#ffffff'
        )
        title_label.pack(pady=20)
        
        # Main content area
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Video feed
        left_panel = tk.Frame(main_frame, bg='#2d2d30', width=800)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video canvas
        self.video_label = tk.Label(left_panel, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons
        control_frame = tk.Frame(left_panel, bg='#2d2d30')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = tk.Button(
            control_frame,
            text="▶ Start Camera",
            command=self.toggle_camera,
            font=('Arial', 14, 'bold'),
            bg='#0e639c',
            fg='white',
            activebackground='#1177bb',
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.snapshot_button = tk.Button(
            control_frame,
            text="📷 Snapshot",
            command=self.take_snapshot,
            font=('Arial', 14),
            bg='#2d2d30',
            fg='white',
            activebackground='#3e3e42',
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.snapshot_button.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Predictions and Stats
        right_panel = tk.Frame(main_frame, bg='#2d2d30', width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Current Prediction
        pred_label = tk.Label(
            right_panel,
            text="Current Detection",
            font=('Arial', 16, 'bold'),
            bg='#2d2d30',
            fg='#ffffff'
        )
        pred_label.pack(pady=(20, 10))
        
        self.prediction_frame = tk.Frame(right_panel, bg='#1e1e1e', height=200)
        self.prediction_frame.pack(fill=tk.X, padx=20, pady=10)
        self.prediction_frame.pack_propagate(False)
        
        self.prediction_letter = tk.Label(
            self.prediction_frame,
            text="--",
            font=('Arial', 80, 'bold'),
            bg='#1e1e1e',
            fg='#4ec9b0'
        )
        self.prediction_letter.pack(expand=True)
        
        self.confidence_label = tk.Label(
            self.prediction_frame,
            text="Confidence: --",
            font=('Arial', 14),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        self.confidence_label.pack()
        
        # Confidence threshold slider
        threshold_frame = tk.Frame(right_panel, bg='#2d2d30')
        threshold_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            threshold_frame,
            text="Confidence Threshold:",
            font=('Arial', 12),
            bg='#2d2d30',
            fg='#ffffff'
        ).pack(anchor=tk.W)
        
        self.threshold_var = tk.DoubleVar(value=0.7)
        self.threshold_slider = ttk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.update_threshold
        )
        self.threshold_slider.pack(fill=tk.X, pady=5)
        
        self.threshold_value_label = tk.Label(
            threshold_frame,
            text="0.70",
            font=('Arial', 12, 'bold'),
            bg='#2d2d30',
            fg='#4ec9b0'
        )
        self.threshold_value_label.pack()
        
        # Statistics
        stats_label = tk.Label(
            right_panel,
            text="Session Statistics",
            font=('Arial', 16, 'bold'),
            bg='#2d2d30',
            fg='#ffffff'
        )
        stats_label.pack(pady=(20, 10))
        
        self.stats_text = tk.Text(
            right_panel,
            height=15,
            font=('Consolas', 11),
            bg='#1e1e1e',
            fg='#ffffff',
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=('Arial', 10),
            bg='#007acc',
            fg='white',
            anchor=tk.W,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_camera(self):
        """Start/stop camera"""
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture"""
        # Check if MediaPipe is available
        if self.detector.mp_hands is None:
            self.update_status("❌ MediaPipe not available - cannot start camera", error=True)
            return
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            self.update_status("❌ Failed to open camera", error=True)
            return
        
        self.detector.hands = self.detector.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.is_camera_on = True
        self.detector.stats['session_start'] = datetime.now()
        
        self.start_button.config(text="⏸ Stop Camera", bg='#a51d2d')
        self.snapshot_button.config(state=tk.NORMAL)
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        self.update_status("📹 Camera started - Show hand gesture to the camera")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_camera_on = False
        
        if self.cap:
            self.cap.release()
        if self.detector.hands:
            self.detector.hands.close()
        
        self.start_button.config(text="▶ Start Camera", bg='#0e639c')
        self.snapshot_button.config(state=tk.DISABLED)
        
        self.update_status("Camera stopped")
    
    def video_loop(self):
        """Main video processing loop"""
        while self.is_camera_on:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.detector.hands.process(frame_rgb)
            
            # Draw hand landmarks and predict
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.detector.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.detector.mp_hands.HAND_CONNECTIONS,
                        self.detector.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.detector.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Extract landmarks and predict
                    landmarks = self.detector.extract_landmarks(hand_landmarks)
                    letter, confidence = self.detector.predict_gesture(landmarks)
                    
                    if letter:
                        # Update stats
                        self.detector.update_stats(letter, confidence)
                        
                        # Display prediction on frame
                        color = (0, 255, 0) if confidence >= self.detector.confidence_threshold else (0, 165, 255)
                        cv2.putText(frame, f"{letter}", (50, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
                        cv2.putText(frame, f"{confidence*100:.1f}%", (50, 160), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                        
                        # Update GUI prediction
                        self.root.after(0, self.update_prediction_display, letter, confidence)
            else:
                # No hand detected
                cv2.putText(frame, "No hand detected", (50, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                self.root.after(0, self.update_prediction_display, "--", 0.0)
            
            # Convert frame for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((780, 585), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update video label
            self.video_label.configure(image=photo)
            self.video_label.image = photo
    
    def update_prediction_display(self, letter, confidence):
        """Update the prediction display"""
        self.prediction_letter.config(text=letter)
        
        if letter == "--":
            self.confidence_label.config(text="Confidence: --")
            self.prediction_letter.config(fg='#808080')
        else:
            self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
            
            if confidence >= self.detector.confidence_threshold:
                self.prediction_letter.config(fg='#4ec9b0')
            else:
                self.prediction_letter.config(fg='#ce9178')
    
    def update_threshold(self, value):
        """Update confidence threshold"""
        self.detector.confidence_threshold = float(value)
        self.threshold_value_label.config(text=f"{float(value):.2f}")
    
    def update_stats_display(self):
        """Update statistics display"""
        if self.detector.stats['session_start']:
            elapsed = datetime.now() - self.detector.stats['session_start']
            elapsed_str = str(elapsed).split('.')[0]
        else:
            elapsed_str = "Not started"
        
        total = self.detector.stats['total_predictions']
        confident = self.detector.stats['confident_predictions']
        accuracy = (confident / total * 100) if total > 0 else 0
        
        # Top 5 detected letters
        top_letters = sorted(
            self.detector.stats['predicted_letters'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        stats_text = f"""
╔══════════════════════════════╗
║     SESSION STATISTICS       ║
╚══════════════════════════════╝

⏱️  Session Time: {elapsed_str}

📊 Predictions
   Total:      {total}
   Confident:  {confident}
   Accuracy:   {accuracy:.1f}%

🏆 Top Detected Letters:
"""
        
        if top_letters:
            for i, (letter, count) in enumerate(top_letters, 1):
                stats_text += f"   {i}. {letter}  →  {count} times\n"
        else:
            stats_text += "   No detections yet\n"
        
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats_text)
        
        # Schedule next update
        self.root.after(1000, self.update_stats_display)
    
    def take_snapshot(self):
        """Take a snapshot of current frame"""
        if self.cap and self.is_camera_on:
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                self.update_status(f"📷 Snapshot saved: {filename}")
    
    def update_status(self, message, error=False):
        """Update status bar"""
        self.status_bar.config(
            text=message,
            bg='#a51d2d' if error else '#007acc'
        )
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ISLDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
