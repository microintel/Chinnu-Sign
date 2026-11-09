
"""
Advanced Real-Time ISL Hand Gesture Recognition System
A/B/V gesture recognition with webcam GUI, predictions and statistics.
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from datetime import datetime
import json
import os


class AdvancedISLDetector:

    def __init__(self, model_path='models/trained/isl_test_model.h5'):
        self.model_path = model_path
        self.model = None

        # MediaPipe initialization
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = None
        except (ImportError, AttributeError) as e:
            print(f"MediaPipe error: {e}")
            self.mp_hands = None
            self.mp_draw = None
            self.hands = None

        # Load trained model labels
        label_path = 'models/trained/isl_test_labels.json'

        try:
            with open(label_path, 'r') as f:
                label_map = json.load(f)

            self.letters = [
                letter for letter, index in
                sorted(label_map.items(), key=lambda item: item[1])
            ]

        except Exception as e:
            print(f"Label loading error: {e}")
            self.letters = ['A', 'B', 'V']

        self.confidence_threshold = 0.7

        # Statistics
        self.stats = {
            'total_predictions': 0,
            'confident_predictions': 0,
            'session_start': None,
            'predicted_letters': {}
        }

        self.load_model()

    def load_model(self):
        """Load trained TensorFlow model."""

        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"Model loaded: {self.model_path}")
                print(f"Available letters: {self.letters}")
            else:
                print(f"Model not found: {self.model_path}")

        except Exception as e:
            print(f"Model loading error: {e}")

    def extract_landmarks(self, hand_landmarks):
        """Extract 63 hand landmark coordinates."""

        if hand_landmarks is None:
            return None

        landmarks = []

        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return np.array(landmarks, dtype=np.float32)

    def predict_gesture(self, landmarks):
        """Predict ISL letter from hand landmarks."""

        if self.model is None or landmarks is None:
            return None, 0.0

        try:
            landmarks = landmarks.reshape(1, 63)

            prediction = self.model.predict(
                landmarks, verbose=0
            )[0]

            if len(prediction) != len(self.letters):
                raise ValueError(
                    "Model output classes do not match label count"
                )

            letter_idx = int(np.argmax(prediction))
            confidence = float(prediction[letter_idx])

            letter = self.letters[letter_idx]

            return letter, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

    def update_stats(self, letter, confidence):
        """Update prediction statistics."""

        self.stats['total_predictions'] += 1

        if confidence >= self.confidence_threshold:
            self.stats['confident_predictions'] += 1

            if letter not in self.stats['predicted_letters']:
                self.stats['predicted_letters'][letter] = 0

            self.stats['predicted_letters'][letter] += 1


class ISLDetectorGUI:

    def __init__(self, root):

        self.root = root
        self.root.title(
            "ISL Hand Gesture Recognition System"
        )
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')

        self.detector = AdvancedISLDetector()

        self.cap = None
        self.is_camera_on = False
        self.video_thread = None

        self.latest_frame = None
        self.latest_prediction = ("--", 0.0)

        self.setup_ui()

        self.update_stats_display()
        self.update_video_display()

    def setup_ui(self):

        title_frame = tk.Frame(
            self.root,
            bg='#2d2d30',
            height=80
        )
        title_frame.pack(
            fill=tk.X,
            padx=10,
            pady=10
        )
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="Indian Sign Language Recognition System",
            font=('Arial', 24, 'bold'),
            bg='#2d2d30',
            fg='white'
        )
        title_label.pack(pady=20)

        main_frame = tk.Frame(
            self.root,
            bg='#1e1e1e'
        )
        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=5
        )

        # Left video panel
        left_panel = tk.Frame(
            main_frame,
            bg='#2d2d30',
            width=800
        )
        left_panel.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(0, 5)
        )

        self.video_label = tk.Label(
            left_panel,
            bg='black'
        )
        self.video_label.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
        )

        # Buttons
        control_frame = tk.Frame(
            left_panel,
            bg='#2d2d30'
        )
        control_frame.pack(
            fill=tk.X,
            padx=10,
            pady=10
        )

        self.start_button = tk.Button(
            control_frame,
            text="Start Camera",
            command=self.toggle_camera,
            font=('Arial', 14, 'bold'),
            bg='#0e639c',
            fg='white',
            padx=20,
            pady=10
        )
        self.start_button.pack(
            side=tk.LEFT,
            padx=5
        )

        self.snapshot_button = tk.Button(
            control_frame,
            text="Snapshot",
            command=self.take_snapshot,
            font=('Arial', 14),
            bg='#444444',
            fg='white',
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.snapshot_button.pack(
            side=tk.LEFT,
            padx=5
        )

        # Right panel
        right_panel = tk.Frame(
            main_frame,
            bg='#2d2d30',
            width=400
        )
        right_panel.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            padx=(5, 0)
        )
        right_panel.pack_propagate(False)

        tk.Label(
            right_panel,
            text="Current Detection",
            font=('Arial', 16, 'bold'),
            bg='#2d2d30',
            fg='white'
        ).pack(pady=20)

        self.prediction_frame = tk.Frame(
            right_panel,
            bg='#1e1e1e',
            height=200
        )
        self.prediction_frame.pack(
            fill=tk.X,
            padx=20,
            pady=10
        )
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
            fg='white'
        )
        self.confidence_label.pack()

        # Confidence slider
        threshold_frame = tk.Frame(
            right_panel,
            bg='#2d2d30'
        )
        threshold_frame.pack(
            fill=tk.X,
            padx=20,
            pady=10
        )

        tk.Label(
            threshold_frame,
            text="Confidence Threshold",
            font=('Arial', 12),
            bg='#2d2d30',
            fg='white'
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
        self.threshold_slider.pack(
            fill=tk.X,
            pady=5
        )

        self.threshold_value_label = tk.Label(
            threshold_frame,
            text="0.70",
            font=('Arial', 12, 'bold'),
            bg='#2d2d30',
            fg='#4ec9b0'
        )
        self.threshold_value_label.pack()

        # Statistics
        tk.Label(
            right_panel,
            text="Session Statistics",
            font=('Arial', 16, 'bold'),
            bg='#2d2d30',
            fg='white'
        ).pack(pady=15)

        self.stats_text = tk.Text(
            right_panel,
            height=15,
            font=('Consolas', 11),
            bg='#1e1e1e',
            fg='white',
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.stats_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=10
        )

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
        self.status_bar.pack(
            side=tk.BOTTOM,
            fill=tk.X
        )

    def toggle_camera(self):

        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):

        if self.detector.mp_hands is None:
            self.update_status(
                "MediaPipe is unavailable",
                error=True
            )
            return

        if self.detector.model is None:
            self.update_status(
                "Model is not loaded. Check model file.",
                error=True
            )
            return

        self.cap = cv2.VideoCapture(0)

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, 1280
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, 720
        )

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.update_status(
                "Could not open webcam",
                error=True
            )
            return

        self.detector.hands = self.detector.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.is_camera_on = True
        self.detector.stats['session_start'] = datetime.now()

        self.start_button.config(
            text="Stop Camera",
            bg='#a51d2d'
        )

        self.snapshot_button.config(
            state=tk.NORMAL
        )

        self.video_thread = threading.Thread(
            target=self.video_loop,
            daemon=True
        )
        self.video_thread.start()

        self.update_status(
            "Camera started - Show your hand gesture"
        )

    def stop_camera(self):

        self.is_camera_on = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.detector.hands is not None:
            self.detector.hands.close()
            self.detector.hands = None

        self.start_button.config(
            text="Start Camera",
            bg='#0e639c'
        )

        self.snapshot_button.config(
            state=tk.DISABLED
        )

        self.update_status("Camera stopped")

    def video_loop(self):
        """Process webcam frames in background thread."""

        while self.is_camera_on and self.cap is not None:

            ret, frame = self.cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.detector.hands.process(
                frame_rgb
            )

            letter = "--"
            confidence = 0.0

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    self.detector.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.detector.mp_hands.HAND_CONNECTIONS
                    )

                    landmarks = self.detector.extract_landmarks(
                        hand_landmarks
                    )

                    letter, confidence = (
                        self.detector.predict_gesture(landmarks)
                    )

                    if letter is not None:

                        self.detector.update_stats(
                            letter,
                            confidence
                        )

                        color = (
                            (0, 255, 0)
                            if confidence >= self.detector.confidence_threshold
                            else (0, 165, 255)
                        )

                        cv2.putText(
                            frame,
                            letter,
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3,
                            color,
                            5
                        )

                        cv2.putText(
                            frame,
                            f"{confidence * 100:.1f}%",
                            (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            color,
                            3
                        )

            else:

                cv2.putText(
                    frame,
                    "No hand detected",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3
                )

            # Store data for Tkinter main thread
            self.latest_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            self.latest_prediction = (
                letter if letter is not None else "--",
                confidence
            )

    def update_video_display(self):
        """Update video and prediction from Tkinter main thread."""

        if self.is_camera_on and self.latest_frame is not None:

            img = Image.fromarray(self.latest_frame)

            img = img.resize(
                (780, 585),
                Image.Resampling.LANCZOS
            )

            photo = ImageTk.PhotoImage(image=img)

            self.video_label.configure(image=photo)
            self.video_label.image = photo

            letter, confidence = self.latest_prediction

            self.update_prediction_display(
                letter,
                confidence
            )

        self.root.after(
            30,
            self.update_video_display
        )

    def update_prediction_display(
        self,
        letter,
        confidence
    ):

        self.prediction_letter.config(
            text=letter
        )

        if letter == "--":

            self.confidence_label.config(
                text="Confidence: --"
            )

            self.prediction_letter.config(
                fg='#808080'
            )

        else:

            self.confidence_label.config(
                text=f"Confidence: {confidence * 100:.1f}%"
            )

            if confidence >= self.detector.confidence_threshold:

                self.prediction_letter.config(
                    fg='#4ec9b0'
                )

            else:

                self.prediction_letter.config(
                    fg='#ce9178'
                )

    def update_threshold(self, value):

        self.detector.confidence_threshold = float(value)

        self.threshold_value_label.config(
            text=f"{float(value):.2f}"
        )

    def update_stats_display(self):

        if self.detector.stats['session_start']:

            elapsed = (
                datetime.now()
                - self.detector.stats['session_start']
            )

            elapsed_str = str(elapsed).split('.')[0]

        else:

            elapsed_str = "Not started"

        total = self.detector.stats['total_predictions']

        confident = self.detector.stats['confident_predictions']

        percentage = (
            confident / total * 100
            if total > 0 else 0
        )

        top_letters = sorted(
            self.detector.stats['predicted_letters'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        stats_text = f"""
SESSION STATISTICS

Session Time: {elapsed_str}

PREDICTIONS
Total:       {total}
Confident:   {confident}
Above Threshold: {percentage:.1f}%

TOP DETECTED LETTERS
"""

        if top_letters:

            for i, (letter, count) in enumerate(
                top_letters, 1
            ):

                stats_text += (
                    f"{i}. {letter} -> {count} times\n"
                )

        else:

            stats_text += "No detections yet\n"

        self.stats_text.delete(
            '1.0',
            tk.END
        )

        self.stats_text.insert(
            '1.0',
            stats_text
        )

        self.root.after(
            1000,
            self.update_stats_display
        )

    def take_snapshot(self):
        """Save the latest webcam frame."""

        if self.latest_frame is not None:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = f"snapshot_{timestamp}.jpg"

            image = cv2.cvtColor(
                self.latest_frame,
                cv2.COLOR_RGB2BGR
            )

            cv2.imwrite(
                filename,
                image
            )

            self.update_status(
                f"Snapshot saved: {filename}"
            )

        else:

            self.update_status(
                "No camera frame available",
                error=True
            )

    def update_status(self, message, error=False):

        self.status_bar.config(
            text=message,
            bg='#a51d2d' if error else '#007acc'
        )

    def on_closing(self):

        self.stop_camera()
        self.root.destroy()


def main():

    root = tk.Tk()

    app = ISLDetectorGUI(root)

    root.protocol(
        "WM_DELETE_WINDOW",
        app.on_closing
    )

    root.mainloop()


if __name__ == "__main__":
    main()

