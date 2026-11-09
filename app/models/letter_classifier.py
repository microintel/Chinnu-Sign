"""
Real-time Letter Detection Module for ISL Hand Gesture Recognition
Detects hand gestures and classifies ISL letters using trained model
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import time
from typing import Optional, Tuple, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import (
    draw_text_with_background,
    setup_logging,
    get_label_mapping
)


class ISLLetterDetector:
    """
    Real-time ISL letter detector using MediaPipe and trained model.
    Maintains sequence buffer for temporal consistency.
    """
    
    def __init__(
        self,
        model_path: str,
        labels: Optional[List[str]] = None,
        sequence_length: int = 30,
        confidence_threshold: float = 0.7,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the letter detector.
        
        Args:
            model_path: Path to trained model file
            labels: List of label names (A-Z)
            sequence_length: Length of sequence buffer
            confidence_threshold: Minimum confidence for prediction
            min_detection_confidence: MediaPipe detection confidence
            min_tracking_confidence: MediaPipe tracking confidence
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize labels
        if labels is None:
            self.labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        else:
            self.labels = labels
        
        self.label_to_idx, self.idx_to_label = get_label_mapping(self.labels)
        
        # Load model
        self.logger = setup_logging()
        self.logger.info(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.logger.info("Model loaded successfully")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize sequence buffer
        self.sequence_buffer = deque(maxlen=sequence_length)
        
        # Initialize camera
        self.cap = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.current_letter = None
        self.current_confidence = 0.0
        self.detection_history = []
        
        # Detection state
        self.hand_detected = False
        
        self.logger.info("ISLLetterDetector initialized")
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        Initialize webcam.
        
        Args:
            camera_id: Camera device ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info(f"Camera {camera_id} initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Landmarks array of shape (63,) or None if no hand detected
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            self.hand_detected = True
            
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        self.hand_detected = False
        return None
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to [0, 1] range.
        
        Args:
            landmarks: Raw landmarks array
        
        Returns:
            Normalized landmarks
        """
        landmarks = landmarks.reshape(21, 3)
        
        # Normalize x and y
        min_vals = landmarks.min(axis=0)
        max_vals = landmarks.max(axis=0)
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (landmarks - min_vals) / range_vals
        
        return normalized.flatten()
    
    def add_to_sequence(self, landmarks: np.ndarray) -> None:
        """
        Add landmarks to sequence buffer.
        
        Args:
            landmarks: Landmarks to add
        """
        # Normalize landmarks
        normalized = self.normalize_landmarks(landmarks)
        
        # Add to buffer
        self.sequence_buffer.append(normalized)
    
    def predict_letter(self) -> Tuple[Optional[str], float]:
        """
        Predict letter from current sequence buffer.
        
        Returns:
            Tuple of (predicted_letter, confidence)
        """
        # Check if buffer is full
        if len(self.sequence_buffer) < self.sequence_length:
            return None, 0.0
        
        # Convert buffer to numpy array
        sequence = np.array(list(self.sequence_buffer))
        sequence = sequence.reshape(1, self.sequence_length, 63)
        
        # Predict
        predictions = self.model.predict(sequence, verbose=0)
        
        # Get best prediction
        confidence = np.max(predictions)
        predicted_idx = np.argmax(predictions)
        
        # Check confidence threshold
        if confidence >= self.confidence_threshold:
            predicted_letter = self.idx_to_label[predicted_idx]
            return predicted_letter, confidence
        
        return None, confidence
    
    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Frame with landmarks drawn
        """
        if not self.hand_detected:
            return frame
        
        # Convert to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def draw_prediction(
        self,
        frame: np.ndarray,
        letter: Optional[str],
        confidence: float
    ) -> np.ndarray:
        """
        Draw prediction on frame.
        
        Args:
            frame: Input frame
            letter: Predicted letter
            confidence: Prediction confidence
        
        Returns:
            Frame with prediction drawn
        """
        h, w = frame.shape[:2]
        
        # Draw hand detection status
        if self.hand_detected:
            status_text = "Hand Detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No Hand Detected"
            status_color = (0, 0, 255)
        
        frame = draw_text_with_background(
            frame,
            status_text,
            (10, 30),
            font_scale=0.7,
            text_color=status_color,
            bg_color=(0, 0, 0)
        )
        
        # Draw buffer status
        buffer_status = f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}"
        frame = draw_text_with_background(
            frame,
            buffer_status,
            (10, 70),
            font_scale=0.6,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0)
        )
        
        # Draw prediction
        if letter is not None:
            # Large letter display
            letter_display = f"Letter: {letter}"
            frame = draw_text_with_background(
                frame,
                letter_display,
                (10, 130),
                font_scale=1.5,
                font_thickness=3,
                text_color=(0, 255, 0),
                bg_color=(0, 0, 0),
                padding=10
            )
            
            # Confidence
            conf_display = f"Confidence: {confidence:.2%}"
            frame = draw_text_with_background(
                frame,
                conf_display,
                (10, 180),
                font_scale=0.8,
                text_color=(0, 255, 0),
                bg_color=(0, 0, 0)
            )
        elif confidence > 0:
            # Show low confidence message
            frame = draw_text_with_background(
                frame,
                f"Low Confidence: {confidence:.2%}",
                (10, 130),
                font_scale=0.8,
                text_color=(0, 165, 255),
                bg_color=(0, 0, 0)
            )
        
        # Draw FPS
        fps = self.calculate_fps()
        fps_text = f"FPS: {fps:.1f}"
        frame = draw_text_with_background(
            frame,
            fps_text,
            (w - 150, 30),
            font_scale=0.7,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0)
        )
        
        # Draw instructions
        instructions = [
            "Q - Quit",
            "R - Reset buffer",
            "S - Save detection"
        ]
        
        y_pos = h - 100
        for instruction in instructions:
            frame = draw_text_with_background(
                frame,
                instruction,
                (10, y_pos),
                font_scale=0.5,
                text_color=(200, 200, 200),
                bg_color=(0, 0, 0)
            )
            y_pos += 25
        
        return frame
    
    def calculate_fps(self) -> float:
        """
        Calculate current FPS.
        
        Returns:
            FPS value
        """
        if self.frame_count == 0:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0.0
    
    def reset_sequence(self) -> None:
        """Reset sequence buffer."""
        self.sequence_buffer.clear()
        self.current_letter = None
        self.current_confidence = 0.0
        self.logger.info("Sequence buffer reset")
    
    def save_detection(self) -> None:
        """Save current detection to history."""
        if self.current_letter is not None:
            detection = {
                'letter': self.current_letter,
                'confidence': self.current_confidence,
                'timestamp': time.time()
            }
            self.detection_history.append(detection)
            self.logger.info(f"Detection saved: {self.current_letter} ({self.current_confidence:.2%})")
    
    def run_detection(self) -> None:
        """Run real-time detection loop."""
        if not self.initialize_camera():
            return
        
        self.logger.info("Starting real-time detection...")
        print("\n" + "=" * 50)
        print("ISL Letter Detection - Press Q to quit")
        print("=" * 50)
        
        self.start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.error("Failed to read frame")
                    break
                
                # Flip frame for mirror view
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                
                # Process landmarks
                if landmarks is not None:
                    # Add to sequence
                    self.add_to_sequence(landmarks)
                    
                    # Predict
                    letter, confidence = self.predict_letter()
                    
                    # Update current detection
                    self.current_letter = letter
                    self.current_confidence = confidence
                else:
                    # No hand detected
                    self.current_letter = None
                    self.current_confidence = 0.0
                
                # Draw landmarks
                frame = self.draw_landmarks(frame)
                
                # Draw prediction
                frame = self.draw_prediction(
                    frame,
                    self.current_letter,
                    self.current_confidence
                )
                
                # Show frame
                cv2.imshow('ISL Letter Detection', frame)
                
                # Update frame count
                self.frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_sequence()
                elif key == ord('s'):
                    self.save_detection()
        
        except KeyboardInterrupt:
            self.logger.info("Detection interrupted by keyboard")
        
        finally:
            self.cleanup()
        
        # Print statistics
        print("\n" + "=" * 50)
        print("Detection Session Summary")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {self.calculate_fps():.1f}")
        print(f"Detections saved: {len(self.detection_history)}")
        
        if self.detection_history:
            print("\nDetection History:")
            for i, det in enumerate(self.detection_history, 1):
                print(f"  {i}. {det['letter']} ({det['confidence']:.2%})")
    
    def cleanup(self) -> None:
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        
        if self.hands is not None:
            self.hands.close()
        
        cv2.destroyAllWindows()
        
        self.logger.info("Resources released")


def main():
    """Main function for letter detection."""
    print("=" * 50)
    print("ISL Real-time Letter Detection")
    print("=" * 50)
    
    # Get model path
    model_path = input("Enter model path (default: ../models/letter_detection/hybrid_model_best.h5): ").strip()
    if not model_path:
        model_path = "../models/letter_detection/hybrid_model_best.h5"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Get parameters
    try:
        conf_threshold = input("Enter confidence threshold (default: 0.7): ").strip()
        confidence_threshold = float(conf_threshold) if conf_threshold else 0.7
    except ValueError:
        confidence_threshold = 0.7
    
    try:
        seq_len = input("Enter sequence length (default: 30): ").strip()
        sequence_length = int(seq_len) if seq_len else 30
    except ValueError:
        sequence_length = 30
    
    # Initialize detector
    try:
        detector = ISLLetterDetector(
            model_path=model_path,
            sequence_length=sequence_length,
            confidence_threshold=confidence_threshold
        )
        
        # Run detection
        detector.run_detection()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
