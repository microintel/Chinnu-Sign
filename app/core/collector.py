"""
Data Collection Module for ISL Hand Gesture Detection
Collects hand landmark data for Indian Sign Language letters A-Z using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from typing import Optional, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import (
    create_directory_if_not_exists,
    save_numpy,
    draw_text_with_background,
    setup_logging
)


class HandLandmarkCollector:
    """
    Collects hand landmark data for ISL hand gestures using MediaPipe.
    Captures 21 hand keypoints (x, y, z) = 63 features per frame.
    """
    
    def __init__(
        self,
        data_dir: str = "../data/raw",
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the Hand Landmark Collector.
        
        Args:
            data_dir: Directory to save collected data
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.data_dir = data_dir
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = None
        self.cap = None
        
        # Statistics
        self.total_samples_collected = 0
        
        # Logging
        self.logger = setup_logging()
        
        self.logger.info("HandLandmarkCollector initialized")
    
    def initialize_camera(self, camera_id: int = 0) -> bool:
        """
        Initialize webcam for data collection.
        
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
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info(f"Camera {camera_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False
    
    def initialize_mediapipe(self) -> bool:
        """
        Initialize MediaPipe Hands model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            self.logger.info("MediaPipe Hands initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing MediaPipe: {e}")
            return False
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from frame.
        
        Args:
            frame: Input frame from camera
        
        Returns:
            Numpy array of shape (63,) containing x, y, z coordinates for 21 landmarks
            Returns None if no hand detected
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Get first hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: Input frame
            landmarks: Hand landmarks array
        
        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None:
            return frame
        
        # Convert BGR to RGB for processing
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
    
    def display_feedback(
        self,
        frame: np.ndarray,
        letter: str,
        count: int,
        total: int,
        status: str = "Ready"
    ) -> np.ndarray:
        """
        Display collection feedback on frame.
        
        Args:
            frame: Input frame
            letter: Current letter being collected
            count: Number of samples collected
            total: Total samples to collect
            status: Current status message
        
        Returns:
            Frame with feedback drawn
        """
        h, w = frame.shape[:2]
        
        # Draw title bar
        frame = draw_text_with_background(
            frame,
            f"ISL Data Collection - Letter: {letter}",
            (10, 30),
            font_scale=0.8,
            text_color=(0, 255, 0),
            bg_color=(0, 0, 0)
        )
        
        # Draw progress
        progress_text = f"Progress: {count}/{total} ({count*100//total if total > 0 else 0}%)"
        frame = draw_text_with_background(
            frame,
            progress_text,
            (10, 70),
            font_scale=0.7,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0)
        )
        
        # Draw status
        status_color = (0, 255, 0) if status == "Captured!" else (255, 255, 255)
        frame = draw_text_with_background(
            frame,
            f"Status: {status}",
            (10, 110),
            font_scale=0.7,
            text_color=status_color,
            bg_color=(0, 0, 0)
        )
        
        # Draw instructions at bottom
        instructions = [
            "SPACE - Capture sample",
            "N - Next letter",
            "Q - Quit"
        ]
        
        y_pos = h - 100
        for instruction in instructions:
            frame = draw_text_with_background(
                frame,
                instruction,
                (10, y_pos),
                font_scale=0.6,
                text_color=(200, 200, 200),
                bg_color=(0, 0, 0)
            )
            y_pos += 30
        
        return frame
    
    def save_landmarks(
        self,
        landmarks: np.ndarray,
        letter: str,
        sample_id: int
    ) -> bool:
        """
        Save landmarks to file.
        
        Args:
            landmarks: Landmarks array to save
            letter: Letter label
            sample_id: Sample identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create letter directory if it doesn't exist
            letter_dir = os.path.join(self.data_dir, letter)
            create_directory_if_not_exists(letter_dir)
            
            # Save landmarks
            filename = f"{letter}_{sample_id:04d}.npy"
            filepath = os.path.join(letter_dir, filename)
            
            np.save(filepath, landmarks)
            
            # Save metadata
            metadata = {
                'letter': letter,
                'sample_id': sample_id,
                'timestamp': datetime.now().isoformat(),
                'landmarks_shape': landmarks.shape,
                'num_keypoints': 21
            }
            
            metadata_file = filepath.replace('.npy', '_meta.txt')
            with open(metadata_file, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            self.total_samples_collected += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving landmarks: {e}")
            return False
    
    def collect_dataset_for_letter(
        self,
        letter: str,
        num_samples: int = 300
    ) -> int:
        """
        Collect dataset for a single letter.
        
        Args:
            letter: Letter to collect (A-Z)
            num_samples: Number of samples to collect
        
        Returns:
            Number of samples successfully collected
        """
        if not self.initialize_camera():
            return 0
        
        if not self.initialize_mediapipe():
            return 0
        
        samples_collected = 0
        status = "Ready - Show hand gesture and press SPACE"
        
        self.logger.info(f"Starting collection for letter '{letter}'")
        
        try:
            while samples_collected < num_samples:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                # Flip frame horizontally for mirror view
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                
                # Draw landmarks
                frame = self.draw_landmarks(frame, landmarks)
                
                # Display feedback
                frame = self.display_feedback(
                    frame,
                    letter,
                    samples_collected,
                    num_samples,
                    status
                )
                
                # Show frame
                cv2.imshow('ISL Data Collection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("Collection interrupted by user")
                    break
                
                elif key == ord('n'):
                    self.logger.info(f"Skipping to next letter (collected {samples_collected} samples)")
                    break
                
                elif key == ord(' '):
                    if landmarks is not None:
                        # Save landmarks
                        if self.save_landmarks(landmarks, letter, samples_collected):
                            samples_collected += 1
                            status = f"Captured! ({samples_collected}/{num_samples})"
                            self.logger.info(f"Saved sample {samples_collected}/{num_samples}")
                        else:
                            status = "Error saving sample!"
                    else:
                        status = "No hand detected! Please show hand gesture"
                
                # Reset status after 1 second
                if status.startswith("Captured") or status.startswith("Error"):
                    # Keep status for a few frames
                    pass
                else:
                    status = "Ready - Show hand gesture and press SPACE"
            
        except KeyboardInterrupt:
            self.logger.info("Collection interrupted by keyboard")
        
        finally:
            self.cleanup()
        
        self.logger.info(f"Collection completed: {samples_collected}/{num_samples} samples for '{letter}'")
        return samples_collected
    
    def collect_full_dataset(
        self,
        letters: Optional[List[str]] = None,
        samples_per_letter: int = 300
    ) -> dict:
        """
        Collect full dataset for all letters.
        
        Args:
            letters: List of letters to collect. If None, collects A-Z
            samples_per_letter: Number of samples to collect per letter
        
        Returns:
            Dictionary with letter-wise collection statistics
        """
        if letters is None:
            letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        
        statistics = {}
        
        for letter in letters:
            print(f"\n{'='*50}")
            print(f"Collecting data for letter: {letter}")
            print(f"{'='*50}\n")
            
            samples = self.collect_dataset_for_letter(letter, samples_per_letter)
            statistics[letter] = {
                'samples_collected': samples,
                'target_samples': samples_per_letter,
                'completion_rate': (samples / samples_per_letter) * 100
            }
            
            print(f"\nCompleted {letter}: {samples}/{samples_per_letter} samples")
            
            if samples < samples_per_letter:
                retry = input(f"\nOnly {samples} samples collected for '{letter}'. Retry? (y/n): ")
                if retry.lower() == 'y':
                    additional_samples = self.collect_dataset_for_letter(
                        letter,
                        samples_per_letter - samples
                    )
                    statistics[letter]['samples_collected'] += additional_samples
        
        # Print final statistics
        print(f"\n{'='*50}")
        print("Data Collection Summary")
        print(f"{'='*50}")
        
        for letter, stats in statistics.items():
            print(f"{letter}: {stats['samples_collected']}/{stats['target_samples']} "
                  f"({stats['completion_rate']:.1f}%)")
        
        total_collected = sum(s['samples_collected'] for s in statistics.values())
        total_target = sum(s['target_samples'] for s in statistics.values())
        
        print(f"\nTotal: {total_collected}/{total_target} samples")
        print(f"Overall completion: {(total_collected/total_target)*100:.1f}%")
        
        return statistics
    
    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        
        if self.hands is not None:
            self.hands.close()
        
        cv2.destroyAllWindows()
        
        self.logger.info("Resources released")


def main():
    """Main function for data collection."""
    print("=" * 50)
    print("ISL Hand Gesture Data Collection System")
    print("=" * 50)
    print("\nThis tool will help you collect hand landmark data")
    print("for Indian Sign Language letters (A-Z).")
    print("\nInstructions:")
    print("1. Show the hand gesture for the displayed letter")
    print("2. Press SPACE to capture a sample")
    print("3. Collect 300 samples per letter for best results")
    print("4. Press N to move to next letter")
    print("5. Press Q to quit\n")
    
    # Get data directory
    data_dir = input("Enter data directory path (default: ../data/raw): ").strip()
    if not data_dir:
        data_dir = "../data/raw"
    
    # Get samples per letter
    try:
        samples_input = input("Enter samples per letter (default: 300): ").strip()
        samples_per_letter = int(samples_input) if samples_input else 300
    except ValueError:
        print("Invalid input. Using default: 300")
        samples_per_letter = 300
    
    # Ask for specific letters or all
    collect_all = input("Collect all letters A-Z? (y/n, default: y): ").strip().lower()
    
    if collect_all == 'n':
        letters_input = input("Enter letters to collect (e.g., A,B,C): ").strip().upper()
        letters = [l.strip() for l in letters_input.split(',') if l.strip()]
    else:
        letters = None
    
    # Initialize collector
    collector = HandLandmarkCollector(data_dir=data_dir)
    
    # Start collection
    input("\nPress ENTER to start collection...")
    
    try:
        statistics = collector.collect_full_dataset(
            letters=letters,
            samples_per_letter=samples_per_letter
        )
        
        # Save statistics
        import json
        stats_file = os.path.join(data_dir, 'collection_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=4)
        
        print(f"\nStatistics saved to: {stats_file}")
        
    except Exception as e:
        print(f"\nError during collection: {e}")
    
    finally:
        collector.cleanup()
    
    print("\nData collection completed!")


if __name__ == "__main__":
    main()
