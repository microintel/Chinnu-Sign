"""
Skeleton Visualizer Module for ISL Hand Gesture Detection
Creates white background skeleton drawings from MediaPipe landmarks
Inspired by ASL project approach to eliminate lighting/background issues
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import setup_logging


class HandSkeletonVisualizer:
    """
    Creates skeleton visualizations of hand landmarks on white background.
    This approach makes the model robust to lighting and background variations.
    """
    
    # MediaPipe hand connections for drawing skeleton
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]
    
    def __init__(self, canvas_size: int = 400, line_thickness: int = 3, 
                 point_radius: int = 4):
        """
        Initialize the skeleton visualizer.
        
        Args:
            canvas_size: Size of the white canvas (square)
            line_thickness: Thickness of skeleton lines
            point_radius: Radius of landmark points
        """
        self.canvas_size = canvas_size
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.logger = setup_logging()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.logger.info("HandSkeletonVisualizer initialized")
    
    def create_white_canvas(self) -> np.ndarray:
        """
        Create a white canvas for drawing skeleton.
        
        Returns:
            White image of size (canvas_size, canvas_size, 3)
        """
        return np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.uint8) * 255
    
    def normalize_landmarks(self, landmarks: List, 
                           bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Normalize landmarks to fit within canvas while maintaining aspect ratio.
        
        Args:
            landmarks: List of hand landmarks from MediaPipe
            bbox: Bounding box (x, y, w, h) of hand region
            
        Returns:
            Normalized landmarks as numpy array
        """
        x, y, w, h = bbox
        
        # Convert landmarks to numpy array
        points = np.array([[lm.x * w + x, lm.y * h + y] 
                          for lm in landmarks])
        
        # Get the center and scale to fit canvas
        center = points.mean(axis=0)
        
        # Scale to use 80% of canvas
        scale_factor = (self.canvas_size * 0.8) / max(w, h)
        
        # Center and scale the points
        centered_points = (points - center) * scale_factor
        canvas_center = self.canvas_size / 2
        normalized_points = centered_points + canvas_center
        
        return normalized_points.astype(int)
    
    def draw_skeleton(self, canvas: np.ndarray, landmarks: np.ndarray,
                     line_color: Tuple[int, int, int] = (0, 255, 0),
                     point_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Draw hand skeleton on canvas.
        
        Args:
            canvas: White canvas to draw on
            landmarks: Normalized landmark coordinates
            line_color: BGR color for skeleton lines
            point_color: BGR color for landmark points
            
        Returns:
            Canvas with drawn skeleton
        """
        # Draw connections (lines)
        for connection in self.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = tuple(landmarks[start_idx])
            end_point = tuple(landmarks[end_idx])
            
            # Draw line
            cv2.line(canvas, start_point, end_point, line_color, 
                    self.line_thickness)
        
        # Draw landmark points
        for point in landmarks:
            cv2.circle(canvas, tuple(point), self.point_radius, 
                      point_color, -1)
        
        return canvas
    
    def process_frame(self, frame: np.ndarray, 
                     hands_detector) -> Tuple[Optional[np.ndarray], 
                                             Optional[np.ndarray]]:
        """
        Process a frame and create skeleton visualization.
        
        Args:
            frame: Input BGR frame from camera
            hands_detector: MediaPipe Hands detector instance
            
        Returns:
            Tuple of (skeleton_image, landmarks_array) or (None, None)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = hands_detector.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        # Get first hand (assuming single hand)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Calculate bounding box
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        bbox = (x_min, y_min, bbox_w, bbox_h)
        
        # Normalize landmarks
        normalized_landmarks = self.normalize_landmarks(
            hand_landmarks.landmark, bbox
        )
        
        # Create white canvas
        skeleton_canvas = self.create_white_canvas()
        
        # Draw skeleton
        skeleton_image = self.draw_skeleton(
            skeleton_canvas, 
            normalized_landmarks
        )
        
        # Extract landmarks as flat array (21 points × 3 coords)
        landmarks_array = np.array([
            [lm.x, lm.y, lm.z] 
            for lm in hand_landmarks.landmark
        ]).flatten()
        
        return skeleton_image, landmarks_array
    
    def create_skeleton_from_landmarks(self, landmarks_array: np.ndarray,
                                      scale: bool = True) -> np.ndarray:
        """
        Create skeleton visualization from landmark array.
        Useful for visualizing stored landmark data.
        
        Args:
            landmarks_array: Flattened array of shape (63,) - 21 landmarks × 3
            scale: Whether to scale landmarks to canvas
            
        Returns:
            Skeleton image on white background
        """
        # Reshape to (21, 3)
        landmarks = landmarks_array.reshape(21, 3)
        
        if scale:
            # Scale x, y coordinates to canvas
            landmarks_2d = landmarks[:, :2]  # Take only x, y
            
            # Normalize to [0, 1] if needed
            if landmarks_2d.max() <= 1.0:
                landmarks_2d = landmarks_2d * self.canvas_size
            
            # Center in canvas
            center = landmarks_2d.mean(axis=0)
            canvas_center = self.canvas_size / 2
            landmarks_2d = landmarks_2d - center + canvas_center
        else:
            landmarks_2d = landmarks[:, :2]
        
        landmarks_2d = landmarks_2d.astype(int)
        
        # Create white canvas
        canvas = self.create_white_canvas()
        
        # Draw skeleton
        skeleton = self.draw_skeleton(canvas, landmarks_2d)
        
        return skeleton
    
    def create_comparison_view(self, original_frame: np.ndarray,
                              skeleton_image: np.ndarray,
                              prediction: Optional[str] = None) -> np.ndarray:
        """
        Create side-by-side comparison of original frame and skeleton.
        
        Args:
            original_frame: Original camera frame
            skeleton_image: Skeleton visualization
            prediction: Optional predicted character to display
            
        Returns:
            Combined image showing both views
        """
        # Resize original frame to match skeleton size
        resized_frame = cv2.resize(original_frame, 
                                   (self.canvas_size, self.canvas_size))
        
        # Create combined view
        combined = np.hstack([resized_frame, skeleton_image])
        
        # Add prediction text if provided
        if prediction:
            text = f"Predicted: {prediction}"
            cv2.putText(combined, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return combined


def demo_skeleton_visualizer():
    """
    Demo function to test skeleton visualizer with webcam.
    """
    import cv2
    
    # Initialize visualizer
    visualizer = HandSkeletonVisualizer(canvas_size=400)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    print("Press 's' to save skeleton image")
    
    save_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        skeleton_img, landmarks = visualizer.process_frame(frame, hands)
        
        if skeleton_img is not None:
            # Create comparison view
            combined = visualizer.create_comparison_view(frame, skeleton_img)
            cv2.imshow("Hand Detection - Original | Skeleton", combined)
        else:
            cv2.imshow("Hand Detection - Original | Skeleton", frame)
            cv2.putText(frame, "No hand detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and skeleton_img is not None:
            # Save skeleton image
            filename = f"skeleton_{save_counter}.jpg"
            cv2.imwrite(filename, skeleton_img)
            print(f"Saved {filename}")
            save_counter += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    demo_skeleton_visualizer()
