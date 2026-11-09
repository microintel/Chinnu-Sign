"""
Utility functions for ISL Hand Gesture Detection System
Provides common helper functions used across multiple modules
"""

import os
import json
import pickle
import numpy as np
import cv2
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import logging


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        log_file: Path to log file. If None, logs to console only.
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('ISL_System')
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directory_if_not_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to: {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from: {filepath}")
    return data


def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved to: {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_numpy(data: np.ndarray, filepath: str) -> None:
    """
    Save numpy array to file.
    
    Args:
        data: Numpy array to save
        filepath: Path to save file
    """
    np.save(filepath, data)
    print(f"Numpy array saved to: {filepath}")


def load_numpy(filepath: str) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        filepath: Path to numpy file
    
    Returns:
        Loaded numpy array
    """
    data = np.load(filepath)
    print(f"Numpy array loaded from: {filepath}")
    return data


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks to [0, 1] range.
    
    Args:
        landmarks: Array of shape (21, 3) or (63,) containing x, y, z coordinates
    
    Returns:
        Normalized landmarks
    """
    landmarks = landmarks.copy()
    
    # Reshape if flattened
    original_shape = landmarks.shape
    if len(landmarks.shape) == 1:
        landmarks = landmarks.reshape(21, 3)
    
    # Normalize x and y to [0, 1]
    min_vals = landmarks.min(axis=0)
    max_vals = landmarks.max(axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    normalized = (landmarks - min_vals) / range_vals
    
    # Restore original shape
    if len(original_shape) == 1:
        normalized = normalized.flatten()
    
    return normalized


def calculate_hand_bbox(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box around hand landmarks.
    
    Args:
        landmarks: Array of shape (21, 3) containing normalized x, y, z coordinates
        image_shape: Tuple of (height, width) of image
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates
    """
    h, w = image_shape[:2]
    
    x_coords = landmarks[:, 0] * w
    y_coords = landmarks[:, 1] * h
    
    x_min = int(max(0, np.min(x_coords) - 20))
    y_min = int(max(0, np.min(y_coords) - 20))
    x_max = int(min(w, np.max(x_coords) + 20))
    y_max = int(min(h, np.max(y_coords) + 20))
    
    return x_min, y_min, x_max, y_max


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    font_thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> np.ndarray:
    """
    Draw text with background rectangle on image.
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position of text
        font_scale: Font scale factor
        font_thickness: Thickness of text
        text_color: RGB color of text
        bg_color: RGB color of background
        padding: Padding around text
    
    Returns:
        Image with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA
    )
    
    return image


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate frames per second.
    
    Args:
        start_time: Start timestamp
        frame_count: Number of frames processed
    
    Returns:
        FPS value
    """
    elapsed_time = datetime.now().timestamp() - start_time
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Format timestamp to readable string.
    
    Args:
        timestamp: Unix timestamp. If None, uses current time.
    
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def generate_session_id() -> str:
    """
    Generate unique session ID based on timestamp.
    
    Returns:
        Session ID string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def validate_landmarks(landmarks: np.ndarray) -> bool:
    """
    Validate that landmarks array has correct shape and values.
    
    Args:
        landmarks: Landmarks array to validate
    
    Returns:
        True if valid, False otherwise
    """
    # Check shape
    if landmarks.shape not in [(21, 3), (63,)]:
        return False
    
    # Check for NaN or Inf
    if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
        return False
    
    return True


def get_label_mapping(letters: List[str] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label-to-index and index-to-label mappings.
    
    Args:
        letters: List of letters. If None, uses A-Z.
    
    Returns:
        Tuple of (label_to_idx, idx_to_label) dictionaries
    """
    if letters is None:
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    label_to_idx = {letter: idx for idx, letter in enumerate(letters)}
    idx_to_label = {idx: letter for letter, idx in label_to_idx.items()}
    
    return label_to_idx, idx_to_label


def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Resize frame to specified dimensions.
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
    
    Returns:
        Resized frame
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def apply_brightness_contrast(
    image: np.ndarray,
    brightness: int = 0,
    contrast: int = 0
) -> np.ndarray:
    """
    Apply brightness and contrast adjustments to image.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
    
    Returns:
        Adjusted image
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image


def count_files_in_directory(directory: str, extension: str = None) -> int:
    """
    Count files in directory with optional extension filter.
    
    Args:
        directory: Path to directory
        extension: File extension to filter (e.g., '.npy')
    
    Returns:
        Number of files
    """
    if not os.path.exists(directory):
        return 0
    
    files = os.listdir(directory)
    
    if extension:
        files = [f for f in files if f.endswith(extension)]
    
    return len(files)


def print_system_info():
    """Print system information for debugging."""
    import sys
    import platform
    
    print("=" * 50)
    print("System Information")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"OpenCV Version: {cv2.__version__}")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError:
        print("TensorFlow: Not installed")
    
    try:
        import mediapipe as mp
        print(f"MediaPipe Version: {mp.__version__}")
    except ImportError:
        print("MediaPipe: Not installed")
    
    print("=" * 50)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test label mapping
    label_to_idx, idx_to_label = get_label_mapping()
    print(f"Labels: {list(label_to_idx.keys())}")
    print(f"Number of classes: {len(label_to_idx)}")
    
    # Test landmark normalization
    test_landmarks = np.random.rand(21, 3)
    normalized = normalize_landmarks(test_landmarks)
    print(f"Normalized landmarks shape: {normalized.shape}")
    print(f"Normalized landmarks range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Test session ID generation
    session_id = generate_session_id()
    print(f"Generated session ID: {session_id}")
    
    # Print system info
    print_system_info()
    
    print("\nAll utility tests passed!")
