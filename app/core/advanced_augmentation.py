"""
Advanced Data Augmentation for Hand Landmark Data
Includes rotation, scaling, noise injection, and synthetic variation
"""

import numpy as np
from typing import Tuple, List
import random


class HandLandmarkAugmentor:
    """Advanced augmentation for hand landmark data"""
    
    def __init__(self, seed=42):
        """Initialize augmentor"""
        np.random.seed(seed)
        random.seed(seed)
    
    def rotate_landmarks(self, landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate hand landmarks around the wrist (first point)
        
        Args:
            landmarks: Array of shape (63,) containing x,y,z coordinates
            angle_deg: Rotation angle in degrees
            
        Returns:
            Rotated landmarks
        """
        landmarks = landmarks.copy().reshape(21, 3)
        angle_rad = np.radians(angle_deg)
        
        # Get wrist position (first landmark)
        center = landmarks[0]
        
        # 2D rotation matrix (rotate in xy plane)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Translate to origin, rotate, translate back
        rotated = []
        for point in landmarks:
            centered = point - center
            rotated_point = rotation_matrix @ centered
            rotated.append(rotated_point + center)
        
        return np.array(rotated).flatten()
    
    def scale_landmarks(self, landmarks: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale hand landmarks (simulate different hand sizes or distances)
        
        Args:
            landmarks: Array of shape (63,)
            scale_factor: Scaling factor (e.g., 0.9 = 90% size, 1.1 = 110% size)
            
        Returns:
            Scaled landmarks
        """
        landmarks = landmarks.copy().reshape(21, 3)
        center = landmarks[0]
        
        scaled = []
        for point in landmarks:
            centered = point - center
            scaled_point = centered * scale_factor
            scaled.append(scaled_point + center)
        
        return np.array(scaled).flatten()
    
    def add_noise(self, landmarks: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to landmarks (simulate camera jitter/tracking errors)
        
        Args:
            landmarks: Array of shape (63,)
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Noisy landmarks
        """
        noise = np.random.normal(0, noise_level, landmarks.shape)
        return landmarks + noise
    
    def translate_landmarks(self, landmarks: np.ndarray, 
                           dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        """
        Translate hand landmarks (simulate different hand positions)
        
        Args:
            landmarks: Array of shape (63,)
            dx: Translation in x direction
            dy: Translation in y direction
            
        Returns:
            Translated landmarks
        """
        landmarks = landmarks.copy().reshape(21, 3)
        translation = np.array([dx, dy, 0])
        
        translated = landmarks + translation
        return translated.flatten()
    
    def flip_horizontal(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Flip landmarks horizontally (mirror effect)
        
        Args:
            landmarks: Array of shape (63,)
            
        Returns:
            Flipped landmarks
        """
        landmarks = landmarks.copy().reshape(21, 3)
        
        # Find x-axis center
        center_x = np.mean(landmarks[:, 0])
        
        # Flip x coordinates
        flipped = landmarks.copy()
        flipped[:, 0] = 2 * center_x - landmarks[:, 0]
        
        return flipped.flatten()
    
    def elastic_deformation(self, landmarks: np.ndarray, alpha: float = 0.02) -> np.ndarray:
        """
        Apply elastic deformation (simulate slight finger movements)
        
        Args:
            landmarks: Array of shape (63,)
            alpha: Deformation strength
            
        Returns:
            Deformed landmarks
        """
        landmarks = landmarks.copy().reshape(21, 3)
        
        # Apply random small deformations to each landmark
        for i in range(len(landmarks)):
            # Stronger deformation for finger tips, weaker for palm
            deformation_strength = alpha * (1 + i / 21)
            deformation = np.random.normal(0, deformation_strength, 3)
            landmarks[i] += deformation
        
        return landmarks.flatten()
    
    def augment(self, landmarks: np.ndarray, 
               num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of landmarks
        
        Args:
            landmarks: Original landmarks array of shape (63,)
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented landmark arrays
        """
        augmented = [landmarks.copy()]  # Include original
        
        for _ in range(num_augmentations):
            aug = landmarks.copy()
            
            # Randomly apply transformations
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                aug = self.rotate_landmarks(aug, angle)
            
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                aug = self.scale_landmarks(aug, scale)
            
            if random.random() > 0.5:
                aug = self.add_noise(aug, noise_level=random.uniform(0.005, 0.015))
            
            if random.random() > 0.5:
                dx = random.uniform(-0.05, 0.05)
                dy = random.uniform(-0.05, 0.05)
                aug = self.translate_landmarks(aug, dx, dy)
            
            if random.random() > 0.3:
                aug = self.elastic_deformation(aug, alpha=random.uniform(0.01, 0.03))
            
            augmented.append(aug)
        
        return augmented
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, 
                       augmentation_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment entire dataset
        
        Args:
            X: Input landmarks of shape (n_samples, 63)
            y: Labels of shape (n_samples,)
            augmentation_factor: Number of augmented versions per sample
            
        Returns:
            Augmented X and y
        """
        X_augmented = []
        y_augmented = []
        
        for i in range(len(X)):
            # Get augmented versions
            augmented_samples = self.augment(X[i], num_augmentations=augmentation_factor)
            
            # Add all versions
            for aug_sample in augmented_samples:
                X_augmented.append(aug_sample)
                y_augmented.append(y[i])
        
        return np.array(X_augmented), np.array(y_augmented)


def demo_augmentation():
    """Demonstrate augmentation capabilities"""
    print("="*70)
    print("  HAND LANDMARK AUGMENTATION DEMO")
    print("="*70)
    
    # Create sample landmark data
    np.random.seed(42)
    sample_landmarks = np.random.rand(63)
    sample_landmarks[:3] = [0.5, 0.5, 0.0]  # Wrist at center
    
    # Initialize augmentor
    augmentor = HandLandmarkAugmentor()
    
    print("\n📊 Original shape:", sample_landmarks.shape)
    
    # Test each augmentation
    print("\n1. Rotation (30°)")
    rotated = augmentor.rotate_landmarks(sample_landmarks, 30)
    print(f"   Output shape: {rotated.shape}")
    
    print("\n2. Scaling (1.2x)")
    scaled = augmentor.scale_landmarks(sample_landmarks, 1.2)
    print(f"   Output shape: {scaled.shape}")
    
    print("\n3. Adding noise")
    noisy = augmentor.add_noise(sample_landmarks, noise_level=0.01)
    print(f"   Output shape: {noisy.shape}")
    
    print("\n4. Translation")
    translated = augmentor.translate_landmarks(sample_landmarks, dx=0.1, dy=-0.05)
    print(f"   Output shape: {translated.shape}")
    
    print("\n5. Horizontal flip")
    flipped = augmentor.flip_horizontal(sample_landmarks)
    print(f"   Output shape: {flipped.shape}")
    
    print("\n6. Elastic deformation")
    deformed = augmentor.elastic_deformation(sample_landmarks, alpha=0.02)
    print(f"   Output shape: {deformed.shape}")
    
    print("\n7. Combined augmentation (5 versions)")
    augmented = augmentor.augment(sample_landmarks, num_augmentations=5)
    print(f"   Generated {len(augmented)} samples")
    
    print("\n8. Dataset augmentation")
    X = np.random.rand(100, 63)
    y = np.random.randint(0, 26, 100)
    X_aug, y_aug = augmentor.augment_dataset(X, y, augmentation_factor=3)
    print(f"   Original dataset: {X.shape}")
    print(f"   Augmented dataset: {X_aug.shape}")
    print(f"   Augmentation multiplier: {X_aug.shape[0] / X.shape[0]:.1f}x")
    
    print("\n✅ Augmentation demo complete!")
    print("="*70)


if __name__ == "__main__":
    demo_augmentation()
