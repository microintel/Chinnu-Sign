"""
Preprocessing Module for ISL Hand Gesture Detection
Handles data loading, normalization, augmentation, and train/val/test splitting
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from typing import Tuple, List, Optional
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.utils import (
    save_pickle,
    load_pickle,
    save_numpy,
    load_numpy,
    create_directory_if_not_exists,
    setup_logging
)


class ISLDataPreprocessor:
    """
    Preprocesses ISL hand landmark data for model training.
    Handles loading, normalization, augmentation, and splitting.
    """
    
    def __init__(self, raw_data_dir: str = "../data/raw"):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_dir: Directory containing raw landmark data
        """
        self.raw_data_dir = raw_data_dir
        self.logger = setup_logging()
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.label_encoder = LabelEncoder()
        self.num_classes = 26
        
        self.logger.info("ISLDataPreprocessor initialized")
    
    def load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load raw landmark data from directories.
        
        Returns:
            Tuple of (X, y, labels) where:
                - X: numpy array of shape (num_samples, 63)
                - y: numpy array of labels
                - labels: list of label names
        """
        self.logger.info("Loading raw data...")
        
        X_list = []
        y_list = []
        labels = []
        
        # Get all letter directories
        if not os.path.exists(self.raw_data_dir):
            raise ValueError(f"Data directory not found: {self.raw_data_dir}")
        
        letter_dirs = sorted([
            d for d in os.listdir(self.raw_data_dir)
            if os.path.isdir(os.path.join(self.raw_data_dir, d))
        ])
        
        if not letter_dirs:
            raise ValueError(f"No letter directories found in {self.raw_data_dir}")
        
        # Load data for each letter
        for letter in letter_dirs:
            letter_dir = os.path.join(self.raw_data_dir, letter)
            
            # Get all .npy files
            npy_files = [f for f in os.listdir(letter_dir) if f.endswith('.npy')]
            
            if not npy_files:
                self.logger.warning(f"No data files found for letter '{letter}'")
                continue
            
            labels.append(letter)
            
            # Load each sample
            for npy_file in npy_files:
                filepath = os.path.join(letter_dir, npy_file)
                
                try:
                    landmarks = np.load(filepath)
                    
                    # Ensure correct shape
                    if landmarks.shape == (21, 3):
                        landmarks = landmarks.flatten()
                    
                    if landmarks.shape[0] == 63:
                        X_list.append(landmarks)
                        y_list.append(letter)
                    else:
                        self.logger.warning(f"Invalid shape {landmarks.shape} for {filepath}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading {filepath}: {e}")
        
        if not X_list:
            raise ValueError("No valid data loaded")
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        
        self.logger.info(f"Loaded {len(X)} samples for {len(labels)} letters")
        
        # Print statistics
        for label in labels:
            count = np.sum(y == label)
            self.logger.info(f"  {label}: {count} samples")
        
        return X, y, labels
    
    def normalize_landmarks(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to [0, 1] range.
        
        Args:
            X: Input data of shape (num_samples, 63)
        
        Returns:
            Normalized data
        """
        self.logger.info("Normalizing landmarks...")
        
        X_normalized = np.zeros_like(X)
        
        for i in range(len(X)):
            landmarks = X[i].reshape(21, 3)
            
            # Normalize x and y coordinates
            min_vals = landmarks.min(axis=0)
            max_vals = landmarks.max(axis=0)
            
            # Avoid division by zero
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            
            normalized = (landmarks - min_vals) / range_vals
            
            X_normalized[i] = normalized.flatten()
        
        return X_normalized
    
    def augment_sample(
        self,
        landmarks: np.ndarray,
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translation_range: float = 0.1,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """
        Apply data augmentation to a single sample.
        
        Args:
            landmarks: Landmarks array of shape (63,)
            rotation_range: Range of rotation in degrees
            scale_range: Range of scaling factors
            translation_range: Range of translation
            noise_level: Level of Gaussian noise
        
        Returns:
            Augmented landmarks
        """
        landmarks = landmarks.copy().reshape(21, 3)
        
        # 1. Rotation (only in xy plane)
        angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        
        # Center the landmarks
        center = landmarks.mean(axis=0)
        landmarks_centered = landmarks - center
        
        # Apply rotation
        landmarks_rotated = np.dot(landmarks_centered, rotation_matrix.T)
        
        # 2. Scaling
        scale_factor = np.random.uniform(*scale_range)
        landmarks_scaled = landmarks_rotated * scale_factor
        
        # 3. Translation
        translation = np.random.uniform(-translation_range, translation_range, size=3)
        landmarks_translated = landmarks_scaled + translation
        
        # Move back to original center
        landmarks_augmented = landmarks_translated + center
        
        # 4. Add Gaussian noise
        noise = np.random.normal(0, noise_level, landmarks_augmented.shape)
        landmarks_augmented += noise
        
        # Clip to valid range
        landmarks_augmented = np.clip(landmarks_augmented, 0, 1)
        
        return landmarks_augmented.flatten()
    
    def augment_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data to increase dataset size.
        
        Args:
            X: Input data
            y: Labels
            augmentation_factor: Number of augmented samples per original sample
        
        Returns:
            Tuple of (X_augmented, y_augmented)
        """
        self.logger.info(f"Augmenting data with factor {augmentation_factor}...")
        
        X_augmented_list = [X]
        y_augmented_list = [y]
        
        for i in range(augmentation_factor):
            X_aug = np.array([
                self.augment_sample(sample)
                for sample in X
            ])
            
            X_augmented_list.append(X_aug)
            y_augmented_list.append(y)
        
        X_augmented = np.vstack(X_augmented_list)
        y_augmented = np.concatenate(y_augmented_list)
        
        self.logger.info(f"Data augmented: {len(X)} -> {len(X_augmented)} samples")
        
        return X_augmented, y_augmented
    
    def create_sequences(
        self,
        X: np.ndarray,
        sequence_length: int = 30
    ) -> np.ndarray:
        """
        Create sequences for LSTM/GRU input.
        Each sequence contains multiple frames of the same sample (repeated).
        
        Args:
            X: Input data of shape (num_samples, 63)
            sequence_length: Length of sequences
        
        Returns:
            Sequences of shape (num_samples, sequence_length, 63)
        """
        self.logger.info(f"Creating sequences of length {sequence_length}...")
        
        # For static gestures, repeat the same frame
        X_sequences = np.repeat(X[:, np.newaxis, :], sequence_length, axis=1)
        
        self.logger.info(f"Created sequences with shape: {X_sequences.shape}")
        
        return X_sequences
    
    def prepare_train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input data
            y: Labels
            test_size: Proportion of test data
            val_size: Proportion of validation data
            random_state: Random seed
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        self.logger.info("Splitting data into train/val/test sets...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        # One-hot encode labels
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Validation set: {len(X_val)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat
    
    def save_preprocessed_data(
        self,
        output_dir: str,
        include_label_encoder: bool = True
    ) -> None:
        """
        Save preprocessed data to files.
        
        Args:
            output_dir: Directory to save preprocessed data
            include_label_encoder: Whether to save label encoder
        """
        self.logger.info(f"Saving preprocessed data to {output_dir}...")
        
        create_directory_if_not_exists(output_dir)
        
        # Save train data
        np.save(os.path.join(output_dir, 'X_train.npy'), self.X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), self.y_train)
        
        # Save validation data
        np.save(os.path.join(output_dir, 'X_val.npy'), self.X_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), self.y_val)
        
        # Save test data
        np.save(os.path.join(output_dir, 'X_test.npy'), self.X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), self.y_test)
        
        # Save label encoder
        if include_label_encoder:
            with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'num_classes': self.num_classes,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'feature_dim': self.X_train.shape[-1],
            'labels': list(self.label_encoder.classes_),
            'data_shape': str(self.X_train.shape)
        }
        
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info("Preprocessed data saved successfully")
        self.logger.info(f"  Train: {self.X_train.shape}")
        self.logger.info(f"  Val: {self.X_val.shape}")
        self.logger.info(f"  Test: {self.X_test.shape}")
    
    def preprocess_pipeline(
        self,
        output_dir: str = "../data/processed",
        normalize: bool = True,
        augment: bool = True,
        augmentation_factor: int = 2,
        create_seq: bool = True,
        sequence_length: int = 30,
        test_size: float = 0.15,
        val_size: float = 0.15
    ) -> None:
        """
        Complete preprocessing pipeline.
        
        Args:
            output_dir: Directory to save processed data
            normalize: Whether to normalize landmarks
            augment: Whether to augment data
            augmentation_factor: Factor for data augmentation
            create_seq: Whether to create sequences
            sequence_length: Length of sequences
            test_size: Proportion of test data
            val_size: Proportion of validation data
        """
        # Load raw data
        X, y, labels = self.load_raw_data()
        
        # Normalize
        if normalize:
            X = self.normalize_landmarks(X)
        
        # Split data first
        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.prepare_train_val_test_split(X, y, test_size, val_size)
        
        # Augment training data only
        if augment:
            # Get original labels for augmentation
            y_train_labels = self.label_encoder.inverse_transform(
                np.argmax(y_train, axis=1)
            )
            
            # Augment
            X_train_aug, y_train_labels_aug = self.augment_data(
                X_train, y_train_labels, augmentation_factor
            )
            
            # Re-encode augmented labels
            y_train_encoded = self.label_encoder.transform(y_train_labels_aug)
            y_train = to_categorical(y_train_encoded, num_classes=self.num_classes)
            X_train = X_train_aug
        
        # Create sequences
        if create_seq:
            X_train = self.create_sequences(X_train, sequence_length)
            X_val = self.create_sequences(X_val, sequence_length)
            X_test = self.create_sequences(X_test, sequence_length)
        
        # Store in instance variables
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        # Save data
        self.save_preprocessed_data(output_dir)
        
        self.logger.info("Preprocessing pipeline completed!")


def main():
    """Main function for preprocessing."""
    print("=" * 50)
    print("ISL Data Preprocessing")
    print("=" * 50)
    
    # Get parameters
    raw_data_dir = input("Enter raw data directory (default: ../data/raw): ").strip()
    if not raw_data_dir:
        raw_data_dir = "../data/raw"
    
    output_dir = input("Enter output directory (default: ../data/processed): ").strip()
    if not output_dir:
        output_dir = "../data/processed"
    
    # Initialize preprocessor
    preprocessor = ISLDataPreprocessor(raw_data_dir=raw_data_dir)
    
    # Run preprocessing pipeline
    try:
        preprocessor.preprocess_pipeline(
            output_dir=output_dir,
            normalize=True,
            augment=True,
            augmentation_factor=2,
            create_seq=True,
            sequence_length=30,
            test_size=0.15,
            val_size=0.15
        )
        
        print("\nPreprocessing completed successfully!")
        print(f"Processed data saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
