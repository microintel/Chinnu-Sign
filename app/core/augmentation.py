"""
Advanced Data Augmentation Pipeline for ISL Project
Heavy augmentation using Albumentations library
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("❌ albumentations not installed. Install with: pip install albumentations")
    sys.exit(1)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ISLDataAugmentor:
    """Advanced data augmentation for ISL dataset"""
    
    def __init__(self, 
                 source_dir: str,
                 target_dir: str,
                 augmentations_per_image: int = 5):
        """
        Initialize augmentor
        
        Args:
            source_dir: Source dataset directory
            target_dir: Where to save augmented data
            augmentations_per_image: Number of augmented versions per image
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.augmentations_per_image = augmentations_per_image
        
        if not self.source_dir.exists():
            raise ValueError(f"Source directory not found: {source_dir}")
        
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'original_count': 0,
            'augmented_count': 0,
            'failed_count': 0,
            'classes': {}
        }
    
    def create_augmentation_pipeline(self, augmentation_level: str = 'heavy') -> A.Compose:
        """
        Create augmentation pipeline
        
        Args:
            augmentation_level: 'light', 'medium', or 'heavy'
            
        Returns:
            Albumentations Compose pipeline
        """
        if augmentation_level == 'light':
            return A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            ])
        
        elif augmentation_level == 'medium':
            return A.Compose([
                A.Rotate(limit=15, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, 
                                    val_shift_limit=15, p=0.5),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ])
        
        else:  # heavy
            return A.Compose([
                # Geometric transforms
                A.Rotate(limit=15, p=0.7),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    rotate=0,
                    scale=1.0,
                    p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    p=0.3
                ),
                
                # Color transforms
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=20, 
                    p=0.6
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                
                # Noise and blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.2),
                
                # Dropout and occlusions
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=20, 
                    max_width=20,
                    min_holes=3,
                    min_height=10,
                    min_width=10,
                    p=0.3
                ),
                
                # Compression artifacts
                A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
                
                # Shadow and lighting
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.2
                ),
            ])
    
    def augment_image(self, 
                     image_path: Path, 
                     pipeline: A.Compose,
                     num_augmentations: int) -> List[np.ndarray]:
        """
        Apply augmentation to single image
        
        Args:
            image_path: Path to source image
            pipeline: Augmentation pipeline
            num_augmentations: Number of augmented versions
            
        Returns:
            List of augmented images
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return []
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate augmentations
            augmented_images = []
            for i in range(num_augmentations):
                augmented = pipeline(image=image)['image']
                augmented_images.append(augmented)
            
            return augmented_images
            
        except Exception as e:
            logger.error(f"Error augmenting {image_path}: {e}")
            return []
    
    def augment_class(self, 
                     class_name: str, 
                     pipeline: A.Compose,
                     max_workers: int = 4) -> int:
        """
        Augment all images in a class
        
        Args:
            class_name: Name of the class
            pipeline: Augmentation pipeline
            max_workers: Number of parallel workers
            
        Returns:
            Number of augmented images created
        """
        # Find class directory
        class_dirs = list(self.source_dir.rglob(class_name))
        class_dir = None
        for d in class_dirs:
            if d.is_dir():
                class_dir = d
                break
        
        if not class_dir:
            logger.warning(f"Class directory not found: {class_name}")
            return 0
        
        # Create target directory
        target_class_dir = self.target_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
            image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            logger.warning(f"No images found in {class_dir}")
            return 0
        
        logger.info(f"Augmenting class '{class_name}': {len(image_files)} images")
        
        augmented_count = 0
        failed_count = 0
        
        # Process images
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for img_path in image_files:
                future = executor.submit(
                    self.augment_image, 
                    img_path, 
                    pipeline, 
                    self.augmentations_per_image
                )
                futures[future] = img_path
            
            # Collect results with progress bar
            with tqdm(total=len(image_files), desc=f"  {class_name}") as pbar:
                for future in as_completed(futures):
                    img_path = futures[future]
                    
                    try:
                        augmented_images = future.result()
                        
                        if augmented_images:
                            # Save augmented images
                            base_name = img_path.stem
                            
                            for i, aug_img in enumerate(augmented_images):
                                # Convert back to BGR for OpenCV
                                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                                
                                # Save
                                output_name = f"{base_name}_aug{i:02d}.jpg"
                                output_path = target_class_dir / output_name
                                cv2.imwrite(str(output_path), aug_img_bgr, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 95])
                                
                                augmented_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {e}")
                        failed_count += 1
                    
                    pbar.update(1)
        
        return augmented_count
    
    def augment_dataset(self, 
                       augmentation_level: str = 'heavy',
                       max_workers: int = 4):
        """
        Augment entire dataset
        
        Args:
            augmentation_level: 'light', 'medium', or 'heavy'
            max_workers: Number of parallel workers
        """
        logger.info("\n" + "="*80)
        logger.info(" "*25 + "DATA AUGMENTATION")
        logger.info("="*80)
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Target: {self.target_dir}")
        logger.info(f"Augmentation level: {augmentation_level}")
        logger.info(f"Augmentations per image: {self.augmentations_per_image}")
        logger.info("="*80)
        
        # Create pipeline
        pipeline = self.create_augmentation_pipeline(augmentation_level)
        
        # Find all classes
        classes = []
        for item in self.source_dir.iterdir():
            if item.is_dir():
                classes.append(item.name)
        
        if not classes:
            logger.error("No classes found in source directory")
            return
        
        classes = sorted(classes)
        logger.info(f"\nFound {len(classes)} classes: {', '.join(classes)}\n")
        
        # Process each class
        start_time = datetime.now()
        
        for class_name in classes:
            augmented_count = self.augment_class(class_name, pipeline, max_workers)
            
            self.stats['classes'][class_name] = {
                'augmented_count': augmented_count
            }
            self.stats['augmented_count'] += augmented_count
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("AUGMENTATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Classes processed: {len(classes)}")
        logger.info(f"Total augmented images: {self.stats['augmented_count']}")
        logger.info(f"Time elapsed: {elapsed:.1f}s")
        logger.info(f"Augmented images saved to: {self.target_dir}")
        logger.info("="*80)
        
        # Save statistics
        self.save_statistics()
    
    def visualize_augmentations(self, 
                               class_name: str,
                               num_samples: int = 3,
                               output_path: str = "data/processed/augmentation_examples.png"):
        """
        Visualize augmentation examples
        
        Args:
            class_name: Class to visualize
            num_samples: Number of original images to show
            output_path: Where to save visualization
        """
        # Find class in source
        class_dirs = list(self.source_dir.rglob(class_name))
        class_dir = None
        for d in class_dirs:
            if d.is_dir():
                class_dir = d
                break
        
        if not class_dir:
            logger.error(f"Class not found: {class_name}")
            return
        
        # Get sample images
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if len(image_files) < num_samples:
            num_samples = len(image_files)
        
        # Create pipeline
        pipeline = self.create_augmentation_pipeline('heavy')
        
        # Create figure
        fig, axes = plt.subplots(num_samples, self.augmentations_per_image + 1,
                                figsize=((self.augmentations_per_image + 1) * 3, num_samples * 3))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Process samples
        for i, img_path in enumerate(image_files[:num_samples]):
            # Read original
            original = cv2.imread(str(img_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Show original
            axes[i, 0].imshow(original)
            axes[i, 0].set_title('Original', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Generate and show augmentations
            for j in range(self.augmentations_per_image):
                augmented = pipeline(image=original)['image']
                axes[i, j + 1].imshow(augmented)
                axes[i, j + 1].set_title(f'Aug {j+1}')
                axes[i, j + 1].axis('off')
        
        plt.suptitle(f'Augmentation Examples - Class: {class_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Augmentation examples saved to: {output_path}")
        plt.close()
    
    def save_statistics(self):
        """Save augmentation statistics"""
        stats_path = self.target_dir / "augmentation_stats.json"
        
        stats_data = {
            'source_dir': str(self.source_dir),
            'target_dir': str(self.target_dir),
            'augmentations_per_image': self.augmentations_per_image,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"✓ Statistics saved to: {stats_path}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment ISL dataset')
    parser.add_argument('--source', type=str, required=True,
                       help='Source dataset directory')
    parser.add_argument('--target', type=str, required=True,
                       help='Target directory for augmented data')
    parser.add_argument('--num-augmentations', type=int, default=5,
                       help='Number of augmentations per image')
    parser.add_argument('--level', type=str, default='heavy',
                       choices=['light', 'medium', 'heavy'],
                       help='Augmentation intensity level')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization of augmentations')
    parser.add_argument('--viz-class', type=str, default='A',
                       help='Class to visualize')
    
    args = parser.parse_args()
    
    try:
        # Create augmentor
        augmentor = ISLDataAugmentor(
            source_dir=args.source,
            target_dir=args.target,
            augmentations_per_image=args.num_augmentations
        )
        
        # Visualize if requested
        if args.visualize:
            logger.info("Generating augmentation visualization...")
            augmentor.visualize_augmentations(args.viz_class)
        
        # Augment dataset
        augmentor.augment_dataset(
            augmentation_level=args.level,
            max_workers=args.workers
        )
        
        logger.info("\n✓ Augmentation complete!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
