"""
Kaggle Dataset Downloader and Explorer for ISL Project
Downloads and analyzes the ISL dataset from Kaggle
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

try:
    import kagglehub
except ImportError:
    print("❌ kagglehub not installed. Install with: pip install kagglehub")
    sys.exit(1)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KaggleISLDatasetDownloader:
    """Download and explore Kaggle ISL dataset"""
    
    def __init__(self, target_dir: str = "data/raw/kaggle_isl"):
        """
        Initialize downloader
        
        Args:
            target_dir: Directory to store downloaded dataset
        """
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = None
        self.stats = {}
        
    def download_dataset(self, dataset_name: str = "ayushmanrayaguru/indian-sign-language") -> str:
        """
        Download ISL dataset from Kaggle
        
        Args:
            dataset_name: Kaggle dataset identifier
            
        Returns:
            Path to downloaded dataset
        """
        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info("This may take a few minutes depending on dataset size...")
        
        try:
            # Download using kagglehub
            path = kagglehub.dataset_download(dataset_name)
            logger.info(f"✓ Dataset downloaded to: {path}")
            
            self.dataset_path = Path(path)
            return str(path)
            
        except Exception as e:
            logger.error(f"❌ Failed to download dataset: {e}")
            logger.error("Make sure you have:")
            logger.error("  1. Kaggle API credentials configured (~/.kaggle/kaggle.json)")
            logger.error("  2. Accepted the dataset terms on Kaggle website")
            raise
    
    def explore_structure(self, dataset_path: str = None) -> Dict:
        """
        Explore dataset directory structure
        
        Args:
            dataset_path: Path to dataset (uses downloaded path if None)
            
        Returns:
            Dictionary with structure information
        """
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path or not self.dataset_path.exists():
            raise ValueError("Dataset path not found. Download dataset first.")
        
        logger.info("\n" + "="*60)
        logger.info("DATASET STRUCTURE EXPLORATION")
        logger.info("="*60)
        
        structure = {
            'root': str(self.dataset_path),
            'folders': [],
            'total_files': 0
        }
        
        # Walk through directory
        for root, dirs, files in os.walk(self.dataset_path):
            rel_path = os.path.relpath(root, self.dataset_path)
            
            # Count image files
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if image_files:
                folder_info = {
                    'path': rel_path,
                    'num_files': len(image_files),
                    'sample_files': image_files[:3]
                }
                structure['folders'].append(folder_info)
                structure['total_files'] += len(image_files)
                
                logger.info(f"\nFolder: {rel_path}")
                logger.info(f"  Files: {len(image_files)}")
                logger.info(f"  Samples: {', '.join(image_files[:3])}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total folders: {len(structure['folders'])}")
        logger.info(f"Total image files: {structure['total_files']}")
        logger.info("="*60)
        
        return structure
    
    def analyze_dataset(self, dataset_path: str = None) -> Dict[str, int]:
        """
        Analyze class distribution in dataset
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Dictionary mapping class names to sample counts
        """
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path or not self.dataset_path.exists():
            raise ValueError("Dataset path not found. Download dataset first.")
        
        logger.info("\n" + "="*60)
        logger.info("CLASS DISTRIBUTION ANALYSIS")
        logger.info("="*60)
        
        classes = {}
        
        # Check common dataset structures
        possible_roots = [
            self.dataset_path,
            self.dataset_path / "train",
            self.dataset_path / "Train",
            self.dataset_path / "images",
            self.dataset_path / "data"
        ]
        
        for root in possible_roots:
            if not root.exists():
                continue
                
            for class_dir in root.iterdir():
                if not class_dir.is_dir():
                    continue
                
                class_name = class_dir.name
                
                # Count image files
                image_files = list(class_dir.glob("*.jpg")) + \
                             list(class_dir.glob("*.jpeg")) + \
                             list(class_dir.glob("*.png")) + \
                             list(class_dir.glob("*.bmp"))
                
                if image_files:
                    classes[class_name] = len(image_files)
        
        # Sort and display
        if classes:
            logger.info("\nClass Distribution:")
            total_samples = 0
            for cls in sorted(classes.keys()):
                count = classes[cls]
                total_samples += count
                logger.info(f"  {cls:>3}: {count:>6} samples")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Total classes: {len(classes)}")
            logger.info(f"Total samples: {total_samples}")
            logger.info(f"Average per class: {total_samples / len(classes):.1f}")
            logger.info(f"Min samples: {min(classes.values())}")
            logger.info(f"Max samples: {max(classes.values())}")
            logger.info("="*60)
            
            self.stats = {
                'classes': classes,
                'total_samples': total_samples,
                'num_classes': len(classes),
                'avg_per_class': total_samples / len(classes),
                'min_samples': min(classes.values()),
                'max_samples': max(classes.values())
            }
        else:
            logger.warning("❌ No classes found. Check dataset structure.")
        
        return classes
    
    def analyze_image_properties(self, dataset_path: str = None, 
                                 max_samples: int = 100) -> Dict:
        """
        Analyze image properties (size, aspect ratio, etc.)
        
        Args:
            dataset_path: Path to dataset
            max_samples: Maximum samples to analyze per class
            
        Returns:
            Dictionary with image statistics
        """
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        logger.info("\n" + "="*60)
        logger.info("IMAGE PROPERTIES ANALYSIS")
        logger.info("="*60)
        
        image_stats = {
            'widths': [],
            'heights': [],
            'aspect_ratios': [],
            'channels': [],
            'file_sizes': []
        }
        
        # Find all images
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend(list(self.dataset_path.rglob(f"*{ext}")))
            image_paths.extend(list(self.dataset_path.rglob(f"*{ext.upper()}")))
        
        # Sample images
        if len(image_paths) > max_samples:
            import random
            image_paths = random.sample(image_paths, max_samples)
        
        logger.info(f"Analyzing {len(image_paths)} sample images...")
        
        for img_path in image_paths:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1
                
                image_stats['widths'].append(w)
                image_stats['heights'].append(h)
                image_stats['aspect_ratios'].append(w / h)
                image_stats['channels'].append(channels)
                image_stats['file_sizes'].append(os.path.getsize(img_path))
                
            except Exception as e:
                logger.warning(f"Error reading {img_path}: {e}")
                continue
        
        # Calculate statistics
        if image_stats['widths']:
            logger.info("\nImage Statistics:")
            logger.info(f"  Width:  {np.mean(image_stats['widths']):.1f} ± {np.std(image_stats['widths']):.1f} px")
            logger.info(f"          (min: {min(image_stats['widths'])}, max: {max(image_stats['widths'])})")
            logger.info(f"  Height: {np.mean(image_stats['heights']):.1f} ± {np.std(image_stats['heights']):.1f} px")
            logger.info(f"          (min: {min(image_stats['heights'])}, max: {max(image_stats['heights'])})")
            logger.info(f"  Aspect: {np.mean(image_stats['aspect_ratios']):.2f} ± {np.std(image_stats['aspect_ratios']):.2f}")
            logger.info(f"  Channels: {int(np.mean(image_stats['channels']))}")
            logger.info(f"  File size: {np.mean(image_stats['file_sizes']) / 1024:.1f} ± {np.std(image_stats['file_sizes']) / 1024:.1f} KB")
        
        return image_stats
    
    def visualize_samples(self, dataset_path: str = None, 
                         samples_per_class: int = 3,
                         output_path: str = "data/processed/dataset_samples.png"):
        """
        Visualize sample images from each class
        
        Args:
            dataset_path: Path to dataset
            samples_per_class: Number of samples to show per class
            output_path: Where to save visualization
        """
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        
        logger.info("\n" + "="*60)
        logger.info("VISUALIZING DATASET SAMPLES")
        logger.info("="*60)
        
        # Get classes
        if not self.stats or 'classes' not in self.stats:
            self.analyze_dataset()
        
        classes = sorted(self.stats['classes'].keys())
        
        # Create figure
        num_classes = len(classes)
        fig, axes = plt.subplots(num_classes, samples_per_class, 
                                figsize=(samples_per_class * 3, num_classes * 3))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        # Plot samples
        for i, cls in enumerate(classes):
            # Find class directory
            class_dirs = list(self.dataset_path.rglob(cls))
            class_dir = None
            for d in class_dirs:
                if d.is_dir():
                    class_dir = d
                    break
            
            if not class_dir:
                continue
            
            # Get image files
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            
            # Plot samples
            for j in range(min(samples_per_class, len(image_files))):
                ax = axes[i, j] if num_classes > 1 else axes[j]
                
                img = cv2.imread(str(image_files[j]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
                
                if j == 0:
                    ax.set_ylabel(cls, fontsize=12, fontweight='bold')
                
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Samples visualization saved to: {output_path}")
        plt.close()
    
    def plot_class_distribution(self, output_path: str = "data/processed/class_distribution.png"):
        """
        Plot class distribution bar chart
        
        Args:
            output_path: Where to save plot
        """
        if not self.stats or 'classes' not in self.stats:
            self.analyze_dataset()
        
        classes = self.stats['classes']
        
        # Create plot
        plt.figure(figsize=(15, 6))
        
        sorted_classes = sorted(classes.items(), key=lambda x: x[0])
        class_names = [c[0] for c in sorted_classes]
        counts = [c[1] for c in sorted_classes]
        
        bars = plt.bar(range(len(class_names)), counts, color='steelblue', alpha=0.8)
        
        # Color bars based on count
        avg_count = np.mean(counts)
        for i, bar in enumerate(bars):
            if counts[i] < avg_count * 0.8:
                bar.set_color('orangered')
            elif counts[i] > avg_count * 1.2:
                bar.set_color('forestgreen')
        
        plt.xlabel('Class', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title('ISL Dataset - Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        
        # Add average line
        plt.axhline(y=avg_count, color='red', linestyle='--', 
                   label=f'Average: {avg_count:.0f}', linewidth=2)
        plt.legend()
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Class distribution plot saved to: {output_path}")
        plt.close()
    
    def save_metadata(self, output_path: str = "data/processed/kaggle_dataset_metadata.json"):
        """
        Save dataset metadata to JSON
        
        Args:
            output_path: Where to save metadata
        """
        metadata = {
            'dataset_path': str(self.dataset_path),
            'download_timestamp': datetime.now().isoformat(),
            'statistics': self.stats
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to: {output_path}")
    
    def run_full_analysis(self):
        """Run complete dataset analysis pipeline"""
        logger.info("\n" + "="*80)
        logger.info(" "*20 + "KAGGLE ISL DATASET ANALYSIS")
        logger.info("="*80)
        
        # Download
        if not self.dataset_path:
            self.download_dataset()
        
        # Explore
        self.explore_structure()
        
        # Analyze
        self.analyze_dataset()
        self.analyze_image_properties()
        
        # Visualize
        self.visualize_samples()
        self.plot_class_distribution()
        
        # Save metadata
        self.save_metadata()
        
        logger.info("\n" + "="*80)
        logger.info("✓ ANALYSIS COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nDataset path: {self.dataset_path}")
        logger.info(f"Total classes: {self.stats.get('num_classes', 0)}")
        logger.info(f"Total samples: {self.stats.get('total_samples', 0)}")
        logger.info("\nNext steps:")
        logger.info("  1. Review visualizations in data/processed/")
        logger.info("  2. Run data augmentation pipeline")
        logger.info("  3. Start custom data collection")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and explore Kaggle ISL dataset')
    parser.add_argument('--dataset', type=str, 
                       default='ayushmanrayaguru/indian-sign-language',
                       help='Kaggle dataset identifier')
    parser.add_argument('--target-dir', type=str, 
                       default='data/raw/kaggle_isl',
                       help='Target directory for dataset')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download if dataset exists')
    parser.add_argument('--dataset-path', type=str,
                       help='Path to existing dataset')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = KaggleISLDatasetDownloader(target_dir=args.target_dir)
    
    try:
        if args.skip_download and args.dataset_path:
            downloader.dataset_path = Path(args.dataset_path)
            logger.info(f"Using existing dataset at: {args.dataset_path}")
        else:
            downloader.download_dataset(args.dataset)
        
        # Run full analysis
        downloader.run_full_analysis()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
