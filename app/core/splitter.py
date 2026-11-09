"""
Stratified Data Splitter for ISL Project
Creates train/validation/test splits with class balance
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Stratified dataset splitter for ISL"""
    
    def __init__(self,
                 source_dir: str,
                 output_dir: str,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize splitter
        
        Args:
            source_dir: Source dataset directory
            output_dir: Output directory for splits
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001:
            raise ValueError("Ratios must sum to 1.0")
        
        # Create output directories
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_class_files(self) -> Dict[str, List[Path]]:
        """
        Get all files organized by class
        
        Returns:
            Dictionary mapping class names to list of file paths
        """
        class_files = {}
        
        # Find all class directories
        for class_dir in sorted(self.source_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            files = []
            
            # Get all image/landmark files
            for file_path in class_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.npy']:
                    files.append(file_path)
            
            if files:
                class_files[class_name] = sorted(files)
        
        return class_files
    
    def split_data(self, copy_files: bool = True) -> Dict:
        """
        Split dataset into train/val/test with stratification
        
        Args:
            copy_files: Whether to copy files (True) or move them (False)
            
        Returns:
            Dictionary with split statistics
        """
        logger.info("\n" + "="*80)
        logger.info(" "*25 + "DATASET SPLITTING")
        logger.info("="*80)
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Ratios: Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={self.test_ratio:.0%}")
        logger.info(f"Random Seed: {self.random_seed}")
        logger.info("="*80)
        
        # Get class files
        class_files = self.get_class_files()
        
        if not class_files:
            logger.error("❌ No class directories found!")
            return {}
        
        logger.info(f"\nFound {len(class_files)} classes")
        
        # Statistics
        stats = {
            'source_dir': str(self.source_dir),
            'output_dir': str(self.output_dir),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'timestamp': datetime.now().isoformat(),
            'classes': {},
            'totals': {
                'train': 0,
                'val': 0,
                'test': 0
            }
        }
        
        # Process each class
        total_files = sum(len(files) for files in class_files.values())
        with tqdm(total=total_files, desc="Splitting files") as pbar:
            for class_name, files in sorted(class_files.items()):
                # Create class directories
                (self.train_dir / class_name).mkdir(exist_ok=True)
                (self.val_dir / class_name).mkdir(exist_ok=True)
                (self.test_dir / class_name).mkdir(exist_ok=True)
                
                # Split files
                np.random.seed(self.random_seed)
                
                # First split: train vs (val+test)
                train_files, temp_files = train_test_split(
                    files,
                    train_size=self.train_ratio,
                    random_state=self.random_seed,
                    shuffle=True
                )
                
                # Second split: val vs test
                val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
                val_files, test_files = train_test_split(
                    temp_files,
                    train_size=val_size,
                    random_state=self.random_seed,
                    shuffle=True
                )
                
                # Copy/move files
                for file_path, split_dir in [(train_files, self.train_dir),
                                              (val_files, self.val_dir),
                                              (test_files, self.test_dir)]:
                    for src_file in file_path:
                        dst_file = split_dir / class_name / src_file.name
                        if copy_files:
                            shutil.copy2(src_file, dst_file)
                        else:
                            shutil.move(str(src_file), dst_file)
                        pbar.update(1)
                
                # Update statistics
                stats['classes'][class_name] = {
                    'total': len(files),
                    'train': len(train_files),
                    'val': len(val_files),
                    'test': len(test_files)
                }
                
                stats['totals']['train'] += len(train_files)
                stats['totals']['val'] += len(val_files)
                stats['totals']['test'] += len(test_files)
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("SPLIT SUMMARY")
        logger.info("="*80)
        logger.info(f"Training:   {stats['totals']['train']:,} samples ({self.train_ratio:.0%})")
        logger.info(f"Validation: {stats['totals']['val']:,} samples ({self.val_ratio:.0%})")
        logger.info(f"Test:       {stats['totals']['test']:,} samples ({self.test_ratio:.0%})")
        logger.info(f"Total:      {sum(stats['totals'].values()):,} samples")
        logger.info("="*80)
        
        # Save metadata
        metadata_file = self.output_dir / 'split_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"\n✓ Metadata saved to: {metadata_file}")
        
        # Create visualization
        self.visualize_splits(stats)
        
        return stats
    
    def visualize_splits(self, stats: Dict):
        """Create visualization of splits"""
        logger.info("\nCreating visualizations...")
        
        # Prepare data
        classes = sorted(stats['classes'].keys())
        train_counts = [stats['classes'][c]['train'] for c in classes]
        val_counts = [stats['classes'][c]['val'] for c in classes]
        test_counts = [stats['classes'][c]['test'] for c in classes]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Split Analysis', fontsize=16, fontweight='bold')
        
        # 1. Stacked bar chart (per class)
        ax1 = axes[0, 0]
        x = np.arange(len(classes))
        width = 0.6
        
        ax1.bar(x, train_counts, width, label='Train', color='#2ecc71')
        ax1.bar(x, val_counts, width, bottom=train_counts, label='Validation', color='#3498db')
        ax1.bar(x, test_counts, width, bottom=np.array(train_counts)+np.array(val_counts), 
                label='Test', color='#e74c3c')
        
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Stacked Distribution per Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Grouped bar chart (comparing splits)
        ax2 = axes[0, 1]
        x2 = np.arange(len(classes))
        width2 = 0.25
        
        ax2.bar(x2 - width2, train_counts, width2, label='Train', color='#2ecc71')
        ax2.bar(x2, val_counts, width2, label='Validation', color='#3498db')
        ax2.bar(x2 + width2, test_counts, width2, label='Test', color='#e74c3c')
        
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Grouped Comparison')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Pie chart (overall distribution)
        ax3 = axes[1, 0]
        totals = [stats['totals']['train'], stats['totals']['val'], stats['totals']['test']]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        ax3.pie(totals, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax3.set_title('Overall Split Distribution')
        
        # 4. Ratio verification (ensure stratification)
        ax4 = axes[1, 1]
        
        train_ratios = [stats['classes'][c]['train'] / stats['classes'][c]['total'] for c in classes]
        val_ratios = [stats['classes'][c]['val'] / stats['classes'][c]['total'] for c in classes]
        test_ratios = [stats['classes'][c]['test'] / stats['classes'][c]['total'] for c in classes]
        
        x3 = np.arange(len(classes))
        ax4.plot(x3, train_ratios, 'o-', label='Train Ratio', color='#2ecc71', linewidth=2)
        ax4.plot(x3, val_ratios, 's-', label='Val Ratio', color='#3498db', linewidth=2)
        ax4.plot(x3, test_ratios, '^-', label='Test Ratio', color='#e74c3c', linewidth=2)
        
        ax4.axhline(y=self.train_ratio, color='#2ecc71', linestyle='--', alpha=0.3)
        ax4.axhline(y=self.val_ratio, color='#3498db', linestyle='--', alpha=0.3)
        ax4.axhline(y=self.test_ratio, color='#e74c3c', linestyle='--', alpha=0.3)
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Ratio')
        ax4.set_title('Stratification Verification (Ratios per Class)')
        ax4.set_xticks(x3)
        ax4.set_xticklabels(classes, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = self.output_dir / 'split_visualization.png'
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Visualization saved to: {viz_file}")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--source', type=str, required=True,
                       help='Source dataset directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for splits')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                       help='Training set ratio (default: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--copy', action='store_true',
                       help='Copy files instead of moving them')
    
    args = parser.parse_args()
    
    # Create splitter
    splitter = DatasetSplitter(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Split dataset
    splitter.split_data(copy_files=args.copy)
    
    logger.info("\n✓ Dataset splitting complete!")


if __name__ == "__main__":
    main()
