"""
Dataset Combination and Analysis Utilities
Combines multiple dataset sources and performs comprehensive analysis
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetCombiner:
    """Combine multiple ISL dataset sources"""
    
    def __init__(self, output_dir: str = 'data/processed/combined'):
        """
        Initialize combiner
        
        Args:
            output_dir: Directory for combined dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sources = []
        self.stats = {
            'sources': {},
            'total_samples': 0,
            'classes': defaultdict(int)
        }
    
    def add_source(self, name: str, path: str, dataset_type: str = 'images'):
        """
        Add a dataset source
        
        Args:
            name: Source name (e.g., 'kaggle', 'custom', 'augmented')
            path: Path to dataset
            dataset_type: 'images' or 'landmarks'
        """
        source_path = Path(path)
        if not source_path.exists():
            logger.warning(f"Source path not found: {path}")
            return
        
        self.sources.append({
            'name': name,
            'path': source_path,
            'type': dataset_type
        })
        
        logger.info(f"✓ Added source '{name}': {path}")
    
    def count_samples_in_source(self, source: Dict) -> Dict[str, int]:
        """
        Count samples in a source
        
        Args:
            source: Source dictionary
            
        Returns:
            Dictionary of class counts
        """
        counts = defaultdict(int)
        source_path = source['path']
        
        # Find class directories
        for class_dir in source_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Count files based on type
            if source['type'] == 'images':
                files = list(class_dir.glob("*.jpg")) + \
                       list(class_dir.glob("*.jpeg")) + \
                       list(class_dir.glob("*.png"))
            else:  # landmarks
                files = list(class_dir.glob("*.npy"))
            
            counts[class_name] = len(files)
        
        return counts
    
    def combine_datasets(self, copy_files: bool = True):
        """
        Combine all sources into unified dataset
        
        Args:
            copy_files: If True, copy files; if False, create symlinks
        """
        logger.info("\n" + "="*80)
        logger.info(" "*25 + "COMBINING DATASETS")
        logger.info("="*80)
        
        if not self.sources:
            logger.error("No sources added")
            return
        
        # Process each source
        for source in self.sources:
            source_name = source['name']
            source_path = source['path']
            
            logger.info(f"\nProcessing source: {source_name}")
            
            # Count samples
            source_counts = self.count_samples_in_source(source)
            self.stats['sources'][source_name] = source_counts
            
            # Copy/link files
            for class_name, count in source_counts.items():
                if count == 0:
                    continue
                
                # Create target directory
                target_dir = self.output_dir / class_name
                target_dir.mkdir(exist_ok=True)
                
                source_class_dir = source_path / class_name
                
                # Get files
                if source['type'] == 'images':
                    files = list(source_class_dir.glob("*.jpg")) + \
                           list(source_class_dir.glob("*.jpeg")) + \
                           list(source_class_dir.glob("*.png"))
                else:
                    files = list(source_class_dir.glob("*.npy"))
                
                # Copy files with progress bar
                desc = f"  {class_name} from {source_name}"
                for file_path in tqdm(files, desc=desc, leave=False):
                    # Create unique filename
                    new_name = f"{source_name}_{file_path.name}"
                    target_path = target_dir / new_name
                    
                    if copy_files:
                        shutil.copy2(file_path, target_path)
                    else:
                        # Create symlink (Windows may require admin)
                        try:
                            if not target_path.exists():
                                target_path.symlink_to(file_path)
                        except:
                            shutil.copy2(file_path, target_path)
                    
                    self.stats['classes'][class_name] += 1
                    self.stats['total_samples'] += 1
        
        # Save metadata
        self.save_metadata()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print combination summary"""
        logger.info("\n" + "="*80)
        logger.info("COMBINATION SUMMARY")
        logger.info("="*80)
        
        # Source statistics
        logger.info("\nSamples per source:")
        for source_name, counts in self.stats['sources'].items():
            total = sum(counts.values())
            logger.info(f"  {source_name:>15}: {total:>6} samples")
        
        # Class statistics
        logger.info("\nSamples per class:")
        for class_name in sorted(self.stats['classes'].keys()):
            count = self.stats['classes'][class_name]
            logger.info(f"  {class_name:>3}: {count:>6} samples")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Total classes: {len(self.stats['classes'])}")
        logger.info(f"Total samples: {self.stats['total_samples']}")
        logger.info(f"Average per class: {self.stats['total_samples'] / len(self.stats['classes']):.1f}")
        logger.info(f"Combined dataset: {self.output_dir}")
        logger.info("="*80)
    
    def save_metadata(self):
        """Save combination metadata"""
        metadata = {
            'combination_timestamp': datetime.now().isoformat(),
            'output_dir': str(self.output_dir),
            'sources': [
                {
                    'name': s['name'],
                    'path': str(s['path']),
                    'type': s['type']
                }
                for s in self.sources
            ],
            'statistics': {
                'total_samples': self.stats['total_samples'],
                'num_classes': len(self.stats['classes']),
                'classes': dict(self.stats['classes']),
                'sources': {k: dict(v) for k, v in self.stats['sources'].items()}
            }
        }
        
        metadata_path = self.output_dir / "combination_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to: {metadata_path}")


class DatasetAnalyzer:
    """Comprehensive dataset analysis"""
    
    def __init__(self, dataset_path: str):
        """
        Initialize analyzer
        
        Args:
            dataset_path: Path to dataset
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")
        
        self.stats = {}
        self.class_distribution = {}
    
    def analyze_class_balance(self) -> pd.DataFrame:
        """
        Analyze class balance
        
        Returns:
            DataFrame with class statistics
        """
        logger.info("\nAnalyzing class balance...")
        
        class_stats = []
        
        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Count files
            files = list(class_dir.glob("*.jpg")) + \
                   list(class_dir.glob("*.png")) + \
                   list(class_dir.glob("*.npy"))
            
            count = len(files)
            
            class_stats.append({
                'class': class_name,
                'count': count
            })
            
            self.class_distribution[class_name] = count
        
        df = pd.DataFrame(class_stats)
        
        if not df.empty:
            df['percentage'] = (df['count'] / df['count'].sum()) * 100
            df = df.sort_values('class')
            
            logger.info(f"\nClass Balance:")
            logger.info(f"  Total classes: {len(df)}")
            logger.info(f"  Total samples: {df['count'].sum()}")
            logger.info(f"  Mean: {df['count'].mean():.1f}")
            logger.info(f"  Std: {df['count'].std():.1f}")
            logger.info(f"  Min: {df['count'].min()} ({df.loc[df['count'].idxmin(), 'class']})")
            logger.info(f"  Max: {df['count'].max()} ({df.loc[df['count'].idxmax(), 'class']})")
        
        return df
    
    def plot_class_distribution(self, output_path: str = None):
        """
        Plot class distribution
        
        Args:
            output_path: Where to save plot
        """
        if not self.class_distribution:
            self.analyze_class_balance()
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Bar chart
        classes = sorted(self.class_distribution.keys())
        counts = [self.class_distribution[c] for c in classes]
        
        ax1 = axes[0]
        bars = ax1.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
        
        # Color by deviation
        mean_count = np.mean(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            if count < mean_count * 0.8:
                bar.set_color('orangered')
            elif count > mean_count * 1.2:
                bar.set_color('forestgreen')
        
        ax1.axhline(y=mean_count, color='red', linestyle='--', 
                   label=f'Mean: {mean_count:.0f}', linewidth=2)
        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        ax2.boxplot([counts], vert=False, widths=0.7)
        ax2.set_xlabel('Sample Count', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution Statistics', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = self.dataset_path / "analysis_class_distribution.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Class distribution plot saved to: {output_path}")
        plt.close()
    
    def analyze_data_quality(self, sample_size: int = 100):
        """
        Analyze data quality metrics
        
        Args:
            sample_size: Number of samples to analyze per class
        """
        logger.info("\nAnalyzing data quality...")
        
        quality_stats = {
            'corrupted_files': 0,
            'empty_files': 0,
            'invalid_landmarks': 0,
            'valid_samples': 0
        }
        
        # Sample files
        all_files = []
        for class_dir in self.dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            files = list(class_dir.glob("*.jpg")) + \
                   list(class_dir.glob("*.png")) + \
                   list(class_dir.glob("*.npy"))
            
            if len(files) > sample_size:
                import random
                files = random.sample(files, sample_size)
            
            all_files.extend(files)
        
        # Check each file
        for file_path in tqdm(all_files, desc="Checking quality"):
            try:
                if file_path.suffix in ['.jpg', '.png']:
                    # Check image
                    img = cv2.imread(str(file_path))
                    if img is None:
                        quality_stats['corrupted_files'] += 1
                    elif img.size == 0:
                        quality_stats['empty_files'] += 1
                    else:
                        quality_stats['valid_samples'] += 1
                
                elif file_path.suffix == '.npy':
                    # Check landmarks
                    data = np.load(file_path)
                    if data.shape[0] != 63:  # 21 landmarks × 3 coords
                        quality_stats['invalid_landmarks'] += 1
                    else:
                        quality_stats['valid_samples'] += 1
            
            except Exception as e:
                quality_stats['corrupted_files'] += 1
        
        # Report
        logger.info("\nQuality Analysis:")
        logger.info(f"  Valid samples: {quality_stats['valid_samples']}")
        logger.info(f"  Corrupted files: {quality_stats['corrupted_files']}")
        logger.info(f"  Empty files: {quality_stats['empty_files']}")
        logger.info(f"  Invalid landmarks: {quality_stats['invalid_landmarks']}")
        
        total = sum(quality_stats.values())
        if total > 0:
            valid_pct = (quality_stats['valid_samples'] / total) * 100
            logger.info(f"  Quality score: {valid_pct:.2f}%")
        
        return quality_stats
    
    def generate_report(self, output_path: str = None):
        """
        Generate comprehensive analysis report
        
        Args:
            output_path: Where to save report
        """
        logger.info("\n" + "="*80)
        logger.info(" "*25 + "DATASET ANALYSIS")
        logger.info("="*80)
        
        # Run analyses
        df_balance = self.analyze_class_balance()
        self.plot_class_distribution()
        quality_stats = self.analyze_data_quality()
        
        # Create report
        if output_path is None:
            output_path = self.dataset_path / "analysis_report.txt"
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" "*25 + "ISL DATASET ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.dataset_path}\n\n")
            
            # Class balance
            f.write("-"*80 + "\n")
            f.write("CLASS BALANCE\n")
            f.write("-"*80 + "\n")
            f.write(df_balance.to_string(index=False))
            f.write("\n\n")
            
            # Quality
            f.write("-"*80 + "\n")
            f.write("DATA QUALITY\n")
            f.write("-"*80 + "\n")
            for key, value in quality_stats.items():
                f.write(f"{key:>20}: {value}\n")
            f.write("\n")
        
        logger.info(f"✓ Analysis report saved to: {output_path}")
        logger.info("="*80)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine and analyze ISL datasets')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine datasets')
    combine_parser.add_argument('--output', type=str, required=True,
                               help='Output directory')
    combine_parser.add_argument('--sources', nargs='+', required=True,
                               help='Source paths (format: name:path:type)')
    combine_parser.add_argument('--copy', action='store_true',
                               help='Copy files instead of symlinking')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.add_argument('--dataset', type=str, required=True,
                               help='Dataset path')
    analyze_parser.add_argument('--output', type=str,
                               help='Output directory for reports')
    
    args = parser.parse_args()
    
    if args.command == 'combine':
        # Combine datasets
        combiner = DatasetCombiner(output_dir=args.output)
        
        for source_spec in args.sources:
            parts = source_spec.split(':')
            if len(parts) == 3:
                name, path, dtype = parts
                combiner.add_source(name, path, dtype)
            else:
                logger.warning(f"Invalid source spec: {source_spec}")
        
        combiner.combine_datasets(copy_files=args.copy)
    
    elif args.command == 'analyze':
        # Analyze dataset
        analyzer = DatasetAnalyzer(dataset_path=args.dataset)
        
        output_path = args.output if args.output else args.dataset
        analyzer.generate_report(output_path)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
