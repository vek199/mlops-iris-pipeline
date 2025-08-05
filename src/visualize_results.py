#!/usr/bin/env python3
"""
Advanced Visualization Script for Poisoning Analysis

This script creates comprehensive visualizations for:
- Model performance comparison (clean vs poisoned)
- Confusion matrices and classification reports
- Data distribution analysis
- Poisoning attack impact visualization
- Feature space analysis

Author: MLOps Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoisoningVisualizer:
    """
    Advanced visualization class for poisoning analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the PoisoningVisualizer.
        
        Args:
            figsize: Default figure size for plots
            dpi: Resolution for saved plots
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
    
    def plot_performance_comparison(self, comparison_data: Dict[str, Any], 
                                  output_path: str = "performance_comparison.png") -> None:
        """
        Create comprehensive performance comparison plots.
        
        Args:
            comparison_data: Results from model comparison
            output_path: Path to save the plot
        """
        logger.info("Creating performance comparison visualization...")
        
        # Extract performance degradation data
        degradation = comparison_data.get("performance_degradation", {})
        
        if not degradation:
            logger.warning("No performance degradation data found")
            return
        
        # Prepare data for plotting
        metrics = list(degradation.keys())
        clean_scores = [degradation[metric]["clean"] for metric in metrics]
        poisoned_scores = [degradation[metric]["poisoned"] for metric in metrics]
        absolute_drops = [degradation[metric]["absolute_drop"] for metric in metrics]
        relative_drops = [degradation[metric]["relative_drop_pct"] for metric in metrics]
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance: Clean vs Poisoned Data', fontsize=16, fontweight='bold')
        
        # 1. Side-by-side bar comparison
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, clean_scores, width, label='Clean Model', color=self.colors[0], alpha=0.8)
        ax1.bar(x + width/2, poisoned_scores, width, label='Poisoned Model', color=self.colors[1], alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Comparison: Clean vs Poisoned')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (clean, poisoned) in enumerate(zip(clean_scores, poisoned_scores)):
            ax1.text(i - width/2, clean + 0.01, f'{clean:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, poisoned + 0.01, f'{poisoned:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Absolute performance drop
        bars2 = ax2.bar(metrics, absolute_drops, color=self.colors[2], alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Absolute Drop')
        ax2.set_title('Absolute Performance Drop')
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, drop in zip(bars2, absolute_drops):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{drop:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Relative performance drop (percentage)
        bars3 = ax3.bar(metrics, relative_drops, color=self.colors[3], alpha=0.7)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Relative Drop (%)')
        ax3.set_title('Relative Performance Drop (%)')
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, drop in zip(bars3, relative_drops):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{drop:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Radar chart for overall comparison
        self._create_radar_chart(ax4, metrics, clean_scores, poisoned_scores)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison plot saved to: {output_path}")
    
    def _create_radar_chart(self, ax, metrics: List[str], clean_scores: List[float], 
                          poisoned_scores: List[float]) -> None:
        """Create a radar chart comparing clean and poisoned model performance."""
        # Number of metrics
        N = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first value at the end to close the radar chart
        clean_scores_plot = clean_scores + [clean_scores[0]]
        poisoned_scores_plot = poisoned_scores + [poisoned_scores[0]]
        
        # Clear the axis for polar plot
        ax.clear()
        
        # Create polar plot
        ax = plt.subplot(2, 2, 4, projection='polar')
        
        # Plot clean model
        ax.plot(angles, clean_scores_plot, 'o-', linewidth=2, label='Clean Model', color=self.colors[0])
        ax.fill(angles, clean_scores_plot, alpha=0.25, color=self.colors[0])
        
        # Plot poisoned model
        ax.plot(angles, poisoned_scores_plot, 'o-', linewidth=2, label='Poisoned Model', color=self.colors[1])
        ax.fill(angles, poisoned_scores_plot, alpha=0.25, color=self.colors[1])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', size=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)
    
    def plot_confusion_matrices(self, clean_artifacts_dir: str, poisoned_artifacts_dir: str,
                              test_data_path: str, output_path: str = "confusion_matrices.png") -> None:
        """
        Plot confusion matrices for clean and poisoned models side by side.
        
        Args:
            clean_artifacts_dir: Directory with clean model artifacts
            poisoned_artifacts_dir: Directory with poisoned model artifacts
            test_data_path: Path to test dataset
            output_path: Path to save the plot
        """
        logger.info("Creating confusion matrix comparison...")
        
        # Load models and data
        clean_model, le_clean, scaler_clean = self._load_model_artifacts(clean_artifacts_dir)
        poisoned_model, le_poisoned, scaler_poisoned = self._load_model_artifacts(poisoned_artifacts_dir)
        
        # Load test data
        df = pd.read_csv(test_data_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Make predictions
        y_encoded = le_clean.transform(y)
        X_scaled_clean = scaler_clean.transform(X)
        X_scaled_poisoned = scaler_poisoned.transform(X)
        
        y_pred_clean = clean_model.predict(X_scaled_clean)
        y_pred_poisoned = poisoned_model.predict(X_scaled_poisoned)
        
        # Create confusion matrices
        cm_clean = confusion_matrix(y_encoded, y_pred_clean)
        cm_poisoned = confusion_matrix(y_encoded, y_pred_poisoned)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Clean model confusion matrix
        sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le_clean.classes_, yticklabels=le_clean.classes_, ax=ax1)
        ax1.set_title('Clean Model\nConfusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Poisoned model confusion matrix
        sns.heatmap(cm_poisoned, annot=True, fmt='d', cmap='Reds',
                   xticklabels=le_poisoned.classes_, yticklabels=le_poisoned.classes_, ax=ax2)
        ax2.set_title('Poisoned Model\nConfusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices plot saved to: {output_path}")
    
    def plot_data_distribution_analysis(self, datasets: List[str], 
                                      output_path: str = "data_distribution.png") -> None:
        """
        Analyze and visualize data distribution across clean and poisoned datasets.
        
        Args:
            datasets: List of dataset paths
            output_path: Path to save the plot
        """
        logger.info("Creating data distribution analysis...")
        
        # Load all datasets
        data_info = []
        for dataset_path in datasets:
            df = pd.read_csv(dataset_path)
            dataset_name = Path(dataset_path).stem
            
            # Get class distribution
            target_col = df.columns[-1]
            class_counts = df[target_col].value_counts()
            
            data_info.append({
                'dataset': dataset_name,
                'total_samples': len(df),
                'class_distribution': dict(class_counts),
                'data': df
            })
        
        # Create visualization
        n_datasets = len(data_info)
        fig, axes = plt.subplots(2, n_datasets, figsize=(5*n_datasets, 10))
        if n_datasets == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot class distributions
        for i, info in enumerate(data_info):
            # Class distribution bar plot
            classes = list(info['class_distribution'].keys())
            counts = list(info['class_distribution'].values())
            
            axes[0, i].bar(classes, counts, color=self.colors[i % len(self.colors)], alpha=0.7)
            axes[0, i].set_title(f"{info['dataset']}\nClass Distribution")
            axes[0, i].set_xlabel('Class')
            axes[0, i].set_ylabel('Count')
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # Add count labels on bars
            for j, (cls, count) in enumerate(zip(classes, counts)):
                axes[0, i].text(j, count + count*0.01, str(count), ha='center', va='bottom')
            
            # Feature distribution (use first feature as example)
            df = info['data']
            feature_col = df.columns[0]  # First feature
            target_col = df.columns[-1]  # Target
            
            for class_name in classes:
                class_data = df[df[target_col] == class_name][feature_col]
                axes[1, i].hist(class_data, alpha=0.6, label=class_name, bins=20)
            
            axes[1, i].set_title(f"{info['dataset']}\nFeature Distribution ({feature_col})")
            axes[1, i].set_xlabel(feature_col)
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Data distribution analysis saved to: {output_path}")
    
    def plot_poison_level_impact(self, results_dir: str, poison_levels: List[float],
                                output_path: str = "poison_impact.png") -> None:
        """
        Visualize the impact of different poison levels on model performance.
        
        Args:
            results_dir: Directory containing results for different poison levels
            poison_levels: List of poison levels (e.g., [0.05, 0.10, 0.50])
            output_path: Path to save the plot
        """
        logger.info("Creating poison level impact visualization...")
        
        # Collect performance data for different poison levels
        performance_data = []
        
        for level in poison_levels:
            level_pct = int(level * 100)
            results_file = Path(results_dir) / f"poisoned_{level_pct}pct_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract key metrics
                performance_data.append({
                    'poison_level': level * 100,  # Convert to percentage
                    'accuracy': results.get('accuracy', 0),
                    'f1_macro': results.get('f1_macro', 0),
                    'precision_macro': results.get('precision_macro', 0),
                    'recall_macro': results.get('recall_macro', 0)
                })
        
        if not performance_data:
            logger.warning("No performance data found for poison level analysis")
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(performance_data)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Impact of Poison Level on Model Performance', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        metric_titles = ['Accuracy', 'F1 Score (Macro)', 'Precision (Macro)', 'Recall (Macro)']
        axes_flat = [ax1, ax2, ax3, ax4]
        
        for ax, metric, title in zip(axes_flat, metrics, metric_titles):
            ax.plot(df['poison_level'], df[metric], marker='o', linewidth=2, 
                   markersize=8, color=self.colors[0])
            ax.set_xlabel('Poison Level (%)')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Poison Level')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for i, row in df.iterrows():
                ax.annotate(f'{row[metric]:.3f}', 
                           (row['poison_level'], row[metric]),
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Poison level impact plot saved to: {output_path}")
    
    def plot_feature_space_analysis(self, clean_data_path: str, poisoned_data_path: str,
                                   output_path: str = "feature_space_analysis.png") -> None:
        """
        Visualize feature space to show how poisoning affects data distribution.
        
        Args:
            clean_data_path: Path to clean dataset
            poisoned_data_path: Path to poisoned dataset
            output_path: Path to save the plot
        """
        logger.info("Creating feature space analysis...")
        
        # Load datasets
        clean_df = pd.read_csv(clean_data_path)
        poisoned_df = pd.read_csv(poisoned_data_path)
        
        # Prepare data
        X_clean = clean_df.iloc[:, :-1]
        y_clean = clean_df.iloc[:, -1]
        X_poisoned = poisoned_df.iloc[:, :-1]
        y_poisoned = poisoned_df.iloc[:, -1]
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        X_clean_pca = pca.fit_transform(X_clean)
        X_poisoned_pca = pca.transform(X_poisoned)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Space Analysis: Clean vs Poisoned Data', fontsize=16, fontweight='bold')
        
        # Encode labels for consistent coloring
        le = LabelEncoder()
        all_labels = np.concatenate([y_clean, y_poisoned])
        le.fit(all_labels)
        y_clean_encoded = le.transform(y_clean)
        y_poisoned_encoded = le.transform(y_poisoned)
        
        # 1. Clean data PCA
        scatter1 = ax1.scatter(X_clean_pca[:, 0], X_clean_pca[:, 1], 
                              c=y_clean_encoded, cmap='viridis', alpha=0.7)
        ax1.set_title('Clean Data (PCA)')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # 2. Poisoned data PCA
        scatter2 = ax2.scatter(X_poisoned_pca[:, 0], X_poisoned_pca[:, 1], 
                              c=y_poisoned_encoded, cmap='viridis', alpha=0.7)
        ax2.set_title('Poisoned Data (PCA)')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add colorbar
        plt.colorbar(scatter1, ax=ax1)
        plt.colorbar(scatter2, ax=ax2)
        
        # 3. Overlay comparison
        ax3.scatter(X_clean_pca[:, 0], X_clean_pca[:, 1], 
                   c='blue', alpha=0.5, label='Clean', s=20)
        ax3.scatter(X_poisoned_pca[:, 0], X_poisoned_pca[:, 1], 
                   c='red', alpha=0.5, label='Poisoned', s=20)
        ax3.set_title('Clean vs Poisoned Data Overlay')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax3.legend()
        
        # 4. Feature importance (if using first two features)
        if X_clean.shape[1] >= 2:
            ax4.scatter(X_clean.iloc[:, 0], X_clean.iloc[:, 1], 
                       c=y_clean_encoded, cmap='viridis', alpha=0.7, label='Clean')
            ax4.scatter(X_poisoned.iloc[:, 0], X_poisoned.iloc[:, 1], 
                       c=y_poisoned_encoded, cmap='plasma', alpha=0.7, 
                       marker='x', s=30, label='Poisoned')
            ax4.set_title('Original Feature Space (First 2 Features)')
            ax4.set_xlabel(X_clean.columns[0])
            ax4.set_ylabel(X_clean.columns[1])
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature space analysis saved to: {output_path}")
    
    def _load_model_artifacts(self, artifacts_dir: str):
        """Load model artifacts from directory."""
        artifacts_path = Path(artifacts_dir)
        
        # Find model file
        model_files = list(artifacts_path.glob("*.joblib"))
        model_files = [f for f in model_files if "label_encoder" not in f.name and "scaler" not in f.name]
        
        if not model_files:
            raise FileNotFoundError(f"No model file found in {artifacts_dir}")
        
        model_path = model_files[0]
        model = joblib.load(model_path)
        
        # Load label encoder and scaler
        encoder_path = artifacts_path / "label_encoder.joblib"
        scaler_path = artifacts_path / "scaler.joblib"
        
        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        
        return model, label_encoder, scaler
    
    def create_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                  output_dir: str = "visualizations") -> None:
        """
        Create a comprehensive visual report with all analysis plots.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            output_dir: Directory to save all visualizations
        """
        logger.info("Creating comprehensive visual report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create different visualizations based on available data
        if "performance_comparison" in analysis_results:
            self.plot_performance_comparison(
                analysis_results["performance_comparison"],
                os.path.join(output_dir, "performance_comparison.png")
            )
        
        if "datasets" in analysis_results:
            self.plot_data_distribution_analysis(
                analysis_results["datasets"],
                os.path.join(output_dir, "data_distribution.png")
            )
        
        logger.info(f"Comprehensive visual report saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive visualizations for poisoning analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare clean vs poisoned model performance
  python src/visualize_results.py --comparison-file comparison_results.json --output-dir visualizations
  
  # Analyze data distribution across datasets
  python src/visualize_results.py --data-distribution --datasets data/iris.csv data/iris_poisoned_10pct.csv
  
  # Feature space analysis
  python src/visualize_results.py --feature-analysis --clean-data data/iris.csv --poisoned-data data/iris_poisoned_10pct.csv
        """
    )
    
    parser.add_argument(
        "--comparison-file",
        type=str,
        help="JSON file with model comparison results"
    )
    parser.add_argument(
        "--data-distribution",
        action="store_true",
        help="Create data distribution analysis"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="List of datasets for distribution analysis"
    )
    parser.add_argument(
        "--feature-analysis",
        action="store_true",
        help="Create feature space analysis"
    )
    parser.add_argument(
        "--clean-data",
        type=str,
        help="Path to clean dataset (for feature analysis)"
    )
    parser.add_argument(
        "--poisoned-data",
        type=str,
        help="Path to poisoned dataset (for feature analysis)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    
    visualizer = PoisoningVisualizer()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate requested visualizations
    if args.comparison_file:
        with open(args.comparison_file, 'r') as f:
            comparison_data = json.load(f)
        
        visualizer.plot_performance_comparison(
            comparison_data,
            os.path.join(args.output_dir, "performance_comparison.png")
        )
    
    if args.data_distribution and args.datasets:
        visualizer.plot_data_distribution_analysis(
            args.datasets,
            os.path.join(args.output_dir, "data_distribution.png")
        )
    
    if args.feature_analysis and args.clean_data and args.poisoned_data:
        visualizer.plot_feature_space_analysis(
            args.clean_data,
            args.poisoned_data,
            os.path.join(args.output_dir, "feature_space_analysis.png")
        )
    
    logger.info(f"Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()