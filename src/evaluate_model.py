#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script for Poisoning Analysis

This script provides comprehensive model evaluation with focus on:
- Poisoning attack impact analysis
- Cross-validation on multiple datasets
- Robustness metrics
- Comparison between clean and poisoned models

Author: MLOps Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class with poisoning analysis capabilities.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.label_encoder = None
        self.scaler = None
        self.evaluation_results = {}
    
    def load_model_artifacts(self, artifacts_dir: str) -> Tuple[Any, LabelEncoder, StandardScaler]:
        """
        Load model and preprocessing artifacts.
        
        Args:
            artifacts_dir (str): Directory containing model artifacts
            
        Returns:
            Tuple of (model, label_encoder, scaler)
        """
        artifacts_path = Path(artifacts_dir)
        
        # Find model file
        model_files = list(artifacts_path.glob("*.joblib"))
        model_files = [f for f in model_files if "label_encoder" not in f.name and "scaler" not in f.name]
        
        if not model_files:
            raise FileNotFoundError(f"No model file found in {artifacts_dir}")
        
        model_path = model_files[0]  # Take the first model found
        model = joblib.load(model_path)
        
        # Load label encoder
        encoder_path = artifacts_path / "label_encoder.joblib"
        label_encoder = joblib.load(encoder_path)
        
        # Load scaler
        scaler_path = artifacts_path / "scaler.joblib"
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded model artifacts from: {artifacts_dir}")
        return model, label_encoder, scaler
    
    def evaluate_on_dataset(self, model: Any, data_path: str, 
                           label_encoder: LabelEncoder, scaler: StandardScaler) -> Dict[str, Any]:
        """
        Evaluate model on a specific dataset.
        
        Args:
            model: Trained model
            data_path (str): Path to evaluation dataset
            label_encoder: Fitted label encoder
            scaler: Fitted scaler
            
        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating model on: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Preprocess
        y_encoded = label_encoder.transform(y)
        X_scaled = scaler.transform(X)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(
            y_encoded, y_pred, y_pred_proba, label_encoder.classes_
        )
        
        # Add dataset info
        metrics["dataset_info"] = {
            "path": data_path,
            "samples": len(df),
            "features": X.shape[1],
            "classes": len(label_encoder.classes_)
        }
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray = None, 
                                       class_names: List[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            # Basic metrics
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro'),
            "recall_macro": recall_score(y_true, y_pred, average='macro'),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
            "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
            "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        if class_names:
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)
            
            metrics["per_class_metrics"] = {}
            for i, class_name in enumerate(class_names):
                metrics["per_class_metrics"][class_name] = {
                    "precision": precision_per_class[i],
                    "recall": recall_per_class[i],
                    "f1": f1_per_class[i]
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        if class_names:
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            metrics["classification_report"] = report
        
        # ROC AUC for multiclass (if probabilities available)
        if y_pred_proba is not None and len(np.unique(y_true)) > 2:
            try:
                # Binarize labels for multiclass ROC AUC
                y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
                auc_scores = []
                for i in range(y_true_binarized.shape[1]):
                    auc = roc_auc_score(y_true_binarized[:, i], y_pred_proba[:, i])
                    auc_scores.append(auc)
                metrics["roc_auc_per_class"] = auc_scores
                metrics["roc_auc_macro"] = np.mean(auc_scores)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Prediction confidence analysis
        if y_pred_proba is not None:
            max_probs = np.max(y_pred_proba, axis=1)
            metrics["prediction_confidence"] = {
                "mean": np.mean(max_probs),
                "std": np.std(max_probs),
                "min": np.min(max_probs),
                "max": np.max(max_probs)
            }
            
            # Low confidence predictions (potential uncertainty due to poisoning)
            low_confidence_threshold = 0.6
            low_confidence_mask = max_probs < low_confidence_threshold
            metrics["low_confidence_predictions"] = {
                "count": np.sum(low_confidence_mask),
                "percentage": np.mean(low_confidence_mask) * 100,
                "threshold": low_confidence_threshold
            }
        
        return metrics
    
    def compare_clean_vs_poisoned(self, clean_artifacts_dir: str, poisoned_artifacts_dir: str,
                                 test_data_path: str) -> Dict[str, Any]:
        """
        Compare performance of models trained on clean vs poisoned data.
        
        Args:
            clean_artifacts_dir (str): Directory with clean model artifacts
            poisoned_artifacts_dir (str): Directory with poisoned model artifacts
            test_data_path (str): Path to test dataset
            
        Returns:
            Comparison results
        """
        logger.info("Comparing clean vs poisoned model performance...")
        
        # Load models
        clean_model, le_clean, scaler_clean = self.load_model_artifacts(clean_artifacts_dir)
        poisoned_model, le_poisoned, scaler_poisoned = self.load_model_artifacts(poisoned_artifacts_dir)
        
        # Evaluate both models
        clean_results = self.evaluate_on_dataset(clean_model, test_data_path, le_clean, scaler_clean)
        poisoned_results = self.evaluate_on_dataset(poisoned_model, test_data_path, le_poisoned, scaler_poisoned)
        
        # Calculate performance degradation
        degradation = {}
        for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            clean_val = clean_results[metric]
            poisoned_val = poisoned_results[metric]
            degradation[metric] = {
                "clean": clean_val,
                "poisoned": poisoned_val,
                "absolute_drop": clean_val - poisoned_val,
                "relative_drop_pct": ((clean_val - poisoned_val) / clean_val) * 100 if clean_val > 0 else 0
            }
        
        comparison = {
            "clean_model_results": clean_results,
            "poisoned_model_results": poisoned_results,
            "performance_degradation": degradation,
            "test_dataset": test_data_path
        }
        
        # Log summary
        logger.info("Performance Comparison Summary:")
        for metric, values in degradation.items():
            logger.info(f"  {metric}: {values['clean']:.4f} → {values['poisoned']:.4f} "
                       f"(drop: {values['relative_drop_pct']:.1f}%)")
        
        return comparison
    
    def evaluate_robustness(self, artifacts_dir: str, datasets: List[str]) -> Dict[str, Any]:
        """
        Evaluate model robustness across multiple datasets.
        
        Args:
            artifacts_dir (str): Directory with model artifacts
            datasets (List[str]): List of dataset paths to evaluate on
            
        Returns:
            Robustness evaluation results
        """
        logger.info(f"Evaluating model robustness across {len(datasets)} datasets...")
        
        # Load model
        model, label_encoder, scaler = self.load_model_artifacts(artifacts_dir)
        
        results = {}
        metrics_across_datasets = {
            "accuracy": [],
            "f1_macro": [],
            "precision_macro": [],
            "recall_macro": []
        }
        
        # Evaluate on each dataset
        for dataset_path in datasets:
            try:
                dataset_name = Path(dataset_path).stem
                eval_result = self.evaluate_on_dataset(model, dataset_path, label_encoder, scaler)
                results[dataset_name] = eval_result
                
                # Collect metrics for robustness analysis
                for metric in metrics_across_datasets:
                    metrics_across_datasets[metric].append(eval_result[metric])
                
                logger.info(f"  {dataset_name}: Accuracy = {eval_result['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate on {dataset_path}: {e}")
                results[Path(dataset_path).stem] = {"error": str(e)}
        
        # Calculate robustness metrics
        robustness_metrics = {}
        for metric_name, values in metrics_across_datasets.items():
            if values:  # Only if we have valid results
                robustness_metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                }
        
        return {
            "individual_results": results,
            "robustness_metrics": robustness_metrics,
            "datasets_evaluated": len([r for r in results.values() if "error" not in r])
        }
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate and save comprehensive evaluation report."""
        report = {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "results": results,
            "summary": self._generate_summary(results)
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to: {output_path}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from evaluation results."""
        summary = {
            "evaluation_type": "unknown",
            "key_findings": []
        }
        
        # Detect evaluation type and generate appropriate summary
        if "performance_degradation" in results:
            summary["evaluation_type"] = "clean_vs_poisoned_comparison"
            degradation = results["performance_degradation"]
            
            worst_metric = max(degradation.items(), key=lambda x: x[1]["relative_drop_pct"])
            summary["worst_affected_metric"] = {
                "metric": worst_metric[0],
                "drop_percentage": worst_metric[1]["relative_drop_pct"]
            }
            
            summary["key_findings"].append(
                f"Worst affected metric: {worst_metric[0]} "
                f"(dropped by {worst_metric[1]['relative_drop_pct']:.1f}%)"
            )
        
        elif "robustness_metrics" in results:
            summary["evaluation_type"] = "robustness_analysis"
            robustness = results["robustness_metrics"]
            
            if "accuracy" in robustness:
                acc_cv = robustness["accuracy"]["coefficient_of_variation"]
                summary["accuracy_stability"] = {
                    "coefficient_of_variation": acc_cv,
                    "stability_level": "high" if acc_cv < 0.1 else "medium" if acc_cv < 0.2 else "low"
                }
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation for poisoning analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model on test data
  python src/evaluate_model.py --artifacts-dir artifacts/random_forest_iris --test-data data/iris.csv
  
  # Compare clean vs poisoned models
  python src/evaluate_model.py --compare-models --clean-artifacts artifacts/clean_model --poisoned-artifacts artifacts/poisoned_model --test-data data/iris.csv
  
  # Robustness evaluation across multiple datasets
  python src/evaluate_model.py --robustness-eval --artifacts-dir artifacts/model --datasets data/iris.csv data/iris_poisoned_5pct.csv data/iris_poisoned_10pct.csv
        """
    )
    
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        help="Directory containing model artifacts"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare clean vs poisoned models"
    )
    parser.add_argument(
        "--clean-artifacts",
        type=str,
        help="Directory with clean model artifacts (for comparison)"
    )
    parser.add_argument(
        "--poisoned-artifacts",
        type=str,
        help="Directory with poisoned model artifacts (for comparison)"
    )
    parser.add_argument(
        "--robustness-eval",
        action="store_true",
        help="Perform robustness evaluation"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="List of datasets for robustness evaluation"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="evaluation_report.json",
        help="Path to save evaluation report"
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    if args.compare_models:
        if not all([args.clean_artifacts, args.poisoned_artifacts, args.test_data]):
            parser.error("--compare-models requires --clean-artifacts, --poisoned-artifacts, and --test-data")
        
        results = evaluator.compare_clean_vs_poisoned(
            args.clean_artifacts, args.poisoned_artifacts, args.test_data
        )
        
    elif args.robustness_eval:
        if not all([args.artifacts_dir, args.datasets]):
            parser.error("--robustness-eval requires --artifacts-dir and --datasets")
        
        results = evaluator.evaluate_robustness(args.artifacts_dir, args.datasets)
        
    else:
        if not all([args.artifacts_dir, args.test_data]):
            parser.error("Standard evaluation requires --artifacts-dir and --test-data")
        
        model, label_encoder, scaler = evaluator.load_model_artifacts(args.artifacts_dir)
        results = evaluator.evaluate_on_dataset(model, args.test_data, label_encoder, scaler)
    
    # Generate and save report
    evaluator.generate_evaluation_report(results, args.output_report)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    if "accuracy" in results:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    elif "performance_degradation" in results:
        print("Performance Degradation Analysis:")
        for metric, values in results["performance_degradation"].items():
            print(f"  {metric}: {values['relative_drop_pct']:.1f}% drop")
    elif "robustness_metrics" in results:
        print("Robustness Analysis:")
        acc_metrics = results["robustness_metrics"].get("accuracy", {})
        if acc_metrics:
            print(f"  Accuracy CV: {acc_metrics.get('coefficient_of_variation', 0):.4f}")
    
    print(f"Report saved to: {args.output_report}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()