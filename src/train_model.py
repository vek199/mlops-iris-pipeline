#!/usr/bin/env python3
"""
Enhanced Model Training Script with MLflow Integration

This script trains machine learning models on the IRIS dataset with support for:
- Multiple algorithms (RandomForest, SVM, Logistic Regression)
- MLflow experiment tracking
- Poisoned data handling
- Comprehensive logging and metrics

Author: MLOps Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A comprehensive model training class with MLflow integration.
    """
    
    def __init__(self, experiment_name: str = "iris_poisoning_experiment"):
        """
        Initialize the ModelTrainer.
        
        Args:
            experiment_name (str): Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
    
    def load_and_prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare the dataset for training.
        
        Args:
            data_path (str): Path to the CSV file
            test_size (float): Proportion of data for testing
            
        Returns:
            Tuple of X_train, X_test, y_train, y_test, y_original
        """
        logger.info(f"Loading data from: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Separate features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Store original labels for analysis
        y_original = y.copy()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Log class distribution
        class_counts = pd.Series(y).value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, y_original
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray, data_path: str) -> Dict[str, Any]:
        """
        Train a single model and return metrics.
        
        Args:
            model_name (str): Name of the model to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            data_path (str): Path to original data file
            
        Returns:
            Dict with model metrics and artifacts
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        
        # Start MLflow run if available
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=f"{model_name}_{Path(data_path).stem}"):
                return self._train_with_mlflow(model, model_name, X_train, y_train, X_test, y_test, data_path)
        else:
            return self._train_without_mlflow(model, model_name, X_train, y_train, X_test, y_test, data_path)
    
    def _train_with_mlflow(self, model, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, data_path: str) -> Dict[str, Any]:
        """Train model with MLflow tracking."""
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("data_source", data_path)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))
        
        # Log model hyperparameters
        for param, value in model.get_params().items():
            mlflow.log_param(f"model_{param}", value)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, value)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
        
        # Log model
        mlflow.sklearn.log_model(model, f"model_{model_name}")
        
        # Save artifacts locally
        artifacts = self._save_artifacts(model, model_name, metrics, data_path)
        
        return {"metrics": metrics, "artifacts": artifacts, "model": model}
    
    def _train_without_mlflow(self, model, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray, data_path: str) -> Dict[str, Any]:
        """Train model without MLflow tracking."""
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()
        
        # Save artifacts locally
        artifacts = self._save_artifacts(model, model_name, metrics, data_path)
        
        return {"metrics": metrics, "artifacts": artifacts, "model": model}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average='macro'),
            "recall_macro": recall_score(y_true, y_pred, average='macro'),
            "f1_macro": f1_score(y_true, y_pred, average='macro'),
            "precision_micro": precision_score(y_true, y_pred, average='micro'),
            "recall_micro": recall_score(y_true, y_pred, average='micro'),
            "f1_micro": f1_score(y_true, y_pred, average='micro'),
        }
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics["classification_report"] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        return metrics
    
    def _save_artifacts(self, model, model_name: str, metrics: Dict, data_path: str) -> Dict[str, str]:
        """Save model artifacts locally."""
        # Create artifacts directory
        artifacts_dir = Path(f"artifacts/{model_name}_{Path(data_path).stem}")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = artifacts_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Save label encoder
        encoder_path = artifacts_dir / "label_encoder.joblib"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save scaler
        scaler_path = artifacts_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save metrics
        metrics_path = artifacts_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_serializable = self._make_serializable(metrics)
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Artifacts saved to: {artifacts_dir}")
        
        return {
            "model_path": str(model_path),
            "encoder_path": str(encoder_path),
            "scaler_path": str(scaler_path),
            "metrics_path": str(metrics_path)
        }
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def train_all_models(self, data_path: str) -> Dict[str, Any]:
        """
        Train all available models on the dataset.
        
        Args:
            data_path (str): Path to the CSV file
            
        Returns:
            Dict with results for all models
        """
        logger.info("Starting training for all models...")
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, y_original = self.load_and_prepare_data(data_path)
        
        results = {}
        
        # Train each model
        for model_name in self.models.keys():
            try:
                result = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test, data_path
                )
                results[model_name] = result
                
                # Log summary
                accuracy = result["metrics"]["accuracy"]
                logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        # Compare models
        self._compare_models(results)
        
        return results
    
    def _compare_models(self, results: Dict[str, Any]) -> None:
        """Compare model performance and log summary."""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        comparison_data = []
        
        for model_name, result in results.items():
            if "error" not in result:
                metrics = result["metrics"]
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "F1 (Macro)": f"{metrics['f1_macro']:.4f}",
                    "Precision (Macro)": f"{metrics['precision_macro']:.4f}",
                    "Recall (Macro)": f"{metrics['recall_macro']:.4f}",
                    "CV Mean": f"{metrics.get('cv_mean', 0):.4f}"
                })
        
        if comparison_data:
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            logger.info(f"\n{comparison_df.to_string(index=False)}")
            
            # Find best model
            best_model = max(comparison_data, key=lambda x: float(x["Accuracy"]))
            logger.info(f"\nBest Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']})")
        
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Train models on IRIS dataset with poisoning experiment support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models on clean data
  python src/train_model.py --data-path data/iris.csv
  
  # Train specific model on poisoned data
  python src/train_model.py --data-path data/iris_poisoned_10pct.csv --model random_forest
  
  # Train with custom experiment name
  python src/train_model.py --data-path data/iris.csv --experiment-name "clean_data_baseline"
        """
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/iris.csv",
        help="Path to the CSV data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["random_forest", "svm", "logistic_regression", "all"],
        default="all",
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="iris_poisoning_experiment",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(experiment_name=args.experiment_name)
    
    # Train models
    if args.model == "all":
        results = trainer.train_all_models(args.data_path)
    else:
        # Load and prepare data
        X_train, X_test, y_train, y_test, _ = trainer.load_and_prepare_data(args.data_path, args.test_size)
        
        # Train single model
        result = trainer.train_single_model(
            args.model, X_train, y_train, X_test, y_test, args.data_path
        )
        results = {args.model: result}
    
    logger.info("Training completed successfully!")
    
    if MLFLOW_AVAILABLE:
        logger.info("Check MLflow UI for detailed experiment tracking")
        logger.info("Run: mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()