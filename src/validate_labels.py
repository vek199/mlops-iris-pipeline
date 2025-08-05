#!/usr/bin/env python3
"""
Label Validation Script for Detecting Data Poisoning

This script implements KNN-based label validation to detect potentially
flipped or mislabeled instances in datasets. It's designed to identify
data poisoning attacks where labels have been maliciously altered.

Author: MLOps Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LabelValidator:
    """
    A class for validating labels using K-Nearest Neighbors approach.
    
    This validator identifies potentially mislabeled instances by comparing
    each point's label with the labels of its k-nearest neighbors.
    """
    
    def __init__(self, k: int = 5, threshold: float = 0.5, normalize_features: bool = True):
        """
        Initialize the Label Validator.
        
        Args:
            k (int): Number of nearest neighbors to consider
            threshold (float): Fraction of neighbors that must disagree to flag a point
            normalize_features (bool): Whether to normalize features before KNN
        """
        self.k = k
        self.threshold = threshold
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.label_encoder = LabelEncoder()
        
    def find_suspicious_labels(self, data_path: str, target_column: str = None) -> Tuple[List[int], Dict]:
        """
        Analyze a dataset to find rows with potentially flipped labels using KNN.
        
        A row is considered suspicious if its label disagrees with a certain threshold
        of its k-nearest neighbors in the feature space.
        
        Args:
            data_path (str): Path to the CSV data file
            target_column (str): Name of target column (if None, uses last column)
            
        Returns:
            Tuple[List[int], Dict]: List of suspicious indices and analysis report
        """
        logger.info(f"Analyzing labels in: {data_path}")
        logger.info(f"Using k={self.k}, threshold={self.threshold}")
        
        # Load the data
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Determine target column
        if target_column is None:
            target_column = df.columns[-1]
        
        logger.info(f"Target column: {target_column}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode labels if they are strings
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Normalize features if requested
        if self.normalize_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Fit KNN classifier (k+1 because closest neighbor is the point itself)
        knn = KNeighborsClassifier(n_neighbors=self.k + 1)
        knn.fit(X_scaled, y_encoded)
        
        # Find neighbors for all points
        distances, indices = knn.kneighbors(X_scaled)
        
        suspicious_indices = []
        detailed_report = []
        
        # Analyze each data point
        for i in range(len(df)):
            original_label = y_encoded[i]
            original_label_name = y.iloc[i]
            
            # Get labels of k nearest neighbors (excluding self at index 0)
            neighbor_indices = indices[i][1:]
            neighbor_labels = y_encoded[neighbor_indices]
            neighbor_distances = distances[i][1:]
            
            # Count disagreements
            num_disagreements = np.sum(neighbor_labels != original_label)
            disagreement_ratio = num_disagreements / self.k
            
            # Check if point is suspicious
            if disagreement_ratio >= self.threshold:
                suspicious_indices.append(i)
                
                # Get neighbor label names for reporting
                neighbor_label_names = [self.label_encoder.inverse_transform([label])[0] 
                                      for label in neighbor_labels]
                
                report_entry = {
                    "index": i,
                    "original_label": original_label_name,
                    "disagreements": num_disagreements,
                    "disagreement_ratio": disagreement_ratio,
                    "neighbor_labels": neighbor_label_names,
                    "neighbor_distances": neighbor_distances.tolist(),
                    "confidence_score": 1 - disagreement_ratio  # Lower is more suspicious
                }
                detailed_report.append(report_entry)
                
                logger.debug(f"Row {i}: Label '{original_label_name}', "
                           f"{num_disagreements}/{self.k} neighbors disagree")
        
        # Generate summary report
        summary_report = {
            "dataset_path": data_path,
            "total_rows": len(df),
            "suspicious_rows": len(suspicious_indices),
            "suspicion_rate": len(suspicious_indices) / len(df),
            "validation_parameters": {
                "k": self.k,
                "threshold": self.threshold,
                "normalize_features": self.normalize_features
            },
            "class_distribution": dict(y.value_counts()),
            "suspicious_indices": suspicious_indices,
            "detailed_analysis": detailed_report
        }
        
        # Log summary
        logger.info(f"Analysis complete:")
        logger.info(f"  Total rows: {len(df)}")
        logger.info(f"  Suspicious rows: {len(suspicious_indices)}")
        logger.info(f"  Suspicion rate: {len(suspicious_indices)/len(df):.2%}")
        
        if suspicious_indices:
            logger.info(f"  Suspicious row indices: {suspicious_indices}")
            
            # Show class-wise suspicion breakdown
            suspicious_labels = [y.iloc[i] for i in suspicious_indices]
            suspicion_by_class = pd.Series(suspicious_labels).value_counts()
            logger.info("  Suspicion by class:")
            for class_name, count in suspicion_by_class.items():
                total_class = (y == class_name).sum()
                logger.info(f"    {class_name}: {count}/{total_class} ({count/total_class:.1%})")
        
        return suspicious_indices, summary_report
    
    def validate_with_confidence_threshold(self, data_path: str, confidence_model_path: str = None) -> Dict:
        """
        Alternative validation method using model confidence.
        
        Train a model on the data and identify points where the model's
        prediction strongly disagrees with the given label.
        
        Args:
            data_path (str): Path to dataset
            confidence_model_path (str): Path to save the confidence model
            
        Returns:
            Dict: Validation report with confidence-based suspicious points
        """
        logger.info("Performing confidence-based validation...")
        
        # Load data
        df = pd.read_csv(data_path)
        target_column = df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Normalize features
        if self.normalize_features:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Train a classifier for confidence estimation
        from sklearn.ensemble import RandomForestClassifier
        confidence_model = RandomForestClassifier(n_estimators=100, random_state=42)
        confidence_model.fit(X_scaled, y_encoded)
        
        # Get prediction probabilities
        pred_probs = confidence_model.predict_proba(X_scaled)
        predictions = confidence_model.predict(X_scaled)
        
        # Find points where model strongly disagrees with given label
        suspicious_confidence = []
        
        for i in range(len(df)):
            true_label = y_encoded[i]
            pred_label = predictions[i]
            max_prob = np.max(pred_probs[i])
            true_label_prob = pred_probs[i][true_label]
            
            # Flag if model predicts different class with high confidence
            if pred_label != true_label and max_prob > 0.8:
                suspicious_confidence.append({
                    "index": i,
                    "true_label": self.label_encoder.inverse_transform([true_label])[0],
                    "predicted_label": self.label_encoder.inverse_transform([pred_label])[0],
                    "prediction_confidence": max_prob,
                    "true_label_confidence": true_label_prob
                })
        
        logger.info(f"Confidence-based validation found {len(suspicious_confidence)} suspicious points")
        
        return {
            "suspicious_points": suspicious_confidence,
            "model_accuracy": (predictions == y_encoded).mean()
        }


def save_validation_report(report: Dict, output_path: str) -> None:
    """Save validation report to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Validation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset labels to detect potential poisoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python src/validate_labels.py --data-path data/iris.csv
  
  # Custom parameters
  python src/validate_labels.py --data-path data/iris_poisoned.csv --k 7 --threshold 0.6
  
  # Save detailed report
  python src/validate_labels.py --data-path data/iris_poisoned_10pct.csv --output-report validation_report.json
        """
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV file to validate"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to consider"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fraction of neighbors that must disagree to flag a point"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        help="Name of target column (default: last column)"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        help="Path to save detailed validation report (JSON)"
    )
    parser.add_argument(
        "--confidence-validation",
        action="store_true",
        help="Also perform confidence-based validation"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize features before KNN"
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = LabelValidator(
        k=args.k,
        threshold=args.threshold,
        normalize_features=not args.no_normalize
    )
    
    # Perform KNN-based validation
    suspicious_indices, report = validator.find_suspicious_labels(
        args.data_path, 
        args.target_column
    )
    
    # Perform confidence-based validation if requested
    if args.confidence_validation:
        confidence_report = validator.validate_with_confidence_threshold(args.data_path)
        report["confidence_validation"] = confidence_report
    
    # Save report if requested
    if args.output_report:
        save_validation_report(report, args.output_report)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"LABEL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {args.data_path}")
    print(f"Total rows: {report['total_rows']}")
    print(f"Suspicious rows: {report['suspicious_rows']}")
    print(f"Suspicion rate: {report['suspicion_rate']:.2%}")
    
    if suspicious_indices:
        print(f"\nSuspicious row indices: {suspicious_indices}")
    else:
        print(f"\n✅ No suspicious labels detected with current parameters")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()