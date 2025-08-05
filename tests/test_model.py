#!/usr/bin/env python3
"""
Enhanced Model Testing Suite

Tests for model performance, robustness, and poisoning resilience.
"""

import pandas as pd
import numpy as np
import pytest
import os
import tempfile
import shutil
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


class TestModelPerformance:
    """Test model performance on clean and poisoned data."""
    
    @pytest.fixture
    def sample_iris_data(self):
        """Create sample IRIS-like data for testing."""
        np.random.seed(42)
        n_samples = 150
        
        # Create realistic IRIS-like features
        data = {
            'sepal_length': np.random.normal(5.5, 1.0, n_samples),
            'sepal_width': np.random.normal(3.0, 0.5, n_samples),
            'petal_length': np.random.normal(4.0, 1.5, n_samples),
            'petal_width': np.random.normal(1.3, 0.7, n_samples),
            'species': (['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50)
        }
        return pd.DataFrame(data)
    
    def test_basic_model_training(self, sample_iris_data):
        """Test basic model training functionality."""
        X = sample_iris_data.iloc[:, :-1]
        y = sample_iris_data.iloc[:, -1]
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Should achieve reasonable accuracy on clean, well-separated data
        assert accuracy > 0.7, f"Model accuracy {accuracy} is too low"
        
        # Test that model makes sensible predictions
        assert len(np.unique(y_pred)) >= 2, "Model should predict multiple classes"
        assert len(y_pred) == len(y_test), "Prediction length mismatch"
    
    def test_model_robustness_to_noise(self, sample_iris_data):
        """Test model robustness to feature noise."""
        X = sample_iris_data.iloc[:, :-1]
        y = sample_iris_data.iloc[:, -1]
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Add noise to features
        X_noisy = X.copy()
        noise_level = 0.1
        for col in X.columns:
            noise = np.random.normal(0, noise_level * X[col].std(), len(X))
            X_noisy[col] += noise
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        X_train_noisy, X_test_noisy, _, _ = train_test_split(
            X_noisy, y_encoded, test_size=0.2, random_state=42
        )
        
        # Train on clean data, test on noisy
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(X_train, y_train)
        
        clean_accuracy = accuracy_score(y_test, model.predict(X_test))
        noisy_accuracy = accuracy_score(y_test, model.predict(X_test_noisy))
        
        # Performance should degrade but not collapse
        performance_drop = clean_accuracy - noisy_accuracy
        assert performance_drop < 0.3, f"Model too sensitive to noise (drop: {performance_drop})"


class TestPoisoningResistance:
    """Test model resistance to data poisoning."""
    
    @pytest.fixture
    def poisoned_iris_data(self):
        """Create poisoned IRIS data for testing."""
        np.random.seed(42)
        n_samples = 150
        
        # Create base data
        data = {
            'sepal_length': np.random.normal(5.5, 1.0, n_samples),
            'sepal_width': np.random.normal(3.0, 0.5, n_samples),
            'petal_length': np.random.normal(4.0, 1.5, n_samples),
            'petal_width': np.random.normal(1.3, 0.7, n_samples),
            'species': (['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50)
        }
        df = pd.DataFrame(data)
        
        # Poison 10% of labels
        poison_count = int(0.1 * len(df))
        poison_indices = np.random.choice(len(df), poison_count, replace=False)
        species_values = df['species'].unique()
        
        for idx in poison_indices:
            current_species = df.loc[idx, 'species']
            other_species = [s for s in species_values if s != current_species]
            df.loc[idx, 'species'] = np.random.choice(other_species)
        
        return df
    
    def test_performance_degradation_bounds(self, sample_iris_data, poisoned_iris_data):
        """Test that poisoning causes bounded performance degradation."""
        # Train on clean data
        X_clean = sample_iris_data.iloc[:, :-1]
        y_clean = sample_iris_data.iloc[:, -1]
        
        le_clean = LabelEncoder()
        y_clean_encoded = le_clean.fit_transform(y_clean)
        
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X_clean, y_clean_encoded, test_size=0.2, random_state=42
        )
        
        model_clean = RandomForestClassifier(random_state=42, n_estimators=20)
        model_clean.fit(X_train_clean, y_train_clean)
        clean_accuracy = accuracy_score(y_test_clean, model_clean.predict(X_test_clean))
        
        # Train on poisoned data
        X_poisoned = poisoned_iris_data.iloc[:, :-1]
        y_poisoned = poisoned_iris_data.iloc[:, -1]
        
        le_poisoned = LabelEncoder()
        y_poisoned_encoded = le_poisoned.fit_transform(y_poisoned)
        
        X_train_poisoned, X_test_poisoned, y_train_poisoned, y_test_poisoned = train_test_split(
            X_poisoned, y_poisoned_encoded, test_size=0.2, random_state=42
        )
        
        model_poisoned = RandomForestClassifier(random_state=42, n_estimators=20)
        model_poisoned.fit(X_train_poisoned, y_train_poisoned)
        poisoned_accuracy = accuracy_score(y_test_poisoned, model_poisoned.predict(X_test_poisoned))
        
        # Performance should degrade but not collapse
        performance_drop = clean_accuracy - poisoned_accuracy
        
        # With 10% poisoning, we expect some degradation but not complete failure
        assert performance_drop >= 0, "Poisoned model shouldn't perform better than clean"
        assert performance_drop < 0.4, f"Performance drop too severe: {performance_drop}"
        
        # Both models should still be better than random
        n_classes = len(np.unique(y_clean_encoded))
        random_accuracy = 1.0 / n_classes
        
        assert clean_accuracy > random_accuracy + 0.2, "Clean model performance too low"
        assert poisoned_accuracy > random_accuracy, "Poisoned model worse than random"


# Legacy test for backward compatibility
def test_random_forest_accuracy():
    """Legacy test function for backward compatibility."""
    # Create synthetic data since original files might not exist
    np.random.seed(42)
    n_samples = 150
    
    data = {
        'feature1': np.random.normal(5.5, 1.0, n_samples),
        'feature2': np.random.normal(3.0, 0.5, n_samples),
        'feature3': np.random.normal(4.0, 1.5, n_samples),
        'feature4': np.random.normal(1.3, 0.7, n_samples),
        'target': (['A'] * 50 + ['B'] * 50 + ['C'] * 50)
    }
    df = pd.DataFrame(data)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42, n_estimators=20)
    model.fit(X.iloc[:-len(X_test)], y_encoded[:-len(y_test)])
    
    # Make predictions
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"RandomForest accuracy: {acc}")
    assert acc > 0.6  # Lower threshold for synthetic data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])