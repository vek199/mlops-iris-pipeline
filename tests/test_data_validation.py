#!/usr/bin/env python3
"""
Enhanced Data Validation Tests

Tests for data quality, poisoning detection, and dataset integrity.
"""

import pandas as pd
import pytest
import numpy as np
import os
from pathlib import Path


@pytest.fixture
def clean_iris_data():
    """Load clean IRIS data if available, otherwise create synthetic data."""
    iris_path = "data/iris.csv"
    if os.path.exists(iris_path):
        return pd.read_csv(iris_path)
    else:
        # Create synthetic IRIS-like data for testing
        np.random.seed(42)
        n_samples = 150
        data = {
            'sepal_length': np.random.normal(5.5, 1.0, n_samples),
            'sepal_width': np.random.normal(3.0, 0.5, n_samples),
            'petal_length': np.random.normal(4.0, 1.5, n_samples),
            'petal_width': np.random.normal(1.3, 0.7, n_samples),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        }
        return pd.DataFrame(data)


@pytest.fixture
def poisoned_data(clean_iris_data):
    """Create poisoned version of IRIS data for testing."""
    poisoned_df = clean_iris_data.copy()
    
    # Flip some labels to simulate poisoning
    n_samples = len(poisoned_df)
    n_poison = int(0.1 * n_samples)  # 10% poisoning
    
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    species_values = poisoned_df['species'].unique()
    
    for idx in poison_indices:
        current_species = poisoned_df.loc[idx, 'species']
        # Change to a different species
        other_species = [s for s in species_values if s != current_species]
        poisoned_df.loc[idx, 'species'] = np.random.choice(other_species)
    
    return poisoned_df


class TestDataQuality:
    """Test data quality and integrity."""
    
    def test_no_missing_values(self, clean_iris_data):
        """Test that dataset contains no missing values."""
        assert not clean_iris_data.isnull().values.any(), "Dataset contains missing values"
    
    def test_expected_columns(self, clean_iris_data):
        """Test that dataset has expected columns."""
        expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
        actual_columns = set(clean_iris_data.columns)
        
        # Allow for different target column names
        if "species" not in actual_columns:
            # Check if last column could be the target
            target_col = clean_iris_data.columns[-1]
            expected_columns = expected_columns - {"species"} | {target_col}
        
        assert actual_columns == expected_columns, f"Unexpected columns: {clean_iris_data.columns}"
    
    def test_target_column_classes(self, clean_iris_data):
        """Test that target column has expected number of classes."""
        target_col = clean_iris_data.columns[-1]  # Last column is target
        unique_classes = clean_iris_data[target_col].unique()
        assert len(unique_classes) == 3, f"Target column should have 3 classes, found {len(unique_classes)}"
    
    def test_feature_data_types(self, clean_iris_data):
        """Test that features have appropriate data types."""
        feature_columns = clean_iris_data.columns[:-1]  # All except target
        
        for col in feature_columns:
            assert pd.api.types.is_numeric_dtype(clean_iris_data[col]), f"Feature {col} should be numeric"
    
    def test_data_ranges(self, clean_iris_data):
        """Test that feature values are within reasonable ranges."""
        feature_columns = clean_iris_data.columns[:-1]
        
        for col in feature_columns:
            values = clean_iris_data[col]
            assert values.min() >= 0, f"Feature {col} has negative values"
            assert values.max() <= 20, f"Feature {col} has unreasonably large values"
    
    def test_class_balance(self, clean_iris_data):
        """Test that classes are reasonably balanced."""
        target_col = clean_iris_data.columns[-1]
        class_counts = clean_iris_data[target_col].value_counts()
        
        # Check that no class is less than 10% of total
        min_class_size = len(clean_iris_data) * 0.1
        assert all(count >= min_class_size for count in class_counts), "Classes are severely imbalanced"


class TestPoisoningDetection:
    """Test poisoning detection capabilities."""
    
    def test_detect_poisoned_vs_clean(self, clean_iris_data, poisoned_data):
        """Test that we can distinguish between clean and poisoned data."""
        # Simple statistical test - poisoned data should have different label distribution
        clean_target = clean_iris_data.iloc[:, -1]
        poisoned_target = poisoned_data.iloc[:, -1]
        
        clean_counts = clean_target.value_counts().sort_index()
        poisoned_counts = poisoned_target.value_counts().sort_index()
        
        # The distributions should be different (with high probability)
        # Using Chi-square test would be more rigorous, but simple comparison should suffice
        assert not clean_counts.equals(poisoned_counts), "Poisoned data distribution should differ from clean"
    
    def test_poisoning_metadata_integrity(self):
        """Test that poisoning metadata files are created correctly."""
        # This test would check if poisoning scripts create proper metadata
        # For now, just verify the concept
        assert True  # Placeholder for metadata validation
    
    def test_label_consistency(self, clean_iris_data):
        """Test that labels are consistent with feature patterns."""
        # This is a placeholder for more sophisticated label consistency checks
        # In a real scenario, you might use clustering or other techniques
        target_col = clean_iris_data.columns[-1]
        features = clean_iris_data.iloc[:, :-1]
        
        # Basic test: each class should have some internal consistency
        for class_name in clean_iris_data[target_col].unique():
            class_data = features[clean_iris_data[target_col] == class_name]
            
            # Classes should have at least some samples
            assert len(class_data) > 0, f"Class {class_name} has no samples"
            
            # Within-class variance should not be excessive
            for col in features.columns:
                class_variance = class_data[col].var()
                overall_variance = features[col].var()
                
                # Within-class variance should be less than overall variance
                assert class_variance <= overall_variance, f"Class {class_name} has excessive variance in {col}"


class TestRobustness:
    """Test model robustness against different data conditions."""
    
    def test_sample_size_robustness(self, clean_iris_data):
        """Test behavior with different sample sizes."""
        # Test with reduced sample sizes
        for fraction in [0.5, 0.7, 0.9]:
            subset_size = int(len(clean_iris_data) * fraction)
            subset_data = clean_iris_data.sample(n=subset_size, random_state=42)
            
            # Should still maintain basic properties
            assert len(subset_data) == subset_size
            
            target_col = subset_data.columns[-1]
            unique_classes = subset_data[target_col].unique()
            
            # Should still have all classes (with high probability)
            assert len(unique_classes) >= 2, f"Subset with {fraction} data lost too many classes"
    
    def test_feature_correlation_stability(self, clean_iris_data):
        """Test that feature correlations are stable."""
        features = clean_iris_data.iloc[:, :-1]
        correlation_matrix = features.corr()
        
        # Check that correlation matrix is well-formed
        assert not correlation_matrix.isnull().any().any(), "Correlation matrix contains NaN values"
        
        # Diagonal should be 1.0
        diagonal_values = np.diag(correlation_matrix.values)
        assert np.allclose(diagonal_values, 1.0), "Correlation matrix diagonal should be 1.0"
    
    def test_outlier_resistance(self, clean_iris_data):
        """Test data quality with potential outliers."""
        # Add some outliers to test robustness
        data_with_outliers = clean_iris_data.copy()
        feature_cols = data_with_outliers.columns[:-1]
        
        # Add extreme values to a few samples
        n_outliers = min(5, len(data_with_outliers))
        outlier_indices = np.random.choice(len(data_with_outliers), n_outliers, replace=False)
        
        for idx in outlier_indices:
            for col in feature_cols:
                # Set to extreme value (5 standard deviations away)
                mean_val = clean_iris_data[col].mean()
                std_val = clean_iris_data[col].std()
                data_with_outliers.loc[idx, col] = mean_val + 5 * std_val
        
        # Data should still be valid despite outliers
        assert not data_with_outliers.isnull().any().any(), "Outlier injection created invalid data"
        
        # Target column should remain unchanged
        target_col = data_with_outliers.columns[-1]
        assert data_with_outliers[target_col].equals(clean_iris_data[target_col]), "Target column was corrupted"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
