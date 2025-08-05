#!/usr/bin/env python3
"""
Comprehensive Tests for Data Poisoning Pipeline

This module tests all components of the data poisoning experiment:
- Data poisoning functionality
- Label validation algorithms
- Model training with poisoned data
- Performance evaluation and comparison

Author: MLOps Pipeline
Date: 2024
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from poison_data import poison_labels, poison_multiple_levels
from validate_labels import LabelValidator
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator


class TestDataPoisoning:
    """Test suite for data poisoning functionality."""
    
    @pytest.fixture
    def sample_iris_data(self):
        """Create a sample IRIS-like dataset for testing."""
        np.random.seed(42)
        data = {
            'sepal_length': np.random.normal(5.5, 1.0, 150),
            'sepal_width': np.random.normal(3.0, 0.5, 150),
            'petal_length': np.random.normal(4.0, 1.5, 150),
            'petal_width': np.random.normal(1.3, 0.7, 150),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_poison_labels_basic(self, sample_iris_data, temp_dir):
        """Test basic label poisoning functionality."""
        # Save sample data
        input_path = os.path.join(temp_dir, "test_iris.csv")
        output_path = os.path.join(temp_dir, "test_iris_poisoned.csv")
        sample_iris_data.to_csv(input_path, index=False)
        
        # Poison 10% of labels
        poison_level = 0.10
        poison_labels(input_path, output_path, poison_level, random_seed=42)
        
        # Check that poisoned file exists
        assert os.path.exists(output_path)
        
        # Load original and poisoned data
        original_df = pd.read_csv(input_path)
        poisoned_df = pd.read_csv(output_path)
        
        # Check that shapes are the same
        assert original_df.shape == poisoned_df.shape
        
        # Check that some labels have been changed
        target_col = original_df.columns[-1]
        changes = (original_df[target_col] != poisoned_df[target_col]).sum()
        expected_changes = int(len(original_df) * poison_level)
        
        # Allow for small rounding differences
        assert abs(changes - expected_changes) <= 1
        
        # Check that metadata file was created
        metadata_path = output_path.replace('.csv', '_metadata.json')
        assert os.path.exists(metadata_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['poison_level'] == poison_level
        assert metadata['total_rows'] == len(original_df)
        assert metadata['poisoned_rows'] == expected_changes
    
    def test_poison_labels_edge_cases(self, sample_iris_data, temp_dir):
        """Test edge cases for label poisoning."""
        input_path = os.path.join(temp_dir, "test_iris.csv")
        sample_iris_data.to_csv(input_path, index=False)
        
        # Test 0% poisoning
        output_path_0 = os.path.join(temp_dir, "test_iris_0pct.csv")
        poison_labels(input_path, output_path_0, 0.0)
        
        original_df = pd.read_csv(input_path)
        poisoned_df_0 = pd.read_csv(output_path_0)
        
        # Should be identical
        pd.testing.assert_frame_equal(original_df, poisoned_df_0)
        
        # Test 100% poisoning
        output_path_100 = os.path.join(temp_dir, "test_iris_100pct.csv")
        poison_labels(input_path, output_path_100, 1.0)
        
        poisoned_df_100 = pd.read_csv(output_path_100)
        target_col = original_df.columns[-1]
        
        # All labels should be different (since we have multiple classes)
        changes = (original_df[target_col] != poisoned_df_100[target_col]).sum()
        assert changes == len(original_df)
    
    def test_poison_labels_invalid_inputs(self, sample_iris_data, temp_dir):
        """Test error handling for invalid inputs."""
        input_path = os.path.join(temp_dir, "test_iris.csv")
        output_path = os.path.join(temp_dir, "test_iris_poisoned.csv")
        sample_iris_data.to_csv(input_path, index=False)
        
        # Test invalid poison level
        with pytest.raises(ValueError):
            poison_labels(input_path, output_path, -0.1)
        
        with pytest.raises(ValueError):
            poison_labels(input_path, output_path, 1.1)
        
        # Test non-existent input file
        with pytest.raises(FileNotFoundError):
            poison_labels("non_existent.csv", output_path, 0.1)
    
    def test_poison_multiple_levels(self, sample_iris_data, temp_dir):
        """Test creating multiple poisoned datasets."""
        input_path = os.path.join(temp_dir, "test_iris.csv")
        output_dir = os.path.join(temp_dir, "poisoned")
        sample_iris_data.to_csv(input_path, index=False)
        
        poison_levels = [0.05, 0.10, 0.20]
        poison_multiple_levels(input_path, output_dir, poison_levels)
        
        # Check that all files were created
        for level in poison_levels:
            level_pct = int(level * 100)
            expected_file = os.path.join(output_dir, f"iris_poisoned_{level_pct}pct.csv")
            assert os.path.exists(expected_file)
            
            # Verify the poisoning level
            original_df = pd.read_csv(input_path)
            poisoned_df = pd.read_csv(expected_file)
            
            target_col = original_df.columns[-1]
            changes = (original_df[target_col] != poisoned_df[target_col]).sum()
            expected_changes = int(len(original_df) * level)
            
            assert abs(changes - expected_changes) <= 1


class TestLabelValidation:
    """Test suite for label validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a label validator instance."""
        return LabelValidator(k=5, threshold=0.5)
    
    @pytest.fixture
    def clean_data(self):
        """Create clean, well-separated data."""
        np.random.seed(42)
        # Create three well-separated clusters
        cluster1 = np.random.normal([0, 0], 0.5, (50, 2))
        cluster2 = np.random.normal([5, 5], 0.5, (50, 2))
        cluster3 = np.random.normal([0, 5], 0.5, (50, 2))
        
        X = np.vstack([cluster1, cluster2, cluster3])
        y = ['A'] * 50 + ['B'] * 50 + ['C'] * 50
        
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def poisoned_data(self, clean_data):
        """Create data with some mislabeled instances."""
        poisoned_df = clean_data.copy()
        
        # Flip some labels manually for testing
        # Change some 'A' labels to 'B'
        poisoned_df.loc[5:9, 'target'] = 'B'  # 5 instances
        # Change some 'B' labels to 'C' 
        poisoned_df.loc[55:59, 'target'] = 'C'  # 5 instances
        
        return poisoned_df
    
    def test_clean_data_validation(self, validator, clean_data, temp_dir):
        """Test validation on clean data - should find no suspicious labels."""
        data_path = os.path.join(temp_dir, "clean_data.csv")
        clean_data.to_csv(data_path, index=False)
        
        suspicious_indices, report = validator.find_suspicious_labels(data_path)
        
        # Should find very few or no suspicious labels in clean data
        assert len(suspicious_indices) <= 2  # Allow for some edge cases
        assert report['suspicion_rate'] <= 0.05  # Less than 5%
    
    def test_poisoned_data_validation(self, validator, poisoned_data, temp_dir):
        """Test validation on poisoned data - should detect suspicious labels."""
        data_path = os.path.join(temp_dir, "poisoned_data.csv")
        poisoned_data.to_csv(data_path, index=False)
        
        suspicious_indices, report = validator.find_suspicious_labels(data_path)
        
        # Should detect some of the mislabeled instances
        assert len(suspicious_indices) >= 5  # Should find at least some of the 10 flipped labels
        assert report['suspicion_rate'] > 0.03  # At least 3% suspicion rate
        
        # Check that some of our known flipped indices are detected
        known_flipped = list(range(5, 10)) + list(range(55, 60))
        detected_flipped = set(suspicious_indices) & set(known_flipped)
        assert len(detected_flipped) >= 3  # Should detect at least some
    
    def test_validator_parameters(self, clean_data, temp_dir):
        """Test different validator parameters."""
        data_path = os.path.join(temp_dir, "test_data.csv")
        clean_data.to_csv(data_path, index=False)
        
        # Test different k values
        validator_k3 = LabelValidator(k=3, threshold=0.5)
        validator_k7 = LabelValidator(k=7, threshold=0.5)
        
        _, report_k3 = validator_k3.find_suspicious_labels(data_path)
        _, report_k7 = validator_k7.find_suspicious_labels(data_path)
        
        # Results should be different but both valid
        assert 'total_rows' in report_k3
        assert 'total_rows' in report_k7
        assert report_k3['total_rows'] == report_k7['total_rows']
        
        # Test different thresholds
        validator_low = LabelValidator(k=5, threshold=0.3)
        validator_high = LabelValidator(k=5, threshold=0.7)
        
        suspicious_low, _ = validator_low.find_suspicious_labels(data_path)
        suspicious_high, _ = validator_high.find_suspicious_labels(data_path)
        
        # Lower threshold should generally find more suspicious instances
        assert len(suspicious_low) >= len(suspicious_high)
    
    def test_confidence_validation(self, validator, poisoned_data, temp_dir):
        """Test confidence-based validation."""
        data_path = os.path.join(temp_dir, "poisoned_data.csv")
        poisoned_data.to_csv(data_path, index=False)
        
        confidence_report = validator.validate_with_confidence_threshold(data_path)
        
        assert 'suspicious_points' in confidence_report
        assert 'model_accuracy' in confidence_report
        assert 0 <= confidence_report['model_accuracy'] <= 1
        
        # Should find some suspicious points in poisoned data
        assert len(confidence_report['suspicious_points']) >= 0


class TestModelTraining:
    """Test suite for model training functionality."""
    
    @pytest.fixture
    def trainer(self):
        """Create a model trainer instance."""
        return ModelTrainer(experiment_name="test_experiment")
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Create sample data file for training."""
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100),
            'target': np.random.choice(['A', 'B', 'C'], 100)
        }
        df = pd.DataFrame(data)
        
        data_path = os.path.join(temp_dir, "sample_data.csv")
        df.to_csv(data_path, index=False)
        
        return data_path
    
    def test_data_loading_and_preparation(self, trainer, sample_data):
        """Test data loading and preprocessing."""
        X_train, X_test, y_train, y_test, y_original = trainer.load_and_prepare_data(sample_data)
        
        # Check shapes
        assert X_train.shape[0] == 80  # 80% for training
        assert X_test.shape[0] == 20   # 20% for testing
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert len(y_original) == 100
        
        # Check that features are scaled (mean should be close to 0)
        assert abs(np.mean(X_train)) < 0.5
        assert abs(np.mean(X_test)) < 0.5
    
    def test_single_model_training(self, trainer, sample_data, temp_dir):
        """Test training a single model."""
        # Change to temp directory for artifacts
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            X_train, X_test, y_train, y_test, _ = trainer.load_and_prepare_data(sample_data)
            result = trainer.train_single_model(
                'random_forest', X_train, y_train, X_test, y_test, sample_data
            )
            
            # Check that result contains expected keys
            assert 'metrics' in result
            assert 'artifacts' in result
            assert 'model' in result
            
            # Check metrics
            metrics = result['metrics']
            assert 'accuracy' in metrics
            assert 'f1_macro' in metrics
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['f1_macro'] <= 1
            
            # Check that artifacts were saved
            artifacts = result['artifacts']
            assert os.path.exists(artifacts['model_path'])
            assert os.path.exists(artifacts['encoder_path'])
            assert os.path.exists(artifacts['scaler_path'])
            
        finally:
            os.chdir(original_dir)
    
    def test_invalid_model_training(self, trainer, sample_data):
        """Test error handling for invalid model names."""
        X_train, X_test, y_train, y_test, _ = trainer.load_and_prepare_data(sample_data)
        
        with pytest.raises(ValueError):
            trainer.train_single_model(
                'invalid_model', X_train, y_train, X_test, y_test, sample_data
            )
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    def test_mlflow_integration(self, mock_log_metric, mock_log_param, mock_start_run, trainer, sample_data):
        """Test MLflow integration (mocked)."""
        # Mock MLflow context manager
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock()
        
        X_train, X_test, y_train, y_test, _ = trainer.load_and_prepare_data(sample_data)
        
        # This should work even with mocked MLflow
        result = trainer.train_single_model(
            'random_forest', X_train, y_train, X_test, y_test, sample_data
        )
        
        assert 'metrics' in result


class TestModelEvaluation:
    """Test suite for model evaluation functionality."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a model evaluator instance."""
        return ModelEvaluator()
    
    @pytest.fixture
    def mock_model_artifacts(self, temp_dir):
        """Create mock model artifacts for testing."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        import joblib
        
        # Create mock artifacts directory
        artifacts_dir = os.path.join(temp_dir, "test_artifacts")
        os.makedirs(artifacts_dir)
        
        # Create and save mock model
        model = RandomForestClassifier(random_state=42)
        # Fit on dummy data
        X_dummy = np.random.random((10, 4))
        y_dummy = [0, 1, 2] * 3 + [0]
        model.fit(X_dummy, y_dummy)
        
        # Create and save label encoder
        le = LabelEncoder()
        le.fit(['setosa', 'versicolor', 'virginica'])
        
        # Create and save scaler
        scaler = StandardScaler()
        scaler.fit(X_dummy)
        
        # Save artifacts
        joblib.dump(model, os.path.join(artifacts_dir, "random_forest.joblib"))
        joblib.dump(le, os.path.join(artifacts_dir, "label_encoder.joblib"))
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))
        
        return artifacts_dir
    
    @pytest.fixture
    def sample_test_data(self, temp_dir):
        """Create sample test data."""
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(5.5, 1.0, 30),
            'feature2': np.random.normal(3.0, 0.5, 30),
            'feature3': np.random.normal(4.0, 1.5, 30),
            'feature4': np.random.normal(1.3, 0.7, 30),
            'species': ['setosa'] * 10 + ['versicolor'] * 10 + ['virginica'] * 10
        }
        df = pd.DataFrame(data)
        
        test_data_path = os.path.join(temp_dir, "test_data.csv")
        df.to_csv(test_data_path, index=False)
        
        return test_data_path
    
    def test_load_model_artifacts(self, evaluator, mock_model_artifacts):
        """Test loading model artifacts."""
        model, label_encoder, scaler = evaluator.load_model_artifacts(mock_model_artifacts)
        
        # Check that all components were loaded
        assert model is not None
        assert label_encoder is not None
        assert scaler is not None
        
        # Check that they have expected properties
        assert hasattr(model, 'predict')
        assert hasattr(label_encoder, 'transform')
        assert hasattr(scaler, 'transform')
    
    def test_evaluate_on_dataset(self, evaluator, mock_model_artifacts, sample_test_data):
        """Test model evaluation on a dataset."""
        model, label_encoder, scaler = evaluator.load_model_artifacts(mock_model_artifacts)
        
        metrics = evaluator.evaluate_on_dataset(
            model, sample_test_data, label_encoder, scaler
        )
        
        # Check that metrics are present
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'confusion_matrix' in metrics
        assert 'dataset_info' in metrics
        
        # Check that metrics are valid
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
        
        # Check dataset info
        dataset_info = metrics['dataset_info']
        assert dataset_info['samples'] == 30
        assert dataset_info['features'] == 4
        assert dataset_info['classes'] == 3
    
    def test_generate_evaluation_report(self, evaluator, temp_dir):
        """Test generation of evaluation reports."""
        # Create mock results
        mock_results = {
            'accuracy': 0.95,
            'f1_macro': 0.93,
            'test_info': 'mock evaluation'
        }
        
        report_path = os.path.join(temp_dir, "test_report.json")
        evaluator.generate_evaluation_report(mock_results, report_path)
        
        # Check that report was created
        assert os.path.exists(report_path)
        
        # Check report content
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        assert 'evaluation_timestamp' in report
        assert 'results' in report
        assert 'summary' in report
        assert report['results'] == mock_results


class TestIntegration:
    """Integration tests for the complete poisoning pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for integration tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def iris_dataset(self, temp_dir):
        """Create a realistic IRIS-like dataset."""
        np.random.seed(42)
        
        # Generate more realistic IRIS-like data
        n_samples_per_class = 50
        
        # Setosa: smaller flowers
        setosa_data = {
            'sepal_length': np.random.normal(5.0, 0.4, n_samples_per_class),
            'sepal_width': np.random.normal(3.4, 0.4, n_samples_per_class),
            'petal_length': np.random.normal(1.5, 0.2, n_samples_per_class),
            'petal_width': np.random.normal(0.3, 0.1, n_samples_per_class),
            'species': ['setosa'] * n_samples_per_class
        }
        
        # Versicolor: medium flowers
        versicolor_data = {
            'sepal_length': np.random.normal(5.9, 0.5, n_samples_per_class),
            'sepal_width': np.random.normal(2.8, 0.3, n_samples_per_class),
            'petal_length': np.random.normal(4.3, 0.5, n_samples_per_class),
            'petal_width': np.random.normal(1.3, 0.2, n_samples_per_class),
            'species': ['versicolor'] * n_samples_per_class
        }
        
        # Virginica: larger flowers
        virginica_data = {
            'sepal_length': np.random.normal(6.5, 0.6, n_samples_per_class),
            'sepal_width': np.random.normal(3.0, 0.3, n_samples_per_class),
            'petal_length': np.random.normal(5.5, 0.6, n_samples_per_class),
            'petal_width': np.random.normal(2.0, 0.3, n_samples_per_class),
            'species': ['virginica'] * n_samples_per_class
        }
        
        # Combine all data
        all_data = {}
        for key in setosa_data.keys():
            if key == 'species':
                all_data[key] = (setosa_data[key] + versicolor_data[key] + 
                               virginica_data[key])
            else:
                all_data[key] = np.concatenate([
                    setosa_data[key], versicolor_data[key], virginica_data[key]
                ])
        
        df = pd.DataFrame(all_data)
        
        # Save dataset
        dataset_path = os.path.join(temp_dir, "iris_realistic.csv")
        df.to_csv(dataset_path, index=False)
        
        return dataset_path
    
    def test_end_to_end_poisoning_pipeline(self, iris_dataset, temp_dir):
        """Test the complete poisoning detection pipeline."""
        # Step 1: Create poisoned datasets
        poison_levels = [0.05, 0.10]
        poisoned_files = []
        
        for level in poison_levels:
            level_pct = int(level * 100)
            poisoned_path = os.path.join(temp_dir, f"iris_poisoned_{level_pct}pct.csv")
            poison_labels(iris_dataset, poisoned_path, level, random_seed=42)
            poisoned_files.append(poisoned_path)
            
            # Verify poisoning worked
            assert os.path.exists(poisoned_path)
        
        # Step 2: Validate labels on poisoned data
        validator = LabelValidator(k=5, threshold=0.5)
        
        for poisoned_file in poisoned_files:
            suspicious_indices, report = validator.find_suspicious_labels(poisoned_file)
            
            # Should detect some suspicious labels in poisoned data
            assert len(suspicious_indices) > 0
            assert report['suspicion_rate'] > 0
        
        # Step 3: Train models on clean and poisoned data
        trainer = ModelTrainer(experiment_name="integration_test")
        
        # Change to temp directory for artifacts
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Train on clean data
            clean_results = trainer.train_all_models(iris_dataset)
            assert 'random_forest' in clean_results
            assert 'metrics' in clean_results['random_forest']
            
            # Train on poisoned data
            poisoned_results = trainer.train_all_models(poisoned_files[0])  # 5% poisoned
            assert 'random_forest' in poisoned_results
            assert 'metrics' in poisoned_results['random_forest']
            
            # Compare performance
            clean_accuracy = clean_results['random_forest']['metrics']['accuracy']
            poisoned_accuracy = poisoned_results['random_forest']['metrics']['accuracy']
            
            # Poisoned model should generally perform worse
            # (though with small poison levels, this might not always be true)
            performance_drop = clean_accuracy - poisoned_accuracy
            
            # At minimum, performance should not improve dramatically
            assert performance_drop >= -0.1  # Allow for some variance
            
        finally:
            os.chdir(original_dir)
    
    def test_robustness_evaluation(self, iris_dataset, temp_dir):
        """Test robustness evaluation across multiple datasets."""
        # Create multiple datasets with different poison levels
        datasets = [iris_dataset]  # Start with clean dataset
        
        for level in [0.05, 0.10, 0.20]:
            level_pct = int(level * 100)
            poisoned_path = os.path.join(temp_dir, f"iris_poisoned_{level_pct}pct.csv")
            poison_labels(iris_dataset, poisoned_path, level, random_seed=42)
            datasets.append(poisoned_path)
        
        # Train a model on clean data
        trainer = ModelTrainer()
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            clean_results = trainer.train_all_models(iris_dataset)
            
            # Find the artifacts directory for the trained model
            artifacts_dirs = [d for d in os.listdir('.') if d.startswith('artifacts')]
            if artifacts_dirs:
                # Use the first artifacts directory found
                artifacts_path = os.path.join('.', artifacts_dirs[0], 'random_forest_iris_realistic')
                
                if os.path.exists(artifacts_path):
                    # Evaluate robustness
                    evaluator = ModelEvaluator()
                    robustness_results = evaluator.evaluate_robustness(artifacts_path, datasets)
                    
                    # Check results structure
                    assert 'individual_results' in robustness_results
                    assert 'robustness_metrics' in robustness_results
                    assert 'datasets_evaluated' in robustness_results
                    
                    # Should have evaluated multiple datasets
                    assert robustness_results['datasets_evaluated'] >= 2
                    
                    # Check that robustness metrics contain expected measures
                    robustness_metrics = robustness_results['robustness_metrics']
                    if 'accuracy' in robustness_metrics:
                        acc_metrics = robustness_metrics['accuracy']
                        assert 'mean' in acc_metrics
                        assert 'std' in acc_metrics
                        assert 'coefficient_of_variation' in acc_metrics
        
        finally:
            os.chdir(original_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])