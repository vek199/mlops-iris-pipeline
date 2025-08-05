#!/usr/bin/env python3
"""
Demo Script for IRIS Data Poisoning Experiment

This script provides a simple demonstration of the data poisoning pipeline
without requiring DVC or MLflow to be installed. It creates poisoned data,
trains models, and shows the impact of poisoning.

Usage: python demo_poisoning_experiment.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import json
from pathlib import Path


def create_iris_dataset():
    """Create and save the IRIS dataset."""
    print("📊 Creating IRIS dataset...")
    
    # Load IRIS dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    
    # Create data directory and save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris.csv', index=False)
    
    print(f"✅ IRIS dataset saved with {len(df)} samples and {len(df['species'].unique())} classes")
    return df


def poison_dataset(df, poison_level=0.10, random_seed=42):
    """Create a poisoned version of the dataset."""
    print(f"⚔️  Creating poisoned dataset with {poison_level*100}% poison level...")
    
    np.random.seed(random_seed)
    poisoned_df = df.copy()
    
    # Calculate number of samples to poison
    num_to_poison = int(len(df) * poison_level)
    poison_indices = np.random.choice(df.index, size=num_to_poison, replace=False)
    
    # Get unique species
    species_list = df['species'].unique().tolist()
    
    # Flip labels
    flipped_count = 0
    for idx in poison_indices:
        original_species = poisoned_df.loc[idx, 'species']
        possible_new_species = [s for s in species_list if s != original_species]
        new_species = np.random.choice(possible_new_species)
        poisoned_df.loc[idx, 'species'] = new_species
        flipped_count += 1
    
    print(f"✅ Poisoned {flipped_count} labels out of {len(df)} samples")
    return poisoned_df


def train_and_evaluate_model(df, model_name="clean"):
    """Train a model and return evaluation metrics."""
    print(f"🎯 Training {model_name} model...")
    
    # Prepare data
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"✅ {model_name.title()} model accuracy: {accuracy:.4f}")
    
    return {
        'model': model,
        'label_encoder': le,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred
    }


def detect_suspicious_labels(df, k=5, threshold=0.5):
    """Simple suspicious label detection using KNN approach."""
    print(f"🔍 Detecting suspicious labels with k={k}, threshold={threshold}...")
    
    from sklearn.neighbors import NearestNeighbors
    
    # Prepare data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)  # +1 because point is neighbor to itself
    distances, indices = nbrs.kneighbors(X)
    
    suspicious_indices = []
    
    for i in range(len(df)):
        # Get neighbors (excluding self at index 0)
        neighbor_indices = indices[i][1:]
        neighbor_labels = y_encoded[neighbor_indices]
        
        # Count disagreements
        disagreements = np.sum(neighbor_labels != y_encoded[i])
        disagreement_ratio = disagreements / k
        
        if disagreement_ratio >= threshold:
            suspicious_indices.append(i)
    
    print(f"🚨 Found {len(suspicious_indices)} suspicious labels out of {len(df)} total")
    print(f"   Suspicion rate: {len(suspicious_indices)/len(df)*100:.1f}%")
    
    return suspicious_indices


def create_visualizations(clean_results, poisoned_results, poison_level):
    """Create comparison visualizations."""
    print("📊 Creating visualizations...")
    
    os.makedirs('visualizations', exist_ok=True)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Clean vs {poison_level*100}% Poisoned Model Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    models = ['Clean', f'Poisoned ({poison_level*100}%)']
    accuracies = [clean_results['accuracy'], poisoned_results['accuracy']]
    
    bars = ax1.bar(models, accuracies, color=['skyblue', 'salmon'], alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Clean model confusion matrix
    cm_clean = clean_results['confusion_matrix']
    species_names = clean_results['label_encoder'].classes_
    
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues',
                xticklabels=species_names, yticklabels=species_names, ax=ax2)
    ax2.set_title('Clean Model\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # 3. Poisoned model confusion matrix
    cm_poisoned = poisoned_results['confusion_matrix']
    
    sns.heatmap(cm_poisoned, annot=True, fmt='d', cmap='Reds',
                xticklabels=species_names, yticklabels=species_names, ax=ax3)
    ax3.set_title(f'Poisoned Model ({poison_level*100}%)\nConfusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Performance metrics comparison
    metrics = ['precision', 'recall', 'f1-score']
    clean_scores = [clean_results['classification_report']['macro avg'][m] for m in metrics]
    poisoned_scores = [poisoned_results['classification_report']['macro avg'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, clean_scores, width, label='Clean', color='skyblue', alpha=0.7)
    ax4.bar(x + width/2, poisoned_scores, width, label=f'Poisoned ({poison_level*100}%)', color='salmon', alpha=0.7)
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Score')
    ax4.set_title('Detailed Performance Metrics')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace('-', ' ').title() for m in metrics])
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('visualizations/poisoning_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to visualizations/poisoning_impact_analysis.png")


def save_results(clean_results, poisoned_results, poison_level, suspicious_indices):
    """Save experimental results to JSON."""
    print("💾 Saving results...")
    
    os.makedirs('reports', exist_ok=True)
    
    # Calculate performance degradation
    performance_drop = clean_results['accuracy'] - poisoned_results['accuracy']
    relative_drop = (performance_drop / clean_results['accuracy']) * 100 if clean_results['accuracy'] > 0 else 0
    
    results = {
        'experiment_config': {
            'poison_level': poison_level,
            'random_seed': 42,
            'model_type': 'RandomForest'
        },
        'performance_metrics': {
            'clean_accuracy': clean_results['accuracy'],
            'poisoned_accuracy': poisoned_results['accuracy'],
            'absolute_drop': performance_drop,
            'relative_drop_percent': relative_drop
        },
        'detection_results': {
            'suspicious_labels_count': len(suspicious_indices),
            'total_samples': 150,  # IRIS dataset size
            'detection_rate': len(suspicious_indices) / 150 * 100
        },
        'summary': {
            'attack_success': 'Yes' if performance_drop > 0.05 else 'Limited',
            'detection_effectiveness': 'Good' if len(suspicious_indices) > poison_level * 150 * 0.5 else 'Poor',
            'overall_impact': 'High' if relative_drop > 20 else 'Medium' if relative_drop > 10 else 'Low'
        }
    }
    
    with open('reports/demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to reports/demo_results.json")
    return results


def print_summary(results):
    """Print a summary of the experiment."""
    print("\n" + "="*60)
    print("🧪 IRIS DATA POISONING EXPERIMENT SUMMARY")
    print("="*60)
    
    config = results['experiment_config']
    metrics = results['performance_metrics']
    detection = results['detection_results']
    summary = results['summary']
    
    print(f"📊 Experiment Configuration:")
    print(f"   Poison Level: {config['poison_level']*100}%")
    print(f"   Model Type: {config['model_type']}")
    
    print(f"\n📈 Performance Impact:")
    print(f"   Clean Model Accuracy: {metrics['clean_accuracy']:.4f}")
    print(f"   Poisoned Model Accuracy: {metrics['poisoned_accuracy']:.4f}")
    print(f"   Performance Drop: {metrics['absolute_drop']:.4f} ({metrics['relative_drop_percent']:.1f}%)")
    
    print(f"\n🔍 Detection Results:")
    print(f"   Suspicious Labels Found: {detection['suspicious_labels_count']}")
    print(f"   Detection Rate: {detection['detection_rate']:.1f}%")
    
    print(f"\n📋 Summary Assessment:")
    print(f"   Attack Success: {summary['attack_success']}")
    print(f"   Detection Effectiveness: {summary['detection_effectiveness']}")
    print(f"   Overall Impact: {summary['overall_impact']}")
    
    print(f"\n💡 Key Insights:")
    if metrics['relative_drop_percent'] > 20:
        print("   - Significant performance degradation observed")
    elif metrics['relative_drop_percent'] > 10:
        print("   - Moderate performance degradation observed")
    else:
        print("   - Limited performance degradation observed")
    
    if detection['detection_rate'] > 70:
        print("   - Good detection capability - most poisoned labels identified")
    elif detection['detection_rate'] > 30:
        print("   - Moderate detection capability - some poisoned labels identified")
    else:
        print("   - Poor detection capability - few poisoned labels identified")
    
    print("\n🛡️  Mitigation Recommendations:")
    print("   1. Implement label validation before training")
    print("   2. Use robust training methods (e.g., regularization)")
    print("   3. Monitor model performance on validation sets")
    print("   4. Establish data provenance tracking")
    print("   5. Consider ensemble methods for better robustness")
    
    print("="*60)


def main():
    """Main demonstration function."""
    print("🎯 IRIS Data Poisoning Experiment Demo")
    print("="*50)
    
    # Configuration
    poison_level = 0.10  # 10% poisoning
    
    # Step 1: Create dataset
    clean_df = create_iris_dataset()
    
    # Step 2: Create poisoned dataset
    poisoned_df = poison_dataset(clean_df, poison_level)
    
    # Step 3: Train models
    clean_results = train_and_evaluate_model(clean_df, "clean")
    poisoned_results = train_and_evaluate_model(poisoned_df, "poisoned")
    
    # Step 4: Detect suspicious labels
    suspicious_indices = detect_suspicious_labels(poisoned_df)
    
    # Step 5: Create visualizations
    create_visualizations(clean_results, poisoned_results, poison_level)
    
    # Step 6: Save and display results
    results = save_results(clean_results, poisoned_results, poison_level, suspicious_indices)
    print_summary(results)
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Check the following directories for outputs:")
    print(f"   - data/ (datasets)")
    print(f"   - reports/ (JSON results)")
    print(f"   - visualizations/ (plots)")


if __name__ == "__main__":
    main()