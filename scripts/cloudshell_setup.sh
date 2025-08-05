#!/bin/bash
"""
Optimized Cloud Shell Setup Script

This script provides a space-efficient setup for Google Cloud Shell
environments with limited storage.
"""

set -e  # Exit on any error

echo "🌩️  Setting up IRIS pipeline for Google Cloud Shell..."
echo "=============================================="

# Check available space
echo "📊 Checking available disk space..."
df -h $HOME

# Clean up any existing Python cache
echo "🧹 Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Create a minimal virtual environment
echo "🐍 Setting up minimal virtual environment..."
python3 -m venv .venv --clear
source .venv/bin/activate

# Upgrade pip to latest version for better dependency resolution
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install minimal requirements first
echo "📦 Installing minimal requirements..."
pip install --no-cache-dir -r requirements-minimal.txt

# Check if space allows for additional packages
available_space=$(df $HOME | tail -1 | awk '{print $4}')
if [ "$available_space" -gt 2000000 ]; then  # More than 2GB available
    echo "💾 Sufficient space available, installing additional packages..."
    
    # Install additional packages one by one with space checks
    echo "📊 Installing seaborn..."
    pip install --no-cache-dir seaborn>=0.11.0 || echo "⚠️  Seaborn installation failed, continuing..."
    
    echo "🧪 Installing pytest-cov..."
    pip install --no-cache-dir pytest-cov>=4.0.0 || echo "⚠️  pytest-cov installation failed, continuing..."
    
    echo "📈 Installing plotly (if space allows)..."
    pip install --no-cache-dir plotly>=5.0.0 || echo "⚠️  Plotly installation failed, continuing..."
    
else
    echo "⚠️  Limited space available, using minimal installation"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data reports visualizations artifacts mlruns

# Download IRIS dataset using built-in sklearn
echo "📥 Preparing IRIS dataset..."
python3 -c "
import pandas as pd
from sklearn.datasets import load_iris
import os

print('Loading IRIS dataset...')
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

os.makedirs('data', exist_ok=True)
df.to_csv('data/iris.csv', index=False)
print(f'✅ IRIS dataset saved with {len(df)} samples')
"

# Create a lightweight demo script for Cloud Shell
echo "🎮 Creating Cloud Shell demo..."
cat > cloudshell_demo.py << 'EOF'
#!/usr/bin/env python3
"""
Lightweight demo for Cloud Shell environments
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json
import os

def quick_demo():
    print("🎯 IRIS Poisoning Quick Demo for Cloud Shell")
    print("=" * 50)
    
    # Load data
    print("📊 Loading IRIS dataset...")
    if os.path.exists('data/iris.csv'):
        df = pd.read_csv('data/iris.csv')
    else:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = [iris.target_names[i] for i in iris.target]
    
    print(f"✅ Dataset loaded: {len(df)} samples, {len(df['species'].unique())} classes")
    
    # Create poisoned version (10%)
    print("⚔️  Creating 10% poisoned dataset...")
    poisoned_df = df.copy()
    poison_count = int(0.1 * len(df))
    poison_indices = np.random.choice(df.index, poison_count, replace=False)
    
    species_list = df['species'].unique()
    for idx in poison_indices:
        original = poisoned_df.loc[idx, 'species']
        new_species = np.random.choice([s for s in species_list if s != original])
        poisoned_df.loc[idx, 'species'] = new_species
    
    print(f"✅ Poisoned {poison_count} labels")
    
    # Train models
    results = {}
    for name, data in [("clean", df), ("poisoned", poisoned_df)]:
        print(f"🎯 Training {name} model...")
        
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)  # Smaller for speed
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = accuracy
        print(f"   {name.title()} accuracy: {accuracy:.4f}")
    
    # Show results
    performance_drop = results["clean"] - results["poisoned"]
    relative_drop = (performance_drop / results["clean"]) * 100
    
    print("\n📊 RESULTS SUMMARY")
    print("=" * 30)
    print(f"Clean model accuracy:    {results['clean']:.4f}")
    print(f"Poisoned model accuracy: {results['poisoned']:.4f}")
    print(f"Performance drop:        {performance_drop:.4f} ({relative_drop:.1f}%)")
    
    if performance_drop > 0.05:
        print("🚨 Significant performance degradation detected!")
    else:
        print("✅ Minimal performance impact observed")
    
    # Save results
    summary = {
        "clean_accuracy": results["clean"],
        "poisoned_accuracy": results["poisoned"],
        "performance_drop": performance_drop,
        "relative_drop_percent": relative_drop,
        "poison_level": 0.1,
        "attack_success": performance_drop > 0.05
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/cloudshell_demo_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to reports/cloudshell_demo_results.json")
    print("🎉 Demo completed successfully!")

if __name__ == "__main__":
    quick_demo()
EOF

chmod +x cloudshell_demo.py

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import sklearn
import pandas
import numpy
print('✅ Core packages working')
print(f'   scikit-learn: {sklearn.__version__}')
print(f'   pandas: {pandas.__version__}')
print(f'   numpy: {numpy.__version__}')
"

echo ""
echo "🎉 Cloud Shell setup complete!"
echo "=============================================="
echo "📊 Final disk usage:"
df -h $HOME
echo ""
echo "🚀 Quick start commands:"
echo "  python cloudshell_demo.py              # Run quick demo"
echo "  python src/poison_data.py --help       # See poisoning options"
echo "  python src/train_model.py --help       # See training options"
echo ""
echo "💡 For full pipeline (if space allows):"
echo "  pip install -r requirements.txt        # Install all packages"
echo "  python demo_poisoning_experiment.py    # Run full demo"
EOF

chmod +x scripts/cloudshell_setup.sh

echo "✅ Cloud Shell setup script created"