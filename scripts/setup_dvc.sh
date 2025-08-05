#!/bin/bash
"""
DVC Setup and Testing Script

This script demonstrates the DVC commands that should be run to set up
data versioning for the iris pipeline project.

Note: This script requires DVC to be installed. If not installed, 
uncomment the installation line below.
"""

set -e  # Exit on any error

echo "🔧 Setting up DVC for IRIS pipeline..."

# Uncomment the following line if DVC is not installed
# pip install dvc[gs]

# Initialize DVC (if not already done)
if [ ! -d ".dvc" ]; then
    echo "📁 Initializing DVC repository..."
    dvc init --no-scm
else
    echo "✅ DVC repository already initialized"
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download IRIS dataset if not present
if [ ! -f "data/iris.csv" ]; then
    echo "📥 Downloading IRIS dataset..."
    python -c "
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

# Load iris dataset
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame
df = pd.DataFrame(iris.data, columns=feature_names)
df['species'] = [target_names[i] for i in iris.target]

# Save to CSV
df.to_csv('data/iris.csv', index=False)
print('✅ IRIS dataset saved to data/iris.csv')
"
fi

# Add data to DVC tracking
echo "📊 Adding data to DVC tracking..."
dvc add data/iris.csv

# Create DVC pipeline stages
echo "🔄 Creating DVC pipeline..."

# Stage 1: Data poisoning
cat > dvc.yaml << 'EOF'
stages:
  poison_data:
    cmd: python src/poison_data.py --multiple-levels --input-path data/iris.csv --output-dir data/poisoned
    deps:
    - src/poison_data.py
    - data/iris.csv
    outs:
    - data/poisoned/
    
  train_clean:
    cmd: python src/train_model.py --data-path data/iris.csv --experiment-name clean_baseline
    deps:
    - src/train_model.py
    - data/iris.csv
    outs:
    - artifacts/
    metrics:
    - artifacts/clean_model/metrics.json
    
  train_poisoned:
    cmd: python src/train_model.py --data-path data/poisoned/iris_poisoned_10pct.csv --experiment-name poisoned_10pct
    deps:
    - src/train_model.py
    - data/poisoned/iris_poisoned_10pct.csv
    outs:
    - artifacts/poisoned_model/
    metrics:
    - artifacts/poisoned_model/metrics.json
    
  validate_labels:
    cmd: python src/validate_labels.py --data-path data/poisoned/iris_poisoned_10pct.csv --output-report validation_report.json
    deps:
    - src/validate_labels.py
    - data/poisoned/iris_poisoned_10pct.csv
    outs:
    - validation_report.json
    
  evaluate_comparison:
    cmd: python src/evaluate_model.py --compare-models --clean-artifacts artifacts/clean_model --poisoned-artifacts artifacts/poisoned_model --test-data data/iris.csv --output-report comparison_report.json
    deps:
    - src/evaluate_model.py
    - artifacts/clean_model/
    - artifacts/poisoned_model/
    - data/iris.csv
    outs:
    - comparison_report.json
EOF

echo "✅ DVC pipeline created in dvc.yaml"

# Create DVC params file for experiment configuration
cat > params.yaml << 'EOF'
# Data poisoning parameters
poisoning:
  levels: [0.05, 0.10, 0.50]
  random_seed: 42

# Model training parameters
training:
  test_size: 0.2
  random_state: 42
  models:
    random_forest:
      n_estimators: 100
      max_depth: null
    svm:
      C: 1.0
      kernel: rbf
    logistic_regression:
      max_iter: 1000

# Label validation parameters
validation:
  knn:
    k: 5
    threshold: 0.5
  confidence:
    threshold: 0.8

# Visualization parameters
visualization:
  figsize: [12, 8]
  dpi: 300
EOF

echo "✅ Parameters file created: params.yaml"

# Create .dvcignore file
cat > .dvcignore << 'EOF'
# Add patterns of files dvc should ignore, which could later be added by `dvc add`.
# The patterns are detected by the same rules git uses for .gitignore files.
# See https://git-scm.com/docs/gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# MLflow
mlruns/
mlartifacts/

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
logs/
EOF

echo "✅ DVC ignore file created: .dvcignore"

# Show DVC status
echo "📊 Current DVC status:"
dvc status || echo "⚠️  DVC status check failed - this is normal if pipeline hasn't been run yet"

echo ""
echo "🎉 DVC setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run the pipeline: dvc repro"
echo "2. Check pipeline status: dvc status"
echo "3. View metrics: dvc metrics show"
echo "4. Compare experiments: dvc plots show"
echo "5. Push data to remote: dvc push (after configuring remote storage)"
echo ""
echo "🔗 Useful DVC commands:"
echo "  dvc dag                    # Show pipeline DAG"
echo "  dvc metrics diff           # Compare metrics across runs"
echo "  dvc plots diff             # Compare plots across runs"
echo "  dvc experiments show       # Show all experiments"
echo "  dvc remote add -d storage <url>  # Add remote storage"