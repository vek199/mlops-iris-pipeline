# 🌩️ Google Cloud Shell Deployment Guide

## 🚨 Storage Space Issue Resolution

If you encounter "No space left on device" errors in Cloud Shell, follow this optimized deployment guide.

## 🛠️ Quick Solutions

### **Option 1: Minimal Installation (Recommended for Cloud Shell)**

```bash
# Clean up space first
rm -rf .venv  # Remove any existing virtual environment
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Use the optimized setup script
bash scripts/cloudshell_setup.sh
```

### **Option 2: Manual Space Management**

```bash
# Check available space
df -h

# Clean up Cloud Shell
rm -rf ~/.cache/pip/*  # Clear pip cache
docker system prune -f  # Clean Docker if using containers

# Create minimal environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages one by one without cache
pip install --no-cache-dir scikit-learn
pip install --no-cache-dir pandas numpy
pip install --no-cache-dir matplotlib
pip install --no-cache-dir pytest
```

### **Option 3: Use Minimal Requirements**

```bash
# Use the lightweight requirements file
pip install --no-cache-dir -r requirements-minimal.txt
```

## 🎯 Space-Efficient Workflow

### **Step 1: Setup (Space-Optimized)**
```bash
# Clone the repository
git clone <your-repo-url>
cd mlops-iris-pipeline

# Run optimized setup
bash scripts/cloudshell_setup.sh
```

### **Step 2: Quick Demo**
```bash
# Activate environment
source .venv/bin/activate

# Run lightweight demo
python cloudshell_demo.py
```

### **Step 3: Core Experiments**
```bash
# Create poisoned data (5% only to save space)
python src/poison_data.py --poison-level 0.05

# Train models (minimal config)
python src/train_model.py --data-path data/iris.csv --model random_forest

# Validate labels
python src/validate_labels.py --data-path data/iris_poisoned.csv
```

## 📊 Resource Management Tips

### **Monitor Space Usage**
```bash
# Check disk usage
df -h

# Check directory sizes
du -sh * | sort -hr

# Clean up regularly
find . -name "*.pyc" -delete
rm -rf __pycache__/
```

### **Optimize Package Installation**
```bash
# Install without cache
pip install --no-cache-dir package_name

# Uninstall unused packages
pip uninstall package_name

# Use minimal versions
pip install scikit-learn==1.0.0  # Instead of latest
```

## 🔧 Troubleshooting

### **Error: "No space left on device"**

**Solution 1: Clean temporary files**
```bash
# Clean pip cache
rm -rf ~/.cache/pip

# Clean conda cache (if using conda)
conda clean --all

# Clean system temp
sudo rm -rf /tmp/*
```

**Solution 2: Use Cloud Shell boost**
```bash
# Enable Cloud Shell boost (more resources)
# Click "Boost Cloud Shell" in the Cloud Shell interface
```

**Solution 3: Use persistent disk**
```bash
# Mount persistent disk for larger storage
# Follow Cloud Shell persistent disk documentation
```

### **Package Installation Fails**

**Solution: Install core packages only**
```bash
# Minimal working set
pip install --no-cache-dir scikit-learn pandas numpy matplotlib

# Skip optional packages
# - seaborn (use matplotlib instead)
# - plotly (use matplotlib instead)  
# - mlflow (use simple logging instead)
# - jupyter (use basic Python scripts)
```

## 🎮 Cloud Shell Demo Script

The `cloudshell_demo.py` provides a lightweight demonstration that works with minimal dependencies:

```python
# Run the demo
python cloudshell_demo.py

# Expected output:
# - IRIS dataset loading
# - 10% data poisoning
# - Model training (clean vs poisoned)
# - Performance comparison
# - Results saved to JSON
```

## 📋 Step-by-Step Cloud Shell Workflow

### **Complete Setup Process:**

1. **Initial Setup**
   ```bash
   mkdir week8 && cd week8
   git clone <your-repo-url>
   cd mlops-iris-pipeline
   ```

2. **Space-Efficient Installation**
   ```bash
   bash scripts/cloudshell_setup.sh
   source .venv/bin/activate
   ```

3. **Quick Verification**
   ```bash
   python cloudshell_demo.py
   ```

4. **Run Core Experiments**
   ```bash
   # Data poisoning
   python src/poison_data.py --poison-level 0.05 --poison-level 0.10
   
   # Model training
   python src/train_model.py --data-path data/iris.csv
   python src/train_model.py --data-path data/iris_poisoned.csv
   
   # Label validation
   python src/validate_labels.py --data-path data/iris_poisoned.csv
   ```

5. **View Results**
   ```bash
   cat reports/cloudshell_demo_results.json
   ls -la artifacts/
   ls -la visualizations/
   ```

## ⚡ Performance Optimizations

### **Reduce Model Complexity**
```python
# In training scripts, use smaller models
RandomForestClassifier(n_estimators=50)  # Instead of 100
```

### **Skip Heavy Visualizations**
```python
# Use simple plots instead of complex visualizations
import matplotlib.pyplot as plt  # Instead of seaborn/plotly
```

### **Batch Operations**
```python
# Process smaller batches of data
chunk_size = 1000  # Process in smaller chunks
```

## 🌟 Alternative: Docker Deployment

If Cloud Shell space remains an issue, consider using Docker:

```bash
# Create Dockerfile for minimal environment
FROM python:3.9-slim

WORKDIR /app
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

COPY . .
CMD ["python", "cloudshell_demo.py"]
```

## 📞 Support

If you continue to experience space issues:

1. **Use the minimal requirements**: `requirements-minimal.txt`
2. **Run the optimized setup**: `scripts/cloudshell_setup.sh`  
3. **Use the lightweight demo**: `cloudshell_demo.py`
4. **Enable Cloud Shell boost** for more resources
5. **Consider local development** for full feature testing

The core functionality will work with the minimal setup, providing the essential data poisoning experiment capabilities!