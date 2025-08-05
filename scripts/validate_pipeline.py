#!/usr/bin/env python3
"""
Pipeline Validation Script

This script validates the DVC pipeline configuration and checks that all
necessary files and dependencies are in place.
"""

import yaml
import json
import os
from pathlib import Path
import sys


def validate_dvc_pipeline():
    """Validate the DVC pipeline configuration."""
    print("🔍 Validating DVC pipeline configuration...")
    
    # Check if dvc.yaml exists
    dvc_file = Path("dvc.yaml")
    if not dvc_file.exists():
        print("❌ dvc.yaml not found")
        return False
    
    # Load and validate DVC configuration
    try:
        with open(dvc_file) as f:
            dvc_config = yaml.safe_load(f)
        
        if "stages" not in dvc_config:
            print("❌ No stages defined in dvc.yaml")
            return False
        
        stages = dvc_config["stages"]
        print(f"✅ Found {len(stages)} stages in pipeline")
        
        # Validate each stage
        for stage_name, stage_config in stages.items():
            print(f"  📋 Validating stage: {stage_name}")
            
            # Check required fields
            if "cmd" not in stage_config:
                print(f"    ❌ Missing 'cmd' in stage {stage_name}")
                return False
            
            # Check dependencies exist (for stages that have them)
            if "deps" in stage_config:
                for dep in stage_config["deps"]:
                    if not dep.startswith("src/"):  # Skip source files check for now
                        continue
                    if not Path(dep).exists():
                        print(f"    ⚠️  Dependency {dep} not found (may be created during pipeline)")
            
            print(f"    ✅ Stage {stage_name} is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating dvc.yaml: {e}")
        return False


def validate_params_file():
    """Validate the parameters file."""
    print("\n🔍 Validating parameters file...")
    
    params_file = Path("params.yaml")
    if not params_file.exists():
        print("❌ params.yaml not found")
        return False
    
    try:
        with open(params_file) as f:
            params = yaml.safe_load(f)
        
        # Check required parameter sections
        required_sections = ["poisoning", "training", "validation", "visualization"]
        for section in required_sections:
            if section not in params:
                print(f"❌ Missing parameter section: {section}")
                return False
            print(f"  ✅ Found section: {section}")
        
        # Validate specific parameters
        if "levels" not in params["poisoning"]:
            print("❌ Missing poisoning levels configuration")
            return False
        
        poison_levels = params["poisoning"]["levels"]
        if not all(0 <= level <= 1 for level in poison_levels):
            print("❌ Invalid poison levels (must be between 0 and 1)")
            return False
        
        print(f"  ✅ Poison levels: {poison_levels}")
        print("✅ Parameters file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating params.yaml: {e}")
        return False


def validate_source_files():
    """Validate that all required source files exist."""
    print("\n🔍 Validating source files...")
    
    required_files = [
        "src/poison_data.py",
        "src/validate_labels.py", 
        "src/train_model.py",
        "src/evaluate_model.py",
        "src/visualize_results.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} not found")
            all_exist = False
    
    return all_exist


def validate_directory_structure():
    """Validate the expected directory structure."""
    print("\n🔍 Validating directory structure...")
    
    required_dirs = ["src", "tests", "scripts"]
    optional_dirs = ["data", "artifacts", "reports", "visualizations", "mlruns"]
    
    all_good = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ not found")
            all_good = False
    
    for dir_name in optional_dirs:
        if Path(dir_name).exists():
            print(f"  ✅ {dir_name}/ (optional)")
        else:
            print(f"  📁 {dir_name}/ (will be created during pipeline execution)")
    
    return all_good


def validate_requirements():
    """Validate requirements file."""
    print("\n🔍 Validating requirements...")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open(req_file) as f:
            requirements = f.read()
        
        # Check for essential packages
        essential_packages = [
            "scikit-learn", "pandas", "numpy", "matplotlib", 
            "pytest", "mlflow", "dvc"
        ]
        
        missing_packages = []
        for package in essential_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing essential packages: {missing_packages}")
            return False
        
        print("✅ All essential packages found in requirements.txt")
        return True
        
    except Exception as e:
        print(f"❌ Error validating requirements.txt: {e}")
        return False


def check_git_status():
    """Check git status and provide recommendations."""
    print("\n🔍 Checking git status...")
    
    if not Path(".git").exists():
        print("⚠️  Not a git repository. Consider running: git init")
        return False
    
    print("✅ Git repository detected")
    
    # Check for .gitignore
    if Path(".gitignore").exists():
        print("✅ .gitignore found")
    else:
        print("⚠️  .gitignore not found. Consider adding one for Python projects")
    
    return True


def generate_setup_commands():
    """Generate recommended setup commands."""
    print("\n📋 Recommended setup commands:")
    print("="*50)
    
    commands = [
        "# Install dependencies",
        "pip install -r requirements.txt",
        "",
        "# Initialize DVC (if not done)",
        "dvc init",
        "",
        "# Create necessary directories",
        "mkdir -p data reports visualizations artifacts",
        "",
        "# Run the complete pipeline",
        "dvc repro",
        "",
        "# Check pipeline status",
        "dvc status",
        "",
        "# View pipeline DAG",
        "dvc dag",
        "",
        "# Show metrics",
        "dvc metrics show",
        "",
        "# Start MLflow UI",
        "mlflow ui --host 0.0.0.0 --port 5000",
    ]
    
    for cmd in commands:
        print(cmd)


def main():
    """Main validation function."""
    print("🧪 IRIS Pipeline Validation")
    print("="*40)
    
    # Change to script directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    validation_results = []
    
    # Run all validations
    validation_results.append(("DVC Pipeline", validate_dvc_pipeline()))
    validation_results.append(("Parameters", validate_params_file()))
    validation_results.append(("Source Files", validate_source_files()))
    validation_results.append(("Directory Structure", validate_directory_structure()))
    validation_results.append(("Requirements", validate_requirements()))
    validation_results.append(("Git Setup", check_git_status()))
    
    # Summary
    print("\n" + "="*40)
    print("📊 VALIDATION SUMMARY")
    print("="*40)
    
    all_passed = True
    for check_name, passed in validation_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<20} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All validations passed! Pipeline is ready to run.")
    else:
        print("\n⚠️  Some validations failed. Please address the issues above.")
    
    generate_setup_commands()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())