#!/usr/bin/env python3
"""
Data Poisoning Script for IRIS Dataset

This script implements label flipping attacks on the IRIS dataset by randomly
changing target labels at specified poison levels (5%, 10%, 50%).

Author: MLOps Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import argparse
import os
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def poison_labels(input_path: str, output_path: str, poison_level: float, random_seed: int = 42) -> None:
    """
    Loads a CSV file and flips target labels for a specified percentage of rows.
    
    This function implements a label flipping attack by randomly selecting
    rows and changing their target labels to different classes. This simulates
    a data poisoning attack commonly seen in adversarial ML scenarios.

    Args:
        input_path (str): Path to the original CSV file
        output_path (str): Path to save the poisoned CSV file
        poison_level (float): Fraction of labels to flip (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility

    Raises:
        ValueError: If poison_level is not between 0.0 and 1.0
        FileNotFoundError: If input file doesn't exist
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Validate inputs
    if not 0.0 <= poison_level <= 1.0:
        raise ValueError(f"Poison level must be between 0.0 and 1.0, got {poison_level}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading dataset from: {input_path}")
    
    # Load the dataset
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Determine target column (assume last column is target)
    target_column = df.columns[-1]
    logger.info(f"Target column: {target_column}")
    
    # Get unique labels in the target column
    unique_labels = df[target_column].unique().tolist()
    logger.info(f"Unique labels: {unique_labels}")
    
    if len(unique_labels) < 2:
        logger.error("Cannot flip labels with less than two unique classes")
        return

    # Calculate number of rows to poison
    num_rows = len(df)
    num_to_poison = int(num_rows * poison_level)

    if num_to_poison == 0 and poison_level > 0:
        logger.warning(f"Poison level {poison_level * 100}% is too low to select any rows")
        df.to_csv(output_path, index=False)
        return

    logger.info(f"Poisoning {num_to_poison} of {num_rows} rows ({poison_level * 100:.2f}%)")
    
    # Create a copy to modify
    df_poisoned = df.copy()

    # Select random row indices to poison without replacement
    poison_indices = np.random.choice(df.index, size=num_to_poison, replace=False)
    
    # Track label changes for reporting
    label_changes = {}
    for label in unique_labels:
        label_changes[label] = {new_label: 0 for new_label in unique_labels if new_label != label}

    # Flip the label for each selected row
    for idx in poison_indices:
        original_label = df_poisoned.loc[idx, target_column]
        
        # Create list of possible new labels (all labels except original)
        possible_new_labels = [label for label in unique_labels if label != original_label]
        
        # Choose a new label randomly
        new_label = random.choice(possible_new_labels)
        
        # Apply the flipped label
        df_poisoned.loc[idx, target_column] = new_label
        
        # Track the change
        label_changes[original_label][new_label] += 1
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save poisoned dataset
    df_poisoned.to_csv(output_path, index=False)
    logger.info(f"Poisoned dataset saved to: {output_path}")
    
    # Report label changes
    logger.info("Label flip summary:")
    for original, changes in label_changes.items():
        for new_label, count in changes.items():
            if count > 0:
                logger.info(f"  {original} -> {new_label}: {count} instances")
    
    # Save metadata about poisoning
    metadata = {
        "original_file": input_path,
        "poisoned_file": output_path,
        "poison_level": poison_level,
        "total_rows": num_rows,
        "poisoned_rows": num_to_poison,
        "random_seed": random_seed,
        "label_changes": label_changes
    }
    
    metadata_path = output_path.replace('.csv', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to: {metadata_path}")


def poison_multiple_levels(input_path: str, output_dir: str = "data/poisoned", 
                          poison_levels: list = None) -> None:
    """
    Create multiple poisoned datasets with different poison levels.
    
    Args:
        input_path (str): Path to original dataset
        output_dir (str): Directory to save poisoned datasets
        poison_levels (list): List of poison levels to create
    """
    if poison_levels is None:
        poison_levels = [0.05, 0.10, 0.50]  # 5%, 10%, 50%
    
    os.makedirs(output_dir, exist_ok=True)
    
    for level in poison_levels:
        level_pct = int(level * 100)
        output_path = os.path.join(output_dir, f"iris_poisoned_{level_pct}pct.csv")
        logger.info(f"Creating {level_pct}% poisoned dataset...")
        poison_labels(input_path, output_path, level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poison a dataset by flipping target labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 5% poisoned dataset
  python src/poison_data.py --poison-level 0.05
  
  # Create 10% poisoned dataset with custom paths
  python src/poison_data.py --input-path data/iris.csv --output-path data/iris_10pct.csv --poison-level 0.10
  
  # Create multiple poison levels
  python src/poison_data.py --multiple-levels
        """
    )
    
    parser.add_argument(
        "--input-path", 
        type=str, 
        default="data/iris.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="data/iris_poisoned.csv",
        help="Path for the poisoned output CSV"
    )
    parser.add_argument(
        "--poison-level", 
        type=float,
        help="Fraction of labels to flip (e.g., 0.05 for 5%%)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--multiple-levels",
        action="store_true",
        help="Create datasets with multiple poison levels (5%%, 10%%, 50%%)"
    )
    
    args = parser.parse_args()
    
    if args.multiple_levels:
        poison_multiple_levels(args.input_path)
    elif args.poison_level is not None:
        poison_labels(args.input_path, args.output_path, args.poison_level, args.random_seed)
    else:
        parser.error("Either --poison-level or --multiple-levels must be specified")