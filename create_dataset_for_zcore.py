#!/usr/bin/env python3
"""
Script to create organized dataset structure from prediction results.

This script reads the prediction results CSV file and organizes images into
directories based on their predicted classes in the format:
data/proc_<dataset_name>/<dataset_name>/train/<class_name>/

The script handles:
- Reading CSV with image_name, all_predictions, and prediction columns
- Extracting all unique classes from predictions
- Creating directory structure
- Copying images to appropriate class directories
"""

import pandas as pd
import shutil
import json
import ast
from pathlib import Path
import argparse
from typing import Set, Dict


def extract_unique_classes(df: pd.DataFrame) -> Set[str]:
    """
    Extract all unique class labels from the dataframe.

    Args:
        df: DataFrame with 'prediction' and 'all_predictions' columns

    Returns:
        Set of unique class labels
    """
    unique_classes = set()

    # Add classes from prediction column
    unique_classes.update(df["prediction"].unique())

    # Add classes from all_predictions column
    for predictions_str in df["all_predictions"]:
        try:
            # Parse the string representation of list of dicts
            predictions = ast.literal_eval(predictions_str)
            for pred_dict in predictions:
                if "label" in pred_dict:
                    unique_classes.add(pred_dict["label"])
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse predictions: {predictions_str}")
            print(f"Error: {e}")
            continue

    return unique_classes


def create_directory_structure(base_path: Path, dataset_name: str, classes: Set[str]) -> None:
    """
    Create the directory structure for the organized dataset.

    Args:
        base_path: Base path where to create the structure
        dataset_name: Name of the dataset
        classes: Set of class names
    """
    proc_dataset_path = base_path / f"proc_{dataset_name}"
    train_path = proc_dataset_path / "train"

    # Create base directories
    train_path.mkdir(parents=True, exist_ok=True)

    # Create class directories
    for class_name in classes:
        class_path = train_path / class_name
        class_path.mkdir(exist_ok=True)
        print(f"Created directory: {class_path}")


def copy_images_to_classes(
    df: pd.DataFrame, source_data_path: Path, target_base_path: Path, dataset_name: str
) -> Dict[str, int]:
    """
    Copy images to their respective class directories based on predictions.

    Args:
        df: DataFrame with image predictions
        source_data_path: Path to source images
        target_base_path: Base path for processed dataset
        dataset_name: Name of the dataset

    Returns:
        Dictionary with class names and image counts
    """
    proc_dataset_path = target_base_path / f"proc_{dataset_name}"
    train_path = proc_dataset_path / "train"

    class_counts = {}
    successful_copies = 0
    failed_copies = 0

    for _, row in df.iterrows():
        image_name = row["image_name"]
        predicted_class = row["prediction"]

        source_image_path = source_data_path / image_name
        target_class_path = train_path / predicted_class
        target_image_path = target_class_path / image_name

        # Check if source image exists
        if not source_image_path.exists():
            print(f"Warning: Source image not found: {source_image_path}")
            failed_copies += 1
            continue

        try:
            # Copy image to target directory
            shutil.copy2(source_image_path, target_image_path)
            successful_copies += 1

            # Update class counts
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1

        except Exception as e:
            print(f"Error copying {image_name} to {predicted_class}: {e}")
            failed_copies += 1

    print("\nCopy summary:")
    print(f"Successful copies: {successful_copies}")
    print(f"Failed copies: {failed_copies}")

    return class_counts


def save_dataset_info(
    target_base_path: Path, dataset_name: str, classes: Set[str], class_counts: Dict[str, int]
) -> None:
    """
    Save dataset information to a JSON file.

    Args:
        target_base_path: Base path for processed dataset
        dataset_name: Name of the dataset
        classes: Set of all classes
        class_counts: Dictionary with class image counts
    """
    proc_dataset_path = target_base_path / f"proc_{dataset_name}"
    info_file = proc_dataset_path / "dataset_info.json"

    info = {
        "dataset_name": dataset_name,
        "total_classes": len(classes),
        "classes": sorted(list(classes)),
        "class_counts": class_counts,
        "total_images": sum(class_counts.values()),
        "structure": f"{dataset_name}/train/<class_name>/",
    }

    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset info saved to: {info_file}")


def process_dataset(dataset_name: str, base_path: str = "/home/nc/zcore_score") -> None:
    """
    Main function to process a dataset and create organized structure.

    Args:
        dataset_name: Name of the dataset to process
        base_path: Base path of the project
    """
    base_path = Path(base_path)

    # Define paths
    results_path = base_path / "classification_results" / dataset_name
    data_path = base_path / "data" / dataset_name
    csv_file = results_path / "huggingface_classification_results.csv"

    # Check if required files exist
    if not csv_file.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_file}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    print(f"Processing dataset: {dataset_name}")
    print(f"Reading results from: {csv_file}")
    print(f"Source images from: {data_path}")

    # Read the CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} image predictions")

    # Extract unique classes
    unique_classes = extract_unique_classes(df)
    print(f"Found {len(unique_classes)} unique classes: {sorted(unique_classes)}")

    # Create directory structure
    create_directory_structure(base_path / "data", dataset_name, unique_classes)

    # Copy images to class directories
    class_counts = copy_images_to_classes(df, data_path, base_path / "data", dataset_name)

    # Print class distribution
    print("\nClass distribution:")
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        print(f"  {class_name}: {count} images")

    # Save dataset information
    save_dataset_info(base_path / "data", dataset_name, unique_classes, class_counts)

    print("\nDataset processing completed!")
    print(f"Organized dataset available at: {base_path}/data/proc_{dataset_name}/train/")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Create organized dataset structure from prediction results")
    parser.add_argument(
        "dataset_name", help="Name of the dataset to process (should match directory names in data/ and results/)"
    )
    parser.add_argument(
        "--base-path", default="/home/nc/zcore_score", help="Base path of the project (default: /home/nc/zcore_score)"
    )

    args = parser.parse_args()

    try:
        process_dataset(args.dataset_name, args.base_path)
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
