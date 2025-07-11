#!/bin/bash

# Script to run the zcore pipeline in sequence
# Usage: ./run_pipeline.sh <dataset_name>

set -e  # Exit on any error

# Check if dataset name is provided
if [ $# -eq 0 ]; then
    echo "Error: Dataset name is required"
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"

echo "Starting zcore pipeline for dataset: $DATASET_NAME"
echo "============================================="

# Step 1: Run zeroshot huggingface classifier
echo "Step 1: Running zeroshot huggingface classifier..."
python3 zeroshot_huggingface_classifier.py --image_dir "$DATASET_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed"
    exit 1
fi
echo "Step 1 completed successfully"
echo

# Step 2: Create dataset for zcore
echo "Step 2: Creating dataset for zcore..."
python3 create_dataset_for_zcore.py "$DATASET_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Step 2 failed"
    exit 1
fi
echo "Step 2 completed successfully"
echo

# Step 3: Run zeroshot coreset selection
echo "Step 3: Running zeroshot coreset selection..."
python3 zeroshot_coreset_selection.py --dataset "proc_$DATASET_NAME" --data_dir ./data --results_dir ./results --embedding yolo --num_workers 10
if [ $? -ne 0 ]; then
    echo "Error: Step 3 failed"
    exit 1
fi
echo "Step 3 completed successfully"
echo

# Step 4: Run top k sampler
echo "Step 4: Running top k sampler..."
python3 top_k_sampler.py --dataset_name "proc_$DATASET_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Step 4 failed"
    exit 1
fi
echo "Step 4 completed successfully"
echo

echo "============================================="
echo "Pipeline completed successfully for dataset: $DATASET_NAME"
