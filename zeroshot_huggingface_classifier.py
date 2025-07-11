from transformers import pipeline
from PIL import Image
import os
import pandas as pd
import argparse
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Zero-shot image classification using HuggingFace transformers")

    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to classify")

    parser.add_argument(
        "--candidate_labels",
        nargs="+",
        default=["fire", "smoke", "no_fire_or_smoke"],
        help="List of candidate labels for classification (default: fire smoke no_fire_or_smoke)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HuggingFace model checkpoint to use (default: openai/clip-vit-large-patch14)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="huggingface_classification_results.csv",
        help="Output CSV file path (default: classification_results.csv)",
    )

    return parser.parse_args()


def main(args):
    """Main function to run the zero-shot image classification."""
    # Initialize the model
    print(f"Loading model: {args.model}")
    detector = pipeline(model=args.model, task="zero-shot-image-classification")

    # Directory containing all images to process
    image_dir = f"data/{args.image_dir}"

    # Validate image directory exists
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory '{image_dir}' does not exist")

    # Get list of all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print(f"No image files found in directory: {image_dir}")
        return

    # Initialize lists to store results
    image_names = []
    all_predictions = []
    predictions = []

    # Process each image
    print(f"Processing images with candidate labels: {args.candidate_labels}")
    for image_file in tqdm(image_files):
        # Load image
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        # Get prediction from classifier
        result = detector(image, candidate_labels=args.candidate_labels)

        # all predictions
        current_predictions = result

        # Get the top prediction (highest score)
        top_prediction = result[0]["label"]  # Get the label with the highest score

        # Store results
        image_names.append(image_file)
        all_predictions.append(current_predictions)
        predictions.append(top_prediction)

    # Create dataframe with results
    results_df = pd.DataFrame(
        {"image_name": image_names, "all_predictions": all_predictions, "prediction": predictions}
    )

    print(f"\nProcessing complete! Results for {len(results_df)} images:")
    print(results_df.head(10))
    print("\nPrediction distribution:")
    print(results_df["prediction"].value_counts())

    # Save results to CSV
    save_dir = f"classification_results/{args.image_dir}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.output)
    results_df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
