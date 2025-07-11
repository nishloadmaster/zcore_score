import argparse


def load_scores_csv(dataset_name: str):
    try:
        import pandas as pd  # Import here to handle cases where pandas isn't installed

        df = pd.read_csv(f"results/{dataset_name}/score.csv")
        return df
    except ImportError:
        print("Error: pandas is required but not installed. Please install with: pip install pandas")
        return None
    except Exception as e:
        print(f"Error loading scores from {dataset_name}: {e}")
        return None


def analyze_dataset(df):
    """Analyze and print dataset statistics"""
    print("CSV loaded successfully.")
    print(df.head())

    # Examine the data structure
    print("Dataset info:")
    print(f"Total samples: {len(df)}")
    print(f"Score range: {df['zcore_score'].min():.4f} to {df['zcore_score'].max():.4f}")
    print("\nUnique class directories:")
    unique_dirs = df["filepath"].apply(lambda x: x.split("/")[-2]).unique()
    print(unique_dirs)

    # Count samples per class
    class_counts = df["filepath"].apply(lambda x: x.split("/")[-2]).value_counts()
    print("\nSamples per class:")
    print(class_counts)


def sample_data_by_score(df, prune_fraction=0.5, prune_type="top_k"):
    """
    Sample data based on zcore scores, keeping the top prune_fraction of samples.

    Args:
        df: DataFrame with 'filepath' and 'zcore_score' columns
        prune_fraction: fraction of data to keep (0.0 to 1.0)
        prune_type: sampling strategy ('top_k' or 'weighted')

    Returns:
        DataFrame with sampled data
    """
    # Sort by zcore_score in descending order (higher scores first)
    df_sorted = df.sort_values("zcore_score", ascending=False)

    # Calculate number of samples to keep
    n_samples = int(len(df_sorted) * prune_fraction)

    if prune_type == "top_k":
        # Take top samples
        sampled_df = df_sorted.head(n_samples).copy()
    elif prune_type == "weighted":
        # Calculate weights based on zcore_score
        weights = df_sorted["zcore_score"] / df_sorted["zcore_score"].sum()
        sampled_df = df_sorted.sample(n=n_samples, weights=weights, random_state=42).copy()
    else:
        raise ValueError("Unsupported prune_type. Use 'top_k' or 'weighted'.")

    print(f"Original dataset: {len(df)} samples")
    print(f"Sampled dataset: {len(sampled_df)} samples ({prune_fraction * 100:.1f}%)")

    # Show class distribution in sampled data
    sampled_class_counts = sampled_df["filepath"].apply(lambda x: x.split("/")[-2]).value_counts()
    print("\nSampled class distribution:")
    print(sampled_class_counts)

    return sampled_df


def main():
    parser = argparse.ArgumentParser(description="Sample data based on zcore scores")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="proc_4d2bb494-3b0f-42d0-ad5e-e9d6053f604e",
    )
    parser.add_argument(
        "--prune_fraction", type=float, default=0.3, help="Fraction of data to keep (0.0 to 1.0, default: 0.3)"
    )
    parser.add_argument(
        "--prune_type",
        type=str,
        default="top_k",
        choices=["top_k", "weighted"],
        help="Sampling strategy: top_k or weighted (default: top_k)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Optional output file path to save the sampled data"
    )

    args = parser.parse_args()

    # Validate prune_fraction
    if not 0.0 <= args.prune_fraction <= 1.0:
        raise ValueError("prune_fraction must be between 0.0 and 1.0")

    # Load the data
    df = load_scores_csv(args.dataset_name)
    if df is None:
        print("Failed to load CSV.")
        return

    # Analyze the dataset
    analyze_dataset(df)

    # Sample the data
    pruned_df = sample_data_by_score(df, prune_fraction=args.prune_fraction, prune_type=args.prune_type)

    print("\nTop 10 samples after pruning:")
    print(pruned_df.head(10))

    # Save output if specified
    if args.output_file:
        pruned_df.to_csv(args.output_file, index=False)
        print(f"\nSampled data saved to: {args.output_file}")


if __name__ == "__main__":
    main()
