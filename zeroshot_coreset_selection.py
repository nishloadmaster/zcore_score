import argparse
import numpy as np
import os
import pandas as pd

import core.coreset as cs

def main(args):

    exp_name, exp_file = cs.experiment_name(args)
    assert not os.path.exists(exp_file), f"{exp_file} already exists."

    embed_data = cs.get_model_embedding(args)
    
    # Handle both old format (just embeddings) and new format (dict with filepaths)
    if isinstance(embed_data, dict):
        embeddings = embed_data['embeddings']
        filepaths = embed_data['filepaths']
    else:
        embeddings = embed_data
        filepaths = None
    
    print(f"Embeddings shape: {embeddings.shape}")
    if filepaths is not None:
        print(f"Number of filepaths: {len(filepaths)}")
    else:
        print("No filepaths available")
    
    scores = cs.zcore_score(args, embeddings)
    print(f"Number of scores: {len(scores)}")
    
    # Create results dictionary with scores and filenames
    results = {
        'scores': scores,
        'filepaths': filepaths
    }
    
    # Save as .npz to store multiple arrays
    exp_file_npz = exp_file.replace('.npy', '.npz')
    np.savez(exp_file_npz, **results)
    print(f"\nZCore scores and filepaths saved at {exp_file_npz}")
    
    # Optionally, also save as CSV for easy viewing
    if filepaths is not None:
        # Validate that filepaths and scores have the same length
        if len(filepaths) == len(scores):
            df = pd.DataFrame({
                'filepath': filepaths,
                'zcore_score': scores
            })
            csv_file = exp_file.replace('.npy', '.csv')
            df.to_csv(csv_file, index=False)
            print(f"Results also saved as CSV at {csv_file}")
        else:
            print(f"Warning: Length mismatch - {len(filepaths)} filepaths vs {len(scores)} scores. CSV not saved.")
            print("This may indicate an issue with embedding extraction. Only .npz file saved.")
    else:
        print("Warning: No filepaths available. Only scores saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Zero-Shot Coreset Selection (ZCore)")

    parser.add_argument("--trial", type=int, default=0)

    # Dataset.
    dataset_choice = ["cifar10", "cifar100", "imagenet", "eurosat10", 
                      "eurosat20", "eurosat40", "eurosat80", "fire_detection"]
    parser.add_argument("--dataset", type=str, choices=dataset_choice)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=2)

    # ZCore Parameters (see paper for more details).
    parser.add_argument("--embedding", type=str, nargs="+", 
                        choices=["resnet18", "clip", "dinov2", "yolo"])
    parser.add_argument("--n_sample", type=int, default=int(1e6))
    parser.add_argument("--sample_dim", type=int, default=2)
    parser.add_argument("--no_rand_init", dest="rand_init", 
                        action="store_false", default=True)
    parser.add_argument("--redund_exp", type=int, default=4)
    parser.add_argument("--redund_nn", type=int, default=1000)

    args = parser.parse_args()
    main(args)

