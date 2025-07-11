import torch
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import argparse
import os
import sys

# Add core modules to path for integration with repository
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def extract_yolo_embeddings(data_path, output_path, model_name="yolov8n.pt", 
                           feature_layer=-2, batch_size=16):
    """
    Extract YOLO embeddings from images in a directory.
    
    Args:
        data_path (str): Path to image directory  
        output_path (str): Path to save embeddings
        model_name (str): YOLO model to use
        feature_layer (int): Layer to extract features from
        batch_size (int): Batch size for processing
    """
    
    # Load YOLOv8 model
    model = YOLO(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Hook to store intermediate output
    features = []

    def hook_fn(module, input, output):
        pooled = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze()
        features.append(pooled.detach().cpu())

    # Register hook on the chosen layer
    handle = model.model.model[feature_layer].register_forward_hook(hook_fn)

    # Dataset
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    dataset = ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Extracting YOLO embeddings from {len(dataset)} images...")
    
    # Run forward pass to extract features
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            _ = model(images)  # Triggers hook

    # Clean up
    handle.remove()

    # Stack all collected features
    if len(features) > 0:
        embeddings = torch.vstack(features).numpy()
    else:
        raise ValueError("No features extracted! Check your data path and model.")

    # Save embeddings and corresponding filenames
    embedding_data = {
        "embeddings": embeddings,
        "image_paths": [sample[0] for sample in dataset.samples],  # full paths
        "model_name": model_name,
        "feature_layer": feature_layer
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(embedding_data, f)
    
    print(f"YOLO embeddings saved to {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embedding_data

def main():
    parser = argparse.ArgumentParser(description="Extract YOLO embeddings for ZCore")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to image directory")
    parser.add_argument("--output_path", type=str, default="yolov8_embeddings.pk",
                       help="Path to save embeddings")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="YOLO model to use")
    parser.add_argument("--feature_layer", type=int, default=-2,
                       help="Layer to extract features from")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path {args.data_path} does not exist!")
    
    extract_yolo_embeddings(
        data_path=args.data_path,
        output_path=args.output_path,
        model_name=args.model,
        feature_layer=args.feature_layer,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
