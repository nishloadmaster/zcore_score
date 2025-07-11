import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import os
import pickle
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image


class YOLOEmbeddingModel:
    """
    Custom FiftyOne-compatible wrapper for YOLO models to extract embeddings
    """

    def __init__(self, model_name="yolov8n.pt", feature_layer=-2):
        self.model = YOLO(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.feature_layer = feature_layer
        self.features = []

        # Setup hook for feature extraction
        self.hook_handle = None
        self._setup_hook()

    def _setup_hook(self):
        """Setup forward hook to extract features from specified layer"""

        def hook_fn(module, input, output):
            pooled = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze()
            self.features.append(pooled.detach().cpu())

        self.hook_handle = self.model.model.model[self.feature_layer].register_forward_hook(hook_fn)

    def _cleanup_hook(self):
        """Remove the forward hook"""
        if self.hook_handle:
            self.hook_handle.remove()

    def __call__(self, image_paths):
        """
        Extract embeddings from images (FiftyOne compatible interface)

        Args:
            image_paths: List of image file paths or single path

        Returns:
            numpy array of embeddings
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        self.features = []
        transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])

        successful_indices = []

        with torch.no_grad():
            for i, img_path in enumerate(image_paths):
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert("RGB")
                    tensor_img = transform(image).unsqueeze(0).to(self.device)

                    # Forward pass to trigger hook
                    _ = self.model(tensor_img)
                    successful_indices.append(i)

                except Exception as e:
                    print(f"Warning: Failed to process {img_path}: {e}")
                    # Continue without adding to features - we'll handle this below

        if len(self.features) == 0:
            raise ValueError("No images could be processed successfully")

        # Get the feature size from successful extractions
        feature_size = self.features[0].shape[0]

        # Create final embeddings array with proper size
        final_embeddings = []
        feature_idx = 0

        for i in range(len(image_paths)):
            if i in successful_indices:
                final_embeddings.append(self.features[feature_idx])
                feature_idx += 1
            else:
                # Add zero embedding for failed images
                final_embeddings.append(torch.zeros(feature_size))

        # Stack all features and convert to numpy
        embeddings = torch.vstack(final_embeddings).numpy()
        return embeddings if len(image_paths) > 1 else embeddings[0]

    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup_hook()


def load_fo_dataset(args):
    if "cifar" in args.dataset:
        dataset = foz.load_zoo_dataset(args.dataset, split="train")
    else:
        if args.dataset == "imagenet":
            path = os.path.join(args.data_dir, "imagenet", "ILSVRC", "Data", "CLS-LOC", "train")
        elif "eurosat" in args.dataset:
            path = os.path.join(args.data_dir, args.dataset, "train")
        elif args.dataset == "fire_detection":
            path = os.path.join(args.data_dir, "fire_detection", "train")
        else:
            raise ValueError(f"{args.dataset} not recognized.")

        dataset = fo.Dataset.from_dir(path, dataset_type=fo.types.ImageClassificationDirectoryTree)

    return dataset


def load_fo_model(args, model_name):
    if model_name == "clip":
        model = foz.load_zoo_model(
            "open-clip-torch",
            clip_model="ViT-L-14",
            pretrained="openai",
        )
    elif model_name == "resnet18":
        model = foz.load_zoo_model("resnet18-imagenet-torch")
    elif model_name == "dinov2":
        model = foz.load_zoo_model("dinov2-vitb14-torch")
    elif model_name == "yolo":
        # YOLO models use a custom wrapper since they're not in FiftyOne zoo
        model = YOLOEmbeddingModel()
    else:
        model = foz.load_zoo_model(model_name)

    return model


def generate_embedding(args, model_name, embed_file):
    dataset = load_fo_dataset(args)
    model = load_fo_model(args, model_name)

    print(f"Generating {args.dataset}-{model_name} embeddings.")

    # Get the filepaths in the same order as embeddings
    filepaths = [sample.filepath for sample in dataset]

    if model_name == "yolo":
        # YOLO models use custom embedding extraction
        model_embed = model(filepaths)
    else:
        # Standard FiftyOne models
        model_embed = dataset.compute_embeddings(model)

    # Save embeddings with metadata
    embed_data = {"embeddings": model_embed, "filepaths": filepaths}

    os.makedirs(os.path.dirname(embed_file), exist_ok=True)
    pickle.dump(embed_data, open(embed_file, "wb"))
    print(f"Model embeddings saved at {embed_file}.")

    return embed_data


def get_model_embedding(args):
    embed_dir = os.path.join(args.data_dir, "preprocess", args.dataset)

    all_filepaths = None

    for i, model_name in enumerate(args.embedding):
        embed_file = os.path.join(embed_dir, f"{model_name}_embedding.pk")
        if os.path.exists(embed_file):
            embed_data = pickle.load(open(embed_file, "rb"))
        else:
            embed_data = generate_embedding(args, model_name, embed_file)

        # Handle both old format (just embeddings) and new format (dict)
        if isinstance(embed_data, dict):
            model_embed = embed_data["embeddings"]
            filepaths = embed_data["filepaths"]
        else:
            # Old format - just embeddings, no filepaths
            model_embed = embed_data
            filepaths = None

        if i == 0:
            total_embed = model_embed
            all_filepaths = filepaths
        else:
            # Ensure same order if filepaths are available
            if all_filepaths is not None and filepaths is not None:
                assert all_filepaths == filepaths, "File order mismatch between embeddings!"
            total_embed = np.concatenate((total_embed, model_embed), axis=1)

    return {"embeddings": total_embed, "filepaths": all_filepaths}
