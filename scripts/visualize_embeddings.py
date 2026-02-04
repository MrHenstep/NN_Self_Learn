"""
SimCLR Embedding Visualization and Similarity Analysis

This script extracts embeddings from pretrained and/or finetuned SimCLR models,
as well as directly-trained ResNet models, produces 2D visualizations (t-SNE/UMAP)
colored by CIFAR-10 class, and computes quantitative similarity statistics to
understand the learned embedding space.

Usage (interactive):
    from scripts.visualize_embeddings import main
    main()
    main(n_samples=1000, method="tsne", model="pretrained", show=True)
    
    # Compare all three training approaches:
    main(model="all", direct_checkpoint="checkpoints/resnet20_cifar10.pth")
    
    # Analyze only a directly-trained model:
    main(model="direct", direct_checkpoint="checkpoints/resnet20_cifar10.pth")

Parameters:
    n_samples: Number of samples for dimensionality reduction (default: 2000)
    method: Reduction method - 'tsne', 'umap', or 'both' (default: both)
    model: Model to analyze - 'pretrained', 'finetuned', 'direct', 'all', or 'both' (default: both)
    space: Embedding space - 'features', 'projections', or 'both' (default: both)
    output_dir: Directory to save visualizations (default: visualizations/)
    show: Show plots interactively in addition to saving
    direct_checkpoint: Path to checkpoint for directly-trained ResNet (required if model='direct' or 'all')
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Literal
import warnings

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
if os.getcwd() != str(_ROOT):
    os.chdir(_ROOT)

from torchvision import datasets
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Try importing UMAP, gracefully handle if not available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn not installed. UMAP visualization will be skipped. "
                  "Install with: pip install umap-learn")

from models.cnn.config import ModelConfig, DataMetadata
from models.cnn.architectures.factory import build_model
from models.cnn.architectures.simclr import SimCLRModel, build_simclr_model
from data_loading.contrastive_transforms import simclr_eval_transform

# CIFAR-10 class names (alphabetical order, matching dataset indices)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Display order: vehicles first (airplane, automobile, ship, truck), then animals
# This groups semantically similar classes together in visualizations
DISPLAY_ORDER = [0, 1, 8, 9, 2, 3, 4, 5, 6, 7]
CIFAR10_CLASSES_ORDERED = [CIFAR10_CLASSES[i] for i in DISPLAY_ORDER]

# CIFAR-10 normalization constants
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616])


def build_backbone() -> nn.Module:
    """Build a ResNet-20 backbone for CIFAR-10."""
    model_cfg = ModelConfig(model_name="resnet", resnet_n=3, use_projection=False)
    data_meta = DataMetadata(
        dataset_key="cifar10",
        num_classes=10,
        input_channels=3,
        input_size=32
    )
    return build_model(model_cfg, data_meta)


def load_pretrained_simclr(
    checkpoint_path: str = "checkpoints/simclr_pretrained.pth",
    device: torch.device = torch.device("cpu"),
) -> SimCLRModel:
    """
    Load pretrained SimCLR model (backbone + projection head).
    
    Args:
        checkpoint_path: Path to the pretrained checkpoint
        device: Device to load model on
        
    Returns:
        SimCLRModel with loaded weights
    """
    backbone = build_backbone()
    simclr_model = build_simclr_model(
        backbone,
        projection_dim=128,
        projection_hidden=512,
        use_bn_in_head=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    simclr_model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    simclr_model.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
    
    simclr_model.to(device)
    simclr_model.eval()
    
    print(f"Loaded pretrained SimCLR from {checkpoint_path}")
    return simclr_model


def load_finetuned_backbone(
    checkpoint_path: str = "checkpoints/simclr_finetuned.pth",
    device: torch.device = torch.device("cpu"),
) -> SimCLRModel:
    """
    Load finetuned model and wrap it back into SimCLR structure for embedding extraction.
    
    The finetuned checkpoint contains only the backbone with a classification head.
    We load the backbone weights and create a new projection head (randomly initialized,
    but we only use the backbone features for finetuned model analysis).
    
    Args:
        checkpoint_path: Path to the finetuned checkpoint
        device: Device to load model on
        
    Returns:
        SimCLRModel with finetuned backbone (projection head is new)
    """
    backbone = build_backbone()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    
    # Wrap in SimCLR structure - projection head is randomly initialized
    # For finetuned model, we primarily analyze the backbone features
    simclr_model = build_simclr_model(
        backbone,
        projection_dim=128,
        projection_hidden=512,
        use_bn_in_head=True
    )
    
    simclr_model.to(device)
    simclr_model.eval()
    
    print(f"Loaded finetuned backbone from {checkpoint_path}")
    if 'test_accuracy' in checkpoint:
        print(f"  (Test accuracy: {checkpoint['test_accuracy']:.2%})")
    
    return simclr_model


def load_direct_resnet(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Load a directly-trained ResNet model (no contrastive pre-training).
    
    This loads a ResNet that was trained end-to-end on classification,
    without any SimCLR pre-training phase.
    
    Args:
        checkpoint_path: Path to the checkpoint file. The checkpoint should
            contain either 'model_state_dict' or be a raw state dict.
        device: Device to load model on
        
    Returns:
        ResNet model with loaded weights
    """
    backbone = build_backbone()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        backbone.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume it's a raw state dict
        backbone.load_state_dict(checkpoint)
    
    backbone.to(device)
    backbone.eval()
    
    print(f"Loaded directly-trained ResNet from {checkpoint_path}")
    if isinstance(checkpoint, dict) and 'test_accuracy' in checkpoint:
        print(f"  (Test accuracy: {checkpoint['test_accuracy']:.2%})")
    
    return backbone


def get_cifar10_test_loader(
    batch_size: int = 256,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """
    Create CIFAR-10 test set DataLoader with evaluation transforms.
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for CIFAR-10 test set
    """
    transform = simclr_eval_transform(CIFAR10_MEAN, CIFAR10_STD, image_size=32)
    
    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


@torch.no_grad()
def extract_embeddings(
    model: SimCLRModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features and projections from the model for all data in the dataloader.
    
    Args:
        model: SimCLRModel to extract embeddings from
        dataloader: DataLoader providing (images, labels) tuples
        device: Device to run inference on
        
    Returns:
        Tuple of (features, projections, labels) tensors:
            - features: (N, 64) backbone features before projection
            - projections: (N, 128) normalized projection head outputs
            - labels: (N,) class labels
    """
    model.eval()
    
    all_features = []
    all_projections = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device)
        
        features, projections = model(images, return_features=True)
        
        all_features.append(features.cpu())
        all_projections.append(projections.cpu())
        all_labels.append(labels)
    
    features = torch.cat(all_features, dim=0)
    projections = torch.cat(all_projections, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Extracted {len(labels)} embeddings")
    print(f"  Features shape: {features.shape}")
    print(f"  Projections shape: {projections.shape}")
    
    return features, projections, labels


@torch.no_grad()
def extract_features_only(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract only features from a model (for models without projection heads).
    
    This is used for directly-trained ResNet models that don't have a
    SimCLR projection head.
    
    Args:
        model: ResNet model with forward_features method
        dataloader: DataLoader providing (images, labels) tuples
        device: Device to run inference on
        
    Returns:
        Tuple of (features, labels) tensors:
            - features: (N, 64) backbone features
            - labels: (N,) class labels
    """
    model.eval()
    
    all_features = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        
        features = model.forward_features(images)
        
        all_features.append(features.cpu())
        all_labels.append(labels)
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Extracted {len(labels)} feature embeddings")
    print(f"  Features shape: {features.shape}")
    
    return features, labels


def subsample_embeddings(
    features: torch.Tensor,
    projections: torch.Tensor,
    labels: torch.Tensor,
    n_samples: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly subsample embeddings for faster dimensionality reduction.
    
    Args:
        features: (N, D1) feature embeddings
        projections: (N, D2) projection embeddings
        labels: (N,) class labels
        n_samples: Number of samples to keep
        seed: Random seed for reproducibility
        
    Returns:
        Subsampled (features, projections, labels) tuple
    """
    n_total = len(labels)
    if n_samples >= n_total:
        return features, projections, labels
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_samples, replace=False)
    indices = torch.from_numpy(indices)
    
    return features[indices], projections[indices], labels[indices]


def reduce_dimensionality(
    embeddings: np.ndarray,
    method: Literal["tsne", "umap"] = "tsne",
    n_components: int = 2,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Reduce embeddings to 2D using t-SNE or UMAP.
    
    Args:
        embeddings: (N, D) array of embeddings
        method: Reduction method ('tsne' or 'umap')
        n_components: Output dimensionality (default 2)
        seed: Random seed
        **kwargs: Additional arguments for the reducer
        
    Returns:
        (N, n_components) reduced embeddings
    """
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=seed,
            perplexity=kwargs.get("perplexity", 30),
            max_iter=kwargs.get("max_iter", 1000),
            init="pca",
            learning_rate="auto",
        )
        reduced = reducer.fit_transform(embeddings)
        
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = UMAP(
            n_components=n_components,
            random_state=seed,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            metric=kwargs.get("metric", "cosine"),
        )
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reduced


def plot_embeddings_2d(
    reduced: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Create a 2D scatter plot of embeddings colored by class.
    
    Args:
        reduced: (N, 2) reduced embeddings
        labels: (N,) class labels
        title: Plot title
        output_path: Path to save figure (if provided)
        show: Whether to display plot interactively
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a colormap with distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for class_idx in range(10):
        mask = labels == class_idx
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=[colors[class_idx]],
            label=CIFAR10_CLASSES[class_idx],
            alpha=0.6,
            s=10,
        )
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best", markerscale=2, fontsize=9)
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compute_similarity_stats(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute intra-class and inter-class cosine similarity statistics.
    
    Args:
        embeddings: (N, D) embedding tensor
        labels: (N,) class labels
        normalize: Whether to L2-normalize embeddings before computing similarity
        
    Returns:
        Dictionary with similarity statistics:
            - intra_mean: Mean similarity within same class
            - intra_std: Std of intra-class similarities
            - inter_mean: Mean similarity between different classes
            - inter_std: Std of inter-class similarities
            - separation: intra_mean - inter_mean (higher is better)
    """
    if normalize:
        embeddings = F.normalize(embeddings, dim=1)
    
    # Compute full similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.t())  # (N, N)
    
    # Create masks for intra-class and inter-class pairs
    labels_row = labels.unsqueeze(1)  # (N, 1)
    labels_col = labels.unsqueeze(0)  # (1, N)
    same_class_mask = (labels_row == labels_col)  # (N, N)
    
    # Exclude diagonal (self-similarity)
    n = len(labels)
    diag_mask = torch.eye(n, dtype=torch.bool)
    same_class_mask = same_class_mask & ~diag_mask
    diff_class_mask = ~(labels_row == labels_col)
    
    # Extract similarities
    intra_sims = sim_matrix[same_class_mask]
    inter_sims = sim_matrix[diff_class_mask]
    
    stats = {
        "intra_mean": intra_sims.mean().item(),
        "intra_std": intra_sims.std().item(),
        "inter_mean": inter_sims.mean().item(),
        "inter_std": inter_sims.std().item(),
    }
    stats["separation"] = stats["intra_mean"] - stats["inter_mean"]
    
    return stats


def compute_class_centroids(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute class centroids (mean embedding per class).
    
    Args:
        embeddings: (N, D) embedding tensor
        labels: (N,) class labels
        normalize: Whether to L2-normalize centroids
        
    Returns:
        (num_classes, D) centroid tensor
    """
    num_classes = 10
    dim = embeddings.shape[1]
    centroids = torch.zeros(num_classes, dim)
    
    for c in range(num_classes):
        mask = labels == c
        centroids[c] = embeddings[mask].mean(dim=0)
    
    if normalize:
        centroids = F.normalize(centroids, dim=1)
    
    return centroids


def plot_class_similarity_heatmap(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    title: str,
    output_path: Optional[Path] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot a 10x10 heatmap of class centroid similarities.
    
    Classes are reordered to group vehicles and animals together
    for easier interpretation of semantic relationships.
    
    Args:
        embeddings: (N, D) embedding tensor
        labels: (N,) class labels
        title: Plot title
        output_path: Path to save figure
        show: Whether to display interactively
        figsize: Figure size
    """
    centroids = compute_class_centroids(embeddings, labels, normalize=True)
    
    # Compute similarity matrix between centroids
    sim_matrix = torch.mm(centroids, centroids.t()).numpy()
    
    # Reorder rows and columns to group vehicles and animals together
    sim_matrix = sim_matrix[DISPLAY_ORDER, :][:, DISPLAY_ORDER]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        xticklabels=CIFAR10_CLASSES_ORDERED,
        yticklabels=CIFAR10_CLASSES_ORDERED,
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        square=True,
    )
    
    ax.set_title(title)
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compute_knn_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> float:
    """
    Evaluate embedding quality using k-NN classification accuracy.
    
    Splits data into train/test, fits k-NN on train embeddings,
    and evaluates on test embeddings.
    
    Args:
        embeddings: (N, D) embedding tensor
        labels: (N,) class labels
        k: Number of neighbors
        train_ratio: Fraction of data for training
        seed: Random seed
        
    Returns:
        k-NN classification accuracy on test set
    """
    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()
    
    # Normalize for cosine similarity
    embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
    
    # Split data
    n = len(labels_np)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_ratio)
    
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    X_train, X_test = embeddings_np[train_idx], embeddings_np[test_idx]
    y_train, y_test = labels_np[train_idx], labels_np[test_idx]
    
    # Fit and evaluate k-NN
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def print_similarity_table(
    stats_dict: Dict[str, Dict[str, float]],
) -> None:
    """
    Print a formatted table of similarity statistics.
    
    Args:
        stats_dict: Nested dict {model_space: {metric: value}}
    """
    print("\n" + "=" * 70)
    print("SIMILARITY STATISTICS")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Intra-class':>12} {'Inter-class':>12} {'Separation':>12}")
    print("-" * 70)
    
    for name, stats in stats_dict.items():
        intra = f"{stats['intra_mean']:.4f}±{stats['intra_std']:.4f}"
        inter = f"{stats['inter_mean']:.4f}±{stats['inter_std']:.4f}"
        sep = f"{stats['separation']:.4f}"
        print(f"{name:<35} {intra:>12} {inter:>12} {sep:>12}")
    
    print("=" * 70)
    print("(Higher separation = better class clustering)")
    print()


def print_knn_table(
    knn_dict: Dict[str, float],
) -> None:
    """
    Print k-NN accuracy results.
    
    Args:
        knn_dict: Dict {model_space: accuracy}
    """
    print("\n" + "=" * 50)
    print("k-NN CLASSIFICATION ACCURACY (k=5)")
    print("=" * 50)
    print(f"{'Configuration':<35} {'Accuracy':>12}")
    print("-" * 50)
    
    for name, acc in knn_dict.items():
        print(f"{name:<35} {acc:>12.2%}")
    
    print("=" * 50)
    print()


def reduce_and_plot(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    title_prefix: str,
    output_dir: Path,
    filename_prefix: str,
    n_samples: int = 2000,
    show: bool = False,
) -> None:
    """
    Subsample, reduce dimensionality, and plot embeddings.
    
    Args:
        embeddings: (N, D) embeddings
        labels: (N,) labels
        method: 'tsne' or 'umap'
        title_prefix: Prefix for plot title
        output_dir: Directory to save plots
        filename_prefix: Prefix for output filename
        n_samples: Number of samples for reduction
        show: Whether to show plots interactively
    """
    # Subsample
    n_total = len(labels)
    if n_samples < n_total:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, size=n_samples, replace=False)
        emb_sub = embeddings[indices].numpy()
        lab_sub = labels[indices].numpy()
        sample_info = f"({n_samples}/{n_total} samples)"
    else:
        emb_sub = embeddings.numpy()
        lab_sub = labels.numpy()
        sample_info = f"(all {n_total} samples)"
    
    method_upper = method.upper()
    print(f"  Running {method_upper} {sample_info}...")
    
    reduced = reduce_dimensionality(emb_sub, method=method)
    
    title = f"{title_prefix} - {method_upper} {sample_info}"
    filename = f"{filename_prefix}_{method}.png"
    output_path = output_dir / filename
    
    plot_embeddings_2d(reduced, lab_sub, title, output_path, show=show)


def main(
    n_samples: int = 2000,
    method: str = "both",
    model: str = "both", 
    space: str = "both",
    output_dir: str = "visualizations",
    show: bool = False,
    batch_size: int = 256,
    direct_checkpoint: Optional[str] = None,
):
    """
    Main function for embedding visualization.
    
    Args:
        n_samples: Number of samples for t-SNE/UMAP (default: 2000)
        method: 'tsne', 'umap', or 'both'
        model: 'pretrained', 'finetuned', 'direct', 'all', or 'both' (legacy, same as 'all' minus 'direct')
        space: 'features', 'projections', or 'both'
        output_dir: Directory to save figures
        show: Show plots interactively
        batch_size: Batch size for embedding extraction
        direct_checkpoint: Path to checkpoint for directly-trained ResNet (required if model='direct' or 'all')
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine which methods to use
    if method == "both":
        methods = ["tsne", "umap"] if UMAP_AVAILABLE else ["tsne"]
        if not UMAP_AVAILABLE:
            print("Note: UMAP not available, using t-SNE only")
    else:
        if method == "umap" and not UMAP_AVAILABLE:
            raise ImportError("UMAP requested but not installed. Install with: pip install umap-learn")
        methods = [method]
    
    # Determine which models to analyze
    if model == "both":
        models_to_analyze = ["pretrained", "finetuned"]
    elif model == "all":
        if direct_checkpoint is None:
            raise ValueError("direct_checkpoint must be provided when model='all'")
        models_to_analyze = ["pretrained", "finetuned", "direct"]
    else:
        models_to_analyze = [model]
    
    # Validate direct_checkpoint if needed
    if "direct" in models_to_analyze and direct_checkpoint is None:
        raise ValueError("direct_checkpoint must be provided when analyzing 'direct' model")
    
    # Determine which spaces to visualize
    spaces = ["features", "projections"] if space == "both" else [space]
    
    # Load data
    print("\nLoading CIFAR-10 test set...")
    test_loader = get_cifar10_test_loader(batch_size=batch_size)
    
    # Storage for statistics
    all_similarity_stats = {}
    all_knn_accuracies = {}
    
    for model_name in models_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name.upper()} model")
        print(f"{'='*60}")
        
        # Load model and extract embeddings based on model type
        if model_name == "pretrained":
            loaded_model = load_pretrained_simclr(device=device)
            features, projections, labels = extract_embeddings(loaded_model, test_loader, device)
        elif model_name == "finetuned":
            loaded_model = load_finetuned_backbone(device=device)
            features, projections, labels = extract_embeddings(loaded_model, test_loader, device)
        elif model_name == "direct":
            loaded_model = load_direct_resnet(direct_checkpoint, device=device)
            features, labels = extract_features_only(loaded_model, test_loader, device)
            projections = None  # No projection head for direct model
        
        for space in spaces:
            # Skip projections for models without meaningful projection heads
            if model_name in ["finetuned", "direct"] and space == "projections":
                print(f"\n--- Skipping {space.upper()} space for {model_name} model ---")
                if model_name == "finetuned":
                    print("  (Projection head was not used during fine-tuning)")
                else:
                    print("  (Direct model has no projection head)")
                continue
            
            embeddings = features if space == "features" else projections
            config_name = f"{model_name}_{space}"
            
            print(f"\n--- {space.upper()} space (dim={embeddings.shape[1]}) ---")
            
            # Compute similarity statistics
            print("Computing similarity statistics...")
            stats = compute_similarity_stats(embeddings, labels, normalize=True)
            all_similarity_stats[config_name] = stats
            
            # Compute k-NN accuracy
            print("Computing k-NN accuracy...")
            knn_acc = compute_knn_accuracy(embeddings, labels, k=5)
            all_knn_accuracies[config_name] = knn_acc
            
            # Dimensionality reduction and plotting
            title_prefix = f"{model_name.capitalize()} {space.capitalize()}"
            filename_prefix = config_name
            
            for curr_method in methods:
                reduce_and_plot(
                    embeddings, labels,
                    method=curr_method,
                    title_prefix=title_prefix,
                    output_dir=output_dir,
                    filename_prefix=filename_prefix,
                    n_samples=n_samples,
                    show=show,
                )
            
            # Class similarity heatmap
            print("  Generating class similarity heatmap...")
            heatmap_title = f"{title_prefix} - Class Similarity"
            heatmap_path = output_dir / f"{filename_prefix}_heatmap.png"
            plot_class_similarity_heatmap(
                embeddings, labels,
                title=heatmap_title,
                output_path=heatmap_path,
                show=show,
            )
    
    # Print summary tables
    print_similarity_table(all_similarity_stats)
    print_knn_table(all_knn_accuracies)
    
    print(f"\nAll visualizations saved to: {output_dir.resolve()}")
    print("Done!")


if __name__ == "__main__":
    # main()
    main(model="all", direct_checkpoint="checkpoints/resnet20_cifar10.pth")