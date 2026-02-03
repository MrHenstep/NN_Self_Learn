"""
Dataset wrappers for contrastive learning.

This module provides dataset wrappers that return multiple augmented views
of each image, as required by contrastive learning methods like SimCLR.
"""

from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image


class TwoViewDataset(Dataset):
    """Dataset wrapper that returns two augmented views of each image.
    
    Wraps an existing dataset and applies a transform twice to each image,
    producing two different augmented views. Used for contrastive learning
    where we need positive pairs from the same image.
    
    Args:
        base_dataset: Underlying dataset to wrap (should return PIL Images)
        transform: Augmentation transform to apply (called twice per image)
        return_label: Whether to also return the original label (default True)
        
    Example:
        >>> base = CIFAR10(root='./data', train=True, download=True)
        >>> transform = simclr_augment_cifar(mean, std)
        >>> dataset = TwoViewDataset(base, transform)
        >>> view1, view2, label = dataset[0]
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Callable,
        return_label: bool = True,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.return_label = return_label
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        # Get raw image and label from base dataset
        # Note: base_dataset should have transform=None or only ToTensor
        img, label = self.base_dataset[idx]
        
        # If img is already a tensor, we need to convert back to PIL
        # This happens if base dataset has transforms
        if hasattr(img, 'numpy'):
            # It's a tensor, need to handle this case
            raise ValueError(
                "TwoViewDataset expects base_dataset to return PIL Images. "
                "Set transform=None on the base dataset."
            )
        
        # Apply transform twice to get two different views
        view1 = self.transform(img)
        view2 = self.transform(img)
        
        if self.return_label:
            return view1, view2, label
        return view1, view2


class MultiViewDataset(Dataset):
    """Dataset wrapper that returns multiple augmented views of each image.
    
    More general version of TwoViewDataset that can return any number of views.
    
    Args:
        base_dataset: Underlying dataset to wrap
        transform: Augmentation transform to apply
        n_views: Number of augmented views to generate
        return_label: Whether to also return the original label
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Callable,
        n_views: int = 2,
        return_label: bool = True,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.n_views = n_views
        self.return_label = return_label
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        img, label = self.base_dataset[idx]
        
        if hasattr(img, 'numpy'):
            raise ValueError(
                "MultiViewDataset expects base_dataset to return PIL Images. "
                "Set transform=None on the base dataset."
            )
        
        views = tuple(self.transform(img) for _ in range(self.n_views))
        
        if self.return_label:
            return views + (label,)
        return views
