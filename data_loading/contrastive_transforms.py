"""
Augmentation pipelines for contrastive learning.

This module provides strong augmentation transforms for self-supervised
contrastive learning methods like SimCLR. These augmentations are designed
to create diverse views of the same image while preserving semantic content.
"""

import torch
from torchvision import transforms


def simclr_augment_cifar(
    mean: torch.Tensor, 
    std: torch.Tensor,
    image_size: int = 32,
    s: float = 1.0,  # Color jitter strength
) -> transforms.Compose:
    """Strong augmentation pipeline for SimCLR on CIFAR-10/100.
    
    Creates diverse views using random cropping, flipping, color distortion,
    grayscale conversion, and Gaussian blur.
    
    Args:
        mean: Normalization mean tensor
        std: Normalization std tensor  
        image_size: Size of output images (default 32 for CIFAR)
        s: Color jitter strength multiplier
        
    Returns:
        Composed transform pipeline
        
    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning of 
        Visual Representations", ICML 2020 (Appendix A)
    """
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    # Color jitter with strength s
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * s,
        contrast=0.8 * s, 
        saturation=0.8 * s,
        hue=0.2 * s
    )
    
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def simclr_augment_imagenet(
    mean: torch.Tensor,
    std: torch.Tensor,
    image_size: int = 224,
    s: float = 1.0,
) -> transforms.Compose:
    """Strong augmentation pipeline for SimCLR on ImageNet-scale datasets.
    
    Similar to CIFAR augmentation but with larger blur kernel and
    different crop scales suitable for higher resolution images.
    
    Args:
        mean: Normalization mean tensor
        std: Normalization std tensor
        image_size: Size of output images (default 224)
        s: Color jitter strength multiplier
        
    Returns:
        Composed transform pipeline
    """
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * s,
        contrast=0.8 * s,
        saturation=0.8 * s,
        hue=0.2 * s
    )
    
    # Kernel size should be odd and ~10% of image size
    kernel_size = int(0.1 * image_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def simclr_eval_transform(
    mean: torch.Tensor,
    std: torch.Tensor,
    image_size: int = 32,
) -> transforms.Compose:
    """Evaluation transform for SimCLR (no augmentation).
    
    Simple center crop and normalization for evaluation.
    
    Args:
        mean: Normalization mean tensor
        std: Normalization std tensor
        image_size: Size of output images
        
    Returns:
        Composed transform pipeline
    """
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])
