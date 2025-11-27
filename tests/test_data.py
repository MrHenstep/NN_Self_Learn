import torch
from torchvision import transforms
from data_loading.config import DATASET_REGISTRY
from data_loading.transforms import cifar10_augment

def test_registry_keys():
    """Ensure key datasets are present in the registry."""
    assert "cifar10" in DATASET_REGISTRY
    assert "mnist" in DATASET_REGISTRY
    assert "tiny_imagenet" in DATASET_REGISTRY

def test_transform_builder():
    """Test that augmentation builders return a Compose object."""
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    
    transform = cifar10_augment(mean, std, image_size=32)
    assert isinstance(transform, transforms.Compose)
    
    # Check that Normalize is likely the last step
    assert isinstance(transform.transforms[-1], transforms.Normalize)
