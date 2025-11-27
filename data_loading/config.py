from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from . import transforms as custom_transforms

@dataclass(frozen=True)
class DatasetConfig:
    key: str
    dataset_cls: Optional[Callable[..., Dataset]]
    default_val_split: int
    augment_builder: Optional[Callable[[torch.Tensor, torch.Tensor, int], transforms.Compose]]
    default_augment: bool
    display_name: str


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    mean: torch.Tensor
    std: torch.Tensor
    image_size: int
    num_channels: int
    class_names: Optional[Sequence[str]]


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "mnist": DatasetConfig(
        key="mnist",
        dataset_cls=datasets.MNIST,
        default_val_split=10_000,
        augment_builder=custom_transforms.mnist_augment,
        default_augment=False,
        display_name="MNIST",
    ),
    "fashion_mnist": DatasetConfig(
        key="fashion_mnist",
        dataset_cls=datasets.FashionMNIST,
        default_val_split=10_000,
        augment_builder=custom_transforms.mnist_augment,
        default_augment=False,
        display_name="FashionMNIST",
    ),
    "cifar10": DatasetConfig(
        key="cifar10",
        dataset_cls=datasets.CIFAR10,
        default_val_split=5_000,
        augment_builder=custom_transforms.cifar10_augment,
        default_augment=True,
        display_name="CIFAR-10",
    ),
    "oxford_pets": DatasetConfig(
        key="oxford_pets",
        dataset_cls=datasets.OxfordIIITPet,
        default_val_split=1_000,
        augment_builder=None,
        default_augment=True,
        display_name="Oxford-IIIT Pets",
    ),
    "tiny_imagenet": DatasetConfig(
        key="tiny_imagenet",
        dataset_cls=None,  # handled separately
        default_val_split=0,
        augment_builder=None,
        default_augment=True,
        display_name="Tiny ImageNet",
    ),
    "imagenet": DatasetConfig(
        key="imagenet",
        dataset_cls=datasets.ImageNet,
        default_val_split=0,
        augment_builder=None,
        default_augment=True,
        display_name="ImageNet",
    ),
}
