from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    dataset_cls: Callable[..., Dataset]
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


def _mnist_augment(mean: torch.Tensor, std: torch.Tensor, image_size: int) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def _cifar10_augment(mean: torch.Tensor, std: torch.Tensor, image_size: int) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.8, 1.25), value="random"),
        transforms.Normalize(mean_list, std_list),
    ])


_DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "mnist": DatasetConfig(
        key="mnist",
        dataset_cls=datasets.MNIST,
        default_val_split=10_000,
        augment_builder=_mnist_augment,
        default_augment=False,
        display_name="MNIST",
    ),
    "fashion_mnist": DatasetConfig(
        key="fashion_mnist",
        dataset_cls=datasets.FashionMNIST,
        default_val_split=10_000,
        augment_builder=_mnist_augment,
        default_augment=False,
        display_name="FashionMNIST",
    ),
    "cifar10": DatasetConfig(
        key="cifar10",
        dataset_cls=datasets.CIFAR10,
        default_val_split=5_000,
        augment_builder=_cifar10_augment,
        default_augment=True,
        display_name="CIFAR-10",
    ),
}


def _compute_mean_std(dataset_cls: Callable[..., Dataset], root: str, download: bool) -> tuple[torch.Tensor, torch.Tensor, int, int, Optional[Sequence[str]]]:
    baseline = dataset_cls(root=root, train=True, download=download, transform=transforms.ToTensor())
    loader = DataLoader(baseline, batch_size=512, shuffle=False, num_workers=0)

    mean = 0.0
    second_moment = 0.0
    count = 0

    for images, _ in loader:
        images = images.float()
        batch_size = images.size(0)
        reshaped = images.view(batch_size, images.size(1), -1)
        mean += reshaped.mean(dim=2).sum(dim=0)
        second_moment += (reshaped ** 2).mean(dim=2).sum(dim=0)
        count += batch_size

    mean /= count
    variance = second_moment / count - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=1e-8))

    sample = baseline[0][0]
    image_size = sample.shape[-1]
    num_channels = sample.shape[0]
    class_names = getattr(baseline, "classes", None)

    return mean, std, image_size, num_channels, class_names


def load_dataset(
    name: str,
    *,
    root: str = "./data",
    val_split: Optional[int] = None,
    augment: Optional[bool] = None,
    download: bool = True,
    verbose: bool = True,
) -> DatasetBundle:
    key = name.lower()
    if key not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(_DATASET_REGISTRY)}")

    config = _DATASET_REGISTRY[key]
    dataset_cls = config.dataset_cls

    mean, std, image_size, num_channels, class_names = _compute_mean_std(dataset_cls, root=root, download=download)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    use_augment = config.default_augment if augment is None else augment
    if use_augment and config.augment_builder is not None:
        train_transform = config.augment_builder(mean, std, image_size)
    else:
        train_transform = base_transform

    raw_train = dataset_cls(root=root, train=True, download=download, transform=base_transform)
    aug_train = dataset_cls(root=root, train=True, download=download, transform=train_transform)
    test_set = dataset_cls(root=root, train=False, download=download, transform=base_transform)

    split = config.default_val_split if val_split is None else val_split
    if split <= 0 or split >= len(raw_train):
        raise ValueError(f"val_split must be in (0, {len(raw_train)}), got {split}")

    train_indices = list(range(0, len(raw_train) - split))
    val_indices = list(range(len(raw_train) - split, len(raw_train)))

    train_subset = Subset(aug_train, train_indices)
    val_subset = Subset(raw_train, val_indices)

    if verbose:
        print(f"{config.display_name} stats: mean={mean.tolist()}, std={std.tolist()}")
        print(f"Image size: {image_size}x{image_size}")
        print("Using data augmentation for training set" if use_augment else "No data augmentation for training set")
        print("Train (subset):", len(train_subset))
        print("Val (subset):", len(val_subset))
        print("Test:", len(test_set))

    return DatasetBundle(
        train=train_subset,
        val=val_subset,
        test=test_set,
        mean=mean,
        std=std,
        image_size=image_size,
        num_channels=num_channels,
        class_names=class_names,
    )


def load_torchvision_data_MNIST(dataset, val_split: int = 10_000, augment: bool = False):
    name_map = {
        datasets.MNIST: "mnist",
        datasets.FashionMNIST: "fashion_mnist",
    }
    key = name_map.get(dataset)
    if key is None:
        raise ValueError("Unsupported dataset class for MNIST loader.")
    bundle = load_dataset(key, val_split=val_split, augment=augment)
    return bundle.train, bundle.val, bundle.test


def load_torchvision_data_cifar10(val_split: int = 5_000, augment: bool = True):
    bundle = load_dataset("cifar10", val_split=val_split, augment=augment)
    return bundle.train, bundle.val, bundle.test
