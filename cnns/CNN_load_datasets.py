from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from collections import defaultdict
import random
import os


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


def _imagenet_train_augment(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def _imagenet_eval_transform(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def _oxford_pets_train_augment(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def _oxford_pets_eval_transform(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def _tiny_imagenet_train_augment(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])


def _tiny_imagenet_eval_transform(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.ToTensor(),
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


def _compute_mean_std_from_dataset(dataset: Dataset, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mean = 0.0
    second_moment = 0.0
    count = 0

    for images, *_ in loader:
        images = images.float()
        bsz = images.size(0)
        reshaped = images.view(bsz, images.size(1), -1)
        mean += reshaped.mean(dim=2).sum(dim=0)
        second_moment += (reshaped ** 2).mean(dim=2).sum(dim=0)
        count += bsz

    mean /= count
    variance = second_moment / count - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=1e-8))
    return mean, std


def _load_oxford_pets(
    config: DatasetConfig,
    *,
    root: str,
    val_split: Optional[int],
    augment: Optional[bool],
    download: bool,
    verbose: bool,
) -> DatasetBundle:
    stat_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    base_trainval = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=download,
        transform=stat_transform,
    )

    mean, std = _compute_mean_std_from_dataset(base_trainval, batch_size=64)
    num_channels = base_trainval[0][0].shape[0]
    image_size = base_trainval[0][0].shape[-1]

    eval_transform = _oxford_pets_eval_transform(mean, std)
    use_augment = config.default_augment if augment is None else augment
    train_transform = _oxford_pets_train_augment(mean, std) if use_augment else eval_transform

    train_dataset = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=False,
        transform=train_transform,
    )
    val_dataset = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=False,
        transform=eval_transform,
    )

    split = config.default_val_split if val_split is None else val_split
    total = len(train_dataset)
    if split <= 0 or split >= total:
        raise ValueError(f"val_split must be in (0, {total}), got {split}")
    rng = random.Random(0)
    label_dataset = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=False,
        transform=None,
    )

    by_class = defaultdict(list)
    for idx in range(total):
        _, label = label_dataset[idx]
        by_class[label].append(idx)

    class_indices_train: list[int] = []
    class_indices_val: list[int] = []
    allocations: list[list] = []  # [label, indices, val_count, remainder]
    sum_val = 0
    for label in sorted(by_class.keys()):
        idxs = by_class[label]
        rng.shuffle(idxs)
        cls_total = len(idxs)
        proportion = (cls_total * split) / total
        base = int(proportion)
        remainder = proportion - base

        val_count = base
        if val_count == 0 and cls_total > 1:
            val_count = 1
        if val_count >= cls_total:
            val_count = cls_total - 1 if cls_total > 1 else cls_total
        val_count = max(val_count, 1 if cls_total > 1 else cls_total)

        allocations.append([label, idxs, val_count, remainder, cls_total])
        sum_val += val_count

    diff = sum_val - split
    if diff > 0:
        sorted_allocs = sorted(range(len(allocations)), key=lambda i: allocations[i][3])
        idx_ptr = 0
        while diff > 0 and idx_ptr < len(sorted_allocs):
            alloc_idx = sorted_allocs[idx_ptr]
            label, idxs, val_count, remainder, cls_total = allocations[alloc_idx]
            if val_count > 1:
                allocations[alloc_idx][2] -= 1
                diff -= 1
            else:
                idx_ptr += 1
        if diff != 0:
            raise RuntimeError("Unable to adjust stratified split to match val_split (reduce phase).")
    elif diff < 0:
        sorted_allocs = sorted(range(len(allocations)), key=lambda i: allocations[i][3], reverse=True)
        idx_ptr = 0
        while diff < 0 and idx_ptr < len(sorted_allocs):
            alloc_idx = sorted_allocs[idx_ptr]
            label, idxs, val_count, remainder, cls_total = allocations[alloc_idx]
            capacity = cls_total - val_count
            if capacity > 1:  # ensure at least one train sample remains
                allocations[alloc_idx][2] += 1
                diff += 1
            else:
                idx_ptr += 1
        if diff != 0:
            raise RuntimeError("Unable to adjust stratified split to match val_split (increase phase).")

    for label, idxs, val_count, _, cls_total in allocations:
        val_sel = idxs[:val_count]
        train_sel = idxs[val_count:]
        class_indices_val.extend(val_sel)
        class_indices_train.extend(train_sel)

    if len(class_indices_val) != split:
        raise RuntimeError(f"Stratified split size mismatch: expected {split}, got {len(class_indices_val)}")
    if verbose:
        unique_train = len({label_dataset[i][1] for i in class_indices_train})
        unique_val = len({label_dataset[i][1] for i in class_indices_val})
        print(f"Stratified split -> train classes: {unique_train}, val classes: {unique_val}")
        # Compute per-class distribution
        def _class_distribution(indices_list: list[int], dataset) -> dict[int, float]:
            counts = defaultdict(int)
            total_samples = len(indices_list)
            for idx in indices_list:
                _, lbl = dataset[idx]
                counts[lbl] += 1
            return {lbl: (count / total_samples) * 100.0 for lbl, count in counts.items()}

        train_distribution = _class_distribution(class_indices_train, label_dataset)
        val_distribution = _class_distribution(class_indices_val, label_dataset)
        test_label_dataset = datasets.OxfordIIITPet(
            root=root,
            split="test",
            target_types="category",
            download=False,
            transform=None,
        )
        test_distribution = _class_distribution(list(range(len(test_label_dataset))), test_label_dataset)

        print("Class distribution (%):")
        for label in sorted(by_class.keys()):
            train_pct = train_distribution.get(label, 0.0)
            val_pct = val_distribution.get(label, 0.0)
            test_pct = test_distribution.get(label, 0.0)
            class_name = label_dataset.classes[label] if hasattr(label_dataset, "classes") else str(label)
            print(
                f"  {class_name:<20} train={train_pct:6.2f}% val={val_pct:6.2f}% test={test_pct:6.2f}%"
            )

    train_subset = Subset(train_dataset, class_indices_train)
    val_subset = Subset(val_dataset, class_indices_val)

    test_set = datasets.OxfordIIITPet(
        root=root,
        split="test",
        target_types="category",
        download=download,
        transform=eval_transform,
    )

    if verbose:
        print(f"Oxford-IIIT Pets stats: mean={mean.tolist()}, std={std.tolist()}")
        print("Input resolution: 224x224 (RandomResizedCrop for training, Resize+CenterCrop for eval)")
        print("Using training augmentation" if use_augment else "Training augmentation disabled")
        print("Train:", len(train_subset))
        print("Val:", len(val_subset))
        print("Test:", len(test_set))

    class_names = getattr(train_dataset, "classes", None)

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


def _parse_tiny_imagenet_val_annotations(val_dir: str) -> dict[str, str]:
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    mapping: dict[str, str] = {}
    with open(annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename, wnid = parts[0], parts[1]
                mapping[filename] = wnid
    return mapping


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        classes: Sequence[str],
        transform: Optional[Callable] = None,
        loader=default_loader,
    ):
        self.samples = samples
        self.transform = transform
        self.loader = loader
        self.classes = list(classes)
        self.targets = [target for _, target in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _gather_tiny_imagenet_samples(root: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict[str, int], list[str]]:
    train_dir = os.path.join(root, "train")
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Tiny ImageNet train directory not found at '{train_dir}'.")

    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    train_samples: list[tuple[str, int]] = []
    for cls_name, cls_idx in class_to_idx.items():
        cls_images = os.path.join(train_dir, cls_name, "images")
        if not os.path.isdir(cls_images):
            continue
        for fname in os.listdir(cls_images):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                train_samples.append((os.path.join(cls_images, fname), cls_idx))

    val_dir = os.path.join(root, "val")
    if not os.path.isdir(val_dir):
        raise RuntimeError(f"Tiny ImageNet val directory not found at '{val_dir}'.")
    mapping = _parse_tiny_imagenet_val_annotations(val_dir)
    val_images_dir = os.path.join(val_dir, "images")
    val_samples: list[tuple[str, int]] = []
    for fname, wnid in mapping.items():
        cls_idx = class_to_idx.get(wnid)
        if cls_idx is None:
            continue
        path = os.path.join(val_images_dir, fname)
        if os.path.isfile(path):
            val_samples.append((path, cls_idx))

    classes = sorted(class_to_idx, key=class_to_idx.get)
    return train_samples, val_samples, class_to_idx, classes


def _load_imagenet(
    config: DatasetConfig,
    *,
    root: str,
    val_split: Optional[int],
    augment: Optional[bool],
    download: bool,
    verbose: bool,
) -> DatasetBundle:
    if download and verbose:
        print("ImageNet download is not supported via torchvision; ignoring download=True.")

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    eval_transform = _imagenet_eval_transform(mean, std)
    use_augment = config.default_augment if augment is None else augment
    train_transform = _imagenet_train_augment(mean, std) if use_augment else eval_transform

    try:
        train_set = datasets.ImageNet(root=root, split="train", transform=train_transform)
    except (RuntimeError, FileNotFoundError) as exc:
        raise RuntimeError(
            "ImageNet training data not found. Ensure the ImageNet dataset is prepared under "
            f"'{root}' following torchvision's folder structure (train/ and val/)."
        ) from exc

    try:
        val_dataset = datasets.ImageNet(root=root, split="val", transform=eval_transform)
    except (RuntimeError, FileNotFoundError) as exc:
        raise RuntimeError(
            "ImageNet validation data not found. Ensure the 'val' split exists under the provided root."
        ) from exc

    total_val = len(val_dataset)
    if val_split is None:
        val_subset: Dataset = val_dataset
        test_set: Dataset = datasets.ImageNet(root=root, split="val", transform=eval_transform)
    else:
        if val_split <= 0 or val_split >= total_val:
            raise ValueError(f"val_split must be in (0, {total_val}), got {val_split}")
        indices = list(range(total_val))
        val_indices = indices[:val_split]
        test_indices = indices[val_split:]
        val_subset = Subset(val_dataset, val_indices)
        test_set = Subset(val_dataset, test_indices)

    if verbose:
        print(f"ImageNet stats: mean={mean.tolist()}, std={std.tolist()}")
        print("Input resolution: 224x224 (RandomResizedCrop for training, Resize+CenterCrop for eval)")
        print("Using training augmentation" if use_augment else "Training augmentation disabled")
        print("Train:", len(train_set))
        print("Val:", len(val_subset))
        print("Test:", len(test_set))

    class_names = getattr(val_dataset, "classes", None)

    return DatasetBundle(
        train=train_set,
        val=val_subset,
        test=test_set,
        mean=mean,
        std=std,
        image_size=224,
        num_channels=3,
        class_names=class_names,
    )


def _load_tiny_imagenet(
    *,
    root: str,
    val_split: Optional[int],
    augment: Optional[bool],
    download: bool,
    verbose: bool,
) -> DatasetBundle:
    if download:
        if verbose:
            print("Automatic download for Tiny ImageNet is not supported. Ensure data is extracted under 'tiny-imagenet-200/'.")

    train_samples, val_samples, class_to_idx, classes = _gather_tiny_imagenet_samples(root)

    stats_dataset = TinyImageNetDataset(train_samples, classes, transform=transforms.ToTensor())
    mean, std = _compute_mean_std_from_dataset(stats_dataset, batch_size=256)

    use_augment = True if augment is None else augment
    train_transform = _tiny_imagenet_train_augment(mean, std) if use_augment else _tiny_imagenet_eval_transform(mean, std)
    eval_transform = _tiny_imagenet_eval_transform(mean, std)

    train_dataset = TinyImageNetDataset(train_samples, classes, transform=train_transform)
    eval_dataset = TinyImageNetDataset(val_samples, classes, transform=eval_transform)

    total_val = len(val_samples)
    split = total_val // 2 if val_split is None else val_split
    if split <= 0 or split >= total_val:
        raise ValueError(f"val_split must be in (0, {total_val}), got {split}")

    rng = random.Random(0)
    indices = list(range(total_val))
    rng.shuffle(indices)
    val_indices = indices[:split]
    test_indices = indices[split:]

    val_subset = Subset(eval_dataset, val_indices)
    test_subset = Subset(eval_dataset, test_indices)

    if verbose:
        print("Tiny ImageNet stats:")
        print(f"  mean={mean.tolist()} std={std.tolist()}")
        print(f"  train samples={len(train_dataset)} val samples={len(val_subset)} test samples={len(test_subset)}")

    mean_image_size = 64  # known constant
    return DatasetBundle(
        train=train_dataset,
        val=val_subset,
        test=test_subset,
        mean=mean,
        std=std,
        image_size=mean_image_size,
        num_channels=3,
        class_names=classes,
    )


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
    if key == "oxford_pets":
        return _load_oxford_pets(
            config,
            root=root,
            val_split=val_split,
            augment=augment,
            download=download,
            verbose=verbose,
        )
    if key == "imagenet":
        return _load_imagenet(
            config,
            root=root,
            val_split=val_split,
            augment=augment,
            download=download,
            verbose=verbose,
        )
    if key == "tiny_imagenet":
        tiny_root = os.path.join(root, "tiny-imagenet-200") if os.path.isdir(os.path.join(root, "tiny-imagenet-200")) else root
        return _load_tiny_imagenet(
            root=tiny_root,
            val_split=val_split,
            augment=augment,
            download=download,
            verbose=verbose,
        )

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


def load_torchvision_data_oxford_pets(
    root: str = "./data",
    val_split: Optional[int] = None,
    augment: bool = True,
    download: bool = True,
    verbose: bool = True,
):
    bundle = load_dataset(
        "oxford_pets",
        root=root,
        val_split=val_split,
        augment=augment,
        download=download,
        verbose=verbose,
    )
    return bundle.train, bundle.val, bundle.test


def load_torchvision_data_imagenet(
    root: str = "./data",
    val_split: Optional[int] = None,
    augment: bool = True,
    verbose: bool = True,
):
    bundle = load_dataset(
        "imagenet",
        root=root,
        val_split=val_split,
        augment=augment,
        download=False,
        verbose=verbose,
    )
    return bundle.train, bundle.val, bundle.test


def load_torchvision_data_tiny_imagenet(
    root: str = "./data",
    val_split: Optional[int] = None,
    augment: bool = True,
    download: bool = False,
    verbose: bool = True,
):
    bundle = load_dataset(
        "tiny_imagenet",
        root=root,
        val_split=val_split,
        augment=augment,
        download=download,
        verbose=verbose,
    )
    return bundle.train, bundle.val, bundle.test
