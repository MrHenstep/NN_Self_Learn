import os
from typing import Callable, Optional, Sequence

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


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


def gather_tiny_imagenet_samples(root: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict[str, int], list[str]]:
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
