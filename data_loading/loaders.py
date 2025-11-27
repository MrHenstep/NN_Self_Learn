import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader

from models.cnn.config import DataConfig, DataMetadata
from . import datasets as ldd

# print("Sys.path at start of loaders.py:", os.sys.path)
# print("__package__ in loaders.py:", __package__)

def build_dataloaders(data_cfg: DataConfig, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader, DataMetadata]:
    key_lower = data_cfg.dataset_key.lower()

    if key_lower == "imagenet":
        root = data_cfg.root or "data/imagenet"

        print(f"Loading ImageNet data from {root} with augment={data_cfg.use_augment}")
        train_ds, val_ds, test_ds = ldd.load_torchvision_data_imagenet(root=root, augment=data_cfg.use_augment)
        
        num_classes = 1000
        input_channels = 3
        input_size = 224
        default_batch = 64
        default_eval_batch = 128
    elif key_lower == "tiny_imagenet":
        root = data_cfg.root or "data"
        
        print(f"Loading Tiny ImageNet data from {root} with augment={data_cfg.use_augment}")
        train_ds, val_ds, test_ds = ldd.load_torchvision_data_tiny_imagenet(root=root, augment=data_cfg.use_augment)

        num_classes = 200
        input_channels = 3
        input_size = 64
        default_batch = 256
        default_eval_batch = 256
    elif key_lower == "oxford_pets":
        root = data_cfg.root or "data/oxford-iiit-pet"
                
        print(f"Loading Oxford Pets data from {root} with augment={data_cfg.use_augment}")
        train_ds, val_ds, test_ds = ldd.load_torchvision_data_oxford_pets(root=root, augment=data_cfg.use_augment)

        num_classes = 37
        input_channels = 3
        input_size = 224
        default_batch = 64
        default_eval_batch = 128
    else:
        bundle = ldd.load_dataset(data_cfg.dataset_key, augment=data_cfg.use_augment)
        train_ds, val_ds, test_ds = bundle.train, bundle.val, bundle.test
        num_classes = len(bundle.class_names) if getattr(bundle, "class_names", None) is not None else 10
        input_channels = bundle.num_channels
        input_size = bundle.image_size
        default_batch = 128
        default_eval_batch = 256

    batch_size = data_cfg.batch_size or default_batch
    val_batch_size = data_cfg.val_batch_size or default_eval_batch
    test_batch_size = data_cfg.test_batch_size or default_eval_batch

    num_workers = data_cfg.num_workers
    if num_workers is None:
        num_workers = (os.cpu_count() or 4) // 2

    pin_memory = data_cfg.pin_memory if data_cfg.pin_memory is not None else (device.type == 'cuda')

    print(f"Using {num_workers} data loader workers.")

    def _build_loader(dataset, *, batch_size, shuffle):
        kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'drop_last': False,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        if num_workers and num_workers > 0:
            if data_cfg.persistent_workers is not None:
                kwargs['persistent_workers'] = data_cfg.persistent_workers
            if data_cfg.prefetch_factor is not None:
                kwargs['prefetch_factor'] = data_cfg.prefetch_factor
        return DataLoader(dataset, **kwargs)

    train_loader = _build_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_ds, batch_size=val_batch_size, shuffle=False)
    test_loader = _build_loader(test_ds, batch_size=test_batch_size, shuffle=False)

    metadata = DataMetadata(
        dataset_key=key_lower,
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
    )

    return train_loader, val_loader, test_loader, metadata
