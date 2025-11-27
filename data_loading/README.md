# Data Loading

This package handles all aspects of dataset preparation, augmentation, and loading. It is designed to be used by the `scripts` to provide data to the `models`.

## Key Components

*   **`loaders.py`**: The main entry point for creating PyTorch DataLoaders. The `build_dataloaders` function takes a configuration and returns train/val/test loaders along with dataset metadata.
*   **`datasets.py`**: High-level dataset management. It coordinates the loading of specific datasets (like Oxford Pets or TinyImageNet) and handles splitting logic.
*   **`config.py`**: Contains the `DATASET_REGISTRY`, which maps dataset names (keys) to their configurations (classes, default splits, etc.).
*   **`transforms.py`**: Defines data augmentation pipelines (e.g., `mnist_augment`, `cifar10_augment`, `imagenet_train_augment`).
*   **`custom_datasets.py`**: Contains custom `Dataset` implementations, such as `TinyImageNetDataset`, for data that doesn't fit standard `torchvision` structures.

## Supported Datasets

The following datasets are currently registered in `config.py`:
*   MNIST
*   FashionMNIST
*   CIFAR-10
*   Oxford-IIIT Pets
*   Tiny ImageNet
*   ImageNet
