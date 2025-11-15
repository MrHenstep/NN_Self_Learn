import torch
import torchvision
import torchvision.transforms as transforms


def load_torchvision_data_MNIST(dataset, val_split: int = 10000, augment: bool = False):


    # 0. Compute mean and std from the training set (with ToTensor only)
    tmp_dataset = dataset(root="./data", train=True, download=True,
                          transform=transforms.ToTensor())
    x_tmp = torch.stack([img for img, _ in tmp_dataset])  # (60000, 1, 28, 28)
    mean = x_tmp.mean([0,2,3])
    std  = x_tmp.std([0,2,3])
    print(f"{dataset.__name__} stats: mean={mean.item():.4f}, std={std.item():.4f}")

    image_size = x_tmp.shape[-1]
    print(f"Image size: {image_size}x{image_size}")

    # 1. Transform: to tensor (scales to [0,1], shape (C,H,W))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())        
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=2),         # small spatial jitter
            transforms.RandomHorizontalFlip(p=0.5),       # left-right symmetry
            transforms.RandomRotation(degrees=8, fill=0),        # slight rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])
        print("Using data augmentation for training set")
    else:
        train_transform = transform
        print("No data augmentation for training set")

    # 2. Download data as dataset objects
    # Create two views of the training set:
    # - raw_train_dataset: non-augmented (used for validation)
    # - aug_train_dataset: possibly augmented (used for training)
    raw_train_dataset = dataset(root="./data", train=True, download=True, transform=transform)
    aug_train_dataset = dataset(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = dataset(root="./data", train=False, download=True, transform=transform)

    # 3. Create Subset objects for train/val so augmentation is applied on-the-fly
    n_total = len(raw_train_dataset)
    train_indices = list(range(0, n_total - val_split))
    val_indices = list(range(n_total - val_split, n_total))

    from torch.utils.data import Subset
    train_subset = Subset(aug_train_dataset, train_indices)
    val_subset = Subset(raw_train_dataset, val_indices)

    print("Train (subset):", len(train_subset))
    print("Val (subset):", len(val_subset))
    print("Test:", len(test_dataset))

    return train_subset, val_subset, test_dataset


def load_torchvision_data_cifar10(val_split: int = 5000, augment: bool = True):
    import torch, torchvision
    from torchvision import transforms

    dataset = torchvision.datasets.CIFAR10

    # 0. Compute mean/std from raw data
    tmp_dataset = dataset(root="./data", train=True, download=True,
                          transform=transforms.ToTensor())
    x_tmp = torch.stack([img for img, _ in tmp_dataset])  # (50000, 3, 32, 32)
    mean = x_tmp.mean([0, 2, 3])
    std  = x_tmp.std([0, 2, 3])
    print(f"{dataset.__name__} stats: mean={mean}, std={std}")

    image_size = x_tmp.shape[-1]
    print(f"Image size: {image_size}x{image_size}")

    # 1. Base transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])

    if augment:
        # Standard CIFAR augmentation: pad=4 -> random crop -> horizontal flip.
        # Apply RandomErasing on the tensor before Normalize so erasure values
        # are in the image value scale (0..1) rather than in normalized space.
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.8, 1.25), value='random'),
            transforms.Normalize(mean.tolist(), std.tolist())
        ])
        print("Using data augmentation for training set (pad=4, crop, flip, RandomErasing before Normalize)")
    else:
        train_transform = transform
        print("No data augmentation for training set")

    # 2. Download data as dataset objects
    raw_train_dataset = dataset(root="./data", train=True, download=True, transform=transform)
    aug_train_dataset = dataset(root="./data", train=True, download=True, transform=train_transform)
    test_dataset  = dataset(root="./data", train=False, download=True, transform=transform)

    # 3. Create Subset objects for train/val so augmentation is applied on-the-fly
    n_total = len(raw_train_dataset)
    train_indices = list(range(0, n_total - val_split))
    val_indices = list(range(n_total - val_split, n_total))

    from torch.utils.data import Subset
    train_subset = Subset(aug_train_dataset, train_indices)
    val_subset = Subset(raw_train_dataset, val_indices)

    print("Train (subset):", len(train_subset))
    print("Val (subset):", len(val_subset))
    print("Test:", len(test_dataset))

    return train_subset, val_subset, test_dataset
