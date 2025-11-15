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

    # 2. Download data
    # Create two views of the training set:
    # - raw_train_dataset: non-augmented (used for computing splits and for validation tensors)
    # - aug_train_dataset: possibly augmented (used for training tensors when augment=True)
    raw_train_dataset = dataset(root="./data", train=True, download=True, transform=transform)
    aug_train_dataset = dataset(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = dataset(root="./data", train=False, download=True, transform=transform)

    # 3. Stack into tensors
    # Use raw_train_dataset to get canonical images/labels and determine splits
    x_all_raw = torch.stack([img for img, _ in raw_train_dataset])
    y_all = torch.tensor([label for _, label in raw_train_dataset])

    # Use augmented dataset for the training portion if augmentation requested
    x_all_aug = torch.stack([img for img, _ in aug_train_dataset])

    x_test = torch.stack([img for img, _ in test_dataset])
    y_test = torch.tensor([label for _, label in test_dataset])

    # 4. Split train/val
    # Keep validation taken from the raw (non-augmented) images; training uses augmented images
    x_train, y_train = x_all_aug[:-val_split], y_all[:-val_split]
    x_val, y_val     = x_all_raw[-val_split:], y_all[-val_split:]

    print("Train:", x_train.shape, y_train.shape)
    print("Val:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


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
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.8, 1.25), value='random')
        ])
        print("Using data augmentation for training set")
    else:
        train_transform = transform
        print("No data augmentation for training set")

    # 2. Download data
    # Create raw (non-augmented) and augmented training datasets so we can
    # build a clean validation set while keeping augmentation baked into
    # the returned training tensors (preserves existing external API).
    raw_train_dataset = dataset(root="./data", train=True, download=True, transform=transform)
    aug_train_dataset = dataset(root="./data", train=True, download=True, transform=train_transform)
    test_dataset  = dataset(root="./data", train=False, download=True, transform=transform)

    # 3. Stack into tensors
    x_all_raw  = torch.stack([img for img, _ in raw_train_dataset])
    y_all      = torch.tensor([label for _, label in raw_train_dataset])
    x_all_aug  = torch.stack([img for img, _ in aug_train_dataset])
    x_test     = torch.stack([img for img, _ in test_dataset])
    y_test     = torch.tensor([label for _, label in test_dataset])

    # 4. Split train/val
    # training tensors come from the augmented dataset (if augment=True),
    # validation tensors come from the raw dataset (no augmentation)
    x_train, y_train = x_all_aug[:-val_split], y_all[:-val_split]
    x_val,   y_val   = x_all_raw[-val_split:], y_all[-val_split:]

    print("Train:", x_train.shape, y_train.shape)
    print("Val:",   x_val.shape,   y_val.shape)
    print("Test:",  x_test.shape,  y_test.shape)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
