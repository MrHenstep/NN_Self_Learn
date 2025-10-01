import torch
import torchvision
import torchvision.transforms as transforms


def load_torchvision_data(dataset, val_split: int = 10000):


    # 0. Compute mean and std from the training set (with ToTensor only)
    tmp_dataset = dataset(root="./data", train=True, download=True,
                          transform=transforms.ToTensor())
    x_tmp = torch.stack([img for img, _ in tmp_dataset])  # (60000, 1, 28, 28)
    mean = x_tmp.mean([0,2,3])
    std  = x_tmp.std([0,2,3])
    print(f"{dataset.__name__} stats: mean={mean.item():.4f}, std={std.item():.4f}")


    # 1. Transform: to tensor (scales to [0,1], shape (C,H,W))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())        
    ])

    # 2. Download Fashion-MNIST
    train_dataset = dataset(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = dataset(
        root="./data", train=False, download=True, transform=transform
    )

    # 3. Stack into tensors
    x_all = torch.stack([img for img, _ in train_dataset])      # (60000, 1, 28, 28)
    y_all = torch.tensor([label for _, label in train_dataset]) # (60000,)

    x_test = torch.stack([img for img, _ in test_dataset])      # (10000, 1, 28, 28)
    y_test = torch.tensor([label for _, label in test_dataset]) # (10000,)

    # 4. Split train/val
    x_train, y_train = x_all[:-val_split], y_all[:-val_split]   # (50000,...)
    x_val, y_val     = x_all[-val_split:], y_all[-val_split:]   # (10000,...)

    print("Train:", x_train.shape, y_train.shape, x_train.min().item(), x_train.max().item())
    print("Val:", x_val.shape, y_val.shape)
    print("Test:", x_test.shape, y_test.shape)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
