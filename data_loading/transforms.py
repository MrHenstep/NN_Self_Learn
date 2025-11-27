import torch
from torchvision import transforms

def mnist_augment(mean: torch.Tensor, std: torch.Tensor, image_size: int) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def cifar10_augment(mean: torch.Tensor, std: torch.Tensor, image_size: int) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.08), ratio=(0.8, 1.25), value="random"),
        transforms.Normalize(mean_list, std_list),
    ])


def imagenet_train_augment(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def imagenet_eval_transform(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def oxford_pets_train_augment(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def oxford_pets_eval_transform(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])


def tiny_imagenet_train_augment(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
    ])


def tiny_imagenet_eval_transform(mean: torch.Tensor, std: torch.Tensor) -> transforms.Compose:
    mean_list = mean.tolist()
    std_list = std.tolist()
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_list, std_list),
    ])
