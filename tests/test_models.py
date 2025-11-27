import torch
import pytest
from models.cnn.architectures.resnet import ResNetCF, ResNetIN
from models.cnn.architectures.cnn_baseline import SimpleCNN
from models.perceptron.MLP import MLP_torch

def test_resnet_cf_shape():
    """Test that ResNetCF produces correct output shape for CIFAR-like inputs."""
    # Smallest possible ResNet (n=1)
    model = ResNetCF(n_classes=10, resnet_n=1, use_projection=True, use_residual=True)
    # Batch of 2, 3 channels, 32x32
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_resnet_in_shape():
    """Test that ResNetIN produces correct output shape for ImageNet-like inputs."""
    # Standard ResNet18-like
    model = ResNetIN(n_classes=100, use_projection=True, use_residual=True)
    # Batch of 2, 3 channels, 224x224
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 100)

def test_simple_cnn_shape():
    """Test SimpleCNN output shape."""
    model = SimpleCNN(input_size=32, num_classes=10, input_channels=3)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

def test_mlp_torch_shape():
    """Test MLP_torch output shape."""
    # MLP_torch is designed for binary classification (output dim 1)
    model = MLP_torch(input_dim=10, hidden_dim=5)
    x = torch.randn(2, 10)
    out = model(x)
    assert out.shape == (2, 1)
