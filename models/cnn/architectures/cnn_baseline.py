import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    """
    Step 2.4(a) — Part 3: Model
    Conv(1→8, 3×3, stride=1, padding=0) → ReLU → MaxPool(2×2) → Flatten → Linear(8·13·13 → 10)
    For MNIST (N, 1, 28, 28) → (N, 10)
    """
    def __init__(self, input_size, num_classes: int = 10, input_channels: int = 1):
        
        super().__init__()

        feature_size = input_size  # 28 for MNIST
        num_channels = input_channels


        # BLOCK 1
        num_channels_1, kernel_size_1, stride_1, padding_1 = 32, 3, 1, 1
        pool_kernel_size, pool_stride = 2, 2

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels_1, kernel_size=kernel_size_1, stride=stride_1, padding=padding_1)
        self.pool  = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        print("Num channels, feature size at start:", num_channels, feature_size)

        num_channels = num_channels_1
        feature_size = ((feature_size + 2 * padding_1 - kernel_size_1) // stride_1) + 1  # after conv layer
        print("Num channels, feature size after conv1:", num_channels, feature_size)
        feature_size = (feature_size - pool_kernel_size) // pool_stride + 1  # after pool layer
        print("Num channels, feature size after pool1:", num_channels, feature_size)

        # # BLOCK 2

        num_channels_2, kernel_size_2, stride_2, padding_2 = 64, 3, 1, 1
        pool_kernel_size_2, pool_stride_2 = 2, 2

        self.conv2 = nn.Conv2d(in_channels=num_channels_1, out_channels=num_channels_2, kernel_size=kernel_size_2, 
        stride=stride_2, padding=padding_2)
        self.pool  = nn.MaxPool2d(kernel_size=pool_kernel_size_2, stride=pool_stride_2)


        num_channels = num_channels_2
        feature_size = ((feature_size + 2 * padding_2 - kernel_size_2) // stride_2) + 1  # after conv layer
        print("Num channels, feature size after conv2:", num_channels, feature_size)
        feature_size = (feature_size - pool_kernel_size_2) // pool_stride_2 + 1  # after pool layer
        print("Num channels, feature size after pool2:", num_channels, feature_size)

        # FINAL CLASSIFIER

        self.fc1   = nn.Linear(num_channels * feature_size * feature_size, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        
        # He/Kaiming init for ReLU layers
        
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        
        # starts at (N, 1, 28, 28) for MNIST

        # BLOCK 1
        x = self.conv1(x)           # size after 1st conv: (N, 16, 28, 28)
        x = F.relu(x)
        x = self.pool(x)            # size after 1st pool: (N, 16, 14, 14)  

        # BLOCK 2
        x = self.conv2(x)           # size after 2nd conv: (N, 32, 14, 14)
        x = F.relu(x)
        x = self.pool(x)            # size after 2nd pool: (N, 32, 7, 7)

        x = torch.flatten(x, 1)       # (N, 32*7*7)

        logits = self.fc1(x)          # (N, 10)
        
        return logits                 # use CrossEntropyLoss on logits


    