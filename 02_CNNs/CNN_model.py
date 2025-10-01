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
    def __init__(self, input_size, num_classes: int = 10):
        
        super().__init__()

        feature_size = input_size  # 28 for MNIST
        num_channels = 1


        # BLOCK 1
        num_channels_1, kernel_size_1, stride_1, padding_1 = 16, 3, 1, 1

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_channels_1, kernel_size=kernel_size_1, stride=stride_1, padding=padding_1)

        feature_size = ((feature_size + 2 * padding_1 - kernel_size_1) // stride_1) + 1  
        num_channels = num_channels_1

        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        feature_size = (feature_size - 2) // 2 + 1  # after pool layer

        # # BLOCK 2

        num_channels_2, kernel_size_2, stride_2, padding_2 = 32, 3, 1, 1

        self.conv2 = nn.Conv2d(in_channels=num_channels_1, out_channels=num_channels_2, kernel_size=kernel_size_2, stride=stride_2, padding=padding_2)

        feature_size = ((feature_size + 2 * padding_2 - kernel_size_2) // stride_2) + 1  
        num_channels = num_channels_2

        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        feature_size = (feature_size - 2) // 2 + 1  # after pool layer

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
        
        # BLOCK 1
        x = self.conv1(x)            # (N, 8, 28, 28)
        x = F.relu(x)
        x = self.pool(x)

        # BLOCK 2
        x = self.conv2(x)            # (N, 16, 14, 14)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)       # (N, 16*7*7)

        logits = self.fc1(x)          # (N, 10)
        
        return logits                 # use CrossEntropyLoss on logits


    