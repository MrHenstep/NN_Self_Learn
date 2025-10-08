import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


class SimpleCNNFlexi(nn.Module):

    #################################################################################################################   
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

    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel_size, pool_stride):
    
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        )
        self.num_conv_blocks += 1
        self.model.add_module(f"conv_block_{self.num_conv_blocks}", block)

        return block

    def _make_classifier(self, in_features, out_features):

        self.num_classifier_layers += 1
        block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, out_features)
        )
        self.model.add_module(f"classifier_{self.num_classifier_layers}", block) 
        return 
    
    def _get_num_features_before_classifier(self, input_channels, input_size):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            dummy = self.model(dummy)
            num_features_before_classifier = dummy.numel()
        return num_features_before_classifier
    
    #################################################################################################################

    def __init__(self, input_size, num_classes: int = 10):
        
        super().__init__()

        # Hardcoded for MNIST
        input_channels = 1

        # Set up model as a sequence of layers
        self.model = nn.Sequential()
        self.num_conv_blocks = 0
        self.num_classifier_layers = 0

        # Common layer parameters
        stride = 1
        padding = 1
        pool_kernel_size = 2
        pool_stride = 2
        kernel_size = 3

        # Conv Block 1
        num_channels_1 = 32
        self._make_conv_block(input_channels, num_channels_1, kernel_size, stride, padding, pool_kernel_size, pool_stride)

        # Conv Block 2
        num_channels_2 = 64
        self._make_conv_block(num_channels_1, num_channels_2, kernel_size, stride, padding, pool_kernel_size, pool_stride)

        # Final Classifier
        num_features_before_classifier = self._get_num_features_before_classifier(input_channels, input_size)
        self._make_classifier(num_features_before_classifier, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, x):
        
        return self.model(x)


    