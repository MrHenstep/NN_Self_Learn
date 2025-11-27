import torch
import torch.nn as nn
import torch.nn.functional as F


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

        return 

    def _make_VGG_block(self, in_channels, out_channels, kernel_size, stride, padding, pool_kernel_size, pool_stride, depth=2):

        block = nn.Sequential()
        
        for layer_index in range(depth):

            if layer_index == 0:
                block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            else:   
                block.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

            block.append(nn.BatchNorm2d(out_channels))
            # block.append(nn.GroupNorm(8, out_channels))  # Alternative to BatchNorm
            block.append(nn.ReLU())

        block.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))

        self.num_conv_blocks += 1
        self.model.add_module(f"conv_block_{self.num_conv_blocks}", block)

        return 

    def _make_dropout(self, p: float = 0.5):
        self.num_dropout_layers += 1
        block = nn.Dropout(p=p)
        self.model.add_module(f"dropout", block)
        return

    def _make_classifier(self, in_features, out_features):

        self.num_classifier_layers += 1
        block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, out_features)
        )
        self.model.add_module(f"classifier_{self.num_classifier_layers}", block) 
        return 
    
    def _make_adaptive_avg_pool(self, output_size=(1,1)):
        block = nn.AdaptiveAvgPool2d(output_size=output_size)
        self.model.add_module(f"adaptive_avg_pool", block)
        return

    def _get_num_features_before_classifier(self, input_channels, input_size):
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            dummy = self.model(dummy)
            num_features_before_classifier = dummy.numel()
        return num_features_before_classifier
    
    #################################################################################################################

    def make_VGG(self):
        
              # Common layer parameters
        stride = 1
        padding = 1
        pool_kernel_size = 2
        pool_stride = 2
        kernel_size = 3

        # Conv Block 1
        num_channels_1 = 64
        depth_1 = 3
        # self._make_conv_block(input_channels, num_channels_1, kernel_size, stride, padding, pool_kernel_size, pool_stride)
        self._make_VGG_block(self.input_channels, num_channels_1, kernel_size, stride, padding, pool_kernel_size, pool_stride, depth=depth_1)


        # Conv Block 2
        num_channels_2 = 2 * num_channels_1
        depth_2 = 3
        # self._make_conv_block(num_channels_1, num_channels_2, kernel_size, stride, padding, pool_kernel_size, pool_stride)
        self._make_VGG_block(num_channels_1, num_channels_2, kernel_size, stride, padding, pool_kernel_size, pool_stride, depth=depth_2)


        # # Conv Block 3
        num_channels_3 = 2 * num_channels_2
        depth_3 = 3
        # self._make_conv_block(num_channels_2, num_channels_3, kernel_size, stride, padding, pool_kernel_size, pool_stride)
        self._make_VGG_block(num_channels_2, num_channels_3, kernel_size, stride, padding, pool_kernel_size, pool_stride, depth=depth_3)

        # Dropout
        self._make_dropout(p=0.3)

        # Adaptive Avg Pool to get fixed size output (1x1)
        self._make_adaptive_avg_pool(output_size=(1,1))

        # Final Classifier
        num_features_before_classifier = self._get_num_features_before_classifier(self.input_channels, self.input_size)
        self._make_classifier(num_features_before_classifier, self.num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    #################################################################################################################
    
    #################################################################################################################



    #################################################################################################################

    def __init__(self, input_channels=3, input_size=32, num_classes=10):
        
        super().__init__()

        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes

        # Set up model as a sequence of layers; model counts are for labelling only
        self.model = nn.Sequential()
        self.num_conv_blocks = 0
        self.num_classifier_layers = 0
        self.num_dropout_layers = 0

    def forward(self, x):
        
        return self.model(x)

    #################################################################################################################
