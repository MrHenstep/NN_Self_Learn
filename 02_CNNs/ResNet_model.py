import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    ResidualBlock(in_channels, out_channels, stride=1, downsample=None)
    A basic ResNet residual block consisting of two 3x3 convolutional layers with BatchNorm
    and a ReLU activation. The first convolution can apply a spatial stride to downsample
    the feature map; the second convolution always uses stride 1. The block adds the
    (input) identity connection to the output of the second BatchNorm. If the input and
    output shapes differ (e.g. different number of channels or spatial size), a `downsample`
    callable/module may be provided to transform the identity to the correct shape before
    addition.
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the block.
        stride (int, optional): Stride for the first convolution. Defaults to 1.
        downsample (callable or nn.Module, optional): Optional transform applied to the
            identity (input) to match the shape of the output before addition. Typically
            a 1x1 convolution + BatchNorm when changing channel count or spatial dimensions.
    Forward:
        x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).
    Returns:
        torch.Tensor: Output tensor of shape (N, out_channels, H_out, W_out), where
        H_out and W_out depend on the stride (H_out = floor(H / stride), etc.).
    Notes:
        - Convolutions use kernel_size=3, padding=1 and bias=False.
        - Each convolution is followed by BatchNorm2d; ReLU is applied after the first
          convolutional block and again after adding the identity.
        - The downsample transform (if provided) is expected to perform any required
          convolutional and normalization operations on the identity branch.
        - ReLU is used with inplace=True in this implementation.
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DownsampleOptionA(nn.Module):
    """Implements ResNet option A downsampling for CIFAR-style ResNets.

    This performs spatial subsampling by striding (via slicing) and channel
    expansion by zero-padding. No learnable parameters.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DownsampleOptionA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        # Spatial subsample by slicing (equivalent to downsampling by integer stride)
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]

        # Channel expansion via zero-padding if needed
        if self.out_channels > self.in_channels:
            pad_ch = self.out_channels - self.in_channels
            pad_shape = (x.size(0), pad_ch, x.size(2), x.size(3))
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)

        return x


class ResNet20(nn.Module):
    def __init__(self, n_classes, use_projection: bool = True):
        super(ResNet20, self).__init__()
        
        # If True use 1x1 conv + BN projection for identity when downsampling
        # If False use option A (zero-pad + subsample) from the CIFAR ResNet paper
        self.use_projection = use_projection

        self.dropout_percentage = 0.5
        self.relu = nn.ReLU()
        
        # STEM BLOCK

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)


        # STAGE-1 using Residual Blocks
        self.layer1 = nn.Sequential(
            ResidualBlock(in_channels=16, out_channels=16, stride=1, downsample=None),
            ResidualBlock(in_channels=16, out_channels=16, stride=1, downsample=None),
            ResidualBlock(in_channels=16, out_channels=16, stride=1, downsample=None)
        )

        # STAGE-2 using Residual Blocks
        if self.use_projection:
            ds = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(32),
            )
        else:
            ds = DownsampleOptionA(in_channels=16, out_channels=32, stride=2)

        self.layer2 = nn.Sequential(
            ResidualBlock(in_channels=16, out_channels=32, stride=2, downsample=ds),
            ResidualBlock(in_channels=32, out_channels=32, stride=1, downsample=None),
            ResidualBlock(in_channels=32, out_channels=32, stride=1, downsample=None),
        )


        # STAGE-3 using Residual Blocks
        if self.use_projection:
            ds2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            )
        else:
            ds2 = DownsampleOptionA(in_channels=32, out_channels=64, stride=2)

        self.layer3 = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=64, stride=2, downsample=ds2),
            ResidualBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
            ResidualBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
        )

       
        # FINAL BLOCK
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=64, out_features=n_classes)

        # Initialize weights (He / Kaiming for convs, sensible defaults for BN/Linear)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        print("ResNet-20 Model Created")
    
    def forward(self, x):

        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)            # (N, 512)
        x = self.fc(x)                     # (N, n_classes) logits; no ReLU/softmax here

        return x