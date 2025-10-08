import torch
# import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

# accesses a layer in SimpleCNN class, conv1

# ----------------------------
# Visualization member methods
# ----------------------------

# USAGE EXAMPLES:
# model.show_conv1_kernels()
# xb, yb = next(iter(test_loader))
# model.show_conv1_feature_maps(xb[:1], device=device)
# model.show_kernel_frequency_response()
# model.print_kernels()

# def show_conv1_kernels(conv_layer):

#     """Show 3×3 conv kernels as signed heatmaps with optional bias."""

#     W = conv_layer.weight.detach().cpu().numpy()  # (8,1,3,3)
#     b = conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else None

#     show_conv_kernels(W, b)


def show_conv_kernel(conv_layer):

    """Show 3×3 conv kernels as signed heatmaps with optional bias."""
    W = conv_layer.weight.detach().cpu().numpy()  # (8,1,3,3)
    b = conv_layer.bias.detach().cpu().numpy() if conv_layer.bias is not None else None

    # normalize each kernel for display
    Wn = W.copy()
    for k in range(Wn.shape[0]):
        w = Wn[k, 0]
        w = (w - w.mean()) / (w.std() + 1e-8)
        Wn[k, 0] = np.clip(w, -2, 2)

    ncols = Wn.shape[0]
    fig, axes = plt.subplots(1, ncols, figsize=(1.8*ncols, 2.2))
    axes = np.atleast_1d(axes)

    for k, ax in enumerate(axes):
        ax.imshow(Wn[k, 0], cmap='gray', vmin=-2, vmax=2, interpolation='nearest')
        title = f"k{k}"
        if b is not None:
            title += f"\nbias={b[k]:+.2f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def show_conv1_feature_maps(model, x_one: torch.Tensor, device: torch.device | None = None):

    """
    Visualize conv1 pre/post-activation feature maps for a single input image.
    x_one: (1, 1, 28, 28)
    """

    model.eval()

    if device is None:
        device = next(model.parameters()).device
    x_one = x_one.to(device)

    z = model.conv1(x_one)   # (1, C, H, W)
    a = F.relu(z)

    z_np = z.squeeze(0).detach().cpu().numpy()  # (C, H, W)
    a_np = a.squeeze(0).detach().cpu().numpy()

    C = z_np.shape[0]
    fig, axes = plt.subplots(2, C, figsize=(2*C, 4))
    if C == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # ensure 2xC indexing if C==1

    for k in range(C):
        axes[0, k].imshow(z_np[k], cmap='gray', interpolation='nearest')
        axes[0, k].set_title(f"pre-Act k{k}", fontsize=10)
        axes[0, k].axis('off')

        axes[1, k].imshow(a_np[k], cmap='gray', interpolation='nearest')
        axes[1, k].set_title(f"post-ReLU k{k}", fontsize=10)
        axes[1, k].axis('off')

    plt.tight_layout()
    plt.show()

def show_kernel_frequency_response(conv_layer):

    """Show a crude 2D FFT magnitude of each 3×3 kernel (zero-padded)."""

    W = conv_layer.weight.detach().cpu().numpy()  # (C_out, 1, 3, 3)
    C = W.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(1.8*C, 2.2))
    axes = np.atleast_1d(axes)

    for k, ax in enumerate(axes):
        w = W[k, 0]
        pad = np.zeros((16, 16), dtype=np.float32)
        pad[:w.shape[0], :w.shape[1]] = w
        fftmag = np.abs(np.fft.fftshift(np.fft.fft2(pad)))
        ax.imshow(fftmag, cmap='magma', interpolation='nearest')
        ax.set_title(f"FFT |k{k}|", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def print_kernels(model, decimals: int = 3):
    
    """Print the raw 3×3 kernel values."""
    
    W = model.conv1.weight.detach().cpu().numpy()
    
    for k in range(W.shape[0]):
        print(f"Kernel k{k}:")
        print(np.round(W[k, 0], decimals))
        print()