import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

import CNN_load_datasets as ldd
import CNN_model as cnnmodel
import CNN_model_2 as cnnflexi
import CNN_visualisation as cnnvis

import time

#########################################################################################

def train_epochs(model, train_loader, criterion, optimizer, scheduler, device, num_epochs: int = 5):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):

        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_acc  += (logits.argmax(1) == yb).sum().item()
            n += bs

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / n
        train_acc  = running_acc / n
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # --- Validate ---
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                bs = xb.size(0)
                val_loss += loss.item() * bs
                val_acc  += (logits.argmax(1) == yb).sum().item()
                n_val += bs
        val_loss /= n_val
        val_acc  /= n_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

    return train_losses, train_accs, val_losses, val_accs

def test_model(model, test_loader, device):
    
    model.eval()
    test_acc, n_test = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            test_acc += (logits.argmax(1) == yb).sum().item()
            n_test += xb.size(0)
    print(f"Test accuracy: {test_acc / n_test:.4f}")

def _find_normalize(transform):
    """Return (mean, std) if a transforms.Normalize is present; else None."""
    if transform is None:
        return None
    # unwrap Compose if needed
    if isinstance(transform, transforms.Compose):
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                return (torch.tensor(t.mean).view(-1,1,1),
                        torch.tensor(t.std).view(-1,1,1))
    elif isinstance(transform, transforms.Normalize):
        return (torch.tensor(transform.mean).view(-1,1,1),
                torch.tensor(transform.std).view(-1,1,1))
    return None

def _prepare_for_display(batch, norm_params=None):
    """
    batch: torch.Tensor (N, C, H, W) in float
    norm_params: (mean, std) as tensors shaped (C,1,1) or None
    returns np.ndarray (N, H, W, C) in [0,1]
    """
    x = batch.detach().cpu()
    if norm_params is not None:
        mean, std = norm_params
        x = x * std + mean  # unnormalize

    # If still not in [0,1], do per-image min-max for display only
    x_min = x.amin(dim=(1,2,3), keepdim=True)
    x_max = x.amax(dim=(1,2,3), keepdim=True)
    needs_rescale = (x_min < 0).any() or (x_max > 1).any()
    if needs_rescale:
        # avoid div-by-zero on constant images
        denom = torch.clamp(x_max - x_min, min=1e-8)
        x = (x - x_min) / denom

    x = torch.clamp(x, 0.0, 1.0)
    x = x.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
    return x

def plot_predictions(model, test_loader, device, class_names=None):
    model.eval()
    torch.manual_seed(0)

    # Try to discover Normalize(mean,std) in the dataset
    norm_params = _find_normalize(getattr(test_loader.dataset, "transform", None))

    # Grab a random batch from the test loader
    xb, yb = next(iter(test_loader))
    xb, yb = xb.to(device), yb.to(device)

    with torch.inference_mode():
        preds = model(xb).argmax(dim=1)

    # Prepare images for display
    is_rgb = (xb.shape[1] == 3)
    imgs_disp = _prepare_for_display(xb, norm_params=norm_params)

    n_rows, n_cols = 4, 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4.5))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes):
        if i >= len(xb):
            ax.axis('off'); continue

        if is_rgb:
            ax.imshow(imgs_disp[i])  # already (H,W,C) in [0,1]
        else:
            # single-channel
            gray = imgs_disp[i][..., 0]
            ax.imshow(gray, cmap='gray', interpolation='nearest')

        pred_i = preds[i].item()
        label_i = yb[i].item()
        ptxt = class_names[pred_i] if class_names else str(pred_i)
        ltxt = class_names[label_i] if class_names else str(label_i)

        color = 'green' if pred_i == label_i else 'red'
        ax.set_title(f"P:{ptxt} / L:{ltxt}", fontsize=9, color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_training_curves(tl, ta, vl, va, num_epochs):
    epochs = np.arange(1, num_epochs+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, tl, label='Train Loss')
    plt.plot(epochs, vl, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, ta, label='Train Acc')
    plt.plot(epochs, va, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

#########################################################################################

if __name__ == "__main__":

    random_seed = 0
    torch.manual_seed(random_seed)

    # 0. pick GPU if available, else CPU ----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # 1. Load MNIST data -------------------------------------------------------

    augment = True
    # data_set = torchvision.datasets.MNIST
    
    # data_set = torchvision.datasets.FashionMNIST
    # (x_train, y_train), (x_val, y_val), (x_test, y_test) = ldd.load_torchvision_data_MNIST(data_set, augment=augment)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = ldd.load_torchvision_data_cifar10(augment=augment)

    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val,   y_val)
    test_ds  = TensorDataset(x_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False, num_workers=4)


    # 2. Create model ------------------------------------------------------------

    input_channels = x_train.shape[1]  
    input_size = x_train.shape[2]  
    
    # model = cnnmodel.SimpleCNN(input_size=input_size, num_classes=10).to(device)
    model = cnnflexi.SimpleCNNFlexi(input_channels=input_channels, input_size=input_size, num_classes=10).to(device)
    
    print(model)
    
    # assert False, "Test stop"

    # 3. Train ---------------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    

    num_epochs = 200

    # scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.2, total_steps=len(train_loader)*num_epochs,pct_start=0.3)


    start_time = time.time()
    
    tl, ta, vl, va = train_epochs(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs)
    
    end_time = time.time()
    print(f"Training time for per epoch: {(end_time - start_time)/num_epochs:.2f} seconds")


    # 4. Test ----------------------------------------------------------------

    test_model(model, test_loader, device)


    # Plot training curves
    plot_training_curves(tl, ta, vl, va, num_epochs)

    # plot_predictions(model, test_loader, device)

    # After training:
    # cnnvis.show_conv1_kernels(model)
    # cnnvis.show_kernel_frequency_response(model)
    # cnnvis.print_kernels(model)

    # xb, yb = next(iter(test_loader))
    # cnnvis.show_conv1_feature_maps(model, xb[:1], device=device)

    # cnnvis.show_conv_kernel(model.model.conv_block_1[0])  # Show kernels of first conv layer
    # cnnvis.show_kernel_frequency_response(model.model.conv_block_1[0])  # Show freq response of first conv layer

    # cnnvis.show_conv_kernel(model.model.conv_block_2[0])  # Show kernels of second conv layer
    # cnnvis.show_kernel_frequency_response(model.model.conv_block_2[0])  # Show freq response of second conv layer