import importlib
import sys
from pathlib import Path
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __package__:
    from . import CNN_load_datasets as ldd
    from . import CNN_model as cnnmodel
    from . import CNN_model_2 as cnnflexi
    from . import CNN_visualisation as cnnvis
    from . import ResNet_model as rn
    ldd = importlib.reload(ldd)
else:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from cnns import CNN_load_datasets as ldd
    from cnns import CNN_model as cnnmodel
    from cnns import CNN_model_2 as cnnflexi
    from cnns import CNN_visualisation as cnnvis
    from cnns import ResNet_model as rn
    ldd = importlib.reload(ldd)

#########################################################################################

def train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs: int = 5):
    """Train for `num_epochs` and return a pandas DataFrame with per-epoch metrics.

    Args:
        model: torch.nn.Module
        train_loader: DataLoader for training
        val_loader: DataLoader for validation (required)
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler (or None)
        device: torch.device
        num_epochs: number of epochs to run

    Returned DataFrame columns: ['epoch','train_loss','train_acc','val_loss','val_acc',
    'train_err','val_err','learning_rate','time_elapsed']
    """
    rows = []

    for epoch in range(num_epochs):

        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        time_start = time.time()

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

        time_end = time.time()
        time_elapsed = time_end - time_start

        lr = optimizer.param_groups[0]['lr']
        train_err = 1.0 - train_acc
        val_err = 1.0 - val_acc

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_err={train_err:.4f} | val_loss={val_loss:.4f} val_err={val_err:.4f} | lr={lr:.6f} | time={time_elapsed:.2f}s")

        rows.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_err': train_err,
            'val_err': val_err,
            'learning_rate': lr,
            'time_elapsed': time_elapsed,
        })

    history_df = pd.DataFrame(rows)
    return history_df

def test_model(model, test_loader, device):
    
    model.eval()
    test_acc, n_test = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            test_acc += (logits.argmax(1) == yb).sum().item()
            n_test += xb.size(0)
    # print(f"Test accuracy: {test_acc / n_test:.4f}")
    print(f"Test error: {1.0 - (test_acc / n_test):.4f}")

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

def plot_training_curves(history_df):
    """Plot training/validation curves from the history DataFrame returned by train_epochs."""
    num_epochs = len(history_df)
    epochs = np.arange(1, num_epochs+1)
    tl = history_df['train_loss']
    vl = history_df['val_loss']
    ta = history_df['train_acc']
    va = history_df['val_acc']
    lr = history_df.get('learning_rate', None)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, tl, label='Train Loss')
    plt.plot(epochs, vl, label='Val Loss')
    if lr is not None:
        plt.plot(epochs, lr * (lr.max() / lr.max()), label='Learning Rate', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1,2,2)
    # Plot error (1 - accuracy) instead of accuracy when available
    te = history_df.get('train_err')
    ve = history_df.get('val_err')
    if te is not None and ve is not None:
        plt.plot(epochs, te, label='Train Error')
        plt.plot(epochs, ve, label='Val Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Training and Validation Error')
    else:
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


    # 1. Select dataset and augmentation ------------------------------------------------

    dataset_key = "CIFAR10"   # options: "cifar10", "mnist", "fashion_mnist"
    model_choice = "resnet20"  # options: "resnet20", "simplecnn", "cnnflexi"
    use_augment: Optional[bool] = None  # set to True/False to override dataset default

    bundle = ldd.load_dataset(dataset_key, augment=use_augment)
    train_ds, val_ds, test_ds = bundle.train, bundle.val, bundle.test

    num_classes = len(bundle.class_names) if bundle.class_names is not None else 10
    input_channels = bundle.num_channels
    input_size = bundle.image_size

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, drop_last=False, num_workers=4)


    # 2. Create model ------------------------------------------------------------

    if model_choice == "simplecnn":
        model = cnnmodel.SimpleCNN(input_size=input_size, num_classes=num_classes)
    elif model_choice == "cnnflexi":
        model = cnnflexi.SimpleCNNFlexi(input_channels=input_channels, input_size=input_size, num_classes=num_classes)
        model.make_VGG()
    elif model_choice == "resnet20":
        if input_channels != 3:
            raise ValueError("ResNet20 expects 3-channel inputs; choose a different model for this dataset.")
        model = rn.ResNet(n_classes=num_classes, resnet_n=3, use_projection=False)
    else:
        raise ValueError(f"Unknown model_choice '{model_choice}'.")

    model = model.to(device)
    # print(model)
    
    # assert False, "Test stop"

    # 3. Train ---------------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    # Build optimizer with two param groups: apply weight decay to weights, but
    # exclude BatchNorm parameters and biases from weight decay (common best-practice).
    # Build optimizer param groups by module type to reliably exclude BatchNorm
    # parameters and all biases from weight decay.
    decay_params = []
    no_decay_params = []

    # Collect ids of BatchNorm parameters
    bn_param_ids = set()
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            for p in module.parameters():
                bn_param_ids.add(id(p))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in bn_param_ids or name.endswith('.bias'):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    print(f"Optimizer params: decay={len(decay_params)} no_decay={len(no_decay_params)} (bn_params={len(bn_param_ids)})")

    optimizer = torch.optim.SGD([
        {'params': decay_params, 'weight_decay': 1e-4},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=0.1, momentum=0.9)
    

    num_epochs = 200

    # scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Use the common ResNet/CIFAR step schedule: drop LR by 10 at epochs 80 and 120
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    # Alternative: cosine annealing
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.2, total_steps=len(train_loader)*num_epochs,pct_start=0.3)

    start_time = time.time()

    history_df = train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs)

    end_time = time.time()
    print(f"Training time for per epoch: {(end_time - start_time)/num_epochs:.2f} seconds")

    # 4. Test ----------------------------------------------------------------

    test_model(model, test_loader, device)

    # Plot training curves
    plot_training_curves(history_df)
