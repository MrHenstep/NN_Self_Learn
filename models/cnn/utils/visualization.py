import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def find_normalize(transform):
    if transform is None:
        return None
    if isinstance(transform, transforms.Compose):
        for t in transform.transforms:
            if isinstance(t, transforms.Normalize):
                return (torch.tensor(t.mean).view(-1,1,1), torch.tensor(t.std).view(-1,1,1))
    elif isinstance(transform, transforms.Normalize):
        return (torch.tensor(transform.mean).view(-1,1,1), torch.tensor(transform.std).view(-1,1,1))
    return None


def prepare_for_display(batch, norm_params=None):
    x = batch.detach().cpu()
    if norm_params is not None:
        mean, std = norm_params
        x = x * std + mean
    x_min = x.amin(dim=(1,2,3), keepdim=True)
    x_max = x.amax(dim=(1,2,3), keepdim=True)
    needs_rescale = (x_min < 0).any() or (x_max > 1).any()
    if needs_rescale:
        denom = torch.clamp(x_max - x_min, min=1e-8)
        x = (x - x_min) / denom
    x = torch.clamp(x, 0.0, 1.0)
    x = x.permute(0, 2, 3, 1).numpy()
    return x


def plot_predictions(model, test_loader, device, class_names=None):
    model.eval()
    torch.manual_seed(0)

    norm_params = find_normalize(getattr(test_loader.dataset, "transform", None))
    xb, yb = next(iter(test_loader))
    xb, yb = xb.to(device), yb.to(device)

    with torch.inference_mode():
        preds = model(xb).argmax(dim=1)

    is_rgb = (xb.shape[1] == 3)
    imgs_disp = prepare_for_display(xb, norm_params=norm_params)

    n_rows, n_cols = 4, 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4.5))
    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes):
        if i >= len(xb):
            ax.axis('off'); continue
        if is_rgb:
            ax.imshow(imgs_disp[i])
        else:
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
        plt.plot(epochs, lr, label='Learning Rate', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1,2,2)
    te = history_df.get('train_err')
    ve = history_df.get('val_err')
    if te is not None and ve is not None:
        plt.plot(epochs, te, label='Train Error')
        plt.plot(epochs, ve, label='Val Error')
        if lr is not None:
            plt.plot(epochs, lr, label='Learning Rate', linestyle='--')
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
