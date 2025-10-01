import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import CNN_load_MNIST as ldmnist
import CNN_model as cnnmodel
import CNN_visualisation as cnnvis

#########################################################################################

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, num_epochs: int = 5):
    for epoch in range(num_epochs):

        # print(f"Epoch {epoch+1}/{num_epochs}, lr={optimizer.param_groups[0]['lr']:.6f}")

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

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

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

def plot_predictions(model, test_loader, device):
    model.eval()

    np.random.seed(random_seed)

    # Grab a random batch from the test loader
    xb, yb = next(iter(test_loader))
    xb, yb = xb.to(device), yb.to(device)

    with torch.no_grad():
        logits = model(xb)
        preds = logits.argmax(dim=1)

    images = xb.cpu().numpy()
    labels = yb.cpu().numpy()
    preds  = preds.cpu().numpy()

    # Plot a grid
    n_rows, n_cols = 4, 8   # adjust grid size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        ax.imshow(images[i,0], cmap='gray', interpolation='nearest')
        
        # choose title color
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(f"P:{preds[i]} / L:{labels[i]}", fontsize=12, color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#########################################################################################


random_seed = 0
torch.manual_seed(random_seed)

# 0. pick GPU if available, else CPU ----------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 1. Load MNIST data -------------------------------------------------------

# (x_train, y_train), (x_val, y_val), (x_test, y_test) = ldmnist.load_torchvision_data(torchvision.datasets.MNIST)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = ldmnist.load_torchvision_data(torchvision.datasets.FashionMNIST)

train_ds = TensorDataset(x_train, y_train)
val_ds   = TensorDataset(x_val,   y_val)
test_ds  = TensorDataset(x_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)


# 2. Create model ------------------------------------------------------------

input_size = x_train.shape[2]  # 28 for MNIST
model = cnnmodel.SimpleCNN(input_size=input_size, num_classes=10).to(device)
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = None
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



# 3. Train ---------------------------------------------------------------

num_epochs = 50
train_epoch(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs)

# 4. Test ----------------------------------------------------------------

test_model(model, test_loader, device)
# plot_predictions(model, test_loader, device)

# After training:
# cnnvis.show_conv1_kernels(model)
cnnvis.show_kernel_frequency_response(model)
# cnnvis.print_kernels(model)

# xb, yb = next(iter(test_loader))
# cnnvis.show_conv1_feature_maps(model, xb[:1], device=device)
