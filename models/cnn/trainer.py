import time
from typing import Optional
import torch
import pandas as pd

from data_loading.augmentation import mixup_batch


def train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs: int = 5, mixup_alpha: Optional[float] = None, ema_model: Optional[torch.nn.Module] = None):
    rows = []
    use_mixup = mixup_alpha is not None and mixup_alpha > 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        time_start = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            if use_mixup:
                xb, targets_a, targets_b, lam = mixup_batch(xb, yb, mixup_alpha)
                logits = model(xb)
                loss_vec = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
                preds = logits.argmax(1)
                batch_acc = lam * (preds == targets_a).float() + (1.0 - lam) * (preds == targets_b).float()
                running_acc += batch_acc.sum().item()
            else:
                logits = model(xb)
                loss_vec = criterion(logits, yb)
                preds = logits.argmax(1)
                running_acc += (preds == yb).sum().item()

            loss = loss_vec.mean()
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)

            bs = xb.size(0)
            running_loss += loss_vec.sum().item()
            n += bs

        train_loss = running_loss / n
        train_acc  = running_acc / n

        model.eval()
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = eval_model(xb)
                loss_vec = criterion(logits, yb)
                bs = xb.size(0)
                val_loss += loss_vec.sum().item()
                val_acc  += (logits.argmax(1) == yb).sum().item()
                n_val += bs
        val_loss /= n_val
        val_acc  /= n_val

        time_elapsed = time.time() - time_start

        lr = optimizer.param_groups[0]['lr']
        train_err = 1.0 - train_acc
        val_err = 1.0 - val_acc

        if epoch == 0:
            print(f"{'epoch':>5} {'train_loss':>10} {'train_err':>10} {'val_loss':>10} {'val_err':>10} {'lr':>10} {'time(s)':>8}")
        print(f"{epoch+1:5d} {train_loss:10.4f} {train_err:10.4f} {val_loss:10.4f} {val_err:10.4f} {lr:10.6f} {time_elapsed:8.2f}")

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

        if scheduler is not None:
            scheduler.step()

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
    print(f"Test error: {1.0 - (test_acc / n_test):.4f}")
