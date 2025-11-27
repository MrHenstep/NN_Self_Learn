import time
import torch
import sys

from .config import ModelConfig, TrainConfig, DataMetadata
# from data_loading.loaders import build_dataloaders
from .architectures.factory import build_model
from .utils.optimization import build_optimizer, build_scheduler, build_ema
from .trainer import train_epochs, test_model
from .utils.visualization import plot_training_curves
from torch.utils.data import DataLoader


def run_training(
    model_cfg: ModelConfig, 
    train_cfg: TrainConfig, 
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    data_meta: DataMetadata
):
    
    # print("Building dataloaders...")
    # train_loader, val_loader, test_loader, data_meta = build_dataloaders(data_cfg, device)

    print("Building model...")
    model = build_model(model_cfg, data_meta).to(device)
    print(model)

    print("Setting up training components...")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing, reduction='none')
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    ema_model = build_ema(model, train_cfg, device)

    print("Starting training...")
    start_time = time.time()
    history_df = train_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=train_cfg.num_epochs,
        mixup_alpha=train_cfg.mixup_alpha,
        ema_model=ema_model,
    )

    epoch_time = (time.time() - start_time) / max(train_cfg.num_epochs, 1)
    print(f"Training time per epoch: {epoch_time:.2f} seconds")

    if train_cfg.test_after_training:
        print("Testing model after training...")
        eval_model = ema_model if ema_model is not None else model
        test_model(eval_model, test_loader, device)

    if train_cfg.plot_curves:
        print("Plotting training curves...")
        plot_training_curves(history_df)

    return history_df, model, ema_model
