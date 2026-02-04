import sys
from pathlib import Path
import subprocess
import datetime
import torch
import os

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# change cwd to _ROOT if it's different - copes with running in interactive window
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

from models.cnn.config import DataConfig, ModelConfig, TrainConfig
from models.cnn.engine import run_training
from data_loading.loaders import build_dataloaders


#########################################################################################

if __name__ == "__main__":

    # Git commit and timestamp info
    try:
        info = subprocess.check_output(["git", "show", "-s", "--format=%H%n%s"], text=True).strip().splitlines()
        if len(info) >= 2:
            print(f"Commit Message: {info[1]}")
    except Exception:
        print("Commit info unavailable.")

    now = datetime.datetime.now()
    print("Start time:", now.strftime("%Y-%m-%d %H:%M:%S")) 

    # Reproducibility
    torch.manual_seed(0)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- Configuration (edit here just before run) ----
    # data_cfg = DataConfig(dataset_key="tiny_imagenet", use_augment=None)
    data_cfg = DataConfig(dataset_key="cifar10", use_augment=True)
    model_cfg = ModelConfig(model_name="resnet", resnet_n=3, use_projection=None, use_residual=True)
    train_cfg = TrainConfig(
        num_epochs=200,
        mixup_alpha=None,
        label_smoothing=0.05,
        optimizer="sgd",
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=False,
        scheduler="warmup_cosine",
        warmup_epochs=7,
        min_lr=1e-3,
        use_ema=False,
        ema_decay=0.99,
        plot_curves=True,
        test_after_training=True,
    )

    print("Building dataloaders...")
    train_loader, val_loader, test_loader, data_meta = build_dataloaders(data_cfg, device)

    history_df, model, ema_model = run_training(
        model_cfg, 
        train_cfg, 
        device,
        train_loader,
        val_loader,
        test_loader,
        data_meta
    )

    # Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Use EMA model if available, otherwise use regular model
    save_model = ema_model if ema_model is not None else model
    
    # Build checkpoint filename from config
    checkpoint_name = f"resnet{6*model_cfg.resnet_n + 2}_{data_cfg.dataset_key}.pth"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Get final validation accuracy from history
    final_val_acc = history_df['val_acc'].iloc[-1] if 'val_acc' in history_df.columns else None
    
    checkpoint = {
        'model_state_dict': save_model.state_dict(),
        'model_config': {
            'model_name': model_cfg.model_name,
            'resnet_n': model_cfg.resnet_n,
            'use_projection': model_cfg.use_projection,
            'use_residual': model_cfg.use_residual,
        },
        'data_config': {
            'dataset_key': data_cfg.dataset_key,
            'num_classes': data_meta.num_classes,
        },
        'train_config': {
            'num_epochs': train_cfg.num_epochs,
            'learning_rate': train_cfg.learning_rate,
            'weight_decay': train_cfg.weight_decay,
        },
        'final_val_acc': final_val_acc,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    if final_val_acc is not None:
        print(f"  Final validation accuracy: {final_val_acc:.2%}")
    
    # Also save training history
    history_path = checkpoint_dir / f"resnet{6*model_cfg.resnet_n + 2}_{data_cfg.dataset_key}_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
