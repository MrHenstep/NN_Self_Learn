import sys
from pathlib import Path
import subprocess
import datetime
import torch

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
    data_cfg = DataConfig(dataset_key="tiny_imagenet", use_augment=None)
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
