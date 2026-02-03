"""
Contrastive Learning Training Script

This script implements SimCLR-style contrastive learning with a two-phase approach:
1. Pre-training: Train encoder with NT-Xent loss on augmented image pairs
2. Fine-tuning: Train classifier on top of frozen or unfrozen encoder

Usage:
    python scripts/train_contrastive.py

Configuration:
    Edit the config sections below to customize the training.
"""

import sys
from pathlib import Path
import subprocess
import datetime
import time
import torch
import torch.nn as nn
import os
import pandas as pd

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Change cwd to _ROOT if it's different
if os.getcwd() != str(_ROOT):
    os.chdir(_ROOT)

from torchvision import datasets

from models.cnn.config import DataConfig, ModelConfig, ContrastiveConfig, DataMetadata
from models.cnn.architectures.factory import build_model
from models.cnn.architectures.simclr import SimCLRModel, build_simclr_model
from models.cnn.losses import NTXentLoss
from models.cnn.utils.optimization import build_optimizer, build_scheduler
from data_loading.loaders import build_dataloaders
from data_loading.contrastive_transforms import simclr_augment_cifar, simclr_eval_transform
from data_loading.contrastive_dataset import TwoViewDataset
from data_loading import datasets as data_datasets


def pretrain_contrastive(
    model: SimCLRModel,
    train_loader: torch.utils.data.DataLoader,
    config: ContrastiveConfig,
    device: torch.device,
) -> pd.DataFrame:
    """
    Phase 1: Contrastive pre-training with NT-Xent loss.
    
    Args:
        model: SimCLRModel (backbone + projection head)
        train_loader: DataLoader yielding (view1, view2, label) tuples
        config: ContrastiveConfig with training hyperparameters
        device: torch device
        
    Returns:
        DataFrame with training history
    """
    model.train()
    model.to(device)
    
    criterion = NTXentLoss(temperature=config.temperature)
    
    # Use SGD with cosine annealing (SimCLR default)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.pretrain_lr,
        momentum=0.9,
        weight_decay=config.pretrain_weight_decay,
    )
    
    # Cosine annealing with warmup
    total_steps = config.pretrain_epochs * len(train_loader)
    warmup_steps = config.warmup_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    rows = []
    global_step = 0
    
    print(f"\n{'='*60}")
    print("PHASE 1: Contrastive Pre-training")
    print(f"{'='*60}")
    print(f"Epochs: {config.pretrain_epochs}, Batch size: {config.pretrain_batch_size}")
    print(f"Temperature: {config.temperature}, Projection dim: {config.projection_dim}")
    print(f"{'='*60}\n")
    
    for epoch in range(config.pretrain_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        time_start = time.time()
        
        for view1, view2, _ in train_loader:
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass: get projections for both views
            z1 = model(view1)
            z2 = model(view2)
            
            # Compute contrastive loss
            loss = criterion(z1, z2)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            n_batches += 1
            global_step += 1
        
        epoch_loss = running_loss / n_batches
        time_elapsed = time.time() - time_start
        lr = optimizer.param_groups[0]['lr']
        
        if epoch == 0:
            print(f"{'epoch':>5} {'loss':>10} {'lr':>12} {'time(s)':>8}")
        print(f"{epoch+1:5d} {epoch_loss:10.4f} {lr:12.6f} {time_elapsed:8.2f}")
        
        rows.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'learning_rate': lr,
            'time_elapsed': time_elapsed,
        })
    
    return pd.DataFrame(rows)


def finetune_classifier(
    model: SimCLRModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: ContrastiveConfig,
    num_classes: int,
    device: torch.device,
) -> tuple[pd.DataFrame, nn.Module]:
    """
    Phase 2: Fine-tune classifier on top of pre-trained backbone.
    
    Args:
        model: Pre-trained SimCLRModel
        train_loader: Standard supervised DataLoader (images, labels)
        val_loader: Validation DataLoader
        config: ContrastiveConfig with fine-tuning hyperparameters
        num_classes: Number of output classes
        device: torch device
        
    Returns:
        Tuple of (history DataFrame, fine-tuned backbone with classifier)
    """
    # Get the backbone and reset/reinitialize the classifier
    backbone = model.get_backbone()
    
    # Reinitialize the classifier head
    feature_dim = model.feature_dim
    backbone.fc = nn.Linear(feature_dim, num_classes)
    nn.init.normal_(backbone.fc.weight, 0, 0.01)
    nn.init.zeros_(backbone.fc.bias)
    backbone.to(device)
    
    # Freeze backbone if doing linear evaluation
    if config.linear_eval:
        for name, param in backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("Linear evaluation: backbone frozen, only training classifier")
        params = backbone.fc.parameters()
    else:
        print("Full fine-tuning: training entire network")
        params = backbone.parameters()
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(
        params,
        lr=config.finetune_lr,
        momentum=0.9,
        weight_decay=config.finetune_weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.finetune_epochs,
        eta_min=1e-4
    )
    
    rows = []
    
    print(f"\n{'='*60}")
    print("PHASE 2: Fine-tuning Classifier")
    print(f"{'='*60}")
    print(f"Epochs: {config.finetune_epochs}, Linear eval: {config.linear_eval}")
    print(f"Learning rate: {config.finetune_lr}")
    print(f"{'='*60}\n")
    
    for epoch in range(config.finetune_epochs):
        backbone.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        time_start = time.time()
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = backbone(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(1) == yb).sum().item()
            n += bs
        
        train_loss = running_loss / n
        train_acc = running_acc / n
        
        # Validation
        backbone.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = backbone(xb)
                loss = criterion(logits, yb)
                bs = xb.size(0)
                val_loss += loss.item() * bs
                val_acc += (logits.argmax(1) == yb).sum().item()
                n_val += bs
        val_loss /= n_val
        val_acc /= n_val
        
        time_elapsed = time.time() - time_start
        lr = optimizer.param_groups[0]['lr']
        
        if epoch == 0:
            print(f"{'epoch':>5} {'train_loss':>10} {'train_acc':>10} {'val_loss':>10} {'val_acc':>10} {'lr':>10} {'time(s)':>8}")
        print(f"{epoch+1:5d} {train_loss:10.4f} {train_acc:10.4f} {val_loss:10.4f} {val_acc:10.4f} {lr:10.6f} {time_elapsed:8.2f}")
        
        rows.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': lr,
            'time_elapsed': time_elapsed,
        })
        
        scheduler.step()
    
    return pd.DataFrame(rows), backbone


def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    """Evaluate model on test set."""
    model.eval()
    test_acc, n_test = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            test_acc += (logits.argmax(1) == yb).sum().item()
            n_test += yb.size(0)
    test_acc /= n_test
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    return test_acc


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
    torch.manual_seed(42)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ========== CONFIGURATION ==========
    
    # Model: ResNet-20 for CIFAR (resnet_n=3 gives 6*3+2=20 layers)
    model_cfg = ModelConfig(
        model_name="resnet",
        resnet_n=3,  # ResNet-20
        use_projection=False,  # Use Option A downsampling for CIFAR
        use_residual=True,
    )
    
    # Contrastive learning config
    contrastive_cfg = ContrastiveConfig(
        # Loss
        temperature=0.5,
        
        # Projection head
        projection_dim=128,
        projection_hidden=512,
        use_bn_in_head=True,
        
        # Pre-training (reduce epochs for testing; use 500+ for good results)
        pretrain_epochs=200,
        pretrain_lr=0.5,
        pretrain_weight_decay=1e-4,
        pretrain_batch_size=256,
        warmup_epochs=10,
        
        # Fine-tuning
        finetune_epochs=200,
        finetune_lr=0.1,
        finetune_weight_decay=1e-3,
        linear_eval=False,  # Set True for linear evaluation only
        
        # Augmentation
        color_jitter_strength=1.0,
    )
    
    # ========== DATA LOADING ==========
    
    print("\nPreparing datasets...")
    
    # Get CIFAR-10 statistics
    data_root = "./data"
    
    # Load raw dataset to compute statistics
    raw_cifar = datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)
    
    # CIFAR-10 standard statistics
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    
    # Create contrastive augmentation transform
    contrastive_transform = simclr_augment_cifar(
        mean=mean,
        std=std,
        image_size=32,
        s=contrastive_cfg.color_jitter_strength,
    )
    
    # Wrap with TwoViewDataset for contrastive training
    contrastive_dataset = TwoViewDataset(
        base_dataset=raw_cifar,
        transform=contrastive_transform,
        return_label=True,
    )
    
    contrastive_loader = torch.utils.data.DataLoader(
        contrastive_dataset,
        batch_size=contrastive_cfg.pretrain_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Important for contrastive learning
    )
    
    # Standard supervised loaders for fine-tuning
    data_cfg = DataConfig(dataset_key="cifar10", use_augment=True)
    train_loader, val_loader, test_loader, data_meta = build_dataloaders(data_cfg, device)
    
    # ========== BUILD MODEL ==========
    
    print("\nBuilding model...")
    
    # Build backbone (ResNet-20)
    backbone = build_model(model_cfg, data_meta)
    print(f"Backbone: ResNet-{6*model_cfg.resnet_n + 2}")
    
    # Wrap with projection head for contrastive learning
    simclr_model = build_simclr_model(
        backbone=backbone,
        projection_dim=contrastive_cfg.projection_dim,
        projection_hidden=contrastive_cfg.projection_hidden,
        use_bn_in_head=contrastive_cfg.use_bn_in_head,
    )
    
    total_params = sum(p.numel() for p in simclr_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # ========== PHASE 1: CONTRASTIVE PRE-TRAINING ==========
    
    pretrain_history = pretrain_contrastive(
        model=simclr_model,
        train_loader=contrastive_loader,
        config=contrastive_cfg,
        device=device,
    )
    
    # Save pre-trained model
    checkpoint_path = Path("checkpoints/simclr_pretrained.pth")
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save({
        'backbone_state_dict': simclr_model.backbone.state_dict(),
        'projection_head_state_dict': simclr_model.projection_head.state_dict(),
        'config': contrastive_cfg,
    }, checkpoint_path)
    print(f"\nPre-trained model saved to {checkpoint_path}")
    
    # ========== PHASE 2: FINE-TUNING ==========
    
    finetune_history, finetuned_model = finetune_classifier(
        model=simclr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=contrastive_cfg,
        num_classes=data_meta.num_classes,
        device=device,
    )
    
    # ========== TESTING ==========
    
    print("\nEvaluating on test set...")
    test_acc = test_model(finetuned_model, test_loader, device)
    
    # Save fine-tuned model
    finetune_path = Path("checkpoints/simclr_finetuned.pth")
    torch.save({
        'model_state_dict': finetuned_model.state_dict(),
        'test_accuracy': test_acc,
        'config': contrastive_cfg,
    }, finetune_path)
    print(f"Fine-tuned model saved to {finetune_path}")
    
    # Save training histories
    pretrain_history.to_csv("checkpoints/simclr_pretrain_history.csv", index=False)
    finetune_history.to_csv("checkpoints/simclr_finetune_history.csv", index=False)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    print("="*60)
