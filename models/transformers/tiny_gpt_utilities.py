import torch
import matplotlib.pyplot as plt
import numpy as np
from models.transformers.tiny_gpt_config import Config

def _as_device(device):
    """Return a torch.device from either a device or string."""
    return device if isinstance(device, torch.device) else torch.device(device)

def get_batch(data, block_size, batch_size, device):
    device = _as_device(device)
    # Ensure dataset is long enough to form at least one (x, y) pair
    if len(data) < 2:
        raise ValueError(f"Dataset too short: len(data)={len(data)}; need at least 2 tokens.")

    # Use the largest feasible window, capped by block_size
    T = min(block_size, len(data) - 1)

    # Calculate the upper bound for random start indices
    # Valid start indices i in [0, len(data) - T - 1]; torch.randint high is exclusive
    high = len(data) - T
    if high <= 0:
        # This should not happen due to T=min(block_size, len(data)-1), but guard anyway
        raise ValueError(f"Invalid batch window: len(data)={len(data)}, T={T}, high={high}")

    # Sample random start positions for each sequence in the batch
    ix = torch.randint(low=0, high=high, size=(batch_size,))
    
    # Create input sequences (context)
    x = torch.stack([data[i:i+T] for i in ix])
    
    # Create target sequences (next token prediction)
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    
    # Move tensors to the specified device
    return x.to(device), y.to(device)

def estimate_loss(model, data, cfg: Config, device=None):
    # Accept explicit runtime device; fall back to cfg.device string
    device = _as_device(device if device is not None else cfg.device)
    # Switch model to evaluation mode (disables dropout, etc.)
    model.eval()
    # Disable gradient computation for inference
    with torch.no_grad():
        # Sample a batch from the data
        X, Y = get_batch(data, cfg.block_size, cfg.batch_size, device)
        # Forward pass to get logits and loss
        logits, loss = model(X, Y)
    # Switch model back to training mode
    model.train()
    # Return scalar loss value
    return loss.item()

def plot_training_curves(history_df, save_path=None):
    """
    Plot training curves for loss and accuracy metrics.
    
    Args:
        history_df: pandas DataFrame with columns: step, train_loss, val_loss, 
                   perplexity, top1_acc, top5_acc
        save_path: Optional path to save the figure (e.g., 'training_curves.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss and Perplexity curves
    ax1 = axes[0]
    line1 = ax1.plot(history_df['step'], history_df['train_loss'], 'o-', label='Train Loss', alpha=0.7)
    line2 = ax1.plot(history_df['step'], history_df['val_loss'], 's-', label='Val Loss', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    
    # Add secondary y-axis for perplexity
    ax1_perp = ax1.twinx()
    line3 = ax1_perp.plot(history_df['step'], history_df['perplexity'], '^--', label='Perplexity', alpha=0.6, color='purple')
    ax1_perp.set_ylabel('Perplexity')
    ax1_perp.set_ylim(bottom=0)
    
    # Combine legends from both axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Plot 2: Accuracy curves
    ax2 = axes[1]
    ax2.plot(history_df['step'], history_df['top1_acc'] * 100, 'o-', label='Top-1 Accuracy', alpha=0.7)
    ax2.plot(history_df['step'], history_df['top5_acc'] * 100, 's-', label='Top-5 Accuracy', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Top-1 and Top-5 Accuracy')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()