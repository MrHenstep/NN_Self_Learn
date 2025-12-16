import torch
import matplotlib.pyplot as plt
import numpy as np
from models.transformers.tiny_gpt_config import Config

def _as_device(device):
    """Return a torch.device from either a device or string."""
    return device if isinstance(device, torch.device) else torch.device(device)

def get_batch(data, block_size, batch_size, device, bos_id=None):
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

    if bos_id is not None and bos_id >= 0:
        # Ensure every window starts with BOS for a stable start-of-sequence signal
        x[:, 0] = bos_id
    
    # Move tensors to the specified device
    return x.to(device), y.to(device)

def estimate_loss(model, data, cfg: Config, device=None, bos_id=None):
    # Accept explicit runtime device; fall back to cfg.device string
    device = _as_device(device if device is not None else cfg.device)
    # Switch model to evaluation mode (disables dropout, etc.)
    model.eval()
    # Disable gradient computation for inference
    with torch.no_grad():
        # Sample a batch from the data
        X, Y = get_batch(data, cfg.block_size, cfg.batch_size, device, bos_id=bos_id)
        # Forward pass to get logits and loss
        logits, loss = model(X, Y)
    # Switch model back to training mode
    model.train()
    # Return scalar loss value
    return loss.item()

def compute_self_repetition_ratio(model, tokenizer, prompt_text, device, num_samples=3, max_tokens=100):
    """
    Generate multiple samples from the same prompt and compute diversity.
    Returns a score (0-1) where 1 = all unique bigrams, 0 = all identical.
    """
    model.eval()
    with torch.no_grad():
        # Encode prompt with BOS if available
        prompt_ids = tokenizer.encode(prompt_text)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None:
            prompt_ids = [bos_id] + prompt_ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        # Generate multiple samples
        samples = []
        for _ in range(num_samples):
            out = model.generate(prompt_tensor, max_new_tokens=max_tokens)
            samples.append(out[0].tolist())
        
        # Collect all bigrams
        unique_bigrams = set()
        total_bigrams = 0
        for sample in samples:
            bigrams = list(zip(sample[:-1], sample[1:]))
            unique_bigrams.update(bigrams)
            total_bigrams += len(bigrams)
        
        # Compute diversity ratio
        diversity = len(unique_bigrams) / max(total_bigrams, 1)
    
    model.train()
    return diversity

def compute_vocab_diversity(model, tokenizer, prompt_text, device, max_tokens=100):
    """
    Generate one sample and measure fraction of vocab used.
    Returns a score (0-1) where 1 = all vocab tokens used, 0 = no diversity.
    """
    model.eval()
    with torch.no_grad():
        # Encode prompt with BOS if available
        prompt_ids = tokenizer.encode(prompt_text)
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None:
            prompt_ids = [bos_id] + prompt_ids
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        
        # Generate sample
        out = model.generate(prompt_tensor, max_new_tokens=max_tokens)
        sample = out[0].tolist()
        
        # Count unique tokens
        unique_tokens = len(set(sample))
        vocab_diversity = unique_tokens / max(tokenizer.vocab_size, 1)
    
    model.train()
    return vocab_diversity

def compute_fullseq_perplexity(model, full_sequences, cfg, device):
    """
    Compute perplexity on full sequences (not random windows).
    full_sequences: list of torch.Tensor, each a complete sequence.
    Returns: average perplexity across sequences.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for seq in full_sequences:
            if len(seq) < 2:
                continue
            # Move to device
            seq = seq.to(device)
            # For full-sequence loss, use the entire sequence as context/target
            # Reshape as a batch of 1
            x = seq[:-1].unsqueeze(0)  # (1, seq_len-1)
            y = seq[1:].unsqueeze(0)   # (1, seq_len-1)
            
            logits, loss = model(x, y)
            total_loss += loss.item()
            count += 1
    
    model.train()
    
    if count == 0:
        return float('inf')
    
    avg_loss = total_loss / count
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def plot_training_curves(history_df, save_path=None):
    """
    Plot training curves for loss, accuracy, and generation quality metrics.
    
    Args:
        history_df: pandas DataFrame with columns: step, train_loss, val_loss, 
                   perplexity, top1_acc, top5_acc, rep_ratio (optional), vocab_diversity (optional)
        save_path: Optional path to save the figure (e.g., 'training_curves.png')
    """
    # Determine layout based on available columns
    has_gen_metrics = 'rep_ratio' in history_df.columns and 'vocab_diversity' in history_df.columns
    num_plots = 4 if has_gen_metrics else 2
    cols = 2
    rows = (num_plots + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if num_plots == 2:
        axes = axes.flatten()  # Ensure axes is always 2D array
    
    # Plot 1: Loss and Perplexity curves
    ax1 = axes[0, 0] if has_gen_metrics else axes[0]
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
    ax2 = axes[0, 1] if has_gen_metrics else axes[1]
    ax2.plot(history_df['step'], history_df['top1_acc'] * 100, 'o-', label='Top-1 Accuracy', alpha=0.7)
    ax2.plot(history_df['step'], history_df['top5_acc'] * 100, 's-', label='Top-5 Accuracy', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Top-1 and Top-5 Accuracy')
    ax2.set_ylim(bottom=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if has_gen_metrics:
        # Plot 3: Self-repetition ratio (higher = more unique bigrams)
        ax3 = axes[1, 0]
        ax3.plot(history_df['step'], history_df['rep_ratio'], 'd-', label='Repetition Diversity', alpha=0.7, color='teal')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Unique bigram ratio')
        ax3.set_title('Self-Repetition Diversity')
        ax3.set_ylim(bottom=0)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Vocab diversity
        ax4 = axes[1, 1]
        ax4.plot(history_df['step'], history_df['vocab_diversity'], 'x-', label='Vocab Diversity', alpha=0.7, color='orange')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Unique tokens / Vocab')
        ax4.set_title('Vocab Diversity in Generation')
        ax4.set_ylim(bottom=0)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()