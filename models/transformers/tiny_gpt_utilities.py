import torch
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