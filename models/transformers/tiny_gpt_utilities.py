import torch
from models.transformers.tiny_gpt_config import Config

def get_batch(data, block_size, batch_size, device):
    # Ensure dataset is long enough to form at least one (x, y) pair
    if len(data) < 2:
        raise ValueError(f"Dataset too short: len(data)={len(data)}; need at least 2 tokens.")

    # Use the largest feasible window, capped by block_size
    T = min(block_size, len(data) - 1)

    # Valid start indices i in [0, len(data) - T - 1]; torch.randint high is exclusive
    high = len(data) - T
    if high <= 0:
        # This should not happen due to T=min(block_size, len(data)-1), but guard anyway
        raise ValueError(f"Invalid batch window: len(data)={len(data)}, T={T}, high={high}")

    ix = torch.randint(low=0, high=high, size=(batch_size,))
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x.to(device), y.to(device)

def estimate_loss(model, data, cfg: Config):
    model.eval()
    with torch.no_grad():
        X, Y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device)
        logits, loss = model(X, Y)
    model.train()
    return loss.item()