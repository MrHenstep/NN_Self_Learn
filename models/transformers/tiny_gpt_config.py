
from dataclasses import dataclass

# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class Config:
    # Model architecture
    vocab_size: int = 0              # Number of distinct tokens in the vocabulary
    n_layer: int = 2                 # Number of transformer blocks
    n_head: int = 2                  # Number of attention heads per block
    n_embd: int = 64                 # Embedding dimension (token + positional)
    block_size: int = 64             # Maximum sequence length model can process
    dropout: float = 0.0             # Dropout probability in attention/MLP

    # Training hyperparameters
    batch_size: int = 32             # Number of sequences per training step
    max_steps: int = 800             # Total number of optimisation steps
    lr: float = 3e-3                 # Learning rate for AdamW optimiser
    eval_interval: int = 100         # Evaluate validation loss every N steps
    eval_tokens: int = 200           # Number of tokens used for quick eval (unused in minimal code)

    # Reproducibility / device
    seed: int = 1234                 # Random seed for reproducibility
    device: str = "cpu"              # Device to train on ("cpu" for this tiny model)