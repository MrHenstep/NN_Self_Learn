# tiny_gpt_char.py
# Minimal, CPU-friendly decoder-only transformer for next-character prediction.
# Run: python tiny_gpt_char.py [--text_file my_corpus.txt] [--max_steps 1000] ...

import sys
from pathlib import Path
import os, time, random

import torch

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.transformers.tiny_gpt_utilities import get_batch, estimate_loss
from models.transformers.tiny_gpt_config import Config
from models.transformers.tiny_gpt_tokenizer import CharTokenizer

from models.transformers.tiny_gpt_utilities import get_batch, estimate_loss
from models.transformers.tiny_gpt_transformerblocks import Block
from models.transformers.tiny_gpt_model import TinyGPT

DEFAULT_TEXT = (
    "To build understanding, we train a tiny transformer on this small text. "
    "It learns next-character probabilities and can generate similar text. "
    "Short sequences, few layers, and small embeddings keep computation light. "
    "You can also supply your own text file for training."
)

def main():


    cfg = Config()  # default
    text_file = None

    # Reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load text
    if text_file and os.path.isfile(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_TEXT

    # Tokenizer & data
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]


    # Ensure vocab size is set before building the model
    cfg.vocab_size = tok.vocab_size
    model = TinyGPT(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"Vocab size: {tok.vocab_size}, Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training
    model.train()
    t0 = time.time()
    for step in range(1, cfg.max_steps + 1):
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, cfg.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, cfg)
            elapsed = time.time() - t0
            print(f"step {step:4d}/{cfg.max_steps} | train loss {loss.item():.3f} | val loss {val_loss:.3f} | {elapsed:.1f}s")

    # Generation demo
    print("\n=== Generation ===")
    start_text = "To "
    start_ids = torch.tensor([tok.encode(start_text)], dtype=torch.long).to(cfg.device)
    # Use eval_tokens for generation length (no gen_tokens in Config)
    out = model.generate(start_ids, max_new_tokens=cfg.eval_tokens)[0].tolist()
    print(tok.decode(out))

if __name__ == "__main__":
    main()
