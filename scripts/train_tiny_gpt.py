# tiny_gpt_char.py
# Minimal, CPU-friendly decoder-only transformer for next-character prediction.
# Run: python tiny_gpt_char.py [--text_file my_corpus.txt] [--max_steps 1000] ...

import sys
from pathlib import Path
import os, time, random

import torch
import torch.nn.functional as F

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# change cwd to _ROOT if it's different - copes with running in interactive window
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

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

def show_predictions(model, tokenizer, data, cfg, device, num_samples=3):
    """Show model predictions vs. ground truth on random samples."""
    model.eval()
    with torch.no_grad():
        X, Y = get_batch(data, cfg.block_size, cfg.batch_size, device)
        logits, _ = model(X, Y)
        probs = F.softmax(logits, dim=-1)  # (B, T, V)
        
        print("\n=== Prediction Examples ===")
        for i in range(min(num_samples, X.size(0))):
            # Pick last position in sequence
            context = tokenizer.decode(X[i].tolist())
            true_next = tokenizer.decode([Y[i, -1].item()])
            
            # Get top-3 predictions
            top_probs, top_idx = probs[i, -1].topk(3)
            pred_chars = [tokenizer.decode([idx.item()]) for idx in top_idx]
            
            print(f"\nSample {i+1}:")
            print(f"  Context: ...{context[-40:]}")  # last 40 chars
            print(f"  True next: '{true_next}'")
            print(f"  Predictions:")
            for j, (char, prob) in enumerate(zip(pred_chars, top_probs)):
                marker = "✓" if char == true_next else " "
                print(f"    {marker} '{char}': {prob.item():.1%}")
    
    model.train()

def calculate_accuracy(model, data, cfg, device):
    """Calculate top-1 and top-5 accuracy on validation data."""
    model.eval()
    with torch.no_grad():
        X, Y = get_batch(data, cfg.block_size, cfg.batch_size, device)
        logits, _ = model(X, Y)
        
        # Top-1 accuracy
        preds = logits.argmax(dim=-1)
        top1_acc = (preds == Y).float().mean().item()
        
        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=-1).indices
        top5_acc = (top5_preds == Y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    model.train()
    return top1_acc, top5_acc

def main():


    # cfg = Config()  # default
    cfg = Config(
        block_size=128,
        n_embd=128,
        n_layer=4,
        n_head=4,
        batch_size=32,
        max_steps=2000
    )
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    cfg.device = device_str
    print(f"Using device: {device_str}")
    
    data_text_file = "tiny_shakespeare/tiny_shakespeare.txt"  
    # data_text_file = None  # use default text

    # Reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device_str == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    

    # Load text
    if data_text_file:
        with open("data/" + data_text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_TEXT

    print(f"Training on {len(text)} characters from: "
          f"{data_text_file if data_text_file else 'default text'}")

    # Tokenizer & data
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]


    # Ensure vocab size is set before building the model
    cfg.vocab_size = tok.vocab_size
    model = TinyGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"Vocab size: {tok.vocab_size}, Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training
    model.train()
    for step in range(1, cfg.max_steps + 1):
        t0 = time.time()
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # if step % cfg.eval_interval == 0 or step == 1:
        #     val_loss = estimate_loss(model, val_data, cfg)
        #     epoch_time = time.time() - t0
        #     print(f"\n step {step:4d}/{cfg.max_steps} | train loss {loss.item():.3f} | val loss {val_loss:.3f} | {epoch_time:.1f}s")

        if step % cfg.eval_interval == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, cfg, device=device)
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            top1, top5 = calculate_accuracy(model, val_data, cfg, device)
            step_time = time.time() - t0
            
            print(f"\nstep {step:4d}/{cfg.max_steps} | Step time {step_time:.1f}s")
            print(f"  train loss: {loss.item():.3f}")
            print(f"  val loss: {val_loss:.3f} | perplexity: {perplexity:.2f}")
            print(f"  accuracy: top-1={top1:.1%}, top-5={top5:.1%}")
            
        #     # Show predictions every few intervals
        #     if step % (cfg.eval_interval * 2) == 0:
        #         show_predictions(model, tok, val_data, cfg, num_samples=2)
                
        #         # Mini generation sample
        #         print("\n=== Mini Generation ===")
        #         start = "To "
        #         start_ids = torch.tensor([tok.encode(start)], dtype=torch.long).to(cfg.device)
        #         out = model.generate(start_ids, max_new_tokens=50)[0].tolist()
        #         print(f"{tok.decode(out)}")
                
            # Generation demo
            print("\n=== Generation ===")
            start_text = "Short "
            start_ids = torch.tensor([tok.encode(start_text)], dtype=torch.long).to(device)
            # Use eval_tokens for generation length (no gen_tokens in Config)
            out = model.generate(start_ids, max_new_tokens=cfg.eval_tokens)[0].tolist()
            print(tok.decode(out))

            print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
