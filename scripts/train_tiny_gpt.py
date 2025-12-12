# tiny_gpt_char.py
# Minimal, CPU-friendly decoder-only transformer for next-character prediction.
# Run: python tiny_gpt_char.py [--text_file my_corpus.txt] [--max_steps 1000] ...

import sys
from pathlib import Path
import os, time, random

import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# change cwd to _ROOT if it's different - copes with running in interactive window
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

from models.transformers.tiny_gpt_utilities import get_batch, estimate_loss, plot_training_curves
from models.transformers.tiny_gpt_config import Config
from models.transformers.tiny_gpt_tokenizer import CharTokenizer
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

    # default
    # cfg = Config()  

    # original minimal config
    cfg = Config(
        block_size=128,
        n_embd=128,
        n_layer=4,
        n_head=4,
        batch_size=32,
        max_steps=2000
    )

    # larger config for use on GPUs
    # cfg = Config(
    #     block_size=256,
    #     n_embd=256,
    #     n_layer=8,
    #     n_head=8,
    #     batch_size=64,
    #     max_steps=20000,
    #     # eval_interval=500,
    # )


    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    cfg.device = device_str
    print(f"Using device: {device_str}")
    
    # Reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device_str == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Load data -------------------------------------------------------------
    
    data_text_file = "tiny_shakespeare/tiny_shakespeare.txt"  
    # data_text_file = None  # use default text

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

    # Build model -----------------------------------------------------------
    
    model = TinyGPT(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps)

    print(f"Vocab size: {tok.vocab_size}, Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training
    model.train()
    t0 = time.time()
    history = []  # Collect metrics
    
    for step in range(1, cfg.max_steps + 1):
        
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.eval_interval == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, cfg, device=device)
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            top1, top5 = calculate_accuracy(model, val_data, cfg, device)
            
            # Log metrics to history
            history.append({
                'step': step,
                'train_loss': loss.item(),
                'val_loss': val_loss,
                'perplexity': perplexity,
                'top1_acc': top1,
                'top5_acc': top5,
            })
            
            print(f"Step {step:4d}/{cfg.max_steps} | tr loss {loss.item():.3f} | val loss {val_loss:.3f} | perp {perplexity:.2f} | top-1 {top1:.1%} | top-5 {top5:.1%}")

    step_time = (time.time() - t0) / cfg.max_steps
    print(f"Average step time: {step_time:.3f}s")
    
    # Convert history to DataFrame and save
    history_df = pd.DataFrame(history)
    # csv_path = "training_history.csv"
    # history_df.to_csv(csv_path, index=False)
    # print(f"\nTraining history saved to {csv_path}")
    # print(f"\nTraining History Summary:")
    # print(history_df.to_string())
    
    # Plot training curves
    plot_path = "training_curves.png"
    plot_training_curves(history_df, save_path=plot_path)

    
    print("\n=== Generation ===\n")
    start_text = "Alas "
    start_ids = torch.tensor([tok.encode(start_text)], dtype=torch.long).to(device)
    # Use eval_tokens for generation length (no gen_tokens in Config)
    out = model.generate(start_ids, max_new_tokens=cfg.eval_tokens)[0].tolist()
    print(tok.decode(out))



if __name__ == "__main__":
    main()
