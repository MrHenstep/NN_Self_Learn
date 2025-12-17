# tiny_gpt_char.py
# Minimal, CPU-friendly decoder-only transformer for next-character prediction.
# Run: python tiny_gpt_char.py [--text_file my_corpus.txt] [--max_steps 1000] ...

import sys
from pathlib import Path
import os, time, random

import torch

import pandas as pd

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# change cwd to _ROOT if it's different - copes with running in interactive window
if os.getcwd() != _ROOT:
    os.chdir(_ROOT)

from models.transformers.tiny_gpt_utilities import (
    get_batch, estimate_loss, plot_training_curves,
    compute_self_repetition_ratio, compute_vocab_diversity
)
from models.transformers.tiny_gpt_config import Config
from models.transformers.tiny_gpt_tokenizer import CharTokenizer, ByteBPETokenizerWrapper, build_or_load_bytebpe
from models.transformers.tiny_gpt_model import TinyGPT
import math

def lr_lambda(step: int, warmup_steps: int, max_steps: int) -> float:
    """Learning rate schedule: linear warmup then cosine decay."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def calculate_accuracy(model, data, cfg, device, bos_id=None):
    """Calculate top-1 and top-5 accuracy on validation data."""
    model.eval()
    with torch.no_grad():
        X, Y = get_batch(data, cfg.block_size, cfg.batch_size, device, bos_id=bos_id)
        logits, _ = model(X, Y)
        
        # Top-1 accuracy
        preds = logits.argmax(dim=-1)
        top1_acc = (preds == Y).float().mean().item()
        
        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=-1).indices
        top5_acc = (top5_preds == Y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    
    model.train()
    return top1_acc, top5_acc


def load_text_file(file_path, file_name):
    with open(file_path + "/" + file_name, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def build_char_tokenizer(text):
    return CharTokenizer(text)

def build_bytebpe_tokenizer(file_path, file_name):
    # Tune BPE for tiny model capacity: smaller vocab, higher min freq
    bpe_vocab_size = 2048
    bpe_min_freq = 5
    bpe_cache_dir = "bpe_tok_2k"

    return build_or_load_bytebpe(
        text_file="data/" + file_name,
        vocab_size=bpe_vocab_size,
        min_frequency=bpe_min_freq,
        cache_dir=bpe_cache_dir,
    )


def get_sample_generation(tok, model, cfg, device, prompt_text="ROMEO:", max_tokens=100):
    
    # Prepend BOS to the prompt when available
    bos_id = getattr(tok, "bos_token_id", None)
    prompt_ids = tok.encode(prompt_text)
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids
    start_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    # Use eval_tokens for generation length (no gen_tokens in Config)
    out = model.generate(start_ids, max_new_tokens=max_tokens)[0].tolist()
    return tok.decode(out)
    # print("\n" + tok.decode(out) + "\n")

def save_checkpoint(
    model,
    tok,
    cfg,
    tokeniser_choice,
    folder,
    file_name,
):
    os.makedirs(folder or ".", exist_ok=True)
    save_path = os.path.join(folder, file_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(cfg),
            "tokeniser_choice": tokeniser_choice,
            "tokenizer_class": tok.__class__.__name__,
            "tokenizer_state": getattr(tok, "state_dict", lambda: None)(),
        },
        save_path,
    )
    print(f"Saved model checkpoint to {save_path}")

def load_checkpoint_for_generation(file_path, file_name, device=None):
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_full_path = os.path.join(file_path or ".", file_name)
    checkpoint = torch.load(checkpoint_full_path, map_location=device)

    cfg = Config(**checkpoint["config"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tok_state = checkpoint.get("tokenizer_state")
    tok_class = checkpoint.get("tokenizer_class", "CharTokenizer")

    if tok_class == "CharTokenizer":
        tok = CharTokenizer.from_state_dict(tok_state)
    elif tok_class == "ByteBPETokenizerWrapper":
        tok = ByteBPETokenizerWrapper.from_state_dict(tok_state)
    else:
        raise ValueError(f"Unsupported tokenizer class in checkpoint: {tok_class}")

    model.eval()
    return model, tok, cfg, device


def generate_from_checkpoint(file_path, file_name, prompt_text="Alas ", max_tokens=80, device=None):
    model, tok, cfg, device = load_checkpoint_for_generation(file_path, file_name, device)
    cfg.eval_tokens = max_tokens
    sample_gen_text = get_sample_generation(tok, model, cfg, device, prompt_text=prompt_text, max_tokens=max_tokens)
    print("\n" + sample_gen_text + "\n")

def setup_device_and_seed(seed: int = 1234):
    """Set device, print start info, and seed RNGs."""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Start time: {current_time}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    random.seed(seed)
    torch.manual_seed(seed)
    if device_str == "cuda":
        torch.cuda.manual_seed_all(seed)

    return device


def prepare_data(file_path: str, file_name: str, tokeniser_choice: str = "char"):
    """Load text, tokenize, prepend BOS if available, and split train/val."""
    text = load_text_file(file_path, file_name)

    print(
        f"Training on {len(text):,} characters from: "
        f"{file_name if file_name else 'default text'}"
    )

    if tokeniser_choice == "char":
        tok = build_char_tokenizer(text)
    else:
        tok = build_bytebpe_tokenizer(file_path, file_name)

    bos_id = getattr(tok, "bos_token_id", None)
    encoded = tok.encode(text)
    if bos_id is not None:
        encoded = [bos_id] + encoded

    data = torch.tensor(encoded, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    return tok, train_data, val_data, bos_id


def create_config():
    """Return training config plus labels for logging/saving."""
    cfg = Config(
      n_head=4, n_embd=128, n_layer=4, 
      block_size=128, batch_size=32, 
      max_steps=1000
    )

    # cfg = Config(
    #     n_head=6, n_embd=6*64, n_layer=6,
    #     block_size=256, batch_size=32,
    #     max_steps=50000,
    #     dropout=0.1, weight_decay=0.1,
    #     eval_interval=1000,
    # )

    model_label = "wiki3"
    generate_interval = 1000
    return cfg, model_label, generate_interval


def build_model_and_optim(cfg, device):
    """Construct model, optimizer, and LR scheduler."""
    model = TinyGPT(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    warmup_steps = max(1, int(0.1 * cfg.max_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup_steps, cfg.max_steps),
    )

    return model, optimizer, scheduler


def train_and_evaluate(
    model,
    optimizer,
    scheduler,
    train_data,
    val_data,
    tok,
    cfg,
    device,
    bos_id,
    generate_interval,
):
    """Run training loop with periodic eval, logging, and sample generation."""
    model.train()
    t0 = time.time()
    history = []

    for step in range(1, cfg.max_steps + 1):

        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device, bos_id=bos_id)

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.eval_interval == 0 or step == 1:

            val_loss = estimate_loss(model, val_data, cfg, device=device, bos_id=bos_id)

            perplexity = torch.exp(torch.tensor(val_loss)).item()

            top1, top5 = calculate_accuracy(model, val_data, cfg, device, bos_id=bos_id)

            rep_ratio = compute_self_repetition_ratio(model, tok, "ROMEO:", device, num_samples=2, max_tokens=80)

            vocab_div = compute_vocab_diversity(model, tok, "ROMEO:", device, max_tokens=80)

            history.append({'step': step, 'train_loss': loss.item(), 'val_loss': val_loss, 'perplexity': perplexity, 'top1_acc': top1, 'top5_acc': top5, 'rep_ratio': rep_ratio, 'vocab_diversity': vocab_div, })

            print(
                f"Step {step:4d}/{cfg.max_steps} | tr loss {loss.item():.3f} | "
                f"val loss {val_loss:.3f} | perp {perplexity:.2f} | "
                f"top-1 {top1:.1%} | top-5 {top5:.1%} | "
                f"rep {rep_ratio:.3f} | vocab {vocab_div:.3f}"
            )

        if step == cfg.max_steps or step == 1 or step % (generate_interval) == 0:
            sample_gen_text = get_sample_generation(tok, model, cfg, device, prompt_text="Alas ", max_tokens=cfg.eval_tokens)
            print("\n" + sample_gen_text + "\n")

    step_time = (time.time() - t0) / cfg.max_steps
    print(f"Average step time: {step_time:.3f}s")

    history_df = pd.DataFrame(history)
    plot_training_curves(history_df, save_path=None)

    return history_df


def main():

    device = setup_device_and_seed()

    file_path = "data"
    file_name = "tiny_shakespeare/tiny_shakespeare.txt"  
    # file_name = "gut_shakespeare/pg100.txt"
    # file_name = "wikitext103/wikitext103_train.txt"

    tokeniser_choice = "char"  
    # tokeniser_choice = "bytebpe"  

    tok, train_data, val_data, bos_id = prepare_data(file_path, file_name, tokeniser_choice)

    cfg, model_label, generate_interval = create_config()

    print("Using config:", cfg)

    cfg.vocab_size = tok.vocab_size

    model, optimizer, scheduler = build_model_and_optim(cfg, device)

    print(f"Vocab size: {tok.vocab_size}, Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history_df = train_and_evaluate(
        model,
        optimizer,
        scheduler,
        train_data,
        val_data,
        tok,
        cfg,
        device,
        bos_id,
        generate_interval,
    )

    return model, tok, tokeniser_choice, cfg, device, history_df, model_label


if __name__ == "__main__":
    
    model, tok, tokeniser_choice, cfg, device, history_df, model_label = main()
    
    base = f"tiny_gpt_checkpoint_{model_label}"
    save_checkpoint(model, tok, cfg, tokeniser_choice, folder="checkpoints", file_name=base + ".pth")
    history_df.to_csv(os.path.join("checkpoints", base + "_history.csv"), index=False)


