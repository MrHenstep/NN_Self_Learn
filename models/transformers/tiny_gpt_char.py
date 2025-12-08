# tiny_gpt_char.py
# Minimal, CPU-friendly decoder-only transformer for next-character prediction.
# Run: python tiny_gpt_char.py [--text_file my_corpus.txt] [--max_steps 1000] ...

import math, argparse, os, time, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class Config:
    # Model sizes kept small for CPU
    vocab_size: int = 0              # set after building tokenizer
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 64             # max sequence length
    dropout: float = 0.0

    # Training
    batch_size: int = 32
    max_steps: int = 800
    lr: float = 3e-3
    eval_interval: int = 100
    eval_tokens: int = 200           # for quick perplexity estimate
    seed: int = 1234
    device: str = "cpu"

# -----------------------------
# 2) Tiny character tokenizer
# -----------------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

# -----------------------------
# 3) Dataset utilities
# -----------------------------
def get_batch(data, block_size, batch_size, device):
    # pick random starting positions
    ix = torch.randint(low=0, high=len(data) - block_size - 1, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def estimate_loss(model, data, cfg: Config):
    model.eval()
    with torch.no_grad():
        X, Y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device)
        logits, loss = model(X, Y)
    model.train()
    return loss.item()

# -----------------------------
# 4) Core transformer blocks
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

        # Causal mask (registered as buffer so it moves with device, not trained)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                             .view(1, 1, cfg.block_size, cfg.block_size))

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * self.scale                              # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v                                                                # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)                          # (B, T, C)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual 1
        x = x + self.mlp(self.ln2(x))   # residual 2
        return x

# -----------------------------
# 5) Tiny GPT model
# -----------------------------
class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "Sequence too long for block_size"

        tok = self.token_emb(idx)                          # (B, T, C)
        pos = self.pos_emb[:, :T, :]                       # (1, T, C)
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                              # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]  # crop if needed
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]           # last time step
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# -----------------------------
# 6) Main: training & demo
# -----------------------------
DEFAULT_TEXT = (
    "To build understanding, we train a tiny transformer on this small text. "
    "It learns next-character probabilities and can generate similar text. "
    "Short sequences, few layers, and small embeddings keep computation light. "
    "You can also supply your own text file for training."
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, default=None,
                        help="Path to a .txt file to train on. If omitted, uses a tiny built-in corpus.")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gen_tokens", type=int, default=200)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load text
    if args.text_file and os.path.isfile(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_TEXT

    # Tokenizer & data
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # Config
    cfg = Config(
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        device="cpu",
    )

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
    out = model.generate(start_ids, max_new_tokens=args.gen_tokens)[0].tolist()
    print(tok.decode(out))

if __name__ == "__main__":
    main()
