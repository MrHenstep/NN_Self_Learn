import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformers.tiny_gpt_config import Config

from models.transformers.tiny_gpt_transformerblocks import Block


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

        # idx starts (B, T)
        tok = self.token_emb(idx)                          # (B, T, C) - maps token to embedding size
        pos = self.pos_emb[:, :T, :]                       # (1, T, C)
        x = tok + pos                                      # (B, T, C) - broadcast addition
        for blk in self.blocks:
            x = blk(x)                                     # (B, T, C)  
        x = self.ln_f(x)                                   # (B, T, C)
        logits = self.head(x)                              # (B, T, V) - maps embed to vocab size

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))        # calculates the loss by taking the probability of getting the correct next token
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        self.eval()
        for _ in range(max_new_tokens):               # generate max_new_tokens tokens one by one
            idx_cond = idx[:, -self.cfg.block_size:]  # crop if more tokens than block_size requested
            logits, _ = self(idx_cond)                # model gives us logits for next token for all tokens in the sequence
            next_logits = logits[:, -1, :]            # but we only want the last token's logits
            probs = F.softmax(next_logits, dim=-1)    # convert logits to probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # sample from the distribution
            idx = torch.cat([idx, next_id], dim=1) # append sampled token to the sequence and loop to the next new token to be generated
        return idx
