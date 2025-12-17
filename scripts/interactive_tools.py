
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

from models.transformers.tiny_gpt_utilities import (
    get_batch, estimate_loss, plot_training_curves,
    compute_self_repetition_ratio, compute_vocab_diversity, compute_fullseq_perplexity
)
from models.transformers.tiny_gpt_config import Config
from models.transformers.tiny_gpt_tokenizer import CharTokenizer, build_or_load_bytebpe
from models.transformers.tiny_gpt_transformerblocks import Block
from models.transformers.tiny_gpt_model import TinyGPT
import math

from scripts.train_tiny_gpt import get_sample_generation, build_bytebpe_tokenizer, load_checkpoint_for_generation, generate_from_checkpoint

# model, tok, cfg, device = load_checkpoint_for_generation("checkpoints", "tiny_gpt_checkpoint.pth")

#show_sample_generation(tok, model, cfg, device, prompt_text="Sojourn ", max_tokens=cfg.eval_tokens)