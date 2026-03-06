# NN_Self_Learn

A methodical walk through neural network architectures — from perceptrons to GPT — built and trained from scratch to develop a ground-up understanding of how these models work.

All models were trained on a Linux machine with an NVIDIA GTX 3090.

## What's Here

The project follows a deliberate progression through increasingly complex architectures, each applied to a concrete task:

**Perceptrons and MLPs** — starting from the basics, implemented in both raw NumPy (for learning) and PyTorch.

**CNNs** — convolutional networks for image classification on CIFAR, including a baseline CNN, an improved variant, and VGG-style architectures. Includes ablation studies (see `CNN_ablation.xlsx`).

**ResNets** — deeper residual networks tackling the same classification problems, exploring how skip connections affect training dynamics.

**Contrastive Learning (SimCLR)** — self-supervised representation learning with a contrastive objective, including custom dataset handling and augmentation pipelines.

**PINNs** — a brief exploration of physics-informed neural networks.

**Transformers / GPT** — a from-scratch GPT implementation with BPE tokenisation (2k vocabulary), trained on Shakespeare and Wikipedia extracts. Includes configurable model size, custom transformer blocks, and text generation. The model generates text that reads convincingly like language, even if the content leaves something to be desired.

## Project Structure

```
├── models/
│   ├── perceptron/          # Perceptron and MLP
│   ├── cnn/
│   │   ├── architectures/   # Baseline CNN, improved CNN, ResNet, SimCLR
│   │   └── utils/           # Visualisation, optimisation, probing
│   ├── pinn/                # Physics-informed neural network
│   └── transformers/        # GPT model, transformer blocks, tokeniser, config
├── scripts/                 # Training and evaluation entry points
│   ├── train_cnn.py
│   ├── train_contrastive.py
│   ├── train_pinn.py
│   ├── train_tiny_gpt.py
│   ├── run_perceptron.py
│   ├── run_mlp.py
│   └── visualize_embeddings.py
├── data_loading/            # Dataset handling, augmentation, contrastive transforms
├── bpe_tok_2k/              # BPE tokeniser vocabulary and merge rules
└── tests/                   # Unit and integration tests
```

## Running

```bash
# Perceptron / MLP
python scripts/run_perceptron.py
python scripts/run_mlp.py

# CNNs / ResNets
python scripts/train_cnn.py

# Contrastive learning (SimCLR)
python scripts/train_contrastive.py

# Physics-informed neural network
python scripts/train_pinn.py

# GPT training and generation
python scripts/train_tiny_gpt.py
```

## Prerequisites

Python 3.x, PyTorch, Torchvision, NumPy, Matplotlib, Pandas, scikit-learn. See `environment.yml`.
