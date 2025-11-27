# Scripts

This directory contains the executable entry points for the project. All scripts are designed to be run from the project root to ensure package imports work correctly.

## Available Scripts

### `train_cnn.py`
The main training script for Convolutional Neural Networks.
*   **Function**: Configures the model, data, and training parameters, builds the DataLoaders, and launches the training engine.
*   **Key Features**:
    *   Supports multiple datasets (CIFAR-10, TinyImageNet, Oxford Pets, etc.).
    *   Configurable via `DataConfig`, `ModelConfig`, and `TrainConfig` dataclasses within the script.
    *   Handles dependency injection of DataLoaders into the model engine.

### `run_mlp.py`
A demonstration script for Multi-Layer Perceptrons (MLP).
*   **Function**: Trains a simple MLP on the XOR problem.
*   **Features**:
    *   Compares a raw NumPy implementation against a PyTorch implementation.
    *   Demonstrates manual backpropagation (in the NumPy version).

### `run_perceptron.py`
A demonstration script for a single Perceptron.
*   **Function**: Trains a Perceptron on the Iris dataset.
*   **Features**:
    *   Visualizes decision boundaries.
    *   Compares NumPy and PyTorch implementations.

## Usage

Run these scripts from the root of the repository:

```bash
# Example
python scripts/train_cnn.py
```
