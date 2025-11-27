# NN_Self_Learn

This repository contains implementations of various neural network architectures, ranging from basic Perceptrons to modern Convolutional Neural Networks (ResNets), implemented in both raw NumPy (for learning purposes) and PyTorch.

## Project Structure

*   **`scripts/`**: The entry points for the project. Contains executable scripts to train and test models.
*   **`models/`**: The core library containing model architectures, training engines, and configuration definitions.
    *   `cnn/`: Convolutional Neural Network implementations (ResNet, VGG-style).
    *   `perceptron/`: Basic Perceptron and MLP implementations.
*   **`data_loading/`**: A standalone package for handling datasets, data augmentation, and DataLoader creation.
*   **`data/`**: Directory where datasets are downloaded and stored (typically ignored by version control).

## Getting Started

The project is designed to be run from the root directory using the scripts in the `scripts/` folder.

### Prerequisites
*   Python 3.x
*   PyTorch
*   Torchvision
*   NumPy
*   Matplotlib
*   Pandas
*   Scikit-learn

### Running a Training Job

To train a CNN (e.g., ResNet on TinyImageNet):

```bash
python scripts/train_cnn.py
```

To run the simple Perceptron or MLP examples:

```bash
python scripts/run_perceptron.py
python scripts/run_mlp.py
```
