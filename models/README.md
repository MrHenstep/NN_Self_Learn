# Models

This package contains the neural network architectures and the logic required to train them. It is designed to be decoupled from specific data loading implementations.

## Subpackages

### `cnn/`
Contains the logic for Convolutional Neural Networks.

*   **`architectures/`**: Defines the model classes.
    *   `resnet.py`: Implementation of ResNet (Custom and ImageNet-style).
    *   `cnn_baseline.py` / `cnn_improved.py`: Simpler CNN architectures for experimentation.
    *   `factory.py`: Helper to instantiate models based on configuration.
*   **`engine.py`**: The high-level orchestrator. Contains `run_training`, which accepts a model configuration and data loaders, then manages the full training lifecycle.
*   **`trainer.py`**: Contains the inner loops for training (`train_epochs`) and evaluation (`test_model`).
*   **`config.py`**: Defines `DataConfig`, `ModelConfig`, and `TrainConfig` dataclasses used to configure experiments.
*   **`utils/`**: Optimization (schedulers, EMA) and visualization utilities.

### `perceptron/`
Contains basic neural network implementations, often used for educational purposes or simple baselines.

*   **`MLP.py`**: Implementation of a Multi-Layer Perceptron. Includes a version using only NumPy (with manual backprop) and a PyTorch version.
*   **`perceptron.py`**: Implementation of a single Perceptron unit.
