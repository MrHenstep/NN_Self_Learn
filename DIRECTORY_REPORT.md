# NN_Self_Learn - Directory Report

This document provides a comprehensive overview of the project's directory structure with summaries for each directory.

---

## Root Level

### `/` (Root Directory)
**Summary:** Main project root containing configuration files, documentation, and top-level package directories.

**Key Files:**
- `README.md` - Primary project documentation
- `environment.yml` - Conda environment specification
- `pytest.ini` - Test configuration
- `.gitignore` - Git ignore rules
- `transformer.xlsx` - Transformer-related reference data

---

## Main Directories

### `/scripts/`
**Summary:** Executable entry points for training and running models. All scripts should be executed from the project root.

**Contents:**
- `train_cnn.py` - Main CNN training script with configurable architectures and datasets
- `run_mlp.py` - MLP demonstration on XOR problem (NumPy vs PyTorch)
- `run_perceptron.py` - Single Perceptron training on Iris dataset with visualization
- `README.md` - Script documentation and usage instructions

**Purpose:** Provides command-line interfaces for model training and experimentation.

---

### `/models/`
**Summary:** Core library containing neural network architectures, training engines, and configuration definitions. Designed to be decoupled from data loading implementations.

**Contents:**
- `cnn/` - Convolutional Neural Network implementations
- `perceptron/` - Basic Perceptron and MLP implementations
- `README.md` - Models package documentation

**Purpose:** Houses all model architectures and training logic.

---

### `/models/cnn/`
**Summary:** Complete CNN implementation with multiple architectures, training engine, and utilities.

**Contents:**
- `architectures/` - Model architecture definitions
- `utils/` - Optimization and visualization utilities
- `engine.py` - High-level training orchestrator
- `trainer.py` - Training and evaluation loops
- `config.py` - Configuration dataclasses (DataConfig, ModelConfig, TrainConfig)

**Purpose:** Provides a modular system for CNN experimentation with various architectures.

---

### `/models/cnn/architectures/`
**Summary:** Definitions of various CNN model architectures from baseline to advanced networks.

**Contents:**
- `resnet.py` - ResNet implementation (custom and ImageNet-style)
- `cnn_baseline.py` - Simple baseline CNN architecture
- `cnn_improved.py` - Enhanced CNN architecture for experimentation
- `factory.py` - Model instantiation helper based on configuration

**Purpose:** Provides multiple CNN architectures for different use cases and datasets.

---

### `/models/cnn/utils/`
**Summary:** Utility modules for optimization, visualization, and model analysis.

**Contents:**
- `optimization.py` - Schedulers and EMA implementations
- `visualization.py` - Training metrics visualization
- `vis_model.py` - Model architecture visualization
- `probe.py` - Model probing and analysis utilities

**Purpose:** Supporting tools for training, debugging, and understanding CNN models.

---

### `/models/perceptron/`
**Summary:** Educational implementations of basic neural network building blocks.

**Contents:**
- `perceptron.py` - Single Perceptron unit implementation
- `MLP.py` - Multi-Layer Perceptron with NumPy (manual backprop) and PyTorch versions

**Purpose:** Demonstrates fundamental neural network concepts with both raw NumPy and PyTorch.

---

### `/data_loading/`
**Summary:** Standalone package for dataset preparation, augmentation, and DataLoader creation.

**Contents:**
- `loaders.py` - Main DataLoader builder entry point
- `datasets.py` - High-level dataset management and splitting
- `config.py` - Dataset registry mapping names to configurations
- `transforms.py` - Data augmentation pipelines for various datasets
- `custom_datasets.py` - Custom Dataset implementations (e.g., TinyImageNet)
- `augmentation.py` - Additional augmentation utilities
- `README.md` - Data loading documentation

**Supported Datasets:** MNIST, FashionMNIST, CIFAR-10, Oxford-IIIT Pets, Tiny ImageNet, ImageNet

**Purpose:** Provides a unified interface for loading and preprocessing diverse datasets.

---

### `/tests/`
**Summary:** Test suite for validating models, data loading, and integration.

**Contents:**
- `test_models.py` - Model architecture tests
- `test_data.py` - Data loading and preprocessing tests
- `test_integration.py` - End-to-end integration tests
- `test_smoke.py` - Quick smoke tests for basic functionality
- `conftest.py` - Pytest configuration and fixtures

**Purpose:** Ensures code quality and correctness through automated testing.

---

## Project Architecture Summary

The project follows a clean separation of concerns:

1. **Scripts Layer** (`/scripts/`) - User-facing entry points
2. **Model Layer** (`/models/`) - Architecture definitions and training logic
3. **Data Layer** (`/data_loading/`) - Dataset handling and preprocessing
4. **Test Layer** (`/tests/`) - Validation and quality assurance

This modular structure allows for:
- Easy experimentation with different architectures and datasets
- Clear dependency injection patterns
- Reusable components across different experiments
- Educational progression from basic (Perceptron) to advanced (ResNet) implementations
