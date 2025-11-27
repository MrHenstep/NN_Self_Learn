import sys
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.cnn.config import DataMetadata

# Add project root to sys.path so tests can import 'models' and 'data_loading'
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

@pytest.fixture
def dummy_data_meta():
    """Returns metadata for a dummy 10-class dataset with 3x32x32 images."""
    return DataMetadata(
        dataset_key="dummy",
        num_classes=10,
        input_channels=3,
        input_size=32
    )

@pytest.fixture
def dummy_dataloader():
    """Returns a DataLoader with 10 random samples (batch_size=2)."""
    # Create 10 samples of 3x32x32 images and labels
    X = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=2)
