import torch
import torch.nn as nn
from models.cnn.trainer import train_epochs
from models.cnn.architectures.cnn_baseline import SimpleCNN

def test_train_loop_one_epoch(dummy_dataloader):
    """
    Integration test: Runs the training loop for 1 epoch on dummy data.
    Verifies that it completes without error and returns a history DataFrame.
    """
    device = torch.device("cpu")
    # Use 3 input channels to match the dummy dataloader (which produces 3x32x32)
    model = SimpleCNN(input_size=32, num_classes=10, input_channels=3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Run for 1 epoch
    history = train_epochs(
        model=model,
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader, # reuse for speed
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        num_epochs=1,
        mixup_alpha=None,
        ema_model=None
    )
    
    assert len(history) == 1
    assert "train_loss" in history.columns
    assert "val_loss" in history.columns
    # Ensure loss is not NaN
    assert not history["train_loss"].isna().any()
