from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    dataset_key: str
    use_augment: Optional[bool] = None
    root: Optional[str] = None
    batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 4

@dataclass
class ModelConfig:
    model_name: str = "resnet"
    resnet_n: int = 3
    use_projection: Optional[bool] = None
    use_residual: bool = True
    flexi_arch: str = "VGG"

@dataclass
class TrainConfig:
    num_epochs: int = 200
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.05
    optimizer: str = "sgd"
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-3
    nesterov: bool = False
    scheduler: Optional[str] = "warmup_cosine"
    warmup_epochs: int = 7
    min_lr: float = 1e-3
    use_ema: bool = True
    ema_decay: float = 0.99
    plot_curves: bool = True
    test_after_training: bool = True

@dataclass
class DataMetadata:
    dataset_key: str
    num_classes: int
    input_channels: int
    input_size: int

# TODO: Future: add (de)serialization helpers for YAML/JSON presets.
