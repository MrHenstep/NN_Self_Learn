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
class ContrastiveConfig:
    """Configuration for contrastive learning (SimCLR-style).
    
    Attributes:
        temperature: Temperature parameter for NT-Xent loss (lower = sharper)
        projection_dim: Output dimension of projection head
        projection_hidden: Hidden dimension of projection head MLP
        use_bn_in_head: Whether to use BatchNorm in projection head
        pretrain_epochs: Number of epochs for contrastive pre-training
        pretrain_lr: Learning rate for pre-training
        pretrain_weight_decay: Weight decay for pre-training
        pretrain_batch_size: Batch size for pre-training (larger is better)
        warmup_epochs: Warmup epochs for pre-training
        finetune_epochs: Number of epochs for fine-tuning
        finetune_lr: Learning rate for fine-tuning
        finetune_weight_decay: Weight decay for fine-tuning
        linear_eval: If True, freeze backbone and only train classifier
        color_jitter_strength: Strength of color jitter augmentation
    """
    # Loss parameters
    temperature: float = 0.5
    
    # Projection head parameters
    projection_dim: int = 128
    projection_hidden: int = 512
    use_bn_in_head: bool = True
    
    # Pre-training parameters
    pretrain_epochs: int = 500
    pretrain_lr: float = 0.5  # SimCLR uses high LR with LARS
    pretrain_weight_decay: float = 1e-4
    pretrain_batch_size: int = 256
    warmup_epochs: int = 10
    
    # Fine-tuning parameters
    finetune_epochs: int = 100
    finetune_lr: float = 0.1
    finetune_weight_decay: float = 1e-3
    linear_eval: bool = False  # If True, freeze backbone
    
    # Augmentation parameters
    color_jitter_strength: float = 1.0

@dataclass
class DataMetadata:
    dataset_key: str
    num_classes: int
    input_channels: int
    input_size: int

# TODO: Future: add (de)serialization helpers for YAML/JSON presets.
