"""
SimCLR architecture for contrastive learning.

This module provides SimCLR-style contrastive learning components:
- SimCLRModel: Wraps a backbone (e.g., ResNet) with a projection head
- ProjectionHead: MLP projection head for mapping features to contrastive space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.
    
    Maps backbone features to a lower-dimensional space where
    contrastive loss is computed. Uses a 2-layer MLP with BatchNorm
    and ReLU activation.
    
    Args:
        input_dim: Dimensionality of input features from backbone
        hidden_dim: Dimensionality of hidden layer
        output_dim: Dimensionality of output projections
        use_bn: Whether to use BatchNorm in the projection head
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 512, 
        output_dim: int = 128,
        use_bn: bool = True
    ):
        super().__init__()
        
        if use_bn:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class SimCLRModel(nn.Module):
    """SimCLR-style contrastive learning model.
    
    Wraps a backbone network (e.g., ResNetCF) with a projection head
    for contrastive pre-training. The projection head maps backbone
    features to a normalized embedding space.
    
    Args:
        backbone: Feature extractor network (must have forward_features method)
        projection_dim: Output dimension of projection head
        projection_hidden: Hidden dimension of projection head MLP
        use_bn_in_head: Whether to use BatchNorm in projection head
        
    Example:
        >>> backbone = ResNetCF(n_classes=10, resnet_n=3)
        >>> model = SimCLRModel(backbone, projection_dim=128)
        >>> x1, x2 = augment(images), augment(images)
        >>> z1, z2 = model(x1), model(x2)
        >>> loss = nt_xent_loss(z1, z2)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        projection_dim: int = 128,
        projection_hidden: int = 512,
        use_bn_in_head: bool = True,
    ):
        super().__init__()
        
        self.backbone = backbone
        
        # Get feature dimension from backbone
        if hasattr(backbone, 'feature_dim'):
            feature_dim = backbone.feature_dim
        else:
            # Fallback: assume 64 for CIFAR ResNet
            feature_dim = 64
        
        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=projection_hidden,
            output_dim=projection_dim,
            use_bn=use_bn_in_head,
        )
        
        self.projection_dim = projection_dim
        self.feature_dim = feature_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning normalized projections.
        
        Args:
            x: Input images of shape (N, C, H, W)
            return_features: If True, also return backbone features
            
        Returns:
            If return_features is False: 
                L2-normalized projections of shape (N, projection_dim)
            If return_features is True:
                Tuple of (features, projections)
        """
        features = self.backbone.forward_features(x)
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        
        if return_features:
            return features, projections
        return projections
    
    def get_backbone(self) -> nn.Module:
        """Return the backbone for fine-tuning or evaluation."""
        return self.backbone


def build_simclr_model(
    backbone: nn.Module,
    projection_dim: int = 128,
    projection_hidden: int = 512,
    use_bn_in_head: bool = True,
) -> SimCLRModel:
    """Factory function to build a SimCLR model from a backbone.
    
    Args:
        backbone: Pre-built backbone network (e.g., from build_model)
        projection_dim: Output dimension of projection head
        projection_hidden: Hidden dimension of projection head MLP
        use_bn_in_head: Whether to use BatchNorm in projection head
        
    Returns:
        SimCLRModel wrapping the backbone with a projection head
    """
    return SimCLRModel(
        backbone=backbone,
        projection_dim=projection_dim,
        projection_hidden=projection_hidden,
        use_bn_in_head=use_bn_in_head,
    )
