"""
Loss functions for contrastive and self-supervised learning.

This module provides:
- NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR
- InfoNCE loss variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(
    z1: torch.Tensor, 
    z2: torch.Tensor, 
    temperature: float = 0.5
) -> torch.Tensor:
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    
    Computes the contrastive loss for two batches of normalized embeddings.
    For each sample i, its positive pair is at position i in the other view.
    All other samples in the batch serve as negatives.
    
    Args:
        z1: First batch of L2-normalized embeddings, shape (N, D)
        z2: Second batch of L2-normalized embeddings, shape (N, D)
        temperature: Temperature parameter for scaling similarities
        
    Returns:
        Scalar loss tensor
        
    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning of 
        Visual Representations", ICML 2020
    """
    batch_size = z1.shape[0]
    device = z1.device
    
    # Concatenate both views: [z1_0, z1_1, ..., z1_N, z2_0, z2_1, ..., z2_N]
    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    
    # Compute pairwise cosine similarity matrix
    # Since z is already L2-normalized, dot product = cosine similarity
    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)
    
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))
    
    # For sample i in z1, its positive is at index i + batch_size (in z2)
    # For sample i in z2, its positive is at index i (in z1)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(batch_size, device=device)
    ])
    
    # Cross-entropy loss treats similarities as logits
    loss = F.cross_entropy(sim, labels)
    
    return loss


class NTXentLoss(nn.Module):
    """NT-Xent loss as a nn.Module.
    
    Args:
        temperature: Temperature parameter for scaling similarities
        
    Example:
        >>> criterion = NTXentLoss(temperature=0.5)
        >>> z1 = F.normalize(encoder(x1), dim=1)
        >>> z2 = F.normalize(encoder(x2), dim=1)
        >>> loss = criterion(z1, z2)
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return nt_xent_loss(z1, z2, self.temperature)


def info_nce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """InfoNCE loss with explicit negative samples.
    
    More general form where negatives are explicitly provided rather than
    using in-batch negatives.
    
    Args:
        query: Query embeddings of shape (N, D)
        positive: Positive key embeddings of shape (N, D)  
        negatives: Negative key embeddings of shape (N, K, D) or (K, D)
        temperature: Temperature parameter
        
    Returns:
        Scalar loss tensor
    """
    # Positive similarity: (N,)
    pos_sim = (query * positive).sum(dim=1) / temperature
    
    # Negative similarities
    if negatives.dim() == 2:
        # (K, D) -> broadcast to all queries
        neg_sim = torch.mm(query, negatives.T) / temperature  # (N, K)
    else:
        # (N, K, D) -> per-query negatives
        neg_sim = torch.bmm(negatives, query.unsqueeze(-1)).squeeze(-1) / temperature  # (N, K)
    
    # Log-sum-exp over positives and negatives
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (N, 1+K)
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    
    return F.cross_entropy(logits, labels)
