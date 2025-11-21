import numpy as np
import torch


def mixup_batch(inputs: torch.Tensor, targets: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, float(lam)
