import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from ..config import TrainConfig


def make_param_groups(model):
    decay_params = []
    no_decay_params = []
    bn_param_ids = set()

    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            for p in module.parameters():
                bn_param_ids.add(id(p))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in bn_param_ids or name.endswith('.bias'):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    print(f"Optimizer params: decay={len(decay_params)} no_decay={len(no_decay_params)} (bn_params={len(bn_param_ids)})")
    return decay_params, no_decay_params, bn_param_ids


def build_optimizer(model, train_cfg: TrainConfig):
    decay_params, no_decay_params, _ = make_param_groups(model)
    opt_name = train_cfg.optimizer.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            [
                {'params': decay_params, 'weight_decay': train_cfg.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=train_cfg.learning_rate,
            momentum=train_cfg.momentum,
            nesterov=train_cfg.nesterov,
        )
    else:
        raise ValueError(f"Unsupported optimizer '{train_cfg.optimizer}'.")
    return optimizer


def build_scheduler(optimizer, train_cfg: TrainConfig):
    if train_cfg.scheduler is None:
        return None
    name = train_cfg.scheduler.lower()
    if name in {"none", ""}:
        return None
    if name == "warmup_cosine":
        base_lr = optimizer.param_groups[0]['lr']

        def _lr_lambda(epoch_idx: int):
            warmup = train_cfg.warmup_epochs
            total = max(train_cfg.num_epochs, 1)
            if epoch_idx < warmup and warmup > 0:
                return float(epoch_idx + 1) / float(warmup)
            if total <= warmup:
                return train_cfg.min_lr / base_lr
            progress = (epoch_idx - warmup) / max(total - warmup, 1)
            progress = float(min(max(progress, 0.0), 1.0))
            cosine_term = 0.5 * (1.0 + np.cos(np.pi * progress))
            scaled_lr = train_cfg.min_lr / base_lr
            return max(cosine_term, scaled_lr)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    raise ValueError(f"Unknown scheduler '{train_cfg.scheduler}'.")


def build_ema(model, train_cfg: TrainConfig, device: torch.device):
    if not train_cfg.use_ema:
        return None
    ema_decay = train_cfg.ema_decay

    def _ema_avg_fn(averaged_param, model_param, num_averaged):
        if num_averaged == 0:
            return model_param
        return ema_decay * averaged_param + (1.0 - ema_decay) * model_param

    ema_model = AveragedModel(model, avg_fn=_ema_avg_fn)
    ema_model.to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model
