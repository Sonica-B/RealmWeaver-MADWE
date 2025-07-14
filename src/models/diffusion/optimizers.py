"""
Optimizers for MADWE diffusion models
"""

import torch
from torch.optim import AdamW, Adam
from typing import Optional, List, Dict, Any


def get_lora_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Get optimizer for LoRA training

    Args:
        model: Model with LoRA parameters
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        betas: Adam beta parameters
        eps: Adam epsilon
        optimizer_type: Type of optimizer (adamw, adam)

    Returns:
        Configured optimizer
    """
    # Get LoRA parameters only
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name.lower():
            lora_params.append(param)

    print(f"Found {len(lora_params)} LoRA parameters to optimize")

    # Create optimizer
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            lora_params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_type.lower() == "adam":
        optimizer = Adam(lora_params, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def get_param_groups(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float = 1e-2,
    no_decay_params: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get parameter groups with different weight decay settings

    Args:
        model: Model to get parameters from
        learning_rate: Base learning rate
        weight_decay: Weight decay for parameters
        no_decay_params: List of parameter name patterns to exclude from decay

    Returns:
        List of parameter groups
    """
    if no_decay_params is None:
        no_decay_params = ["bias", "norm", "ln_"]

    decay_params = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have weight decay
        should_decay = True
        for pattern in no_decay_params:
            if pattern in name.lower():
                should_decay = False
                break

        if should_decay:
            decay_params.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {"params": decay_params, "lr": learning_rate, "weight_decay": weight_decay},
        {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
    ]

    print(
        f"Parameter groups: {len(decay_params)} with decay, {len(no_decay)} without decay"
    )

    return param_groups


class CosineAnnealingWithWarmup:
    """Learning rate scheduler with warmup and cosine annealing"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        warmup_init_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.warmup_init_lr_ratio = warmup_init_lr_ratio

        # Store base learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr_scale = (
                self.warmup_init_lr_ratio
                + (1 - self.warmup_init_lr_ratio)
                * self.current_step
                / self.warmup_steps
            )
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )

        # Update learning rates
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * lr_scale

    def get_last_lr(self):
        """Get current learning rates"""
        return [group["lr"] for group in self.optimizer.param_groups]
