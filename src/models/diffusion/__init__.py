"""
Diffusion models module for MADWE
"""

from .lora_trainer import LoRATrainer
from .inference import LoRAInference, MultiLoRAInference, quick_generate
from .optimizers import get_lora_optimizer, get_param_groups, CosineAnnealingWithWarmup

# Backwards compatibility aliases
DiffusionInference = LoRAInference
FastInference = MultiLoRAInference

__all__ = [
    "LoRATrainer",
    "LoRAInference",
    "MultiLoRAInference",
    "quick_generate",
    "get_lora_optimizer",
    "get_param_groups",
    "CosineAnnealingWithWarmup",
    "DiffusionInference",  # Alias
    "FastInference",  # Alias
]
