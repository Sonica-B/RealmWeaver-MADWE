#!/usr/bin/env python3
"""
Launch script for LoRA training
"""
import yaml
import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.diffusion import LoRATrainer


def load_config(config_path: str, biome: str) -> dict:
    """Load and merge configuration"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create base config
    train_config = {
        "biome": biome,
        "model_id": config["model"]["base_model_id"],
        "mixed_precision": config["model"]["mixed_precision"],
        # LoRA settings
        "lora_rank": config["lora"]["rank"],
        "lora_alpha": config["lora"]["alpha"],
        "lora_dropout": config["lora"]["dropout"],
        # Training settings
        "batch_size": config["training"]["batch_size"],
        "gradient_accumulation_steps": config["training"][
            "gradient_accumulation_steps"
        ],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "num_epochs": config["training"]["num_epochs"],
        "num_train_steps": config["training"]["num_train_steps"],
        "warmup_steps": config["training"]["warmup_steps"],
        "max_grad_norm": config["training"]["max_grad_norm"],
        # Loss settings
        "use_style_loss": config["loss"]["use_style_loss"],
        "style_loss_weight": config["loss"]["style_loss_weight"],
        # Data settings
        "resolution": config["data"]["resolution"],
        "num_workers": config["data"]["num_workers"],
        # Logging
        "log_every": config["logging"]["log_every"],
        "val_every": config["logging"]["val_every"],
        "save_every": config["logging"]["save_every"],
    }

    # Apply biome-specific overrides
    if biome in config.get("biomes", {}):
        biome_config = config["biomes"][biome]
        train_config.update(biome_config)

    return train_config


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for MADWE")
    parser.add_argument("--biome", type=str, required=True, help="Biome to train")
    parser.add_argument(
        "--config", type=str, default="configs/training/lora_config.yaml"
    )
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="data/models/lora")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.biome)
    config["data_dir"] = args.data_dir
    config["output_dir"] = f"{args.output_dir}/lora_{args.biome}"

    if args.resume:
        config["resume_from"] = args.resume

    print(f"Training LoRA for {args.biome} biome")
    print(f"Configuration: {args.config}")
    print(f"Output: {config['output_dir']}")

    # Initialize and train
    trainer = LoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
