#!/usr/bin/env python3
"""
LoRA Training for MADWE Project
Trains biome-specific LoRA adapters for game asset generation
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np

from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.utils.dataloader import DynamicGameAssetDataset
from src.models.diffusion.optimizers import get_lora_optimizer


class LoRATrainer:
    """LoRA trainer for biome-specific style adaptation"""

    def __init__(self, config: Dict):
        self.config = config
        self.accelerator = self._setup_accelerator()
        self.device = self.accelerator.device

        # Setup paths
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self._setup_models()
        self._setup_lora()
        self._setup_training()

    def _setup_accelerator(self) -> Accelerator:
        """Setup accelerator with mixed precision"""
        project_config = ProjectConfiguration(
            project_dir=self.config["output_dir"],
            logging_dir=os.path.join(self.config["output_dir"], "logs"),
        )

        return Accelerator(
            mixed_precision=(
                "fp16" if self.config.get("mixed_precision", True) else "no"
            ),
            gradient_accumulation_steps=self.config.get(
                "gradient_accumulation_steps", 4
            ),
            project_config=project_config,
            log_with="tensorboard",
        )

    def _setup_models(self):
        """Load base SDXL models"""
        model_id = self.config.get(
            "model_id", "stabilityai/stable-diffusion-xl-base-1.0"
        )

        # Load models
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=(
                torch.float16
                if self.config.get("mixed_precision", True)
                else torch.float32
            ),
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=(
                torch.float16
                if self.config.get("mixed_precision", True)
                else torch.float32
            ),
        )

        self.text_encoder_1 = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=(
                torch.float16
                if self.config.get("mixed_precision", True)
                else torch.float32
            ),
        )

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_id,
            subfolder="text_encoder_2",
            torch_dtype=(
                torch.float16
                if self.config.get("mixed_precision", True)
                else torch.float32
            ),
        )

        # Load tokenizers
        from transformers import CLIPTokenizer

        self.tokenizer_1 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )

        # Freeze VAE and text encoders
        self.vae.requires_grad_(False)
        self.text_encoder_1.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        # Move to device
        self.vae.to(self.device)
        self.text_encoder_1.to(self.device)
        self.text_encoder_2.to(self.device)

    def _setup_lora(self):
        """Configure LoRA for UNet"""
        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 8),
            lora_alpha=self.config.get("lora_alpha", 8),
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=self.config.get("lora_dropout", 0.1),
        )

        # Add LoRA to UNet
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.to(self.device)

        # Print trainable parameters
        self.unet.print_trainable_parameters()

    def _setup_training(self):
        """Setup optimizer, scheduler, and dataloaders"""
        # Optimizer
        self.optimizer = get_lora_optimizer(
            self.unet,
            learning_rate=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-2),
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
            subfolder="scheduler",
        )

        # Dataloaders
        self._setup_dataloaders()

        # Learning rate scheduler
        self.lr_scheduler = get_scheduler(
            self.config.get("lr_scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 500),
            num_training_steps=self.config.get("num_train_steps", 10000),
        )

        # Prepare for distributed training
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )

    def _setup_dataloaders(self):
        """Create training and validation dataloaders"""
        # Training dataset
        train_dataset = DynamicGameAssetDataset(
            data_dir=Path(self.config["data_dir"]),
            split="train",
            filters={"biome": self.config.get("biome", None)},
            transform=self._get_transforms(),
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 1),
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )

        # Validation dataset
        val_dataset = DynamicGameAssetDataset(
            data_dir=Path(self.config["data_dir"]),
            split="val",
            filters={"biome": self.config.get("biome", None)},
            transform=self._get_transforms(),
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=2
        )

    def _get_transforms(self):
        """Get image transforms"""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.config.get("resolution", 512),
                        self.config.get("resolution", 512),
                    )
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt for SDXL"""
        # Tokenize
        tokens_1 = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        tokens_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode
        with torch.no_grad():
            encoder_output_1 = self.text_encoder_1(tokens_1, output_hidden_states=True)
            encoder_output_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)

        # Get embeddings
        text_embeds = torch.cat(
            [encoder_output_1.hidden_states[-2], encoder_output_2.hidden_states[-2]],
            dim=-1,
        )

        pooled_embeds = encoder_output_2.text_embeds

        return text_embeds, pooled_embeds

    def train(self):
        """Main training loop"""
        global_step = 0

        # Initialize tracking
        self.accelerator.init_trackers(project_name="madwe-lora", config=self.config)

        for epoch in range(self.config.get("num_epochs", 10)):
            self.unet.train()
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.get('num_epochs', 10)}",
            )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.unet):
                    # Get images and prompts
                    images = batch["image"].to(self.device)
                    prompts = batch.get(
                        "prompt",
                        [
                            f"{batch['metadata'][i]['biome']} game asset"
                            for i in range(len(batch["image"]))
                        ],
                    )

                    # Encode images to latents
                    with torch.no_grad():
                        latents = self.vae.encode(images).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor

                    # Sample noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=self.device,
                    ).long()

                    # Add noise to latents
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # Encode prompts
                    text_embeds_list = []
                    pooled_embeds_list = []
                    for prompt in prompts:
                        text_embeds, pooled_embeds = self._encode_prompt(prompt)
                        text_embeds_list.append(text_embeds)
                        pooled_embeds_list.append(pooled_embeds)

                    text_embeds = torch.cat(text_embeds_list, dim=0)
                    pooled_embeds = torch.cat(pooled_embeds_list, dim=0)

                    # Create time embeddings
                    time_ids = torch.tensor(
                        [
                            [
                                self.config.get("resolution", 512),
                                self.config.get("resolution", 512),
                                0,
                                0,
                                self.config.get("resolution", 512),
                                self.config.get("resolution", 512),
                            ]
                        ],
                        device=self.device,
                    ).repeat(latents.shape[0], 1)

                    # Predict noise
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_embeds,
                            "time_ids": time_ids,
                        },
                    ).sample

                    # Calculate loss
                    loss = F.mse_loss(model_pred, noise, reduction="mean")

                    # Add style consistency loss if enabled
                    if self.config.get("use_style_loss", True):
                        style_loss = self._compute_style_loss(model_pred, noise)
                        loss = (
                            loss
                            + self.config.get("style_loss_weight", 0.1) * style_loss
                        )

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.config.get("max_grad_norm", 1.0) > 0:
                        self.accelerator.clip_grad_norm_(
                            self.unet.parameters(),
                            self.config.get("max_grad_norm", 1.0),
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Logging
                    if global_step % self.config.get("log_every", 50) == 0:
                        self.accelerator.log(
                            {
                                "loss": loss.item(),
                                "lr": self.lr_scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "step": global_step,
                            }
                        )

                    # Update progress
                    progress_bar.set_postfix(loss=loss.item())
                    global_step += 1

                    # Save checkpoint
                    if global_step % self.config.get("save_every", 500) == 0:
                        self._save_checkpoint(global_step)

                    # Validation
                    if global_step % self.config.get("val_every", 1000) == 0:
                        self._validate()

        # Final save
        self._save_checkpoint(global_step, final=True)
        self.accelerator.end_training()

    def _compute_style_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute style consistency loss using Gram matrices"""

        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram / (c * h * w)

        pred_gram = gram_matrix(pred)
        target_gram = gram_matrix(target)

        return F.mse_loss(pred_gram, target_gram)

    def _save_checkpoint(self, step: int, final: bool = False):
        """Save LoRA checkpoint"""
        save_path = (
            self.output_dir / f"checkpoint-{step}"
            if not final
            else self.output_dir / "final"
        )
        save_path.mkdir(exist_ok=True)

        # Save LoRA weights
        self.accelerator.unwrap_model(self.unet).save_pretrained(save_path)

        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "step": step,
                "config": self.config,
            },
            save_path / "training_state.pt",
        )

        print(f"Saved checkpoint to {save_path}")

    def _validate(self):
        """Run validation"""
        self.unet.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                images = batch["image"].to(self.device)
                prompts = batch.get(
                    "prompt",
                    [
                        f"{batch['metadata'][i]['biome']} game asset"
                        for i in range(len(batch["image"]))
                    ],
                )

                # Same forward pass as training
                latents = (
                    self.vae.encode(images).latent_dist.sample()
                    * self.vae.config.scaling_factor
                )
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 1000, (latents.shape[0],), device=self.device
                ).long()
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                text_embeds_list = []
                pooled_embeds_list = []
                for prompt in prompts:
                    text_embeds, pooled_embeds = self._encode_prompt(prompt)
                    text_embeds_list.append(text_embeds)
                    pooled_embeds_list.append(pooled_embeds)

                text_embeds = torch.cat(text_embeds_list, dim=0)
                pooled_embeds = torch.cat(pooled_embeds_list, dim=0)

                time_ids = torch.tensor(
                    [[512, 512, 0, 0, 512, 512]], device=self.device
                ).repeat(latents.shape[0], 1)

                model_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,
                        "time_ids": time_ids,
                    },
                ).sample

                loss = F.mse_loss(model_pred, noise)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        self.accelerator.log({"val_loss": avg_val_loss})
        print(f"Validation loss: {avg_val_loss:.4f}")

        self.unet.train()


def main():
    """Main training entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA for MADWE")
    parser.add_argument("--biome", type=str, required=True, help="Biome to train on")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed", help="Data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/models/lora", help="Output directory"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--resolution", type=int, default=512, help="Training resolution"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Use mixed precision"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation",
    )

    args = parser.parse_args()

    # Create config
    config = {
        "biome": args.biome,
        "data_dir": args.data_dir,
        "output_dir": os.path.join(args.output_dir, f"lora_{args.biome}"),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "lora_rank": args.lora_rank,
        "resolution": args.resolution,
        "mixed_precision": args.mixed_precision,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_steps": 10000,
        "warmup_steps": 500,
        "save_every": 500,
        "val_every": 1000,
        "log_every": 50,
        "use_style_loss": True,
        "style_loss_weight": 0.1,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lora_alpha": args.lora_rank,
        "lora_dropout": 0.1,
        "num_workers": 4,
    }

    print(f"Training LoRA for biome: {args.biome}")
    print(f"Output directory: {config['output_dir']}")

    # Train
    trainer = LoRATrainer(config)
    trainer.train()

    print("Training complete!")


if __name__ == "__main__":
    main()
