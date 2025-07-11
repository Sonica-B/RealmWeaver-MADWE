"""
LoRA training for Stable Diffusion 
"""

import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from ...utils.dataloader import DynamicConditionalDataset


class LoRATrainer:
    """LoRA trainer for game assets with Python 3.11 enhancements"""

    def __init__(self, config_path: str | Path):
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.accelerator = self._setup_accelerator()
        self.models: Dict[str, Any] = {}
        set_seed(self.config.get("seed", 42))

    def _setup_accelerator(self) -> Accelerator:
        project_config = ProjectConfiguration(
            project_dir=self.config["output_dir"],
            logging_dir=self.config["logging_dir"],
            automatic_checkpoint_naming=True,
            total_limit=2,  # Keep only 2 checkpoints
        )

        return Accelerator(
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            mixed_precision=self.config["mixed_precision"],
            log_with=self.config.get("log_with", "tensorboard"),
            project_config=project_config,
        )

    def setup_models(self) -> None:
        """Load and configure models with latest APIs"""
        model_id = self.config["pretrained_model_name"]

        # Load scheduler
        self.models["noise_scheduler"] = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        # Load tokenizer
        self.models["tokenizer"] = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )

        # Load models with specific components
        self.models["text_encoder"] = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder"
        )

        self.models["vae"] = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

        self.models["unet"] = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet"
        )

        # Move to appropriate dtype
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Freeze non-LoRA models and cast to appropriate dtype
        self.models["vae"].requires_grad_(False)
        self.models["text_encoder"].requires_grad_(False)
        self.models["unet"].requires_grad_(False)

        self.models["vae"].to(self.accelerator.device, dtype=weight_dtype)
        self.models["text_encoder"].to(self.accelerator.device, dtype=weight_dtype)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_alpha"],
            target_modules=self.config["target_modules"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            task_type=TaskType.DIFFUSION_IMAGE_GENERATION,
        )

        # Add LoRA to UNet
        self.models["unet"] = get_peft_model(self.models["unet"], lora_config)
        self.models["unet"].print_trainable_parameters()

        # Enable gradient checkpointing if specified
        if self.config.get("gradient_checkpointing", False):
            self.models["unet"].enable_gradient_checkpointing()
            self.models["text_encoder"].gradient_checkpointing_enable()

    def create_dataloader(
        self, condition: Dict[str, str], asset_type: str = "textures"
    ) -> DataLoader:
        """Create dataloader for specific conditions"""
        dataset = DynamicConditionalDataset(
            data_dir=Path(self.config["data_dir"]),
            asset_type=asset_type,
            condition=condition,
            tokenizer=self.models["tokenizer"],
            size=self.config["resolution"],
        )

        return DataLoader(
            dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=torch.cuda.is_available(),
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with mixed precision"""
        with self.accelerator.autocast():
            # Encode images to latents
            latents = (
                self.models["vae"]
                .encode(batch["pixel_values"].to(self.models["vae"].dtype))
                .latent_dist.sample()
            )
            latents = latents * self.models["vae"].config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample timesteps
            timesteps = torch.randint(
                0,
                self.models["noise_scheduler"].config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()

            # Add noise
            noisy_latents = self.models["noise_scheduler"].add_noise(
                latents, noise, timesteps
            )

            # Get text embeddings
            encoder_hidden_states = self.models["text_encoder"](
                batch["input_ids"], return_dict=False
            )[0]

            # Predict noise
            model_pred = self.models["unet"](
                noisy_latents, timesteps, encoder_hidden_states, return_dict=False
            )[0]

            # Calculate loss
            if self.models["noise_scheduler"].config.prediction_type == "epsilon":
                target = noise
            elif (
                self.models["noise_scheduler"].config.prediction_type == "v_prediction"
            ):
                target = self.models["noise_scheduler"].get_velocity(
                    latents, noise, timesteps
                )
            else:
                raise ValueError(
                    f"Unknown prediction type {self.models['noise_scheduler'].config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def train(
        self,
        condition: Dict[str, str],
        asset_type: str = "textures",
        num_epochs: int | None = None,
    ) -> None:
        """Train LoRA for specific conditions with modern training loop"""
        if num_epochs is None:
            num_epochs = self.config.get("num_train_epochs", 10)

        # Setup
        self.setup_models()
        dataloader = self.create_dataloader(condition, asset_type)

        # Optimizer with optional 8-bit
        optimizer_class = torch.optim.AdamW
        if self.config.get("use_8bit_adam", False):
            try:
                import bitsandbytes as bnb

                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                print("bitsandbytes not available, using standard AdamW")

        optimizer = optimizer_class(
            self.models["unet"].parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config["adam_beta1"], self.config["adam_beta2"]),
            weight_decay=self.config["adam_weight_decay"],
            eps=self.config["adam_epsilon"],
        )

        # Learning rate scheduler
        from transformers import get_scheduler

        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config.get("lr_warmup_steps", 500),
            num_training_steps=len(dataloader) * num_epochs,
        )

        # Prepare with accelerator
        self.models["unet"], optimizer, dataloader, lr_scheduler = (
            self.accelerator.prepare(
                self.models["unet"], optimizer, dataloader, lr_scheduler
            )
        )

        # Initialize trackers
        if self.accelerator.is_main_process:
            condition_str = "_".join(f"{k}_{v}" for k, v in condition.items())
            self.accelerator.init_trackers(
                project_name=f"madwe-lora-{condition_str}", config=self.config
            )

        # Training loop
        global_step = 0

        for epoch in range(num_epochs):
            self.models["unet"].train()
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.models["unet"]):
                    loss = self.train_step(batch)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.models["unet"].parameters(),
                            self.config.get("max_grad_norm", 1.0),
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_step += 1

                    if self.accelerator.is_main_process:
                        progress_bar.set_postfix(
                            {
                                "loss": loss.detach().item(),
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                        )

                        # Log metrics
                        self.accelerator.log(
                            {
                                "train_loss": loss.detach().item(),
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "global_step": global_step,
                            },
                            step=global_step,
                        )

                        # Save checkpoint
                        if global_step % self.config["save_steps"] == 0:
                            self.save_checkpoint(condition, asset_type, global_step)

                        # Validation
                        if global_step % self.config.get("validation_steps", 100) == 0:
                            self.validate(condition, asset_type, global_step)

        # Save final model
        self.accelerator.wait_for_everyone()
        self.save_checkpoint(condition, asset_type, global_step, final=True)
        self.accelerator.end_training()

    def save_checkpoint(
        self, condition: Dict[str, str], asset_type: str, step: int, final: bool = False
    ) -> None:
        """Save LoRA checkpoint with metadata"""
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            condition_str = "_".join(f"{k}_{v}" for k, v in condition.items())
            save_path = (
                Path(self.config["output_dir"]) / f"lora_{asset_type}_{condition_str}"
            )
            if final:
                save_path = save_path / "final"
            else:
                save_path = save_path / f"checkpoint-{step}"

            # Save LoRA weights
            unwrapped_model = self.accelerator.unwrap_model(self.models["unet"])
            unwrapped_model.save_pretrained(save_path)

            # Save training state
            self.accelerator.save_state(save_path / "training_state")

            print(f"Saved checkpoint to {save_path}")

    def validate(self, condition: Dict[str, str], asset_type: str, step: int) -> None:
        """Generate validation images"""
        self.models["unet"].eval()

        # Create pipeline for inference
        pipeline = StableDiffusionPipeline(
            vae=self.models["vae"],
            text_encoder=self.models["text_encoder"],
            tokenizer=self.models["tokenizer"],
            unet=self.accelerator.unwrap_model(self.models["unet"]),
            scheduler=self.models["noise_scheduler"],
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipeline.set_progress_bar_config(disable=True)

        # Generate validation images
        condition_str = "_".join(f"{k} {v}" for k, v in condition.items())
        validation_prompts = [
            f"{asset_type} asset, {condition_str}, detailed, high quality",
            f"{asset_type} for game, {condition_str}, game ready, seamless",
            f"{asset_type} pattern, {condition_str}, stylized game art",
        ]

        images = []
        with torch.no_grad():
            for prompt in validation_prompts:
                image = pipeline(
                    prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=torch.Generator(
                        device=self.accelerator.device
                    ).manual_seed(42),
                ).images[0]
                images.append(image)

        # Log images
        if self.accelerator.is_main_process:
            self.accelerator.log(
                {
                    f"validation/{asset_type}_{condition_str}": [
                        wandb.Image(img, caption=prompt)
                        for img, prompt in zip(images, validation_prompts)
                    ]
                },
                step=step,
            )

        self.models["unet"].train()


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA for MADWE")
    parser.add_argument(
        "--config", type=str, default="configs/training/lora_config.yaml"
    )
    parser.add_argument(
        "--asset-type",
        type=str,
        default="textures",
        choices=["textures", "sprites", "gameplay"],
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Condition string in format key1_value1,key2_value2 (e.g., genre_platformer,style_pixel)",
    )
    args = parser.parse_args()

    # Parse condition string
    condition = {}
    for pair in args.condition.split(","):
        key, value = pair.split("_", 1)
        condition[key] = value

    trainer = LoRATrainer(args.config)
    trainer.train(condition, args.asset_type)


if __name__ == "__main__":
    main()
