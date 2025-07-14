"""
Inference utilities for MADWE diffusion models with LoRA
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Union
import numpy as np
from PIL import Image

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors import safe_open


class LoRAInference:
    """Inference with trained LoRA adapters"""

    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # Load base pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant="fp16" if self.dtype == torch.float16 else None,
        ).to(self.device)

        # Optimize scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Enable optimizations
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()

        self.current_lora = None

    def load_lora(self, lora_path: Union[str, Path], scale: float = 1.0):
        """Load LoRA adapter"""
        lora_path = Path(lora_path)

        if not lora_path.exists():
            raise ValueError(f"LoRA path not found: {lora_path}")

        # Unload previous LoRA if exists
        if self.current_lora:
            self.pipe.unload_lora_weights()

        # Load new LoRA
        self.pipe.load_lora_weights(lora_path)
        self.pipe.fuse_lora(lora_scale=scale)

        self.current_lora = lora_path
        print(f"Loaded LoRA: {lora_path}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images with current LoRA"""

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        # Generate
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        ).images

        return images

    def generate_batch(
        self, prompts: List[str], negative_prompts: Optional[List[str]] = None, **kwargs
    ) -> List[Image.Image]:
        """Generate batch of images"""
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)

        all_images = []
        for prompt, neg_prompt in zip(prompts, negative_prompts):
            images = self.generate(prompt, neg_prompt, **kwargs)
            all_images.extend(images)

        return all_images


class MultiLoRAInference:
    """Inference with multiple LoRA adapters for different biomes"""

    def __init__(
        self,
        base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_dir: Union[str, Path] = "data/models/lora",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_dir = Path(lora_dir)

        # Base inference engine
        self.inference = LoRAInference(base_model_id, device)

        # Discover available LoRAs
        self.available_loras = self._discover_loras()
        print(f"Found {len(self.available_loras)} LoRA adapters")

    def _discover_loras(self) -> Dict[str, Path]:
        """Find all trained LoRA adapters"""
        loras = {}

        if self.lora_dir.exists():
            for lora_path in self.lora_dir.iterdir():
                if lora_path.is_dir() and lora_path.name.startswith("lora_"):
                    biome = lora_path.name.replace("lora_", "")
                    checkpoint_path = lora_path / "final"
                    if checkpoint_path.exists():
                        loras[biome] = checkpoint_path
                    else:
                        # Find latest checkpoint
                        checkpoints = list(lora_path.glob("checkpoint-*"))
                        if checkpoints:
                            latest = max(
                                checkpoints, key=lambda p: int(p.name.split("-")[1])
                            )
                            loras[biome] = latest

        return loras

    def generate_for_biome(
        self, biome: str, prompt: str, enhance_prompt: bool = True, **kwargs
    ) -> List[Image.Image]:
        """Generate using biome-specific LoRA"""

        if biome not in self.available_loras:
            print(f"Warning: No LoRA found for biome '{biome}', using base model")
            return self.inference.generate(prompt, **kwargs)

        # Load biome LoRA
        self.inference.load_lora(self.available_loras[biome])

        # Enhance prompt with biome context if requested
        if enhance_prompt:
            prompt = f"{biome} biome style, {prompt}"

        return self.inference.generate(prompt, **kwargs)

    def generate_comparison(
        self, prompt: str, biomes: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, List[Image.Image]]:
        """Generate same prompt with different biome LoRAs"""

        if biomes is None:
            biomes = list(self.available_loras.keys())

        results = {}

        # Generate with base model
        results["base"] = self.inference.generate(prompt, **kwargs)

        # Generate with each biome
        for biome in biomes:
            if biome in self.available_loras:
                results[biome] = self.generate_for_biome(biome, prompt, **kwargs)

        return results


def quick_generate(
    prompt: str,
    lora_path: Optional[Union[str, Path]] = None,
    biome: Optional[str] = None,
    **kwargs,
) -> Image.Image:
    """Quick generation helper function"""

    if biome:
        # Use multi-LoRA inference
        multi_inference = MultiLoRAInference()
        images = multi_inference.generate_for_biome(biome, prompt, **kwargs)
    else:
        # Use single LoRA or base model
        inference = LoRAInference()
        if lora_path:
            inference.load_lora(lora_path)
        images = inference.generate(prompt, **kwargs)

    return images[0] if images else None
