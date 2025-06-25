"""
Diffusion model inference utilities 
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, LCMScheduler
from peft import PeftModel
import time
import warnings


class DiffusionInference:
    """Optimized inference for diffusion models with LoRA"""

    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        enable_xformers: bool = True,
    ):

        self.base_model = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.enable_xformers = enable_xformers
        self.pipe: StableDiffusionPipeline | None = None
        self.lora_models: Dict[str, Path] = {}

    def load_base_model(self) -> None:
        """Load base Stable Diffusion model with latest API"""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
        )

        # Use DPM solver for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe = self.pipe.to(self.device)

        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            if self.enable_xformers:
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    warnings.warn(f"xformers not available: {e}")

            # Enable VAE slicing for lower memory usage
            self.pipe.enable_vae_slicing()

    def load_lora(self, lora_path: Path | str, adapter_name: str) -> None:
        """Load LoRA adapter with PEFT"""
        if self.pipe is None:
            self.load_base_model()

        lora_path = Path(lora_path)

        # Load LoRA weights into UNet
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet, str(lora_path), adapter_name=adapter_name
        )

        self.lora_models[adapter_name] = lora_path
        print(f"Loaded LoRA adapter: {adapter_name}")

    def set_active_lora(self, adapter_name: str) -> None:
        """Set active LoRA adapter"""
        if adapter_name not in self.lora_models:
            raise ValueError(f"LoRA adapter '{adapter_name}' not loaded")

        self.pipe.unet.set_adapter(adapter_name)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image with timing info"""

        if self.pipe is None:
            self.load_base_model()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        start_time = time.perf_counter()

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            **kwargs,
        )

        generation_time = time.perf_counter() - start_time

        return {
            "image": result.images[0],
            "time": generation_time,
            "fps": 1.0 / generation_time,
            "settings": {
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "size": (width, height),
                "seed": seed,
            },
        }

    def generate_batch(
        self, prompts: List[str], batch_size: int = 1, **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple images with batching"""
        results = []

        # Process in batches for memory efficiency
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]

            # Generate batch
            batch_results = self.pipe(prompt=batch_prompts, **kwargs)

            # Convert to individual results
            for j, image in enumerate(batch_results.images):
                results.append({"image": image, "prompt": batch_prompts[j]})

        return results


class FastInference:
    """Optimized inference with LCM for real-time generation"""

    def __init__(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5",
    ):
        self.base_model = base_model
        self.lcm_lora_id = lcm_lora_id
        self.pipe: StableDiffusionPipeline | None = None

    def setup_lcm(self) -> None:
        """Setup LCM for 4-step generation"""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=True,
        )

        # Load LCM-LoRA
        self.pipe.load_lora_weights(self.lcm_lora_id)

        # Use LCM scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe = self.pipe.to("cuda")

        # Enable optimizations
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_vae_slicing()

    def generate_fast(
        self, prompt: str, num_inference_steps: int = 4, guidance_scale: float = 0.0
    ) -> Dict[str, Any]:
        """Generate with minimal steps using LCM"""

        if self.pipe is None:
            self.setup_lcm()

        start = time.perf_counter()

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,  # LCM uses 0.0 guidance
            width=512,
            height=512,
        )

        elapsed = time.perf_counter() - start

        return {
            "image": result.images[0],
            "time": elapsed,
            "fps": 1.0 / elapsed,
            "method": "LCM",
        }

    def benchmark(self, prompt: str, runs: int = 10) -> Dict[str, float]:
        """Benchmark generation speed"""
        times = []

        # Warmup
        self.generate_fast(prompt)

        # Benchmark runs
        for _ in range(runs):
            result = self.generate_fast(prompt)
            times.append(result["time"])

        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "mean_fps": 1.0 / np.mean(times),
        }
