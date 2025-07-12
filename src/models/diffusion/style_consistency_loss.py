"""
Style Consistency Loss and Biome-Specific Training
Day 6: Monday, June 10 - Hierarchical WFC Implementation
Implements style consistency loss and texture variation pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import json
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
import yaml
from tqdm import tqdm
import wandb
from collections import defaultdict

# Import LPIPS for perceptual loss
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available, using basic perceptual loss")


@dataclass
class BiomeConfig:
    """Configuration for biome-specific training"""

    name: str
    primary_colors: List[Tuple[int, int, int]]
    texture_patterns: List[str]
    style_references: List[str]
    consistency_weight: float = 1.0
    diversity_weight: float = 0.5
    variation_params: Dict[str, Any] = None


class StyleConsistencyLoss(nn.Module):
    """Custom loss for maintaining style consistency within biomes"""

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        use_lpips: bool = True,
        use_gram_matrix: bool = True,
        use_color_histogram: bool = True,
    ):
        super().__init__()

        # Perceptual loss using LPIPS or VGG
        if use_lpips and LPIPS_AVAILABLE:
            self.perceptual_loss = lpips.LPIPS(net="vgg")
        else:
            self.perceptual_loss = self._create_vgg_loss()

        self.use_gram_matrix = use_gram_matrix
        self.use_color_histogram = use_color_histogram

        # Feature extractor for style features
        self.feature_extractor = feature_extractor or self._create_feature_extractor()

        # Loss weights
        self.weights = {
            "perceptual": 1.0,
            "style": 0.5,
            "color": 0.3,
            "texture": 0.4,
            "spatial": 0.2,
        }

    def _create_vgg_loss(self) -> nn.Module:
        """Create VGG-based perceptual loss"""
        from torchvision.models import vgg16

        class VGGPerceptualLoss(nn.Module):
            def __init__(self):
                super().__init__()
                vgg = vgg16(pretrained=True).features
                self.layers = nn.ModuleList(
                    [
                        vgg[:4],  # relu1_2
                        vgg[4:9],  # relu2_2
                        vgg[9:16],  # relu3_3
                        vgg[16:23],  # relu4_3
                    ]
                )

                # Freeze parameters
                for param in self.parameters():
                    param.requires_grad = False

            def forward(self, x):
                features = []
                for layer in self.layers:
                    x = layer(x)
                    features.append(x)
                return features

        return VGGPerceptualLoss()

    def _create_feature_extractor(self) -> nn.Module:
        """Create feature extractor for style analysis"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style representation"""
        b, c, h, w = features.shape
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def color_histogram_loss(
        self, pred: torch.Tensor, target: torch.Tensor, bins: int = 32
    ) -> torch.Tensor:
        """Compute color histogram matching loss"""
        batch_size = pred.shape[0]
        loss = 0.0

        for i in range(batch_size):
            for c in range(3):  # RGB channels
                # Compute histograms
                pred_hist = torch.histc(pred[i, c], bins=bins, min=0, max=1)
                target_hist = torch.histc(target[i, c], bins=bins, min=0, max=1)

                # Normalize histograms
                pred_hist = pred_hist / pred_hist.sum()
                target_hist = target_hist / target_hist.sum()

                # Earth Mover's Distance approximation
                loss += torch.sum(
                    torch.abs(pred_hist.cumsum(0) - target_hist.cumsum(0))
                )

        return loss / (batch_size * 3)

    def texture_coherence_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Ensure texture patterns are coherent"""
        # Compute spatial gradients
        grad_x = torch.abs(features[:, :, :, 1:] - features[:, :, :, :-1])
        grad_y = torch.abs(features[:, :, 1:, :] - features[:, :, :-1, :])

        # Encourage smooth gradients (coherent textures)
        smoothness = torch.mean(grad_x) + torch.mean(grad_y)

        # Compute texture complexity using frequency analysis
        fft = torch.fft.fft2(features)
        magnitude = torch.abs(fft)

        # High frequency content indicates detailed texture
        high_freq = magnitude[:, :, features.shape[2] // 4 :, features.shape[3] // 4 :]
        texture_complexity = torch.mean(high_freq)

        # Balance between smoothness and complexity
        return smoothness - 0.1 * texture_complexity

    def spatial_consistency_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Ensure spatial layout consistency"""
        # Downsample for spatial structure comparison
        pool = nn.AdaptiveAvgPool2d((8, 8))
        pred_spatial = pool(pred)
        target_spatial = pool(target)

        # Compute structural similarity
        return F.mse_loss(pred_spatial, target_spatial)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        biome_refs: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute style consistency loss

        Args:
            pred: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]
            biome_refs: Reference images for the biome style

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Perceptual loss
        if LPIPS_AVAILABLE and isinstance(self.perceptual_loss, lpips.LPIPS):
            losses["perceptual"] = self.perceptual_loss(pred, target).mean()
        else:
            # VGG perceptual loss
            pred_features = self.perceptual_loss(pred)
            target_features = self.perceptual_loss(target)

            perceptual_loss = 0
            for pf, tf in zip(pred_features, target_features):
                perceptual_loss += F.mse_loss(pf, tf)
            losses["perceptual"] = perceptual_loss / len(pred_features)

        # Style loss using Gram matrices
        if self.use_gram_matrix:
            style_features_pred = self.feature_extractor(pred)
            style_features_target = self.feature_extractor(target)

            gram_pred = self.gram_matrix(style_features_pred)
            gram_target = self.gram_matrix(style_features_target)

            losses["style"] = F.mse_loss(gram_pred, gram_target)

        # Color histogram matching
        if self.use_color_histogram:
            losses["color"] = self.color_histogram_loss(pred, target)

        # Texture coherence
        texture_features = self.feature_extractor(pred)
        losses["texture"] = self.texture_coherence_loss(texture_features)

        # Spatial consistency
        losses["spatial"] = self.spatial_consistency_loss(pred, target)

        # If biome references provided, ensure consistency with biome style
        if biome_refs:
            biome_consistency = 0
            for ref in biome_refs:
                ref_features = self.feature_extractor(ref.unsqueeze(0))
                pred_features = self.feature_extractor(pred)

                # Compare Gram matrices for style
                ref_gram = self.gram_matrix(ref_features)
                pred_gram = self.gram_matrix(pred_features)

                biome_consistency += F.mse_loss(
                    pred_gram, ref_gram.expand_as(pred_gram)
                )

            losses["biome_consistency"] = biome_consistency / len(biome_refs)

        # Weighted total loss
        total_loss = sum(self.weights.get(k, 1.0) * v for k, v in losses.items())
        losses["total"] = total_loss

        return losses


class TextureVariationPipeline:
    """Pipeline for generating texture variations while maintaining style"""

    def __init__(
        self,
        base_model_path: str,
        biome_configs: Dict[str, BiomeConfig],
        output_dir: Path,
    ):
        self.base_model_path = base_model_path
        self.biome_configs = biome_configs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()

        # Variation strategies
        self.variation_strategies = {
            "color_shift": self.apply_color_shift,
            "pattern_blend": self.apply_pattern_blend,
            "detail_level": self.apply_detail_variation,
            "lighting": self.apply_lighting_variation,
            "weathering": self.apply_weathering,
        }

    def setup_models(self):
        """Setup diffusion models and components"""
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device)

        # Setup for training
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler

    def generate_base_texture(
        self, biome: str, pattern: str, seed: int = 42
    ) -> torch.Tensor:
        """Generate base texture for a biome"""
        config = self.biome_configs[biome]

        # Construct prompt
        prompt = f"{pattern} texture, {biome} biome, "
        prompt += (
            f"colors: {', '.join([f'rgb{c}' for c in config.primary_colors[:3]])}, "
        )
        prompt += "high quality, seamless, game asset, 4k detail"

        # Generate
        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            image = self.pipeline(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
                height=512,
                width=512,
            ).images[0]

        # Convert to tensor
        return transforms.ToTensor()(image).unsqueeze(0).to(self.device)

    def apply_color_shift(
        self, texture: torch.Tensor, params: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply color shifting while maintaining style"""
        shift_range = params.get("shift_range", 0.1)
        preserve_luminance = params.get("preserve_luminance", True)

        # Convert to HSV for better color control
        texture_np = texture.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Apply controlled color shift
        hsv = self._rgb_to_hsv(texture_np)

        # Shift hue
        hue_shift = np.random.uniform(-shift_range, shift_range)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 1.0

        # Slight saturation variation
        sat_factor = np.random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 1)

        if preserve_luminance:
            # Preserve original luminance
            original_v = hsv[:, :, 2].copy()
            rgb_shifted = self._hsv_to_rgb(hsv)
            hsv_new = self._rgb_to_hsv(rgb_shifted)
            hsv_new[:, :, 2] = original_v
            rgb_shifted = self._hsv_to_rgb(hsv_new)
        else:
            rgb_shifted = self._hsv_to_rgb(hsv)

        return (
            torch.from_numpy(rgb_shifted.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )

    def apply_pattern_blend(
        self, texture: torch.Tensor, params: Dict[str, Any]
    ) -> torch.Tensor:
        """Blend multiple patterns together"""
        blend_mode = params.get("blend_mode", "overlay")
        blend_strength = params.get("blend_strength", 0.5)
        secondary_pattern = params.get("secondary_pattern")

        if secondary_pattern is None:
            return texture

        # Generate or load secondary pattern
        if isinstance(secondary_pattern, str):
            # Generate secondary pattern
            secondary = self.generate_base_texture(
                params.get("biome", "forest"),
                secondary_pattern,
                seed=np.random.randint(0, 10000),
            )
        else:
            secondary = secondary_pattern

        # Apply blend
        if blend_mode == "overlay":
            blended = self._overlay_blend(texture, secondary, blend_strength)
        elif blend_mode == "multiply":
            blended = texture * secondary * (1 + blend_strength)
        elif blend_mode == "screen":
            blended = 1 - (1 - texture) * (1 - secondary) * (1 - blend_strength)
        else:  # linear
            blended = texture * (1 - blend_strength) + secondary * blend_strength

        return torch.clamp(blended, 0, 1)

    def apply_detail_variation(
        self, texture: torch.Tensor, params: Dict[str, Any]
    ) -> torch.Tensor:
        """Vary the level of detail in the texture"""
        detail_scale = params.get("detail_scale", 1.0)
        sharpness = params.get("sharpness", 1.0)

        # Frequency separation
        low_freq = F.avg_pool2d(texture, kernel_size=3, stride=1, padding=1)
        high_freq = texture - low_freq

        # Adjust detail level
        enhanced_detail = high_freq * detail_scale

        # Reconstruct with adjusted detail
        result = low_freq + enhanced_detail

        # Apply sharpening if needed
        if sharpness != 1.0:
            kernel = (
                torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
                .view(1, 1, 3, 3)
                .to(self.device)
            )

            sharpened = F.conv2d(result, kernel.repeat(3, 1, 1, 1), padding=1, groups=3)
            result = result + (sharpened - result) * (sharpness - 1)

        return torch.clamp(result, 0, 1)

    def apply_lighting_variation(
        self, texture: torch.Tensor, params: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply lighting variations"""
        light_direction = params.get("light_direction", [1, 1, 1])
        ambient_strength = params.get("ambient_strength", 0.3)

        # Create normal map approximation from texture
        gray = torch.mean(texture, dim=1, keepdim=True)

        # Sobel filters for gradients
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(self.device)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(self.device)
        )

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)

        # Approximate normals
        normals = torch.stack(
            [grad_x.squeeze(1), grad_y.squeeze(1), torch.ones_like(grad_x.squeeze(1))],
            dim=1,
        )
        normals = F.normalize(normals, dim=1)

        # Light direction
        light_dir = (
            torch.tensor(light_direction, dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(self.device)
        )
        light_dir = F.normalize(light_dir, dim=1)

        # Lambertian shading
        n_dot_l = torch.sum(normals * light_dir, dim=1, keepdim=True)
        n_dot_l = torch.clamp(n_dot_l, 0, 1)

        # Apply lighting
        lit_texture = texture * (ambient_strength + (1 - ambient_strength) * n_dot_l)

        return torch.clamp(lit_texture, 0, 1)

    def apply_weathering(
        self, texture: torch.Tensor, params: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply weathering effects"""
        weathering_type = params.get("type", "age")
        intensity = params.get("intensity", 0.5)

        if weathering_type == "age":
            # Desaturate and darken
            gray = torch.mean(texture, dim=1, keepdim=True)
            aged = texture * (1 - intensity * 0.5) + gray * intensity * 0.5
            aged = aged * (1 - intensity * 0.2)  # Darken
            return aged

        elif weathering_type == "moss":
            # Add green tinting in crevices
            gray = torch.mean(texture, dim=1, keepdim=True)
            moss_mask = (1 - gray) * intensity  # Darker areas get more moss

            moss_color = torch.tensor([0.2, 0.5, 0.1]).view(1, 3, 1, 1).to(self.device)
            mossed = texture * (1 - moss_mask) + moss_color * moss_mask
            return mossed

        elif weathering_type == "rust":
            # Add rust coloring
            rust_color = torch.tensor([0.7, 0.3, 0.1]).view(1, 3, 1, 1).to(self.device)

            # Create rust pattern
            noise = torch.rand_like(texture[:, :1, :, :])
            rust_mask = (noise > (1 - intensity)).float()

            rusted = texture * (1 - rust_mask) + rust_color * rust_mask
            return rusted

        return texture

    def generate_variations(
        self,
        biome: str,
        base_pattern: str,
        num_variations: int = 5,
        strategies: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """Generate multiple variations of a texture"""
        if strategies is None:
            strategies = list(self.variation_strategies.keys())

        # Generate base texture
        base_texture = self.generate_base_texture(biome, base_pattern)
        variations = [base_texture]

        config = self.biome_configs[biome]

        for i in range(num_variations - 1):
            # Start with base
            varied = base_texture.clone()

            # Apply random combination of strategies
            num_strategies = np.random.randint(1, min(4, len(strategies) + 1))
            selected_strategies = np.random.choice(
                strategies, num_strategies, replace=False
            )

            for strategy in selected_strategies:
                if strategy in self.variation_strategies:
                    # Get parameters for this strategy
                    if config.variation_params and strategy in config.variation_params:
                        params = config.variation_params[strategy].copy()
                    else:
                        params = self._get_default_params(strategy)

                    params["biome"] = biome

                    # Apply variation
                    varied = self.variation_strategies[strategy](varied, params)

            variations.append(varied)

        return variations

    def _get_default_params(self, strategy: str) -> Dict[str, Any]:
        """Get default parameters for variation strategy"""
        defaults = {
            "color_shift": {"shift_range": 0.1, "preserve_luminance": True},
            "pattern_blend": {"blend_mode": "overlay", "blend_strength": 0.3},
            "detail_level": {"detail_scale": 1.2, "sharpness": 1.1},
            "lighting": {"light_direction": [1, 1, 1], "ambient_strength": 0.3},
            "weathering": {"type": "age", "intensity": 0.3},
        }
        return defaults.get(strategy, {})

    def _overlay_blend(
        self, base: torch.Tensor, overlay: torch.Tensor, opacity: float
    ) -> torch.Tensor:
        """Photoshop-style overlay blend"""
        result = torch.where(
            base < 0.5, 2 * base * overlay, 1 - 2 * (1 - base) * (1 - overlay)
        )
        return base * (1 - opacity) + result * opacity

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV"""
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / [179, 255, 255]

    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB"""
        hsv_scaled = (hsv * [179, 255, 255]).astype(np.uint8)
        return cv2.cvtColor(hsv_scaled, cv2.COLOR_HSV2RGB).astype(np.float32) / 255


class BiomeSpecificTrainer:
    """Trainer for biome-specific LoRA models with style consistency"""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.setup_biome_configs()
        self.setup_training_environment()

    def setup_biome_configs(self):
        """Setup configurations for each biome"""
        self.biome_configs = {
            "forest": BiomeConfig(
                name="forest",
                primary_colors=[(34, 89, 34), (85, 107, 47), (139, 69, 19)],
                texture_patterns=["bark", "moss", "leaves", "dirt"],
                style_references=["realistic", "painterly", "stylized"],
                consistency_weight=1.0,
                diversity_weight=0.5,
                variation_params={
                    "color_shift": {"shift_range": 0.1, "preserve_luminance": True},
                    "weathering": {"type": "moss", "intensity": 0.4},
                },
            ),
            "desert": BiomeConfig(
                name="desert",
                primary_colors=[(238, 203, 173), (205, 133, 63), (244, 164, 96)],
                texture_patterns=["sand", "rock", "sandstone", "dunes"],
                style_references=["realistic", "stylized"],
                consistency_weight=1.2,
                diversity_weight=0.3,
                variation_params={
                    "lighting": {
                        "light_direction": [1, 2, 0.5],
                        "ambient_strength": 0.4,
                    },
                    "weathering": {"type": "age", "intensity": 0.5},
                },
            ),
            "snow": BiomeConfig(
                name="snow",
                primary_colors=[(255, 250, 250), (176, 224, 230), (192, 192, 192)],
                texture_patterns=["ice", "snow", "frost", "frozen_ground"],
                style_references=["realistic", "crystalline"],
                consistency_weight=1.1,
                diversity_weight=0.4,
            ),
            "volcanic": BiomeConfig(
                name="volcanic",
                primary_colors=[(139, 0, 0), (255, 69, 0), (64, 64, 64)],
                texture_patterns=["lava", "obsidian", "ash", "magma"],
                style_references=["realistic", "glowing"],
                consistency_weight=1.3,
                diversity_weight=0.6,
            ),
            "underwater": BiomeConfig(
                name="underwater",
                primary_colors=[(0, 119, 190), (0, 150, 199), (144, 224, 239)],
                texture_patterns=["coral", "sand", "rocks", "seaweed"],
                style_references=["realistic", "dreamy"],
                consistency_weight=1.0,
                diversity_weight=0.7,
            ),
            "sky": BiomeConfig(
                name="sky",
                primary_colors=[(135, 206, 250), (255, 255, 255), (255, 215, 0)],
                texture_patterns=["clouds", "aurora", "stars", "mist"],
                style_references=["realistic", "ethereal"],
                consistency_weight=0.8,
                diversity_weight=0.8,
            ),
        }

    def setup_training_environment(self):
        """Setup accelerate and training environment"""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            mixed_precision=self.config["mixed_precision"],
            log_with="wandb",
        )

        # Initialize style consistency loss
        self.style_loss = StyleConsistencyLoss()

        # Setup output directories
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_biome_lora(self, biome: str, dataset_path: Path):
        """Train LoRA for specific biome with style consistency"""
        config = self.biome_configs[biome]

        # Log to wandb
        if self.accelerator.is_main_process:
            wandb.init(project="madwe-biome-training", name=f"{biome}-lora")

        # Setup model components
        from ..models.diffusion.lora_trainer import LoRATrainer

        trainer = LoRATrainer(self.config["training_config_path"])

        # Custom training loop with style consistency
        self._train_with_style_consistency(trainer, config, dataset_path)

    def _train_with_style_consistency(
        self, trainer, config: BiomeConfig, dataset_path: Path
    ):
        """Training loop with style consistency loss"""
        # This would integrate with the existing LoRATrainer
        # but add our custom style consistency loss
        pass


# Import OpenCV for color space conversions
try:
    import cv2
except ImportError:
    print("Warning: OpenCV not available, some features will be limited")
    cv2 = None


def main():
    """Test biome-specific training and variation generation"""
    import argparse

    parser = argparse.ArgumentParser(description="Biome-specific texture training")
    parser.add_argument("--mode", choices=["train", "generate"], required=True)
    parser.add_argument("--biome", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default="configs/training/biome_config.yaml"
    )
    parser.add_argument("--output-dir", type=str, default="data/biome_textures")

    args = parser.parse_args()

    if args.mode == "train":
        trainer = BiomeSpecificTrainer(args.config)
        trainer.train_biome_lora(args.biome, Path(f"data/processed/train/{args.biome}"))

    elif args.mode == "generate":
        # Test variation generation
        biome_configs = {
            args.biome: BiomeConfig(
                name=args.biome,
                primary_colors=[(100, 150, 100), (150, 100, 50), (50, 50, 150)],
                texture_patterns=["pattern1", "pattern2"],
                style_references=["style1"],
            )
        }

        pipeline = TextureVariationPipeline(
            "runwayml/stable-diffusion-v1-5", biome_configs, Path(args.output_dir)
        )

        variations = pipeline.generate_variations(
            args.biome, "test_pattern", num_variations=5
        )

        print(f"Generated {len(variations)} variations for {args.biome}")


if __name__ == "__main__":
    main()
