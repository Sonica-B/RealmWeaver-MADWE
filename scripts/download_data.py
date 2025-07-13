#!/usr/bin/env python3
"""
AI-Powered Game Asset Generation using FLUX.1-schnell
Generates high-quality game assets with FLUX.1 model from Black Forest Labs
"""
import os

os.environ["HF_HOME"] = "D:\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface\\transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:\\huggingface\\hub"
os.environ["HF_DATASETS_CACHE"] = "D:\\huggingface\\datasets"

import numpy as np
from PIL import Image
from pathlib import Path
import random
import json
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List, Dict
import torch
from diffusers import FluxPipeline
import time
import os

from huggingface_hub import login

login("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")


class FluxAssetGenerator:
    """Generate game assets using FLUX.1-schnell model"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading FLUX.1-schnell model...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir="D:\\huggingface\\hub",
            use_auth_token="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",
        )

        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()  # Save VRAM
        else:
            self.pipe = self.pipe.to(self.device)

        print(f"FLUX.1-schnell loaded on {self.device}")

        # Game asset prompt templates
        self.texture_prompts = {
            "forest": {
                "bark": "photorealistic tree bark texture, seamless tileable pattern, high detail oak bark with deep grooves and ridges, natural brown colors with moss patches, 4k game texture, PBR ready, diffuse map",
                "leaves": "dense forest canopy texture from below, seamless tileable, green maple and oak leaves overlapping, dappled sunlight filtering through, game environment texture, high detail foliage",
                "moss": "lush green moss texture covering rocks, seamless tileable pattern, tiny detailed moss plants, moist appearance, forest floor game texture, photorealistic detail",
                "undergrowth": "dense forest undergrowth texture, ferns and small plants, seamless tileable pattern, various green shades, game environment ground cover, high detail vegetation",
            },
            "desert": {
                "sand": "smooth desert sand dunes texture, seamless tileable pattern, fine grain sand with wind ripples, golden beige color, game environment texture, photorealistic detail",
                "rock": "weathered sandstone texture, seamless tileable, layered sedimentary rock with erosion, orange and red bands, game-ready cliff texture, photorealistic detail",
                "cracked": "dry cracked mud texture, seamless tileable pattern, drought-affected soil with polygon cracks, brown and gray, game ground texture, photorealistic detail",
            },
            "snow": {
                "fresh": "pristine fresh snow texture, seamless tileable pattern, soft powder snow with subtle shadows, pure white with blue tints, game environment texture, photorealistic",
                "ice": "frozen lake ice texture, seamless tileable, transparent ice with cracks and bubbles, blue-white coloration, game environment surface, photorealistic detail",
                "frost": "delicate frost pattern texture, seamless tileable, ice crystals on glass effect, intricate branching patterns, game window or surface texture",
            },
            "volcanic": {
                "lava": "flowing lava texture, seamless tileable pattern, molten rock with bright orange cracks, dark cooled surface with glowing fissures, game environment texture",
                "ash": "volcanic ash texture, seamless tileable, fine gray powder with darker particles, post-eruption ground cover, game environment texture, photorealistic",
                "obsidian": "polished obsidian texture, seamless tileable, volcanic glass with sharp reflections, deep black with highlights, game-ready material",
            },
            "underwater": {
                "coral": "vibrant coral reef texture, seamless tileable pattern, mixed hard and soft corals, bright tropical colors, underwater game environment, photorealistic",
                "sand": "underwater sand texture with ripples, seamless tileable, light filtering creates patterns, fine white sand, ocean floor game texture",
                "rocks": "underwater rock formation texture, seamless tileable, algae-covered stone surface, green and brown tones, submarine game environment",
            },
            "cyberpunk": {
                "metal": "scratched chrome metal texture, seamless tileable, industrial surface with wear marks, reflective metallic finish, cyberpunk game asset, PBR material",
                "neon": "neon sign glow texture, seamless tileable pattern, bright pink and blue light strips, cyberpunk city aesthetic, game environment texture",
                "tech": "circuit board texture, seamless tileable, detailed PCB with components, green substrate with gold traces, high-tech game texture",
            },
        }

        self.sprite_prompts = {
            "character": {
                "warrior": "fantasy RPG warrior character sprite, full body game asset, heavy plate armor with ornate details, battle-worn appearance, standing pose, white background, high detail game art",
                "mage": "wizard character sprite for games, flowing robes with magical symbols, staff in hand, mystical aura, white background, detailed character art, game asset",
                "archer": "elven archer sprite, leather armor with nature motifs, bow ready, graceful pose, white background, game character asset, high detail",
            },
            "item": {
                "sword": "legendary flaming sword sprite, ornate fantasy weapon, magical fire effects, glowing runes on blade, white background, game item asset, high detail",
                "potion": "health potion sprite, glass bottle with red liquid, cork stopper, magical glow effect, white background, RPG game item, detailed asset",
                "gem": "magical crystal gem sprite, multifaceted jewel with inner light, sparkling effects, white background, game treasure item, high detail",
            },
            "effect": {
                "explosion": "cartoon explosion sprite, stylized blast with smoke rings, bright orange and yellow colors, white background, game effect animation frame",
                "magic": "magical spell effect sprite, swirling energy with sparkles, blue and purple colors, white background, game VFX asset",
                "fire": "fire effect sprite, realistic flames with smoke, orange and red colors, white background, game particle effect",
            },
        }

    def generate_texture(
        self, biome: str, pattern: str, size: Tuple[int, int] = (1024, 1024)
    ) -> np.ndarray:
        """Generate texture using FLUX.1-schnell"""
        prompt = self.texture_prompts.get(biome, {}).get(
            pattern, f"{pattern} texture for {biome}, seamless tileable, game asset"
        )

        with torch.no_grad():
            image = self.pipe(
                prompt,
                height=size[1],
                width=size[0],
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(
                    random.randint(0, 999999)
                ),
            ).images[0]

        return np.array(image)

    def generate_sprite(
        self, category: str, sprite_type: str, size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """Generate sprite using FLUX.1-schnell"""
        prompt = self.sprite_prompts.get(category, {}).get(
            sprite_type, f"{sprite_type} sprite, game asset, white background"
        )

        with torch.no_grad():
            image = self.pipe(
                prompt,
                height=size[1],
                width=size[0],
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator(self.device).manual_seed(
                    random.randint(0, 999999)
                ),
            ).images[0]

        # Convert to RGBA for transparency
        img_array = np.array(image)
        if img_array.shape[2] == 3:
            # Simple white background removal
            rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
            rgba[:, :, :3] = img_array

            # Create alpha mask (non-white pixels)
            gray = np.mean(img_array, axis=2)
            alpha = np.where(gray < 250, 255, 0).astype(np.uint8)
            rgba[:, :, 3] = alpha

            return rgba

        return img_array


def generate_flux_assets(
    output_dir: Path | str,
    num_samples: int = 50,
    texture_size: Tuple[int, int] = (1024, 1024),
    sprite_size: Tuple[int, int] = (512, 512),
) -> None:
    """Generate game assets using FLUX.1-schnell"""
    output_dir = Path(output_dir)
    generator = FluxAssetGenerator()

    # Define asset categories
    texture_configs = {
        "forest": ["bark", "leaves", "moss", "undergrowth"],
        "desert": ["sand", "rock", "cracked"],
        "snow": ["fresh", "ice", "frost"],
        "volcanic": ["lava", "ash", "obsidian"],
        "underwater": ["coral", "sand", "rocks"],
        "cyberpunk": ["metal", "neon", "tech"],
    }

    sprite_configs = {
        "character": ["warrior", "mage", "archer"],
        "item": ["sword", "potion", "gem"],
        "effect": ["explosion", "magic", "fire"],
    }

    print("=== FLUX.1-schnell Game Asset Generation ===")
    print(f"Texture size: {texture_size}")
    print(f"Sprite size: {sprite_size}")
    print(f"Samples per type: {num_samples}")
    print("=" * 40)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "generator": "FLUX.1-schnell",
        "total_generated": 0,
        "assets": [],
    }

    # Generate textures
    for biome, patterns in texture_configs.items():
        for pattern in patterns:
            output_path = output_dir / "raw" / "textures" / biome
            output_path.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(num_samples), desc=f"Textures/{biome}/{pattern}"):
                try:
                    texture = generator.generate_texture(biome, pattern, texture_size)

                    img = Image.fromarray(texture.astype(np.uint8))
                    filename = f"{biome}_{pattern}_{i:04d}.png"
                    filepath = output_path / filename
                    img.save(filepath, optimize=True)

                    metadata["assets"].append(
                        {
                            "filename": filename,
                            "filepath": str(filepath.relative_to(output_dir)),
                            "type": "texture",
                            "biome": biome,
                            "pattern": pattern,
                            "size": texture_size,
                        }
                    )
                    metadata["total_generated"] += 1

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    print(f"\nError: {e}")

    # Generate sprites
    for category, types in sprite_configs.items():
        for sprite_type in types:
            output_path = output_dir / "raw" / "sprites" / category
            output_path.mkdir(parents=True, exist_ok=True)

            for i in tqdm(
                range(num_samples // 2), desc=f"Sprites/{category}/{sprite_type}"
            ):
                try:
                    sprite = generator.generate_sprite(
                        category, sprite_type, sprite_size
                    )

                    img = Image.fromarray(sprite.astype(np.uint8))
                    filename = f"{category}_{sprite_type}_{i:04d}.png"
                    filepath = output_path / filename
                    img.save(filepath, optimize=True)

                    metadata["assets"].append(
                        {
                            "filename": filename,
                            "filepath": str(filepath.relative_to(output_dir)),
                            "type": "sprite",
                            "category": category,
                            "sprite_type": sprite_type,
                            "size": sprite_size,
                        }
                    )
                    metadata["total_generated"] += 1

                    time.sleep(0.5)

                except Exception as e:
                    print(f"\nError: {e}")

    # Save metadata
    metadata_path = output_dir / "metadata" / "flux_generation.json"
    metadata_path.parent.mkdir(exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Complete ===")
    print(f"Generated: {metadata['total_generated']} assets")
    print(f"Saved to: {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate game assets using FLUX.1-schnell"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per type")
    parser.add_argument("--texture-size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--sprite-size", type=int, nargs=2, default=[512, 512])

    args = parser.parse_args()

    print("FLUX.1-schnell Game Asset Generator")
    print("Note: Requires ~40GB VRAM or CPU offloading")

    generate_flux_assets(
        args.output_dir,
        num_samples=args.num_samples,
        texture_size=tuple(args.texture_size),
        sprite_size=tuple(args.sprite_size),
    )


if __name__ == "__main__":
    main()
