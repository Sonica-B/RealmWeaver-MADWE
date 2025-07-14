#!/usr/bin/env python3
"""
Enhanced Sprite Generation using SDXL-Turbo
Fixed logic for colored, detailed sprites instead of outlines
"""
import os
import sys
import json
import random
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Dict

os.environ["HF_HOME"] = "D:\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface\\transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:\\huggingface\\hub"

try:
    import torch
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)

# Force GPU
if not torch.cuda.is_available():
    print("ERROR: GPU not available. This script requires CUDA.")
    sys.exit(1)


class SpriteGenerator:
    def __init__(self):
        self.device = "cuda"

        print("Loading SDXL-Turbo for sprite generation...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir="D:\\huggingface\\hub",
        ).to(self.device)

        # Use DPM++ for better quality
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        # Enhanced sprite prompts with color enforcement
        self.sprite_prompts = {
            # CHARACTER SPRITES - Multiple styles
            "pixel_characters": {
                "8bit_hero": "Pixel art RPG hero sprite, limited NES color palette, sword and shield equipped, idle animation frame, clear pixel boundaries, no anti-aliasing, authentic retro constraints, white background for transparency, game ready sprite sheet format",
                "16bit_mage": "SNES style wizard sprite, flowing robes with detailed shading, magical staff with glowing orb, spell casting pose, expanded color palette, subtle dithering for gradients, professional pixel art techniques",
                "modern_pixel": "modern indie pixel art character, detailed animations, dynamic lighting effects on pixels, contemporary color theory, sub-pixel animation techniques, mix of retro aesthetic with modern sensibilities",
            },
            "hand_drawn_characters": {
                "ink_warrior": "hand-drawn ink illustration warrior character, bold brush strokes, dynamic action pose, flowing cape and hair, crosshatching for shadows, expressive line weight variation, scanned traditional art aesthetic, high contrast black and white with red accents",
                "watercolor_elf": "ethereal elf character painted in watercolor style, soft color bleeds, translucent clothing layers, delicate features, botanical elements in hair, dreamy atmosphere, traditional media texture, white background",
                "sketch_rogue": "rough pencil sketch style rogue character, construction lines visible, dynamic gesture drawing, daggers and throwing knives, hood casting shadow over face, energetic line work, concept art quality",
            },
            "vector_characters": {
                "minimal_knight": "minimalist vector art knight, geometric shapes only, limited color palette, bold silhouette, modern flat design principles, scalable graphics quality, perfect for mobile games, clean composition",
                "gradient_ranger": "vector ranger with subtle gradients, bow drawn in action pose, layered clothing design, nature-inspired color scheme, smooth curves and sharp angles balanced, professional vector illustration",
                "iso_soldier": "isometric vector soldier sprite, 3/4 view perspective, modular armor pieces, color-coded team variations possible, perfect proportions for tile-based games, strategic game aesthetic",
            },
            # ITEM SPRITES - Various styles
            "fantasy_items": {
                "legendary_sword": "epic flaming longsword game item, ornate cross-guard with runic inscriptions, blade emanating magical fire, dragon motifs on hilt, glowing gems embedded, particle effects for flames, legendary rarity aura, detailed metalwork, center composition on transparent background",
                "healing_potion": "major healing potion bottle, swirling red liquid with golden sparkles, cork sealed with wax, glass refracting light beautifully, medieval bottle shape, magical glow from within, bubble effects, item rarity indicator",
                "artifact_amulet": "ancient magical amulet, intricate Celtic knot design, central gem pulsating with power, weathered gold chain, mystical symbols etched, soft magical aura, lore-rich appearance, museum quality detail",
            },
            "sci_fi_items": {
                "plasma_rifle": "futuristic plasma rifle weapon sprite, sleek design with glowing energy cores, holographic sight, heat dissipation vents, alien technology aesthetic, neon accent lights, modular attachments visible, high-tech materials",
                "nano_medkit": "advanced medical nanite injector, transparent canister showing swirling nanobots, digital display showing vitals, ergonomic grip design, sterile white with medical crosses, soft blue glow, emergency red accents",
                "quantum_key": "quantum encryption key device, crystalline structure with data streams inside, impossible geometry, holographic interface projecting, chrome and glass construction, particle effects, cyberpunk aesthetic",
            },
            # EFFECT SPRITES
            "magical_effects": {
                "fire_burst": "magical fire explosion sprite sheet, 8 frame sequence, dynamic flame shapes, orange to yellow to white heat gradient, smoke wisps, ember particles, impact shockwave, spell effect quality, transparent background",
                "ice_shatter": "frost spell impact effect, crystalline ice formations shattering, blue to white gradient, sharp geometric fragments, freezing mist, magical snowflakes, dynamic motion blur, frame-by-frame animation ready",
                "lightning_strike": "electric spell effect sprite, branching lightning bolt, bright white core with purple edges, electrical arcs, ground impact scorch, atmospheric glow, energy crackling, particle systems included",
            },
            "environmental_effects": {
                "toxic_cloud": "poisonous gas cloud sprite, sickly green wisps, bubbling toxic particles, semi-transparent layers, organic flowing shapes, hazard warning aesthetic, animated swirling motion, environmental danger",
                "portal_vortex": "dimensional portal effect, swirling energy vortex, space-time distortion, multiple color layers, particle streams, reality warping at edges, mystical transportation, animated sequence ready",
                "explosion_dust": "realistic explosion with debris, dust cloud expansion, rock fragments, shockwave distortion, fire at center transitioning to smoke, military/realistic style, multiple animation frames",
            },
            # UI ELEMENTS
            "game_ui": {
                "health_bar": "RPG health bar UI element, ornate frame design, red fill with gradient, decorative end caps, empty state included, damage animation ready, medieval fantasy styling, multiple size variations",
                "ability_button": "skill ability button frame, circular design with rune border, empty center for ability icon, cooldown sweep indicator, pressed state variation, magical glow effect, MOBA game style",
                "dialogue_box": "RPG dialogue window frame, wooden texture with metal corners, parchment background, decorative Celtic patterns, name plate area, semi-transparent option, visual novel quality",
            },
        }

    def generate_sprite(
        self, prompt: str, size: Tuple[int, int] = (768, 768)
    ) -> np.ndarray:
        """Generate sprite with improved settings"""

        # Enhanced prompt for colored output
        enhanced_prompt = f"{prompt}, single object or character, digital art, full color illustration, vibrant bright colors, saturated colors, detailed shading, professional game art"

        # Strong negative to avoid sketches
        negative_prompt = "sketch, multiple characters, multiple objects, line art, drawing, outline, pencil, black and white, monochrome, grayscale, simple, minimalist, uncolored, linework only, coloring book, wireframe"

        with torch.no_grad():
            # Use small guidance scale for SDXL-Turbo
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=6,  # Slightly more steps
                guidance_scale=2.0,  # Small positive guidance
                height=size[1],
                width=size[0],
                generator=torch.Generator(self.device).manual_seed(
                    random.randint(0, 999999)
                ),
            ).images[0]

        # Resize to standard sprite size
        image = image.resize((512, 512), Image.Resampling.LANCZOS)

        return self._process_sprite_alpha(np.array(image))

    def _process_sprite_alpha(self, img_array: np.ndarray) -> np.ndarray:
        """Process sprite with improved alpha extraction"""
        if img_array.shape[2] == 3:
            rgba = np.zeros((512, 512, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_array

            # Better white background removal
            # Calculate how "white" each pixel is
            white_threshold = 240
            is_white = np.all(img_array > white_threshold, axis=2)

            # Create alpha channel
            alpha = np.ones((512, 512), dtype=np.uint8) * 255
            alpha[is_white] = 0

            # Smooth edges using PIL
            alpha_img = Image.fromarray(alpha)
            alpha_img = alpha_img.filter(ImageFilter.SMOOTH_MORE)
            alpha_img = alpha_img.filter(ImageFilter.SMOOTH)

            rgba[:, :, 3] = np.array(alpha_img)
            return rgba

        return img_array

    def generate_category(self, category: str, sprite_type: str) -> np.ndarray:
        """Generate sprite from category"""
        prompt = self.sprite_prompts.get(category, {}).get(
            sprite_type,
            f"colorful {sprite_type} game sprite, vibrant colors, detailed art, white background",
        )
        return self.generate_sprite(prompt)


def generate_all_sprites(output_dir: Path, samples_per_type: int = 10):
    """Generate all sprite categories"""
    output_dir = Path(output_dir)
    generator = SpriteGenerator()

    sprite_categories = [
        "pixel_characters",
        "fantasy_characters",
        "cute_characters",
        "fantasy_items",
        "sci_fi_items",
        "ui_elements",
    ]

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "generator": "SDXL-Turbo Sprites Enhanced",
        "total_generated": 0,
        "assets": [],
    }

    print("\n=== Enhanced Sprite Generation ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Settings: 6 steps, guidance=1.5")
    print(f"Samples per type: {samples_per_type}")
    print("=" * 40)

    for category in sprite_categories:
        for sprite_type in generator.sprite_prompts.get(category, {}).keys():
            output_path = output_dir / "raw" / "sprites" / category
            output_path.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(samples_per_type), desc=f"{category}/{sprite_type}"):
                try:
                    sprite = generator.generate_category(category, sprite_type)

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
                            "subtype": sprite_type,
                        }
                    )
                    metadata["total_generated"] += 1

                except Exception as e:
                    print(f"\nError: {e}")

    metadata_path = output_dir / "metadata"
    metadata_path.mkdir(exist_ok=True)

    with open(metadata_path / "sprite_generation.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {metadata['total_generated']} sprites")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    generate_all_sprites(args.output_dir, args.samples)
