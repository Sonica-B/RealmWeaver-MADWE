#!/usr/bin/env python3
"""
Texture Generation using SDXL-Turbo
Keeps original texture generation logic unchanged
"""
import os
import sys
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Dict

os.environ["HF_HOME"] = "D:\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface\\transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:\\huggingface\\hub"

try:
    import torch
    from diffusers import AutoPipelineForText2Image
except ImportError as e:
    print(f"Missing: {e}")
    sys.exit(1)

# Force GPU
if not torch.cuda.is_available():
    print("ERROR: GPU not available. This script requires CUDA.")
    sys.exit(1)


class TextureGenerator:
    def __init__(self):
        self.device = "cuda"

        print("Loading SDXL-Turbo for texture generation...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir="D:\\huggingface\\hub",
        ).to(self.device)

        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_tiling()

        # COMPREHENSIVE texture prompts
        self.texture_prompts = {
            # FANTASY RPG TEXTURES
            "fantasy_forest": {
                "ancient_bark": "ultra detailed seamless tileable ancient oak tree bark texture, deeply carved grooves and ridges, intricate lichen patches in emerald green and gray, weathered wood grain with age rings visible, moss growing in crevices, tiny mushrooms sprouting, dappled sunlight creating subtle shadows, photorealistic 4K game texture, PBR material with normal mapping details, high frequency detail preservation",
                "enchanted_leaves": "magical forest canopy texture seamless tileable, overlapping maple oak and birch leaves, bioluminescent veins glowing softly blue and purple, dewdrops catching rainbow light, fairy dust particles, translucent leaves showing intricate vein patterns, autumn colors transitioning from green to gold to crimson, volumetric light rays filtering through, ultra high detail foliage texture",
                "mystical_moss": "fantasy glowing moss texture seamless pattern, bioluminescent spores emitting soft cyan light, velvety micro vegetation, tiny magical flowers blooming, moisture droplets refracting light, phosphorescent patches, deep forest floor coverage, intricate fractal growth patterns, subsurface scattering effect, game ready PBR texture",
            },
            # CYBERPUNK TEXTURES
            "cyberpunk_city": {
                "neon_metal": "futuristic chrome metal panel texture seamless, holographic iridescent surface, scratches revealing circuitry underneath, neon pink and cyan light reflections, rain droplets beading on surface, corporate logos etched in, warning labels in Japanese, rust patches with electrical burns, industrial wear marks, ray traced reflections, 4K game texture",
                "tech_wall": "cyberpunk building facade texture tileable, LED strip lights running through concrete, exposed fiber optic cables glowing data streams, graffiti tags with UV reactive paint, moisture stains from acid rain, ventilation grates with steam, holographic advertisement residue, bullet holes and plasma burns, dystopian urban decay, photorealistic detail",
                "circuit_floor": "high tech circuit board floor texture seamless, multilayer PCB with gold traces, SMD components, cooling gel pools, data flow visualization, quantum processors glowing, diagnostic LEDs blinking patterns, carbon fiber reinforcement mesh, anti-static coating with wear patterns, industrial sci-fi aesthetic",
            },
            # HORROR DARK TEXTURES
            "horror_mansion": {
                "blood_wall": "disturbing mansion wall texture seamless, old wallpaper peeling revealing blood stains underneath, scratch marks from desperate fingers, mold growing in organic patterns, mysterious symbols carved deep, rust from old nails, cobwebs in corners, Victorian damask pattern corrupted, psychological horror atmosphere, desaturated colors with deep reds",
                "cursed_wood": "haunted floorboard texture tileable, aged dark oak with supernatural wear, ghostly footprints burned in, strange liquids seeping between boards, occult symbols carved by unknown hands, wood grain forming screaming faces, splinters that seem to move, creaking under invisible weight, horror game aesthetic",
                "nightmare_fabric": "possessed curtain fabric texture seamless, heavy velvet with moving patterns, faces appearing in the weave, blood seeping through fibers, moth holes forming pentagram shapes, dust particles floating eerily, shadows that don't match light sources, Victorian gothic horror style",
            },
            # CASUAL MOBILE TEXTURES
            "casual_bright": {
                "candy_surface": "cheerful candy shop floor texture seamless tileable, glossy checkered pattern in bubblegum pink and mint green, sugar crystal sparkles, lollipop swirl patterns, gummy bear imprints, sprinkles scattered about, high saturation colors, soft shadows, mobile game optimized, family friendly aesthetic",
                "toy_blocks": "playful wooden toy block texture seamless, primary colors red blue yellow green, soft rounded edges, child safe design, alphabet letters embossed, number stamps, worn paint showing wood grain, finger smudges, wholesome nostalgic feeling, casual game art style",
                "cloud_pattern": "dreamy sky texture tileable, fluffy cumulus clouds in pastel blue sky, rainbow gradients at edges, cartoon style rendering, paper airplane trails, floating bubbles with rainbow reflections, sun rays creating god rays effect, optimized for mobile performance",
            },
            # RETRO ARCADE TEXTURES
            "retro_arcade": {
                "pixel_grid": "authentic arcade cabinet side art texture seamless, 80s neon geometric patterns, scanline effects, CRT phosphor glow simulation, vector graphics inspired designs, chrome gradients, laser grid backgrounds, synthwave color palette hot pink cyan yellow, grain and noise for authenticity",
                "arcade_floor": "retro arcade carpet texture tileable, cosmic bowling alley pattern, blacklight reactive designs, geometric shapes in neon colors, worn pathways from foot traffic, spilled soda stains, gum spots, 90s aesthetic, nostalgic pattern design",
                "cabinet_metal": "arcade machine metal texture seamless, brushed aluminum with finger prints, coin slot wear marks, button indentations, speaker grill patterns, scratches from keys, sticker residue, authentic wear and patina, industrial gaming aesthetic",
            },
            # UNDERWATER OCEAN TEXTURES
            "underwater_depths": {
                "coral_reef": "vibrant coral reef texture seamless tileable, brain coral staghorn and table coral varieties, anemones swaying with current, clownfish hiding spots, barnacles and sea urchins, bioluminescent organisms, caustic light patterns dancing, rich biodiversity, photorealistic underwater photography style",
                "ocean_sand": "deep sea floor sand texture seamless, ripples from underwater currents, seashell fragments, starfish imprints, seaweed debris, light refracting through water creating moving patterns, tiny bubbles rising, marine sediment layers, underwater atmosphere",
                "kelp_forest": "dense kelp forest texture tileable, giant kelp fronds swaying, sunlight filtering creating dappled shadows, small fish darting between stalks, sea otters resting spots, underwater fog effect, rich green and brown tones, dynamic underwater ecosystem",
            },
            # STEAMPUNK TEXTURES
            "steampunk_industrial": {
                "brass_gears": "intricate steampunk brass mechanism texture seamless, interlocking gears and cogs, steam valve handles, pressure gauges with cracked glass, copper pipes with verdigris patina, rivets and bolts, oil stains and grease marks, Victorian industrial aesthetic, mechanical complexity",
                "leather_metal": "steampunk leather and metal texture tileable, aged brown leather with brass studs, buckles and straps, stitching patterns, wear marks from use, metal plates with engraved serial numbers, steam punk goggles imprint, industrial revolution era styling",
                "steam_pipes": "complex pipe system texture seamless, copper and brass pipes interconnected, steam vents releasing pressure, valve wheels, temperature gauges, condensation droplets, rust and patina, industrial maze of plumbing, Victorian engineering aesthetic",
            },
            # POST-APOCALYPTIC TEXTURES
            "post_apocalypse": {
                "rusted_metal": "devastated wasteland metal texture seamless, heavy rust and corrosion, bullet holes and shrapnel damage, makeshift repairs with scrap, radiation warning signs faded, blood stains and burn marks, structural collapse evidence, harsh survival environment",
                "toxic_ground": "contaminated wasteland ground texture tileable, cracked earth with toxic waste seepage, mutated plant growth, radiation crystals forming, debris from civilization, ash and fallout dust, tire tracks from survivor vehicles, desolate atmosphere",
                "bunker_concrete": "nuclear bunker wall texture seamless, reinforced concrete with rebar exposed, blast damage and scorch marks, emergency instructions stenciled, water stains and mold, scratched countdown marks, survival shelter aesthetic",
            },
        }

    def generate_texture(
        self, category: str, texture_type: str, size: Tuple[int, int] = (768, 768)
    ) -> np.ndarray:
        """Generate texture with SDXL-Turbo and color enforcement"""
        prompt = self.texture_prompts.get(category, {}).get(
            texture_type,
            f"high quality seamless {texture_type} texture for {category} game, tileable, detailed",
        )

        # Add color enforcement and style prompts
        enhanced_prompt = f"{prompt}, vibrant colors, full color artwork, digital painting style, highly saturated, rich color palette, professional game texture, photographic quality"

        # Strong negative prompts
        negative_prompt = "black and white, grayscale, monochrome, desaturated, sketch, line art, drawing, simple, low quality, blurry"

        with torch.no_grad():
            image = self.pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=6,  # More steps for quality
                guidance_scale=2.0,  # Small guidance for color
                height=size[1],
                width=size[0],
                generator=torch.Generator(self.device).manual_seed(
                    random.randint(0, 999999)
                ),
            ).images[0]

        # Resize to 512x512 for tileable textures
        image = image.resize((512, 512), Image.Resampling.LANCZOS)

        return np.array(image)


def generate_all_textures(output_dir: Path, samples_per_type: int = 10):
    """Generate all texture categories"""
    output_dir = Path(output_dir)
    generator = TextureGenerator()

    texture_categories = [
        "fantasy_forest",
        "cyberpunk_city",
        "horror_mansion",
        "casual_bright",
        "retro_arcade",
        "underwater_depths",
    ]

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "generator": "SDXL-Turbo Textures",
        "total_generated": 0,
        "assets": [],
    }

    print("\n=== Texture Generation ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Samples per type: {samples_per_type}")
    print("=" * 40)

    for category in texture_categories:
        for texture_type in generator.texture_prompts.get(category, {}).keys():
            output_path = output_dir / "raw" / "textures" / category
            output_path.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(samples_per_type), desc=f"{category}/{texture_type}"):
                try:
                    texture = generator.generate_texture(category, texture_type)

                    img = Image.fromarray(texture.astype(np.uint8))
                    filename = f"{category}_{texture_type}_{i:04d}.png"
                    filepath = output_path / filename
                    img.save(filepath, optimize=True)

                    metadata["assets"].append(
                        {
                            "filename": filename,
                            "filepath": str(filepath.relative_to(output_dir)),
                            "type": "texture",
                            "category": category,
                            "subtype": texture_type,
                        }
                    )
                    metadata["total_generated"] += 1

                except Exception as e:
                    print(f"\nError: {e}")

    metadata_path = output_dir / "metadata"
    metadata_path.mkdir(exist_ok=True)

    with open(metadata_path / "texture_generation.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {metadata['total_generated']} textures")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    generate_all_textures(args.output_dir, args.samples)
