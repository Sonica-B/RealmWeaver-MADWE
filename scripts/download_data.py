#!/usr/bin/env python3
"""
Comprehensive 2D Game Asset Generator using SDXL-Turbo
Generates assets across ALL 2D game genres, themes, and art styles
"""
import os
import sys
import time
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List, Dict

os.environ["HF_HOME"] = "D:\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface\\transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:\\huggingface\\hub"
os.environ["HF_DATASETS_CACHE"] = "D:\\huggingface\\datasets"

try:
    import torch
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
except ImportError as e:
    print(f"Missing: {e}")
    print("Run: pip install torch diffusers transformers accelerate sentencepiece")
    sys.exit(1)


class ComprehensiveGameAssetGenerator:
    """Ultra-detailed prompt generation for all 2D game art styles"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading SDXL-Turbo for ultra-fast generation...")
        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None,
                use_safetensors=True,
                cache_dir="D:\\huggingface\\hub",
            )

            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_tiling()

            print("SDXL-Turbo loaded! 1-4 step generation ready")

        except Exception as e:
            print(f"Error: {e}")
            self.pipe = None

        # COMPREHENSIVE texture prompts with extreme detail
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

        # COMPREHENSIVE sprite prompts with extreme detail
        self.sprite_prompts = {
            # CHARACTER SPRITES - Multiple styles
            "pixel_characters": {
                "8bit_hero": "8-bit pixel art RPG hero sprite, 32x32 pixels, limited NES color palette, sword and shield equipped, idle animation frame, clear pixel boundaries, no anti-aliasing, authentic retro constraints, white background for transparency, game ready sprite sheet format",
                "16bit_mage": "16-bit SNES style wizard sprite, 64x64 pixels, flowing robes with detailed shading, magical staff with glowing orb, spell casting pose, expanded color palette, subtle dithering for gradients, professional pixel art techniques",
                "modern_pixel": "modern indie pixel art character, 128x128 pixels, detailed animations, dynamic lighting effects on pixels, contemporary color theory, sub-pixel animation techniques, mix of retro aesthetic with modern sensibilities",
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

    def generate_asset(
        self, prompt: str, size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """Generate using SDXL-Turbo with 1-4 steps"""
        if self.pipe is None:
            return self._generate_synthetic(size)

        try:
            # SDXL-Turbo optimal settings
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=4,  # 1-4 steps for Turbo
                    guidance_scale=0.0,  # Disabled for Turbo
                    height=size[1],
                    width=size[0],
                    generator=torch.Generator(self.device).manual_seed(
                        random.randint(0, 999999)
                    ),
                ).images[0]

            return np.array(image)

        except Exception as e:
            print(f"Generation error: {e}")
            return self._generate_synthetic(size)

    def generate_texture(
        self, category: str, texture_type: str, size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """Generate texture with detailed prompt"""
        prompt = self.texture_prompts.get(category, {}).get(
            texture_type,
            f"high quality seamless {texture_type} texture for {category} game, tileable, detailed",
        )
        return self.generate_asset(prompt, size)

    def generate_sprite(
        self, category: str, sprite_type: str, size: Tuple[int, int] = (512, 512)
    ) -> np.ndarray:
        """Generate sprite with detailed prompt"""
        prompt = self.sprite_prompts.get(category, {}).get(
            sprite_type,
            f"high quality {sprite_type} sprite for {category}, centered, transparent background",
        )

        img_array = self.generate_asset(prompt, size)

        # Convert to RGBA for sprites
        if img_array.shape[2] == 3:
            rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
            rgba[:, :, :3] = img_array

            # Better alpha extraction
            gray = np.mean(img_array, axis=2)
            # More aggressive alpha for better transparency
            alpha = np.where(gray < 240, 255, 0).astype(np.uint8)

            # Smooth alpha edges
            from scipy.ndimage import gaussian_filter

            alpha = gaussian_filter(alpha.astype(float), sigma=1.0)
            alpha = (alpha * 255).astype(np.uint8)

            rgba[:, :, 3] = alpha
            return rgba

        return img_array

    def _generate_synthetic(self, size: Tuple[int, int]) -> np.ndarray:
        """Fallback synthetic generation"""
        return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def generate_comprehensive_assets(output_dir: Path, samples_per_type: int = 10) -> None:
    """Generate comprehensive game assets across all styles"""
    output_dir = Path(output_dir)
    generator = ComprehensiveGameAssetGenerator()

    # All texture categories to generate
    texture_categories = [
        "fantasy_forest",
        "cyberpunk_city",
        "horror_mansion",
        "casual_bright",
        "retro_arcade",
        "underwater_depths",
        "steampunk_industrial",
        "post_apocalypse",
    ]

    # All sprite categories to generate
    sprite_categories = [
        "pixel_characters",
        "hand_drawn_characters",
        "vector_characters",
        "fantasy_items",
        "sci_fi_items",
        "magical_effects",
        "environmental_effects",
        "game_ui",
    ]

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "generator": "SDXL-Turbo Comprehensive",
        "total_generated": 0,
        "assets": [],
    }

    print("\n=== Comprehensive Game Asset Generation ===")
    print(f"Generating across ALL 2D game styles and genres")
    print(f"Samples per type: {samples_per_type}")
    print(f"Using SDXL-Turbo for 4-step generation")
    print("=" * 50)

    # Generate textures
    for category in texture_categories:
        for texture_type in generator.texture_prompts.get(category, {}).keys():
            output_path = output_dir / "raw" / "textures" / category
            output_path.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(samples_per_type), desc=f"{category}/{texture_type}"):
                try:
                    texture = generator.generate_texture(
                        category, texture_type, (512, 512)
                    )

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
                            "art_style": category.split("_")[0],
                            "size": [512, 512],
                        }
                    )
                    metadata["total_generated"] += 1

                except Exception as e:
                    print(f"\nError: {e}")

    # Generate sprites
    for category in sprite_categories:
        for sprite_type in generator.sprite_prompts.get(category, {}).keys():
            output_path = output_dir / "raw" / "sprites" / category
            output_path.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(samples_per_type), desc=f"{category}/{sprite_type}"):
                try:
                    sprite = generator.generate_sprite(
                        category, sprite_type, (512, 512)
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
                            "subtype": sprite_type,
                            "art_style": category.split("_")[0],
                            "size": [512, 512],
                        }
                    )
                    metadata["total_generated"] += 1

                except Exception as e:
                    print(f"\nError: {e}")

    # Save metadata
    metadata_path = output_dir / "metadata"
    metadata_path.mkdir(exist_ok=True)

    with open(metadata_path / "comprehensive_generation.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Generation Complete ===")
    print(f"Total assets generated: {metadata['total_generated']}")
    print(f"Texture categories: {len(texture_categories)}")
    print(f"Sprite categories: {len(sprite_categories)}")
    print(f"Output directory: {output_dir}")

    # Summary statistics
    print("\n=== Asset Breakdown ===")
    for cat in texture_categories:
        count = len([a for a in metadata["assets"] if a["category"] == cat])
        print(f"{cat}: {count} textures")
    for cat in sprite_categories:
        count = len([a for a in metadata["assets"] if a["category"] == cat])
        print(f"{cat}: {count} sprites")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive 2D game assets"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument(
        "--samples", type=int, default=10, help="Samples per asset type"
    )

    args = parser.parse_args()

    # Check environment
    print("=== SDXL-Turbo Comprehensive Asset Generator ===")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU: Not available, using CPU")

    # Verify scipy for alpha smoothing
    try:
        import scipy
    except ImportError:
        print("\nInstalling scipy for better alpha channels...")
        os.system("pip install scipy")

    generate_comprehensive_assets(args.output_dir, samples_per_type=args.samples)


if __name__ == "__main__":
    main()
