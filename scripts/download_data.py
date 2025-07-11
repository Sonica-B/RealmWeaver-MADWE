"""
Download and generate synthetic game assets - Python 3.11 compatible
Fully dynamic asset generation with comprehensive game coverage
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import random


class DynamicGameAssetGenerator:
    """Generate game assets dynamically based on discovered categories"""

    def __init__(self, config_path: Path | str | None = None):
        # Load from config or use defaults
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = self._get_default_config()

        self.categories = config["categories"]
        self.asset_structure = config["asset_structure"]

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default comprehensive configuration"""
        return {
            "categories": {
                "textures": {
                    "genres": {
                        "platformer": [
                            "side_scrolling",
                            "metroidvania",
                            "puzzle_platformer",
                            "auto_runner",
                        ],
                        "rpg": [
                            "action_rpg",
                            "turn_based",
                            "roguelike",
                            "jrpg",
                            "tactical",
                        ],
                        "action": [
                            "beat_em_up",
                            "hack_n_slash",
                            "run_n_gun",
                            "twin_stick",
                        ],
                        "puzzle": ["match_3", "physics", "logic", "tile_matching"],
                        "shooter": [
                            "bullet_hell",
                            "side_scrolling",
                            "gallery",
                            "shmup",
                        ],
                        "fighting": ["2d_fighter", "arena_brawler", "combo_fighter"],
                        "adventure": ["point_click", "visual_novel", "text_adventure"],
                        "strategy": ["rts", "turn_based", "tower_defense", "4x"],
                        "simulation": ["life_sim", "farming", "city_builder", "tycoon"],
                        "sports": ["arcade", "realistic", "extreme"],
                        "racing": ["arcade", "kart", "time_trial"],
                        "idle": ["clicker", "incremental", "automation"],
                        "card": ["collectible", "deckbuilder", "solitaire"],
                        "rhythm": ["music", "dance", "beat_matching"],
                        "educational": ["math", "language", "science", "puzzle"],
                    },
                    "styles": {
                        "pixel_art": [
                            "8bit",
                            "16bit",
                            "32bit",
                            "modern_pixel",
                            "micro",
                        ],
                        "vector": ["flat", "gradient", "geometric", "minimalist"],
                        "hand_drawn": [
                            "sketch",
                            "ink",
                            "watercolor",
                            "chalk",
                            "pencil",
                        ],
                        "cel_shaded": ["anime", "cartoon", "comic", "outlined"],
                        "realistic": ["photorealistic", "semi_realistic", "painterly"],
                        "abstract": ["geometric", "fluid", "surreal", "glitch"],
                        "retro": ["synthwave", "vaporwave", "art_deco", "vintage"],
                        "gothic": ["dark", "victorian", "medieval", "horror"],
                        "isometric": ["pixel_iso", "vector_iso", "detailed_iso"],
                        "low_poly": ["faceted", "angular", "crystalline"],
                        "monochrome": ["silhouette", "noir", "minimalist"],
                        "stylized": ["exaggerated", "whimsical", "fantastical"],
                    },
                    "themes": {
                        "fantasy": ["medieval", "magical", "mythical", "fairy_tale"],
                        "sci_fi": ["cyberpunk", "space", "futuristic", "alien"],
                        "horror": ["gothic", "psychological", "cosmic", "zombie"],
                        "nature": ["forest", "ocean", "desert", "arctic", "jungle"],
                        "urban": ["modern_city", "dystopian", "underground", "neon"],
                        "historical": [
                            "ancient",
                            "medieval",
                            "renaissance",
                            "industrial",
                        ],
                        "abstract": ["geometric", "surreal", "psychedelic", "void"],
                        "steampunk": ["mechanical", "victorian_tech", "brass", "gears"],
                        "post_apocalyptic": [
                            "wasteland",
                            "ruins",
                            "survivor",
                            "mutant",
                        ],
                    },
                },
                "sprites": {
                    "types": {
                        "characters": [
                            "hero",
                            "enemy",
                            "npc",
                            "boss",
                            "companion",
                            "mount",
                        ],
                        "items": [
                            "weapon",
                            "armor",
                            "consumable",
                            "key",
                            "treasure",
                            "tool",
                        ],
                        "effects": [
                            "explosion",
                            "magic",
                            "particle",
                            "weather",
                            "impact",
                        ],
                        "ui": ["button", "icon", "frame", "bar", "cursor", "popup"],
                        "environment": [
                            "prop",
                            "decoration",
                            "hazard",
                            "interactive",
                            "background",
                        ],
                    },
                    "animations": {
                        "idle": ["breathing", "floating", "swaying", "pulsing"],
                        "movement": ["walk", "run", "jump", "climb", "swim", "fly"],
                        "combat": ["attack", "defend", "dodge", "special", "death"],
                        "interaction": ["use", "pickup", "activate", "talk"],
                        "emotion": ["happy", "sad", "angry", "surprised", "thinking"],
                    },
                },
                "gameplay": {
                    "mechanics": {
                        "platforming": ["jumping", "wall_jump", "double_jump", "dash"],
                        "combat": ["melee", "ranged", "magic", "combo"],
                        "puzzle": ["switch", "push_block", "key_door", "pattern"],
                        "collection": ["coins", "powerups", "secrets", "achievements"],
                        "progression": [
                            "experience",
                            "skill_tree",
                            "upgrades",
                            "unlocks",
                        ],
                    },
                    "level_elements": {
                        "terrain": ["solid", "platform", "slope", "destructible"],
                        "hazards": ["spikes", "lava", "enemy", "trap"],
                        "interactive": ["door", "switch", "portal", "checkpoint"],
                        "decorative": [
                            "background",
                            "foreground",
                            "particle",
                            "lighting",
                        ],
                    },
                },
            },
            "asset_structure": {
                "textures": {
                    "pattern": "seamless tiling texture",
                    "elements": [
                        "material properties",
                        "surface detail",
                        "color variation",
                        "environmental wear",
                    ],
                },
                "sprites": {
                    "pattern": "game-ready sprite asset",
                    "elements": [
                        "clear silhouette",
                        "readable at small size",
                        "animation-friendly design",
                    ],
                },
                "gameplay": {
                    "pattern": "functional game element",
                    "elements": [
                        "visual clarity",
                        "gameplay purpose",
                        "feedback indication",
                    ],
                },
            },
        }

    def generate_dynamic_prompt(self, asset_type: str, metadata: Dict[str, str]) -> str:
        """Generate detailed prompt based on dynamic metadata"""
        prompt_parts = []

        # Base description
        base_pattern = self.asset_structure.get(asset_type, {}).get(
            "pattern", "game asset"
        )
        prompt_parts.append(f"Professional {base_pattern}")

        # Add all metadata attributes
        for key, value in metadata.items():
            if key != "asset_type":
                # Convert underscores to spaces and format nicely
                formatted_key = key.replace("_", " ")
                formatted_value = value.replace("_", " ")
                prompt_parts.append(f"{formatted_key}: {formatted_value}")

        # Add asset-specific elements
        elements = self.asset_structure.get(asset_type, {}).get("elements", [])
        if elements:
            prompt_parts.append(f"featuring {', '.join(elements)}")

        # Technical requirements
        prompt_parts.extend(
            [
                "production quality",
                "game-ready asset",
                "optimized for real-time rendering",
                "consistent art direction",
                "professional polish",
            ]
        )

        return ", ".join(prompt_parts)

    def discover_categories(self, base_path: Path) -> Dict[str, List[str]]:
        """Dynamically discover existing categories from directory structure"""
        categories = {}

        for asset_type in ["textures", "sprites", "gameplay"]:
            type_path = base_path / asset_type
            if type_path.exists():
                # Get all subdirectories
                subdirs = [d.name for d in type_path.iterdir() if d.is_dir()]
                if subdirs:
                    categories[asset_type] = subdirs

        return categories

    def create_asset_image(
        self,
        asset_type: str,
        metadata: Dict[str, Any],
        size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """Create asset based on type and metadata"""
        # Generate base color from metadata hash
        metadata_str = json.dumps(metadata, sort_keys=True)
        color_seed = hash(metadata_str) % 1000000
        np.random.seed(color_seed)

        base_color = (
            np.random.randint(50, 206),
            np.random.randint(50, 206),
            np.random.randint(50, 206),
        )

        img_array = np.zeros((*size, 3), dtype=np.uint8)

        if asset_type == "textures":
            # Create tileable texture pattern
            self._generate_texture_pattern(img_array, base_color, metadata)
        elif asset_type == "sprites":
            # Create sprite with clear shape
            self._generate_sprite_pattern(img_array, base_color, metadata)
        elif asset_type == "gameplay":
            # Create gameplay element
            self._generate_gameplay_pattern(img_array, base_color, metadata)
        else:
            # Default pattern
            img_array[:] = base_color

        return Image.fromarray(img_array)

    def _generate_texture_pattern(
        self,
        img_array: np.ndarray,
        base_color: Tuple[int, int, int],
        metadata: Dict[str, Any],
    ) -> None:
        """Generate texture-specific patterns"""
        height, width = img_array.shape[:2]

        # Different patterns based on metadata
        if "pixel" in str(metadata.get("style", "")):
            # Pixel pattern
            block_size = 16
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    variation = np.random.randint(-30, 30)
                    color = np.clip(np.array(base_color) + variation, 0, 255)
                    img_array[i : i + block_size, j : j + block_size] = color
        else:
            # Smooth gradient pattern
            for i in range(height):
                for j in range(width):
                    # Create seamless tiling
                    u = i / height * 2 * np.pi
                    v = j / width * 2 * np.pi

                    noise = (np.sin(u) * np.cos(v) + 1) / 2
                    variation = int(noise * 60 - 30)

                    img_array[i, j] = np.clip(np.array(base_color) + variation, 0, 255)

    def _generate_sprite_pattern(
        self,
        img_array: np.ndarray,
        base_color: Tuple[int, int, int],
        metadata: Dict[str, Any],
    ) -> None:
        """Generate sprite-specific patterns"""
        height, width = img_array.shape[:2]
        center_x, center_y = width // 2, height // 2

        # White/transparent background
        img_array[:] = (255, 255, 255)

        # Create sprite shape based on type
        sprite_type = metadata.get("type", "character")

        if "character" in sprite_type:
            # Character silhouette
            # Body
            body_h = height // 3
            body_w = width // 4
            y1 = center_y - body_h // 2
            y2 = center_y + body_h // 2
            x1 = center_x - body_w // 2
            x2 = center_x + body_w // 2
            img_array[y1:y2, x1:x2] = base_color

            # Head
            head_r = width // 8
            y_coords, x_coords = np.ogrid[:height, :width]
            head_mask = (x_coords - center_x) ** 2 + (
                y_coords - (center_y - body_h // 2 - head_r)
            ) ** 2 <= head_r**2
            img_array[head_mask] = np.clip(np.array(base_color) * 0.9, 0, 255)

        elif "item" in sprite_type:
            # Item shape (diamond/gem)
            for i in range(height):
                for j in range(width):
                    if abs(i - center_y) + abs(j - center_x) < min(height, width) // 3:
                        distance = abs(i - center_y) + abs(j - center_x)
                        fade = 1.0 - (distance / (min(height, width) // 3))
                        img_array[i, j] = np.clip(
                            np.array(base_color) * (0.7 + 0.3 * fade), 0, 255
                        )
        else:
            # Generic circular sprite
            radius = min(height, width) // 3
            y_coords, x_coords = np.ogrid[:height, :width]
            mask = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 <= radius**2
            img_array[mask] = base_color

    def _generate_gameplay_pattern(
        self,
        img_array: np.ndarray,
        base_color: Tuple[int, int, int],
        metadata: Dict[str, Any],
    ) -> None:
        """Generate gameplay element patterns"""
        height, width = img_array.shape[:2]

        # Create functional-looking elements
        mechanic = metadata.get("mechanic", "platform")

        if "platform" in mechanic:
            # Platform tile
            img_array[:] = base_color
            # Add edge highlights
            img_array[:5, :] = np.clip(np.array(base_color) * 1.3, 0, 255)
            img_array[-5:, :] = np.clip(np.array(base_color) * 0.7, 0, 255)
        elif "hazard" in mechanic:
            # Hazard pattern (spikes)
            img_array[:] = (200, 200, 200)  # Gray base
            # Create triangle patterns
            for x in range(0, width, width // 8):
                for y in range(height // 2, height):
                    if (y - height // 2) < (
                        width // 16 - abs(x + width // 16 - (x % (width // 8)))
                    ):
                        img_array[y, x] = base_color
        else:
            # Generic gameplay element
            img_array[:] = base_color
            # Add some detail
            img_array[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = (
                np.clip(np.array(base_color) * 0.8, 0, 255)
            )

    def generate_comprehensive_assets(
        self, output_dir: Path | str, samples_per_category: int = 10
    ) -> None:
        """Generate assets based on dynamic categories"""
        output_dir = Path(output_dir)
        raw_dir = output_dir / "raw"

        print("Generating comprehensive game assets...")

        metadata_all = {
            "generated_at": datetime.now().isoformat(),
            "structure": "raw/[asset_type]/[category_combinations]/",
            "assets": [],
        }

        # Process each asset type
        for asset_type, type_categories in self.categories.items():
            print(f"\nGenerating {asset_type}...")

            # Create all combinations for this asset type
            combinations = self._generate_combinations(type_categories)

            with tqdm(
                total=len(combinations) * samples_per_category, desc=f"{asset_type}"
            ) as pbar:

                for combo in combinations:
                    # Create directory
                    combo_path = "_".join(f"{k}_{v}" for k, v in combo.items())
                    asset_dir = raw_dir / asset_type / combo_path
                    asset_dir.mkdir(parents=True, exist_ok=True)

                    # Generate samples
                    for i in range(samples_per_category):
                        # Generate prompt
                        prompt = self.generate_dynamic_prompt(asset_type, combo)

                        # Create image
                        img = self.create_asset_image(asset_type, combo)

                        # Save
                        filename = f"{asset_type}_{combo_path}_{i:04d}.png"
                        filepath = asset_dir / filename
                        img.save(filepath, optimize=True)

                        # Store metadata
                        metadata_all["assets"].append(
                            {
                                "filename": filename,
                                "path": str(filepath.relative_to(output_dir)),
                                "asset_type": asset_type,
                                "metadata": combo,
                                "prompt": prompt,
                            }
                        )

                        pbar.update(1)

        # Save metadata
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        with open(metadata_dir / "generated_assets.json", "w") as f:
            json.dump(metadata_all, f, indent=2)

        print(f"\nGenerated {len(metadata_all['assets'])} total assets")
        print(f"Metadata saved to {metadata_dir / 'generated_assets.json'}")

    def _generate_combinations(
        self, categories: Dict[str, List[str]]
    ) -> List[Dict[str, str]]:
        """Generate all valid combinations from categories"""
        combinations = []

        # Get category names and their values
        cat_names = list(categories.keys())
        cat_values = [categories[name] for name in cat_names]

        # Generate cartesian product
        import itertools

        for combo in itertools.product(*cat_values):
            combo_dict = {cat_names[i]: combo[i] for i in range(len(cat_names))}
            combinations.append(combo_dict)

        # Limit combinations if too many
        if len(combinations) > 100:
            # Sample subset
            combinations = random.sample(combinations, 100)

        return combinations


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate game assets for MADWE")
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument("--samples", type=int, default=10, help="Samples per category")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode with fewer samples"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = DynamicGameAssetGenerator(args.config)

    # Quick mode adjustments
    if args.quick:
        args.samples = 2
        # Reduce categories for quick testing
        for asset_type in generator.categories:
            for cat in generator.categories[asset_type]:
                if isinstance(generator.categories[asset_type][cat], list):
                    generator.categories[asset_type][cat] = generator.categories[
                        asset_type
                    ][cat][:2]

    # Generate assets
    generator.generate_comprehensive_assets(args.output_dir, args.samples)


if __name__ == "__main__":
    main()
