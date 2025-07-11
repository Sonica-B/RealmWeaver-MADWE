"""
Download and generate synthetic game assets - Python 3.11 compatible
Fully dynamic asset generation with comprehensive game coverage
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import random
from scipy.ndimage import gaussian_filter


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
                    "character_types": ["hero", "enemy", "npc", "boss"],
                    "item_types": ["weapon", "armor", "consumable", "powerup"],
                    "effect_types": ["explosion", "magic", "particle", "impact"],
                    "animations": ["idle", "walk", "run", "jump", "attack"],
                },
                "gameplay": {
                    "element_types": [
                        "platform",
                        "hazard",
                        "collectible",
                        "interactive",
                    ],
                    "mechanics": ["jumping", "combat", "puzzle", "exploration"],
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
        # Generate unique seed for each image
        metadata_str = json.dumps(metadata, sort_keys=True) + str(random.random())
        color_seed = hash(metadata_str) % 1000000
        np.random.seed(color_seed)
        random.seed(color_seed)

        # More varied base colors
        color_schemes = [
            # Dark schemes
            [(20, 20, 30), (40, 40, 60), (60, 60, 90)],
            [(30, 20, 20), (60, 40, 40), (90, 60, 60)],
            [(20, 30, 20), (40, 60, 40), (60, 90, 60)],
            # Bright schemes
            [(100, 50, 50), (150, 75, 75), (200, 100, 100)],
            [(50, 100, 50), (75, 150, 75), (100, 200, 100)],
            [(50, 50, 100), (75, 75, 150), (100, 100, 200)],
            # Neutral schemes
            [(80, 80, 80), (120, 120, 120), (160, 160, 160)],
            [(100, 90, 80), (140, 120, 100), (180, 150, 120)],
        ]

        scheme = random.choice(color_schemes)
        base_color = scheme[0]
        accent_color = scheme[1]
        highlight_color = scheme[2]

        img_array = np.zeros((*size, 3), dtype=np.uint8)

        if asset_type == "textures":
            # Create tileable texture pattern
            self._generate_texture_pattern(
                img_array, base_color, accent_color, highlight_color, metadata
            )
        elif asset_type == "sprites":
            # Create sprite with clear shape
            self._generate_sprite_pattern(
                img_array, base_color, accent_color, highlight_color, metadata
            )
        elif asset_type == "gameplay":
            # Create gameplay element
            self._generate_gameplay_pattern(
                img_array, base_color, accent_color, highlight_color, metadata
            )
        else:
            # Default pattern
            img_array[:] = base_color

        # Add post-processing effects randomly
        if random.random() > 0.5:
            # Add noise
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def _generate_texture_pattern(
        self,
        img_array: np.ndarray,
        base_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int],
        highlight_color: Tuple[int, int, int],
        metadata: Dict[str, Any],
    ) -> None:
        """Generate texture-specific patterns with variety"""
        height, width = img_array.shape[:2]

        # Base fill with variation
        img_array[:] = base_color

        # Add base texture
        texture_type = random.choice(["noise", "gradient", "pattern"])
        if texture_type == "noise":
            noise = np.random.normal(0, 20, (height, width, 3))
            img_array[:] = np.clip(img_array + noise, 0, 255)

        # Different patterns based on metadata
        style = str(metadata.get("styles", "")).lower()
        genre = str(metadata.get("genres", "")).lower()

        pattern_choice = random.randint(0, 3)

        if "pixel" in style:
            # Varied pixel patterns
            block_sizes = [8, 16, 32]
            block_size = random.choice(block_sizes)

            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    if pattern_choice == 0:
                        # Checkerboard with variation
                        if (i // block_size + j // block_size) % 2 == 0:
                            color = random.choice([base_color, accent_color])
                        else:
                            color = random.choice([accent_color, highlight_color])
                        # Add random variation
                        variation = np.random.randint(-20, 20, 3)
                        color = np.clip(np.array(color) + variation, 0, 255)
                    elif pattern_choice == 1:
                        # Random pattern
                        if random.random() > 0.5:
                            color = np.clip(
                                np.array(base_color) + np.random.randint(-40, 40, 3),
                                0,
                                255,
                            )
                        else:
                            color = accent_color
                    else:
                        # Gradient blocks
                        progress = (i + j) / (height + width)
                        color = (
                            np.array(base_color) * (1 - progress)
                            + np.array(accent_color) * progress
                        )
                        color = np.clip(color + np.random.randint(-15, 15, 3), 0, 255)

                    img_array[i : i + block_size, j : j + block_size] = color

        elif "vector" in style or "flat" in style:
            # Varied geometric patterns
            if pattern_choice == 0:
                # Random rectangles
                num_shapes = np.random.randint(10, 25)
                for _ in range(num_shapes):
                    x1 = np.random.randint(0, width - 20)
                    y1 = np.random.randint(0, height - 20)
                    w = np.random.randint(20, min(100, width - x1))
                    h = np.random.randint(20, min(100, height - y1))
                    color = random.choice([base_color, accent_color, highlight_color])
                    color = np.clip(
                        np.array(color) + np.random.randint(-30, 30, 3), 0, 255
                    )
                    img_array[y1 : y1 + h, x1 : x1 + w] = color
            elif pattern_choice == 1:
                # Stripes with variation
                stripe_width = random.randint(20, 50)
                vertical = random.choice([True, False])
                for i in range(0, width if vertical else height, stripe_width):
                    color = accent_color if (i // stripe_width) % 2 == 0 else base_color
                    color = np.clip(
                        np.array(color) + np.random.randint(-20, 20, 3), 0, 255
                    )
                    if vertical:
                        img_array[:, i : i + stripe_width] = color
                    else:
                        img_array[i : i + stripe_width, :] = color
            else:
                # Circles/dots pattern
                num_circles = random.randint(15, 40)
                for _ in range(num_circles):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    radius = np.random.randint(10, 40)
                    y_coords, x_coords = np.ogrid[:height, :width]
                    mask = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius**2
                    color = random.choice([accent_color, highlight_color])
                    color = np.clip(
                        np.array(color) + np.random.randint(-30, 30, 3), 0, 255
                    )
                    img_array[mask] = color

        elif "realistic" in style or "hand" in style:
            # Organic varied textures
            # Multiple layers of noise
            for _ in range(random.randint(2, 4)):
                scale = random.choice([2, 4, 8])
                noise = np.random.randn(
                    height // scale, width // scale, 3
                ) * random.randint(15, 35)
                noise = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)
                noise = gaussian_filter(noise, sigma=random.uniform(1, 4))
                img_array[:] = np.clip(img_array + noise, 0, 255)

            # Add varied detail spots
            num_spots = np.random.randint(5, 20)
            for _ in range(num_spots):
                x, y = np.random.randint(0, width), np.random.randint(0, height)
                radius = np.random.randint(10, 50)
                y_coords, x_coords = np.ogrid[:height, :width]
                distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
                mask = distances <= radius
                # Soft edges
                alpha = np.clip(1 - distances / radius, 0, 1)
                color = random.choice([accent_color, highlight_color])
                variation = np.random.randint(-40, 40, 3)
                spot_color = np.clip(np.array(color) + variation, 0, 255)
                for c in range(3):
                    img_array[:, :, c][mask] = (
                        alpha[mask] * spot_color[c]
                        + (1 - alpha[mask]) * img_array[:, :, c][mask]
                    )
        else:
            # Default: Varied tile patterns
            tile_sizes = [32, 48, 64]
            tile_size = random.choice(tile_sizes)

            for i in range(0, height, tile_size):
                for j in range(0, width, tile_size):
                    # Each tile is unique
                    tile_pattern = random.randint(0, 3)

                    if tile_pattern == 0:
                        # Solid with variation
                        tile_color = np.clip(
                            np.array(base_color) + np.random.randint(-30, 30, 3), 0, 255
                        )
                        img_array[i : i + tile_size, j : j + tile_size] = tile_color
                    elif tile_pattern == 1:
                        # Gradient tile
                        for y in range(tile_size):
                            progress = y / tile_size
                            color = (
                                np.array(base_color) * (1 - progress)
                                + np.array(accent_color) * progress
                            )
                            img_array[i + y : i + y + 1, j : j + tile_size] = color
                    elif tile_pattern == 2:
                        # Diagonal pattern
                        for y in range(tile_size):
                            for x in range(tile_size):
                                if (x + y) % (tile_size // 4) < tile_size // 8:
                                    img_array[i + y, j + x] = accent_color
                                else:
                                    img_array[i + y, j + x] = base_color
                    else:
                        # Center highlight
                        img_array[i : i + tile_size, j : j + tile_size] = base_color
                        center_size = tile_size // 3
                        center_start = tile_size // 3
                        img_array[
                            i + center_start : i + center_start + center_size,
                            j + center_start : j + center_start + center_size,
                        ] = accent_color

                    # Always add borders with variation
                    border_width = random.randint(1, 3)
                    border_color = np.clip(
                        np.array(base_color) * random.uniform(0.6, 0.8), 0, 255
                    )
                    img_array[i : i + border_width, j : j + tile_size] = border_color
                    img_array[
                        i + tile_size - border_width : i + tile_size, j : j + tile_size
                    ] = border_color
                    img_array[i : i + tile_size, j : j + border_width] = border_color
                    img_array[
                        i : i + tile_size, j + tile_size - border_width : j + tile_size
                    ] = border_color

    def _generate_sprite_pattern(
        self,
        img_array: np.ndarray,
        base_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int],
        highlight_color: Tuple[int, int, int],
        metadata: Dict[str, Any],
    ) -> None:
        """Generate sprite-specific patterns"""
        height, width = img_array.shape[:2]
        center_x, center_y = width // 2, height // 2

        # White/transparent background
        img_array[:] = (255, 255, 255)

        # Get sprite type from metadata
        sprite_type = ""
        for key in ["character_types", "item_types", "effect_types", "types"]:
            if key in metadata:
                sprite_type = str(metadata[key])
                break

        if (
            "character" in sprite_type
            or "hero" in sprite_type
            or "enemy" in sprite_type
        ):
            # Character sprite with more detail
            # Body
            body_h = height // 3
            body_w = width // 4
            y1 = center_y - body_h // 2
            y2 = center_y + body_h // 2
            x1 = center_x - body_w // 2
            x2 = center_x + body_w // 2

            # Main body
            img_array[y1:y2, x1:x2] = base_color

            # Add shading
            for i in range(body_w // 4):
                shade_color = np.clip(
                    np.array(base_color) * (0.8 + i * 0.2 / (body_w // 4)), 0, 255
                )
                img_array[y1:y2, x1 + i : x1 + i + 1] = shade_color
                img_array[y1:y2, x2 - i - 1 : x2 - i] = shade_color

            # Head
            head_r = width // 8
            y_coords, x_coords = np.ogrid[:height, :width]
            head_center_y = center_y - body_h // 2 - head_r
            head_mask = (x_coords - center_x) ** 2 + (
                y_coords - head_center_y
            ) ** 2 <= head_r**2
            img_array[head_mask] = np.clip(np.array(base_color) * 0.9, 0, 255)

            # Arms (simple rectangles)
            arm_w = body_w // 4
            arm_h = body_h // 2
            # Left arm
            img_array[y1 : y1 + arm_h, x1 - arm_w : x1] = np.clip(
                np.array(base_color) * 0.85, 0, 255
            )
            # Right arm
            img_array[y1 : y1 + arm_h, x2 : x2 + arm_w] = np.clip(
                np.array(base_color) * 0.85, 0, 255
            )

            # Legs
            leg_w = body_w // 3
            leg_h = body_h // 2
            # Left leg
            img_array[y2 : y2 + leg_h, center_x - leg_w - 5 : center_x - 5] = np.clip(
                np.array(base_color) * 0.8, 0, 255
            )
            # Right leg
            img_array[y2 : y2 + leg_h, center_x + 5 : center_x + leg_w + 5] = np.clip(
                np.array(base_color) * 0.8, 0, 255
            )

        elif "item" in sprite_type or "weapon" in sprite_type or "armor" in sprite_type:
            # Item sprite (gem/equipment style)
            if "weapon" in sprite_type:
                # Sword-like shape
                blade_w = width // 8
                blade_h = height // 2
                handle_w = width // 6
                handle_h = height // 6

                # Blade
                blade_x1 = center_x - blade_w // 2
                blade_x2 = center_x + blade_w // 2
                blade_y1 = center_y - blade_h
                blade_y2 = center_y

                for y in range(blade_y1, blade_y2):
                    progress = (y - blade_y1) / blade_h
                    current_w = int(blade_w * (0.2 + 0.8 * progress))
                    x1 = center_x - current_w // 2
                    x2 = center_x + current_w // 2

                    # Metallic gradient
                    center_intensity = 1.0 - abs(2 * progress - 1)
                    blade_color = np.array([180, 180, 200]) * (
                        0.7 + 0.3 * center_intensity
                    )
                    img_array[y, x1:x2] = blade_color

                # Handle
                handle_x1 = center_x - handle_w // 2
                handle_x2 = center_x + handle_w // 2
                img_array[blade_y2 : blade_y2 + handle_h, handle_x1:handle_x2] = (
                    base_color
                )

                # Guard
                guard_w = width // 3
                guard_h = height // 20
                guard_x1 = center_x - guard_w // 2
                guard_x2 = center_x + guard_w // 2
                img_array[
                    blade_y2 - guard_h // 2 : blade_y2 + guard_h // 2, guard_x1:guard_x2
                ] = np.clip(np.array(base_color) * 0.8, 0, 255)

            else:
                # Gem/crystal shape
                vertices = 6  # Hexagonal gem
                gem_radius = min(width, height) // 3

                for angle_idx in range(vertices):
                    angle1 = angle_idx * 2 * np.pi / vertices
                    angle2 = (angle_idx + 1) * 2 * np.pi / vertices

                    # Create triangular facets
                    for r in range(gem_radius):
                        for theta in np.linspace(angle1, angle2, max(2, r)):
                            x = int(center_x + r * np.cos(theta))
                            y = int(center_y + r * np.sin(theta))

                            if 0 <= x < width and 0 <= y < height:
                                # Facet shading
                                facet_intensity = 0.6 + 0.4 * np.sin(angle_idx)
                                gem_color = np.array(base_color) * facet_intensity
                                img_array[y, x] = np.clip(gem_color, 0, 255)

        elif (
            "effect" in sprite_type
            or "explosion" in sprite_type
            or "magic" in sprite_type
        ):
            # Particle effect sprite
            num_particles = np.random.randint(15, 30)

            for _ in range(num_particles):
                # Random particle position
                angle = np.random.random() * 2 * np.pi
                distance = np.random.random() * min(width, height) // 3
                x = int(center_x + distance * np.cos(angle))
                y = int(center_y + distance * np.sin(angle))

                # Particle size based on distance
                particle_size = max(
                    2, int(10 * (1 - distance / (min(width, height) // 3)))
                )

                # Draw particle with glow
                for dy in range(-particle_size, particle_size + 1):
                    for dx in range(-particle_size, particle_size + 1):
                        px, py = x + dx, y + dy
                        if 0 <= px < width and 0 <= py < height:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= particle_size:
                                intensity = 1.0 - (dist / particle_size)
                                # Glowing effect
                                glow_color = (
                                    np.array(base_color)
                                    + np.array([50, 50, 0]) * intensity
                                )
                                current_color = img_array[py, px]
                                if not np.array_equal(current_color, [255, 255, 255]):
                                    # Additive blending
                                    img_array[py, px] = np.clip(
                                        current_color + glow_color * intensity * 0.5,
                                        0,
                                        255,
                                    )
                                else:
                                    img_array[py, px] = np.clip(glow_color, 0, 255)
        else:
            # Generic circular sprite with shading
            radius = min(height, width) // 3
            y_coords, x_coords = np.ogrid[:height, :width]

            for r in range(radius, 0, -1):
                mask = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2 <= r**2
                # Create gradient from center to edge
                shade_factor = r / radius
                shaded_color = np.array(base_color) * (0.5 + 0.5 * shade_factor)
                img_array[mask] = np.clip(shaded_color, 0, 255)

    def _generate_gameplay_pattern(
        self,
        img_array: np.ndarray,
        base_color: Tuple[int, int, int],
        accent_color: Tuple[int, int, int],
        highlight_color: Tuple[int, int, int],
        metadata: Dict[str, Any],
    ) -> None:
        """Generate gameplay element patterns with high variety"""
        height, width = img_array.shape[:2]

        # Create functional-looking elements
        element_type = metadata.get(
            "element_types", metadata.get("mechanics", "platform")
        )

        if "platform" in str(element_type):
            # Varied platform tiles
            platform_style = random.randint(0, 3)

            if platform_style == 0:
                # Stone/brick pattern
                img_array[:] = base_color
                brick_height = random.randint(20, 40)
                brick_width = random.randint(40, 80)

                for i in range(0, height, brick_height):
                    offset = (brick_width // 2) if (i // brick_height) % 2 == 1 else 0
                    for j in range(-brick_width, width + brick_width, brick_width):
                        x_start = j + offset
                        x_end = min(x_start + brick_width, width)
                        if x_start < width and x_end > 0:
                            # Each brick has unique color
                            brick_color = np.clip(
                                np.array(base_color) + np.random.randint(-25, 25, 3),
                                0,
                                255,
                            )
                            img_array[
                                i : min(i + brick_height, height),
                                max(0, x_start) : x_end,
                            ] = brick_color

                            # Mortar lines
                            mortar_color = np.array(base_color) * 0.6
                            if i + brick_height < height:
                                img_array[
                                    i + brick_height - 2 : i + brick_height,
                                    max(0, x_start) : x_end,
                                ] = mortar_color
                            if x_end < width:
                                img_array[
                                    i : min(i + brick_height, height), x_end - 2 : x_end
                                ] = mortar_color

            elif platform_style == 1:
                # Wood plank pattern
                img_array[:] = [101, 67, 33]  # Brown base
                plank_width = random.randint(30, 60)

                for j in range(0, width, plank_width):
                    plank_color = np.clip(
                        np.array([101, 67, 33]) + np.random.randint(-20, 20, 3), 0, 255
                    )
                    img_array[:, j : min(j + plank_width, width)] = plank_color

                    # Wood grain
                    for i in range(0, height, 3):
                        if random.random() > 0.7:
                            grain_color = np.clip(
                                plank_color * random.uniform(0.8, 1.2), 0, 255
                            )
                            img_array[i : i + 1, j : min(j + plank_width, width)] = (
                                grain_color
                            )

                    # Plank gaps
                    if j + plank_width < width:
                        img_array[:, j + plank_width - 2 : j + plank_width] = [
                            51,
                            34,
                            17,
                        ]

            elif platform_style == 2:
                # Metal/tech pattern
                img_array[:] = [100, 100, 120]  # Metal base

                # Add panels
                panel_size = random.randint(40, 80)
                for i in range(0, height, panel_size):
                    for j in range(0, width, panel_size):
                        # Panel with rivets
                        panel_color = np.clip(
                            np.array([100, 100, 120]) + np.random.randint(-15, 15, 3),
                            0,
                            255,
                        )
                        img_array[
                            i : min(i + panel_size, height),
                            j : min(j + panel_size, width),
                        ] = panel_color

                        # Rivets at corners
                        rivet_positions = [
                            (5, 5),
                            (5, panel_size - 5),
                            (panel_size - 5, 5),
                            (panel_size - 5, panel_size - 5),
                        ]
                        for ry, rx in rivet_positions:
                            if i + ry < height and j + rx < width:
                                for dy in range(-2, 3):
                                    for dx in range(-2, 3):
                                        if (dx * dx + dy * dy) <= 4:
                                            img_array[i + ry + dy, j + rx + dx] = [
                                                60,
                                                60,
                                                80,
                                            ]
            else:
                # Natural/dirt pattern
                img_array[:] = [92, 64, 51]  # Dirt base

                # Add texture
                for _ in range(100):
                    x = random.randint(0, width - 1)
                    y = random.randint(0, height - 1)
                    size = random.randint(2, 8)
                    color_var = random.randint(-30, 30)
                    dirt_color = np.clip(
                        [92, 64, 51] + np.array([color_var, color_var, color_var]),
                        0,
                        255,
                    )
                    for dy in range(-size, size + 1):
                        for dx in range(-size, size + 1):
                            if 0 <= y + dy < height and 0 <= x + dx < width:
                                if dx * dx + dy * dy <= size * size:
                                    img_array[y + dy, x + dx] = dirt_color

            # Common edge effects
            edge_style = random.randint(0, 2)
            if edge_style == 0:
                # Highlight top, shadow bottom
                img_array[:5, :] = np.clip(img_array[:5, :] * 1.3, 0, 255)
                img_array[-5:, :] = np.clip(img_array[-5:, :] * 0.7, 0, 255)
            elif edge_style == 1:
                # Worn edges
                for i in range(5):
                    alpha = i / 5
                    img_array[i, :] = img_array[i, :] * (0.8 + 0.2 * alpha)
                    img_array[-(i + 1), :] = img_array[-(i + 1), :] * (
                        0.6 + 0.4 * alpha
                    )

        elif "hazard" in str(element_type):
            # Highly varied hazard patterns
            hazard_type = random.randint(0, 4)

            if hazard_type == 0:
                # Sharp spikes with variation
                img_array[:] = (40, 40, 40)  # Dark base

                spike_count = random.randint(6, 10)
                spike_width = width // spike_count
                spike_height_base = height // 2

                for i in range(spike_count):
                    # Each spike is unique
                    spike_height = spike_height_base + random.randint(
                        -height // 6, height // 6
                    )
                    spike_offset = random.randint(-spike_width // 4, spike_width // 4)
                    spike_center = i * spike_width + spike_width // 2 + spike_offset

                    # Draw spike with gradient
                    for y in range(spike_height):
                        spike_half_width = max(
                            1, (spike_height - y) * spike_width // (2 * spike_height)
                        )

                        for x in range(
                            max(0, spike_center - spike_half_width),
                            min(width, spike_center + spike_half_width),
                        ):
                            # Distance from center for shading
                            dist_from_center = abs(x - spike_center) / max(
                                1, spike_half_width
                            )
                            intensity = (
                                1.0 - (y / spike_height) * 0.5 - dist_from_center * 0.3
                            )

                            # Metallic shine on one side
                            if x < spike_center:
                                spike_color = np.array([180, 60, 60]) * intensity
                            else:
                                spike_color = np.array([220, 80, 80]) * intensity * 1.1

                            img_array[height - y - 1, x] = np.clip(spike_color, 0, 255)

            elif hazard_type == 1:
                # Lava/fire pattern
                img_array[:] = (20, 0, 0)  # Dark red base

                # Add lava bubbles
                for _ in range(random.randint(20, 40)):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    radius = random.randint(10, 30)

                    y_coords, x_coords = np.ogrid[:height, :width]
                    distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
                    mask = distances <= radius

                    # Hot center, cooler edges
                    for py in range(max(0, y - radius), min(height, y + radius + 1)):
                        for px in range(max(0, x - radius), min(width, x + radius + 1)):
                            dist = np.sqrt((px - x) ** 2 + (py - y) ** 2)
                            if dist <= radius:
                                intensity = 1.0 - (dist / radius)
                                heat = intensity * intensity  # Quadratic falloff
                                lava_color = (
                                    min(255, 255 * heat + 100),
                                    min(255, 200 * heat),
                                    min(255, 50 * heat),
                                )
                                # Blend with existing
                                current = img_array[py, px]
                                img_array[py, px] = np.clip(
                                    current * 0.3 + np.array(lava_color) * 0.7, 0, 255
                                )

            elif hazard_type == 2:
                # Electric/energy field
                img_array[:] = (10, 10, 30)  # Dark blue base

                # Lightning bolts
                for _ in range(random.randint(3, 8)):
                    start_x = random.randint(0, width)
                    start_y = 0
                    end_x = random.randint(0, width)
                    end_y = height

                    # Draw jagged line
                    points = [(start_x, start_y)]
                    segments = random.randint(5, 10)
                    for s in range(1, segments):
                        progress = s / segments
                        base_x = start_x + (end_x - start_x) * progress
                        base_y = start_y + (end_y - start_y) * progress
                        offset_x = random.randint(-30, 30)
                        points.append((int(base_x + offset_x), int(base_y)))
                    points.append((end_x, end_y))

                    # Draw lightning
                    for i in range(len(points) - 1):
                        x1, y1 = points[i]
                        x2, y2 = points[i + 1]

                        # Bresenham's line
                        dx = abs(x2 - x1)
                        dy = abs(y2 - y1)
                        x, y = x1, y1
                        x_inc = 1 if x2 > x1 else -1
                        y_inc = 1 if y2 > y1 else -1

                        if dx > dy:
                            error = dx / 2
                            while x != x2:
                                if 0 <= x < width and 0 <= y < height:
                                    # Bright core
                                    img_array[y, x] = (150, 150, 255)
                                    # Glow
                                    for dy in [-1, 0, 1]:
                                        for dx in [-1, 0, 1]:
                                            if (
                                                0 <= x + dx < width
                                                and 0 <= y + dy < height
                                            ):
                                                current = img_array[y + dy, x + dx]
                                                glow = np.array([100, 100, 200])
                                                img_array[y + dy, x + dx] = np.clip(
                                                    current * 0.5 + glow * 0.5, 0, 255
                                                )
                                error -= dy
                                if error < 0:
                                    y += y_inc
                                    error += dx
                                x += x_inc

            elif hazard_type == 3:
                # Poisonous/acid pattern
                img_array[:] = (30, 50, 30)  # Dark green base

                # Bubbling acid
                for _ in range(random.randint(30, 50)):
                    x = random.randint(0, width)
                    y = random.randint(0, height)
                    radius = random.randint(5, 20)

                    # Bubble
                    y_coords, x_coords = np.ogrid[:height, :width]
                    distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
                    mask = distances <= radius

                    bubble_color = np.array([80, 200, 80]) + np.random.randint(
                        -40, 40, 3
                    )
                    img_array[mask] = np.clip(bubble_color, 0, 255)

                    # Bubble shine
                    shine_mask = (x_coords - (x - radius // 3)) ** 2 + (
                        y_coords - (y - radius // 3)
                    ) ** 2 <= (radius // 3) ** 2
                    img_array[shine_mask] = np.clip(bubble_color * 1.3, 0, 255)

            else:
                # Saw blades / mechanical hazard
                img_array[:] = (60, 60, 60)  # Metal base

                # Circular saw blades
                blade_count = random.randint(2, 4)
                for i in range(blade_count):
                    blade_x = (i + 0.5) * width // blade_count
                    blade_y = height // 2 + random.randint(-height // 4, height // 4)
                    blade_radius = random.randint(40, 80)

                    # Draw blade
                    teeth = 16
                    for angle_i in range(teeth):
                        angle = angle_i * 2 * np.pi / teeth

                        # Tooth points
                        inner_x = int(blade_x + blade_radius * 0.7 * np.cos(angle))
                        inner_y = int(blade_y + blade_radius * 0.7 * np.sin(angle))
                        outer_x = int(blade_x + blade_radius * np.cos(angle))
                        outer_y = int(blade_y + blade_radius * np.sin(angle))

                        # Draw tooth
                        for t in np.linspace(0, 1, 20):
                            x = int(inner_x + (outer_x - inner_x) * t)
                            y = int(inner_y + (outer_y - inner_y) * t)
                            if 0 <= x < width and 0 <= y < height:
                                img_array[y, x] = (180, 180, 200)

                    # Center hub
                    y_coords, x_coords = np.ogrid[:height, :width]
                    hub_mask = (x_coords - blade_x) ** 2 + (
                        y_coords - blade_y
                    ) ** 2 <= (blade_radius // 4) ** 2
                    img_array[hub_mask] = (100, 100, 120)

        elif "collectible" in str(element_type) or "powerup" in str(element_type):
            # Varied collectible patterns
            collectible_type = random.randint(0, 3)
            img_array[:] = (255, 255, 255)  # White background

            center_x, center_y = width // 2, height // 2

            if collectible_type == 0:
                # Coin style
                coin_radius = min(width, height) // 3

                # Outer ring
                y_coords, x_coords = np.ogrid[:height, :width]
                distances = np.sqrt(
                    (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
                )
                ring_mask = (distances <= coin_radius) & (
                    distances >= coin_radius * 0.8
                )
                img_array[ring_mask] = (200, 170, 0)

                # Inner circle
                inner_mask = distances <= coin_radius * 0.8
                img_array[inner_mask] = (255, 215, 0)

                # Shine effect
                for y in range(height):
                    for x in range(width):
                        if inner_mask[y, x]:
                            dist_from_top_left = np.sqrt(
                                (x - center_x + coin_radius // 3) ** 2
                                + (y - center_y + coin_radius // 3) ** 2
                            )
                            if dist_from_top_left < coin_radius // 2:
                                shine_factor = 1 - dist_from_top_left / (
                                    coin_radius // 2
                                )
                                current = img_array[y, x]
                                img_array[y, x] = np.clip(
                                    current + shine_factor * 50, 0, 255
                                )

            elif collectible_type == 1:
                # Gem/crystal
                # Multi-faceted gem
                gem_size = min(width, height) // 3
                facets = random.randint(6, 8)

                for i in range(facets):
                    angle1 = i * 2 * np.pi / facets
                    angle2 = (i + 1) * 2 * np.pi / facets

                    # Facet color varies
                    facet_color = np.array(base_color) + np.array(
                        [
                            random.randint(-30, 30),
                            random.randint(-30, 30),
                            random.randint(20, 60),  # More blue
                        ]
                    )
                    facet_color = np.clip(facet_color, 0, 255)

                    # Draw triangular facet
                    for r in range(gem_size):
                        for t in np.linspace(0, 1, max(2, r)):
                            angle = angle1 + (angle2 - angle1) * t
                            x = int(center_x + r * np.cos(angle))
                            y = int(center_y + r * np.sin(angle))

                            if 0 <= x < width and 0 <= y < height:
                                # Depth shading
                                depth_factor = 1 - (r / gem_size) * 0.3
                                img_array[y, x] = facet_color * depth_factor

            elif collectible_type == 2:
                # Star powerup
                star_size = min(width, height) // 3
                points = 5

                # Create star shape
                for angle_i in range(points * 2):
                    angle = angle_i * np.pi / points
                    if angle_i % 2 == 0:
                        radius = star_size
                    else:
                        radius = star_size // 2

                    x = int(center_x + radius * np.cos(angle - np.pi / 2))
                    y = int(center_y + radius * np.sin(angle - np.pi / 2))

                    # Draw lines from center to point
                    for t in np.linspace(0, 1, 50):
                        px = int(center_x + (x - center_x) * t)
                        py = int(center_y + (y - center_y) * t)
                        if 0 <= px < width and 0 <= py < height:
                            # Yellow gradient
                            intensity = 1 - t * 0.3
                            star_color = (255 * intensity, 215 * intensity, 0)
                            img_array[py, px] = star_color

            else:
                # Heart/health
                heart_size = min(width, height) // 3

                # Draw heart shape
                for y in range(height):
                    for x in range(width):
                        # Heart equation
                        nx = (x - center_x) / heart_size
                        ny = (y - center_y + heart_size // 3) / heart_size

                        if (nx**2 + ny**2 - 1) ** 3 - nx**2 * ny**3 <= 0:
                            # Inside heart
                            dist_from_center = np.sqrt(nx**2 + ny**2)
                            intensity = 1 - dist_from_center * 0.3
                            heart_color = (
                                255 * intensity,
                                100 * intensity,
                                100 * intensity,
                            )
                            img_array[y, x] = np.clip(heart_color, 0, 255)

        elif "interactive" in str(element_type):
            # Interactive elements with variety
            interactive_type = random.randint(0, 2)

            if interactive_type == 0:
                # Button/switch
                img_array[:] = (80, 80, 80)  # Gray frame

                # Button states
                pressed = random.choice([True, False])
                button_margin = 15

                if pressed:
                    # Pressed state
                    button_color = (150, 50, 50)  # Red when pressed
                    depth = 2
                else:
                    # Normal state
                    button_color = (50, 150, 50)  # Green when ready
                    depth = 8

                # Draw button
                button_area = img_array[
                    button_margin:-button_margin, button_margin:-button_margin
                ]
                button_area[:] = button_color

                # 3D effect
                for i in range(depth):
                    fade = i / depth
                    # Top and left highlights
                    button_area[i, i:] = np.clip(
                        np.array(button_color) * (1.3 - fade * 0.3), 0, 255
                    )
                    button_area[i:, i] = np.clip(
                        np.array(button_color) * (1.2 - fade * 0.2), 0, 255
                    )
                    # Bottom and right shadows
                    button_area[-i - 1, i:-i] = np.clip(
                        np.array(button_color) * (0.7 + fade * 0.3), 0, 255
                    )
                    button_area[i:-i, -i - 1] = np.clip(
                        np.array(button_color) * (0.6 + fade * 0.4), 0, 255
                    )

            elif interactive_type == 1:
                # Lever/handle
                img_array[:] = (60, 40, 20)  # Wood base

                # Lever position
                angle = random.randint(-45, 45)
                lever_length = min(width, height) // 3
                lever_width = 20

                # Draw base
                base_size = 40
                base_x = width // 2
                base_y = height // 2
                img_array[
                    base_y - base_size // 2 : base_y + base_size // 2,
                    base_x - base_size // 2 : base_x + base_size // 2,
                ] = (40, 30, 15)

                # Draw lever
                angle_rad = np.radians(angle)
                for i in range(lever_length):
                    x = int(base_x + i * np.sin(angle_rad))
                    y = int(base_y - i * np.cos(angle_rad))

                    for dx in range(-lever_width // 2, lever_width // 2):
                        lx = int(x + dx * np.cos(angle_rad))
                        ly = int(y + dx * np.sin(angle_rad))

                        if 0 <= lx < width and 0 <= ly < height:
                            # Metallic lever
                            img_array[ly, lx] = (120, 120, 140)

                # Lever knob
                knob_x = int(base_x + lever_length * np.sin(angle_rad))
                knob_y = int(base_y - lever_length * np.cos(angle_rad))
                y_coords, x_coords = np.ogrid[:height, :width]
                knob_mask = (x_coords - knob_x) ** 2 + (y_coords - knob_y) ** 2 <= 15**2
                img_array[knob_mask] = (160, 160, 180)

            else:
                # Door/portal
                img_array[:] = (40, 40, 60)  # Dark frame

                # Door style
                door_margin = 20
                door_area = img_array[
                    door_margin:-door_margin, door_margin:-door_margin
                ]

                # Magical portal effect
                center_x = door_area.shape[1] // 2
                center_y = door_area.shape[0] // 2
                max_radius = min(door_area.shape[0], door_area.shape[1]) // 2

                for radius in range(max_radius, 0, -5):
                    y_coords, x_coords = np.ogrid[
                        : door_area.shape[0], : door_area.shape[1]
                    ]
                    mask = (x_coords - center_x) ** 2 + (
                        y_coords - center_y
                    ) ** 2 <= radius**2

                    # Swirling colors
                    hue = (radius / max_radius) * 360
                    portal_color = np.array(
                        [
                            100 + 100 * np.sin(np.radians(hue)),
                            100 + 100 * np.sin(np.radians(hue + 120)),
                            100 + 100 * np.sin(np.radians(hue + 240)),
                        ]
                    )
                    door_area[mask] = np.clip(portal_color, 0, 255)

        else:
            # Generic gameplay element with maximum variety
            generic_type = random.randint(0, 3)

            if generic_type == 0:
                # Checkpoint flag
                img_array[:] = (200, 200, 200)  # Light background

                # Flag pole
                pole_x = width // 4
                img_array[:, pole_x - 2 : pole_x + 2] = (100, 100, 100)

                # Flag
                flag_width = width // 2
                flag_height = height // 3
                flag_color = random.choice(
                    [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                )

                for y in range(flag_height):
                    for x in range(flag_width):
                        # Waving effect
                        wave = np.sin((x / flag_width) * np.pi * 2) * 10
                        flag_y = int(height // 4 + y + wave)
                        if 0 <= flag_y < height and pole_x + x < width:
                            img_array[flag_y, pole_x + x] = flag_color

            elif generic_type == 1:
                # Moving platform track
                img_array[:] = (60, 60, 60)  # Metal track

                # Rails
                rail_height = 10
                img_array[height // 2 - rail_height : height // 2 + rail_height, :] = (
                    40,
                    40,
                    40,
                )

                # Track details
                for x in range(0, width, 30):
                    img_array[
                        height // 2 - rail_height : height // 2 + rail_height, x : x + 5
                    ] = (80, 80, 80)

            elif generic_type == 2:
                # Spring/bouncer
                img_array[:] = (100, 100, 100)  # Metal base

                # Spring coils
                coil_count = 8
                coil_height = height // (coil_count + 2)

                for i in range(coil_count):
                    y = (i + 1) * coil_height
                    # Coil with 3D effect
                    for x in range(width // 4, 3 * width // 4):
                        progress = (x - width // 4) / (width // 2)
                        coil_y = int(y + np.sin(progress * np.pi * 2) * 5)
                        if 0 <= coil_y < height:
                            # Metallic shine
                            intensity = 0.7 + 0.3 * np.sin(progress * np.pi)
                            img_array[coil_y - 2 : coil_y + 2, x] = (
                                np.array([180, 180, 200]) * intensity
                            )

            else:
                # Mystery box
                img_array[:] = base_color

                # Box with question mark
                box_margin = 30
                box_area = img_array[box_margin:-box_margin, box_margin:-box_margin]
                box_area[:] = accent_color

                # Question mark
                q_height = box_area.shape[0] // 2
                q_width = box_area.shape[1] // 3
                q_center_x = box_area.shape[1] // 2
                q_center_y = box_area.shape[0] // 2

                # Draw ?
                for y in range(q_height):
                    for x in range(q_width):
                        # Simple ? shape
                        if (
                            y < q_height // 3 and abs(x - q_width // 2) < q_width // 3
                        ) or (
                            q_height // 2 < y < 2 * q_height // 3 and x > q_width // 2
                        ):
                            box_area[
                                q_center_y - q_height // 2 + y,
                                q_center_x - q_width // 2 + x,
                            ] = highlight_color
                        elif (
                            3 * q_height // 4 < y
                            and abs(x - q_width // 2) < q_width // 6
                        ):
                            box_area[
                                q_center_y - q_height // 2 + y,
                                q_center_x - q_width // 2 + x,
                            ] = highlight_color

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

        # For more reasonable directory names, limit depth
        if len(cat_names) > 2:
            # Use only first two levels for directory names
            primary_cat = cat_names[0]
            secondary_cat = cat_names[1] if len(cat_names) > 1 else None

            for primary_val in categories[primary_cat]:
                if secondary_cat:
                    for secondary_val in categories[secondary_cat]:
                        combinations.append(
                            {primary_cat: primary_val, secondary_cat: secondary_val}
                        )
                else:
                    combinations.append({primary_cat: primary_val})
        else:
            # Original logic for simpler structures
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
