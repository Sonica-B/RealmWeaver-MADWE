"""
Enhanced synthetic game asset generator with detailed textures and sprites
Generates production-quality game assets with proper seamless tiling, detailed sprites, and biome-specific patterns
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import json
from datetime import datetime
import math
import random
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import noise  # For Perlin noise


class DetailedAssetGenerator:
    """Enhanced asset generator with sophisticated patterns and game-ready quality"""

    def __init__(self, size: Tuple[int, int] = (512, 512)):
        self.size = size
        self.width, self.height = size

        # Enhanced biome configurations with detailed properties
        self.biome_configs = {
            "forest": {
                "primary_colors": [
                    (34, 89, 34),
                    (0, 120, 0),
                    (85, 107, 47),
                    (46, 125, 50),
                ],
                "accent_colors": [
                    (139, 69, 19),
                    (160, 82, 45),
                    (205, 133, 63),
                ],  # Browns for bark/dirt
                "patterns": [
                    "bark_texture",
                    "leaf_pattern",
                    "moss_growth",
                    "wood_grain",
                ],
                "sprite_themes": [
                    "woodland_creature",
                    "forest_guardian",
                    "nature_spirit",
                ],
                "items": ["acorn", "mushroom", "crystal_shard", "wooden_staff"],
            },
            "desert": {
                "primary_colors": [
                    (238, 203, 173),
                    (205, 133, 63),
                    (244, 164, 96),
                    (255, 193, 7),
                ],
                "accent_colors": [(160, 82, 45), (210, 180, 140), (255, 218, 185)],
                "patterns": [
                    "sand_dunes",
                    "rock_formation",
                    "ancient_ruins",
                    "wind_erosion",
                ],
                "sprite_themes": ["desert_nomad", "sand_elemental", "ancient_guardian"],
                "items": ["scimitar", "golden_scarab", "sand_crystal", "ancient_coin"],
            },
            "snow": {
                "primary_colors": [
                    (255, 250, 250),
                    (176, 224, 230),
                    (135, 206, 235),
                    (245, 245, 245),
                ],
                "accent_colors": [(70, 130, 180), (100, 149, 237), (192, 192, 192)],
                "patterns": [
                    "ice_crystals",
                    "snow_drift",
                    "frozen_surface",
                    "frost_pattern",
                ],
                "sprite_themes": ["ice_warrior", "frost_mage", "snow_spirit"],
                "items": ["ice_shard", "frost_orb", "frozen_blade", "snow_crystal"],
            },
            "volcanic": {
                "primary_colors": [
                    (139, 0, 0),
                    (255, 69, 0),
                    (178, 34, 34),
                    (255, 140, 0),
                ],
                "accent_colors": [(64, 64, 64), (105, 105, 105), (255, 215, 0)],
                "patterns": ["lava_flow", "obsidian", "magma_cracks", "volcanic_rock"],
                "sprite_themes": ["fire_demon", "magma_golem", "flame_dancer"],
                "items": ["lava_gem", "obsidian_blade", "fire_crystal", "molten_core"],
            },
            "underwater": {
                "primary_colors": [
                    (0, 119, 190),
                    (0, 150, 199),
                    (0, 180, 216),
                    (129, 204, 255),
                ],
                "accent_colors": [(0, 71, 119), (46, 134, 171), (144, 224, 239)],
                "patterns": [
                    "coral_reef",
                    "water_caustics",
                    "seaweed",
                    "bubble_stream",
                ],
                "sprite_themes": ["merfolk", "sea_guardian", "aqua_elemental"],
                "items": ["pearl", "trident", "coral_staff", "sea_crystal"],
            },
            "sky": {
                "primary_colors": [
                    (135, 206, 250),
                    (176, 224, 230),
                    (255, 255, 255),
                    (173, 216, 230),
                ],
                "accent_colors": [(255, 215, 0), (255, 182, 193), (221, 160, 221)],
                "patterns": ["cloud_formation", "aurora", "star_field", "wind_current"],
                "sprite_themes": ["sky_dancer", "wind_rider", "celestial_being"],
                "items": ["feather", "wind_gem", "cloud_essence", "star_fragment"],
            },
        }

    def generate_perlin_noise(
        self, width: int, height: int, scale: float = 0.1, octaves: int = 6
    ) -> np.ndarray:
        """Generate Perlin noise for natural-looking textures"""
        noise_array = np.zeros((width, height))

        for i in range(width):
            for j in range(height):
                noise_array[i][j] = noise.pnoise2(
                    i * scale,
                    j * scale,
                    octaves=octaves,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=width,
                    repeaty=height,
                    base=0,
                )

        # Normalize to 0-1 range
        noise_array = (noise_array - noise_array.min()) / (
            noise_array.max() - noise_array.min()
        )
        return noise_array

    def create_seamless_texture(self, texture_func: callable, biome: str) -> np.ndarray:
        """Create seamless tiling texture using overlap blending"""
        # Generate larger texture
        overlap = 64
        large_size = (self.width + overlap * 2, self.height + overlap * 2)

        # Generate base texture
        base_texture = texture_func(biome)

        # Create seamless edges
        result = np.copy(base_texture)

        # Blend horizontal edges
        for i in range(overlap):
            alpha = i / overlap
            result[i, :] = (1 - alpha) * base_texture[
                -overlap + i, :
            ] + alpha * base_texture[i, :]
            result[-overlap + i, :] = (1 - alpha) * base_texture[
                i, :
            ] + alpha * base_texture[-overlap + i, :]

        # Blend vertical edges
        for j in range(overlap):
            alpha = j / overlap
            result[:, j] = (1 - alpha) * result[:, -overlap + j] + alpha * result[:, j]
            result[:, -overlap + j] = (1 - alpha) * result[:, j] + alpha * result[
                :, -overlap + j
            ]

        return result

    def create_detailed_texture(self, biome: str, pattern: str) -> np.ndarray:
        """Create detailed texture based on biome and pattern"""
        config = self.biome_configs[biome]
        texture = np.zeros((self.width, self.height, 3))

        if pattern == "bark_texture":
            # Create vertical bark lines with variation
            base_color = random.choice(config["accent_colors"])
            for i in range(self.width):
                column_variation = random.uniform(0.8, 1.2)
                for j in range(self.height):
                    # Add vertical streaks
                    streak = math.sin(j * 0.1) * 0.3 + 0.7
                    noise_val = self.generate_perlin_noise(32, 32, 0.2)[i % 32, j % 32]

                    color_factor = streak * column_variation * (0.7 + 0.3 * noise_val)
                    texture[i, j] = [int(c * color_factor) for c in base_color]

        elif pattern == "sand_dunes":
            # Create wavy sand dune patterns
            base_color = random.choice(config["primary_colors"])
            dune_noise = self.generate_perlin_noise(
                self.width, self.height, 0.02, octaves=4
            )

            for i in range(self.width):
                for j in range(self.height):
                    # Create dune ridges
                    dune_factor = math.sin(i * 0.05 + dune_noise[i, j] * 10) * 0.5 + 0.5
                    shadow = 1.0 if dune_factor > 0.5 else 0.85

                    texture[i, j] = [
                        int(c * shadow * (0.9 + 0.1 * dune_noise[i, j]))
                        for c in base_color
                    ]

        elif pattern == "ice_crystals":
            # Create crystalline ice patterns
            base_color = random.choice(config["primary_colors"])
            crystal_centers = [
                (random.randint(0, self.width), random.randint(0, self.height))
                for _ in range(15)
            ]

            for i in range(self.width):
                for j in range(self.height):
                    # Find nearest crystal center
                    min_dist = float("inf")
                    for cx, cy in crystal_centers:
                        dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                        min_dist = min(min_dist, dist)

                    # Create crystalline effect
                    crystal_factor = max(0, 1 - min_dist / 50)
                    sparkle = random.uniform(0.9, 1.1) if crystal_factor > 0.7 else 1.0

                    texture[i, j] = [
                        int(c * (0.8 + 0.2 * crystal_factor) * sparkle)
                        for c in base_color
                    ]

        elif pattern == "lava_flow":
            # Create flowing lava patterns
            primary = random.choice(config["primary_colors"])
            accent = random.choice(config["accent_colors"])

            flow_noise = self.generate_perlin_noise(
                self.width, self.height, 0.05, octaves=3
            )
            heat_noise = self.generate_perlin_noise(
                self.width, self.height, 0.1, octaves=2
            )

            for i in range(self.width):
                for j in range(self.height):
                    # Create lava flow effect
                    flow = flow_noise[i, j]
                    heat = heat_noise[i, j]

                    if flow > 0.6:  # Hot lava
                        color = primary
                        brightness = 0.8 + 0.2 * heat
                    else:  # Cooled lava
                        color = accent
                        brightness = 0.6 + 0.2 * heat

                    texture[i, j] = [int(c * brightness) for c in color]

        elif pattern == "water_caustics":
            # Create underwater light patterns
            base_color = random.choice(config["primary_colors"])

            # Generate multiple layers of caustics
            caustic1 = self.generate_perlin_noise(
                self.width, self.height, 0.03, octaves=2
            )
            caustic2 = self.generate_perlin_noise(
                self.width, self.height, 0.05, octaves=2
            )

            for i in range(self.width):
                for j in range(self.height):
                    # Combine caustic layers
                    caustic = (caustic1[i, j] + caustic2[i, j]) * 0.5
                    brightness = 0.7 + 0.3 * caustic

                    # Add wave distortion
                    wave = math.sin(i * 0.1) * math.sin(j * 0.1) * 0.1 + 0.9

                    texture[i, j] = [int(c * brightness * wave) for c in base_color]

        else:
            # Default pattern with Perlin noise
            base_color = random.choice(config["primary_colors"])
            pattern_noise = self.generate_perlin_noise(self.width, self.height, 0.05)

            for i in range(self.width):
                for j in range(self.height):
                    brightness = 0.5 + pattern_noise[i, j] * 0.5
                    texture[i, j] = [int(c * brightness) for c in base_color]

        return texture.astype(np.uint8)

    def create_detailed_character_sprite(
        self, biome: str, character_type: str
    ) -> np.ndarray:
        """Create detailed character sprite with proper anatomy"""
        sprite = np.zeros((self.height, self.width, 4), dtype=np.uint8)  # RGBA
        config = self.biome_configs[biome]

        # Character dimensions
        center_x, center_y = self.width // 2, self.height // 2

        # Create character based on type
        if character_type == "woodland_creature":
            # Draw body
            body_color = (101, 67, 33)  # Brown
            self._draw_ellipse(sprite, center_x, center_y + 60, 40, 60, body_color)

            # Draw head
            self._draw_circle(sprite, center_x, center_y - 20, 35, body_color)

            # Draw ears
            self._draw_triangle(
                sprite, center_x - 25, center_y - 40, 15, 25, body_color
            )
            self._draw_triangle(
                sprite, center_x + 25, center_y - 40, 15, 25, body_color
            )

            # Draw eyes
            eye_color = (255, 255, 255)
            self._draw_circle(sprite, center_x - 12, center_y - 20, 8, eye_color)
            self._draw_circle(sprite, center_x + 12, center_y - 20, 8, eye_color)
            self._draw_circle(sprite, center_x - 12, center_y - 20, 4, (0, 0, 0))
            self._draw_circle(sprite, center_x + 12, center_y - 20, 4, (0, 0, 0))

            # Draw arms
            self._draw_ellipse(sprite, center_x - 35, center_y + 40, 15, 40, body_color)
            self._draw_ellipse(sprite, center_x + 35, center_y + 40, 15, 40, body_color)

            # Draw legs
            self._draw_ellipse(
                sprite, center_x - 15, center_y + 100, 15, 30, body_color
            )
            self._draw_ellipse(
                sprite, center_x + 15, center_y + 100, 15, 30, body_color
            )

        elif character_type == "fire_demon":
            # Fiery colors
            body_color = (178, 34, 34)  # Firebrick
            flame_color = (255, 69, 0)  # Orange red

            # Draw flaming body
            self._draw_flame_body(sprite, center_x, center_y, body_color, flame_color)

            # Draw horns
            horn_color = (139, 0, 0)
            self._draw_triangle(
                sprite, center_x - 20, center_y - 60, 10, 30, horn_color
            )
            self._draw_triangle(
                sprite, center_x + 20, center_y - 60, 10, 30, horn_color
            )

            # Draw glowing eyes
            self._draw_circle(sprite, center_x - 15, center_y - 30, 10, flame_color)
            self._draw_circle(sprite, center_x + 15, center_y - 30, 10, flame_color)

        elif character_type == "ice_warrior":
            # Icy colors
            armor_color = (176, 224, 230)  # Powder blue
            ice_color = (135, 206, 235)  # Sky blue

            # Draw armored body
            self._draw_rectangle(
                sprite, center_x - 30, center_y - 20, 60, 80, armor_color
            )

            # Draw helmet
            self._draw_ellipse(sprite, center_x, center_y - 50, 35, 40, ice_color)

            # Draw ice spikes on shoulders
            self._draw_triangle(sprite, center_x - 40, center_y - 10, 15, 30, ice_color)
            self._draw_triangle(sprite, center_x + 40, center_y - 10, 15, 30, ice_color)

            # Draw legs
            self._draw_rectangle(
                sprite, center_x - 20, center_y + 60, 15, 40, armor_color
            )
            self._draw_rectangle(
                sprite, center_x + 5, center_y + 60, 15, 40, armor_color
            )

        # Apply anti-aliasing
        sprite = self._apply_antialiasing(sprite)

        return sprite

    def create_detailed_item_sprite(self, biome: str, item_type: str) -> np.ndarray:
        """Create detailed item sprite"""
        sprite = np.zeros((self.height, self.width, 4), dtype=np.uint8)  # RGBA
        config = self.biome_configs[biome]

        center_x, center_y = self.width // 2, self.height // 2

        if item_type == "crystal_shard":
            # Draw crystal with facets
            crystal_color = random.choice(config["primary_colors"])
            self._draw_crystal(sprite, center_x, center_y, 40, 80, crystal_color)

        elif item_type == "wooden_staff":
            # Draw staff with details
            wood_color = (139, 69, 19)
            gem_color = random.choice(config["primary_colors"])

            # Staff body
            self._draw_rectangle(
                sprite, center_x - 5, center_y - 100, 10, 200, wood_color
            )

            # Top ornament
            self._draw_circle(sprite, center_x, center_y - 100, 20, gem_color)

        elif item_type == "fire_crystal":
            # Glowing fire crystal
            core_color = (255, 69, 0)
            glow_color = (255, 140, 0)

            # Create glow effect
            for radius in range(50, 20, -5):
                alpha = (50 - radius) / 30
                self._draw_circle_glow(
                    sprite, center_x, center_y, radius, glow_color, alpha
                )

            # Draw crystal core
            self._draw_crystal(sprite, center_x, center_y, 30, 60, core_color)

        return sprite

    def _draw_circle(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        radius: int,
        color: Tuple[int, int, int],
    ):
        """Draw a filled circle"""
        for i in range(max(0, x - radius), min(self.width, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.height, y + radius + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                    image[j, i] = (*color, 255)

    def _draw_ellipse(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int],
    ):
        """Draw a filled ellipse"""
        for i in range(max(0, x - width), min(self.width, x + width + 1)):
            for j in range(max(0, y - height), min(self.height, y + height + 1)):
                if ((i - x) / width) ** 2 + ((j - y) / height) ** 2 <= 1:
                    image[j, i] = (*color, 255)

    def _draw_rectangle(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int],
    ):
        """Draw a filled rectangle"""
        for i in range(max(0, x), min(self.width, x + width)):
            for j in range(max(0, y), min(self.height, y + height)):
                image[j, i] = (*color, 255)

    def _draw_triangle(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int],
    ):
        """Draw a filled triangle"""
        for i in range(max(0, x - width // 2), min(self.width, x + width // 2 + 1)):
            for j in range(max(0, y), min(self.height, y + height)):
                # Check if point is inside triangle
                rel_x = abs(i - x)
                max_width_at_y = width * (1 - (j - y) / height) / 2
                if rel_x <= max_width_at_y:
                    image[j, i] = (*color, 255)

    def _draw_crystal(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int],
    ):
        """Draw a crystal shape with facets"""
        # Draw main crystal body (hexagon-like)
        facets = 6
        for angle in range(0, 360, 360 // facets):
            rad = math.radians(angle)
            x1 = int(x + width * math.cos(rad))
            y1 = int(y + height * 0.3 * math.sin(rad))

            # Draw triangular facet
            self._draw_triangle_facet(image, x, y - height // 2, x1, y1, color)
            self._draw_triangle_facet(image, x, y + height // 2, x1, y1, color)

    def _draw_triangle_facet(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
    ):
        """Draw a triangular facet with shading"""
        # Simple triangle fill between three points
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Shade based on position
        shade_factor = 0.7 + 0.3 * (cx / self.width)
        shaded_color = tuple(int(c * shade_factor) for c in color)

        # Fill triangle (simplified)
        for i in range(min(x1, x2), max(x1, x2)):
            for j in range(min(y1, y2), max(y1, y2)):
                if 0 <= i < self.width and 0 <= j < self.height:
                    image[j, i] = (*shaded_color, 255)

    def _draw_flame_body(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        body_color: Tuple[int, int, int],
        flame_color: Tuple[int, int, int],
    ):
        """Draw a flaming body effect"""
        # Draw main body
        self._draw_ellipse(image, x, y, 40, 60, body_color)

        # Add flame effects around body
        flame_points = 20
        for i in range(flame_points):
            angle = (i / flame_points) * 2 * math.pi
            flame_x = int(x + 45 * math.cos(angle))
            flame_y = int(y + 65 * math.sin(angle))

            # Random flame height
            flame_height = random.randint(10, 25)
            self._draw_triangle(image, flame_x, flame_y, 8, flame_height, flame_color)

    def _draw_circle_glow(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        radius: int,
        color: Tuple[int, int, int],
        alpha: float,
    ):
        """Draw a glowing circle with transparency"""
        for i in range(max(0, x - radius), min(self.width, x + radius + 1)):
            for j in range(max(0, y - radius), min(self.height, y + radius + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                    # Blend with existing color
                    existing = image[j, i]
                    if existing[3] > 0:  # Has alpha
                        # Blend colors
                        for c in range(3):
                            image[j, i, c] = int(
                                existing[c] * (1 - alpha) + color[c] * alpha
                            )
                    else:
                        image[j, i] = (*color, int(255 * alpha))

    def _apply_antialiasing(self, image: np.ndarray) -> np.ndarray:
        """Apply simple antialiasing to sprite"""
        # Convert to PIL for easier filtering
        pil_image = Image.fromarray(image, mode="RGBA")

        # Apply slight blur to edges
        pil_image = pil_image.filter(ImageFilter.SMOOTH_MORE)

        # Convert back to numpy
        return np.array(pil_image)


def generate_enhanced_assets(
    output_dir: Path | str, num_samples: int = 100, size: Tuple[int, int] = (512, 512)
) -> None:
    """Generate enhanced synthetic game assets with detailed patterns and sprites"""

    output_dir = Path(output_dir)
    generator = DetailedAssetGenerator(size)

    biomes = ["forest", "desert", "snow", "volcanic", "underwater", "sky"]
    asset_types = ["textures", "sprites"]

    print("Generating enhanced synthetic game assets with detailed patterns...")
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "num_samples": num_samples,
        "size": size,
        "generator_version": "2.0_enhanced",
        "features": [
            "seamless_tiling",
            "biome_specific_patterns",
            "detailed_sprites",
            "proper_anatomy",
            "game_ready_items",
            "perlin_noise_textures",
        ],
        "assets": [],
    }

    for asset_type in asset_types:
        for biome in biomes:
            biome_dir = output_dir / "raw" / asset_type / biome
            biome_dir.mkdir(parents=True, exist_ok=True)

            config = generator.biome_configs[biome]

            for i in tqdm(range(num_samples), desc=f"{asset_type}/{biome}"):
                if asset_type == "textures":
                    # Generate detailed texture
                    pattern = random.choice(config["patterns"])
                    img_array = generator.create_detailed_texture(biome, pattern)

                    # Ensure seamless tiling
                    img_array = generator.create_seamless_texture(
                        lambda b, p=pattern: generator.create_detailed_texture(b, p),
                        biome,
                    )

                else:  # sprites
                    sprite_category = random.choice(["character", "item"])

                    if sprite_category == "character":
                        character_type = random.choice(config["sprite_themes"])
                        img_array = generator.create_detailed_character_sprite(
                            biome, character_type
                        )
                    else:  # item
                        item_type = random.choice(config["items"])
                        img_array = generator.create_detailed_item_sprite(
                            biome, item_type
                        )

                # Convert to PIL Image and save
                if img_array.shape[-1] == 4:  # RGBA
                    img = Image.fromarray(img_array.astype(np.uint8), "RGBA")
                else:  # RGB
                    img = Image.fromarray(img_array.astype(np.uint8), "RGB")

                filename = f"{biome}_{asset_type}_{i:04d}.png"
                filepath = biome_dir / filename
                img.save(filepath, optimize=True)

                # Enhanced metadata
                asset_metadata = {
                    "filename": filename,
                    "path": str(filepath.relative_to(output_dir)),
                    "asset_type": asset_type,
                    "biome": biome,
                    "size": size,
                    "seamless_tiling": asset_type == "textures",
                    "has_transparency": img_array.shape[-1] == 4,
                }

                if asset_type == "textures":
                    asset_metadata["pattern_type"] = pattern
                    asset_metadata["noise_layers"] = 3
                else:
                    asset_metadata["sprite_category"] = sprite_category
                    if sprite_category == "character":
                        asset_metadata["character_type"] = character_type
                    else:
                        asset_metadata["item_type"] = item_type

                metadata["assets"].append(asset_metadata)

    # Save enhanced metadata
    metadata_path = output_dir / "metadata" / "enhanced_assets.json"
    metadata_path.parent.mkdir(exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(metadata['assets'])} enhanced assets")
    print(f"Enhanced metadata saved to {metadata_path}")
    print("\nEnhanced features:")
    print("- Seamless tiling textures with proper edge blending")
    print("- Biome-specific patterns (bark, sand dunes, ice crystals, etc.)")
    print("- Detailed character sprites with anatomy and equipment")
    print("- Game-ready item sprites with proper visual effects")
    print("- Multi-layer Perlin noise for realistic textures")
    print("- RGBA support for sprite transparency")


def main():
    """Main function with enhanced generation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate enhanced game assets for MADWE"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Output directory for assets"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples per category"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size (width height)",
    )
    parser.add_argument(
        "--enhanced", action="store_true", default=True, help="Use enhanced generation"
    )

    args = parser.parse_args()

    if args.enhanced:
        generate_enhanced_assets(
            args.output_dir, num_samples=args.num_samples, size=tuple(args.size)
        )
    else:
        print("Using enhanced generation by default")
        generate_enhanced_assets(
            args.output_dir, num_samples=args.num_samples, size=tuple(args.size)
        )


if __name__ == "__main__":
    main()
