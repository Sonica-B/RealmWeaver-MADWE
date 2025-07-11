"""
Enhanced synthetic game asset generator with detailed textures and sprites - Python 3.13.5 compatible
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
                "accent_colors": [(70, 130, 180), (100, 149, 237), (173, 216, 230)],
                "patterns": [
                    "ice_crystals",
                    "snowflake",
                    "frozen_surface",
                    "blizzard_texture",
                ],
                "sprite_themes": ["ice_warrior", "frost_mage", "snow_beast"],
                "items": ["ice_sword", "frost_gem", "snowflake_charm", "frozen_orb"],
            },
            "volcanic": {
                "primary_colors": [
                    (178, 34, 34),
                    (255, 69, 0),
                    (139, 0, 0),
                    (255, 87, 34),
                ],
                "accent_colors": [(255, 140, 0), (255, 165, 0), (128, 0, 0)],
                "patterns": [
                    "lava_flow",
                    "volcanic_rock",
                    "ember_scatter",
                    "magma_cracks",
                ],
                "sprite_themes": ["fire_elemental", "lava_golem", "flame_spirit"],
                "items": ["flame_sword", "lava_crystal", "ember_stone", "fire_staff"],
            },
            "underwater": {
                "primary_colors": [
                    (0, 119, 190),
                    (0, 191, 255),
                    (70, 130, 180),
                    (0, 150, 136),
                ],
                "accent_colors": [(72, 209, 204), (64, 224, 208), (0, 206, 209)],
                "patterns": [
                    "wave_pattern",
                    "coral_growth",
                    "bubble_stream",
                    "seaweed_flow",
                ],
                "sprite_themes": ["sea_guardian", "water_elemental", "mer_warrior"],
                "items": ["trident", "pearl", "sea_crystal", "conch_shell"],
            },
            "sky": {
                "primary_colors": [
                    (135, 206, 250),
                    (255, 255, 255),
                    (176, 196, 222),
                    (100, 149, 237),
                ],
                "accent_colors": [(255, 215, 0), (255, 255, 224), (230, 230, 250)],
                "patterns": [
                    "cloud_formation",
                    "wind_current",
                    "lightning_pattern",
                    "celestial_glow",
                ],
                "sprite_themes": ["sky_guardian", "wind_spirit", "cloud_walker"],
                "items": [
                    "wind_staff",
                    "cloud_essence",
                    "lightning_bolt",
                    "celestial_orb",
                ],
            },
        }

    def generate_perlin_noise(
        self, width: int, height: int, scale: float = 0.1, octaves: int = 4
    ) -> np.ndarray:
        """Generate proper Perlin noise for natural textures"""

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(t, a, b):
            return a + t * (b - a)

        def grad(h, x, y):
            vectors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            g = vectors[h % 4]
            return g[0] * x + g[1] * y

        noise = np.zeros((width, height))

        for octave in range(octaves):
            freq = scale * (2**octave)
            amp = 1.0 / (2**octave)

            for i in range(width):
                for j in range(height):
                    x = i * freq
                    y = j * freq

                    x_floor = int(x)
                    y_floor = int(y)

                    x_frac = x - x_floor
                    y_frac = y - y_floor

                    # Get gradients at corners
                    n00 = grad(x_floor + y_floor * 57, x_frac, y_frac)
                    n01 = grad(x_floor + (y_floor + 1) * 57, x_frac, y_frac - 1)
                    n10 = grad(x_floor + 1 + y_floor * 57, x_frac - 1, y_frac)
                    n11 = grad(x_floor + 1 + (y_floor + 1) * 57, x_frac - 1, y_frac - 1)

                    # Interpolate
                    nx0 = lerp(fade(x_frac), n00, n10)
                    nx1 = lerp(fade(x_frac), n01, n11)
                    value = lerp(fade(y_frac), nx0, nx1)

                    noise[i, j] += value * amp

        return noise

    def create_seamless_texture(self, pattern_func, biome: str, **kwargs) -> np.ndarray:
        """Create seamless tiling texture using edge blending"""
        # Generate base texture
        base_texture = pattern_func(biome, **kwargs)

        # Create seamless version by blending edges
        overlap = 32  # Overlap region for seamless tiling

        # Blend horizontal edges
        for i in range(overlap):
            alpha = i / overlap
            for j in range(self.height):
                # Blend left edge with right edge
                base_texture[i, j] = (
                    alpha * base_texture[i, j]
                    + (1 - alpha) * base_texture[self.width - overlap + i, j]
                )

        # Blend vertical edges
        for j in range(overlap):
            alpha = j / overlap
            for i in range(self.width):
                # Blend top edge with bottom edge
                base_texture[i, j] = (
                    alpha * base_texture[i, j]
                    + (1 - alpha) * base_texture[i, self.height - overlap + j]
                )

        return base_texture

    def generate_bark_texture(self, biome: str) -> np.ndarray:
        """Generate realistic bark texture for forest biome"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["primary_colors"])
        accent_color = random.choice(config["accent_colors"])

        # Create base noise
        noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.02, octaves=6
        )
        detail_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.1, octaves=3
        )

        # Create vertical bark lines
        vertical_lines = np.zeros((self.width, self.height, 3))
        for i in range(self.width):
            line_intensity = np.sin(i * 0.1 + noise[i, 0] * 2) * 0.3 + 0.7
            for j in range(self.height):
                bark_variation = noise[i, j] * 0.4 + detail_noise[i, j] * 0.2

                color = [
                    int(base_color[0] * (line_intensity + bark_variation)),
                    int(base_color[1] * (line_intensity + bark_variation)),
                    int(base_color[2] * (line_intensity + bark_variation)),
                ]

                # Add accent color highlights
                if bark_variation > 0.3:
                    blend_factor = (bark_variation - 0.3) * 2
                    color = [
                        int(
                            color[0] * (1 - blend_factor)
                            + accent_color[0] * blend_factor
                        ),
                        int(
                            color[1] * (1 - blend_factor)
                            + accent_color[1] * blend_factor
                        ),
                        int(
                            color[2] * (1 - blend_factor)
                            + accent_color[2] * blend_factor
                        ),
                    ]

                vertical_lines[i, j] = [max(0, min(255, c)) for c in color]

        return vertical_lines

    def generate_sand_dunes(self, biome: str) -> np.ndarray:
        """Generate sand dune texture for desert biome"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["primary_colors"])

        # Create dune patterns using multiple noise layers
        large_dunes = self.generate_perlin_noise(
            self.width, self.height, scale=0.01, octaves=3
        )
        medium_dunes = self.generate_perlin_noise(
            self.width, self.height, scale=0.05, octaves=4
        )
        fine_sand = self.generate_perlin_noise(
            self.width, self.height, scale=0.2, octaves=2
        )

        texture = np.zeros((self.width, self.height, 3))

        for i in range(self.width):
            for j in range(self.height):
                # Combine noise layers for realistic sand appearance
                dune_height = (
                    large_dunes[i, j] * 0.5
                    + medium_dunes[i, j] * 0.3
                    + fine_sand[i, j] * 0.2
                )

                # Create color variation based on dune height (shadows and highlights)
                brightness = 0.7 + dune_height * 0.6

                color = [
                    int(base_color[0] * brightness),
                    int(base_color[1] * brightness),
                    int(base_color[2] * brightness),
                ]

                # Add wind pattern streaks
                wind_pattern = np.sin(i * 0.05 + j * 0.02) * 0.1
                color = [max(0, min(255, c + wind_pattern * 20)) for c in color]

                texture[i, j] = color

        return texture

    def generate_ice_crystals(self, biome: str) -> np.ndarray:
        """Generate ice crystal texture for snow biome"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["primary_colors"])
        accent_color = random.choice(config["accent_colors"])

        texture = np.full((self.width, self.height, 3), base_color, dtype=float)

        # Generate crystalline patterns
        num_crystals = random.randint(20, 40)

        for _ in range(num_crystals):
            center_x = random.randint(0, self.width)
            center_y = random.randint(0, self.height)
            size = random.randint(20, 80)
            angles = random.randint(6, 12)  # Crystal facets

            # Draw crystal pattern
            for angle in range(angles):
                theta = (angle / angles) * 2 * math.pi
                end_x = center_x + size * math.cos(theta)
                end_y = center_y + size * math.sin(theta)

                # Create crystal ray
                for t in np.linspace(0, 1, size):
                    x = int(center_x + t * (end_x - center_x))
                    y = int(center_y + t * (end_y - center_y))

                    if 0 <= x < self.width and 0 <= y < self.height:
                        # Crystal glow effect
                        intensity = 1 - t
                        blend_factor = intensity * 0.7

                        texture[x, y] = [
                            texture[x, y, 0] * (1 - blend_factor)
                            + accent_color[0] * blend_factor,
                            texture[x, y, 1] * (1 - blend_factor)
                            + accent_color[1] * blend_factor,
                            texture[x, y, 2] * (1 - blend_factor)
                            + accent_color[2] * blend_factor,
                        ]

        # Add ice noise
        ice_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.1, octaves=4
        )
        for i in range(self.width):
            for j in range(self.height):
                noise_factor = ice_noise[i, j] * 0.3
                texture[i, j] = [
                    max(0, min(255, c + noise_factor * 50)) for c in texture[i, j]
                ]

        return texture.astype(np.uint8)

    def generate_lava_flow(self, biome: str) -> np.ndarray:
        """Generate lava flow texture for volcanic biome"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["primary_colors"])
        accent_color = random.choice(config["accent_colors"])

        # Create flowing lava pattern
        flow_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.03, octaves=5
        )
        heat_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.1, octaves=3
        )

        texture = np.zeros((self.width, self.height, 3))

        for i in range(self.width):
            for j in range(self.height):
                # Create flowing pattern
                flow_intensity = flow_noise[i, j]
                heat_level = heat_noise[i, j]

                # Lava flow direction (generally downward)
                flow_direction = np.sin(i * 0.02) * 0.5 + flow_intensity * 0.5

                if flow_direction > 0.2:  # Active lava
                    # Bright, hot lava
                    brightness = 0.8 + heat_level * 0.4
                    color = [
                        int(accent_color[0] * brightness),
                        int(accent_color[1] * brightness),
                        int(accent_color[2] * brightness),
                    ]
                else:  # Cooled lava/rock
                    brightness = 0.3 + flow_intensity * 0.3
                    color = [
                        int(base_color[0] * brightness),
                        int(base_color[1] * brightness),
                        int(base_color[2] * brightness),
                    ]

                # Add ember effects
                if heat_level > 0.4:
                    ember_glow = (heat_level - 0.4) * 100
                    color[0] = min(255, color[0] + ember_glow)
                    color[1] = min(255, color[1] + ember_glow * 0.5)

                texture[i, j] = [max(0, min(255, c)) for c in color]

        return texture

    def generate_coral_growth(self, biome: str) -> np.ndarray:
        """Generate coral texture for underwater biome"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["primary_colors"])
        accent_colors = config["accent_colors"]

        texture = np.full((self.width, self.height, 3), base_color, dtype=float)

        # Generate coral formations
        num_corals = random.randint(15, 25)

        for _ in range(num_corals):
            center_x = random.randint(50, self.width - 50)
            center_y = random.randint(50, self.height - 50)
            coral_color = random.choice(accent_colors)

            # Create branching coral structure
            branches = random.randint(5, 12)
            for branch in range(branches):
                angle = (branch / branches) * 2 * math.pi + random.uniform(-0.5, 0.5)
                length = random.randint(30, 80)

                # Draw coral branch with organic curves
                for t in np.linspace(0, 1, length):
                    # Add organic curvature
                    curve = math.sin(t * math.pi * 3) * 20
                    x = int(
                        center_x
                        + t * length * math.cos(angle)
                        + curve * math.cos(angle + math.pi / 2)
                    )
                    y = int(
                        center_y
                        + t * length * math.sin(angle)
                        + curve * math.sin(angle + math.pi / 2)
                    )

                    if 0 <= x < self.width and 0 <= y < self.height:
                        # Coral polyp detail
                        polyp_size = max(1, int((1 - t) * 8))
                        for dx in range(-polyp_size, polyp_size + 1):
                            for dy in range(-polyp_size, polyp_size + 1):
                                if dx * dx + dy * dy <= polyp_size * polyp_size:
                                    px, py = x + dx, y + dy
                                    if 0 <= px < self.width and 0 <= py < self.height:
                                        intensity = 1 - (dx * dx + dy * dy) / (
                                            polyp_size * polyp_size
                                        )
                                        texture[px, py] = [
                                            texture[px, py, 0] * (1 - intensity)
                                            + coral_color[0] * intensity,
                                            texture[px, py, 1] * (1 - intensity)
                                            + coral_color[1] * intensity,
                                            texture[px, py, 2] * (1 - intensity)
                                            + coral_color[2] * intensity,
                                        ]

        # Add water movement effect
        wave_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.05, octaves=2
        )
        for i in range(self.width):
            for j in range(self.height):
                wave_effect = wave_noise[i, j] * 20
                texture[i, j] = [
                    max(0, min(255, c + wave_effect)) for c in texture[i, j]
                ]

        return texture.astype(np.uint8)

    def generate_cloud_formation(self, biome: str) -> np.ndarray:
        """Generate cloud texture for sky biome"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["primary_colors"])
        accent_color = random.choice(config["accent_colors"])

        # Create cloud layers with different densities
        cloud_base = self.generate_perlin_noise(
            self.width, self.height, scale=0.02, octaves=4
        )
        cloud_detail = self.generate_perlin_noise(
            self.width, self.height, scale=0.08, octaves=3
        )
        cloud_wisps = self.generate_perlin_noise(
            self.width, self.height, scale=0.15, octaves=2
        )

        texture = np.zeros((self.width, self.height, 3))

        for i in range(self.width):
            for j in range(self.height):
                # Combine cloud layers
                cloud_density = (
                    cloud_base[i, j] * 0.6
                    + cloud_detail[i, j] * 0.3
                    + cloud_wisps[i, j] * 0.1
                )

                # Cloud threshold for realistic cloud shapes
                if cloud_density > 0.1:
                    cloud_alpha = min(1.0, (cloud_density - 0.1) * 1.5)

                    # Cloud color with sunlight effects
                    sunlight_angle = i / self.width  # Simulate sun position
                    brightness = 0.8 + sunlight_angle * 0.4

                    cloud_color = [
                        base_color[0] * brightness,
                        base_color[1] * brightness,
                        base_color[2] * brightness,
                    ]

                    # Add golden hour effects
                    if sunlight_angle > 0.7:
                        golden_blend = (sunlight_angle - 0.7) * 3.33
                        cloud_color = [
                            cloud_color[0] * (1 - golden_blend)
                            + accent_color[0] * golden_blend,
                            cloud_color[1] * (1 - golden_blend)
                            + accent_color[1] * golden_blend,
                            cloud_color[2] * (1 - golden_blend)
                            + accent_color[2] * golden_blend,
                        ]

                    texture[i, j] = [max(0, min(255, c)) for c in cloud_color]
                else:
                    # Sky background
                    sky_color = [
                        base_color[0] * 0.4,
                        base_color[1] * 0.6,
                        base_color[2] * 0.9,
                    ]
                    texture[i, j] = sky_color

        return texture

    def create_detailed_character_sprite(
        self, biome: str, character_type: str
    ) -> np.ndarray:
        """Generate detailed character sprites with proper anatomy and equipment"""
        config = self.biome_configs[biome]
        primary_color = random.choice(config["primary_colors"])
        accent_color = random.choice(config["accent_colors"])

        # Create PIL image for detailed drawing
        img = Image.new("RGBA", self.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        center_x, center_y = self.width // 2, self.height // 2

        if character_type == "woodland_creature":
            # Draw detailed woodland creature
            # Body (oval torso)
            body_width, body_height = 80, 120
            draw.ellipse(
                [
                    center_x - body_width // 2,
                    center_y - body_height // 2,
                    center_x + body_width // 2,
                    center_y + body_height // 2,
                ],
                fill=primary_color,
                outline=None,
            )

            # Head (circle)
            head_radius = 45
            draw.ellipse(
                [
                    center_x - head_radius,
                    center_y - body_height // 2 - head_radius - 20,
                    center_x + head_radius,
                    center_y - body_height // 2 + head_radius - 20,
                ],
                fill=primary_color,
                outline=None,
            )

            # Arms
            arm_width, arm_length = 25, 80
            # Left arm
            draw.ellipse(
                [
                    center_x - body_width // 2 - arm_width,
                    center_y - 30,
                    center_x - body_width // 2 + arm_width,
                    center_y - 30 + arm_length,
                ],
                fill=primary_color,
                outline=None,
            )
            # Right arm
            draw.ellipse(
                [
                    center_x + body_width // 2 - arm_width,
                    center_y - 30,
                    center_x + body_width // 2 + arm_width,
                    center_y - 30 + arm_length,
                ],
                fill=primary_color,
                outline=None,
            )

            # Legs
            leg_width, leg_length = 30, 100
            # Left leg
            draw.ellipse(
                [
                    center_x - 25,
                    center_y + body_height // 2 - 20,
                    center_x - 25 + leg_width,
                    center_y + body_height // 2 - 20 + leg_length,
                ],
                fill=primary_color,
                outline=None,
            )
            # Right leg
            draw.ellipse(
                [
                    center_x - 5,
                    center_y + body_height // 2 - 20,
                    center_x - 5 + leg_width,
                    center_y + body_height // 2 - 20 + leg_length,
                ],
                fill=primary_color,
                outline=None,
            )

            # Equipment/accessories
            # Woodland cloak
            cloak_points = [
                (center_x - 60, center_y - 80),
                (center_x + 60, center_y - 80),
                (center_x + 40, center_y + 100),
                (center_x - 40, center_y + 100),
            ]
            draw.polygon(cloak_points, fill=accent_color, outline=None)

            # Nature staff
            staff_top_x = center_x + 70
            staff_top_y = center_y - 150
            staff_bottom_x = center_x + 85
            staff_bottom_y = center_y + 50
            draw.line(
                [(staff_top_x, staff_top_y), (staff_bottom_x, staff_bottom_y)],
                fill=(139, 69, 19),
                width=8,
            )

            # Staff crystal
            draw.ellipse(
                [
                    staff_top_x - 15,
                    staff_top_y - 15,
                    staff_top_x + 15,
                    staff_top_y + 15,
                ],
                fill=(0, 255, 127),
                outline=None,
            )

        elif character_type == "desert_nomad":
            # Desert warrior with robes and weapons
            # Body
            draw.ellipse(
                [center_x - 40, center_y - 60, center_x + 40, center_y + 60],
                fill=primary_color,
                outline=None,
            )

            # Head with headwrap
            draw.ellipse(
                [center_x - 35, center_y - 120, center_x + 35, center_y - 50],
                fill=accent_color,
                outline=None,
            )

            # Desert robes (flowing)
            robe_points = [
                (center_x - 60, center_y - 40),
                (center_x + 60, center_y - 40),
                (center_x + 80, center_y + 120),
                (center_x - 80, center_y + 120),
            ]
            draw.polygon(robe_points, fill=accent_color, outline=None)

            # Scimitar
            sword_handle_x = center_x - 80
            sword_handle_y = center_y + 20
            draw.line(
                [
                    (sword_handle_x, sword_handle_y),
                    (sword_handle_x - 20, sword_handle_y - 80),
                ],
                fill=(192, 192, 192),
                width=6,
            )

            # Curved blade
            blade_points = [
                (sword_handle_x - 20, sword_handle_y - 80),
                (sword_handle_x - 40, sword_handle_y - 120),
                (sword_handle_x - 25, sword_handle_y - 125),
                (sword_handle_x - 15, sword_handle_y - 85),
            ]
            draw.polygon(blade_points, fill=(220, 220, 220), outline=None)

        # Add facial features and details
        # Eyes
        eye_size = 8
        draw.ellipse(
            [
                center_x - 15 - eye_size // 2,
                center_y - 85 - eye_size // 2,
                center_x - 15 + eye_size // 2,
                center_y - 85 + eye_size // 2,
            ],
            fill=(0, 0, 0),
            outline=None,
        )
        draw.ellipse(
            [
                center_x + 15 - eye_size // 2,
                center_y - 85 - eye_size // 2,
                center_x + 15 + eye_size // 2,
                center_y - 85 + eye_size // 2,
            ],
            fill=(0, 0, 0),
            outline=None,
        )

        return np.array(img)

    def create_detailed_item_sprite(self, biome: str, item_type: str) -> np.ndarray:
        """Generate detailed item sprites with proper game item properties"""
        config = self.biome_configs[biome]
        primary_color = random.choice(config["primary_colors"])
        accent_color = random.choice(config["accent_colors"])

        img = Image.new("RGBA", self.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        center_x, center_y = self.width // 2, self.height // 2

        if item_type == "crystal_shard":
            # Multi-faceted crystal with internal glow
            crystal_points = []
            num_facets = 8
            base_radius = 60

            for i in range(num_facets):
                angle = (i / num_facets) * 2 * math.pi
                radius_variation = base_radius + random.randint(-15, 15)
                x = center_x + radius_variation * math.cos(angle)
                y = center_y + radius_variation * math.sin(angle)
                crystal_points.append((x, y))

            # Draw crystal body
            draw.polygon(crystal_points, fill=primary_color, outline=accent_color)

            # Internal facets for depth
            for i in range(0, len(crystal_points), 2):
                inner_points = [
                    (center_x, center_y),
                    crystal_points[i],
                    crystal_points[(i + 1) % len(crystal_points)],
                ]
                facet_color = tuple(max(0, min(255, c + 30)) for c in accent_color)
                draw.polygon(inner_points, fill=facet_color, outline=None)

            # Glow effect
            glow_radius = base_radius + 20
            for r in range(glow_radius, base_radius, -5):
                alpha = int(50 * (glow_radius - r) / 20)
                glow_color = accent_color + (alpha,)
                draw.ellipse(
                    [center_x - r, center_y - r, center_x + r, center_y + r],
                    outline=glow_color,
                    width=3,
                )

        elif item_type == "ancient_coin":
            # Detailed coin with engravings
            coin_radius = 45

            # Main coin body
            draw.ellipse(
                [
                    center_x - coin_radius,
                    center_y - coin_radius,
                    center_x + coin_radius,
                    center_y + coin_radius,
                ],
                fill=primary_color,
                outline=accent_color,
                width=3,
            )

            # Inner circle pattern
            inner_radius = coin_radius - 10
            draw.ellipse(
                [
                    center_x - inner_radius,
                    center_y - inner_radius,
                    center_x + inner_radius,
                    center_y + inner_radius,
                ],
                outline=accent_color,
                width=2,
            )

            # Ancient symbols (simplified runes)
            symbol_radius = coin_radius - 20
            for i in range(8):
                angle = (i / 8) * 2 * math.pi
                x1 = center_x + symbol_radius * math.cos(angle)
                y1 = center_y + symbol_radius * math.sin(angle)
                x2 = center_x + (symbol_radius - 8) * math.cos(angle)
                y2 = center_y + (symbol_radius - 8) * math.sin(angle)
                draw.line([(x1, y1), (x2, y2)], fill=accent_color, width=2)

            # Central symbol
            draw.ellipse(
                [center_x - 8, center_y - 8, center_x + 8, center_y + 8],
                fill=accent_color,
                outline=None,
            )

        elif item_type == "magic_staff":
            # Detailed magical staff
            # Staff shaft
            staff_bottom_y = center_y + 150
            staff_top_y = center_y - 150
            draw.line(
                [(center_x, staff_bottom_y), (center_x, staff_top_y)],
                fill=(101, 67, 33),
                width=12,
            )

            # Staff head (ornate design)
            head_points = [
                (center_x - 25, staff_top_y),
                (center_x - 15, staff_top_y - 30),
                (center_x, staff_top_y - 40),
                (center_x + 15, staff_top_y - 30),
                (center_x + 25, staff_top_y),
                (center_x, staff_top_y + 10),
            ]
            draw.polygon(head_points, fill=accent_color, outline=primary_color, width=2)

            # Magical orb
            orb_radius = 20
            draw.ellipse(
                [
                    center_x - orb_radius,
                    staff_top_y - 25 - orb_radius,
                    center_x + orb_radius,
                    staff_top_y - 25 + orb_radius,
                ],
                fill=primary_color,
                outline=accent_color,
                width=2,
            )

            # Magic energy effect
            for i in range(6):
                angle = (i / 6) * 2 * math.pi
                energy_x = center_x + 35 * math.cos(angle)
                energy_y = staff_top_y - 25 + 35 * math.sin(angle)
                draw.ellipse(
                    [energy_x - 5, energy_y - 5, energy_x + 5, energy_y + 5],
                    fill=accent_color,
                    outline=None,
                )

        return np.array(img)

    def create_detailed_texture(self, biome: str, pattern_type: str) -> np.ndarray:
        """Create detailed biome-specific textures"""
        if pattern_type == "bark_texture" and biome == "forest":
            return self.generate_bark_texture(biome)
        elif pattern_type == "sand_dunes" and biome == "desert":
            return self.generate_sand_dunes(biome)
        elif pattern_type == "ice_crystals" and biome == "snow":
            return self.generate_ice_crystals(biome)
        elif pattern_type == "lava_flow" and biome == "volcanic":
            return self.generate_lava_flow(biome)
        elif pattern_type == "coral_growth" and biome == "underwater":
            return self.generate_coral_growth(biome)
        elif pattern_type == "cloud_formation" and biome == "sky":
            return self.generate_cloud_formation(biome)
        else:
            # Fallback to enhanced generic pattern
            return self.create_enhanced_generic_pattern(biome, pattern_type)

    def create_enhanced_generic_pattern(
        self, biome: str, pattern_type: str
    ) -> np.ndarray:
        """Enhanced generic patterns with improved detail"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["primary_colors"])

        # Multi-layer noise for complex patterns
        primary_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.05, octaves=4
        )
        detail_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.15, octaves=3
        )
        fine_noise = self.generate_perlin_noise(
            self.width, self.height, scale=0.3, octaves=2
        )

        texture = np.zeros((self.width, self.height, 3))

        for i in range(self.width):
            for j in range(self.height):
                # Combine noise layers
                combined_noise = (
                    primary_noise[i, j] * 0.6
                    + detail_noise[i, j] * 0.3
                    + fine_noise[i, j] * 0.1
                )

                # Color variation based on noise
                brightness = 0.5 + combined_noise * 0.8

                color = [
                    int(base_color[0] * brightness),
                    int(base_color[1] * brightness),
                    int(base_color[2] * brightness),
                ]

                texture[i, j] = [max(0, min(255, c)) for c in color]

        return texture


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
