#!/usr/bin/env python3
"""
Enhanced Game Asset Generation Script for MADWE Project
Generates high-quality, diverse game assets covering all 2D game genres and art styles
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
import random
import json
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, List, Dict
import math
import colorsys
from noise import pnoise2, pnoise3


class AdvancedAssetGenerator:
    """Enhanced generator for high-quality, diverse game assets"""

    def __init__(self, size: Tuple[int, int] = (512, 512)):
        self.width, self.height = size
        self.seed = random.randint(0, 1000000)

        # Expanded art styles covering all 2D game genres
        self.art_styles = {
            "pixel_art": {
                "description": "8-bit to 32-bit retro style",
                "pixel_sizes": [1, 2, 4, 8, 16],
                "color_palettes": {
                    "gameboy": [
                        (155, 188, 15),
                        (139, 172, 15),
                        (48, 98, 48),
                        (15, 56, 15),
                    ],
                    "nes": [
                        (0, 0, 0),
                        (255, 255, 255),
                        (255, 0, 0),
                        (0, 255, 0),
                        (0, 0, 255),
                    ],
                    "c64": [(0, 0, 0), (255, 255, 255), (136, 0, 0), (170, 255, 238)],
                    "pico8": [(0, 0, 0), (29, 43, 83), (126, 37, 83), (0, 135, 81)],
                    "gb_green": [
                        (15, 56, 15),
                        (48, 98, 48),
                        (139, 172, 15),
                        (155, 188, 15),
                    ],
                },
            },
            "isometric": {
                "description": "2.5D isometric perspective for strategy and RPG games",
                "angles": [30, 45, 60],
                "tile_types": ["floor", "wall", "roof", "prop", "character"],
            },
            "vector": {
                "description": "Clean, scalable graphics for casual and mobile games",
                "styles": ["flat", "material", "gradient", "geometric"],
            },
            "hand_drawn": {
                "description": "Artistic, sketch-like style for indie games",
                "techniques": ["pencil", "ink", "watercolor", "charcoal", "crayon"],
            },
            "realistic": {
                "description": "Photo-realistic textures for immersive games",
                "materials": ["metal", "wood", "stone", "fabric", "glass", "organic"],
            },
            "cartoon": {
                "description": "Vibrant, exaggerated style for platformers and kids games",
                "features": [
                    "cel_shading",
                    "thick_outlines",
                    "bright_colors",
                    "simple_shapes",
                ],
            },
            "anime": {
                "description": "Japanese animation style for JRPGs and visual novels",
                "elements": ["chibi", "mecha", "fantasy", "school", "cyberpunk"],
            },
            "gothic": {
                "description": "Dark, medieval style for horror and dark fantasy",
                "themes": ["medieval", "victorian", "lovecraftian", "dark_souls"],
            },
            "sci_fi": {
                "description": "Futuristic style for space and cyberpunk games",
                "variants": ["neon", "holographic", "tech", "alien", "dystopian"],
            },
            "minimalist": {
                "description": "Simple, clean style for puzzle and abstract games",
                "approaches": ["geometric", "monochrome", "flat_color", "silhouette"],
            },
        }

        # Comprehensive game genres with specific asset requirements
        self.game_genres = {
            "platformer": {
                "sprites": ["hero", "enemy", "collectible", "platform", "hazard"],
                "textures": ["ground", "wall", "background", "foreground", "parallax"],
            },
            "rpg": {
                "sprites": ["warrior", "mage", "rogue", "npc", "monster", "boss"],
                "textures": [
                    "dungeon",
                    "town",
                    "overworld",
                    "battle_bg",
                    "ui_elements",
                ],
            },
            "strategy": {
                "sprites": ["unit", "building", "resource", "terrain_feature"],
                "textures": ["terrain", "grid", "fog_of_war", "minimap"],
            },
            "shooter": {
                "sprites": ["player", "enemy", "bullet", "powerup", "explosion"],
                "textures": ["arena", "cover", "skybox", "particle_effects"],
            },
            "puzzle": {
                "sprites": ["piece", "block", "gem", "mechanism", "cursor"],
                "textures": ["board", "background", "ui_frame", "effect"],
            },
            "fighting": {
                "sprites": ["fighter", "special_move", "combo_effect", "health_bar"],
                "textures": ["arena", "stage", "crowd", "impact_effect"],
            },
            "racing": {
                "sprites": ["vehicle", "obstacle", "powerup", "checkpoint"],
                "textures": ["track", "terrain", "skyline", "road_surface"],
            },
            "survival": {
                "sprites": ["survivor", "zombie", "resource", "weapon", "shelter"],
                "textures": ["wasteland", "ruins", "nature", "weather"],
            },
            "metroidvania": {
                "sprites": ["explorer", "upgrade", "boss", "secret", "ability"],
                "textures": ["cave", "temple", "laboratory", "alien_world"],
            },
            "roguelike": {
                "sprites": ["hero", "monster", "loot", "trap", "merchant"],
                "textures": ["dungeon_floor", "wall_variant", "room_decoration"],
            },
        }

        # Enhanced biome configurations with genre-specific variations
        self.biome_configs = {
            "forest": {
                "base_colors": [
                    (34, 139, 34),
                    (0, 100, 0),
                    (85, 107, 47),
                    (107, 142, 35),
                ],
                "detail_colors": [
                    (139, 69, 19),
                    (160, 82, 45),
                    (0, 128, 0),
                    (154, 205, 50),
                ],
                "patterns": ["bark", "leaves", "moss", "undergrowth", "canopy"],
                "sprites": {
                    "creatures": ["deer", "wolf", "bear", "fairy", "treant"],
                    "items": ["mushroom", "herb", "acorn", "branch", "flower"],
                },
            },
            "desert": {
                "base_colors": [
                    (238, 203, 173),
                    (255, 228, 181),
                    (255, 218, 185),
                    (222, 184, 135),
                ],
                "detail_colors": [
                    (210, 180, 140),
                    (188, 143, 143),
                    (244, 164, 96),
                    (205, 133, 63),
                ],
                "patterns": [
                    "sand_dunes",
                    "cracked_earth",
                    "sandstone",
                    "oasis",
                    "mirage",
                ],
                "sprites": {
                    "creatures": ["scorpion", "snake", "camel", "mummy", "djinn"],
                    "items": ["cactus", "skull", "treasure", "water_flask", "scarab"],
                },
            },
            "snow": {
                "base_colors": [
                    (255, 250, 250),
                    (245, 245, 245),
                    (240, 248, 255),
                    (230, 230, 250),
                ],
                "detail_colors": [
                    (176, 224, 230),
                    (175, 238, 238),
                    (135, 206, 235),
                    (173, 216, 230),
                ],
                "patterns": [
                    "ice_crystals",
                    "snow_drift",
                    "frozen_lake",
                    "blizzard",
                    "aurora",
                ],
                "sprites": {
                    "creatures": [
                        "yeti",
                        "penguin",
                        "polar_bear",
                        "ice_elemental",
                        "frost_sprite",
                    ],
                    "items": [
                        "icicle",
                        "snowflake",
                        "frozen_gem",
                        "ice_shard",
                        "frost_flower",
                    ],
                },
            },
            "volcanic": {
                "base_colors": [(139, 0, 0), (178, 34, 34), (220, 20, 60), (255, 0, 0)],
                "detail_colors": [
                    (255, 69, 0),
                    (255, 140, 0),
                    (255, 165, 0),
                    (255, 215, 0),
                ],
                "patterns": [
                    "lava_flow",
                    "obsidian",
                    "ash_fall",
                    "magma_cracks",
                    "volcanic_rock",
                ],
                "sprites": {
                    "creatures": [
                        "phoenix",
                        "salamander",
                        "lava_golem",
                        "fire_imp",
                        "magma_worm",
                    ],
                    "items": [
                        "obsidian_shard",
                        "fire_crystal",
                        "lava_rock",
                        "ember",
                        "volcanic_gem",
                    ],
                },
            },
            "underwater": {
                "base_colors": [
                    (0, 119, 190),
                    (70, 130, 180),
                    (100, 149, 237),
                    (0, 191, 255),
                ],
                "detail_colors": [
                    (64, 224, 208),
                    (72, 209, 204),
                    (0, 206, 209),
                    (127, 255, 212),
                ],
                "patterns": ["coral_reef", "seaweed", "bubbles", "current", "seafloor"],
                "sprites": {
                    "creatures": [
                        "shark",
                        "octopus",
                        "jellyfish",
                        "mermaid",
                        "sea_dragon",
                    ],
                    "items": [
                        "pearl",
                        "seashell",
                        "trident",
                        "anchor",
                        "treasure_chest",
                    ],
                },
            },
            "cyberpunk": {
                "base_colors": [
                    (25, 25, 25),
                    (50, 50, 50),
                    (75, 75, 75),
                    (100, 100, 100),
                ],
                "detail_colors": [
                    (255, 0, 255),
                    (0, 255, 255),
                    (255, 255, 0),
                    (0, 255, 0),
                ],
                "patterns": [
                    "circuit",
                    "hologram",
                    "neon_grid",
                    "data_stream",
                    "glitch",
                ],
                "sprites": {
                    "creatures": [
                        "android",
                        "drone",
                        "cyber_punk",
                        "hacker",
                        "ai_construct",
                    ],
                    "items": ["chip", "battery", "usb", "holodisk", "cyberdeck"],
                },
            },
        }

        # Pattern function mapping
        self.pattern_functions = {
            "bark": self.generate_bark_texture,
            "leaves": self.generate_leaves_pattern,
            "moss": self.generate_moss_texture,
            "undergrowth": self.generate_undergrowth,
            "canopy": self.generate_canopy_pattern,
            "sand_dunes": self.generate_sand_dunes,
            "cracked_earth": self.generate_cracked_earth,
            "sandstone": self.generate_sandstone,
            "oasis": self.generate_oasis_pattern,
            "mirage": self.generate_mirage_effect,
            "ice_crystals": self.generate_ice_crystals,
            "snow_drift": self.generate_snow_drift,
            "frozen_lake": self.generate_frozen_lake,
            "blizzard": self.generate_blizzard_effect,
            "aurora": self.generate_aurora_pattern,
            "lava_flow": self.generate_lava_flow,
            "obsidian": self.generate_obsidian,
            "ash_fall": self.generate_ash_fall,
            "magma_cracks": self.generate_magma_cracks,
            "volcanic_rock": self.generate_volcanic_rock,
            "coral_reef": self.generate_coral_reef,
            "seaweed": self.generate_seaweed,
            "bubbles": self.generate_bubble_pattern,
            "current": self.generate_water_current,
            "seafloor": self.generate_seafloor,
            "circuit": self.generate_circuit_pattern,
            "hologram": self.generate_hologram,
            "neon_grid": self.generate_neon_grid,
            "data_stream": self.generate_data_stream,
            "glitch": self.generate_glitch_effect,
        }

    def generate_enhanced_texture(
        self, biome: str, pattern_name: str, art_style: str
    ) -> np.ndarray:
        """Generate texture with specific pattern and art style"""
        config = self.biome_configs[biome]
        base_color = random.choice(config["base_colors"])
        detail_colors = config["detail_colors"]

        # Get pattern generation function from mapping
        if pattern_name in self.pattern_functions:
            texture = self.pattern_functions[pattern_name](base_color, detail_colors)
        else:
            # Fallback to advanced noise-based texture
            texture = self.generate_advanced_noise_texture(base_color, detail_colors)

        # Apply art style post-processing with reduced intensity to preserve patterns
        if art_style != "realistic":  # Skip post-processing for realistic style
            texture = self.apply_art_style(texture, art_style)

        # Ensure seamless tiling
        texture = self.make_seamless(texture)

        # Enhance detail and contrast without destroying patterns
        texture = self.enhance_texture_quality(texture)

        return texture

    def generate_bark_texture(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate realistic bark texture with vertical grooves and detail"""
        texture = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        # Create vertical bark lines with variation
        for x in range(self.width):
            # Main bark column
            column_variation = pnoise2(x * 0.02, 0, octaves=3, repeatx=self.width) * 20

            for y in range(self.height):
                # Vertical noise for bark grooves
                groove_noise = pnoise2(
                    x * 0.05,
                    y * 0.2,
                    octaves=4,
                    repeatx=self.width,
                    repeaty=self.height,
                )
                detail_noise = pnoise2(
                    x * 0.1, y * 0.1, octaves=6, repeatx=self.width, repeaty=self.height
                )

                # Combine noises for bark pattern
                combined = (
                    groove_noise * 0.7 + detail_noise * 0.3 + column_variation * 0.01
                )

                # Create depth illusion
                if combined < -0.3:  # Deep grooves
                    color_factor = 0.3
                elif combined < 0:  # Medium depth
                    color_factor = 0.6
                else:  # Raised areas
                    color_factor = 1.0 + combined * 0.3

                # Apply color with variation
                r = int(base_color[0] * color_factor * (0.9 + random.random() * 0.2))
                g = int(base_color[1] * color_factor * (0.9 + random.random() * 0.2))
                b = int(base_color[2] * color_factor * (0.9 + random.random() * 0.2))

                texture[y, x] = [
                    min(255, max(0, r)),
                    min(255, max(0, g)),
                    min(255, max(0, b)),
                ]

        return texture

    def generate_leaves_pattern(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate detailed leaf pattern texture"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Add multiple layers of leaves
        num_leaves = random.randint(50, 100)

        for _ in range(num_leaves):
            leaf_x = random.randint(0, self.width)
            leaf_y = random.randint(0, self.height)
            leaf_size = random.randint(20, 60)
            leaf_angle = random.uniform(0, 2 * math.pi)
            leaf_color = random.choice(detail_colors)

            # Create leaf shape with veins
            for dx in range(-leaf_size, leaf_size):
                for dy in range(-leaf_size // 2, leaf_size // 2):
                    # Rotate coordinates
                    rx = dx * math.cos(leaf_angle) - dy * math.sin(leaf_angle)
                    ry = dx * math.sin(leaf_angle) + dy * math.cos(leaf_angle)

                    # Leaf shape equation
                    leaf_shape = abs(rx) / leaf_size + (ry / (leaf_size / 2)) ** 2

                    if leaf_shape <= 1:
                        px = int(leaf_x + rx) % self.width
                        py = int(leaf_y + ry) % self.height

                        # Add leaf detail
                        vein_pattern = (
                            math.sin(rx * 0.2) * 0.2 + math.sin(ry * 0.3) * 0.1
                        )
                        brightness = 0.8 + vein_pattern

                        r = int(leaf_color[0] * brightness)
                        g = int(leaf_color[1] * brightness)
                        b = int(leaf_color[2] * brightness)

                        # Blend with existing
                        texture[py, px] = [
                            int(texture[py, px, 0] * 0.3 + r * 0.7),
                            int(texture[py, px, 1] * 0.3 + g * 0.7),
                            int(texture[py, px, 2] * 0.3 + b * 0.7),
                        ]

        return texture

    def generate_sand_dunes(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate realistic sand dune patterns"""
        texture = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        # Create dune ridges
        for y in range(self.height):
            for x in range(self.width):
                # Large scale dunes
                dune_large = pnoise2(
                    x * 0.01,
                    y * 0.01,
                    octaves=3,
                    repeatx=self.width,
                    repeaty=self.height,
                )
                # Medium ripples
                dune_medium = pnoise2(
                    x * 0.05,
                    y * 0.05,
                    octaves=4,
                    repeatx=self.width,
                    repeaty=self.height,
                )
                # Fine sand texture
                dune_fine = pnoise2(
                    x * 0.2, y * 0.2, octaves=2, repeatx=self.width, repeaty=self.height
                )

                # Combine scales
                height = dune_large * 0.6 + dune_medium * 0.3 + dune_fine * 0.1

                # Simulate light and shadow
                light_angle = math.pi / 4  # 45 degree sun angle
                dx = (
                    pnoise2(
                        (x + 1) * 0.01,
                        y * 0.01,
                        repeatx=self.width,
                        repeaty=self.height,
                    )
                    - dune_large
                )
                dy = (
                    pnoise2(
                        x * 0.01,
                        (y + 1) * 0.01,
                        repeatx=self.width,
                        repeaty=self.height,
                    )
                    - dune_large
                )

                slope = math.sqrt(dx * dx + dy * dy)
                illumination = max(0, math.cos(math.atan2(dy, dx) - light_angle)) * (
                    1 - slope * 2
                )

                brightness = 0.7 + height * 0.2 + illumination * 0.3

                # Apply color with variation
                color_var = (
                    random.choice(detail_colors)
                    if random.random() < 0.1
                    else base_color
                )

                r = int(color_var[0] * brightness)
                g = int(color_var[1] * brightness)
                b = int(color_var[2] * brightness)

                texture[y, x] = [
                    min(255, max(0, r)),
                    min(255, max(0, g)),
                    min(255, max(0, b)),
                ]

        return texture

    def generate_ice_crystals(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate ice crystal patterns"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Generate multiple ice crystals
        num_crystals = random.randint(20, 40)

        for _ in range(num_crystals):
            cx = random.randint(50, self.width - 50)
            cy = random.randint(50, self.height - 50)
            size = random.randint(30, 80)
            branches = 6  # Hexagonal symmetry

            for branch in range(branches):
                angle = (branch / branches) * 2 * math.pi

                # Main branch
                for r in range(size):
                    x = int(cx + r * math.cos(angle))
                    y = int(cy + r * math.sin(angle))

                    if 0 <= x < self.width and 0 <= y < self.height:
                        # Crystal gradient
                        intensity = 1.0 - (r / size) * 0.5
                        crystal_color = random.choice(detail_colors)

                        # Add refraction effect
                        refraction = math.sin(r * 0.3) * 0.2 + 0.8

                        r_val = int(crystal_color[0] * intensity * refraction)
                        g_val = int(crystal_color[1] * intensity * refraction)
                        b_val = int(crystal_color[2] * intensity * refraction)

                        texture[y, x] = [r_val, g_val, b_val]

                        # Sub-branches
                        if r % 10 == 0 and r > 10:
                            sub_angle1 = angle + math.pi / 6
                            sub_angle2 = angle - math.pi / 6
                            sub_length = (size - r) // 2

                            for sub_r in range(sub_length):
                                # First sub-branch
                                sx1 = int(x + sub_r * math.cos(sub_angle1))
                                sy1 = int(y + sub_r * math.sin(sub_angle1))

                                # Second sub-branch
                                sx2 = int(x + sub_r * math.cos(sub_angle2))
                                sy2 = int(y + sub_r * math.sin(sub_angle2))

                                for sx, sy in [(sx1, sy1), (sx2, sy2)]:
                                    if 0 <= sx < self.width and 0 <= sy < self.height:
                                        sub_intensity = intensity * (
                                            1.0 - sub_r / sub_length
                                        )
                                        texture[sy, sx] = [
                                            int(crystal_color[0] * sub_intensity * 0.8),
                                            int(crystal_color[1] * sub_intensity * 0.8),
                                            int(crystal_color[2] * sub_intensity * 0.8),
                                        ]

        return texture

    def generate_lava_flow(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate animated lava flow texture"""
        texture = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                # Multiple noise octaves for lava movement
                flow1 = pnoise2(
                    x * 0.02,
                    y * 0.02 - self.seed * 0.1,
                    octaves=4,
                    repeatx=self.width,
                    repeaty=self.height,
                )
                flow2 = pnoise2(
                    x * 0.05,
                    y * 0.05 + self.seed * 0.1,
                    octaves=3,
                    repeatx=self.width,
                    repeaty=self.height,
                )
                turbulence = pnoise2(
                    x * 0.1, y * 0.1, octaves=2, repeatx=self.width, repeaty=self.height
                )

                # Combine flows
                lava_heat = (flow1 * 0.5 + flow2 * 0.3 + turbulence * 0.2 + 1) / 2

                # Temperature-based coloring
                if lava_heat > 0.8:  # Hottest - white/yellow
                    color = (255, 255, 200)
                elif lava_heat > 0.6:  # Very hot - bright orange
                    t = (lava_heat - 0.6) / 0.2
                    color = (int(255), int(200 + t * 55), int(0 + t * 200))
                elif lava_heat > 0.4:  # Hot - orange/red
                    t = (lava_heat - 0.4) / 0.2
                    color = (int(200 + t * 55), int(50 + t * 150), int(0))
                else:  # Cooled - dark red/black
                    t = lava_heat / 0.4
                    color = (int(50 + t * 150), int(0 + t * 50), int(0))

                # Add cracks in cooled lava
                crack = pnoise2(
                    x * 0.2, y * 0.2, octaves=1, repeatx=self.width, repeaty=self.height
                )
                if lava_heat < 0.5 and crack > 0.6:
                    color = (20, 20, 20)  # Dark cracks

                texture[y, x] = color

        return texture

    def generate_coral_reef(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate colorful coral reef texture"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Generate various coral formations
        num_coral_patches = random.randint(15, 30)

        for _ in range(num_coral_patches):
            coral_type = random.choice(["brain", "staghorn", "table", "soft"])
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height)
            size = random.randint(40, 100)
            coral_color = random.choice(detail_colors)

            if coral_type == "brain":
                # Brain coral pattern
                for dx in range(-size // 2, size // 2):
                    for dy in range(-size // 2, size // 2):
                        if dx * dx + dy * dy <= (size // 2) ** 2:
                            x = (cx + dx) % self.width
                            y = (cy + dy) % self.height

                            # Meandering pattern
                            pattern = math.sin(dx * 0.2) * math.cos(dy * 0.2) + pnoise2(
                                x * 0.1, y * 0.1, octaves=2
                            )

                            if pattern > 0:
                                brightness = 0.8 + pattern * 0.2
                                texture[y, x] = [
                                    int(coral_color[0] * brightness),
                                    int(coral_color[1] * brightness),
                                    int(coral_color[2] * brightness),
                                ]

            elif coral_type == "staghorn":
                # Branching coral
                branches = random.randint(5, 12)
                for b in range(branches):
                    angle = (b / branches) * 2 * math.pi + random.uniform(-0.3, 0.3)
                    length = random.randint(size // 2, size)

                    for r in range(length):
                        # Branch curves
                        curve = math.sin(r * 0.1) * 10
                        x = (
                            int(cx + r * math.cos(angle) + curve * math.sin(angle))
                            % self.width
                        )
                        y = (
                            int(cy + r * math.sin(angle) + curve * math.cos(angle))
                            % self.height
                        )

                        # Branch thickness
                        thickness = max(1, (length - r) // 10)

                        for t in range(-thickness, thickness + 1):
                            px = (x + int(t * math.sin(angle))) % self.width
                            py = (y - int(t * math.cos(angle))) % self.height

                            intensity = 1.0 - abs(t) / (thickness + 1)
                            texture[py, px] = [
                                int(coral_color[0] * intensity),
                                int(coral_color[1] * intensity),
                                int(coral_color[2] * intensity),
                            ]

        return texture

    def generate_circuit_pattern(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate cyberpunk circuit board pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Generate circuit paths
        num_main_paths = random.randint(5, 10)

        for _ in range(num_main_paths):
            # Start position
            if random.random() < 0.5:
                # Horizontal path
                start_x = 0
                start_y = random.randint(0, self.height)
                direction = (1, 0)
            else:
                # Vertical path
                start_x = random.randint(0, self.width)
                start_y = 0
                direction = (0, 1)

            x, y = start_x, start_y
            path_color = random.choice(detail_colors)
            path_width = random.randint(2, 5)

            while 0 <= x < self.width and 0 <= y < self.height:
                # Draw path segment
                for w in range(-path_width // 2, path_width // 2 + 1):
                    if direction[0] != 0:  # Horizontal
                        py = y + w
                        if 0 <= py < self.height:
                            texture[py, x] = path_color
                    else:  # Vertical
                        px = x + w
                        if 0 <= px < self.width:
                            texture[y, px] = path_color

                # Add components
                if random.random() < 0.1:
                    # Circuit node
                    node_size = random.randint(5, 10)
                    for dx in range(-node_size, node_size + 1):
                        for dy in range(-node_size, node_size + 1):
                            if dx * dx + dy * dy <= node_size * node_size:
                                px = x + dx
                                py = y + dy
                                if 0 <= px < self.width and 0 <= py < self.height:
                                    texture[py, px] = (255, 255, 255)  # Bright node

                # Move along path
                x += direction[0] * random.randint(1, 5)
                y += direction[1] * random.randint(1, 5)

                # Occasionally change direction
                if random.random() < 0.2:
                    if direction[0] != 0:  # Was horizontal
                        direction = (0, random.choice([-1, 1]))
                    else:  # Was vertical
                        direction = (random.choice([-1, 1]), 0)

        # Add glowing effect
        texture = self.add_glow_effect(texture, detail_colors)

        return texture

    def generate_advanced_noise_texture(
        self,
        base_color: Tuple[int, int, int],
        detail_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        """Generate complex multi-layered noise texture"""
        texture = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        # Multiple noise layers with different scales
        scales = [0.01, 0.05, 0.1, 0.2, 0.5]
        weights = [0.4, 0.25, 0.2, 0.1, 0.05]

        for y in range(self.height):
            for x in range(self.width):
                combined_noise = 0

                # Combine multiple octaves
                for scale, weight in zip(scales, weights):
                    noise_val = pnoise2(
                        x * scale,
                        y * scale,
                        octaves=4,
                        persistence=0.5,
                        lacunarity=2.0,
                        repeatx=self.width,
                        repeaty=self.height,
                    )
                    combined_noise += noise_val * weight

                # Normalize to 0-1
                combined_noise = (combined_noise + 1) / 2

                # Apply color mapping
                if combined_noise < 0.3:
                    color = base_color
                elif combined_noise < 0.6:
                    # Blend between base and first detail color
                    t = (combined_noise - 0.3) / 0.3
                    color = self.blend_colors(base_color, detail_colors[0], t)
                else:
                    # Use random detail color
                    color = random.choice(detail_colors)

                # Add micro-detail
                micro_noise = pnoise2(x * 0.5, y * 0.5, octaves=1)
                brightness = 0.9 + micro_noise * 0.2

                texture[y, x] = [
                    min(255, int(color[0] * brightness)),
                    min(255, int(color[1] * brightness)),
                    min(255, int(color[2] * brightness)),
                ]

        return texture

    def generate_enhanced_sprite(
        self, biome: str, sprite_type: str, art_style: str
    ) -> np.ndarray:
        """Generate high-quality sprite with specific type and art style"""
        # Create RGBA array for transparency
        sprite = np.zeros((self.width, self.height, 4), dtype=np.uint8)

        config = self.biome_configs[biome]

        if sprite_type in config["sprites"]["creatures"]:
            sprite = self.generate_creature_sprite(biome, sprite_type, art_style)
        elif sprite_type in config["sprites"]["items"]:
            sprite = self.generate_item_sprite(biome, sprite_type, art_style)
        else:
            # Generate based on game genre
            genre = random.choice(list(self.game_genres.keys()))
            if sprite_type in self.game_genres[genre]["sprites"]:
                sprite = self.generate_genre_sprite(genre, sprite_type, art_style)

        # Apply art style post-processing
        sprite = self.apply_sprite_art_style(sprite, art_style)

        return sprite

    def generate_creature_sprite(
        self, biome: str, creature_type: str, art_style: str
    ) -> np.ndarray:
        """Generate detailed creature sprite"""
        sprite = np.zeros((self.width, self.height, 4), dtype=np.uint8)

        # Get biome colors
        config = self.biome_configs[biome]
        primary_color = random.choice(config["base_colors"])
        detail_color = random.choice(config["detail_colors"])

        # Create with PIL for better drawing capabilities
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        center_x = self.width // 2
        center_y = self.height // 2

        # Generate based on creature type
        if creature_type == "wolf":
            # Body
            body_width = 120
            body_height = 80
            draw.ellipse(
                [
                    center_x - body_width // 2,
                    center_y - body_height // 2,
                    center_x + body_width // 2,
                    center_y + body_height // 2,
                ],
                fill=primary_color + (255,),
                outline=detail_color + (255,),
                width=3,
            )

            # Head
            head_size = 50
            head_x = center_x - body_width // 2 - 20
            draw.ellipse(
                [
                    head_x - head_size // 2,
                    center_y - head_size // 2,
                    head_x + head_size // 2,
                    center_y + head_size // 2,
                ],
                fill=primary_color + (255,),
                outline=detail_color + (255,),
                width=3,
            )

            # Snout
            snout_points = [
                (head_x - head_size // 2 - 20, center_y),
                (head_x - head_size // 2, center_y - 10),
                (head_x - head_size // 2, center_y + 10),
            ]
            draw.polygon(snout_points, fill=detail_color + (255,))

            # Ears
            ear_points1 = [
                (head_x - 10, center_y - head_size // 2),
                (head_x - 20, center_y - head_size // 2 - 20),
                (head_x, center_y - head_size // 2 - 15),
            ]
            ear_points2 = [
                (head_x + 10, center_y - head_size // 2),
                (head_x + 20, center_y - head_size // 2 - 20),
                (head_x, center_y - head_size // 2 - 15),
            ]
            draw.polygon(
                ear_points1, fill=primary_color + (255,), outline=detail_color + (255,)
            )
            draw.polygon(
                ear_points2, fill=primary_color + (255,), outline=detail_color + (255,)
            )

            # Legs
            leg_width = 15
            leg_height = 60
            leg_positions = [
                (center_x - body_width // 3, center_y + body_height // 2),
                (center_x + body_width // 3, center_y + body_height // 2),
            ]

            for leg_x, leg_y in leg_positions:
                draw.rectangle(
                    [
                        leg_x - leg_width // 2,
                        leg_y,
                        leg_x + leg_width // 2,
                        leg_y + leg_height,
                    ],
                    fill=primary_color + (255,),
                    outline=detail_color + (255,),
                    width=2,
                )

            # Tail
            tail_points = [
                (center_x + body_width // 2, center_y),
                (center_x + body_width // 2 + 60, center_y - 20),
                (center_x + body_width // 2 + 80, center_y - 40),
                (center_x + body_width // 2 + 70, center_y - 10),
                (center_x + body_width // 2 + 50, center_y + 10),
            ]
            draw.polygon(
                tail_points,
                fill=primary_color + (255,),
                outline=detail_color + (255,),
                width=2,
            )

            # Eyes
            eye_size = 8
            draw.ellipse(
                [
                    head_x - 15,
                    center_y - 10,
                    head_x - 15 + eye_size,
                    center_y - 10 + eye_size,
                ],
                fill=(255, 255, 0, 255),
            )
            draw.ellipse(
                [
                    head_x - 15 + 2,
                    center_y - 10 + 2,
                    head_x - 15 + 6,
                    center_y - 10 + 6,
                ],
                fill=(0, 0, 0, 255),
            )

        elif creature_type == "phoenix":
            # Flaming bird sprite
            # Body
            body_width = 100
            body_height = 120

            # Main body shape
            body_points = [
                (center_x, center_y - body_height // 2),
                (center_x - body_width // 2, center_y),
                (center_x, center_y + body_height // 2),
                (center_x + body_width // 2, center_y),
            ]
            draw.polygon(
                body_points, fill=(255, 140, 0, 255), outline=(255, 69, 0, 255), width=3
            )

            # Wings
            wing_span = 200
            wing_height = 150

            # Left wing
            left_wing_points = [
                (center_x - body_width // 2, center_y - 20),
                (center_x - wing_span // 2, center_y - wing_height // 2),
                (center_x - wing_span // 2 - 30, center_y),
                (center_x - wing_span // 2, center_y + wing_height // 2),
                (center_x - body_width // 2, center_y + 20),
            ]
            draw.polygon(
                left_wing_points,
                fill=(255, 69, 0, 255),
                outline=(178, 34, 34, 255),
                width=2,
            )

            # Right wing
            right_wing_points = [
                (center_x + body_width // 2, center_y - 20),
                (center_x + wing_span // 2, center_y - wing_height // 2),
                (center_x + wing_span // 2 + 30, center_y),
                (center_x + wing_span // 2, center_y + wing_height // 2),
                (center_x + body_width // 2, center_y + 20),
            ]
            draw.polygon(
                right_wing_points,
                fill=(255, 69, 0, 255),
                outline=(178, 34, 34, 255),
                width=2,
            )

            # Head with crest
            head_size = 40
            draw.ellipse(
                [
                    center_x - head_size // 2,
                    center_y - body_height // 2 - head_size // 2,
                    center_x + head_size // 2,
                    center_y - body_height // 2 + head_size // 2,
                ],
                fill=(255, 215, 0, 255),
                outline=(255, 140, 0, 255),
                width=2,
            )

            # Fire crest
            crest_points = [
                (center_x - 20, center_y - body_height // 2 - head_size // 2),
                (center_x - 10, center_y - body_height // 2 - head_size),
                (center_x, center_y - body_height // 2 - head_size - 10),
                (center_x + 10, center_y - body_height // 2 - head_size),
                (center_x + 20, center_y - body_height // 2 - head_size // 2),
            ]
            draw.polygon(
                crest_points, fill=(255, 0, 0, 255), outline=(255, 140, 0, 255), width=2
            )

            # Tail feathers
            tail_length = 100
            num_feathers = 7
            for i in range(num_feathers):
                angle = (i - num_feathers // 2) * 0.2
                feather_end_x = center_x + math.sin(angle) * tail_length
                feather_end_y = (
                    center_y + body_height // 2 + math.cos(angle) * tail_length
                )

                feather_points = [
                    (center_x, center_y + body_height // 2),
                    (feather_end_x - 10, feather_end_y),
                    (feather_end_x, feather_end_y + 10),
                    (feather_end_x + 10, feather_end_y),
                ]

                # Gradient effect for feathers
                if i % 2 == 0:
                    feather_color = (255, 140, 0, 200)
                else:
                    feather_color = (255, 69, 0, 200)

                draw.polygon(
                    feather_points, fill=feather_color, outline=(178, 34, 34, 255)
                )

        elif creature_type == "mermaid":
            # Upper body (human-like)
            torso_width = 60
            torso_height = 80

            # Torso
            draw.ellipse(
                [
                    center_x - torso_width // 2,
                    center_y - torso_height,
                    center_x + torso_width // 2,
                    center_y,
                ],
                fill=(255, 218, 185, 255),
                outline=(205, 133, 63, 255),
                width=2,
            )

            # Head
            head_size = 35
            draw.ellipse(
                [
                    center_x - head_size // 2,
                    center_y - torso_height - head_size,
                    center_x + head_size // 2,
                    center_y - torso_height,
                ],
                fill=(255, 228, 196, 255),
                outline=(205, 133, 63, 255),
                width=2,
            )

            # Hair (flowing)
            hair_color = (70, 130, 180, 255)
            for i in range(12):
                angle = (i / 12) * math.pi - math.pi / 2
                hair_length = random.randint(40, 80)
                end_x = center_x + math.cos(angle) * hair_length
                end_y = (
                    center_y
                    - torso_height
                    - head_size // 2
                    + math.sin(angle) * hair_length
                )

                # Wavy hair strands
                control_x = (
                    center_x
                    + math.cos(angle) * hair_length // 2
                    + random.randint(-10, 10)
                )
                control_y = (
                    center_y
                    - torso_height
                    - head_size // 2
                    + math.sin(angle) * hair_length // 2
                )

                # Draw hair strand as series of lines
                points = []
                for t in range(11):
                    t = t / 10
                    x = (
                        (1 - t) ** 2 * center_x
                        + 2 * (1 - t) * t * control_x
                        + t**2 * end_x
                    )
                    y = (
                        (1 - t) ** 2 * (center_y - torso_height - head_size // 2)
                        + 2 * (1 - t) * t * control_y
                        + t**2 * end_y
                    )
                    points.append((x, y))

                draw.line(points, fill=hair_color, width=3)

            # Arms
            arm_length = 50
            draw.line(
                [
                    (center_x - torso_width // 2, center_y - torso_height // 2),
                    (
                        center_x - torso_width // 2 - arm_length,
                        center_y - torso_height // 2 + 20,
                    ),
                ],
                fill=(255, 218, 185, 255),
                width=8,
            )
            draw.line(
                [
                    (center_x + torso_width // 2, center_y - torso_height // 2),
                    (
                        center_x + torso_width // 2 + arm_length,
                        center_y - torso_height // 2 + 20,
                    ),
                ],
                fill=(255, 218, 185, 255),
                width=8,
            )

            # Fish tail
            tail_width = 80
            tail_length = 120

            # Main tail shape
            tail_points = [
                (center_x - torso_width // 2, center_y),
                (center_x - tail_width // 2, center_y + tail_length // 2),
                (center_x, center_y + tail_length),
                (center_x + tail_width // 2, center_y + tail_length // 2),
                (center_x + torso_width // 2, center_y),
            ]
            draw.polygon(
                tail_points,
                fill=(0, 191, 255, 255),
                outline=(0, 119, 190, 255),
                width=3,
            )

            # Scales pattern
            scale_size = 10
            for y in range(center_y, center_y + tail_length, scale_size):
                for x in range(
                    center_x - tail_width // 2, center_x + tail_width // 2, scale_size
                ):
                    # Check if point is inside tail
                    rel_y = (y - center_y) / tail_length
                    max_x_offset = tail_width // 2 * (1 - rel_y)

                    if abs(x - center_x) <= max_x_offset:
                        draw.arc(
                            [
                                x - scale_size // 2,
                                y - scale_size // 2,
                                x + scale_size // 2,
                                y + scale_size // 2,
                            ],
                            start=0,
                            end=180,
                            fill=(0, 150, 190, 255),
                            width=1,
                        )

            # Tail fin
            fin_points = [
                (center_x - 40, center_y + tail_length),
                (center_x - 60, center_y + tail_length + 40),
                (center_x, center_y + tail_length + 20),
                (center_x + 60, center_y + tail_length + 40),
                (center_x + 40, center_y + tail_length),
            ]
            draw.polygon(
                fin_points, fill=(0, 150, 190, 255), outline=(0, 100, 150, 255), width=2
            )

        # Convert back to numpy array
        sprite = np.array(img)

        return sprite

    def generate_item_sprite(
        self, biome: str, item_type: str, art_style: str
    ) -> np.ndarray:
        """Generate detailed item sprite"""
        sprite = np.zeros((self.width, self.height, 4), dtype=np.uint8)

        # Get biome colors
        config = self.biome_configs[biome]
        primary_color = random.choice(config["base_colors"])
        detail_color = random.choice(config["detail_colors"])

        # Create with PIL
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        center_x = self.width // 2
        center_y = self.height // 2

        if "sword" in item_type:
            # Blade
            blade_width = 20
            blade_length = 180

            # Main blade
            blade_points = [
                (center_x - blade_width // 2, center_y + 80),
                (center_x - blade_width // 2, center_y - blade_length // 2),
                (center_x, center_y - blade_length // 2 - 20),  # Tip
                (center_x + blade_width // 2, center_y - blade_length // 2),
                (center_x + blade_width // 2, center_y + 80),
            ]

            # Metallic gradient effect
            blade_color = (192, 192, 192, 255)
            highlight_color = (255, 255, 255, 255)
            shadow_color = (128, 128, 128, 255)

            draw.polygon(blade_points, fill=blade_color)

            # Blade highlights
            highlight_points = [
                (center_x - blade_width // 4, center_y + 80),
                (center_x - blade_width // 4, center_y - blade_length // 2 + 10),
                (center_x, center_y - blade_length // 2 - 15),
                (center_x + 2, center_y - blade_length // 2 - 15),
                (center_x + 2, center_y + 80),
            ]
            draw.polygon(highlight_points, fill=highlight_color)

            # Guard
            guard_width = 80
            guard_height = 15
            draw.rectangle(
                [
                    center_x - guard_width // 2,
                    center_y + 65,
                    center_x + guard_width // 2,
                    center_y + 65 + guard_height,
                ],
                fill=detail_color + (255,),
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Handle
            handle_width = 25
            handle_length = 60
            draw.rectangle(
                [
                    center_x - handle_width // 2,
                    center_y + 80,
                    center_x + handle_width // 2,
                    center_y + 80 + handle_length,
                ],
                fill=primary_color + (255,),
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Handle wrap detail
            wrap_spacing = 8
            for y in range(center_y + 80, center_y + 80 + handle_length, wrap_spacing):
                draw.line(
                    [
                        (center_x - handle_width // 2, y),
                        (center_x + handle_width // 2, y),
                    ],
                    fill=(0, 0, 0, 180),
                    width=2,
                )

            # Pommel
            pommel_size = 30
            draw.ellipse(
                [
                    center_x - pommel_size // 2,
                    center_y + 80 + handle_length,
                    center_x + pommel_size // 2,
                    center_y + 80 + handle_length + pommel_size,
                ],
                fill=detail_color + (255,),
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Gem in pommel
            gem_size = 15
            gem_color = self.get_biome_gem_color(biome)
            draw.ellipse(
                [
                    center_x - gem_size // 2,
                    center_y + 80 + handle_length + pommel_size // 2 - gem_size // 2,
                    center_x + gem_size // 2,
                    center_y + 80 + handle_length + pommel_size // 2 + gem_size // 2,
                ],
                fill=gem_color + (255,),
            )

        elif "staff" in item_type:
            # Staff shaft
            shaft_width = 15
            shaft_length = 200

            # Wooden texture
            for i in range(shaft_length):
                y = center_y - shaft_length // 2 + i
                # Wood grain effect
                grain_offset = math.sin(i * 0.1) * 2

                draw.line(
                    [
                        (center_x - shaft_width // 2 + grain_offset, y),
                        (center_x + shaft_width // 2 + grain_offset, y),
                    ],
                    fill=primary_color + (255,),
                )

            # Staff outline
            draw.rectangle(
                [
                    center_x - shaft_width // 2,
                    center_y - shaft_length // 2,
                    center_x + shaft_width // 2,
                    center_y + shaft_length // 2,
                ],
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Ornamental head
            head_size = 60
            head_y = center_y - shaft_length // 2 - 30

            # Crystal/orb holder
            holder_points = [
                (center_x - 20, head_y + 20),
                (center_x - 30, head_y),
                (center_x - 20, head_y - 20),
                (center_x, head_y - 30),
                (center_x + 20, head_y - 20),
                (center_x + 30, head_y),
                (center_x + 20, head_y + 20),
                (center_x, head_y + 10),
            ]
            draw.polygon(
                holder_points,
                fill=detail_color + (255,),
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Magical orb
            orb_color = self.get_biome_magic_color(biome)
            # Outer glow
            for i in range(5, 0, -1):
                glow_alpha = 50 + i * 20
                glow_size = head_size // 2 + i * 5
                draw.ellipse(
                    [
                        center_x - glow_size,
                        head_y - glow_size,
                        center_x + glow_size,
                        head_y + glow_size,
                    ],
                    fill=orb_color + (glow_alpha,),
                )

            # Inner orb
            draw.ellipse(
                [
                    center_x - head_size // 2,
                    head_y - head_size // 2,
                    center_x + head_size // 2,
                    head_y + head_size // 2,
                ],
                fill=orb_color + (255,),
                outline=(255, 255, 255, 255),
                width=2,
            )

            # Energy swirls
            for i in range(3):
                angle = (i / 3) * 2 * math.pi
                swirl_x = center_x + math.cos(angle) * 15
                swirl_y = head_y + math.sin(angle) * 15
                draw.ellipse(
                    [swirl_x - 5, swirl_y - 5, swirl_x + 5, swirl_y + 5],
                    fill=(255, 255, 255, 200),
                )

        elif "gem" in item_type or "crystal" in item_type:
            # Multi-faceted gem
            gem_size = 80

            # Define facets
            top_point = (center_x, center_y - gem_size)
            left_point = (center_x - gem_size, center_y)
            right_point = (center_x + gem_size, center_y)
            bottom_point = (center_x, center_y + gem_size // 2)

            # Get gem color based on biome
            gem_color = self.get_biome_gem_color(biome)

            # Draw facets with shading
            facets = [
                # Top facets
                (
                    [
                        top_point,
                        (center_x - gem_size // 2, center_y - gem_size // 2),
                        (center_x, center_y),
                    ],
                    1.2,
                ),
                (
                    [
                        top_point,
                        (center_x + gem_size // 2, center_y - gem_size // 2),
                        (center_x, center_y),
                    ],
                    1.0,
                ),
                # Side facets
                (
                    [
                        left_point,
                        (center_x - gem_size // 2, center_y - gem_size // 2),
                        (center_x, center_y),
                    ],
                    0.8,
                ),
                (
                    [
                        right_point,
                        (center_x + gem_size // 2, center_y - gem_size // 2),
                        (center_x, center_y),
                    ],
                    0.9,
                ),
                # Bottom facets
                ([left_point, (center_x, center_y), bottom_point], 0.7),
                ([right_point, (center_x, center_y), bottom_point], 0.85),
                # Center bottom
                ([(center_x, center_y), bottom_point, left_point], 0.6),
                ([(center_x, center_y), bottom_point, right_point], 0.75),
            ]

            for facet_points, brightness in facets:
                facet_color = tuple(int(c * brightness) for c in gem_color)
                draw.polygon(
                    facet_points, fill=facet_color + (255,), outline=(0, 0, 0, 100)
                )

            # Add sparkles
            num_sparkles = 8
            for _ in range(num_sparkles):
                sparkle_x = center_x + random.randint(-gem_size, gem_size)
                sparkle_y = center_y + random.randint(-gem_size, gem_size // 2)

                # Check if inside gem bounds
                if (
                    abs(sparkle_x - center_x) / gem_size
                    + abs(sparkle_y - center_y) / (gem_size * 0.75)
                    < 1
                ):
                    # Draw sparkle
                    sparkle_size = random.randint(2, 5)
                    draw.line(
                        [
                            (sparkle_x - sparkle_size, sparkle_y),
                            (sparkle_x + sparkle_size, sparkle_y),
                        ],
                        fill=(255, 255, 255, 255),
                        width=2,
                    )
                    draw.line(
                        [
                            (sparkle_x, sparkle_y - sparkle_size),
                            (sparkle_x, sparkle_y + sparkle_size),
                        ],
                        fill=(255, 255, 255, 255),
                        width=2,
                    )

        # Convert back to numpy array
        sprite = np.array(img)

        return sprite

    def generate_genre_sprite(
        self, genre: str, sprite_type: str, art_style: str
    ) -> np.ndarray:
        """Generate sprite based on game genre"""
        sprite = np.zeros((self.width, self.height, 4), dtype=np.uint8)

        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        center_x = self.width // 2
        center_y = self.height // 2

        if genre == "platformer" and sprite_type == "hero":
            # Platformer hero with distinctive silhouette
            # Head
            head_size = 60
            draw.ellipse(
                [
                    center_x - head_size // 2,
                    center_y - 100 - head_size // 2,
                    center_x + head_size // 2,
                    center_y - 100 + head_size // 2,
                ],
                fill=(255, 220, 177, 255),
                outline=(0, 0, 0, 255),
                width=3,
            )

            # Body
            body_width = 50
            body_height = 70
            draw.rectangle(
                [
                    center_x - body_width // 2,
                    center_y - 100 + head_size // 2,
                    center_x + body_width // 2,
                    center_y - 100 + head_size // 2 + body_height,
                ],
                fill=(0, 100, 200, 255),
                outline=(0, 0, 0, 255),
                width=3,
            )

            # Arms
            arm_width = 15
            arm_length = 50
            # Left arm
            draw.rectangle(
                [
                    center_x - body_width // 2 - arm_width,
                    center_y - 70,
                    center_x - body_width // 2,
                    center_y - 70 + arm_length,
                ],
                fill=(255, 220, 177, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )
            # Right arm
            draw.rectangle(
                [
                    center_x + body_width // 2,
                    center_y - 70,
                    center_x + body_width // 2 + arm_width,
                    center_y - 70 + arm_length,
                ],
                fill=(255, 220, 177, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Legs
            leg_width = 20
            leg_length = 60
            # Left leg
            draw.rectangle(
                [
                    center_x - leg_width - 5,
                    center_y + body_height - 30,
                    center_x - 5,
                    center_y + body_height - 30 + leg_length,
                ],
                fill=(50, 50, 150, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )
            # Right leg
            draw.rectangle(
                [
                    center_x + 5,
                    center_y + body_height - 30,
                    center_x + leg_width + 5,
                    center_y + body_height - 30 + leg_length,
                ],
                fill=(50, 50, 150, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Shoes
            shoe_width = 30
            shoe_height = 15
            draw.ellipse(
                [
                    center_x - leg_width - 5 - 5,
                    center_y + body_height + leg_length - 30,
                    center_x - 5 + 5,
                    center_y + body_height + leg_length - 30 + shoe_height,
                ],
                fill=(139, 69, 19, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )
            draw.ellipse(
                [
                    center_x + 5 - 5,
                    center_y + body_height + leg_length - 30,
                    center_x + leg_width + 5 + 5,
                    center_y + body_height + leg_length - 30 + shoe_height,
                ],
                fill=(139, 69, 19, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )

            # Face features
            # Eyes
            eye_size = 10
            draw.ellipse(
                [
                    center_x - 15,
                    center_y - 100,
                    center_x - 15 + eye_size,
                    center_y - 100 + eye_size,
                ],
                fill=(255, 255, 255, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )
            draw.ellipse(
                [
                    center_x + 5,
                    center_y - 100,
                    center_x + 5 + eye_size,
                    center_y - 100 + eye_size,
                ],
                fill=(255, 255, 255, 255),
                outline=(0, 0, 0, 255),
                width=2,
            )
            # Pupils
            draw.ellipse(
                [center_x - 12, center_y - 97, center_x - 12 + 4, center_y - 97 + 4],
                fill=(0, 0, 0, 255),
            )
            draw.ellipse(
                [center_x + 8, center_y - 97, center_x + 8 + 4, center_y - 97 + 4],
                fill=(0, 0, 0, 255),
            )

            # Smile
            draw.arc(
                [center_x - 20, center_y - 90, center_x + 20, center_y - 70],
                start=0,
                end=180,
                fill=(0, 0, 0, 255),
                width=2,
            )

        elif genre == "rpg" and sprite_type == "warrior":
            # Armored warrior
            # Helmet
            helmet_width = 70
            helmet_height = 80
            draw.ellipse(
                [
                    center_x - helmet_width // 2,
                    center_y - 120,
                    center_x + helmet_width // 2,
                    center_y - 120 + helmet_height,
                ],
                fill=(192, 192, 192, 255),
                outline=(128, 128, 128, 255),
                width=3,
            )

            # Visor slit
            draw.rectangle(
                [center_x - 20, center_y - 90, center_x + 20, center_y - 80],
                fill=(0, 0, 0, 255),
            )

            # Armor body
            armor_width = 80
            armor_height = 100
            # Chest plate
            chest_points = [
                (center_x - armor_width // 2, center_y - 40),
                (center_x - armor_width // 2 + 10, center_y - 50),
                (center_x + armor_width // 2 - 10, center_y - 50),
                (center_x + armor_width // 2, center_y - 40),
                (center_x + armor_width // 2, center_y + armor_height // 2),
                (center_x, center_y + armor_height // 2 + 10),
                (center_x - armor_width // 2, center_y + armor_height // 2),
            ]
            draw.polygon(
                chest_points,
                fill=(160, 160, 160, 255),
                outline=(96, 96, 96, 255),
                width=3,
            )

            # Shoulder pads
            pad_size = 35
            # Left shoulder
            draw.ellipse(
                [
                    center_x - armor_width // 2 - 20,
                    center_y - 50,
                    center_x - armor_width // 2 + pad_size - 20,
                    center_y - 50 + pad_size,
                ],
                fill=(192, 192, 192, 255),
                outline=(128, 128, 128, 255),
                width=3,
            )
            # Right shoulder
            draw.ellipse(
                [
                    center_x + armor_width // 2 - pad_size + 20,
                    center_y - 50,
                    center_x + armor_width // 2 + 20,
                    center_y - 50 + pad_size,
                ],
                fill=(192, 192, 192, 255),
                outline=(128, 128, 128, 255),
                width=3,
            )

            # Arms (chain mail)
            arm_width = 25
            for y in range(center_y - 20, center_y + 40, 4):
                # Left arm
                draw.line(
                    [
                        (center_x - armor_width // 2 - arm_width, y),
                        (center_x - armor_width // 2, y),
                    ],
                    fill=(128, 128, 128, 255),
                    width=2,
                )
                # Right arm
                draw.line(
                    [
                        (center_x + armor_width // 2, y),
                        (center_x + armor_width // 2 + arm_width, y),
                    ],
                    fill=(128, 128, 128, 255),
                    width=2,
                )

            # Sword in hand
            sword_length = 100
            sword_angle = math.pi / 6  # 30 degrees
            sword_start_x = center_x + armor_width // 2 + arm_width
            sword_start_y = center_y + 20
            sword_end_x = sword_start_x + sword_length * math.cos(sword_angle)
            sword_end_y = sword_start_y - sword_length * math.sin(sword_angle)

            # Sword blade
            draw.line(
                [(sword_start_x, sword_start_y), (sword_end_x, sword_end_y)],
                fill=(220, 220, 220, 255),
                width=8,
            )
            # Sword handle
            handle_length = 20
            handle_end_x = sword_start_x - handle_length * math.cos(sword_angle)
            handle_end_y = sword_start_y + handle_length * math.sin(sword_angle)
            draw.line(
                [(sword_start_x, sword_start_y), (handle_end_x, handle_end_y)],
                fill=(139, 69, 19, 255),
                width=10,
            )

            # Legs (armored)
            leg_width = 30
            leg_height = 80
            # Left leg
            draw.rectangle(
                [
                    center_x - leg_width - 10,
                    center_y + armor_height // 2,
                    center_x - 10,
                    center_y + armor_height // 2 + leg_height,
                ],
                fill=(160, 160, 160, 255),
                outline=(96, 96, 96, 255),
                width=2,
            )
            # Right leg
            draw.rectangle(
                [
                    center_x + 10,
                    center_y + armor_height // 2,
                    center_x + leg_width + 10,
                    center_y + armor_height // 2 + leg_height,
                ],
                fill=(160, 160, 160, 255),
                outline=(96, 96, 96, 255),
                width=2,
            )

        elif genre == "shooter" and sprite_type == "bullet":
            # Projectile with motion blur
            bullet_length = 40
            bullet_width = 10

            # Motion blur effect
            for i in range(5):
                blur_alpha = 255 - i * 40
                blur_offset = i * 15

                # Bullet shape
                bullet_points = [
                    (
                        center_x - blur_offset - bullet_length // 2,
                        center_y - bullet_width // 2,
                    ),
                    (
                        center_x - blur_offset + bullet_length // 2 - 5,
                        center_y - bullet_width // 2,
                    ),
                    (center_x - blur_offset + bullet_length // 2, center_y),
                    (
                        center_x - blur_offset + bullet_length // 2 - 5,
                        center_y + bullet_width // 2,
                    ),
                    (
                        center_x - blur_offset - bullet_length // 2,
                        center_y + bullet_width // 2,
                    ),
                ]

                draw.polygon(bullet_points, fill=(255, 215, 0, blur_alpha))

            # Main bullet
            main_bullet_points = [
                (center_x - bullet_length // 2, center_y - bullet_width // 2),
                (center_x + bullet_length // 2 - 5, center_y - bullet_width // 2),
                (center_x + bullet_length // 2, center_y),
                (center_x + bullet_length // 2 - 5, center_y + bullet_width // 2),
                (center_x - bullet_length // 2, center_y + bullet_width // 2),
            ]
            draw.polygon(
                main_bullet_points,
                fill=(255, 255, 0, 255),
                outline=(255, 140, 0, 255),
                width=2,
            )

            # Hot tip effect
            tip_glow_size = 15
            for i in range(3):
                glow_alpha = 100 - i * 30
                draw.ellipse(
                    [
                        center_x + bullet_length // 2 - tip_glow_size + i * 5,
                        center_y - tip_glow_size + i * 5,
                        center_x + bullet_length // 2 + tip_glow_size - i * 5,
                        center_y + tip_glow_size - i * 5,
                    ],
                    fill=(255, 100, 0, glow_alpha),
                )

        sprite = np.array(img)
        return sprite

    def apply_art_style(self, texture: np.ndarray, art_style: str) -> np.ndarray:
        """Apply specific art style post-processing to texture"""
        if art_style == "pixel_art":
            # Reduce to pixel art resolution
            style_config = self.art_styles["pixel_art"]
            pixel_size = random.choice(style_config["pixel_sizes"])

            if pixel_size > 1:
                # Downsample and upsample for pixelated effect
                small_width = self.width // pixel_size
                small_height = self.height // pixel_size

                # Create small version
                img = Image.fromarray(texture.astype(np.uint8))
                small_img = img.resize((small_width, small_height), Image.NEAREST)

                # Apply color palette
                palette_name = random.choice(
                    list(style_config["color_palettes"].keys())
                )
                palette = style_config["color_palettes"][palette_name]

                # Quantize to palette
                pixels = np.array(small_img)
                for y in range(small_height):
                    for x in range(small_width):
                        pixel = pixels[y, x]
                        # Find closest palette color
                        min_dist = float("inf")
                        closest_color = palette[0]

                        for palette_color in palette:
                            dist = sum(
                                (p - c) ** 2 for p, c in zip(pixel[:3], palette_color)
                            )
                            if dist < min_dist:
                                min_dist = dist
                                closest_color = palette_color

                        pixels[y, x] = (
                            closest_color + (255,) if len(pixel) > 3 else closest_color
                        )

                # Scale back up
                result_img = Image.fromarray(pixels.astype(np.uint8))
                result_img = result_img.resize((self.width, self.height), Image.NEAREST)
                texture = np.array(result_img)

        elif art_style == "hand_drawn":
            # Add sketch-like effects
            img = Image.fromarray(texture.astype(np.uint8))

            # Edge detection for sketch lines
            edges = img.filter(ImageFilter.FIND_EDGES)
            edges = ImageEnhance.Contrast(edges).enhance(2.0)

            # Combine with original
            img = Image.blend(img, edges, 0.3)

            # Add texture
            texture_overlay = self.create_paper_texture()
            img = Image.blend(img, Image.fromarray(texture_overlay), 0.1)

            texture = np.array(img)

        elif art_style == "cartoon":
            # Cel shading effect
            img = Image.fromarray(texture.astype(np.uint8))

            # Posterize colors
            img = img.convert("RGB")
            pixels = np.array(img)

            # Reduce color levels
            levels = 4
            pixels = (pixels // (256 // levels)) * (256 // levels)

            # Add black outlines
            edges = Image.fromarray(pixels).filter(ImageFilter.FIND_EDGES)
            edges = edges.point(lambda x: 0 if x > 30 else 255)

            # Combine
            result = Image.fromarray(pixels)
            edges_array = np.array(edges.convert("RGBA"))
            result_array = np.array(result.convert("RGBA"))

            # Apply edges
            mask = edges_array[:, :, 0] < 128
            result_array[mask] = [0, 0, 0, 255]

            texture = result_array

        return texture

    def apply_sprite_art_style(self, sprite: np.ndarray, art_style: str) -> np.ndarray:
        """Apply art style to sprite with transparency preservation"""
        # Preserve alpha channel
        alpha = sprite[:, :, 3] if sprite.shape[2] == 4 else None

        # Apply style to RGB channels
        if alpha is not None:
            rgb = sprite[:, :, :3]
            rgb_styled = self.apply_art_style(rgb, art_style)

            # Recombine with alpha
            if rgb_styled.shape[2] == 3:
                sprite = np.dstack([rgb_styled, alpha])
            else:
                sprite = rgb_styled
                sprite[:, :, 3] = alpha
        else:
            sprite = self.apply_art_style(sprite, art_style)

        return sprite

    def make_seamless(self, texture: np.ndarray) -> np.ndarray:
        """Make texture seamlessly tileable with minimal pattern disruption"""
        img = Image.fromarray(texture.astype(np.uint8))
        width, height = img.size

        # Create seamless by blending edges
        blend_size = min(width, height) // 16  # Reduced from //8 for less blending

        # Horizontal blending
        left_strip = img.crop((0, 0, blend_size, height))
        right_strip = img.crop((width - blend_size, 0, width, height))

        for x in range(blend_size):
            alpha = x / blend_size
            for y in range(height):
                left_pixel = left_strip.getpixel((x, y))
                right_pixel = right_strip.getpixel((x, y))

                # Smoother blending function
                blend_alpha = 0.5 * (1 - math.cos(alpha * math.pi))
                blended = tuple(
                    int(l * (1 - blend_alpha) + r * blend_alpha)
                    for l, r in zip(left_pixel, right_pixel)
                )
                img.putpixel((x, y), blended)
                img.putpixel((width - blend_size + x, y), blended)

        # Vertical blending
        top_strip = img.crop((0, 0, width, blend_size))
        bottom_strip = img.crop((0, height - blend_size, width, height))

        for y in range(blend_size):
            alpha = y / blend_size
            for x in range(width):
                top_pixel = top_strip.getpixel((x, y))
                bottom_pixel = bottom_strip.getpixel((x, y))

                # Smoother blending function
                blend_alpha = 0.5 * (1 - math.cos(alpha * math.pi))
                blended = tuple(
                    int(t * (1 - blend_alpha) + b * blend_alpha)
                    for t, b in zip(top_pixel, bottom_pixel)
                )
                img.putpixel((x, y), blended)
                img.putpixel((x, height - blend_size + y), blended)

        return np.array(img)

    def enhance_texture_quality(self, texture: np.ndarray) -> np.ndarray:
        """Enhance texture detail and contrast without destroying patterns"""
        img = Image.fromarray(texture.astype(np.uint8))

        # Mild contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # Reduced from 1.3

        # Mild sharpness enhancement
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.05)  # Reduced from 1.2

        # Add very subtle noise for detail
        pixels = np.array(img)
        noise = np.random.normal(0, 2, pixels.shape)  # Reduced from 5
        pixels = np.clip(pixels + noise, 0, 255)

        return pixels.astype(np.uint8)

    def create_paper_texture(self) -> np.ndarray:
        """Create paper texture overlay"""
        texture = np.full((self.width, self.height, 3), 240, dtype=np.uint8)

        # Add fiber pattern
        for _ in range(1000):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = x1 + random.randint(-50, 50)
            y2 = y1 + random.randint(-50, 50)

            color = random.randint(200, 250)

            # Simple line algorithm
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps > 0:
                for i in range(steps):
                    t = i / steps
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))

                    if 0 <= x < self.width and 0 <= y < self.height:
                        texture[y, x] = [color, color, color]

        return texture

    def add_glow_effect(
        self, texture: np.ndarray, glow_colors: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Add glowing effect to bright pixels"""
        img = Image.fromarray(texture.astype(np.uint8))

        # Create glow mask from bright pixels
        pixels = np.array(img)
        brightness = np.mean(pixels, axis=2)
        bright_mask = brightness > 200

        # Create glow layer
        glow_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow_layer)

        # Add glow around bright pixels
        for y in range(self.height):
            for x in range(self.width):
                if bright_mask[y, x]:
                    glow_color = random.choice(glow_colors)
                    for radius in range(10, 0, -2):
                        alpha = int(255 * (1 - radius / 10) * 0.5)
                        draw.ellipse(
                            [x - radius, y - radius, x + radius, y + radius],
                            fill=glow_color + (alpha,),
                        )

        # Blend glow with original
        result = Image.alpha_composite(img.convert("RGBA"), glow_layer)

        return np.array(result.convert("RGB"))

    def blend_colors(
        self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], t: float
    ) -> Tuple[int, int, int]:
        """Blend two colors with parameter t (0-1)"""
        return tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))

    def get_biome_gem_color(self, biome: str) -> Tuple[int, int, int]:
        """Get appropriate gem color for biome"""
        gem_colors = {
            "forest": (34, 139, 34),  # Emerald
            "desert": (255, 215, 0),  # Gold
            "snow": (135, 206, 235),  # Ice blue
            "volcanic": (220, 20, 60),  # Ruby
            "underwater": (64, 224, 208),  # Turquoise
            "cyberpunk": (138, 43, 226),  # Neon purple
        }
        return gem_colors.get(biome, (192, 192, 192))

    def get_biome_magic_color(self, biome: str) -> Tuple[int, int, int]:
        """Get appropriate magic color for biome"""
        magic_colors = {
            "forest": (50, 205, 50),  # Nature green
            "desert": (255, 140, 0),  # Solar orange
            "snow": (0, 191, 255),  # Frost blue
            "volcanic": (255, 69, 0),  # Fire red
            "underwater": (0, 206, 209),  # Aqua
            "cyberpunk": (255, 0, 255),  # Cyber magenta
        }
        return magic_colors.get(biome, (148, 0, 211))

    # Add missing pattern generation methods
    def generate_moss_texture(self, base_color, detail_colors):
        """Generate moss texture pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create patchy moss growth
        num_patches = random.randint(30, 50)
        for _ in range(num_patches):
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height)
            radius = random.randint(20, 60)
            moss_color = random.choice(detail_colors)

            for dx in range(-radius, radius):
                for dy in range(-radius, radius):
                    if dx * dx + dy * dy <= radius * radius:
                        x = (cx + dx) % self.width
                        y = (cy + dy) % self.height

                        # Organic edge with noise
                        edge_noise = pnoise2(x * 0.1, y * 0.1, octaves=2)
                        if (
                            dx * dx + dy * dy
                            <= (radius * (0.8 + edge_noise * 0.2)) ** 2
                        ):
                            # Blend with existing
                            alpha = 1 - (dx * dx + dy * dy) / (radius * radius)
                            texture[y, x] = [
                                int(
                                    texture[y, x, 0] * (1 - alpha)
                                    + moss_color[0] * alpha
                                ),
                                int(
                                    texture[y, x, 1] * (1 - alpha)
                                    + moss_color[1] * alpha
                                ),
                                int(
                                    texture[y, x, 2] * (1 - alpha)
                                    + moss_color[2] * alpha
                                ),
                            ]

        return texture

    def generate_undergrowth(self, base_color, detail_colors):
        """Generate forest undergrowth pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Add various plant elements
        num_plants = random.randint(40, 80)
        for _ in range(num_plants):
            plant_type = random.choice(["fern", "grass", "shrub"])
            px = random.randint(0, self.width)
            py = random.randint(0, self.height)
            plant_color = random.choice(detail_colors)

            if plant_type == "fern":
                # Draw fern fronds
                num_fronds = random.randint(5, 8)
                for i in range(num_fronds):
                    angle = (i / num_fronds) * math.pi - math.pi / 2
                    length = random.randint(30, 60)

                    for r in range(length):
                        x = int(px + r * math.cos(angle)) % self.width
                        y = int(py + r * math.sin(angle)) % self.height

                        # Fern leaflets
                        if r % 5 == 0:
                            for side in [-1, 1]:
                                leaflet_angle = angle + side * math.pi / 4
                                leaflet_length = (length - r) // 3

                                for lr in range(leaflet_length):
                                    lx = (
                                        int(x + lr * math.cos(leaflet_angle))
                                        % self.width
                                    )
                                    ly = (
                                        int(y + lr * math.sin(leaflet_angle))
                                        % self.height
                                    )
                                    texture[ly, lx] = plant_color

                        texture[y, x] = plant_color

            elif plant_type == "grass":
                # Draw grass blades
                num_blades = random.randint(5, 15)
                for _ in range(num_blades):
                    blade_curve = random.uniform(-0.5, 0.5)
                    blade_height = random.randint(20, 40)

                    for h in range(blade_height):
                        curve_offset = int(blade_curve * h)
                        x = (px + curve_offset) % self.width
                        y = (py - h) % self.height

                        if 0 <= y < self.height:
                            texture[y, x] = plant_color

        return texture

    def generate_canopy_pattern(self, base_color, detail_colors):
        """Generate tree canopy pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create overlapping circular canopy sections
        num_canopies = random.randint(10, 20)
        for _ in range(num_canopies):
            cx = random.randint(0, self.width)
            cy = random.randint(0, self.height)
            radius = random.randint(50, 100)
            canopy_color = random.choice(detail_colors)

            # Create leafy texture
            for angle in np.linspace(0, 2 * math.pi, 36):
                for r in range(radius // 2, radius):
                    # Add variation
                    r_var = r + random.randint(-10, 10)
                    x = int(cx + r_var * math.cos(angle)) % self.width
                    y = int(cy + r_var * math.sin(angle)) % self.height

                    # Gradient from center
                    intensity = 1 - (r / radius) * 0.5
                    color = [int(c * intensity) for c in canopy_color]

                    # Draw small leaf cluster
                    cluster_size = random.randint(3, 8)
                    for dx in range(-cluster_size // 2, cluster_size // 2):
                        for dy in range(-cluster_size // 2, cluster_size // 2):
                            if dx * dx + dy * dy <= (cluster_size // 2) ** 2:
                                px = (x + dx) % self.width
                                py = (y + dy) % self.height
                                texture[py, px] = color

        return texture

    def generate_cracked_earth(self, base_color, detail_colors):
        """Generate cracked desert earth pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create Voronoi-like crack pattern
        num_centers = random.randint(20, 40)
        centers = [
            (random.randint(0, self.width), random.randint(0, self.height))
            for _ in range(num_centers)
        ]

        # Fill regions
        for y in range(self.height):
            for x in range(self.width):
                # Find nearest center
                min_dist = float("inf")
                second_min_dist = float("inf")

                for cx, cy in centers:
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist < min_dist:
                        second_min_dist = min_dist
                        min_dist = dist
                    elif dist < second_min_dist:
                        second_min_dist = dist

                # Create cracks at boundaries
                if second_min_dist - min_dist < 3:
                    texture[y, x] = [40, 30, 20]  # Dark crack color
                else:
                    # Add texture variation
                    noise = pnoise2(x * 0.05, y * 0.05, octaves=2)
                    brightness = 0.8 + noise * 0.4
                    color = random.choice([base_color] + detail_colors[:1])
                    texture[y, x] = [min(255, int(c * brightness)) for c in color]

        return texture

    def generate_sandstone(self, base_color, detail_colors):
        """Generate sandstone texture with layers"""
        texture = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        # Create horizontal stratification
        layer_height = random.randint(5, 15)
        current_y = 0

        while current_y < self.height:
            layer_color = random.choice([base_color] + detail_colors)
            layer_thickness = random.randint(layer_height // 2, layer_height * 2)

            for y in range(current_y, min(current_y + layer_thickness, self.height)):
                for x in range(self.width):
                    # Add noise to layer
                    noise = pnoise2(x * 0.02, y * 0.1, octaves=3)
                    waviness = int(noise * 5)

                    if 0 <= y + waviness < self.height:
                        # Grain texture
                        grain = pnoise2(x * 0.1, y * 0.1, octaves=1)
                        brightness = 0.9 + grain * 0.2

                        texture[y + waviness, x] = [
                            min(255, int(layer_color[0] * brightness)),
                            min(255, int(layer_color[1] * brightness)),
                            min(255, int(layer_color[2] * brightness)),
                        ]

            current_y += layer_thickness

        return texture

    def generate_oasis_pattern(self, base_color, detail_colors):
        """Generate oasis water pattern in desert"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create oasis center
        center_x = self.width // 2
        center_y = self.height // 2
        oasis_radius = random.randint(80, 120)

        # Water colors
        water_colors = [(0, 119, 190), (70, 130, 180), (100, 149, 237)]

        for y in range(self.height):
            for x in range(self.width):
                dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                if dist < oasis_radius:
                    # Water in center
                    if dist < oasis_radius * 0.7:
                        # Add ripples
                        ripple = math.sin(dist * 0.2) * 0.1 + 0.9
                        water_color = random.choice(water_colors)
                        texture[y, x] = [int(c * ripple) for c in water_color]
                    else:
                        # Wet sand transition
                        t = (dist - oasis_radius * 0.7) / (oasis_radius * 0.3)
                        wet_sand = self.blend_colors(
                            water_colors[0], detail_colors[0], t
                        )
                        texture[y, x] = wet_sand

                # Add palm trees around oasis
                if oasis_radius < dist < oasis_radius * 1.5:
                    if random.random() < 0.001:
                        # Simple palm representation
                        palm_height = 30
                        for h in range(palm_height):
                            py = y - h
                            if 0 <= py < self.height:
                                texture[py, x] = [101, 67, 33]  # Brown trunk

        return texture

    def generate_mirage_effect(self, base_color, detail_colors):
        """Generate heat mirage effect"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create wavy distortion pattern
        for y in range(self.height):
            for x in range(self.width):
                # Heat wave distortion
                wave1 = math.sin(x * 0.02 + y * 0.01) * 10
                wave2 = math.sin(x * 0.01 - y * 0.02) * 5

                # Sample from distorted position
                sample_x = int(x + wave1) % self.width
                sample_y = int(y + wave2) % self.height

                # Create shimmering effect
                shimmer = pnoise2(x * 0.05, y * 0.05, octaves=2)
                brightness = 1.0 + shimmer * 0.2

                # Blend between colors for mirage
                if shimmer > 0.3:
                    color = detail_colors[0]  # Sky-like color for mirage
                else:
                    color = base_color

                texture[y, x] = [min(255, int(c * brightness)) for c in color]

        return texture

    def generate_snow_drift(self, base_color, detail_colors):
        """Generate snow drift patterns"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create smooth drifts
        for y in range(self.height):
            for x in range(self.width):
                # Large scale drifts
                drift1 = pnoise2(x * 0.01, y * 0.01, octaves=3)
                drift2 = pnoise2(x * 0.02, y * 0.03, octaves=2)

                height = (drift1 * 0.7 + drift2 * 0.3 + 1) / 2

                # Shadow in valleys
                if height < 0.4:
                    shadow_color = detail_colors[0]
                    t = height / 0.4
                    color = self.blend_colors(shadow_color, base_color, t)
                else:
                    # Bright on peaks
                    t = (height - 0.4) / 0.6
                    color = self.blend_colors(base_color, (255, 255, 255), t * 0.3)

                texture[y, x] = color

        return texture

    def generate_frozen_lake(self, base_color, detail_colors):
        """Generate frozen lake surface"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Ice cracks
        num_cracks = random.randint(5, 15)
        for _ in range(num_cracks):
            start_x = random.randint(0, self.width)
            start_y = random.randint(0, self.height)

            # Crack propagation
            x, y = start_x, start_y
            crack_length = random.randint(50, 150)
            angle = random.uniform(0, 2 * math.pi)

            for i in range(crack_length):
                # Add some randomness to crack direction
                angle += random.uniform(-0.3, 0.3)
                x += int(math.cos(angle) * 2)
                y += int(math.sin(angle) * 2)

                if 0 <= x < self.width and 0 <= y < self.height:
                    # Main crack
                    texture[y, x] = detail_colors[0]

                    # Crack width variation
                    width = random.randint(1, 3)
                    for w in range(-width, width + 1):
                        for h in range(-width, width + 1):
                            px = x + w
                            py = y + h
                            if 0 <= px < self.width and 0 <= py < self.height:
                                if abs(w) + abs(h) <= width:
                                    texture[py, px] = detail_colors[0]

        # Add frost patterns
        frost_noise = np.zeros((self.width, self.height))
        for y in range(self.height):
            for x in range(self.width):
                frost = pnoise2(x * 0.1, y * 0.1, octaves=4)
                if frost > 0.3:
                    frost_alpha = (frost - 0.3) / 0.7
                    current = texture[y, x]
                    white = (255, 255, 255)
                    texture[y, x] = [
                        int(c * (1 - frost_alpha) + w * frost_alpha)
                        for c, w in zip(current, white)
                    ]

        return texture

    def generate_blizzard_effect(self, base_color, detail_colors):
        """Generate blizzard/snow storm effect"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Wind direction
        wind_angle = random.uniform(-math.pi / 4, math.pi / 4)
        wind_speed = random.uniform(10, 30)

        # Snow particles
        num_particles = random.randint(500, 1000)
        for _ in range(num_particles):
            # Start position
            if random.random() < 0.5:
                x = random.randint(0, self.width)
                y = 0
            else:
                x = 0
                y = random.randint(0, self.height)

            # Particle trail
            particle_length = random.randint(20, 60)
            particle_size = random.randint(1, 4)

            for i in range(particle_length):
                # Move particle
                x += int(wind_speed * math.cos(wind_angle))
                y += int(wind_speed * math.sin(wind_angle) + 5)  # Gravity

                if 0 <= x < self.width and 0 <= y < self.height:
                    # Fade effect
                    alpha = 1 - i / particle_length

                    for dx in range(-particle_size, particle_size + 1):
                        for dy in range(-particle_size, particle_size + 1):
                            px = x + dx
                            py = y + dy

                            if 0 <= px < self.width and 0 <= py < self.height:
                                if dx * dx + dy * dy <= particle_size * particle_size:
                                    current = texture[py, px]
                                    white = (255, 255, 255)
                                    texture[py, px] = [
                                        int(c * (1 - alpha * 0.7) + w * alpha * 0.7)
                                        for c, w in zip(current, white)
                                    ]

        return texture

    def generate_aurora_pattern(self, base_color, detail_colors):
        """Generate aurora borealis pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Aurora colors
        aurora_colors = [(0, 255, 127), (50, 205, 50), (0, 191, 255), (138, 43, 226)]

        # Create flowing bands
        num_bands = random.randint(3, 6)
        for band in range(num_bands):
            band_y = self.height // 4 + band * 30
            band_color = random.choice(aurora_colors)

            for x in range(self.width):
                # Wavy pattern
                wave = math.sin(x * 0.02 + band * 0.5) * 30
                wave += math.sin(x * 0.05) * 10

                band_center = band_y + int(wave)

                # Band thickness varies
                thickness = 20 + int(math.sin(x * 0.03) * 10)

                for y in range(
                    max(0, band_center - thickness),
                    min(self.height, band_center + thickness),
                ):
                    # Gaussian falloff
                    dist = abs(y - band_center)
                    intensity = math.exp(-((dist / thickness) ** 2))

                    if intensity > 0.1:
                        current = texture[y, x]
                        # Additive blending for glow
                        texture[y, x] = [
                            min(255, int(c + band_color[i] * intensity * 0.5))
                            for i, c in enumerate(current)
                        ]

        return texture

    def generate_obsidian(self, base_color, detail_colors):
        """Generate obsidian glass texture"""
        texture = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        # Create sharp, glassy reflections
        for y in range(self.height):
            for x in range(self.width):
                # Multiple layers of reflection
                reflection1 = pnoise2(x * 0.01, y * 0.01, octaves=2)
                reflection2 = pnoise2(x * 0.05, y * 0.02, octaves=3)

                # Sharp transitions for glassy look
                if reflection1 > 0.3:
                    color = (20, 20, 20)  # Very dark base
                elif reflection2 > 0.4:
                    color = detail_colors[0]  # Highlight color
                else:
                    # Gradient between
                    t = (reflection1 + 1) / 2
                    color = self.blend_colors((10, 10, 10), (60, 60, 60), t)

                texture[y, x] = color

        # Add sharp edges/cracks
        num_edges = random.randint(10, 20)
        for _ in range(num_edges):
            edge_start = (random.randint(0, self.width), random.randint(0, self.height))
            edge_end = (random.randint(0, self.width), random.randint(0, self.height))

            # Draw sharp line
            steps = max(
                abs(edge_end[0] - edge_start[0]), abs(edge_end[1] - edge_start[1])
            )
            if steps > 0:
                for i in range(steps):
                    t = i / steps
                    x = int(edge_start[0] + t * (edge_end[0] - edge_start[0]))
                    y = int(edge_start[1] + t * (edge_end[1] - edge_start[1]))

                    if 0 <= x < self.width and 0 <= y < self.height:
                        texture[y, x] = (200, 200, 200)  # Bright edge

        return texture

    def generate_ash_fall(self, base_color, detail_colors):
        """Generate volcanic ash fall pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Layers of ash
        for layer in range(3):
            layer_color = [(40 + layer * 20,) * 3][0]  # Progressively lighter gray

            for y in range(self.height):
                for x in range(self.width):
                    # Ash accumulation pattern
                    accumulation = pnoise2(x * 0.02, y * 0.02, octaves=3 + layer)

                    if accumulation > 0.2 - layer * 0.1:
                        # Smooth blending
                        alpha = (accumulation + 1) / 2
                        current = texture[y, x]
                        texture[y, x] = [
                            int(c * (1 - alpha * 0.5) + layer_color[i] * alpha * 0.5)
                            for i, c in enumerate(current)
                        ]

        # Falling ash particles
        num_particles = random.randint(200, 400)
        for _ in range(num_particles):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            size = random.randint(1, 3)

            # Ash particles are gray
            gray_level = random.randint(80, 150)
            ash_color = (gray_level, gray_level, gray_level)

            for dx in range(-size, size + 1):
                for dy in range(-size, size + 1):
                    px = (x + dx) % self.width
                    py = (y + dy) % self.height

                    if dx * dx + dy * dy <= size * size:
                        texture[py, px] = ash_color

        return texture

    def generate_magma_cracks(self, base_color, detail_colors):
        """Generate glowing magma cracks in rock"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create crack network
        num_main_cracks = random.randint(3, 7)
        crack_points = []

        for _ in range(num_main_cracks):
            start = (random.randint(0, self.width), random.randint(0, self.height))
            end = (random.randint(0, self.width), random.randint(0, self.height))

            # Interpolate crack path
            steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
            if steps > 0:
                for i in range(steps):
                    t = i / steps
                    x = int(start[0] + t * (end[0] - start[0]))
                    y = int(start[1] + t * (end[1] - start[1]))

                    # Add some wobble
                    wobble = int(pnoise2(i * 0.1, 0, octaves=2) * 5)
                    x = (x + wobble) % self.width

                    crack_points.append((x, y))

        # Draw cracks with glow
        for x, y in crack_points:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Glow radius
                glow_radius = random.randint(5, 10)

                for dx in range(-glow_radius, glow_radius + 1):
                    for dy in range(-glow_radius, glow_radius + 1):
                        px = (x + dx) % self.width
                        py = (y + dy) % self.height

                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist <= glow_radius:
                            # Glow intensity
                            intensity = 1 - dist / glow_radius

                            # Hot center, cooler edges
                            if dist < 2:
                                glow_color = (255, 255, 200)  # White hot
                            elif dist < glow_radius * 0.5:
                                glow_color = (255, 140, 0)  # Orange
                            else:
                                glow_color = (139, 0, 0)  # Dark red

                            current = texture[py, px]
                            texture[py, px] = [
                                int(c * (1 - intensity) + g * intensity)
                                for c, g in zip(current, glow_color)
                            ]

        return texture

    def generate_volcanic_rock(self, base_color, detail_colors):
        """Generate porous volcanic rock texture"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Create porous surface
        for y in range(self.height):
            for x in range(self.width):
                # Rock texture
                rock_noise = pnoise2(x * 0.05, y * 0.05, octaves=4)

                # Pores/holes
                pore_noise = pnoise2(x * 0.2, y * 0.2, octaves=2)

                if pore_noise > 0.5:
                    # Dark pore
                    color = (20, 20, 20)
                else:
                    # Rock surface with variation
                    brightness = 0.7 + rock_noise * 0.3
                    rock_color = random.choice([base_color] + detail_colors[:1])
                    color = [int(c * brightness) for c in rock_color]

                texture[y, x] = color

        return texture

    def generate_seaweed(self, base_color, detail_colors):
        """Generate flowing seaweed pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Multiple seaweed strands
        num_strands = random.randint(20, 40)

        for _ in range(num_strands):
            # Base position
            base_x = random.randint(0, self.width)
            base_y = self.height - random.randint(0, 50)

            # Strand properties
            strand_length = random.randint(100, 200)
            strand_width = random.randint(5, 15)
            seaweed_color = random.choice(detail_colors)

            # Current influence
            current_strength = random.uniform(0.02, 0.05)
            current_offset = random.uniform(0, 2 * math.pi)

            for i in range(strand_length):
                # Flowing motion
                flow = math.sin(i * 0.1 + current_offset) * 30 * current_strength * i
                x = int(base_x + flow) % self.width
                y = base_y - i

                if 0 <= y < self.height:
                    # Strand width varies
                    width = strand_width * (1 - i / strand_length)

                    for w in range(int(-width / 2), int(width / 2) + 1):
                        px = (x + w) % self.width

                        # Edge softness
                        edge_alpha = 1 - abs(w) / (width / 2 + 1)

                        current = texture[y, px]
                        texture[y, px] = [
                            int(c * (1 - edge_alpha * 0.7) + s * edge_alpha * 0.7)
                            for c, s in zip(current, seaweed_color)
                        ]

        return texture

    def generate_bubble_pattern(self, base_color, detail_colors):
        """Generate underwater bubble pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Bubble streams
        num_streams = random.randint(5, 10)

        for _ in range(num_streams):
            stream_x = random.randint(0, self.width)
            stream_base_y = self.height

            # Bubbles in stream
            num_bubbles = random.randint(20, 40)

            for i in range(num_bubbles):
                # Bubble properties
                bubble_y = stream_base_y - i * random.randint(10, 20)
                bubble_x = stream_x + int(math.sin(i * 0.3) * 20)
                bubble_size = random.randint(3, 10)

                if 0 <= bubble_y < self.height:
                    # Draw bubble
                    for dx in range(-bubble_size, bubble_size + 1):
                        for dy in range(-bubble_size, bubble_size + 1):
                            px = (bubble_x + dx) % self.width
                            py = bubble_y + dy

                            if 0 <= py < self.height:
                                dist = math.sqrt(dx * dx + dy * dy)

                                if dist <= bubble_size:
                                    # Bubble edge
                                    if dist > bubble_size - 2:
                                        color = (200, 200, 255)
                                    # Bubble interior (transparent look)
                                    else:
                                        t = dist / bubble_size
                                        color = self.blend_colors(
                                            base_color, (220, 220, 255), 1 - t
                                        )

                                    texture[py, px] = color

        return texture

    def generate_water_current(self, base_color, detail_colors):
        """Generate water current flow pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Flow direction
        flow_angle = random.uniform(0, 2 * math.pi)

        for y in range(self.height):
            for x in range(self.width):
                # Current flow lines
                flow_offset = x * math.cos(flow_angle) + y * math.sin(flow_angle)

                # Multiple current layers
                current1 = math.sin(flow_offset * 0.02) * 0.5 + 0.5
                current2 = math.sin(flow_offset * 0.05 + 1.5) * 0.5 + 0.5
                current3 = math.sin(flow_offset * 0.1 + 3.0) * 0.5 + 0.5

                # Combine currents
                combined = current1 * 0.5 + current2 * 0.3 + current3 * 0.2

                # Apply color gradient
                if combined > 0.7:
                    color = detail_colors[0]
                elif combined > 0.4:
                    t = (combined - 0.4) / 0.3
                    color = self.blend_colors(base_color, detail_colors[0], t)
                else:
                    color = base_color

                # Add slight turbulence
                turb = pnoise2(x * 0.05, y * 0.05, octaves=2)
                brightness = 0.9 + turb * 0.2

                texture[y, x] = [min(255, int(c * brightness)) for c in color]

        return texture

    def generate_seafloor(self, base_color, detail_colors):
        """Generate seafloor sand and rock pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Sand ripples
        for y in range(self.height):
            for x in range(self.width):
                # Ripple pattern
                ripple = math.sin(x * 0.1) * math.sin(y * 0.02) * 0.3

                # Sand texture
                sand_noise = pnoise2(x * 0.05, y * 0.05, octaves=3)

                # Rocks and shells
                rock_noise = pnoise2(x * 0.02, y * 0.02, octaves=2)

                if rock_noise > 0.6:
                    # Rock
                    color = (80, 80, 100)
                else:
                    # Sandy areas
                    brightness = 0.8 + ripple + sand_noise * 0.2
                    sand_color = random.choice([base_color] + detail_colors[:1])
                    color = [int(c * brightness) for c in sand_color]

                texture[y, x] = color

        # Add occasional shells/starfish
        num_objects = random.randint(5, 15)
        for _ in range(num_objects):
            obj_x = random.randint(20, self.width - 20)
            obj_y = random.randint(20, self.height - 20)
            obj_type = random.choice(["shell", "starfish"])

            if obj_type == "shell":
                # Simple shell shape
                shell_size = random.randint(10, 20)
                shell_color = random.choice(
                    [(255, 228, 196), (255, 218, 185), (255, 192, 203)]
                )

                for angle in np.linspace(0, math.pi, 20):
                    r = shell_size * (1 - angle / math.pi)
                    x = int(obj_x + r * math.cos(angle))
                    y = int(obj_y + r * math.sin(angle) * 0.5)

                    if 0 <= x < self.width and 0 <= y < self.height:
                        texture[y, x] = shell_color

            elif obj_type == "starfish":
                # Five-armed starfish
                starfish_size = random.randint(15, 25)
                starfish_color = random.choice(
                    [(255, 127, 80), (255, 99, 71), (255, 140, 0)]
                )

                for arm in range(5):
                    arm_angle = (arm / 5) * 2 * math.pi

                    for r in range(starfish_size):
                        x = int(obj_x + r * math.cos(arm_angle))
                        y = int(obj_y + r * math.sin(arm_angle))

                        # Arm width
                        width = max(1, (starfish_size - r) // 3)

                        for w in range(-width, width + 1):
                            wx = int(x + w * math.sin(arm_angle))
                            wy = int(y - w * math.cos(arm_angle))

                            if 0 <= wx < self.width and 0 <= wy < self.height:
                                texture[wy, wx] = starfish_color

        return texture

    def generate_hologram(self, base_color, detail_colors):
        """Generate holographic projection pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Holographic scan lines
        for y in range(self.height):
            # Horizontal scan line effect
            if y % 4 < 2:
                brightness_mod = 1.2
            else:
                brightness_mod = 0.8

            for x in range(self.width):
                # Holographic interference pattern
                interference1 = math.sin(x * 0.1) * math.sin(y * 0.1)
                interference2 = math.sin(x * 0.05 + 1) * math.sin(y * 0.05 + 1)

                combined = (interference1 + interference2) * 0.5

                # Color shifting effect
                hue_shift = (x + y) / (self.width + self.height)
                r, g, b = colorsys.hsv_to_rgb(hue_shift, 0.8, 1.0)

                # Apply holographic color
                holo_color = (int(r * 255), int(g * 255), int(b * 255))

                # Blend with base
                alpha = (combined + 1) / 2 * 0.7
                color = [
                    int(
                        base_color[i] * (1 - alpha)
                        + holo_color[i] * alpha * brightness_mod
                    )
                    for i in range(3)
                ]

                texture[y, x] = color

        # Add glitch artifacts
        num_glitches = random.randint(3, 8)
        for _ in range(num_glitches):
            glitch_y = random.randint(0, self.height - 10)
            glitch_height = random.randint(2, 10)
            glitch_offset = random.randint(-20, 20)

            # Shift section horizontally
            for y in range(glitch_y, min(glitch_y + glitch_height, self.height)):
                row = texture[y, :].copy()
                texture[y, :] = np.roll(row, glitch_offset, axis=0)

        return texture

    def generate_neon_grid(self, base_color, detail_colors):
        """Generate cyberpunk neon grid pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Grid parameters
        grid_size = 32
        line_width = 2

        # Perspective transformation for depth
        horizon_y = self.height // 3

        for y in range(horizon_y, self.height):
            # Perspective scaling
            depth = (y - horizon_y) / (self.height - horizon_y)
            perspective_scale = 0.2 + depth * 0.8

            # Horizontal lines
            if y % int(grid_size * perspective_scale) < line_width:
                for x in range(self.width):
                    # Neon glow
                    glow_color = random.choice(detail_colors)
                    texture[y, x] = glow_color

            # Vertical lines with perspective
            for x in range(self.width):
                # Calculate perspective x position
                center_x = self.width // 2
                offset_x = (x - center_x) * perspective_scale
                grid_x = center_x + offset_x

                if abs((grid_x % grid_size) - grid_size // 2) < line_width:
                    if 0 <= int(grid_x) < self.width:
                        texture[y, x] = random.choice(detail_colors)

        # Add glow effect to lines
        img = Image.fromarray(texture)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        blurred = np.array(img)

        # Combine original with glow
        for y in range(self.height):
            for x in range(self.width):
                if np.sum(texture[y, x]) > np.sum(base_color):
                    # Keep bright lines
                    continue
                else:
                    # Add glow to dark areas
                    texture[y, x] = [
                        min(255, b + texture[y, x, i] // 4)
                        for i, b in enumerate(blurred[y, x])
                    ]

        return texture

    def generate_data_stream(self, base_color, detail_colors):
        """Generate flowing data/matrix-like pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Data streams
        num_streams = random.randint(20, 40)

        for _ in range(num_streams):
            stream_x = random.randint(0, self.width)
            stream_speed = random.uniform(0.5, 2.0)
            stream_color = random.choice(detail_colors)

            # Characters in stream
            stream_length = random.randint(50, 150)
            start_y = random.randint(-stream_length, self.height)

            for i in range(stream_length):
                y = int(start_y + i * stream_speed) % self.height

                if 0 <= y < self.height:
                    # Brightness falloff
                    if i < 10:
                        brightness = i / 10
                    elif i > stream_length - 20:
                        brightness = (stream_length - i) / 20
                    else:
                        brightness = 1.0

                    # Random "character" representation
                    char_size = 8
                    char_pattern = random.randint(0, 255)

                    for bit in range(8):
                        if char_pattern & (1 << bit):
                            px = (stream_x + (bit % 3) - 1) % self.width
                            py = y + (bit // 3) - 1

                            if 0 <= py < self.height:
                                color = [int(c * brightness) for c in stream_color]
                                texture[py, px] = color

        return texture

    def generate_glitch_effect(self, base_color, detail_colors):
        """Generate digital glitch effect pattern"""
        texture = np.full((self.width, self.height, 3), base_color, dtype=np.uint8)

        # Base noise
        for y in range(self.height):
            for x in range(self.width):
                noise = random.random()
                if noise > 0.95:
                    texture[y, x] = random.choice(detail_colors)
                else:
                    texture[y, x] = base_color

        # Glitch blocks
        num_blocks = random.randint(5, 15)
        for _ in range(num_blocks):
            block_x = random.randint(0, self.width - 50)
            block_y = random.randint(0, self.height - 50)
            block_w = random.randint(10, 50)
            block_h = random.randint(5, 30)

            # Glitch type
            glitch_type = random.choice(["shift", "corrupt", "color"])

            if glitch_type == "shift":
                # Pixel shift
                shift_x = random.randint(-20, 20)
                shift_y = random.randint(-5, 5)

                block_data = texture[
                    block_y : block_y + block_h, block_x : block_x + block_w
                ].copy()

                for y in range(block_h):
                    for x in range(block_w):
                        src_x = (x + shift_x) % block_w
                        src_y = (y + shift_y) % block_h

                        if (
                            block_y + y < self.height
                            and block_x + x < self.width
                            and block_y + src_y < self.height
                            and block_x + src_x < self.width
                        ):
                            texture[block_y + y, block_x + x] = block_data[src_y, src_x]

            elif glitch_type == "corrupt":
                # Data corruption
                for y in range(block_y, min(block_y + block_h, self.height)):
                    for x in range(block_x, min(block_x + block_w, self.width)):
                        if random.random() < 0.3:
                            texture[y, x] = random.choice(detail_colors)

            elif glitch_type == "color":
                # Color channel shift
                for y in range(block_y, min(block_y + block_h, self.height)):
                    for x in range(block_x, min(block_x + block_w, self.width)):
                        r, g, b = texture[y, x]
                        # Shift channels
                        texture[y, x] = [g, b, r]

        # Scan lines
        for y in range(0, self.height, 4):
            if random.random() < 0.3:
                for x in range(self.width):
                    texture[y, x] = [max(0, c - 50) for c in texture[y, x]]

        return texture


def generate_enhanced_game_assets(
    output_dir: Path | str,
    num_samples: int = 100,
    size: Tuple[int, int] = (512, 512),
    quality_threshold: float = 0.8,
) -> None:
    """
    Generate high-quality, diverse game assets covering all genres and art styles

    Args:
        output_dir: Output directory for generated assets
        num_samples: Number of samples per combination
        size: Image dimensions
        quality_threshold: Minimum quality score for asset acceptance
    """
    output_dir = Path(output_dir)
    generator = AdvancedAssetGenerator(size)

    # Define generation combinations
    asset_types = ["textures", "sprites"]
    biomes = list(generator.biome_configs.keys())
    art_styles = list(generator.art_styles.keys())
    genres = list(generator.game_genres.keys())

    print("=== Enhanced Game Asset Generation ===")
    print(f"Generating {num_samples} samples per combination")
    print(f"Biomes: {len(biomes)}")
    print(f"Art styles: {len(art_styles)}")
    print(f"Game genres: {len(genres)}")
    print(f"Total combinations: {len(asset_types) * len(biomes) * len(art_styles)}")
    print("=" * 40)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "generator_version": "3.0_advanced",
        "quality_threshold": quality_threshold,
        "features": [
            "all_2d_game_genres",
            "comprehensive_art_styles",
            "seamless_textures",
            "detailed_sprites",
            "quality_filtering",
            "genre_specific_assets",
            "advanced_patterns",
        ],
        "stats": {
            "total_generated": 0,
            "accepted": 0,
            "rejected": 0,
            "rejection_reasons": {},
        },
        "assets": [],
    }

    # Generate assets with quality control
    for asset_type in asset_types:
        for biome in biomes:
            for art_style in art_styles[:3]:  # Limit art styles for initial generation
                # Create output directory
                output_path = output_dir / "raw" / asset_type / f"{biome}_{art_style}"
                output_path.mkdir(parents=True, exist_ok=True)

                # Progress tracking
                generated_count = 0
                attempts = 0
                max_attempts = num_samples * 2  # Allow for rejections

                pbar = tqdm(total=num_samples, desc=f"{asset_type}/{biome}/{art_style}")

                while generated_count < num_samples and attempts < max_attempts:
                    attempts += 1

                    try:
                        if asset_type == "textures":
                            # Select pattern type
                            config = generator.biome_configs[biome]
                            pattern_name = random.choice(config["patterns"])

                            # Generate texture - use 'realistic' style to preserve pattern details
                            if (
                                art_style == "realistic" or random.random() < 0.5
                            ):  # 50% chance to skip style processing
                                img_array = generator.generate_enhanced_texture(
                                    biome, pattern_name, "realistic"
                                )
                            else:
                                img_array = generator.generate_enhanced_texture(
                                    biome, pattern_name, art_style
                                )

                            # Quality checks
                            quality_score = evaluate_texture_quality(img_array)

                            asset_metadata = {
                                "asset_type": "texture",
                                "pattern": pattern_name,
                                "seamless": True,
                                "quality_score": quality_score,
                            }

                        else:  # sprites
                            # Select sprite type based on biome or genre
                            if random.random() < 0.7:  # 70% biome-specific
                                config = generator.biome_configs[biome]
                                sprite_category = random.choice(["creatures", "items"])
                                sprite_type = random.choice(
                                    config["sprites"][sprite_category]
                                )
                            else:  # 30% genre-specific
                                genre = random.choice(
                                    list(generator.game_genres.keys())
                                )
                                sprite_type = random.choice(
                                    generator.game_genres[genre]["sprites"]
                                )

                            # Generate sprite
                            img_array = generator.generate_enhanced_sprite(
                                biome, sprite_type, art_style
                            )

                            # Quality checks
                            quality_score = evaluate_sprite_quality(img_array)

                            asset_metadata = {
                                "asset_type": "sprite",
                                "sprite_type": sprite_type,
                                "has_transparency": True,
                                "quality_score": quality_score,
                            }

                        # Quality threshold check
                        if quality_score < quality_threshold:
                            reason = get_rejection_reason(quality_score, img_array)
                            metadata["stats"]["rejected"] += 1
                            metadata["stats"]["rejection_reasons"][reason] = (
                                metadata["stats"]["rejection_reasons"].get(reason, 0)
                                + 1
                            )
                            continue

                        # Save accepted asset
                        if img_array.shape[2] == 4:  # RGBA
                            img = Image.fromarray(img_array.astype(np.uint8), "RGBA")
                        else:  # RGB
                            img = Image.fromarray(img_array.astype(np.uint8), "RGB")

                        filename = f"{biome}_{art_style}_{asset_type}_{generated_count:04d}.png"
                        filepath = output_path / filename
                        img.save(filepath, optimize=True, quality=95)

                        # Update metadata
                        asset_metadata.update(
                            {
                                "filename": filename,
                                "filepath": str(filepath.relative_to(output_dir)),
                                "biome": biome,
                                "art_style": art_style,
                                "size": size,
                                "file_size": filepath.stat().st_size,
                            }
                        )

                        metadata["assets"].append(asset_metadata)
                        metadata["stats"]["accepted"] += 1
                        generated_count += 1
                        pbar.update(1)

                    except Exception as e:
                        print(f"\nError generating asset: {e}")
                        continue

                pbar.close()
                metadata["stats"]["total_generated"] += attempts

    # Save metadata
    metadata_path = output_dir / "metadata" / "enhanced_generation_metadata.json"
    metadata_path.parent.mkdir(exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n=== Generation Summary ===")
    print(f"Total attempts: {metadata['stats']['total_generated']}")
    print(f"Accepted: {metadata['stats']['accepted']}")
    print(f"Rejected: {metadata['stats']['rejected']}")
    print(
        f"Acceptance rate: {metadata['stats']['accepted'] / metadata['stats']['total_generated'] * 100:.1f}%"
    )
    print("\nRejection reasons:")
    for reason, count in metadata["stats"]["rejection_reasons"].items():
        print(f"  {reason}: {count}")
    print(f"\nMetadata saved to: {metadata_path}")


def evaluate_texture_quality(texture: np.ndarray) -> float:
    """Evaluate texture quality based on multiple criteria"""
    scores = []

    # 1. Detail level (edge detection)
    gray = np.mean(texture[:, :, :3], axis=2)
    edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
    detail_score = min(1.0, edges / (texture.shape[0] * texture.shape[1] * 50))
    scores.append(detail_score)

    # 2. Color variety
    unique_colors = len(np.unique(texture.reshape(-1, texture.shape[2]), axis=0))
    color_score = min(1.0, unique_colors / 1000)
    scores.append(color_score)

    # 3. Contrast
    contrast = gray.std() / 128
    scores.append(min(1.0, contrast * 2))

    # 4. Pattern coherence (using FFT)
    fft = np.fft.fft2(gray)
    fft_magnitude = np.abs(fft)
    pattern_score = (
        np.sum(fft_magnitude > np.mean(fft_magnitude) * 2) / fft_magnitude.size
    )
    scores.append(min(1.0, pattern_score * 10))

    return np.mean(scores)


def evaluate_sprite_quality(sprite: np.ndarray) -> float:
    """Evaluate sprite quality based on multiple criteria"""
    scores = []

    # 1. Non-empty content
    if sprite.shape[2] == 4:  # RGBA
        alpha = sprite[:, :, 3]
        content_ratio = np.sum(alpha > 0) / alpha.size
        scores.append(min(1.0, content_ratio * 5))  # Expect ~20% coverage
    else:
        scores.append(1.0)

    # 2. Shape coherence (connected components)
    if sprite.shape[2] == 4:
        binary = (sprite[:, :, 3] > 128).astype(np.uint8)
        # Simple connected component check
        components = measure_connected_components(binary)
        coherence_score = 1.0 if components <= 3 else 0.5
        scores.append(coherence_score)
    else:
        scores.append(1.0)

    # 3. Color quality
    rgb = sprite[:, :, :3]
    color_variety = len(np.unique(rgb.reshape(-1, 3), axis=0))
    color_score = min(1.0, color_variety / 100)
    scores.append(color_score)

    # 4. Edge quality
    if sprite.shape[2] == 4:
        # Check for clean edges
        alpha_edges = (
            np.abs(np.diff(sprite[:, :, 3].astype(float), axis=0)).sum()
            + np.abs(np.diff(sprite[:, :, 3].astype(float), axis=1)).sum()
        )
        edge_score = min(1.0, alpha_edges / (sprite.shape[0] * sprite.shape[1] * 10))
        scores.append(edge_score)
    else:
        scores.append(1.0)

    return np.mean(scores)


def measure_connected_components(binary: np.ndarray) -> int:
    """Simple connected components counter"""
    # Simplified version - just count large connected regions
    visited = np.zeros_like(binary, dtype=bool)
    components = 0

    def flood_fill(x, y):
        if x < 0 or x >= binary.shape[1] or y < 0 or y >= binary.shape[0]:
            return 0
        if visited[y, x] or binary[y, x] == 0:
            return 0

        visited[y, x] = True
        size = 1

        # 4-connectivity
        size += flood_fill(x + 1, y)
        size += flood_fill(x - 1, y)
        size += flood_fill(x, y + 1)
        size += flood_fill(x, y - 1)

        return size

    # Count components larger than threshold
    min_size = binary.size * 0.01  # At least 1% of image

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if not visited[y, x] and binary[y, x] > 0:
                component_size = flood_fill(x, y)
                if component_size > min_size:
                    components += 1

    return components


def get_rejection_reason(score: float, img_array: np.ndarray) -> str:
    """Determine specific rejection reason"""
    if score < 0.3:
        return "insufficient_detail"
    elif score < 0.5:
        # Check for specific issues
        if img_array.shape[2] == 4:
            alpha = img_array[:, :, 3]
            if np.sum(alpha > 0) < alpha.size * 0.05:
                return "too_sparse"

        # Check color variety
        unique_colors = len(
            np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)
        )
        if unique_colors < 10:
            return "low_color_variety"

        return "low_contrast"
    else:
        return "quality_threshold"


def main():
    """Main function for enhanced asset generation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate high-quality game assets for MADWE project"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for generated assets",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples per combination"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image size (width height)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.7,
        help="Minimum quality score for acceptance (0-1)",
    )

    args = parser.parse_args()

    print("MADWE Enhanced Asset Generation System v3.0")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Samples per combination: {args.num_samples}")
    print(f"Image size: {args.size[0]}x{args.size[1]}")
    print(f"Quality threshold: {args.quality_threshold}")
    print("=" * 50)

    generate_enhanced_game_assets(
        args.output_dir,
        num_samples=args.num_samples,
        size=tuple(args.size),
        quality_threshold=args.quality_threshold,
    )


if __name__ == "__main__":
    main()
