"""
Download and generate synthetic game assets - Python 3.13.5 compatible
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
import json
from datetime import datetime


def generate_synthetic_assets(
    output_dir: Path | str, num_samples: int = 100, size: Tuple[int, int] = (512, 512)
) -> None:
    """Generate synthetic game assets for each biome"""

    output_dir = Path(output_dir)
    biomes = ["forest", "desert", "snow", "volcanic", "underwater", "sky"]
    asset_types = ["textures", "sprites"]

    # Enhanced biome color palettes
    palettes = {
        "forest": [(34, 139, 34), (0, 100, 0), (85, 107, 47), (46, 125, 50)],
        "desert": [(238, 203, 173), (205, 133, 63), (244, 164, 96), (255, 193, 7)],
        "snow": [(255, 250, 250), (176, 224, 230), (135, 206, 235), (245, 245, 245)],
        "volcanic": [(178, 34, 34), (255, 69, 0), (139, 0, 0), (255, 87, 34)],
        "underwater": [(0, 119, 190), (0, 191, 255), (70, 130, 180), (0, 150, 136)],
        "sky": [(135, 206, 250), (255, 255, 255), (176, 196, 222), (100, 149, 237)],
    }

    print("Generating synthetic game assets...")
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "num_samples": num_samples,
        "size": size,
        "assets": [],
    }

    for asset_type in asset_types:
        for biome in biomes:
            biome_dir = output_dir / "raw" / asset_type / biome
            biome_dir.mkdir(parents=True, exist_ok=True)

            colors = palettes[biome]

            for i in tqdm(range(num_samples), desc=f"{asset_type}/{biome}"):
                img = create_asset_image(asset_type, colors, size)
                filename = f"{biome}_{asset_type}_{i:04d}.png"
                filepath = biome_dir / filename
                img.save(filepath, optimize=True)

                metadata["assets"].append(
                    {
                        "filename": filename,
                        "path": str(filepath.relative_to(output_dir)),
                        "asset_type": asset_type,
                        "biome": biome,
                        "size": size,
                    }
                )

    # Save metadata
    metadata_path = output_dir / "metadata" / "generated_assets.json"
    metadata_path.parent.mkdir(exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated {len(metadata['assets'])} assets")
    print(f"Metadata saved to {metadata_path}")


def create_asset_image(
    asset_type: str,
    colors: List[Tuple[int, int, int]],
    size: Tuple[int, int] = (512, 512),
) -> Image.Image:
    """Create synthetic asset image with improved patterns"""

    base_color = colors[np.random.randint(len(colors))]
    img_array = np.zeros((*size, 3), dtype=np.uint8)

    if asset_type == "textures":
        # Create more sophisticated texture patterns
        pattern_type = np.random.choice(["tiles", "noise", "gradient", "mixed"])

        if pattern_type == "tiles":
            # Tiled texture with variations
            tile_sizes = [32, 64, 128]
            tile_size = np.random.choice(tile_sizes)

            for i in range(0, size[0], tile_size):
                for j in range(0, size[1], tile_size):
                    variation = np.random.randint(-30, 30, 3)
                    color = np.clip(np.array(base_color) + variation, 0, 255)

                    # Add subtle gradient within tile
                    tile = np.ones((tile_size, tile_size, 3)) * color
                    gradient = np.linspace(0.9, 1.1, tile_size)
                    tile = tile * gradient[:, np.newaxis, np.newaxis]

                    img_array[i : i + tile_size, j : j + tile_size] = np.clip(
                        tile, 0, 255
                    )

        elif pattern_type == "noise":
            # Perlin-like noise pattern
            noise = np.random.randn(size[0] // 4, size[1] // 4, 3) * 30
            from scipy.ndimage import zoom

            noise = zoom(noise, (4, 4, 1), order=1)
            img_array = np.clip(base_color + noise, 0, 255).astype(np.uint8)

        elif pattern_type == "gradient":
            # Gradient pattern
            gradient_x = np.linspace(0, 1, size[0])
            gradient_y = np.linspace(0, 1, size[1])
            gradient = np.outer(gradient_x, gradient_y)

            for c in range(3):
                channel = base_color[c] * (0.7 + 0.3 * gradient)
                img_array[:, :, c] = np.clip(channel, 0, 255)

        else:  # mixed
            # Combination of patterns
            img_array[:] = base_color
            # Add random shapes
            num_shapes = np.random.randint(5, 15)
            for _ in range(num_shapes):
                x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
                radius = np.random.randint(20, 80)
                variation = np.random.randint(-40, 40, 3)
                color = np.clip(np.array(base_color) + variation, 0, 255)

                y_coords, x_coords = np.ogrid[: size[0], : size[1]]
                mask = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius**2
                img_array[mask] = color

    else:  # sprites
        # Create sprite-like shapes with transparency
        img_array[:] = (255, 255, 255)  # White background

        # Main sprite shape
        sprite_type = np.random.choice(["character", "item", "effect"])

        if sprite_type == "character":
            # Simple character silhouette
            center_x, center_y = size[0] // 2, size[1] // 2

            # Body
            body_radius = np.random.randint(60, 100)
            y_coords, x_coords = np.ogrid[: size[0], : size[1]]
            body_mask = (x_coords - center_x) ** 2 + (
                y_coords - center_y
            ) ** 2 <= body_radius**2
            img_array[body_mask] = base_color

            # Head
            head_y = center_y - body_radius - 30
            head_radius = np.random.randint(30, 50)
            head_mask = (x_coords - center_x) ** 2 + (
                y_coords - head_y
            ) ** 2 <= head_radius**2
            img_array[head_mask] = np.clip(np.array(base_color) * 0.9, 0, 255)

        elif sprite_type == "item":
            # Item shapes (gems, coins, etc.)
            num_facets = np.random.randint(4, 8)
            center = np.array([size[0] // 2, size[1] // 2])
            radius = np.random.randint(50, 100)

            for i in range(num_facets):
                angle = i * 2 * np.pi / num_facets
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))

                # Draw triangular facets
                variation = np.random.randint(-20, 20, 3)
                color = np.clip(np.array(base_color) + variation, 0, 255)

                # Simple filled shape
                y_coords, x_coords = np.ogrid[: size[0], : size[1]]
                dist_from_center = np.sqrt(
                    (x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2
                )
                dist_from_point = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
                mask = (dist_from_center < radius) & (dist_from_point < radius // 2)
                img_array[mask] = color

        else:  # effect
            # Particle effects
            num_particles = np.random.randint(20, 50)
            for _ in range(num_particles):
                x, y = np.random.randint(50, size[0] - 50, 2)
                radius = np.random.randint(5, 20)
                variation = np.random.randint(-50, 50, 3)
                color = np.clip(np.array(base_color) + variation, 0, 255)

                y_coords, x_coords = np.ogrid[: size[0], : size[1]]
                mask = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius**2

                # Soft edges
                distances = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
                alpha = np.clip(1 - distances / radius, 0, 1)

                for c in range(3):
                    img_array[:, :, c][mask] = (
                        alpha[mask] * color[c]
                        + (1 - alpha[mask]) * img_array[:, :, c][mask]
                    )

    return Image.fromarray(img_array.astype(np.uint8))


def download_real_assets(output_dir: Path | str) -> None:
    """Placeholder for downloading real game assets"""
    print("\nNote: Real asset downloading not implemented")
    print("To use real Dragon Hills assets:")
    print("1. Extract game assets using appropriate tools")
    print("2. Organize by biome and asset type")
    print("3. Place in data/raw/{asset_type}/{biome}/")
    print("\nAsset structure:")
    print("  - textures/forest/*.png")
    print("  - textures/desert/*.png")
    print("  - sprites/characters/*.png")
    print("  - etc.")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate game assets for MADWE")
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
        "--download-real", action="store_true", help="Instructions for real assets"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.download_real:
        download_real_assets(output_dir)
    else:
        generate_synthetic_assets(
            output_dir, num_samples=args.num_samples, size=tuple(args.size)
        )

        print(f"\nSynthetic assets generated in {output_dir / 'raw'}")
        print("Replace with real game assets for production use")


if __name__ == "__main__":
    main()
