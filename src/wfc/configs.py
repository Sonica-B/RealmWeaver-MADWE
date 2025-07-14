#!/usr/bin/env python3
import json
from pathlib import Path

# 1. Point to your raw texture root
RAW_ROOT = Path("data/raw/textures")

# 2. Point to where your configs live (will be created if needed)
OUT_ROOT = Path("configs/biomes")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 3. For each sub‐folder in data/raw/textures (each biome)…
for biome_dir in RAW_ROOT.iterdir():
    if not biome_dir.is_dir():
        continue

    biome_name = biome_dir.name
    print(f"Generating config for biome: {biome_name}")

    # 4. Gather all .png files in that folder
    tiles = sorted([p.name for p in biome_dir.glob("*.png")])
    if not tiles:
        print(f"  → WARNING: no .png files found under {biome_dir}")
        continue

    # 5. Build the adjacency map (here: allow any tile next to any other)
    adjacency = {t: tiles for t in tiles}

    # 6. Optional: uniform weights
    weights = [1] * len(tiles)

    # 7. Assemble the config dict
    cfg = {
        "tiles": tiles,
        "adjacency": adjacency,
        "weights": weights
    }

    # 8. Write it out as JSON
    out_file = OUT_ROOT / f"{biome_name}.json"
    with open(out_file, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  → Wrote {out_file}")

print("Done.")
