import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Cardinal directions: North, East, South, West
Cardinal = Tuple[int, int]
DIRS: List[Cardinal] = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class OverlapModel:
    """Simple tiled WFC that fills a H×W grid with tile indices."""
    def __init__(self,
                 tiles: List[np.ndarray],
                 tile_names: List[str],
                 *,
                 out_size: Tuple[int, int]):
        self.tiles = tiles
        self.names = tile_names
        self.T = len(tiles)
        self.W, self.H = out_size

        self.wave = np.ones((self.H, self.W, self.T), dtype=bool)
        self.allowed = np.ones((self.T, 4, self.T), dtype=bool)

        self.weights = np.ones(self.T, dtype=float)
        self.weight_log = self.weights * np.log(self.weights)

    def load_rules(self,
                   adjacency: Dict[str, List[str]],
                   weights: List[float] = None):
        idx = {n:i for i,n in enumerate(self.names)}
        for src, dests in adjacency.items():
            s = idx[src]
            mask = np.zeros(self.T, dtype=bool)
            for d in dests:
                mask[idx[d]] = True
            for di in range(4):
                self.allowed[s, di] = mask
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
            self.weight_log = self.weights * np.log(self.weights)

    def _observe(self) -> bool:
        probs = self.wave * self.weights
        sums = probs.sum(axis=2)
        if (sums == 0).any():
            return False
        ent = (probs * (np.log(sums[:,:,None]) - np.log(self.weights+1e-9))).sum(axis=2)
        undecided = (self.wave.sum(axis=2) > 1)
        if not undecided.any():
            return True
        ent[~undecided] = np.inf
        yx = np.argwhere(np.isclose(ent, ent.min()))
        y, x = random.choice(yx)
        choices = np.where(self.wave[y,x])[0]
        pick = random.choices(choices, weights=self.weights[choices])[0]
        self.wave[y,x,:] = False
        self.wave[y,x,pick] = True
        self._stack = [(x,y)]
        return True

    def _propagate(self) -> bool:
        while self._stack:
            x,y = self._stack.pop()
            for d, (dx,dy) in enumerate(DIRS):
                nx, ny = x+dx, y+dy
                if not (0 <= nx < self.W and 0 <= ny < self.H): continue
                valid = np.zeros(self.T, dtype=bool)
                for t in np.where(self.wave[y,x])[0]:
                    valid |= self.allowed[t,d]
                before = self.wave[ny,nx].copy()
                self.wave[ny,nx] &= valid
                if not self.wave[ny,nx].any():
                    raise RuntimeError("Contradiction during propagate")
                if not np.array_equal(before, self.wave[ny,nx]):
                    self._stack.append((nx,ny))
        return (self.wave.sum(axis=2)==1).all()

    def run(self, max_iter=10_000) -> np.ndarray:
        for _ in range(max_iter):
            if not self._observe():
                raise RuntimeError("Contradiction on observe")
            if self._propagate():
                break
        return self.wave.argmax(axis=2)

    def render(self, grid: np.ndarray) -> Image.Image:
        th, tw, _ = self.tiles[0].shape
        canvas = Image.new("RGB", (self.W*tw, self.H*th))
        for y in range(self.H):
            for x in range(self.W):
                tile = Image.fromarray(self.tiles[int(grid[y,x])])
                canvas.paste(tile, (x*tw, y*th))
        return canvas


if __name__ == "__main__":
    import json, argparse, os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Basic WFC prototype with JSON export")
    parser.add_argument("--config", default="configs/biomes/forest.json",
                        help="Input biome spec (tiles + adjacency + weights)")
    parser.add_argument("--demo", action="store_true",
                        help="Save a PNG preview to data/samples/")
    parser.add_argument("--export-json", default=None,
                        help="Path to write flat tile-list JSON for Unity")
    args = parser.parse_args()

    # 1. Load config & tiles
    cfg = json.load(open(args.config))
    biome = Path(args.config).stem
    TILES_ROOT = Path("data/raw/textures")

    tiles = []
    for t in cfg["tiles"]:
        img_path = TILES_ROOT / biome / t
        if not img_path.exists():
            raise FileNotFoundError(f"Missing tile: {img_path}")
        tiles.append(np.array(Image.open(img_path).convert("RGB")))

    # 2. Run WFC
    model = OverlapModel(tiles, cfg["tiles"], out_size=(32,32))
    model.load_rules(cfg["adjacency"], cfg.get("weights"))
    grid = model.run()

    # 3. Render & save image
    img = model.render(grid)
    out_img_dir = Path("data/samples")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_img_dir / f"wfc_demo_{biome}.png"
    img.save(img_path)
    print(f"Saved demo image: {img_path}")

    # 4. Export flat JSON if requested
    if args.export_json:
        # Flatten grid → list of filenames
        flat = []
        for y in range(model.H):
            for x in range(model.W):
                idx = int(grid[y,x])
                flat.append(cfg["tiles"][idx])

        payload = {
            "tiles": flat,
            "width": model.W,
            "height": model.H
        }

        export_path = Path(args.export_json)
        os.makedirs(export_path.parent, exist_ok=True)
        with open(export_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Exported tile-list JSON: {export_path}")
