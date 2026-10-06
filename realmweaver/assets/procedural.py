"""CPU adapter for the `Generator` seam: deterministic value noise on a torus, coloured from the tile class palette.

Textures are periodic by construction (the noise lattice wraps), so they are seamless without post-processing.
Sprites are the same noise cut to a centred blob over a white background with a hard alpha mask. Used by tests,
as the bench baseline, and on any machine without a GPU.
"""

from __future__ import annotations

import time
import zlib

import numpy as np

from realmweaver.biomes import load_biome
from realmweaver.types import Asset, AssetSpec

_GREY = ("#555555", "#808080", "#aaaaaa")  # palette for a subject the biome does not define
_MAX_OCTAVES = 6  # lattices of 4, 8, ... cells; octaves finer than 2 px per cell are never sampled


def _value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Periodic value noise in [0, 1]: a random `cells x cells` lattice bilinearly interpolated on a torus.

    Each axis gets a random phase, so the lattice lines of the octaves never pile up on the tile edge.
    """
    lattice = rng.random((cells, cells), dtype=np.float32)
    phase = rng.random(2, dtype=np.float32) * cells
    pos = np.arange(size, dtype=np.float32)[None, :] * (cells / size) + phase[:, None]  # (2, size): y, x
    idx = pos.astype(np.intp)
    t = pos - idx
    t = t * t * (3 - 2 * t)  # smoothstep hides the lattice lines
    idx %= cells
    nxt = (idx + 1) % cells  # the lattice point after the last one is the first: that is the torus
    rows = lattice[:, idx[1]]
    rows += (lattice[:, nxt[1]] - rows) * t[1]  # (cells, size): interpolated along x
    out = rows[idx[0]]
    out += (rows[nxt[0]] - out) * t[0][:, None]  # (size, size): then along y
    return out


def _fbm(rng: np.random.Generator, size: int) -> np.ndarray:
    """Octaves with doubling frequency and halving amplitude, min-max normalised to [0, 1]."""
    total = np.zeros((size, size), np.float32)
    for octave in range(_MAX_OCTAVES):
        cells = 4 << octave
        if cells > size // 2:
            break
        total += _value_noise(rng, size, cells) * (0.5**octave)
    lo, hi = float(total.min()), float(total.max())
    return (total - lo) / max(hi - lo, 1e-6)


def _colour_ramp(noise: np.ndarray, palette: tuple[str, ...]) -> np.ndarray:
    """Map noise in [0, 1] through the palette's colours as evenly spaced stops via a 256-entry LUT; uint8 HxWx3."""
    stops = np.array([[int(c.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] for c in palette], np.float32)
    x = np.linspace(0.0, 1.0, len(stops), dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(grid, x, stops[:, channel]) for channel in range(3)], axis=-1)
    return np.rint(lut).astype(np.uint8)[np.rint(noise * 255).astype(np.uint8)]  # one gather per pixel


def _sprite(rgb: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Cut a centred blob with a noise-wobbled radius; white and transparent outside it; uint8 HxWx4."""
    size = rgb.shape[0]
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2
    radius = np.hypot(axis[:, None], axis[None, :]) / (size / 2)  # 0 at the centre, ~1.41 at a corner
    inside = radius < 0.6 + 0.3 * (noise - 0.5)  # blob radius wobbles within [0.45, 0.75]
    out = np.dstack([rgb, np.where(inside, 255, 0).astype(np.uint8)])
    out[~inside, :3] = 255  # white under transparent pixels, the same convention as a keyed diffusion sprite
    return out


class ProceduralGenerator:
    """`Generator` adapter that needs no model: same spec, same pixels, on any machine."""

    def __init__(self, seed_salt: int = 0) -> None:
        self.seed_salt = seed_salt

    def generate(self, spec: AssetSpec) -> Asset:
        start = time.perf_counter()
        tile = load_biome(spec.biome).tiles.get(spec.subject)
        palette = tile.palette if tile is not None else _GREY
        subject_hash = zlib.crc32(f"{spec.biome}/{spec.kind}/{spec.subject}".encode())
        rng = np.random.default_rng([spec.seed % (1 << 32), self.seed_salt % (1 << 32), subject_hash])
        noise = _fbm(rng, spec.size)
        image = _colour_ramp(noise, palette)
        if spec.kind == "sprite":
            image = _sprite(image, noise)
        return Asset(spec=spec, image=image, latency_s=time.perf_counter() - start)
