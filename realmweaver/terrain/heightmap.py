"""Heightmap sources: seed-consistent terrain with random access by cell coordinates.

`DiffusionHeightmap` reads `xandergos/terrain-diffusion-30m` (MIT) through the model's own HTTP API, because its
package lives in its own venv (`python -m terrain_diffusion.inference.api` there); `NoiseHeightmap` is value-noise
fBm in numpy for tests and as the fallback. Both return a `Terrain`: height in metres above sea level, temperature in
degrees C and moisture as annual precipitation in mm, as float32 HxW arrays indexed `[y, x]`.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Terrain:
    """One window of a heightmap source: `x0, y0` is its top-left cell on the source's plane and `cell_m` the metres
    per cell."""

    height: np.ndarray
    temperature: np.ndarray
    moisture: np.ndarray
    cell_m: float
    x0: int = 0
    y0: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.height.shape[0]), int(self.height.shape[1])


class HeightmapSource(Protocol):
    """The terrain seam: random access by cell coordinates on an unbounded plane. The same seed and window always
    give the same arrays, and a window equals the matching crop of any larger window (seed consistency)."""

    cell_m: float

    def sample(self, x0: int, y0: int, width: int, height: int) -> Terrain: ...


class DiffusionHeightmap:
    """`xandergos/terrain-diffusion-30m` behind its API server (`GET /terrain`): elevation as int16 metres plus the
    WorldClim-style climate layers, native 30 m per cell. `scale` > 1 asks the server for its bilinear upsample (4
    gives 7.5 m cells), which adds no detail. Windows are assembled from `tile` x `tile` requests aligned to the
    tile grid (None requests the exact window). Raises RuntimeError naming the server when it is not reachable."""

    NATIVE_M = 30.0

    def __init__(
        self, url: str, seed: int, scale: int = 1, tile: int | None = 1024, timeout_s: float = 900.0
    ) -> None:
        self.url = url.rstrip("/")
        self.seed = int(seed)
        self.scale = int(scale)
        self.tile = tile
        self.timeout_s = timeout_s
        self.cell_m = self.NATIVE_M / self.scale

    def sample(self, x0: int, y0: int, width: int, height: int) -> Terrain:
        # ponytail: the pipeline's Laplacian denoise re-estimates the low band over the requested window, so a cell's
        # value depends on the window's extent (up to 9 m on a coast, see 12-terrain-spike.md) and margins do not
        # help; aligned tiles make every request for a cell byte-identical, and the server caches them. A float32
        # worker with a window-free denoise is the upgrade path.
        if self.tile is None:
            elev, temp, precip = self._fetch(x0, y0, width, height)
            return Terrain(elev, temp, precip, self.cell_m, x0, y0)
        t = self.tile
        tx0, ty0, tx1, ty1 = x0 // t, y0 // t, (x0 + width - 1) // t, (y0 + height - 1) // t
        rows = [
            [self._fetch(tx * t, ty * t, t, t) for tx in range(tx0, tx1 + 1)] for ty in range(ty0, ty1 + 1)
        ]
        oy, ox = y0 - ty0 * t, x0 - tx0 * t
        layers = (np.block([[cell[k] for cell in row] for row in rows]) for k in range(3))
        elev, temp, precip = (a[oy : oy + height, ox : ox + width].copy() for a in layers)
        return Terrain(elev, temp, precip, self.cell_m, x0, y0)

    def _fetch(self, x0: int, y0: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # the server indexes rows first: i is y, j is x; `seed` is a no-op once the server runs that seed
        query = urllib.parse.urlencode(
            {"i1": y0, "j1": x0, "i2": y0 + height, "j2": x0 + width, "scale": self.scale, "seed": self.seed}
        )
        try:
            with urllib.request.urlopen(f"{self.url}/terrain?{query}", timeout=self.timeout_s) as resp:
                raw, h, w = resp.read(), int(resp.headers["X-Height"]), int(resp.headers["X-Width"])
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"terrain-diffusion API at {self.url} rejected the window: {e.read()!r}"
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"terrain-diffusion API not reachable at {self.url}; start "
                f"`python -m terrain_diffusion.inference.api` in its venv: {e}"
            ) from e
        # ponytail: the server floors elevation to whole metres (int16 on the wire), so plains carry 1 m steps and
        # flats that stop D8 flow; a worker returning the pipeline's float32 elevation is the upgrade path.
        elev = np.frombuffer(raw[: h * w * 2], dtype="<i2").reshape(h, w).astype(np.float32)
        climate = np.frombuffer(raw[h * w * 2 :], dtype="<f4").reshape(
            h, w, 4
        )  # temp, t_season, precip, p_cv
        log.info("terrain-diffusion window %dx%d at (%d, %d), scale %d", w, h, x0, y0, self.scale)
        return elev, climate[..., 0].copy(), climate[..., 2].copy()


class NoiseHeightmap:
    """Value-noise fBm on hashed lattice points, so every cell is a pure function of (seed, x, y): random access and
    seed consistency hold exactly, and there is no erosion (the model owns realism). Height is a continent term (sea
    where it is negative), rolling hills and a ridged mountain term on land, scaled by `relief_m`; temperature falls
    6.5 degrees per km of height and rises with southing; moisture is a second noise field lifted by height."""

    def __init__(self, seed: int = 0, cell_m: float = 30.0, relief_m: float = 900.0) -> None:
        self.seed, self.cell_m, self.relief_m = int(seed), float(cell_m), float(relief_m)

    def sample(self, x0: int, y0: int, width: int, height: int) -> Terrain:
        ys, xs = np.mgrid[y0 : y0 + height, x0 : x0 + width].astype(np.float64)
        land = (_fbm(xs, ys, self.seed, 1 / 420.0, 3) - 0.38) * 3.0  # about -1..1, negative is sea
        hills = _fbm(xs, ys, self.seed + 17, 1 / 36.0, 5) - 0.5
        ridges = 1.0 - np.abs(2.0 * _fbm(xs, ys, self.seed + 29, 1 / 110.0, 4) - 1.0)
        h = self.relief_m * (0.45 * land + 0.25 * hills + 0.6 * ridges**2 * np.clip(land, 0.0, 1.0))
        km_y = ys * self.cell_m / 1000.0
        temperature = 16.0 - 6.5 * np.maximum(h, 0.0) / 1000.0 + 0.2 * km_y
        temperature += 6.0 * (_fbm(xs, ys, self.seed + 41, 1 / 200.0, 2) - 0.5)
        moisture = (
            150.0 + 1700.0 * _fbm(xs, ys, self.seed + 53, 1 / 160.0, 4) ** 1.5 + 0.3 * np.maximum(h, 0.0)
        )
        return Terrain(
            h.astype(np.float32),
            temperature.astype(np.float32),
            moisture.astype(np.float32),
            self.cell_m,
            x0,
            y0,
        )


def _fbm(
    x: np.ndarray, y: np.ndarray, seed: int, frequency: float, octaves: int, gain: float = 0.5
) -> np.ndarray:
    """Fractional Brownian motion of value noise in about [0, 1]; lacunarity 2."""
    total, norm, amplitude = np.zeros_like(x), 0.0, 1.0
    for octave in range(octaves):
        total += amplitude * _value_noise(x * frequency, y * frequency, seed + 101 * octave)
        norm += amplitude
        amplitude, frequency = amplitude * gain, frequency * 2.0
    return total / norm


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smoothstep-interpolated hashed lattice values in [0, 1)."""
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx, fy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    top = _hash(ix, iy, seed) * (1.0 - fx) + _hash(ix + 1, iy, seed) * fx
    bottom = _hash(ix, iy + 1, seed) * (1.0 - fx) + _hash(ix + 1, iy + 1, seed) * fx
    return top * (1.0 - fy) + bottom * fy


def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Lattice point to [0, 1): a 32-bit integer mix of (x, y, seed) that wraps on purpose."""
    h = ix.astype(np.uint32) * np.uint32(0x9E3779B1)
    h ^= iy.astype(np.uint32) * np.uint32(0x85EBCA77)
    h ^= np.uint32((seed * 0xC2B2AE3D) & 0xFFFFFFFF)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h.astype(np.float64) / 4294967296.0
