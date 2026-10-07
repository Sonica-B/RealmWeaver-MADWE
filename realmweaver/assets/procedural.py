"""CPU adapter for the `Generator` seam: deterministic value noise on a torus, coloured from the tile class palette.

Textures are periodic by construction (the noise lattice wraps), so they are seamless without post-processing.
Sprites are the same noise cut to a centred blob over a white background with a hard alpha mask. A mesh is a unit
cube wearing its subject's texture, the placeholder until a mesh model sits behind the seam. Used by tests, as the
bench baseline, and on any machine without a GPU.
"""

from __future__ import annotations

import time
import zlib

import numpy as np

from realmweaver import gltf
from realmweaver.biomes import load_biome
from realmweaver.types import Asset, AssetSpec, MeshPayload, Payload, SpritePayload, TexturePayload

_GREY = ("#555555", "#808080", "#aaaaaa")  # palette for a subject the biome does not define
_MAX_OCTAVES = 6  # lattices of 4, 8, ... cells; octaves finer than 2 px per cell are never sampled
_KINDS = ("texture", "sprite", "mesh")
# A face's four corners, counter-clockwise seen from outside, and the texture corner each one wears.
_QUAD = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], np.float32)
_UVS = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], np.float32)


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


def _cube() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A unit cube as positions [24, 3], normals [24, 3], uvs [24, 2] and indices [36]: four vertices per face, so
    every face carries its own normal and the full texture."""
    positions, normals = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            if sign < 0:
                u, v = v, u  # mirrored, so the quad stays counter-clockwise seen from outside
            quad = np.zeros((4, 3), np.float32)
            quad[:, axis], quad[:, u], quad[:, v] = 0.5 * sign, 0.5 * _QUAD[:, 0], 0.5 * _QUAD[:, 1]
            normal = np.zeros((4, 3), np.float32)
            normal[:, axis] = sign
            positions.append(quad)
            normals.append(normal)
    indices = (np.arange(0, 24, 4)[:, None] + np.array([[0, 1, 2, 0, 2, 3]])).ravel()
    return np.concatenate(positions), np.concatenate(normals), np.tile(_UVS, (6, 1)), indices


def _mesh(texture: np.ndarray, name: str) -> MeshPayload:
    """The placeholder mesh: a unit cube wearing `texture` on every face."""
    # ponytail: every mesh subject is the same cube in its subject's texture, enough to carry a GLB through the
    # bridge end to end; a mesh model adapter (TRELLIS.2 or TripoSG, docs/research/07) is the upgrade path.
    positions, normals, uvs, indices = _cube()
    png, _ = TexturePayload(texture).encode()
    glb = gltf.textured_mesh(positions, normals, uvs, indices, png, name)
    watertight = gltf.is_watertight(positions, indices.reshape(-1, 3))
    return MeshPayload(
        glb, triangles=len(indices) // 3, watertight=watertight, textures={"baseColor": texture}
    )


class ProceduralGenerator:
    """`Generator` adapter that needs no model: same spec, same payload, on any machine."""

    def __init__(self, seed_salt: int = 0) -> None:
        self.seed_salt = seed_salt

    def generate(self, spec: AssetSpec) -> Asset:
        start = time.perf_counter()
        if spec.kind not in _KINDS:
            raise RuntimeError(
                f"ProceduralGenerator makes {', '.join(_KINDS)} assets, not {spec.kind}; a {spec.kind} needs a "
                "model adapter behind the Generator seam"
            )
        tile = load_biome(spec.biome).tiles.get(spec.subject)
        palette = tile.palette if tile is not None else _GREY
        pixels = "sprite" if spec.kind == "sprite" else "texture"  # a mesh wears its subject's texture
        subject_hash = zlib.crc32(f"{spec.biome}/{pixels}/{spec.subject}".encode())
        rng = np.random.default_rng([spec.seed % (1 << 32), self.seed_salt % (1 << 32), subject_hash])
        noise = _fbm(rng, spec.size)
        image = _colour_ramp(noise, palette)
        payload: Payload
        if spec.kind == "sprite":
            payload = SpritePayload(_sprite(image, noise))
        elif spec.kind == "mesh":
            payload = _mesh(image, spec.subject or "mesh")
        else:
            payload = TexturePayload(image)
        return Asset(spec=spec, payload=payload, latency_s=time.perf_counter() - start)
