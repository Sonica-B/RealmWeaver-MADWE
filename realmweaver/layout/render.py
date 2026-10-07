"""Flat-colour preview of a layout, one block of `cell` x `cell` pixels per tile."""

from __future__ import annotations

import numpy as np

from realmweaver.types import Layout

_FALLBACK = "#808080"  # classes without a palette entry render grey, like the biome loader's default palette


def render(layout: Layout, palettes: dict[str, str], cell: int = 8) -> np.ndarray:
    """uint8 (H*cell) x (W*cell) x 3 image; `palettes` maps tile class to a hex colour such as the first palette entry."""
    colours = np.array(
        [_rgb(palettes.get(name, _FALLBACK)) for name in layout.tileset.classes], dtype=np.uint8
    )
    image = colours[layout.grid]
    return np.repeat(np.repeat(image, cell, axis=0), cell, axis=1)


def _rgb(colour: str) -> tuple[int, int, int]:
    r, g, b = bytes.fromhex(colour.lstrip("#")[:6])
    return r, g, b
