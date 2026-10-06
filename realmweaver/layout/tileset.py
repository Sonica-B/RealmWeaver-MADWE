"""Tilesets learned from a biome's example map: every adjacent pair present is allowed, counts become weights."""

from __future__ import annotations

import numpy as np

from realmweaver.types import DIRS, TileSet


def tileset_from_example(map_text: str, legend: dict[str, str]) -> TileSet:
    """`allowed[a, d, b]` is True iff class `b` appears in direction `d` of class `a` somewhere in the map."""
    rows = [line.strip() for line in map_text.splitlines() if line.strip()]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("example map rows must all have the same length")
    classes = list(dict.fromkeys(legend.values()))
    index = {name: i for i, name in enumerate(classes)}
    grid = np.array([[index[legend[ch]] for ch in row] for row in rows], dtype=np.int32)
    height, width = grid.shape
    count = len(classes)
    allowed = np.zeros((count, 4, count), dtype=bool)
    ys, xs = np.indices(grid.shape)
    for d, (dx, dy) in enumerate(DIRS):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        allowed[grid[inside], d, grid[ny[inside], nx[inside]]] = True
    counts = np.bincount(grid.ravel(), minlength=count).astype(np.float64)
    return TileSet(classes=classes, allowed=allowed, weights=counts / counts.sum())
