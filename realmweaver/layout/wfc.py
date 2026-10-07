"""Simple-tiled wave function collapse over a `TileSet`: min-entropy observation, AC-3 propagation, restarts.

The wave is `bool[H, W, T]`; cell `(y, x)` may still become class `t` while `wave[y, x, t]` is True.
"""

from __future__ import annotations

import logging

import numpy as np

from realmweaver.types import DIRS, Layout, TileSet, opposite

log = logging.getLogger(__name__)

_NOISE = 1e-6  # tie-breaker on entropy; far below any real entropy gap


class Contradiction(RuntimeError):
    """Some cell has no allowed tile class left, even after restarts and the centre-block retry."""


def solve(
    tileset: TileSet,
    width: int,
    height: int,
    seed: int,
    fixed: dict[tuple[int, int], str] | None = None,
    max_restarts: int = 5,
) -> Layout:
    """Solve a `height` x `width` layout. `fixed` maps `(x, y)` to a class name that cell must take."""
    init = _blank_wave(tileset, width, height)
    for (x, y), name in (fixed or {}).items():
        init[y, x] = False
        init[y, x, tileset.index(name)] = True
    return Layout(grid=_solve(tileset, init, seed, max_restarts), tileset=tileset)


def solve_chunk(
    tileset: TileSet, size: int, seed: int, neighbours: dict[str, Layout], max_restarts: int = 5
) -> Layout:
    """Solve a `size` x `size` chunk whose edges are compatible with the ready neighbours (`"N"`, `"E"`, `"S"`, `"W"`).

    Each border cell is restricted to the classes allowed next to the neighbour's touching edge cell, in both
    directions, so `allowed` holds across the border exactly as it does inside a layout.
    """
    init = _blank_wave(tileset, size, size)
    allowed = tileset.allowed
    for d, key in enumerate("NESW"):
        if key not in neighbours:
            continue
        other = neighbours[key].grid
        if other.shape != (size, size):
            raise ValueError(f"neighbour {key} is {other.shape[1]}x{other.shape[0]}, chunk is {size}x{size}")
        ours = [init[0, :], init[:, -1], init[-1, :], init[:, 0]][d]
        theirs = [other[-1, :], other[:, 0], other[0, :], other[:, -1]][d]
        mask = allowed[:, d, theirs].T & allowed[theirs, opposite(d), :]
        # ponytail: diagonal chunks are never constrained against each other, so two ready neighbours can make
        # incompatible demands on a corner cell. The earlier neighbour (N, E, S, W order) keeps that cell and the
        # seam shows one mismatch there; the upgrade path is a corner protocol in the world agent.
        ok = (ours & mask).any(axis=1)
        if not ok.all():
            log.warning(
                "chunk border %s: %d cell(s) keep an earlier neighbour's constraint", key, int((~ok).sum())
            )
        ours[ok] &= mask[ok]
    return Layout(grid=_solve(tileset, init, seed, max_restarts), tileset=tileset)


def violations(layout: Layout) -> int:
    """Brute-force count of adjacent pairs the tileset forbids; independent of the solver's matrix ops."""
    grid, allowed = layout.grid, layout.tileset.allowed
    height, width = grid.shape
    bad = 0
    for y in range(height):
        for x in range(width):
            for d in (1, 2):  # E and S: each unordered pair exactly once
                dx, dy = DIRS[d]
                if x + dx < width and y + dy < height:
                    a, b = int(grid[y, x]), int(grid[y + dy, x + dx])
                    bad += not (allowed[a, d, b] and allowed[b, opposite(d), a])
    return bad


def _blank_wave(tileset: TileSet, width: int, height: int) -> np.ndarray:
    # Classes with zero weight (in the legend, absent from the map) can never be placed.
    return np.broadcast_to(tileset.weights > 0, (height, width, len(tileset))).copy()


def _solve(tileset: TileSet, init: np.ndarray, seed: int, max_restarts: int) -> np.ndarray:
    grid = _restarts(tileset, init, seed, max_restarts)
    height, width, _ = init.shape
    ch, cw = height // 2, width // 2
    if grid is None and ch and cw:
        # ponytail: no backjumping. Shrink instead: solve the centre block on its own, pin it, and retry the
        # full grid around it. Upgrade path is POMS-style boundary erosion or true backtracking.
        y0, x0 = (height - ch) // 2, (width - cw) // 2
        core = _restarts(tileset, init[y0 : y0 + ch, x0 : x0 + cw], seed, max_restarts)
        if core is not None:
            pinned = init.copy()
            pinned[y0 : y0 + ch, x0 : x0 + cw] = np.eye(len(tileset), dtype=bool)[core]
            grid = _restarts(tileset, pinned, seed + max_restarts + 1, max_restarts)
    if grid is None:
        raise Contradiction(
            f"no {width}x{height} layout after {max_restarts} restarts and the centre-block retry"
        )
    return grid


def _restarts(tileset: TileSet, init: np.ndarray, seed: int, max_restarts: int) -> np.ndarray | None:
    for k in range(max_restarts + 1):
        try:
            return _attempt(tileset, init, np.random.default_rng(seed + k))
        except Contradiction:
            continue
    return None


def _attempt(tileset: TileSet, init: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    allowed, weights = tileset.allowed, tileset.weights
    wave = init.copy()
    height, width, count = wave.shape
    wlogw = np.where(weights > 0, weights * np.log(np.where(weights > 0, weights, 1.0)), 0.0)
    _propagate(wave, allowed, [(y, x) for y in range(height) for x in range(width)])
    while True:
        undecided = wave.sum(axis=-1) > 1
        if not undecided.any():
            return wave.argmax(axis=-1).astype(np.int32)
        # Shannon entropy of the weights still possible in each cell: log(S) - sum(w log w) / S.
        total = wave @ weights
        entropy = np.log(total) - (wave @ wlogw) / total + rng.random((height, width)) * _NOISE
        entropy[~undecided] = np.inf
        y, x = np.unravel_index(int(np.argmin(entropy)), (height, width))
        p = weights * wave[y, x]
        wave[y, x] = False
        wave[y, x, rng.choice(count, p=p / p.sum())] = True
        _propagate(wave, allowed, [(y, x)])


def _propagate(wave: np.ndarray, allowed: np.ndarray, stack: list[tuple[int, int]]) -> None:
    """AC-3: pop a changed cell, shrink each neighbour to the classes some remaining class of ours supports."""
    height, width, _ = wave.shape
    while stack:
        y, x = stack.pop()
        for d, (dx, dy) in enumerate(DIRS):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            current = wave[ny, nx]
            shrunk = current & (wave[y, x] @ allowed[:, d, :])
            if (shrunk != current).any():
                if not shrunk.any():
                    raise Contradiction(f"cell (x={nx}, y={ny}) has no allowed class left")
                current[:] = shrunk
                stack.append((ny, nx))
