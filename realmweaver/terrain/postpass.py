"""The terrain post-pass, numpy only: D8 rivers, slope-cost roads, Whittaker biomes, settlement sites on flat land
near water, WFC village layouts and two silhouette landmarks. Arrays are HxW indexed `[y, x]`, heights in metres."""

from __future__ import annotations

import heapq
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from realmweaver.layout import solve, tileset_from_example
from realmweaver.types import Layout

log = logging.getLogger(__name__)

BIOMES: tuple[str, ...] = (
    "desert",
    "forest",
    "sky",
    "snow",
    "underwater",
    "volcanic",
)  # == sorted(biome_names())
SEA_LEVEL_M = 0.0
SKY_M = 2500.0  # above the clouds: the sky biome stands in for the design's sky islands until they exist
D8: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)  # (dx, dy)
_D8_DX = np.array([d[0] for d in D8])
_D8_DY = np.array([d[1] for d in D8])
_D8_LEN = np.hypot(_D8_DX, _D8_DY)

# Whittaker table over (temperature C, precipitation mm/yr) mapped to our biome names; first match wins, forest
# otherwise. Ranges are [low, high) with None open.
_WHITTAKER: tuple[tuple[str, tuple[float | None, float | None], tuple[float | None, float | None]], ...] = (
    ("snow", (None, 3.0), (None, None)),  # tundra and taiga
    ("desert", (None, None), (None, 250.0)),  # temperate and subtropical desert
    ("volcanic", (20.0, None), (None, 600.0)),  # hot shrubland and savanna: our barren basalt look
)

VILLAGE_LEGEND = {"g": "grass", "f": "field", "r": "road", "h": "house", "p": "plaza"}
VILLAGE_MAP = """
ffffggggffff
ffggrrrrggff
ggrrrhhrrrgg
grrhhrrhhrrg
grrrrpprrrrg
grrrrpprrrrg
grrhhrrhhrrg
ggrrrhhrrrgg
ffggrrrrggff
ffffggggffff
"""  # houses touch only roads and houses, so the solver can never drop one in open grass
# hand weights (the glossary's "tightened by hand rules"): the example's counts over-weight road and starve houses
VILLAGE_WEIGHTS = {"grass": 0.40, "field": 0.15, "road": 0.15, "house": 0.27, "plaza": 0.03}


# -- rivers ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rivers:
    """D8 flow: `direction[y, x]` indexes `D8` for the receiving neighbour (-1 at sinks and in the sea),
    `accumulation` is the rain draining through each cell in cells of mean rain, `mask` the river cells."""

    direction: np.ndarray
    accumulation: np.ndarray
    mask: np.ndarray


def rivers(
    height: np.ndarray,
    rain: np.ndarray | None = None,
    threshold: float = 100.0,
    sea_level: float = SEA_LEVEL_M,
) -> Rivers:
    """D8 flow direction (steepest drop per unit distance, strictly downhill), flow accumulation weighted by `rain`
    (unit mean; ones when None), river cells where the accumulation reaches `threshold` above sea level."""
    h = np.asarray(height, dtype=np.float64)
    rows, cols = h.shape
    padded = np.pad(h, 1, mode="edge")  # an edge neighbour at equal height is never the steepest drop
    best = np.zeros_like(h)
    direction = np.full(h.shape, -1, dtype=np.int8)
    for d, (dx, dy) in enumerate(D8):
        drop = (h - padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]) / _D8_LEN[d]
        steeper = drop > best
        best[steeper], direction[steeper] = drop[steeper], d
    direction[h <= sea_level] = -1
    weights = (
        np.ones(h.shape)
        if rain is None
        else np.asarray(rain, dtype=np.float64) / max(float(np.mean(rain)), 1e-9)
    )
    acc, flat_dir = weights.ravel().copy(), direction.ravel()
    # ponytail: a Python loop in descending height order (a cell drains to a strictly lower one, so its total is
    # final when visited) and no depression filling, so a pit or a flat ends its stream. Priority-flood (Barnes 2014)
    # with numba is the upgrade path.
    for idx in np.argsort(h, axis=None)[::-1]:
        d = flat_dir[idx]
        if d >= 0:
            acc[idx + _D8_DY[d] * cols + _D8_DX[d]] += acc[idx]
    accumulation = acc.reshape(h.shape).astype(np.float32)
    mask = (accumulation >= threshold) & (h > sea_level)
    sinks = int(((direction < 0) & (h > sea_level)).sum())
    log.info("rivers: %d river cells of %d, %d land sinks", int(mask.sum()), mask.size, sinks)
    return Rivers(direction, accumulation, mask)


# -- biomes ---------------------------------------------------------------------------------------------------


def biomes(
    height: np.ndarray,
    temperature: np.ndarray,
    moisture: np.ndarray,
    sea_level: float = SEA_LEVEL_M,
    sky_m: float = SKY_M,
) -> np.ndarray:
    """int8 HxW of indices into `BIOMES`: the Whittaker table over (temperature, precipitation) with two elevation
    overrides, underwater at or below `sea_level` and sky above `sky_m`."""
    h, t, m = (np.asarray(a, dtype=np.float64) for a in (height, temperature, moisture))
    out = np.full(h.shape, BIOMES.index("forest"), dtype=np.int8)
    for name, (t_lo, t_hi), (m_lo, m_hi) in reversed(_WHITTAKER):  # earlier rules land last and win
        hit = np.ones(h.shape, dtype=bool)
        for value, lo, hi in ((t, t_lo, t_hi), (m, m_lo, m_hi)):
            if lo is not None:
                hit &= value >= lo
            if hi is not None:
                hit &= value < hi
        out[hit] = BIOMES.index(name)
    out[h > sky_m] = BIOMES.index("sky")
    out[h <= sea_level] = BIOMES.index("underwater")
    return out


def biome_shares(class_map: np.ndarray) -> dict[str, float]:
    """Fraction of cells per biome name, every name present, summing to one."""
    counts = np.bincount(np.asarray(class_map).ravel(), minlength=len(BIOMES))
    return {name: float(c) / float(counts.sum()) for name, c in zip(BIOMES, counts, strict=True)}


# -- settlements ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """A settlement site: its cell, height, the mean rise over run of the 3x3 around it, and the Chebyshev distance
    in cells to the nearest river or sea cell."""

    x: int
    y: int
    height_m: float
    slope: float
    water_cells: int


def slope_map(height: np.ndarray, cell_m: float) -> np.ndarray:
    """Rise over run per cell from central differences, edges replicated."""
    h = np.pad(np.asarray(height, dtype=np.float64), 1, mode="edge")
    dzdx = (h[1:-1, 2:] - h[1:-1, :-2]) / (2.0 * cell_m)
    dzdy = (h[2:, 1:-1] - h[:-2, 1:-1]) / (2.0 * cell_m)
    return np.hypot(dzdx, dzdy)


def settlement_sites(
    height: np.ndarray,
    rivers: Rivers,
    n: int,
    cell_m: float,
    max_slope: float = 0.08,
    max_water_cells: int = 5,
    min_separation: int = 12,
    sea_level: float = SEA_LEVEL_M,
) -> list[Site]:
    """Up to `n` sites on flat land near water, best first: 3x3 mean slope at most `max_slope`, a river or sea cell
    within `max_water_cells`, above sea level and off the water, at least `min_separation` cells apart (Chebyshev).
    Score = flatness + half the water proximity, so flat beats close."""
    h = np.asarray(height, dtype=np.float64)
    local = _box_mean(slope_map(h, cell_m))
    water = rivers.mask | (h <= sea_level)
    dist = _chebyshev_distance(water, max_water_cells)
    score = (1.0 - local / max_slope) + 0.5 * (1.0 - dist / max_water_cells)
    score[(local > max_slope) | (dist > max_water_cells) | water] = -np.inf
    sites: list[Site] = []
    for _ in range(n):
        idx = int(np.argmax(score))
        if not np.isfinite(score.flat[idx]):
            break
        y, x = divmod(idx, h.shape[1])
        sites.append(Site(x, y, float(h[y, x]), float(local[y, x]), int(dist[y, x])))
        y0, x0 = max(0, y - min_separation), max(0, x - min_separation)
        score[y0 : y + min_separation + 1, x0 : x + min_separation + 1] = -np.inf
    log.info("settlement sites: %d of %d requested", len(sites), n)
    return sites


def settlement_layout(site: Site, size: int = 12, seed: int = 0) -> Layout:
    """A `size` x `size` village micro-layout from the WFC over `VILLAGE_MAP`: the plaza is fixed at the centre with
    a road cross through it (the example map alone gives a soup), houses and fields follow the learned adjacency.
    The seed is folded with the site's cell, so each village differs and repeats across runs."""
    tileset = tileset_from_example(VILLAGE_MAP, VILLAGE_LEGEND)
    tileset.weights = np.array([VILLAGE_WEIGHTS[name] for name in tileset.classes], dtype=np.float64)
    tileset.weights /= tileset.weights.sum()
    village_seed = (seed * 1_000_003 + site.y * 65_536 + site.x) & 0x7FFF_FFFF
    centre, arm = size // 2, max(1, size // 3)
    fixed = {(centre, centre): "plaza"}
    for k in range(centre - arm, centre + arm + 1):
        fixed.setdefault((k, centre), "road")
        fixed.setdefault((centre, k), "road")
    return solve(tileset, size, size, seed=village_seed, fixed=fixed)


# -- roads ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Road:
    """An 8-connected path of `(x, y)` cells from settlement `a` to settlement `b` (indices into the site list) and
    its total cost; `cost` is infinite for a straight-line fallback."""

    a: int
    b: int
    cells: np.ndarray
    cost: float


@dataclass(frozen=True)
class Roads:
    roads: tuple[Road, ...]

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        for road in self.roads:
            out[road.cells[:, 1], road.cells[:, 0]] = True
        return out


def roads(
    settlements: Sequence[Site],
    height: np.ndarray,
    cell_m: float,
    rivers: Rivers | None = None,
    sea_level: float = SEA_LEVEL_M,
) -> Roads:
    """One least-cost path per edge of the settlements' Euclidean minimum spanning tree, so every settlement is
    reached with the fewest roads. A step costs its length times (1 + (slope / 0.1)^2): a 10 % grade doubles it and
    a 30 % grade is ten times a flat step; a river cell costs four times (a ford); the sea is impassable."""
    h = np.asarray(height, dtype=np.float64)
    if len(settlements) < 2:
        return Roads(())
    water = rivers.mask if rivers is not None else np.zeros(h.shape, dtype=bool)
    points = [(s.x, s.y) for s in settlements]
    out = []
    for a, b in _spanning_tree(points):
        cells, cost = _least_cost_path(h, points[a], points[b], cell_m, water, sea_level)
        if cells is None:
            # ponytail: a straight line when the sea separates two sites; a bridge or ferry rule is the upgrade path.
            log.warning("roads: no land path from settlement %d to %d, drawing a straight line", a, b)
            cells, cost = _line(points[a], points[b]), float("inf")
        out.append(Road(a, b, cells, cost))
    log.info("roads: %d roads, %d cells", len(out), sum(len(r.cells) for r in out))
    return Roads(tuple(out))


# -- landmarks ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Landmark:
    """A silhouette-first point of interest: `kind` is "summit" or "river_mouth"."""

    x: int
    y: int
    kind: str
    height_m: float


def landmarks(height: np.ndarray, rivers: Rivers) -> list[Landmark]:
    """The summit (highest cell) and the river mouth (the river cell with the largest accumulation; only when the
    window has a river)."""
    h = np.asarray(height, dtype=np.float64)
    y, x = np.unravel_index(int(np.argmax(h)), h.shape)
    out = [Landmark(int(x), int(y), "summit", float(h[y, x]))]
    if rivers.mask.any():
        y, x = np.unravel_index(int(np.argmax(np.where(rivers.mask, rivers.accumulation, -np.inf))), h.shape)
        out.append(Landmark(int(x), int(y), "river_mouth", float(h[y, x])))
    return out


# -- private --------------------------------------------------------------------------------------------------


def _box_mean(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    rows, cols = a.shape
    return (
        sum(p[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    )


def _chebyshev_distance(mask: np.ndarray, limit: int) -> np.ndarray:
    """Chebyshev distance to the nearest True cell, capped at `limit + 1`, by repeated 3x3 dilation."""
    dist = np.where(mask, 0, limit + 1).astype(np.int32)
    reach = mask.copy()
    rows, cols = mask.shape
    for k in range(1, limit + 1):
        p = np.pad(reach, 1, mode="constant")
        grown = reach.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= p[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
        dist[grown & ~reach] = k
        reach = grown
    return dist


def _spanning_tree(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Prim's minimum spanning tree over Euclidean distance, as (parent, child) index pairs."""
    p = np.asarray(points, dtype=np.float64)
    done = np.zeros(len(p), dtype=bool)
    done[0] = True
    dist = np.hypot(*(p - p[0]).T)
    parent = np.zeros(len(p), dtype=int)
    edges = []
    for _ in range(len(p) - 1):
        dist[done] = np.inf
        j = int(np.argmin(dist))
        edges.append((int(parent[j]), j))
        done[j] = True
        dj = np.hypot(*(p - p[j]).T)
        closer = (dj < dist) & ~done
        parent[closer], dist[closer] = j, dj[closer]
    return edges


def _least_cost_path(
    h: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    cell_m: float,
    water: np.ndarray,
    sea_level: float,
) -> tuple[np.ndarray | None, float]:
    """Dijkstra over the 8-connected grid with the slope cost of `roads`; None when the goal is unreachable."""
    rows, cols = h.shape
    s, g = start[1] * cols + start[0], goal[1] * cols + goal[0]
    flat_h, flat_water, passable = h.ravel(), water.ravel(), (h > sea_level).ravel()
    best = np.full(h.size, np.inf)
    best[s] = 0.0
    prev = np.full(h.size, -1, dtype=np.int64)
    heap = [(0.0, s)]
    # ponytail: pure-Python Dijkstra, fine for a 133x133 region and slow past 512x512; scipy.sparse.csgraph or a
    # coarse-to-fine search is the upgrade path.
    while heap:
        cost, idx = heapq.heappop(heap)
        if idx == g:
            break
        if cost > best[idx]:
            continue
        y, x = divmod(idx, cols)
        for d in range(8):
            nx, ny = x + D8[d][0], y + D8[d][1]
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            n = ny * cols + nx
            if not passable[n]:
                continue
            length = _D8_LEN[d] * cell_m
            slope = abs(flat_h[n] - flat_h[idx]) / length
            step = length * (1.0 + (slope / 0.1) ** 2) * (4.0 if flat_water[n] else 1.0)
            if cost + step < best[n]:
                best[n], prev[n] = cost + step, idx
                heapq.heappush(heap, (cost + step, n))
    if not np.isfinite(best[g]):
        return None, float("inf")
    path = [g]
    while path[-1] != s:
        path.append(int(prev[path[-1]]))
    return np.array([(i % cols, i // cols) for i in reversed(path)], dtype=np.int32), float(best[g])


def _line(a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(np.int32)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(np.int32)
    return np.stack([xs, ys], axis=1)
