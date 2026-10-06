"""Predictor: order-2 Markov chain over eight quantised headings, falling back to order-1, then constant velocity.

The predicted heading distribution becomes visit probabilities for the eight neighbouring chunks by casting
rays from the last observed position: a heading's probability goes to whichever chunk its ray leaves into.
"""

from __future__ import annotations

import math
from collections import Counter, deque

Key = tuple[int, int]

SECTORS = 8
# Neighbour offset of each heading sector: sector j points along angle j * 45 degrees.
RING: tuple[Key, ...] = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
# Constant-velocity prior: probability mass by sector offset from the last heading (offset 0 = keep going).
_PRIOR = (0.6, 0.15, 0.05, 0.0, 0.0, 0.0, 0.05, 0.15)
_PSEUDO = 1.0  # pseudo-count of the prior inside the smoothed Markov estimate
_RAYS = (-math.pi / 12, 0.0, math.pi / 12)  # three rays per 45-degree sector
_MIN_STEP = 1e-6  # smaller displacements carry no heading


def _sector(angle: float) -> int:
    return round(angle / (math.pi / 4)) % SECTORS


def _time_to_edge(offset: float, v: float, size: float) -> float:
    """Travel time from `offset` inside [0, size) to the edge in the direction of `v`."""
    if v > 1e-12:
        return (size - offset) / v
    if v < -1e-12:
        return offset / -v
    return math.inf


def _exit_chunk(px: float, py: float, cx: int, cy: int, size: float, angle: float) -> Key:
    """The chunk a straight ray from (px, py) at `angle` enters when it leaves chunk (cx, cy)."""
    vx, vy = math.cos(angle), math.sin(angle)
    tx, ty = _time_to_edge(px - cx * size, vx, size), _time_to_edge(py - cy * size, vy, size)
    sx, sy = (1 if vx > 0 else -1), (1 if vy > 0 else -1)
    if abs(tx - ty) < 1e-9:
        return (cx + sx, cy + sy)
    return (cx + sx, cy) if tx < ty else (cx, cy + sy)


class Predictor:
    """Feed `observe` with player positions in tile units; `rank` returns the next-chunk distribution."""

    def __init__(self, chunk_size: int = 16) -> None:
        self.chunk_size = chunk_size
        self._pos: tuple[float, float] | None = None
        self._headings: deque[int] = deque(maxlen=2)
        self._order2: dict[tuple[int, int], Counter[int]] = {}
        self._order1: dict[int, Counter[int]] = {}

    def observe(self, pos: tuple[float, float]) -> None:
        x, y = float(pos[0]), float(pos[1])
        if self._pos is not None:
            dx, dy = x - self._pos[0], y - self._pos[1]
            if math.hypot(dx, dy) >= _MIN_STEP:
                h, history = _sector(math.atan2(dy, dx)), tuple(self._headings)
                if len(history) == 2:
                    self._order2.setdefault(history, Counter())[h] += 1
                if history:
                    self._order1.setdefault(history[-1], Counter())[h] += 1
                self._headings.append(h)
        self._pos = (x, y)

    def heading_distribution(self) -> list[float] | None:
        """P(next heading sector): order-2 counts if that context was seen, else order-1, else the constant-velocity
        prior around the last heading; counts are smoothed with the prior. None before any movement."""
        if not self._headings:
            return None
        last = self._headings[-1]
        prior = [_PRIOR[(j - last) % SECTORS] for j in range(SECTORS)]
        contexts = (
            self._order2.get(tuple(self._headings)) if len(self._headings) == 2 else None,
            self._order1.get(last),
        )
        for counts in contexts:
            if counts:
                n = counts.total()
                return [(counts[j] + _PSEUDO * prior[j]) / (n + _PSEUDO) for j in range(SECTORS)]
        return prior

    def rank(self, current: Key, k: int = 8) -> list[tuple[Key, float]]:
        """The neighbours of `current` most likely entered next, best first, with probabilities summing to <= 1."""
        dist = self.heading_distribution()
        if dist is None:
            return self.ring_baseline(current)[:k]
        cx, cy, size = current[0], current[1], float(self.chunk_size)
        px, py = (cx + 0.5) * size, (cy + 0.5) * size
        if (
            self._pos is not None
            and cx * size <= self._pos[0] < (cx + 1) * size
            and cy * size <= self._pos[1] < (cy + 1) * size
        ):
            px, py = self._pos
        probs: Counter[Key] = Counter()
        for j, pj in enumerate(dist):
            for offset in _RAYS:
                probs[_exit_chunk(px, py, cx, cy, size, j * math.pi / 4 + offset)] += pj / len(_RAYS)
        ring = [(cx + dx, cy + dy) for dx, dy in RING]
        return [(c, float(probs[c])) for c in sorted(ring, key=lambda c: -probs[c])[:k]]

    def ring_baseline(self, current: Key) -> list[tuple[Key, float]]:
        """The 8-neighbour ring, uniform 1/8 each, in heading-sector order."""
        return [((current[0] + dx, current[1] + dy), 1 / SECTORS) for dx, dy in RING]
