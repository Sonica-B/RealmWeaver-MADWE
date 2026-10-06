"""Shared types that cross the seams between layout, assets, world and bridge.

Images are numpy uint8 arrays: HxWx3 for textures, HxWx4 for sprites.
Directions are N, E, S, W = 0..3; `allowed[a, d, b]` reads "class b may sit in direction d of class a".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import Literal, Protocol

import numpy as np

Kind = Literal["texture", "sprite"]
Tier = Literal["draft", "refine"]

DIRS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIR_NAMES: tuple[str, ...] = ("N", "E", "S", "W")


def opposite(direction: int) -> int:
    return (direction + 2) % 4


@dataclass(frozen=True)
class AssetSpec:
    """Everything needed to (re)generate one asset. `subject` is the tile class for textures, the prop name for sprites."""

    biome: str
    kind: Kind = "texture"
    subject: str = ""
    size: int = 512
    seed: int = 0
    steps: int = 4
    seamless: bool = True
    tier: Tier = "draft"

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def id(self) -> str:
        return sha1(self.canonical().encode("utf-8")).hexdigest()[:16]


@dataclass
class Asset:
    spec: AssetSpec
    image: np.ndarray
    style_vec: np.ndarray | None = None
    latency_s: float = 0.0

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def nbytes(self) -> int:
        return int(self.image.nbytes)


class Generator(Protocol):
    """The asset agent seam. Two adapters: DiffusionGenerator (GPU) and ProceduralGenerator (CPU)."""

    def generate(self, spec: AssetSpec) -> Asset: ...


@dataclass
class TileSet:
    classes: list[str]
    allowed: np.ndarray  # bool [T, 4, T]
    weights: np.ndarray  # float [T], sums to 1

    def index(self, name: str) -> int:
        return self.classes.index(name)

    def __len__(self) -> int:
        return len(self.classes)


@dataclass
class Layout:
    grid: np.ndarray  # int32 HxW of indices into tileset.classes
    tileset: TileSet

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    def class_at(self, x: int, y: int) -> str:
        return self.tileset.classes[int(self.grid[y, x])]

    def class_rows(self) -> list[list[str]]:
        return [[self.tileset.classes[int(v)] for v in row] for row in self.grid]


@dataclass
class Chunk:
    cx: int
    cy: int
    biome: str
    layout: Layout
    asset_ids: dict[str, str] = field(default_factory=dict)  # tile class -> asset id
    state: Literal["pending", "draft", "ready"] = "pending"

    @property
    def key(self) -> tuple[int, int]:
        return (self.cx, self.cy)
