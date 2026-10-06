"""Biome definitions: one folder per biome holding `biome.yaml` (style, tile classes, props, legend, example map)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent


@dataclass(frozen=True)
class TileClass:
    name: str
    prompt: str
    palette: tuple[str, ...]


@dataclass(frozen=True)
class Biome:
    name: str
    style: str
    negative: str
    coherence_threshold: float
    legend: dict[str, str]
    tiles: dict[str, TileClass]
    props: dict[str, str]
    lora: str | None
    example_map: str

    def tile_classes(self) -> list[str]:
        return list(self.tiles)


def biome_names() -> list[str]:
    return sorted(p.parent.name for p in _ROOT.glob("*/biome.yaml"))


@cache
def load_biome(name: str) -> Biome:
    path = _ROOT / name / "biome.yaml"
    if not path.is_file():
        raise KeyError(f"unknown biome {name!r}; available: {biome_names()}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tiles = {
        cls: TileClass(cls, str(spec["prompt"]), tuple(str(c) for c in spec.get("palette", ["#808080"])))
        for cls, spec in raw["tiles"].items()
    }
    legend = {str(k): str(v) for k, v in raw["legend"].items()}
    example_map = "\n".join(line for line in str(raw["map"]).splitlines() if line.strip())
    missing = {v for v in legend.values()} - set(tiles)
    if missing:
        raise ValueError(f"biome {name}: legend classes without tile definitions: {sorted(missing)}")
    unknown = {ch for ch in example_map if ch != "\n"} - set(legend)
    if unknown:
        raise ValueError(f"biome {name}: map characters without legend entries: {sorted(unknown)}")
    return Biome(
        name=str(raw["name"]),
        style=str(raw["style"]),
        negative=str(raw.get("negative", "")),
        coherence_threshold=float(raw.get("coherence_threshold", 0.5)),
        legend=legend,
        tiles=tiles,
        props={str(k): str(v) for k, v in (raw.get("props") or {}).items()},
        lora=raw.get("lora"),
        example_map=example_map,
    )


__all__ = ["Biome", "TileClass", "biome_names", "load_biome"]
