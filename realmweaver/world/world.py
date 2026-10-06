"""The `World` seam: chunks solved against their neighbours, one coherence-checked asset per tile class, a
predictor-driven prewarm queue and a byte-capped cache, all recorded in the world state graph.

Generation is synchronous: `request_chunk` returns once the chunk's layout and assets exist, together with the
transition it caused, and `tick` generates at most `budget` chunks before returning: predicted ones as drafts
first and, once nothing is left to prewarm, draft chunks refined nearest the player first.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Literal

import numpy as np

from realmweaver.biomes import load_biome
from realmweaver.config import settings
from realmweaver.layout import Contradiction, solve_chunk, tileset_from_example
from realmweaver.metrics import histogram_embed
from realmweaver.types import Asset, AssetSpec, Chunk, Generator, Layout, Tier
from realmweaver.world.graph import Region, WorldStateGraph
from realmweaver.world.predictor import Predictor
from realmweaver.world.scheduler import Scheduler

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3  # one generation plus at most two regenerations before the anchor fallback
_STEPS = {"draft": 4, "refine": 8}  # the anytime knob per quality tier

Key = tuple[int, int]
Embed = Callable[[np.ndarray], np.ndarray]
Transition = Literal["created", "reused", "refined"]


@dataclass(frozen=True)
class ChunkResult:
    """`request_chunk`'s answer: the chunk and what the request did to it, so no caller diffs graph state."""

    chunk: Chunk
    transition: Transition


def _stable_seed(*parts: object) -> int:
    """32-bit seed from the parts' text; `hash()` is salted per process and would not survive save/load."""
    return int.from_bytes(sha1(":".join(map(str, parts)).encode()).digest()[:4], "big")


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    p50, p95 = np.percentile(values, [50, 95])
    return {"n": len(values), "mean": float(np.mean(values)), "p50": float(p50), "p95": float(p95)}


class World:
    """One biome, one region, chunks on demand. `graph` is the source of truth and is written only here;
    `region(name)` reads the region map and `stats()` is what the bench reads."""

    def __init__(
        self,
        biome: str,
        generator: Generator,
        embed: Embed = histogram_embed,
        chunk_size: int = 16,
        seed: int = 0,
        cache_bytes: int | None = None,
    ) -> None:
        self.biome, self.generator, self.embed = biome, generator, embed
        self.chunk_size, self.seed = chunk_size, seed
        definition = load_biome(biome)
        self.threshold = definition.coherence_threshold
        self.tileset = tileset_from_example(definition.example_map, definition.legend)
        self._region = biome  # ponytail: one region per world; several biomes per world is the upgrade path
        self._graph = WorldStateGraph(seed=seed, chunk_size=chunk_size)
        self._graph.add_region(self._region, biome)
        self.predictor = Predictor(chunk_size)
        self.scheduler = Scheduler(cache_bytes=settings().cache_bytes if cache_bytes is None else cache_bytes)
        self.asset_size = settings().asset_size
        self._assets: dict[str, Asset] = {}  # resident pixels by asset id
        self._player_chunk: Key | None = None
        self._counts = dict.fromkeys(
            ("prewarm_hits", "prewarm_misses", "regenerations", "fallbacks", "refined"), 0
        )
        self._chunk_latencies: list[float] = []

    @property
    def graph(self) -> WorldStateGraph:
        """The world state graph, read by tests and the bench; written only through `World`."""
        return self._graph

    def request_chunk(self, cx: int, cy: int, tier: Tier = "draft") -> ChunkResult:
        """The chunk at (cx, cy) with its transition: `created` when it was absent, `refined` when
        `tier="refine"` upgraded a draft's assets in place, `reused` when it was returned as it was."""
        key, chunk = (cx, cy), self._graph.chunk(cx, cy)
        if chunk is not None and (tier == "draft" or chunk.state == "ready"):
            self.scheduler.touch(key)
            return ChunkResult(chunk, "reused")
        start, created = time.perf_counter(), chunk is None
        if chunk is None:
            chunk = Chunk(cx, cy, self.biome, self._solve(cx, cy))
            self._graph.add_chunk(chunk, self._region)
        try:
            self._assign_assets(chunk, tier)
        except Exception:
            self.scheduler.abort(key)
            if created:
                self._drop(key)
            raise
        self._graph.set_state(key, "ready" if tier == "refine" else "draft")
        self.scheduler.done(key, self._graph.chunk_bytes(key))
        self._chunk_latencies.append(time.perf_counter() - start)
        self._evict(self._player_chunk or key)
        return ChunkResult(chunk, "created" if created else "refined")

    def chunk(self, cx: int, cy: int) -> Chunk | None:
        """The resident chunk at (cx, cy), None when absent; `request_chunk` generates it."""
        return self._graph.chunk(cx, cy)

    def chunks(self) -> list[Chunk]:
        """Every resident chunk, whatever its region."""
        return list(self._graph.chunks.values())

    def region(self, name: str) -> Region:
        """The region map's answer for `name`: its biome and resident chunk keys; KeyError when unknown."""
        return self._graph.region(name)

    def asset(self, asset_id: str) -> Asset:
        """The asset with its pixels, regenerated from the recorded spec when not resident (after `load`)."""
        if asset_id not in self._assets:
            self._assets[asset_id] = self._asset_for(self._graph.asset_spec(asset_id))
        return self._assets[asset_id]

    def observe_player(self, x: float, y: float) -> None:
        """Player position in tile units; entering a chunk counts as a prewarm hit when it is already present."""
        self.predictor.observe((x, y))
        key = (int(x // self.chunk_size), int(y // self.chunk_size))
        if key != self._player_chunk:
            self._counts["prewarm_hits" if self._graph.chunk(*key) else "prewarm_misses"] += 1
            self._player_chunk = key
        self.scheduler.touch(key)

    def tick(self, budget: int = 1) -> list[Key]:
        """Prewarm: queue the player's absent neighbours at their visit probability and generate up to `budget` of
        them; budget left once the queue is empty refines that many draft chunks, nearest the player first.
        Returns the keys generated or refined."""
        if self._player_chunk is None:
            return []
        current = self._player_chunk
        self._evict(current)
        cost = float(np.mean(self._chunk_latencies)) if self._chunk_latencies else 0.0  # measured chunk cost
        for key, p in self.predictor.rank(current, k=8):
            if self._graph.chunk(*key) is None:
                self.scheduler.submit(key, p, cost_s=cost or None)
        done: list[Key] = []
        # ponytail: generation runs right here, so `max_in_flight` only ever sees one job; the upgrade path is a
        # worker thread per in-flight slot that hands finished chunks back through `scheduler.done`.
        while len(done) < budget and (key := self.scheduler.next()) is not None:
            self.request_chunk(*key)
            done.append(key)
        drafts = [key for key, chunk in self._graph.chunks.items() if chunk.state == "draft"]
        drafts.sort(key=lambda k: max(abs(k[0] - current[0]), abs(k[1] - current[1])))
        for key in drafts[: budget - len(done)]:  # idle: nothing left to prewarm
            self.request_chunk(*key, tier="refine")
            self._counts["refined"] += 1
            done.append(key)
        return done

    def save(self, path: str | Path) -> None:
        meta = {"biome": self.biome, "chunk_size": self.chunk_size, "seed": self.seed}
        Path(path).write_text(json.dumps({**meta, "graph": self._graph.to_json()}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, generator: Generator, embed: Embed = histogram_embed) -> World:
        """Rebuild a saved world; asset pixels are regenerated from their specs the first time `asset` asks."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        world = cls(data["biome"], generator, embed, data["chunk_size"], data["seed"])
        world._graph = WorldStateGraph.from_json(data["graph"])
        for key in world._graph.chunks:  # ponytail: counted as resident before their pixels are regenerated
            world.scheduler.done(key, world._graph.chunk_bytes(key))
        return world

    def stats(self) -> dict:
        return {
            "chunks": len(self._graph.chunks),
            "assets": self._graph.count("Asset"),
            "resident_assets": len(self._assets),
            "cache_bytes": self.scheduler.bytes_used,
            "cache_cap_bytes": self.scheduler.cache_bytes,
            "in_flight": self.scheduler.in_flight,
            "pending": self.scheduler.pending,
            "seam_violations": self._graph.seam_violations(),
            **self._counts,
            "chunk_latency_s": _summary(self._chunk_latencies),
            "asset_latency_s": _summary([a.latency_s for a in self._assets.values()]),
        }

    # -- private ------------------------------------------------------------------------------------------

    def _solve(self, cx: int, cy: int) -> Layout:
        seed = _stable_seed(self.seed, cx, cy, "layout")
        layouts = {side: c.layout for side, c in self._graph.neighbours(cx, cy).items()}
        try:
            return solve_chunk(self.tileset, self.chunk_size, seed, layouts)
        except Contradiction:
            # ponytail: a chunk that cannot meet every neighbour's border is solved unconstrained and the seam
            # shows; the upgrade path is re-solving the neighbours' border strips together with it.
            log.warning("chunk (%d, %d): borders contradict, solving without neighbours", cx, cy)
            return solve_chunk(self.tileset, self.chunk_size, seed, {})

    def _asset_for(self, spec: AssetSpec) -> Asset:
        asset = self._assets.get(spec.id)
        if asset is None:
            asset = self.generator.generate(spec)
            asset.style_vec = self.embed(asset.preview(asset.spec.size))
        return asset

    def _assign_assets(self, chunk: Chunk, tier: Tier) -> None:
        """One asset per tile class present, most frequent class first so it anchors the region's style. A candidate
        below the coherence threshold is regenerated with a new seed, at most twice; when every candidate fails the
        class is anchored to the region's best asset of that class, else the best candidate becomes the anchor."""
        counts = np.bincount(chunk.layout.grid.ravel(), minlength=len(self.tileset))
        for index in np.argsort(-counts, kind="stable"):
            if counts[index] == 0:
                continue
            cls, best = self.tileset.classes[int(index)], None
            for attempt in range(MAX_ATTEMPTS):
                self._counts["regenerations"] += attempt > 0
                seed = _stable_seed(self.seed, chunk.cx, chunk.cy, cls) + attempt
                spec = AssetSpec(
                    self.biome, subject=cls, size=self.asset_size, seed=seed, steps=_STEPS[tier], tier=tier
                )
                asset = self._asset_for(spec)
                score = self._graph.candidate_coherence(asset.style_vec, chunk.key, cls)
                if best is None or score > best[1]:
                    best = (asset, score)
                if score >= self.threshold:
                    break
            asset, score = best
            make_anchor = needs_fallback = score < self.threshold
            if needs_fallback:
                self._counts["fallbacks"] += 1
                log.info("%s %s: coherence %.2f < %.2f, anchoring", chunk.key, cls, score, self.threshold)
                existing = self._graph.best_asset(self._region, cls)
                if existing is not None:  # the region's best asset of the class takes over; no new anchor
                    asset, score, make_anchor = self.asset(existing), None, False
            self._assets[asset.id] = asset
            for orphan in self._graph.add_asset(asset, chunk.key, cls, coherence=score, anchor=make_anchor):
                self._assets.pop(orphan, None)

    def _evict(self, current: Key) -> None:
        for key in self.scheduler.evict_if_needed(current):
            self._drop(key)

    def _drop(self, key: Key) -> None:
        if self._graph.chunk(*key) is not None:
            for asset_id in self._graph.remove_chunk(key):
                self._assets.pop(asset_id, None)
