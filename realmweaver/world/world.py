"""The `World` seam: chunks solved against their neighbours, one coherence-checked asset per tile class, a
predictor-driven prewarm queue and a byte-capped cache, all recorded in the world state graph.

The world owns what is generating: `submit` hands a chunk to the runner (one job per key, player requests ahead
of prewarm) and returns its handle, `request_chunk` is that handle's result, `tick` schedules prewarm through the
same runner and `on_ready` listeners hear of every chunk created or refined. With the inline runner every call
has finished before it returns; with a threaded one the generator runs on at most `runner.slots` worker threads
while the graph, scheduler and caches are touched under one lock that generation itself never holds.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
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
from realmweaver.world.runner import InlineRunner, Priority, Runner
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


Listener = Callable[[ChunkResult], None]


def _settled(result: ChunkResult) -> Future[ChunkResult]:
    handle: Future[ChunkResult] = Future()
    handle.set_result(result)
    return handle


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
        runner: Runner | None = None,
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
        self.runner = runner or InlineRunner()
        self.scheduler = Scheduler(
            max_in_flight=self.runner.slots,
            cache_bytes=settings().cache_bytes if cache_bytes is None else cache_bytes,
        )
        self.asset_size = settings().asset_size
        # One lock over the graph, the scheduler, the predictor and the caches below; `_asset_for` lets go of it
        # while the generator and the embedder run, so the runner's slots overlap there and never elsewhere.
        self._lock = threading.RLock()
        self._assets: dict[str, Asset] = {}  # resident pixels by asset id
        self._jobs: dict[Key, Future[ChunkResult]] = {}  # the generation in flight per key
        self._listeners: list[Listener] = []
        self._warmed: list[Key] = []  # prewarmed or refined since the last tick
        self._player_chunk: Key | None = None
        self._counts = dict.fromkeys(
            ("prewarm_hits", "prewarm_misses", "regenerations", "fallbacks", "refined"), 0
        )
        self._chunk_latencies: list[float] = []

    @property
    def graph(self) -> WorldStateGraph:
        """The world state graph, read by tests and the bench; written only through `World`."""
        return self._graph

    def submit(
        self, cx: int, cy: int, tier: Tier = "draft", priority: Priority = "player"
    ) -> Future[ChunkResult]:
        """The handle of the chunk at (cx, cy): settled at once with `reused` when the chunk already satisfies
        `tier`, the handle already in flight for the key when there is one, else a new job on the runner, player
        requests ahead of prewarm. Its result is the `ChunkResult`; the inline runner settles it before returning."""
        key = (cx, cy)
        with self._lock:
            chunk = self._graph.chunk(cx, cy)
            if chunk is not None and (tier == "draft" or chunk.state == "ready"):
                self.scheduler.touch(key)
                return _settled(ChunkResult(chunk, "reused"))
            running = self._jobs.get(key)
            if running is not None:
                # ponytail: a refine asked while a draft is in flight gets the draft; the next tick refines it.
                return running
            handle = self.runner.submit(priority, partial(self._run, key, tier))
            if not handle.done():  # the inline runner has run it already; a worker needs this lock first
                self._jobs[key] = handle
            return handle

    def request_chunk(self, cx: int, cy: int, tier: Tier = "draft") -> ChunkResult:
        """The chunk at (cx, cy) with its transition: `created` when it was absent, `refined` when
        `tier="refine"` upgraded a draft's assets in place, `reused` when it was returned as it was. Waits for
        the runner, so never call it from a thread the runner's own job would need."""
        return self.submit(cx, cy, tier).result()

    def on_ready(self, listener: Listener) -> None:
        """Call `listener` with every chunk created or refined, on the thread that generated it and before its
        handle settles; a listener that raises is logged and the chunk still counts as done."""
        self._listeners.append(listener)

    def chunk(self, cx: int, cy: int) -> Chunk | None:
        """The resident chunk at (cx, cy), None when absent; `request_chunk` generates it."""
        with self._lock:
            return self._graph.chunk(cx, cy)

    def chunks(self) -> list[Chunk]:
        """Every resident chunk, whatever its region."""
        with self._lock:
            return list(self._graph.chunks.values())

    def region(self, name: str) -> Region:
        """The region map's answer for `name`: its biome and resident chunk keys; KeyError when unknown."""
        with self._lock:
            return self._graph.region(name)

    def asset(self, asset_id: str) -> Asset:
        """The asset with its pixels, regenerated from the recorded spec when not resident (after `load`)."""
        with self._lock:
            if asset_id not in self._assets:
                self._assets[asset_id] = self._asset_for(self._graph.asset_spec(asset_id))
            return self._assets[asset_id]

    def observe_player(self, x: float, y: float) -> None:
        """Player position in tile units; entering a chunk counts as a prewarm hit when it is already present."""
        with self._lock:
            self.predictor.observe((x, y))
            key = (int(x // self.chunk_size), int(y // self.chunk_size))
            if key != self._player_chunk:
                self._counts["prewarm_hits" if self._graph.chunk(*key) else "prewarm_misses"] += 1
                self._player_chunk = key
            self.scheduler.touch(key)

    def tick(self, budget: int = 1) -> list[Key]:
        """Prewarm: queue the player's absent neighbours at their visit probability and hand up to `budget` of
        them to the runner, at most `scheduler.max_in_flight` (the runner's slots) at a time; budget left while
        nothing is in flight refines that many draft chunks, nearest the player first. Returns the keys whose
        prewarm or refine finished since the previous call: with the inline runner, the ones this call generated."""
        with self._lock:
            if self._player_chunk is None:
                return []
            current = self._player_chunk
            self._evict(current)
            # the measured chunk cost prices each prediction; the scheduler's prior stands in until there is one
            cost = float(np.mean(self._chunk_latencies)) if self._chunk_latencies else 0.0
            for key, p in self.predictor.rank(current, k=8):
                if self._graph.chunk(*key) is None and key not in self._jobs:
                    self.scheduler.submit(key, p, cost_s=cost or None)
            scheduled = 0
            while scheduled < budget and (key := self.scheduler.next()) is not None:
                if key in self._jobs:  # asked for by the player since it was queued: that job caches it
                    self.scheduler.abort(key)
                    continue
                self._prewarm(key, "draft")
                scheduled += 1
            if not self._jobs:  # idle: nothing left to prewarm and nothing generating
                drafts = [key for key, chunk in self._graph.chunks.items() if chunk.state == "draft"]
                drafts.sort(key=lambda k: max(abs(k[0] - current[0]), abs(k[1] - current[1])))
                for key in drafts[: budget - scheduled]:
                    self._prewarm(key, "refine")
            warmed, self._warmed = self._warmed, []
            return warmed

    def save(self, path: str | Path) -> None:
        meta = {"biome": self.biome, "chunk_size": self.chunk_size, "seed": self.seed}
        with self._lock:
            data = json.dumps({**meta, "graph": self._graph.to_json()})
        Path(path).write_text(data, encoding="utf-8")

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
        with self._lock:
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

    def _prewarm(self, key: Key, tier: Tier) -> None:
        """Hand a prewarm (or an idle refine) to the runner; its key joins `_warmed` once it is done."""

        def finished(handle: Future[ChunkResult]) -> None:
            if handle.exception() is not None:  # /chunk reports the generator's error; prewarm just notes it
                log.error("%s of chunk %s failed: %s", tier, key, handle.exception())
                return
            with self._lock:
                self._warmed.append(key)
                self._counts["refined"] += tier == "refine"

        self.submit(*key, tier, priority="prewarm").add_done_callback(finished)

    def _run(self, key: Key, tier: Tier) -> ChunkResult:
        """One runner job: generate, leave `_jobs`, then tell the listeners before the handle settles."""
        try:
            result = self._generate(key, tier)
        finally:
            with self._lock:
                self._jobs.pop(key, None)
        if result.transition != "reused":
            for listener in list(self._listeners):
                try:
                    listener(result)
                except Exception:
                    log.exception("ready listener failed on chunk %s", key)
        return result

    def _generate(self, key: Key, tier: Tier) -> ChunkResult:
        cx, cy = key
        with self._lock:
            chunk = self._graph.chunk(cx, cy)
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
        """Under the lock: the resident asset for `spec`, else one generated and embedded with the lock let go
        of meanwhile, so another slot's bookkeeping (and every read) proceeds while this one generates."""
        asset = self._assets.get(spec.id)
        if asset is None:
            self._lock.release()
            try:
                asset = self.generator.generate(spec)
                asset.style_vec = self.embed(asset.preview(asset.spec.size))
            finally:
                self._lock.acquire()
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
