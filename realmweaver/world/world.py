"""The `World` seam: chunks solved against their neighbours, one coherence-checked asset per tile class, a
predictor-driven prewarm queue and a byte-capped cache, all recorded in the world state graph.

The world owns what is generating: `submit` hands a chunk to the runner (one job per key, player requests ahead
of prewarm) and returns its handle, `request_chunk` is that handle's result, `tick` schedules prewarm through the
same runner and `on_ready` listeners hear of every chunk created or refined. With the inline runner every call
has finished before it returns; with a threaded one the generator runs on at most `runner.slots` worker threads.
The graph, the scheduler, the predictor and the caches are touched under one lock that is never held while the
generator or the embedder runs: what to generate is decided under it, generated with it let go of, recorded under
it again. `close` stops the runner.
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


def _satisfies(chunk: Chunk | None, tier: Tier) -> bool:
    """A resident draft chunk answers a draft request, a ready one any request; a pending one is still generating."""
    return chunk is not None and chunk.state != "pending" and (tier == "draft" or chunk.state == "ready")


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
    `region(name)` and `region_at(cx, cy)` read the region map and `stats()` is what the bench reads."""

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
        self.tileset = tileset_from_example(definition.example_map, definition.legend)  # compiled once
        # ponytail: one region per biome and one biome per World, so `region_at` answers the same region for every
        # key; several regions per world, from a biome-noise region map, land with E3 (terrain).
        self._region = biome
        self._graph = WorldStateGraph(seed=seed, chunk_size=chunk_size)
        self._graph.add_region(self._region, biome, self.tileset, self.threshold)
        self.predictor = Predictor(chunk_size)
        self.runner = runner or InlineRunner()
        self.scheduler = Scheduler(
            max_in_flight=self.runner.slots,
            cache_bytes=settings().cache_bytes if cache_bytes is None else cache_bytes,
        )
        self.asset_size = settings().asset_size
        # One lock over the graph, the scheduler, the predictor and the caches below. It is never held while the
        # generator or the embedder runs: `submit` and `tick` hand jobs to the runner with it let go of (the
        # inline runner runs them right there) and `_asset_for` generates with it let go of, so the runner's slots
        # overlap in generation and never in bookkeeping.
        self._lock = threading.RLock()
        self._assets: dict[str, Asset] = {}  # resident pixels by asset id
        self._jobs: dict[Key, Future[ChunkResult]] = {}  # the generation in flight per key
        self._listeners: list[Listener] = []
        self._warmed: list[Key] = []  # prewarmed or refined since the last tick
        self._player_chunk: Key | None = None
        self._unconstrained: set[tuple[str, str]] = (
            set()
        )  # region pairs whose borders go unconstrained, logged
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
        """The handle of the chunk at (cx, cy): the handle already in flight for the key when there is one, settled
        at once with `reused` when the chunk already satisfies `tier`, else a new job on the runner, player
        requests ahead of prewarm. Its result is the `ChunkResult`; the inline runner settles it before returning.
        Cancelling a handle still queued frees its key; after `close` a handle settles with the runner's error."""
        key = (cx, cy)
        with self._lock:
            running = self._jobs.get(key)
            if running is not None:
                return running
            chunk = self._graph.chunk(cx, cy)
            if _satisfies(chunk, tier):
                self.scheduler.touch(key)
                return _settled(ChunkResult(chunk, "reused"))
            handle: Future[ChunkResult] = Future()
            self._jobs[key] = handle  # claimed under the lock: one job per key, whoever asks next waits on it
            handle.add_done_callback(partial(self._forget, key))
        # with the lock let go of: the inline runner runs the job here and now, on this thread
        return self.runner.submit(priority, partial(self._run, key, tier), handle)

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
        """The region map's answer for `name`: its biome, tileset, coherence threshold and resident chunk keys;
        KeyError when unknown."""
        with self._lock:
            return self._graph.region(name)

    def region_at(self, cx: int, cy: int) -> Region:
        """The region map's answer for the chunk at (cx, cy), present or not: the region recorded for a resident
        chunk, else the world's own region (the single-biome map), which always carries its tileset and
        coherence threshold."""
        with self._lock:
            return self._graph.region(self._graph.region_of(cx, cy) or self._region)

    def asset(self, asset_id: str) -> Asset:
        """The asset with its pixels, regenerated from the recorded spec when not resident (after `load`)."""
        with self._lock:
            if asset_id in self._assets:
                return self._assets[asset_id]
            spec = self._graph.asset_spec(asset_id)
        asset = self._asset_for(spec)  # generated with the lock let go of
        with self._lock:
            return self._assets.setdefault(asset_id, asset)

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
        while scheduled < budget and (key := self._next_prewarm()) is not None:
            self._prewarm(key, "draft")  # with the lock let go of: the inline runner generates here and now
            scheduled += 1
        with self._lock:  # idle: nothing left to prewarm and nothing generating
            drafts = [] if self._jobs else [k for k, c in self._graph.chunks.items() if c.state == "draft"]
            drafts.sort(key=lambda k: max(abs(k[0] - current[0]), abs(k[1] - current[1])))
        for key in drafts[: budget - scheduled]:
            self._prewarm(key, "refine")
        with self._lock:
            warmed, self._warmed = self._warmed, []
            return warmed

    def close(self) -> None:
        """Stop the runner: jobs still queued are cancelled (their handles say so), running ones finish first, and
        `submit` after this settles its handle with the runner's RuntimeError."""
        self.runner.close()

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

    def _lock_held(self) -> bool:
        """Test hook: True when the calling thread holds the world lock, which a generator must never see."""
        return self._lock._is_owned()

    def _forget(self, key: Key, handle: Future[ChunkResult]) -> None:
        """Done callback of every job handle: the key is free again once its job settled, however it settled."""
        with self._lock:
            if self._jobs.get(key) is handle:
                del self._jobs[key]

    def _next_prewarm(self) -> Key | None:
        """The scheduler's best pending key that no job covers already (one the player asked for since it was
        queued is dropped: that job caches it); None when nothing is pending or the in-flight cap is reached."""
        with self._lock:
            while (key := self.scheduler.next()) is not None:
                if key not in self._jobs:
                    return key
                self.scheduler.abort(key)
            return None

    def _prewarm(self, key: Key, tier: Tier) -> None:
        """Hand a prewarm (or an idle refine) to the runner; its key joins `_warmed` once it is done."""

        def finished(handle: Future[ChunkResult]) -> None:
            if handle.cancelled():  # the runner closed with it still queued: its in-flight slot is free again
                with self._lock:
                    self.scheduler.abort(key)
                return
            if handle.exception() is not None:  # /chunk reports the generator's error; prewarm just notes it
                log.error("%s of chunk %s failed: %s", tier, key, handle.exception())
                return
            with self._lock:
                if handle.result().transition == "reused":  # a player's request made it meanwhile
                    self.scheduler.abort(key)
                else:
                    self._warmed.append(key)
                    self._counts["refined"] += tier == "refine"

        self.submit(*key, tier, priority="prewarm").add_done_callback(finished)

    def _run(self, key: Key, tier: Tier) -> ChunkResult:
        """One runner job: generate, then tell the listeners before the handle settles and `_forget` frees the key."""
        result = self._generate(key, tier)
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
            if _satisfies(chunk, tier):
                self.scheduler.touch(key)
                return ChunkResult(chunk, "reused")
            start, created, region = time.perf_counter(), chunk is None, self.region_at(cx, cy)
            if chunk is None:
                chunk = Chunk(cx, cy, region.biome, self._solve(cx, cy, region))
                self._graph.add_chunk(chunk, region.name)
        try:
            self._assign_assets(chunk, tier, region)  # generates with the lock let go of, records under it
        except Exception:
            with self._lock:
                self.scheduler.abort(key)
                if created:
                    self._drop(key)
            raise
        with self._lock:
            self._graph.set_state(key, "ready" if tier == "refine" else "draft")
            self.scheduler.done(key, self._graph.chunk_bytes(key))
            self._chunk_latencies.append(time.perf_counter() - start)
            self._evict(self._player_chunk or key)
            return ChunkResult(chunk, "created" if created else "refined")

    def _solve(self, cx: int, cy: int, region: Region) -> Layout:
        """The chunk's layout, its borders constrained by the neighbours that solve with the region's tileset; a
        neighbour of another tileset is no constraint, which the first such pair of regions logs."""
        seed = _stable_seed(self.seed, cx, cy, "layout")
        layouts: dict[str, Layout] = {}
        for side, other in self._graph.neighbours(cx, cy).items():
            theirs = self.region_at(other.cx, other.cy)
            if region.shares_tileset(theirs):
                layouts[side] = other.layout
            elif (region.name, theirs.name) not in self._unconstrained:
                self._unconstrained.add((region.name, theirs.name))
                log.warning(
                    "regions %s and %s solve with different tilesets: their borders go unconstrained",
                    region.name, theirs.name,
                )  # fmt: skip
        try:
            return solve_chunk(region.tileset, self.chunk_size, seed, layouts)
        except Contradiction:
            # ponytail: a chunk that cannot meet every neighbour's border is solved unconstrained and the seam
            # shows; the upgrade path is re-solving the neighbours' border strips together with it.
            log.warning("chunk (%d, %d): borders contradict, solving without neighbours", cx, cy)
            return solve_chunk(region.tileset, self.chunk_size, seed, {})

    def _asset_for(self, spec: AssetSpec) -> Asset:
        """The resident asset for `spec`, else one generated and embedded now with the lock let go of, so another
        slot's bookkeeping (and every read) proceeds meanwhile; the caller records it under the lock."""
        with self._lock:
            asset = self._assets.get(spec.id)
        if asset is None:
            asset = self.generator.generate(spec)
            asset.style_vec = self.embed(asset.preview(asset.spec.size))
        return asset

    def _assign_assets(self, chunk: Chunk, tier: Tier, region: Region) -> None:
        """One asset per tile class present, most frequent class first so it anchors the region's style. A candidate
        below the region's coherence threshold is regenerated with a new seed, at most twice; when every candidate
        fails the class is anchored to the region's best asset of that class, else the best candidate becomes the
        anchor. Each candidate is generated with the lock let go of and scored and recorded under it."""
        tileset, threshold = region.tileset, region.coherence_threshold
        counts = np.bincount(chunk.layout.grid.ravel(), minlength=len(tileset))
        for index in np.argsort(-counts, kind="stable"):
            if counts[index] == 0:
                continue
            cls, best = tileset.classes[int(index)], None
            for attempt in range(MAX_ATTEMPTS):
                seed = _stable_seed(self.seed, chunk.cx, chunk.cy, cls) + attempt
                spec = AssetSpec(
                    region.biome, subject=cls, size=self.asset_size, seed=seed, steps=_STEPS[tier], tier=tier
                )
                asset = self._asset_for(spec)
                with self._lock:
                    self._counts["regenerations"] += attempt > 0
                    score = self._graph.candidate_coherence(asset.style_vec, chunk.key, cls)
                if best is None or score > best[1]:
                    best = (asset, score)
                if score >= threshold:
                    break
            asset, score = best
            make_anchor = needs_fallback = score < threshold
            if needs_fallback:
                with self._lock:
                    self._counts["fallbacks"] += 1
                    log.info("%s %s: coherence %.2f < %.2f, anchoring", chunk.key, cls, score, threshold)
                    existing = self._graph.best_asset(region.name, cls)
                if existing is not None:  # the region's best asset of the class takes over; no new anchor
                    asset, score, make_anchor = self.asset(existing), None, False
            with self._lock:
                self._assets[asset.id] = asset
                for orphan in self._graph.add_asset(
                    asset, chunk.key, cls, coherence=score, anchor=make_anchor
                ):
                    self._assets.pop(orphan, None)

    def _evict(self, current: Key) -> None:
        for key in self.scheduler.evict_if_needed(current):
            self._drop(key)

    def _drop(self, key: Key) -> None:
        if self._graph.chunk(*key) is not None:
            for asset_id in self._graph.remove_chunk(key):
                self._assets.pop(asset_id, None)
