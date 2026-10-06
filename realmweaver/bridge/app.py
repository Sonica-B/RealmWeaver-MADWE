"""FastAPI routes over the `World` seam: encoded assets (PNG textures and sprites, GLB meshes), chunk JSON, player
position, benchmark report, WebSocket ready events and the operator page. Every call into the world or the generator
runs in the threadpool under one lock (neither the world state graph nor the GPU adapter is thread-safe) so the event
loop stays free while a chunk generates. Assets are encoded once and served from a cache without that lock, and
prewarm runs on one background task that yields to `/chunk` and `/generate` between chunks, so the player's own
request never queues behind a prediction (story 24)."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from realmweaver.biomes import biome_names, load_biome
from realmweaver.config import settings
from realmweaver.metrics import histogram_embed, tileability
from realmweaver.metrics.bench import latest_report
from realmweaver.types import Asset, AssetSpec, Chunk, Generator, ImagePayload, Kind, Tier
from realmweaver.world import World

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
IMMUTABLE = "public, max-age=31536000, immutable"
MAX_GENERATED = 256  # ponytail: /generate results kept resident, oldest out; the upgrade path is a byte cap like the world's
MAX_ENCODED = 512  # ponytail: encoded assets kept by count, oldest out; the upgrade path is a byte cap shared with the world's
Event = dict[str, Any]
Key = tuple[int, int]
Position = tuple[float, float]


class GenerateRequest(BaseModel):  # AssetSpec fields; `subject` is the tile class, the prop or the mesh name
    biome: str
    kind: Kind = "texture"
    subject: str = ""
    size: int = Field(default=512, ge=8, le=2048, multiple_of=8)
    seed: int = 0
    steps: int = Field(default=4, ge=1, le=64)
    seamless: bool = True
    tier: Tier = "draft"


class PlayerPosition(BaseModel):
    x: float
    y: float


def _kv(mapping: dict[str, str]) -> list[dict[str, str]]:
    return [{"k": k, "v": v} for k, v in mapping.items()]


def chunk_json(chunk: Chunk) -> dict[str, Any]:
    """The binding Unity payload (tests/fixtures/chunk_example.json): every map has a flat twin."""
    layout, assets = chunk.layout, dict(chunk.asset_ids)
    prefabs = {cls: f"Prefab_{cls}" for cls in assets}
    return {
        "cx": chunk.cx, "cy": chunk.cy, "biome": chunk.biome, "size": layout.width, "state": chunk.state,
        "tiles": layout.class_rows(), "assets": assets, "prefabs": prefabs, "classes": list(layout.tileset.classes),
        "tilesFlat": layout.grid.ravel().tolist(), "assetList": _kv(assets), "prefabList": _kv(prefabs),
    }  # fmt: skip


def ready_event(chunk: Chunk) -> Event:
    assets = dict(chunk.asset_ids)
    return {"type": "ready", "chunk": [chunk.cx, chunk.cy], "assets": assets, "assetList": _kv(assets)}


def _adapters(
    generator: Generator | None, device: str
) -> tuple[Generator, Callable[[np.ndarray], np.ndarray]]:
    """The generator (by device when none is given) and its embedder: DINOv2 behind diffusion, else the histogram."""
    if generator is None and device.startswith("cuda"):
        from realmweaver.assets.diffusion import DiffusionGenerator

        generator = DiffusionGenerator()
    elif generator is None:
        from realmweaver.assets import ProceduralGenerator

        generator = ProceduralGenerator()
    if type(generator).__name__ != "DiffusionGenerator":
        return generator, histogram_embed
    from realmweaver.assets.embed import DinoEmbedder

    return generator, DinoEmbedder()


class Bridge:
    """World, generator and subscribers behind the routes. The world is built on first use so `/health` answers
    before a diffusion model loads. `lock` serialises every call into the world or the generator; `encoded` holds
    the bytes and media type of every asset announced through chunk JSON, a ready event or `/generate`, served
    without it."""

    def __init__(self, world: World | None, generator: Generator | None) -> None:
        self._world, self._generator = world, world.generator if world else generator
        # ponytail: one lock over the world state graph and the generator, so one generation runs at a time and the
        # scheduler's second in-flight slot stays idle; the upgrade path is a worker thread per slot.
        self.lock = threading.Lock()
        self.listeners: set[asyncio.Queue[Event]] = set()
        self.generated: OrderedDict[str, Asset] = OrderedDict()
        self.encoded: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        # Event-loop-only state: `priority` counts /chunk and /generate calls in flight, `positions` holds player
        # positions the prewarm task has not observed yet, `prewarmed` what it prewarmed since the last /player.
        self.priority = 0
        self.positions: list[Position] = []
        self.prewarmed: list[Key] = []
        self._prewarm: asyncio.Task[None] | None = None
        self.biome = world.biome if world else settings().biome
        self.device = str(getattr(self._generator, "device", "cpu")) if self._generator else settings().device

    @property
    def loaded(self) -> bool:
        return self._world is not None

    @property
    def generator_kind(self) -> str:
        if self._generator is None:
            return "diffusion" if self.device.startswith("cuda") else "procedural"
        return type(self._generator).__name__.removesuffix("Generator").lower() or "custom"

    @property
    def world(self) -> World:
        if self._world is None:
            self._generator, embed = _adapters(self._generator, self.device)
            self._world = World(self.biome, self._generator, embed=embed, chunk_size=settings().chunk_size)
            log.info("world ready: %s on %s, biome %s", self.generator_kind, self.device, self.biome)
        return self._world

    # -- locked world calls (threadpool) --------------------------------------------------------------------

    def chunk(self, cx: int, cy: int, tier: Tier) -> tuple[dict[str, Any], Event | None]:
        with self.lock:
            result = self.world.request_chunk(cx, cy, tier)
            chunk = result.chunk
            response = chunk_json(chunk), ready_event(chunk) if result.transition != "reused" else None
            assets = self._unencoded([chunk])
        for asset in assets:
            self._encode(asset)
        return response

    def prewarm(self, positions: list[Position]) -> tuple[list[Key], list[Event]]:
        """One prewarm step: observe the queued positions, then generate at most one predicted chunk."""
        with self.lock:
            for x, y in positions:
                self.world.observe_player(x, y)
            warmed = self.world.tick(1)
            chunks = [c for key in warmed if (c := self.world.chunk(*key)) is not None]
            events, assets = [ready_event(c) for c in chunks], self._unencoded(chunks)
        for asset in assets:
            self._encode(asset)
        return warmed, events

    def generate(self, spec: AssetSpec) -> dict[str, Any]:
        with self.lock:
            asset = self.generated.get(spec.id) or self.world.generator.generate(spec)
            self.generated[spec.id] = asset
            self.generated.move_to_end(spec.id)
            while len(self.generated) > MAX_GENERATED:
                self.generated.popitem(last=False)
        payload = asset.payload  # seams are a property of pixels; a mesh or a clip reports none
        seam = tileability(payload.image) if isinstance(payload, ImagePayload) else None
        return {"id": asset.id, "latency_s": asset.latency_s, "tileability": seam}

    def chunks(self) -> list[dict[str, Any]]:  # every chunk resident in the world
        with self.lock:
            resident = self.world.chunks() if self.loaded else []
        return [{"cx": c.cx, "cy": c.cy, "state": c.state, "biome": c.biome} for c in resident]

    def stats(self) -> dict[str, Any]:  # empty until the first generation builds the world
        with self.lock:
            return self.world.stats() if self.loaded else {}

    def _unencoded(self, chunks: list[Chunk]) -> list[Asset]:
        """Under the lock: the chunks' assets without a cached encoding; resident after `request_chunk`, so dict
        lookups."""
        return [self.world.asset(i) for c in chunks for i in c.asset_ids.values() if i not in self.encoded]

    # -- encoded assets, no world lock ----------------------------------------------------------------------

    def asset_bytes(self, asset_id: str, media_type: str) -> bytes:
        """The cached encoding, else one made now; the lock is taken only for an id this bridge never announced.
        404 for an unknown id or an asset encoded as another media type (a mesh asked for as a PNG)."""
        encoded = self.encoded.get(asset_id)
        if encoded is None:
            asset = self.generated.get(asset_id)
            if asset is None:
                with self.lock:  # e.g. an id from a saved world, regenerated from its recorded spec
                    try:
                        asset = self.world.asset(asset_id)
                    except KeyError:
                        raise HTTPException(404, f"unknown asset {asset_id}") from None
            encoded = self._encode(asset)
        data, actual = encoded
        if actual != media_type:
            raise HTTPException(404, f"asset {asset_id} is {actual}, not {media_type}")
        return data

    def _encode(self, asset: Asset) -> tuple[bytes, str]:
        """Encode once into `encoded` (assets never change) and serve from there, on whichever thread asks."""
        encoded = self.encoded.get(asset.id)
        if encoded is None:
            # ponytail: encoding on the calling thread delays a fresh chunk's response by its assets' encode time;
            # the upgrade path is encoding in the step that generated them.
            encoded = self.encoded[asset.id] = asset.encode()
            while len(self.encoded) > MAX_ENCODED:
                self.encoded.popitem(last=False)
        return encoded

    # -- event loop only ------------------------------------------------------------------------------------

    def player(self, x: float, y: float) -> list[Key]:
        """Queue the position for the prewarm task and return what it prewarmed since the previous call."""
        self.positions.append((x, y))
        if self._prewarm is None or self._prewarm.done():
            self._prewarm = asyncio.create_task(self._prewarm_loop())
        warmed, self.prewarmed = self.prewarmed, []
        return warmed

    async def _prewarm_loop(self) -> None:
        while self.positions:
            # A /chunk or /generate is waiting for the lock: the player's own request goes first (story 24).
            if self.priority:
                await asyncio.sleep(0.02)
                continue
            batch, self.positions = self.positions, []
            try:
                warmed, events = await run_in_threadpool(self.prewarm, batch)
            except Exception:  # e.g. a generator naming its missing requirement; /chunk reports it to callers
                log.exception("prewarm failed")
                return
            self.prewarmed += warmed
            for event in events:
                self.broadcast(event)

    async def first(self, call: Callable[..., Any], *args: Any) -> Any:
        """Run a locked world call in the threadpool with priority over prewarm."""
        self.priority += 1
        try:
            return await run_in_threadpool(call, *args)
        finally:
            self.priority -= 1

    def broadcast(self, event: Event) -> None:
        for queue in self.listeners:
            queue.put_nowait(event)


def create_app(world: World | None = None, generator: Generator | None = None) -> FastAPI:
    """The bridge app. Without `world`, one is built on first use around `generator` or the default adapter:
    `DiffusionGenerator` + `DinoEmbedder` on a CUDA device, else `ProceduralGenerator` + `histogram_embed`."""
    app = FastAPI(title="RealmWeaver bridge", docs_url=None, redoc_url=None)
    bridge = Bridge(world, generator)
    app.state.bridge = bridge

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok", "generator": bridge.generator_kind, "device": bridge.device, "loaded": bridge.loaded,
            "biome": bridge.biome, "chunk_size": bridge.world.chunk_size if bridge.loaded else settings().chunk_size,
        }  # fmt: skip

    @app.get("/biomes")
    async def biomes() -> list[dict[str, Any]]:
        return [
            {"name": b.name, "tiles": b.tile_classes(), "props": list(b.props),
             "palette": {cls: list(t.palette) for cls, t in b.tiles.items()}}
            for b in map(load_biome, biome_names())
        ]  # fmt: skip

    @app.post("/generate")
    async def generate(body: GenerateRequest) -> dict[str, Any]:
        try:
            load_biome(body.biome)
        except KeyError as e:
            raise HTTPException(404, str(e)) from None
        return await bridge.first(bridge.generate, AssetSpec(**body.model_dump()))

    async def asset(asset_id: str, media_type: str) -> Response:
        data = await run_in_threadpool(bridge.asset_bytes, asset_id, media_type)
        return Response(data, media_type=media_type, headers={"Cache-Control": IMMUTABLE})

    @app.get("/asset/{asset_id}.png")
    async def asset_png(asset_id: str) -> Response:
        return await asset(asset_id, "image/png")

    @app.get("/asset/{asset_id}.glb")
    async def asset_glb(asset_id: str) -> Response:
        return await asset(asset_id, "model/gltf-binary")

    @app.get("/chunk/{cx}/{cy}")
    async def chunk(cx: int, cy: int, tier: Tier = "draft") -> dict[str, Any]:
        payload, event = await bridge.first(bridge.chunk, cx, cy, tier)
        if event is not None:
            bridge.broadcast(event)
        return payload

    @app.get("/chunks")
    async def chunks() -> list[dict[str, Any]]:
        return await run_in_threadpool(bridge.chunks)

    @app.post("/player")
    async def player(body: PlayerPosition) -> dict[str, Any]:
        return {"prewarmed": [list(key) for key in bridge.player(body.x, body.y)]}

    @app.get("/report")
    async def report() -> dict[str, Any]:
        data = await run_in_threadpool(latest_report, settings().reports_dir)
        return {"available": False} if data is None else {**data, "available": True}

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return await run_in_threadpool(bridge.stats)

    @app.websocket("/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        queue: asyncio.Queue[Event] = asyncio.Queue()
        bridge.listeners.add(queue)
        pump = asyncio.create_task(_pump(ws, queue))
        try:
            await ws.send_json({"type": "hello"})
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass  # clients send nothing; receiving is how a disconnect shows up
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            pump.cancel()
            bridge.listeners.discard(queue)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request: Request, exc: RuntimeError) -> Response:
        # a generator naming its missing requirement reaches the caller as JSON instead of a bare 500
        log.error("%s %s failed: %s", request.method, request.url.path, exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return app


async def _pump(ws: WebSocket, queue: asyncio.Queue[Event]) -> None:
    try:
        while True:
            await ws.send_json(await queue.get())
    except Exception:  # the subscriber left mid-send; the receive loop is already cleaning up
        log.debug("events subscriber left mid-send", exc_info=True)
