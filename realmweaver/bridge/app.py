"""FastAPI routes over the `World` seam: encoded assets (PNG textures and sprites, GLB meshes), chunk JSON, player
position, benchmark report, WebSocket ready events and the operator page. The world owns what generates when: `/chunk`
submits at player priority and waits for the handle in the threadpool, `/player` schedules prewarm and returns at
once, and `/generate` takes a runner slot like a chunk does, so the generator never runs on two threads it cannot
share. Every chunk the world finishes reaches the bridge through `World.on_ready`, which encodes its assets once
into a cache (one lock over it, since worker and threadpool threads both fill it) and fans the ready event out to
the WebSocket subscribers (story 24). The app's lifespan closes the world's runner on shutdown."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from realmweaver import wire
from realmweaver.biomes import biome_names, load_biome
from realmweaver.config import settings
from realmweaver.metrics import histogram_embed, tileability
from realmweaver.metrics.bench import latest_report
from realmweaver.types import Asset, AssetSpec, Generator, ImagePayload, Kind, Tier
from realmweaver.world import ChunkResult, ThreadRunner, World

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
IMMUTABLE = "public, max-age=31536000, immutable"
MAX_GENERATED = 256  # ponytail: /generate results kept resident, oldest out; the upgrade path is a byte cap like the world's
MAX_ENCODED = 512  # ponytail: encoded assets kept by count, oldest out; the upgrade path is a byte cap shared with the world's
Event = dict[str, Any]
Key = tuple[int, int]
Subscriber = tuple[
    asyncio.AbstractEventLoop, asyncio.Queue[Event]
]  # a WebSocket subscriber's queue and its loop


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
    before a diffusion model loads. `encoded` holds the bytes and media type of every asset announced through chunk
    JSON, a ready event or `/generate`, filled by the thread that finished the chunk and served from any; it and
    `generated` are touched under `_cache_lock` only."""

    def __init__(self, world: World | None, generator: Generator | None) -> None:
        self._world, self._generator = world, world.generator if world else generator
        self.subscribers: set[Subscriber] = set()
        self.generated: OrderedDict[str, Asset] = OrderedDict()
        self.encoded: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self._cache_lock = threading.Lock()  # `generated` and `encoded`: worker threads and the threadpool
        self._building: asyncio.Future[None] | None = None  # event-loop state: the one lazy build under way
        self.biome = world.biome if world else settings().biome
        self.device = str(getattr(self._generator, "device", "cpu")) if self._generator else settings().device
        if world is not None:
            world.on_ready(self._ready)

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
            raise RuntimeError("the world is not built yet; the route awaits `load()` first")
        return self._world

    async def load(self) -> World:
        """The world, built by one threadpool call however many requests arrive while it loads."""
        if self._world is None:
            if self._building is None:
                self._building = asyncio.ensure_future(run_in_threadpool(self._build))
            try:
                await asyncio.shield(self._building)
            finally:
                self._building = None
        return self.world

    def _build(self) -> None:
        self._generator, embed = _adapters(self._generator, self.device)
        slots = 2 if self.generator_kind == "procedural" else 1  # one diffusion pipeline, one thread
        world = World(
            self.biome,
            self._generator,
            embed=embed,
            chunk_size=settings().chunk_size,
            runner=ThreadRunner(slots),
        )
        world.on_ready(self._ready)
        self._world = world
        log.info(
            "world ready: %s on %s, biome %s, %d slot(s)", self.generator_kind, self.device, self.biome, slots
        )

    # -- world calls (threadpool) ---------------------------------------------------------------------------

    def chunk(self, cx: int, cy: int, tier: Tier) -> dict[str, Any]:
        """The chunk JSON (`wire.chunk_payload`) once the world's player-priority job is done (at once when the
        chunk is resident); its ready event, if any, went out through `_ready` before the handle settled."""
        return wire.chunk_payload(self.world.submit(cx, cy, tier, priority="player").result().chunk)

    def player(self, x: float, y: float) -> list[Key]:
        """Observe the position, schedule one prewarm, return what the world prewarmed since the previous call."""
        self.world.observe_player(x, y)
        return self.world.tick(1)

    def generate(self, spec: AssetSpec) -> dict[str, Any]:
        with self._cache_lock:
            asset = self.generated.get(spec.id)
        if asset is None:  # on a runner slot: the generator is never called beside a chunk's generation
            asset = self.world.runner.submit("player", lambda: self.world.generator.generate(spec)).result()
        with self._cache_lock:
            self.generated[spec.id] = asset
            self.generated.move_to_end(spec.id)
            while len(self.generated) > MAX_GENERATED:
                self.generated.popitem(last=False)
        payload = asset.payload  # seams are a property of pixels; a mesh or a clip reports none
        seam = tileability(payload.image) if isinstance(payload, ImagePayload) else None
        return {"id": asset.id, "latency_s": asset.latency_s, "tileability": seam}

    def chunks(self) -> list[dict[str, Any]]:  # every chunk resident in the world
        resident = self.world.chunks() if self.loaded else []
        return [{"cx": c.cx, "cy": c.cy, "state": c.state, "biome": c.biome} for c in resident]

    def stats(self) -> dict[str, Any]:  # empty until the first generation builds the world
        return self.world.stats() if self.loaded else {}

    def close(self) -> None:
        """App shutdown: stop the world's runner (queued jobs cancelled, running ones finished first)."""
        if self._world is not None:
            self._world.close()

    # -- encoded assets -------------------------------------------------------------------------------------

    def asset_bytes(self, asset_id: str, media_type: str) -> bytes:
        """The cached encoding, else one made now (an id from a saved world is regenerated from its recorded spec).
        404 for an unknown id or an asset encoded as another media type (a mesh asked for as a PNG)."""
        with self._cache_lock:
            encoded = self.encoded.get(asset_id)
            asset = None if encoded is not None else self.generated.get(asset_id)
        if encoded is None:
            if asset is None:
                try:
                    asset = self.world.asset(asset_id) if self.loaded else None
                except KeyError:
                    asset = None
            if asset is None:
                raise HTTPException(404, f"unknown asset {asset_id}")
            encoded = self._encode(asset)
        data, actual = encoded
        if actual != media_type:
            raise HTTPException(404, f"asset {asset_id} is {actual}, not {media_type}")
        return data

    def _encode(self, asset: Asset) -> tuple[bytes, str]:
        """Encode once into `encoded` (assets never change) and serve from there, on whichever thread asks; the
        encoding itself runs with the lock let go of, and the first of two racing encoders is the one kept."""
        with self._cache_lock:
            encoded = self.encoded.get(asset.id)
        if encoded is None:
            fresh = asset.encode()
            with self._cache_lock:
                encoded = self.encoded.setdefault(asset.id, fresh)
                while len(self.encoded) > MAX_ENCODED:
                    self.encoded.popitem(last=False)
        return encoded

    # -- the world's ready listener and the WebSocket fan-out, from any thread -------------------------------

    def _ready(self, result: ChunkResult) -> None:
        """`World.on_ready`: on the thread that generated the chunk, encode its new assets, then announce it."""
        chunk, assets = result.chunk, dict(result.chunk.asset_ids)
        for asset_id in assets.values():
            self._encode(self.world.asset(asset_id))
        self.broadcast(wire.ready_event(chunk, assets))

    def broadcast(self, event: Event) -> None:
        for loop, queue in list(self.subscribers):
            loop.call_soon_threadsafe(queue.put_nowait, event)


def create_app(world: World | None = None, generator: Generator | None = None) -> FastAPI:
    """The bridge app. Without `world`, one is built on first use around `generator` or the default adapter:
    `DiffusionGenerator` + `DinoEmbedder` on a CUDA device, else `ProceduralGenerator` + `histogram_embed`.
    Its lifespan closes the world's runner on shutdown."""
    bridge = Bridge(world, generator)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await run_in_threadpool(bridge.close)  # joins the worker threads: off the event loop

    app = FastAPI(title="RealmWeaver bridge", docs_url=None, redoc_url=None, lifespan=lifespan)
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
        await bridge.load()
        return await run_in_threadpool(bridge.generate, AssetSpec(**body.model_dump()))

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
        await bridge.load()
        return await run_in_threadpool(bridge.chunk, cx, cy, tier)

    @app.get("/chunks")
    async def chunks() -> list[dict[str, Any]]:
        return await run_in_threadpool(bridge.chunks)

    @app.post("/player")
    async def player(body: PlayerPosition) -> dict[str, Any]:
        await bridge.load()
        warmed = await run_in_threadpool(bridge.player, body.x, body.y)
        return {"prewarmed": [list(key) for key in warmed]}

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
        subscriber: Subscriber = (asyncio.get_running_loop(), queue)
        bridge.subscribers.add(subscriber)
        forward = asyncio.create_task(_forward(ws, queue))
        try:
            await ws.send_json({"type": "hello"})
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass  # clients send nothing; receiving is how a disconnect shows up
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            forward.cancel()
            bridge.subscribers.discard(subscriber)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request: Request, exc: RuntimeError) -> Response:
        # a generator naming its missing requirement reaches the caller as JSON instead of a bare 500
        log.error("%s %s failed: %s", request.method, request.url.path, exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return app


async def _forward(ws: WebSocket, queue: asyncio.Queue[Event]) -> None:
    """Send the subscriber every event `broadcast` queued for it, until it leaves."""
    try:
        while True:
            await ws.send_json(await queue.get())
    except Exception:  # the subscriber left mid-send; the receive loop is already cleaning up
        log.debug("events subscriber left mid-send", exc_info=True)
