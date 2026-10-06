"""FastAPI routes over the `World` seam: PNG assets, chunk JSON, player position, benchmark report, WebSocket ready
events and the operator page. Every call into the world or the generator runs in the threadpool under one lock (neither
the world state graph nor the GPU adapter is thread-safe) so the event loop stays free while a chunk generates."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from realmweaver.biomes import biome_names, load_biome
from realmweaver.config import settings
from realmweaver.metrics import histogram_embed, tileability
from realmweaver.types import Asset, AssetSpec, Chunk, Generator, Tier
from realmweaver.world import World

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
IMMUTABLE = "public, max-age=31536000, immutable"
MAX_GENERATED = 256  # ponytail: /generate results kept resident, oldest out; the upgrade path is a byte cap like the world's
Event = dict[str, Any]


class GenerateRequest(BaseModel):  # AssetSpec fields; `subject` is the tile class or the prop name
    biome: str
    kind: Literal["texture", "sprite"] = "texture"
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


def latest_report(reports_dir: Path) -> dict[str, Any]:
    """The newest `bench-*.json` (the file name carries the timestamp) or an honest `{"available": False}`."""
    files = sorted(reports_dir.glob("bench-*.json"))
    if not files:
        return {"available": False}
    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("benchmark report %s is unreadable", files[-1], exc_info=True)
        return {"available": False}
    return {**data, "available": True, "file": files[-1].name}


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
    before a diffusion model loads; `lock` serialises every call into the world or the generator."""

    def __init__(self, world: World | None, generator: Generator | None) -> None:
        self._world, self._generator = world, world.generator if world else generator
        self.lock = threading.Lock()
        self.listeners: set[asyncio.Queue[Event]] = set()
        self.generated: OrderedDict[str, Asset] = OrderedDict()
        self.biome = world.biome if world else os.environ.get("REALMWEAVER_BIOME", "forest")
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
            state = getattr(self.world.graph.chunk(cx, cy), "state", None)
            chunk = self.world.request_chunk(cx, cy, tier)
            return chunk_json(chunk), ready_event(chunk) if chunk.state != state else None

    def player(self, x: float, y: float) -> tuple[list[tuple[int, int]], list[Event]]:
        with self.lock:
            self.world.observe_player(x, y)
            warmed = self.world.tick(1)
            chunks = (self.world.graph.chunk(*key) for key in warmed)
            return warmed, [ready_event(c) for c in chunks if c is not None]

    def generate(self, spec: AssetSpec) -> dict[str, Any]:
        with self.lock:
            asset = self.generated.get(spec.id) or self.world.generator.generate(spec)
            self.generated[spec.id] = asset
            self.generated.move_to_end(spec.id)
            while len(self.generated) > MAX_GENERATED:
                self.generated.popitem(last=False)
        return {"id": asset.id, "latency_s": asset.latency_s, "tileability": tileability(asset.image)}

    def png(self, asset_id: str) -> bytes:
        with self.lock:
            try:
                asset = self.generated.get(asset_id) or self.world.asset(asset_id)
            except KeyError:
                raise HTTPException(404, f"unknown asset {asset_id}") from None
        buf = io.BytesIO()
        Image.fromarray(asset.image).save(buf, format="PNG")
        return buf.getvalue()

    def stats(self) -> dict[str, Any]:  # empty until the first generation builds the world
        with self.lock:
            return self.world.stats() if self.loaded else {}

    def broadcast(self, event: Event) -> None:  # event loop only
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
        return await run_in_threadpool(bridge.generate, AssetSpec(**body.model_dump()))

    @app.get("/asset/{asset_id}.png")
    async def asset(asset_id: str) -> Response:
        data = await run_in_threadpool(bridge.png, asset_id)
        return Response(data, media_type="image/png", headers={"Cache-Control": IMMUTABLE})

    @app.get("/chunk/{cx}/{cy}")
    async def chunk(cx: int, cy: int, tier: Tier = "draft") -> dict[str, Any]:
        payload, event = await run_in_threadpool(bridge.chunk, cx, cy, tier)
        if event is not None:
            bridge.broadcast(event)
        return payload

    @app.post("/player")
    async def player(body: PlayerPosition) -> dict[str, Any]:
        warmed, events = await run_in_threadpool(bridge.player, body.x, body.y)
        for event in events:
            bridge.broadcast(event)
        return {"prewarmed": [list(key) for key in warmed]}

    @app.get("/report")
    async def report() -> dict[str, Any]:
        return await run_in_threadpool(latest_report, settings().reports_dir)

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
