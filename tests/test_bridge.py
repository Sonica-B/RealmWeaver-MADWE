"""The bridge seam: HTTP routes, chunk JSON against the Unity fixture, WebSocket ready events, the operator page.

The plan's six tests come first, unchanged in substance (ruff format splits their one-liners). The world behind the
TestClient uses the procedural generator with 64 px assets. Tests that post `/player` use the client as a context
manager: prewarm runs on a background task, which needs one event loop across the requests and the WebSocket.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from realmweaver.assets import ProceduralGenerator
from realmweaver.bridge import create_app
from realmweaver.world import World

FIXTURE = Path(__file__).parent / "fixtures" / "chunk_example.json"
ASSET_SIZE = 64


@pytest.fixture(autouse=True)
def _small_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REALMWEAVER_ASSET_SIZE", str(ASSET_SIZE))


def _client() -> TestClient:
    return TestClient(create_app(World("forest", ProceduralGenerator(), chunk_size=8, seed=1)))


# --- the plan's tests ---------------------------------------------------------------------------------


def test_health_and_biomes():
    c = _client()
    assert c.get("/health").json()["generator"] == "procedural"
    assert any(b["name"] == "forest" and "grass" in b["tiles"] for b in c.get("/biomes").json())


def test_generate_then_fetch_png():
    c = _client()
    r = c.post("/generate", json={"biome": "forest", "subject": "grass", "size": 64, "seed": 1}).json()
    png = c.get(f"/asset/{r['id']}.png")
    assert png.headers["content-type"] == "image/png"
    assert np.asarray(Image.open(io.BytesIO(png.content))).shape == (64, 64, 3)


def test_chunk_json_shape_and_prefab_map():
    c = _client()
    j = c.get("/chunk/0/0").json()
    assert (
        len(j["tiles"]) == 8
        and set(j["assets"]) == set(j["prefabs"])
        and j["prefabs"]["grass"] == "Prefab_grass"
    )


def test_player_prewarm_and_ws_ready_event():
    with _client() as c:
        c.get("/chunk/0/0")
        with c.websocket_connect("/events") as ws:
            assert ws.receive_json()["type"] == "hello"
            for x in range(0, 40, 2):
                c.post("/player", json={"x": x, "y": 4})
            assert ws.receive_json()["type"] == "ready"


def test_report_absent_is_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("REALMWEAVER_REPORTS_DIR", str(tmp_path))
    c = _client()
    assert c.get("/report").json() == {"available": False}


def test_index_served():
    assert "RealmWeaver" in _client().get("/").text


# --- beyond the plan ----------------------------------------------------------------------------------


def test_chunk_response_carries_every_fixture_key_and_is_consistent():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    j = _client().get("/chunk/2/-1?tier=draft").json()
    assert set(j) == set(fixture), "the chunk JSON is the Unity contract: exactly the fixture's keys"
    assert (j["cx"], j["cy"], j["biome"], j["size"], j["state"]) == (2, -1, "forest", 8, "draft")
    assert len(j["tilesFlat"]) == 64 and all(0 <= i < len(j["classes"]) for i in j["tilesFlat"])
    rows = [[j["classes"][i] for i in j["tilesFlat"][r * 8 : (r + 1) * 8]] for r in range(8)]
    assert rows == j["tiles"], "tilesFlat is the row-major flattening of tiles"
    assert {kv["k"]: kv["v"] for kv in j["assetList"]} == j["assets"]
    assert {kv["k"]: kv["v"] for kv in j["prefabList"]} == j["prefabs"]
    assert all(name == f"Prefab_{cls}" for cls, name in j["prefabs"].items())
    assert {cls for row in j["tiles"] for cls in row} == set(j["assets"]), "one asset per tile class present"


def test_chunk_assets_are_served_as_immutable_png():
    c = _client()
    j = c.get("/chunk/0/0").json()
    for asset_id in j["assets"].values():
        r = c.get(f"/asset/{asset_id}.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert np.asarray(Image.open(io.BytesIO(r.content))).shape == (ASSET_SIZE, ASSET_SIZE, 3)


def test_unknown_asset_is_404():
    assert _client().get("/asset/0123456789abcdef.png").status_code == 404


def test_generate_sprite_keeps_alpha_and_reports_measurements():
    c = _client()
    body = {"biome": "forest", "kind": "sprite", "subject": "mushroom", "size": 64, "seed": 3}
    r = c.post("/generate", json=body)
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"id", "latency_s", "tileability"} and j["latency_s"] >= 0
    assert j["tileability"] == 0.0, "a sprite's wrapped edges are all white: zero seam gradient"
    png = Image.open(io.BytesIO(c.get(f"/asset/{j['id']}.png").content))
    assert png.mode == "RGBA" and np.asarray(png).shape == (64, 64, 4)


def test_generate_mesh_then_fetch_glb():
    c = _client()
    body = {"biome": "forest", "kind": "mesh", "subject": "rock", "size": 16, "seed": 1}
    r = c.post("/generate", json=body)
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"id", "latency_s", "tileability"} and j["tileability"] is None, "no seams on a mesh"
    glb = c.get(f"/asset/{j['id']}.glb")
    assert glb.status_code == 200 and glb.headers["content-type"] == "model/gltf-binary"
    assert (
        glb.headers["cache-control"] == "public, max-age=31536000, immutable" and glb.content[:4] == b"glTF"
    )
    assert c.get(f"/asset/{j['id']}.png").status_code == 404, "a mesh has no PNG representation"


def test_generate_is_content_addressed_and_validated():
    c = _client()
    body = {"biome": "forest", "subject": "grass", "size": 64, "seed": 1}
    assert c.post("/generate", json=body).json()["id"] == c.post("/generate", json=body).json()["id"]
    assert c.post("/generate", json={"biome": "atlantis", "subject": "grass"}).status_code == 404
    assert c.post("/generate", json={"biome": "forest", "subject": "grass", "size": 60}).status_code == 422
    assert c.post("/generate", json={"biome": "forest", "tier": "ultra"}).status_code == 422


def test_refine_tier_marks_chunk_ready_and_bad_tier_is_rejected():
    c = _client()
    assert c.get("/chunk/0/0").json()["state"] == "draft"
    assert c.get("/chunk/0/0?tier=refine").json()["state"] == "ready"
    assert c.get("/chunk/0/0?tier=final").status_code == 422


def test_ws_ready_event_follows_a_direct_chunk_request():
    c = _client()
    with c.websocket_connect("/events") as ws:
        assert ws.receive_json() == {"type": "hello"}
        j = c.get("/chunk/1/1").json()
        event = ws.receive_json()
        assert event["type"] == "ready" and event["chunk"] == [1, 1]
        assert event["assets"] == j["assets"] and event["assetList"] == j["assetList"]


def test_bridge_never_reads_the_world_state_graph(monkeypatch: pytest.MonkeyPatch):
    """Ready events come from `request_chunk`'s transition, listings and prewarm from `World` reads: every
    route still works while reading `graph`, which is for tests and the bench, raises."""

    def forbidden(_world):
        raise AssertionError("the bridge read world.graph")

    monkeypatch.setattr(World, "graph", property(forbidden), raising=False)
    with _client() as c, c.websocket_connect("/events") as ws:
        assert ws.receive_json() == {"type": "hello"}
        assert c.get("/chunk/0/0").json()["state"] == "draft"  # created: a ready event
        assert ws.receive_json()["chunk"] == [0, 0]
        assert c.get("/chunk/0/0?tier=refine").json()["state"] == "ready"  # refined: another
        assert ws.receive_json()["chunk"] == [0, 0]
        assert c.get("/chunk/0/0").json()["state"] == "ready"  # reused: none, so the next event is (1, 0)'s
        assert c.get("/chunk/1/0").json()["state"] == "draft"
        assert ws.receive_json()["chunk"] == [1, 0]
        assert {(j["cx"], j["cy"]) for j in c.get("/chunks").json()} == {(0, 0), (1, 0)}
        warmed: list[list[int]] = []
        for x in range(0, 40, 2):
            warmed += c.post("/player", json={"x": x, "y": 4}).json()["prewarmed"]
        deadline = time.monotonic() + 5
        while not warmed and time.monotonic() < deadline:
            time.sleep(0.05)
            warmed += c.post("/player", json={"x": 38, "y": 4}).json()["prewarmed"]
        assert warmed, "prewarm failed: it logs instead of raising"


def test_player_response_lists_prewarmed_chunks_that_then_exist():
    with _client() as c:
        c.get("/chunk/0/0")
        warmed: list[list[int]] = []
        for x in range(0, 40, 2):
            r = c.post("/player", json={"x": x, "y": 4.0})
            assert r.status_code == 200
            warmed += r.json()["prewarmed"]  # what the background task prewarmed since the previous call
        deadline = time.monotonic() + 10
        while not warmed and time.monotonic() < deadline:
            time.sleep(0.05)
            warmed += c.post("/player", json={"x": 38, "y": 4.0}).json()["prewarmed"]
        assert warmed and all(len(k) == 2 for k in warmed)
        for cx, cy in warmed:
            assert c.get(f"/chunk/{cx}/{cy}").json()["state"] in {"draft", "ready"}
        assert c.get("/stats").json()["chunks"] >= 1 + len({tuple(k) for k in warmed})


def test_resident_png_is_served_while_the_generator_lock_is_held():
    app = create_app(World("forest", ProceduralGenerator(), chunk_size=8, seed=1))
    c = TestClient(app)
    ids = list(c.get("/chunk/0/0").json()["assets"].values())
    generated = c.post("/generate", json={"biome": "forest", "subject": "rock", "size": 64, "seed": 2}).json()
    ids.append(generated["id"])
    codes: list[int] = []
    fetch = threading.Thread(target=lambda: codes.extend(c.get(f"/asset/{i}.png").status_code for i in ids))
    with app.state.bridge.lock:  # a chunk is generating: world calls wait, PNGs of announced assets must not
        fetch.start()
        fetch.join(timeout=5)
        assert not fetch.is_alive(), "GET /asset waited for the generator lock"
    assert codes == [200] * len(ids)


def test_chunks_lists_every_resident_chunk():
    c = _client()
    assert c.get("/chunks").json() == []
    c.get("/chunk/0/0")
    c.get("/chunk/1/0?tier=refine")
    listed = {(j["cx"], j["cy"]): j for j in c.get("/chunks").json()}
    assert set(listed) == {(0, 0), (1, 0)}
    assert listed[(0, 0)]["state"] == "draft" and listed[(1, 0)]["state"] == "ready"
    assert all(set(j) == {"cx", "cy", "state", "biome"} and j["biome"] == "forest" for j in listed.values())


def test_report_returns_the_newest_bench_file(tmp_path, monkeypatch):
    monkeypatch.setenv("REALMWEAVER_REPORTS_DIR", str(tmp_path))
    (tmp_path / "bench-20260101-000000.json").write_text(json.dumps({"schema": 1, "run": "old"}))
    (tmp_path / "bench-20260301-120000.json").write_text(json.dumps({"schema": 1, "run": "new"}))
    (tmp_path / "notes.json").write_text("{}")
    j = _client().get("/report").json()
    assert j["available"] is True and j["run"] == "new"


def test_stats_and_health_describe_the_injected_world():
    c = _client()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["generator"] == "procedural" and h["device"] == "cpu"
    assert h["biome"] == "forest" and h["chunk_size"] == 8
    s = c.get("/stats").json()
    assert {"chunks", "assets", "cache_bytes", "prewarm_hits", "seam_violations"} <= set(s)


def test_default_world_is_built_lazily_from_the_environment(monkeypatch):
    monkeypatch.setenv("REALMWEAVER_DEVICE", "cpu")
    monkeypatch.setenv("REALMWEAVER_BIOME", "desert")
    monkeypatch.setenv("REALMWEAVER_CHUNK_SIZE", "8")
    c = TestClient(create_app())
    h = c.get("/health").json()
    assert h["generator"] == "procedural" and h["device"] == "cpu" and h["loaded"] is False
    assert c.get("/chunks").json() == [] and c.get("/health").json()["loaded"] is False
    j = c.get("/chunk/0/0").json()
    assert j["biome"] == "desert" and j["size"] == 8 and "sand" in j["classes"]
    assert c.get("/health").json()["loaded"] is True


def test_module_exposes_one_lazy_app():
    import realmweaver.bridge as bridge

    assert isinstance(bridge.app, FastAPI) and bridge.app is bridge.app


def test_operator_page_is_self_contained():
    html = _client().get("/").text
    assert "<title>RealmWeaver</title>" in html
    assert not re.search(r"<script[^>]+src=", html), "no external scripts"
    assert not re.search(r"<link[^>]+href=\"?https?://", html), "no external stylesheets"
    assert "prefers-color-scheme" in html and "No benchmark report yet" in html
    assert "live counters, not benchmarks" in html, "the stats panel is labelled as state, not a benchmark"
    assert "'/chunks'" in html, "the page hydrates the map from the resident chunks on load"
