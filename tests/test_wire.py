"""The wire seam: one field table defines chunk JSON, the ready event, the Unity DTOs and the README contract.

Expected values are independent of the table: the hand-authored fixture (tests/fixtures/chunk_example.json), the
ready event the plan's contract addendum documents, and the files the table is rendered into, which must not drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from realmweaver import wire
from realmweaver.assets import ProceduralGenerator
from realmweaver.bridge import create_app
from realmweaver.cli import main
from realmweaver.types import Chunk, Layout, TileSet
from realmweaver.world import ThreadRunner, World

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "chunk_example.json"
CLIENT_CS = ROOT / "unity" / "com.realmweaver.client" / "Runtime" / "RealmWeaverClient.cs"
CLIENT_README = ROOT / "unity" / "com.realmweaver.client" / "README.md"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def chunk_from(fixture: dict) -> Chunk:
    """The fixture's chunk as the world holds it: a layout whose grid indexes its tileset's classes."""
    classes = list(fixture["classes"])
    n = len(classes)
    tileset = TileSet(classes, np.ones((n, 4, n), dtype=bool), np.full(n, 1 / n))
    grid = np.array([[classes.index(c) for c in row] for row in fixture["tiles"]], dtype=np.int32)
    layout = Layout(grid, tileset)
    return Chunk(
        fixture["cx"], fixture["cy"], fixture["biome"], layout, dict(fixture["assets"]), fixture["state"]
    )


# --- the payloads -------------------------------------------------------------------------------------


def test_chunk_payload_is_the_fixture(fixture: dict) -> None:
    payload = wire.chunk_payload(chunk_from(fixture))
    assert payload == fixture, (
        "the fixture is the binding chunk JSON: every key and value, flat twins included"
    )
    assert list(payload) == list(wire.CHUNK.keys), "keys come out in table order"


def test_chunk_payload_takes_the_prefab_map(fixture: dict) -> None:
    payload = wire.chunk_payload(chunk_from(fixture), prefab_name="Tile_{}".format)
    assert payload["prefabs"]["grass"] == "Tile_grass"
    assert payload["prefabList"] == [{"k": cls, "v": f"Tile_{cls}"} for cls in fixture["assets"]]


def test_ready_event_is_the_documented_message(fixture: dict) -> None:
    assets = {"grass": "b47d7ceb24ef4926"}
    assert wire.ready_event(chunk_from(fixture), assets) == {
        "type": "ready",
        "chunk": [0, 0],
        "assets": {"grass": "b47d7ceb24ef4926"},
        "assetList": [{"k": "grass", "v": "b47d7ceb24ef4926"}],
    }


def test_fixture_keys_are_the_table_keys(fixture: dict) -> None:
    assert set(fixture) == set(wire.CHUNK.keys)


def test_every_map_has_one_flat_twin() -> None:
    for message in wire.MESSAGES:
        for f in message.fields:
            twins = [g.name for g in message.fields if g.twin is not None and g.twin.of == f.name]
            if f.cs is None:
                assert len(twins) == 1, (
                    f"{message.dto}.{f.name} is a map Unity cannot read: it needs one flat twin"
                )
            else:
                assert not twins, f"{message.dto}.{f.name} is read by Unity; nothing should twin it"


def test_bridge_serves_the_wire_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REALMWEAVER_ASSET_SIZE", "64")
    world = World("forest", ProceduralGenerator(), chunk_size=8, seed=1, runner=ThreadRunner(slots=2))
    c = TestClient(create_app(world))
    with c.websocket_connect("/events") as ws:
        assert ws.receive_json() == {"type": "hello"}
        j = c.get("/chunk/0/0").json()
        event = ws.receive_json()
    chunk = world.request_chunk(0, 0).chunk
    assert j == wire.chunk_payload(chunk)
    assert event == wire.ready_event(chunk, chunk.asset_ids)


# --- the generated files ------------------------------------------------------------------------------


def test_csharp_dtos_are_the_client_file_region() -> None:
    assert wire.generated(CLIENT_CS.read_text(encoding="utf-8")) == wire.render_csharp_dtos(), (
        "RealmWeaverClient.cs drifted from the wire table: run `uv run realmweaver wire --write`"
    )


def test_readme_contract_is_the_readme_region() -> None:
    rendered = wire.render_contract_markdown()
    assert wire.generated(CLIENT_README.read_text(encoding="utf-8")) == rendered, (
        "the client README drifted from the wire table: run `uv run realmweaver wire --write`"
    )
    for message in wire.MESSAGES:
        assert all(f"`{key}`" in rendered for key in message.keys), f"{message.dto} keys documented"


def test_regenerate_replaces_only_the_region() -> None:
    text = "head\n    // <wire-generated> note\nold\n    // </wire-generated>\ntail\n"
    assert wire.generated(text) == "old\n"
    assert (
        wire.regenerate(text, "new\n")
        == "head\n    // <wire-generated> note\nnew\n    // </wire-generated>\ntail\n"
    )
    md = "<!-- <wire-generated> -->\n<!-- </wire-generated> -->\n"
    assert (
        wire.generated(md) == ""
        and wire.regenerate(md, "x\n") == "<!-- <wire-generated> -->\nx\n<!-- </wire-generated> -->\n"
    )
    with pytest.raises(ValueError):
        wire.generated("no markers\n")
    with pytest.raises(ValueError):
        wire.generated(text + text)


def test_cli_wire_prints_and_rewrites(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["wire"]) == 0
    assert capsys.readouterr().out == wire.render_csharp_dtos() + wire.render_contract_markdown()
    for rel in wire.GENERATED:  # a stale checkout: the regions hold junk, the rest is the real file
        stale = tmp_path / rel
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text(
            wire.regenerate((ROOT / rel).read_text(encoding="utf-8"), "stale\n"), encoding="utf-8"
        )
    assert main(["wire", "--write", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / rel) for rel in wire.GENERATED]
    for rel in wire.GENERATED:
        assert (tmp_path / rel).read_text(encoding="utf-8") == (ROOT / rel).read_text(encoding="utf-8"), (
            "--write regenerates exactly what the checkout holds"
        )
