"""Contract test: the Unity client's C# DTOs match the bridge's chunk JSON (tests/fixtures/chunk_example.json).

JsonUtility maps JSON keys to public field names one to one, so a DTO field that is not a key in the payload is a
silent null on the Unity side. The bridge test asserts its live `/chunk` response carries the same keys as the fixture.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIENT_CS = ROOT / "unity" / "com.realmweaver.client" / "Runtime" / "RealmWeaverClient.cs"
FIXTURE = ROOT / "tests" / "fixtures" / "chunk_example.json"

CHUNK_KEYS = {"cx", "cy", "size", "biome", "state", "classes", "tilesFlat", "assetList", "prefabList"}

# The WS /events "ready" message documented in the client README; the chunk fixture carries no event.
READY_EVENT_EXAMPLE = {"type": "ready", "chunk": [0, 0], "assets": [{"k": "grass", "v": "b47d7ceb24ef4926"}]}

# Line comments are stripped first; DTO bodies then hold only fields (no braces), so matching to the first '}' is exact.
COMMENT_RE = re.compile(r"//[^\n]*")
CLASS_RE = re.compile(r"\[Serializable\]\s*public\s+class\s+(\w+)\s*\{([^}]*)\}")
FIELD_RE = re.compile(r"^\s*public\s+(?!static\b|const\b)([\w<>\[\]]+)\s+(\w+)\s*(?:=[^;]*)?;", re.MULTILINE)


def serializable_fields(source: str) -> dict[str, dict[str, str]]:
    """{class name: {field name: C# type}} for every [Serializable] class in a C# source file."""
    code = COMMENT_RE.sub("", source)
    return {
        name: {field: ctype for ctype, field in FIELD_RE.findall(body)}
        for name, body in CLASS_RE.findall(code)
    }


@pytest.fixture(scope="module")
def dtos() -> dict[str, dict[str, str]]:
    return serializable_fields(CLIENT_CS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chunk() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_parser_finds_the_three_dtos(dtos: dict[str, dict[str, str]]) -> None:
    assert set(dtos) == {"ChunkDto", "KV", "ReadyEvent"}
    assert dtos["KV"] == {"k": "string", "v": "string"}
    assert set(dtos["ChunkDto"]) == CHUNK_KEYS
    assert dtos["ChunkDto"]["tilesFlat"] == "int[]"
    assert dtos["ChunkDto"]["classes"] == "string[]"
    assert dtos["ChunkDto"]["assetList"] == dtos["ChunkDto"]["prefabList"] == "List<KV>"


def test_chunk_dto_fields_are_fixture_keys(dtos: dict[str, dict[str, str]], chunk: dict) -> None:
    missing = set(dtos["ChunkDto"]) - set(chunk)
    assert not missing, f"ChunkDto fields absent from the chunk JSON: {sorted(missing)}"


def test_kv_fields_match_list_items(dtos: dict[str, dict[str, str]], chunk: dict) -> None:
    for key in ("assetList", "prefabList"):
        assert chunk[key], f"{key} must not be empty"
        for item in chunk[key]:
            assert set(item) == set(dtos["KV"]), f"{key} item {item} is not a {{k, v}} pair"


def test_ready_event_fields(dtos: dict[str, dict[str, str]]) -> None:
    assert set(dtos["ReadyEvent"]) <= set(READY_EVENT_EXAMPLE)
    assert all(set(a) == set(dtos["KV"]) for a in READY_EVENT_EXAMPLE["assets"])


def test_fixture_is_internally_consistent(chunk: dict) -> None:
    size = chunk["size"]
    assert len(chunk["tilesFlat"]) == size * size
    assert all(0 <= i < len(chunk["classes"]) for i in chunk["tilesFlat"])
    rows = [[chunk["classes"][i] for i in chunk["tilesFlat"][r * size : (r + 1) * size]] for r in range(size)]
    assert rows == chunk["tiles"], "tilesFlat must be the row-major flattening of tiles"
    assert {kv["k"]: kv["v"] for kv in chunk["assetList"]} == chunk["assets"]
    assert {kv["k"]: kv["v"] for kv in chunk["prefabList"]} == chunk["prefabs"]
    assert all(name == f"Prefab_{cls}" for cls, name in chunk["prefabs"].items())
    assert chunk["state"] in {"pending", "draft", "ready"}
