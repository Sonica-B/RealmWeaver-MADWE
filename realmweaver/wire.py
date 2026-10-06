"""The wire: every message the bridge sends Unity, defined once as a field table (ADR-0003).

`CHUNK` is the `GET /chunk/{cx}/{cy}` payload, `READY` the WebSocket `ready` event and `KV` the pair both carry maps
as. A row names a JSON key, its JSON type, the C# type `JsonUtility` reads it as (None for a map: JsonUtility reads
neither dictionaries nor jagged arrays, so a map key stays for other clients and a flat twin carries it to Unity),
its one-line meaning and, for a flat twin, the rule that derives it from its map. The one table yields the payloads
(`chunk_payload`, `ready_event`), the `[Serializable]` DTOs of `RealmWeaverClient.cs` (`render_csharp_dtos`) and the
client README's contract tables (`render_contract_markdown`); `realmweaver wire --write` rewrites both files between
their `<wire-generated>` marker lines and tests/test_wire.py fails when either drifts. tests/fixtures/chunk_example.json
is the reference chunk payload.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realmweaver.types import Chunk

__all__ = [
    "CHUNK",
    "GENERATED",
    "KV",
    "MESSAGES",
    "PREFAB_NAME",
    "READY",
    "Field",
    "Message",
    "Twin",
    "chunk_payload",
    "generated",
    "indices",
    "kv",
    "ready_event",
    "regenerate",
    "render_contract_markdown",
    "render_csharp_dtos",
]

Values = dict[str, Any]
PREFAB_NAME: Callable[[str], str] = "Prefab_{}".format  # the default prefab map: tile class -> prefab name


@dataclass(frozen=True)
class Twin:
    """How a flat field stands in for a map: the map's key and the rule deriving the field from the message values."""

    of: str
    rule: Callable[[Values], Any]


@dataclass(frozen=True)
class Field:
    """One row: JSON key (also the C# field name), JSON type, C# type (None: a map Unity ignores), one-line meaning
    (the C# trailing comment and the README note) and, for a flat field, the twin rule that fills it."""

    name: str
    json: str
    cs: str | None
    doc: str = ""
    twin: Twin | None = None


@dataclass(frozen=True)
class Message:
    """One wire message: its C# DTO name, the `<summary>` line and its fields in wire order."""

    dto: str
    summary: str
    fields: tuple[Field, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def encode(self, **values: Any) -> Values:
        """The message in key order from its primary values, every flat twin derived by its rule."""
        for f in self.fields:
            if f.twin is not None:
                values[f.name] = f.twin.rule(values)
        if set(values) != set(self.keys):
            missing, unknown = set(self.keys) - set(values), set(values) - set(self.keys)
            raise TypeError(f"{self.dto}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        return {key: values[key] for key in self.keys}


def kv(of: str) -> Twin:
    """The flat twin of the object at `of`: `[{"k", "v"}]` in its order, which JsonUtility reads as `List<KV>`."""
    return Twin(of, lambda v: [{"k": key, "v": value} for key, value in v[of].items()])


def indices(of: str, into: str) -> Twin:
    """The flat twin of the rows of names at `of`: row-major indices into the list of names at `into`."""
    return Twin(of, lambda v: [v[into].index(name) for row in v[of] for name in row])


# fmt: off
KV = Message(
    "KV",
    "One key/value pair: the JsonUtility-readable form of a JSON object used as a map.",
    (
        Field("k", "string", "string"),
        Field("v", "string", "string"),
    ),
)

CHUNK = Message(
    "ChunkDto",
    "GET /chunk/{cx}/{cy}. Field names are the JSON keys; tests/test_unity_protocol.py checks them.",
    (
        Field("cx", "int", "int"),
        Field("cy", "int", "int"),
        Field("size", "int", "int", "tiles per side"),
        Field("biome", "string", "string"),
        Field("state", "string", "string", "pending | draft | ready"),
        Field("tiles", "[[string]]", None, "rows of tile class names, row 0 first"),
        Field("classes", "string[]", "string[]", "tile class names; tilesFlat indexes into this"),
        Field("tilesFlat", "int[]", "int[]", "size*size entries, row-major, row 0 first", indices("tiles", "classes")),
        Field("assets", "{string: string}", None, "tile class -> asset id"),
        Field("assetList", '[{"k", "v"}]', "List<KV>", "tile class -> asset id, fetched as /asset/<id>.png", kv("assets")),
        Field("prefabs", "{string: string}", None, "tile class -> prefab name"),
        Field("prefabList", '[{"k", "v"}]', "List<KV>", 'tile class -> prefab name, "Prefab_<tileClass>"', kv("prefabs")),
    ),
)

READY = Message(
    "ReadyEvent",
    'WS /events message. Only type == "ready" is acted on; "hello" parses with chunk == null.',
    (
        Field("type", "string", "string"),
        Field("chunk", "int[]", "int[]", "[cx, cy]"),
        Field("assets", "{string: string}", None, "tile class -> asset id"),
        Field("assetList", '[{"k", "v"}]', "List<KV>", "tile class -> asset id", kv("assets")),
    ),
)
# fmt: on

MESSAGES = (KV, CHUNK, READY)  # in the order RealmWeaverClient.cs declares them


# -- the payloads -----------------------------------------------------------------------------------------


def chunk_payload(chunk: Chunk, prefab_name: Callable[[str], str] = PREFAB_NAME) -> Values:
    """`GET /chunk/{cx}/{cy}`: the chunk as Unity and the operator page read it; `prefab_name` is the prefab map."""
    layout, assets = chunk.layout, dict(chunk.asset_ids)
    return CHUNK.encode(
        cx=chunk.cx,
        cy=chunk.cy,
        size=layout.width,
        biome=chunk.biome,
        state=chunk.state,
        tiles=layout.class_rows(),
        classes=list(layout.tileset.classes),
        assets=assets,
        prefabs={cls: prefab_name(cls) for cls in assets},
    )


def ready_event(chunk: Chunk, assets: Mapping[str, str]) -> Values:
    """The `WS /events` message for a chunk the world finished: its coordinates and its assets by tile class."""
    return READY.encode(type="ready", chunk=[chunk.cx, chunk.cy], assets=dict(assets))


# -- the generated files ----------------------------------------------------------------------------------


def render_csharp_dtos() -> str:
    """The `[Serializable]` classes of RealmWeaverClient.cs: one per message, a map key left out (JsonUtility would
    read it as null), the trailing comments aligned."""
    width = max(len(f"public {f.cs} {f.name};") for m in MESSAGES for f in m.fields if f.cs)
    blocks = []
    for m in MESSAGES:
        lines = [
            f"    /// <summary>{m.summary}</summary>",
            "    [Serializable]",
            f"    public class {m.dto}",
            "    {",
        ]
        for f in m.fields:
            if f.cs:
                decl = f"public {f.cs} {f.name};"
                lines.append(f"        {decl:<{width}} // {f.doc}" if f.doc else f"        {decl}")
        blocks.append("\n".join([*lines, "    }"]))
    return "\n\n".join(blocks) + "\n"


def render_contract_markdown() -> str:
    """The client README's contract: one table per message, a map's note naming the flat twin Unity reads instead."""
    out: list[str] = []
    for m in MESSAGES:
        out += [f"### `{m.dto}`", "", _md(m.summary), "", "| Key | JSON | C# | Notes |", "|---|---|---|---|"]
        for f in m.fields:
            cs = f"`{f.cs}`" if f.cs else "ignored"
            out.append(f"| `{f.name}` | `{f.json}` | {cs} | {_md(_note(m, f))} |")
        out.append("")
    return "\n".join([*out, ""])  # a blank line closes the last table before the end marker


def _note(m: Message, f: Field) -> str:
    parts = [f.doc] if f.doc else []
    if f.twin is not None:
        parts.append(f"flat twin of `{f.twin.of}`")
    if f.cs is None:
        twins = " and ".join(f"`{g.name}`" for g in m.fields if g.twin is not None and g.twin.of == f.name)
        parts.append(f"Unity reads {twins} instead; kept for other clients")
    return "; ".join(parts)


def _md(text: str) -> str:
    """Text safe inside a Markdown table cell: pipes and angle brackets escaped."""
    return text.replace("|", "\\|").replace("<", "\\<")


_REGION = re.compile(
    r"^[ \t]*(?://|<!--) <wire-generated>[^\n]*\n(.*?)^[ \t]*(?://|<!--) </wire-generated>[^\n]*\n",
    re.MULTILINE | re.DOTALL,
)

GENERATED: dict[Path, Callable[[], str]] = {  # file under the checkout -> what its generated region holds
    Path("unity/com.realmweaver.client/Runtime/RealmWeaverClient.cs"): render_csharp_dtos,
    Path("unity/com.realmweaver.client/README.md"): render_contract_markdown,
}


def _region(text: str) -> re.Match[str]:
    found = list(_REGION.finditer(text))
    if len(found) != 1:
        raise ValueError(f"expected one <wire-generated> region, found {len(found)}")
    return found[0]


def generated(text: str) -> str:
    """What a file holds between its `<wire-generated>` and `</wire-generated>` marker lines (exactly one pair)."""
    return _region(text).group(1)


def regenerate(text: str, body: str) -> str:
    """`text` with its generated region replaced by `body`; the marker lines and everything around them stay."""
    m = _region(text)
    return text[: m.start(1)] + body + text[m.end(1) :]
