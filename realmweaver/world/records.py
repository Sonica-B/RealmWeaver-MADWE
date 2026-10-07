"""The record schema of the world state graph (ADR-0004), declared once: every node kind with the attributes a
record of it must and may carry, and every edge kind with the (source kind, target kind) pairs it may join, the
attributes it carries, the cardinalities it imposes and whether it is symmetric. `WorldStateGraph.add` and
`link` check each write against these tables, `validate` re-checks the whole graph against them and `from_json`
checks a loaded file against them, so a new kind for the game is one more row here and nothing else.

Attribute types are JSON's (str, int, float, bool, list, dict, None, or a union of them), so a record always
round-trips through node-link JSON; an int satisfies a float attribute, a bool satisfies only a bool one. A node
id is `<kind, lower-case>:<name>` (`region:forest`, `chunk:0,0`, `npc:Mara Vell`); the World node is `world`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import UnionType

from realmweaver.types import opposite

WORLD = "world"  # the one World node's id
Spec = type | UnionType  # a JSON type or a union of them, as `isinstance` reads it


@dataclass(frozen=True)
class Record:
    """One node as a read record: its id, kind and a copy of its attributes (writes go through the graph)."""

    id: str
    kind: str
    attrs: dict[str, object]

    def __getitem__(self, name: str) -> object:
        return self.attrs[name]


@dataclass(frozen=True)
class NodeKind:
    """What a record of this kind carries: the `required` attributes (name -> type) and the `optional` ones."""

    required: Mapping[str, Spec] = field(default_factory=dict)
    optional: Mapping[str, Spec] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeKind:
    """The (source kind, target kind) `pairs` the edge may join and the attributes it carries; `one_in` names the
    kinds every node of which has exactly one incoming edge of this kind, `one_out` likewise outgoing; `mirror`,
    when set, makes the edge symmetric: `link` adds the reverse edge with `mirror(attrs)` and `validate` expects
    it."""

    pairs: frozenset[tuple[str, str]]
    required: Mapping[str, Spec] = field(default_factory=dict)
    optional: Mapping[str, Spec] = field(default_factory=dict)
    one_in: frozenset[str] = frozenset()
    one_out: frozenset[str] = frozenset()
    mirror: Callable[[dict], dict] | None = None


_GENERATED = {"description": str, "provenance": dict | None}  # what a generated or authored record may carry

NODE_KINDS: dict[str, NodeKind] = {
    # the chunk pipeline (E1-E2): what `World` writes
    "World": NodeKind({"seed": int, "chunk_size": int}),
    "Region": NodeKind(
        {"name": str, "biome": str},
        {"style": list | None, "tileset": dict | None, "coherence_threshold": float | None},
    ),
    "Chunk": NodeKind(
        {"cx": int, "cy": int, "biome": str, "size": int, "state": str, "region": str, "grid": list, "asset_ids": dict}
    ),
    "Tile": NodeKind({"cx": int, "cy": int, "x": int, "y": int, "tile_class": str}),
    "Asset": NodeKind(
        {"region": str, "spec": dict, "nbytes": int},
        {"latency_s": float, "coherence": float | None, "anchored": bool, "style": list | None, **_GENERATED},
    ),
    # terrain (E3): a 3D region window and what the post-pass placed in it
    "Region3D": NodeKind(
        {"name": str, "cells": list, "cell_m": float, "biome_shares": dict},
        {"river_cells": int, "road_cells": int, **_GENERATED},
    ),
    "Settlement": NodeKind(
        {"name": str, "x": int, "y": int, "biome": str},
        {"cell_m": float, "height_m": float, "slope": float, "water_cells": int, "layout": dict | None, **_GENERATED},
    ),
    "Landmark": NodeKind({"name": str, "category": str, "x": int, "y": int}, {"height_m": float, "biome": str, **_GENERATED}),
    # the game (E7-E8): NPCs with their memories, quests with their objectives, factions, events and saves
    "NPC": NodeKind({"name": str}, {"role": str, "home": str, "work": str, "persona": dict, **_GENERATED}),
    "Memory": NodeKind({"text": str, "t": float, "importance": float, "source": str}, {"reflection": bool}),
    "Quest": NodeKind({"name": str}, {"state": str, "reward": dict, **_GENERATED}),
    "Objective": NodeKind({"name": str}, {"done": bool, **_GENERATED}),
    "Faction": NodeKind({"name": str}, {"reputation": float, **_GENERATED}),
    "Event": NodeKind({"name": str, "t": float}, {**_GENERATED}),
    "Save": NodeKind({"name": str, "t": float}, {"path": str, **_GENERATED}),
}  # fmt: skip

_KNOWABLE = ("NPC", "Settlement", "Landmark", "Region", "Region3D", "Quest", "Faction", "Asset")

EDGE_KINDS: dict[str, EdgeKind] = {
    "CONTAINS": EdgeKind(
        frozenset({
            ("World", "Region"), ("Region", "Chunk"), ("Chunk", "Tile"),
            ("World", "Region3D"), ("Region3D", "Settlement"), ("Region3D", "Landmark"),
            ("Region", "NPC"), ("Settlement", "NPC"), ("NPC", "Memory"),
            ("World", "Quest"), ("Quest", "Objective"), ("World", "Faction"), ("World", "Event"), ("World", "Save"),
        }),
        one_in=frozenset(NODE_KINDS) - {"World", "Asset"},  # every record but the world and the assets has one parent
    ),
    "ADJACENT": EdgeKind(
        frozenset({("Tile", "Tile"), ("Chunk", "Chunk"), ("Region", "Region"), ("Settlement", "Settlement")}),
        {"dir": int},
        {"road_cells": int, "cost": float | None},
        mirror=lambda attrs: {**attrs, "dir": opposite(attrs["dir"])},
    ),
    "INSTANCE_OF": EdgeKind(frozenset({("Tile", "Asset")}), one_out=frozenset({"Tile"})),
    "STYLE_ANCHOR": EdgeKind(frozenset({("Region", "Asset")})),
    "KNOWS": EdgeKind(frozenset(("NPC", kind) for kind in _KNOWABLE)),
    "ASSIGNED": EdgeKind(
        frozenset({("Quest", "NPC"), ("Objective", "NPC"), ("Objective", "Settlement"), ("Objective", "Landmark")})
    ),
    "MEMBER_OF": EdgeKind(frozenset({("NPC", "Faction"), ("Settlement", "Faction")})),
    "TRIGGERS": EdgeKind(frozenset({("Event", "Quest"), ("Event", "Event"), ("Objective", "Event"), ("Quest", "Event")})),
}  # fmt: skip


def node_problems(
    kind: str, node_id: str, attrs: Mapping[str, object], *, partial: bool = False
) -> list[str]:
    """Every way `attrs` fails to be a `kind` record with id `node_id`: an unknown kind, an id without the kind's
    prefix, a missing required attribute (unless `partial`, for an update), an unknown attribute, a wrongly typed
    one or one that is not plain JSON. Empty when it is a record."""
    schema = NODE_KINDS.get(kind)
    if schema is None:
        return [f"unknown node kind {kind!r}"]
    prefix = WORLD if kind == "World" else f"{kind.lower()}:"
    problems = [] if node_id.startswith(prefix) else [f"a {kind} id starts with {prefix!r}, got {node_id!r}"]
    return problems + _attr_problems(attrs, schema.required, schema.optional, partial=partial)


def edge_problems(edge_kind: str, src_kind: str, dst_kind: str, attrs: Mapping[str, object]) -> list[str]:
    """Every way an `edge_kind` edge from a `src_kind` record to a `dst_kind` one with `attrs` breaks the schema."""
    schema = EDGE_KINDS.get(edge_kind)
    if schema is None:
        return [f"unknown edge kind {edge_kind!r}"]
    problems = (
        []
        if (src_kind, dst_kind) in schema.pairs
        else [f"{edge_kind} may not join a {src_kind} to a {dst_kind}"]
    )
    return problems + _attr_problems(attrs, schema.required, schema.optional)


def _attr_problems(
    attrs: Mapping[str, object],
    required: Mapping[str, Spec],
    optional: Mapping[str, Spec],
    *,
    partial: bool = False,
) -> list[str]:
    problems = (
        [] if partial else [f"missing required attribute {name!r}" for name in required if name not in attrs]
    )
    for name, value in attrs.items():
        spec = required.get(name, optional.get(name))
        if spec is None:
            problems.append(f"unknown attribute {name!r}")
        elif not _typed(value, spec):
            problems.append(f"attribute {name!r} should be {_name(spec)}, got {type(value).__name__}")
        else:
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                problems.append(f"attribute {name!r} is not plain JSON: {e}")
    return problems


def _typed(value: object, spec: Spec) -> bool:
    members = spec.__args__ if isinstance(spec, UnionType) else (spec,)
    if isinstance(value, bool):  # a bool is an int to Python, not to a record
        return bool in members
    if isinstance(value, int) and float in members:  # JSON has one number type
        return True
    return isinstance(value, spec)


def _name(spec: Spec) -> str:
    return getattr(spec, "__name__", str(spec))
