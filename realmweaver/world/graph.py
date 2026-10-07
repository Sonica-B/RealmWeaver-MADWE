"""World state graph: the typed property graph that is a world's source of truth (ADR-0004).

Every node is a record of a kind in `records.NODE_KINDS` and every edge one of `records.EDGE_KINDS`, typed by its
MultiDiGraph key: `add` and `link` are the two writes, checked against the schema before anything changes, and the
typed helpers (`add_region`, `add_chunk`, `add_asset`, ...) are wrappers over them. Node ids are strings and every
attribute is plain JSON, so the graph round-trips through node-link JSON. Style vectors are whatever the injected
embedder returned.
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

import networkx as nx
import numpy as np

from realmweaver.types import DIR_NAMES, DIRS, Asset, AssetSpec, Chunk, Layout, TileSet, opposite
from realmweaver.world.records import EDGE_KINDS, NODE_KINDS, WORLD, Record, edge_problems, node_problems

log = logging.getLogger(__name__)

STYLE_ALPHA = 0.2  # EMA weight of each new asset in its region's style vector
_EPS = 1e-9
ChunkState = Literal["pending", "draft", "ready"]
Direction = Literal["out", "in"]


@dataclass(frozen=True)
class Region:
    """A contiguous set of chunks sharing one biome (the glossary's Region) as a read record: `id` is the
    Region node's id, stable across save and load; `chunks` the keys of the chunks it CONTAINS; `tileset` the
    compiled tileset its chunks are solved with (None until `add_region` or the first chunk recorded one) and
    `coherence_threshold` the biome's, as `World` recorded it (None when it was added without one)."""

    id: str
    name: str
    biome: str
    chunks: frozenset[tuple[int, int]]
    tileset: TileSet | None = None
    coherence_threshold: float | None = None

    def shares_tileset(self, other: Region) -> bool:
        """True when both regions solve with the same tile classes and adjacency table, so a border between them
        can be constrained; False when either has no tileset yet."""
        a, b = self.tileset, other.tileset
        if a is None or b is None:
            return False
        return a is b or (a.classes == b.classes and bool(np.array_equal(a.allowed, b.allowed)))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), _EPS))


def _region_node(name: str) -> str:
    return f"region:{name}"


def _chunk_node(cx: int, cy: int) -> str:
    return f"chunk:{cx},{cy}"


def _tile_node(cx: int, cy: int, x: int, y: int) -> str:
    return f"tile:{cx},{cy},{x},{y}"


def _asset_node(asset_id: str) -> str:
    return f"asset:{asset_id}"


class WorldStateGraph:
    """`networkx.MultiDiGraph` in `g`, the live `Chunk` objects in `chunks`, plus an index of asset style vectors."""

    def __init__(self, seed: int = 0, chunk_size: int = 16) -> None:
        self.g = nx.MultiDiGraph()
        self.chunks: dict[tuple[int, int], Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}  # asset id -> style vector
        self._styles: dict[str, np.ndarray] = {}  # region name -> EMA style vector
        self._tilesets: dict[str, TileSet] = {}  # region name -> the compiled tileset its node holds as JSON
        self.add("World", WORLD, seed=seed, chunk_size=chunk_size)

    # -- records: the two writes and the three reads every kind goes through ------------------------------

    def add(self, kind: str, node_id: str, **attrs: object) -> None:
        """Write one record: a node of `kind` with id `node_id` and plain-JSON attributes, checked against
        `NODE_KINDS` before anything is written; ValueError names what is wrong (the kind, the id's prefix, a
        missing, unknown or wrongly typed attribute, or an id already in the graph)."""
        if node_id in self.g:
            raise ValueError(f"{node_id}: already in the graph")
        problems = node_problems(kind, node_id, attrs)
        if problems:
            raise ValueError(f"{node_id}: {problems[0]}")
        self.g.add_node(node_id, kind=kind, **attrs)

    def link(self, edge_kind: str, src: str, dst: str, **attrs: object) -> None:
        """Write one `edge_kind` edge from record `src` to record `dst`, checked against `EDGE_KINDS` (the kind,
        the pair of record kinds, the attributes); a symmetric kind (ADJACENT) gets its mirror edge back as well.
        ValueError names what is wrong, including an endpoint that is no record."""
        for node in (src, dst):
            if node not in self.g:
                raise ValueError(f"{edge_kind} edge {src} -> {dst}: no record {node!r}")
        problems = edge_problems(edge_kind, self.g.nodes[src]["kind"], self.g.nodes[dst]["kind"], attrs)
        if problems:
            raise ValueError(f"{edge_kind} edge {src} -> {dst}: {problems[0]}")
        self.g.add_edge(src, dst, key=edge_kind, **attrs)
        if (mirror := EDGE_KINDS[edge_kind].mirror) is not None:
            self.g.add_edge(dst, src, key=edge_kind, **mirror(attrs))

    def get(self, node_id: str) -> Record | None:
        """The record with id `node_id`; None when the graph holds none."""
        return self._record(node_id) if node_id in self.g else None

    def nodes(self, kind: str) -> list[Record]:
        """Every record of `kind`, in insertion order; ValueError for a kind the schema does not know."""
        if kind not in NODE_KINDS:
            raise ValueError(f"unknown node kind {kind!r}")
        return [self._record(n) for n, d in self.g.nodes(data=True) if d["kind"] == kind]

    def neighbours(
        self, node_id: str, edge_kind: str | None = None, direction: Direction = "out"
    ) -> list[Record]:
        """The records at the far end of the node's `edge_kind` edges (every kind when None), outgoing unless
        `direction="in"`, each once in edge order; KeyError for an unknown node, ValueError for an unknown kind."""
        if node_id not in self.g:
            raise KeyError(f"no record {node_id!r}")
        if edge_kind is not None and edge_kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind {edge_kind!r}")
        edges = (
            self.g.out_edges(node_id, keys=True)
            if direction == "out"
            else self.g.in_edges(node_id, keys=True)
        )
        far = 1 if direction == "out" else 0
        return [self._record(n) for n in dict.fromkeys(e[far] for e in edges if edge_kind in (None, e[2]))]

    # -- writes -------------------------------------------------------------------------------------------

    def add_region(
        self, name: str, biome: str, tileset: TileSet | None = None, coherence_threshold: float | None = None
    ) -> None:
        """A region of `biome`; `tileset` (else the first chunk's) and `coherence_threshold` go on its record."""
        rnode = _region_node(name)
        self.add("Region", rnode, name=name, biome=biome, coherence_threshold=coherence_threshold)
        self.link("CONTAINS", WORLD, rnode)
        if tileset is not None:
            self._set_tileset(name, tileset)

    def add_chunk(self, chunk: Chunk, region: str) -> None:
        """Add the chunk and its tiles, with ADJACENT edges inside it and across borders to chunks already present."""
        rnode = _region_node(region)
        if rnode not in self.g:
            raise KeyError(f"unknown region {region!r}")
        self._set_tileset(region, chunk.layout.tileset)
        cx, cy, n = chunk.cx, chunk.cy, chunk.layout.width
        cnode, rows = _chunk_node(cx, cy), chunk.layout.class_rows()
        self.add(
            "Chunk", cnode, cx=cx, cy=cy, biome=chunk.biome, size=n, state=chunk.state, region=region,
            grid=chunk.layout.grid.tolist(), asset_ids=chunk.asset_ids,
        )  # fmt: skip
        self.link("CONTAINS", rnode, cnode)
        tiles = [_tile_node(cx, cy, x, y) for y in range(n) for x in range(n)]
        for i, t in enumerate(tiles):
            self.add("Tile", t, cx=cx, cy=cy, x=i % n, y=i // n, tile_class=rows[i // n][i % n])
            self.link("CONTAINS", cnode, t)
        for i, t in enumerate(tiles):
            if i % n + 1 < n:
                self.link("ADJACENT", t, tiles[i + 1], dir=1)
            if i // n + 1 < n:
                self.link("ADJACENT", t, tiles[i + n], dir=2)
        for d, (dx, dy) in enumerate(DIRS):
            other = self.chunks.get((cx + dx, cy + dy))
            if other is None:
                continue
            self.link("ADJACENT", cnode, _chunk_node(other.cx, other.cy), dir=d)
            for i in range(n):
                ours = ((i, 0), (n - 1, i), (i, n - 1), (0, i))[d]
                theirs = ((i, n - 1), (0, i), (i, 0), (n - 1, i))[d]
                self.link(
                    "ADJACENT", _tile_node(cx, cy, *ours), _tile_node(other.cx, other.cy, *theirs), dir=d
                )
        self.chunks[chunk.key] = chunk

    def add_asset(
        self,
        asset: Asset,
        chunk_key: tuple[int, int],
        tile_class: str,
        coherence: float | None = None,
        anchor: bool = False,
    ) -> list[str]:
        """Make `asset` the INSTANCE_OF target of every `tile_class` tile of the chunk and fold its style vector into
        the region's EMA. The region's first asset, and any `anchor`, gets a STYLE_ANCHOR edge. Returns the ids of
        assets this replacement left without tiles; they are removed from the graph."""
        chunk = self.chunks[chunk_key]
        region = self.g.nodes[_chunk_node(*chunk_key)]["region"]
        anode = _asset_node(asset.id)
        if anode not in self.g:
            style = None if asset.style_vec is None else [float(v) for v in asset.style_vec]
            self.add(
                "Asset", anode, region=region, spec=asdict(asset.spec), nbytes=asset.nbytes,
                latency_s=asset.latency_s, coherence=coherence, anchored=False, style=style,
            )  # fmt: skip
            if style is not None:
                self._vecs[asset.id] = np.asarray(style, dtype=np.float64)
                self._update_style(region, self._vecs[asset.id])
        elif coherence is not None:
            self._set(anode, coherence=coherence)
        old = chunk.asset_ids.get(tile_class)
        for t in self._tiles_of(chunk_key, tile_class):
            if old is not None:
                self.g.remove_edges_from([(t, _asset_node(old), "INSTANCE_OF")])
            self.link("INSTANCE_OF", t, anode)
        chunk.asset_ids[tile_class] = asset.id
        dropped = self._drop_orphans([old] if old and old != asset.id else [])
        rnode = _region_node(region)
        if anchor or not self.neighbours(rnode, "STYLE_ANCHOR"):
            self.link("STYLE_ANCHOR", rnode, anode)
            self._set(anode, anchored=True)
        return dropped

    def set_state(self, chunk_key: tuple[int, int], state: ChunkState) -> None:
        self.chunks[chunk_key].state = state
        self._set(_chunk_node(*chunk_key), state=state)

    def remove_chunk(self, chunk_key: tuple[int, int]) -> list[str]:
        """Remove the chunk and its tiles; returns the ids of assets no tile uses any more (also removed)."""
        chunk = self.chunks.pop(chunk_key)
        cnode = _chunk_node(*chunk_key)
        self.g.remove_nodes_from([t.id for t in self.neighbours(cnode, "CONTAINS")])
        self.g.remove_node(cnode)
        return self._drop_orphans(set(chunk.asset_ids.values()))

    # -- reads --------------------------------------------------------------------------------------------

    def chunk(self, cx: int, cy: int) -> Chunk | None:
        return self.chunks.get((cx, cy))

    def region(self, name: str) -> Region:
        """The region record for `name`; KeyError when the graph holds no such region."""
        rnode = _region_node(name)
        if rnode not in self.g:
            raise KeyError(f"unknown region {name!r}")
        node = self.g.nodes[rnode]
        keys = ((r["cx"], r["cy"]) for r in self.neighbours(rnode, "CONTAINS") if r.kind == "Chunk")
        return Region(
            rnode, name, node["biome"], frozenset(keys),
            tileset=self._tilesets.get(name), coherence_threshold=node.get("coherence_threshold"),
        )  # fmt: skip

    def region_of(self, cx: int, cy: int) -> str | None:
        """The name of the region that CONTAINS the chunk at (cx, cy); None when the chunk is absent."""
        chunk = self.chunks.get((cx, cy))
        return None if chunk is None else self.g.nodes[_chunk_node(cx, cy)]["region"]

    def neighbour_chunks(self, cx: int, cy: int) -> dict[str, Chunk]:
        """Chunks present on each side, keyed `"N"`, `"E"`, `"S"`, `"W"` as `solve_chunk` names its neighbours."""
        found = ((DIR_NAMES[d], self.chunks.get((cx + dx, cy + dy))) for d, (dx, dy) in enumerate(DIRS))
        return {name: c for name, c in found if c is not None}

    def region_style(self, region: str) -> np.ndarray | None:
        return self._styles.get(region)

    def coherence(self, asset_id: str) -> float:
        """0.6 cos(asset, region style) + 0.4 mean cos(asset, assets of the tiles ADJACENT to this asset's tiles)."""
        anode = _asset_node(asset_id)
        tiles = {t.id for t in self.neighbours(anode, "INSTANCE_OF", direction="in")}
        return self._score(self._vecs[asset_id], self.g.nodes[anode]["region"], tiles)

    def candidate_coherence(
        self, style_vec: np.ndarray, chunk_key: tuple[int, int], tile_class: str
    ) -> float:
        """The coherence a not-yet-added asset would score as the asset of `tile_class` in the chunk."""
        region = self.g.nodes[_chunk_node(*chunk_key)]["region"]
        vec = np.asarray(style_vec, dtype=np.float64)
        return self._score(vec, region, set(self._tiles_of(chunk_key, tile_class)))

    def best_asset(self, region: str, tile_class: str | None = None) -> str | None:
        """The region's highest-coherence asset (of `tile_class` if given); None when it holds none."""
        nodes = self.g.nodes
        ranked = [
            (nodes[_asset_node(a)].get("coherence") or 0.0, a)
            for a in self._vecs
            if nodes[_asset_node(a)]["region"] == region
            and (tile_class is None or nodes[_asset_node(a)]["spec"]["subject"] == tile_class)
        ]
        return max(ranked)[1] if ranked else None

    def asset_spec(self, asset_id: str) -> AssetSpec:
        return AssetSpec(**self.g.nodes[_asset_node(asset_id)]["spec"])

    def chunk_bytes(self, chunk_key: tuple[int, int]) -> int:
        ids = set(self.chunks[chunk_key].asset_ids.values())
        return sum(int(self.g.nodes[_asset_node(a)]["nbytes"]) for a in ids)

    def count(self, kind: str) -> int:
        return len(self.nodes(kind))

    def seam_violations(self) -> int:
        """Cross-chunk adjacent pairs the tileset forbids (a corner cell may keep an earlier neighbour's rule); a
        border between chunks of different tilesets has no rule to break and is not counted."""
        bad = 0
        for (cx, cy), chunk in self.chunks.items():
            allowed, grid = chunk.layout.tileset.allowed, chunk.layout.grid
            for d in (1, 2):  # E and S: each border once
                other = self.chunks.get((cx + DIRS[d][0], cy + DIRS[d][1]))
                if other is None or other.layout.tileset.classes != chunk.layout.tileset.classes:
                    continue
                ours = grid[:, -1] if d == 1 else grid[-1, :]
                theirs = other.layout.grid[:, 0] if d == 1 else other.layout.grid[0, :]
                bad += int((~(allowed[ours, d, theirs] & allowed[theirs, opposite(d), ours])).sum())
        return bad

    def validate(self) -> list[str]:
        """Structural and semantic checks; returns the problems found, so an empty list means consistent.

        Structure is the record schema's: every node and edge is a record of a known kind with its attributes;
        every kind `EDGE_KINDS` gives exactly one incoming CONTAINS edge (its parent) or exactly one outgoing
        INSTANCE_OF edge (a tile's asset) has it; every symmetric edge has its mirror. Semantics are the layout's:
        a tile's asset depicts its tile class, ADJACENT edges join tiles (chunks) one step apart in the edge's
        direction, and every neighbouring pair inside a chunk's layout has its edge.
        """
        g, problems = self.g, self._schema_problems()
        for n, d in g.nodes(data=True):
            kind = d.get("kind")
            for name, schema in EDGE_KINDS.items():
                if kind in schema.one_in and (found := self._degree(n, name, "in")) != 1:
                    problems.append(f"{n}: expected exactly one incoming {name} edge, found {found}")
                if kind in schema.one_out and (found := self._degree(n, name, "out")) != 1:
                    problems.append(f"{n}: expected exactly one outgoing {name} edge, found {found}")
            if kind == "Tile":
                for a in self.neighbours(n, "INSTANCE_OF"):
                    if a["spec"]["subject"] != d["tile_class"]:
                        problems.append(
                            f"{n}: INSTANCE_OF {a.id} does not depict tile class {d['tile_class']}"
                        )
            if kind in ("Tile", "Chunk"):
                size = d["size"] if kind == "Chunk" else g.nodes[_chunk_node(d["cx"], d["cy"])]["size"]
                here = self._position(d, size)
                for _, v, k, e in g.out_edges(n, keys=True, data=True):
                    if k != "ADJACENT" or e.get("dir") not in range(4):
                        continue  # not a record: reported by the schema check above
                    there = self._position(g.nodes[v], size)
                    if (there[0] - here[0], there[1] - here[1]) != DIRS[e["dir"]]:
                        problems.append(
                            f"{n}: ADJACENT edge to {v} does not point one step in direction {e['dir']}"
                        )
        for u, v, k, a in g.edges(keys=True, data=True):
            schema = EDGE_KINDS.get(k)
            if schema is None or schema.mirror is None or any(name not in a for name in schema.required):
                continue  # not a record: reported by the schema check above
            if g.get_edge_data(v, u, k) != schema.mirror(a):
                problems.append(f"{u} -{k}-> {v}: missing its mirror edge")
        for (cx, cy), chunk in self.chunks.items():
            n, rows = chunk.layout.width, chunk.layout.class_rows()
            for y in range(n):
                for x in range(n):
                    t = _tile_node(cx, cy, x, y)
                    if t not in g:
                        problems.append(f"{t}: missing from the graph")
                    elif g.nodes[t]["tile_class"] != rows[y][x]:
                        problems.append(
                            f"{t}: class {g.nodes[t]['tile_class']} differs from the layout's {rows[y][x]}"
                        )
                    elif (x + 1 < n and not g.has_edge(t, _tile_node(cx, cy, x + 1, y), "ADJACENT")) or (
                        y + 1 < n and not g.has_edge(t, _tile_node(cx, cy, x, y + 1), "ADJACENT")
                    ):
                        problems.append(f"{t}: missing an ADJACENT edge the layout implies")
        return problems

    # -- persistence --------------------------------------------------------------------------------------

    def to_json(self) -> dict:
        return nx.node_link_data(self.g, edges="edges")

    @classmethod
    def from_json(cls, data: dict) -> WorldStateGraph:
        """Rebuild a graph from `to_json`'s data; ValueError naming the record when a node or edge breaks the
        schema (a file from another version, or not a world state graph at all)."""
        graph = cls.__new__(cls)
        graph.g = nx.node_link_graph(copy.deepcopy(data), directed=True, multigraph=True, edges="edges")
        graph.chunks, graph._vecs, graph._styles, graph._tilesets = {}, {}, {}, {}
        problems = graph._schema_problems()
        if problems:
            raise ValueError(f"not a world state graph: {problems[0]}")
        for r in graph.nodes("Region"):  # regions first: their chunks share the one tileset built per region
            if r.attrs.get("style") is not None:
                graph._styles[r["name"]] = np.asarray(r["style"], dtype=np.float64)
            if (ts := r.attrs.get("tileset")) is not None:
                graph._tilesets[r["name"]] = TileSet(
                    list(ts["classes"]), np.asarray(ts["allowed"], bool), np.asarray(ts["weights"])
                )
        for r in graph.nodes("Asset"):
            if r.attrs.get("style") is not None:
                graph._vecs[r.id.removeprefix("asset:")] = np.asarray(r["style"], dtype=np.float64)
        for r in graph.nodes("Chunk"):
            layout = Layout(np.asarray(r["grid"], dtype=np.int32), graph._tilesets[r["region"]])
            graph.chunks[(r["cx"], r["cy"])] = Chunk(
                r["cx"], r["cy"], r["biome"], layout, r["asset_ids"], r["state"]
            )  # the record's `asset_ids` is the node's own dict, as `add_chunk` records it: they stay one
        return graph

    # -- private ------------------------------------------------------------------------------------------

    def _record(self, node_id: str) -> Record:
        data = self.g.nodes[node_id]
        return Record(node_id, data["kind"], {k: v for k, v in data.items() if k != "kind"})

    def _set(self, node_id: str, **attrs: object) -> None:
        """Update attributes of an existing record, checked against its kind's schema like `add`."""
        node = self.g.nodes[node_id]
        problems = node_problems(node["kind"], node_id, attrs, partial=True)
        if problems:
            raise ValueError(f"{node_id}: {problems[0]}")
        node.update(attrs)

    def _degree(self, node_id: str, edge_kind: str, direction: Direction) -> int:
        edges = (
            self.g.in_edges(node_id, keys=True) if direction == "in" else self.g.out_edges(node_id, keys=True)
        )
        return sum(1 for _, _, k in edges if k == edge_kind)

    def _schema_problems(self) -> list[str]:
        """Every node and edge that is not a record of the schema, named."""
        g, problems = self.g, []
        for n, d in g.nodes(data=True):
            attrs = {k: v for k, v in d.items() if k != "kind"}
            problems += [f"{n}: {p}" for p in node_problems(d.get("kind"), n, attrs)]
        for u, v, k, a in g.edges(keys=True, data=True):
            found = edge_problems(k, g.nodes[u].get("kind"), g.nodes[v].get("kind"), a)
            problems += [f"{u} -{k}-> {v}: {p}" for p in found]
        return problems

    @staticmethod
    def _position(node: dict, size: int) -> tuple[int, int]:
        """A tile's or a chunk's place on the tile grid (chunks at their own coordinates)."""
        if node["kind"] == "Tile":
            return (node["cx"] * size + node["x"], node["cy"] * size + node["y"])
        return (node["cx"], node["cy"])

    def _set_tileset(self, name: str, tileset: TileSet) -> None:
        """Record the region's tileset once: its JSON on the node, the object in `_tilesets`; a later chunk's must
        list the same classes."""
        rnode = _region_node(name)
        recorded = self.g.nodes[rnode].get("tileset")
        if recorded is None:
            self._set(
                rnode,
                tileset={
                    "classes": list(tileset.classes),
                    "allowed": tileset.allowed.tolist(),
                    "weights": tileset.weights.tolist(),
                },
            )
            self._tilesets[name] = tileset
        elif recorded["classes"] != list(tileset.classes):
            raise ValueError(
                f"region {name!r} tileset classes {recorded['classes']} != chunk's {tileset.classes}"
            )

    def _tiles_of(self, chunk_key: tuple[int, int], tile_class: str) -> list[str]:
        chunk = self.chunks[chunk_key]
        ys, xs = np.nonzero(chunk.layout.grid == chunk.layout.tileset.index(tile_class))
        return [_tile_node(*chunk_key, int(x), int(y)) for x, y in zip(xs, ys, strict=True)]

    def _asset_of(self, tile: str) -> str | None:
        ids = (
            v.removeprefix("asset:") for _, v, k in self.g.out_edges(tile, keys=True) if k == "INSTANCE_OF"
        )
        return next((a for a in ids if a in self._vecs), None)

    def _update_style(self, region: str, vec: np.ndarray) -> None:
        current = self._styles.get(region)
        self._styles[region] = (
            vec.copy() if current is None else (1 - STYLE_ALPHA) * current + STYLE_ALPHA * vec
        )
        self._set(_region_node(region), style=self._styles[region].tolist())

    def _score(self, vec: np.ndarray, region: str, tiles: set[str]) -> float:
        """A term without data drops out and the other takes its weight; no data at all scores 1.0."""
        style = self._styles.get(region)
        near: Counter[str] = Counter()
        for t in tiles:
            for _, v, k in self.g.out_edges(t, keys=True):
                if k == "ADJACENT" and v not in tiles and (a := self._asset_of(v)) is not None:
                    near[a] += 1
        region_term = None if style is None else _cos(vec, style)
        near_term = (
            sum(n * _cos(vec, self._vecs[a]) for a, n in near.items()) / near.total() if near else None
        )
        if region_term is None:
            return 1.0 if near_term is None else near_term
        return region_term if near_term is None else 0.6 * region_term + 0.4 * near_term

    def _drop_orphans(self, asset_ids: list[str] | set[str]) -> list[str]:
        dropped = []
        for a in asset_ids:
            anode = _asset_node(a)
            if anode in self.g and not self._degree(anode, "INSTANCE_OF", "in"):
                self.g.remove_node(anode)
                self._vecs.pop(a, None)
                dropped.append(a)
        return dropped
