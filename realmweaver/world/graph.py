"""World state graph: the typed property graph that is a world's source of truth (ADR-0004).

Nodes are World, Region, Chunk, Tile and Asset (attribute `kind`); edges are CONTAINS, ADJACENT, INSTANCE_OF and
STYLE_ANCHOR, typed by their MultiDiGraph key. Node ids are strings and every attribute is plain JSON, so the
graph round-trips through node-link JSON. Style vectors are whatever the injected embedder returned.
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

log = logging.getLogger(__name__)

STYLE_ALPHA = 0.2  # EMA weight of each new asset in its region's style vector
WORLD = "world"
_PARENT_KIND = {"Region": "World", "Chunk": "Region", "Tile": "Chunk"}
_EPS = 1e-9
ChunkState = Literal["pending", "draft", "ready"]


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


def _adjacent(u: str, v: str, d: int) -> list[tuple[str, str, str, dict]]:
    """ADJACENT is symmetric: one edge each way, each carrying the direction from its source to its target."""
    return [(u, v, "ADJACENT", {"dir": d}), (v, u, "ADJACENT", {"dir": opposite(d)})]


class WorldStateGraph:
    """`networkx.MultiDiGraph` in `g`, the live `Chunk` objects in `chunks`, plus an index of asset style vectors."""

    def __init__(self, seed: int = 0, chunk_size: int = 16) -> None:
        self.g = nx.MultiDiGraph()
        self.g.add_node(WORLD, kind="World", seed=seed, chunk_size=chunk_size)
        self.chunks: dict[tuple[int, int], Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}  # asset id -> style vector
        self._styles: dict[str, np.ndarray] = {}  # region name -> EMA style vector
        self._tilesets: dict[str, TileSet] = {}  # region name -> the compiled tileset its node holds as JSON

    # -- writes -------------------------------------------------------------------------------------------

    def add_region(
        self, name: str, biome: str, tileset: TileSet | None = None, coherence_threshold: float | None = None
    ) -> None:
        """A region of `biome`; `tileset` (else the first chunk's) and `coherence_threshold` go on its record."""
        self.g.add_node(
            _region_node(name), kind="Region", name=name, biome=biome, style=None, tileset=None,
            coherence_threshold=coherence_threshold,
        )  # fmt: skip
        self.g.add_edge(WORLD, _region_node(name), key="CONTAINS")
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
        self.g.add_node(
            cnode, kind="Chunk", cx=cx, cy=cy, biome=chunk.biome, size=n, state=chunk.state, region=region,
            grid=chunk.layout.grid.tolist(), asset_ids=chunk.asset_ids,
        )  # fmt: skip
        self.g.add_edge(rnode, cnode, key="CONTAINS")
        tiles = [_tile_node(cx, cy, x, y) for y in range(n) for x in range(n)]
        self.g.add_nodes_from(
            (t, dict(kind="Tile", cx=cx, cy=cy, x=i % n, y=i // n, tile_class=rows[i // n][i % n]))
            for i, t in enumerate(tiles)
        )
        edges: list[tuple[str, str, str, dict]] = [(cnode, t, "CONTAINS", {}) for t in tiles]
        for i, t in enumerate(tiles):
            x, y = i % n, i // n
            if x + 1 < n:
                edges += _adjacent(t, tiles[i + 1], 1)
            if y + 1 < n:
                edges += _adjacent(t, tiles[i + n], 2)
        for d, (dx, dy) in enumerate(DIRS):
            other = self.chunks.get((cx + dx, cy + dy))
            if other is None:
                continue
            edges += _adjacent(cnode, _chunk_node(other.cx, other.cy), d)
            for i in range(n):
                ours = ((i, 0), (n - 1, i), (i, n - 1), (0, i))[d]
                theirs = ((i, n - 1), (0, i), (i, 0), (n - 1, i))[d]
                edges += _adjacent(_tile_node(cx, cy, *ours), _tile_node(other.cx, other.cy, *theirs), d)
        self.g.add_edges_from(edges)
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
            self.g.add_node(
                anode, kind="Asset", region=region, spec=asdict(asset.spec), nbytes=asset.nbytes,
                latency_s=asset.latency_s, coherence=coherence, anchored=False, style=style,
            )  # fmt: skip
            if style is not None:
                self._vecs[asset.id] = np.asarray(style, dtype=np.float64)
                self._update_style(region, self._vecs[asset.id])
        elif coherence is not None:
            self.g.nodes[anode]["coherence"] = coherence
        old = chunk.asset_ids.get(tile_class)
        for t in self._tiles_of(chunk_key, tile_class):
            if old is not None:
                self.g.remove_edges_from([(t, _asset_node(old), "INSTANCE_OF")])
            self.g.add_edge(t, anode, key="INSTANCE_OF")
        chunk.asset_ids[tile_class] = asset.id
        dropped = self._drop_orphans([old] if old and old != asset.id else [])
        rnode = _region_node(region)
        if anchor or not any(k == "STYLE_ANCHOR" for _, _, k in self.g.out_edges(rnode, keys=True)):
            self.g.add_edge(rnode, anode, key="STYLE_ANCHOR")
            self.g.nodes[anode]["anchored"] = True
        return dropped

    def set_state(self, chunk_key: tuple[int, int], state: ChunkState) -> None:
        self.chunks[chunk_key].state = state
        self.g.nodes[_chunk_node(*chunk_key)]["state"] = state

    def remove_chunk(self, chunk_key: tuple[int, int]) -> list[str]:
        """Remove the chunk and its tiles; returns the ids of assets no tile uses any more (also removed)."""
        chunk = self.chunks.pop(chunk_key)
        cnode = _chunk_node(*chunk_key)
        self.g.remove_nodes_from([v for _, v, k in self.g.out_edges(cnode, keys=True) if k == "CONTAINS"])
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
        nodes = self.g.nodes
        keys = (
            (nodes[v]["cx"], nodes[v]["cy"])
            for _, v, k in self.g.out_edges(rnode, keys=True)
            if k == "CONTAINS"
        )
        return Region(
            rnode, name, nodes[rnode]["biome"], frozenset(keys),
            tileset=self._tilesets.get(name), coherence_threshold=nodes[rnode].get("coherence_threshold"),
        )  # fmt: skip

    def region_of(self, cx: int, cy: int) -> str | None:
        """The name of the region that CONTAINS the chunk at (cx, cy); None when the chunk is absent."""
        chunk = self.chunks.get((cx, cy))
        return None if chunk is None else self.g.nodes[_chunk_node(cx, cy)]["region"]

    def neighbours(self, cx: int, cy: int) -> dict[str, Chunk]:
        """Chunks present on each side, keyed `"N"`, `"E"`, `"S"`, `"W"` as `solve_chunk` names its neighbours."""
        found = ((DIR_NAMES[d], self.chunks.get((cx + dx, cy + dy))) for d, (dx, dy) in enumerate(DIRS))
        return {name: c for name, c in found if c is not None}

    def region_style(self, region: str) -> np.ndarray | None:
        return self._styles.get(region)

    def coherence(self, asset_id: str) -> float:
        """0.6 cos(asset, region style) + 0.4 mean cos(asset, assets of the tiles ADJACENT to this asset's tiles)."""
        anode = _asset_node(asset_id)
        tiles = {u for u, _, k in self.g.in_edges(anode, keys=True) if k == "INSTANCE_OF"}
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
            (nodes[_asset_node(a)]["coherence"] or 0.0, a)
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
        return sum(1 for _, d in self.g.nodes(data=True) if d.get("kind") == kind)

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

        Every Region, Chunk and Tile has exactly one CONTAINS parent of the right kind; every Tile has exactly one
        INSTANCE_OF edge, to an Asset of its tile class; ADJACENT edges join tiles (chunks) one step apart in the
        edge's direction, and every neighbouring pair inside a chunk's layout has its edge.
        """
        g, problems = self.g, []
        for n, d in g.nodes(data=True):
            kind = d.get("kind")
            if kind in _PARENT_KIND:
                parents = [u for u, _, k in g.in_edges(n, keys=True) if k == "CONTAINS"]
                if len(parents) != 1 or g.nodes[parents[0]].get("kind") != _PARENT_KIND[kind]:
                    problems.append(
                        f"{n}: expected one CONTAINS parent of kind {_PARENT_KIND[kind]}, found {parents}"
                    )
            if kind == "Tile":
                assets = [v for _, v, k in g.out_edges(n, keys=True) if k == "INSTANCE_OF"]
                if len(assets) != 1:
                    problems.append(f"{n}: expected exactly one INSTANCE_OF edge, found {len(assets)}")
                elif g.nodes[assets[0]]["spec"]["subject"] != d["tile_class"]:
                    problems.append(
                        f"{n}: INSTANCE_OF {assets[0]} does not depict tile class {d['tile_class']}"
                    )
            if kind in ("Tile", "Chunk"):
                size = d["size"] if kind == "Chunk" else g.nodes[_chunk_node(d["cx"], d["cy"])]["size"]
                here = (
                    (d["cx"] * size + d["x"], d["cy"] * size + d["y"])
                    if kind == "Tile"
                    else (d["cx"], d["cy"])
                )
                for _, v, k, e in g.out_edges(n, keys=True, data=True):
                    if k != "ADJACENT":
                        continue
                    o = g.nodes[v]
                    there = (
                        (o["cx"] * size + o["x"], o["cy"] * size + o["y"])
                        if kind == "Tile"
                        else (o["cx"], o["cy"])
                    )
                    if (there[0] - here[0], there[1] - here[1]) != DIRS[e["dir"]]:
                        problems.append(
                            f"{n}: ADJACENT edge to {v} does not point one step in direction {e['dir']}"
                        )
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
        graph = cls.__new__(cls)
        graph.g = nx.node_link_graph(copy.deepcopy(data), directed=True, multigraph=True, edges="edges")
        graph.chunks, graph._vecs, graph._styles, graph._tilesets = {}, {}, {}, {}
        nodes = graph.g.nodes
        for n, d in nodes(data=True):  # regions first: their chunks share the one tileset built per region
            kind = d.get("kind")
            if kind == "Region":
                if d["style"] is not None:
                    graph._styles[d["name"]] = np.asarray(d["style"], dtype=np.float64)
                if d["tileset"] is not None:
                    ts = d["tileset"]
                    graph._tilesets[d["name"]] = TileSet(
                        list(ts["classes"]), np.asarray(ts["allowed"], bool), np.asarray(ts["weights"])
                    )
            elif kind == "Asset" and d["style"] is not None:
                graph._vecs[n.removeprefix("asset:")] = np.asarray(d["style"], dtype=np.float64)
        for d in (d for _, d in nodes(data=True) if d.get("kind") == "Chunk"):
            layout = Layout(np.asarray(d["grid"], dtype=np.int32), graph._tilesets[d["region"]])
            graph.chunks[(d["cx"], d["cy"])] = Chunk(
                d["cx"], d["cy"], d["biome"], layout, d["asset_ids"], d["state"]
            )
        return graph

    # -- private ------------------------------------------------------------------------------------------

    def _set_tileset(self, name: str, tileset: TileSet) -> None:
        """Record the region's tileset once: its JSON on the node, the object in `_tilesets`; a later chunk's must
        list the same classes."""
        node = self.g.nodes[_region_node(name)]
        if node["tileset"] is None:
            node["tileset"] = {
                "classes": list(tileset.classes),
                "allowed": tileset.allowed.tolist(),
                "weights": tileset.weights.tolist(),
            }
            self._tilesets[name] = tileset
        elif node["tileset"]["classes"] != list(tileset.classes):
            raise ValueError(
                f"region {name!r} tileset classes {node['tileset']['classes']} != chunk's {tileset.classes}"
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
        self.g.nodes[_region_node(region)]["style"] = self._styles[region].tolist()

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
            if anode in self.g and not any(
                k == "INSTANCE_OF" for _, _, k in self.g.in_edges(anode, keys=True)
            ):
                self.g.remove_node(anode)
                self._vecs.pop(a, None)
                dropped.append(a)
        return dropped
