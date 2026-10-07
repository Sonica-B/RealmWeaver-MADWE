"""Writes a terrain region into the world state graph (ADR-0004): one Region3D record for the window, one Region per
biome present (ADJACENT where their patches touch), and Settlement and Landmark records the Region3D CONTAINS, with
settlements ADJACENT along roads. The masks themselves stay out of the graph (they are file assets); the records
carry shares, counts and the village layouts. Every record goes in through `WorldStateGraph.add` and `link`, which
check it against the record schema.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np

from realmweaver.terrain.postpass import BIOMES, Landmark, Rivers, Roads, Site, biome_shares
from realmweaver.types import Layout
from realmweaver.world import WORLD, WorldStateGraph

log = logging.getLogger(__name__)


def write_region(
    graph: WorldStateGraph,
    region_name: str,
    biome_map: np.ndarray,
    rivers: Rivers,
    roads: Roads,
    settlements: Sequence[Site],
    layouts: Sequence[Layout] | None = None,
    landmarks: Sequence[Landmark] = (),
    cell_m: float = 30.0,
    provenance: dict | None = None,
) -> str:
    """Add the region's records and edges; returns the Region3D node id. `layouts` pairs with `settlements` and
    `provenance` (model, licence, seed) is stored as given."""
    shares = biome_shares(biome_map)
    present = [name for name in BIOMES if shares[name] > 0]
    rnode = f"region3d:{region_name}"
    graph.add(
        "Region3D", rnode,
        name=region_name, cells=[int(biome_map.shape[0]), int(biome_map.shape[1])], cell_m=float(cell_m),
        biome_shares=shares, river_cells=int(rivers.mask.sum()), road_cells=int(roads.mask(biome_map.shape).sum()),
        provenance=provenance,
    )  # fmt: skip
    graph.link("CONTAINS", WORLD, rnode)
    # ponytail: one Region per biome present rather than per contiguous patch (the glossary's wording); labelling
    # connected components with a minimum area is the upgrade path.
    region_ids = {}
    for name in present:
        graph.add_region(f"{region_name}/{name}", name)
        region_ids[name] = graph.region(f"{region_name}/{name}").id
    centroids = {name: _centroid(biome_map == BIOMES.index(name)) for name in present}
    for a, b in _touching(biome_map):
        a_name, b_name = BIOMES[a], BIOMES[b]
        bearing = _bearing(centroids[a_name], centroids[b_name])
        graph.link("ADJACENT", region_ids[a_name], region_ids[b_name], dir=bearing)
    settlement_ids = []
    for i, site in enumerate(settlements):
        layout = (
            None
            if layouts is None
            else {"classes": list(layouts[i].tileset.classes), "grid": layouts[i].grid.tolist()}
        )
        sid = f"settlement:{region_name}/{i}"
        graph.add(
            "Settlement", sid,
            name=f"{region_name}/settlement-{i}", x=int(site.x), y=int(site.y), cell_m=float(cell_m),
            height_m=float(site.height_m), slope=float(site.slope), water_cells=int(site.water_cells),
            biome=BIOMES[int(biome_map[site.y, site.x])], layout=layout,
        )  # fmt: skip
        graph.link("CONTAINS", rnode, sid)
        settlement_ids.append(sid)
    for road in roads.roads:
        a, b = settlements[road.a], settlements[road.b]
        cost = float(road.cost) if math.isfinite(road.cost) else None
        graph.link(
            "ADJACENT", settlement_ids[road.a], settlement_ids[road.b],
            dir=_bearing((a.x, a.y), (b.x, b.y)), road_cells=int(len(road.cells)), cost=cost,
        )  # fmt: skip
    for i, mark in enumerate(landmarks):
        lid = f"landmark:{region_name}/{i}"
        graph.add(
            "Landmark", lid,
            name=f"{region_name}/{mark.kind}", category=mark.kind, x=int(mark.x), y=int(mark.y),
            height_m=float(mark.height_m), biome=BIOMES[int(biome_map[mark.y, mark.x])],
        )  # fmt: skip
        graph.link("CONTAINS", rnode, lid)
    log.info(
        "write_region %s: %d biomes, %d settlements, %d roads, %d landmarks",
        region_name, len(present), len(settlements), len(roads.roads), len(landmarks),
    )  # fmt: skip
    return rnode


def _touching(class_map: np.ndarray) -> list[tuple[int, int]]:
    """Unordered pairs of distinct classes that share a 4-neighbour edge somewhere in the map."""
    pairs = []
    for a, b in ((class_map[:, :-1], class_map[:, 1:]), (class_map[:-1, :], class_map[1:, :])):
        differ = a != b
        pairs.append(np.stack([np.minimum(a, b)[differ], np.maximum(a, b)[differ]], axis=1))
    unique = np.unique(np.concatenate(pairs), axis=0) if pairs else np.zeros((0, 2), dtype=int)
    return [(int(a), int(b)) for a, b in unique]


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _bearing(p: tuple[float, float], q: tuple[float, float]) -> int:
    """The DIRS index (N, E, S, W) of the dominant axis from `p` to `q`; east when they coincide."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    if abs(dx) >= abs(dy):
        return 1 if dx >= 0 else 3
    return 2 if dy > 0 else 0
