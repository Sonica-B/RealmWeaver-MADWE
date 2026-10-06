"""Terrain post-pass on the CPU noise heightmap: rivers flow downhill, roads connect every settlement, biome shares sum
to one and follow temperature, sites are flat and near water, villages solve, and a region writes into the graph."""

from itertools import combinations

import numpy as np
import pytest

from realmweaver.biomes import biome_names
from realmweaver.layout import violations
from realmweaver.terrain import (
    BIOMES,
    D8,
    DiffusionHeightmap,
    NoiseHeightmap,
    biome_shares,
    biomes,
    landmarks,
    rivers,
    roads,
    settlement_layout,
    settlement_sites,
    write_region,
)
from realmweaver.world import WorldStateGraph

SIZE = 96
THRESHOLD = 60.0
MAX_SLOPE, MAX_WATER, SEPARATION = 0.08, 5, 12


@pytest.fixture(scope="module")
def terrain():
    return NoiseHeightmap(seed=7).sample(0, 0, SIZE, SIZE)


@pytest.fixture(scope="module")
def flow(terrain):
    return rivers(terrain.height, rain=terrain.moisture, threshold=THRESHOLD)


@pytest.fixture(scope="module")
def sites(terrain, flow):
    return settlement_sites(
        terrain.height, flow, n=4, cell_m=terrain.cell_m, max_slope=MAX_SLOPE, max_water_cells=MAX_WATER,
        min_separation=SEPARATION,
    )  # fmt: skip


def test_noise_heightmap_is_seed_consistent_with_random_access():
    crop = NoiseHeightmap(seed=3).sample(10, 20, 32, 24)
    whole = NoiseHeightmap(seed=3).sample(0, 0, 64, 64)
    assert crop.shape == (24, 32) and crop.height.dtype == np.float32
    assert np.array_equal(crop.height, whole.height[20:44, 10:42])
    assert np.array_equal(crop.moisture, whole.moisture[20:44, 10:42])
    assert not np.array_equal(crop.height, NoiseHeightmap(seed=4).sample(10, 20, 32, 24).height)


def test_rivers_flow_downhill(terrain, flow):
    ys, xs = np.nonzero(flow.direction >= 0)
    dx, dy = np.array(D8)[flow.direction[ys, xs]].T
    assert (terrain.height[ys + dy, xs + dx] <= terrain.height[ys, xs]).all()
    assert flow.mask.any() and flow.accumulation[flow.mask].min() >= THRESHOLD
    assert (terrain.height[flow.mask] > 0).all() and (flow.direction[terrain.height <= 0] == -1).all()


def test_roads_connect_all_settlements(terrain, flow, sites):
    assert len(sites) >= 3
    net = roads(sites, terrain.height, terrain.cell_m, rivers=flow)
    parent = list(range(len(sites)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for road in net.roads:
        assert tuple(road.cells[0]) == (sites[road.a].x, sites[road.a].y)
        assert tuple(road.cells[-1]) == (sites[road.b].x, sites[road.b].y)
        assert (np.abs(np.diff(road.cells, axis=0)).max(axis=1) == 1).all()  # 8-connected, no repeats
        assert (terrain.height[road.cells[:, 1], road.cells[:, 0]] > 0).all() and np.isfinite(road.cost)
        parent[find(road.a)] = find(road.b)
    assert len({find(i) for i in range(len(sites))}) == 1
    assert net.mask(terrain.shape).sum() >= len(sites)


def test_biome_shares_sum_to_one_and_follow_temperature(terrain):
    assert set(BIOMES) == set(biome_names())
    warm = biome_shares(biomes(terrain.height, terrain.temperature, terrain.moisture))
    cold = biome_shares(biomes(terrain.height, terrain.temperature - 25.0, terrain.moisture))
    assert sum(warm.values()) == pytest.approx(1.0) and sum(cold.values()) == pytest.approx(1.0)
    assert cold["snow"] > warm["snow"] and cold["forest"] < warm["forest"]
    assert warm["underwater"] == pytest.approx(float((terrain.height <= 0).mean()))


def test_settlement_sites_are_flat_and_near_water(sites):
    assert all(s.slope <= MAX_SLOPE and s.water_cells <= MAX_WATER and s.height_m > 0 for s in sites)
    assert all(max(abs(a.x - b.x), abs(a.y - b.y)) > SEPARATION for a, b in combinations(sites, 2))


def test_settlement_layout_is_a_village(sites):
    layout = settlement_layout(sites[0], size=12, seed=1)
    assert layout.grid.shape == (12, 12) and violations(layout) == 0 and layout.class_at(6, 6) == "plaza"
    assert {"road", "house"} <= {c for row in layout.class_rows() for c in row}
    assert np.array_equal(layout.grid, settlement_layout(sites[0], size=12, seed=1).grid)


def test_diffusion_heightmap_names_its_missing_server():
    with pytest.raises(RuntimeError, match="terrain-diffusion API"):
        DiffusionHeightmap("http://127.0.0.1:9", seed=1, timeout_s=0.3).sample(0, 0, 4, 4)


def test_write_region_adds_records_and_keeps_the_graph_valid(terrain, flow, sites):
    class_map = biomes(terrain.height, terrain.temperature, terrain.moisture)
    net = roads(sites, terrain.height, terrain.cell_m, rivers=flow)
    marks = landmarks(terrain.height, flow)
    shares = biome_shares(class_map)
    g = WorldStateGraph(seed=0)
    node = write_region(
        g, "spike", class_map, flow, net, sites,
        layouts=[settlement_layout(s, seed=1) for s in sites], landmarks=marks,
        provenance={"source": "NoiseHeightmap", "seed": 7},
    )  # fmt: skip
    assert node == "region3d:spike" and g.count("Region3D") == 1
    assert g.count("Region") == sum(1 for v in shares.values() if v > 0)
    assert g.count("Settlement") == len(sites) and g.count("Landmark") == len(marks) == 2
    assert g.validate() == []
    along_roads = [
        (u, v) for u, v, k in g.g.edges(keys=True) if k == "ADJACENT" and u.startswith("settlement:")
    ]
    assert len(along_roads) == 2 * len(net.roads)
    again = WorldStateGraph.from_json(g.to_json())
    top = max(shares, key=shares.get)
    assert again.count("Settlement") == len(sites) and again.region(f"spike/{top}").biome == top
