"""The `World` seam: chunks with assets, borders, persistence and prewarm on the CPU adapters."""

import numpy as np
import pytest

from realmweaver.assets import ProceduralGenerator
from realmweaver.layout import violations
from realmweaver.world import World


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    """The plan's tests save `tmp_world.json` into the working directory; keep that out of the repo."""
    monkeypatch.chdir(tmp_path)


def test_request_chunk_maps_every_tile_class_to_an_asset():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=1)
    c = w.request_chunk(0, 0)
    classes = {c.layout.class_at(x, y) for x in range(8) for y in range(8)}
    assert classes <= set(c.asset_ids) and violations(c.layout) == 0 and c.state == "draft"
    assert all(w.asset(i).image.shape[:2] == (512, 512) for i in c.asset_ids.values())


def test_neighbour_chunks_share_a_valid_border_and_graph_validates():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=1)
    w.request_chunk(0, 0)
    w.request_chunk(1, 0)
    assert w.graph.validate() == []
    ts = w.request_chunk(0, 0).layout.tileset
    a, b = w.request_chunk(0, 0).layout, w.request_chunk(1, 0).layout
    assert all(ts.allowed[ts.index(a.class_at(7, y)), 1, ts.index(b.class_at(0, y))] for y in range(8))


def test_save_and_load_round_trip_is_identical():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=2)
    w.request_chunk(0, 0)
    w.save("tmp_world.json")
    w2 = World.load("tmp_world.json", ProceduralGenerator())
    assert np.array_equal(w2.request_chunk(0, 0).layout.grid, w.request_chunk(0, 0).layout.grid)


def test_tick_prewarms_the_chunk_ahead_of_the_player():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=3)
    w.request_chunk(0, 0)
    for x in range(0, 40, 2):
        w.observe_player(x, 4.0)  # walking east inside chunk 0..4
    warmed = w.tick(budget=1)
    assert warmed and warmed[0][0] > 0  # a chunk to the east


# --- beyond the plan ---------------------------------------------------------------------------------


@pytest.fixture
def small_assets(monkeypatch):
    """64 px assets make a chunk cost milliseconds; the plan's tests above keep the default asset size."""
    monkeypatch.setenv("REALMWEAVER_ASSET_SIZE", "64")


def test_a_tiny_cache_evicts_far_chunks_from_graph_and_store(small_assets):
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=4, cache_bytes=1)
    w.observe_player(2.0, 2.0)
    w.request_chunk(0, 0)
    far = w.request_chunk(5, 5)  # over the cap: the chunk farthest from the player goes at once
    assert far.state == "draft" and w.graph.chunk(5, 5) is None and w.graph.chunk(0, 0) is not None
    s = w.stats()
    assert s["chunks"] == 1 and s["resident_assets"] == s["assets"] and w.graph.validate() == []


def test_refine_tier_upgrades_a_draft_chunk_in_place(small_assets):
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=4)
    draft = w.request_chunk(0, 0)
    ready = w.request_chunk(0, 0, tier="refine")
    assert ready is draft and ready.state == "ready" and w.graph.validate() == []
    assert w.request_chunk(0, 0, tier="refine") is ready and w.stats()["chunks"] == 1
    assert all(w.asset(i).image.shape == (64, 64, 3) for i in ready.asset_ids.values())


def test_assets_are_regenerated_from_their_specs_after_load(small_assets, tmp_path):
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=5)
    c = w.request_chunk(0, 0)
    w.save(tmp_path / "world.json")
    w2 = World.load(tmp_path / "world.json", ProceduralGenerator())
    assert w2.stats()["resident_assets"] == 0 and w2.stats()["chunks"] == 1 and w2.graph.validate() == []
    asset_id = next(iter(c.asset_ids.values()))
    assert np.array_equal(w2.asset(asset_id).image, w.asset(asset_id).image)
    assert w2.stats()["resident_assets"] == 1 and w2.request_chunk(0, 0).asset_ids == c.asset_ids


def test_unknown_asset_or_biome_raises_key_error(small_assets):
    w = World("forest", ProceduralGenerator(), chunk_size=8)
    with pytest.raises(KeyError):
        w.asset("no-such-asset")
    with pytest.raises(KeyError):
        World("atlantis", ProceduralGenerator())


def test_same_seed_reproduces_the_world_and_another_seed_differs(small_assets):
    a = World("forest", ProceduralGenerator(), chunk_size=8, seed=11).request_chunk(0, 0)
    b = World("forest", ProceduralGenerator(), chunk_size=8, seed=11).request_chunk(0, 0)
    c = World("forest", ProceduralGenerator(), chunk_size=8, seed=12).request_chunk(0, 0)
    assert np.array_equal(a.layout.grid, b.layout.grid) and a.asset_ids == b.asset_ids
    assert not np.array_equal(a.layout.grid, c.layout.grid) and a.asset_ids != c.asset_ids


def test_observe_player_counts_prewarm_hits_and_misses(small_assets):
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=6)
    w.request_chunk(0, 0)
    w.observe_player(1.0, 1.0)  # entering a chunk that is ready
    w.observe_player(3.0, 1.0)  # still inside it: not a new entry
    w.observe_player(20.0, 1.0)  # chunk (2, 0) does not exist yet
    assert w.stats()["prewarm_hits"] == 1 and w.stats()["prewarm_misses"] == 1


def test_stats_reports_counts_bytes_and_latencies(small_assets):
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=7)
    assert w.tick() == [] and w.stats()["chunk_latency_s"] == {"n": 0}
    c = w.request_chunk(0, 0)
    s = w.stats()
    keys = {"chunks", "assets", "cache_bytes", "prewarm_hits", "prewarm_misses", "regenerations", "fallbacks"}
    assert keys <= set(s) and s["chunks"] == 1 and s["seam_violations"] == 0
    assert s["cache_bytes"] == sum(w.asset(i).nbytes for i in set(c.asset_ids.values()))
    assert s["chunk_latency_s"]["n"] == 1 and s["chunk_latency_s"]["p50"] > 0


def test_tick_generates_at_most_budget_chunks_and_the_heading_first(small_assets):
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=8)
    for x in range(7):
        w.observe_player(float(x), 4.0)  # walking east inside chunk (0, 0)
    warmed = w.tick(budget=3)
    assert len(warmed) == 3 and warmed[0] == (1, 0) and all(w.graph.chunk(*k) is not None for k in warmed)
    assert w.tick(budget=0) == [] and w.stats()["chunks"] == 3 and w.graph.validate() == []


def test_chunk_solved_against_four_neighbours_is_the_same_object_on_every_request(small_assets):
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=9)
    for key in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        w.request_chunk(*key)
    mid = w.request_chunk(0, 0)
    assert mid is w.request_chunk(0, 0) and set(w.graph.neighbours(0, 0)) == {"N", "E", "S", "W"}
    # diagonal neighbours are never constrained against each other, so a corner may keep one mismatch
    assert w.stats()["seam_violations"] <= 4 and w.graph.validate() == []
