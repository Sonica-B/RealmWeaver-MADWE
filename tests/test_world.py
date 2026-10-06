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
    """64 px assets keep these chunks small; the plan's tests above keep the default asset size."""
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


# --- story 12: drift, regeneration with the next seed, anchor fallback -----------------------------------

_DIM = 128  # style vectors below are unit vectors: e0 is the region's first asset, every other call gets its own axis


class _Attempts:
    """`Generator` adapter over the procedural one that groups the specs it sees into runs: the world retries a
    tile class with the next seed, so a spec whose seed follows the previous one's for the same subject is a
    retry, and `attempt` is the index of the latest spec within its run."""

    def __init__(self) -> None:
        self.inner, self.runs = ProceduralGenerator(), []

    def generate(self, spec):
        last = self.runs[-1][-1] if self.runs else None
        if last is not None and last.subject == spec.subject and spec.seed == last.seed + 1:
            self.runs[-1].append(spec)
        else:
            self.runs.append([spec])
        return self.inner.generate(spec)

    @property
    def attempt(self) -> int:
        return len(self.runs[-1]) - 1


def _embed_by_attempt(gen: _Attempts, cosines: tuple[float, ...]):
    """Style vectors with a chosen cosine to the first asset's vector e0: the first asset embeds to e0, every later
    one to `cosines[attempt]` times e0 plus its own orthogonal axis, so its coherence is at most that cosine."""
    calls = 0

    def embed(_image: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        c = 1.0 if calls == 1 else cosines[min(gen.attempt, len(cosines) - 1)]
        vec = np.zeros(_DIM)
        vec[0], vec[calls] = c, np.sqrt(1.0 - c * c)
        return vec

    return embed


def test_below_threshold_asset_is_regenerated_with_the_next_seed_until_it_passes(small_assets):
    gen = _Attempts()
    w = World("forest", gen, _embed_by_attempt(gen, cosines=(0.0, 1.0)), chunk_size=8, seed=21)
    c = w.request_chunk(0, 0)
    present = {c.layout.class_at(x, y) for x in range(8) for y in range(8)}
    s = w.stats()
    assert s["regenerations"] == len(present) - 1 and s["fallbacks"] == 0  # the first asset sets the style
    assert present <= set(c.asset_ids) and w.graph.validate() == []
    assert [len(run) for run in gen.runs] == [1] + [2] * (len(present) - 1)
    for first, second in gen.runs[
        1:
    ]:  # the orthogonal first try fails; the next seed's asset is the one kept
        assert second.seed == first.seed + 1 and c.asset_ids[first.subject] == second.id
    assert all(w.graph.coherence(a) >= w.threshold for a in c.asset_ids.values())


def test_when_every_attempt_fails_the_best_candidate_is_anchored_and_later_chunks_reuse_the_regions_best(
    small_assets,
):
    gen = _Attempts()
    w = World("forest", gen, _embed_by_attempt(gen, cosines=(0.1, 0.3, 0.2)), chunk_size=8, seed=22)
    first = w.request_chunk(0, 0)
    present = {first.layout.class_at(x, y) for x in range(8) for y in range(8)}
    s = w.stats()
    assert s["fallbacks"] == len(present) - 1 and s["regenerations"] == 2 * (len(present) - 1)
    assert present <= set(first.asset_ids) and w.graph.validate() == []
    for run in gen.runs[
        1:
    ]:  # three tries and no asset of the class in the region yet: the best try (0.3) stays
        assert len(run) == 3 and first.asset_ids[run[0].subject] == run[1].id
    second = w.request_chunk(1, 0)
    shared = set(second.asset_ids) & set(first.asset_ids)
    assert shared and w.graph.validate() == []
    for cls in shared:  # every try fails again, so the class is anchored to the region's best asset of it
        assert second.asset_ids[cls] == first.asset_ids[cls] == w.graph.best_asset("forest", cls)
    assert w.stats()["fallbacks"] == len(present) - 1 + len(second.asset_ids)


def test_tick_refines_the_nearest_draft_chunks_once_nothing_is_left_to_prewarm(small_assets):
    # one constant style vector keeps every candidate coherent, so only the refine pass itself is under test
    w = World("forest", ProceduralGenerator(), embed=lambda _image: np.ones(4), chunk_size=8, seed=13)
    w.request_chunk(0, 0)
    w.observe_player(4.0, 4.0)  # in (0, 0) with no heading yet: every neighbour is a prewarm candidate
    warmed = w.tick(budget=1)  # draft first: a neighbour is generated, nothing refined
    assert warmed and w.stats()["refined"] == 0
    for key in [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]:
        w.request_chunk(*key)
    refined = w.tick(budget=2)
    assert refined[0] == (0, 0) and len(refined) == 2 and w.stats()["refined"] == 2
    here = w.graph.chunk(0, 0)
    specs = [w.graph.asset_spec(a) for a in here.asset_ids.values()]
    assert here.state == "ready" and all(s.tier == "refine" and s.steps == 8 for s in specs)
    assert w.graph.chunk(*refined[1]).state == "ready" and len(w.tick(budget=10)) == 7 and w.tick() == []
    assert w.stats()["refined"] == 9 and w.stats()["chunks"] == 9 and w.graph.validate() == []
    assert all(c.state == "ready" for c in w.graph.chunks.values())
