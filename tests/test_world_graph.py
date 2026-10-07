"""World state graph: the record schema behind every write, typed nodes and edges, region style, coherence,
validation and JSON round trip."""

import numpy as np
import pytest

from realmweaver.assets import ProceduralGenerator
from realmweaver.biomes import load_biome
from realmweaver.layout import solve_chunk, tileset_from_example
from realmweaver.metrics import histogram_embed
from realmweaver.types import AssetSpec, Chunk, Layout
from realmweaver.world import EDGE_KINDS, NODE_KINDS, WorldStateGraph

SIZE = 4


def _chunk(cx, cy, seed, neighbours=None):
    b = load_biome("forest")
    ts = tileset_from_example(b.example_map, b.legend)
    return Chunk(cx, cy, "forest", solve_chunk(ts, SIZE, seed=seed, neighbours=neighbours or {}))


def _asset(subject, seed=0):
    a = ProceduralGenerator().generate(AssetSpec("forest", subject=subject, size=32, seed=seed))
    a.style_vec = histogram_embed(a.image)
    return a


def _fill(g, chunk):
    """One asset per class present; the same class gets the same spec, so chunks share asset ids."""
    assets = {}
    for cls in sorted({chunk.layout.class_at(x, y) for x in range(SIZE) for y in range(SIZE)}):
        assets[cls] = _asset(cls)
        g.add_asset(assets[cls], chunk.key, cls)
    return assets


def _filled_graph():
    g = WorldStateGraph(seed=0, chunk_size=SIZE)
    g.add_region("forest", "forest")
    c = _chunk(0, 0, seed=1)
    g.add_chunk(c, "forest")
    return g, c, _fill(g, c)


def test_validate_catches_a_tile_without_an_asset():
    g = WorldStateGraph(seed=0, chunk_size=SIZE)
    g.add_region("forest", "forest")
    g.add_chunk(_chunk(0, 0, seed=1), "forest")
    problems = g.validate()
    assert problems and all("INSTANCE_OF" in p for p in problems)
    g2, _, _ = _filled_graph()
    assert g2.validate() == []


def test_coherence_is_a_cosine_blend_in_minus_one_to_one():
    g, c, _ = _filled_graph()
    for asset_id in c.asset_ids.values():
        assert -1.0 <= g.coherence(asset_id) <= 1.0
    style = g.region_style("forest")
    assert style is not None and style.shape == (48,)


def _game_records(g):
    """One record of every kind the chunk pipeline does not write, joined by every edge kind, as the game will
    write them: a 3D region with a settlement and a landmark, an NPC with a memory, a faction, a quest with an
    objective, an event and a save."""
    g.add("Region3D", "region3d:vale", name="vale", cells=[8, 8], cell_m=30.0, biome_shares={"forest": 1.0})
    g.link("CONTAINS", "world", "region3d:vale")
    g.add("Settlement", "settlement:emberfall", name="Emberfall", x=3, y=4, biome="forest")
    g.link("CONTAINS", "region3d:vale", "settlement:emberfall")
    g.add("Landmark", "landmark:tower", name="Grey Tower", category="tower", x=6, y=1)
    g.link("CONTAINS", "region3d:vale", "landmark:tower")
    g.add("NPC", "npc:mara", name="Mara Vell", role="innkeeper")
    g.link("CONTAINS", "settlement:emberfall", "npc:mara")
    g.add(
        "Memory", "memory:mara/0", text="Rats gnawed three grain sacks", t=6.0, importance=0.5, source="world"
    )
    g.link("CONTAINS", "npc:mara", "memory:mara/0")
    g.add("Faction", "faction:hearthguild", name="Hearthguild")
    g.link("CONTAINS", "world", "faction:hearthguild")
    g.add("Quest", "quest:rats", name="Rats in the Cellar", description="clear the cellar", state="offered")
    g.link("CONTAINS", "world", "quest:rats")
    g.add("Objective", "objective:rats/0", name="Kill five rats")
    g.link("CONTAINS", "quest:rats", "objective:rats/0")
    g.add("Event", "event:fair", name="Harvest fair", t=216.0)
    g.link("CONTAINS", "world", "event:fair")
    g.add("Save", "save:auto-0", name="autosave 0", t=18.5)
    g.link("CONTAINS", "world", "save:auto-0")
    g.link("KNOWS", "npc:mara", "quest:rats")
    g.link("KNOWS", "npc:mara", "landmark:tower")
    g.link("ASSIGNED", "quest:rats", "npc:mara")
    g.link("ASSIGNED", "objective:rats/0", "settlement:emberfall")
    g.link("MEMBER_OF", "npc:mara", "faction:hearthguild")
    g.link("TRIGGERS", "event:fair", "quest:rats")


def test_add_and_link_reject_what_the_record_schema_forbids():
    g = WorldStateGraph(seed=0, chunk_size=SIZE)
    g.add_region("forest", "forest")
    with pytest.raises(ValueError, match="Dragon"):
        g.add("Dragon", "dragon:smaug", name="Smaug")
    with pytest.raises(ValueError, match="'name'"):
        g.add("NPC", "npc:pell", role="stable boy")
    with pytest.raises(ValueError, match="'x'"):
        g.add("Settlement", "settlement:x", name="x", x="1", y=2, biome="forest")
    with pytest.raises(ValueError, match="'colour'"):
        g.add("NPC", "npc:pell", name="Pell", colour="red")
    with pytest.raises(ValueError, match="'npc:'"):
        g.add("NPC", "person:pell", name="Pell")
    g.add("NPC", "npc:pell", name="Pell")
    with pytest.raises(ValueError, match="npc:pell"):
        g.add("NPC", "npc:pell", name="Pell")
    with pytest.raises(ValueError, match="INSTANCE_OF"):
        g.link("INSTANCE_OF", "npc:pell", "region:forest")  # only a Tile is an INSTANCE_OF an Asset
    with pytest.raises(ValueError, match="'dir'"):
        g.link("ADJACENT", "region:forest", "region:forest")
    with pytest.raises(ValueError, match="npc:ghost"):
        g.link("KNOWS", "npc:pell", "npc:ghost")
    with pytest.raises(ValueError, match="HAUNTS"):
        g.link("HAUNTS", "npc:pell", "region:forest")
    assert (
        g.get("npc:pell").kind == "NPC" and g.get("npc:pell")["name"] == "Pell" and g.get("npc:ghost") is None
    )
    assert [r.id for r in g.nodes("NPC")] == ["npc:pell"] and g.neighbours("npc:pell") == []
    assert g.nodes("Quest") == [] and g.count("NPC") == 1


def test_json_round_trip_preserves_every_record_kind():
    g, c, _ = _filled_graph()
    _game_records(g)
    assert g.validate() == []
    g2 = WorldStateGraph.from_json(g.to_json())
    assert g2.to_json() == g.to_json() and g2.validate() == []
    for kind in NODE_KINDS:
        assert g2.nodes(kind) == g.nodes(kind) != []
    assert {e["key"] for e in g2.to_json()["edges"]} == set(EDGE_KINDS)
    assert np.array_equal(g2.chunks[c.key].layout.grid, c.layout.grid)
    assert np.allclose(g2.region_style("forest"), g.region_style("forest"))
    assert g2.get("npc:mara")["role"] == "innkeeper"
    assert [r.id for r in g2.neighbours("npc:mara", "KNOWS")] == ["quest:rats", "landmark:tower"]
    assert [r.id for r in g2.neighbours("npc:mara", "CONTAINS", direction="in")] == ["settlement:emberfall"]
    assert [r.id for r in g2.neighbours("quest:rats", direction="in")] == ["world", "npc:mara", "event:fair"]


def test_from_json_rejects_a_record_the_schema_forbids():
    g, _, _ = _filled_graph()
    _game_records(g)
    data = g.to_json()
    del next(n for n in data["nodes"] if n["id"] == "npc:mara")["name"]
    with pytest.raises(ValueError, match="npc:mara.*'name'"):
        WorldStateGraph.from_json(data)


# --- beyond the plan ---------------------------------------------------------------------------------


def _cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_neighbours_are_keyed_by_side_and_only_present_chunks_count():
    g = WorldStateGraph(seed=0, chunk_size=SIZE)
    g.add_region("forest", "forest")
    west = _chunk(0, 0, seed=1)
    g.add_chunk(west, "forest")
    east = _chunk(1, 0, seed=2, neighbours={"W": west.layout})
    g.add_chunk(east, "forest")
    assert list(g.neighbour_chunks(0, 0)) == ["E"] and g.neighbour_chunks(0, 0)["E"] is east
    assert list(g.neighbour_chunks(1, 0)) == ["W"] and g.neighbour_chunks(5, 5) == {}
    assert g.chunk(1, 0) is east and g.chunk(2, 2) is None
    assert g.seam_violations() == 0
    assert [r.id for r in g.neighbours("chunk:0,0", "ADJACENT")] == ["chunk:1,0"]
    assert [r.id for r in g.neighbours("chunk:1,0", "ADJACENT")] == ["chunk:0,0"]


def test_seam_violations_counts_forbidden_border_pairs():
    b = load_biome("forest")
    ts = tileset_from_example(b.example_map, b.legend)
    g = WorldStateGraph(seed=0, chunk_size=SIZE)
    g.add_region("forest", "forest")
    grass = Layout(np.full((SIZE, SIZE), ts.index("grass"), np.int32), ts)
    water = Layout(np.full((SIZE, SIZE), ts.index("water"), np.int32), ts)
    g.add_chunk(Chunk(0, 0, "forest", grass), "forest")
    g.add_chunk(Chunk(1, 0, "forest", water), "forest")  # grass never touches water in the forest map
    assert g.seam_violations() == SIZE


def test_remove_chunk_drops_its_tiles_and_only_the_assets_no_tile_uses():
    g, c, _ = _filled_graph()
    other = _chunk(1, 0, seed=2, neighbours={"W": c.layout})
    g.add_chunk(other, "forest")
    _fill(g, other)
    shared = set(c.asset_ids.values()) & set(other.asset_ids.values())
    only_other = set(other.asset_ids.values()) - set(c.asset_ids.values())
    assert set(g.remove_chunk((1, 0))) == only_other
    assert all(g.get(f"asset:{a}") for a in shared) and not any(g.get(f"asset:{a}") for a in only_other)
    assert g.chunk(1, 0) is None and {(t["cx"], t["cy"]) for t in g.nodes("Tile")} == {(0, 0)}
    assert g.validate() == [] and g.neighbour_chunks(0, 0) == {}


def test_replacing_a_class_asset_drops_the_orphan_and_keeps_one_anchor():
    g, c, _ = _filled_graph()
    cls = next(iter(c.asset_ids))  # the first asset added: the region's STYLE_ANCHOR
    old, new = c.asset_ids[cls], _asset(cls, seed=7)
    assert g.add_asset(new, c.key, cls, coherence=0.9) == [old]
    assert c.asset_ids[cls] == new.id and g.get(f"asset:{old}") is None and g.validate() == []
    anchors = g.neighbours("region:forest", "STYLE_ANCHOR")
    assert [a.id for a in anchors] == [f"asset:{new.id}"] and anchors[0]["coherence"] == 0.9


def test_region_style_is_an_ema_with_alpha_0_2_in_insertion_order():
    g = WorldStateGraph(seed=0, chunk_size=SIZE)
    g.add_region("forest", "forest")
    c = _chunk(0, 0, seed=1)
    g.add_chunk(c, "forest")
    expected = None
    for cls in sorted({c.layout.class_at(x, y) for x in range(SIZE) for y in range(SIZE)}):
        a = _asset(cls)
        g.add_asset(a, c.key, cls)
        expected = a.style_vec if expected is None else 0.8 * expected + 0.2 * a.style_vec
    assert np.allclose(g.region_style("forest"), expected) and g.region_style("elsewhere") is None


def test_candidate_coherence_matches_a_grid_scan():
    g, c, assets = _filled_graph()
    rows, cls = c.layout.class_rows(), c.layout.class_at(0, 0)
    vec = _asset(cls, seed=9).style_vec
    cosines = [
        _cosine(vec, assets[rows[y + dy][x + dx]].style_vec)
        for y in range(SIZE)
        for x in range(SIZE)
        if rows[y][x] == cls
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if 0 <= x + dx < SIZE and 0 <= y + dy < SIZE and rows[y + dy][x + dx] != cls
    ]
    region_term = _cosine(vec, g.region_style("forest"))
    expected = 0.6 * region_term + 0.4 * np.mean(cosines) if cosines else region_term
    assert g.candidate_coherence(vec, c.key, cls) == pytest.approx(expected)
    own = assets[cls].style_vec
    assert g.coherence(assets[cls].id) == pytest.approx(g.candidate_coherence(own, c.key, cls))


def test_coherence_with_no_region_style_and_no_neighbours_is_one():
    g = WorldStateGraph(seed=0, chunk_size=SIZE)
    g.add_region("forest", "forest")
    c = _chunk(0, 0, seed=1)
    g.add_chunk(c, "forest")
    assert g.candidate_coherence(np.ones(48), c.key, c.layout.class_at(0, 0)) == 1.0


def test_validate_flags_an_asset_of_another_tile_class():
    g, c, _ = _filled_graph()
    g.add_asset(_asset("no-such-class"), c.key, c.layout.class_at(0, 0))
    assert any("does not depict" in p for p in g.validate())


def test_best_asset_prefers_the_highest_coherence_and_filters_by_class():
    g, c, assets = _filled_graph()
    classes = list(assets)
    for i, cls in enumerate(classes):
        g.add_asset(assets[cls], c.key, cls, coherence=0.1 * (i + 1))  # re-adding only updates the score
    assert g.best_asset("forest") == assets[classes[-1]].id
    assert g.best_asset("forest", classes[0]) == assets[classes[0]].id
    assert g.best_asset("forest", "no-such-class") is None and g.best_asset("elsewhere") is None
