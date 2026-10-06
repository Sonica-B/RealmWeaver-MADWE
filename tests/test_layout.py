import logging

import numpy as np
import pytest

from realmweaver.biomes import biome_names, load_biome
from realmweaver.layout import Contradiction, render, solve, solve_chunk, tileset_from_example, violations
from realmweaver.types import Layout, opposite

LEGEND = {"g": "grass", "w": "water", "s": "shore"}
MAP = "ggggg\ngsssg\ngswsg\ngsssg\nggggg"  # water never touches grass directly


def test_example_adjacency_is_exactly_the_pairs_present():
    ts = tileset_from_example(MAP, LEGEND)
    g, w, s = (ts.index(n) for n in ("grass", "water", "shore"))
    assert ts.allowed[g, 2, s] and ts.allowed[s, 2, w]  # grass->S->shore, shore->S->water
    assert not ts.allowed[g, :, w].any() and not ts.allowed[w, :, g].any()
    assert ts.weights.sum() == pytest.approx(1.0) and ts.weights[g] > ts.weights[w]


def test_solve_has_zero_violations_and_is_deterministic():
    ts = tileset_from_example(MAP, LEGEND)
    a, b = solve(ts, 24, 24, seed=7), solve(ts, 24, 24, seed=7)
    assert a.grid.shape == (24, 24) and violations(a) == 0
    assert np.array_equal(a.grid, b.grid) and not np.array_equal(a.grid, solve(ts, 24, 24, seed=8).grid)


def test_fixed_cells_are_honoured():
    ts = tileset_from_example(MAP, LEGEND)
    lay = solve(ts, 8, 8, seed=1, fixed={(0, 0): "water", (7, 7): "grass"})
    assert lay.class_at(0, 0) == "water" and lay.class_at(7, 7) == "grass" and violations(lay) == 0


def test_chunk_border_matches_neighbour():
    ts = tileset_from_example(MAP, LEGEND)
    west = solve_chunk(ts, 8, seed=3, neighbours={})
    east = solve_chunk(ts, 8, seed=4, neighbours={"W": west})
    for y in range(8):
        a, b = ts.index(west.class_at(7, y)), ts.index(east.class_at(0, y))
        assert ts.allowed[a, 1, b]  # west cell -> E -> east cell


def test_impossible_fixed_cells_raise_contradiction():
    ts = tileset_from_example(MAP, LEGEND)
    with pytest.raises(Contradiction):
        solve(ts, 2, 1, seed=0, fixed={(0, 0): "water", (1, 0): "grass"})


def test_every_shipped_biome_solves():
    for name in biome_names():
        b = load_biome(name)
        ts = tileset_from_example(b.example_map, b.legend)
        assert violations(solve(ts, 16, 16, seed=0)) == 0


# --- beyond the plan ---------------------------------------------------------------------------------


def test_tileset_classes_follow_legend_order_and_weights_are_counts():
    ts = tileset_from_example(MAP, LEGEND)
    assert ts.classes == ["grass", "water", "shore"]
    assert ts.weights.tolist() == pytest.approx([16 / 25, 1 / 25, 8 / 25])
    assert ts.allowed.shape == (3, 4, 3) and ts.allowed.dtype == bool
    assert not ts.allowed[ts.index("water"), :, ts.index("water")].any()  # a lone cell learns no self-pair


def test_ragged_example_map_is_rejected():
    with pytest.raises(ValueError):
        tileset_from_example("ggg\ngg", LEGEND)


def test_legend_class_absent_from_map_is_never_placed():
    ts = tileset_from_example(MAP, {**LEGEND, "x": "lava"})
    lay = solve(ts, 12, 12, seed=2)
    assert violations(lay) == 0 and not (lay.grid == ts.index("lava")).any()


def test_violations_counts_forbidden_pairs_in_a_hand_made_grid():
    ts = tileset_from_example(MAP, LEGEND)
    g, w, s = (ts.index(n) for n in ("grass", "water", "shore"))
    good = Layout(grid=np.array([[g, s, w], [g, s, s]], dtype=np.int32), tileset=ts)
    bad = Layout(grid=np.array([[g, w, w], [g, s, g]], dtype=np.int32), tileset=ts)
    assert violations(good) == 0
    assert violations(bad) == 3  # grass|water, water|water, water/grass counted by hand


def test_render_paints_each_cell_with_its_class_colour():
    ts = tileset_from_example(MAP, LEGEND)
    lay = solve(ts, 6, 4, seed=5)
    img = render(lay, {"grass": "#3f7d3a", "water": "#2c6f9e"}, cell=3)  # shore has no entry: grey
    assert img.shape == (12, 18, 3) and img.dtype == np.uint8
    expected = {"grass": (0x3F, 0x7D, 0x3A), "water": (0x2C, 0x6F, 0x9E), "shore": (0x80, 0x80, 0x80)}
    for y in range(4):
        for x in range(6):
            block = img[y * 3 : (y + 1) * 3, x * 3 : (x + 1) * 3]
            assert (block == expected[lay.class_at(x, y)]).all()


def test_solve_chunk_is_deterministic_per_seed():
    ts = tileset_from_example(MAP, LEGEND)
    a, b = solve_chunk(ts, 8, seed=3, neighbours={}), solve_chunk(ts, 8, seed=3, neighbours={})
    assert np.array_equal(a.grid, b.grid)


def test_chunk_with_four_ready_neighbours_matches_every_border():
    b = load_biome("forest")
    ts = tileset_from_example(b.example_map, b.legend)
    around = {key: solve_chunk(ts, 16, seed=10 + i, neighbours={}) for i, key in enumerate("NESW")}
    mid = solve_chunk(ts, 16, seed=99, neighbours=around)
    assert violations(mid) == 0
    for i in range(16):
        pairs = [  # (our edge cell, the neighbour's touching cell, direction from ours to theirs)
            (mid.grid[0, i], around["N"].grid[15, i], 0),
            (mid.grid[i, 15], around["E"].grid[i, 0], 1),
            (mid.grid[15, i], around["S"].grid[0, i], 2),
            (mid.grid[i, 0], around["W"].grid[i, 15], 3),
        ]
        for ours, theirs, d in pairs:
            assert ts.allowed[ours, d, theirs] and ts.allowed[theirs, opposite(d), ours]


def test_chunk_rejects_neighbour_of_another_size():
    ts = tileset_from_example(MAP, LEGEND)
    with pytest.raises(ValueError):
        solve_chunk(ts, 8, seed=0, neighbours={"N": solve_chunk(ts, 4, seed=0, neighbours={})})


def test_chunk_corner_where_two_neighbours_disagree_keeps_the_first_neighbours_rule(caplog):
    b = load_biome("volcanic")
    ts = tileset_from_example(b.example_map, b.legend)
    basalt, lava, ash = (ts.index(n) for n in ("basalt", "lava", "ash"))
    north = np.full((16, 16), basalt, dtype=np.int32)
    north[15, 0] = lava  # above our (0, 0): demands lava or cooled lava below it
    west = np.full((16, 16), basalt, dtype=np.int32)
    west[0, 15] = ash  # left of our (0, 0): demands ash or basalt to its east
    assert not (ts.allowed[lava, 2, :] & ts.allowed[ash, 1, :]).any()  # the two demands are incompatible
    with caplog.at_level(logging.WARNING, logger="realmweaver.layout.wfc"):
        lay = solve_chunk(ts, 16, seed=0, neighbours={"N": Layout(north, ts), "W": Layout(west, ts)})
    assert "chunk border W: 1 cell(s)" in caplog.text
    assert violations(lay) == 0 and ts.allowed[lava, 2, lay.grid[0, 0]]  # N wins at the corner
    assert all(ts.allowed[basalt, 2, lay.grid[0, x]] for x in range(1, 16))  # rest of the N edge matches
    assert all(ts.allowed[basalt, 1, lay.grid[y, 0]] for y in range(1, 16))  # rest of the W edge matches


@pytest.mark.parametrize("name", biome_names())
def test_shipped_biome_map_is_complete_and_solves_across_seeds(name):
    b = load_biome(name)
    ts = tileset_from_example(b.example_map, b.legend)
    assert set(ts.classes) == set(b.tiles) and (ts.weights > 0).all()  # every tile class is drawn in the map
    assert ts.allowed.any(axis=2).all()  # every class has a neighbour in every direction: placeable anywhere
    for seed in range(1, 4):
        assert violations(solve(ts, 16, 16, seed=seed)) == 0


# biome -> (hazard class, the only classes it may touch): the adjacency structure each map encodes
RINGED = {
    "forest": ("water", {"water", "shore", "dirt"}),  # the forest river has a dirt bank on its north side
    "desert": ("water", {"water", "shore"}),
    "snow": ("frozen_lake", {"frozen_lake", "ice"}),
    "volcanic": ("lava", {"lava", "cooled_lava"}),
    "underwater": ("deep_water", {"deep_water", "rock"}),
    "sky": ("void", {"void", "cloud"}),
}


@pytest.mark.parametrize(("name", "hazard", "ring"), [(k, *v) for k, v in RINGED.items()])
def test_biome_maps_ring_their_hazard_class(name, hazard, ring):
    b = load_biome(name)
    ts = tileset_from_example(b.example_map, b.legend)
    touching = {ts.classes[i] for i in np.flatnonzero(ts.allowed[ts.index(hazard)].any(axis=0))}
    assert touching == ring
