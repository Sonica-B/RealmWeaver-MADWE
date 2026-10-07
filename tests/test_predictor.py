"""Predictor: order-2 Markov over quantised headings against the uniform 8-ring baseline."""

import numpy as np
import pytest

from realmweaver.world import Predictor


def _walk(seed, n=400, chunk=16):
    rng = np.random.default_rng(seed)
    pos = np.zeros(2)
    heading = rng.uniform(0, 2 * np.pi)
    out = []
    for _ in range(n):
        heading += rng.normal(0, 0.25)
        pos += np.array([np.cos(heading), np.sin(heading)])
        out.append(pos.copy())
    return out


def test_markov_predictor_beats_ring_baseline_on_synthetic_walks():
    hits_m = hits_r = total = 0
    for seed in range(20):
        p = Predictor()
        path = _walk(seed)
        cur = None
        for pos in path:
            ck = (int(pos[0] // 16), int(pos[1] // 16))
            if cur is not None and ck != cur:
                total += 1
                hits_m += ck in [c for c, _ in p.rank(cur, k=3)]
                hits_r += ck in [c for c, _ in p.ring_baseline(cur)[:3]]
            p.observe(tuple(pos))
            cur = ck
    assert total > 100 and hits_m / total > hits_r / total + 0.15


# --- beyond the plan ---------------------------------------------------------------------------------


def _observe(p, points):
    for x, y in points:
        p.observe((float(x), float(y)))


def test_rank_before_any_movement_is_the_uniform_ring():
    p = Predictor()
    assert p.rank((3, 4)) == p.ring_baseline((3, 4)) and p.heading_distribution() is None
    p.observe((50.0, 70.0))  # one position is still no heading
    assert p.rank((3, 4), k=3) == p.ring_baseline((3, 4))[:3]


def test_ring_baseline_is_uniform_over_the_eight_neighbours():
    ring = Predictor().ring_baseline((0, 0))
    around = {(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)} - {(0, 0)}
    assert len(ring) == 8 and {c for c, _ in ring} == around and all(p == 1 / 8 for _, p in ring)


def test_walking_east_puts_the_east_chunk_first_with_most_of_the_mass():
    p = Predictor(chunk_size=16)
    _observe(p, [(x, 8) for x in range(15)])
    ranked = p.rank((0, 0), k=8)
    assert ranked[0][0] == (1, 0) and ranked[0][1] > 0.8 and len(ranked) == 8
    assert sum(prob for _, prob in ranked) == pytest.approx(1.0)
    assert {c for c, _ in ranked} == {c for c, _ in p.ring_baseline((0, 0))}


def test_order_two_context_beats_constant_velocity_on_a_staircase():
    p = Predictor(chunk_size=16)
    pos = [0.0, 0.0]
    for i in range(16):  # east, north, east, north, ...: the heading alternates every step
        pos[i % 2] += 1.0
        p.observe(tuple(pos))
    dist = p.heading_distribution()
    # the last step went north; constant velocity would keep going north, the order-2 context says east
    assert dist is not None and dist[0] == pytest.approx((6 + 0.05) / 7) and dist[0] > dist[2]


def test_order_one_fallback_and_smoothing_match_hand_computed_values():
    p = Predictor(chunk_size=16)
    _observe(p, [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)])  # E, E, E, N
    dist = p.heading_distribution()  # context (E, N) and N were never seen: constant velocity around N
    assert dist[2] == pytest.approx(0.6) and dist[1] == pytest.approx(0.15) and dist[6] == pytest.approx(0.0)
    p.observe((4.0, 1.0))  # E: context (N, E) is new, order-1 knows E -> {E: 2, N: 1}
    dist = p.heading_distribution()
    assert dist[0] == pytest.approx((2 + 0.6) / 4) and dist[2] == pytest.approx((1 + 0.05) / 4)


def test_stationary_observations_carry_no_heading():
    p = Predictor()
    _observe(p, [(0, 0), (1, 0), (1, 0), (1, 0)])
    assert p.heading_distribution()[0] == pytest.approx(0.6)


def test_rank_from_a_chunk_the_player_is_not_in_casts_from_its_centre():
    p = Predictor(chunk_size=16)
    _observe(p, [(x, 8) for x in range(10)])
    chunk, prob = p.rank((5, 5), k=1)[0]
    assert chunk == (6, 5) and prob > 0.5
