"""Quality metrics at the seam: tileability, tiling score, style consistency, histogram embedding."""

import numpy as np
import pytest

from realmweaver.assets import ProceduralGenerator
from realmweaver.metrics import histogram_embed, style_consistency, tileability, tiling_score
from realmweaver.types import AssetSpec


def _seamless():
    return ProceduralGenerator().generate(AssetSpec("forest", subject="grass", size=64, seed=1)).image


def _cut(img):  # destroy the seam by rolling a non-periodic gradient in
    g = np.linspace(0, 255, img.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * 0.3 + g * 0.7, 0, 255).astype(np.uint8)


def test_tileability_ranks_seamless_above_cut():
    s = _seamless()
    assert tileability(s) < tileability(_cut(s)) and tileability(s) < 1.5


def test_tiling_score_lower_for_seamless():
    s = _seamless()
    assert tiling_score(s) < tiling_score(_cut(s))


def test_style_consistency_is_higher_when_biomes_differ():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 8)) + 3
    b = rng.normal(size=(5, 8)) - 3
    close = {"x": a, "y": a + rng.normal(scale=0.01, size=(5, 8))}
    assert style_consistency({"x": a, "y": b}) > style_consistency(close)


def test_histogram_embed_is_unit_norm_48d():
    v = histogram_embed(_seamless())
    assert v.shape == (48,) and abs(np.linalg.norm(v) - 1) < 1e-6


# --- beyond the plan ---------------------------------------------------------------------------------


def _ramp(n: int = 64) -> np.ndarray:
    """Horizontal 0..255 ramp: a hard wrapped seam on the vertical edge, none on the horizontal one."""
    return np.repeat(np.linspace(0, 255, n, dtype=np.float32)[None, :], n, axis=0)


def test_tileability_of_a_ramp_matches_hand_computed_ratio():
    # seam pairs: 64 vertical-edge steps of 255 and 64 horizontal-edge steps of 0 -> mean 127.5
    # interior pairs: 64*63 horizontal steps of 255/63 and 63*64 vertical steps of 0 -> mean 255/126
    assert tileability(_ramp()) == pytest.approx(63.0, rel=1e-4)


def test_tileability_is_one_for_a_flat_image():
    assert tileability(np.full((16, 16, 3), 90, np.uint8)) == 1.0


def test_tiling_score_of_a_ramp_matches_hand_computed_value():
    # rolled by half, the vertical seam band holds 6 ramp steps of 255/63 and one jump of 255 (7 adjacent
    # pairs); the horizontal band holds only zeros; the mean of the two band MSEs is normalised by variance
    step = 255 / 63
    expected = ((6 * step**2 + 255**2) / 7 / 2) / np.var(_ramp())
    assert tiling_score(_ramp()) == pytest.approx(expected, rel=1e-3)


def test_tiling_score_is_zero_for_a_flat_image():
    assert tiling_score(np.full((16, 16, 3), 90, np.uint8)) == 0.0


def test_style_consistency_identical_within_orthogonal_across_is_one():
    x = np.array([[2.0, 0.0], [5.0, 0.0]])
    y = np.array([[0.0, 1.0], [0.0, 3.0]])
    assert style_consistency({"x": x, "y": y}) == pytest.approx(1.0)


def test_style_consistency_of_one_shared_style_is_zero():
    v = np.array([[1.0, 2.0, 3.0]] * 3)
    assert style_consistency({"x": v, "y": v}) == pytest.approx(0.0)  # within 1 minus cross 1


def test_histogram_embed_counts_pixels_per_bin():
    v = histogram_embed(np.tile(np.array([255, 0, 0], np.uint8), (4, 4, 1)))
    expected = np.zeros(48)
    expected[[15, 16, 32]] = 1 / np.sqrt(3)  # R in its top bin, G and B in their bottom bins
    assert np.allclose(v, expected)


def test_metrics_accept_sprites_with_alpha():
    img = ProceduralGenerator().generate(AssetSpec("forest", kind="sprite", subject="log", size=32)).image
    assert np.isfinite(tileability(img)) and np.isfinite(tiling_score(img))
    assert histogram_embed(img).shape == (48,)
