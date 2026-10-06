"""Procedural generator, prompt building and alpha keying: the CPU side of the `Generator.generate` seam."""

import numpy as np
import pytest

from realmweaver.assets import ProceduralGenerator, alpha_from_white, build_prompt
from realmweaver.biomes import load_biome
from realmweaver.types import AssetSpec


def test_texture_is_uint8_rgb_of_requested_size_and_deterministic():
    g = ProceduralGenerator()
    a = g.generate(AssetSpec("forest", subject="grass", size=64, seed=3))
    b = g.generate(AssetSpec("forest", subject="grass", size=64, seed=3))
    assert a.image.shape == (64, 64, 3) and a.image.dtype == np.uint8
    assert np.array_equal(a.image, b.image) and a.id == b.id and a.latency_s >= 0


def test_different_seed_or_subject_changes_image():
    g = ProceduralGenerator()
    s = AssetSpec("forest", subject="grass", size=64, seed=3)
    assert not np.array_equal(
        g.generate(s).image, g.generate(AssetSpec("forest", subject="grass", size=64, seed=4)).image
    )
    assert not np.array_equal(
        g.generate(s).image, g.generate(AssetSpec("forest", subject="water", size=64, seed=3)).image
    )


def test_sprite_has_alpha_and_transparent_corners():
    spec = AssetSpec("forest", kind="sprite", subject="mushroom", size=64)
    img = ProceduralGenerator().generate(spec).image
    assert img.shape == (64, 64, 4) and img[0, 0, 3] == 0 and img[32, 32, 3] == 255


def test_build_prompt_mentions_tile_and_style():
    b = load_biome("forest")
    pos, neg = build_prompt(b, AssetSpec("forest", subject="grass"))
    assert "grass" in pos and b.style in pos and "seamless" in pos and neg == b.negative


def test_alpha_from_white_keys_background():
    rgb = np.full((8, 8, 3), 255, np.uint8)
    rgb[2:6, 2:6] = (200, 30, 30)
    out = alpha_from_white(rgb)
    assert out.shape == (8, 8, 4) and out[0, 0, 3] == 0 and out[3, 3, 3] == 255


# --- beyond the plan ---------------------------------------------------------------------------------


def test_seed_salt_changes_image_but_not_id():
    spec = AssetSpec("forest", subject="grass", size=32, seed=3)
    a, b = ProceduralGenerator().generate(spec), ProceduralGenerator(seed_salt=1).generate(spec)
    assert a.id == b.id and not np.array_equal(a.image, b.image)


def test_texture_wraps_without_a_seam():
    """Noise sampled on a torus: the wrapped-edge step is no larger than a typical interior step."""
    spec = AssetSpec("forest", subject="dirt", size=64, seed=7)
    img = ProceduralGenerator().generate(spec).image.astype(np.float32)
    seam = np.abs(img[:, -1] - img[:, 0]).mean() + np.abs(img[-1, :] - img[0, :]).mean()
    interior = np.abs(np.diff(img, axis=1)).mean() + np.abs(np.diff(img, axis=0)).mean()
    assert seam < 2 * interior


def test_unknown_subject_falls_back_to_grey():
    img = ProceduralGenerator().generate(AssetSpec("forest", subject="no-such-tile", size=32)).image
    assert np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2])
    assert img.min() < img.max()  # shades of grey, not one flat value


def test_unknown_biome_raises_key_error():
    with pytest.raises(KeyError):
        ProceduralGenerator().generate(AssetSpec("atlantis", subject="grass", size=16))


def test_large_texture_has_requested_shape_and_measured_latency():
    a = ProceduralGenerator().generate(AssetSpec("forest", subject="rock", size=512, seed=1))
    assert a.image.shape == (512, 512, 3) and a.image.dtype == np.uint8 and a.latency_s > 0


def test_sprite_background_is_white_under_transparent_pixels():
    img = ProceduralGenerator().generate(AssetSpec("forest", kind="sprite", subject="fern", size=32)).image
    assert tuple(img[0, 0]) == (255, 255, 255, 0)


def test_build_prompt_texture_is_exact():
    b = load_biome("forest")
    pos, neg = build_prompt(b, AssetSpec("forest", subject="water"))
    assert pos == f"{b.tiles['water'].prompt}, {b.style}, seamless tileable texture" and neg == b.negative


def test_build_prompt_for_sprite_uses_prop_description():
    b = load_biome("forest")
    pos, neg = build_prompt(b, AssetSpec("forest", kind="sprite", subject="mushroom"))
    assert pos == f"{b.props['mushroom']}, {b.style}" and "seamless" not in pos and neg == b.negative


def test_build_prompt_unknown_prop_uses_subject_itself():
    b = load_biome("forest")
    pos, _ = build_prompt(b, AssetSpec("forest", kind="sprite", subject="brass lantern"))
    assert pos == f"brass lantern, {b.style}"


def test_alpha_from_white_respects_tolerance_and_keeps_colour():
    rgb = np.full((4, 4, 3), 230, np.uint8)  # sqrt(3 * 25**2) ~ 43 away from white
    assert alpha_from_white(rgb)[0, 0, 3] == 255 and alpha_from_white(rgb, tol=60)[0, 0, 3] == 0
    assert np.array_equal(alpha_from_white(rgb)[..., :3], rgb)
