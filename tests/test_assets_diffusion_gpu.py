"""GPU seam tests for the diffusion generator. Downloads SD1.5, the Hyper-SD LoRAs and DINOv2-small on first run."""

from pathlib import Path

import numpy as np
import pytest
import torch

from realmweaver.types import AssetSpec

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from realmweaver.assets.diffusion import DiffusionGenerator  # noqa: E402
from realmweaver.assets.embed import DinoEmbedder  # noqa: E402
from realmweaver.assets.lora import train_biome_lora  # noqa: E402
from realmweaver.metrics.quality import tileability  # noqa: E402

FOREST_TEXTURES = Path("data/raw/textures/fantasy_forest")


@pytest.fixture(scope="module")
def gen():
    g = DiffusionGenerator()
    g.warmup("forest")
    return g


def test_texture_is_seamless_and_fast(gen):
    a = gen.generate(AssetSpec("forest", subject="grass", seed=1))
    assert a.image.shape == (512, 512, 3) and a.image.dtype == np.uint8
    assert tileability(a.image) < 1.3 and a.latency_s < 3.0


def test_same_seed_same_image(gen):
    s = AssetSpec("forest", subject="rock", seed=5)
    assert np.array_equal(gen.generate(s).image, gen.generate(s).image)
    assert not np.array_equal(
        gen.generate(s).image, gen.generate(AssetSpec("forest", subject="rock", seed=6)).image
    )


def test_refine_tier_uses_cfg_and_stays_seamless(gen):
    a = gen.generate(AssetSpec("forest", subject="water", seed=3, steps=8, tier="refine"))
    assert a.image.shape == (512, 512, 3) and tileability(a.image) < 1.3


def test_sprite_has_transparent_corner(gen):
    a = gen.generate(AssetSpec("forest", kind="sprite", subject="mushroom", seed=2))
    assert a.image.shape == (512, 512, 4) and a.image[0, 0, 3] == 0


def test_pool_makes_zero_new_device_allocs_per_image(gen):
    gen.generate(AssetSpec("forest", subject="grass", seed=9))
    gen.reset_stats()
    gen.generate(AssetSpec("forest", subject="grass", seed=10))
    s = gen.allocator_stats()
    assert s["num_alloc_retries"] == 0 and s["num_device_alloc"] == 0
    assert s["max_memory_allocated"] > 0 and s["max_memory_reserved"] >= s["max_memory_allocated"]


def test_pool_reuses_prompt_embeds_per_biome_subject_tier(gen):
    before = gen.pool.stats()["embeds"]
    gen.generate(AssetSpec("forest", subject="dirt", seed=1))
    gen.generate(AssetSpec("forest", subject="dirt", seed=2))
    after = gen.pool.stats()
    assert after["embeds"] == before and after["total_bytes"] > 0  # warmup already cached every forest prompt
    assert after["graphs"] >= 3 and after["graph_bytes"] > 0  # draft, refine and sprite UNet steps captured


def test_dino_embedder_is_unit_norm_384d(gen):
    emb = DinoEmbedder()
    v = emb(gen.generate(AssetSpec("forest", subject="grass", seed=4)).image)
    assert v.shape == (384,) and v.dtype == np.float32 and abs(np.linalg.norm(v) - 1.0) < 1e-5


@pytest.mark.skipif(not FOREST_TEXTURES.is_dir(), reason="forest reference textures not present")
def test_biome_lora_trains_saves_and_loads(tmp_path):
    out = train_biome_lora("forest", FOREST_TEXTURES, tmp_path / "forest", steps=3)
    assert (out / "pytorch_lora_weights.safetensors").is_file()
    g = DiffusionGenerator(biome_loras={"forest": str(out)})
    a = g.generate(AssetSpec("forest", subject="grass", seed=1))
    assert "forest" in g.active_adapters and tileability(a.image) < 1.3
