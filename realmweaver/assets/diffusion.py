"""SD1.5 + Hyper-SD few-step generator: seamless textures via circular padding, alpha-keyed sprites, memory pool.

Model settings follow the Hyper-SD model card: the fixed-step LoRAs run DDIM with `timestep_spacing="trailing"`;
the plain 4-step LoRA (`draft`) takes no classifier-free guidance and the 8-step CFG-preserved LoRA (`refine`)
takes guidance 5. A biome LoRA, when configured, is stacked as a third adapter named after the biome.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from realmweaver.assets.pool import Embeds, MemoryPool
from realmweaver.assets.seamless import set_seamless
from realmweaver.biomes import Biome, load_biome
from realmweaver.config import Settings
from realmweaver.config import settings as load_settings
from realmweaver.types import Asset, AssetSpec, Generator, Tier

log = logging.getLogger(__name__)

# tier -> (guidance scale, minimum steps). AssetSpec.steps defaults to the draft count, so refine raises it to the
# 8 steps its LoRA was distilled for; larger explicit step counts are honoured.
_TIERS: dict[Tier, tuple[float, int]] = {"draft": (1.0, 1), "refine": (5.0, 8)}
# Sprites start from noise mixed with this fraction of the VAE latent of a white canvas. The biome style suffix
# in the prompt otherwise wins over "white background" and the model paints a full scene (measured in
# docs/research/05-gpu-measurements.md: 0.0 gives no white border, 0.15 gives a blank canvas).
_WHITE_BIAS = 0.05


class DiffusionGenerator(Generator):
    """The GPU asset agent: `generate(spec) -> Asset` on SD1.5 fp16 with every weight resident on the device.

    `pool=True` hands the pipeline one reused latent buffer, caches prompt embeds per (biome, kind, subject, tier)
    and replays the UNet step from CUDA graphs captured per (adapters, padding, shape). `compile=True` hands the
    UNet forward to `torch.compile(mode="reduce-overhead")` instead of the pool's graphs (needs triton; off by
    default). `biome_loras` maps a biome name to a LoRA directory, file or hub id and falls back to the biome's
    `lora:` field.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        pool: bool = True,
        compile: bool = False,
        biome_loras: dict[str, str] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        if not self.settings.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "DiffusionGenerator needs a CUDA device (settings().device); use ProceduralGenerator on CPU"
            )
        self.device = torch.device(self.settings.device)
        self._pipe = self._load_pipeline()
        self._pool = MemoryPool(self.settings.asset_size, device=self.settings.device) if pool else None
        self._biome_loras = dict(biome_loras or {})
        self._biome_adapters: dict[str, bool] = {}
        self._active: tuple[str, ...] = ()
        self._seamless: bool | None = None
        self._white: dict[int, torch.Tensor] = {}
        self._eager_unet = self._pipe.unet.forward
        if compile:
            self._pipe.unet.forward = torch.compile(self._eager_unet, mode="reduce-overhead", dynamic=False)
            log.info("UNet forward wrapped in torch.compile; the first generation pays the cold start")
        elif self._pool is not None:
            self._pipe.unet.forward = self._graphed_unet

    @property
    def pool(self) -> MemoryPool | None:
        return self._pool

    @property
    def active_adapters(self) -> tuple[str, ...]:
        return self._active

    def _load_pipeline(self) -> Any:
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        s = self.settings
        t0 = time.perf_counter()
        pipe = StableDiffusionPipeline.from_pretrained(
            s.model_id, dtype=torch.float16, safety_checker=None, requires_safety_checker=False
        ).to(self.device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.set_progress_bar_config(disable=True)
        pipe.load_lora_weights(s.fast_lora_repo, weight_name=s.draft_lora_file, adapter_name="draft")
        pipe.load_lora_weights(s.fast_lora_repo, weight_name=s.refine_lora_file, adapter_name="refine")
        log.info(
            "loaded %s + %s draft/refine LoRAs on %s in %.1fs",
            s.model_id,
            s.fast_lora_repo,
            self.device,
            time.perf_counter() - t0,
        )
        return pipe

    def _ensure_biome_adapter(self, biome: Biome) -> bool:
        """Load the biome LoRA once (`biome_loras`, else the biome's `lora:` field); True when one exists."""
        if biome.name not in self._biome_adapters:
            path = self._biome_loras.get(biome.name) or biome.lora
            if path:
                self._pipe.load_lora_weights(path, adapter_name=biome.name)
                self._active = ()  # loading changes diffusers' active set; re-select on the next generate
                log.info("loaded biome LoRA %r from %s", biome.name, path)
            self._biome_adapters[biome.name] = bool(path)
        return self._biome_adapters[biome.name]

    def _activate(self, tier: Tier, biome: Biome) -> None:
        names = (tier, biome.name) if self._ensure_biome_adapter(biome) else (tier,)
        if names != self._active:
            self._pipe.set_adapters(list(names), [1.0] * len(names))
            self._active = names

    def _set_seamless(self, on: bool) -> None:
        if on != self._seamless:
            set_seamless(self._pipe.unet, on)
            set_seamless(self._pipe.vae, on)
            self._seamless = on

    def _graphed_unet(
        self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor]:
        """UNet forward replacement: replay the graph the pool holds for the current adapters, padding and shapes."""
        key = (self._active, self._seamless, tuple(sample.shape), tuple(encoder_hidden_states.shape))
        return (self._pool.step(key, self._eager_unet, sample, timestep, encoder_hidden_states, **kwargs),)

    def _white_latent(self, size: int) -> torch.Tensor:
        """Scaled VAE latent of a white `size`x`size` canvas, encoded once per size (with zeros padding)."""
        if size not in self._white:
            with torch.inference_mode():
                canvas = torch.ones(1, 3, size, size, device=self.device, dtype=torch.float16)
                dist = self._pipe.vae.encode(canvas).latent_dist
                self._white[size] = dist.mean * self._pipe.vae.config.scaling_factor
        return self._white[size]

    def _embeds(self, spec: AssetSpec, biome: Biome) -> Embeds:
        """Prompt embeds for `spec`, cached in the pool per (biome, kind, subject, tier); tiers differ in CFG."""

        def make() -> Embeds:
            from realmweaver.assets.prompts import build_prompt

            positive, negative = build_prompt(biome, spec)
            cfg = _TIERS[spec.tier][0] > 1.0
            with torch.inference_mode():  # cached embeds must not pin the text encoder's autograd state
                embeds, negative_embeds = self._pipe.encode_prompt(
                    positive, self.device, 1, cfg, negative_prompt=negative
                )
            return embeds, negative_embeds

        if self._pool is None:
            return make()
        return self._pool.embeds((spec.biome, spec.kind, spec.subject, spec.tier), make)

    def generate(self, spec: AssetSpec) -> Asset:
        t0 = time.perf_counter()
        biome = load_biome(spec.biome)
        guidance, min_steps = _TIERS[spec.tier]
        self._activate(spec.tier, biome)
        self._set_seamless(spec.seamless and spec.kind == "texture")
        embeds, negative_embeds = self._embeds(spec, biome)
        latents = generator = None
        if self._pool is not None and spec.size == self._pool.size:
            latents = self._pool.seeded_latents(spec.seed)
        else:  # ponytail: sizes other than settings().asset_size bypass the pool and draw fresh device noise
            generator = torch.Generator(self.device).manual_seed(spec.seed)
        if spec.kind == "sprite":  # start near a white canvas so alpha keying has a background to remove
            if latents is None:
                shape = (1, 4, spec.size // 8, spec.size // 8)
                latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float16)
            latents.mul_(1.0 - _WHITE_BIAS).add_(self._white_latent(spec.size), alpha=_WHITE_BIAS)
        with torch.inference_mode():
            out = self._pipe(
                prompt_embeds=embeds,
                negative_prompt_embeds=negative_embeds,
                height=spec.size,
                width=spec.size,
                num_inference_steps=max(spec.steps, min_steps),
                guidance_scale=guidance,
                generator=generator,
                latents=latents,
                output_type="pt",
                return_dict=False,
            )[0]
        rgb = (out[0].permute(1, 2, 0) * 255).round().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        if spec.kind == "sprite":
            from realmweaver.assets.alpha import alpha_from_white

            image = alpha_from_white(rgb)
        else:
            image = rgb
        return Asset(spec=spec, image=image, latency_s=time.perf_counter() - t0)

    def warmup(self, biome: str) -> None:
        """Load the biome LoRA, cache prompt embeds for every tile class and prop in both tiers, and run one asset
        per tier plus one sprite so cuDNN autotuning, the allocator cache and the captured UNet graphs are paid
        before the first real request."""
        b = load_biome(biome)
        for tier in _TIERS:
            self._activate(tier, b)
            if self._pool is not None:
                for cls in b.tiles:
                    self._embeds(AssetSpec(biome, subject=cls, tier=tier), b)
                for prop in b.props:
                    self._embeds(AssetSpec(biome, kind="sprite", subject=prop, tier=tier), b)
        first = next(iter(b.tiles))
        self.generate(AssetSpec(biome, subject=first, tier="draft"))
        self.generate(AssetSpec(biome, subject=first, tier="refine", steps=8))
        if b.props:
            self.generate(AssetSpec(biome, kind="sprite", subject=next(iter(b.props))))
        self.reset_stats()

    def allocator_stats(self) -> dict[str, int]:
        """Peak bytes and cumulative allocator counters from `torch.cuda.memory_stats()` since `reset_stats`."""
        s = torch.cuda.memory_stats(self.device)
        return {
            "max_memory_allocated": int(s["allocated_bytes.all.peak"]),
            "max_memory_reserved": int(s["reserved_bytes.all.peak"]),
            "num_alloc_retries": int(s["num_alloc_retries"]),
            "num_device_alloc": int(s["num_device_alloc"]),
        }

    def reset_stats(self) -> None:
        """Zero the peak and accumulated allocator counters so the next `allocator_stats` covers new work only."""
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.reset_accumulated_memory_stats(self.device)
