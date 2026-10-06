"""Memory pool: one pre-allocated, graph-captured generation slot reused across assets.

Holds the device latent buffer the pipeline is handed as `latents=`, a pinned host noise buffer filled from a seeded
CPU generator (so a seed gives the same noise on any GPU), the prompt-embed cache keyed by
(biome, kind, subject, tier), and the UNet steps captured as CUDA graphs per (adapters, padding, shape). Weights
stay resident; after warm-up nothing here is re-allocated per image and the UNet step replays without Python
dispatch, which is where an unfused PEFT LoRA spends most of its time.

Allocator setting: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported before torch is imported, only when
neither it nor `PYTORCH_ALLOC_CONF` is already set and only off Windows: torch 2.11 on Windows prints
"expandable_segments not supported on this platform" and ignores it. The setting is read at the first CUDA
allocation, so importing this module anywhere before the pipeline loads is early enough.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import Any

if sys.platform != "win32" and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

log = logging.getLogger(__name__)

# (prompt_embeds, negative_prompt_embeds or None without CFG)
Embeds = tuple[torch.Tensor, torch.Tensor | None]
GraphKey = tuple[Any, ...]


class CapturedStep:
    """One UNet forward captured as a CUDA graph for fixed input shapes.

    Replay copies the live inputs into the static buffers captured with the graph and returns the static output,
    which the caller must consume before the next replay (the diffusion loop does).
    """

    def __init__(
        self,
        forward: Callable[..., Any],
        sample: torch.Tensor,
        timestep: torch.Tensor,
        embeds: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        device = sample.device
        self.sample, self.timestep, self.embeds = sample.clone(), timestep.clone(), embeds.clone()
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(2):  # warm cuBLAS/cuDNN workspaces on a side stream before capturing
                forward(self.sample, self.timestep, encoder_hidden_states=self.embeds, **kwargs)
        torch.cuda.current_stream(device).wait_stream(side)
        before = torch.cuda.memory_allocated(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.out = forward(self.sample, self.timestep, encoder_hidden_states=self.embeds, **kwargs)[0]
        self.nbytes = int(torch.cuda.memory_allocated(device) - before)

    def __call__(self, sample: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        self.sample.copy_(sample)
        self.timestep.copy_(timestep)
        self.embeds.copy_(embeds)
        self.graph.replay()
        return self.out


class MemoryPool:
    """Reusable latent, noise, prompt-embed and captured-step buffers for `size`x`size` SD1.5 generations."""

    def __init__(
        self, size: int, batch: int = 1, device: str = "cuda", dtype: torch.dtype = torch.float16
    ) -> None:
        # ponytail: SD1.5 latent geometry (4 channels, /8) is fixed here; read it from unet/vae configs for SDXL.
        shape = (batch, 4, size // 8, size // 8)
        self.size, self.batch, self.dtype = size, batch, dtype
        self.device = torch.device(device)
        self.latents = torch.zeros(shape, device=self.device, dtype=dtype)
        self.noise = torch.zeros(shape, dtype=torch.float32, pin_memory=self.device.type == "cuda")
        self.prompt_embeds: dict[tuple[str, ...], Embeds] = {}
        # ponytail: one captured graph per (adapters, padding, shape), each holding its own activation memory and
        # never evicted; share one mempool or LRU-evict once many biome LoRAs are resident at once.
        self.steps: dict[GraphKey, CapturedStep | None] = {}

    def seeded_latents(self, seed: int) -> torch.Tensor:
        """Refill the device latents in place with N(0,1) noise from a CPU generator seeded with `seed`."""
        self.noise.normal_(generator=torch.Generator().manual_seed(seed))
        self.latents.copy_(self.noise)
        return self.latents

    def embeds(self, key: tuple[str, ...], make: Callable[[], Embeds]) -> Embeds:
        """Return the cached prompt embeds for `key`, computing them with `make` on first use."""
        if key not in self.prompt_embeds:
            self.prompt_embeds[key] = make()
        return self.prompt_embeds[key]

    def step(
        self,
        key: GraphKey,
        forward: Callable[..., Any],
        sample: torch.Tensor,
        timestep: torch.Tensor,
        embeds: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run one UNet step through the graph captured for `key`, capturing it on first use.

        A capture that fails (an op the graph cannot record) is logged once and that key falls back to eager.
        """
        if key not in self.steps:
            try:
                self.steps[key] = CapturedStep(forward, sample, timestep, embeds, **kwargs)
                log.info("captured UNet step %s (%.0f MiB)", key, self.steps[key].nbytes / 2**20)
            except Exception:  # noqa: BLE001 - any capture failure means eager for this key
                log.warning("CUDA graph capture failed for %s; running this step eagerly", key, exc_info=True)
                self.steps[key] = None
        captured = self.steps[key]
        if captured is None:
            return forward(sample, timestep, encoder_hidden_states=embeds, **kwargs)[0]
        return captured(sample, timestep, embeds)

    def stats(self) -> dict[str, int]:
        """Bytes held by each buffer plus the number of cached prompt-embed entries and captured graphs."""
        embed_bytes = sum(t.nbytes for pair in self.prompt_embeds.values() for t in pair if t is not None)
        graph_bytes = sum(s.nbytes for s in self.steps.values() if s is not None)
        latent_bytes, noise_bytes = int(self.latents.nbytes), int(self.noise.nbytes)
        return {
            "latents_bytes": latent_bytes,
            "noise_bytes": noise_bytes,
            "embed_bytes": int(embed_bytes),
            "embeds": len(self.prompt_embeds),
            "graphs": sum(s is not None for s in self.steps.values()),
            "graph_bytes": int(graph_bytes),
            "total_bytes": latent_bytes + noise_bytes + int(embed_bytes) + int(graph_bytes),
        }
