# 05 — GPU measurements: SD1.5 + Hyper-SD generator, memory pool, biome LoRA (2026-10-06)

Raw numbers only, per ADR-0005. Everything below was measured on this machine with the commands shown; nothing is
quoted from a model card or estimated. Scripts live outside the package (the benchmark proper is Task 7).

## Environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227 MiB (11.94 GiB), driver 595.97, on AC power |
| CPU / OS | Intel Core Ultra 9 285H, Windows 11 Home 10.0.26200 |
| Python / torch | 3.14.3, torch 2.11.0+cu128 (CUDA 12.8, cuDNN 91900), allocator backend `native` |
| diffusers / peft / transformers | 0.41.0 / 0.21.2 / 5.18.0 |
| models | `stable-diffusion-v1-5/stable-diffusion-v1-5` (cached fp32 snapshot, cast to fp16 at load), `ByteDance/Hyper-SD` `Hyper-SD15-4steps-lora.safetensors` + `Hyper-SD15-8steps-CFG-lora.safetensors` (128.3 MiB fp16 each, rank 64), `facebook/dinov2-small` |
| `expandable_segments` | **not available**: torch 2.11 on Windows prints `expandable_segments not supported on this platform` and ignores it; `pool.py` only exports it off Windows |
| `torch.compile` | **not measured**: no triton on this machine and the venv's torch lacks `functorch` (`torch._dynamo` raises `ModuleNotFoundError`), so `compile=True` was not exercised |

## Settings verified against the Hyper-SD model card

Read from `huggingface.co/ByteDance/Hyper-SD` (raw README, 2026-10-06):

| LoRA | card recipe | used here |
|---|---|---|
| `Hyper-SD15-{2,4,8}steps-lora` | `DDIMScheduler.from_config(..., timestep_spacing="trailing")`, `guidance_scale=0`, no `eta` | **draft** = 4-step LoRA, DDIM trailing, guidance 1.0 (diffusers disables CFG for any `guidance_scale <= 1`, so 1.0 and 0 are the same computation), eta 0 |
| `Hyper-SD15-1step-lora` (unified) | `TCDScheduler`, `eta=1.0` ("lower eta results in more detail for multi-steps inference") | not used; TCD/eta is only recommended for the unified LoRA, not for the fixed-step files |
| `Hyper-SD15-8steps-CFG-lora` | "support 5~8 guidance scales" (no SD1.5 code block on the card) | **refine** = 8-step CFG LoRA, DDIM trailing, guidance 5.0, eta 0, `steps = max(spec.steps, 8)` |

Deviation from the task brief: the brief suggested a TCD/LCM-style scheduler for the draft tier; the card prescribes
DDIM(trailing) for the fixed-step LoRAs, so that is what runs.

## Sprite background: measured, not prompted

`build_prompt` ends the sprite prompt with the biome style (`"... white background, top-down 2D game texture,
painterly fantasy forest, ..."`), and SD1.5 paints a full forest scene regardless of tier or CFG. The generator
therefore starts sprites from `noise * (1 - b) + b * z_white`, where `z_white` is the scaled VAE latent of a white
canvas. Sweep over `b` (forest props mushroom/log/fern, seeds 0-2, 512 px; *border<40* = share of edge pixels within
RGB distance 40 of white, *corner keyed* = images whose (0,0) pixel keys as background at tol 40, *interior white* =
share of interior pixels that are white, 1.00 means the prop vanished):

| variant | border<40 | corner keyed | interior white |
|---|---|---|---|
| no bias, draft 4 steps, prompt tail "isolated on a plain white background" | 0.04 | 0/9 | - |
| no bias, refine 8 steps CFG 5 + scenery negative | 0.01 | 0/9 | - |
| mix b=0.05, draft | 0.78 | 9/9 | 0.74 |
| mix b=0.05, refine CFG 5 | 0.92 | 9/9 | 0.74 |
| mix b=0.10, draft | 0.88 | 8/9 | 0.91 |
| mix b=0.10, draft, no prompt tail | 0.90 | 9/9 | 0.91 |
| mix b=0.15, draft | 0.97 | 9/9 | 0.99 (blank) |
| mix b=0.20, draft | 0.99 | 9/9 | 1.00 (blank) |
| additive b=0.10, draft | 0.91 | 9/9 | 0.94 |
| additive b=0.20 / 0.30, draft | 1.00 / 0.98 | 9/9 / 6/9 | 1.00 (blank) |

`_WHITE_BIAS = 0.05` (mix) is the operating point; the prompt tail changes nothing measurable, so prompts stay as
`build_prompt` builds them. The remaining non-white edge pixels are props touching the border; `alpha_from_white`
then keys the background at its default tolerance.

## Where the time and memory go (eager pipeline, draft adapter, 512 px, medians of 10 after warm-up)

| stage | ms | note |
|---|---|---|
| UNet step, circular padding, draft LoRA unfused (as shipped before graphs) | 123.5 | PEFT dispatch for rank-64 LoRA on 4 projections of every attention block |
| UNet step, zeros padding | 104.8 | circular pad copies cost ~15 % |
| UNet step, adapters disabled | 57.9 | |
| UNet step, draft LoRA fused (`fuse_lora`) | 59.8 | fusing recovers it but tier/biome switching would fuse/unfuse fp16 weights per switch |
| UNet step, SDPA backend forced: efficient / math / flash / cuDNN | 116.4 / 170.6 / unavailable / unavailable | flash and cuDNN attention raise on this sm_120 build |
| UNet step, `cudnn.benchmark=True` | 110.2 | |
| UNet step, **manual `torch.cuda.CUDAGraph` replay** | **54.6** | capture 0.14 s; `max|out - eager| = 0.0000` at t=999 and t=749 |
| VAE decode 512 px, circular / zeros | 114.7 / 108.7 | GPU-bound; left eager |
| postprocess + `.cpu().numpy()` | 0.3 | |
| `pipe()` 4 steps, latent out / pt out; `generate()` | 505.0 / 585.6 / 594.2 | |

Peak-memory transients per stage (pool on, same process): `encode_prompt` 0.00 GB, UNet batch 1 0.08 GB, UNet
batch 2 (CFG) 0.17 GB, VAE decode 0.72 GB, whole `generate()` 0.72 GB for both tiers. The VAE decode sets the peak.

`nvidia-smi` sampled every 5 s during sustained eager generation: 46-53 % GPU utilisation at 94-98 W, 2.3-2.6 GHz,
74 °C, no throttle reasons active, i.e. the eager loop was about half CPU-bound. Replaying the UNet step from a CUDA
graph is what the memory pool does now (`MemoryPool.step`, one graph per (adapters, padding, shape)).

Two regressions found and fixed on the way (first full pass, eager, separate processes):

| run | draft p50 / p95 s | refine p50 / p95 s | peak alloc / reserved GB |
|---|---|---|---|
| pool on, eager UNet, embeds cached outside `inference_mode` | 0.758 / 0.905 | 1.400 / 2.356 | 3.72 / 4.37 |
| pool off, eager UNet | 0.652 / 0.814 | 1.478 / 2.192 | 3.03 / 3.67 |

1. Resident memory after warm-up was 3.00 GB with the pool vs 2.28 GB without: every cached prompt embed had
   been encoded outside `inference_mode`, so each one pinned the text encoder's autograd state (18 prompts, 0.72
   GB). `_embeds` now encodes under `inference_mode`.
2. Latency: the in-process A/B (same instance, pool toggled, 10 textures per arm, 3 rounds) gave p50 0.63-0.81 s in
   *both* arms, so the eager pool was neither faster nor slower; the cost was the unfused PEFT LoRA path (table
   above), which the graph replay removes without fusing.

Reference point, same machine and loop, SD1.5 + `latent-consistency/lcm-lora-sdv1-5`, LCMScheduler, 4 steps,
guidance 1.0, circular padding (the pre-rewrite configuration): p50 **1.293 s unfused**, **0.445 s fused**, peak
2.88 GB. The 0.61 s baseline quoted in ADR-0001 is therefore a fused-LoRA number.

## Results (final code: graph-captured pool, embeds under `inference_mode`)

20 warm 512 px forest textures per row, seeds 100-119, subjects cycling grass/dirt/water/shore/tree/rock. Pipeline
load 18.4-18.5 s in every run (fp32 snapshot cast to fp16); `warmup("forest")` 4.3 s pool on, 3.7 s pool off, 5.1 s
with the forest LoRA (18 prompt embeds, 3 captured UNet graphs: draft texture, refine texture, sprite).

| config | tier | steps | p50 s | p95 s | peak allocated GB | peak reserved GB | `num_device_alloc` over 20 images | `num_alloc_retries` over 20 |
|---|---|---|---|---|---|---|---|---|
| pool on | draft | 4 | **0.382** | 0.402 | 3.04 | 4.45 | 1 (first image after warm-up, then 0) | 0 |
| pool on | refine | 8 | **0.932** | 0.992 | 3.04 | 4.45 | 0 | 0 |
| pool off (eager UNet, embeds re-encoded, fresh device noise) | draft | 4 | 0.582 | 0.641 | 3.00 | 3.62 | 0 | 0 |
| pool off | refine | 8 | 1.171 | 1.208 | 3.00 | 3.62 | 0 | 0 |
| pool on + forest LoRA stacked (`biome_loras`) | draft | 4 | 0.374 | 0.386 | 3.04 | 4.45 | 1 (first image, then 0) | 0 |
| pool on + forest LoRA stacked | refine | 8 | 0.957 | 0.968 | 3.04 | 4.45 | 0 | 0 |

- Against the pre-rewrite floor (0.61 s/asset, 3.09 GB peak, ADR-0001): draft p50 0.382 s and 3.04 GB peak
  allocated. Peak *reserved* is 0.8 GB higher with the pool on because each captured graph keeps its private
  activation pool (`graph_bytes` in `pool.stats()` counts only the live static buffers, 9.7 MB for 3 graphs).
- Graph replay vs eager UNet on the same pool latents and embeds: images **identical** (max abs diff 0) in both
  pool-on runs; a single eager `generate()` in the same process took 0.53 s / 0.67 s.
- The one `num_device_alloc` on the first draft image after warm-up follows the sprite generated last in
  `warmup` (zeros padding, different block sizes); the next 19 images and the whole refine row allocate nothing
  from the device. `num_alloc_retries` is 0 everywhere.
- Pool buffers: latents 32 KB (fp16 device), noise 64 KB (pinned fp32 host), 18 embeds 3.19 MB.

Tileability (`realmweaver.metrics.quality.tileability`, 1.0 = seam looks like interior; 10 textures each, seeds
200-209, mean with min-max):

| | seamless on, draft | seamless off, draft | seamless on, refine |
|---|---|---|---|
| no biome LoRA | 1.005 (0.696-1.192) | 4.613 (1.255-9.001) | 1.005 (0.813-1.382) |
| forest LoRA stacked | 1.010 (0.765-1.229) | 3.766 (1.470-5.677) | 1.021 (0.709-1.229) |

One refine texture without the LoRA measured 1.382; every other seamless value is below 1.3 and every
`seamless=False` value is above it.

### Forest biome LoRA

`train_biome_lora("forest", Path("data/raw/textures/fantasy_forest"), Path("models/lora/forest"), steps=200)`:
30 images (ancient_bark, enchanted_leaves, mystical_moss x10) plus their mirrors = 60 pre-encoded latents, captions
from `build_prompt` with the file name as subject (`"ancient bark, top-down 2D game texture, painterly fantasy
forest, soft diffuse light, rich detail, seamless tileable texture"`), rank 8 on `to_q,to_k,to_v,to_out.0`, batch 2,
AdamW lr 1e-4, fp16 autocast with GradScaler, circular padding on during training.

- Wall-clock: **71.2 s** for 200 steps including the pipeline load and pre-encoding (about 6.3 s per 20 steps).
- Mean loss per 20-step window: 0.1746, 0.1752, 0.2081, 0.1428, 0.1705, 0.1609, 0.1657, 0.2023, 0.1895, 0.1341.
  Flat within the noise of random-timestep epsilon MSE; 200 steps is a smoke-level run, not a converged style
  adapter.
- Output: `models/lora/forest/pytorch_lora_weights.safetensors` + `train_log.json`; it loads through
  `DiffusionGenerator(biome_loras={"forest": "models/lora/forest"})` as adapter `forest` stacked on `draft`/`refine`
  (rows above), with tileability unchanged within noise.

### Tests

`uv run pytest tests/test_seamless_cpu.py -v`: 2 passed (4.9 s, dominated by importing torch); the whole CPU suite
`uv run pytest -q -m "not gpu"`: 72 passed in 12.1 s wall (note: plain `pytest -q` on a CUDA machine also runs the
`gpu`-marked file, which downloads models and takes minutes).
`uv run pytest tests/test_assets_diffusion_gpu.py -v -m gpu`: 8 passed (seamless draft texture < 1.3 and < 3 s,
same seed same image, refine CFG seamless, sprite corner alpha 0, zero `num_device_alloc`/retries on the second
image, embed reuse and 3 captured graphs, DINOv2 384-d unit norm, 3-step LoRA train/save/load).

## Commands

All from the repo root, each measurement in its own process (so the allocator counters of one run never include
another resident pipeline); `--no-sync` keeps `uv` from re-syncing the venv mid-run.

```
uv run --no-sync python measure_gpu.py --pool on  --out measure_pool_on.json
uv run --no-sync python measure_gpu.py --pool off --out measure_pool_off.json
uv run --no-sync python train_lora.py --steps 200
uv run --no-sync python measure_gpu.py --pool on --lora models/lora/forest --out measure_lora.json
```

`measure_gpu.py` builds `DiffusionGenerator(pool=...)`, calls `warmup("forest")`, then per tier resets the allocator
counters and generates 20 textures (subjects cycling grass/dirt/water/shore/tree/rock, seeds 100-119; draft 4 steps,
refine 8 steps); it records `Asset.latency_s` (p50 = median, p95 = the 19th of 20 sorted), `max_memory_allocated`
and `max_memory_reserved` over the 20, and the per-image deltas of `num_device_alloc` and `num_alloc_retries` from
`torch.cuda.memory_stats()`. With the pool on it also checks that graph replay reproduces the eager UNet (same pool
latents and embeds) and measures `tileability` on 10 textures (seeds 200-209) with `seamless=True` and
`seamless=False`. `train_lora.py` calls `train_biome_lora("forest", Path("data/raw/textures/fantasy_forest"),
Path("models/lora/forest"), steps=200)` and prints the wall-clock and the mean loss per 20 steps from
`models/lora/forest/train_log.json`.

## Appendix: the two scripts, verbatim (kept out of the package; Task 7 owns the benchmark)

### measure_gpu.py

```python
"""GPU measurements for docs/research/05-gpu-measurements.md.

Usage (from the repo root):  uv run python measure_gpu.py --pool on|off [--lora models/lora/forest] [--out x.json]
Prints a markdown table and writes the raw numbers as JSON. Run pool on and off in separate processes so the
allocator counters and peaks are not polluted by a second resident pipeline.
"""

import argparse
import json
import platform
import statistics
import sys
import time

import numpy as np
import torch

from realmweaver.assets.diffusion import DiffusionGenerator
from realmweaver.metrics.quality import tileability
from realmweaver.types import AssetSpec

SUBJECTS = ["grass", "dirt", "water", "shore", "tree", "rock"]
N = 20


def run_tier(gen: DiffusionGenerator, tier: str, steps: int) -> dict:
    gen.reset_stats()
    lat, allocs, retries = [], [], []
    for i in range(N):
        before = gen.allocator_stats()
        a = gen.generate(AssetSpec("forest", subject=SUBJECTS[i % len(SUBJECTS)], seed=100 + i, tier=tier, steps=steps))
        after = gen.allocator_stats()
        lat.append(a.latency_s)
        allocs.append(after["num_device_alloc"] - before["num_device_alloc"])
        retries.append(after["num_alloc_retries"] - before["num_alloc_retries"])
    s = gen.allocator_stats()
    lat_sorted = sorted(lat)
    return {
        "n": N,
        "steps": steps,
        "p50_s": statistics.median(lat),
        "p95_s": lat_sorted[int(round(0.95 * (N - 1)))],
        "mean_s": statistics.fmean(lat),
        "peak_allocated_gb": s["max_memory_allocated"] / 2**30,
        "peak_reserved_gb": s["max_memory_reserved"] / 2**30,
        "num_device_alloc_per_image": allocs,
        "num_alloc_retries_per_image": retries,
    }


def run_tileability(gen: DiffusionGenerator, seamless: bool, tier: str = "draft") -> dict:
    vals = []
    for i in range(10):
        a = gen.generate(AssetSpec("forest", subject=SUBJECTS[i % len(SUBJECTS)], seed=200 + i, seamless=seamless, tier=tier))
        vals.append(float(tileability(a.image)))
    return {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals), "values": vals}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["on", "off"], default="on")
    ap.add_argument("--lora", default=None, help="biome LoRA dir for forest (adds a LoRA row)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    gen = DiffusionGenerator(pool=args.pool == "on", biome_loras={"forest": args.lora} if args.lora else None)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    gen.warmup("forest")
    warm_s = time.perf_counter() - t0

    res = {
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "allocator_backend": torch.cuda.get_allocator_backend(),
        },
        "pool": args.pool,
        "lora": args.lora,
        "load_s": load_s,
        "warmup_s": warm_s,
        "draft": run_tier(gen, "draft", 4),
        "refine": run_tier(gen, "refine", 8),
    }
    if args.pool == "on":
        res["pool_stats"] = gen.pool.stats()
        # graph replay must reproduce the eager UNet bit for bit (same pool latents, same embeds)
        probe = AssetSpec("forest", subject="tree", seed=11)
        graphed = gen.generate(probe).image
        gen._pipe.unet.forward = gen._eager_unet
        t0 = time.perf_counter()
        eager = gen.generate(probe).image
        eager_s = time.perf_counter() - t0
        gen._pipe.unet.forward = gen._graphed_unet
        res["graph_vs_eager"] = {"identical": bool(np.array_equal(graphed, eager)), "max_abs_diff": int(np.abs(graphed.astype(int) - eager.astype(int)).max()), "eager_generate_s": eager_s}
        res["tileability_seamless_on"] = run_tileability(gen, True)
        res["tileability_seamless_off"] = run_tileability(gen, False)
        res["tileability_refine_seamless_on"] = run_tileability(gen, True, tier="refine")
    import diffusers, peft, transformers  # noqa: E401

    res["env"].update({"diffusers": diffusers.__version__, "peft": peft.__version__, "transformers": transformers.__version__})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    def row(name: str, r: dict) -> str:
        da, ra = r["num_device_alloc_per_image"], r["num_alloc_retries_per_image"]
        return (
            f"| {name} | {r['steps']} | {r['p50_s']:.3f} | {r['p95_s']:.3f} | {r['peak_allocated_gb']:.2f} | "
            f"{r['peak_reserved_gb']:.2f} | {sum(da)} ({max(da)} max) | {sum(ra)} |"
        )

    print(f"\npool={args.pool} lora={args.lora} load {load_s:.1f}s warmup {warm_s:.1f}s  {res['env']}")
    print("| tier (pool " + args.pool + ") | steps | p50 s | p95 s | peak alloc GB | peak reserved GB | num_device_alloc over 20 | num_alloc_retries over 20 |")
    print("|---|---|---|---|---|---|---|---|")
    print(row("draft", res["draft"]))
    print(row("refine", res["refine"]))
    if args.pool == "on":
        for k in ("tileability_seamless_on", "tileability_seamless_off", "tileability_refine_seamless_on"):
            t = res[k]
            print(f"{k}: mean {t['mean']:.3f} min {t['min']:.3f} max {t['max']:.3f}  values {np.round(t['values'], 3).tolist()}")
        print("pool stats:", res["pool_stats"])
        print("graph vs eager:", res["graph_vs_eager"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### train_lora.py

```python
"""Train the forest biome LoRA and print the wall-clock plus the loss trend (mean per 20-step window).

Usage (from the repo root):  uv run python train_lora.py [--steps 200]
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

from realmweaver.assets.lora import train_biome_lora

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--out", default="models/lora/forest")
args = ap.parse_args()

out = train_biome_lora("forest", Path("data/raw/textures/fantasy_forest"), Path(args.out), steps=args.steps)
rec = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
loss = rec["loss"]
print(f"\nimages={rec['images']} rank={rec['rank']} steps={rec['steps']} batch={rec['batch']} lr={rec['lr']} wall={rec['wall_s']:.1f}s")
print("| steps | mean loss |")
print("|---|---|")
for i in range(0, len(loss), 20):
    print(f"| {i + 1}-{min(i + 20, len(loss))} | {statistics.fmean(loss[i : i + 20]):.4f} |")
print("saved:", sorted(p.name for p in out.iterdir()))
```
