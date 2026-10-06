# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

delegated (inferred, user asked for full autonomy): Python 3.14 + FastAPI backend; one static HTML/CSS/JS operator page served by the backend, no front-end build step; Unity 6 C# client scripts consume the same API.

## Users

(inferred from brief and repo) Small indie game teams and technical artists building 2D tile-based worlds in Unity on a single consumer NVIDIA GPU (6–12 GB VRAM), who need many biome-consistent textures and sprites without hand-authoring each one. Secondary audience: reviewers and recruiters evaluating the author's portfolio project through its dashboard, benchmarks and pitch video.

## Product Purpose

RealmWeaver (MADWE, "Multi-Agent Diffusion World Engine") turns a biome description into coherent, tileable game assets and a constraint-valid tile layout in seconds, streams them into Unity, and reports measured quality and speed. Success means: an asset in a few seconds on a laptop GPU, a layout with zero adjacency violations, visible style coherence across a biome, and every number on the dashboard produced by a reproducible benchmark.

## Positioning

Not another text-to-image button. The product is the coordination layer around few-step diffusion: a world state graph that keeps biome style coherent across chunks, a predictor that pre-warms the chunks the player is likely to reach, pooled GPU memory that keeps latency flat, and a Unity bridge that maps results to prefabs. Claims ship with a benchmark report, never as prose.

## Operating Context

Windows 11 laptop with an NVIDIA GPU; Hugging Face model cache; a Unity 6 project consuming assets over a local WebSocket/HTTP bridge; CLI commands for generate / layout / bench; JSON benchmark reports under `reports/`; GitHub Actions CI running the CPU test suite.

## Capabilities and Constraints

- Biomes are the unit of style: forest, desert, snow, volcanic, underwater, sky (repo configs) and dataset categories such as fantasy_forest, cyberpunk_city, steampunk_industrial.
- Terminology: biome, tile, chunk, asset, adjacency rule, world state graph, prewarm, style coherence, tileability.
- Hard constraints: must run within a 6–12 GB VRAM budget; few-step inference; models must carry licenses that permit the use; no gated models that need manual terms acceptance for the default path.
- Metrics that must be real measurements: FID/KID against the repo dataset, tileability seam score, CLIP style-consistency, seconds per asset, assets per minute, peak VRAM, allocation overhead with and without pooling.
- Undecided (pending research notes in `docs/research/`): primary diffusion model tier and optional quality tier.

## Brand Commitments

Name: RealmWeaver, subtitle MADWE. Authors listed in README: Ankit Gole and Shreya Boyane. No logo asset exists; none should be invented as a brand claim.

## Evidence on Hand

- ~1,000 generated texture and sprite PNGs under `data/raw` and `data/processed` (train/val/test splits) with biome-style category folders.
- Biome adjacency configs under `configs/biomes/` (currently permissive placeholders).
- No benchmark results exist yet. The figures in the original project pitch (73% effort reduction, FID 32.4, 2.7 s/asset, 120 textures/min, 94% tileability, 0.89 style consistency, 31 FPS, 6.2 GB VRAM) are targets, not evidence, and must not appear on any surface until measured.

## Product Principles

1. Measured, not claimed: every performance or quality number on a surface links to a benchmark report.
2. Coherence over single-image quality: a biome reads as one world.
3. Latency budget first: interactive generation on a laptop GPU beats slower, prettier output.
4. Smallest surface that works: one page, one API, one Unity script set.
5. Honest degradation: when the GPU is absent, the system runs with a procedural fallback and says so.

## Accessibility & Inclusion

Operator page must be keyboard-navigable, colour never the sole signal for status, and text contrast meets WCAG AA.
