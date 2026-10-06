# ADR-0005: Every number is a benchmark output; KID+FID at fixed n, p50/p95, allocator counters

Date: 2026-10-06. Status: accepted.

## Context
The original pitch quoted FID 32.4, 2.7 s/asset, 120 textures/min, 94% tileability, 0.89 style consistency, 31 FPS and 6.2 GB VRAM with no measurement anywhere in the repo.

## Decision
`realmweaver bench` writes `reports/bench-<timestamp>.json` with environment, versions, seeds, n, and: KID (subset mean ± std) and FID at fixed n against `data/raw/textures`; tileability as seam/interior gradient ratio plus Tiling Score; style consistency as within- minus cross-biome DINO cosine; latency p50/p95 after warm-up; assets per minute over a timed run; `max_memory_allocated`, `max_memory_reserved`, `num_alloc_retries`, `num_device_alloc` with the memory pool on and off; predictor hit-rate vs the 8-ring baseline over ≥20 synthetic-walk seeds. README and dashboard show only values read from a report, beside the original targets.

## Consequences
The reference set is the repo's 240 SDXL-Turbo textures; the caveat is recorded in every report. Unity FPS is out of scope until a Unity build exists.
