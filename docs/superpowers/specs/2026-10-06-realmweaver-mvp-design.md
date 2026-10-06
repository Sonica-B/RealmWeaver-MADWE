# RealmWeaver MVP — Design Spec (2026-10-06)

Status: self-approved under the user's "perform autonomously" instruction; every assumption is marked *(assumed)*. Vocabulary: `GLOSSARY.md`. Research: `docs/research/00-codebase-audit.md`, `docs/research/01-model-landscape-2026.md`, `docs/research/02-pcg-and-systems-literature.md`.

## Problem Statement

A small team building a 2D tile world in Unity needs hundreds of biome-consistent, seamless textures and props, plus valid tile layouts, and cannot hand-author them. The 2025 repo promised this (multi-agent diffusion, WFC, a world state graph, predictive generation, memory pooling, a Unity bridge, measured FID/latency) but shipped stubs: no tests, a permissive WFC over tiles that no longer exist, an unframed TCP stub, empty graph and prediction packages, three different base models, a leaked token, and no measurements (audit: roughly 20% of scope, nothing runs end to end).

## Solution

One Python package, `realmweaver`, with a CLI, a FastAPI bridge, an operator page and a Unity client script set:

- **Asset agent** — SD1.5 at 512px with a few-step LoRA (Hyper-SD15 / TCD, 4–8 steps) and stacked rank-8 biome LoRAs; seamless textures via circular padding; a procedural fallback adapter for CPU/tests. Optional quality tier: FLUX.2-klein-4B behind the same seam.
- **Layout agent** — own numpy WFC (simple tiled model, AC-3 propagation, min-entropy, restart-then-shrink on contradiction), adjacency and weights learned from a per-biome ASCII example map, chunked with border constraints.
- **World state graph** — networkx typed property graph (World → Region → Chunk → Tile → Asset), DINOv2-small style vectors, coherence score with regenerate-then-anchor fallback. The graph is the source of truth; JSON on disk.
- **Predictor + scheduler** — order-2 Markov over 8 headings, priority = P(visit)/cost, ≤2 in flight, byte-capped LRU, draft-then-refine tiers.
- **Memory pool** — resident weights, static latent/noise/embedding buffers, optional `torch.compile(mode="reduce-overhead")`; allocator counters reported before/after.
- **Bridge** — HTTP PNG + chunk JSON, WebSocket player position and ready events; Unity `AssetStreamer` + `TileCatalog` prefab map.
- **Metrics** — tileability (seam/interior gradient ratio, Tiling Score), style consistency (within- minus cross-biome DINO cosine), KID + FID at fixed n, p50/p95 latency, assets/min, peak VRAM, allocator stats, predictor hit-rate vs 8-ring baseline. One JSON report per run under `reports/`; README and dashboard only show numbers from a report.

Agents are **roles** (asset, layout, critic, predictor), not a class hierarchy. Each role is a module with one interface.

## User Stories

1. As a technical artist, I want `realmweaver generate --biome forest --tile grass`, so that I get a seamless 512px texture in about a second.
2. As a technical artist, I want every texture to tile, so that I never see seams on a tilemap.
3. As a technical artist, I want a biome LoRA trained from a folder of reference images, so that generated assets match my art direction.
4. As a level designer, I want `realmweaver layout --biome forest --size 32`, so that I get a valid tile grid with no illegal adjacencies.
5. As a level designer, I want to author layout rules as a small ASCII example map, so that I never edit 10,000-line JSON.
6. As a level designer, I want chunk borders to match their neighbours, so that an endless world has no cracks.
7. As a Unity developer, I want a C# script that fetches a chunk and its textures, so that the world appears in my scene without manual import.
8. As a Unity developer, I want tile classes mapped to prefabs in a ScriptableObject, so that I control what each class instantiates.
9. As a Unity developer, I want generation to stay in Python, so that the game process never loads a diffusion model.
10. As a player, I want chunks ahead of me generated before I reach them, so that I do not see pop-in or stalls.
11. As an engine owner, I want a world state graph on disk, so that the same seed and history reproduce the same world.
12. As an engine owner, I want assets that drift from the biome style to be regenerated, so that a biome reads as one world.
13. As an engine owner, I want a memory pool, so that allocation overhead and peak VRAM are lower and latency is flat.
14. As an engine owner, I want `realmweaver bench`, so that FID/KID, tileability, style consistency, latency, throughput and VRAM are measured with seeds and counts.
15. As a reviewer, I want the README to show measured numbers next to the original targets, so that claims are honest.
16. As an operator, I want one web page to generate, inspect assets, see the world map and read the last benchmark, so that I can demo without a terminal.
17. As an operator, I want the system to run without a GPU using the procedural adapter and say so, so that tests and demos never fake GPU output.
18. As a maintainer, I want CPU-only tests in CI, so that every PR is checked without a GPU runner.
19. As a maintainer, I want a Docker image with GPU access, so that the whole stack runs identically on another machine.
20. As a maintainer, I want `uv sync` to produce the environment, so that setup is one command.
21. As a security-conscious owner, I want no secrets in the repo, so that the committed token incident does not repeat.
22. As a developer, I want a quality tier behind the same generator seam, so that I can swap in FLUX.2-klein without touching callers.
23. As a developer, I want sprites with a transparent background, so that props drop onto tiles.
24. As a developer, I want request priorities, so that the chunk under the player always wins over prewarm.
25. As a developer, I want ready events over WebSocket, so that Unity swaps placeholders without polling.

## Implementation Decisions

**Package and tooling.** Single package `realmweaver` (flat, importable), `pyproject.toml` with uv, Python ≥3.11, optional extra `gpu`. Ruff replaces black/flake8. CLI `realmweaver` via a small argparse module. Docker image from `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, compose with `gpus: all`, HF cache and `reports/` mounted. CI: ruff + pytest on CPU. The old `src/`, `scripts/`, `configs/biomes/*.json`, `setup.py`, `requirements.txt` and `setup_environment.bat` are deleted; the 48 prompt strings from the generator scripts are mined into the biome YAMLs; `data/raw` stays as the FID/KID reference set and `data/processed` leaves git.

**Types (one shared module).** `AssetSpec(biome, kind, subject, size, seed, steps, seamless, tier)`, `Asset(id, spec, image uint8 HxWx3|4, style_vec optional)`, `TileSet(classes, allowed[T,4,T], weights)`, `Layout(grid int HxW, tileset)`, `Chunk(cx, cy, biome, layout, asset_ids, state)`. Asset id = sha1 of the canonical spec.

**Generator seam.** `Generator` protocol with one method `generate(spec) -> Asset`. Adapters: `DiffusionGenerator` (SD1.5 fp16, few-step LoRA, biome LoRA stacking with fixed max rank, circular padding when `spec.seamless`, memory pool, draft 4 steps / refine 8 steps), `ProceduralGenerator` (deterministic value-noise + biome palette, CPU, seamless by construction), and later `KleinGenerator` for the quality tier. Everything else only sees the protocol.

**Biome definition.** `realmweaver/biomes/<name>/biome.yaml` (prompt style, negative prompt, per-tile-class subjects, palette for the procedural adapter, optional LoRA path, coherence threshold) including a `map:` block (12×12 ASCII example map) and its legend. Six biomes ship: forest, desert, snow, volcanic, underwater, sky *(assumed from the repo configs)*.

**WFC.** `TileSet.from_example(map_text, legend)` learns `allowed` and weights. `solve(tileset, w, h, seed, fixed={})` runs min-entropy observe + AC-3 propagate with boolean matrix ops; contradiction → restart with new seed (≤5) → shrink to sub-block. `solve_chunk` copies the touching row/column of ready neighbours as `fixed` cells. No PyPI WFC dependency (ADR-0002).

**World state graph.** networkx `MultiDiGraph`, node types World/Region/Chunk/Tile/Asset, edges CONTAINS/ADJACENT/INSTANCE_OF/STYLE_ANCHOR. Style vector = DINOv2-small embedding (`facebook/dinov2-small`) via an injected `embed(image) -> np.ndarray`; tests inject a colour-histogram embedder. Coherence = 0.6·cos(asset, region style) + 0.4·mean cos(asset, adjacent assets); below the biome threshold → regenerate (≤2) → anchor fallback (reuse the region's best asset). Persisted as JSON node-link; `World.load/save`.

**Prediction and scheduling.** `Predictor.observe(pos)`; `Predictor.rank(current_chunk) -> [(chunk, p)]` from order-2 Markov over 8 headings with order-1 then constant-velocity fallback. `Scheduler` priority = p / estimated cost; ≤2 in flight; byte-capped LRU with distance-aware eviction; draft first, refine when idle.

**Memory pool.** `MemoryPool(size, batch)` holds latent/noise/prompt-embedding tensors reused via `copy_()`; pipeline keeps weights resident (no CPU offload); allocator settings `expandable_segments:True`; bench records `num_alloc_retries`, `num_device_alloc`, `max_memory_allocated`, reserved−allocated gap per image with pool on and off.

**Bridge.** FastAPI: `GET /health`, `GET /biomes`, `POST /generate` → asset id, `GET /asset/{id}.png`, `GET /chunk/{cx}/{cy}` → tiles, asset ids, prefab map, `POST /player` → prewarm, `GET /report` → latest benchmark JSON, `WS /events` → `{type: ready, chunk, asset}`; `/` serves the operator page. Unity: `RealmWeaverClient.cs` (UnityWebRequestTexture, non-readable textures), `AssetStreamer.cs` (priority queue, ≤4 coroutines, placeholder, byte-capped LRU with Destroy), `TileCatalog.cs` (ScriptableObject tile class → prefab), `RealmWeaverEvents.cs` (NativeWebSocket). Unity cannot be compiled on this machine; the protocol is verified from Python and the scripts are reviewed against Unity 6.3 docs. The `origin/unity` branch's C# is evaluated for salvage first (`docs/research/03-origin-unity-salvage.md`).

**Metrics.** `tileability(img)` = wrapped seam gradient / interior gradient (1.0 = seamless; report share of assets ≤ 1.2) plus Tiling Score; `style_consistency(vecs_by_biome)`; `kid_fid(real_dir, fake_imgs, n)` with torchmetrics against `data/raw/textures` (240 SDXL-Turbo images, the only reference set; caveat recorded in reports); bench writes `reports/bench-<timestamp>.json` with environment, seeds, n, per-metric values, and p50/p95.

**Sprites.** Props are generated on a white background and alpha-keyed by colour distance *(ponytail corner: naive keying; upgrade to a matting model when it matters)*.

**Security.** The HF token in history must be revoked by the owner; `.gitignore` keeps secret files out; no history rewrite without explicit permission; CI runs a secret scan.

## Testing Decisions

A good test exercises a module through its interface with an independent expected value, never its internals. Seams under test (self-approved):

- `layout`: example map → tileset adjacency equals the pairs present; solved grids have zero adjacency violations (checked by an independent brute-force checker); determinism per seed; borders honoured; contradiction path recovers.
- `assets`: procedural adapter shape/dtype/determinism and seamlessness; `tileability` ranks a seamless synthetic texture above a cut one; alpha keying removes white.
- `world`: request → chunk with every tile mapped to an asset; graph invariants; coherence in range; drift → regeneration; predictor beats the 8-ring baseline on synthetic walks (hit-rate assertion over ≥20 seeds); scheduler never exceeds 2 in flight and evicts by bytes.
- `bridge`: TestClient endpoints, PNG decodes, WebSocket emits ready.
- `cli`: layout PNG written; bench with procedural adapter writes a report.
- GPU: `@pytest.mark.gpu` smoke for `DiffusionGenerator` (skipped without CUDA).

Prior art: none in repo (no tests existed). Fixtures stay tiny; no mocks of internals.

## Out of Scope

3D assets; running diffusion inside Unity (Sentis); multiplayer; LLM narrative/quest agents; animated character sheets and ControlNet poses; a matting model for sprites; cloud deployment; training LoRAs at scale (one short demo run per biome is enough); replacing or regenerating the dataset; git history rewrite.

## Further Notes

Measured on this machine before any rewrite (SD1.5 + LCM-LoRA, 4 steps, 512px, circular padding): 0.61 s/asset warm, 3.09 GB peak VRAM, seam gradient equal to interior gradient. These are the floor the new generator must not regress. Original pitch numbers (FID 32.4, 2.7 s/asset, 120 textures/min, 94% tileable, 0.89 style, 31 FPS, 6.2 GB) are targets listed beside measurements in the README; none is quoted as achieved until the bench says so.
