# Standards review — 2026-10-06 (`feat/mvp-2026` @ cdc140a vs `main`)

Scope: `git diff main...HEAD -- realmweaver tests unity tools pyproject.toml Dockerfile compose.yaml .github` (14 commits, 65 files); read-only.
Clean: no `print`/machine paths/secrets in library code; public functions typed; tests mock nothing internal; README and operator page show report values only.

## Hard violations
- `metrics/bench.py:159-160` — reads `generator._biome_loras` and builds a second model via `type(generator)(settings=…)` (rule 2 seam breach, rule 3) — take a `pool_ab: Callable[[bool], Generator]` factory from the CLI.
- `assets/__init__.py:4`, `metrics/__init__.py:3` — public surfaces omit `DiffusionGenerator`, `DinoEmbedder`, `set_seamless`, `train_biome_lora`, `run_bench`, `kid_fid`, yet `cli.py:39,124,134`, `bridge/app.py:93,102`, `tests/test_bench.py:12`, `tests/test_seamless_cpu.py:3` import those private modules (rule 2) — PEP 562 lazy re-exports.
- `bridge/app.py:116` — `os.environ.get("REALMWEAVER_BIOME")` bypasses `realmweaver.config` (rule 5) — add `Settings.biome`.
- `assets/diffusion.py:57`, `assets/embed.py:19`, `metrics/fid.py:72` — constructors build models a caller could inject (rule 3) — optional `pipe=`/`model=`/`extractor=`, built only when `None`.
- `assets/procedural.py:75` "in milliseconds"; `tests/test_world.py:57`, `tests/test_bridge.py:4` "costs milliseconds"; `tests/test_assets_diffusion_gpu.py:33` `latency_s < 3.0` — performance claims outside a report (rule 10) — delete the words; drop the assert or read the report's p95.
- `tools/report_table.py:59` — literal "per 16x16 chunk" while `bench.py:195` `params` omits the measured `chunk_size` (rule 10) — record `chunk`, format the label from it.
- `pyproject.toml:18,56` — no `gpu` extra (torch/diffusers/peft unconditional); `Dockerfile:7` uses pip (ADR-0006: extra `gpu`, `uv sync` only) — add the extra or amend the ADR.
- `cli.py:90` PIL `resize` off any file/HTTP edge; `world/world.py:134-147` `stats()`, `assets/diffusion.py:212` `allocator_stats()` cross seams as `dict` (rule 4) — numpy reshape-mean; `WorldStats`/`AllocatorStats` dataclasses.
- `world/scheduler.py:61` O(n²) eviction; `bridge/app.py:113` one lock serialises generation (glossary: two in flight); `bridge/app.py:168` PNG re-encoded per hit under `immutable` — no `# ponytail:` (rule 8) — add it.

## Judgement calls
- Duplicated Code — `cli.py:32` vs `bridge/app.py:88` `_adapters`; `metrics/bench.py:219` vs `bridge/app.py:75` `latest_report` — one `assets.default_adapters(device)`, one `metrics.latest_report`.
- Shotgun Surgery — tier→steps in `types.py:36`, `cli.py:28`, `world/world.py:31`, `assets/diffusion.py:27`, `bridge/app.py:45`; `_TIERS[spec.tier][0]` positional tuple (Primitive Obsession) — `TIER_STEPS` beside `Tier`; `TierConfig(guidance, min_steps)`.
- Primitive Obsession — `bridge/app.py:100,127` dispatch on `type(generator).__name__`; `world/graph.py:145` `set_state(state: str)` + `type: ignore` — `kind` on the `Generator` protocol; `ChunkState` Literal.
- Long Function — `metrics/bench.py:132` `run_bench` (85 lines), `assets/lora.py:62` `train_biome_lora` (83), `bridge/app.py:180` `create_app` (84) — extract `_pool_ab`/`_notes`, `_train_loop`/`_save`, `APIRouter`s.
- Message Chain / Feature Envy — `bridge/app.py:141,149` `self.world.graph.chunk(...)` reaches through `World`; `cli.py:81-91` `_chunk_image` lives on `chunk.layout.tileset` — add `World.chunk()`; move the stitcher beside `layout.render`.
- Speculative Generality — `world/scheduler.py:28` `bytes_hint` never read; `cost_s` never fed from `_chunk_latencies`; `assets/diffusion.py:48` `compile=` untested; `config.py:42` dead bool branch — delete or wire.
- Mysterious Name — `world/graph.py:57` attribute `g` (`world.graph.g.nodes`); `world/world.py:204-212` `anchor` means both "fallback needed" and "make STYLE_ANCHOR"; `bridge/__init__.py:15` `del app` + `__getattr__` shadows the submodule — `graph`; `needs_fallback`/`make_anchor`; rename module `routes.py`.
- CUDA-graph pool — `assets/pool.py:56,61` capture `**kwargs`; `pool.py:64-65` replay copies only sample/timestep/embeds and `diffusion.py:127` keys neither, so a tensor kwarg (`timestep_cond`, `added_cond_kwargs`) would replay stale; `diffusion.py:166` the pool latent is re-multiplied by `init_noise_sigma` inside diffusers (the zero-alloc test passes via the caching allocator); thread safety rests on `bridge/app.py:113` — assert kwargs tensor-free at capture or key/copy them; document "not thread-safe".
- World loop — `world/world.py:204-212` a below-threshold candidate with no class asset is added `anchor=True`, so a failing asset becomes the region's STYLE_ANCHOR — add it un-anchored. `world/world.py:45` p95 via `np.percentile` vs `metrics/bench.py:80` nearest rank — one helper.
- Vocabulary — "asset agent"/"world agent" (`types.py:65`, `cli.py:28-43`, `assets/diffusion.py:35`, `layout/wfc.py:60`) vs glossary `Generator`/`World` (ADR-0001 says "asset agent" too) — pick one in GLOSSARY.md.
- Config hygiene — `assets/pool.py:24` import-time `os.environ` mutation; `metrics/bench.py:33`, `tools/report_table.py:84` cwd-relative paths ignoring `settings().reports_dir`; `.github/workflows/ci.yml:21` lints without `tools/` — move to `config.py`; `Settings.reference_dir`; add `tools`.

## Test/lint run
- `REALMWEAVER_DEVICE=cpu uv run pytest -q -m "not gpu"` → `143 passed, 9 deselected, 1 warning in 17.59s` (`-o addopts=""` shows the line; rule 7's 60 s met).
- `uv run ruff check realmweaver tests tools` → `All checks passed!`; `ruff format --check` → `44 files already formatted`.
