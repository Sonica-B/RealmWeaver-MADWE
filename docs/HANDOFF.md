# Handoff (2026-10-06, end of session 1)

Branch `feat/mvp-2026`, PR against `main`. Everything in the plan (`docs/superpowers/plans/2026-10-06-realmweaver-mvp.md`, Tasks 0–9) is implemented, reviewed on two axes (`docs/review/`), fixed, and measured.

## State
- CPU suite: 150 tests (~20 s), `REALMWEAVER_DEVICE=cpu uv run pytest -q -m "not gpu"`; GPU suite: 11 tests, `uv run pytest -q -m gpu`.
- Latest benchmark on the shipped code: `reports/bench-20261006-045936.json` (n=100, six biomes); README "Measured" table is generated from it by `tools/report_table.py --write` (picks the newest report by its own timestamp).
- Docker: `docker compose build` then `docker compose run --rm realmweaver realmweaver bench --n 10 --biomes forest` ran on the GPU inside the container (`reports/bench-20261006-111729.json`, UTC clock).
- Operator page verified in a browser (generate, chunk request, prewarm, events) — `docs/images/operator-page.png`.
- Forest biome LoRA trained once (200 steps, 71 s) to prove `realmweaver train-lora`; weights live under `models/` (gitignored).

## Known limits (honest)
- FID (265.6) is against the repo's own 240 non-tileable SDXL-Turbo textures — not comparable to the 2025 pitch figure; KID 0.039 is the usable statistic.
- Style consistency 0.097 is the within-minus-cross-biome DINO cosine; DINO features follow tile class more than biome, so the number is small by construction.
- Tileable share 80% at the strict 1.2 seam ratio (mean ratio ~1.06); the quality tier (FLUX.2-klein) is designed behind the `Generator` seam but not wired.
- Prewarm runs synchronously inside `World.tick`; the bridge moves it to a background worker that yields to player requests.
- Unity package was reviewed against Unity 6.3 docs and protocol-tested from Python; it has not been compiled in an Editor (none on this machine).

## Owner actions
- Revoke the Hugging Face token in pushed history (commit `5636a3e`; see `docs/research/00-codebase-audit.md` §6). History rewrite only on explicit request.
- Merge the PR when satisfied; CI runs ruff + CPU tests + a secret scan on Linux.

## If resuming work
Next candidates, in order of value: wire `KleinGenerator` (FLUX.2-klein-4B, Apache-2.0) behind the seam and A/B it in the bench; compile the Unity package in Unity 6.3 and record frame times; raise tileable share by gating refine-tier regeneration on the seam ratio.
