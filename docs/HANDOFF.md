# Handoff (2026-10-06, end of day)

Branch `feat/mvp-2026`, PR #4 against `main` (Auto-fix on). Two bodies of work sit on the branch:

## 1. RealmWeaver MVP (done)
150+ CPU and 11 GPU tests, measured benchmark `reports/bench-20261006-045936.json`, Docker GPU path verified, operator page verified. See `README.md` (generated Measured table) and the earlier sections of this file's history in git.

## 2. Emberfall: Crimson-Desert-like open-world action RPG (planning + Phase 0 done)
- Plan: `docs/superpowers/specs/2026-10-06-emberfall-game-design.md` (pillars, scope tiers M0–M4, 16 epics covering all 127 Crimson Desert systems, architecture, risks, testing). Research: `docs/research/06` (Crimson Desert), `07` (2026 stack), `08` (NPC runtime spike), `09` (TRELLIS.2 mesh spike), `10` (licence audit), `11` (spec tiers), `12` (terrain spike).
- Tracker: GitHub Issues — spec #5 (`ready-for-agent`), wayfinder map #6 with 16 epics (#7–#22) and decision tickets (#23–#35). Resolved today: #29 mesh spike, #30 terrain spike, #31 NPC runtime spike, #32 payload-typed Asset, #34 spec tiers, #35 licence audit. Open frontier: #23 engine choice (owner), #26 character/regions (owner), then #25/#27/#28/#33/#24.
- Engine evolution landed (architecture review order): C1 payload-typed Asset, C3 World behind a region map, C5 Runner seam + thin bridge, C8 one wire module (`realmweaver wire --write` regenerates the Unity DTO block and README contract). C4 record-driven graph landed; C6 (Biome record) and C7 (geometry from World) held.
- New packages: `realmweaver/npc` (personas, memory stream, JSON actions, grounding verifier, local LLM adapter; measured 0.38 s/turn with Qwen3-4B Q4 via llama.cpp), `realmweaver/terrain` (heightmap sources, rivers/roads/biomes/settlements post-pass, graph writer; real model ran: 1024² tile in 0.2 s warm), `realmweaver/gltf.py`, `realmweaver/wire.py`. Spike tooling in `tools/`; external installs at `D:\tools\ComfyUI` (TRELLIS.2) and `D:\tools\terrain-diffusion` (own venvs).
- Licence consequences already applied to the plan: HY-Motion and Hunyuan3D excluded; SPAR3D placeholder-only; TRELLIS.2 with BiRefNet instead of RMBG-2.0; Hyper-SD files allowlisted by name.

## Owner actions (blocking the next phase)
1. Decide the engine (#23): Unreal 5.8 two-week spike (recommended) or Unity 6.3 (keeps the shipped client). Both need an account the agent cannot create; then do #33 (engine account + empty project).
2. Decide the playable character and the five regions (#26); art direction (#25) and the chapter outline (#27) follow.
3. Cloud GPU budget (#24) for 1024³ meshes.
4. Revoke the Hugging Face token in history (`5636a3e`).

## If resuming work
- Verify `REALMWEAVER_DEVICE=cpu uv run pytest -q -m "not gpu" -o addopts=""` is green and `git status` is clean; commit anything an interrupted agent left (check `docs/research/`, `realmweaver/world/records.py` for C4).
- Next engine tasks without an engine: C4 if unfinished, C6 Biome record, mesh post-processing as a batch job with BiRefNet matting (E2), priority-flood depression filling in terrain (E3), NPC runtime served by `llama-server` as a sidecar with the latency budget (E7), content-studio approval queues with provenance (E13/E14).
- Once the engine exists: E4 client foundation (streaming cells, bridge v2 WebSocket client, save/load), then the M1 vertical slice per spec #5.
