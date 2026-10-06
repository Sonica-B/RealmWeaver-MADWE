# Engine evolution review — 2026-10-06

Range `41c9e8a..88f5fe1` (C1, C3, C5, NPC runtime); lines at HEAD (uncommitted C8 `wire.py` work is shifting `bridge/app.py`, `cli.py`); paths under `realmweaver/`. No `print`/machine paths; GLB container correct.

## Standards — hard violations
- `npc/runtime.py:25,253-270` — rule 2: new cross-package seam npc→`WorldStateGraph`, via private module path and raw networkx `graph.g`/`graph.chunks` — take a `World`; use `region()`/`chunks()`.
- `npc/runtime.py:30,40,65-70` — rule 4: facts cross that seam as `"<kind>: <Name> - <detail>"` strings parsed by regex — frozen `Fact(kind, name, detail)`.
- `GLOSSARY.md` — rule 1: runner/slot/job/handle, transition, persona, memory stream, grounding verifier, action schema are in code but not the glossary (spec §8 "Glossary additions" never landed); `Listener` is a callable at `world/world.py:55`, a tuple at `bridge/app.py:38` — add terms; rename one.
- `gltf.py:168-169`, `npc/llm_local.py:72-74` — rule 10: "minutes for a million-triangle mesh", "stalls for minutes" in code; `docs/research/08-npc-runtime-spike.md` p50/p95/VRAM come from untracked `npc_spike.py` (ADR-0005 not re-runnable) — drop durations; commit the harness.

## Standards — judgement calls
- `npc/memory.py:104-116` + `runtime.py:64` — reflection folds `source="player"` observations into a trusted `self` memory: an injected name becomes sayable after the next reflection — exclude player observations from reflection.
- `npc/runtime.py:98-115,74-77,89-93` — verifier skips `give.text`, `end.text`, `report_crime.*`; `knows()` accepts any composite of known words ("Tomas Dusk") — check every text; whole-phrase match.
- `npc/runtime.py:167` — utterance interpolated raw: newlines forge MEMORIES lines; no length cap vs `n_ctx` — `json.dumps`, strip control characters, cap.
- `world/world.py:314-325,355,130-142` — `_asset_for` releases one RLock level; entered re-entrantly (anchor fallback via `asset()`; every InlineRunner job, run inside `submit`'s lock) the lock stays held through generation (contra docstring l.8) — call `_asset_for(spec)` directly; run inline jobs outside the lock.
- `world/runner.py:48-73`, `bridge/app.py:149`, `cli.py:118` — `ThreadRunner` never closed (`tests/test_bridge.py:38-46` leaks threads per client); `submit` after `close` never settles; a cancelled `World.submit` Future leaves `_jobs[key]` set forever (`world.py:141,255`) — `World.close()`, lifespan, `set_exception` after close, read-only handle.
- `bridge/app.py:170-176,209-216` — `generated`/`encoded` mutated from threadpool and worker threads unlocked (`move_to_end` after concurrent `popitem` raises) — one lock.
- `npc/__init__.py:28-57`, `persona.py:98-244`, `actions.py:103` — 27 exports incl. prompt internals; five personas as library code; `action_text` dead — trim, fixtures to `tests/`, delete.

## Spec — missing/partial
- C3 `world/world.py:95` still "`ponytail: one region per world`"; `Region` (`graph.py:31-38`) lacks tileset/threshold; no `region_at`; `_solve:303-312` ignores neighbour regions — card: "World takes a region map … a Region record carries biome, compiled tileset and threshold, the border solve asks the neighbour's region".
- C5 `world.py:150-153,255-257`, `app.py:34,106` — only `on_ready`; failed prewarm just logs; no progress/cancel; `generated` cache kept — card: "events() → ready · progress · failed", "long jobs get progress and cancellation".
- C1 `types.py:184-189`, `bench.py:192-195`, `cli.py:60`, `app.py:177-178` — `payload.image` still read in three modules plus `isinstance` switch; no audio kind (`Kind:22`; E1 lists audio) — card: "callers ask the payload, never the pixels".
- E7/§5 — no input filter ("profanity/injection filters on player input — M0"), no latency budget, async turn, "cache of common lines" or planning; facts are fixtures, not graph nodes.
- §7 `tests/test_asset_payloads.py:168-186` — watertight only; "triangle budget, UV coverage" untested.

## Spec — scope creep
- `npc/persona.py:50-95,166` — schedule templates (E7 "Daily schedules — M1") and `voice_id` (Voice — M2) in M0.
- `npc/llm_local.py:99-100,130-134` — grammar decoding kept after the spike's own "No full-schema GBNF grammar".

## Spec — looks wrong
- `npc/llm_local.py:1-7,25-26` — in-process `llama-cpp-python`, `Qwen/Qwen3-4B-GGUF`, undeclared in `pyproject.toml`; §5: "llama.cpp `llama-server` with `Qwen/Qwen3.5-4B`", "LLM, TTS and STT share one sidecar process" — ADR or align.
- `npc/runtime.py:98-115` — capitalised names, two action kinds; E7: "every fact in a reply must trace to a graph node or the NPC's memory"; numbers and lowercase claims pass.

## Test/lint run
- `REALMWEAVER_DEVICE=cpu uv run pytest -q -m "not gpu" -o addopts="" --ignore=tests/test_terrain.py` → `188 passed, 11 deselected, 2 warnings in 28.39s`
- `uv run ruff check realmweaver tests --exclude realmweaver/terrain --exclude tests/test_terrain.py` → `All checks passed!`
