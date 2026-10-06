# 00 — Codebase Audit: RealmWeaver-MADWE (2026-10-06)

Scope: `main` @ `983ec61` (2025-07-25): 32 commits on main, 68 across all refs. Method: full read of every source file, AST lint for undefined names/unused imports, import of `base_agent.py` on Python 3.11 and 3.14, PyYAML parse of the training config, PIL census of every image in `data/`, masked secret scan of full git history. No model or GPU code executed. Produced by a read-only audit agent; saved by the session lead.

## 1. Per-file audit

### src/agents/base_agent.py (721 lines)
- Claims: foundation for all agents (`Message`, `MessageBus`, `StateManager`, `BaseAgent`, `AgentCluster`, `EchoAgent`).
- Real: stdlib asyncio, no stubs; topic subscriptions stored but never used; `StateManager` never used. No base model.
- Bugs: L143 uses `Tuple` without importing it, `NameError` at import on Python 3.11-3.13 (verified on 3.11); heartbeat replies are heartbeats, so agent pairs ping-pong forever; `publish` awaits `Queue.put` while holding a `threading.RLock`; `save_state` json-dumps a `datetime`.
- Keep: `Message` schema and lifecycle ABC as design reference only.

### src/agents/asset_agent.py (829 lines)
- Claims: LangGraph pipeline validate / prompt / generate / post-process / quality check (3 retries) / save, priority batching.
- Real: prompt builder, quality heuristic (Laplacian variance, contrast, colour std, Canny density), retry graph, semaphore batching. Not a `BaseAgent` subclass; validation result ignored.
- Base model: SD1.5 `runwayml/stable-diffusion-v1-5` (repo id removed from the Hub in 2024) passed to SDXL-only inference classes.
- Cannot run: imports `DiffusionInference`/`FastInference` that `inference.py` never defines; `langgraph` not in requirements and old import paths; calls non-existent `generate_fast`, `load_lora(adapter_name=)`, `model_id=`; treats the list from `generate()` as one image; `_make_seamless` duplicates the edge band instead of blending the wrap seam.
- Keep: prompt template, quality heuristics, retry-loop shape.

### src/agents/character_agent.py (996 lines)
- Real: `CharacterAttributes`/`Equipment` dataclasses, prompt builders, equipment/colour/biome tables. Fake: ControlNet loaded but `_generate_with_pose_control` falls back to plain generation; equipment "masks" are fixed rectangles; three poses in a custom keypoint layout.
- Base model: SD1.5 + `lllyasviel/sd-controlnet-openpose` (1.4 GB downloaded, never used).
- Cannot run: inherits base_agent's `NameError`; same missing imports and keyword mismatch; `request.metadata` not a dataclass field; state save fails on enums; 6 unused imports.

### src/models/diffusion/style_consistency_loss.py (758 lines)
- Real loss terms: LPIPS-VGG/VGG16 perceptual, Gram matrices, histogram colour distance, pooled spatial MSE, FFT texture coherence. Real post-fx: overlay blend, frequency-split detail, relighting from Sobel normals, weathering. Stubs: `_train_with_style_consistency` is `pass`; Gram features come from a never-trained CNN.
- Bugs: `torch.histc` has no gradient; LPIPS range mismatch; HSV helpers apply 8-bit ranges to float32; wrong relative import; default config missing; hard `import wandb`; nothing imports it.

### src/models/diffusion/emergent_gameplay_loss.py (751 lines)
- Essentially fake: `_has_hidden_areas`/`_has_combat_zones` return `torch.rand(1) > 0.5`; `_has_multiple_paths` always True; `PlayerPredictor` hard-coded mock; cluster/narrative extraction return `[]`; emergence/player terms carry no gradient; `GameplayPattern.to_tensor` raises `IndexError`. Keep: nothing.

### src/models/diffusion/optimizers.py (157 lines)
- `get_lora_optimizer`/`get_param_groups` fine; `CosineAnnealingWithWarmup` produces a 0-dim tensor LR, has no `state_dict`, divides by zero when total == warmup; unused.

### scripts/benchmark.py (272 lines)
- Genuine measurements (warmup, synchronize, perf_counter, 3 runs, peak memory, JSON) against the dead SD1.5 id; interactive `input()` blocks headless runs; "fps" means images/s.

### scripts/generate_textures.py (329) / generate_sprites.py (392) / download_data.py (438)
- Prompt banks: `texture_prompts[category][subtype]` 8x3 = 24 (fantasy_forest, cyberpunk_city, horror_mansion, casual_bright, retro_arcade, underwater_depths, steampunk_industrial, post_apocalypse); `sprite_prompts` 8x3 = 24 (pixel_characters, hand_drawn_characters, vector_characters, fantasy_items, sci_fi_items, magical_effects, environmental_effects, game_ui). Duplicated across files; mined to `04-prompt-bank.json`.
- Base model: SDXL-Turbo; history: 3,691-line procedural generator (`df09f8e`), then FLUX.1-schnell with a hard-coded HF token and FLUX-dev settings (`5636a3e`), then SDXL-Turbo (`fb6055b`).
- Bugs: guidance 2.0 + negative prompt on Turbo (meant for 0.0); 768px then shrink; "seamless" only as prompt words; `optimize_prompt` counts words not CLIP tokens; seeds random and unrecorded; `sys.exit` at module level without CUDA; `D:/huggingface` hard-coded; sprite background removal assumes white backgrounds that are not white (dataset sprites ~0% transparent); LANCZOS shrink destroys pixel grids; `download_data.py` has a `_generate_synthetic` fallback that writes uniform-noise PNGs as assets, `os.system("pip install scipy")` at runtime, and an alpha overflow bug.

### scripts/preprocess_data.py (493 lines)
- Real; ran 2025-07-13: 284 valid images, 80/10/10 per category, `random_state=42`, 2 augmentations per training texture.
- Bugs: seamless score subtracts uint8 (wraps) and compares 10-px bands instead of meeting columns; "high-frequency detail" read from the low-frequency centre of a shifted spectrum; `matchTemplate(gray, gray)` is 1x1 so `pattern_regularity` always 0.0; not a real pHash; unseeded augmentations; needs sklearn (missing); writes a metadata filename the dataloader never reads; stale (32 newer sprites never split).

### scripts/train_lora.py (91) + configs/training/lora_config.yaml (84)
- PyYAML reads `1e-4` as a string so AdamW crashes (verified); `--resume` key never read; biome overrides match no dataset category; biome filter is a no-op (dataloader parses `fantasy_forest` as `{fantasy: forest}`); the prompt fallback indexes a missing `metadata` key eagerly, `KeyError`; conflicting stop criteria; SDXL trained at 512. No training run ever happened; no weights anywhere.

### scripts/setup_environment.bat, .github/workflows/ci.yml, setup.py, requirements.txt
- Batch file requires exactly Python 3.11, installs CPU-only torch, installs unused torch-tensorrt, copies a missing `.env.example`.
- CI cannot pass: `pywin32` has no Linux wheels; flake8/pytest target non-existent paths; zero tests on any branch.
- `setup.py` installs top-level `models`/`agents` while code imports `src.…`; console scripts broken; `install_requires` empty.
- requirements: unused onnx, onnxruntime-gpu, bitsandbytes, pandas, seaborn, matplotlib, colorama; missing torch, langgraph, networkx, scikit-learn, lpips.

### configs/biomes/*.json (7 files, 62,406 lines)
- Fully connected, undirected, uniform weight 1: no constraint at all; reference 617 tile PNGs deleted in `fb6055b`; produced by `src/wfc/configs.py`, which writes files on import. Keep: nothing.

### data/processed/metadata/*
- `preprocessing_metadata.json` (423 KB): `is_seamless` true for 0 of 192 textures; `pattern_regularity` 0.0 everywhere; no captions; `dataloader.py` never reads it.

## 2. Dependency graph (main)
```
scripts/train_lora.py -> src.models.diffusion (package __init__)
src/models/diffusion/__init__.py -> .lora_trainer, .inference, .optimizers (defines aliases DiffusionInference/FastInference)
src/models/diffusion/lora_trainer -> src.utils.dataloader, src.models.diffusion.optimizers
src/agents/asset_agent -> ..models.diffusion.inference   [ImportError] (+ langgraph missing)
src/agents/character_agent -> ..agents.base_agent [NameError on Py<=3.13], ..models.diffusion.inference [ImportError]
Orphans: asset_agent, character_agent, style_consistency_loss, emergent_gameplay_loss, utils/prompt_optimizer, wfc/base_wfc, wfc/configs, unity_bridge/ipc_server
Empty: agents/environment_agent.py, agents/narrative_agent.py; docstring-only packages models/nwsg, models/player_prediction, generation, generation/wfc
```

## 3. Cross-module contract breaks
1. Agents -> inference: class names, keyword arguments, methods, return types and model family all disagree.
2. Captions never reach training: three different metadata filenames across generators, preprocessing and dataloader.
3. Biome filter vs directory names: the filter silently passes everything.
4. YAML -> trainer: learning rates arrive as strings.
5. Trainer saves PEFT adapter format; inference expects diffusers LoRA format; no conversion.
6. Biome names differ everywhere: {forest, desert, snow, volcanic, underwater, sky} vs {forest, desert, cyberpunk, underwater} vs dataset {fantasy_forest, cyberpunk_city, ...}.

## 4. Base-model inventory
| Location | Model |
|---|---|
| asset/character agents, style loss, benchmark, sprite fallback | SD1.5 `runwayml/stable-diffusion-v1-5` (dead id; mirror `stable-diffusion-v1-5/stable-diffusion-v1-5`) |
| character_agent | + `lllyasviel/sd-controlnet-openpose` (never used) |
| lora_trainer, inference, lora_config.yaml | SDXL base 1.0 |
| generate_textures / generate_sprites / download_data | SDXL-Turbo (Stability non-commercial licence) |
| download_data @ `5636a3e` (history) | FLUX.1-schnell (Apache-2.0) |
| origin/unity asset_agent / inference | SD2.1 / SD1.5 |

## 5. Hard-coded paths and secrets
- `D:\huggingface\...` in generate_textures.py L21-23/L144, generate_sprites.py L21-23/L146/L156, download_data.py L18-21/L76.
- Working tree: no secrets; no `.env` or key files.

## 6. Git history check
- 68 commits across refs; `origin/unity` has 35 commits not on main; `origin/develop` 1.
- `51a9a35` ("key.txt") only edits `.gitignore` (renames the ignore entry, un-ignoring `keys.txt`).
- **A secret is in history:** one Hugging Face user access token (`hf_` prefix, 37 chars) committed in `5636a3e` (2025-07-13) in `scripts/keys.txt` (`HF_TOKEN = "hf_..."`) and twice in `scripts/download_data.py` (`login("hf_...")`, `use_auth_token="hf_..."`); removed from files in `fb6055b`/`983ec61`; still reachable from `main`, `origin/main`, `origin/develop` on GitHub.
- Action: revoke at huggingface.co/settings/tokens first; history rewrite (`git filter-repo`, also purging 1,217 deleted legacy images; `.git` pack is 360 MiB) is optional and needs owner approval.

## 7. Dataset census (every file read with PIL)
Raw: 316 PNGs, all 512x512. Textures (RGB, 30 each, 3 subtypes x 10): casual_bright, cyberpunk_city, fantasy_forest, horror_mansion, post_apocalypse, retro_arcade, steampunk_industrial, underwater_depths = 240. Sprites (RGBA): fantasy_items 30, pixel_characters 30, sci_fi_items 16 = 76.
Processed: 668 PNGs (284 originals byte-identical to raw + 384 augmentations); sci_fi_items absent.
Quality: textures not tileable (0/192 pass even the buggy check); sprites ~0% transparent (opaque dark/grey backgrounds); `pixel_characters` is not pixel art (~1% flat 2x2 blocks, ~145k colours per image); no noise images found. Legacy: 1,217 procedural images deleted in `fb6055b` remain in history; `data/` is 426 MB on disk.

## 8. origin/unity (spot-checked, not audited)
35 commits (2025-07-28 to 07-30, AnkitGole), 112 files, +13,390/-2,562 excluding data: Unity project `MADWE-Unity/` (~2.1k lines C#), `environment_agent.py` (594), `narrative_agent.py` (601, HF causal LM), `multi_agent_coordinator.py`, `protocols.py`, `tools.py`, `unity_bridge/communication.py` (299), `wfc/hierarchical_wfc.py` (216), `models/nwsg/graph_network.py` (820, rtree-indexed graph), 625-line ONNX/TensorRT optimizers, `run_madwe.py`. Entry point broken at import (`MessageBus` undefined; `LoRAConfig` missing); SD2.1; no tests. Evaluated separately in `03-origin-unity-salvage.md`.

## 9. Keep / Rewrite / Delete
**Keep:** `data/raw/*` (reference set) and the 48 prompts; `preprocess_data.py` ideas (after metric fixes); benchmark harness shape; `get_lora_optimizer`/`get_param_groups`; dataloader directory-discovery idea.
**Rewrite:** LoRA trainer (SD1.5, PEFT, correct save/load format); dataset generation (one config-driven path, circular padding, recorded seeds); asset agent (prompt/quality/retry on one inference API); world graph, WFC, bridge, CI, packaging.
**Delete:** emergent_gameplay_loss; `configs/biomes/*.json`; `data/processed/**` from git; empty agents and placeholder packages; `prompt_optimizer`; cosine scheduler; `wfc/configs.py`; old `setup.py`/`requirements.txt`/`setup_environment.bat`.

## 10. Executive summary
1. Main implements roughly 20% of the MADWE scope; nothing runs end to end; no model was ever trained.
2. Data generation ~55%: 316 images at 512px, but textures don't tile, sprites aren't transparent, "pixel art" isn't pixel art, seeds unrecorded.
3. Preprocessing ~65%: ran once, buggy metrics, stale.
4. LoRA training ~25%: crashes on config and first batch; biome filter is a no-op.
5. Inference ~45%: SDXL `LoRAInference` reasonable; no LoRA to load; agents call a different API.
6. Custom losses ~10%: style loss salvageable as metrics; gameplay loss is random/mock.
7. Multi-agent layer ~10%: crashes at import; environment/narrative agents empty on main.
8. WFC ~15%; NWSG and player prediction 0% on main (exist only on origin/unity, unverified).
9. Unity integration ~10%: unused IPC stub; origin/unity has a project and ~2.1k lines C# worth reviewing.
10. Tooling ~5%: zero tests, CI can't pass, Windows-only paths, three base models on main, HF token in pushed history (revoke today).
