# 03 — `origin/unity` salvage review (2026-10-06)

**Question.** The remote branch `origin/unity` (35 commits by AnkitGole, 2025-07-28 → 2025-07-30, never merged) contains a Unity project and a second generation of the Python agent code. Is any of it worth carrying into the `realmweaver` rebuild described in `docs/superpowers/specs/2026-10-06-realmweaver-mvp-design.md`?

**Method.** Read-only: `git ls-tree`, `git show origin/unity:<path>`, `git diff --stat main...origin/unity`. Nothing was checked out. Every C# file, `ProjectVersion.txt`, `manifest.json`, `packages-lock.json`, the two scenes, the manager prefab, and the Python modules named in the brief were read in full; `asset_agent.py`, `multi_agent_coordinator.py`, `tools.py`, `optimization_framework.py`, the test scripts and `.gitignore` were skimmed because they bear on the verdicts. Line numbers below refer to the branch versions.

**Verdict legend.** SALVAGE = copy as-is or near as-is. ADAPT = extract with `git show`, then rework per the listed changes. IGNORE = do not extract; at most borrow an idea.

---

## 0. TL;DR

| # | Item | Verdict | One-line reason |
|---|------|---------|-----------------|
| 1a | `MADWE-Unity/` as the base of our Unity client (project, scenes, prefabs, materials, settings, packages) | **IGNORE** | Unity 2023.2.20f1, Built-in RP, 3D cubes on the XZ plane, `manifest.json` pins `com.unity.ml-agents` to an absolute path on the author's disk, no folder `.meta` files, and two compile errors in the committed scripts. A fresh Unity 6.3 2D project is faster. |
| 1b | `Assets/Scripts/AssetLoadingSystem.cs` | **ADAPT** → `AssetStreamer.cs` | Its coroutine + `UnityWebRequestTexture` + LRU-with-`Destroy` plumbing is ~60 % of the spec's `AssetStreamer`; needs HTTP URL, `nonReadable`, byte cap, priority queue, placeholder; ~200 lines of material/StreamingAssets/base64 code deleted. |
| 1c | `Assets/Scripts/TilePrefabManager.cs` lines 5-20 (+210-215) | **ADAPT** → `TileCatalog.cs` | The two `[Serializable]` mapping classes are the right shape for a ScriptableObject prefab map; key by tile-class name instead of int id. Everything else in the file (pooling, per-tile `renderer.material` tinting) is dropped. |
| 1d | `MADWEController.cs`, `TileManager.cs`, `AdvancedTileManager.cs`, `TileLOD.cs`, `SimpleTileTest.cs`, `UnityMainThreadDispatcher.cs`, `Editor/MADWESetupEditor.cs` | **IGNORE** | Raw-TCP transports (two incompatible ones), `JsonUtility` on a `List<List<int>>` that can never deserialize, 3D LOD, a boundary "smoothing" hack that offsets edge tiles vertically, an editor window that builds primitive cubes. None maps onto HTTP + WebSocket. |
| 2 | `src/models/nwsg/graph_network.py` | **IGNORE** | Dict-of-sets graph plus a 3-D R\*-tree over a world that is an integer chunk grid, an untrained random-weight `nn.TransformerEncoder` whose output no caller uses, a "synchronizer" that syncs with nobody. No typed edges, no hierarchy, no style vectors. networkx covers everything real in it. |
| 3 | `src/wfc/hierarchical_wfc.py` | **IGNORE** | Placeholder by its own docstring. `collapse()` is `np.random.choice`: no propagation, no entropy, no adjacency; `Tile.edges` are never read; `NeuralConstraintLearner` is never trained or called. A regression from `main`'s `base_wfc.py`. |
| 4 | `src/unity_bridge/communication.py`, `src/agents/protocols.py` (and `ipc_server.py`) | **IGNORE** | An in-process agent bus envelope, not a wire format. The one real Unity transport (`ipc_server.py`) listens on port 12345 with little-endian frames while both C# clients dial port 5005 (one newline-framed, one big-endian). Nothing was ever exchanged end to end. |
| 5 | `src/agents/environment_agent.py`, `src/agents/narrative_agent.py` | **IGNORE** | No prompt tables. Environment agent calls a WFC method that does not exist, inserts 4-tuple bounds into a 3-D index (every insert fails silently), `_send_to_unity` is `pass`. Narrative agent (DialoGPT, template strings) is out of scope and fails at import. |
| 6 | `.gitignore` Unity block, `tools.py`, `multi_agent_coordinator.py`, `optimization_framework.py`, test scripts | **IGNORE** | The ignore block drops folder `.meta` files (`*.meta` / `!*.*.meta`) and un-ignores a non-existent `unity_project/`; use the canonical GitHub `Unity.gitignore` instead. The rest is scaffolding with broken imports. |

**Do not merge the branch.** It rewrites `src/agents/*` (which the spec deletes), adds no data or configs, and its only two extractable files are fetched with `git show`. The generator scripts the spec mines prompts from (`scripts/generate_textures.py`, `generate_sprites.py`, `train_lora.py`) were added on `main` after the branch point (merge-base `2c7b663` has only four scripts), so the branch neither contains nor deletes them.

---

## 1. Branch facts

- Range `2c7b663..b025dfe` (merge-base → branch head): 35 commits, 112 files, +13,390 / −2,562.
- Adds `MADWE-Unity/` (9 C# scripts, 13 prefabs, 10 materials, 2 scenes, full `ProjectSettings/`, `Packages/`), 5 scripts, 8 new `src/` modules, and rewrites `asset_agent.py` (+503/−795), `base_agent.py` (+263/−677), `character_agent.py` (+149/−960), `environment_agent.py`, `narrative_agent.py`.
- Does not touch `data/`, `configs/`, `requirements.txt`, `README.md`. Consequently `rtree` and `chromadb`, which the new modules import at module load, are not declared anywhere.
- Commit messages are a day-by-day log ("Day 4 Integration", "Day 7 - Complete Production Code", "Changed the entire protocol system"); the protocol was changed twice in three days and the Python side never caught up (section 4).

---

## 2. Item 1 — the Unity project `MADWE-Unity/`

### 2.1 Facts

| Fact | Value | Source |
|------|-------|--------|
| Editor | **2023.2.20f1** (end-of-life; spec targets Unity 6.3) | `ProjectSettings/ProjectVersion.txt` |
| Render pipeline | **Built-in** (no SRP asset in `GraphicsSettings.asset`; materials use shader fileID 46 = Standard; scripts call `Shader.Find("Standard")`) | `GraphicsSettings.asset`, `Assets/*_Material.mat`, `AssetLoadingSystem.cs:171` |
| World model | **3-D**: tiles are cubes placed at `(x*size, 0, y*size)` on the XZ plane | `AdvancedTileManager.cs:193-197` |
| Packages | `com.unity.ml-agents` from **`file:D:/Projects/RealmWeaver-MADWE/ml-agents/...`** (absolute path on the author's machine; folder is git-ignored), pulling `com.unity.sentis 2.1.0`; `com.unity.visualscripting`, `timeline`, `ugui`, `collab-proxy`. **No `com.unity.nuget.newtonsoft-json`** in manifest or lock. | `Packages/manifest.json`, `packages-lock.json:84-93` |
| Scene wiring | `MADWE_TestScene.unity`: GameObject `MADWESystem` carries `AssetLoadingSystem` + `TilePrefabManager` + `AdvancedTileManager` + `MADWEController`; GameObject `TileManager` carries a second `TilePrefabManager` whose `defaultTilePrefab` is `TilePrefab.prefab`. Both `biomePrefabSets` are **empty**, so the 11 hand-built tile prefabs under `Resources/Prefabs/Tiles/{Forest,Desert,Snow}` are never referenced; every tile is the default cube with a colour tint. | scene lines 173-199, 297-301 |
| Folder `.meta` files | **None committed** (`Scripts.meta`, `Scenes.meta`, `Resources.meta`, … are absent); opening the project regenerates them and dirties the tree. | `git ls-tree -r origin/unity -- MADWE-Unity` |

### 2.2 What transport and JSON shape each script expects

Three transports coexist on the branch and none of them talk to each other.

| Side | File | Transport | Framing | JSON lib | Inbound message types (shape) | Outbound |
|------|------|-----------|---------|----------|-------------------------------|----------|
| C# | `MADWEController.cs` (in the scene) | raw TCP client to `127.0.0.1:5005` (L124), background `Thread`, polls `stream.DataAvailable` + `Thread.Sleep(10)` (L180-199), hops to main thread via `UnityMainThreadDispatcher` | **newline-delimited** JSON (L159, L187) | `JsonUtility` | `tile_update` → `data:{tiles:int[][], position:[x,y], size:[w,h], biome}` (L47-53); `asset_update` → `data:{asset_type:"texture"\|"material", asset_name, asset_data: base64 PNG or JSON string, metadata:{biome,name,size}}` (L63-78); `test` | `{type, data:{status,message,sent_time}, timestamp}` with `type` ∈ `connection`, `tile_update_complete`, `asset_loaded`, `test_response` |
| C# | `TileManager.cs` (not in the scene) | raw TCP client to `127.0.0.1:5005` (L18-19) | **4-byte big-endian length prefix** (L96-103, L268-271) | **Newtonsoft.Json** (L7) | `chunk_generated` / `tile_update` → `data:{width, height, tiles:[{x,y,id,name,biome,rotation,prefab}]}` (L187-204, L311-320); `performance_stats` → `{avg_latency, generation_fps}` | `unity_connected`; `generate_chunk` → `data:{position:[x,y], biome, size:[32,32]}` (L243-257) |
| Python | `src/unity_bridge/ipc_server.py` (`UnityBridge`) | TCP **server on port 12345** (L17, L66) | 4-byte **native-endian** (`struct.pack('I')`, little-endian on x86) length prefix (L86, L167) | `json` | dispatches on `message['type']` to handlers registered with `register_handler` — **never called anywhere on the branch** | `terrain_generated {data:int[][], metadata}`, `asset_generated {asset_type, data, metadata}` (L219-232, L250-259) — names neither C# client knows |
| Python | `src/agents/tools.py` | HTTP `POST http://127.0.0.1:5005/reload` (L30-31) | — | — | — | nothing serves it |

Also: `EnvironmentAgent._send_to_unity` is `pass` (L437-440), `UnityMessageHandlers.handle_generate_terrain` reads `self.request_cache`, an attribute that lives on `UnityBridge`, not on the handlers class (L213 → `AttributeError`). No Python code on the branch ever emits `tile_update` or `asset_update`, so the Unity scripts were exercised, if at all, by hand-written test senders that were not committed.

### 2.3 Does it compile and run?

As committed, Assembly-CSharp does **not** compile, so nothing in the project can run:

1. `TileManager.cs:7` — `using Newtonsoft.Json;` with no Newtonsoft package in the resolved lock (`packages-lock.json` has no `com.unity.nuget.newtonsoft-json`; `collab-proxy 2.8.2` declares no dependencies). Error CS0246.
2. `SimpleTileTest.cs:19` — calls `tileManager.CreateTileGrid(testGrid, "forest")` with `(int[,], string)`; the only overload is `CreateTileGrid(int width, int height)` (`TileManager.cs:181`). Error CS1503.
3. `Packages/manifest.json` — `com.unity.ml-agents` resolves to `file:D:/Projects/RealmWeaver-MADWE/ml-agents/...`; on any other machine Package Manager fails to resolve and the editor opens in safe mode. The spec explicitly rules out Sentis/ML-Agents in the game process.

Even with those fixed, the in-scene controller cannot render a chunk:

4. `MADWEController.cs:49` declares `public List<List<int>> tiles;` and deserialises it with `JsonUtility.FromJson<TileUpdateData>` (L246). Unity's serializer does not support nested containers, so `tiles` is always `null`; `tileData.data.tiles.Count` (L250) throws, the `catch` logs it, and no tile is ever placed. The one message type the controller exists for is dead on arrival.
5. `TilePrefabManager.ApplyBiomeStyle` (L240-249) calls `renderer.material.SetColor` on every placed tile — `renderer.material` instantiates a material copy per tile (256 per 16×16 chunk), which breaks batching and leaks materials. Should be `MaterialPropertyBlock` or shared materials.
6. `AdvancedTileManager.ApplyBoundarySmoothing` (L204-220) adds a sinusoidal **Y offset** to every edge tile "to blend chunks" — in a tile world this manufactures the cracks the spec's border constraints exist to prevent.
7. `AdvancedTileManager.PlaceTile` (L180) does a `chunksContainer.Find("Chunk_x_y_biome")` string lookup per tile (O(children) per tile); `ClearChunk` (L240) repeats it.
8. `TilePrefabManager.ReturnToPool` (L196-199) identifies the pool by `instance.name.StartsWith(prefab.name)` — fragile ("Pine" vs "PineTile") and wrong once two prefabs share a prefix.
9. `AssetLoadingSystem.PreloadAssets` (L56-72) tries to load 12 textures from `StreamingAssets/Textures/{biome}/{name}.png` that do not exist in the repo → 12 error logs at start.
10. Three incompatible tile-id vocabularies: `hierarchical_wfc.py` (forest 0-3 grass/tree/path/water, desert 10-12 sand/dune/cactus, cyberpunk 20-22), `TilePrefabManager.GetTileColor` (desert 10-13 sand/dune/rock/oasis, snow 20-22 snow/ice/pine), `environment_agent._generate_simple_tiles` (forest 0-2 grass/tree/**rock**, desert 3-5 sand/cactus/rock), and `TileManager.InitializePrefabCache` (string keys, biomes forest/desert/**cyberpunk**, arrays that must have exactly four entries or throw).

### 2.4 Unity APIs used

`TcpClient`/`NetworkStream`/`Thread` (System.Net.Sockets), `JsonUtility`, Newtonsoft (unresolved), `UnityWebRequestTexture.GetTexture` + `DownloadHandlerTexture.GetContent` on a `file://` URL, `Texture2D.LoadImage` (readable textures), `Shader.Find("Standard")`, `Material.SetTexture/EnableKeyword`, `GameObject.CreatePrimitive`, `Instantiate`/`Destroy`, coroutines with `yield return null`, `Camera.main` distance LOD, `FindFirstObjectByType` (2023.1+, fine on Unity 6), `DontDestroyOnLoad`, Editor: `EditorWindow`, `MenuItem`, `AssetDatabase.CreateFolder/CreateAsset`, `PrefabUtility.SaveAsPrefabAsset`. No Tilemap, no SpriteRenderer, no ScriptableObject, no WebSocket, no HTTP client.

### 2.5 Adapt or write fresh? Per planned file

The spec's Unity client is `RealmWeaverClient.cs` (HTTP GET `/chunk/{cx}/{cy}` JSON, GET `/asset/{id}.png` via `UnityWebRequestTexture`, non-readable), `AssetStreamer.cs` (priority queue, ≤4 coroutines, placeholder, byte-capped LRU with `Destroy`), `TileCatalog.cs` (ScriptableObject tile class → prefab), `RealmWeaverEvents.cs` (WebSocket `/events`, NativeWebSocket).

| Planned file | Closest branch file | Reusable share | Verdict and reasoning |
|---|---|---|---|
| `RealmWeaverClient.cs` | `MADWEController.cs` | ~0 | **Write fresh (~80 lines).** Everything in the controller is TCP/thread/newline plumbing. Only the lesson survives: keep DTOs flat for `JsonUtility` (section 7). |
| `AssetStreamer.cs` | `AssetLoadingSystem.cs` | ~170 of 371 lines | **ADAPT.** Keep the cache dictionary + `LinkedList` LRU + `Destroy` on evict (L141-159), the `UnityWebRequestTexture` coroutine (L100-139), the callback API with cache-hit LRU bump (L74-98), the load-queue coroutine with in-flight counter decremented in the callback (L249-280), `ClearCache`/`OnDestroy` (L288-308). Against a fresh ~180-line file this is roughly a wash in line count; adapting wins because the retained part is exactly where fresh streamers get bugs (eviction destroying a texture still bound to a renderer, in-flight counter never decremented on error, LRU not bumped on hit). |
| `TileCatalog.cs` | `TilePrefabManager.cs:5-20` | 16 lines | **ADAPT.** `TilePrefabSet{biome, mappings[]}` and `TilePrefabMapping{tileId, tileName, prefabVariations[], probability}` are the right serializable shape; wrap in a `[CreateAssetMenu] ScriptableObject`, key by tile-class **name** (the spec's tile classes are names), drop `tileId`. Keep `GetVariationIndex` (L210-215): deterministic prefab variation from world position. Drop the object pool and tinting. |
| `RealmWeaverEvents.cs` | `UnityMainThreadDispatcher.cs` | 0 | **Write fresh (~60 lines).** NativeWebSocket already marshals to the main thread through `DispatchMessageQueue()` in `Update`; a dispatcher is unnecessary. |
| chunk placement (inside `RealmWeaverClient` or a small `ChunkRenderer`) | `AdvancedTileManager.cs` | idea only | **IGNORE.** Borrow the structure (one container GameObject per chunk, `Dictionary<Vector2Int, Chunk>`, yield every N instantiations, clear-before-replace, L68-153) but write it for XY with `SpriteRenderer`/`Tilemap`; drop LOD, boundary smoothing, and the per-tile `Find`. |

### 2.6 Extraction commands (the only two)

```bash
# AssetStreamer skeleton — then apply the edit list below
git show origin/unity:MADWE-Unity/Assets/Scripts/AssetLoadingSystem.cs > <unity-client>/Assets/RealmWeaver/AssetStreamer.cs

# TileCatalog data classes + deterministic variation picker
git show origin/unity:MADWE-Unity/Assets/Scripts/TilePrefabManager.cs | sed -n '5,20p'
git show origin/unity:MADWE-Unity/Assets/Scripts/TilePrefabManager.cs | sed -n '210,215p'
```

(On Windows Git Bash, `git show origin/unity:.gitignore`-style paths that start with `.` get mangled by MSYS path conversion; prefix with `MSYS_NO_PATHCONV=1` or run from PowerShell.)

**Edit list for `AssetStreamer.cs` (line numbers of the extracted file):**

1. Delete the StreamingAssets fields (L9-11), `preloadCommonAssets` (L16), `materialCache`/`materialLRU` (L23, L29), `PreloadAssets` (L56-72), the whole material section (L161-247), the Python-data path and `MaterialData` (L310-370), and `using System.IO`.
2. `LoadTextureAsync` (L100-139): replace the `file://` platform branches (L102-114) with `string url = $"{baseUrl}/asset/{assetId}.png";` and call `UnityWebRequestTexture.GetTexture(url, nonReadable: true)`; keep `wrapMode = Repeat`; drop `anisoLevel`/`filterMode` or expose them.
3. Byte cap: replace `maxCachedTextures` (L14) with `long maxCacheBytes` (e.g. 256 MB); on insert add `tex.width * tex.height * 4` (or `Profiler.GetRuntimeMemorySizeLong(tex)`) to a running total and evict from the LRU head while over the cap (L144-155 becomes a `while`). Spec also wants distance-aware eviction on the Python side; the Unity side only needs bytes.
4. Priority: replace `Queue<AssetLoadRequest>` (L24) with a small binary heap or `SortedSet<(int priority, long seq, AssetLoadRequest)>` (`System.Collections.Generic.PriorityQueue` is not in Unity's .NET Standard 2.1 profile). `maxConcurrentLoads = 4` (L20).
5. Placeholder: hand a shared 2×2 placeholder texture to the callback immediately, then the real one when the request completes; `RealmWeaverEvents` triggers a re-request on `{type: ready}`.
6. Decrement `currentLoadCount` on the error path as well (L273-280 already routes through the callback, so keep that structure).

---

## 3. Item 2 — `src/models/nwsg/graph_network.py` (821 lines)

**Verdict: IGNORE.** There is no data model or algorithm beyond what `networkx` plus a `dict` give us, and the spec's graph (typed `MultiDiGraph`, World → Region → Chunk → Tile → Asset, CONTAINS/ADJACENT/INSTANCE_OF/STYLE_ANCHOR edges, DINOv2 style vectors, coherence score, JSON node-link) overlaps it nowhere.

What it is:

- `WorldNode` (L41-94): id, `NodeType` ∈ {region, biome, chunk, tile, entity, agent, player, narrative} — no World, no Asset — 3-D position, **6-tuple** 3-D bounds, untyped `connections: Set[str]`, an optional `torch.Tensor` embedding.
- `SpatialIndex` (L117-185): `rtree` 3-D R\*-tree. Our world is an integer chunk grid; neighbours are `(cx±1, cy±1)` arithmetic and a dict lookup. `rtree` needs `libspatialindex` and is not in `requirements.txt`. `get_state` (L159-166) pickles only `(rtree_id, node_id)` pairs under a key called `bounds`; `load_state` re-inserts from the node dict, so the pickled "spatial index data" carries nothing.
- `StateSynchronizer` (L187-232): a `deque` of local updates with "last-write-wins" resolution. No remote peer exists; `process_synchronization` (L585-614) re-applies the graph's own updates to itself.
- `NeuralWorldStateGraph` (L235-727): dict-of-sets adjacency (`add_edge`, BFS `get_connected_nodes`), a 1-second TTL query cache, snapshots with a "checksum" that hashes only the sorted node ids and a timestamp (L108-114, not content), rollback, `validate_consistency`, and a 3-layer `nn.TransformerEncoder` (L281-292). Node embeddings are `torch.randn` (L312) or a 10-feature vector zero-padded to 64 and passed through an untrained `nn.Linear` (L629-641). `compute_node_embeddings` (L616-652) has no caller; its output is not used for any decision.
- The only integration, `EnvironmentAgent._process_generation_queue` (L258-271), builds `WorldNode(bounds=4-tuple)`; `SpatialIndex.insert` with `dimension = 3` raises, `add_node`'s blanket `except` (L336-338) returns `False`, so every chunk insertion fails silently.

Borrowable idea only: the invariant checks in `validate_consistency` (orphan edges, node↔edge mismatch, index coverage) are a reasonable template for the spec's "graph invariants" tests — but as test assertions over the networkx graph, not as code.

---

## 4. Item 3 — `src/wfc/hierarchical_wfc.py` (217 lines)

**Verdict: IGNORE — placeholder, and a regression from `main`.**

- `NeuralConstraintLearner` (L34-58) is docstring'd `"""Placeholder for neural constraint learning"""`: an MLP encoder and a sigmoid pair scorer, never trained, never called. `generate_hierarchical_world(..., use_neural_constraints)` accepts the flag and ignores it (L147-193).
- `WaveFunctionCollapse.collapse` (L115-124) is literally `np.random.choice(tile_ids, size=self.grid_size, p=probabilities)` — comment: *"Simplified collapse - just random selection with weights"*. No wave, no entropy, no propagation, no contradiction handling. `Tile.edges` (L28-31) are defined for every tile and read by nothing.
- `_generate_regions` (L195-202) is `np.random.choice([101,102,103])`; the "hierarchy" is region id → biome name → random fill, with a hard-coded 128-tile region and 32-tile chunk.
- `EnvironmentAgent` calls `self.hierarchical_wfc.generate(size=..., start_level='tile', constraints=...)` (L288-292); no such method exists.
- `main`'s `src/wfc/base_wfc.py` (156 lines, unchanged on the branch) at least does min-entropy observation and stack propagation (see `02-pcg-and-systems-literature.md`); the spec replaces it with the AC-3 numpy solver anyway. The forest/desert weights here (grass 3.0, tree 1.0, path 0.5, water 0.2) are not worth keeping because the spec learns weights from the example map.

---

## 5. Item 4 — `src/unity_bridge/communication.py` (300 lines), `src/agents/protocols.py` (85 lines)

**Verdict: IGNORE as a wire format.** Keep nothing.

- `communication.py` is an **in-process** agent message bus: `Message` envelope (`message_id`, `sender_id`, `recipient_id`, `message_type`, `priority`, `payload`, `timestamp`, `ttl`, `correlation_id`, `metadata`, L45-93) with `to_json`/`from_json`; `MessageQueue` (priority heap with TTL expiry — `_clean_expired` pops and re-pushes the entire heap on every `get`, L142-153); `EventSubscription` with `*` glob patterns; `MessageRouter` with a `ThreadPoolExecutor`. The `MessageType` enum (L22-33) mixes bus primitives with agent verbs (`GENERATE_CONTENT`, `VALIDATE_COHERENCE`, `PREDICT_PLAYER`). This is exactly the class-hierarchy-plus-bus design the spec rejects ("Agents are roles, not a class hierarchy"); the spec's transport is HTTP request/response (correlation is implicit) plus one WebSocket event `{type: ready, chunk, asset}`.
- `protocols.py` defines `TerrainGenerationRequest(position, size, biome, adjacent_chunks, constraints)`, `AssetGenerationRequest(asset_type, style, resolution, prompt, variations)`, `CoordinationMessage`, `StateUpdateMessage`, `PredictionRequest(player_id, current_position, recent_actions, time_horizon, confidence_threshold)`. `AssetGenerationRequest` is a strict subset of the spec's `AssetSpec` (which adds `seed`, `steps`, `seamless`, `tier` and derives the content id). `narrative_agent.py:18` imports `NarrativeGenerationRequest` from here; it does not exist, so that module fails at import.
- `ipc_server.py` (the actual socket code) is covered in section 2.2: wrong port, wrong endianness, unregistered handlers, an `AttributeError` in the one handler. Not a starting point for FastAPI.

---

## 6. Item 5 — `src/agents/environment_agent.py` (595 lines), `src/agents/narrative_agent.py` (602 lines)

**Verdict: IGNORE.** Neither contains prompt tables.

`environment_agent.py`:
- Imports `HierarchicalWFC` and `NeuralWorldStateGraph` (L18-19) and uses both through APIs that do not work (sections 3-4). Chunk tiles come from `_generate_simple_tiles` (L310-335): weighted random per cell, in a Python double loop.
- Biome list is `forest, desert, snow, cyberpunk, dungeon` (L49); `valid_transitions` (L524-536) pairs them by hand. Two of five are off-spec; volcanic/underwater/sky are absent.
- `_on_player_moved` (L363-387) requests every chunk in a `(2r+1)²` square with Manhattan-distance priority — this is the **8-ring baseline** the spec's predictor must beat, which is a two-line comprehension in a test, not a reason to keep the file.
- `_send_to_unity` is `pass` (L437-440); consensus voting (L556-582) always accepts except for `chunk_size` changes.

`narrative_agent.py`:
- Out of scope per the spec ("LLM narrative/quest agents"). Loads `microsoft/DialoGPT-medium` at init (L63, L85-89), uses `int(time.time()) % len(items)` as randomness (L521-529), string-replace "personality filters" (L350-364), and six template sentences (L505-519, L546-555). Broken as committed: missing `NarrativeGenerationRequest` import (L18), `asdict` used without import (L441), `self.publish_event` (L393, L486) where `BaseAgent` only has `emit_event`.

The only diffusion prompts on the whole branch are five one-liners in `asset_agent.py:346-352` (`forest`, `desert`, `cyberpunk`, `dungeon`, `underwater`) feeding a `stabilityai/stable-diffusion-2-1` pipeline (L68) — a fourth base model. They are generic and inferior to the 48 prompts in `main`'s `scripts/generate_textures.py` / `generate_sprites.py`, which the spec already mines into the biome YAMLs. IGNORE.

---

## 7. Other files looked at

| File | Verdict | Note |
|------|---------|------|
| `.gitignore` (Unity block, L56-102) | IGNORE | `*.meta` + `!*.*.meta` ignores **folder** `.meta` files, which Unity requires in version control (the committed project has none); the `!unity_project/**` un-ignores point at a directory that does not exist. Use GitHub's `Unity.gitignore` for the new client directory. |
| `src/agents/tools.py` | IGNORE | Opens a `chromadb.PersistentClient` at import time (not in requirements); `reload_unity` posts to an HTTP endpoint nothing serves; `run_wfc` shells out to `venv\Scripts\python.exe`. |
| `src/agents/multi_agent_coordinator.py` | IGNORE | Imports `UnityBridge` from `unity_bridge.communication` (it lives in `ipc_server.py`) and `MessageBus` from `base_agent` (does not exist); constructs `EnvironmentAgent(agent_id=, config=, message_bus=)` against a `(config, router)` signature. Never ran. |
| `src/performance/optimization_framework.py` | IGNORE | An FPS/latency monitor that flips quality flags; no CUDA, no allocator, no pooling — nothing for the spec's memory pool. |
| `scripts/test_agent_communication.py`, `test_simple_message.py`, `test_wfc.py`, `run_environment_agent.py`, `run_madwe.py` | IGNORE | The first two exercise only the in-process router (no sockets); `test_wfc.py` plots the random fill; the runners call constructors with the wrong signatures. |
| 11 tile prefabs, `TilePrefab.prefab`, `MADWEManager.prefab`, 10 `.mat` files, 2 scenes | IGNORE | Primitive cubes/cylinders/spheres with Standard-shader tints for a 3-D world; the spec's tiles are textured 2-D quads/sprites driven by generated PNGs. |

---

## 8. Design inputs for the new bridge that fall out of this review

1. **Keep chunk JSON flat so `JsonUtility` can parse it without Newtonsoft.** `JsonUtility` cannot deserialise `List<List<int>>`, dictionaries, or multidimensional arrays (the exact failure in `MADWEController.cs:49`). Shape `GET /chunk/{cx}/{cy}` as `{cx, cy, biome, w, h, tiles:int[] (row-major), classes:string[], assets:[{tile_class, asset_id}], prefabs:[{tile_class, prefab}]}` — arrays of primitives and of flat `[Serializable]` objects only.
2. **No raw TCP, no framing code, no receive thread.** `UnityWebRequest` coroutines run on the main thread; NativeWebSocket (`https://github.com/endel/NativeWebSocket.git#upm`) marshals events through `DispatchMessageQueue()` in `Update`. `UnityMainThreadDispatcher` is unnecessary.
3. **Textures:** request with `UnityWebRequestTexture.GetTexture(url, nonReadable: true)`, set `wrapMode = Repeat`, cap the cache in bytes, `Destroy` on evict and in `OnDestroy` — the parts of `AssetLoadingSystem.cs` worth keeping.
4. **Never `renderer.material` per tile.** Use shared materials or `MaterialPropertyBlock`; swapping a tile's texture after a `ready` event should set the block, not instantiate a material.
5. **Fresh Unity 6.3 2D (URP) project** in the client directory, with the canonical Unity `.gitignore` (folder `.meta` files committed). Add NativeWebSocket as a git UPM dependency; do not add ML-Agents/Sentis.
6. **One tile-class vocabulary**, owned by the biome YAML and exposed by the bridge as names, ends the three-way integer-id drift documented in section 2.3 item 10.

---

## 9. Action checklist

- [ ] Do not merge or cherry-pick `origin/unity`.
- [ ] `git show origin/unity:MADWE-Unity/Assets/Scripts/AssetLoadingSystem.cs` → `AssetStreamer.cs`; apply the six-step edit list in section 2.6.
- [ ] `git show origin/unity:MADWE-Unity/Assets/Scripts/TilePrefabManager.cs | sed -n '5,20p'` (+ `210,215p`) → `TileCatalog.cs` as a `ScriptableObject` keyed by tile-class name.
- [ ] Write `RealmWeaverClient.cs` and `RealmWeaverEvents.cs` fresh (~80 + ~60 lines) against the flat JSON shape in section 8.
- [ ] Leave `MADWE-Unity/`, `graph_network.py`, `hierarchical_wfc.py`, `communication.py`, `protocols.py`, `ipc_server.py`, the agents, `tools.py`, the coordinator and the test scripts on the branch. Record the decision in an ADR if the branch is ever deleted.
