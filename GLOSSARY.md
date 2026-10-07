# RealmWeaver Glossary

Domain language for RealmWeaver (MADWE). Code, tests, specs and ADRs use these words exactly.

| Term | Meaning |
|---|---|
| **Biome** | The unit of visual style and layout rules: forest, desert, snow, volcanic, underwater, sky. A biome owns a tileset, a prompt style, an optional LoRA adapter and a style vector. |
| **Tile class** | A semantic kind of tile inside a biome (grass, water, shore, rock, path, tree). Layout reasons about tile classes; assets give them pixels. |
| **Tileset** | A biome's tile classes plus the adjacency table and weights the solver uses. Learned from an example map, optionally tightened by hand rules. |
| **Example map** | A small hand-authored ASCII grid per biome. Every adjacent pair in it is an allowed adjacency; frequencies become weights. The single source of layout truth. |
| **Adjacency rule** | `allowed[a, dir, b]`: tile class `b` may sit in direction `dir` of tile class `a`. Directional, not symmetric by default. |
| **Layout** | A solved grid of tile classes for one chunk, produced by the WFC solver. Zero adjacency violations by construction. |
| **Chunk** | A fixed-size square of tiles (default 16×16) addressed by integer chunk coordinates `(cx, cy)`. The unit of generation, caching, prediction and streaming. |
| **Border constraint** | Tile classes fixed along a chunk edge so it matches an already-solved neighbour chunk. |
| **Contradiction** | A WFC state where some cell has no allowed tile class left. Handled by restart with a new seed, then by shrinking the problem. |
| **Asset** | One generated asset carrying the payload of its kind: a seamless texture for a tile class, a sprite for a prop, a mesh or an animation clip. Identified by a content id derived from its asset spec. |
| **Asset spec** | The complete request for one asset: biome, tile class or prompt, kind (texture / sprite / mesh / animation), size, seed, steps, seamless flag, quality tier. |
| **Payload** | The kind-typed content of an asset: texture or sprite pixels, a mesh's GLB bytes with its textures, an animation's clip bytes. It encodes itself (bytes plus media type: PNG or GLB) and renders a preview thumbnail; consumers ask the payload, never the pixels. |
| **Generator** | The module that turns an asset spec into an asset of its kind. Two adapters sit at this seam: the diffusion generator (GPU) and the procedural generator (CPU fallback, used by tests). |
| **Quality tier** | `draft` (few-step, used for prewarm) or `refine` (more steps, replaces draft when idle). Steps are the anytime knob. |
| **Seamless** | A texture whose left/right and top/bottom edges continue each other. Produced with circular padding in the generator's convolutions. |
| **Tileability** | Measured seam quality: ratio of wrapped-edge gradient to interior gradient, 1.0 = edge indistinguishable from interior. Reported per asset and as a percentage of assets above threshold. |
| **Style vector** | A DINOv2 image embedding of an asset; a biome's style vector is the running mean of its assets. |
| **Coherence score** | 0.6 × cosine(asset, biome style vector) + 0.4 × mean cosine(asset, adjacent assets). Below the biome threshold the asset is regenerated, at most twice. |
| **Style consistency** | Benchmark statistic: mean within-biome cosine minus cross-biome cosine of style vectors. |
| **World state graph** | The typed property graph that is the source of truth for a world: World → Region → Chunk → Tile → Asset nodes with CONTAINS, ADJACENT, INSTANCE_OF and STYLE_ANCHOR edges, plus the game's Region3D, Settlement, Landmark, NPC, Memory, Quest, Objective, Faction, Event and Save nodes with KNOWS, ASSIGNED, MEMBER_OF and TRIGGERS edges. Agents write to it only through validated records. |
| **Record schema** | The one declaration (`realmweaver/world/records.py`) of every node kind with its required and optional attributes and every edge kind with the (source kind, target kind) pairs it may join. `WorldStateGraph.add` and `link` check each write against it; `validate`, `to_json` and `from_json` derive from it. A **record** is one node read back: its id, kind and attributes. |
| **Region** | A contiguous set of chunks sharing one biome. |
| **Prewarm** | Generating a chunk before the player requests it, based on the predictor. |
| **Predictor** | Order-2 Markov model over eight quantised headings that ranks neighbour chunks by visit probability, falling back to order-1 then constant velocity. |
| **Scheduler** | The queue that orders chunk requests by priority = visit probability / cost, keeps at most two generations in flight, and evicts from the byte-capped LRU cache. |
| **Memory pool** | Pre-allocated latent, noise and embedding buffers reused across generations, with resident weights, so the allocator makes zero new device allocations per image. |
| **Bridge** | The FastAPI service Unity talks to: HTTP for encoded assets (PNG, GLB) and chunk JSON, WebSocket for player position and ready events. |
| **Prefab map** | The table from tile class to Unity prefab, shipped as a ScriptableObject on the Unity side and as JSON on the bridge side. |
| **Benchmark report** | A JSON file under `reports/` with seeds, counts, versions and every metric the README or dashboard shows. |
| **Runner** | The seam that decides on which thread a generation job runs: `World.submit` hands a job to the runner and gets back its handle (a `Future`); `slots` is how many jobs run at once. Two adapters: inline (the calling thread) and threaded (one worker per slot, player requests ahead of prewarm). |
| **Transition** | What a chunk request did to the chunk, reported in its `ChunkResult`: `created` (absent before), `refined` (a draft's assets upgraded in place) or `reused` (returned as it was). |
| **Region record** | The read record `region(name)` returns for a Region node: its node id (stable across save and load), name, biome and the keys of the chunks it contains; the region map's compiled tileset and coherence threshold join it with C3. |
| **Persona** | An NPC's identity, goals, voice and memory seed: name, role, faction, home and work places, the 24 h schedule, goals, memory seed, a voice id and an authored fallback line. A frozen record the runtime reads and never writes. |
| **Memory stream** | An NPC's timestamped memories: observations tagged by source (`world`, `self` or `player`) and reflections, retrieved by recency × importance × relevance. Reflections fold only the NPC's own observations, never player speech. |
| **Grounding verifier** | Checks every NPC statement against graph facts: each text-bearing field of an action may name only what the facts, persona or the NPC's own memories hold and claim only numbers they state; `give` and `offer_quest` must reference declared items and quests. A rejected action is retried once with the reason, then the fallback line is used. |
| **Action schema** | The JSON tool calls an NPC may emit: `say`, `give`, `offer_quest`, `refuse`, `report_crime` and `end`, a discriminated union validated before the grounding verifier sees it. |
| **Provenance record** | What a generated artifact carries about its making: model, licence, seed, prompt and reviewer, so every shipped asset, quest and NPC line can be traced and licence-checked. |
