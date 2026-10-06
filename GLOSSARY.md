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
| **Asset** | One generated image: a seamless texture for a tile class or a sprite for a prop. Identified by a content id derived from its asset spec. |
| **Asset spec** | The complete request for one asset: biome, tile class or prompt, kind (texture / sprite), size, seed, steps, seamless flag, quality tier. |
| **Generator** | The module that turns an asset spec into pixels. Two adapters sit at this seam: the diffusion generator (GPU) and the procedural generator (CPU fallback, used by tests). |
| **Quality tier** | `draft` (few-step, used for prewarm) or `refine` (more steps, replaces draft when idle). Steps are the anytime knob. |
| **Seamless** | A texture whose left/right and top/bottom edges continue each other. Produced with circular padding in the generator's convolutions. |
| **Tileability** | Measured seam quality: ratio of wrapped-edge gradient to interior gradient, 1.0 = edge indistinguishable from interior. Reported per asset and as a percentage of assets above threshold. |
| **Style vector** | A DINOv2 image embedding of an asset; a biome's style vector is the running mean of its assets. |
| **Coherence score** | 0.6 × cosine(asset, biome style vector) + 0.4 × mean cosine(asset, adjacent assets). Below the biome threshold the asset is regenerated, at most twice. |
| **Style consistency** | Benchmark statistic: mean within-biome cosine minus cross-biome cosine of style vectors. |
| **World state graph** | The typed property graph that is the source of truth for a world: World → Region → Chunk → Tile → Asset nodes with CONTAINS, ADJACENT, INSTANCE_OF and STYLE_ANCHOR edges. Agents write to it only through validated records. |
| **Region** | A contiguous set of chunks sharing one biome. |
| **Prewarm** | Generating a chunk before the player requests it, based on the predictor. |
| **Predictor** | Order-2 Markov model over eight quantised headings that ranks neighbour chunks by visit probability, falling back to order-1 then constant velocity. |
| **Scheduler** | The queue that orders chunk requests by priority = visit probability / cost, keeps at most two generations in flight, and evicts from the byte-capped LRU cache. |
| **Memory pool** | Pre-allocated latent, noise and embedding buffers reused across generations, with resident weights, so the allocator makes zero new device allocations per image. |
| **Bridge** | The FastAPI service Unity talks to: HTTP for PNG assets and chunk JSON, WebSocket for player position and ready events. |
| **Prefab map** | The table from tile class to Unity prefab, shipped as a ScriptableObject on the Unity side and as JSON on the bridge side. |
| **Benchmark report** | A JSON file under `reports/` with seeds, counts, versions and every metric the README or dashboard shows. |
