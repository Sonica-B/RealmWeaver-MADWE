# 02 — PCG & Systems Literature Review for RealmWeaver-MADWE

**Scope.** Literature 2020–2026 (weighted toward 2024–2026) relevant to a real-time game-world
generation engine that orchestrates diffusion asset generators with constraint-based layout,
coordinated through a world-state graph, with speculative pre-generation, GPU memory pooling,
honest benchmarking, and a Unity client.

**Method.** Web search (standard + extended), arXiv abstract pages, PyPI JSON API, PyTorch and
Unity documentation, and vendor blogs, all retrieved on 2026-10-06. Every claim below carries a
numbered reference to the list at the end. arXiv ids that were taken from memory rather than
fetched during this review are marked "(unverified id)".

**Repo context that shaped the recommendations** (read during the review):

| File | What exists today | Gap vs. literature |
|---|---|---|
| `src/wfc/base_wfc.py` (156 lines) | Simple-tiled WFC in numpy; entropy heuristic; stack propagation; raises on contradiction | Adjacency mask is applied identically to all 4 directions (`load_rules`), no backtracking/restart, Python-level loop over tiles in `_propagate`, no boundary/seed constraints, no chunk border fixing |
| `src/models/diffusion/inference.py` | SDXL base 1.0, fp16, `enable_model_cpu_offload()`, VAE slicing/tiling, 1024², 30 steps, LoRA fuse per biome | Offloading defeats any memory-pooling/CUDA-graph strategy; 1024² SDXL at 30 steps is seconds-to-tens-of-seconds per texture on an 8 GB RTX 3060 (README target) |
| `src/unity_bridge/ipc_server.py` (55 lines) | Raw TCP, `recv(4096)` single read, JSON echo | No message framing, no binary path, no texture transport |
| `src/models/nwsg/`, `src/models/player_prediction/` | Empty packages | Need a data model and a predictor (Sections 2, 3) |

---

## 1. Wave Function Collapse and constraint-based PCG

### 1.1 Foundations

* **Gumin's WaveFunctionCollapse (2016)** [1] defines the two canonical models: the *overlapping
  model* (learn N×N patterns and their co-occurrence from an example bitmap; output contains only
  patterns present in the input) and the *simple-tiled model* (explicit tiles + adjacency table).
  Loop: pick the cell with minimum Shannon entropy, collapse it by weighted sampling, propagate.
  The reference implementation has **no backtracking**: a contradiction simply fails the run.
  Documented shipped uses: Bad North, Caves of Qud, Townscaper, The Matrix Awakens.
* **Karth & Smith, "WaveFunctionCollapse is Constraint Solving in the Wild" (FDG 2017)** [2] and the
  journal extension **"WaveFunctionCollapse: Content Generation via Constraint Solving and Machine
  Learning" (IEEE ToG 2021)** [3] give the rational reconstruction: WFC = a CSP with a learned
  binary adjacency constraint, solved by minimum-remaining-values + arc-consistency propagation.
  They probe the impact of backtracking, location/choice heuristics, and pattern classification.
* **Merrell, Model Synthesis (2007) and "Comparing Model Synthesis and WFC" (2021)** [4][5]: the
  two are "two versions of the same algorithm". Differences that matter for large outputs: model
  synthesis *modifies in blocks* (solve a sub-block while the rest is fixed), which lets it succeed
  on large grids where global WFC repeatedly fails. The AC-3-style propagator used by WFC is the
  same family as Merrell's.

### 1.2 Solver internals: propagation, backtracking, heuristics

* Propagation in stock WFC is **AC-3** (O(a·d³)); more advanced solvers use **AC-4** (O(a·d²), optimal
  but memory-hungry, which is why it is rarely used in game-side WFC) [6][7].
* **Backtracking** is a cheap, well-understood add-on. Karth's research implementation
  `ikarth/wfc_2019f` [8] exposes `backtracking=True/False`, location heuristics (entropy, lexical,
  hilbert, spiral, anti-entropy, random) and choice heuristics (weighted, random, rarest). The
  Karth & Smith experiments show the entropy heuristic plus modest backtracking is the sweet spot;
  for most game tilesets, *restart-on-contradiction* with a different seed is sufficient and simpler
  than true backjumping.
* **DeBroglie** (C#, Boris the Brave) [9] is the most complete practical constraint catalogue for
  WFC: Border, FixedTile, Path/connectivity, MaxConsecutive, Mirror/Symmetry, Count, Separation,
  PairSeparation, and custom constraints via `SetContradiction()`; backtracking is a solver option.
  This list is the right menu of "extra constraints" to budget for (we need FixedTile for chunk
  borders, Border for region edges, Count for density targets).

### 1.3 Scaling WFC to large / streaming worlds

* **Nested WFC (N-WFC)** — Nie, Zheng, Zhuang, Song, IEEE CoG 2023 (Best Paper nominee), journal
  version IEEE ToG 2024, arXiv 2308.07307 [10]. Several fixed-size internal WFCs are nested in an
  exterior process; with a *complete / sub-complete tileset* preparation strategy it generates
  "aperiodic and deterministic infinite content" with minimal conflict/backtracking, reducing time
  complexity from exponential to polynomial. This is exactly the chunked-generation regime we need:
  generate chunk-by-chunk, fixing the already-generated border cells of neighbours.
* **Punch Out Model Synthesis (POMS)** — Zzyzek, arXiv 2501.14786 (Jan 2025) [11]. Progressive
  sub-block realisation with *stochastic boundary erosion* when a sub-block fails; introduces the
  notion of **tile correlation length** to size blocks. Handles larger grids with less solution
  bias than Merrell's modify-in-blocks and Gumin's WFC.
* **"A Markovian Framing of WFC"** — Yiu … Togelius, Merino, Earle, arXiv 2509.09919 (Sep 2025)
  [12]. WFC as an MDP: the external optimiser (RL/evolution) maximises objectives while WFC
  propagation guarantees adjacency constraints. Decoupling beats joint constraint+objective
  evolution. Useful if we later want "make this region playable/dense" objectives on top of WFC.
* **Evolutionary WFC** — Rajesh, Khalifa, Togelius, arXiv 2607.02082 (Jul 2026) [13]: evolve the
  small *input example* rather than the level; WFC is the genotype→phenotype map. Works where
  properties are local; global constraints stay hard.
* **Learning Local Constraints for RL content generators** — Bhaumik, Togelius, Yannakakis,
  Khalifa, arXiv 2605.13570 (May 2026) [14]: WFC-learned adjacency constrains the action space of a
  PCGRL generator (Lode Runner), giving visual coherence + global playability.

### 1.4 Hierarchy, semantics, biomes/regions, and rewrite systems

* **Hierarchical Semantic WFC (HSWFC)** — Alaka & Bidarra, FDG 2023 [15]: adds *meta-tiles*
  (semantic traits) organised into a taxonomy ("meta-tree"); cells can collapse to a meta-tile first
  (e.g. "forest") and later refine to a concrete tile. A user study showed significantly lower
  designer cognitive load than flat WFC. This is the cleanest published formalisation of
  *biome-then-tile* two-level WFC.
* **"Procedural Generation of 3D Maps with WFC: Optimization and Advanced Constraints"**
  (Eurographics digital library) [16] catalogues practical extensions: tile weights, *layer-based
  generation*, unique-tile appearance constraints, automated neighbour creation.
* Theses on terrain WFC report that **biome-per-chunk** assignment makes biome borders visible
  [17]; the fix in the literature is a coarse biome layer whose border cells are transition
  meta-tiles (HSWFC) or a boundary blend of allowed tiles.
* **WFC on remote-sensing elevation** — Dajkhosh, arXiv 2412.04688 (Dec 2024) [18]: overlapping
  WFC on SRTM slope maps preserves input terrain statistics (histogram, mean/std) — a cheap
  evaluation template for "does our WFC output match the tile statistics of the example".
* **MarkovJunior** — Gumin (2022) [19]: a probabilistic language of rewrite rules with inference by
  constraint propagation; **Markov Senior** (arXiv 2408.05959, CoG 2024) [20] learns MarkovJunior
  grammars from user-specified content. Relevant as the "next step up" from WFC if we need
  programmatic structures (roads, rivers), but it is a different, heavier paradigm.

### 1.5 WFC/tile generation with diffusion models and LLMs

* No paper named "DiffWFC" was found in this review. The closest hybrids are:
  * **Text-to-level diffusion for Super Mario Bros** — Schrum et al., AIIDE 2025, arXiv 2507.00184
    [21]: diffusion directly over tile grids with automatic captions; a small from-scratch
    transformer text encoder beat large pretrained encoders.
  * **Procedural level generation with diffusion models from a single example** (2024) [22]:
    unconditional diffusion over one Mario/Minecraft level at multiple scales.
  * **MultiGen** — Po, Zhang, Hertz, Wetzstein, Wadhwa, Ruiz, arXiv 2603.06679 (Mar 2026) [23]:
    a diffusion game engine decomposed into Memory / Observation / Dynamics with an *explicit
    external memory* editable by the designer; multiplayer-consistent rollouts.
  * **SynCity** — Engstler et al., ICCV 2025, arXiv 2503.16420 [24] and **SynCity 3000** (arXiv
    2607.05392) [25]: tile-by-tile 3D world generation with Flux + TRELLIS, each tile generated
    with the adjacent tiles as 2D context and then blended. **BlockFusion** — Wu et al., SIGGRAPH
    2024, arXiv 2401.17053 [26]: diffusion over latent tri-planes extended block-by-block via
    extrapolation from overlapping regions. Both are the "diffusion side" of our design: *layout
    decides what, diffusion decides how it looks, and neighbours condition the generation.*
  * **Tiled Diffusion** — Madar & Fried, CVPR 2025, arXiv 2412.15185 [27]: imposes self-,
    one-to-one and many-to-many tiling constraints inside the diffusion process (seamless textures,
    360°). Directly applicable to tileable biome textures.
* **LLM tile placement**: Word2World (arXiv 2405.06686) [28], Word2Minecraft (arXiv 2503.16536)
  [29], MarioGPT (arXiv 2302.05981) [30], zero-shot dual-agent 3D map generation (arXiv 2512.10501)
  [31], narrative-to-scene (arXiv 2509.04481) [32]. Consensus: LLMs are good at *semantic layout
  and story-driven tile semantics*, weak at *hard local adjacency*; every system post-validates
  or repairs with a constraint pass. That argues for LLM/agents → region graph, WFC → tiles.

### 1.6 Python package audit (checked on PyPI 2026-10-06)

| Name | Status | Verdict |
|---|---|---|
| `wfc` (PyPI) | **Not WFC at all** — "WebForms Core Commander" by Elanat, v2.2.0 [33] | Do not install |
| `pywfc` (PyPI) | v0.0.0, single release 2021-04-02, classes Wave/State/Rule, no overlapping model, no backtracking, no dependencies declared [34] | Abandoned toy |
| `wavefunctioncollapse`, `wave-function-collapse` (PyPI) | 404 — do not exist [35] | — |
| `mxgmn` | Not a pip package; Gumin's code is C# [1] | — |
| `ikarth/wfc_2019f` (GitHub only) | Research-grade overlapping model, backtracking, 7 location + 4 choice heuristics, numpy/imageio, Python 3.10 [8] | Good for reading; not pip-installable, overlapping-only, slow |
| DeBroglie (C#/.NET) [9] | Mature constraints + backtracking | Right feature list, wrong language |

**Verdict:** no dependable Python package exists. A ~200-line numpy simple-tiled solver with
directional adjacency, AC-3 propagation, min-entropy (+ tiny noise) heuristic, restart-based
backtracking, and FixedTile/Border constraints is the correct build-vs-buy decision; it also keeps
the chunk-border protocol (Section 1.3) under our control.

---

## 2. World-state graph / scene-graph coordination for generative worlds

### 2.1 Scene-graph-conditioned generation

* **SG2IM** — Johnson, Gupta, Fei-Fei, CVPR 2018, arXiv 1804.01622 (unverified id) [36]: graph
  conv over (object, relation, object) triples → layout → image. The lineage of "graph → layout →
  pixels".
* **FlowScene** — Yang et al., arXiv 2603.19598 (Mar 2026) [37]: tri-branch rectified flow
  conditioned on a *multimodal scene graph* that jointly generates layout, shapes and textures;
  object information is exchanged *along graph edges during generation*, and the paper explicitly
  evaluates **style consistency** against language- and graph-conditioned baselines.
* **WorldGen** — Wang et al. (Meta), arXiv 2511.16825 (Nov 2025) [38]: LLM scene-layout reasoning
  → procedural generation → diffusion 3D generation → object-aware decomposition; "fine-grained
  control over layout, scale, and style". Architecturally the closest published analogue to
  MADWE's agents → WFC → diffusion pipeline.
* **3D Scene Generation: A Survey** — arXiv 2505.05474 [39]; **WorldExplorer** — arXiv 2506.01799
  [40]; SynCity/BlockFusion [24][26]; **VectorWorld** (streaming world model via diffusion flow on
  vector graphs, arXiv 2603.17652) [41].

### 2.2 Neural game engines and interactive world models (what they teach about state)

| System | Date | Key fact for us |
|---|---|---|
| GameNGen, arXiv 2408.14837 [42] | Aug 2024 | Diffusion next-frame DOOM at >20 FPS on one TPU; conditioning augmentation needed for stable autoregression |
| DIAMOND, arXiv 2405.12399 (unverified id) [43] | 2024 | Diffusion world models beat discrete tokens on Atari fidelity |
| Oasis (Decart/Etched) [44] | Oct 2024 | Real-time Minecraft-like; degrades after dozens of frames (no memory) |
| Genie, arXiv 2402.15391 (unverified id) [45]; Genie 2 [46]; Genie 3 [47]; Project Genie [48] | 2024–Jan 2026 | Genie 2 adds memory of off-screen regions; Genie 3: 11B autoregressive, 720p/24 fps, consistent for minutes; backtracking does not regenerate over old areas |
| PlayGen, arXiv 2412.00887 [49] | Dec 2024 | Autoregressive DiT + playability-based evaluation; 1000+ frames on an RTX 2060 |
| Promptable Game Models, arXiv 2303.13472 [50] | 2023 | Game = evolution of an explicit environment *state* driven by agent actions; NL prompts over states |
| GameFactory, arXiv 2501.08325 [51]; Matrix-Game 1/2/3, arXiv 2506.18701 / 2508.13009 / 2604.08995 [52][53][54]; Hunyuan-GameCraft, arXiv 2506.17201 [55] | 2025–2026 | 13–17B models; Matrix-Game 2.0 at 25 FPS; GameCraft 6.6 FPS with hybrid history conditioning; 3.0 adds long-horizon memory |
| MultiGen, arXiv 2603.06679 [23] | Mar 2026 | **Explicit external memory independent of context window**, updated by user edits |
| StatePlay, arXiv 2607.26754 [56] | Jul 2026 | Jointly predicts explicit game state (HP, timers) with frames; +18.6% mechanics fidelity vs. no explicit state |
| Scalable Generative Game Engine, arXiv 2602.00608 [57] | Jan 2026 | "Resolution wall" of neural engines; 720×480 at 26–48 FPS only with cluster-scale co-design |
| WorldRoamBench, arXiv 2606.31672 [58]; WBench, arXiv 2605.25874 [59] | 2026 | Define sliding-window drift, memory-retention and action-fidelity metrics; no model satisfies all |

The consistent lesson across 2024–2026: **coherence comes from explicit, external state** (Genie 2
memory, MultiGen memory, StatePlay state head), not from a bigger context window. MADWE's hybrid
(explicit graph + per-asset diffusion) sidesteps the resolution wall and drift problems of
end-to-end neural engines entirely, which is a legitimate positioning argument.

### 2.3 Graph world models for agents and games

* **Generative Agents** — Park et al., arXiv 2304.03442 (unverified id) [60]: memory stream +
  reflection + planning; the "world" is a tree of areas/objects with states.
* **AriGraph** — Anokhin et al., IJCAI 2025, arXiv 2407.04363 [61]: semantic KG + episodic edges
  as an LLM agent's world model; beats full-history/RAG in TextWorld.
* **KG world models of textual environments** — Ammanabrolu & Riedl (NeurIPS 2021, arXiv
  2106.09608, unverified id) [62]: world state as (entity, relation, entity) tuples added/deleted
  per step.
* **Schema-governed LLM pipeline for RPG narrative** — MDPI *Systems* 14(2):175, 2026 [63]: a
  formal schema of characters/quests/state transitions; LLM emits JSON that is normalised and
  validated for schema compliance, reachability and reference consistency. This is the pattern
  for `agents/` → graph writes.
* **Dormans 2010, "Adventures in Level Design: Generating Missions and Spaces"** [64]: graph
  grammars produce a mission graph, then a space graph — the classic regions→spaces hierarchy.
* **"Neural world state graph"**: no paper with that exact name was found. The closest formal
  objects are AriGraph (semantic+episodic graph), StatePlay's state head, MultiGen's external
  memory, and VectorWorld's vector graph. The name is therefore available, but the report should
  define it precisely (below) rather than imply prior art.

### 2.4 Style-coherence scoring in the literature

* **StyleAligned** — Hertz et al., arXiv 2312.02133 [65] evaluates set style consistency as the
  **pairwise mean cosine similarity of DINO ViT-B/8 embeddings**, explicitly preferring DINO over
  CLIP because CLIP conflates content with style. Follow-ups (AlignedGen 2509.17088 [66],
  Only-Style 2506.09916 [67]) keep the DINO-based protocol. The repo's
  `style_consistency_loss.py` should be checked against this protocol.
* FlowScene [37] propagates object information along graph edges and reports style consistency as
  a headline metric — the graph-edge-propagation idea is published.

### 2.5 Minimal formal structure that gives coherence (recommended data model)

A **typed property graph** (in-memory `networkx.DiGraph`, persisted as JSON; no database):

Node types (all have `id`, `seed`, `created_at`, `status ∈ {planned, generating, ready, rejected}`):

| Node | Properties |
|---|---|
| `World` | `global_seed`, `style_prompt`, `palette[8×RGB]` |
| `Region` | `biome`, `bbox`, `lora_id`, `style_vec` (768-d DINO, float16, EMA of accepted assets, initialised from one anchor image), `prompt_suffix` |
| `Chunk` | `(cx, cy)`, `grid[H×W] tile ids`, `wfc_seed`, `border_fixed{N,E,S,W}` |
| `Tile` | `tile_class`, `prefab_address`, `rotation`, `is_transition` |
| `Asset` | `kind ∈ {texture, sprite, prop}`, `path`, `sha256`, `gen_seed`, `steps`, `style_vec`, `coherence_score`, `size_px` |

Edge types: `CONTAINS` (World→Region→Chunk→Tile), `ADJACENT` (Chunk↔Chunk with `shared_border`
tile ids; Region↔Region with `blend_width`), `INSTANCE_OF` (Tile→Asset), `DERIVED_FROM`
(Asset→Asset for variants/inpaints), `STYLE_ANCHOR` (Region→Asset).

Embedding stored: **one** vector type, DINOv2/DINO ViT-B 768-d per Asset and per Region. (A CLIP
text embedding of the region prompt is cheap to add later but is not needed for the coherence
check.)

Coherence score for a new asset `a` in region `r` with adjacent ready assets `N(a)`:
`s = 0.6·cos(a.style_vec, r.style_vec) + 0.4·mean_{b∈N(a)} cos(a.style_vec, b.style_vec)`; accept
if `s ≥ τ_r`, where `τ_r` is calibrated once per biome as the 5th percentile of pairwise
similarities among the biome's reference images. On reject: retry with a new seed (max 2), then
fall back to the region anchor asset. Consistency check runs in three cheap passes: structural
(all `ADJACENT` chunk borders satisfy the adjacency table), semantic (every `Tile` has an `Asset`
of the right `tile_class`), stylistic (score above). Region–region borders use a linear blend of
the two `style_vec`s by distance to the boundary as the generation target for transition tiles.

---

## 3. Predictive / speculative pre-generation for latency hiding

### 3.1 What the literature says

* **Outatime** — Lee et al. (Microsoft), MobiSys 2015 [68]: speculative execution for cloud gaming
  masks up to 250 ms RTT with a **Markov model over recent user input** plus state-space
  subsampling, misprediction compensation and bandwidth compression; players "overwhelmingly
  prefer" it. The template for *predict → speculatively generate → swap in*.
* **Song, Kotz, Jain, He, IEEE TMC 2006** [69]: on two years of Wi-Fi traces (6,000 users),
  low-order Markov predictors matched or beat compression-based (LZ/PPM/SPM) predictors; the
  **order-2 Markov predictor with fallback** was best (median ≈ 72% next-cell accuracy for long
  traces). This is the strongest empirical argument for *not* training a GRU first.
* **NLPMM** — arXiv 2003.07037 [70]: Markov next-location prediction with time/object similarity;
  order-1 Markov alone is the weakest baseline when longer prefixes matter (top-1 ≈ 0.42–0.50),
  which is why order-2 with fallback is the recommended minimum.
* **Predictive scheduling for VR** — Hou et al., INFOCOM 2020, arXiv 1912.12672 [71]: two-phase
  (proactive + deadline) scheduling that stays robust when motion prediction is inaccurate — i.e.
  design the scheduler so a wrong prediction costs only wasted work, never a stall.
* Engine practice: velocity-based look-ahead (`lookahead ≥ max_speed × (t_io + t_decode +
  t_activate) + margin`), prediction-time settings and time-based cache retention in OpenMW's
  terrain/cell cache [72][73][74].
* **Anytime behaviour**: Zilberstein's anytime algorithms [75]; for diffusion, the step count
  *is* the anytime knob — few-step distillations (LCM arXiv 2310.04378, SDXL-Turbo/ADD arXiv
  2311.17042, unverified ids) [76][77] let a chunk ship a 4-step draft and later a 20-step refine.
* Caching: LRU is fine for a first version; ARC (Megiddo & Modha, FAST 2003) [78] if scan
  patterns (long straight runs) pollute LRU. Distance-based eviction (evict the chunk farthest
  from the predicted position) is the game-engine norm [72][74].

### 3.2 Simplest predictor that should beat "generate the 8 neighbours"

Note the baseline is already strong *if* generation is faster than chunk traversal. With SDXL at
seconds per texture on an 8 GB card, it is not; the predictor's value is **budget concentration**
(generate the 2–3 chunks the player will actually enter next, at higher quality) and **horizon
extension** (chunks two steps ahead along the predicted heading). Recommended:

1. Quantise movement into 8 headings + {stop}; state = last two headings (order-2), with fallback
   to order-1 and then to constant-velocity extrapolation (Song et al. [69]).
2. Predict a distribution over the next 2 chunk transitions; priority of chunk c =
   `P(visit c within horizon) / estimated_cost(c)`; feed a priority queue with a hard budget of
   in-flight generations (2 on an 8 GB GPU).
3. Keep the 8-neighbour ring as the *fallback* set (lowest priority, draft quality).

### 3.3 Offline evaluation protocol (hit-rate on synthetic walks)

* Walkers on a WFC map: (a) random-waypoint, (b) Lévy-walk-like (heavy-tailed run lengths), (c)
  goal-directed A* with heading noise, (d) replayed human sessions when available.
* Metrics per walk: **hit-rate** (fraction of chunk entries whose assets were `ready`), **stall
  ms** (time spent on placeholder), **waste ratio** (generated chunks never entered within T s),
  **GPU-seconds per visited chunk**.
* Simulate generation time with a fixed service-time model (measured p50/p95 from Section 5),
  run ≥ 20 seeds × 3 walker types, report mean ± 95% CI; predictors compared: 8-neighbour,
  velocity-only, order-1, order-2+fallback, (optional) tiny GRU trained on the synthetic walks.

---

## 4. GPU memory pooling for diffusion inference

### 4.1 PyTorch allocator facts (docs 2.14 [79], devlog 2026-06-01 [80])

* The **caching allocator** keeps freed blocks; `nvidia-smi` shows reserved, not live, memory.
  Use `memory_allocated()/max_memory_allocated()` for tensors and
  `memory_reserved()/max_memory_reserved()` for the allocator; `reset_peak_memory_stats()` between
  experiments [79].
* `PYTORCH_ALLOC_CONF` (alias `PYTORCH_CUDA_ALLOC_CONF`) options: `expandable_segments:True`
  (virtual-memory-backed segments that grow; largely eliminates allocation-order fragmentation),
  `max_split_size_mb`, `garbage_collection_threshold`, `roundup_power2_divisions`,
  `large_segment_size_mb` (default 20 MB), `per_process_memory_fraction`, `backend:cudaMallocAsync`
  [79]. Settings can also be changed at runtime with
  `torch.cuda.memory._set_allocator_settings("expandable_segments:True")` [81].
* Fragmentation persists even with expandable segments when (a) live blocks pin splits in place
  and (b) allocations cross the 1 MB small/large pool boundary; the recommended mitigations are to
  free completely between phases, keep sizes consistently on one side of 1 MB, and for CUDA
  graphs capture in decreasing-size order or rely on complete freeing between captures [80].
* `torch.cuda.MemPool(allocator)` + `torch.cuda.use_mem_pool(pool)` (PyTorch ≥ 2.5) scope a
  region of code to its own pool; `CUDAPluggableAllocator` swaps the whole allocator for a custom
  `.so` and must be loaded before the first allocation [79]. Neither is needed for a single-process
  diffusion server; they matter for multi-tenant or NCCL/managed-memory cases.
* **CUDA graphs**: static shapes and static addresses, capture on a side stream after warm-up,
  copy inputs into long-lived tensors before replay; graphs share a private pool and can share it
  across sequentially replayed graphs (`pool=g1.pool()`); `make_graphed_callables` for partial
  capture [79].
* **Measurement tools**: `torch.cuda.memory._record_memory_history()` →
  `_dump_snapshot("x.pickle")` → pytorch.org/memory_viz shows the active-memory timeline and the
  allocator-state history with stack traces [82]; `torch.cuda.memory_stats()` exposes
  `num_device_alloc`, `num_device_free`, `num_alloc_retries`, `num_ooms`, `allocated_bytes`,
  `reserved_bytes`, `active_bytes`, `inactive_split_bytes`, `requested_bytes` [83]; the PyTorch
  "Understanding GPU Memory" series [84] walks through snapshots.

### 4.2 torch.compile / CUDA graphs for diffusers (2025)

* PyTorch blog "torch.compile and Diffusers: a hands-on guide" (Jul 2025) [85]: compiling only the
  denoiser with `fullgraph=True` gave **1.5×** (Flux on H100, 6.7 s → 4.5 s); **regional
  compilation** (`compile_repeated_blocks`) cut cold compile from 67.4 s to 9.6 s (2.4 s warm) at
  the same speed-up; `dynamic=True` avoids resolution recompiles; LoRA hot-swap with a fixed
  `max_rank` avoids recompiles when switching biome LoRAs; NF4 weights + compile = 1.5× at 15 GB.
* diffusers' "Accelerate inference" guide [86]: `mode="max-autotune"` compiles to a CUDA graph;
  `mode="reduce-overhead"` is the lighter CUDA-graph mode; keep scheduler `sigmas` on CPU to avoid
  device syncs; use `return_dict=False` for fullgraph; `channels_last` for UNet/VAE.
* Compiled/CUDA-graphed pipelines require a **fixed resolution, batch size and step schedule** —
  which the asset generator can guarantee (e.g. 512² textures, batch 1–2, fixed steps).

### 4.3 What "memory pooling" should concretely mean in MADWE

The current `inference.py` uses `enable_model_cpu_offload()`, so every texture pays PCIe traffic
and allocator churn, and no CUDA graph can be captured. "Memory pooling" should therefore be
defined as **one resident, pre-allocated, graph-captured generation slot**:

1. Weights resident on GPU (SD1.5 fp16 ≈ 2 GB, or SDXL with NF4 UNet for 8 GB cards) — no offload.
2. Pre-allocated static buffers: latents, noise, prompt embeddings, timestep tensor, VAE output —
   allocated once, reused via `copy_()`; batch and resolution fixed per asset class.
3. UNet step captured as a CUDA graph (`torch.compile(mode="reduce-overhead", fullgraph=True)` on
   the UNet only, or manual `torch.cuda.graph`), VAE decode compiled separately.
4. Allocator pinned with `expandable_segments:True` and a `garbage_collection_threshold` so the
   steady-state reserved size is flat.
5. Optional: a `MemPool` for the "draft" (few-step) and "refine" slots if both must be resident.

**Before/after numbers to report** (same seeds, same prompts, n = 50 after 5 warm-ups):
`num_device_alloc` and `num_alloc_retries` per image (target: 0 after warm-up), peak
`max_memory_allocated` and `max_memory_reserved`, the reserved−allocated gap (fragmentation),
p50/p95 per-texture latency, and textures/minute. A memory_viz snapshot before and after is the
qualitative evidence.

---

## 5. Quality / throughput benchmarking methodology

### 5.1 Distribution metrics with small n

* **FID is biased in n and the bias is model-dependent** — Chong & Forsyth, CVPR 2020, arXiv
  1911.07023 [87]: comparisons at any fixed n can flip; they propose extrapolating to FID∞.
* **KID** (MMD with polynomial kernel) is unbiased and usable at small n — Bińkowski et al.,
  ICLR 2018, arXiv 1801.01401 (unverified id) [88]; report mean ± std over random subsets.
* **clean-fid** — Parmar et al., CVPR 2022, arXiv 2104.11222 (unverified id) [89]: resizing/
  quantisation differences across libraries change FID materially; pin the implementation.
* **FD-DINOv2** — Stein et al., NeurIPS 2023, arXiv 2306.04675 (unverified id) [90]: Inception
  features unfairly penalise diffusion models; DINOv2 features agree better with humans.
* **CMMD** — Jayasumana et al., CVPR 2024, arXiv 2401.09603 [91]: CLIP embeddings + Gaussian-RBF
  MMD; unbiased, sample-efficient, consistent across n; FID disagreed with humans and failed to
  track iterative improvements.
* Practical rule: with a few hundred generated tiles per biome, report **KID (or CMMD) at a fixed n
  with subset std**, optionally FD-DINOv2; never a single-n FID headline.

### 5.2 Tileability and style-consistency metrics

* **TexTile** — Rodriguez-Pardo et al., CVPR 2024, arXiv 2403.12961 [92]: a differentiable
  classifier-based tileability score; usable as a loss and as an evaluation; existing metrics were
  "uncorrelated" with tileability. Tiled Diffusion [27] is the matching generator-side constraint.
  A cheap in-house complement: mean absolute gradient across the wrapped seam divided by the
  interior mean gradient (a relative-total-variation style seam score, cf. the RTV metric in [93]).
* **Style consistency**: pairwise mean DINO cosine within a biome set (StyleAligned protocol [65]);
  report alongside the cross-biome mean as a contrast so a degenerate "everything looks the same"
  generator cannot score well.
* **Long-horizon consistency** (if a world walk is evaluated): borrow WorldRoamBench's
  sliding-window drift and revisit/memory-retention definitions [58].

### 5.3 Throughput, latency and FPS

* Per-asset latency: p50/p95 over n ≥ 50 after warm-up, fixed seeds and prompts, fixed
  (resolution, steps, batch, scheduler); diffusers' own benchmark workflow exports a consolidated
  CSV per configuration [85][86] — mirror that format. MLPerf Inference (arXiv 1911.02549,
  unverified id) [94] is the canonical reference for scenario definitions (single-stream latency
  percentiles vs. offline throughput).
* textures/minute = images / wall-clock over a ≥ 5-minute steady-state run (not from one batch).
* Unity: **FrameTimingManager** gives CPU main/render-thread and GPU frame times per frame
  (`FrameTiming`) and bottleneck attribution [95]; the **Performance Testing package 3.0.x**
  (`com.unity.test-framework.performance`) records frame-time samples in Play Mode tests [96].
  Report p50/p99 frame time (ms) during a scripted, deterministic walk with assets streaming in —
  never a mean FPS.
* Reproducibility checklist for every number: git commit, model/LoRA hashes, seeds, n, GPU +
  driver + CUDA + torch versions, allocator config, warm-up count, CI method (bootstrap, 1,000
  resamples), and the raw CSV committed under `docs/research/bench/`.

---

## 6. Unity integration (2026)

### 6.1 Versions

* **Unity 6.3 LTS** is the current long-term-support release (support until December 2027;
  Unity 6.0 LTS until October 2026) [97]. Non-LTS update releases continue (6.4, 6.5 and 6.6
  manuals are live; a 6000.7 manual exists in draft) [98][99]. Target **6.3 LTS**; nothing in the
  design needs a 6.4+ feature.
* **Inference Engine (Sentis) 2.6.1**, April 2026: ONNX opset ≤ 25; CPU, GPUCompute and GPUPixel
  backends [100]. **ML-Agents 4.x** (4.0.0 Sep 2025 … 4.1.0 for 6000.6) is built on Inference
  Engine and talks to Python over gRPC [101][102].

### 6.2 Can Sentis run an SD1.5-class UNet in-engine?

Evidence is thin and negative for real-time use: the only public data point found is a 2023
thread where importing a 1.6 GB fp16 SD-2.1 UNet took 8–15 minutes and ~20 GB of disk/page file
on a 12 GB-RAM machine; Unity acknowledged "inefficiencies for larger models" and later reduced
import RAM [103]. No official SD sample exists; Unity's Hugging Face integration showcases small
models pre-converted to `.sentis` [104]. Sentis executes via compute shaders without
flash-attention or tensor-core GEMM paths, so even if the 860M-parameter UNet imports, it will be
several times slower than CUDA + torch.compile and will contend with the game's own GPU budget.
**Verdict: keep diffusion in the Python process; Sentis is appropriate only for tiny in-engine
models (e.g. a movement predictor or a style scorer), and even those are simpler in Python.**

### 6.3 Transport options

| Option | 2026 status | Fit |
|---|---|---|
| Raw TCP + JSON (current stub) | Works; stub has no framing (`recv(4096)` once) and no binary path | Keep only with 4-byte length prefix |
| **WebSocket** | `NativeWebSocket` wraps `System.Net.WebSockets.ClientWebSocket`, no DLLs, WebGL/mobile support [105]; Python side `websockets`/FastAPI | Best for bidirectional events (player position, "asset ready") |
| **HTTP + PNG** | `UnityWebRequestTexture.GetTexture(uri, nonReadable)` decodes **on a worker thread** and uploads directly [106]; `ImageConversion.LoadImage` is main-thread and stalls 100 ms–s for large images [107][108] | Best for asset payloads |
| gRPC | `Grpc.Core` deprecated; `grpc-dotnet` needs HTTP/2, which Unity lacks natively → requires `YetAnotherHttpHandler` [109] | Over-engineered here |
| ZeroMQ/NetMQ | Works with `AsyncIO.ForceDotNet.Force()` and manual context teardown; historic IL2CPP `Socket.Select` issues [110] | Unnecessary dependency |
| ML-Agents side channels | gRPC-based, tied to the training loop [101] | Wrong abstraction |

### 6.4 Texture formats and runtime loading

* PNG → `ARGB32`/`RGB24` uncompressed in VRAM (512² RGBA = 1 MB; 1024² = 4 MB). Fine for a
  prototype; GPU-compressed formats cut VRAM 4–8×.
* Raw RGBA via `Texture2D.LoadRawTextureData` + `Apply(updateMipmaps:false,
  makeNoLongerReadable:true)` avoids decode entirely at 4× the bandwidth of PNG [111].
* **KTX2 / Basis Universal** via `com.unity.cloud.ktx` 3.x transcodes on the C# job system to
  BC7/ASTC/ETC2 per platform with async `LoadFromBytes/LoadFromUrl` [112][113]. This is the upgrade
  path once VRAM or transfer becomes the bottleneck; it needs a Basis encoder in Python
  (`basisu` CLI or `pybasisu`).
* **Addressables 2.x** (2.2–2.7 for Unity 6) loads prefabs/materials asynchronously from local or
  remote bundles; `LoadAssetAsync<Texture2D>` with coroutines or `await handle.Task` [114]. Use it
  for the static tile prefab catalogue, not for generated textures.
* Mipmaps: `UnityWebRequestTexture` produces no mipmaps unless `DownloadedTextureParams`
  requests them; for tile textures viewed at varying distance, request mipmaps or use KTX2 with
  embedded levels [106].

### 6.5 Minimal C# client design

```
AssetStreamer : MonoBehaviour
  ├─ WebSocket (NativeWebSocket) "ws://127.0.0.1:5005/events"
  │     → sends {t, chunk, heading} at 5 Hz;  ← receives {asset_id, chunk, tile, url, sha256}
  ├─ Queue<AssetJob> prioritised by distance to player
  ├─ ≤ 4 concurrent coroutines: UnityWebRequestTexture.GetTexture(url, nonReadable:true)
  │     → Texture2D → MaterialPropertyBlock("_BaseMap") on the chunk's renderer
  │       (or Sprite.Create for 2D) ; placeholder texture until then
  └─ LRU<asset_id, Texture2D> capped by bytes; Destroy(tex) on evict; re-fetch on miss
TileCatalog : ScriptableObject   (tile_class → Addressables prefab address + material slot)
```

Python side: FastAPI (or aiohttp) serving `GET /asset/{id}.png` from the generated-asset cache and
a `/events` WebSocket; the generation scheduler (Section 3) pushes "ready" messages. The existing
`ipc_server.py` becomes unnecessary; if TCP is kept for any reason, frame every message with a
4-byte big-endian length.

---

## DESIGN RECOMMENDATIONS

**1. WFC / layout**
- Write our own ~200-line numpy simple-tiled solver; no PyPI package is viable (`wfc` is a web
  framework, `pywfc` is a 2021 v0.0.0 toy) [33][34].
- Directional adjacency table `allowed[T,4,T]` learned from example maps (overlapping-model
  statistics) plus hand rules; fix the current same-mask-all-directions bug.
- AC-3 propagation vectorised as boolean matmul; min-entropy + tiny noise heuristic.
- Backtracking = restart with new seed (max 5), then shrink to a sub-block (POMS-style erosion) [11].
- Chunked generation with FixedTile borders copied from ready neighbours (Nested WFC) [10].
- Two levels: coarse biome meta-tiles (HSWFC) → concrete tiles; transition tiles at region borders [15].
- Keep DeBroglie's Border/FixedTile/Count constraints in scope; skip Path and Symmetry [9].
- Validate output statistics against the example tile histogram [18].

**2. World-state graph**
- Typed property graph: World→Region→Chunk→Tile→Asset nodes; CONTAINS, ADJACENT, INSTANCE_OF,
  DERIVED_FROM, STYLE_ANCHOR edges; `networkx` + JSON, no database.
- One embedding type: 768-d DINO per Asset and an EMA `style_vec` per Region [65].
- Coherence score = 0.6·cos(asset, region) + 0.4·mean cos(asset, adjacent assets); threshold per
  biome from reference 5th percentile; two retries then anchor fallback.
- Three-pass consistency check: structural (borders), semantic (tile class), stylistic (score).
- Agents write to the graph only via schema-validated JSON [63]; the graph, not the LLM context, is
  the source of truth (MultiGen/StatePlay lesson) [23][56].

**3. Speculative pre-generation**
- Order-2 Markov over 8 quantised headings with fallback to order-1 then constant velocity [69].
- Priority = P(visit within 2 chunks)/cost; ≤ 2 in-flight generations; 8-neighbour ring as
  draft-quality fallback.
- Two quality tiers: 4-step draft first, 20-step refine later (steps = anytime knob) [76].
- LRU by bytes, distance-aware eviction; measure hit-rate, stall ms, waste ratio on ≥ 20 seeds of
  random-waypoint, Lévy and A* walkers with a measured service-time model.

**4. GPU memory pooling**
- Drop `enable_model_cpu_offload`; weights resident (SD1.5 fp16, or SDXL NF4 on 8 GB).
- Static latent/noise/embedding buffers reused via `copy_()`; fixed 512² / batch / steps.
- UNet under `torch.compile(mode="reduce-overhead", fullgraph=True)`; regional compile to cut
  cold start [85]; LoRA hot-swap with fixed `max_rank`.
- `expandable_segments:True` + `garbage_collection_threshold`.
- Report before/after: `num_device_alloc` and `num_alloc_retries` per image (target 0),
  `max_memory_allocated/reserved`, reserved−allocated gap, p50/p95 latency, memory_viz snapshots [79][82][83].

**5. Benchmarking**
- KID (subset mean ± std) and CMMD at fixed n; never single-n FID [87][88][91]; pin clean-fid-style
  preprocessing [89].
- TexTile + wrapped-seam gradient ratio for tileability [92]; within- vs cross-biome DINO cosine
  for style [65].
- Latency p50/p95, textures/minute over ≥ 5-minute steady state, n ≥ 50 after 5 warm-ups, 3 runs.
- Unity: FrameTimingManager p50/p99 frame time during a scripted walk via the Performance Testing
  package [95][96]; commit raw CSVs, seeds, hashes, versions.

**6. Unity integration**
- Target Unity 6.3 LTS [97]; keep diffusion in Python — Sentis cannot run an SD UNet usefully [103].
- HTTP GET PNG via `UnityWebRequestTexture` (worker-thread decode, `nonReadable:true`) for assets;
  NativeWebSocket JSON for position/ready events [105][106].
- Replace the 55-line TCP stub with FastAPI (`/asset/{id}.png`, `/events`); if TCP stays, add a
  4-byte length prefix.
- `AssetStreamer` MonoBehaviour: priority queue, ≤ 4 coroutines, MaterialPropertyBlock apply,
  placeholder texture, byte-capped LRU with `Destroy`.
- `TileCatalog` ScriptableObject maps tile class → Addressables prefab [114]; KTX2/Basis via
  `com.unity.cloud.ktx` only when VRAM/transfer is measured as the bottleneck [112].

---

## References

1. Gumin, M. WaveFunctionCollapse (2016). https://github.com/mxgmn/WaveFunctionCollapse
2. Karth, I., Smith, A. M. "WaveFunctionCollapse is Constraint Solving in the Wild." FDG 2017. https://escholarship.org/uc/item/3rm1w0mn
3. Karth, I., Smith, A. M. "WaveFunctionCollapse: Content Generation via Constraint Solving and Machine Learning." IEEE Trans. Games 2021. https://doi.org/10.1109/TG.2021.3076368
4. Merrell, P. Model Synthesis (2007–). https://paulmerrell.org/model-synthesis/
5. Merrell, P. "Comparing Model Synthesis and Wave Function Collapse" (2021). https://paulmerrell.org/wp-content/uploads/2021/07/comparison.pdf
6. Boris the Brave. "Wave Function Collapse Explained" (2020). https://boristhebrave.com/2020/04/13/wave-function-collapse-explained
7. Boris the Brave. "Advanced Table Constraints" (AC-3 vs AC-4). https://www.boristhebrave.com/?p=1124
8. Karth, I. wfc_2019f (research implementation). https://github.com/ikarth/wfc_2019f
9. DeBroglie constraints documentation. https://boristhebrave.github.io/DeBroglie/articles/constraints.html
10. Nie, Y., Zheng, S., Zhuang, Z., Song, X. "Extend Wave Function Collapse to Large-Scale Content Generation" (Nested WFC). IEEE CoG 2023; IEEE ToG 2024. arXiv:2308.07307. https://arxiv.org/abs/2308.07307
11. Zzyzek, Z. "Punch Out Model Synthesis." arXiv:2501.14786 (2025). https://arxiv.org/abs/2501.14786
12. Yiu, F. et al. "A Markovian Framing of WaveFunctionCollapse." arXiv:2509.09919 (2025). https://arxiv.org/abs/2509.09919
13. Rajesh, D., Khalifa, A., Togelius, J. "Evolutionary Wave Function Collapse." arXiv:2607.02082 (2026). https://arxiv.org/abs/2607.02082
14. Bhaumik, D., Togelius, J., Yannakakis, G. N., Khalifa, A. "Learning Local Constraints for Reinforcement-Learned Content Generators." arXiv:2605.13570 (2026). https://arxiv.org/abs/2605.13570
15. Alaka, S., Bidarra, R. "Hierarchical Semantic Wave Function Collapse." FDG 2023. https://publications.graphics.tudelft.nl/papers/23 ; slides https://graphics.tudelft.nl/Publications-new/2024/AB24/HSWFC.CL.slides.pdf
16. "Procedural Generation of 3D Maps with Wave Function Collapse: Optimization and Advanced Constraints." Eurographics Digital Library. https://diglib.eg.org/items/112ca5b1-8e8b-49f0-b28b-25164aad9104/full
17. University of Malta thesis on unlimited-size WFC terrain with per-chunk biomes. https://www.um.edu.mt/library/oar/bitstream/123456789/107911/1/2120IDGIDG500005068161_1.PDF
18. Dajkhosh, S. "Utilizing WaveFunctionCollapse for Procedural Generation of Terrains using Remotely Sensed Elevation Data." arXiv:2412.04688 (2024). https://arxiv.org/abs/2412.04688
19. Gumin, M. MarkovJunior (2022). https://github.com/mxgmn/MarkovJunior
20. "Markov Senior — Learning Markov Junior Grammars to Generate User-specified Content." arXiv:2408.05959; IEEE CoG 2024. https://arxiv.org/abs/2408.05959
21. Schrum, J. et al. "Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros." AIIDE 2025. arXiv:2507.00184. https://arxiv.org/abs/2507.00184
22. "Procedural Level Generation with Diffusion Models from a Single Example" (2024). Summary: https://liner.com/review/procedural-level-generation-with-diffusion-models-from-single-example
23. Po, R. et al. "MultiGen: Level-Design for Editable Multiplayer Worlds in Diffusion Game Engines." arXiv:2603.06679 (2026). https://arxiv.org/abs/2603.06679
24. Engstler, P. et al. "SynCity: Training-Free Generation of 3D Worlds." ICCV 2025. arXiv:2503.16420. https://arxiv.org/abs/2503.16420
25. "SynCity 3000: Bootstrapping Scene-Scale 3D Diffusion." arXiv:2607.05392 (2026). https://arxiv.org/abs/2607.05392
26. Wu, Z. et al. "BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation." SIGGRAPH 2024. arXiv:2401.17053. https://arxiv.org/abs/2401.17053
27. Madar, O., Fried, O. "Tiled Diffusion." CVPR 2025. arXiv:2412.15185. https://arxiv.org/abs/2412.15185
28. Nasir, M. U. et al. "Word2World: Generating Stories and Worlds through Large Language Models." arXiv:2405.06686 (2024). https://arxiv.org/abs/2405.06686
29. "Word2Minecraft: Generating 3D Game Levels through Large Language Models." arXiv:2503.16536 (2025). https://arxiv.org/abs/2503.16536
30. Sudhakaran, S. et al. "MarioGPT: Open-Ended Text2Level Generation through LLMs." arXiv:2302.05981. https://arxiv.org/abs/2302.05981
31. "Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for PCG." arXiv:2512.10501 (2025). https://arxiv.org/abs/2512.10501
32. "Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments." arXiv:2509.04481 (2025). https://arxiv.org/abs/2509.04481
33. PyPI `wfc` (WebForms Core Commander, Elanat, v2.2.0, 2026-10-05). https://pypi.org/project/wfc/
34. PyPI `pywfc` v0.0.0 (2021-04-02). https://pypi.org/project/pywfc/ ; https://github.com/FoxNerdSaysMoo/PyWFC
35. PyPI `wavefunctioncollapse`, `wave-function-collapse`: HTTP 404 on 2026-10-06.
36. Johnson, J., Gupta, A., Fei-Fei, L. "Image Generation from Scene Graphs." CVPR 2018. arXiv:1804.01622 (unverified id).
37. Yang, Z. et al. "FlowScene: Style-Consistent Indoor Scene Generation with Multimodal Graph Rectified Flow." arXiv:2603.19598 (2026). https://arxiv.org/abs/2603.19598
38. Wang, D. et al. "WorldGen: From Text to Traversable and Interactive 3D Worlds." arXiv:2511.16825 (2025). https://arxiv.org/abs/2511.16825
39. "3D Scene Generation: A Survey." arXiv:2505.05474 (2025). https://arxiv.org/abs/2505.05474
40. "WorldExplorer: Towards Generating Fully Navigable 3D Scenes." arXiv:2506.01799 (2025). https://arxiv.org/abs/2506.01799
41. "VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs." arXiv:2603.17652 (2026). https://arxiv.org/abs/2603.17652
42. Valevski, D. et al. "Diffusion Models Are Real-Time Game Engines" (GameNGen). ICLR 2025. arXiv:2408.14837. https://arxiv.org/abs/2408.14837
43. Alonso, E. et al. "Diffusion for World Modeling: Visual Details Matter in Atari" (DIAMOND). NeurIPS 2024. arXiv:2405.12399 (unverified id).
44. Decart/Etched, "Oasis" (31 Oct 2024). Coverage: https://techcrunch.com/2024/10/31/decarts-ai-simulates-a-real-time-playable-version-of-minecraft ; https://www.infoq.com/news/2024/11/decart-etched-oasis
45. Bruce, J. et al. "Genie: Generative Interactive Environments." ICML 2024. arXiv:2402.15391 (unverified id).
46. Google DeepMind, "Genie 2: A large-scale foundation world model" (4 Dec 2024). https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
47. Google DeepMind, "Genie 3: A new frontier for world models" (5 Aug 2025). https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
48. The Register, "Google's Project Genie turns prompts into interactive worlds" (29 Jan 2026). https://www.theregister.com/2026/01/29/googles_project_genie_ai
49. Yang, Y. et al. "Playable Game Generation" (PlayGen). arXiv:2412.00887 (2024). https://arxiv.org/abs/2412.00887
50. Menapace, W. et al. "Promptable Game Models: Text-Guided Game Simulation via Masked Diffusion Models." arXiv:2303.13472. https://arxiv.org/abs/2303.13472
51. "GameFactory: Creating New Games with Generative Interactive Videos." arXiv:2501.08325 (2025). https://arxiv.org/abs/2501.08325
52. "Matrix-Game: Interactive World Foundation Model." arXiv:2506.18701 (2025). https://arxiv.org/abs/2506.18701
53. "Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model." arXiv:2508.13009 (2025). https://arxiv.org/abs/2508.13009
54. "Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory." arXiv:2604.08995 (2026). https://arxiv.org/abs/2604.08995
55. "Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition." arXiv:2506.17201 (2025). https://arxiv.org/abs/2506.17201
56. Lin, Z. et al. "StatePlay: State-Aware Game World Models for Mechanics-Consistent Generation." arXiv:2607.26754 (2026). https://arxiv.org/abs/2607.26754
57. Zeng, W. et al. "Scalable Generative Game Engine: Breaking the Resolution Wall via Hardware-Algorithm Co-Design." arXiv:2602.00608 (2026). https://arxiv.org/abs/2602.00608
58. Xu, T.-B. et al. "WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of Interactive World Models." arXiv:2606.31672 (2026). https://arxiv.org/abs/2606.31672
59. "WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation." arXiv:2605.25874 (2026). https://arxiv.org/abs/2605.25874
60. Park, J. S. et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023. arXiv:2304.03442 (unverified id).
61. Anokhin, P. et al. "AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents." IJCAI 2025. arXiv:2407.04363. https://www.ijcai.org/proceedings/2025/0002
62. Ammanabrolu, P., Riedl, M. "Learning Knowledge Graph-based World Models of Textual Environments." NeurIPS 2021. arXiv:2106.09608 (unverified id). https://openreview.net/pdf?id=o24k_XfIe6_
63. "Game Knowledge Management System: Schema-Governed LLM Pipeline for Executable Narrative Generation in RPGs." Systems 14(2):175 (2026). https://doi.org/10.3390/systems14020175
64. Dormans, J. "Adventures in Level Design: Generating Missions and Spaces for Action Adventure Games." PCG Workshop, FDG 2010. https://doi.org/10.1145/1814256.1814257
65. Hertz, A. et al. "Style Aligned Image Generation via Shared Attention." CVPR 2024. arXiv:2312.02133. https://arxiv.org/abs/2312.02133
66. "AlignedGen: Aligning Style Across Generated Images." arXiv:2509.17088 (2025). https://arxiv.org/abs/2509.17088
67. "Only-Style: Stylistic Consistency in Image Generation without Content Leakage." arXiv:2506.09916 (2025). https://arxiv.org/abs/2506.09916
68. Lee, K. et al. "Outatime: Using Speculation to Enable Low-Latency Continuous Interaction for Mobile Cloud Gaming." MobiSys 2015. https://www.microsoft.com/en-us/research/publication/outatime-using-speculation-to-enable-low-latency-continuous-interaction-for-cloud-gaming/
69. Song, L., Kotz, D., Jain, R., He, X. "Evaluating Next-Cell Predictors with Extensive Wi-Fi Mobility Data." IEEE TMC 5(12), 2006. https://www.cs.dartmouth.edu/dfk/research/song-jpredict/
70. "NLPMM: a Next Location Predictor with Markov Modeling." arXiv:2003.07037. https://arxiv.org/abs/2003.07037
71. Hou, I-H. et al. "Predictive Scheduling for Virtual Reality." INFOCOM 2020. arXiv:1912.12672. https://arxiv.org/abs/1912.12672
72. "Open-world streaming" engineering notes (look-ahead invariant). https://www.skills.sh/firzus/agent-skills/open-world-streaming
73. Cinevva, "Open world in the browser, part 4: streaming before fancy" (Feb 2026). https://app.cinevva.com/blog/2026-02-25-open-world-browser-part-04-streaming-before-fancy
74. OpenMW terrain/cell cache: prediction time and cache retention settings. https://gitlab.com/OpenMW/openmw/-/merge_requests/3388
75. Zilberstein, S. "Using Anytime Algorithms in Intelligent Systems." AI Magazine 17(3), 1996.
76. Luo, S. et al. "Latent Consistency Models." arXiv:2310.04378 (unverified id).
77. Sauer, A. et al. "Adversarial Diffusion Distillation" (SDXL-Turbo). arXiv:2311.17042 (unverified id).
78. Megiddo, N., Modha, D. "ARC: A Self-Tuning, Low Overhead Replacement Cache." FAST 2003.
79. PyTorch 2.14 CUDA semantics: memory management, PYTORCH_ALLOC_CONF, CUDAPluggableAllocator, MemPool, CUDA graphs. https://docs.pytorch.org/docs/2.14/notes/cuda.html
80. PyTorch devlog, "When does fragmentation occur in the CUDA caching allocator?" (2026-06-01). https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/
81. PyTorch commit "expandable_segments <-> other allocator options" (#134338) and `_set_allocator_settings`. https://android.googlesource.com/platform/external/pytorch/+/d91b49dbaa%5E%21
82. PyTorch, "Understanding CUDA Memory Usage" (memory snapshot / memory_viz). https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html
83. PyTorch, `torch.cuda.memory_stats` reference (keys listed from the stable docs). https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
84. PyTorch blog, "Understanding GPU Memory 1: Visualizing All Allocations over Time" (2023). https://pytorch.org/blog/understanding-gpu-memory-1/
85. PyTorch blog, "torch.compile and Diffusers: A Hands-On Guide to Peak Performance" (Jul 2025). https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
86. Hugging Face diffusers, "Accelerate inference" (torch.compile, regional compilation, CUDA graphs). https://huggingface.co/docs/diffusers/optimization/fp16
87. Chong, M. J., Forsyth, D. "Effectively Unbiased FID and Inception Score and where to find them." CVPR 2020. arXiv:1911.07023. https://arxiv.org/abs/1911.07023
88. Bińkowski, M. et al. "Demystifying MMD GANs" (KID). ICLR 2018. arXiv:1801.01401 (unverified id).
89. Parmar, G., Zhang, R., Zhu, J.-Y. "On Aliased Resizing and Surprising Subtleties in GAN Evaluation" (clean-fid). CVPR 2022. arXiv:2104.11222 (unverified id).
90. Stein, G. et al. "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models." NeurIPS 2023. arXiv:2306.04675 (unverified id).
91. Jayasumana, S. et al. "Rethinking FID: Towards a Better Evaluation Metric for Image Generation" (CMMD). CVPR 2024. arXiv:2401.09603. https://arxiv.org/abs/2401.09603
92. Rodriguez-Pardo, C. et al. "TexTile: A Differentiable Metric for Texture Tileability." CVPR 2024. arXiv:2403.12961. https://arxiv.org/abs/2403.12961
93. "TiPGAN: High-quality tileable textures synthesis with intrinsic priors" (RTV seam metric). Computer-Aided Design, 2025. https://www.sciencedirect.com/science/article/pii/S0010448525000284
94. Reddi, V. J. et al. "MLPerf Inference Benchmark." ISCA 2020. arXiv:1911.02549 (unverified id).
95. Unity Manual, Frame Timing Manager (Unity 6.x). https://docs.unity3d.com/Manual/frame-timing-manager.html
96. Unity Performance Testing package (com.unity.test-framework.performance 3.0.x). https://docs.unity3d.com/Packages/com.unity.test-framework.performance@latest ; https://docs.unity3d.com/Manual/test-framework/course/LostCrypt/performance-tests.html
97. Unity, "Unity 6 releases / LTS support" (6.3 LTS supported until Dec 2027). https://unity.com/releases/unity-6
98. Unity Manual, "What's new in Unity 6.4". https://docs.unity3d.com/6000.4/Documentation/Manual/WhatsNewUnity64.html
99. Unity Manual (current label "Unity 6.6 (6000.6)"), ML-Agents package page. https://docs.unity3d.com/Manual/com.unity.ml-agents.html
100. Unity Inference Engine (Sentis) 2.6 changelog (2.6.1, Apr 2026; ONNX opset ≤ 25). https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/changelog/CHANGELOG.html
101. Unity ML-Agents custom side channels (gRPC communicator). https://docs.unity3d.com/Packages/com.unity.ml-agents@latest
102. Unity Discussions, "ML-Agents 4.0.0 is now available" (Sep 2025). https://discussions.unity.com/t/ml-agents-4-0-0-is-now-available/1681770
103. Unity Discussions, "Taking over 8 minutes and counting to import a 1.6GB onnx model" (SD-2.1 UNet, 2023). https://discussions.unity.com/t/taking-over-8-minutes-and-counting-to-import-a-1-6gb-onnx-model/264442
104. Unity blog, "Hugging Face AI models and more Sentis updates." https://unity.com/blog/games/hugging-face-ai-models-and-more-sentis-updates
105. endel/NativeWebSocket (System.Net.WebSockets-based, WebGL support). https://github.com/endel/NativeWebSocket
106. Unity ScriptReference, `UnityWebRequestTexture.GetTexture` (6000.3). https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Networking.UnityWebRequestTexture.GetTexture.html
107. Unity ScriptReference, `ImageConversion.LoadImage` (6000.3). https://docs.unity3d.com/6000.3/Documentation/ScriptReference/ImageConversion.LoadImage.html
108. Lavik, M. "Async texture import in Unity engine." https://matiaslavik.codeberg.page/unity-async-texture-import/
109. Cysharp/YetAnotherHttpHandler (HTTP/2 + grpc-dotnet for Unity). https://github.com/Cysharp/YetAnotherHttpHandler ; grpc-dotnet Unity status https://github.com/grpc/grpc-dotnet
110. NetMQ in Unity issues (AsyncIO.ForceDotNet.Force, IL2CPP). https://github.com/zeromq/netmq/issues/631
111. Unity ScriptReference, `Texture2D.LoadRawTextureData` / `Texture2D.Apply`. https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Texture2D.LoadRawTextureData.html
112. Unity KTX/Basis Universal package (com.unity.cloud.ktx 3.x). https://docs.unity3d.com/Packages/com.unity.cloud.ktx@3.0/api/KtxUnity.html
113. Unity Discussions, "KTX 2.0 and basis file runtime import with KtxUnity." https://discussions.unity.com/t/ktx-2-0-and-basis-file-runtime-import-with-ktxunity/816735
114. Unity Addressables 2.x, "Asynchronous loading" / "Load Addressable assets." https://docs.unity3d.com/Packages/com.unity.addressables@2.7/manual/load-addressable-assets.html
