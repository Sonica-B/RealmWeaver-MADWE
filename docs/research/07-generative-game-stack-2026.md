# 07 - Generative game stack for an open-world 3D action RPG (state of the art as of 2026-10-06)

Scope: what a **1-3 person team** can use in 2026 to grow RealmWeaver (SD1.5 textures, numpy WFC, world state graph,
FastAPI/WebSocket bridge, Unity UPM client) into an **open-world 3D action RPG in the vein of Crimson Desert**, authored
on an **RTX 5070 Ti Laptop GPU (12 GB GDDR7, 192-bit, 677 GB/s, 5,888 CUDA cores, 140 W TGP)** on Windows 11, with
optional 24 GB-class cloud GPUs for the heavy models. Seven topics, each with a comparison table, a recommendation block
and citations, then a single STACK RECOMMENDATION.

How to read the numbers (same convention as `01-model-landscape-2026.md`):

- `[M]` measured by a cited third party on the named GPU; `[C]` vendor / model-card / README claim; `[E]` my estimate
  derived from a cited measurement (arithmetic shown). Every `[E]` must be re-measured on the 5070 Ti before a design
  decision is frozen (validation plan in section 9).
- Hugging Face (HF) ids, download counts, licence tags and update dates were read from the HF Hub API on 2026-10-06.
- Licence summaries are my reading of the linked texts, not legal advice; the two Tencent licences and the AMASS
  licence are the ones most likely to bite and are quoted verbatim where it matters.

---

## 0. TL;DR

1. **Engine: Unreal Engine 5.8** (released 2026-06-17, the last planned UE5; UE6 early access "by end of 2027").
   It is the only engine whose runtime absorbs generative output without hand work: **Nanite** renders the 500 K-1 M
   triangle meshes that TRELLIS.2 / Hunyuan3D emit with no retopology or LOD authoring, **World Partition + HLOD**
   streams the open world, **PCG** (production-ready since 5.7) scatters vegetation/rocks from Python-authored masks,
   **Motion Matching** (production-ready since 5.4) plus the free 500-clip Game Animation Sample turns retargeted clips
   into a locomotion system, **StateTree** is the default AI framework in 5.8, and the open-source **Audio2Face-3D UE
   plugin** does lip-sync. Cost: a 5 % royalty only above US $1 M gross per product, an Epic account, and a heavier
   laptop workflow. **Unity 6.3 LTS** is the fallback if a two-week spike shows UE iteration times are unworkable on the
   laptop; **Godot 4.7** is the no-account prototyping engine, not the shipping target for this scope.
2. **3D assets: `microsoft/TRELLIS.2-4B` (MIT)** through ComfyUI's native nodes (merged 2026-08-22, Windows): 512³ in
   ~45-60 s on 8 GB `[M]`, 1024³ needs 16 GB (cloud or 4-bit), full PBR GLB. Fallbacks: `tencent/Hunyuan3D-2.1` shape
   (10 GB, Windows) with the 2.1 PBR painter on a 24 GB cloud GPU (21 GB), `VAST-AI/TripoSG` (MIT, 8 GB) for clean
   watertight shapes, `stabilityai/stable-point-aware-3d` for sub-second props (Stability Community Licence, free under
   US $1 M revenue). Tencent licences exclude the **EU, UK and South Korea** and cap at **1 M MAU**; avoid them for
   anything you cannot swap later.
3. **Terrain: Terrain Diffusion (SIGGRAPH 2026, MIT, `xandergos/terrain-diffusion-30m`)** for seed-consistent,
   randomly-accessible 30 m/px heightmaps + climate layers, then erosion in Gaea 2 Indie (US $99) or an own GPU pass,
   biome/road/settlement logic in the existing Python world graph (WFC / MarkovJunior), import as UE Landscape tiles,
   PCG for scatter. World models (Genie 3, Marble, HunyuanWorld) are reference/skybox tools, not asset authoring.
4. **Animation: Mixamo + Game Animation Sample clips as the base library**, retargeted to the UE5 Mannequin; bespoke
   clips from **`tencent/HY-Motion-1.0`** on a 24 GB cloud GPU (24-26 GB `[C]`, Tencent community licence - verify the
   territory clause before shipping). **Do not ship clips from HumanML3D/AMASS-trained models (MoMask, T2M-GPT,
   MotionGPT3): AMASS is non-commercial.** Rig generated characters with **`VAST-AI/UniRig`** (MIT, 8 GB, seconds).
   Faces: Audio2Face-3D (SDK MIT, weights NVIDIA Open Model Licence).
5. **NPC brain: `google/gemma-4-E4B-it` (Apache-2.0, Apr 2026, ~3 GB at Q4)** or `Qwen/Qwen3.5-4B` (Apache-2.0,
   Feb 2026) served by **llama.cpp `llama-server --jinja`** with JSON-schema grammars; `[E]` ~1 s per 60-token turn on
   the 5070 Ti laptop (from a measured 65.6 tok/s for an 8B Q4). Voice: **Kokoro-82M** (Apache-2.0, ~75 ms on a
   4060 Ti `[M]`) or **Chatterbox-Turbo 350M** (MIT, ~75-200 ms, now NVIDIA ACE's TTS); ears: **whisper.cpp small.en**
   (MIT, 0.13 s for 9.4 s of audio on CUDA `[M]`). vLLM is not a Windows option (WSL2 only).
6. **Bridge: keep HTTP + WebSocket (ADR-0003)**; UE has a built-in WebSockets module; gRPC (TurboLink) only if a
   measured need appears. Engine tick budget 16.7 ms: bridge reader on its own thread, <= 1 ms/frame of main-thread
   apply, agent decisions at 1-5 Hz, LLM turns fully asynchronous.
7. **VRAM split during play (12 GB): game <= 6.0 GB, LLM 4B Q4 + 8 K KV ~3.5 GB, TTS + STT ~1.0 GB, headroom 1.5 GB.**
   Nothing diffusion-sized is resident during play; all meshes, terrain, rigs, clips and voice banks are pre-baked.
8. **Reality check:** Valheim = 5 people, ~3.5 years to Early Access and 9 years to 1.0; Sons of the Forest = 10-15
   people, ~4 years; Enshrouded = ~58-75 people, own engine, ~2.75 years in EA; Dragon's Dogma 2 = 392 core credits,
   ~4 years; Crimson Desert = ~200 (peak < 300) people, ~7 years. Generative tooling compresses asset and animation
   production (the 19 % "asset generation" use in the GDC 2026 survey), not design, combat feel, QA or optimisation.

---

## 1. Generative 3D assets (image/text -> mesh + PBR)

### 1.1 Open-weight models

| Model | HF id | Params | Licence (commercial?) | VRAM | Time | Output / game-readiness | Windows | Verdict |
|---|---|---|---|---|---|---|---|---|
| **TRELLIS.2** (Microsoft, 2025-12) | `microsoft/TRELLIS.2-4B` (8.8 M dl, 1,291 likes; `Comfy-Org/TRELLIS.2` repack 300 K dl/month) | 4 B | **MIT** (yes; deps have own licences) | official: >= 24 GB, Linux-only tested `[C]`; ComfyUI native: 512³ ~8 GB (RTX 3060, 45-60 s), 1024³ ~16 GB (RTX 4070, 90-120 s), 1536³ ~24 GB `[M]`; GGUF Q4 peaks 6-6.5 GB on 6 GB cards `[M]`; int8 ConvRot checkpoint 5.25 GB | H100: 512³ 3 s, 1024³ 17 s, 1536³ 60 s `[C]` | GLB with **base color, roughness, metallic, opacity**; `texture_size` 4096, `decimation_target` 1,000,000 (default); O-Voxel handles open surfaces / non-manifold / interiors (good for foliage, cloth; collision needs a separate hull); triangles, auto-UV, no LODs | **Yes via ComfyUI core (PR merged 2026-08-22)**; one-click Windows installers exist; official repo needs FlexGEMM/O-Voxel/CuMesh/nvdiffrast/nvdiffrec CUDA kernels (CUDA 12.4) | **Primary** |
| TRELLIS v1 (2024-12) | `microsoft/TRELLIS-image-large` (1.2 B; 40.4 M dl), `-text-xlarge` (2.0 B) | 1.2 B | MIT (except diffoctreerast / Flexicubes derivatives) | >= 16 GB `[C]` | ~1-3 min class `[M]` (third-party) | radiance field / 3DGS / mesh; GLB textures 1024 default; no PBR split | "not fully tested" (issue #3), community forks | superseded by TRELLIS.2 |
| **Hunyuan3D 2.1** (Tencent, 2025-06-13) | `tencent/Hunyuan3D-2.1` (684 K dl; `Comfy-Org/hunyuan3D_2.1_repackaged`) | Shape 3.3 B + Paint 2 B | **Tencent Hunyuan 3D 2.1 Community Licence**: commercial allowed; **> 1 M MAU must request a licence**; **Territory excludes the EU, UK and South Korea**; outputs may not train other models | **10 GB shape, 21 GB paint, 29 GB both** `[C]`; `low_vram_mode` flag | minutes class `[M]` (third-party: 3-8 min for 2.0 at 24 GB) | mesh + **PBR (base color, normal, roughness, metallic, AO)**; default **500 K faces**; training code released | **"supports MacOS, Windows, Linux"** `[C]`; ComfyUI nodes (mesh on 6 GB+, paint 20 GB+) | Shape locally, paint on cloud; licence-gated |
| Hunyuan3D 2.0 (2025-01) | `tencent/Hunyuan3D-2` (3.6 M dl), `-2mini` (0.6 B), `-2mv` (multi-view), Turbo/Fast distillations | 1.1 B shape + 1.3 B paint | Tencent Hunyuan 3D 2.0 Community Licence (same 1 M MAU + EU/UK/SK exclusion) | **6 GB shape, 16 GB shape + texture** `[C]` | Turbo variants "faster" (no numbers) | GLB/OBJ via trimesh; diffuse texture (PBR via MaterialMVP) ; Blender add-on + local API server | Windows bundle `sdbds/Hunyuan3D-2-for-windows` | low-VRAM fallback (mini-Turbo) |
| Hunyuan3D 2.5 / 3.0 / 3.1 | none | - | hosted only (no HF repos as of 2026-10-06; HF search for "Hunyuan3D" returns no 3.x) | - | - | - | - | API only |
| **TripoSG** (VAST, 2025-03) | `VAST-AI/TripoSG` (1.44 B; 40.7 K dl), `TripoSG-scribble` (512 tokens) | 1.5 B rectified flow | **MIT** | **>= 8 GB** `[C]` | not stated | **watertight mesh only** (no texture); `--faces` cap; diffusers `TripoSGPipeline` | no notes (pure PyTorch, should run) | clean shapes; texture elsewhere |
| TripoSF (VAST, 2025-04) | `VAST-AI/TripoSF` | VAE | MIT | - | - | high-res shape VAE (research) | - | watch |
| **Step1X-3D** (StepFun, 2025-05) | `stepfun-ai/Step1X-3D` | geometry 1.3 B + texture 3.5 B (SDXL-based) | **Apache-2.0** | **27-29 GB** `[C]` | **152 s** full pipeline, 50 steps `[C]` | watertight TSDF mesh + texture GLB | no notes; ComfyUI "on roadmap" | cloud-only; best licence |
| Hi3DGen / Stable3DGen (2025-03) | `Stable-X/trellis-normal-v0-1` (138 K dl) | TRELLIS-size | **MIT** (NVIDIA deps removed "to enable commercial use") | TRELLIS-class | - | geometry only (normal-bridged), WIP | - | niche (hard-surface detail) |
| **SPAR3D** (Stability, 2025-01) | `stabilityai/stable-point-aware-3d` (gated; 2.0 B; 276 K dl) | 2 B | **Stability AI Community Licence**: free below **US $1 M** annual revenue | **~10.5 GB default, ~7 GB with `SPAR3D_LOW_VRAM=1`** `[C]` | **< 1 s** `[C]` | textured UV-unwrapped GLB, per-object roughness/metallic, remesh none/tri/quad, vertex-count target | **experimental** (VS 2022 + CUDA extensions) | sub-second props / placeholders |
| Stable Fast 3D (2024-08) | `stabilityai/stable-fast-3d` (gated; 1.0 B; 282 K dl) | 1 B | Stability Community | ~6-7 GB `[E]` | < 1 s `[C]` | same family, lower quality | experimental | superseded by SPAR3D |
| Direct3D-S2 (DreamTech, 2025-05) | `wushuang98/Direct3D-S2` | - | MIT | 512: >= 10 GB; 1024: ~24 GB `[C]` | - | shape only (sparse SDF) | ComfyUI node | shape alternative |
| SAM 3D Objects (Meta, 2025-11-19) | `facebook/sam-3d-objects` (gated; 46 K dl) | - | **SAM Licence**: commercial allowed, bans military/weapons uses, trade-control clauses, attribution in publications | not stated | - | mesh + Gaussian splat (+ pose/layout) from **photos with masks** | not stated | photo-to-prop, not concept-to-asset |
| Sparc3D (NeurIPS 2025) | code only | - | code open; **no checkpoints** (commercial via Hitem3D) | ~8 GB for 1024³ on A100 `[C]` | - | high-res shapes | - | skip |
| Unique3D (2024-05) | `Wuvin/Unique3D` | - | MIT | - | - | multi-view + mesh | - | superseded |
| CraftsMan (2024-05) | `wyysf/CraftsMan` | - | "other" | - | - | mesh + refinement | - | superseded |
| Hunyuan3D-Part / -Omni (2025-09) | `tencent/Hunyuan3D-Part`, `-Omni` | - | Tencent (no licence tag on -Part) | - | - | part segmentation / generation; skeleton & bbox controls | - | useful for rigging/LOD splits later |

Sources: HF Hub API records (ids above); [TRELLIS.2 repo](https://github.com/microsoft/TRELLIS.2);
[ComfyUI native TRELLIS.2/Pixal3D (2026-08-22)](https://comfyui-wiki.com/en/news/2026-08-22-trellis2-pixal3d-native-comfyui),
[instasd VRAM table](https://www.instasd.com/post/trellis-2-pixal3d-native-comfyui-guide), [ComfyUI docs](https://docs.comfy.org/tutorials/3d/pixal3d),
[Comfy-Org/TRELLIS.2](https://huggingface.co/Comfy-Org/TRELLIS.2), [6 GB GGUF guide](https://dailytopai.com/article/how-to-run-trellis-2-3d-ai-locally-on-just-6gb-vram-with-gguf-44.html),
[low-VRAM guide](https://trellis2.app/blog/trellis-2-low-vram); [TRELLIS v1 repo](https://github.com/microsoft/TRELLIS);
[Hunyuan3D-2.1 repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), [2.1 LICENSE](https://huggingface.co/tencent/Hunyuan3D-2.1/raw/main/LICENSE),
[Hunyuan3D-2 repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [2.0 LICENSE](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2/main/LICENSE),
[version comparison](https://triposr.org/blog/hunyuan3d-versions); [TripoSG](https://github.com/VAST-AI-Research/TripoSG); [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D);
[Stable3DGen](https://github.com/Stable-X/Hi3DGen); [SPAR3D card](https://huggingface.co/stabilityai/stable-point-aware-3d), [SPAR3D repo](https://github.com/Stability-AI/stable-point-aware-3d);
[Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2); [SAM 3D](https://ai.meta.com/blog/sam-3d/), [SAM licence](https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/LICENSE);
[Sparc3D controversy](https://www.vset3d.com/the-sparc3d-controversy-from-open-source-promise-to-paid-hitem3d-platform/).

### 1.2 Commercial APIs

| Service | Pricing (2026) | Game-readiness features | Output licence | Notes |
|---|---|---|---|---|
| **Tripo** (VAST) | 1 credit = US $0.01; text-to-3D 10-20, image-to-3D 20-30, multiview 20-30, texture 10-30, **retopology 10-30, auto-rig 25, animation retarget 10/clip**, segmentation 40-50; 2,000 free API credits on signup | Smart Mesh P1.0 (2026-03): clean low-poly topology in ~2 s; universal rigging + animation library | free tier outputs **public, CC BY 4.0** (ToS also reserves broad rights - read it); paid tiers private with full commercial rights | same lab as TripoSG/UniRig |
| **Meshy** | Free 100 credits/mo; Pro US $20/mo (1,000); Studio US $60/mo (4,000); Meshy 6 generation 20 credits untextured / 30 textured | topology (quad/tri), target polycount 1 K-300 K, symmetry, remesh; 4 K base color (8 K option); auto-rig + 600+ animations | free tier **CC BY 4.0** (attribution); paid plans private commercial | widest tooling |
| **Rodin / Hyper3D** (Gen-2 / 2.5) | Creator ~US $30/mo (30 credits), Business ~US $120/mo (208); ~US $0.40/gen via fal.ai; base 0.5 credit + up to 2.5 for extreme texture tiers | **quad topology**, PBR | commercial rights on all tiers | strongest topology |

Sources: [Tripo API pricing](https://developers.tripo3d.ai/en/pricing), [Tripo rigging](https://developers.tripo3d.ai/en/models/rig), [Tripo free-tier licence note](https://www.aifreeapi.com/en/posts/tripo-3d);
[Meshy pricing](https://www.meshy.ai/pricing), [Meshy commercial-use help](https://help.meshy.ai/en/articles/16102098-can-i-use-meshy-assets-commercially), [Meshy 6 API changelog](https://docs.meshy.ai/en/api/changelog);
[Rodin pricing review](https://makerstack.co/reviews/hyper3d-rodin-review/), [Rodin on fal](https://fal.ai/models/fal-ai/hyper3d/rodin), [Hyper3D API](https://hyper3d.ai/features/api).

### 1.3 Game-readiness: what the raw outputs are and what has to happen after

- **Topology.** Every open model emits triangle soups from a voxel/SDF/TSDF extraction: TRELLIS.2 decimates to a
  target (default 1 M), Hunyuan3D defaults to 500 K faces, SPAR3D/TripoSG expose a face/vertex target. None emits
  animation-grade quads; Rodin and Meshy (paid) do. **With Nanite (UE) you ship the decimated triangle mesh directly**
  and only need a collision hull (UE auto-convex / simplified mesh); without Nanite (Unity/Godot) you need a headless
  Blender pass (decimate -> Quadriflow for deformables -> `xatlas` UV -> bake) plus 3-4 LODs via `meshoptimizer`/
  `gltfpack`, which is where a texture-bake queue in the Python sidecar earns its keep.
- **Watertightness.** Hunyuan3D (SDF), TripoSG, Step1X-3D (TSDF) and SPAR3D are watertight; TRELLIS.2's O-Voxel is
  explicitly *not* restricted to watertight (open surfaces, non-manifold, interiors), which is better for foliage,
  cloth and hollow props but means collision must be generated separately.
- **Textures.** TRELLIS.2 and Hunyuan3D 2.1 give PBR splits; TripoSG/Direct3D-S2/Hi3DGen give geometry only (texture
  with Hunyuan Paint, MV-Adapter, or the existing SD1.5 pipeline projected in Blender); SPAR3D gives a baked
  albedo + scalar roughness/metallic. Expect 1-4 K atlases with visible seams on the back of single-image inputs;
  multi-view inputs (Hunyuan3D-2mv, Tripo multiview) fix the backside.
- **Style consistency** across hundreds of props is the unsolved part: condition every generation on the same
  reference sheet (FLUX.2 klein / SD1.5 biome LoRA concept art -> image-to-3D), run the existing DINOv2 coherence
  score on rendered turntables, and reject below threshold (section 5 of `02-pcg-and-systems-literature.md`).
- **Throughput budget on the laptop.** At 512³ TRELLIS.2 is ~1 min/asset on 8 GB-class hardware `[M]`; `[E]` 40-50 s
  on the 5070 Ti (677 GB/s vs the 3060's 360 GB/s, not linear because of CPU-side O-Voxel conversion). 1,000 props is a
  day of unattended generation; 1024³ hero assets go to a 24 GB cloud GPU (~1.5-2 min each `[M]`).

### 1.4 Recommendation (3D assets)

**Primary: `microsoft/TRELLIS.2-4B` (MIT) via ComfyUI-core nodes on Windows**, 512³ locally for props (8 GB class,
~1 min), 1024³ hero assets and characters on a rented 24 GB GPU; export GLB with PBR, keep the 1 M-triangle decimation
for Nanite, generate collision in-engine. **Shape fallback: `VAST-AI/TripoSG` (MIT, 8 GB)** when TRELLIS.2 topology
is too noisy for a deformable; **sub-second placeholders: SPAR3D** (Stability Community Licence; fine while revenue
< US $1 M, re-license or replace before then). **Hunyuan3D 2.1** is the quality fallback for PBR painting of other
models' geometry, but its licence excludes EU/UK/SK players and caps at 1 M MAU: keep every Hunyuan-derived asset
tagged in the world graph so it can be regenerated with an MIT model if the game ever ships there. Skip Step1X-3D
locally (27-29 GB) despite the best licence; it is a good cloud batch option for watertight hero geometry. Use a
commercial API (Tripo for rig + retopo at ~US $0.55 per rigged character; Rodin for quads) only for the handful of
characters and weapons that need animation-grade topology.

---

## 2. Terrain and world generation

### 2.1 Heightmaps, erosion, biomes

| Approach | Tool / model | Licence / price | What it gives | Runs on the laptop? | Verdict |
|---|---|---|---|---|---|
| **Diffusion terrain, infinite + seed-consistent** | **Terrain Diffusion / InfiniteDiffusion** (SIGGRAPH 2026; `xandergos/terrain-diffusion-30m`, `-90m`) | **MIT** | heightmap + climate layers (temperature, precipitation mean/variability); 30 m/px model "for playable worlds" (7.7 km coarse model), 90 m/px for continents; **O(1) random access, lazy unbounded sampling, deterministic per seed** ("learned successor to Perlin noise") | yes (SD-class UNets) - VRAM not published, `[E]` < 6 GB | **Primary heightmap source**: the same "generate where the player is heading" idea as the existing predictor |
| Text-driven terrain | MESA (CVPRW 2025; SD2.1 LDM on Copernicus DEM 30 m / Major TOM) | open weights on HF | 2.5D terrain from text | yes | secondary; climate-conditioned |
| Joint heightmap + texture | TerraFusion (2025), Geodiffussr (2025-11) | research | heightmap + albedo together | yes | watch |
| Noise + erosion (classic) | FBM/simplex noise -> hydraulic/thermal erosion (Mei 2007 GPU; Olsen 2004), hydrology-first terrain (Genevaux 2013) | - | fully controllable, no licence | yes (CPU/GPU, Python/numba or compute shader) | keep as the deterministic fallback and for local edits |
| Erosion DCCs | **Gaea 2** (Indie US $99, 8 K cap; Community free non-commercial 1 K; Gaea 3 due 2026, "world space" infinite nodes in 2.4); World Machine (Indie US $119, Pro US $299 tiled export; free 1,025² non-commercial); World Creator 2026.4 (Indie US $99 < US $100 K revenue; Community edition export-disabled) | commercial | best-in-class erosion, masks, PBR splat maps, exports to Unity/UE/Blender | Windows native | Gaea 2 Indie is the value pick |
| Engine terrain | **UE Landscape + World Partition + PCG** (PCG production-ready 5.7 with Editor Mode; Biome Core v2 in 5.6; Nanite Foliage 5.7; **Mesh Terrain experimental in 5.8** for caves/overhangs); Unity Terrain + Terrain Tools package (erosion, noise, brush masks, Toolbox heightmap import; 6.4 adds terrain material drag-drop); Godot **Terrain3D 1.0** (GDExtension, Apr 2025; dynamic collision, 10 foliage LODs, 3D projection on cliffs) | engine licence | heightfield terrain + scatter | yes | all three ingest 16-bit heightmaps and splat masks from Python |

### 2.2 Roads, rivers, settlements, dungeons, scatter

| Need | Technique | Implementation path | Notes |
|---|---|---|---|
| Rivers | flow accumulation on the heightmap (D8) + carve; or hydrology-first (Genevaux 2013) | numpy / `pysheds`-style flow accumulation in the sidecar | Terrain Diffusion's precipitation layer seeds sources |
| Roads | Parish & Muller 2001 L-system / A* on slope-cost between settlements; splines | Python -> engine splines (UE Landscape Splines deform terrain; Unity splines package; Godot Path3D) | keep road graph in the world state graph |
| Biomes | Whittaker diagram over (temperature, precipitation, elevation) | direct from Terrain Diffusion climate layers | matches existing biome YAMLs |
| Settlements / dungeons | **WFC / Model Synthesis** (existing numpy WFC), **MarkovJunior** rewrite grammars, graph grammars (Dormans mission/space; cyclic dungeon generation as in Unexplored) | existing layout agent; see `02-pcg-and-systems-literature.md` sections 1.4-1.5 | generated layouts instantiate kits of TRELLIS.2 modules |
| Vegetation / rocks | Poisson-disk (Bridson 2007) + ecosystem competition (Deussen 1998; Lane & Prusinkiewicz 2002) on biome/slope/moisture masks | masks from Python -> **UE PCG graphs** (GPU, production-ready) / Unity Terrain detail + GPU Resident Drawer / Terrain3D instancer | author rules once in PCG, feed masks |
| Whole-scene generators | **Infinigen** (Princeton, BSD-3; Blender-based terrain/plants/creatures, Infinigen Indoors): Windows experimental "minimal" mode, WSL recommended; exports OBJ/FBX/USD | reference-quality nature kits, offline in Blender | heavy but free kit source |
| World models | **Genie 3 / Project Genie** (Google Labs since 2026-01-29, AI Ultra US subscribers, **no public API**); **Marble** (World Labs; 3DGS `.spz`/`.ply`, mesh + collider export, Unity/UE/Blender guides, paid); **HunyuanWorld 1.0** (`tencent/HunyuanWorld-1`, Tencent community licence; layered mesh export, Unity/UE); **WorldGen** (`LeoXie/WorldGen`, Apache-2.0; text/image -> navigable Gaussian-splat scene, low-VRAM mode); Meta's WorldGen (CVPR 2026; traversable mesh worlds, paper) ; **worldgen-forge** (MIT, v0.2, 0 stars; terrain + layout + TRELLIS.2 assets -> UE 5.8/Unity/Godot/glTF, Windows-first) | - | none authors game-grade static content: splat scenes are not editable meshes, HunyuanWorld meshes are panoramic shells. Use them for vista skyboxes, concept blockouts and reference. worldgen-forge is the closest to RealmWeaver's architecture and worth reading, not depending on |

Sources: [terrain-diffusion repo](https://github.com/xandergos/terrain-diffusion), [project page](https://xandergos.github.io/terrain-diffusion/); [MESA](https://openaccess.thecvf.com/content/CVPR2025W/MORSE/papers/Borne--Pons_MESA_Text-Driven_Terrain_Generation_Using_Latent_Diffusion_and_Global_Copernicus_CVPRW_2025_paper.pdf);
[TerraFusion](https://arxiv.org/abs/2505.04050); [Gaea 2.2](https://www.cgchannel.com/2025/07/quadspinner-releases-gaea-2-2/), [Gaea 3](https://www.cgchannel.com/2025/12/quadspinner-unveils-gaea-3-0/); [World Machine purchase](https://www.world-machine.com/purchase.php);
[World Creator 2026.4](https://www.cgchannel.com/2026/04/world-creator-2026-4-is-out-with-a-new-free-community-edition/); [UE 5.7 release](https://www.unrealengine.com/news/unreal-engine-5-7-is-now-available), [UE 5.8 Mesh Terrain](https://80.lv/articles/unreal-engine-5-8-is-out-today-with-big-optimization-improvements-and-mesh-terrain);
[Unity Terrain Tools](https://docs.unity3d.com/6000.2/Documentation/Manual/TerrainTools.html); [Terrain3D 1.0](https://gamefromscratch.com/terrain3d-for-godot-hits-1-0-release/), [Terrain3D releases](https://github.com/TokisanGames/Terrain3D/releases);
[Infinigen](https://github.com/princeton-vl/infinigen), [Infinigen Indoors](https://www.cgchannel.com/2025/06/open-source-tool-infinigen-indoors-generates-procedural-3d-interiors/); [Project Genie](https://en.wikipedia.org/wiki/Project_Genie_(website)), [Genie access](https://fenxi.fr/en/blog/genie-3-google-deepmind-world-model-2/);
[Marble export](https://help.scenario.com/articles/4774055514-marble-by-world-labs-the-essentials), [World Labs blog](https://www.worldlabs.ai/blog/bigger-better-worlds); [HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0);
[WorldGen (LeoXie)](https://huggingface.co/LeoXie/WorldGen), [WorldGen CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_WorldGen_From_Text_to_Traversable_and_Interactive_3D_Worlds_CVPR_2026_paper.pdf); [worldgen-forge](https://github.com/Ardynai/worldgen-forge).

### 2.3 Recommendation (terrain)

Replace the 2D tile world with a **two-layer terrain pipeline owned by the Python sidecar**: (1) **Terrain Diffusion
30 m** generates heightmap + climate per 7.7 km region, seed-consistent and randomly accessible, so the existing
predictor/prewarm logic transfers unchanged; (2) a deterministic post-pass (flow-accumulation rivers, slope-cost roads
between WFC-placed settlements, Whittaker biomes) writes the world state graph. Erode hero regions in **Gaea 2 Indie**
and bake splat masks. Export 16-bit PNG tiles + masks to **UE Landscape (World Partition)**, scatter with **PCG**
graphs driven by the masks, and instantiate settlement kits from TRELLIS.2 modules. Treat Genie 3 / Marble /
HunyuanWorld as concept and skybox tools only. Keep Infinigen as an offline kit generator for nature assets if the
3D-model route produces inconsistent foliage.

---

## 3. Characters and animation

### 3.1 Comparison

| Stage | Option | Licence / cost | Hardware | Practicality for 1-3 people |
|---|---|---|---|---|
| Base humans | **MakeHuman / MPFB2** (Blender 4.2+ add-on) | AGPL app, **CC0** output and bundled assets | CPU | free, parametric, rigged, exportable |
| | **MetaHuman Creator** (in UE 5.6+, parametric body; 5.7 Python/Blueprint batch) | UE EULA; since 2025-06 usable in **any engine** as a "non-engine product", no royalty | GPU | best fidelity; Unreal-native |
| | Image-to-3D character (TRELLIS.2 1024³) + auto-rig | MIT | 16 GB -> cloud | stylised NPC variety |
| Auto-rig | **UniRig** (`VAST-AI/UniRig`, SIGGRAPH 2025) | **MIT** | >= 8 GB; 1-5 s/mesh `[M]` | in/out OBJ/FBX/GLB/VRM, skeleton + skinning weights; full Rig-XL checkpoints "planned" |
| | RigAnything (Adobe), Make-It-Animatable (FBX export) | research | - | watch |
| | Tripo auto-rig (25 credits) / Meshy auto-rig + 600 animations | US $0.25 per rig | cloud | fastest path for hero characters |
| Clip library | **Mixamo** | free, commercial OK, no redistribution of raw assets; Adobe account | - | still the default library in 2026 |
| | **Game Animation Sample** (UE; 500+ clips, updated for 5.8) | UE EULA (engine products) | - | motion-matching database out of the box |
| Text-to-motion | **HY-Motion 1.0** (`tencent/HY-Motion-1.0`, 1.0 B; Lite 0.46 B; 2025-12-31) | HF tag `tencent-hunyuan-community` (same family as Hunyuan3D: expect EU/UK/SK exclusion and MAU cap; `License.txt` in repo - verify before shipping) | **26 GB / 24 GB Lite** `[C]`; "supports macOS, Windows, Linux"; clips < 5 s recommended | skeleton output; ComfyUI node exports FBX; trained on 3,000 h + 400 h curated; no loops, no multi-person |
| | MoMask (CVPR 2024), T2M-GPT (2023), MotionGPT3 (2025), LGTM (2024), MotionLCM | code MIT/academic, but **trained on HumanML3D = AMASS + HumanAct12** | 4-8 GB | **AMASS licence: non-commercial scientific research only** -> do not ship their clips commercially |
| | OpenT2M (2026-03; 1 M+ sequences) | "freely available to researchers" | - | research dataset; check terms |
| Retargeting | UE **IK Retargeter** (auto-retarget 5.4+), Unity Humanoid, Blender (Rokoko Studio Live plugin, Auto-Rig Pro) | free / US $40 | CPU | SMPL -> Mannequin via Blender; **SMPL-X body model itself is non-commercial (Meshcapade licenses it)** - never ship SMPL meshes, only retargeted motion |
| Facial / lip-sync | **Audio2Face-3D** (open-sourced 2025-09): SDK C++/Python **MIT**, model weights **NVIDIA Open Model Licence**, Audio2Emotion custom (A2F-only), **UE5 plugin v2.5 (5.5/5.6, MIT)**, Maya ACE 2.0, training framework Apache-2.0, NIM container | free | RTX GPU | Unity/Godot via the SDK or NIM gRPC; no official plugin |
| | uLipSync (Unity, MIT, MFCC real-time), OVRLipSync (Meta; UE/Unity), Rhubarb (MIT CLI, 2D visemes) | free | CPU | cheap fallbacks |
| Commercial animation | Meshy animate, Tripo retarget (10 credits/clip), Unity AI (Muse Animate retired; Unity AI beta, points-based) | credits | cloud | fine for one-offs |

Sources: [MPFB2](https://github.com/makehumancommunity/mpfb2), [MakeHuman licence](https://static.makehumancommunity.org/about/license.html); [MetaHuman any-engine](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/), [MetaHuman EULA](https://www.unrealengine.com/eula/mhc);
[UniRig](https://github.com/VAST-AI-Research/UniRig), [UniRig news](https://comfyui-wiki.com/en/news/2025-04-12-unirig-one-model-to-rig-them-all); [RigAnything](https://arxiv.org/abs/2502.09615);
[Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html); [GASP 5.8](https://www.unrealengine.com/tech-blog/download-the-latest-game-animation-sample-project-now-updated-for-ue-5-8), [GASP docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/game-animation-sample-project-in-unreal-engine);
[HY-Motion repo](https://github.com/Tencent-Hunyuan/HY-Motion-1.0), [HY-Motion card](https://huggingface.co/tencent/HY-Motion-1.0), [HY-Motion paper](https://arxiv.org/abs/2512.23464); [MoMask](https://arxiv.org/abs/2312.00063);
[AMASS licence](https://amass.is.tue.mpg.de/license.html), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [SMPL-X licence](https://smpl-x.is.tue.mpg.de/modellicense.html); [OpenT2M](https://arxiv.org/abs/2603.18623);
[Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D), [Audio2Face-3D SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK), [open-sourcing news](https://www.opensourceforu.com/2025/09/nvidia-moves-audio2face-technology-to-open-source/); [uLipSync](https://github.com/hecomi/uLipSync); [Rhubarb](https://github.com/DanielSWolf/rhubarb-lip-sync).

### 3.2 Recommendation (characters and animation)

**Humans:** MetaHuman Creator for the player and named NPCs (free in any engine since June 2025; Unreal-native rig,
Audio2Face-ready), **MPFB2 (CC0)** for crowd bodies when variety matters more than fidelity. **Creatures and
stylised NPCs:** TRELLIS.2 at 1024³ (cloud) -> **UniRig** (MIT, seconds) -> cleanup in Blender -> UE IK Retargeter.
**Clips:** Mixamo + Game Animation Sample as the motion-matching database; **HY-Motion 1.0** on a 24 GB cloud GPU for
bespoke actions (attack variants, gestures), retargeted off the SMPL skeleton to the Mannequin (never ship SMPL
meshes). **Hard rule: no HumanML3D/AMASS-trained generators (MoMask, T2M-GPT, MotionGPT3, LGTM) in a commercial
build** - AMASS is research-only. **Faces:** Audio2Face-3D UE plugin for lip-sync from the TTS stream; uLipSync-style
viseme fallbacks for crowd NPCs.

---

## 4. Agentic NPCs

### 4.1 Architecture and frameworks

| Layer | Option | Licence / price | 2026 status | Fit |
|---|---|---|---|---|
| Cognitive loop | **Generative Agents** (Park et al. 2023): memory stream, retrieval by recency x importance x relevance, periodic reflection, hierarchical planning | paper / MIT reference code | still the reference design; every 2025-26 product is a cut-down version | implement in the Python sidecar on top of the world state graph (memories = graph edges with timestamps) |
| Runtime SDKs | **NVIDIA ACE** (on-device models, 2026-03: Riva ASR v1.1, **Nemotron 3 Nano 4B** SLM, **Chatterbox 350 M TTS**, Audio2Face-3D; earlier Mistral NeMo Minitron 0.5 B in inZOI at **1 GB VRAM**, min RTX 3060/4060 8 GB, recommended 12 GB) | free SDK / NIM | shipped in inZOI (Smart Zoi, 2025-03), NARAKA, Mecha BREAK, Dead Meat (2026) | best on-device blueprint; UE plugins, Unity via NIM |
| | **Inworld** | TTS-2 US $25/M chars (US $12.50 on the US $1.5 K/mo Growth plan), TTS-2 Flash US $15 -> US $7 | pivoted from character engine to voice/runtime | cloud voice only |
| | **Convai** | Free / Indie US $29 / Pro US $99 / Scale US $499 / Business US $1,199 per month; Unity + UE plugins | reviews cite persistent latency | cloud, per-month |
| | Fortnite UEFN NPC tool (Gemini 3.1 Flash Lite + ElevenLabs) | UEFN only | Darth Vader NPC (2025-05) drew a SAG-AFTRA unfair-labour charge | not usable outside UEFN |
| In-engine local inference | **LLMUnity** (undream.ai, Apache-2.0, v3.0.3 2026-03, llama.cpp/LlamaLib, GGUF, RAG) ; llama-cpp-unity; **NobodyWho** (Godot GDExtension, v11, Godot 4.5+); Unity Inference Engine 2.6.1 (ONNX; small models only) | free | mature for Unity/Godot; UE has no equivalent - use the sidecar | sidecar (`llama-server`) keeps engine-agnostic |

### 4.2 Local LLMs that fit beside a game on 12 GB

| Model | HF id | Params | Licence | GGUF Q4 weights | Context / modalities | Notes |
|---|---|---|---|---|---|---|
| **Gemma 4 E4B-it** (Google, 2026-04-02) | `google/gemma-4-E4B-it` (34.4 M dl); `unsloth/gemma-4-E4B-it-GGUF` (5.7 M dl) | 8.0 B total / ~4.5 B active | **Apache-2.0** | ~3 GB (4.5 GB "4-bit RAM" incl. activations `[C]`) | 128 K; text + image + audio in | day-0 llama.cpp; RTX 3060 Q8 ~45 t/s `[M]` |
| Gemma 4 E2B-it | `google/gemma-4-E2B-it` (20.3 M dl) | 5.1 B total / ~2 B active | Apache-2.0 | ~2 GB `[E]` | 128 K | crowd/ambient NPC tier |
| **Qwen3.5-4B** (Alibaba, 2026-02-27) | `Qwen/Qwen3.5-4B` (8.0 M dl); `-2B`, `-0.8B`, `-9B` | 4 B | **Apache-2.0** | ~2.5 GB `[E]` | image-text-to-text | strong tool calling lineage |
| Qwen3-4B-Instruct-2507 / Qwen3-8B | `Qwen/Qwen3-4B-Instruct-2507` (63.6 M dl), `Qwen/Qwen3-8B` (129 M dl) | 4 B / 8 B | Apache-2.0 | 2.5 / 5 GB | 32 K+ | previous generation, widest GGUF coverage |
| Nemotron 3 Nano 4B (NVIDIA, 2026-03) | via NVIDIA ACE | 4 B | NVIDIA open model terms (check) | - | "game agent capabilities", thinking budget | the ACE-blessed choice |
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | 3.8 B | **MIT** | 2.3 GB | 128 K | strong reasoning per byte |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` (gated) | 3.2 B | Llama 3.2 licence (attribution, 700 M MAU cap) | 2 GB | 128 K | superseded |
| Gemma 3n E4B | `google/gemma-3n-E4B-it` (gated) | 7.9 B total | Gemma terms | 3 GB | audio in | superseded by Gemma 4 (Apache) |

**Latency per turn on the 5070 Ti laptop.** Measured for Llama 3.1 8B Q4_K_M on the RTX 5070 Ti Laptop GPU: prompt
processing **2,911 tok/s**, generation **65.6 tok/s**, TTFT **455 ms** `[M]` (hardware-corner llama-bench). A 4 B Q4
model has roughly half the weight bytes, so `[E]` ~110-130 tok/s generation and ~5 K tok/s prefill. A typical NPC turn
(2 K-token persona + memory context, 60-token reply) is therefore `[E]` 0.4 s prefill + 0.5 s generation ~= **1 s**,
or ~0.5 s with KV-cache reuse of the static persona prefix (`llama-server` prompt caching); an 8 B model is ~2 s.
inZOI's 0.5 B SLM shows the floor: 1 GB VRAM and near-instant turns at the cost of coherence.

**Structured output / tool calling.** `llama-server --jinja` exposes OpenAI-style tools and `response_format` with a
JSON schema compiled to a GBNF grammar, so NPC actions (`{"act":"give_item","item_id":...}`) are guaranteed
well-formed; two open 2026 bugs (tool-call `arguments` returned as an object, occasionally malformed arguments for
simple schemas) mean the sidecar must still validate every action against the world graph. **vLLM has no native
Windows build** (WSL2, Docker Model Runner, or the community `SystemPanic/vllm-windows` fork); llama.cpp is the
Windows path and its CUDA build is the one to ship.

### 4.3 Voice

| Role | Model | HF id | Licence | Footprint / latency |
|---|---|---|---|---|
| **TTS (default)** | **Kokoro-82M** | `hexgrad/Kokoro-82M` (128 M dl) | **Apache-2.0** | weights < 1 GB, 2-3 GB total `[M]`; RTF ~0.04; 75 ms on an RTX 4060 Ti, 22 ms on a 5090 `[M]`; 54 preset voices, no cloning |
| TTS (cloned / emotive) | **Chatterbox / Chatterbox-Turbo (350 M)** | `ResembleAI/chatterbox` (22.9 M dl) | **MIT** | ~75 ms-< 200 ms `[C]`; 4 GB+ recommended; 23 languages; zero-shot cloning; adopted as NVIDIA ACE's TTS (2026-03) |
| TTS (multilingual) | Qwen3-TTS 0.6 B / 1.7 B (2026-01) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Apache-2.0 | 4-6 GB; ~97 ms streaming `[C]`; 3 s voice cloning |
| TTS (dialogue) | Dia-1.6B | `nari-labs/Dia-1.6B-0626` | Apache-2.0 | heavier; two-speaker scripts; not real-time on a shared GPU |
| TTS to avoid | XTTS-v2 | - | **CPML non-commercial** | - |
| **STT** | **whisper.cpp** (small.en / distil) | ggml | **MIT** | 9.4 s of audio in **0.13 s** on CUDA `[M]`; ~0.5 GB |
| STT (fast, multilingual) | Parakeet-TDT 0.6 B v3 | `nvidia/parakeet-tdt-0.6b-v3` (3.2 M dl) | **CC-BY-4.0** | 25 languages; NeMo/ONNX runtimes (sherpa-onnx on Windows) |

### 4.4 Shipped games and the known failure modes

| Game | Year | Brain | Voice | What broke / what it taught |
|---|---|---|---|---|
| Suck Up! (Proxima) | 2023-, 1.0 listed 2025-10 | cloud LLM | voice in | persuasion mechanics are exploitable; cloud cost per conversation is the business model |
| Vaudeville (Bumblebee) | EA 2023, 1.0 listed 2025-11 | cloud LLM | mic optional | detective premise tolerates latency; reviews flag slow turns |
| 1001 Nights / Book of Infinity (Ada Eden) | TBA | cloud LLM | text | the LLM is the mechanic (storytelling to the king), not a chat add-on |
| Whispers from the Star (Anuttacon) | 2025-08 | cloud | voice | survival-by-conversation; strongest design integration |
| Where Winds Meet | 2025-11 | cloud | voice | **players completed quests by describing actions never taken**; Steam AI disclosure added only after 28 days |
| inZOI Smart Zoi (Krafton + NVIDIA ACE) | 2025-03 | **on-device 0.5 B SLM, 1 GB VRAM** | - | nudge-by-text, not free chat; proves the on-device budget |
| Fortnite Darth Vader (Epic) | 2025-05 | Gemini + ElevenLabs | voice | moderation incidents within hours; SAG-AFTRA unfair-labour charge; Epic still shipped a UEFN NPC tool |
| Ubisoft Teammates (prototype) | 2025-11 | Gemini | voice | 80-person R&D team; not a product |
| Mantella (Skyrim/Fallout mod) | ongoing | user's choice (local or cloud) | TTS pipeline | the community reference for local pipelines and their setup pain |

Failure modes to design against: (1) **hallucinated quests / unearned completions** - the LLM must never be the
authority on world state; it emits *proposals* validated against the graph (Where Winds Meet). (2) **Cost scaling with
players** - cloud tokens per conversation turn a one-time purchase into a liability; on-device is the only fixed-cost
path. (3) **Latency** - > 1.5 s turns feel broken (Convai reviews); hide with "thinking" barks, streaming TTS, and
pre-generated openers. (4) **Moderation** - profanity/jailbreaks within hours (Vader); keep an allow-listed action
schema, a small classifier gate, and a canned fallback. (5) **Disclosure** - Steam requires AI-content disclosure;
the Where Winds Meet delay became the story. GDC's 2026 survey: 52 % of studios use generative AI, only 5 % put AI
output in front of players, and 52 % of developers think it harms the industry - expect scrutiny.

Sources: [Generative Agents](https://arxiv.org/abs/2304.03442); [NVIDIA ACE 2026 models](https://developer.nvidia.com/blog/nvidia-rtx-innovations-are-powering-the-next-era-of-game-development/), [ACE + Qwen3](https://developer.nvidia.com/blog/nvidia-ace-adds-open-source-qwen3-slm-for-on-device-deployment-in-pc-games/), [inZOI 1 GB VRAM](https://www.pcgamer.com/hardware/graphics-cards/with-nvidia-ace-taking-up-1-gb-of-vram-in-inzoi-team-green-will-need-to-up-its-memory-game-if-ai-npcs-take-off-in-pc-gaming/), [inZOI ACE launch](https://www.nvidia.com/en-us/geforce/news/nvidia-ace-naraka-bladepoint-inzoi-launch-this-month/);
[Inworld pricing](https://inworld.ai/resources/tts-api-pricing-comparison), [Inworld 2026 status](https://arcanumrpgs.com/blog/inworld-ai/); [Convai pricing](https://softwarefinder.com/artificial-intelligence/convai), [Convai latency review](https://scribehow.com/page/Convai_Review_2026_The_Most_Technically_Impressive_AI_Character_Platform__Held_Back_by_a_Latency_Problem_That_Wont_Go_Away__QJHPcZlvRkOQFemjqtlo4g);
[Fortnite UEFN NPCs](https://wccftech.com/fortnite-uefn-ai-npc-conversations-gemini-elevenlabs/), [Vader SAG-AFTRA](https://gamerant.com/fortnite-darth-vader-ai-npc/); [Ubisoft Teammates](https://www.gamedeveloper.com/business/ubisoft-s-first-playable-generative-ai-experience-is-an-r-d-experiment-called-teammates-);
[LLMUnity](https://undream.ai/LLMUnity/), [NobodyWho](https://godotassetlibrary.com/asset/MwOzul/nobodywho-local-llms-in-your-game), [Unity Inference Engine](https://docs.unity3d.com/6000.3/Documentation/Manual/com.unity.ai.inference.html);
[Gemma 4 GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF), [Gemma 4 local guide](https://unsloth.ai/docs/models/gemma-4), [Gemma 4 on RTX](https://markaicode.com/benchmarks/gemma-4-llamacpp-benchmark/); [5070 Ti laptop llama-bench](https://www.hardware-corner.net/gpu-ranking-local-llm/);
[llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [tool-call bug #20198](https://github.com/ggml-org/llama.cpp/issues/20198), [#22072](https://github.com/ggml-org/llama.cpp/issues/22072), [vLLM GPU install](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/), [vLLM on Windows 2026](https://fazm.ai/t/vllm-windows-support-2026);
[Kokoro latency](https://gigagpu.com/kokoro-tts-latency-by-gpu/), [Chatterbox repo](https://github.com/resemble-ai/chatterbox), [Chatterbox-Turbo](https://www.therundown.ai/tools/chatterbox-turbo), [Qwen3-TTS report](https://arxiv.org/abs/2601.15621), [open TTS 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models), [whisper.cpp vs faster-whisper](https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026);
[games with AI NPCs (2026 survey)](https://arcanumrpgs.com/blog/games-with-ai-npcs/); [GDC 2026 survey digest](https://ziva.sh/blogs/gdc-2026-generative-ai-game-development-data).

### 4.5 Recommendation (NPCs)

Run **one `llama-server` process in the Python sidecar** with **`google/gemma-4-E4B-it` Q4_K_M** (Apache-2.0; swap to
`Qwen/Qwen3.5-4B` if tool-call adherence measures better), `--jinja`, JSON-schema-constrained actions, a cached
persona prefix per NPC, and a hard per-turn budget (<= 1.2 s p95; otherwise fall back to a pre-generated line). Memory
stream + reflection live in the world state graph (graph as source of truth, ADR-0004), so the engine never trusts the
LLM: every action is a proposal validated by rules. Voice: **Kokoro-82M** for everyone, **Chatterbox-Turbo** for the
~20 named characters with cloned voices; **whisper.cpp small.en** for push-to-talk. Lip-sync from the TTS waveform via
Audio2Face-3D (UE) or MFCC visemes. Pre-generate opener lines, barks and quest text offline; reserve live generation
for player-initiated conversation. Budget: ~5 GB VRAM for the whole NPC stack (section 8).

---

## 5. Game engines for this path

| Criterion | **Unreal Engine 5.8** (2026-06-17) | **Unity 6.3 LTS** (2025-12-04) / 6.4 (2026-03-20) | **Godot 4.7** (2026-06-18) |
|---|---|---|---|
| Price / terms | free; **5 % royalty on gross above US $1 M per product**; Fab assets separate | **Personal free below US $200 K/yr**; Pro **US $2,310/yr** (US $210/mo); Enterprise above US $25 M; Runtime Fee cancelled 2024-09; Unity AI points-based (beta in 6.3, Muse sunset) | MIT, free |
| Account needed to install / build | **Epic account** for the launcher and for GitHub source access | **Unity ID + Hub sign-in**; Personal must go online every 30 days (Named User Licensing); offline activation only for Enterprise/Industry; CI needs licence activation (GameCI) | **none** (zip download) |
| Open-world runtime | **World Partition + HLOD, Nanite (+ Nanite Foliage 5.7), Lumen + Lumen Lite (5.8, 2x faster), MegaLights (prod 5.8), PCG (prod 5.7)**, Landscape + Mesh Terrain (experimental 5.8) | Terrain + Terrain Tools, Addressables + additive scenes (no world partition), GPU Resident Drawer / occlusion (Unity 6), Mesh LOD preview (6.4), ECS core packages (6.4) | Terrain3D add-on, manual threaded chunk streaming, visibility ranges, Jolt default (4.6), HDR output (4.7), no HLOD/Nanite equivalent |
| Animation | **Motion Matching (prod 5.4) + Game Animation Sample (500+ clips)**, IK Retargeter, Control Rig, **Mover 2.0 (beta 5.8)**, **Mutable** customisation, MetaHuman Creator in-editor | Animator/Animation Rigging; **Kinematica dead (last update 2020)**, MxM unsupported since 2025-11, open-source Unity MotionMatching (MIT) | AnimationTree, modular IK (4.6), motion-matching GDExtension (4.4, early), MetaHuman import allowed |
| AI / gameplay | **StateTree default in 5.8**, Behavior Trees, **Mass** (overhauled 5.8, off-thread entity creation), Smart Objects | **Unity Behavior** (official BT, free, 6000.0.16+), Behavior Designer Pro / NodeCanvas (paid), ML-Agents 4.x | **LimboAI 1.8.1** (BT + HSM, 2026-08), Beehave |
| Python / sidecar | editor Python (automation only); built-in **WebSockets** module; gRPC via TurboLink (C++/BP, streaming) | NativeWebSocket; gRPC via grpc-dotnet + YetAnotherHttpHandler (gRPC path landing 2026); ML-Agents gRPC precedent | built-in WebSocketPeer; py4godot / godot-python GDExtensions (prototype-grade); LibGodot embedding (4.6) |
| Lip-sync / faces | **Audio2Face-3D UE plugin (MIT)**, MetaHuman Animator | uLipSync, OVRLipSync, A2F via SDK/NIM | manual |
| Headless / CI | `UnrealEditor-Cmd ... -run=Cook` / UAT BuildCookRun on Windows or Linux runners; large (> 100 GB) | `-batchmode -nographics`; GameCI containers; licence activation in CI | `godot --headless --export-release` (trivial) |
| Laptop friendliness (12 GB, Windows) | heaviest: shader compile, DDC, Lumen VRAM; Lumen Lite helps; 1080p dev viewport | moderate | lightest |
| Fit for generative pipelines | **best**: Nanite eats 1 M-tri generated meshes, PCG eats masks, motion matching eats clip piles | good with manual LOD/retopo and scatter tooling | needs the most hand work |
| Roadmap risk | UE6 early access "by end of 2027" (merges UEFN, Verse, Scene Graph, MCP pipeline); 5.8 is the last planned UE5 | Unity 7 beta 2026-12, launch Q1 2027 ("direct continuation of Unity 6") | 4.8 in development |

Sources: [UE 5.8 release](https://www.unrealengine.com/news/unreal-engine-5-8-is-now-available), [UE 5.8 for indies](https://www.strayspark.studio/blog/unreal-engine-5-8-preview-indie-features-2026), [State of Unreal 2026](https://www.unrealengine.com/news/state-of-unreal-2026-top-news-from-the-show), [UE6 timeline](https://ludusengine.com/blog/unreal-engine-6-release-date), [Mover 2.0 guide](https://www.strayspark.studio/blog/mover-2-0-vs-character-movement-component-ue5-7-2026), [UE EULA](https://www.unrealengine.com/eula/unreal), [UE licensing 2026](https://300mind.studio/blog/unreal-engine-licensing-royalties/), [Mutable](https://dev.epicgames.com/documentation/unreal-engine/mutable-faq-in-unreal-engine);
[Unity 6.3 LTS](https://unity.com/blog/unity-6-3-lts-is-now-available), [Unity 6.4](https://www.cgchannel.com/2026/03/unity-releases-unity-6-4-and-unity-studio/), [Unity 7](https://gamedev.net/news/4582-unity-7-to-launch-in-q1-2027-new-engine-is-a-direct-continuation-of-unity-6/), [Runtime Fee cancelled](https://unity.com/blog/unity-is-canceling-the-runtime-fee), [Unity 2026 pricing](https://www.cgchannel.com/2025/11/price-of-paid-unity-subscriptions-to-rise-but-free-subs-extended/), [NUL FAQ](https://support.unity.com/hc/en-us/articles/23956583533844-Named-User-licensing-NUL-FAQ), [Unity Behavior](https://docs.unity3d.com/6000.0/Documentation/Manual/com.unity.behavior.html), [Kinematica status](https://discussions.unity.com/t/what-happened-to-kinematica/862682), [Unity AI beta](https://discussions.unity.com/t/unity-ai-beta-2026-is-here/1703625);
[Godot 4.6](https://godotengine.org/releases/4.6/), [Godot 4.7](https://godotengine.org/releases/4.7/), [Godot headless export](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_dedicated_servers.html), [LimboAI](https://github.com/limbonaut/limboai/releases), [godot-motion-matching](https://github.com/GuilhermeGSousa/godot-motion-matching), [py4godot](https://github.com/niklas2902/py4godot);
[TurboLink](https://github.com/thejinchao/turbolink), [YetAnotherHttpHandler](https://github.com/Cysharp/YetAnotherHttpHandler), [ML-Agents gRPC](https://github.com/Unity-Technologies/ml-agents/blob/main/ml-agents-envs/mlagents_envs/rpc_communicator.py).

### 5.1 Recommendation (engine)

**Unreal Engine 5.8**, decided by a two-week spike with three exit criteria: (a) a 4 km² World Partition map with
2,000 TRELLIS.2 Nanite props and PCG foliage runs at >= 60 fps at 1080p on the laptop with <= 6 GB VRAM; (b) a cook of
that map completes on the laptop in under 30 minutes; (c) the FastAPI/WebSocket bridge port (chunk JSON + asset URLs
-> `UWebSocket` + runtime glTF import) takes under a week. If (a) or (b) fails, **Unity 6.3 LTS** keeps the existing
UPM client and asset store controllers, at the price of manual LOD/retopo for every generated mesh and no motion
matching. Godot is reserved for sidecar tooling and tests (no account, headless export, LibGodot), not the product.
The Python sidecar, world graph and generation queue are engine-agnostic by design (ADR-0003/0004) and survive either
decision.

---

## 6. Combat, animation systems and open-world runtime techniques

| Need | Unreal 5.8 | Unity 6.3 | Godot 4.7 | Keep in Python? |
|---|---|---|---|---|
| Character controller | CharacterMovementComponent (default) or **Mover 2.0 (beta; networked prediction)**; GASP character | Opsive UCC / Invector TPC (paid), Kinematic Character Controller (free), own CharacterController | CharacterBody3D + Jolt | no |
| Locomotion / animation | **Motion Matching (PoseSearch)** + Chooser + GASP database; Animation Blueprints; Control Rig | Animator state machines, Animancer (paid), open-source MotionMatching (MIT, UPC); Animation Rigging | AnimationTree, motion-matching addon (early), modular IK | no |
| Hit reactions / ragdoll | Chaos + **PhysicsControl** component (physics-driven blends), Contextual Animation | PuppetMaster (paid), ragdoll blending, Active Ragdoll samples | PhysicalBoneSimulator3D | no |
| Streaming | **World Partition + HLOD + Nanite + Data Layers** | Addressables + additive scene streaming; GPU Resident Drawer; impostor assets (Amplify Impostors) | `ResourceLoader.load_threaded_request` chunking; visibility ranges; impostor add-ons | chunk scheduling stays in the sidecar (existing predictor) |
| LOD / impostors | Nanite (none needed); HLOD for far | Mesh LOD + impostors; LOD generation via `meshoptimizer` in the bake queue | LOD meshes + visibility ranges | LOD bake in Python (`gltfpack`) |
| Save system | `USaveGame` + plugins (SPUD/EMS) | Easy Save 3 (paid), own serializer | ConfigFile/JSON/ResourceSaver | **yes**: the world state graph *is* the save; engine persists only deltas (player, inventory, dynamic actors) and a graph snapshot id |
| Quest / dialogue tooling | **Yarn Spinner** (MIT; Unity/Godot/Unreal), **ink** (MIT; Inkpot for UE), articy:draft X (free single-user; UE/Unity exporters) | Yarn Spinner, ink (official), Dialogue System for Unity (Pixel Crushers, paid), articy | Yarn Spinner, Dialogue Manager (MIT), godot-ink | authored dialogue in Yarn/ink files; LLM lines fill slots the writer marks as open |
| Enemy AI | **StateTree (default framework in 5.8)**, Behavior Trees, **Mass** for crowds, Smart Objects, EQS | Unity Behavior (free), Behavior Designer Pro, utility-AI assets | LimboAI (BT + HSM), Beehave, utility add-ons | combat AI stays in-engine (deterministic, 60 Hz); strategic/social decisions (1-5 Hz) come from the sidecar |
| Bridge protocol | built-in WebSockets module; TurboLink gRPC (C++/BP streaming) if needed | NativeWebSocket; grpc-dotnet via YetAnotherHttpHandler | WebSocketPeer | **HTTP + WebSocket (ADR-0003)**, JSON for control, binary (MessagePack / raw) for arrays; ML-Agents' gRPC exists as the precedent if streaming tensors ever matter |

Tick budget and sync model: 60 fps = **16.7 ms** per frame. The bridge reader runs on its own thread and hands the
main thread a bounded queue; the main thread applies <= 1 ms of messages per frame (spawns, graph deltas, NPC
intents). Player state goes to the sidecar at 5 Hz (existing `POST /player`), NPC strategic decisions at 1-5 Hz, LLM
turns are fully asynchronous (0.5-2 s) with engine-side "thinking" animations, and pre-generation (the existing Markov
prewarm) covers chunks, barks and openers. Engine-local WebSocket round trips are sub-millisecond to low-millisecond in
principle; a Godot-3-era issue (#23898) reported 30-50 ms local round trips, and 4.x's `WebSocketPeer` exposes
`set_no_delay` for exactly this - measure the real number in each engine rather than assuming it. Determinism
rule: the engine owns physics/combat; the sidecar owns the world graph, generation and NPC cognition; conflicts are
resolved by the graph (ADR-0004).

Sources: [UE Mover](https://dev.epicgames.com/documentation/unreal-engine/mover-in-unreal-engine), [StateTree quick start](https://dev.epicgames.com/documentation/unreal-engine/statetree-quick-start-guide), [Mass 5.8](https://portal.productboard.com/epicgames/1-unreal-engine-public-roadmap/tabs/129-unreal-engine-5-8), [GASP motion matching](https://uhiyama-lab.com/en/notes/ue/motion-matching-game-animation-sample/);
[Unity MotionMatching (UPC)](https://upcommons.upc.edu/entities/publication/a7a391ec-542c-45ca-adab-e90fc36ba0a7), [MxM status](https://assetstore.unity.com/packages/tools/animation/motion-matching-for-unity-145624/reviews), [Unity Behavior](https://docs.unity3d.com/6000.0/Documentation/Manual/com.unity.behavior.html);
[Godot WebSocketPeer](https://docs.godotengine.org/en/stable/classes/class_websocketpeer.html), [Godot WebSocket latency issue](https://github.com/godotengine/godot/issues/23898); [Yarn Spinner](https://yarnspinner.dev), [ink](https://github.com/inkle/ink), [Inkpot](https://github.com/The-Chinese-Room/Inkpot);
[ML-Agents gRPC design thread](https://forum.unity.com/threads/ml-agents-grpc-design-choice.992710/), [TurboLink](https://github.com/thejinchao/turbolink).

### 6.1 Recommendation (runtime)

Buy nothing until the engine spike is done. On UE: GASP + Motion Matching for locomotion, CMC now and Mover 2.0 when
it leaves beta, PhysicsControl for hit reactions, StateTree + Mass for enemies, World Partition/HLOD/Nanite for
streaming, Yarn Spinner (MIT) for authored dialogue with LLM-filled slots, and the world graph as the save. Keep the
bridge on HTTP + WebSocket; add gRPC only after a profiler shows JSON parsing above 1 ms/frame.

---

## 7. Cost / time reality

| Game | Studio size | Engine | Timeline | Outcome |
|---|---|---|---|---|
| **Valheim** | **5 people** at EA | Unity | side project from 2017 -> EA 2021-02-02 (~3.5 yrs) -> **1.0 2026-09-09** (~9 yrs) | 5 M copies in a month, 10 M+ by 2022-07 |
| **Sons of the Forest** | **10-15 people** (ex-VFX) | Unity | announced 2019-12 -> EA 2023-02-23 -> 1.0 2024-02-22 (~4 yrs) | multi-million seller |
| **Enshrouded** | **~58 at EA, 75 now** ("doubled since development began") | own **Holistic Engine** (voxel) | announced 2023-05 -> EA 2024-01-24 -> 1.0 2026-10-15 (~2.75 yrs EA; engine work predates) | 1 M in a week, 5 M players by 2026-01 |
| **Dragon's Dogma 2** | **392 core credits** (1,000+ total credits) | RE Engine | ~4 yrs active development after DMC5 (2019) -> 2024-03 | AAA benchmark |
| **Crimson Desert** | **~200 (peak < 300)** | own BlackSpace Engine | **~7 years** -> 2026-03-19 | the reference target |

What this means: the smallest open-world survival-action games that shipped took **5-15 people and 3.5-4 years to a
sellable Early Access**, with in-house tooling and a narrow combat scope; Crimson-Desert-class scope took ~200 people
for 7 years. Asset volume is the multiplier: an open-world title streams **thousands to tens of thousands of unique
assets** (foliage, props, buildings, kit pieces) versus hundreds per level in a closed-world game `[C]` (industry
summaries; no public per-title counts). Generative tooling attacks exactly that multiplier - props, textures,
variants, rigs, clips, voice - while leaving design, combat feel, level flow, QA and optimisation at human speed. The
GDC 2026 data say the same thing from the other side: 52 % of studios use generative AI, but asset generation is only
19 % of uses and 5 % put AI output in front of players.

Sources: [Valheim](https://en.wikipedia.org/wiki/Valheim), [Valheim 1.0 interview](https://gamesbeat.com/valheim-finally-hits-1-0-after-five-years-in-early-access-interview/); [Sons of the Forest](https://en.wikipedia.org/wiki/Sons_of_the_Forest), [Endnight profile](https://xgamingserver.com/blog/the-developers-of-sons-of-the-forest-endnight-games-studio-profile/);
[Enshrouded](https://en.wikipedia.org/wiki/Enshrouded), [Keen Games about](https://www.keengames.com/about), [Enshrouded FAQ (engine)](https://enshrouded.com/en-US/FAQ); [Dragon's Dogma 2 credits](https://automaton-media.com/en/news/dragons-dogma-2s-credits-suggest-a-shockingly-small-number-of-development-staff/);
[Crimson Desert](https://en.wikipedia.org/wiki/Crimson_Desert), [Pearl Abyss CEDEC talk](https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process); [open vs closed world production](https://nipsapp.com/open-world-and-closed-world-games/); [GDC 2026 survey digest](https://ziva.sh/blogs/gdc-2026-generative-ai-game-development-data).

### 7.1 Recommendation (scope)

Plan a **Valheim-shaped** first milestone, not a Crimson-Desert-shaped one: one 4-16 km² region, 3-4 biomes, one
settlement kit, 8-12 enemy archetypes, one weapon class with motion-matched combat, 20 named NPCs with local LLM
dialogue and 200 crowd NPCs on canned lines. With generative tooling the asset wall for that scope is weeks of
unattended GPU time plus curation, which is the only reason 1-3 people can attempt it; budget 18-24 months to an Early
Access-quality vertical slice and treat every number in this document as something to re-measure in month one.

---

## 8. STACK RECOMMENDATION

| Decision | Choice | Why / constraints |
|---|---|---|
| **Engine** | **Unreal Engine 5.8** (fallback Unity 6.3 LTS after the section 5.1 spike) | Nanite absorbs 1 M-tri generated meshes without retopo/LOD; World Partition + HLOD streams; PCG (prod) scatters from Python masks; Motion Matching (prod) + 500 free clips; StateTree default; Audio2Face UE plugin (MIT); MetaHuman free in-engine. Cost: Epic account, 5 % above US $1 M, heavier laptop workflow |
| **3D asset model** | **`microsoft/TRELLIS.2-4B` (MIT)** via ComfyUI-core on Windows: 512³ locally (8 GB class, ~1 min), 1024³ on a 24 GB cloud GPU; `VAST-AI/TripoSG` (MIT) for watertight deformables; SPAR3D for sub-second placeholders; Hunyuan3D 2.1 only as a tagged, replaceable PBR-paint fallback (EU/UK/SK exclusion, 1 M MAU) | full PBR GLB; keep decimation target for Nanite; collision generated in-engine; commercial APIs (Tripo rig/retopo, Rodin quads) for the few animation-grade heroes |
| **Terrain** | **Terrain Diffusion 30 m (MIT, `xandergos/terrain-diffusion-30m`)** heightmap + climate -> Python post-pass (rivers, roads, Whittaker biomes, WFC/MarkovJunior settlements written to the world graph) -> Gaea 2 Indie erosion for hero regions -> UE Landscape tiles + PCG masks | seed-consistent O(1) random access matches the existing predictor/prewarm model; world models (Genie 3, Marble, HunyuanWorld) are reference/skybox only |
| **Animation path** | MetaHuman (humans) / TRELLIS.2 + **UniRig (MIT)** (creatures) -> UE IK Retargeter -> **Mixamo + Game Animation Sample** motion-matching database; bespoke clips from **`tencent/HY-Motion-1.0`** on a 24 GB cloud GPU, retargeted off SMPL; **no AMASS/HumanML3D-trained generators in the build**; faces via Audio2Face-3D | all clip generation is offline; the engine only ever sees retargeted FBX |
| **NPC LLM + voice** | **`google/gemma-4-E4B-it` Q4_K_M (Apache-2.0)** via `llama-server --jinja` + JSON-schema actions validated against the world graph (`Qwen/Qwen3.5-4B` as the A/B); **Kokoro-82M** (Apache-2.0) default TTS, **Chatterbox-Turbo** (MIT) for cloned named voices; **whisper.cpp small.en** (MIT) STT | `[E]` ~1 s per turn on the 5070 Ti (from 65.6 tok/s measured for 8B Q4); Generative-Agents memory/reflection in the sidecar; vLLM excluded (no Windows) |
| **Bridge protocol** | **HTTP + WebSocket (ADR-0003)**: JSON control messages, binary for arrays/heightmaps, PNG/KTX2/GLB over HTTP from the cache; UE built-in WebSockets module; gRPC (TurboLink) only if profiling demands streaming | reader thread + bounded queue, <= 1 ms/frame main-thread apply, player state 5 Hz, agent decisions 1-5 Hz, LLM async |
| **Pre-bake vs live** | **Pre-bake (offline, local 512³ or cloud 1024³):** all meshes, PBR textures, collision, terrain tiles, erosion, splat masks, settlement layouts, rigs, animation clips, NPC personas, authored dialogue, voice banks for scripted lines. **Live (in play, CPU or <= 5 GB GPU):** LLM dialogue for player-initiated talk, TTS for unscripted lines, WFC micro-layouts (interiors/dungeon rooms) on CPU, chunk scheduling/prewarm, graph updates. **Dev/creator mode only:** SD1.5 texture generation (3.1 GB measured), 3D generation | nothing diffusion-sized is resident during play |
| **VRAM budget (12 GB, during play)** | **Game (UE 5.8, 1080p, Lumen Lite + Nanite) <= 6.0 GB** `[E]`; **LLM** Gemma 4 E4B Q4 weights ~3.0 GB + 8 K KV cache ~0.5 GB = **3.5 GB**; **TTS** Kokoro ~0.5 GB; **STT** whisper small.en ~0.5 GB; **OS/compositor + headroom ~1.5 GB** | inZOI proves a 1 GB SLM beside a full UE5 game; if the game needs more, drop to Gemma 4 E2B (~2 GB) or run STT on CPU. Dev mode swaps the LLM out for SD1.5 (3.1 GB) or TRELLIS.2 512³ (~8 GB, game closed) |

---

## 9. Validation plan and open questions (measure before freezing)

1. **Engine spike (2 weeks):** UE 5.8 vs Unity 6.3 on the laptop - fps/VRAM at 1080p with 2,000 Nanite props + PCG
   foliage, cook time, bridge port effort (section 5.1 criteria).
2. **TRELLIS.2 on the 5070 Ti:** p50/p95 seconds and peak VRAM at 512³ (bf16 vs int8 ConvRot vs GGUF Q5) in ComfyUI
   core; mesh quality vs TripoSG on 50 prop prompts rendered through the existing DINOv2 coherence score.
3. **Terrain Diffusion 30 m:** VRAM and seconds per 1,024² tile; seed-consistency across adjacent tiles; river/road
   pass quality on 20 seeds.
4. **LLM turn latency:** llama-bench Gemma 4 E4B Q4_K_M and Qwen3.5-4B on the laptop; tool-call adherence on a
   200-prompt NPC action suite with JSON schema; prompt-cache hit rate with per-NPC persona prefixes.
5. **Co-residency:** run the game at target settings with llama-server + Kokoro + whisper.cpp resident and log peak
   VRAM, frame-time p95, and turn latency simultaneously (the whole section 8 budget is `[E]` until this runs).
6. **HY-Motion licence:** read `License.txt` in the repo (the raw path 404s; the HF card tags
   `tencent-hunyuan-community`); confirm territory and MAU clauses before any clip ships. Same review for every
   Hunyuan-derived asset tag in the world graph.
7. **UniRig on TRELLIS.2 meshes:** skinning quality on 20 creatures; whether the "planned" Rig-XL/VRoid checkpoints have
   shipped.
8. **Audio2Face-3D:** confirm RTX laptop latency inside UE 5.8 (plugin v2.5 targets 5.5/5.6) and the NVIDIA Open Model
   Licence terms for shipped weights.
9. **Steam AI disclosure text** drafted with the first build, not after launch.

---

## 10. Sources (all accessed 2026-10-06)

3D: [TRELLIS.2 repo](https://github.com/microsoft/TRELLIS.2), [TRELLIS.2-4B card](https://huggingface.co/microsoft/TRELLIS.2-4B), [Comfy-Org/TRELLIS.2](https://huggingface.co/Comfy-Org/TRELLIS.2), [ComfyUI native TRELLIS.2 news](https://comfyui-wiki.com/en/news/2026-08-22-trellis2-pixal3d-native-comfyui), [instasd guide](https://www.instasd.com/post/trellis-2-pixal3d-native-comfyui-guide), [ComfyUI Pixal3D docs](https://docs.comfy.org/tutorials/3d/pixal3d), [TRELLIS.2 6 GB GGUF](https://dailytopai.com/article/how-to-run-trellis-2-3d-ai-locally-on-just-6gb-vram-with-gguf-44.html), [trellis2-gguf](https://huggingface.co/ilintar/trellis2-gguf), [LocalAI GGUF](https://huggingface.co/LocalAI-io/TRELLIS.2-4B-GGUF), [TRELLIS v1 repo](https://github.com/microsoft/TRELLIS), [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large), [Hunyuan3D-2.1 repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), [Hunyuan3D-2.1 card](https://huggingface.co/tencent/Hunyuan3D-2.1), [2.1 LICENSE](https://huggingface.co/tencent/Hunyuan3D-2.1/raw/main/LICENSE), [Hunyuan3D-2 repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), [2.0 LICENSE](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2/main/LICENSE), [Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini), [Hunyuan3D-Part](https://huggingface.co/tencent/Hunyuan3D-Part), [Hunyuan3D versions](https://triposr.org/blog/hunyuan3d-versions), [Hunyuan3D 2.1 ComfyUI VRAM](https://www.tencentcloud.com/techpedia/146879?lang=en), [TripoSG repo](https://github.com/VAST-AI-Research/TripoSG), [TripoSG card](https://huggingface.co/VAST-AI/TripoSG), [TripoSF](https://huggingface.co/VAST-AI/TripoSF), [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D), [Stable3DGen](https://github.com/Stable-X/Hi3DGen), [trellis-normal](https://huggingface.co/Stable-X/trellis-normal-v0-1), [SPAR3D card](https://huggingface.co/stabilityai/stable-point-aware-3d), [SPAR3D repo](https://github.com/Stability-AI/stable-point-aware-3d), [SF3D](https://huggingface.co/stabilityai/stable-fast-3d), [Unique3D](https://huggingface.co/Wuvin/Unique3D), [CraftsMan](https://huggingface.co/wyysf/CraftsMan), [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), [SAM 3D blog](https://ai.meta.com/blog/sam-3d/), [sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects), [SAM licence](https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/LICENSE), [Sparc3D](https://lizhihao6.github.io/Sparc3D/), [Sparc3D/Hitem3D](https://www.vset3d.com/the-sparc3d-controversy-from-open-source-promise-to-paid-hitem3d-platform/), [Tripo API pricing](https://developers.tripo3d.ai/en/pricing), [Tripo rigging](https://developers.tripo3d.ai/en/models/rig), [Meshy pricing](https://www.meshy.ai/pricing), [Meshy commercial use](https://help.meshy.ai/en/articles/16102098-can-i-use-meshy-assets-commercially), [Meshy API changelog](https://docs.meshy.ai/en/api/changelog), [Rodin review](https://makerstack.co/reviews/hyper3d-rodin-review/), [Rodin on fal](https://fal.ai/models/fal-ai/hyper3d/rodin), [2026 3D model timeline](https://app.cinevva.com/guides/ai-3d-model-generation-timeline-2026).

Terrain / worlds: [terrain-diffusion](https://github.com/xandergos/terrain-diffusion), [InfiniteDiffusion page](https://xandergos.github.io/terrain-diffusion/), [MESA](https://openaccess.thecvf.com/content/CVPR2025W/MORSE/papers/Borne--Pons_MESA_Text-Driven_Terrain_Generation_Using_Latent_Diffusion_and_Global_Copernicus_CVPRW_2025_paper.pdf), [TerraFusion](https://arxiv.org/abs/2505.04050), [Geodiffussr](https://arxiv.org/abs/2511.23029), [Gaea 2.2](https://www.cgchannel.com/2025/07/quadspinner-releases-gaea-2-2/), [Gaea 3](https://www.cgchannel.com/2025/12/quadspinner-unveils-gaea-3-0/), [World Machine](https://www.world-machine.com/purchase.php), [World Creator 2026.4](https://www.cgchannel.com/2026/04/world-creator-2026-4-is-out-with-a-new-free-community-edition/), [Unity Terrain Tools](https://docs.unity3d.com/6000.2/Documentation/Manual/TerrainTools.html), [UE 5.7 release](https://www.unrealengine.com/news/unreal-engine-5-7-is-now-available), [UE 5.6 release](https://forums.unrealengine.com/t/unreal-engine-5-6-released/2538952), [UE 5.8 Mesh Terrain](https://80.lv/articles/unreal-engine-5-8-is-out-today-with-big-optimization-improvements-and-mesh-terrain), [Terrain3D 1.0](https://gamefromscratch.com/terrain3d-for-godot-hits-1-0-release/), [Terrain3D releases](https://github.com/TokisanGames/Terrain3D/releases), [Infinigen](https://github.com/princeton-vl/infinigen), [Infinigen licence](https://github.com/princeton-vl/infinigen/blob/main/LICENSE), [Infinigen Indoors](https://www.cgchannel.com/2025/06/open-source-tool-infinigen-indoors-generates-procedural-3d-interiors/), [Project Genie](https://en.wikipedia.org/wiki/Project_Genie_(website)), [Genie 3 access](https://fenxi.fr/en/blog/genie-3-google-deepmind-world-model-2/), [Marble essentials](https://help.scenario.com/articles/4774055514-marble-by-world-labs-the-essentials), [World Labs blog](https://www.worldlabs.ai/blog/bigger-better-worlds), [HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0), [HunyuanWorld-1 card](https://huggingface.co/tencent/HunyuanWorld-1), [WorldGen (LeoXie)](https://huggingface.co/LeoXie/WorldGen), [WorldGen CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_WorldGen_From_Text_to_Traversable_and_Interactive_3D_Worlds_CVPR_2026_paper.pdf), [worldgen-forge](https://github.com/Ardynai/worldgen-forge), [WorldClaw](https://arxiv.org/abs/2608.05248).

Characters / animation: [MPFB2](https://github.com/makehumancommunity/mpfb2), [MPFB 2 news](https://www.cgchannel.com/2025/03/check-out-open-source-blender-character-generation-plugin-mpfb-2/), [MakeHuman licence](https://static.makehumancommunity.org/about/license.html), [MetaHuman any engine](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/), [MetaHuman EULA](https://www.unrealengine.com/eula/mhc), [MetaHuman 5.6](https://forums.unrealengine.com/t/metahuman-5-6-release/2539165), [UniRig repo](https://github.com/VAST-AI-Research/UniRig), [UniRig card](https://huggingface.co/VAST-AI/UniRig), [RigAnything](https://arxiv.org/abs/2502.09615), [Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html), [Mixamo licence guide](https://www.licenseorg.com/guide/3d-assets/mixamo), [GASP](https://www.unrealengine.com/blog/game-animation-sample), [GASP 5.8](https://www.unrealengine.com/tech-blog/download-the-latest-game-animation-sample-project-now-updated-for-ue-5-8), [HY-Motion repo](https://github.com/Tencent-Hunyuan/HY-Motion-1.0), [HY-Motion card](https://huggingface.co/tencent/HY-Motion-1.0), [HY-Motion paper](https://arxiv.org/abs/2512.23464), [MoMask](https://arxiv.org/abs/2312.00063), [LGTM](https://arxiv.org/abs/2405.03485), [OpenT2M](https://arxiv.org/abs/2603.18623), [AMASS licence](https://amass.is.tue.mpg.de/license.html), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [SMPL-X licence](https://smpl-x.is.tue.mpg.de/modellicense.html), [Motion-X licence](https://motion-x-dataset.github.io/static/license/Motion-X%20License.pdf), [Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D), [Audio2Face-3D SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK), [A2F open source](https://www.opensourceforu.com/2025/09/nvidia-moves-audio2face-technology-to-open-source/), [uLipSync](https://github.com/hecomi/uLipSync), [Rhubarb](https://github.com/DanielSWolf/rhubarb-lip-sync).

NPCs / LLM / voice: [Generative Agents](https://arxiv.org/abs/2304.03442), [NVIDIA ACE 2026 models](https://developer.nvidia.com/blog/nvidia-rtx-innovations-are-powering-the-next-era-of-game-development/), [ACE Qwen3 SLM](https://developer.nvidia.com/blog/nvidia-ace-adds-open-source-qwen3-slm-for-on-device-deployment-in-pc-games/), [ACE in inZOI](https://www.nvidia.com/en-us/geforce/news/nvidia-ace-naraka-bladepoint-inzoi-launch-this-month/), [inZOI 1 GB VRAM](https://www.pcgamer.com/hardware/graphics-cards/with-nvidia-ace-taking-up-1-gb-of-vram-in-inzoi-team-green-will-need-to-up-its-memory-game-if-ai-npcs-take-off-in-pc-gaming/), [ACE explained 2026](https://arcanumrpgs.com/blog/nvidia-ace/), [Inworld TTS pricing](https://inworld.ai/resources/tts-api-pricing-comparison), [Inworld 2026](https://arcanumrpgs.com/blog/inworld-ai/), [Convai pricing](https://softwarefinder.com/artificial-intelligence/convai), [Convai latency](https://scribehow.com/page/Convai_Review_2026_The_Most_Technically_Impressive_AI_Character_Platform__Held_Back_by_a_Latency_Problem_That_Wont_Go_Away__QJHPcZlvRkOQFemjqtlo4g), [Fortnite UEFN NPCs](https://wccftech.com/fortnite-uefn-ai-npc-conversations-gemini-elevenlabs/), [Vader NPC](https://gamerant.com/fortnite-darth-vader-ai-npc/), [Kotaku on Fortnite AI](https://kotaku.com/fortnite-darth-vader-ai-npc-epic-disney-1851783545), [Ubisoft Teammates](https://www.gamedeveloper.com/business/ubisoft-s-first-playable-generative-ai-experience-is-an-r-d-experiment-called-teammates-), [games with AI NPCs 2026](https://arcanumrpgs.com/blog/games-with-ai-npcs/), [LLMUnity](https://undream.ai/LLMUnity/), [llama-cpp-unity](https://github.com/lookbe/llama-cpp-unity), [NobodyWho](https://godotassetlibrary.com/asset/MwOzul/nobodywho-local-llms-in-your-game), [Unity Inference Engine](https://docs.unity3d.com/6000.3/Documentation/Manual/com.unity.ai.inference.html), [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it), [Gemma 4 GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF), [Gemma 4 run guide](https://unsloth.ai/docs/models/gemma-4), [Gemma 4 llama.cpp bench](https://markaicode.com/benchmarks/gemma-4-llamacpp-benchmark/), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B-it), [GPU ranking llama-bench](https://www.hardware-corner.net/gpu-ranking-local-llm/), [RTX 5070/Ti local LLM](https://openclawdc.com/blog/best-local-llm-rtx-5070/), [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [JSON schema constrain](https://www.simplified.guide/llama-cpp/server-constrain-json-schema), [issue #20198](https://github.com/ggml-org/llama.cpp/issues/20198), [issue #22072](https://github.com/ggml-org/llama.cpp/issues/22072), [vLLM GPU install](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/), [vLLM on Windows 2026](https://fazm.ai/t/vllm-windows-support-2026), [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [Kokoro latency](https://gigagpu.com/kokoro-tts-latency-by-gpu/), [Chatterbox card](https://huggingface.co/ResembleAI/chatterbox), [Chatterbox repo](https://github.com/resemble-ai/chatterbox), [Chatterbox-Turbo](https://www.therundown.ai/tools/chatterbox-turbo), [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base), [Qwen3-TTS report](https://arxiv.org/abs/2601.15621), [Dia](https://huggingface.co/nari-labs/Dia-1.6B-0626), [open TTS 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models), [TTS 2026 compared](https://www.tryspeakeasy.io/blog/open-source-text-to-speech-2026), [whisper.cpp vs faster-whisper](https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026), [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [GDC 2026 survey digest](https://ziva.sh/blogs/gdc-2026-generative-ai-game-development-data), [AI in game dev 2026](https://www.genzopia.com/blog/ai-in-game-development-statistics-2026).

Engines / runtime: [UE 5.8 release](https://www.unrealengine.com/news/unreal-engine-5-8-is-now-available), [UE 5.8 indie preview](https://www.strayspark.studio/blog/unreal-engine-5-8-preview-indie-features-2026), [UE 5.8 roadmap](https://portal.productboard.com/epicgames/1-unreal-engine-public-roadmap/tabs/129-unreal-engine-5-8), [State of Unreal 2026](https://www.unrealengine.com/news/state-of-unreal-2026-top-news-from-the-show), [UE6 timeline](https://ludusengine.com/blog/unreal-engine-6-release-date), [Mover docs](https://dev.epicgames.com/documentation/unreal-engine/mover-in-unreal-engine), [Mover 2.0 guide](https://www.strayspark.studio/blog/mover-2-0-vs-character-movement-component-ue5-7-2026), [StateTree](https://dev.epicgames.com/documentation/unreal-engine/statetree-quick-start-guide), [Mutable FAQ](https://dev.epicgames.com/documentation/unreal-engine/mutable-faq-in-unreal-engine), [UE EULA](https://www.unrealengine.com/eula/unreal), [UE licensing 2026](https://300mind.studio/blog/unreal-engine-licensing-royalties/), [UE 5.7 PCG](https://videocardz.com/newz/unreal-engine-5-7-rolls-out-with-new-pcg-nanite-foliage-and-metahuman-tools), [Unity 6.3 LTS](https://unity.com/blog/unity-6-3-lts-is-now-available), [Unity 6.4 + Studio](https://www.cgchannel.com/2026/03/unity-releases-unity-6-4-and-unity-studio/), [Unity 7](https://gamedev.net/news/4582-unity-7-to-launch-in-q1-2027-new-engine-is-a-direct-continuation-of-unity-6/), [Runtime Fee cancelled](https://unity.com/blog/unity-is-canceling-the-runtime-fee), [Unity pricing 2026](https://www.cgchannel.com/2025/11/price-of-paid-unity-subscriptions-to-rise-but-free-subs-extended/), [Unity NUL FAQ](https://support.unity.com/hc/en-us/articles/23956583533844-Named-User-licensing-NUL-FAQ), [Unity Hub licences](https://docs.unity.com/en-us/hub/manage-license), [Unity Behavior](https://docs.unity3d.com/6000.0/Documentation/Manual/com.unity.behavior.html), [Kinematica status](https://discussions.unity.com/t/what-happened-to-kinematica/862682), [MxM reviews](https://assetstore.unity.com/packages/tools/animation/motion-matching-for-unity-145624/reviews), [Unity MotionMatching (UPC)](https://upcommons.upc.edu/entities/publication/a7a391ec-542c-45ca-adab-e90fc36ba0a7), [Unity AI beta](https://discussions.unity.com/t/unity-ai-beta-2026-is-here/1703625), [Unity AI cost thread](https://discussions.unity.com/t/unity-ai-is-way-out-of-reality-on-cost/1724381), [Godot 4.6](https://godotengine.org/releases/4.6/), [Godot 4.7](https://godotengine.org/releases/4.7/), [Godot HDR 4.7](https://godotengine.org/article/hdr-output-arrives-in-godot-4-7/), [Godot headless export](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_dedicated_servers.html), [Godot CLI](https://docs.godotengine.org/en/4.4/tutorials/editor/command_line_tutorial.html), [LimboAI releases](https://github.com/limbonaut/limboai/releases), [godot-motion-matching](https://github.com/GuilhermeGSousa/godot-motion-matching), [py4godot](https://github.com/niklas2902/py4godot), [godot-python docs](https://godot-python-extension.readthedocs.io/), [Godot WebSocketPeer](https://docs.godotengine.org/en/stable/classes/class_websocketpeer.html), [Godot WS latency issue](https://github.com/godotengine/godot/issues/23898), [TurboLink](https://github.com/thejinchao/turbolink), [YetAnotherHttpHandler](https://github.com/Cysharp/YetAnotherHttpHandler), [Unity gRPC 6.3 thread](https://discussions.unity.com/t/grpc-in-unity-6-3-better-now-or-nothing-changed/1698788), [ML-Agents rpc_communicator](https://github.com/Unity-Technologies/ml-agents/blob/main/ml-agents-envs/mlagents_envs/rpc_communicator.py), [ML-Agents gRPC design](https://forum.unity.com/threads/ml-agents-grpc-design-choice.992710/), [Yarn Spinner](https://yarnspinner.dev), [ink](https://github.com/inkle/ink), [Inkpot](https://github.com/The-Chinese-Room/Inkpot).

Cost / time: [Valheim](https://en.wikipedia.org/wiki/Valheim), [Valheim 1.0](https://gamesbeat.com/valheim-finally-hits-1-0-after-five-years-in-early-access-interview/), [Valheim 5 people](https://gamingbolt.com/valheim-was-made-by-a-team-of-just-5-people-developer-is-in-the-process-of-expanding), [Sons of the Forest](https://en.wikipedia.org/wiki/Sons_of_the_Forest), [Endnight profile](https://xgamingserver.com/blog/the-developers-of-sons-of-the-forest-endnight-games-studio-profile/), [Enshrouded](https://en.wikipedia.org/wiki/Enshrouded), [Keen Games](https://www.keengames.com/about), [Enshrouded 3 M players](https://wnhub.io/news/other/item-44557), [Enshrouded FAQ](https://enshrouded.com/en-US/FAQ), [Dragon's Dogma 2 credits](https://automaton-media.com/en/news/dragons-dogma-2s-credits-suggest-a-shockingly-small-number-of-development-staff/), [DD2 development](https://game8.co/articles/latest/dragons-dogma-2-took-12-years-to-make-because-of-scheduling-confilcts), [Crimson Desert](https://en.wikipedia.org/wiki/Crimson_Desert), [Pearl Abyss CEDEC](https://www.invenglobal.com/articles/24095/pearl-abyss-unveils-crimson-desert-development-process), [Crimson Desert team shift](https://tribune.com.pk/story/2599900/crimson-desert-development-team-shifts-to-new-project-after-release), [open vs closed world production](https://nipsapp.com/open-world-and-closed-world-games/), [RTX 5070 Ti Laptop specs](https://www.notebookcheck.net/Nvidia-GeForce-RTX-5070-Ti-Laptop-Benchmarks-and-Specs.934945.0.html).
