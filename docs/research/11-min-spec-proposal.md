# 11 - Minimum spec tiers and performance budget (proposal, resolves #34)

Status: **proposal** (research ticket #34; depends on the engine choice #23, which is not decided here).
Date: 2026-10-06. Tags as in doc 07: `[M]` measured by someone and cited, `[E]` our estimate, `[C]` vendor claim.
Every engine-specific number below is re-validated by the engine spike (#23) before it becomes an ADR;
until then each target "stays a target" (ADR-0005).

## 0. TL;DR - the three tiers

| | **Minimum** | **Recommended** | **Showcase** |
|---|---|---|---|
| GPU (NVIDIA / AMD / Intel) | GTX 1660 SUPER / RX 5600 XT / Arc A580 (RTX 2060 is the comfortable floor) | RTX 3060 12 GB / RX 6700 XT / Arc B580. **8 GB sub-profile:** RTX 4060, 5060, 3060 Ti, 3070, RX 7600, RX 9060 XT 8 GB | RTX 4070 Ti SUPER / RTX 5070 Ti / RX 9070 XT |
| VRAM | 6 GB | 12 GB (8 GB sub-profile, see §3) | 16 GB |
| CPU | 6 cores: Ryzen 5 2600 / i5-8400 | 8 cores: Ryzen 7 5700X / i7-12700 (6c/12t Ryzen 5 5600 works) | 8 cores: Ryzen 7 7700X / i7-13700K |
| RAM | 16 GB | 32 GB (16 GB with the LLM off) | 32 GB |
| Storage | 50 GB SSD (SATA ok) | 50 GB SSD | 50 GB NVMe |
| Target | 1080p output at 30 fps, 50 % internal res | 1080p at 60 fps, 67 % internal res (720p) | 1440p at 60 fps, DLSS/FSR Quality; 4K60 with Performance + FG |
| Upscaler | TSR (engine default); FSR 3.1 / XeSS SR on non-RTX | TSR; DLSS 4 on RTX; FSR 4 on RX 7000/9000; XeSS on Arc | DLSS 4 / FSR 4 (+ frame generation optional) |
| Graphics features | Nanite ON; Lumen Lite SW (Low); cascaded shadow maps (VSM off); MegaLights off; PCG foliage 50 %; 1K texture cap | Nanite ON; Lumen Lite SW (Medium); VSM Medium; MegaLights off; PCG 100 %; 2K textures | Nanite ON; full Lumen (HW RT on RTX/RDNA3+) High; VSM High; MegaLights ON; PCG 100 %; 4K hero textures |
| Agentic NPCs | **LLM off** -> authored lines + pre-baked Kokoro voice bank; STT off | 4B LLM resident (Gemma 4 E4B or Qwen3.5-4B, Q4_K_M, 4K ctx) + Kokoro GPU + whisper.cpp small. 8 GB sub-profile: 2B LLM, TTS on CPU | 4B at 8K ctx or 9B Q4; Kokoro; whisper |
| VRAM split (GB) | game <= 5.0 (driver 0.5, no AI) | game <= 6.0 / LLM 3.5 / TTS 0.5 / STT 0.5 / headroom 1.5 | game <= 8.0 / LLM 5.5 / TTS 0.5 / STT 0.5 / headroom 1.5 |
| Steam coverage (GPU, Sept 2026) | **~71 %** of surveyed GPUs | **~26 %** with 12 GB+ (<= 33 % if the 16 GB 4060 Ti/5060 Ti/9060 XT variants are counted); **~59 %** including the 8 GB sub-profile | **~9 %** |

Decision in one sentence: ship three tiers where the agentic-NPC differentiator is *full* on 12 GB
(~26 % of Steam), *light* on 8 GB (another ~33 %), and *authored-line fallback* on 6 GB (the remaining
~12 %), so the game runs on ~71 % of Steam GPUs and the LLM runs in some form on ~59 % - instead of the
~26 % the plan's single 12 GB budget would reach.

## 1. Evidence

### 1.1 Steam Hardware Survey (September 2026 = the survey current in October 2026)

The survey published at the start of October is the September 2026 one
([store.steampowered.com/hwsurvey](https://store.steampowered.com/hwsurvey/), read 2026-10-06; the October
survey appears in early November). Buckets `[M]`:

| Dimension | Shares | Derived |
|---|---|---|
| VRAM | 16 GB 27.21 %, 8 GB 26.71 %, 12 GB 13.06 %, 24 GB 5.50 %, 4 GB 5.13 %, 6 GB 5.12 %, 2 GB 3.69 %, 10 GB 1.91 % | >= 6 GB 84.7 %, >= 8 GB 79.6 %, >= 12 GB 50.1 %, >= 16 GB 37.1 % (the 16 GB bucket is padded by Apple/iGPU shared memory; use the per-card sums below) |
| System RAM | 32 GB 42.22 %, 16 GB 37.82 %, 8 GB 6.47 %, 64 GB 3.71 %, 24 GB 2.56 % | >= 16 GB 88.7 %, >= 32 GB 48.3 % |
| CPU cores | 8c 30.06 %, 6c 27.23 %, 4c 10.79 %, 10c 8.19 %, 16c 5.78 %, 14c 5.23 %, 12c 4.78 % | >= 6 cores 86.5 %, >= 8 cores 59.3 % |
| Resolution | 1080p 47.91 %, 1440p 27.04 %, 2560x1600 5.22 %, 4K 4.79 %, 3440x1440 2.91 % | 1080p is still the plurality; 1440p grew +5.2 pts in one month |
| OS / API | Windows 11 71.34 %, Windows 10 23.61 %; DirectX 12 GPUs 92.19 % | Windows 10 22H2 stays supported (UE 5.8 still ships it); DX12 + SM 6.6 is the Nanite floor |
| Top GPUs | RTX 5070 5.86 %, RTX 5060 4.20 %, 5060 Ti 3.87 %, 4060 3.72 %, 3060 3.54 %, 4060 Laptop 3.35 %, 3050 2.77 %, 4060 Ti 2.73 % | the single most common card is a **12 GB** card; the most common *class* is 8 GB |

Coverage method: sum of per-card shares on the
[video-card page](https://store.steampowered.com/hwsurvey/videocard/) for every card at >= 0.10 % that meets the
tier, rescaled x0.95 to the main-page base (the video-card page normalises ~5 % of systems out: RTX 5070 reads
6.15 % there vs 5.86 % on the main page). Laptop GPUs are counted at their real VRAM (RTX 3060 Laptop = 6 GB,
4070 Laptop = 8 GB, 5070 Ti Laptop = 12 GB). Cards sold in two VRAM sizes (4060 Ti, 5060 Ti, 9060 XT) are
counted at 8 GB; that is the conservative bound and the +7 pt upper bound is given with it.

| Tier set | Raw sum | Rescaled | Members (examples) |
|---|---|---|---|
| Showcase: >= RTX 4070 Ti SUPER class, >= 16 GB | 9.2 % | **~9 %** | 5070 Ti, 5080, 4070 Ti S, 4080/S, 4090, 5090, RX 9070/9070 XT, 7900 XT/XTX |
| Recommended-12: >= RTX 3060 class and >= 12 GB | 26.9 % | **~26 %** | + 5070, 3060 12 GB, 4070/S/Ti, 3080 Ti, RX 6700 XT, 6750 XT, 7700 XT, 7800 XT, 6800 XT, 5070 Ti Laptop, 4080 Laptop, 3090 |
| Recommended-8: >= RTX 3060 class with 8-11 GB | 34.6 % | **~33 %** | 5060, 5060 Ti, 4060 (+Laptop), 4060 Ti, 5060 Laptop, 3060 Ti, 3070/Ti, 3080 10 GB, 4070 Laptop, 5070 Laptop, RX 6600/XT, 7600, 9060 XT, 5700 XT, 2070 S, 2080/S/Ti, 1080 Ti |
| Minimum-only: 6-8 GB below RTX 3060 class | 13.0 % | **~12 %** | 3050, 3060 Laptop, 4050 Laptop, 2060, 2060 S, 2070, 1660/S/Ti, 1070/Ti, 1080, 3050 6 GB Laptop |
| **Minimum total** (all of the above) | 74.4 % | **~71 %** | adding GTX 1060 6 GB would give ~73 % |
| Excluded | ~17 % | | GTX 1650/1050 Ti/1050 (4 GB), 3050 4 GB laptops, 2050, RX 580/570, RX 6500 XT, every iGPU (Iris Xe, Radeon Graphics, UHD), GTX 970 |

### 1.2 Comparable titles

| Game (engine, year) | Minimum | Recommended | Install | Notes |
|---|---|---|---|---|
| **Crimson Desert** (BlackSpace, 2026) | GTX 1060 / RX 5500 XT, R5 2600X / i5-8500, 16 GB -> 1080p30 upscaled from 900p | RTX 2080 / RX 6700 XT, R5 5600 / i5-11600K, 16 GB -> 1080p60 or 4K30 | 150 GB SSD stated, ~121 GB installed | doc 06 §1.4; High = RTX 4070 -> 1440p60; Ultra = RTX 5070 Ti -> 4K60; Steam Deck "Verified" but sub-30 fps measured |
| Black Myth: Wukong (UE5, 2024) | GTX 1060 6 GB / RX 580 8 GB, i5-8400 / R5 1600, 16 GB -> 1080p30 | RTX 2060 / RX 5700 XT / Arc A750, i7-9700 / R5 5500, 16 GB -> 1080p60 | 130 GB | Nanite + Lumen shipping on Pascal at 1080p30 ([gamerequirementslab](https://gamerequirementslab.com/black-myth-wukong-system-requirements)) |
| S.T.A.L.K.E.R. 2 (UE5, 2024) | GTX 1060 6 GB / RX 580 8 GB, i5-7600K / R5 1600X, 8 GB -> 1080p30 Low | RTX 2070 S / RX 5700 XT / GTX 1080 Ti, i7-9700K / R7 3700X, 16 GB -> 1080p60 | 150 GB SSD | ~6 GB VRAM at lowest, < 10 GB at 4K; 8 GB cards "unplayable" at 1080p Epic ([TechSpot](https://www.techspot.com/review/2926-stalker-2-benchmark/)); RTX 4060 = 1080p60 Medium |
| Avowed (UE5, 2025) | GTX 1070 / RX 5700 / Arc A580, i5-8400 / R5 2600, 16 GB | RTX 3080 / RX 6800 XT, i7-10700K / R5 5600X, 16 GB | 75 GB | ([Destructoid](https://destructoid.com/minimum-and-recommended-pc-requirements-for-avowed)) |
| Clair Obscur: Expedition 33 (UE5, 2025) | GTX 1660 6 GB / RX 5600 XT / Arc A380, i7-8700K / R5 1600X, 8 GB | RTX 3070 / RX 6800 XT, i7-12700K / R5 5600X, 16 GB | 55 GB | the small-team UE5 comparable; 55 GB is our install-size reference ([Windows Central](https://www.windowscentral.com/gaming/clair-obscur-expedition-33-pc-system-requirements-and-specs-can-you-run-this-turn-based-rpg-adventure)) |
| Oblivion Remastered (UE5, 2025) | GTX 1070 Ti / RX 5700, i7-6800K / R5 2600X, 16 GB | RTX 2080 / RX 6800 XT, i5-10600K / R5 3600X, **32 GB** | 125 GB | > 8 GB VRAM at 1080p with SW Lumen; RTX 3060 ~40 fps with sub-30 lows at 1080p SW Lumen, RTX 4060 63 fps ([Notebookcheck](https://www.notebookcheck.net/The-Elder-Scrolls-IV-Oblivion-Remastered-review-Technology-and-benchmarks.1016625.0.html), [HardwareTimes](https://hardwaretimes.com/elder-scrolls-oblivion-remastered-gpu-benchmarks-performance/)) |
| Valheim (Unity, 2021) | GTX 950 / HD 7970, quad core, 8 GB | GTX 1060 / RX 580, i5 3 GHz, 16 GB | 1 GB | Unity fallback reference: no Nanite/Lumen, hand LODs ([systemrequirementslab](https://www.systemrequirementslab.com/cyri/requirements/valheim/20524)) |
| Sons of the Forest (Unity, 2023) | GTX 1060 3 GB / RX 570, i5-8400 / R3 3300X, 12 GB | GTX 1080 Ti / RX 5700 XT, i7-8700K / R5 3600X, 16 GB | 20 GB | Unity open world at GTX 1080 Ti recommended ([gamepressure](https://www.gamepressure.com/newsroom/sons-of-the-forest-system-requirements-revealed/z25058)) |

Reading: every 2024-26 UE5 open world puts Minimum at GTX 1060-1070 Ti with 6-8 GB for 1080p30 (upscaled) and
Recommended at RTX 2060-3070 for 1080p60; nobody ships below 6 GB VRAM; install sizes run 55-150 GB. The
Unity references sit one class lower because they carry no virtualised geometry or GI.

### 1.3 UE 5.8 feature costs on RTX 3060 / 4060 class at 1080p

| Feature | Cost | Source |
|---|---|---|
| Nanite (cull + raster + base pass) | guideline **< 4 ms** on RTX 3060 at native 1080p for 60 fps `[M-guideline]` | [Stray Spark, Nanite/foliage notes](https://medium.com/@shinsoj/notes-on-foliage-in-unreal-5-3522b6eb159f) |
| Virtual Shadow Maps, open-world test scene, UE 5.7.1 | **3.4 ms** total on RTX 3060 (0.4 ms page alloc, 2.1 ms render) before tuning `[M]` | [Stray Spark VSM open worlds](https://www.strayspark.studio/blog/virtual-shadow-map-optimization-open-worlds-ue5-7) |
| Lumen high quality vs **Lumen Lite** (new in 5.8) | 7-9 ms vs **3.5-4.5 ms** on a 28-CU GPU at 1080p (the Steam Machine; RTX 3060 has 28 SMs) `[M]`; Epic: "twice as fast as Lumen High", Switch 2 at 60 fps `[C]` | [Stray Spark 1080p60 budget](https://www.strayspark.studio/blog/optimize-unreal-engine-5-8-steam-machine-1080p-60fps), [Guru3D on 5.8](https://www.guru3d.com/story/unreal-engine-58-debuts-lumen-lite-and-productionready-megalights/) |
| Lumen reflections | keep **< 1 ms** at 1080p60 on RTX 3060 `[M-guideline]` | [Stray Spark Lumen/VSM case study](https://medium.com/@shinsoj/case-study-unreal-5-lighting-tech-in-immortals-of-aveum-f6a51305d700) |
| MegaLights (production in 5.8) | flat per-pixel floor independent of light count; pays off above "tens of shadowed lights"; shadow rays need RT hardware to be fast `[C]` | [Epic MegaLights docs](https://dev.epicgames.com/documentation/unreal-engine/megalights-in-unreal-engine), [Stray Spark MegaLights budget](https://www.strayspark.studio/blog/megalights-performance-budget-ue5-8) |
| Nanite hardware floor | DX12 with SM 6.6 64-bit atomics (or Vulkan `VK_KHR_shader_atomic_int64`): NVIDIA Maxwell+, AMD GCN+, Intel Arc `[C]` | [Epic hardware specs](https://dev.epicgames.com/documentation/unreal-engine/hardware-and-software-specifications-for-unreal-engine) |
| Upscalers, UE 5.8 | TSR built in; DLSS 4.5 plugin (SR, FG, MFG, RR, DLAA), FSR 4.1.1 plugin (RX 7000+ for ML upscaling, FSR 3.1 path below), XeSS 3.1 plugin (SR + MFG; XeLL broken on 5.8.0/5.8.1) `[C]` | [AMD GPUOpen](https://gpuopen.com/learn/amd-fsr-plugin-updated-for-unreal-engine-58/), [VideoCardz XeSS 3.1](https://videocardz.com/newz/intel-xess-plugin-3-1-adds-unreal-engine-5-8-support), [DLSS 4.5 for 5.8](https://x.com/DrkFX/status/2072992551045308848) |
| Upscalers, Unity 6 | STP built in (any GPU); DLSS 3.7 and FSR 2 native in HDRP; DLSS 4 / FSR 4 / XeSS only via third-party assets as of 6.1 `[C]` - re-check on 6.3 LTS at the spike | [Unity discussions, state of upscaling](https://discussions.unity.com/t/state-of-upscaling-features/1663052) |

### 1.4 Co-resident AI footprint (the half of the budget no comparable title has)

| Component | Footprint | Source |
|---|---|---|
| 4B LLM, Q4_K_M, llama.cpp | Qwen3-4B: ~2.9 GiB at 4K context, ~4 GB at 8K, 6.8 GiB at 32K (FP16 KV) `[M]` | [localllm.in llama.cpp VRAM guide](https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms), [KV cache guide](https://insiderllm.com/guides/kv-cache-optimization-guide/) |
| Gemma 4 E4B-it, Q4_K_M | **GGUF file 5.0-5.3 GB** (the per-layer-embedding tables live in the file); ~2.7 GB of weights on the GPU when PLE stays in host RAM, "4+ GB recommended" `[C]` | [Unsloth Gemma 4 docs](https://unsloth.ai/docs/models/gemma-4/qat), [llmrun.dev](https://llmrun.dev/model/google-gemma-4-e4b-it-assistant) |
| Kokoro-82M TTS | ONNX fp16 169 MB / int8 88 MB; "any GPU with 1 GB" runs it; near-real-time on CPU `[C]`; the PyTorch path measured 2-3 GB total in doc 07 because of the torch CUDA context | [kokoro-82M-onnx](https://huggingface.co/suryatmodulus/kokoro-82M-onnx-opt), [fastkokoro](https://pypi.org/project/fastkokoro/0.6.0/) |
| Chatterbox / Chatterbox-Turbo TTS | **5 GB (Turbo) to 6.5-7 GB** VRAM `[C]` - does not fit any live budget below 24 GB | [Chatterbox-TTS-Server docs](https://github.com/devnen/Chatterbox-TTS-Server/blob/main/documentation.md) |
| whisper small STT | ~0.5 GB int8 / 0.95 GB fp16 (faster-whisper); whisper.cpp small ~0.5 GB, 9.4 s of audio in 0.13 s on CUDA `[M]` (doc 07) | [llmapi faster-whisper-small](https://llmapi.ai/models/systran-faster-whisper-small/) |
| Precedent: inZOI "Smart Zoi" (NVIDIA ACE, 0.5B SLM, TensorRT-LLM, ~1 GB) | **+2-4 GB VRAM** in practice; stutter reports even below 90 % GPU load; the game **auto-drops every graphics setting one level** when the SLM is on; requires RTX 3060+ `[M]` | [inZOI wiki](https://thegameswiki.com/inzoi/wiki/smart-zoi), [Steam thread](https://steamcommunity.com/app/2456740/discussions/0/599660572366213860/) |

Three consequences for the plan's §5 split (game 6.0 / LLM 3.5 / TTS 0.5 / STT 0.5 / headroom 1.5):

1. **LLM 3.5 GB holds only at <= 4K context with a q8 KV cache**, and for Gemma 4 E4B only if llama.cpp keeps
   the PLE tables in host RAM (measure at the spike; Qwen3.5-4B at ~2.5 GB of weights is the safer fit).
2. **TTS 0.5 GB means Kokoro through ONNX Runtime, not PyTorch.** Chatterbox-Turbo is an offline voice-bank
   baker only; a live clone needs 5 GB that no tier has.
3. **Headroom 1.5 GB is where the WDDM reserve (~0.4), the CUDA contexts (~0.3 each) and the player's
   Discord/browser (~0.5-1.0) go.** Three separate sidecar processes would eat ~0.9 GB of it in contexts alone,
   so the runtime sidecar hosts LLM, TTS and STT in **one** process (one CUDA context).

## 2. The 8 GB question (asked by the ticket)

Viable for graphics, not for the full AI budget. 8 GB is the largest card class on Steam (26.7 % of systems;
~33 % of GPUs are 3060-class cards with 8-11 GB) and the comparables' 1080p60 Recommended class
(RTX 4060 = S.T.A.L.K.E.R. 2 Medium 1080p60; RTX 3070 = Clair Obscur recommended). But the game alone needs
5-6 GB at 1080p Medium (S.T.A.L.K.E.R. 2 lowest = 6 GB; Oblivion Remastered > 8 GB), so there is no room for a
4B model. What the **Recommended-8** sub-profile drops, in order:

| Item | 12 GB profile | 8 GB sub-profile | Why |
|---|---|---|---|
| LLM | 4B Q4_K_M, 4K ctx, 3.5 GB | **2B**: Gemma 4 E2B-it Q4 (~2 GB incl. KV; doc 07 §4.2 "crowd tier") | coherence drops one notch; the grounding verifier and JSON-schema actions stay |
| TTS | Kokoro on GPU, 0.5 GB | **Kokoro on CPU** (ONNX int8, 2 threads) | near-real-time on CPU `[C]`; named-NPC lines are pre-baked anyway |
| STT | whisper.cpp small on GPU, 0.5 GB | whisper.cpp small on GPU 0.5 GB (push-to-talk burst) or CPU | |
| Game VRAM | <= 6.0 GB | **<= 4.0 GB**: textures one notch down, VSM one notch down, Nanite streaming pool 512 MB | the inZOI rule: one notch down when the model is resident |
| Headroom | 1.5 GB | 1.0 GB (+ 0.5 driver) | |
| Player-facing | "Living NPCs: Full" | "Living NPCs: Light" (default on 8 GB) or "Off" (authored lines) | the settings menu says what it costs |

Without this sub-profile the differentiator reaches ~26 % of Steam; with it ~59 %. The 6 GB Minimum tier
gets authored lines only: a 6-core CPU running a 4B model at ~10 tok/s beside the game thread is stutter, not a
feature.

## 3. Tier definitions (the decision)

### Minimum - "it runs" (~71 % of Steam GPUs)

- GPU: GTX 1660 SUPER 6 GB / RX 5600 XT 6 GB / Arc A580 8 GB; DX12 with SM 6.6 (Nanite floor). GTX 1060 6 GB
  may run at 720p30 Low but is unsupported.
- CPU 6 cores (Ryzen 5 2600 / i5-8400); RAM 16 GB; 50 GB SSD (SATA is enough for World Partition cells);
  Windows 10 22H2 / 11 64-bit.
- 1080p output at 30 fps from 50 % internal resolution (TSR; FSR 3.1 or XeSS SR selectable). Nanite on
  (every generated mesh is Nanite; the fallback-mesh path is not maintained), Lumen Lite SW Low, cascaded shadow
  maps (VSM off), MegaLights off, PCG foliage density 50 %, 1K texture cap, no RT.
- AI: LLM off; NPC dialogue from the authored line bank (every spine NPC ships >= 40 authored lines, every
  generated NPC >= 12 pre-generated and reviewed) with the pre-baked Kokoro voice bank; STT off; the world-graph
  service still runs (1 CPU thread).

### Recommended - "the game as designed" (~26 % at 12 GB; ~59 % with the 8 GB sub-profile)

- GPU: RTX 3060 12 GB / RX 6700 XT 12 GB / Arc B580 12 GB (RTX 5070, 4070, 4060 Ti 16 GB, 5060 Ti 16 GB,
  RX 9060 XT 16 GB all qualify). 8 GB sub-profile per §2.
- CPU 8 cores (Ryzen 7 5700X / i7-12700; a 6c/12t Ryzen 5 5600 passes with the sidecar pinned to 2 threads);
  RAM 32 GB (16 GB passes with the LLM off: UE5 8-12 GB + GGUF 5 GB mapped + Python sidecar 2-3 GB does not fit
  16 GB with a browser open); 50 GB SSD.
- 1080p at 60 fps from 67 % internal (1280x720; TSR default, DLSS 4 Quality on RTX, FSR 4 on RX 7000/9000,
  XeSS on Arc). Nanite on, Lumen Lite SW Medium, VSM Medium, MegaLights off, PCG 100 %, 2K textures.
- AI: Gemma 4 E4B-it or Qwen3.5-4B Q4_K_M at 4K context with q8 KV (A/B per the plan), Kokoro ONNX on GPU,
  whisper.cpp small on GPU; one sidecar process. VRAM: game <= 6.0 / LLM 3.5 / TTS 0.5 / STT 0.5 / headroom
  1.5 = 12.0 (unchanged from the plan §5, now with the three caveats of §1.4).

### Showcase - "the trailer" (~9 %)

- GPU: RTX 4070 Ti SUPER 16 GB / RTX 5070 Ti 16 GB / RX 9070 XT 16 GB; CPU 8 cores (Ryzen 7 7700X /
  i7-13700K); RAM 32 GB; 50 GB NVMe.
- 1440p at 60 fps with DLSS 4 / FSR 4 Quality (4K60 with Performance mode + frame generation as an option).
  Nanite on, full Lumen High (hardware RT where present, SW otherwise), VSM High, MegaLights on (night-time
  settlements with dozens of shadowed lights are the showcase), PCG 100 %, 4K hero textures.
- AI: the 4B model at 8K context (~4 GB) or Qwen3.5-9B Q4 (~5.5 GB); Kokoro; whisper. VRAM: game <= 8.0 /
  LLM 5.5 / TTS 0.5 / STT 0.5 / headroom 1.5 = 16.0.

### Why these GPUs and not one class lower/higher

- Minimum at GTX 1660 SUPER (not GTX 1060) because our team has no optimisation budget for a Pascal 1080p30
  pass that Wukong and S.T.A.L.K.E.R. 2 had; the three GTX 16-series cards add ~2.4 pts over an RTX 2060 floor
  at no engineering cost (TSR/FSR run everywhere).
- Recommended keeps the plan's RTX 3060 because it is the plan's assumption, the 12 GB 3060 is still 3.54 % of
  Steam, and it is the card the Lumen/VSM/Nanite measurements above were taken on.
- Showcase at 4070 Ti SUPER class because full Lumen HW + MegaLights at 1440p60 does not fit a 4070 / 9070-class
  card without dropping one of them (§4), and 16 GB is what a 9B model plus 1440p render targets need.

## 4. Budgets

### 4.1 GPU frame-time split, Recommended (RTX 3060 12 GB, 1080p output, 67 % internal, 60 fps = 16.7 ms)

| Pass | ms | Basis |
|---|---|---|
| Nanite: cull + raster + base pass (2,000 props + Nanite foliage) | 3.5 | < 4 ms guideline at native 1080p; we render 720p internal and keep the margin for foliage `[E]` |
| Virtual Shadow Maps (1 directional + <= 2 shadowed locals, Medium, directional LOD bias +1) | 2.5 | 3.4 ms measured before tuning `[M]`, target after tuning `[E]` |
| Lumen Lite SW GI + reflections (Medium) | 3.5 | 3.5-4.5 ms on 28 CUs at 1080p `[M]` |
| Direct lighting (no MegaLights), sky/atmosphere, volumetric clouds, fog, translucency, water | 2.0 | `[E]` |
| Post-process + TSR 720p->1080p + UI | 1.5 | `[E]` |
| **Game GPU total** | **13.0** | |
| LLM/TTS co-residency reserve (WDDM time-slices during a turn) | 2.0 | 12 %; inZOI stutters without one `[E]` |
| Slack | 1.7 | |
| **Frame** | **16.7** | |

CPU, same tier: game thread <= 12 ms (sim 3.0, motion matching 2.5, NPC StateTree/Mass 2.0, World Partition +
HLOD streaming 1.5, physics 1.5, bridge apply <= 1.0 per #23, misc 0.5); render thread <= 10 ms; RHI thread
<= 8 ms; the sidecar is pinned to 2 physical cores (llama.cpp 1 thread when GPU-offloaded, whisper burst,
Kokoro CPU on the 8 GB sub-profile) so it never shares a core with the game thread.

**Dialogue-mode rule.** During an LLM turn (<= 1 s per the plan) the floor is 45 fps (22 ms) on Recommended;
generation is paced to <= 40 tok/s in the open world (a 60-token line streams into TTS over ~1.5 s) and unpaced
in the dialogue camera, where the close framing cuts the render cost by roughly a third. The spike measures
1 % lows during a turn, not just the idle average.

### 4.2 Minimum (GTX 1660 SUPER 6 GB, 1080p output, 50 % internal, 30 fps = 33.3 ms)

Nanite 6.0, cascaded shadow maps 2.5, Lumen Lite SW Low 4.0, lighting/atmosphere 3.0, post + upscale 2.0 =
**17.5 ms game**, no AI reserve, 15.8 ms slack `[E]` (the 1660 SUPER is ~0.55x an RTX 3060; the slack covers
power-limited laptop 2060/3050-class parts). The game must stay under **5.0 GB** VRAM here (0.5 GB driver
reserve on a 6 GB card), which is why textures cap at 1K and VSM is off.

### 4.3 Showcase (RTX 4070 Ti SUPER 16 GB, 1440p output, DLSS/FSR Quality = 960p internal, 60 fps)

Nanite 3.0, VSM High 3.0, Lumen HW RT High 4.5, MegaLights 2.0, lighting/atmosphere/water 1.5, post + DLSS 1.0
= **15.0 ms game**, AI reserve 1.0 (same LLM work on a 2.2x faster GPU), slack 0.7 `[E]`. This is why a 4070 /
9070-class card cannot be Showcase: at 1440p60 it must drop MegaLights or fall back to Lumen Lite.

### 4.4 Storage

50 GB SSD budget (hard cap 60 GB, measured at every milestone build): ~20 GB Nanite meshes (2,000 props at
~10 MB per 1M-tri Nanite mesh), ~12 GB BC7 textures (1K small / 2K hero), ~1 GB terrain + splat masks,
~1.5 GB motion-matching database + rigs, ~2 GB audio + voice banks (Opus), ~1 GB shaders, **~6 GB AI models
(the 5 GB Gemma 4 E4B GGUF, whisper small, Kokoro)**, ~5 GB slack. Clair Obscur's 55 GB is the comparable; the
120-150 GB class of Crimson Desert/Wukong/S.T.A.L.K.E.R. 2 is not ours.

## 5. What the engine spike (#23) must re-validate

Every number in §1.3 and §4 is engine-specific and was measured on UE 5.7/5.8 by third parties. The spike
report replaces them with `realmweaver bench` outputs (ADR-0005):

1. fps and 1 % lows at 1080p/67 % with the 2,000-prop PCG map, LLM idle **and** during a turn; VRAM of the game
   alone (the <= 6.0 GB criterion already in #23).
2. **Derate rule for the dev laptop**: the RTX 5070 Ti Laptop GPU is ~1.5-1.6x an RTX 3060 at 1080p (it trails
   a desktop 4070 Ti by ~13 %, and a 4070 is ~54 % faster than a 3060 -
   [checkfps](https://checkfps.io/en/compare/rtx-5070-ti-laptop-vs-rtx-4070-ti), [Notebookcheck 3060](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3060-Desktop-GPU-Benchmarks-and-Specs.579000.0.html)).
   So **1440p60 on the laptop ~ 1080p60 on an RTX 3060** (1.78x the pixels); the spike's 60 fps criterion is
   run at 1440p on the laptop, or at 1080p on a borrowed 3060 / a 4060 8 GB (for the sub-profile).
3. Lumen Lite vs Lumen SW Medium cost on the actual biome content; VSM vs CSM on the Minimum profile; whether
   MegaLights is needed anywhere below Showcase.
4. LLM residency: VRAM of Gemma 4 E4B Q4_K_M at 4K ctx with q8 KV in one process beside the game; the fps hit of
   a 60-token turn at 40 tok/s paced vs unpaced; Qwen3.5-4B as the A/B; Gemma 4 E2B for the 8 GB sub-profile.
5. Upscaler plugin state at the chosen engine version (DLSS 4.5 / FSR 4.1.1 / XeSS 3.1 on UE 5.8; STP /
   DLSS 3.7 / FSR 2 on Unity 6.3 LTS). If the spike falls back to Unity, Minimum can drop one GPU class (no
   Nanite/Lumen) at the price of hand LODs, and the §4 tables are rebuilt from scratch.

## 6. Open items and risks

- The survey rescaling (x0.95) and the "8 GB" count for dual-VRAM SKUs are method choices; the honest range
  for Recommended-12 is 26-33 %.
- The 32 GB RAM line for Recommended is driven by the sidecar (GGUF mapping + Python); if the spike shows the
  stack fits 16 GB with the LLM on, drop it to 16 GB (+40 pts of RAM coverage).
- Gemma 4 E4B's 5 GB GGUF vs "~3 GB" in the plan: the plan's number is the GPU-resident part, the file is not.
  Install size and host RAM carry the difference.
- Windows 10 is 23.6 % of Steam and out of Microsoft support since 2025-10; we ship on it for 1.0 and
  re-check the share at M2.
- Nothing here is measured by us yet; it is a proposal with sourced third-party numbers, which is what
  ADR-0005 allows a research ticket to produce.

## 7. Sources (all accessed 2026-10-06)

Steam: [hwsurvey](https://store.steampowered.com/hwsurvey/), [videocard page](https://store.steampowered.com/hwsurvey/videocard/), [Tom's Hardware on the Sept 2026 survey](https://www.tomshardware.com/pc-components/16gb-gpus-and-8-core-cpus-officially-become-the-most-popular-configs-on-steam-latest-hardware-survey-shows-modern-gamings-growing-hunger-for-more-resources), [Igor's Lab 16 GB vs 8 GB](https://www.igorslab.de/en/16-gb-vram-overtakes-8-gb-steam-survey-gaming-pcs/), [TechPowerUp 32 GB RAM](https://www.techpowerup.com/353334/steam-hardware-survey-32-gb-overtakes-16-gb-as-the-most-common-system-ram-configuration).
Comparables: doc 06 §1.4 (Crimson Desert), [Corsair Crimson Desert specs](https://www.corsair.com/us/en/explorer/gamer/gaming-pcs/crimson-desert-system-requirements-and-recommended-specs/), [Wukong](https://gamerequirementslab.com/black-myth-wukong-system-requirements), [S.T.A.L.K.E.R. 2 specs](https://www.guru3d.com/story/stalker-2-heart-of-chornobyl-pc-system-requirements-announced/), [S.T.A.L.K.E.R. 2 benchmark](https://www.techspot.com/review/2926-stalker-2-benchmark/), [Avowed](https://destructoid.com/minimum-and-recommended-pc-requirements-for-avowed), [Clair Obscur](https://www.windowscentral.com/gaming/clair-obscur-expedition-33-pc-system-requirements-and-specs-can-you-run-this-turn-based-rpg-adventure), [Oblivion Remastered specs](https://box.co.uk/blog/elder-scrolls-oblivion-remastered-pc-system-requirements), [Oblivion benchmarks](https://www.notebookcheck.net/The-Elder-Scrolls-IV-Oblivion-Remastered-review-Technology-and-benchmarks.1016625.0.html), [Valheim](https://www.systemrequirementslab.com/cyri/requirements/valheim/20524), [Sons of the Forest](https://www.gamepressure.com/newsroom/sons-of-the-forest-system-requirements-revealed/z25058).
UE 5.8: [Lumen Lite / MegaLights (Guru3D)](https://www.guru3d.com/story/unreal-engine-58-debuts-lumen-lite-and-productionready-megalights/), [Stray Spark 1080p60 budget](https://www.strayspark.studio/blog/optimize-unreal-engine-5-8-steam-machine-1080p-60fps), [Stray Spark Lumen vs Lite](https://www.strayspark.studio/blog/lumen-vs-lumen-lite-ue5-8-performance), [Stray Spark VSM](https://www.strayspark.studio/blog/virtual-shadow-map-optimization-open-worlds-ue5-7), [Stray Spark MegaLights](https://www.strayspark.studio/blog/megalights-performance-budget-ue5-8), [Epic MegaLights](https://dev.epicgames.com/documentation/unreal-engine/megalights-in-unreal-engine), [Epic Lumen performance guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-performance-guide-for-unreal-engine), [Epic hardware specs](https://dev.epicgames.com/documentation/unreal-engine/hardware-and-software-specifications-for-unreal-engine), [Tom Looman 5.8 performance](https://tomlooman.com/unreal-engine-5-8-performance-highlights/), [FSR 4.1.1 plugin](https://gpuopen.com/learn/amd-fsr-plugin-updated-for-unreal-engine-58/), [XeSS 3.1 plugin](https://videocardz.com/newz/intel-xess-plugin-3-1-adds-unreal-engine-5-8-support), [Unity upscaling state](https://discussions.unity.com/t/state-of-upscaling-features/1663052).
AI footprint: doc 07 §4.2-4.3, [llama.cpp VRAM guide](https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms), [Unsloth Gemma 4](https://unsloth.ai/docs/models/gemma-4/qat), [Kokoro ONNX](https://huggingface.co/suryatmodulus/kokoro-82M-onnx-opt), [Chatterbox server docs](https://github.com/devnen/Chatterbox-TTS-Server/blob/main/documentation.md), [faster-whisper small](https://llmapi.ai/models/systran-faster-whisper-small/), [inZOI Smart Zoi](https://thegameswiki.com/inzoi/wiki/smart-zoi), [inZOI stutter thread](https://steamcommunity.com/app/2456740/discussions/0/599660572366213860/).
GPU scaling: [5070 Ti Laptop vs 4070 Ti](https://checkfps.io/en/compare/rtx-5070-ti-laptop-vs-rtx-4070-ti), [RTX 3060 benchmarks](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3060-Desktop-GPU-Benchmarks-and-Specs.579000.0.html), [RTX 4070 benchmarks](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4070-Desktop-GPU-Benchmarks-and-Specs.742254.0.html).
