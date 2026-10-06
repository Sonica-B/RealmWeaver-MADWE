# 01 - Model landscape for fast 2D game-asset generation (state of the art as of 2026-10-06)

Scope: few-step text-to-image models that can produce **seamless tileable textures, pixel-art sprites and props** on a
consumer NVIDIA GPU with **6-12 GB VRAM on Windows**, with **LoRA style adapters** (the "biome LoRAs" of RealmWeaver).
The original project used SD1.5 + rank-8 LoRA + latent consistency (LCM-LoRA); this document decides whether and how to
move on from that.

How to read the numbers:

- `[M]` measured by a cited third party on the named GPU; `[C]` vendor/model-card claim; `[E]` my estimate derived from a
  cited measurement (arithmetic shown). Estimates must be re-validated on the target GPU before any design decision is
  frozen (a validation script is proposed in section 11).
- All Hugging Face (HF) download counts, parameter counts and licenses were read from the HF Hub API on 2026-10-06.
- PyPI versions were read from the PyPI JSON API on 2026-10-06.

---

## 1. TL;DR

1. **Nothing released in 2025-2026 fits 6 GB at 1-3 s/asset with on-device LoRA training except the SD1.5 / SDXL
   UNet families.** The new generation (FLUX.2 [klein] 4B, Z-Image-Turbo, Qwen-Image-Lightning, Sana-Sprint) is far
   better at prompt following, but every one of them pairs a 2-6 B DiT with a 2-4 B LLM text encoder (Qwen3-4B,
   Gemma-2-2B), so the honest consumer floor is 12 GB for bf16/fp8 and ~8 GB only with 4-bit weights plus CPU offload,
   which pushes 1024px latency to 15-30 s on 8 GB cards [M].
2. **Best primary MVP model: SD1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) at 512px with a few-step LoRA
   (Hyper-SD15 8-step CFG-LoRA or TCD-SD15-LoRA) plus biome LoRAs**, ~1 s/asset on an RTX 3060 [E], 2-3 GB VRAM,
   circular-padding seamlessness works natively (convolutional UNet), and the pixel-art / sprite-sheet ecosystem is
   the deepest. If the floor is 8 GB rather than 6 GB, **SDXL + SDXL-Lightning 4-step at 768-1024px** (openrail++)
   is the stronger sibling (~2-4 s on a 3060 [E]) with `nerijs/pixel-art-xl` (950 K downloads).
3. **Best quality tier: FLUX.2 [klein] 4B** (`black-forest-labs/FLUX.2-klein-4B`, Apache-2.0, 4 steps, native
   `Flux2KleinPipeline` since diffusers 0.37, official `train_dreambooth_lora_flux2_klein.py`), 5.8 s/1024px on an
   RTX 3060 12 GB [M], ~2 s at 512px [E]. **Z-Image-Turbo** (`Tongyi-MAI/Z-Image-Turbo`, Apache-2.0, 8 NFE, 6 B) is the
   alternative when 16 GB is available and has the single most-downloaded pixel-art LoRA on the Hub
   (`tarn59/pixel_art_style_lora_z_image_turbo`, 937 K downloads).
4. **Tileability:** keep circular padding (UNet + VAE, asymmetric X/Y) for the UNet tier; for the DiT tier use the
   training-free *Tiled Diffusion* latent constraint (CVPR 2025) or seam inpainting. Measure with the Tiled-Diffusion
   **Tiling Score** (seam abs-diff), a seam-vs-interior gradient ratio, and the differentiable **TexTile** classifier
   (`pip install textile-metric`, CVPR 2024).
5. **Pins (2026-10-06):** torch 2.14.1+cu130, diffusers 0.41.0, transformers 5.18.0, peft 0.21.2, accelerate 1.15.0,
   huggingface-hub 1.33.0 (diffusers caps `<2.0`), torchmetrics 1.9.0, bitsandbytes 0.50.2 (Windows wheels), gguf 0.19.0.
   torchao and xformers ship no Windows wheels; use SDPA + optional triton-windows for `torch.compile`.

---

## 2. Hardware / latency framing

Target: RTX 3060 (12 GB; the 8 GB "3060 Ti/4060" class is the floor) and RTX 4070 (12 GB), Windows 11.

Reference throughputs used for all `[E]` estimates below:

| Measurement | Value | Source |
|---|---|---|
| SD1.5, 512x512, fp16, RTX 3060 12 GB | ~5 it/s (3.5-6 it/s range) [M] | [st-hakky GPU comparison](https://book.st-hakky.com/en/data-science/gpu-comparison-stable-diffusion), [grisk comments](https://grisk.itch.io/stable-diffusion-gui/comments?after=7) |
| SDXL base, 1024x1024, fp16, RTX 3060 12 GB | ~1.2-1.5 it/s; 30 steps = 11 s; ~8.5 GB VRAM [M] | [MSI SDXL benchmark](https://www.msi.com/blog/stable-diffusion-xl-best-value-rtx-graphics-card) |
| FLUX.2 [klein] 4B, 1024x1024, 4 steps | RTX 3060 12 GB ~5.8 s; RTX 4070 ~3.2 s; RTX 4070 Ti ~2.4 s; RTX 3090 ~2.1 s; RTX 4090 ~1.2 s [M] | [apatero klein consumer-GPU guide](https://apatero.com/blog/flux-2-klein-consumer-gpu-guide) |
| FLUX.2 [klein] 4B, 1024x1024, 4 steps, GGUF Q4_K_M + offload, RTX 3060 Ti 8 GB | 15-30 s [M] | [localaimaster: FLUX on 6-8 GB](https://localaimaster.com/blog/run-flux-on-low-vram-gpu) |
| FLUX.1-dev Q4_K_S, 20 steps, 1024px, RTX 3060 Ti 8 GB | 90-150 s [M] | same |
| Z-Image-Turbo FP8, 1024x1024, RTX 4060 8 GB | 15-20 s; SVDQ int4 on RTX 3080 mobile 8-12 s [M] | [zimage.run low-VRAM guide](https://zimage.run/blog/z-image-turbo-quantized-low-vram-guide) |
| Z-Image-Turbo bf16, 1024x1024, RTX 4090 | 2.3 s [C] | [ComfyUI-wiki release note](https://comfyui-wiki.com/en/news/2025-11-27-alibaba-z-image-turbo-release) |
| Qwen-Image-2512 Q4_K_M + Lightning 4-step, 1024px, RTX 4070 | ~8 s (needs ~14 GB) [M] | [dev.to Qwen-Image-2512 GGUF guide](https://dev.to/gary_yan_86eb77d35e0070f5/qwen-image-2512-gguf-complete-guide-to-running-ai-image-generation-on-consumer-hardware-1l6c) |
| SANA-Sprint 1.6B, 1024px, 1-2 steps | 0.1 s on H100, 0.31 s on RTX 4090 [C] | [SANA-Sprint paper / diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_sprint) |
| Nitro-E 304M distilled, 512px | 0.16 s on a Strix Halo iGPU [C] | [AMD-AGI/Nitro-E](https://github.com/AMD-AGI/Nitro-E) |

Implication: a 4-step SD1.5 pass at 512px costs ~0.8 s of UNet time on a 3060 plus ~0.2 s of text-encoder/VAE work
(`[E]` 1.0-1.2 s); a 4-step SDXL pass at 1024px costs ~3 s of UNet time plus ~0.5-1 s VAE decode (`[E]` 3.5-4 s;
~2 s at 768px because cost scales roughly with pixel count). FLUX.2 klein 4B at 512px should be ~3-4x faster than
its 1024px figure (`[E]` ~1.5-2 s on a 3060 12 GB) but only when the whole pipeline is resident in VRAM.

---

## 3. Family-by-family

### 3.1 SD1.5 + few-step adapters (the incumbent)

| Adapter | HF id | License | Steps / settings | Notes |
|---|---|---|---|---|
| Base | `stable-diffusion-v1-5/stable-diffusion-v1-5` (859.5 M params, 78.2 M downloads) | CreativeML OpenRAIL-M | - | [HF](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| LCM-LoRA | `latent-consistency/lcm-lora-sdv1-5` (6.1 M dl) | openrail++ | 4-8 steps, guidance 1-2, `LCMScheduler` | [HF](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5), [LCM-LoRA blog](https://huggingface.co/blog/lcm_lora) |
| Hyper-SD | `ByteDance/Hyper-SD` (4.6 M dl): `Hyper-SD15-1step-lora` (unified 1-8 steps, TCDScheduler, eta 1.0), `-2steps-lora`, `-4steps-lora` (guidance 0, DDIM trailing), `-8steps-CFG-lora` / `-12steps-CFG-lora` (guidance 5-8 preserved) | SD1.5/SDXL files: ByteDance OpenRAIL-style license dated 2024-04-11 with Attachment-A use restrictions (commercial use allowed); FLUX files: FLUX.1-dev non-commercial | see left | [HF card](https://huggingface.co/ByteDance/Hyper-SD), [LICENSE.md](https://huggingface.co/ByteDance/Hyper-SD/blob/main/LICENSE.md). The authors "strongly recommend" the 8-step CFG LoRA as the default for SD1.5/SDXL. |
| TCD | `h1t/TCD-SD15-LoRA` (33 K dl) | MIT | 4 steps, guidance 0, `TCDScheduler`, eta 0.3 | [HF](https://huggingface.co/h1t/TCD-SD15-LoRA) |
| DMD2 | `tianweiy/DMD2` (1.1 M dl; SD1.5 + SDXL 1/4-step UNet and LoRA) | **CC-BY-NC-4.0 (non-commercial)** | 4 steps, guidance 0, `LCMScheduler` with timesteps `[999,749,499,249]` | [HF](https://huggingface.co/tianweiy/DMD2) |
| SD-Turbo | `stabilityai/sd-turbo` (865.9 M params, 21.6 M dl) | Stability non-commercial research / membership (same family as SDXL-Turbo) | 1-4 steps, guidance 0, 512px | [HF](https://huggingface.co/stabilityai/sd-turbo) |

VRAM: fp16 weights are ~2.1 GB (UNet 1.7 GB + CLIP-L 0.25 GB + VAE 0.17 GB); a 512px run stays under ~3 GB `[E]`.
Latency on RTX 3060: `[E]` 1.0-1.2 s for 4 steps at 512px (section 2). LoRA training with the diffusers
`train_dreambooth_lora.py` script fits in 6-8 GB with 8-bit Adam and gradient checkpointing (standard practice since
2023; the diffusers DreamBooth README documents these flags:
[examples/dreambooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)).

### 3.2 SDXL + few-step adapters

| Adapter | HF id | License | Steps / settings | Notes |
|---|---|---|---|---|
| Base | `stabilityai/stable-diffusion-xl-base-1.0` (2.57 B, 126 M dl) | openrail++ | - | [HF](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| SDXL-Turbo | `stabilityai/sdxl-turbo` (2.57 B, 32.7 M dl) | `sai-nc-community` (commercial via Stability membership) | 1-4 steps, guidance 0, 512px | [HF](https://huggingface.co/stabilityai/sdxl-turbo) |
| SDXL-Lightning | `ByteDance/SDXL-Lightning` (5.6 M dl): `sdxl_lightning_{1,2,4,8}step_unet.safetensors` (best) or `_lora.safetensors` (for other SDXL finetunes) | openrail++ | match step count, `EulerDiscreteScheduler(timestep_spacing="trailing")`, guidance 0, 1024px | [HF](https://huggingface.co/ByteDance/SDXL-Lightning) |
| LCM-LoRA | `latent-consistency/lcm-lora-sdxl` | openrail++ | 4-8 steps, guidance 1-2 | [blog](https://huggingface.co/blog/lcm_lora); `nerijs/pixel-art-xl` recommends 8 steps, guidance 1.5, LoRA weight 1.2 with it |
| Hyper-SDXL | `ByteDance/Hyper-SD`: `Hyper-SDXL-1step-Unet`, `-1step-lora` (unified), `-2/4steps-lora`, `-8/12steps-CFG-lora` | ByteDance OpenRAIL-style (see 3.1) | as SD1.5 | [HF](https://huggingface.co/ByteDance/Hyper-SD) |
| DMD2-SDXL | `tianweiy/DMD2` `dmd2_sdxl_4step_lora_fp16.safetensors`, `dmd2_sdxl_1step_unet_fp16.bin` | **CC-BY-NC-4.0** | 4 steps, guidance 0 | [HF](https://huggingface.co/tianweiy/DMD2) |
| TCD-SDXL | `h1t/TCD-SDXL-LoRA` | MIT | 4-8 steps, eta 0.3 | [project page](https://mhh0318.github.io/tcd/) |

VRAM: ~7 GB of fp16 weights (UNet 5.1 GB + two text encoders 1.5 GB + VAE); ~8.5 GB at 1024px in a WebUI [M] (MSI).
On an 8 GB card use `enable_model_cpu_offload()` or an fp8/NF4 UNet; on 6 GB it needs sequential offload and stops
being "fast". Latency on RTX 3060: `[E]` 3.5-4 s per 4-step 1024px image, ~2 s at 768px. SDXL LoRA training needs
roughly 12 GB with the stock diffusers script (8 GB is possible only with aggressive tricks), so biome-LoRA training
would move to a 12 GB+ machine or the cloud.

### 3.3 Stable Diffusion 3.5

| Model | HF id | Params | License | Steps | VRAM / notes |
|---|---|---|---|---|---|
| SD3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` (gated; 5.0 M dl) | 2.5 B MMDiT-X + CLIP-L/G + T5-XXL | Stability Community License: free for < $1 M annual revenue, enterprise above | 40 steps, guidance 4.5 (no official turbo; `ByteDance/Hyper-SD` ships Hyper-SD3 CFG LoRAs for SD3-Medium, not 3.5) | T5-XXL is 9.5 GB in bf16; drop it or NF4 it to stay near 6-7 GB. [HF](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| SD3.5 Large Turbo | `stabilityai/stable-diffusion-3.5-large-turbo` (gated; 739 K dl) | 8.15 B | Stability Community License | 4 steps, guidance 0 (HF blog example uses `guidance_scale=1.0`) | 16 GB transformer in bf16; needs NF4 (bitsandbytes, natively supported) to sit near 12 GB; not an 8 GB option. [HF](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo), [HF blog](https://huggingface.co/blog/sd3-5) |

Verdict: superseded by FLUX.2 klein / Z-Image for the same VRAM; no reason to pick SD3.5 for this project.

### 3.4 FLUX.1 and FLUX.2

| Model | HF id | Params | License | Steps | VRAM / latency |
|---|---|---|---|---|---|
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` (gated, 23.2 M dl) | 11.9 B + T5-XXL + CLIP-L | Apache-2.0 | 4 steps, guidance 0 | bf16 transformer 24 GB; GGUF `city96/FLUX.1-schnell-gguf` (Apache-2.0): Q4_K_S ~6.8 GB, Q3_K_S ~5.2 GB, Q2_K ~4.0 GB; always pair with a GGUF T5 (fp16 T5 is ~9 GB alone) [M]; 4 steps on an 8 GB card `[E]` 20-30 s from the dev 20-step figure (90-150 s). Nunchaku SVDQuant int4 runs FLUX.1 in "just 4 GiB" with offload and is 3.0x faster than NF4 [C]. [schnell HF](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [localaimaster](https://localaimaster.com/blog/run-flux-on-low-vram-gpu), [Nunchaku](https://github.com/nunchaku-tech/nunchaku) |
| FLUX.1-dev distillations | `ByteDance/Hyper-SD` `Hyper-FLUX.1-dev-8steps-lora` (LoRA scale 0.125, guidance 3.5); `alimama-creative/FLUX.1-Turbo-Alpha` (8 steps, 808 K dl); `Freepik/flux.1-lite-8B` (8.16 B pruned dev) | - | **all inherit the FLUX.1-dev Non-Commercial License** | 8 steps | Excluded for a commercial product. [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD), [Turbo-Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha), [flux.1-lite](https://huggingface.co/Freepik/flux.1-lite-8B) |
| `lllyasviel/flux1-dev-bnb-nf4` | NF4 FLUX.1-dev | - | FLUX.1-dev NC | - | non-commercial; listed for completeness |
| FLUX.2 [dev] | `black-forest-labs/FLUX.2-dev` (2025-11-25) | 32 B, Mistral-Small-3.2-24B text encoder | FLUX Non-Commercial | 50 | "H100-equivalent" [C]; irrelevant here. [flux2 repo](https://github.com/black-forest-labs/flux2) |
| **FLUX.2 [klein] 4B** (2026-01-15) | `black-forest-labs/FLUX.2-klein-4B` (distilled, 3.88 B, 3.1 M dl) and `black-forest-labs/FLUX.2-klein-base-4B` (undistilled) | 3.88 B rectified-flow transformer; text encoder **Qwen3-4B** (`Qwen3ForCausalLM`, hidden 2560, 36 layers, verified from `text_encoder/config.json`); FLUX.2 VAE with 32 latent channels (`AutoencoderKLFlux2`) | **Apache-2.0** | distilled: 4 steps, `guidance_scale=1.0` (ignored); base: ~50 steps, guidance 4.0 | "~13 GB VRAM (RTX 3090/4070 and above)" [C]; FP8 ~4 GB transformer, GGUF Q4_K_M ~2.6 GB [M]; FP8 "up to 1.6x faster", NVFP4 "up to 2.7x faster" and "55 % less VRAM" [C]. RTX 3060 12 GB 5.8 s, RTX 4070 3.2 s at 1024px [M]; 15-30 s on 8 GB with Q4 + offload [M]. `Flux2KleinPipeline` since **diffusers 0.37.0** (model_index says `0.37.0.dev0`). [HF](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [BFL blog](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence), [diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2) |
| FLUX.2 [klein] 9B | `black-forest-labs/FLUX.2-klein-9B` (9.08 B, gated), `-base-9B`, `-9b-fp8` | 9.08 B | **FLUX Non-Commercial License** | 4 steps | ~29 GB bf16 [C]; excluded (license + VRAM). [HF](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |

LoRA training on klein: the official HF guide trains on **klein-base-4B** with ostris/ai-toolkit (~1,800 steps, < 24 GB,
~1 h on an RTX 4090, 15-40 images for a style LoRA) and states the adapter "still loads on the distilled model
afterward" and usually looks better there ([HF blog](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)).
Community minimum is 12 GB (RTX 3060 12 GB / 4060 Ti 16 GB) with 32 GB system RAM
([runcomfy 16 GB guide](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-klein-16gb-vram-training),
[BFL training docs](https://docs.bfl.ai/flux_2/flux2_klein_training)). diffusers ships
`examples/dreambooth/train_dreambooth_lora_flux2_klein.py`
([listing](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)). Caveat for pixel art: the
`svntax-dev` sprite-sheet LoRA author reports the distilled 4B "ruins the quality of the pixels and the consistency"
and recommends inference on the base model ([card](https://huggingface.co/svntax-dev/pixel_spritesheet_4walk_small_lora_v1)),
whereas `Limbicnation/pixel-art-lora` was trained for and runs on the distilled model at 4 steps
([card](https://huggingface.co/Limbicnation/pixel-art-lora)). Treat "distilled + pixel LoRA" as something to A/B test.

### 3.5 Families released 2025-2026

| Model | HF id (params) | Released | License (commercial?) | Steps | VRAM bf16 / fp8 / 4-bit | diffusers native? | PEFT LoRA training? |
|---|---|---|---|---|---|---|---|
| **Z-Image-Turbo** (Alibaba Tongyi) | `Tongyi-MAI/Z-Image-Turbo` (6.15 B; 9.1 M dl, 5.4 K likes) | 2025-11-26 | Apache-2.0 (yes) | 8 NFE (card example `num_inference_steps=9`, `guidance_scale=0.0`; DiffSynth: steps 8, cfg 1) | S3-DiT 6 B + **Qwen3-4B** text encoder + FLUX.1 16-ch VAE (verified from configs). "fits within 16 GB" [C]; FP8 ~6 GB transformer, SVDQ int4 4-5 GB [M]; DiffSynth: "minimum 8 GB" with CPU offload [C]; FP8 "causes noticeable quality degradation" per DiffSynth [C] | `ZImagePipeline` since diffusers 0.36 | Yes: `train_dreambooth_lora_z_image.py`; ai-toolkit (12 GB reported workable, [issue #550](https://github.com/ostris/ai-toolkit/issues/550)); DiffSynth-Studio. Turbo-trained LoRAs lose the 8-step ability unless you train with `ostris/zimage_turbo_training_adapter` (737 K dl) or load `DiffSynth-Studio/Z-Image-Turbo-DistillPatch` at inference ([card](https://huggingface.co/DiffSynth-Studio/Z-Image-Turbo-DistillPatch), [strategies blog](https://huggingface.co/blog/kelseye/training-strategies-of-z-image-turbo)) |
| Z-Image (base) | `Tongyi-MAI/Z-Image` (6.15 B) | 2026-01-27 | Apache-2.0 | 28-50, guidance 3-5 | as above | yes | yes ([repo](https://github.com/Tongyi-MAI/Z-Image)) |
| Qwen-Image + Lightning | `Qwen/Qwen-Image` (20.4 B), `lightx2v/Qwen-Image-Lightning` (4/8-step LoRAs, 7.4 M dl); `Qwen/Qwen-Image-2512` + `lightx2v/Qwen-Image-2512-Lightning` | 2025-08 / 2025-12 | Apache-2.0 (yes) | 4 or 8 steps, `true_cfg_scale=1.0`, shift log(3) | 40 GB bf16; GGUF Q4_K_M 13.1 GB (~14 GB), Q2_K ~8 GB [M]; Qwen2.5-VL-7B text encoder | `QwenImagePipeline` | yes (`train_dreambooth_lora_qwen_image.py`) - but no consumer-8 GB path |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` (7.1 B visual component, RGBA output) | 2026-09-30 | **license: other** (check before use) | - | new; `QwenImage21Pipeline` in diffusers 0.41.0 | yes | `train_dreambooth_lora_qwenimage21.py` exists |
| SANA 1.5 / SANA-Sprint (NVIDIA) | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers`; `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`, `Sana_Sprint_0.6B_1024px_diffusers` | 2025-01 / 2025-03 | Apache-2.0 (yes) | Sprint: 1-4 steps (default 2), guidance 4.5 embedded, `max_timesteps` tunable | 1.6 B DiT (3.2 GB) + Gemma-2-2B text encoder (5.2 GB) + DC-AE f32 VAE: `[E]` ~9 GB bf16, ~5 GB with 8-bit text encoder + transformer (docs show the bitsandbytes recipe) | `SanaSprintPipeline`, `SanaSprintImg2ImgPipeline` | `train_dreambooth_lora_sana.py` targets SANA; LoRA on the sCM-distilled Sprint is untested here. No game-asset LoRAs exist. DC-AE's 32x compression is a risk for crisp pixel edges (untested). |
| HiDream-I1-Fast | `HiDream-ai/HiDream-I1-Fast` (17.1 B MoE; 1.0 M dl) | 2025-04 | MIT (yes) | 16 steps | needs Llama-3.1-8B-Instruct as text encoder; > 30 GB bf16, ~16 GB NF4 [E] | `HiDreamImagePipeline` | `train_dreambooth_lora_hidream.py` | too heavy |
| Lumina-Image 2.0 | `Alpha-VLLM/Lumina-Image-2.0` (2.61 B) | 2025-01 | Apache-2.0 | 30-50 (no few-step variant) | + Gemma-2-2B; ~10 GB bf16 [E] | `Lumina2Pipeline` | `train_dreambooth_lora_lumina2.py` | not few-step |
| PixArt-Sigma | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (0.61 B); LCM only for alpha: `PixArt-alpha/PixArt-LCM-XL-2-1024-MS` | 2024 | openrail++ | 20 (Sigma) / 4 (alpha-LCM) | + T5-XXL 4.7 B (9.5 GB bf16; 8-bit ~5 GB) | yes | community scripts | T5 dominates VRAM |
| Kolors | `Kwai-Kolors/Kolors-diffusers` | 2024-07 | Apache-2.0 | 25-50 | SDXL-size UNet + ChatGLM3-6B encoder (~12 GB bf16) | `KolorsPipeline` | yes (SDXL-style) | not few-step |
| CogView4-6B | `zai-org/CogView4-6B` (6.37 B) | 2025-03 | Apache-2.0 | 50 | + GLM-4-9B encoder (> 20 GB) | `CogView4Pipeline` | community | too heavy |
| Chroma1-HD / Chroma1-Flash | `lodestones/Chroma1-HD` (8.9 B), `lodestones/Chroma1-Flash` (8.9 B few-step; card has no settings) | 2025-08/10 | Apache-2.0 | HD ~20+; Flash few-step | FLUX-class 8.9 B (~18 GB bf16) | `ChromaPipeline` | ai-toolkit | too heavy for 12 GB without 4-bit |
| LongCat-Image | `meituan-longcat/LongCat-Image` (6 B) | 2025-12 | Apache-2.0 | ~50 | 6 B + VLM encoder | via diffusers main | - | not few-step |
| Ovis-Image-7B | `ATH-MaaS/Ovis-Image-7B` (7.37 B) | 2025-11 | Apache-2.0 | 50 | 7 B | `OvisImagePipeline` | - | text-rendering focus |
| Nitro-E (AMD) | `amd/Nitro-E` (304 M; `Nitro-E-512px`, `-512px-dist`, `-512px-GRPO`) | 2025-11 | MIT (yes) | 20 / 4 (dist) | tiny (`[E]` ~2 GB); 512px only | **no** - custom `core.tools.inference_pipe.init_pipe` from the GitHub repo (pinned to diffusers 0.32.2) | no documented LoRA path | fastest option by far but tooling is bespoke |
| GLM-Image | `zai-org/GLM-Image` (6.93 B) | 2026-01-15 | MIT | - | 7 B | `GlmImagePipeline` (0.37) | - | not few-step |
| ERNIE-Image-Turbo (Baidu) | `baidu/ERNIE-Image-Turbo` (8.03 B); GGUF `unsloth/ERNIE-Image-Turbo-GGUF` (284 K dl) | 2026-04 | Apache-2.0 | 8 steps, guidance 1.0 (DMD + RL distilled) | "24 GB" bf16 [C]; GGUF/NVFP4 community builds | `ErnieImagePipeline` (0.38) | not documented | worth watching; too big for 12 GB without 4-bit |
| Krea-2-Turbo | `krea/Krea-2-Turbo` (12.8 B, gated) | 2026-06-22 | Krea 2 Community License (commercial allowed with AUP + content filtering) | 8 steps, guidance 0, mu 1.15 | 12.8 B | `Krea2Pipeline` (0.39), single-file in 0.41 | LoRA trainer space exists | too big |
| Ideogram 4 | `ideogram-ai/ideogram-4-fp8`, `-nf4` (9.28 B, gated) | 2026-06 | license: other | - | 9.3 B | `Ideogram4Pipeline` (0.39) | `train_dreambooth_lora_ideogram4.py` | license unclear |
| Boogu-Image-0.1 | `Boogu/Boogu-Image-0.1-Turbo` (10.3 B; also `-Base`, `-Edit`, fp8 variants) | 2026-07-24 | Apache-2.0 | Turbo few-step | memory tiers "12 GB to 80 GB" [C] | `BooguImageTurboPipeline` | - | too big for 8 GB |
| Ming-Image-0.1-Design | `inclusionAI/Ming-Image-0.1-Design` (6.15 B) | 2026-09-23 | MIT | - | 6 B | custom | - | UI/infographic specialist |
| Anima (CircleStone x Comfy) | `circlestone-labs/Anima` (2 B on Cosmos-Predict2; Anima-Turbo 8-12 steps CFG 1) | 2026 | **circlestone-labs non-commercial** | 8-12 (Turbo) | 2 B + Qwen3-0.6B encoder | single-file/ComfyUI; diffusers 0.39 lists Anima | ComfyUI trainers | excluded (license) |
| DreamLite | `carlofkl/DreamLite-mobile` (390 M) | 2026-06 | **CC-BY-NC-4.0** | - | tiny | `DreamLiteMobilePipeline` | - | excluded (license) |
| DiffusionGemma 26B-A4B | `google/diffusiongemma-26B-A4B-it` | 2026-07 | Gemma terms | - | 26 B MoE | 0.39 | - | too big |

Sources: HF Hub API records for each id (linked above), [diffusers releases](https://github.com/huggingface/diffusers/releases),
[Z-Image repo](https://github.com/Tongyi-MAI/Z-Image), [DiffSynth Z-Image doc](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md),
[Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning), [SANA-Sprint docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_sprint),
[Nitro-E](https://huggingface.co/amd/Nitro-E), [ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo), [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo),
[Anima](https://huggingface.co/circlestone-labs/Anima), [BentoML 2026 guide](https://www.bentoml.com/blog/a-guide-to-open-source-image-generation-models).

---

## 4. Master comparison table (candidates that can plausibly run on 6-12 GB)

| Candidate | HF id | Params (denoiser + TE) | License (commercial) | Steps | VRAM bf16/fp16 | VRAM fp8 / NF4 / GGUF | RTX 3060 12 GB latency | RTX 4070 latency | Native diffusers | LoRA training on consumer GPU | Pixel/texture LoRA ecosystem | Seamless via circular padding |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SD1.5 + Hyper-SD15 / TCD | `stable-diffusion-v1-5/stable-diffusion-v1-5` + `ByteDance/Hyper-SD` / `h1t/TCD-SD15-LoRA` | 0.86 B + 0.12 B | OpenRAIL-M / ByteDance OpenRAIL / MIT (yes) | 2-8 | ~3 GB @512 | n/a (not needed) | `[E]` 1.0-1.2 s @512 (4 steps) | `[E]` 0.5-0.7 s | yes | yes, 6-8 GB | deep (sprite-sheet ckpt, PixelArtRedmond-1.5) | yes (UNet + VAE convs) |
| SDXL + Lightning | `stabilityai/stable-diffusion-xl-base-1.0` + `ByteDance/SDXL-Lightning` | 2.57 B + 0.8 B | openrail++ (yes) | 1-8 | ~7 GB weights, ~8.5 GB @1024 [M] | fp8 UNet ~2.6 GB | `[E]` 3.5-4 s @1024, ~2 s @768 | `[E]` 1.5-2 s @1024 | yes | ~12 GB | deepest for SDXL (`nerijs/pixel-art-xl` 950 K dl) | yes |
| FLUX.2 klein 4B | `black-forest-labs/FLUX.2-klein-4B` | 3.88 B + Qwen3-4B | Apache-2.0 (yes) | 4 | ~13 GB whole pipeline [C] | fp8 ~4 GB, Q4_K_M ~2.6 GB (transformer) [M]; TE needs 8-bit/4-bit or CPU | 5.8 s @1024 [M]; `[E]` 1.5-2 s @512; 15-30 s on 8 GB w/ offload [M] | 3.2 s @1024 [M] | yes (0.37+) | 12 GB min (ai-toolkit), official script | growing (pixel-art-lora, 3 sprite-sheet LoRAs) | VAE only; needs latent-constraint method |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | 6.15 B + Qwen3-4B | Apache-2.0 (yes) | 8 | ~16 GB [C] | fp8 ~6 GB, SVDQ int4 4-5 GB [M] | `[E]` 8-15 s @1024 (4060 8 GB FP8: 15-20 s [M]) | ~3-4 s @1024 (vendor-adjacent claim, low confidence) | yes (0.36+) | 12 GB with adapter/DistillPatch | strongest single pixel LoRA (937 K dl) | VAE only |
| SANA-Sprint 1.6B | `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` | 1.6 B + Gemma-2-2B | Apache-2.0 (yes) | 1-2 | `[E]` ~9 GB | 8-bit both: `[E]` ~5 GB | `[E]` ~1-1.5 s @1024 (0.31 s on 4090 [C]) | `[E]` ~0.7 s | yes | SANA script exists; Sprint untested | none | VAE only (DC-AE) |
| Nitro-E 512 dist | `amd/Nitro-E` | 0.30 B | MIT (yes) | 4 | `[E]` ~2 GB | n/a | `[E]` < 0.3 s | - | no (custom pipeline) | none documented | none | unknown |
| FLUX.1-schnell (GGUF/Nunchaku) | `black-forest-labs/FLUX.1-schnell` + `city96/FLUX.1-schnell-gguf` | 11.9 B + T5-XXL | Apache-2.0 (yes) | 4 | 24 GB+ | Q4_K_S 6.8 GB [M]; Nunchaku int4 ~4 GiB w/ offload [C] | `[E]` 20-30 s @1024 on 8 GB | - | yes (GGUF via `from_single_file`) | 24 GB class | large but mostly FLUX.1-dev (NC) LoRAs | VAE only |
| Qwen-Image(-2512) + Lightning | `Qwen/Qwen-Image-2512` + `lightx2v/Qwen-Image-2512-Lightning` | 20 B + Qwen2.5-VL-7B | Apache-2.0 (yes) | 4/8 | 40 GB | Q4_K_M ~14 GB [M] | n/a on 12 GB | ~8 s @1024 with Q4_K_M + 4-step (14 GB) [M] | yes | 24 GB class | pixel LoRA exists (`prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA`) | VAE only |

---

## 5. Game-asset LoRAs and checkpoints (HF ids, read 2026-10-06)

### 5.1 SD1.5 tier

| Asset | HF id | Downloads | License | Notes |
|---|---|---|---|---|
| Pixel-art sprite-sheet checkpoint (4 views) | `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` | 87.6 K | Apache-2.0 | full SD1.5 checkpoint; triggers `PixelartFSS` (front), `PixelartRSS` (right), `PixelartBSS` (back), `PixelartLSS` (left). [card](https://huggingface.co/Onodofthenorth/SD_PixelArt_SpriteSheet_Generator) |
| PixelArtRedmond 1.5 LoRA | `artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5` | 19.3 K | "other" (check) | [card](https://huggingface.co/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5) |
| pixel-art-style (full ckpt) | `kohbanye/pixel-art-style` | 19.8 K | no license tag | [card](https://huggingface.co/kohbanye/pixel-art-style) |
| diffusiondb-pixelart LoRA | `SedatAl/pixel-art-LoRa` | 249 | OpenRAIL-M | trained on `jainr3/diffusiondb-pixelart` (a useful public dataset for your own LoRAs) |
| Game-asset merge | `Yntec/GameAssetsDigitalUnitsCreationKit` | 219 | OpenRAIL-M | |

### 5.2 SDXL tier

| Asset | HF id | Downloads | License | Notes |
|---|---|---|---|---|
| **pixel-art-xl** | `nerijs/pixel-art-xl` | 950.7 K, 656 likes | CreativeML OpenRAIL-M | no trigger word; with LCM-LoRA use 8 steps, guidance 1.5, LoRA weight 1.2; "don't use refiner"; downscale 8x nearest-neighbour for pixel-perfect output. [card](https://huggingface.co/nerijs/pixel-art-xl) |
| PixelArt.Redmond (SDXL) | `artificialguybr/PixelArtRedmond` | 60.7 K | CreativeML OpenRAIL-M | trigger `Pixel Art, PixArFK`. [card](https://huggingface.co/artificialguybr/PixelArtRedmond) |
| pixel-art slider | `ntc-ai/SDXL-LoRA-slider.pixel-art` | 560 | MIT | strength slider |
| spritesheet slider | `ntc-ai/SDXL-LoRA-slider.spritesheet` | 68 | MIT | |
| Hand-painted 2D seamless textures | `Muapi/hand-painted-2d-seamless-textures` | 64 | openrail++ | CivitAI mirror; the only SDXL "seamless" LoRA found on HF besides `gl4dius4pple/seamlesstexture_LoRA` (9 dl) |
| Illustrious pixel-art checkpoint | `John6666/illustrious-pixel-art-from-hades-v4-series-v-40-sdxl` | 32 | (Illustrious terms) | tags "game assets" |

### 5.3 Z-Image-Turbo tier

| Asset | HF id | Downloads | License | Notes |
|---|---|---|---|---|
| **Pixel art style** | `tarn59/pixel_art_style_lora_z_image_turbo` | 937.7 K | Apache-2.0 | trigger `Pixel art style.`; 871 adapters now exist for Z-Image-Turbo. [card](https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo) |
| 3D MMORPG style | `DK9/3D_MMORPG_style_z-image-turbo_lora` | 426 | Apache-2.0 | props/UI look |
| Training adapter | `ostris/zimage_turbo_training_adapter` | 737.6 K | Apache-2.0 | required for training LoRAs that keep 8-step inference |
| DistillPatch | `DiffSynth-Studio/Z-Image-Turbo-DistillPatch` | - | Apache-2.0 | inference-time fix for LoRAs trained without the adapter |
| (none) | - | - | - | **no seamless-texture or tile LoRA for Z-Image was found on HF** (searches for "Z-Image texture/tile/isometric/game" returned nothing) |

### 5.4 FLUX.2 [klein] 4B tier (Apache-2.0 base)

| Asset | HF id | Downloads | License | Notes |
|---|---|---|---|---|
| **pixel-art-lora** | `Limbicnation/pixel-art-lora` | 21.1 K | Apache-2.0 | for the distilled 4B; prompt `pixel art sprite, <desc>, game asset, transparent background`; strength 0.85-1.4, 4 steps, guidance 1.0, 512x512; rank 64, 1,000 steps, 500 CC0 images; requires diffusers >= 0.37. [card](https://huggingface.co/Limbicnation/pixel-art-lora) |
| pixel-art (base-4B) | `adirik/pixel-art-lora-flux.2-klein-4B` | 949 | Apache-2.0 | trained on klein-base-4B |
| 4-direction walk sprite sheet (32x32) | `svntax-dev/pixel_spritesheet_4walk_small_lora_v1` | 4.0 K | Apache-2.0 | base-4B only ("distilled ruins the pixels"); 512x512, downscale 4x with k-centroid. [card](https://huggingface.co/svntax-dev/pixel_spritesheet_4walk_small_lora_v1) |
| 2x2 multi-view sprite sheet (I2I) | `fal/flux-2-klein-4b-spritesheet-lora` | - (106 likes) | Apache-2.0 | isometric/top-down/side views from a reference image. [card](https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora) |
| sprite-sheet mirror (I2I) | `xocialize/spritesheet-FLUX.2-klein-4B-lora` | 92 | Apache-2.0 | |
| combat sprite sheet 32x48 | `svntax-dev/pixel_spritesheet_4walk_combat_32x48_v1` | 465 | Apache-2.0 | base is `Qwen/Qwen-Image-Edit-2511` (20 B) |
| game-asset tiles (hex/iso/top-down) | `crafiq/flux-2-klein-9b-game-asset-tiles-lora` | 754 | Apache-2.0 LoRA on a **non-commercial 9B base** | |

### 5.5 FLUX.1-dev tier (base is non-commercial; listed for reference / dataset ideas)

`gokaygokay/Flux-Seamless-Texture-LoRA` (9.2 K dl, trigger `smlstxtr`, prompt format `smlstxtr, <prompt>, seamless texture`),
`gokaygokay/Flux-2D-Game-Assets-LoRA` (6.4 K), `gokaygokay/Flux-Game-Assets-LoRA-v2` (557, 129 likes),
`UmeAiRT/FLUX.1-dev-LoRA-Modern_Pixel_art` (MIT), `glif-loradex-trainer/AP123_flux_dev_2DHD_pixel_art`,
`Muapi/top-down-pixel-art-flux-lora`, `Muapi/pixel-game-assets-flux-by-dever`, `milliyin/pixel_art_characters_lora_flux_nf4` (CC-BY-NC).

### 5.6 Other

`prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA` (Apache-2.0, 1.2 K dl), `tarn59/hidream-pixel-art-lora`,
`PrunaAI/p-image-pixel-art-lora` (for Pruna's API-only "p-image"; base not on HF).

Takeaway: pixel-art adapters exist for every tier; **seamless-texture adapters exist only for SDXL (weak) and
FLUX.1-dev (non-commercial)**, so tileability must come from the sampling technique (section 6) plus your own biome
LoRAs, not from a downloaded LoRA.

---

## 6. Tileable / seamless texture techniques and how to measure them

### 6.1 Techniques

| Technique | What it does | Works on | Training-free | Source |
|---|---|---|---|---|
| **Circular padding ("seamless mode")** | switch every `nn.Conv2d` in the UNet and VAE to `padding_mode="circular"`; the asymmetric variant pads X and Y independently so you can tile only horizontally (side-scroller strips) or only vertically | any convolutional model: SD1.5, SDXL (UNet + VAE); for DiTs only the VAE decoder has convs, so it does **not** make FLUX.2/Z-Image/Sana latents periodic | yes | [A1111 asymmetric-tiling extension (tjm35)](https://github.com/tjm35/asymmetric-tiling-sd-webui/), [CompVis issue #250](https://github.com/CompVis/stable-diffusion/issues/250), ComfyUI `Seamless`/`SeamlessVae` nodes ([comfy_mtb model.py](https://huggingface.co/spaces/Dreamspire/flux-style-shaping/blob/main/custom_nodes/comfy_mtb/nodes/model.py)) |
| **Tiled Diffusion** (Madar & Fried, CVPR 2025) | at every denoising step, copies/averages latent regions across paired edges with a context window so tiles are consistent; supports self-tiling, one-to-one and many-to-many (Wang tiles) | SD1.5, SD2.0, SDXL, SD3 (an MMDiT, so the approach ports to FLUX.2 klein / Z-Image via `callback_on_step_end`), ControlNet | yes | [paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Madar_Tiled_Diffusion_CVPR_2025_paper.pdf), [arXiv 2412.15185](https://arxiv.org/html/2412.15185v1), [code](https://github.com/madaror/tiled-diffusion) |
| **ControlTile** (Huang et al., June 2026) | applies 2D affine transforms to RoPE relative positions in a DiT plus a disjoint attention mask to control the tiling pattern | DiT models; ships a LoRA checkpoint; two-phase training on a Blender dataset | **no** (needs training) | [arXiv 2606.22945](https://arxiv.org/abs/2606.22945), [code](https://github.com/junrongh/ControlTile) |
| Seamless Tile Inpainting | generate, roll by half, inpaint the cross-shaped seam | any inpainting-capable model | yes | [width.ai intro](https://width.ai/post/introduction-to-seamless-tile-inpainting); evaluated as "STI" in Tiled Diffusion (worse FID than circular padding) |
| TileGen (SIGGRAPH Asia 2022) | StyleGAN variant with circular padding for always-tileable SVBRDF materials | GAN, material-specific | n/a | [project](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2022_TileGen/index.html), [code](https://github.com/xilongzhou/TileGen) |
| Prompt/LoRA-only ("seamless texture", `smlstxtr`) | relies on training data; no guarantee | FLUX.1-dev LoRA | yes | [gokaygokay LoRA](https://huggingface.co/gokaygokay/Flux-Seamless-Texture-LoRA) |

Tiled Diffusion's own numbers (1,000 LAION captions, SD1.5): self-tiling Tiling Score 0.03 for both Tiled Diffusion
(FID 47.9) and plain circular padding "AT" (FID 49.2) versus 0.28-0.31 without a tiling constraint; seam inpainting
reached 0.03 only at FID 59.2 ([arXiv](https://arxiv.org/html/2412.15185v1)). So for a UNet model circular padding is
essentially as good as the fancier method and costs nothing.

Minimal diffusers implementation of asymmetric circular padding (UNet + VAE):

```python
import torch.nn as nn
import torch.nn.functional as F

def make_seamless(module: nn.Module, x: bool = True, y: bool = True):
    """Patch every Conv2d so padding wraps around on the chosen axes."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and max(m.padding) > 0:
            px, py = m.padding[1], m.padding[0]
            m.padding = (0, 0)           # we pad manually
            def fwd(inp, _m=m, _px=px, _py=py):
                inp = F.pad(inp, (_px, _px, 0, 0), mode="circular" if x else "constant")
                inp = F.pad(inp, (0, 0, _py, _py), mode="circular" if y else "constant")
                return F.conv2d(inp, _m.weight, _m.bias, _m.stride, 0, _m.dilation, _m.groups)
            m.forward = fwd

make_seamless(pipe.unet); make_seamless(pipe.vae)   # SD1.5 / SDXL
```

For the DiT tier (klein / Z-Image) patch only the VAE this way and add a Tiled-Diffusion-style step callback that
rolls the latent by a random offset before each step and rolls it back after (or copies boundary bands), then verify
with the metrics below.

### 6.2 Measuring tileability

| Metric | How | Library | Notes |
|---|---|---|---|
| **Tiling Score (TS)** | mean absolute pixel difference across the tile boundary at the junction and at two offsets, after tiling the image 2x2; 0 = perfect | numpy / torch (10 lines) | defined in Tiled Diffusion; lower is better; 0.03 is "seamless" on SD1.5 |
| Seam-vs-interior gradient ratio | `torch.roll(img, (H/2, W/2))`, compute Sobel/finite-difference energy in a 2-4 px band around the new centre cross, divide by the mean gradient energy of the interior; ratio ~1 = no seam | torch | my proposal (not a published metric); scale-free and cheap enough for CI |
| LPIPS on seam crops | LPIPS between a seam-centred crop of the rolled image and a same-size interior crop | `lpips` 0.1.4 | perceptual version of the above |
| FFT periodicity | magnitude spectrum of the 2x2-tiled image; a visible seam adds energy at harmonics of the tile frequency compared with the untiled spectrum | torch.fft | diagnostic only; no published threshold |
| **TexTile** | differentiable binary classifier trained on human tileability annotations; usable as a loss | `pip install textile-metric` (0.0.11, 2026-09-30, MIT; weights auto-download from HF) | [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Rodriguez-Pardo_TexTile_A_Differentiable_Metric_for_Texture_Tileability_CVPR_2024_paper.pdf), [repo](https://github.com/crp94/textile) |

Recommendation: report TS and the gradient ratio for every generated tile (hard gate), and TexTile as a soft score.

---

## 7. Quality metrics: what is current and pip-installable on Windows (2026-10-06)

| Metric | Library / model | Version (date) | Windows | Notes |
|---|---|---|---|---|
| FID / KID / IS | `torchmetrics[image]` (uses `torch-fidelity`) | torchmetrics 1.9.0 (2026-03-09); torch-fidelity 0.4.0 (2026-02-17) | pure Python, yes | `torchmetrics.image.fid.FrechetInceptionDistance`; for small asset sets (< 1k) prefer KID or report FID with the caveat that it is biased at low N |
| FID (clean) | `clean-fid` | 0.1.35 (2022-12-18) | yes | still the reference implementation for resize-consistent FID; unmaintained but works with torch 2.x |
| FID (reference) | `pytorch-fid` | 0.3.0 (2023-01-05) | yes | |
| CLIP score | `torchmetrics.multimodal.CLIPScore` (needs `transformers>=4.43`) or `open-clip-torch` | open-clip-torch 3.3.0 (2026-02-27) | yes | |
| Style consistency | cosine between CLIP image embeddings (`openai/clip-vit-large-patch14` via transformers) and/or DINO embeddings | - | yes | compute mean pairwise cosine inside a biome vs. across biomes |
| DINOv2 similarity | `facebook/dinov2-base` (Apache-2.0) via transformers | - | yes | safest license |
| DINOv3 similarity | `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated; transformers >= 4.56) | - | yes | custom **DINOv3 License**: commercial use allowed, requires "Built with DINOv3" attribution, bans military/ITAR uses ([license](https://ai.meta.com/resources/models-and-libraries/dinov3-license), [card](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)) |
| PickScore | `yuvalkirstain/PickScore_v1` (CLIP-H, 986 M, 16 M dl) via transformers | - | yes | simplest preference model to run; also packaged in `imscore` |
| HPSv2 | `imscore` 0.0.11 (2025-08-19; transformers >= 4.53) rather than `hpsv2` 1.2.0 (2024; pins `protobuf<4`, conflicts with the modern stack) | - | yes | [imscore](https://github.com/RE-N-Y/imscore) |
| ImageReward | `image-reward` 1.5 (2023-07-11; pins `timm==0.6.13`, `fairscale`) | - | yes, **in a separate venv** | [repo](https://github.com/AI-Machine-Vision-Lab/ImageReward-Evaluating-Human-Preferences-for-Text-to-Image-Generation) |
| HPSv3 | `hpsv3` 1.0.0 (2025-08-06; pins `transformers==4.45.2`; Qwen2-VL-7B backbone) | - | separate venv; 7B VLM is heavy for an 8 GB GPU | [repo](https://github.com/MizzenAI/HPSv3), [paper](https://arxiv.org/html/2508.03789v1) |
| LPIPS | `lpips` 0.1.4 (2021) | - | yes | seam metric, diversity |

Sources: PyPI JSON for each package (read 2026-10-06), [torchmetrics releases](https://github.com/Lightning-AI/torchmetrics/releases),
[HPSv3 PyPI](https://pypi.org/project/hpsv3/), [image-evaluator/imscore search](https://github.com/RE-N-Y/imscore).

Practical note: HPSv3 and ImageReward each pin an old `transformers`; keep them in `envs/eval-legacy` and run them as a
batch job over saved PNGs, never in the generation environment.

---

## 8. Inference acceleration on Windows (status 2026-10-06)

| Technique / lib | Current? | Windows (NVIDIA) | Verdict for this project |
|---|---|---|---|
| **SDPA** (`torch.nn.functional.scaled_dot_product_attention`) | default in diffusers | yes | always on; replaces xformers |
| **torch.compile** | torch 2.14.1 | needs **triton-windows** (3.8.0.post29, 2026-09-28; wheels bundle a minimal CUDA toolchain + TinyCC; RTX 30/40/50 supported, Turing only on <= 3.2) | worth ~1.3-2x on the UNet after warm-up; compile once per resolution; [triton-windows](https://github.com/woct0rdho/triton-windows) |
| xformers | 0.0.35 (2026-02-20, "rely on upstream FA3", torch >= 2.10) | **no Windows wheels** on PyPI or GitHub release | skip |
| bitsandbytes NF4 / int8 | 0.50.2 (2026-08-27) | **yes** (official CUDA + ROCm Windows builds) | use for Qwen3-4B / T5 text encoders and the klein/Z-Image transformer on 8 GB cards; diffusers `PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", ...)` |
| torchao int8/fp8 | 0.18.0 (2026-08-03) | **no win_amd64 wheels** (Linux only) | skip on Windows; use bnb or fp8 safetensors + `layerwise casting` |
| GGUF via diffusers | diffusers `from_single_file` + `gguf` 0.19.0 (2026-05-06); optional 10 % faster CUDA kernels via `kernels` 0.17.2 with `DIFFUSERS_GGUF_CUDA_KERNELS=true` | yes | best 8 GB path for klein-4B / Z-Image transformers (Q8_0 or Q5_K); model classes only, not pipelines ([docs](https://huggingface.co/docs/diffusers/main/en/quantization/gguf)) |
| Nunchaku SVDQuant (4-bit W4A4) | v1.2.1 (2026-01-25); supports FLUX.1, FLUX.2 + klein, Z-Image(-Turbo), Qwen-Image(+Lightning), SANA 1.6B, PixArt-Sigma; INT4 on Turing+, NVFP4 on RTX 50 | **yes**: `nunchaku-1.2.1+cu128/cu130torch2.8-2.11-cp310..313-win_amd64.whl` on GitHub releases; **no wheel for torch 2.14 yet** | the fastest 4-bit path (3x vs NF4, 3.6x memory cut) but it forces torch <= 2.11; keep as an optional "turbo" env. Diffusers 0.40 added a "Nunchaku Lite" loader for prequantized checkpoints (needs `kernels`) ([repo](https://github.com/nunchaku-tech/nunchaku), [release](https://github.com/nunchaku-tech/nunchaku/releases)) |
| DeepCache | 0.1.1 (2023-12-30) | yes | UNet-only (SD1.5/SDXL); still works but unmaintained; with 4-step samplers there is little to cache - skip |
| TeaCache | GitHub only (ali-vilab); not on PyPI | - | superseded inside diffusers by native caching |
| diffusers native caching: **FirstBlockCache**, MagCache, TaylorSeer, PAB, FasterCache, SeaCache | diffusers 0.37-0.41 (`pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold=0.2))`) | yes | useful only for the 8-step Z-Image / 50-step base models; at 4 steps caching mostly hurts quality ([docs](https://huggingface.co/docs/diffusers/main/en/optimization/cache)) |
| ParaAttention / FBCache original | para-attn 0.3.38 (2025-07-05), sdist only | no wheels | use diffusers' FirstBlockCache instead |
| stable-fast | 1.0.5 (2024-05-10) | - | abandoned; skip |
| onnxruntime-directml | 1.24.4 (2026-03-17) | yes | diffusers **deprecated its ONNX pipelines in 0.41.0** in favour of Optimum (2.3.0); only relevant for non-NVIDIA GPUs |
| TensorRT | torch-tensorrt 2.14.0 (2026-09-23, Windows wheels, requires torch 2.14.x); `tensorrt-rtx` metapackage 1.6.1 (2026-07-28, no Windows wheel on PyPI - NVIDIA ships it separately); NVIDIA Model-Optimizer diffusers examples | partial | production-only optimisation for a frozen SD1.5/SDXL graph; not for the MVP |
| SageAttention | 1.0.6 (2024-11-20), no Windows wheels on PyPI (community builds exist) | - | skip |
| SDNQ backend | `sdnq` 0.2.7 (2026-09-24), diffusers 0.40+ | pure PyTorch | alternative to bnb for on-the-fly int8/int4; untested here |

Sources: PyPI JSON (read 2026-10-06), [diffusers quantization overview](https://huggingface.co/docs/diffusers/main/en/quantization/overview),
[diffusers v0.41.0 notes](https://github.com/huggingface/diffusers/releases), [Nunchaku release](https://github.com/nunchaku-tech/nunchaku/releases),
[xformers release](https://github.com/facebookresearch/xformers/releases), [onnxruntime-directml](https://pypi.org/project/onnxruntime-directml/),
[TensorRT-RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html).

---

## 9. Exact package versions to pin (PyPI, 2026-10-06)

```text
# --- core (Windows x64, Python 3.12 recommended; wheels exist for cp310-cp315) ---
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.14.1+cu130            # 2026-09-30; win_amd64 wheels on the cu130 index (PyPI torch is CPU-only on Windows)
torchvision==0.29.1+cu130      # paired release on the same index
diffusers==0.41.0              # 2026-10-06 (released today; 0.40.0 of 2026-08-20 is the fallback)
transformers==5.18.0           # 2026-09-30; Qwen3 / Gemma-2 / DINOv3 all supported
huggingface-hub==1.33.0        # 2026-09-24; diffusers 0.41.0 requires >=1.32,<2.0 (2.1.1 is NOT allowed)
safetensors==0.8.0             # 2026-06-09; required >=0.8.0 by both diffusers and transformers
accelerate==1.15.0             # 2026-09-09
peft==0.21.2                   # 2026-10-01
sentencepiece==0.2.2
protobuf==7.36.2
bitsandbytes==0.50.2           # 2026-08-27; official Windows CUDA wheels
gguf==0.19.0                   # 2026-05-06
kernels==0.17.2                # optional GGUF CUDA kernels
triton-windows==3.8.0.post29   # 2026-09-28; optional, enables torch.compile on Windows (check README table for torch 2.14 pairing)
# --- metrics ---
torchmetrics==1.9.0            # 2026-03-09
torch-fidelity==0.4.0          # 2026-02-17
clean-fid==0.1.35
lpips==0.1.4
open-clip-torch==3.3.0
imscore==0.0.11                # PickScore / HPSv2 / MPS on transformers>=4.53
textile-metric==0.0.11         # 2026-09-30
# --- legacy eval venv (separate) ---
hpsv3==1.0.0 (transformers==4.45.2), image-reward==1.5 (timm==0.6.13)
# --- optional turbo venv ---
nunchaku 1.2.1 (GitHub wheel, needs torch<=2.11 + cu128/cu130), torch-tensorrt==2.14.0
```

Compatibility facts behind these pins: diffusers 0.41.0 declares `huggingface-hub<2.0,>=1.32.0`, `safetensors>=0.8.0`,
`torch>=2.6`, `peft>=0.17.0`, `accelerate>=0.31.0`, Python >= 3.10 ([PyPI](https://pypi.org/pypi/diffusers/0.41.0/json));
transformers 5.18.0 declares `huggingface-hub<3.0,>=1.31.0`, `tokenizers<0.24,>=0.23.1`, `torch>=2.5`
([PyPI](https://pypi.org/pypi/transformers/5.18.0/json)); PyTorch 2.14.1 was published 2026-09-30
([release](https://github.com/pytorch/pytorch/releases)) and the cu130 index carries `torch-2.14.1+cu130-cp31x-win_amd64.whl`
([index](https://download.pytorch.org/whl/cu130/torch/)).

---

## 10. FINAL RECOMMENDATION

**(a) Primary MVP model (6-8 GB Windows GPU, ~1-3 s/asset, biome LoRAs).**
Keep **SD1.5** (`stable-diffusion-v1-5/stable-diffusion-v1-5`, OpenRAIL-M) at 512x512 but replace LCM-LoRA with
**`ByteDance/Hyper-SD` `Hyper-SD15-8steps-CFG-lora`** (CFG-preserving, better prompt adherence; 4-step `Hyper-SD15-4steps-lora`
when speed matters) or **`h1t/TCD-SD15-LoRA`** (MIT, 4 steps, eta 0.3) and stack the project's rank-8 biome LoRAs on top.
Why: it is the only option in 2026 that simultaneously (1) fits 6 GB with headroom (~3 GB), (2) lands at ~1 s/asset on an
RTX 3060 (5 it/s measured at 512px -> 0.8 s for 4 steps), (3) trains biome LoRAs on the same 6-8 GB card, (4) gets true
seamlessness for free via circular padding in a convolutional UNet (Tiled Diffusion shows circular padding reaches
Tiling Score 0.03, the same as their method), and (5) has the sprite-sheet checkpoint and pixel LoRAs ready. Pixel
art is natively low-resolution, so 512px plus 4-8x nearest/k-centroid downscale loses nothing. Avoid DMD2 (CC-BY-NC)
and SD/SDXL-Turbo (Stability non-commercial terms). If the hard floor is 8 GB and biome LoRAs can be trained on a 12 GB
machine, promote **SDXL + `ByteDance/SDXL-Lightning` 4-step UNet** (openrail++) at 768-1024px (`[E]` 2-4 s on a 3060)
with `nerijs/pixel-art-xl`; it is the same pipeline shape and the same tiling trick.

**(b) Optional quality tier.** **FLUX.2 [klein] 4B** (`black-forest-labs/FLUX.2-klein-4B`, Apache-2.0, 4 steps,
guidance 1.0, Qwen3-4B text encoder, `Flux2KleinPipeline` in diffusers >= 0.37). Measured 5.8 s at 1024px on an
RTX 3060 12 GB and 3.2 s on a 4070, `[E]` ~2 s at 512px; on 8 GB load the transformer as GGUF Q8_0/Q5_K + NF4 text
encoder and accept 15-30 s per 1024px image, or offload the text encoder to CPU. Train biome LoRAs on
`FLUX.2-klein-base-4B` (official `train_dreambooth_lora_flux2_klein.py` or ai-toolkit, 12 GB minimum, ~1 h per LoRA on a
4090) and load them on the distilled model; A/B the base model for pixel art because one sprite-sheet author reports
the distilled model damages pixel consistency. **Z-Image-Turbo** (`Tongyi-MAI/Z-Image-Turbo`, Apache-2.0, 8 NFE) is the
alternative for 16 GB cards and owns the most popular pixel LoRA; train its LoRAs with `ostris/zimage_turbo_training_adapter`
or apply `DiffSynth-Studio/Z-Image-Turbo-DistillPatch`. Skip SANA-Sprint (no asset ecosystem, DC-AE f32 risk) and
Nitro-E (custom pipeline, no LoRA path) unless you need sub-0.5 s previews.

**(c) LoRAs to pull from HF.** SD1.5 tier: `Onodofthenorth/SD_PixelArt_SpriteSheet_Generator` (Apache-2.0, 4-view
sprite sheets), `artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5`, dataset `jainr3/diffusiondb-pixelart`
for your own training. SDXL tier: `nerijs/pixel-art-xl` (950 K dl; 8 steps, guidance 1.5, weight 1.2 with LCM-LoRA),
`artificialguybr/PixelArtRedmond`, `ntc-ai/SDXL-LoRA-slider.pixel-art`, `Muapi/hand-painted-2d-seamless-textures`.
klein tier: `Limbicnation/pixel-art-lora` (distilled 4B), `svntax-dev/pixel_spritesheet_4walk_small_lora_v1` (base 4B),
`fal/flux-2-klein-4b-spritesheet-lora` (multi-view I2I). Z-Image tier: `tarn59/pixel_art_style_lora_z_image_turbo`,
`DK9/3D_MMORPG_style_z-image-turbo_lora`. No commercial-safe seamless-texture LoRA exists for any tier; tileability must
come from (e).

**(d) Pins (2026-10-06):** torch 2.14.1+cu130, torchvision 0.29.1+cu130, diffusers 0.41.0, transformers 5.18.0,
huggingface-hub 1.33.0 (diffusers caps `<2.0`), safetensors 0.8.0, accelerate 1.15.0, peft 0.21.2, bitsandbytes 0.50.2,
gguf 0.19.0, torchmetrics 1.9.0, torch-fidelity 0.4.0, clean-fid 0.1.35, lpips 0.1.4, open-clip-torch 3.3.0,
imscore 0.0.11, textile-metric 0.0.11, triton-windows 3.8.0.post29 (optional). Isolate hpsv3 1.0.0 and image-reward 1.5
in a legacy venv; keep Nunchaku 1.2.1 in an optional torch-2.11 venv. Do not plan on xformers or torchao on Windows.

**(e) Tileability technique and metric.** Asymmetric circular padding on every Conv2d of the UNet **and** VAE (section
6.1 snippet), X/Y selectable, applied for all steps; for the DiT tier patch the VAE and add a Tiled-Diffusion-style
latent roll/copy in `callback_on_step_end`. Gate every tile on the Tiled-Diffusion **Tiling Score** (<= 0.05 target)
and a seam-vs-interior gradient ratio (<= 1.2), and log **TexTile** (`textile-metric`) as the learned score; use FID/KID
+ CLIP score for overall quality, and mean pairwise CLIP/DINOv2 cosine inside each biome for style consistency.

---

## 11. Open questions and the validation plan

1. Every `[E]` latency above must be measured on the real target GPU: a 20-sample benchmark of SD1.5+Hyper-SD15 (512px),
   SDXL-Lightning (768/1024px), and klein-4B (512px, bf16 on 12 GB; GGUF Q8 + NF4 text encoder on 8 GB), reporting
   median and p95 with warm caches.
2. Pixel-fidelity A/B for klein: distilled vs base with `Limbicnation/pixel-art-lora`, judged by downscale-then-upscale
   PSNR against the raw output (a proxy for "pixel-perfect" edges) and by TS.
3. Confirm that `Hyper-SD15-8steps-CFG-lora` composes with rank-8 biome LoRAs without re-tuning LoRA scales (fuse order,
   scale 1.0 vs 0.125 as recommended for FLUX/SD3 variants only).
4. torch.compile on Windows via triton-windows 3.8 with torch 2.14: verify the pairing (the README table stops at
   torch 2.10 -> triton 3.6).
5. Qwen-Image-2.1 (2026-09-30) ships with "license: other"; re-check once the terms are published - a 7 B model with
   native RGBA output would be attractive for sprites if it is commercial-friendly.

---

## 12. Sources (all accessed 2026-10-06)

Model cards / repos: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [Z-Image repo](https://github.com/Tongyi-MAI/Z-Image),
[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B),
[FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), [flux2 GitHub](https://github.com/black-forest-labs/flux2), [BFL klein blog](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence),
[klein LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora), [BFL klein training docs](https://docs.bfl.ai/flux_2/flux2_klein_training),
[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf),
[SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning), [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD), [Hyper-SD LICENSE](https://huggingface.co/ByteDance/Hyper-SD/blob/main/LICENSE.md),
[LCM-LoRA SD1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5), [LCM-LoRA blog](https://huggingface.co/blog/lcm_lora), [TCD-SD15](https://huggingface.co/h1t/TCD-SD15-LoRA), [DMD2](https://huggingface.co/tianweiy/DMD2),
[SD1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [SDXL-Turbo](https://huggingface.co/stabilityai/sdxl-turbo), [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo),
[SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), [SD3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo), [SD3.5 blog](https://huggingface.co/blog/sd3-5),
[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512), [Qwen-Image-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning), [Qwen-Image-2512-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning), [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
[SANA-Sprint 1.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers), [SANA-Sprint 0.6B](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers), [SANA 1.5](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers), [SANA-Sprint docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_sprint),
[HiDream-I1-Fast](https://huggingface.co/HiDream-ai/HiDream-I1-Fast), [Lumina-Image 2.0](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0), [PixArt-Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS), [PixArt-LCM](https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS),
[Kolors](https://huggingface.co/Kwai-Kolors/Kolors-diffusers), [CogView4](https://huggingface.co/zai-org/CogView4-6B), [Chroma1-HD](https://huggingface.co/lodestones/Chroma1-HD), [Chroma1-Flash](https://huggingface.co/lodestones/Chroma1-Flash), [LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image), [Ovis-Image](https://huggingface.co/ATH-MaaS/Ovis-Image-7B),
[Nitro-E](https://huggingface.co/amd/Nitro-E), [Nitro-E GitHub](https://github.com/AMD-AGI/Nitro-E), [GLM-Image](https://huggingface.co/zai-org/GLM-Image), [ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo), [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo), [Ideogram 4](https://huggingface.co/ideogram-ai/ideogram-4-fp8), [Boogu-Image-0.1-Turbo](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo), [Ming-Image](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), [Anima](https://huggingface.co/circlestone-labs/Anima), [DreamLite](https://huggingface.co/carlofkl/DreamLite-mobile),
[flux.1-lite-8B](https://huggingface.co/Freepik/flux.1-lite-8B), [FLUX.1-Turbo-Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha).
LoRAs: [pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl), [PixelArtRedmond](https://huggingface.co/artificialguybr/PixelArtRedmond), [PixelArtRedmond 1.5](https://huggingface.co/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5), [SpriteSheet generator](https://huggingface.co/Onodofthenorth/SD_PixelArt_SpriteSheet_Generator),
[tarn59 Z-Image pixel](https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo), [Limbicnation klein pixel](https://huggingface.co/Limbicnation/pixel-art-lora), [svntax sprite sheet](https://huggingface.co/svntax-dev/pixel_spritesheet_4walk_small_lora_v1), [fal klein spritesheet](https://huggingface.co/fal/flux-2-klein-4b-spritesheet-lora),
[gokaygokay seamless](https://huggingface.co/gokaygokay/Flux-Seamless-Texture-LoRA), [gokaygokay 2D game assets](https://huggingface.co/gokaygokay/Flux-2D-Game-Assets-LoRA), [ostris Z-Image adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter), [DistillPatch](https://huggingface.co/DiffSynth-Studio/Z-Image-Turbo-DistillPatch), [Z-Image training strategies](https://huggingface.co/blog/kelseye/training-strategies-of-z-image-turbo), [ai-toolkit](https://github.com/ostris/ai-toolkit), [ai-toolkit 12 GB issue](https://github.com/ostris/ai-toolkit/issues/550), [DiffSynth Z-Image doc](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md), [runcomfy klein 16 GB](https://www.runcomfy.com/trainer/ai-toolkit/flux-2-klein-16gb-vram-training).
Benchmarks: [apatero klein guide](https://apatero.com/blog/flux-2-klein-consumer-gpu-guide), [localaimaster FLUX low-VRAM](https://localaimaster.com/blog/run-flux-on-low-vram-gpu), [zimage.run low-VRAM](https://zimage.run/blog/z-image-turbo-quantized-low-vram-guide), [ComfyUI-wiki Z-Image](https://comfyui-wiki.com/en/news/2025-11-27-alibaba-z-image-turbo-release), [st-hakky GPU table](https://book.st-hakky.com/en/data-science/gpu-comparison-stable-diffusion), [MSI SDXL](https://www.msi.com/blog/stable-diffusion-xl-best-value-rtx-graphics-card), [Qwen-Image GGUF guide](https://dev.to/gary_yan_86eb77d35e0070f5/qwen-image-2512-gguf-complete-guide-to-running-ai-image-generation-on-consumer-hardware-1l6c).
Tiling: [Tiled Diffusion CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Madar_Tiled_Diffusion_CVPR_2025_paper.pdf), [arXiv 2412.15185](https://arxiv.org/html/2412.15185v1), [code](https://github.com/madaror/tiled-diffusion), [ControlTile 2606.22945](https://arxiv.org/abs/2606.22945), [ControlTile code](https://github.com/junrongh/ControlTile), [TexTile CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Rodriguez-Pardo_TexTile_A_Differentiable_Metric_for_Texture_Tileability_CVPR_2024_paper.pdf), [TexTile repo](https://github.com/crp94/textile), [TileGen](https://people.engr.tamu.edu/nimak/Papers/SIGAsia2022_TileGen/index.html), [asymmetric tiling](https://github.com/tjm35/asymmetric-tiling-sd-webui/), [seamless tile inpainting](https://width.ai/post/introduction-to-seamless-tile-inpainting).
Libraries: [diffusers releases](https://github.com/huggingface/diffusers/releases), [diffusers caching](https://huggingface.co/docs/diffusers/main/en/optimization/cache), [diffusers quantization](https://huggingface.co/docs/diffusers/main/en/quantization/overview), [diffusers GGUF](https://huggingface.co/docs/diffusers/main/en/quantization/gguf), [diffusers dreambooth scripts](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth), [Flux2 docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2),
[PyTorch releases](https://github.com/pytorch/pytorch/releases), [cu130 index](https://download.pytorch.org/whl/cu130/torch/), [transformers releases](https://github.com/huggingface/transformers/releases), [peft releases](https://github.com/huggingface/peft/releases), [accelerate releases](https://github.com/huggingface/accelerate/releases), [torchmetrics releases](https://github.com/Lightning-AI/torchmetrics/releases), [bitsandbytes 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/releases), [huggingface_hub releases](https://github.com/huggingface/huggingface_hub/releases), [safetensors 0.8.0](https://github.com/huggingface/safetensors/releases),
[triton-windows](https://github.com/woct0rdho/triton-windows), [Nunchaku](https://github.com/nunchaku-tech/nunchaku), [xformers 0.0.35](https://github.com/facebookresearch/xformers/releases), [DeepCache](https://github.com/horseee/deepcache), [ParaAttention](https://github.com/chengzeyi/ParaAttention), [onnxruntime-directml](https://pypi.org/project/onnxruntime-directml/), [TensorRT-RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html), [HPSv3](https://github.com/MizzenAI/HPSv3), [imscore](https://github.com/RE-N-Y/imscore), [ImageReward](https://github.com/AI-Machine-Vision-Lab/ImageReward-Evaluating-Human-Preferences-for-Text-to-Image-Generation), [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license), [DINOv3 ViT-B](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), [PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1).
