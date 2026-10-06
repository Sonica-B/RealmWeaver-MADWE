# ADR-0001: SD1.5 + few-step LoRA is the primary model; FLUX.2-klein is the optional quality tier

Date: 2026-10-06. Status: accepted.

## Context
The asset agent must produce a 512px seamless texture in about a second on a 6–12 GB Windows GPU, accept rank-8 biome LoRAs trained on the same card, and ship under licences that allow use. `docs/research/01-model-landscape-2026.md` compares SD1.5/SDXL distillations, SD3.5, FLUX.1/2, Z-Image-Turbo, Qwen-Image, Sana and HiDream.

## Decision
- Primary: `stable-diffusion-v1-5/stable-diffusion-v1-5` (fp16, 512px) with `ByteDance/Hyper-SD` `Hyper-SD15-8steps-CFG-lora` (4-step variant for draft tier) or `h1t/TCD-SD15-LoRA` (MIT), biome LoRAs stacked at rank 8. Seamlessness from circular padding on every Conv2d of UNet and VAE.
- Quality tier: `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, 4 steps, `Flux2KleinPipeline`), text encoder offloaded or NF4 on 12 GB, behind the same `Generator` seam.
- Not chosen: SD/SDXL-Turbo and DMD2 (non-commercial), Z-Image-Turbo (16 GB class), Sana-Sprint and Nitro-E (no LoRA/asset ecosystem).

## Consequences
Measured floor before rewrite on RTX 5070 Ti laptop: 0.61 s/asset warm, 3.09 GB peak (LCM-LoRA). torch stays 2.11+cu128 (what the machine and the Docker base image have); the report's 2.14 pin is not adopted until measured. Gated models are never on the default path.
