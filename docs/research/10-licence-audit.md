# 10 - Licence audit of every model on the default path (as of 2026-10-06)

Resolves [#35](https://github.com/Sonica-B/RealmWeaver-MADWE/issues/35). Serves E14 Content production pipeline (#20)
and E15 Platform and release (#21). Input: the default path in `07-generative-game-stack-2026.md` section 8 and the
spec's hard rules ("opt-out of any gated or non-commercial model on the default path", "no AMASS/HumanML3D-trained
generator in a commercial build", "any model with regional limits -> tagged fallbacks only; licence check in CI for
provenance records").

How to read:

- `[P]` = I read the primary text (licence file, EULA, model card front matter) on 2026-10-06; `[S]` = secondary
  source only, because the primary page refuses automated fetches (Epic and Adobe pages were read through a text
  proxy or press coverage) - a human must open those pages in a browser before the M4 legal pass.
- HF licence tags, gated flags and `extra_gated_eu_disallowed` flags were read from the HF Hub API on 2026-10-06.
- "Commercial use" is about the **model**; "Outputs" is about the generated textures/meshes/clips/voice we ship.
  Everything here is my reading of the linked texts, not legal advice.
- Assumed release: worldwide Steam, Windows, single-player, pre-baked content plus a small runtime AI stack
  (4B LLM, TTS, STT). Nothing diffusion-sized ships with the game.

---

## 0. TL;DR - verdicts

| Model | Verdict |
|---|---|
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | **default path** (CreativeML OpenRAIL-M) |
| `ByteDance/Hyper-SD` - `Hyper-SD15-*-lora.safetensors` only | **default path** (CreativeML OpenRAIL++-M, ByteDance 2024-04-11); the `Hyper-FLUX.1-dev-*` files in the same repo are **non-commercial**, the `Hyper-SD3-*` files are Stability Community - never pull those |
| `h1t/TCD-SD15-LoRA` (ADR-0001 alternate) | **default path** (MIT) |
| `facebook/dinov2-small` | **default path** (Apache-2.0) |
| `microsoft/TRELLIS.2-4B` | **default path** (MIT) **with three pipeline fixes**: swap `briaai/RMBG-2.0` (gated, non-commercial) for `ZhengPeng7/BiRefNet` (MIT); keep `nvdiffrast`/`nvdiffrec` (NVIDIA non-commercial source licences) out of the production environment; accept the DINOv3 licence for the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` encoder |
| `VAST-AI/TripoSG` | **default path** (MIT) |
| `stabilityai/stable-point-aware-3d` (SPAR3D) | **fallback only** - gated, commercial use needs registration with Stability, licence terminates above US $1 M annual revenue, "Powered by Stability AI" notice; use for placeholders that never ship |
| `xandergos/terrain-diffusion-30m` | **default path** (MIT; Copernicus DEM credit line recommended) |
| `google/gemma-4-E4B-it` | **default path** (Apache-2.0; the Gemma Terms of Use do **not** apply to Gemma 4) |
| `Qwen/Qwen3.5-4B` | **default path** (Apache-2.0) |
| `hexgrad/Kokoro-82M` | **default path for pre-generated voice banks**; its `espeak-ng` dependency is GPL-3.0 and must not be linked into the game process |
| `ResembleAI/chatterbox` / `chatterbox-turbo` | **default path** (MIT; every output carries Resemble's Perth watermark) |
| whisper.cpp + `openai/whisper` weights (`ggerganov/whisper.cpp` ggml files) | **default path** (MIT code and weights) |
| Mixamo characters and animations (Adobe) | **default path** (royalty-free commercial use; raw files never redistributed) |
| Epic Game Animation Sample (Fab) | **default path** (Fab Standard License at the time of reading; record the licence field at download) |
| MetaHuman (UE 5.6+) | **default path** (Unreal Engine EULA; MetaHuman renders/curves never enter any training or evaluation set) |
| `tencent/HY-Motion-1.0` | **excluded** - licence §5(c) forbids using or displaying **Outputs** outside the Territory (no EU, UK, South Korea), 1 M MAU cap, training data undisclosed (cannot certify AMASS-free); previs only |
| `VAST-AI/UniRig` | **default path** (MIT; trained on CC-BY-4.0 Articulation-XL2.0) |
| `tencent/Hunyuan3D-2.1` | **excluded from shipped assets** (same §5(c) Output-outside-Territory clause and 1 M MAU cap); internal previs only, not a shippable fallback |

Models that **must leave the default path**: HY-Motion-1.0 (excluded), Hunyuan3D-2.1 (excluded from anything that
ships; the plan's "tagged fallback" is not enough because the licence travels with the Output), SPAR3D (fallback only).
Three default-path models stay only with the fixes above: TRELLIS.2 (helper models), Hyper-SD (file allowlist), Kokoro
(GPL isolation).

---

## 1. Audit table

| # | Model / source | Licence (link) | Commercial use | Generated outputs | Attribution / notices | Regional or MAU limits | Gated | Training-data caveat | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `stable-diffusion-v1-5/stable-diffusion-v1-5` (859 M, 78 M dl) | CreativeML OpenRAIL-M `[P]` ([text](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE), HF tag `creativeml-openrail-m`) | yes - royalty-free grant, use-based restrictions (Attachment A) | §6: "Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output" | only when **distributing the model or derivatives** (e.g. our biome LoRAs): copy of licence, Attachment A as enforceable terms, keep notices, mark modified files | none | no | LAION-2B-en aesthetic subsets (web-scraped, includes copyrighted images; subject of ongoing US litigation) - risk sits in outputs that copy training images or named artists' styles | **default** |
| 2 | `ByteDance/Hyper-SD`, files `Hyper-SD15-{1step,2steps,4steps,8steps,8steps-CFG,12steps-CFG}-lora.safetensors` | repo `LICENSE.md` `[P]` is three licences: "For Flux.1-DEV-related models" -> FLUX.1 [dev] **Non-Commercial**; "For SD3-related models" -> Stability AI Community; "For other SD-related models" -> **CreativeML Open RAIL++-M, Copyright (c) 2024 Bytedance Inc., dated 2024-04-11**. HF card has **no licence tag**; [discussion #79](https://huggingface.co/ByteDance/Hyper-SD/discussions/79) (2025-04) unanswered | yes for the SD1.5/SDXL LoRAs (OpenRAIL++-M) | OpenRAIL++-M §: "Licensor claims no rights in the Output" | same pass-through as row 1 if the LoRA or a merge is distributed | none | no | distilled from SD1.5 (row 1 caveat inherited) | **default, SD15 files only**; FLUX files excluded, SD3 files Stability-only |
| 3 | `h1t/TCD-SD15-LoRA` (alternate few-step LoRA, ADR-0001) | MIT `[P]` (HF tag) | yes | no claim | MIT notice if redistributed | none | no | distilled from SD1.5 | **default** |
| 4 | `facebook/dinov2-small` (22 M) | Apache-2.0 `[P]` (HF tag; GitHub README: "DINOv2 code and model weights are released under the Apache License 2.0") | yes | n/a - embeddings only (ADR-0004 style vectors, coherence score) | Apache LICENSE + NOTICE if the weights ship in the game | none | no | LVD-142M, curated from web-crawled images; no generation, so no output risk | **default** |
| 5 | `microsoft/TRELLIS.2-4B` (4 B) | MIT `[P]` (HF tag, GitHub "This model and code are released under the MIT License") - **but** `pipeline.json` hard-wires `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated; [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/): commercial yes, "Built with DINOv3" only when distributing the weights, military/ITAR ban), `briaai/RMBG-2.0` (gated, `license: other`; BRIA terms restrict commercial use to licensees), `microsoft/TRELLIS-image-large` (MIT); README lists `nvdiffrast` ("Nvidia Source Code License (1-Way Commercial)": "may be used ... non-commercially" = research/evaluation only) and `nvdiffrec` (same wording) for **rendering** | yes for the model; **no** for RMBG-2.0 and the NVIDIA renderers | MIT: no claim | MIT notice if code redistributed; nothing for internal DINOv3 use | none | DINOv3 encoder gated (accept Meta's licence); RMBG-2.0 gated | ~800 K assets from Objaverse-XL, ABO, HSSD + TexVerse (per-asset CC licences incl. NC/SA variants) - review queue must catch near-copies of recognisable Sketchfab assets | **default with fixes**: BiRefNet (MIT) instead of RMBG-2.0; no nvdiffrast/nvdiffrec in the production env (GLB export goes through `o_voxel.postprocess.to_glb`, MIT); accept DINOv3 licence |
| 6 | `VAST-AI/TripoSG` (1.4 B) | MIT `[P]` (HF tag) | yes | no claim | MIT notice if redistributed | none | no | 2 M curated Image-SDF pairs from Objaverse(-XL) + ShapeNet (paper §4, §5.2.1); same near-copy caveat as row 5; check its background-removal helper the same way | **default** |
| 7 | `stabilityai/stable-point-aware-3d` (2 B) | [Stability AI Community License](https://stability.ai/community-license-agreement) `[P]` (2024-07-05 text; HF `license: other`, **gated**) | yes **if** < US $1 M annual revenue (you + affiliates, any source) **and** registered at stability.ai/community-license; above that the licence terminates - enterprise licence needed | §IV(c)(iii): "You own any outputs"; §IV(b): outputs may not train a competing foundational model | when distributing the model, a derivative, "or a product or service that uses any portion of them": copy of agreement, Notice file "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved", and "Powered by Stability AI" prominently displayed; AUP applies | revenue cap, no territory clause | **yes** (terms acceptance) | Objaverse (HF dataset tag) | **fallback only** - spec M0 rule (gated) + registration + revenue trigger; placeholders never ship |
| 8 | `xandergos/terrain-diffusion-30m` | MIT `[P]` (HF tag; paper arXiv 2512.08309) | yes | no claim | MIT notice if redistributed; Copernicus asks that products "produced using Copernicus WorldDEM-30" carry "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" - add to credits (cheap safe harbour) | none | no | trained on Copernicus GLO-30 DEM (free incl. commercial, attribution on distribution of the data) | **default** |
| 9 | `google/gemma-4-E4B-it` (8.0 B eff., 4 B active) | **Apache-2.0** `[P]` (HF `license: apache-2.0`, `license_link` -> [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license) = Apache 2.0 text; the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) list Gemma 1-3n and say "For Gemma 4 terms, see the Gemma 4 license") | yes | Apache is silent; Google's Prohibited Use Policy is linked as policy, not as a licence condition | Apache LICENSE + NOTICE shipped with the GGUF in the installer; "Gemma" is a Google trademark (nominative use only) | none | no (HF) | web/code/images/audio, cutoff Jan 2025 | **default** |
| 10 | `Qwen/Qwen3.5-4B` (4.7 B) | Apache-2.0 `[P]` (HF tag + repo LICENSE) | yes | no claim | Apache LICENSE + NOTICE if the GGUF ships | none | no | undisclosed web/code mix | **default** |
| 11 | `hexgrad/Kokoro-82M` (82 M) | Apache-2.0 `[P]` (weights); inference `kokoro`/`misaki` Apache; **`espeak-ng` G2P fallback is GPL-3.0** | yes ("Kokoro has been deployed in numerous projects and commercial APIs") | no claim | Apache notice if shipped; credit the CC BY training audio the card lists (Koniwa CC BY 3.0 < 1 h, SIWIS CC BY 4.0 < 11 h) | none | no | "trained exclusively on permissive/non-copyrighted audio data" (public domain, Apache/MIT audio, synthetic audio from closed TTS) - cleanest TTS provenance on the list | **default for offline voice banks**; no in-process espeak-ng in the game |
| 12 | `ResembleAI/chatterbox` (multilingual) and `chatterbox-turbo` (350 M, en) | MIT `[P]` (HF tags) | yes | no claim; **every output carries the Perth neural watermark** ("Every audio file generated by Chatterbox includes Resemble AI's Perth ... Watermarker") | MIT notice; disclose the watermark in the AI-content disclosure | none | no | "0.5M hours of cleaned data", sources undisclosed; prompts "sourced from freely available data on the internet"; README acknowledges Llama 3 - verify the T3 backbone is architecture-only (if initialised from Llama 3 weights, the Llama 3 Community License "Built with Llama" notice applies) | **default**; clone only consenting voices |
| 13 | whisper.cpp (`ggml-org/whisper.cpp`) + `openai/whisper` weights (`ggerganov/whisper.cpp` ggml conversions) | MIT `[P]` - openai/whisper README: "Whisper's code and model weights are released under the MIT License"; whisper.cpp LICENSE "Copyright (c) 2023-2026 The ggml authors" | yes | n/a (transcripts) | MIT notices (OpenAI 2022, ggml authors) | none | no | 680 k h web audio (OpenAI) - not a shipping concern | **default** |
| 14 | Mixamo characters + animations (Adobe) | [Adobe General Terms of Use](https://www.adobe.com/legal/terms.html) §3.6 Content Files (licence to use in your End Use, distribute "only in connection with your End Use", never "on a stand-alone basis") `[P]`; [Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html) (last updated 2021-09-14) `[S]`: "You can use both characters and animations royalty free for personal, commercial, and non-profit projects"; Adobe staff FAQ (community, 2022-09-29) `[S]`: no "blueprints, templates, or asset packages for video game engines which redistribute character or animation raw files", no stock-site packages, team sharing allowed | yes, royalty-free | you own your creations; Adobe keeps rights in its content | none required | Adobe ID required (no Enterprise/Federated IDs; not offered to China-country-code accounts); no MAU cap; "technology preview", no SLA | Adobe ID | Adobe's own mocap | **default** - clips embedded in cooked game assets; never raw FBX in an SDK or mod kit |
| 15 | Epic Game Animation Sample ([Fab listing](https://www.fab.com/listings/880e319a-a59e-4ed2-b268-b32dac7fa016), Epic Games, free, UE 5.4-5.8, "500+ game-ready animations compatible with the UE5 Mannequin Skeleton") | listing licence field reads **"Standard License"** `[S]` -> [Fab EULA](https://www.fab.com/eula) (2024-10-01) `[S]`: use/modify Content, distribute Projects with Content incorporated (end users may use it "solely as incorporated in the Project in object code"), no standalone distribution except to collaborators, engine-agnostic. In June 2024 (UE Marketplace) it was "only licensed for use with Unreal Engine" - the licence field at download time governs | yes | you own the game; animation files are Fab Content | none | Personal tier < US $100 k revenue in the digital-content industry vs Professional - both free for this listing | Epic account | Epic mocap | **default**; record the Fab licence field in provenance at download |
| 16 | MetaHuman (Creator, Animator, characters) with UE 5.6+ | [Unreal Engine EULA](https://www.unrealengine.com/en-US/eula/unreal) `[S]` (read via proxy): §4(a)(i) Non-Engine Products royalty-free; §6(e) must not "use, or permit others to use, MetaHuman digital characters and animation curves (or any rendered output thereof ...) to build or enhance any database or training or testing any artificial intelligence, machine learning, deep learning, neural networks"; §7(a) MetaHuman names/logos are Epic trademarks; §7(d) rights/consent for any likeness in Cloud Input Data. [MetaHuman Creator EULA](https://www.unrealengine.com/en-US/eula/mhc) `[S]`: "If you're using MetaHuman with Unreal Engine 5.6 and beyond, the Unreal Engine EULA applies." [UE 5.6 release post](https://www.unrealengine.com/news/unreal-engine-5-6-is-now-available) (2025-06-03): "new licensing options in the UE EULA that enable MetaHumans to be used with any engine or creative software"; MetaHumans tradeable on Fab (`.mhpkg`) | yes; engine royalty 5 % above US $1 M lifetime gross per product applies to Emberfall as a UE product regardless | you own the depiction of your characters; Epic notices retained | keep Epic proprietary notices in the product | none (royalty, not territory) | Epic account | n/a (hand-authored); **never** use MetaHuman renders/curves in any training or evaluation set | **default** (humans) |
| 17 | `tencent/HY-Motion-1.0` (1.0 B; Lite 0.46 B) | [Tencent HY-MOTION 1.0 Community License](https://huggingface.co/tencent/HY-Motion-1.0/blob/main/LICENSE.txt) (2025-12-30) `[P]`: "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"; grant "for the Territory only" | yes inside the Territory, < 1 M MAU | §6(d) no rights claimed, **but §5(c): "You must not use, reproduce, modify, distribute, or display the ... Works, Output or results ... outside the Territory. Any such use outside the Territory is unlicensed"**; §5(b) outputs may not improve other AI models; AUP #12: machine-generated content placed in public must be identified | copy of agreement to recipients, Notice file ("Tencent HY-MOTION 1.0 is licensed under ..."), "Powered by Tencent HY" encouraged, use restrictions passed through | **Territory excludes EU, UK, South Korea; 1 M MAU trigger** (§4); HF `extra_gated_eu_disallowed: true` | EU-disallowed flag | 3,000 h pretrain (12 M HunyuanVideo clips -> reconstruction + ~500 h mocap/animation assets) + 400 h curated; no dataset named -> **cannot certify AMASS/HumanML3D-free**; SMPL-H 22-joint skeleton | **excluded** (shipping); previs only, and not even that for EU/UK/KR team members |
| 18 | `VAST-AI/UniRig` | MIT `[P]` (HF tag; GitHub `VAST-AI-Research/UniRig`) | yes | skeleton + skin weights for **our** mesh; no claim | MIT notice if redistributed | none | no | [Articulation-XL2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0) (CC-BY-4.0, rigged Objaverse-XL assets) | **default** |
| 19 | `tencent/Hunyuan3D-2.1` (fallback in the plan) | [Tencent Hunyuan 3D 2.1 Community License](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE) `[P]`: Territory "excluding the territory of the European Union, United Kingdom and South Korea" | yes inside the Territory, < 1 M MAU | §5(c): "You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan 3D 2.1 Works, **Output** or results ... outside the Territory"; §5(b) no training other models; AUP #12 disclosure | agreement copy, Notice file, "Powered by Tencent Hunyuan" encouraged | EU/UK/KR excluded; 1 M MAU; `extra_gated_eu_disallowed: true` | EU-disallowed flag | undisclosed | **excluded from shipped assets**; previs only |
| 20 | (on the animation path, not in the ticket) `nvidia/Audio2Face-3D-v3.0` | NVIDIA Open Model License `[P]` ("ready for commercial/non-commercial use") | yes | - | licence copy on redistribution | none | no | NVIDIA capture (< 10 k h) | audit when adopted |

---

## 2. The clauses that decide the verdicts

**Tencent (HY-Motion-1.0, Hunyuan3D-2.1).** Both licences grant rights "for the Territory only" and both add, in
§5(c), that the **Output** may not be used, distributed or displayed outside the Territory. A clip or mesh generated
with either model therefore cannot be in a build sold to EU, UK or South Korean players - the restriction travels
with the asset, so "tagged fallback" cannot make it shippable; only geo-fencing the game could, and Steam does not
make that practical for a single SKU. HY-Motion adds the AMASS problem: the report names no public dataset, only
"approximately 500 hours of motion sequences" from "motion capture data and 3D animation assets" plus video-derived
motion, so the spec's hard rule cannot be verified. Both repos set `extra_gated_eu_disallowed: true` on HF, which
also blocks an EU-based collaborator from downloading them at all. Verdict: excluded; keep them in the research
notes as previs tools for non-EU team members, never in the content factory's shippable path.

**Stability (SPAR3D).** Commercial use is free below US $1 M annual revenue, but (a) the repo is gated, which the
spec's M0 rule already excludes from the default path, (b) commercial users "must register with Stability AI",
(c) the licence "shall terminate" the day you or an affiliate cross US $1 M from any source, and (d) any product
"that uses any portion of them" must carry the Notice file and "Powered by Stability AI". For sub-second placeholders
that never ship this is tolerable; for shipped props it is four obligations TRELLIS.2 and TripoSG do not have.
Verdict: fallback only (placeholders).

**TRELLIS.2.** The model is MIT, but the reference pipeline is not: `pipeline.json` loads `briaai/RMBG-2.0`
(gated, BRIA's terms restrict commercial use to licensees) for background removal and `facebook/dinov3-vitl16-pretrain-lvd1689m`
(gated, DINOv3 License: commercial OK, no obligations unless the weights are redistributed) for conditioning, and the
README installs `nvdiffrast` and `nvdiffrec`, whose NVIDIA source licences allow only non-commercial research or
evaluation use. The README scopes the two NVIDIA packages to rendering ("Utilized for rendering generated 3D assets",
"split-sum renderer for PBR materials"); GLB export runs through `o_voxel.postprocess.to_glb` (MIT). Production
therefore: BiRefNet (MIT) or `rembg` for matting, DINOv3 licence accepted on the HF account that runs the factory,
no `nvdiffrast`/`nvdiffrec` in the production lockfile (previews rendered in the engine instead), and the ComfyUI
Windows path checked for the same three helpers before it is used for shippable assets.

**Hyper-SD.** The HF card carries no licence tag and the licence question in the community tab is unanswered, which
is why the plan flagged it. The repo's `LICENSE.md` settles it: three licences keyed by file family, and the SD1.5
LoRAs fall under "For other SD-related models ... Copyright (c) 2024 Bytedance Inc. License dated April 11, 2024",
a CreativeML Open RAIL++-M text with the standard Attachment A use restrictions. The FLUX LoRAs next to them are
FLUX.1 [dev] Non-Commercial. CI must allowlist by file name, not by repo.

**Kokoro.** Weights and inference are Apache-2.0 and the training data is the cleanest on this list, but `misaki`
falls back to `espeak-ng` (GPL-3.0) for out-of-dictionary words. Linking it into the game executable would put the
game under GPL-3.0 obligations. The plan already splits voice into pre-generated banks and live TTS: Kokoro stays on
the offline side, live lines go through Chatterbox-Turbo (MIT, no GPL dependency).

**Gemma 4.** The ticket asks about the Gemma Terms. They apply to Gemma 1 through 3n and the specialised variants;
the Terms page itself says "For Gemma 4 terms, see the Gemma 4 license", which is Apache-2.0. So no distribution
notice, no remote-restriction clause and no prohibited-use contract for Gemma 4 - only the Apache LICENSE/NOTICE when
the GGUF ships, and Google's Prohibited Use Policy as a policy we choose to follow.

**Epic.** Two things changed since the stack document was written. The Game Animation Sample's Fab listing now shows
"Standard License" (engine-agnostic, no standalone redistribution) rather than the UE-only terms it had on the UE
Marketplace in 2024; and since UE 5.6 (2025-06-03) MetaHuman is governed by the Unreal Engine EULA, with Epic
stating MetaHumans can be used "with any engine or creative software" and sold on Fab. The EULA clause that bites
us is §6(e): MetaHuman characters, animation curves and renders may not be used to build datasets or to train or
test AI. For a pipeline built around LoRA training and a DINOv2 style gate that means: MetaHuman content never
enters a training set or a style reference board, and the legal pass should confirm that scoring in-game frames
that contain MetaHumans with the style gate is not "testing AI" in Epic's sense.

**Mixamo.** Royalty-free for commercial games, no attribution, but the raw character/animation files may not be
redistributed as files (asset packs, blueprints, stock sites). Cooked animation assets inside the game are fine;
a modding SDK that exposes the FBX library is not. Adobe calls the service a "technology preview" with no SLA, so
archive the downloaded library now (team sharing is allowed).

---

## 3. Models that must leave the default path, and what replaces them

| Leaving | Why | Replacement already on the path |
|---|---|---|
| `tencent/HY-Motion-1.0` (bespoke text-to-motion) | Output barred outside EU/UK/KR-excluded Territory; 1 M MAU; training data undisclosed (AMASS rule unverifiable) | Mixamo + Game Animation Sample clips in the Motion Matching database; bespoke clips by mocap (Rokoko/Move.ai) or hand-keyed in the engine; no open text-to-motion model on the list is both clean of AMASS and territory-free |
| `tencent/Hunyuan3D-2.1` (quality fallback for PBR painting) | same Output-outside-Territory clause; 1 M MAU | TRELLIS.2 texture stage (`slat_flow_imgshape2tex`) for other models' geometry; TripoSG for watertight shapes |
| `stabilityai/stable-point-aware-3d` (sub-second placeholders) | gated, registration, US $1 M revenue trigger, "Powered by Stability AI" | keep as **fallback only** for placeholders that never reach the ship manifest; TRELLIS.2 at 512³ (~45-60 s) for anything that ships |
| `briaai/RMBG-2.0` (inside TRELLIS.2's default pipeline) | gated, commercial use needs a BRIA licence | `ZhengPeng7/BiRefNet` (MIT) or `rembg` |
| `nvdiffrast`, `nvdiffrec` (TRELLIS.2 preview renderers) | NVIDIA non-commercial source licences | engine-side previews; `o_voxel` GLB export is MIT |
| `Hyper-FLUX.1-dev-*` LoRAs (same repo as Hyper-SD15) | FLUX.1 [dev] Non-Commercial | `Hyper-SD15-*` (OpenRAIL++-M) or `h1t/TCD-SD15-LoRA` (MIT); `tianweiy/DMD2` stays excluded (CC-BY-NC-4.0) |
| `espeak-ng` in the game process (Kokoro G2P fallback) | GPL-3.0 | offline voice banks; Chatterbox-Turbo for live lines |

---

## 4. Obligations checklist for a commercial release

**A. Ship-manifest gate (CI, E15)**

1. Every shipped asset has a provenance record: `model_id`, HF revision SHA, LoRA/checkpoint **file name**, `licence_id`,
   `verdict`, prompt/seed, reviewer id (spec M0).
2. Only `verdict: default` enters the ship manifest. `fallback` needs an owner waiver stored in the record (HITL);
   `excluded` fails the build.
3. Blocklist (build fails on sight): `tencent/HY-Motion-1.0`, `tencent/Hunyuan3D-2.1` and any other
   `tencent-hunyuan-community` model, `tianweiy/DMD2`, any FLUX.1-dev-derived file (incl. `Hyper-FLUX.1-dev-*`),
   `briaai/RMBG-2.0`, SMPL/SMPL-X meshes, any AMASS/HumanML3D-trained motion model (MoMask, T2M-GPT, MotionGPT3,
   LGTM, MotionLCM).
4. Production lockfile must not contain `nvdiffrast` or `nvdiffrec`; the factory's model cache must not contain RMBG-2.0.
5. Training-set manifests (biome LoRAs, style boards, DINO references) must not include MetaHuman content or Fab
   assets tagged NoAI.

**B. Third-party notices (installer `THIRD_PARTY_NOTICES` + in-game credits)**

6. Apache-2.0 LICENSE + NOTICE for every shipped weight file under it: Gemma 4 GGUF, Qwen3.5 GGUF (if shipped),
   DINOv2 (if resident at runtime), Kokoro + `misaki` (if the engine ever runs them).
7. MIT notices: OpenAI whisper (code + weights), whisper.cpp (ggml authors), llama.cpp, Chatterbox (Resemble AI),
   TRELLIS.2 (Microsoft), TripoSG and UniRig (VAST), TCD LoRA, terrain-diffusion (xandergos), BiRefNet.
8. CreativeML OpenRAIL-M / OpenRAIL++-M: no obligation for generated textures. If the SD1.5 checkpoint, Hyper-SD15
   LoRAs or our biome LoRAs are ever distributed (published content factory, mod tools), ship the licence text and
   make Attachment A use restrictions enforceable terms of that distribution.
9. Credits lines: "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH
   2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" (terrain); Koniwa (CC BY 3.0)
   and SIWIS (CC BY 4.0) as Kokoro training audio; Epic proprietary notices for MetaHuman content; Unreal Engine
   logo/notice per Epic's brand rules.
10. Only if SPAR3D stays in any pipeline: Stability Notice file + "Powered by Stability AI" + registration at
    stability.ai/community-license.

**C. Generation-time content policy (E14)**

11. One policy = union of OpenRAIL Attachment A, Google's Prohibited Use Policy and Stability's AUP (Tencent's is
    moot): no real-person likeness (spec), no minors in sexual contexts, no hate/harassment content, no disinformation
    artefacts; prompts and seeds logged in provenance so a takedown can be traced.
12. No living artists' names or branded products/characters in prompts; biome/style LoRAs trained on own art or
    licensed references only; the review queue checks 3D outputs for near-copies of recognisable Objaverse/Sketchfab
    assets.

**D. Runtime AI in the shipped game**

13. Steam AI-content disclosure (Steamworks content survey): pre-generated content (textures, meshes, terrain, clips,
    voice banks) and live-generated content (NPC dialogue via Gemma 4/Qwen3.5, live TTS via Chatterbox-Turbo, STT via
    whisper.cpp) with the guardrails described; mention the Perth watermark on TTS audio.
14. No OpenRAIL-licensed model ships at runtime (keeps the game EULA free of pass-through use restrictions) - matches
    the plan's "nothing diffusion-sized is resident during play".
15. GPL isolation: no `espeak-ng` inside the game process; if runtime G2P is ever needed, run it as a separate GPL
    executable or use a dictionary-only G2P.
16. Voice cloning only with written consent; default to built-in or own-recorded voices.

**E. Accounts and asset intake**

17. Epic account (UE, Fab, MetaHuman); record each Fab asset's licence field (Standard / legacy UE Marketplace /
    CC-BY) and NoAI tag at download; no Fab asset redistributed standalone.
18. Adobe ID for Mixamo (not Enterprise/Federated); archive the clip library; raw FBX never leaves the team.
19. HF account used by the factory accepts the DINOv3 licence (TRELLIS.2 encoder); SPAR3D gate only if used.

**F. Training-data hygiene (hard rules)**

20. No AMASS/HumanML3D-trained motion generator in any build; HY-Motion stays out until Tencent discloses its data
    and drops the Territory clause (do not plan on it).
21. SMPL/SMPL-X bodies never ship; only retargeted motion on the UE5 Mannequin.
22. MetaHuman renders and animation curves excluded from every dataset and evaluation set (UE EULA §6(e)).

**G. Triggers to monitor**

23. UE royalty: 5 % above US $1 M lifetime gross per product (quarterly report to Epic once crossed).
24. Stability: US $1 M annual revenue ends the community licence (only if SPAR3D is used).
25. Fab: Personal -> Professional tier at US $100 k digital-content revenue for paid assets.
26. Any future model with a Territory or MAU clause: `fallback` at best, never in the ship manifest.

---

## 5. CI allowlist (lift into the provenance checker)

```yaml
# licence_id -> verdict for the default path; a provenance record must match on model_id (+ file glob where given)
models:
  - {model_id: stable-diffusion-v1-5/stable-diffusion-v1-5, licence_id: creativeml-openrail-m, verdict: default}
  - {model_id: ByteDance/Hyper-SD, file_glob: "Hyper-SD15-*-lora.safetensors", licence_id: creativeml-openrail-plus-plus-m, verdict: default}
  - {model_id: ByteDance/Hyper-SD, file_glob: "Hyper-FLUX*", licence_id: flux-1-dev-non-commercial, verdict: excluded}
  - {model_id: ByteDance/Hyper-SD, file_glob: "Hyper-SD3-*", licence_id: stability-community, verdict: excluded}
  - {model_id: h1t/TCD-SD15-LoRA, licence_id: mit, verdict: default}
  - {model_id: latent-consistency/lcm-lora-sdv1-5, licence_id: openrail-plus-plus, verdict: default}
  - {model_id: tianweiy/DMD2, licence_id: cc-by-nc-4.0, verdict: excluded}
  - {model_id: facebook/dinov2-small, licence_id: apache-2.0, verdict: default}
  - {model_id: microsoft/TRELLIS.2-4B, licence_id: mit, verdict: default,
     requires: [facebook/dinov3-vitl16-pretrain-lvd1689m], forbids_env: [nvdiffrast, nvdiffrec, briaai/RMBG-2.0]}
  - {model_id: facebook/dinov3-vitl16-pretrain-lvd1689m, licence_id: dinov3-license, verdict: default, gated: true, redistribute: false}
  - {model_id: briaai/RMBG-2.0, licence_id: bria-rmbg-2.0, verdict: excluded, gated: true}
  - {model_id: ZhengPeng7/BiRefNet, licence_id: mit, verdict: default}
  - {model_id: microsoft/TRELLIS-image-large, licence_id: mit, verdict: default}
  - {model_id: VAST-AI/TripoSG, licence_id: mit, verdict: default}
  - {model_id: stabilityai/stable-point-aware-3d, licence_id: stability-community, verdict: fallback, gated: true,
     revenue_cap_usd: 1000000, notice: "Powered by Stability AI"}
  - {model_id: xandergos/terrain-diffusion-30m, licence_id: mit, verdict: default, credit: copernicus-glo-30}
  - {model_id: google/gemma-4-E4B-it, licence_id: apache-2.0, verdict: default, ships_at_runtime: true}
  - {model_id: Qwen/Qwen3.5-4B, licence_id: apache-2.0, verdict: default, ships_at_runtime: true}
  - {model_id: hexgrad/Kokoro-82M, licence_id: apache-2.0, verdict: default, offline_only: true, gpl_dependency: espeak-ng}
  - {model_id: ResembleAI/chatterbox, licence_id: mit, verdict: default, watermark: perth}
  - {model_id: ResembleAI/chatterbox-turbo, licence_id: mit, verdict: default, watermark: perth, ships_at_runtime: true}
  - {model_id: openai/whisper, licence_id: mit, verdict: default, ships_at_runtime: true}
  - {model_id: ggerganov/whisper.cpp, licence_id: mit, verdict: default, ships_at_runtime: true}
  - {model_id: adobe/mixamo, licence_id: adobe-general-terms, verdict: default, redistribute_raw: false}
  - {model_id: epic/game-animation-sample, licence_id: fab-standard, verdict: default, redistribute_raw: false, record_fab_licence_field: true}
  - {model_id: epic/metahuman, licence_id: unreal-engine-eula, verdict: default, training_use: forbidden}
  - {model_id: VAST-AI/UniRig, licence_id: mit, verdict: default}
  - {model_id: tencent/HY-Motion-1.0, licence_id: tencent-hy-motion-1.0-community, verdict: excluded,
     territory_excludes: [EU, UK, KR], mau_cap: 1000000, outputs_territory_bound: true}
  - {model_id: tencent/Hunyuan3D-2.1, licence_id: tencent-hunyuan-3d-2.1-community, verdict: excluded,
     territory_excludes: [EU, UK, KR], mau_cap: 1000000, outputs_territory_bound: true}
rules:
  - ship manifest accepts verdict == default only; fallback requires record.waiver.owner; excluded fails the build
  - a record for ByteDance/Hyper-SD must carry file_name and match the default file_glob
  - production lockfile must not contain any forbids_env package; model cache must not contain an excluded model
  - training-set manifests must not reference epic/metahuman or any Fab asset tagged NoAI
```

---

## 6. Open items for the human legal pass (M4)

1. Read the current Unreal Engine EULA and Fab EULA in a browser (Epic pages refuse automated fetches; I read them
   through a text proxy): confirm §4(a)(i), §6(e), §7(d) and that the Game Animation Sample listing still says
   "Standard License" at the time of download.
2. Confirm with Epic (forum or licensing@) that running a DINOv2 style gate over frames containing MetaHumans is not
   "testing AI" under §6(e); default until then: exclude MetaHuman content from all datasets and reference boards.
3. Accept the RMBG-2.0 gate once to read BRIA's terms, then delete the weights - it stays excluded either way.
4. Chatterbox: verify the T3 backbone is Llama-3 architecture only (no Llama 3 weights); if weights are inherited, add
   the Llama 3 Community License notice.
5. Mixamo: the helpx FAQ was last updated 2021-09-14; confirm nothing newer supersedes the Adobe staff FAQ of 2022.
6. If HY-Motion is ever reconsidered: ask hunyuan3d@tencent.com for the training-data provenance and a territory
   waiver in writing; without both it stays excluded.

---

## Sources

HF cards and files: [SD1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [OpenRAIL-M text](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE),
[Hyper-SD LICENSE.md](https://huggingface.co/ByteDance/Hyper-SD/blob/main/LICENSE.md), [Hyper-SD discussion #79](https://huggingface.co/ByteDance/Hyper-SD/discussions/79),
[TCD-SD15-LoRA](https://huggingface.co/h1t/TCD-SD15-LoRA), [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5), [DMD2](https://huggingface.co/tianweiy/DMD2),
[DINOv2](https://huggingface.co/facebook/dinov2-small) / [GitHub](https://github.com/facebookresearch/dinov2),
[TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) (`pipeline.json`), [TRELLIS.2 GitHub](https://github.com/microsoft/TRELLIS.2), [TRELLIS.2 paper](https://arxiv.org/abs/2512.14692),
[nvdiffrast licence](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt), [nvdiffrec licence](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt),
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/), [dinov3-vitl16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet),
[TripoSG](https://huggingface.co/VAST-AI/TripoSG) / [paper](https://arxiv.org/abs/2502.06608),
[SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d), [Stability Community License](https://stability.ai/community-license-agreement), [SPAR3D GitHub LICENSE](https://github.com/Stability-AI/stable-point-aware-3d/blob/main/LICENSE.md),
[terrain-diffusion-30m](https://huggingface.co/xandergos/terrain-diffusion-30m) / [paper](https://arxiv.org/abs/2512.08309), [Copernicus DEM terms](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM),
[Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it), [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license), [Gemma Terms of Use](https://ai.google.dev/gemma/terms),
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) / [Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) / [GitHub](https://github.com/resemble-ai/chatterbox),
[openai/whisper](https://github.com/openai/whisper), [whisper.cpp LICENSE](https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE), [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp),
[HY-Motion-1.0 LICENSE.txt](https://huggingface.co/tencent/HY-Motion-1.0/blob/main/LICENSE.txt) / [report](https://arxiv.org/abs/2512.23464), [UniRig](https://huggingface.co/VAST-AI/UniRig), [Articulation-XL2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0),
[Hunyuan3D-2.1 LICENSE](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE), [Audio2Face-3D](https://huggingface.co/nvidia/Audio2Face-3D-v3.0).

Adobe and Epic: [Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html), [Adobe staff FAQ on licensing](https://community.adobe.com/t5/mixamo-discussions/mixamo-faq-licensing-royalties-ownership-eula-and-tos/m-p/13234775), [Adobe General Terms](https://www.adobe.com/legal/terms.html);
[Game Animation Sample on Fab](https://www.fab.com/listings/880e319a-a59e-4ed2-b268-b32dac7fa016), [Fab EULA](https://www.fab.com/eula), [Fab licences and pricing](https://dev.epicgames.com/documentation/en-us/fab/licenses-and-pricing-in-fab), [GASP docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/game-animation-sample-project-in-unreal-engine), [cgchannel 2024-06-12](https://www.cgchannel.com/2024/06/get-500-free-game-ready-animations-for-use-in-unreal-engine/);
[Unreal Engine EULA](https://www.unrealengine.com/en-US/eula/unreal), [MetaHuman Creator EULA (<= 5.5)](https://www.unrealengine.com/en-US/eula/mhc), [UE 5.6 release post](https://www.unrealengine.com/news/unreal-engine-5-6-is-now-available), [cgchannel 2025-06-04](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/), [digitalproduction 2025-06-05](https://digitalproduction.com/2025/06/05/metahumans-graduate-ready-for-unity-godot-and-the-fab-cash-register/).
