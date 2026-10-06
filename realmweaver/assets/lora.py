"""Biome LoRA training: rank-8 PEFT adapters on the SD1.5 UNet attention projections, saved in diffusers format.

Every image is captioned with `build_prompt`: the file name minus the folder prefix and index is the subject, so
`fantasy_forest/fantasy_forest_ancient_bark_0003.png` becomes "ancient bark, <biome style>, seamless tileable
texture". Latents (each image plus its mirror) and prompt embeds are encoded once with the frozen VAE and text
encoder; then `steps` AdamW updates run under fp16 autocast. The output directory holds
`pytorch_lora_weights.safetensors` (loads with `load_lora_weights`) and `train_log.json` (wall-clock, per-step loss).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from realmweaver.assets.seamless import set_seamless
from realmweaver.biomes import Biome, load_biome
from realmweaver.config import settings
from realmweaver.types import AssetSpec

log = logging.getLogger(__name__)

_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
_BATCH = 2  # ponytail: fixed micro-batch sized for a 12 GB card; expose it when an 8 GB card needs 1
_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _subject(stem: str, prefix: str) -> str:
    """`fantasy_forest_ancient_bark_0003` with folder `fantasy_forest` -> `ancient bark`."""
    if stem.startswith(prefix + "_"):
        stem = stem[len(prefix) + 1 :]
    return re.sub(r"[_\s]*\d+$", "", stem).replace("_", " ").strip() or stem


def _encode_dataset(
    pipe: Any, biome: Biome, paths: list[Path], resolution: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latents for each image and its mirror, with matching prompt embeds; VAE and text encoder stay frozen."""
    from realmweaver.assets.prompts import build_prompt

    latents, embeds = [], []
    scale, rng = pipe.vae.config.scaling_factor, torch.Generator(device).manual_seed(0)
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].to(device, torch.float16) / 127.5 - 1
            caption = build_prompt(biome, AssetSpec(biome.name, subject=_subject(p.stem, p.parent.name)))[0]
            emb = pipe.encode_prompt(caption, device, 1, False)[0]
            for view in (x, x.flip(-1)):
                latents.append(pipe.vae.encode(view).latent_dist.sample(generator=rng) * scale)
                embeds.append(emb)
    return torch.cat(latents), torch.cat(embeds)


def train_biome_lora(
    biome: str,
    image_dir: Path,
    out_dir: Path,
    rank: int = 8,
    steps: int = 200,
    lr: float = 1e-4,
    resolution: int = 512,
) -> Path:
    """Train a UNet-attention LoRA for `biome` on the images under `image_dir`; returns `out_dir`."""
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from diffusers.training_utils import cast_training_params
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    s = settings()
    if not s.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("train_biome_lora needs a CUDA device (settings().device)")
    device = torch.device(s.device)
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no images under {image_dir}")
    b = load_biome(biome)
    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        s.model_id, dtype=torch.float16, safety_checker=None, requires_safety_checker=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    latents, embeds = _encode_dataset(pipe, b, paths, resolution, device)

    unet = pipe.unet
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=_TARGETS)
    )
    cast_training_params(unet, dtype=torch.float32)
    # The generator runs textures with circular padding, so the adapter trains under the same padding.
    set_seamless(unet, True)
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    n_train = int(noise_sched.config.num_train_timesteps)
    rng = torch.Generator(device).manual_seed(0)
    losses: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, len(latents), (_BATCH,), generator=rng, device=device)
        x0 = latents[idx]
        noise = torch.randn(x0.shape, generator=rng, device=device, dtype=x0.dtype)
        t = torch.randint(0, n_train, (_BATCH,), generator=rng, device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(
                noise_sched.add_noise(x0, noise, t), t, encoder_hidden_states=embeds[idx], return_dict=False
            )[0]
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
        if step % 20 == 0 or step == steps - 1:
            log.info("lora %s step %d/%d loss %.4f", biome, step + 1, steps, losses[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(out_dir, unet_lora_layers=state, safe_serialization=True)
    wall = time.perf_counter() - t0
    record = {
        "biome": biome,
        "images": len(paths),
        "rank": rank,
        "steps": steps,
        "lr": lr,
        "batch": _BATCH,
        "resolution": resolution,
        "wall_s": wall,
        "loss": losses,
    }
    (out_dir / "train_log.json").write_text(json.dumps(record), encoding="utf-8")
    log.info("saved %s LoRA (rank %d, %d steps, %.0fs) to %s", biome, rank, steps, wall, out_dir)
    return out_dir
