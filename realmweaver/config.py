"""Process settings. Environment overrides use the REALMWEAVER_ prefix (e.g. REALMWEAVER_DEVICE=cpu)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch missing or broken: CPU it is
        return False


@dataclass
class Settings:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    fast_lora_repo: str = "ByteDance/Hyper-SD"
    draft_lora_file: str = "Hyper-SD15-4steps-lora.safetensors"
    refine_lora_file: str = "Hyper-SD15-8steps-CFG-lora.safetensors"
    device: str = "cuda"
    cache_bytes: int = 512 * 1024 * 1024
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")
    chunk_size: int = 16
    asset_size: int = 512


def settings() -> Settings:
    """Build settings from defaults + environment. Cheap; call it where needed instead of caching globally."""
    s = Settings(device="cuda" if _cuda_available() else "cpu")
    for f in fields(Settings):
        raw = os.environ.get(f"REALMWEAVER_{f.name.upper()}")
        if raw is None:
            continue
        current = getattr(s, f.name)
        if isinstance(current, bool):
            value: object = raw.lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, Path):
            value = Path(raw)
        else:
            value = raw
        setattr(s, f.name, value)
    return s
