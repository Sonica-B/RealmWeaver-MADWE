"""FID and KID at a fixed n against the reference textures: torchmetrics over torch-fidelity's InceptionV3.

Both metrics see uint8 RGB resized to 299 px here, so the number does not depend on the asset size. FID is
biased at small n; KID is the unbiased statistic and comes back as subset mean and std (ADR-0005). The Inception
weights are fetched from torch-fidelity's GitHub release on first use; when that fails the result is
`{"skipped": reason}` instead of an exception, so a bench run offline still writes its report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from realmweaver.config import settings

log = logging.getLogger(__name__)

_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_INCEPTION_PX = 299
_MAX_SUBSET = 50  # KID subset size ceiling; a smaller set uses every image in each subset
_BATCH = 16  # images per feature-extractor call, bounded for CPU memory


def reference_images(real_dir: Path) -> list[Path]:
    """Image files under `real_dir`, recursively, in a stable order."""
    return sorted(p for p in Path(real_dir).rglob("*") if p.suffix.lower() in _SUFFIXES)


def _device() -> str:
    device = settings().device
    if device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            return device
    return "cpu"


def _inception_batch(images: list[np.ndarray]) -> Any:
    """uint8 `[N, 3, 299, 299]` from HxWx3|4 uint8 images (alpha dropped), bilinear with antialiasing."""
    import torch
    import torch.nn.functional as F

    out = []
    for img in images:
        # np.array copies: PIL-backed arrays are read-only and torch wants writable memory
        x = torch.from_numpy(np.array(img[..., :3])).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(_INCEPTION_PX, _INCEPTION_PX), mode="bilinear", antialias=True)
        out.append(x.round().clamp_(0, 255).to(torch.uint8))
    return torch.cat(out)


def kid_fid(real_dir: Path, fakes: list[np.ndarray], n: int, seed: int, extractor: Any | None = None) -> dict:
    """FID and KID between `n` reference images under `real_dir` and `n` of `fakes`, both picked with `seed`.

    `n` is capped by what either side has and recorded in the result; fewer than two images on a side skips.
    `extractor` is a feature network for torchmetrics' `feature=` (uint8 `[N, 3, 299, 299]` -> `[N, d]`); the
    InceptionV3 pool3 features are fetched when it is None.
    """
    paths = reference_images(real_dir)
    n = min(n, len(paths), len(fakes))
    if n < 2:
        return {"skipped": f"need 2+ images per side: {len(paths)} under {real_dir}, {len(fakes)} generated"}
    try:
        # ponytail: Inception pool3 features by default; FD-DINOv2 or CMMD judge diffusion output more fairly
        # (docs/research/02 section 5.1) and slot in through `extractor`, unreported until a bench records them.
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

        device, feature = _device(), 2048 if extractor is None else extractor
        fid = FrechetInceptionDistance(feature=feature).to(device)
        kid = KernelInceptionDistance(feature=feature, subset_size=min(_MAX_SUBSET, n)).to(device)
    except Exception as e:  # the weights come over the network at construction; offline means no metric
        log.warning("FID/KID skipped: %s", e)
        return {"skipped": f"inception features unavailable: {type(e).__name__}: {e}"}
    from PIL import Image

    rng = np.random.default_rng(seed)
    real = [np.asarray(Image.open(paths[i]).convert("RGB")) for i in rng.choice(len(paths), n, replace=False)]
    fake = [fakes[i] for i in rng.choice(len(fakes), n, replace=False)]
    for images, is_real in ((real, True), (fake, False)):
        for start in range(0, n, _BATCH):
            batch = _inception_batch(images[start : start + _BATCH]).to(device)
            fid.update(batch, real=is_real)
            kid.update(batch, real=is_real)
    kid_mean, kid_std = kid.compute()
    return {"fid": float(fid.compute()), "kid_mean": float(kid_mean), "kid_std": float(kid_std), "n": n}
