"""Quality metrics on numpy images and style vectors. CPU only; the bench reports what these return.

Tileability and tiling score look for seams on float grey. Style consistency works on style vectors (`[n, d]`
rows per biome) and the histogram embedding is the CPU stand-in for a DINOv2 style vector.
"""

from __future__ import annotations

import numpy as np

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
_BINS = 16  # per RGB channel: a 48-dim histogram embedding
_EPS = 1e-6


def _grey(img: np.ndarray) -> np.ndarray:
    """Float32 HxW luma from an HxW, HxWx3 or HxWx4 image (alpha ignored)."""
    a = np.asarray(img, dtype=np.float32)
    return a[..., :3] @ _LUMA if a.ndim == 3 else a


def tileability(img: np.ndarray) -> float:
    """Mean |gradient| across the wrapped seams divided by mean |gradient| between interior neighbours.

    The seam pairs are last column vs first column and last row vs first row. 1.0 means the wrapped edge is
    indistinguishable from the interior; larger means a visible seam; a flat image scores 1.0.
    """
    g = _grey(img)
    seam = np.concatenate([np.abs(g[:, -1] - g[:, 0]), np.abs(g[-1, :] - g[0, :])]).mean()
    interior = np.concatenate([np.abs(np.diff(g, axis=1)).ravel(), np.abs(np.diff(g, axis=0)).ravel()]).mean()
    return float(seam / interior) if interior >= _EPS else 1.0


def tiling_score(img: np.ndarray, band: int = 4) -> float:
    """Tiled-Diffusion-style seam score, lower is better.

    The image is rolled by half its size so the wrapped edges meet in a centre cross; the MSE of adjacent-pixel
    steps inside a band of `band` px either side of that cross is normalised by the global variance. A flat
    image scores 0; a seam-free image scores its own interior roughness; a hard seam scores well above it.
    """
    g = _grey(img)
    var = float(g.var())
    if var < _EPS:
        return 0.0
    cy, cx = g.shape[0] // 2, g.shape[1] // 2
    k = max(1, min(band, cy, cx))
    rolled = np.roll(g, (cy, cx), axis=(0, 1))
    across_vertical = np.diff(rolled[:, cx - k : cx + k], axis=1)
    across_horizontal = np.diff(rolled[cy - k : cy + k, :], axis=0)
    return float((np.mean(across_vertical**2) + np.mean(across_horizontal**2)) / 2 / var)


def style_consistency(vecs_by_biome: dict[str, np.ndarray]) -> float:
    """Mean within-biome pairwise cosine minus mean cross-biome pairwise cosine of L2-normalised style vectors.

    Higher means biomes look like themselves and unlike each other; a generator whose output all looks the same
    scores about 0. Pairs are pooled across biomes; a side of the contrast with no pairs counts as 0.
    """
    mats = [np.atleast_2d(np.asarray(v, dtype=np.float64)) for v in vecs_by_biome.values() if len(v)]
    if not mats:
        return 0.0
    labels = np.repeat(np.arange(len(mats)), [len(m) for m in mats])
    m = np.concatenate(mats)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), _EPS)
    i, j = np.triu_indices(len(m), 1)
    cos, same = (m @ m.T)[i, j], labels[i] == labels[j]
    within = cos[same].mean() if same.any() else 0.0
    cross = cos[~same].mean() if (~same).any() else 0.0
    return float(within - cross)


def histogram_embed(img: np.ndarray) -> np.ndarray:
    """48-dim colour histogram (16 bins per RGB channel), L2-normalised; the CPU stand-in for DINOv2."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    bins = a[..., :3].reshape(-1, 3).astype(np.int64) * _BINS // 256
    hist = np.concatenate([np.bincount(bins[:, c], minlength=_BINS) for c in range(3)]).astype(np.float64)
    return hist / max(float(np.linalg.norm(hist)), _EPS)
