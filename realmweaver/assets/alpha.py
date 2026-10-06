"""Alpha keying for sprites generated on a white background."""

from __future__ import annotations

import numpy as np


def alpha_from_white(rgb: np.ndarray, tol: int = 40) -> np.ndarray:
    """Return a uint8 HxWx4 sprite: alpha 0 where the RGB distance to white is below `tol`, 255 elsewhere."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"expected an HxWx3 image, got shape {rgb.shape}")
    colour = rgb[..., :3].astype(np.uint8)
    # ponytail: naive keying. Ceiling: near-white pixels inside the prop punch holes and the edge stays hard.
    # Upgrade path: a matting model (e.g. rembg) at the file edge, returning the same HxWx4 contract.
    distance = np.linalg.norm(255.0 - colour.astype(np.float32), axis=-1)
    alpha = np.where(distance < tol, 0, 255).astype(np.uint8)
    return np.dstack([colour, alpha])
