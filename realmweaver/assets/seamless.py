"""Seamless switch: circular padding on every padded Conv2d so a texture's edges continue each other.

Used on the UNet and the VAE of the diffusion generator (textures on, sprites off). Pure attribute toggle, so it is
cheap enough to flip per asset.
"""

from __future__ import annotations

import torch


def set_seamless(module: torch.nn.Module, on: bool) -> int:
    """Set `padding_mode` to `"circular"` (or back to `"zeros"`) on every Conv2d with padding > 0.

    Returns the number of convolutions touched. Convs with string padding (`"same"`/`"valid"`) are left alone.
    """
    mode = "circular" if on else "zeros"
    changed = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d) and isinstance(m.padding, tuple) and max(m.padding) > 0:
            m.padding_mode = mode
            changed += 1
    return changed
