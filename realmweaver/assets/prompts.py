"""Prompt construction: a biome's style plus a tile class or prop description becomes the diffusion prompt pair."""

from __future__ import annotations

from realmweaver.biomes import Biome
from realmweaver.types import AssetSpec

TEXTURE_SUFFIX = "seamless tileable texture"


def build_prompt(biome: Biome, spec: AssetSpec) -> tuple[str, str]:
    """Return `(positive, negative)` prompts for `spec` in `biome`.

    Textures: ``"<tile prompt>, <biome style>, seamless tileable texture"``; sprites: ``"<prop description>,
    <biome style>"``. A subject the biome does not define is used verbatim as its own description. The negative
    prompt is the biome's.
    """
    if spec.kind == "sprite":
        description = biome.props.get(spec.subject, spec.subject)
        return f"{description}, {biome.style}", biome.negative
    tile = biome.tiles.get(spec.subject)
    description = tile.prompt if tile is not None else spec.subject
    return f"{description}, {biome.style}, {TEXTURE_SUFFIX}", biome.negative
