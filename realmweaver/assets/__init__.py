"""Asset generation: the `Generator` adapters, prompt building, alpha keying, the memory pool, seamless patching,
LoRA training and the style embedder.

The torch-backed names (`DiffusionGenerator`, `DinoEmbedder`, `MemoryPool`, `set_seamless`, `train_biome_lora`)
are re-exported lazily (PEP 562), so importing this package needs no torch; the CPU adapter and helpers do not.
"""

from importlib import import_module
from typing import Any

from realmweaver.assets.alpha import alpha_from_white
from realmweaver.assets.procedural import ProceduralGenerator
from realmweaver.assets.prompts import build_prompt

_LAZY = {
    "DiffusionGenerator": "realmweaver.assets.diffusion",
    "DinoEmbedder": "realmweaver.assets.embed",
    "MemoryPool": "realmweaver.assets.pool",
    "set_seamless": "realmweaver.assets.seamless",
    "train_biome_lora": "realmweaver.assets.lora",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiffusionGenerator",
    "DinoEmbedder",
    "MemoryPool",
    "ProceduralGenerator",
    "alpha_from_white",
    "build_prompt",
    "set_seamless",
    "train_biome_lora",
]
