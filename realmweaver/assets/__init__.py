"""Asset generation: the CPU `Generator` adapter, prompt building and alpha keying.

The diffusion generator, seamless patching, memory pool, LoRA training and the embedder live in sibling modules
and are imported from them directly; they need torch, this surface does not.
"""

from realmweaver.assets.alpha import alpha_from_white
from realmweaver.assets.procedural import ProceduralGenerator
from realmweaver.assets.prompts import build_prompt

__all__ = ["ProceduralGenerator", "alpha_from_white", "build_prompt"]
