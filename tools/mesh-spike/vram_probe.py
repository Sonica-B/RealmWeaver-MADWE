"""Spike-only measurement nodes: reset / report torch peak VRAM from inside the ComfyUI process.

Two pass-through nodes so they can be wired into any graph:
  VRAMPeakReset  -> call at the start of the graph (after LoadImage), resets torch peak stats
  VRAMPeakReport -> call at the end (takes the final mesh), writes a JSON with peak numbers
"""

import json
import time

import torch


class _Any(str):
    def __ne__(self, other):  # wildcard type: matches any socket
        return False


ANY = _Any("*")


class VRAMPeakReset:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY,), "tag": ("STRING", {"default": "run"})}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "utils/spike"

    @classmethod
    def IS_CHANGED(cls, value, tag):
        return tag

    def run(self, value, tag):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        _STATE[tag] = time.perf_counter()
        return (value,)


class VRAMPeakReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY,),
                "tag": ("STRING", {"default": "run"}),
                "path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    CATEGORY = "utils/spike"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, value, tag, path):
        return tag

    def run(self, value, tag, path):
        stats = {"tag": tag}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            stats["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            stats["max_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
            stats["allocated_now_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        t0 = _STATE.get(tag)
        stats["seconds_since_reset"] = round(time.perf_counter() - t0, 2) if t0 else None
        text = json.dumps(stats)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return {"ui": {"text": [text]}, "result": (text,)}


_STATE: dict = {}

NODE_CLASS_MAPPINGS = {"VRAMPeakReset": VRAMPeakReset, "VRAMPeakReport": VRAMPeakReport}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMPeakReset": "VRAM Peak Reset (spike)",
    "VRAMPeakReport": "VRAM Peak Report (spike)",
}
