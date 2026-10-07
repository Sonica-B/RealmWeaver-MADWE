"""Local LLM adapters. `LocalLlm` runs a GGUF model in-process through llama-cpp-python (every layer on the GPU
unless the device is `cpu`, e.g. `REALMWEAVER_DEVICE=cpu`); `FakeLlm` replays scripted replies for tests. Both are
`Callable[[str], str]`, the only thing `NpcRuntime` needs.

The default model is `Qwen/Qwen3-4B-GGUF` Q4_K_M (Apache-2.0, ungated), the spike's measured baseline
(docs/research/08). Prompts are wrapped in Qwen3's ChatML template with an empty think block, which is how the
model card switches thinking off. Replies are free-form text that `parse_action` validates: the spike rejected
grammar-constrained decoding, so there is no grammar path.
"""
# ponytail: one model in this process; the game-design spec's `llama-server` sidecar shared with TTS and STT is the
# upgrade path and gets an ADR when it lands.

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType

from realmweaver.config import settings

log = logging.getLogger(__name__)

DEFAULT_REPO = "Qwen/Qwen3-4B-GGUF"
DEFAULT_FILE = "Qwen3-4B-Q4_K_M.gguf"
DEFAULT_CTX = 4096
DEFAULT_MAX_TOKENS = 160
STOP: tuple[str, ...] = ("<|im_end|>",)
INSTALL_HINT = (
    "llama-cpp-python is not installed: `uv sync --extra npc` builds the CPU wheel; the prebuilt CUDA wheel is "
    "`uv pip install --python .venv llama-cpp-python --extra-index-url "
    "https://abetlen.github.io/llama-cpp-python/whl/cu124`"
)


class FakeLlm:
    """Scripted replies in order (the last one repeats), or a function of the prompt; keeps every prompt seen."""

    def __init__(self, replies: Iterable[str] | Callable[[str], str]) -> None:
        self._fn = replies if callable(replies) else None
        self._replies = [] if callable(replies) else list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._fn is not None:
            return self._fn(prompt)
        if not self._replies:
            raise RuntimeError("FakeLlm has no replies")
        return self._replies[min(len(self.prompts), len(self._replies)) - 1]


def chat_prompt(prompt: str) -> str:
    """Qwen3 ChatML with the think block pre-closed (thinking off)."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _import_llama_cpp() -> ModuleType:
    # The prebuilt CUDA wheel resolves cudart/cublas through PATH; without a CUDA toolkit, torch's copies serve.
    if "CUDA_PATH" not in os.environ:
        spec = importlib.util.find_spec("torch")
        if spec is not None and spec.submodule_search_locations:
            lib = str(Path(next(iter(spec.submodule_search_locations))) / "lib")
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    try:
        import llama_cpp
    except ImportError as e:
        raise RuntimeError(INSTALL_HINT) from e
    return llama_cpp


class LocalLlm:
    """A GGUF model behind `__call__(prompt) -> str`. Sampling is seeded so a run repeats; `llama` is the backend
    handle, for a harness that reads llama.cpp's own counters."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        n_ctx: int = DEFAULT_CTX,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        seed: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        lib = _import_llama_cpp()
        self.device = device or settings().device
        self.model_path, self.max_tokens, self.temperature = str(model_path), max_tokens, temperature
        self.n_gpu_layers = -1 if self.device.startswith("cuda") else 0
        self.llama = lib.Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=n_ctx,
            seed=seed,
            n_threads=n_threads,
            verbose=verbose,
        )
        log.info(
            "loaded %s on %s (gpu layers %d, context %d)",
            self.model_path,
            self.device,
            self.n_gpu_layers,
            n_ctx,
        )

    @property
    def name(self) -> str:
        return Path(self.model_path).name

    def __call__(self, prompt: str) -> str:
        out = self.llama(
            chat_prompt(prompt), max_tokens=self.max_tokens, temperature=self.temperature, stop=list(STOP)
        )
        return out["choices"][0]["text"]


def load_local_llm(repo_id: str = DEFAULT_REPO, filename: str = DEFAULT_FILE, **kwargs: object) -> LocalLlm:
    """Download (or reuse from the Hugging Face cache) the GGUF and load it; kwargs go to `LocalLlm`."""
    from huggingface_hub import hf_hub_download

    return LocalLlm(hf_hub_download(repo_id, filename), **kwargs)  # type: ignore[arg-type]
