"""Local LLM adapters. `LocalLlm` runs a GGUF model through llama-cpp-python (every layer on the GPU unless the
device is `cpu`, e.g. `REALMWEAVER_DEVICE=cpu`); `FakeLlm` replays scripted replies for tests. Both are
`Callable[[str], str]`, the only thing `NpcRuntime` needs.

The spike model is `Qwen/Qwen3-4B-GGUF` Q4_K_M (Apache-2.0, ungated). Prompts are wrapped in Qwen3's ChatML
template with an empty think block, which is how the model card switches thinking off.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
    "llama-cpp-python is not installed; the prebuilt CUDA wheel is "
    "`uv pip install --python .venv llama-cpp-python --extra-index-url "
    "https://abetlen.github.io/llama-cpp-python/whl/cu124` (CPU: the `/whl/cpu` index)"
)


@dataclass
class LlmCall:
    """One generation: token counts and the backend's own prefill / decode timers (`wall_s` includes both)."""

    prompt_tokens: int
    prefill_tokens: int  # prompt tokens actually evaluated; the rest came from the prefix cache
    completion_tokens: int
    prefill_s: float
    decode_s: float
    wall_s: float


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


def grammar_schema(schema: dict) -> dict:
    """The schema without string length bounds: llama.cpp unrolls `maxLength: 400` into 400 nested optionals and
    its grammar engine then stalls for minutes on one reply (measured in the spike). pydantic still enforces the
    bounds on the parsed action."""
    if isinstance(schema, dict):
        return {k: grammar_schema(v) for k, v in schema.items() if k not in ("minLength", "maxLength")}
    if isinstance(schema, list):
        return [grammar_schema(v) for v in schema]
    return schema


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
    """A GGUF model behind `__call__(prompt) -> str`; `calls` keeps per-generation counts and timings.

    `json_schema` (the action schema) compiles to a GBNF grammar so every reply is well-formed JSON; the
    runtime still verifies the content. Sampling is seeded so a measurement run repeats.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        n_ctx: int = DEFAULT_CTX,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        seed: int = 0,
        json_schema: dict | None = None,
        n_threads: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._lib = _import_llama_cpp()
        self.device = device or settings().device
        self.model_path, self.max_tokens, self.temperature = str(model_path), max_tokens, temperature
        self.n_gpu_layers = -1 if self.device.startswith("cuda") else 0
        start = time.perf_counter()
        self._llama = self._lib.Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=n_ctx,
            seed=seed,
            n_threads=n_threads,
            verbose=verbose,
        )
        self.load_s = time.perf_counter() - start
        self._grammar = (
            self._lib.LlamaGrammar.from_json_schema(json.dumps(grammar_schema(json_schema)), verbose=False)
            if json_schema
            else None
        )
        self.calls: list[LlmCall] = []
        log.info(
            "loaded %s on %s in %.1f s (grammar=%s)",
            self.model_path,
            self.device,
            self.load_s,
            bool(json_schema),
        )

    @property
    def name(self) -> str:
        return Path(self.model_path).name

    def __call__(self, prompt: str) -> str:
        ctx = (
            self._llama._ctx.ctx
        )  # ponytail: private handle for llama_perf_context; no public timer in 0.3.x
        self._lib.llama_perf_context_reset(ctx)
        start = time.perf_counter()
        out = self._llama(
            chat_prompt(prompt),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=list(STOP),
            grammar=self._grammar,
        )
        wall = time.perf_counter() - start
        perf, usage = self._lib.llama_perf_context(ctx), out["usage"]
        self.calls.append(
            LlmCall(
                usage["prompt_tokens"],
                int(perf.n_p_eval),
                usage["completion_tokens"],
                perf.t_p_eval_ms / 1000.0,
                perf.t_eval_ms / 1000.0,
                wall,
            )
        )
        return out["choices"][0]["text"]


def load_local_llm(repo_id: str = DEFAULT_REPO, filename: str = DEFAULT_FILE, **kwargs: object) -> LocalLlm:
    """Download (or reuse from the Hugging Face cache) the GGUF and load it; kwargs go to `LocalLlm`."""
    from huggingface_hub import hf_hub_download

    return LocalLlm(hf_hub_download(repo_id, filename), **kwargs)  # type: ignore[arg-type]
