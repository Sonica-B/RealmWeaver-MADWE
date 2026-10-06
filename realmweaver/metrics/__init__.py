"""Quality metrics: tileability, tiling score, style consistency and the histogram embedding, numpy only; plus
the benchmark (`run_bench`, `latest_report`) and FID/KID (`kid_fid`), re-exported lazily (PEP 562) because they
pull in `realmweaver.world` and torchmetrics."""

from importlib import import_module
from typing import Any

from realmweaver.metrics.quality import histogram_embed, style_consistency, tileability, tiling_score

_LAZY = {
    "kid_fid": "realmweaver.metrics.fid",
    "latest_report": "realmweaver.metrics.bench",
    "run_bench": "realmweaver.metrics.bench",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "histogram_embed",
    "kid_fid",
    "latest_report",
    "run_bench",
    "style_consistency",
    "tileability",
    "tiling_score",
]
