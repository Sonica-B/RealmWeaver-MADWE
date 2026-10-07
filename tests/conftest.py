"""Shared test configuration. GPU tests mark themselves with `pytest.mark.gpu` and skip without CUDA."""

import os
import threading
from pathlib import Path

import pytest

from realmweaver.assets import ProceduralGenerator
from realmweaver.types import Asset, AssetSpec


@pytest.fixture(autouse=True)
def _isolated_reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep benchmark reports written by tests out of the repo's reports/ folder."""
    if "REALMWEAVER_REPORTS_DIR" not in os.environ:
        monkeypatch.setenv("REALMWEAVER_REPORTS_DIR", str(tmp_path / "reports"))


class GatedGenerator:
    """`Generator` adapter over the procedural one whose every `generate` call waits for `gate` (open at first);
    `started` is set once a call is inside the generator, so a test knows a generation holds a runner slot."""

    def __init__(self) -> None:
        self.inner, self.gate, self.started = ProceduralGenerator(), threading.Event(), threading.Event()
        self.gate.set()

    def generate(self, spec: AssetSpec) -> Asset:
        self.started.set()
        self.gate.wait(timeout=10)
        return self.inner.generate(spec)


@pytest.fixture
def gated_generator() -> GatedGenerator:
    return GatedGenerator()
