"""Shared test configuration. GPU tests mark themselves with `pytest.mark.gpu` and skip without CUDA."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep benchmark reports written by tests out of the repo's reports/ folder."""
    if "REALMWEAVER_REPORTS_DIR" not in os.environ:
        monkeypatch.setenv("REALMWEAVER_REPORTS_DIR", str(tmp_path / "reports"))
