"""The `realmweaver` command via `main(argv) -> int` on the CPU adapters: files, output, exit codes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from realmweaver.cli import main
from realmweaver.types import AssetSpec


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settle the device without importing torch; the subcommands under test pick the CPU adapters anyway."""
    monkeypatch.setenv("REALMWEAVER_DEVICE", "cpu")


# --- the plan's tests ---------------------------------------------------------------------------------


def test_layout_writes_a_png_of_size_times_8_px(tmp_path):
    p = tmp_path / "layout.png"
    assert main(["layout", "--biome", "forest", "--size", "16", "--seed", "1", "--out", str(p)]) == 0
    assert Image.open(p).size == (16 * 8, 16 * 8)


def test_generate_writes_a_64px_png(tmp_path):
    p = tmp_path / "grass.png"
    argv = ["generate", "--biome", "forest", "--tile", "grass", "--procedural", "--size", "64"]
    assert main([*argv, "--out", str(p)]) == 0
    img = Image.open(p)
    assert img.size == (64, 64) and img.mode == "RGB"


def test_bench_procedural_writes_one_report_under_the_reports_dir(capsys):
    reports = Path(os.environ["REALMWEAVER_REPORTS_DIR"])  # conftest points it at a tmp dir
    assert main(["bench", "--procedural", "--n", "2"]) == 0
    written = list(reports.glob("bench-*.json"))
    assert len(written) == 1 and json.loads(written[0].read_text(encoding="utf-8"))["schema"] == 1
    assert capsys.readouterr().out.strip() == str(written[0]), "bench prints the report path and nothing else"


def test_unknown_subcommand_is_a_usage_error():
    assert main(["nope"]) == 2


# --- beyond the plan ---------------------------------------------------------------------------------


def test_world_preview_with_a_walk_writes_png_and_prints_prewarm_stats(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REALMWEAVER_ASSET_SIZE", "64")
    monkeypatch.setenv("REALMWEAVER_CHUNK_SIZE", "8")
    p = tmp_path / "world.png"
    assert main(["world", "--biome", "forest", "--chunks", "2", "--walk", "20", "--out", str(p)]) == 0
    img = np.asarray(Image.open(p))
    assert img.shape == (2 * 8 * 32, 2 * 8 * 32, 3)  # chunks x tiles x 32 px per tile
    assert len({tuple(px) for px in img.reshape(-1, 3)[::97]}) > 1  # assets drawn, not one flat colour
    stats = json.loads(capsys.readouterr().out.splitlines()[0])
    assert {"prewarm_hits", "prewarm_misses", "chunks", "assets"} <= set(stats)
    assert stats["prewarm_hits"] + stats["prewarm_misses"] >= 1 and stats["chunks"] >= 4


def test_generate_prop_writes_an_rgba_sprite(tmp_path):
    p = tmp_path / "mushroom.png"
    argv = ["generate", "--biome", "forest", "--prop", "mushroom", "--procedural", "--size", "64"]
    assert main([*argv, "--out", str(p)]) == 0
    img = Image.open(p)
    assert img.mode == "RGBA" and img.size == (64, 64)
    a = np.asarray(img)
    assert a[0, 0, 3] == 0 and a[32, 32, 3] == 255


def test_generate_prints_the_canonical_asset_id_latency_and_tileability(tmp_path, capsys):
    p = tmp_path / "water.png"
    argv = ["generate", "--biome", "forest", "--tile", "water", "--procedural", "--size", "32", "--seed", "5"]
    assert main([*argv, "--out", str(p)]) == 0
    out = capsys.readouterr().out
    expected = AssetSpec("forest", "texture", "water", size=32, seed=5, steps=4, seamless=True, tier="draft")
    assert f"id={expected.id}" in out and "latency_s=" in out and "tileability=" in out and str(p) in out


def test_layout_json_holds_the_rows_of_tile_classes(tmp_path):
    png, js = tmp_path / "l.png", tmp_path / "l.json"
    argv = ["layout", "--biome", "desert", "--size", "12", "--seed", "3", "--out", str(png)]
    assert main([*argv, "--json", str(js)]) == 0
    rows = json.loads(js.read_text(encoding="utf-8"))
    assert len(rows) == 12 and all(len(r) == 12 for r in rows)
    assert {cls for r in rows for cls in r} <= {"sand", "dune", "rock", "water", "shore", "cactus"}


def test_usage_errors_return_2_and_help_returns_0(tmp_path):
    assert main([]) == 2
    assert main(["generate", "--biome", "forest"]) == 2  # neither --tile nor --prop
    assert main(["generate", "--biome", "forest", "--tile", "grass", "--prop", "log"]) == 2  # both
    assert main(["layout", "--biome", "atlantis", "--out", str(tmp_path / "x.png")]) == 2  # unknown biome
    assert main(["train-lora", "--biome", "forest"]) == 2  # --images missing
    assert main(["--help"]) == 0


def test_runtime_failure_returns_1_with_one_line_on_stderr(tmp_path, capsys):
    (tmp_path / "blocker").write_text("not a directory", encoding="utf-8")
    out = tmp_path / "blocker" / "layout.png"  # the parent cannot be created
    assert main(["layout", "--biome", "forest", "--size", "4", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("realmweaver: error:")


def test_serve_builds_the_bridge_app_and_hands_it_to_uvicorn(monkeypatch):
    import uvicorn
    from fastapi import FastAPI

    monkeypatch.setenv("REALMWEAVER_ASSET_SIZE", "64")
    monkeypatch.setenv("REALMWEAVER_CHUNK_SIZE", "8")
    served: dict = {}

    def fake_run(app, **kwargs):  # the server loop itself is uvicorn's to test
        served.update(app=app, **kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    assert main(["serve", "--procedural", "--biome", "desert", "--host", "127.0.0.1", "--port", "8010"]) == 0
    assert isinstance(served["app"], FastAPI) and (served["host"], served["port"]) == ("127.0.0.1", 8010)
    assert {"/chunk/{cx}/{cy}", "/asset/{asset_id}.png", "/player"} <= {r.path for r in served["app"].routes}
