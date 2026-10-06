"""Benchmark at the seam: one JSON report holding every metric, written here by the CPU baseline generator."""

import json
import platform
import re
from pathlib import Path

import pytest

from realmweaver.assets import ProceduralGenerator
from realmweaver.config import settings
from realmweaver.metrics.bench import REFERENCE_DIR, TARGETS_2025, latest_report, run_bench
from realmweaver.metrics.fid import kid_fid
from realmweaver.types import AssetSpec

TOP_LEVEL = {
    "schema",
    "timestamp",
    "env",
    "params",
    "latency_s",
    "assets_per_min",
    "tileability",
    "tiling_score_mean",
    "style_consistency",
    "vram",
    "allocator",
    "fid_kid",
    "wfc_ms_per_chunk",
    "predictor",
    "targets_2025",
    "notes",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> tuple[Path, dict]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("REALMWEAVER_DEVICE", "cpu")
        out = tmp_path_factory.mktemp("reports")
        path = run_bench(
            ProceduralGenerator(), ["forest"], n=4, seed=0, out_dir=out, with_fid=False, pool_ab=False
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_report_file_has_every_top_level_key(report):
    path, rep = report
    assert re.fullmatch(r"bench-\d{8}-\d{6}\.json", path.name) and path.parent.name.startswith("reports")
    assert set(rep) == TOP_LEVEL and rep["schema"] == 1


def test_latency_is_measured_after_warm_up(report):
    lat = report[1]["latency_s"]
    assert 0 <= lat["p50"] <= lat["p95"] and lat["mean"] >= 0 and report[1]["assets_per_min"] > 0


def test_predictor_beats_ring_baseline(report):
    pred = report[1]["predictor"]
    assert pred["markov_hit_rate"] > pred["ring_hit_rate"] and pred["seeds"] == 20


def test_procedural_textures_are_all_tileable(report):
    rep = report[1]
    assert rep["tileability"]["share_leq_1_2"] == 1.0 and rep["tileability"]["mean_ratio"] > 0
    assert rep["tiling_score_mean"] >= 0 and rep["wfc_ms_per_chunk"] > 0


def test_latest_report_returns_the_newest_by_name(report, tmp_path):
    path, rep = report
    assert latest_report(path.parent) == rep
    older = path.parent / "bench-20000101-000000.json"
    older.write_text(json.dumps({"schema": 1, "stale": True}), encoding="utf-8")
    assert latest_report(path.parent) == rep
    assert latest_report(tmp_path / "nowhere") is None and latest_report(tmp_path) is None


# --- beyond the plan ---------------------------------------------------------------------------------


def test_params_env_and_targets_are_recorded(report):
    rep = report[1]
    assert rep["params"] == {"n": 4, "seed": 0, "biomes": ["forest"], "size": 512, "steps": 4}
    env = rep["env"]
    assert env["generator"] == "ProceduralGenerator" and env["python"] == platform.python_version()
    assert env["gpu"] is None and env["vram_gb"] is None and env["torch"] and env["diffusers"]
    assert rep["targets_2025"] == TARGETS_2025 and rep["fid_kid"] is None and rep["allocator"] is None
    assert rep["vram"] == {"peak_allocated_gb": None, "peak_reserved_gb": None}


def test_notes_carry_the_reference_set_caveat(report):
    notes = report[1]["notes"]
    assert notes and all(isinstance(n, str) for n in notes)
    assert any(str(REFERENCE_DIR) in n for n in notes)


def test_two_biomes_give_a_positive_style_contrast(tmp_path, monkeypatch):
    monkeypatch.setenv("REALMWEAVER_DEVICE", "cpu")
    path = run_bench(
        ProceduralGenerator(),
        ["forest", "desert"],
        n=4,
        seed=1,
        out_dir=tmp_path,
        with_fid=False,
        pool_ab=False,
    )
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert rep["params"]["biomes"] == ["forest", "desert"] and rep["style_consistency"] > 0
    assert -1.0 <= rep["style_consistency"] <= 1.0


def test_kid_fid_skips_without_reference_images(tmp_path):
    fakes = [
        ProceduralGenerator().generate(AssetSpec("forest", subject="grass", size=32, seed=s)).image
        for s in range(2)
    ]
    out = kid_fid(tmp_path, fakes, n=2, seed=0)
    assert set(out) == {"skipped"} and str(tmp_path) in out["skipped"]


@pytest.mark.gpu
def test_kid_fid_reports_fid_and_kid_at_fixed_n():
    torch = pytest.importorskip("torch")
    if not (settings().device.startswith("cuda") and torch.cuda.is_available()):
        pytest.skip("needs CUDA (and REALMWEAVER_DEVICE not forced to cpu)")
    g = ProceduralGenerator()
    fakes = [g.generate(AssetSpec("forest", subject="grass", size=64, seed=s)).image for s in range(8)]
    out = kid_fid(REFERENCE_DIR, fakes, n=8, seed=0)
    assert set(out) == {"fid", "kid_mean", "kid_std", "n"} and out["n"] == 8
    assert out["fid"] >= 0 and out["kid_std"] >= 0 and all(abs(v) < float("inf") for v in out.values())
    assert kid_fid(REFERENCE_DIR, fakes, n=8, seed=0)["fid"] == pytest.approx(out["fid"])
