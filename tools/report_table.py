"""Render the README "Measured" table from the newest benchmark report. Never edit the numbers by hand.

Usage: uv run python tools/report_table.py [--write]   (prints the table; --write splices it into README.md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

START, END = "<!-- bench-table:start -->", "<!-- bench-table:end -->"


def _fmt(v: object, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def table(report: dict) -> str:
    t = report.get("targets_2025", {})
    lat = report.get("latency_s", {})
    tile = report.get("tileability", {})
    vram = report.get("vram", {})
    fid = report.get("fid_kid") or {}
    pred = report.get("predictor", {})
    alloc = report.get("allocator") or {}
    rows = [
        (
            "Seconds per asset (draft, p50 / p95)",
            f"{_fmt(lat.get('p50'))} / {_fmt(lat.get('p95'))}",
            _fmt(t.get("s_per_asset")),
        ),
        ("Assets per minute", _fmt(report.get("assets_per_min"), 1), _fmt(t.get("textures_per_min"))),
        (
            "Tileable share (seam ratio <= 1.2)",
            f"{100 * tile.get('share_leq_1_2', 0):.0f}%",
            f"{_fmt(t.get('tileable_pct'))}%",
        ),
        (
            "Style consistency (within - cross biome cosine)",
            _fmt(report.get("style_consistency"), 3),
            _fmt(t.get("style")),
        ),
        (
            "FID / KID vs repo textures",
            f"{_fmt(fid.get('fid'), 1)} / {_fmt(fid.get('kid_mean'), 4)} (n={fid.get('n', 'n/a')})",
            _fmt(t.get("fid")),
        ),
        ("Peak VRAM allocated (GB)", _fmt(vram.get("peak_allocated_gb")), _fmt(t.get("vram_gb"))),
        (
            "Prewarm predictor hit-rate (Markov vs 8-ring)",
            f"{_fmt(pred.get('markov_hit_rate'))} vs {_fmt(pred.get('ring_hit_rate'))}",
            "n/a",
        ),
        ("WFC solve per 16x16 chunk (ms)", _fmt(report.get("wfc_ms_per_chunk"), 1), "n/a"),
    ]
    if alloc.get("pool_on") and alloc.get("pool_off"):
        on, off = alloc["pool_on"], alloc["pool_off"]
        rows.append(
            (
                "Latency p50 with memory pool on / off (s)",
                f"{_fmt(on.get('latency_s', {}).get('p50'))} / {_fmt(off.get('latency_s', {}).get('p50'))}",
                "n/a",
            )
        )
    env = report.get("env", {})
    head = (
        f"Measured {report.get('timestamp', '')} on {env.get('gpu', 'unknown GPU')} "
        f"({env.get('generator', '?')}, n={report.get('params', {}).get('n', '?')}, "
        f"seed={report.get('params', {}).get('seed', '?')}). Source: `{report.get('_path', 'reports/')}`.\n\n"
    )
    body = "| Metric | Measured | 2025 pitch target |\n|---|---|---|\n" + "\n".join(
        f"| {a} | {b} | {c} |" for a, b, c in rows
    )
    notes = "\n\n" + "\n".join(f"- {n}" for n in report.get("notes", []))
    return head + body + notes


def main() -> int:
    reports = sorted(Path("reports").glob("bench-*.json"))
    if not reports:
        print("no reports/bench-*.json found", file=sys.stderr)
        return 1
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    report["_path"] = reports[-1].as_posix()
    text = table(report)
    if "--write" in sys.argv:
        readme = Path("README.md")
        content = readme.read_text(encoding="utf-8")
        if START not in content or END not in content:
            print("README.md lacks bench-table markers", file=sys.stderr)
            return 1
        pre, rest = content.split(START, 1)
        _, post = rest.split(END, 1)
        readme.write_text(f"{pre}{START}\n{text}\n{END}{post}", encoding="utf-8")
        print(f"README.md updated from {reports[-1].name}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
