"""Benchmark: one `reports/bench-<timestamp>.json` per run holding every number the README or dashboard may show.

ADR-0005: nothing is claimed that this module did not measure. Latency is the generator's own per-asset time
after uncounted warm-ups, throughput is wall-clock over the timed loop, quality from `realmweaver.metrics` on
the generated textures, VRAM and allocator counters from `allocator_stats`, WFC and predictor from seeded loops.
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from realmweaver.biomes import load_biome
from realmweaver.config import settings
from realmweaver.layout import solve, tileset_from_example
from realmweaver.metrics.fid import kid_fid, reference_images
from realmweaver.metrics.quality import histogram_embed, style_consistency, tileability, tiling_score
from realmweaver.types import Asset, AssetSpec, Generator
from realmweaver.world import Predictor

log = logging.getLogger(__name__)

SCHEMA = 1
# ponytail: the FID/KID reference set is a path relative to the repo root, where the bench runs from; a
# `Settings.reference_dir` beside `reports_dir` is the upgrade path.
REFERENCE_DIR = Path("data/raw/textures")
# The 2025 pitch numbers (plan, ADR-0005): written into every report as targets beside the measurements.
TARGETS_2025 = dict(fid=32.4, s_per_asset=2.7, textures_per_min=120, tileable_pct=94, style=0.89, vram_gb=6.2)
_WARMUPS = 3  # uncounted generations before the timed loop
_TILEABLE_MAX = 1.2  # tileability ratio at or below which a texture counts as tileable
_WFC_SEEDS = 20
_WALK_SEEDS = 20
_WALK_STEPS = 400
_HEADING_NOISE = 0.25  # sd of the per-step heading change on a synthetic walk (tests/test_predictor.py)
_TOP_K = 3  # a hit: the chunk entered next is among the predictor's top-k
_GB = float(2**30)


def _specs(biomes: list[str], n: int, seed: int, size: int) -> list[AssetSpec]:
    """`n` draft seamless textures cycling the biomes and, within a biome, its tile classes; seeds `seed + i`."""
    # ponytail: textures in the draft tier only; sprites and the refine tier are not benchmarked. A tier/kind
    # parameter here is the upgrade path.
    tiles = {b: load_biome(b).tile_classes() for b in biomes}
    out = []
    for i in range(n):
        b, k = biomes[i % len(biomes)], i // len(biomes)
        out.append(AssetSpec(b, subject=tiles[b][k % len(tiles[b])], size=size, seed=seed + i))
    return out


def _timed_run(generator: Generator, specs: list[AssetSpec]) -> tuple[list[Asset], float, dict | None]:
    """Warm up, then generate `specs` in one timed loop: (assets, wall seconds, allocator stats or None).

    The stats add `resident_bytes`, allocated before the loop, and `reserved_minus_allocated_bytes`, the mean over
    the loop's images of the allocator's reserved minus allocated bytes read right after each one.
    """
    if callable(getattr(generator, "warmup", None)):
        for b in dict.fromkeys(s.biome for s in specs):
            generator.warmup(b)
    for i in range(_WARMUPS):
        generator.generate(specs[i % len(specs)])
    if not callable(getattr(generator, "allocator_stats", None)):
        t0 = time.perf_counter()
        assets = [generator.generate(s) for s in specs]
        return assets, time.perf_counter() - t0, None
    import torch  # already imported by any generator that has allocator stats

    generator.reset_stats()
    device, assets, gaps = generator.device, [], []
    resident = int(torch.cuda.memory_allocated(device))
    t0 = time.perf_counter()
    for s in specs:
        assets.append(generator.generate(s))
        gaps.append(torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    wall = time.perf_counter() - t0
    stats = {"resident_bytes": resident, **generator.allocator_stats()}
    return assets, wall, {**stats, "reserved_minus_allocated_bytes": int(statistics.fmean(gaps))}


def _allocator_ab(make_generator: Callable[[bool], Generator], specs: list[AssetSpec]) -> dict[str, dict]:
    """The two allocator arms, pool on then off: each a fresh generator from `make_generator`, measured like the
    main run and released (with the allocator cache emptied) before the next one is built."""
    arms = {}
    for pool in (True, False):
        arm = make_generator(pool)
        assets, _, stats = _timed_run(arm, specs)
        if stats is None:
            raise RuntimeError(
                f"make_generator({pool}) built {type(arm).__name__}, which has no allocator stats"
            )
        import torch  # the arm has allocator stats, so torch is already imported

        del arm
        gc.collect()
        torch.cuda.empty_cache()
        arms["pool_on" if pool else "pool_off"] = {**stats, "latency_s": _latency(assets), "n": len(assets)}
    return arms


def _latency(assets: list[Asset]) -> dict[str, float]:
    lat = sorted(a.latency_s for a in assets)
    p95 = lat[round(0.95 * (len(lat) - 1))]  # nearest rank, as docs/research/05 measured
    return {"p50": float(statistics.median(lat)), "p95": float(p95), "mean": float(statistics.fmean(lat))}


def _wfc_ms_per_chunk(biomes: list[str], size: int, seed: int) -> float:
    """Mean wall ms of one `size`x`size` solve over `_WFC_SEEDS` seeds per biome."""
    times = []
    for b in biomes:
        biome = load_biome(b)
        ts = tileset_from_example(biome.example_map, biome.legend)
        for s in range(seed, seed + _WFC_SEEDS):
            t0 = time.perf_counter()
            solve(ts, size, size, seed=s)
            times.append(time.perf_counter() - t0)
    return float(statistics.fmean(times) * 1000)


def _walk(seed: int) -> np.ndarray:
    """`[_WALK_STEPS, 2]` positions of unit steps with a random-walk heading, as tests/test_predictor.py walks."""
    rng = np.random.default_rng(seed)
    heading = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, _HEADING_NOISE, _WALK_STEPS))
    return np.cumsum(np.stack([np.cos(heading), np.sin(heading)], axis=1), axis=0)


def _predictor(seed: int, chunk: int) -> dict:
    """Top-k hit-rate of the Markov predictor against the uniform 8-ring over `_WALK_SEEDS` synthetic walks."""
    hits_markov = hits_ring = total = 0
    for s in range(seed, seed + _WALK_SEEDS):
        p, cur = Predictor(chunk_size=chunk), None
        for x, y in _walk(s):
            key = (int(x // chunk), int(y // chunk))
            if cur is not None and key != cur:
                total += 1
                hits_markov += key in [c for c, _ in p.rank(cur, k=_TOP_K)]
                hits_ring += key in [c for c, _ in p.ring_baseline(cur)[:_TOP_K]]
            p.observe((float(x), float(y)))
            cur = key
    return {"markov_hit_rate": hits_markov / total, "ring_hit_rate": hits_ring / total, "seeds": _WALK_SEEDS}


def _env(name: str, device: str) -> dict:
    gpu = vram = None
    if device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(device)
            vram = torch.cuda.get_device_properties(device).total_memory / _GB
    env = {"python": platform.python_version(), "torch": version("torch"), "diffusers": version("diffusers")}
    return {**env, "gpu": gpu, "vram_gb": vram, "generator": name}


def run_bench(
    generator: Generator,
    biomes: list[str],
    n: int,
    seed: int,
    out_dir: Path,
    with_fid: bool,
    embed: Callable[[np.ndarray], np.ndarray] | None = None,
    make_generator: Callable[[bool], Generator] | None = None,
) -> Path:
    """Measure everything on `generator` and write `out_dir/bench-<YYYYMMDD-HHMMSS>.json`; returns its path.

    `n` is the total number of draft seamless textures, cycled over `biomes`. `embed` maps an image to a style
    vector (`DinoEmbedder` on a GPU; the histogram embedding when None). `with_fid` adds FID/KID against
    `REFERENCE_DIR`. `make_generator(pool_on)` builds one allocator A/B arm: when it is given and `generator` has
    allocator stats, both arms are measured after the main run (it is ignored otherwise).
    """
    cfg, name, stamp = settings(), type(generator).__name__, datetime.now().astimezone()
    size, chunk = cfg.asset_size, cfg.chunk_size
    specs = _specs(biomes, n, seed, size)
    log.info("bench: %d textures over %s with %s", n, biomes, name)
    assets, wall, stats = _timed_run(generator, specs)
    allocator = None
    if make_generator is not None and stats is not None:
        # ponytail: both arms run in this process with `generator`'s pipeline still resident, so their peaks
        # include those weights (subtract resident_bytes). Separate processes per arm would be cleaner.
        allocator = _allocator_ab(make_generator, specs)
    embed = embed or histogram_embed
    images = [a.payload.image for a in assets]  # the bench measures textures: seams and style need the pixels
    ratios = [tileability(img) for img in images]
    share = float(np.mean([r <= _TILEABLE_MAX for r in ratios]))
    groups = {b: [embed(a.payload.image) for a in assets if a.spec.biome == b] for b in biomes}
    vecs = {b: np.stack(v) for b, v in groups.items() if v}
    fid_kid = kid_fid(REFERENCE_DIR, images, n=len(images), seed=seed) if with_fid else None
    notes = [
        f"fid_kid reference set: {REFERENCE_DIR} ({len(reference_images(REFERENCE_DIR))} SDXL-Turbo textures "
        "from the 2025 repo, the only one available); FID is biased at small n, KID (subset mean, std) is the "
        "unbiased statistic, its std degenerate to 0 while n is at or below the subset size; neither is a "
        "ground-truth style target",
        f"latency_s: the generator's per-asset time over {len(assets)} draft seamless textures after {_WARMUPS} "
        "uncounted warm-ups; assets_per_min: textures / wall-clock of that loop",
        "tileability: wrapped-seam over interior gradient ratio, 1.0 = seamless; share_leq_1_2: share at or "
        f"below {_TILEABLE_MAX}; tiling_score_mean: Tiled-Diffusion seam score, lower is better",
        f"wfc_ms_per_chunk: mean wall ms per {chunk}x{chunk} solve, {_WFC_SEEDS} seeds per biome",
        f"predictor: top-{_TOP_K} hit-rate over {_WALK_SEEDS} synthetic walks of {_WALK_STEPS} steps, heading "
        f"noise sd {_HEADING_NOISE}; the ring baseline takes its first {_TOP_K} neighbours",
        "targets_2025 are the 2025 pitch numbers, not measurements",
    ]
    if stats is None:
        notes.append(f"vram and allocator need the diffusion generator; this run used {name}")
    if allocator is not None:
        notes.append(
            "allocator: each arm is a fresh generator from make_generator, pool on then off, run with the "
            "benchmarked generator's pipeline still resident (see resident_bytes); reserved_minus_allocated_bytes "
            "is the mean over its timed images of bytes reserved minus allocated right after each image"
        )
    report = {
        "schema": SCHEMA,
        "timestamp": stamp.isoformat(timespec="seconds"),
        "env": _env(name, cfg.device),
        "params": {
            "n": n,
            "seed": seed,
            "biomes": list(biomes),
            "size": size,
            "steps": specs[0].steps,
            "chunk": chunk,
        },
        "latency_s": _latency(assets),
        "assets_per_min": len(assets) / wall * 60,
        "tileability": {"mean_ratio": float(statistics.fmean(ratios)), "share_leq_1_2": share},
        "tiling_score_mean": float(statistics.fmean(tiling_score(img) for img in images)),
        "style_consistency": style_consistency(vecs),
        "vram": {
            "peak_allocated_gb": stats["max_memory_allocated"] / _GB if stats else None,
            "peak_reserved_gb": stats["max_memory_reserved"] / _GB if stats else None,
        },
        "allocator": allocator,
        "fid_kid": fid_kid,
        "wfc_ms_per_chunk": _wfc_ms_per_chunk(biomes, chunk, seed),
        "predictor": _predictor(seed, chunk),
        "targets_2025": dict(TARGETS_2025),
        "notes": notes,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"bench-{stamp:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def latest_report(dir: Path) -> dict | None:
    """The newest `bench-*.json` under `dir` (names sort by timestamp), or None when there is none."""
    paths = sorted(Path(dir).glob("bench-*.json"))
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None
