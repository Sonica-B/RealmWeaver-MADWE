# RealmWeaver MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stub repo with a working `realmweaver` package: few-step SD1.5 asset agent with biome LoRAs and seamless textures, numpy WFC layouts, a world state graph with predictive prewarm and memory pooling, a FastAPI bridge with an operator page and Unity client scripts, and a benchmark that measures every claim.

**Architecture:** One flat package with four deep modules behind four seams — `Generator.generate(spec)`, `layout.solve(...)`, `World`, and the bridge routes. Agents are roles over modules, not a class hierarchy. GPU code is optional everywhere; tests run on CPU with the procedural generator and a histogram embedder.

**Tech Stack:** Python ≥3.11, torch 2.11+cu128, diffusers 0.41, peft 0.21, transformers 5.x, numpy, networkx, PyYAML, FastAPI + uvicorn + websockets, torchmetrics (FID/KID), Pillow, pytest, ruff, uv, Docker (`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`), Unity 6.3 LTS C#.

**Spec:** `docs/superpowers/specs/2026-10-06-realmweaver-mvp-design.md` (plus `GLOSSARY.md`, `CODING_STANDARDS.md`, `docs/adr/*`, `docs/research/*`).

## Global Constraints

- Python ≥3.11; torch stays `2.11.0+cu128`; never add `--system-site-packages`.
- Package root is `realmweaver/` at repo root; tests in `tests/`; CPU test suite must finish in < 60 s; GPU tests carry `@pytest.mark.gpu` and skip without CUDA.
- Images cross seams as `numpy.ndarray` uint8 `HxWx3` (texture) or `HxWx4` (sprite); PIL only at file/HTTP edges.
- Vocabulary from `GLOSSARY.md`; no "service/component/boundary" words; deliberate shortcuts carry `# ponytail:` comments.
- No hard-coded machine paths, no secrets, no `print` in library code, no literal performance numbers anywhere.
- Default models: `stable-diffusion-v1-5/stable-diffusion-v1-5` + `ByteDance/Hyper-SD` (`Hyper-SD15-4steps-lora.safetensors` for draft, `Hyper-SD15-8steps-CFG-lora.safetensors` for refine); no gated models on the default path.
- Commits: conventional (`feat:`, `test:`, `fix:`, `docs:`), each task ends with a commit on branch `feat/mvp-2026`, message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Each task agent owns only the files listed in its task; shared files (`pyproject.toml`, `realmweaver/types.py`, `realmweaver/biomes/__init__.py`) are owned by Task 0 and changed only by the lead.

---

## Execution waves

| Wave | Tasks (parallel) | Depends on |
|---|---|---|
| 0 | Task 0 skeleton (lead) | — |
| 1 | Task 1 layout · Task 2 procedural assets + metrics core · Task 3 diffusion generator (GPU) · Task 6 Unity client | Task 0 |
| 2 | Task 4 world · Task 7 benchmark | Tasks 1, 2 |
| 3 | Task 5 bridge + operator page · Task 8 CLI | Tasks 4, 7 |
| 4 | Task 9 integration, Docker verification, README, two-axis review, final bench | all |

---

### Task 0: Skeleton (lead only)

**Files:**
- Create: `pyproject.toml`, `realmweaver/__init__.py`, `realmweaver/types.py`, `realmweaver/config.py`, `realmweaver/biomes/__init__.py`, `realmweaver/biomes/forest/biome.yaml` (other five biomes: Task 1), `tests/conftest.py`, `.github/workflows/ci.yml`, `Dockerfile`, `compose.yaml`, `.dockerignore`
- Remove from git: `src/`, `scripts/`, `configs/`, `setup.py`, `requirements.txt`, `.pre-commit-config.yaml`, `data/processed/`

**Interfaces (Produces, used by every task):**

```python
# realmweaver/types.py
Kind = Literal["texture", "sprite"]; Tier = Literal["draft", "refine"]
@dataclass(frozen=True)
class AssetSpec:
    biome: str; kind: Kind = "texture"; subject: str = ""; size: int = 512
    seed: int = 0; steps: int = 4; seamless: bool = True; tier: Tier = "draft"
    def canonical(self) -> str            # sorted-key JSON of all fields
    @property
    def id(self) -> str                   # sha1(canonical)[:16]
@dataclass
class Asset:
    spec: AssetSpec; image: np.ndarray; style_vec: np.ndarray | None = None; latency_s: float = 0.0
    @property
    def id(self) -> str                   # == spec.id
class Generator(Protocol):
    def generate(self, spec: AssetSpec) -> Asset: ...
@dataclass
class TileSet:
    classes: list[str]; allowed: np.ndarray  # bool [T,4,T] dirs N,E,S,W ; weights: np.ndarray float [T]
    def index(self, name: str) -> int
@dataclass
class Layout:
    grid: np.ndarray                       # int32 HxW, values index tileset.classes
    tileset: TileSet
    def class_at(self, x: int, y: int) -> str
@dataclass
class Chunk:
    cx: int; cy: int; biome: str; layout: Layout
    asset_ids: dict[str, str] = field(default_factory=dict)   # tile class -> asset id
    state: Literal["pending", "draft", "ready"] = "pending"
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))   # N, E, S, W as (dx, dy); opposite(d) = (d + 2) % 4
```

```python
# realmweaver/biomes/__init__.py
@dataclass
class TileClass: name: str; prompt: str; palette: list[str]
@dataclass
class Biome:
    name: str; style: str; negative: str; coherence_threshold: float
    legend: dict[str, str]                 # map char -> tile class
    tiles: dict[str, TileClass]; props: dict[str, str]; lora: str | None
    example_map: str                       # the `map:` block of biome.yaml
def biome_names() -> list[str]
def load_biome(name: str) -> Biome        # raises KeyError for unknown name
```

```python
# realmweaver/config.py
@dataclass
class Settings:
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    fast_lora_repo: str = "ByteDance/Hyper-SD"
    draft_lora_file: str = "Hyper-SD15-4steps-lora.safetensors"
    refine_lora_file: str = "Hyper-SD15-8steps-CFG-lora.safetensors"
    device: str = "cuda" if available else "cpu"
    cache_bytes: int = 512 * 1024 * 1024
    reports_dir: Path = Path("reports")
    chunk_size: int = 16
def settings() -> Settings                # reads REALMWEAVER_* env overrides on each call
```

Biome YAML schema (`realmweaver/biomes/forest/biome.yaml`): keys `name, style, negative, coherence_threshold, lora, legend (char -> class), tiles (class -> {prompt, palette[]}), props (name -> prompt), map (12x12 ASCII block; every adjacent pair is an allowed adjacency)`. See the forest file for the canonical example.

- [ ] Steps (lead): write the files above; `uv sync --extra dev`; `pytest` runs an empty suite green; remove the 2025 tree from git; commit `chore: new realmweaver skeleton, remove 2025 stubs`.

### Task 1: Layout agent (WFC)

**Files:**
- Create: `realmweaver/layout/__init__.py`, `realmweaver/layout/tileset.py`, `realmweaver/layout/wfc.py`, `realmweaver/layout/render.py`
- Create: `realmweaver/biomes/{desert,snow,volcanic,underwater,sky}/biome.yaml` maps if missing (legend + 12x12 map; forest exists)
- Test: `tests/test_layout.py`

**Interfaces:**
- Consumes: `TileSet`, `Layout`, `DIRS` from `realmweaver.types`; `load_biome`.
- Produces:
  - `tileset_from_example(map_text: str, legend: dict[str, str]) -> TileSet` — `allowed[a, d, b]` is True iff class `b` appears in direction `d` of class `a` somewhere in the map; `weights[t]` = count of `t` / total.
  - `class Contradiction(RuntimeError)`
  - `solve(tileset: TileSet, width: int, height: int, seed: int, fixed: dict[tuple[int, int], str] | None = None, max_restarts: int = 5) -> Layout`
  - `solve_chunk(tileset, size: int, seed: int, neighbours: dict[str, Layout]) -> Layout` — `neighbours` keys `"N","E","S","W"`; copies the touching edge row/column of each ready neighbour into `fixed` so that `allowed` holds across the border (the fixed cells are the *neighbour's* edge classes placed one step outside; implement by constraining the solver's border cells to classes allowed next to them).
  - `violations(layout: Layout) -> int` — independent brute-force count of adjacent pairs not in `allowed` (used by tests, bench and the critic role).
  - `render(layout: Layout, palettes: dict[str, str], cell: int = 8) -> np.ndarray` uint8 HxWx3 preview using the first palette colour per class.

Algorithm notes (not negotiable): wave `bool[H, W, T]`; entropy = Shannon entropy over allowed weights + tiny seeded noise; propagation keeps a stack and uses `allowed[t, d]` OR-reduced over the source cell's possibilities (boolean matmul `wave[y,x] @ allowed[:, d, :]`); on contradiction raise, outer loop restarts with `seed + k`; after `max_restarts` solve the centre `(w//2, h//2)` block and raise `Contradiction` only if that fails too.

- [ ] **Step 1: failing tests**

```python
# tests/test_layout.py
import numpy as np, pytest
from realmweaver.layout import tileset_from_example, solve, solve_chunk, violations, Contradiction
from realmweaver.types import DIRS
LEGEND = {"g": "grass", "w": "water", "s": "shore"}
MAP = "gggg\ngssg\ngwsg\ngssg"          # water never touches grass directly

def test_example_adjacency_is_exactly_the_pairs_present():
    ts = tileset_from_example(MAP, LEGEND)
    g, w, s = (ts.index(n) for n in ("grass", "water", "shore"))
    assert ts.allowed[g, 2, s] and ts.allowed[s, 2, w]      # grass->S->shore, shore->S->water
    assert not ts.allowed[g, :, w].any() and not ts.allowed[w, :, g].any()
    assert ts.weights.sum() == pytest.approx(1.0) and ts.weights[g] > ts.weights[w]

def test_solve_has_zero_violations_and_is_deterministic():
    ts = tileset_from_example(MAP, LEGEND)
    a, b = solve(ts, 24, 24, seed=7), solve(ts, 24, 24, seed=7)
    assert a.grid.shape == (24, 24) and violations(a) == 0
    assert np.array_equal(a.grid, b.grid) and not np.array_equal(a.grid, solve(ts, 24, 24, seed=8).grid)

def test_fixed_cells_are_honoured():
    ts = tileset_from_example(MAP, LEGEND)
    lay = solve(ts, 8, 8, seed=1, fixed={(0, 0): "water", (7, 7): "grass"})
    assert lay.class_at(0, 0) == "water" and lay.class_at(7, 7) == "grass" and violations(lay) == 0

def test_chunk_border_matches_neighbour():
    ts = tileset_from_example(MAP, LEGEND)
    west = solve_chunk(ts, 8, seed=3, neighbours={})
    east = solve_chunk(ts, 8, seed=4, neighbours={"W": west})
    for y in range(8):
        a, b = ts.index(west.class_at(7, y)), ts.index(east.class_at(0, y))
        assert ts.allowed[a, 1, b]                                   # west cell -> E -> east cell

def test_impossible_fixed_cells_raise_contradiction():
    ts = tileset_from_example(MAP, LEGEND)
    with pytest.raises(Contradiction):
        solve(ts, 2, 1, seed=0, fixed={(0, 0): "water", (1, 0): "grass"})

def test_every_shipped_biome_solves():
    from realmweaver.biomes import biome_names, load_biome
    for name in biome_names():
        b = load_biome(name); ts = tileset_from_example(b.example_map, b.legend)
        assert violations(solve(ts, 16, 16, seed=0)) == 0
```

- [ ] **Step 2:** `pytest tests/test_layout.py -v` → fails with `ModuleNotFoundError`.
- [ ] **Step 3:** implement `tileset.py` (`tileset_from_example`), `wfc.py` (`solve`, `solve_chunk`, `violations`, `Contradiction`), `render.py`; export from `__init__.py`.
- [ ] **Step 4:** `pytest tests/test_layout.py -v` → all pass; `ruff check realmweaver tests` clean.
- [ ] **Step 5:** commit `feat(layout): numpy WFC with example-map tilesets and chunk borders`.

### Task 2: Procedural generator, seamless utilities, quality metrics core (CPU)

**Files:**
- Create: `realmweaver/assets/__init__.py`, `realmweaver/assets/procedural.py`, `realmweaver/assets/prompts.py`, `realmweaver/assets/alpha.py`
- Create: `realmweaver/metrics/__init__.py`, `realmweaver/metrics/quality.py`
- Test: `tests/test_assets_procedural.py`, `tests/test_metrics_quality.py`

**Interfaces:**
- Consumes: `AssetSpec`, `Asset`, `Generator` from `realmweaver.types`; `load_biome`.
- Produces:
  - `class ProceduralGenerator:  def __init__(self, seed_salt: int = 0); def generate(self, spec: AssetSpec) -> Asset` — deterministic value-noise blended from the tile class palette (`biome.tiles[spec.subject].palette`, fallback grey); textures are periodic by construction (noise sampled on a torus); sprites return HxWx4 with a centred blob and transparent corners; `latency_s` measured.
  - `build_prompt(biome: Biome, spec: AssetSpec) -> tuple[str, str]` — `(positive, negative)`: `f"{tile.prompt}, {biome.style}, seamless tileable texture"` for textures, `f"{biome.props[subject]}, {biome.style}"` for sprites; negative = `biome.negative`.
  - `alpha_from_white(rgb: np.ndarray, tol: int = 40) -> np.ndarray` HxWx4 — alpha 0 where colour distance to white < tol, else 255 *(ponytail: naive keying)*.
  - `tileability(img: np.ndarray) -> float` — mean |∇| across the wrapped seams (last col vs first col, last row vs first row) divided by mean |∇| of interior neighbours; 1.0 means seam looks like interior; computed on float grey.
  - `tiling_score(img: np.ndarray) -> float` — Tiled-Diffusion-style score: MSE between the image rolled by half its size and itself at the seam band, normalised by global variance (lower is better).
  - `style_consistency(vecs_by_biome: dict[str, np.ndarray]) -> float` — mean within-biome cosine minus mean cross-biome cosine (each `np.ndarray [n, d]`, L2-normalised inside).
  - `histogram_embed(img: np.ndarray) -> np.ndarray` — 48-dim RGB histogram embedding (16 bins per channel), L2-normalised; the test/CPU stand-in for DINOv2.

- [ ] **Step 1: failing tests**

```python
# tests/test_assets_procedural.py
import numpy as np
from realmweaver.assets import ProceduralGenerator, build_prompt, alpha_from_white
from realmweaver.biomes import load_biome
from realmweaver.types import AssetSpec

def test_texture_is_uint8_rgb_of_requested_size_and_deterministic():
    g = ProceduralGenerator()
    a = g.generate(AssetSpec("forest", subject="grass", size=64, seed=3))
    b = g.generate(AssetSpec("forest", subject="grass", size=64, seed=3))
    assert a.image.shape == (64, 64, 3) and a.image.dtype == np.uint8
    assert np.array_equal(a.image, b.image) and a.id == b.id and a.latency_s >= 0

def test_different_seed_or_subject_changes_image():
    g = ProceduralGenerator(); s = AssetSpec("forest", subject="grass", size=64, seed=3)
    assert not np.array_equal(g.generate(s).image, g.generate(AssetSpec("forest", subject="grass", size=64, seed=4)).image)
    assert not np.array_equal(g.generate(s).image, g.generate(AssetSpec("forest", subject="water", size=64, seed=3)).image)

def test_sprite_has_alpha_and_transparent_corners():
    img = ProceduralGenerator().generate(AssetSpec("forest", kind="sprite", subject="mushroom", size=64)).image
    assert img.shape == (64, 64, 4) and img[0, 0, 3] == 0 and img[32, 32, 3] == 255

def test_build_prompt_mentions_tile_and_style():
    b = load_biome("forest"); pos, neg = build_prompt(b, AssetSpec("forest", subject="grass"))
    assert "grass" in pos and b.style in pos and "seamless" in pos and neg == b.negative

def test_alpha_from_white_keys_background():
    rgb = np.full((8, 8, 3), 255, np.uint8); rgb[2:6, 2:6] = (200, 30, 30)
    out = alpha_from_white(rgb)
    assert out.shape == (8, 8, 4) and out[0, 0, 3] == 0 and out[3, 3, 3] == 255
```

```python
# tests/test_metrics_quality.py
import numpy as np
from realmweaver.assets import ProceduralGenerator
from realmweaver.metrics import tileability, tiling_score, style_consistency, histogram_embed
from realmweaver.types import AssetSpec

def _seamless(): return ProceduralGenerator().generate(AssetSpec("forest", subject="grass", size=64, seed=1)).image
def _cut(img):   # destroy the seam by rolling a non-periodic gradient in
    g = np.linspace(0, 255, img.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * 0.3 + g * 0.7, 0, 255).astype(np.uint8)

def test_tileability_ranks_seamless_above_cut():
    s = _seamless(); assert tileability(s) < tileability(_cut(s)) and tileability(s) < 1.5

def test_tiling_score_lower_for_seamless():
    s = _seamless(); assert tiling_score(s) < tiling_score(_cut(s))

def test_style_consistency_is_higher_when_biomes_differ():
    rng = np.random.default_rng(0); a = rng.normal(size=(5, 8)) + 3; b = rng.normal(size=(5, 8)) - 3
    assert style_consistency({"x": a, "y": b}) > style_consistency({"x": a, "y": a + rng.normal(scale=0.01, size=(5, 8))})

def test_histogram_embed_is_unit_norm_48d():
    v = histogram_embed(_seamless()); assert v.shape == (48,) and abs(np.linalg.norm(v) - 1) < 1e-6
```

- [ ] **Step 2:** run both files → `ModuleNotFoundError`.
- [ ] **Step 3:** implement; keep `procedural.py` under 120 lines (value noise: hash-based lattice `np.sin`-free, bilinear on a torus via `np.roll`).
- [ ] **Step 4:** both files pass; `ruff` clean.
- [ ] **Step 5:** commit `feat(assets,metrics): procedural generator, prompts, alpha keying, tileability and style metrics`.

### Task 3: Diffusion generator, memory pool, LoRA training (GPU)

**Files:**
- Create: `realmweaver/assets/diffusion.py`, `realmweaver/assets/seamless.py`, `realmweaver/assets/pool.py`, `realmweaver/assets/lora.py`, `realmweaver/assets/embed.py`
- Test: `tests/test_assets_diffusion_gpu.py` (marked `gpu`), `tests/test_seamless_cpu.py`

**Interfaces:**
- Consumes: `AssetSpec`, `Asset`, `Generator`; `build_prompt`, `alpha_from_white` (Task 2); `settings()`; `tileability` (Task 2).
- Produces:
  - `set_seamless(module: torch.nn.Module, on: bool) -> int` — sets `padding_mode = "circular"` (or `"zeros"`) on every `Conv2d` with padding > 0, returns count changed.
  - `class MemoryPool: def __init__(self, size: int, batch: int = 1, device="cuda", dtype=torch.float16); latents, noise, prompt_embeds buffers; def stats(self) -> dict` (bytes held).
  - `class DiffusionGenerator(Generator)`: `__init__(self, settings: Settings | None = None, pool: bool = True, compile: bool = False, biome_loras: dict[str, str] | None = None)`; `generate(spec) -> Asset`; `warmup(biome: str) -> None`; `allocator_stats() -> dict` (`max_memory_allocated`, `max_memory_reserved`, `num_alloc_retries`, `num_device_alloc` from `torch.cuda.memory_stats()`), `reset_stats()`.
    - Loads SD1.5 fp16 once, `safety_checker=None`, weights resident (no cpu offload); loads both Hyper-SD LoRAs as adapters `"draft"`/`"refine"` and switches with `set_adapters`; stacks a biome LoRA adapter (name = biome) when configured; scheduler per Hyper-SD card (TCD/LCM-style for 4-step, DDIM with `timestep_spacing="trailing"` for 8-step CFG, guidance 1.0 draft / 5.0 refine); `spec.seamless` toggles `set_seamless` on UNet and VAE; sprites run with `seamless=False` and get `alpha_from_white`; generator seeded by `spec.seed`; pool supplies `latents=` and reused embeds when `pool=True`.
  - `train_biome_lora(biome: str, image_dir: Path, out_dir: Path, rank: int = 8, steps: int = 200, lr: float = 1e-4, resolution: int = 512) -> Path` — PEFT LoRA on UNet attention (`to_q,to_k,to_v,to_out.0`), caption = `build_prompt` positive, saves with `pipe.unet.save_attn_procs`-compatible diffusers format (`StableDiffusionPipeline.save_lora_weights`) so `load_lora_weights(out_dir)` works.
  - `class DinoEmbedder: def __init__(self, model_id="facebook/dinov2-small", device=None); def __call__(self, img: np.ndarray) -> np.ndarray` (384-d, L2-normalised; CLS token).

- [ ] **Step 1: tests**

```python
# tests/test_seamless_cpu.py
import torch
from realmweaver.assets.seamless import set_seamless
def test_set_seamless_flips_every_padded_conv():
    m = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Conv2d(4, 4, 1), torch.nn.Conv2d(4, 4, 3, padding=1))
    assert set_seamless(m, True) == 2 and all(c.padding_mode == "circular" for c in m if isinstance(c, torch.nn.Conv2d) and c.padding[0])
    assert set_seamless(m, False) == 2 and m[0].padding_mode == "zeros"
```

```python
# tests/test_assets_diffusion_gpu.py
import numpy as np, pytest, torch
from realmweaver.assets.diffusion import DiffusionGenerator
from realmweaver.metrics import tileability
from realmweaver.types import AssetSpec
pytestmark = pytest.mark.gpu
if not torch.cuda.is_available(): pytest.skip("needs CUDA", allow_module_level=True)

@pytest.fixture(scope="module")
def gen(): g = DiffusionGenerator(); g.warmup("forest"); return g

def test_texture_is_seamless_and_fast(gen):
    a = gen.generate(AssetSpec("forest", subject="grass", seed=1))
    assert a.image.shape == (512, 512, 3) and tileability(a.image) < 1.3 and a.latency_s < 3.0

def test_same_seed_same_image(gen):
    s = AssetSpec("forest", subject="rock", seed=5)
    assert np.array_equal(gen.generate(s).image, gen.generate(s).image)

def test_sprite_has_transparent_corner(gen):
    a = gen.generate(AssetSpec("forest", kind="sprite", subject="mushroom", seed=2))
    assert a.image.shape[2] == 4 and a.image[0, 0, 3] == 0

def test_pool_makes_zero_new_device_allocs_per_image(gen):
    gen.generate(AssetSpec("forest", subject="grass", seed=9)); gen.reset_stats()
    gen.generate(AssetSpec("forest", subject="grass", seed=10)); s = gen.allocator_stats()
    assert s["num_alloc_retries"] == 0
```

- [ ] **Step 2:** `pytest tests/test_seamless_cpu.py -v` fails; GPU file skipped on CPU.
- [ ] **Step 3:** implement; run `pytest -m gpu -v` on this machine (RTX 5070 Ti) and record latency, peak VRAM, allocator stats for pool on/off into `docs/research/05-gpu-measurements.md` (raw numbers only).
- [ ] **Step 4:** `train_biome_lora("forest", Path("data/raw/textures/fantasy_forest"), Path("models/lora/forest"), steps=200)` runs end to end on the GPU in under ~10 min; the saved LoRA loads in `DiffusionGenerator(biome_loras={"forest": "models/lora/forest"})` and `tileability` stays < 1.3.
- [ ] **Step 5:** commit `feat(assets): SD1.5 Hyper-SD generator with circular padding, memory pool and biome LoRA training`.

### Task 4: World state graph, predictor, scheduler

**Files:**
- Create: `realmweaver/world/__init__.py`, `realmweaver/world/graph.py`, `realmweaver/world/predictor.py`, `realmweaver/world/scheduler.py`, `realmweaver/world/world.py`
- Test: `tests/test_world_graph.py`, `tests/test_predictor.py`, `tests/test_scheduler.py`, `tests/test_world.py`

**Interfaces:**
- Consumes: `Generator`, `AssetSpec`, `Asset`, `Chunk`, `Layout`; `tileset_from_example`, `solve_chunk`, `violations` (Task 1); `ProceduralGenerator`, `histogram_embed` (Task 2); `load_biome`, `settings()`.
- Produces:
  - `class WorldStateGraph` (wraps `networkx.MultiDiGraph`): `add_region(name, biome) `, `add_chunk(chunk: Chunk, region: str)`, `add_asset(asset: Asset, chunk_key: tuple[int,int], tile_class: str)`, `region_style(region) -> np.ndarray | None` (EMA α=0.2 of asset style vecs), `coherence(asset_id) -> float` (0.6·cos(asset, region style) + 0.4·mean cos(asset, assets of ADJACENT tiles)), `neighbours(cx, cy) -> dict[str, Chunk]`, `to_json() -> dict`, `@classmethod from_json(d)`, `validate() -> list[str]` (every Tile has exactly one INSTANCE_OF, every Chunk one CONTAINS parent, ADJACENT edges match layout).
  - `class Predictor: observe(pos: tuple[float, float]) -> None; rank(current: tuple[int,int], k: int = 8) -> list[tuple[tuple[int,int], float]]` — order-2 Markov over 8 quantised headings (fallback order-1, then constant velocity), returns neighbour chunks with probabilities summing to ≤1; `ring_baseline(current) -> list[...]` uniform 1/8 over the 8 neighbours.
  - `class Scheduler: __init__(cost_s: float = 1.0, max_in_flight: int = 2, cache_bytes: int)`, `submit(key, priority, bytes_hint)`, `next() -> key | None`, `done(key, nbytes)`, `touch(key)`, `evict_if_needed(current) -> list[key]` (LRU by bytes, evicting far chunks first), `in_flight -> int`.
  - `class World: __init__(biome: str, generator: Generator, embed: Callable[[np.ndarray], np.ndarray] = histogram_embed, chunk_size: int = 16, seed: int = 0, cache_bytes: int | None = None)`; `request_chunk(cx, cy, tier="draft") -> Chunk` (solves layout with ready neighbours, generates one asset per distinct tile class with `AssetSpec(biome, subject=cls, seed=hash(seed,cx,cy,cls))`, checks coherence, regenerates ≤2 with new seed, else anchors to the region's best asset, records latency), `asset(asset_id) -> Asset`, `observe_player(x, y) -> None`, `tick(budget: int = 1) -> list[tuple[int,int]]` (prewarm by predictor via scheduler), `save(path)`, `load(path, generator)`, `stats() -> dict` (chunks, assets, cache bytes, prewarm hits/misses).

- [ ] **Step 1: tests (abridged — write all four files)**

```python
# tests/test_world.py
import numpy as np
from realmweaver.assets import ProceduralGenerator
from realmweaver.layout import violations
from realmweaver.world import World

def test_request_chunk_maps_every_tile_class_to_an_asset():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=1)
    c = w.request_chunk(0, 0)
    classes = {c.layout.class_at(x, y) for x in range(8) for y in range(8)}
    assert classes <= set(c.asset_ids) and violations(c.layout) == 0 and c.state == "draft"
    assert all(w.asset(i).image.shape[:2] == (512, 512) for i in c.asset_ids.values())

def test_neighbour_chunks_share_a_valid_border_and_graph_validates():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=1)
    w.request_chunk(0, 0); w.request_chunk(1, 0)
    assert w.graph.validate() == []
    ts = w.request_chunk(0, 0).layout.tileset
    a, b = w.request_chunk(0, 0).layout, w.request_chunk(1, 0).layout
    assert all(ts.allowed[ts.index(a.class_at(7, y)), 1, ts.index(b.class_at(0, y))] for y in range(8))

def test_save_and_load_round_trip_is_identical():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=2); w.request_chunk(0, 0)
    w.save("tmp_world.json"); w2 = World.load("tmp_world.json", ProceduralGenerator())
    assert np.array_equal(w2.request_chunk(0, 0).layout.grid, w.request_chunk(0, 0).layout.grid)

def test_tick_prewarms_the_chunk_ahead_of_the_player():
    w = World("forest", ProceduralGenerator(), chunk_size=8, seed=3); w.request_chunk(0, 0)
    for x in range(0, 40, 2): w.observe_player(x, 4.0)          # walking east inside chunk 0..4
    warmed = w.tick(budget=1)
    assert warmed and warmed[0][0] > 0                           # a chunk to the east
```

```python
# tests/test_predictor.py
import numpy as np, pytest
from realmweaver.world import Predictor

def _walk(seed, n=400, chunk=16):
    rng = np.random.default_rng(seed); pos = np.zeros(2); heading = rng.uniform(0, 2*np.pi); out=[]
    for _ in range(n):
        heading += rng.normal(0, 0.25); pos += np.array([np.cos(heading), np.sin(heading)]); out.append(pos.copy())
    return out

def test_markov_predictor_beats_ring_baseline_on_synthetic_walks():
    hits_m = hits_r = total = 0
    for seed in range(20):
        p = Predictor(); path = _walk(seed); cur = None
        for pos in path:
            ck = (int(pos[0] // 16), int(pos[1] // 16))
            if cur is not None and ck != cur:
                total += 1; hits_m += ck in [c for c, _ in p.rank(cur, k=3)]; hits_r += ck in [c for c, _ in p.ring_baseline(cur)[:3]]
            p.observe(tuple(pos)); cur = ck
    assert total > 100 and hits_m / total > hits_r / total + 0.15
```

Scheduler tests: never more than `max_in_flight` keys returned by `next()` before `done`; `evict_if_needed` evicts the farthest LRU key once bytes exceed the cap; `submit` with higher priority is returned first. Graph tests: `validate()` catches a tile without asset; `coherence` ∈ [-1, 1]; `to_json`/`from_json` round-trip equality of nodes and edges.

- [ ] **Step 2:** run → fail. **Step 3:** implement (`world.py` is the deep module; graph/predictor/scheduler are private helpers but tested through their own interfaces as listed). **Step 4:** pass + ruff. **Step 5:** commit `feat(world): world state graph, Markov prewarm predictor, byte-capped scheduler`.

### Task 5: Bridge (FastAPI + WebSocket) and operator page

**Files:**
- Create: `realmweaver/bridge/__init__.py`, `realmweaver/bridge/app.py`, `realmweaver/bridge/static/index.html` (single file, inline CSS/JS, no build step)
- Test: `tests/test_bridge.py`

**Interfaces:**
- Consumes: `World`, `AssetSpec`, `load_biome`, `biome_names`, `settings()`; latest report from `settings().reports_dir` (Task 7 schema: `reports/bench-*.json`).
- Produces: `create_app(world: World | None = None, generator: Generator | None = None) -> FastAPI` (default: `ProceduralGenerator` when CUDA is absent, `DiffusionGenerator` otherwise, chosen lazily on first request; `/health` reports which). Routes:
  - `GET /health` → `{"status":"ok","generator":"procedural|diffusion","device":...}`
  - `GET /biomes` → `[{"name","tiles":[...],"props":[...]}]`
  - `POST /generate` body `AssetSpec` fields → `{"id", "latency_s", "tileability"}`
  - `GET /asset/{id}.png` → PNG bytes (`image/png`, `Cache-Control: immutable`)
  - `GET /chunk/{cx}/{cy}?tier=draft|refine` → `{"cx","cy","biome","size","tiles":[[cls,...]...],"assets":{cls:id},"prefabs":{cls:"Prefab_"+cls},"state"}`
  - `POST /player` body `{"x","y"}` → `{"prewarmed":[[cx,cy],...]}` (calls `observe_player` then `tick(1)`)
  - `GET /report` → latest bench JSON or `{"available": false}`
  - `WS /events` → server pushes `{"type":"ready","chunk":[cx,cy],"assets":{...}}` after each chunk/prewarm completes and `{"type":"hello"}` on connect.
  - `GET /` → operator page.
- Operator page (impeccable, **Operate** mode): biome select + tile class + seed → Generate; asset grid (latest first, tileability badge, shows 2×2 tiled preview on hover); world canvas: chunks coloured by tile class, click to request a chunk, arrow keys move the player marker and POST `/player`; metrics panel reads `/report` (shows "no benchmark yet" honestly); event log from the WebSocket. Keyboard navigable, AA contrast, no numbers that are not from `/report`. Run `impeccable detect --json realmweaver/bridge/static/index.html` once at the end and fix findings.

- [ ] **Step 1: tests**

```python
# tests/test_bridge.py
import io, numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from realmweaver.assets import ProceduralGenerator
from realmweaver.bridge import create_app
from realmweaver.world import World

def _client(): return TestClient(create_app(World("forest", ProceduralGenerator(), chunk_size=8, seed=1)))

def test_health_and_biomes():
    c = _client(); assert c.get("/health").json()["generator"] == "procedural"
    assert any(b["name"] == "forest" and "grass" in b["tiles"] for b in c.get("/biomes").json())

def test_generate_then_fetch_png():
    c = _client(); r = c.post("/generate", json={"biome": "forest", "subject": "grass", "size": 64, "seed": 1}).json()
    png = c.get(f"/asset/{r['id']}.png"); assert png.headers["content-type"] == "image/png"
    assert np.asarray(Image.open(io.BytesIO(png.content))).shape == (64, 64, 3)

def test_chunk_json_shape_and_prefab_map():
    c = _client(); j = c.get("/chunk/0/0").json()
    assert len(j["tiles"]) == 8 and set(j["assets"]) == set(j["prefabs"]) and j["prefabs"]["grass"] == "Prefab_grass"

def test_player_prewarm_and_ws_ready_event():
    c = _client(); c.get("/chunk/0/0")
    with c.websocket_connect("/events") as ws:
        assert ws.receive_json()["type"] == "hello"
        for x in range(0, 40, 2): c.post("/player", json={"x": x, "y": 4})
        assert ws.receive_json()["type"] == "ready"

def test_report_absent_is_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("REALMWEAVER_REPORTS_DIR", str(tmp_path)); c = _client()
    assert c.get("/report").json() == {"available": False}

def test_index_served():
    assert "RealmWeaver" in _client().get("/").text
```

- [ ] **Step 2:** fail. **Step 3:** implement app (keep under 250 lines) and page. **Step 4:** pass + ruff + impeccable detect. **Step 5:** commit `feat(bridge): FastAPI asset/chunk/player routes, WebSocket events, operator page`.

### Task 6: Unity client scripts

**Files:**
- Create: `unity/com.realmweaver.client/package.json`, `unity/com.realmweaver.client/Runtime/RealmWeaver.Client.asmdef`, `unity/com.realmweaver.client/Runtime/RealmWeaverClient.cs`, `.../AssetStreamer.cs`, `.../TileCatalog.cs`, `.../ChunkRenderer.cs`, `.../RealmWeaverEvents.cs`, `unity/com.realmweaver.client/README.md`
- Test: `tests/test_unity_protocol.py` (Python-side contract test that the C# JSON models match the bridge payloads; parses the C# `[Serializable]` field names with a regex and compares with `/chunk` keys)

**Interfaces:**
- Consumes: bridge payloads from Task 5 (exact keys above). Unity 6.3 LTS APIs only: `UnityWebRequestTexture.GetTexture(url, nonReadable: true)`, `UnityWebRequest.Get`, `JsonUtility`, coroutines, `Texture2D`, `MaterialPropertyBlock`, `Object.Destroy`. WebSocket via `NativeWebSocket` package (document the git URL in package.json `dependencies` and the README; fall back to HTTP polling of `/chunk` state if absent — compile-guard with `#if REALMWEAVER_WS`).
- Produces (C#): `RealmWeaverClient : MonoBehaviour` (`baseUrl`, `GetChunk(int cx,int cy, Action<ChunkDto>)`, `GetTexture(string id, Action<Texture2D>)`), `AssetStreamer : MonoBehaviour` (priority queue keyed by chunk distance to player, ≤4 concurrent requests, `placeholder` texture, byte-capped LRU with `Destroy`), `TileCatalog : ScriptableObject` (`List<Entry{string tileClass; GameObject prefab}>`, `Resolve(string)`), `ChunkRenderer : MonoBehaviour` (instantiates prefabs per tile, applies textures through `MaterialPropertyBlock` `_BaseMap`/`_MainTex`), `RealmWeaverEvents : MonoBehaviour` (connects `/events`, swaps placeholders on `ready`). DTOs: `ChunkDto{cx,cy,biome,size,tiles(string[][] via flattened "tilesFlat"),assets(List<KV>),prefabs(List<KV>),state}` — because `JsonUtility` cannot read dictionaries or jagged arrays, the bridge also emits `"tilesFlat": [...]` and `"assetList": [{"k","v"}]`, `"prefabList": [{"k","v"}]` (Task 5 must add these three keys; record in both READMEs).

- [ ] **Step 1:** write `tests/test_unity_protocol.py` asserting every `[Serializable]` DTO field in `RealmWeaverClient.cs` is a key emitted by `/chunk` (run the bridge TestClient from Task 5) → fails (no C# files).
- [ ] **Step 2:** write the scripts (≈350 lines total), `package.json` (`"name": "com.realmweaver.client", "unity": "6000.3"`), README with install steps (Package Manager → Add from disk), scene setup, and the prefab naming convention `Prefab_<tileClass>`.
- [ ] **Step 3:** test passes; commit `feat(unity): UPM client package: chunk fetch, texture streaming, prefab catalog, ready events`.

### Task 7: Benchmark

**Files:**
- Create: `realmweaver/metrics/bench.py`, `realmweaver/metrics/fid.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `Generator` (`ProceduralGenerator` or `DiffusionGenerator`), `World`, `Predictor`, `tileability`, `tiling_score`, `style_consistency`, `histogram_embed`/`DinoEmbedder`, `settings()`.
- Produces:
  - `kid_fid(real_dir: Path, fakes: list[np.ndarray], n: int, seed: int) -> dict` → `{"fid": float, "kid_mean": float, "kid_std": float, "n": int}` via `torchmetrics.image.fid.FrechetInceptionDistance` and `KernelInceptionDistance(subset_size=min(50, n))` at 299px; CPU-capable; skipped (returns `{"skipped": reason}`) when inception weights cannot be downloaded.
  - `run_bench(generator, biomes: list[str], n: int, seed: int, out_dir: Path, with_fid: bool, pool_ab: bool) -> Path` writes `reports/bench-<YYYYMMDD-HHMMSS>.json`:
    `{"schema": 1, "timestamp", "env": {"python","torch","diffusers","gpu","vram_gb","generator"}, "params": {"n","seed","biomes","size","steps"}, "latency_s": {"p50","p95","mean"}, "assets_per_min", "tileability": {"mean_ratio","share_leq_1_2"}, "tiling_score_mean", "style_consistency", "vram": {"peak_allocated_gb","peak_reserved_gb"}, "allocator": {"pool_on": {...}, "pool_off": {...}} | null, "fid_kid": {...} | null, "wfc_ms_per_chunk", "predictor": {"markov_hit_rate","ring_hit_rate","seeds"}, "targets_2025": {"fid":32.4,"s_per_asset":2.7,"textures_per_min":120,"tileable_pct":94,"style":0.89,"vram_gb":6.2}, "notes": [...caveats incl. reference-set caveat...]}`
  - `latest_report(dir) -> dict | None`.

- [ ] **Step 1:** `tests/test_bench.py`: `run_bench(ProceduralGenerator(), ["forest"], n=4, seed=0, out_dir=tmp_path, with_fid=False, pool_ab=False)` writes a file whose JSON has every top-level key above, `latency_s.p50 >= 0`, `predictor.markov_hit_rate > predictor.ring_hit_rate`, `tileability.share_leq_1_2 == 1.0` for procedural; `latest_report(tmp_path)` returns it. → fail.
- [ ] **Step 2:** implement (`bench.py` < 200 lines; `fid.py` < 80). **Step 3:** pass + ruff. **Step 4:** commit `feat(metrics): benchmark report with latency, tileability, style, VRAM, allocator A/B, FID/KID, predictor hit-rate`.

### Task 8: CLI

**Files:**
- Create: `realmweaver/cli.py`; Modify: `pyproject.toml` `[project.scripts] realmweaver = "realmweaver.cli:main"` (lead applies)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None) -> int` with subcommands:
  - `generate --biome B --tile CLASS | --prop NAME [--seed N] [--size 512] [--tier draft|refine] [--out path.png] [--procedural]`
  - `layout --biome B --size 32 --seed N --out layout.png [--json layout.json]`
  - `world --biome B --chunks 3 --out world.png [--walk 200]` renders a stitched preview of a 3×3 chunk world (procedural unless `--gpu`)
  - `serve [--host 127.0.0.1] [--port 8008] [--procedural]`
  - `bench [--n 50] [--biomes forest,desert] [--fid] [--pool-ab] [--procedural]`
  - `train-lora --biome B --images DIR [--steps 200] [--rank 8]`
  - exit code 0 on success, 2 on usage error; never prints numbers that are not in the report it just wrote.

- [ ] **Step 1:** tests: `main(["layout","--biome","forest","--size","16","--seed","1","--out",str(p)]) == 0` and the PNG is 16·8 px square; `main(["generate","--biome","forest","--tile","grass","--procedural","--size","64","--out",str(p)])` writes a 64×64 PNG; `main(["bench","--procedural","--n","2"])` creates one `reports/bench-*.json` under `REALMWEAVER_REPORTS_DIR`; `main(["nope"]) == 2`. → fail.
- [ ] **Step 2:** implement (< 180 lines, argparse only). **Step 3:** pass + ruff. **Step 4:** commit `feat(cli): generate, layout, world, serve, bench, train-lora`.

### Task 9: Integration, Docker, docs, review, final measurements (lead + reviewers)

**Files:**
- Modify: `README.md` (rewrite), `Dockerfile`, `compose.yaml`, `.github/workflows/ci.yml`, `docs/research/05-gpu-measurements.md`, `reports/bench-*.json` (committed, one GPU run), `unity/.../README.md`

- [ ] **Step 1:** `uv run pytest -q` green on CPU in < 60 s; `uv run pytest -m gpu -q` green on the laptop.
- [ ] **Step 2:** `docker compose build && docker compose run --rm realmweaver pytest -q` green; `docker compose run --rm --gpus all realmweaver realmweaver bench --n 20 --pool-ab` writes a report inside the mounted `reports/` (proves the container sees the GPU).
- [ ] **Step 3:** `realmweaver bench --n 50 --biomes forest,desert,snow --fid --pool-ab` on the laptop; commit the report; README "Measured" table reads from it (generate the table with a tiny script `tools/report_table.py`, never by hand).
- [ ] **Step 4:** two-axis review (code-review skill) of `main...feat/mvp-2026`: Standards axis vs `CODING_STANDARDS.md` + smell baseline, Spec axis vs this spec; fix every hard violation and missing requirement; re-run tests.
- [ ] **Step 5:** README: what it is, 60-second quickstart (`uv sync`, `realmweaver serve`, open `http://127.0.0.1:8008`), architecture diagram (Mermaid), measured table + 2025 targets, Unity install, Docker, research links, security note (token revoked by owner), licence of each model.
- [ ] **Step 6:** commit, push branch, open PR to `main` with the report numbers and the review summary.

---

## Self-review (lead)

- Spec coverage: stories 1–25 map to Tasks 1 (4–6), 2/3 (1–3, 17, 23), 4 (10–13, 24), 5 (16, 25), 6 (7–9), 7 (14, 15), 8 (CLI), 9 (18–22). Story 22 (quality tier) is satisfied by the `Generator` seam; `KleinGenerator` itself is a follow-up and documented as such in the README.
- Type consistency: `AssetSpec.subject` is the tile class for textures and the prop name for sprites everywhere; chunk keys are `(cx, cy)` ints; directions are N,E,S,W = 0..3 with `DIRS` from `realmweaver.types`; `allowed[a, d, b]` reads "b may sit in direction d of a".
- Placeholder scan: none; every task lists concrete tests and signatures. Implementations are written by the task agent under TDD because the interfaces, tests and algorithms above fix the behaviour.
