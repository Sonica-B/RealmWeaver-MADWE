"""The `realmweaver` command: generate, layout, world, serve, bench and train-lora over the package seams.

Every number it prints was measured by the call it just made or sits in the report whose path it prints.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from realmweaver.assets import ProceduralGenerator
from realmweaver.biomes import biome_names, load_biome
from realmweaver.config import settings
from realmweaver.layout import render, solve, tileset_from_example
from realmweaver.metrics import histogram_embed, tileability
from realmweaver.types import AssetSpec, Chunk, Generator, TexturePayload
from realmweaver.world import ThreadRunner, World

log = logging.getLogger(__name__)

_STEPS = {"draft": 4, "refine": 8}  # the anytime knob per quality tier, as the world agent sets it
_TILE_PX = 32  # a world preview draws every tile as its asset downscaled to this many pixels
_ROOT = Path(__file__).resolve().parent.parent  # the checkout: `wire --write` rewrites its Unity package


def _adapters(mode: str, embed: bool = False) -> tuple[Generator, Callable[[np.ndarray], np.ndarray]]:
    """The asset agent and the style embedder for `procedural`, `auto` (diffusion + DINOv2 on a CUDA device,
    else procedural + histogram) or `gpu` (diffusion, or its RuntimeError naming the missing CUDA device)."""
    device = settings().device
    if mode == "procedural" or (mode == "auto" and not device.startswith("cuda")):
        log.info("asset agent: ProceduralGenerator + histogram_embed on cpu")
        return ProceduralGenerator(), histogram_embed
    from realmweaver.assets import DiffusionGenerator, DinoEmbedder

    generator = DiffusionGenerator()
    log.info("asset agent: DiffusionGenerator%s on %s", " + DinoEmbedder" if embed else "", device)
    return generator, DinoEmbedder() if embed else histogram_embed


def _save(path: Path, data: bytes) -> Path:
    """Create the parent and write encoded bytes: the command's one file edge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _generate(args: argparse.Namespace) -> int:
    generator, _ = _adapters("procedural" if args.procedural else "auto")
    kind, name = ("sprite", args.prop) if args.prop else ("texture", args.tile)
    steps, size, seamless = _STEPS[args.tier], args.size or settings().asset_size, kind == "texture"
    spec = AssetSpec(args.biome, kind, name, size, args.seed, steps, seamless, args.tier)
    asset = generator.generate(spec)
    data, _ = asset.encode()
    out = _save(args.out or Path(f"{args.biome}_{name}.png"), data)
    seam = f" tileability={tileability(asset.payload.image):.3f}" if seamless else ""  # a texture property
    print(f"{out} id={asset.id} latency_s={asset.latency_s:.3f}{seam}")
    return 0


def _layout(args: argparse.Namespace) -> int:
    b = load_biome(args.biome)
    layout = solve(tileset_from_example(b.example_map, b.legend), args.size, args.size, args.seed)
    image = render(layout, {name: tile.palette[0] for name, tile in b.tiles.items()})
    print(_save(args.out, TexturePayload(image).encode()[0]))  # plain pixels through the one PNG encoder
    if args.json:
        args.json.write_text(json.dumps(layout.class_rows()), encoding="utf-8")
        print(args.json)
    return 0


def _walk(steps: int, chunks: int, chunk_size: int) -> Iterator[tuple[float, float]]:
    """A synthetic player in tile units: one tile a step, clockwise round the outer chunks' centres."""
    lo, side = chunk_size // 2, max(1, (chunks - 1) * chunk_size)
    for i in range(steps):
        leg, u = divmod(i % (4 * side), side)
        x, y = ((lo + u, lo), (lo + side, lo + u), (lo + side - u, lo + side), (lo, lo + side - u))[leg]
        yield x + 0.5, y + 0.5


def _chunk_image(world: World, chunk: Chunk, thumbs: dict[str, np.ndarray]) -> np.ndarray:
    """One chunk drawn tile by tile from its assets, each downscaled once into `thumbs`."""
    px, ids = _TILE_PX, chunk.asset_ids
    for asset_id in set(ids.values()) - set(thumbs):
        thumbs[asset_id] = world.asset(asset_id).preview(px)
    blank = np.zeros((px, px, 3), np.uint8)  # for tile classes absent from this chunk
    tiles = np.stack([thumbs[ids[c]] if c in ids else blank for c in chunk.layout.tileset.classes])
    n = chunk.layout.grid.shape[0]
    return tiles[chunk.layout.grid].transpose(0, 2, 1, 3, 4).reshape(n * px, n * px, 3)


def _world(args: argparse.Namespace) -> int:
    generator, embed = _adapters("gpu" if args.gpu else "procedural", embed=True)
    world = World(args.biome, generator, embed, chunk_size=settings().chunk_size)
    if args.walk:
        for x, y in _walk(args.walk, args.chunks, world.chunk_size):
            world.observe_player(x, y)
            world.tick(budget=1)
        print(json.dumps(world.stats()))
    thumbs: dict[str, np.ndarray] = {}
    side = range(args.chunks)
    rows = [[_chunk_image(world, world.request_chunk(cx, cy).chunk, thumbs) for cx in side] for cy in side]
    image = np.concatenate([np.concatenate(row, axis=1) for row in rows])
    print(_save(args.out, TexturePayload(image).encode()[0]))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from realmweaver.bridge import create_app

    generator, embed = _adapters("procedural" if args.procedural else "auto", embed=True)
    slots = 2 if isinstance(generator, ProceduralGenerator) else 1  # one diffusion pipeline, one thread
    runner = ThreadRunner(slots)  # the bridge serves while the world generates on its own slots
    world = World(args.biome, generator, embed, chunk_size=settings().chunk_size, runner=runner)
    uvicorn.run(create_app(world=world, generator=generator), host=args.host, port=args.port)
    return 0


def _diffusion_arm(pool: bool) -> Generator:
    """One more diffusion pipeline with the memory pool on or off: the bench builds its allocator A/B arms with it."""
    from realmweaver.assets import DiffusionGenerator

    return DiffusionGenerator(pool=pool)


def _bench(args: argparse.Namespace) -> int:
    from realmweaver.metrics import run_bench

    generator, embed = _adapters("procedural" if args.procedural else "auto", embed=True)
    biomes = [name.strip() for name in args.biomes.split(",") if name.strip()]
    arms = _diffusion_arm if args.pool_ab and not args.procedural else None
    out = settings().reports_dir
    print(run_bench(generator, biomes, args.n, 0, out, with_fid=args.fid, embed=embed, make_generator=arms))
    return 0


def _train_lora(args: argparse.Namespace) -> int:
    from realmweaver.assets import train_biome_lora

    out = args.out or settings().models_dir / "lora" / args.biome
    print(train_biome_lora(args.biome, args.images, out, rank=args.rank, steps=args.steps))
    return 0


def _wire(args: argparse.Namespace) -> int:
    """Print what the wire table renders; with --write, rewrite the `<wire-generated>` regions under --root."""
    from realmweaver import wire

    for rel, renderer in wire.GENERATED.items():
        if args.write:
            path = args.root / rel
            text = wire.regenerate(path.read_text(encoding="utf-8"), renderer())
            path.write_text(text, encoding="utf-8", newline="\n")
            print(path)
        else:
            print(renderer(), end="")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="realmweaver", description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    biomes = biome_names()
    pick = f"one of {', '.join(biomes)}"

    def command(name: str, run: Callable[..., int], text: str, biome: bool = True) -> argparse.ArgumentParser:
        c = sub.add_parser(name, help=text, description=text)
        c.set_defaults(run=run)
        if biome:
            c.add_argument("--biome", choices=biomes, metavar="B", required=True, help=pick)
        return c

    cpu = {"action": "store_true", "help": "CPU generator even when CUDA is available"}
    g = command("generate", _generate, "one asset: a seamless tile texture or an alpha-keyed prop sprite")
    what = g.add_mutually_exclusive_group(required=True)
    what.add_argument("--tile", metavar="CLASS", help="tile class, e.g. grass")
    what.add_argument("--prop", metavar="NAME", help="prop name, e.g. mushroom")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--size", type=int, help="pixels per side (default: the asset_size setting, 512)")
    g.add_argument("--tier", choices=tuple(_STEPS), default="draft")
    g.add_argument("--out", type=Path, metavar="PNG", help="default: <biome>_<subject>.png")
    g.add_argument("--procedural", **cpu)

    lay = command("layout", _layout, "one WFC layout as a flat-colour PNG, 8 px per tile, plus optional JSON")
    lay.add_argument("--size", type=int, default=32, help="tiles per side")
    lay.add_argument("--seed", type=int, default=0)
    lay.add_argument("--out", type=Path, required=True, metavar="PNG")
    lay.add_argument("--json", type=Path, metavar="JSON", help="also write the rows of tile classes")

    w = command("world", _world, "a stitched preview of a chunks x chunks world drawn from its assets")
    w.add_argument("--chunks", type=int, default=3, help="chunks per side")
    w.add_argument("--out", type=Path, required=True, metavar="PNG")
    w.add_argument("--walk", type=int, default=0, metavar="N", help="walk a player N steps, then print stats")
    w.add_argument("--gpu", action="store_true", help="diffusion generator + DINOv2 (needs CUDA)")

    s = command("serve", _serve, "the bridge: operator page, chunk and asset routes, WebSocket events", False)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8008)
    s.add_argument("--procedural", **cpu)
    s.add_argument("--biome", choices=biomes, metavar="B", default="forest", help=f"{pick} (default forest)")

    b = command("bench", _bench, "write a benchmark report under reports_dir and print its path", False)
    b.add_argument("--n", type=int, default=50, help="textures in total, cycled over the biomes")
    b.add_argument("--biomes", default="forest,desert", help="comma-separated biome names")
    b.add_argument("--fid", action="store_true", help="also FID/KID against the reference textures")
    b.add_argument("--pool-ab", action="store_true", help="also allocator stats with the pool on and off")
    b.add_argument("--procedural", **cpu)

    t = command("train-lora", _train_lora, "train a biome LoRA on a folder of images (needs CUDA)")
    t.add_argument("--images", type=Path, required=True, metavar="DIR")
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--out", type=Path, metavar="DIR", help="default: <models_dir>/lora/<biome>")

    wr = command("wire", _wire, "the Unity DTOs and README contract rendered from the wire table", False)
    wr.add_argument("--write", action="store_true", help="rewrite the <wire-generated> regions in place")
    wr.add_argument("--root", type=Path, default=_ROOT, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """0 on success, 2 on a usage error (argparse), 1 on a runtime failure with one line on stderr."""
    logging.basicConfig(stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("realmweaver").setLevel(logging.INFO)
    # the world agent notes every anchored asset at INFO; `world --walk` prints the totals from `stats()`
    logging.getLogger("realmweaver.world").setLevel(logging.WARNING)
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_:  # argparse already printed the usage error (2) or the help text (0)
        return exit_.code if isinstance(exit_.code, int) else 2
    try:
        return args.run(args)
    except Exception as exc:
        print(f"realmweaver: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
