"""Shared types that cross the seams between layout, assets, world and bridge.

An `Asset` carries one payload for its kind: pixels for a texture (uint8 HxWx3) or a sprite (uint8 HxWx4), a GLB
with its textures for a mesh, a clip for an animation. Consumers call `encode()` (bytes plus media type) and
`preview(size)` (uint8 HxWx3), never the pixels; the metrics that need an image array take `payload.image`.
Directions are N, E, S, W = 0..3; `allowed[a, d, b]` reads "class b may sit in direction d of class a".
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import ClassVar, Literal, Protocol

import numpy as np
from PIL import Image  # the one PNG encoder: `encode` is the file and HTTP edge

from realmweaver import gltf

Kind = Literal["texture", "sprite", "mesh", "animation"]
Tier = Literal["draft", "refine"]
PNG = "image/png"

DIRS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIR_NAMES: tuple[str, ...] = ("N", "E", "S", "W")


def opposite(direction: int) -> int:
    return (direction + 2) % 4


@dataclass(frozen=True)
class AssetSpec:
    """Everything needed to (re)generate one asset. `subject` is the tile class for textures, the prop name for
    sprites, meshes and animations."""

    biome: str
    kind: Kind = "texture"
    subject: str = ""
    size: int = 512
    seed: int = 0
    steps: int = 4
    seamless: bool = True
    tier: Tier = "draft"

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def id(self) -> str:
        return sha1(self.canonical().encode("utf-8")).hexdigest()[:16]


@dataclass
class ImagePayload:
    """Pixels, uint8 HxWxC with C fixed by the kind. PNG on the wire; a box-filtered (else nearest) RGB thumbnail."""

    image: np.ndarray
    channels: ClassVar[int] = 0

    def __post_init__(self) -> None:
        shape = tuple(self.image.shape)
        if self.image.dtype != np.uint8 or len(shape) != 3 or shape[2] != self.channels:
            raise ValueError(
                f"{type(self).__name__} needs uint8 HxWx{self.channels}, got {self.image.dtype} {shape}"
            )

    @property
    def nbytes(self) -> int:
        return int(self.image.nbytes)

    def encode(self) -> tuple[bytes, str]:
        # ponytail: PIL's default compression on the calling thread; a faster encoder is the upgrade path.
        buf = io.BytesIO()
        Image.fromarray(self.image).save(buf, format="PNG")
        return buf.getvalue(), PNG

    def preview(self, size: int) -> np.ndarray:
        rgb, (h, w) = self.image[..., :3], self.image.shape[:2]
        if (h, w) == (size, size):
            return rgb
        if h % size == 0 and w % size == 0:  # box filter
            blocks = rgb.reshape(size, h // size, size, w // size, 3)
            return blocks.mean(axis=(1, 3)).round().astype(np.uint8)
        return rgb[np.arange(size) * h // size][:, np.arange(size) * w // size]


@dataclass
class TexturePayload(ImagePayload):
    """A seamless tile texture: uint8 HxWx3."""

    channels: ClassVar[int] = 3


@dataclass
class SpritePayload(ImagePayload):
    """An alpha-keyed prop sprite: uint8 HxWx4, white under transparent pixels."""

    channels: ClassVar[int] = 4


@dataclass
class MeshPayload:
    """A triangle mesh as glTF binary, its textures also held as uint8 HxWx3 arrays keyed by material slot (for
    example "baseColor"), and what the generator measured: `triangles` and whether the surface is `watertight`."""

    glb_bytes: bytes
    triangles: int
    watertight: bool
    textures: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return len(self.glb_bytes) + sum(int(t.nbytes) for t in self.textures.values())

    def encode(self) -> tuple[bytes, str]:
        return self.glb_bytes, gltf.MEDIA_TYPE

    def preview(self, size: int) -> np.ndarray:
        # ponytail: a flat-shaded silhouette of the GLB's first primitive in the base colour texture's mean colour;
        # a textured render of every primitive is the upgrade path.
        base = self.textures.get("baseColor")
        tint = (
            None if base is None else tuple(int(v) for v in base.reshape(-1, base.shape[-1])[:, :3].mean(0))
        )
        return gltf.silhouette(self.glb_bytes, size, tint)


@dataclass
class AnimationPayload:
    """A skeletal clip as glTF binary (its skin and animation channels; the bridge carries GLB, never FBX):
    `frames` keyframes sampled at `fps`."""

    clip_bytes: bytes
    fps: float
    frames: int

    @property
    def nbytes(self) -> int:
        return len(self.clip_bytes)

    def encode(self) -> tuple[bytes, str]:
        return self.clip_bytes, gltf.MEDIA_TYPE

    def preview(self, size: int) -> np.ndarray:
        # ponytail: the rest pose's silhouette, the background alone for a skeleton-only clip; posing the skin at
        # the clip's middle frame is the upgrade path.
        return gltf.silhouette(self.clip_bytes, size)


Payload = TexturePayload | SpritePayload | MeshPayload | AnimationPayload
_PAYLOAD_KIND: dict[type, Kind] = {
    TexturePayload: "texture",
    SpritePayload: "sprite",
    MeshPayload: "mesh",
    AnimationPayload: "animation",
}


@dataclass
class Asset:
    """One generated asset: the spec that made it and the payload of its kind. `encode` and `preview` are how the
    bridge, the CLI and the embedders read it; `image` stays for texture and sprite callers."""

    spec: AssetSpec
    payload: Payload
    style_vec: np.ndarray | None = None
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        if _PAYLOAD_KIND.get(type(self.payload)) != self.spec.kind:
            raise TypeError(f"a {self.spec.kind} spec cannot carry a {type(self.payload).__name__}")

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def nbytes(self) -> int:
        return self.payload.nbytes

    @property
    def image(self) -> np.ndarray:
        """The pixels of a texture or sprite; a mesh or animation has none (`preview` renders one)."""
        if isinstance(self.payload, ImagePayload):
            return self.payload.image
        raise TypeError(f"a {self.spec.kind} asset has no image; call preview() or encode()")

    def encode(self) -> tuple[bytes, str]:
        """The asset as bytes for a file or the wire, with its media type: PNG for pixels, GLB otherwise."""
        return self.payload.encode()

    def preview(self, size: int) -> np.ndarray:
        """A uint8 `size` x `size` x 3 thumbnail: the pixels resampled, or a mesh or clip rendered."""
        return self.payload.preview(size)


class Generator(Protocol):
    """The asset agent seam: a spec of any kind in, an asset carrying that kind's payload out. Two adapters:
    DiffusionGenerator (GPU: textures and sprites) and ProceduralGenerator (CPU: textures, sprites and a
    placeholder mesh). An adapter raises RuntimeError naming what it lacks for a kind it does not make."""

    def generate(self, spec: AssetSpec) -> Asset: ...


@dataclass
class TileSet:
    classes: list[str]
    allowed: np.ndarray  # bool [T, 4, T]
    weights: np.ndarray  # float [T], sums to 1

    def index(self, name: str) -> int:
        return self.classes.index(name)

    def __len__(self) -> int:
        return len(self.classes)


@dataclass
class Layout:
    grid: np.ndarray  # int32 HxW of indices into tileset.classes
    tileset: TileSet

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    def class_at(self, x: int, y: int) -> str:
        return self.tileset.classes[int(self.grid[y, x])]

    def class_rows(self) -> list[list[str]]:
        return [[self.tileset.classes[int(v)] for v in row] for row in self.grid]


@dataclass
class Chunk:
    cx: int
    cy: int
    biome: str
    layout: Layout
    asset_ids: dict[str, str] = field(default_factory=dict)  # tile class -> asset id
    state: Literal["pending", "draft", "ready"] = "pending"

    @property
    def key(self) -> tuple[int, int]:
        return (self.cx, self.cy)
