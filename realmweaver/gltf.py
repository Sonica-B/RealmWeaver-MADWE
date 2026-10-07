"""glTF 2.0 binary (GLB) with numpy only: the container a mesh or an animation payload carries across the seams.

`textured_mesh` writes one triangle primitive with its base colour PNG, `first_primitive` reads positions and
triangles back, `is_watertight` measures closure and `silhouette` renders a flat-shaded preview.
"""

from __future__ import annotations

import json
import struct

import numpy as np

MEDIA_TYPE = "model/gltf-binary"
_MAGIC = b"glTF"
_FLOAT, _USHORT, _UINT = 5126, 5123, 5125
_DTYPES = {5121: "<u1", _USHORT: "<u2", _UINT: "<u4", _FLOAT: "<f4"}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER, _REPEAT = 34962, 34963, 10497
_TRIANGLES = (
    4  # the primitive mode `first_primitive` reads (and the glTF default); points, lines, strips are refused
)


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    about_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return about_x @ about_y


# The preview camera: yaw 45 degrees then pitch 30, looking down -z, so a box shows three faces; one light from the
# upper left in front of it.
_VIEW = _rotation(np.radians(45.0), np.radians(30.0))
_LIGHT = np.array([-0.3, 0.6, 0.75]) / np.linalg.norm([-0.3, 0.6, 0.75])
_GREY = (190, 190, 190)
_BACKGROUND = (28, 30, 34)


def pack(doc: dict, binary: bytes = b"") -> bytes:
    """One GLB: the 12-byte header, the JSON chunk (space padded to 4) and the BIN chunk (zero padded)."""
    text = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    header = struct.pack("<4sII", _MAGIC, 2, 28 + len(text) + len(binary))
    json_chunk = struct.pack("<I4s", len(text), b"JSON") + text
    bin_chunk = struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return header + json_chunk + bin_chunk


def unpack(glb: bytes) -> tuple[dict, bytes]:
    """The JSON document and the BIN chunk (empty when absent) of a GLB; ValueError for anything else."""
    if len(glb) < 20 or glb[:4] != _MAGIC:
        raise ValueError("not a GLB: the glTF magic is missing")
    length, offset, chunks = min(struct.unpack_from("<I", glb, 8)[0], len(glb)), 12, {}
    while offset + 8 <= length:
        n, kind = struct.unpack_from("<I4s", glb, offset)
        chunks[kind] = glb[offset + 8 : offset + 8 + n]
        offset += 8 + n
    if b"JSON" not in chunks:
        raise ValueError("not a GLB: the JSON chunk is missing")
    return json.loads(chunks[b"JSON"]), chunks.get(b"BIN\0", b"")


def textured_mesh(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
    png: bytes,
    name: str = "mesh",
) -> bytes:
    """A GLB holding one node with one triangle mesh (float32 positions, normals and uvs; uint16 or uint32 indices)
    and one metallic-roughness material whose base colour is `png`, wrapped with REPEAT so a seamless texture tiles."""
    positions, normals, uvs = (np.ascontiguousarray(a, dtype="<f4") for a in (positions, normals, uvs))
    indices = np.ascontiguousarray(indices, dtype="<u2" if len(positions) <= 0xFFFF else "<u4")
    views: list[dict] = []
    blob = bytearray()
    for data, target in (
        (positions.tobytes(), _ARRAY_BUFFER),
        (normals.tobytes(), _ARRAY_BUFFER),
        (uvs.tobytes(), _ARRAY_BUFFER),
        (indices.tobytes(), _ELEMENT_ARRAY_BUFFER),
        (png, None),
    ):
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        views.append(view if target is None else {**view, "target": target})
        blob += data + b"\0" * (-len(data) % 4)
    bounds = {"min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()}
    accessors = [
        {"bufferView": 0, "componentType": _FLOAT, "count": len(positions), "type": "VEC3", **bounds},
        {"bufferView": 1, "componentType": _FLOAT, "count": len(normals), "type": "VEC3"},
        {"bufferView": 2, "componentType": _FLOAT, "count": len(uvs), "type": "VEC2"},
        {
            "bufferView": 3,
            "componentType": _USHORT if indices.itemsize == 2 else _UINT,
            "count": len(indices),
            "type": "SCALAR",
        },
    ]
    primitive = {"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}
    pbr = {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}
    doc = {
        "asset": {"version": "2.0", "generator": "realmweaver"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [primitive]}],
        "materials": [{"name": name, "pbrMetallicRoughness": pbr}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"wrapS": _REPEAT, "wrapT": _REPEAT}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "bufferViews": views,
        "accessors": accessors,
        "buffers": [{"byteLength": len(blob)}],
    }
    return pack(doc, bytes(blob))


def _accessor(doc: dict, binary: bytes, index: int) -> np.ndarray:
    """An accessor's values as [count, width] from its buffer view."""
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    width, start = _WIDTH[acc["type"]], view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    # ponytail: a tightly packed view is assumed; an interleaved one (byteStride) is read wrong. Our own GLBs never
    # interleave; honouring byteStride is the upgrade path when a mesh model's output does.
    values = np.frombuffer(binary, _DTYPES[acc["componentType"]], acc["count"] * width, start)
    return values.reshape(-1, width)


def first_primitive(glb: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Float32 [N, 3] positions and int64 [M, 3] triangles of the first mesh's first primitive; both empty when the
    GLB holds no mesh (a skeleton-only clip). ValueError for a primitive not drawn as a triangle list."""
    doc, binary = unpack(glb)
    if not doc.get("meshes"):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    primitive = doc["meshes"][0]["primitives"][0]
    mode = primitive.get("mode", _TRIANGLES)
    if mode != _TRIANGLES:
        raise ValueError(
            f"first primitive has mode {mode}, not TRIANGLES ({_TRIANGLES}): only triangle lists are read"
        )
    positions = _accessor(doc, binary, primitive["attributes"]["POSITION"]).astype(np.float32)
    if "indices" in primitive:
        faces = _accessor(doc, binary, primitive["indices"]).astype(np.int64).reshape(-1, 3)
    else:
        faces = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)
    return positions, faces


def is_watertight(positions: np.ndarray, faces: np.ndarray) -> bool:
    """Closed surface: with coincident vertices merged, every edge belongs to exactly two triangles."""
    _, merged = np.unique(np.asarray(positions, dtype=np.float32), axis=0, return_inverse=True)
    tri = merged.reshape(-1)[np.asarray(faces).reshape(-1, 3)]
    edges = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return len(tri) > 0 and bool((counts == 2).all())


def silhouette(glb: bytes, size: int, colour: tuple[int, int, int] | None = None) -> np.ndarray:
    """Flat-shaded orthographic render of the GLB's first primitive fitted to 80% of a `size` px square, in `colour`
    (grey when None) over a dark background; uint8 HxWx3. A GLB without a mesh renders the background alone."""
    positions, faces = first_primitive(glb)
    img = np.full((size, size, 3), _BACKGROUND, np.uint8)
    if not len(faces):
        return img
    pts = positions.astype(np.float64) @ _VIEW.T
    lo, hi = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    xy = (pts[:, :2] - (lo + hi) / 2) * (0.8 * size / max(float((hi - lo).max()), 1e-6)) + size / 2
    xy[:, 1] = size - xy[:, 1]  # y points up in view space and down the rows of an image
    tri = pts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    shade = 0.3 + 0.7 * np.abs(normals @ _LIGHT)
    base = np.array(_GREY if colour is None else colour, np.float64)
    # ponytail: a Python loop over faces painted far to near (painter's algorithm), one `_fill` per triangle, so
    # it suits placeholders and previews of small props only; a vectorised z-buffer is the upgrade path.
    for f in np.argsort(tri[:, :, 2].mean(axis=1)):
        _fill(img, xy[faces[f]], np.clip(base * shade[f], 0, 255).round().astype(np.uint8))
    return img


def _fill(img: np.ndarray, corners: np.ndarray, colour: np.ndarray) -> None:
    """Paint the pixels whose centres lie inside the screen-space triangle `corners` ([3, 2] x, y)."""
    size = img.shape[0]
    (x0, y0), (x1, y1), (x2, y2) = corners
    x_lo, x_hi = max(int(np.floor(min(x0, x1, x2))), 0), min(int(np.ceil(max(x0, x1, x2))), size - 1)
    y_lo, y_hi = max(int(np.floor(min(y0, y1, y2))), 0), min(int(np.ceil(max(y0, y1, y2))), size - 1)
    if x_lo > x_hi or y_lo > y_hi:
        return
    ys, xs = np.mgrid[y_lo : y_hi + 1, x_lo : x_hi + 1]
    xs, ys = xs + 0.5, ys + 0.5
    w0 = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
    w1 = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
    w2 = (x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)
    inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
    img[y_lo : y_hi + 1, x_lo : x_hi + 1][inside] = colour
