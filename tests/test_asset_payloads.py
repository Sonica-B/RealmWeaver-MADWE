"""The payload-typed `Asset` behind the `Generator` seam: `encode` and `preview` per kind, the `image` property
contract, the kind/payload invariant, and the procedural placeholder cube as a valid GLB.

GLBs are parsed here by the glTF 2.0 container spec (header, then length-typed chunks), not by the package.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from realmweaver.assets import ProceduralGenerator
from realmweaver.types import AnimationPayload, Asset, AssetSpec, MeshPayload, SpritePayload, TexturePayload

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
GLB = "model/gltf-binary"


def _glb(data: bytes) -> tuple[dict, bytes]:
    """JSON document and BIN chunk of a GLB, checked against the container spec as it is read."""
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2 and length == len(data)
    offset, chunks = 12, {}
    while offset < length:
        n, kind = struct.unpack_from("<I4s", data, offset)
        chunks[kind] = data[offset + 8 : offset + 8 + n]
        offset += 8 + n
    assert list(chunks) == [b"JSON", b"BIN\x00"] and all(len(c) % 4 == 0 for c in chunks.values())
    return json.loads(chunks[b"JSON"]), chunks[b"BIN\x00"]


def _pack(doc: dict, binary: bytes) -> bytes:
    """A GLB from its parts, by the container spec: header, space-padded JSON chunk, zero-padded BIN chunk."""
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    body = (
        struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    )
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _accessor_values(doc: dict, binary: bytes, index: int, width: int) -> np.ndarray:
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.frombuffer(binary, "<f4", acc["count"] * width, start).reshape(-1, width)


def _texture(size: int = 32, seed: int = 1) -> Asset:
    return ProceduralGenerator().generate(AssetSpec("forest", subject="grass", size=size, seed=seed))


def _sprite(size: int = 32) -> Asset:
    return ProceduralGenerator().generate(AssetSpec("forest", kind="sprite", subject="mushroom", size=size))


def _mesh(size: int = 16, seed: int = 2) -> Asset:
    return ProceduralGenerator().generate(
        AssetSpec("forest", kind="mesh", subject="rock", size=size, seed=seed)
    )


# --- encode ------------------------------------------------------------------------------------------


def test_texture_encodes_as_png_that_decodes_to_the_same_pixels():
    a = _texture()
    data, media_type = a.encode()
    assert media_type == "image/png" and data[:8] == PNG_MAGIC
    decoded = np.asarray(Image.open(io.BytesIO(data)))
    assert decoded.shape == (32, 32, 3) and np.array_equal(decoded, a.payload.image)


def test_sprite_encodes_as_rgba_png_with_transparent_corners():
    data, media_type = _sprite().encode()
    png = Image.open(io.BytesIO(data))
    assert media_type == "image/png" and png.mode == "RGBA"
    pixels = np.asarray(png)
    assert pixels.shape == (32, 32, 4) and pixels[0, 0, 3] == 0 and pixels[16, 16, 3] == 255


def test_mesh_and_animation_encode_their_bytes_as_gltf_binary():
    mesh = _mesh()
    assert mesh.encode() == (mesh.payload.glb_bytes, GLB)
    clip = AnimationPayload(clip_bytes=mesh.payload.glb_bytes, fps=30.0, frames=45)
    assert clip.encode() == (mesh.payload.glb_bytes, GLB)


# --- preview -----------------------------------------------------------------------------------------


def test_preview_at_the_image_size_is_the_rgb_pixels():
    texture, sprite = _texture(64), _sprite(32)
    assert np.array_equal(texture.preview(64), texture.payload.image)
    assert np.array_equal(sprite.preview(32), sprite.payload.image[..., :3])


def test_preview_box_filters_when_the_side_divides():
    img = np.zeros((4, 4, 3), np.uint8)
    img[:2, 2:] = 255
    img[0, 2] = 251  # (251 + 3 * 255) / 4 = 254
    img[2:, :2] = 100
    img[2:, 2:] = (200, 0, 0)
    expected = [[[0, 0, 0], [254, 254, 254]], [[100, 100, 100], [200, 0, 0]]]
    assert TexturePayload(img).preview(2).tolist() == expected


def test_preview_samples_nearest_otherwise_and_upscales():
    img = np.zeros((6, 6, 3), np.uint8)
    img[..., 0] = np.arange(6)[:, None] * 10 + np.arange(6)[None, :]  # red = 10 * row + column
    rows = [[0, 1, 3, 4], [10, 11, 13, 14], [30, 31, 33, 34], [40, 41, 43, 44]]
    assert TexturePayload(img).preview(4)[..., 0].tolist() == rows
    small = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(TexturePayload(small).preview(4), np.repeat(np.repeat(small, 2, axis=0), 2, axis=1))


def test_mesh_preview_is_a_flat_shaded_silhouette_of_the_cube():
    img = _mesh().preview(32)
    assert img.shape == (32, 32, 3) and img.dtype == np.uint8
    background = img[0, 0]
    assert np.array_equal(background, img[-1, -1]) and not np.array_equal(background, img[16, 16])
    # a cube in three-quarter view shows three faces, each one flat shade, over the background
    assert len({tuple(px) for px in img.reshape(-1, 3)}) == 4


def test_animation_preview_is_a_thumbnail_of_its_rest_pose():
    clip = AnimationPayload(clip_bytes=_mesh(8).payload.glb_bytes, fps=24.0, frames=12)
    img = clip.preview(16)
    assert img.shape == (16, 16, 3) and img.dtype == np.uint8
    assert not np.array_equal(img[0, 0], img[8, 8])


# --- the Asset record --------------------------------------------------------------------------------


def test_image_property_is_the_pixels_of_a_texture_or_sprite_and_raises_for_other_kinds():
    texture, sprite, mesh = _texture(), _sprite(), _mesh()
    assert texture.image is texture.payload.image and texture.image.shape == (32, 32, 3)
    assert sprite.image is sprite.payload.image and sprite.image.shape == (32, 32, 4)
    with pytest.raises(TypeError, match="mesh"):
        _ = mesh.image
    clip = Asset(AssetSpec("forest", kind="animation", subject="walk"), AnimationPayload(b"", 30.0, 0))
    with pytest.raises(TypeError, match="animation"):
        _ = clip.image


def test_nbytes_is_the_payloads_size():
    assert _texture(8).nbytes == 8 * 8 * 3 and _sprite(8).nbytes == 8 * 8 * 4
    mesh = _mesh(8)
    assert mesh.nbytes == len(mesh.payload.glb_bytes) + 8 * 8 * 3
    assert Asset(AssetSpec("forest", kind="animation"), AnimationPayload(b"abc", 30.0, 1)).nbytes == 3


def test_image_payloads_check_dtype_and_channels():
    with pytest.raises(ValueError):
        TexturePayload(np.zeros((4, 4, 4), np.uint8))
    with pytest.raises(ValueError):
        SpritePayload(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        TexturePayload(np.zeros((4, 4, 3), np.float32))


def test_asset_rejects_a_payload_of_another_kind():
    with pytest.raises(TypeError):
        Asset(
            AssetSpec("forest", kind="sprite", subject="log"), TexturePayload(np.zeros((2, 2, 3), np.uint8))
        )
    with pytest.raises(TypeError):
        Asset(AssetSpec("forest", kind="mesh", subject="rock"), AnimationPayload(b"", 30.0, 0))


def test_kind_is_part_of_the_content_id():
    ids = {AssetSpec("forest", kind=k, subject="rock").id for k in ("texture", "sprite", "mesh", "animation")}
    assert len(ids) == 4 and '"kind":"mesh"' in AssetSpec("forest", kind="mesh").canonical()


# --- the procedural cube -----------------------------------------------------------------------------


def test_procedural_mesh_is_one_textured_cube_primitive_in_a_valid_glb():
    a = _mesh(16, seed=2)
    assert isinstance(a.payload, MeshPayload) and a.payload.triangles == 12 and a.payload.watertight is True
    doc, binary = _glb(a.payload.glb_bytes)
    assert doc["asset"]["version"] == "2.0" and doc["buffers"][0]["byteLength"] == len(binary)
    assert len(doc["meshes"]) == 1 and len(doc["meshes"][0]["primitives"]) == 1
    prim = doc["meshes"][0]["primitives"][0]
    assert {"POSITION", "NORMAL", "TEXCOORD_0"} <= set(prim["attributes"])
    position = doc["accessors"][prim["attributes"]["POSITION"]]
    assert (
        position["type"] == "VEC3" and position["count"] == 24
    )  # four vertices per face: one texture per face
    assert position["min"] == [-0.5] * 3 and position["max"] == [0.5] * 3
    assert doc["accessors"][prim["indices"]]["count"] == 36
    corners = _accessor_values(doc, binary, prim["attributes"]["POSITION"], 3)
    expected = {(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)}
    assert {tuple(map(float, v)) for v in corners} == expected
    assert prim.get("mode", 4) == 4, "TRIANGLES: the one mode the package reads back"
    uvs = _accessor_values(doc, binary, prim["attributes"]["TEXCOORD_0"], 2).reshape(6, 4, 2)
    unit_square = {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    assert all({tuple(map(float, p)) for p in face} == unit_square for face in uvs), (
        "each face covers the texture"
    )


def test_non_triangle_primitives_are_refused_by_name():
    from realmweaver import gltf

    doc, binary = _glb(_mesh(8).payload.glb_bytes)
    doc["meshes"][0]["primitives"][0]["mode"] = 1  # LINES
    lines = _pack(doc, binary)
    with pytest.raises(ValueError, match="TRIANGLES"):
        gltf.first_primitive(lines)
    with pytest.raises(ValueError, match="TRIANGLES"):
        MeshPayload(lines, triangles=0, watertight=False).preview(8)


def test_procedural_cube_wears_its_subjects_texture():
    a = _mesh(16, seed=2)
    doc, binary = _glb(a.payload.glb_bytes)
    assert doc["materials"][prim_material(doc)]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    image = doc["images"][doc["textures"][0]["source"]]
    assert image["mimeType"] == "image/png"
    view = doc["bufferViews"][image["bufferView"]]
    png = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
    decoded = np.asarray(Image.open(io.BytesIO(png)))
    assert decoded.shape == (16, 16, 3) and np.array_equal(decoded, a.payload.textures["baseColor"])
    texture = ProceduralGenerator().generate(AssetSpec("forest", subject="rock", size=16, seed=2))
    assert np.array_equal(decoded, texture.payload.image), (
        "the cube's texture is the subject's seamless texture"
    )


def prim_material(doc: dict) -> int:
    return doc["meshes"][0]["primitives"][0]["material"]


def test_procedural_mesh_is_deterministic_and_seed_changes_the_texture():
    a, b, c = _mesh(8, seed=1), _mesh(8, seed=1), _mesh(8, seed=2)
    assert a.payload.glb_bytes == b.payload.glb_bytes and a.id == b.id
    assert a.payload.glb_bytes != c.payload.glb_bytes and a.id != c.id


def test_procedural_generator_names_the_kind_it_cannot_make():
    with pytest.raises(RuntimeError, match="animation"):
        ProceduralGenerator().generate(AssetSpec("forest", kind="animation", subject="walk"))
