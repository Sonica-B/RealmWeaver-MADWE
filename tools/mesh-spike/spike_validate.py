"""Validate the spike GLBs with trimesh + numpy: watertight, triangles, bbox, UV coverage, texture sizes.

Usage: python spike_validate.py <glb> [<glb> ...]  -> prints one JSON per file and writes <glb>.validation.json
"""

import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def uv_coverage(uv, faces, grid=1024):
    """Fraction of the [0,1]^2 atlas covered by UV triangles (rasterised at grid^2)."""
    if uv is None or len(uv) == 0:
        return None
    img = Image.new("1", (grid, grid), 0)
    draw = ImageDraw.Draw(img)
    pts = np.clip(uv, 0, 1) * (grid - 1)
    for tri in faces:
        p = pts[tri]
        draw.polygon([tuple(x) for x in p], fill=1)
    return float(np.asarray(img, dtype=np.uint8).mean())


def tex_size(mat, attr):
    img = getattr(mat, attr, None)
    return list(img.size) if isinstance(img, Image.Image) else None


def validate(path):
    scene = trimesh.load(path, force="scene", process=False)
    geoms = list(scene.geometry.values())
    raw = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    out = {"file": path, "size_mb": round(os.path.getsize(path) / 2**20, 2), "geometries": len(geoms)}
    out["triangles"] = int(raw.faces.shape[0])
    out["vertices_as_stored"] = int(raw.vertices.shape[0])
    ext = raw.bounding_box.extents
    out["bbox_extents"] = [round(float(x), 4) for x in ext]
    out["bbox_min"] = [round(float(x), 4) for x in raw.bounds[0]]
    out["bbox_max"] = [round(float(x), 4) for x in raw.bounds[1]]
    out["watertight_as_stored"] = bool(raw.is_watertight)
    # geometry-level watertightness: merge vertices by position only (UV seams split vertices in a GLB)
    merged = trimesh.Trimesh(vertices=raw.vertices.copy(), faces=raw.faces.copy(), process=False)
    merged.merge_vertices(merge_tex=True, merge_norm=True)
    merged.remove_unreferenced_vertices()
    out["vertices_merged"] = int(merged.vertices.shape[0])
    out["watertight_merged"] = bool(merged.is_watertight)
    out["winding_consistent"] = bool(merged.is_winding_consistent)
    out["euler_number"] = int(merged.euler_number)
    edges_sorted = np.sort(merged.edges, axis=1)
    _, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    out["boundary_edges"] = int((counts == 1).sum())
    out["nonmanifold_edges"] = int((counts > 2).sum())
    out["connected_components"] = int(len(merged.split(only_watertight=False)))
    out["volume_if_watertight"] = round(float(merged.volume), 6) if merged.is_watertight else None
    vis = raw.visual
    uv = getattr(vis, "uv", None)
    out["has_uv"] = uv is not None and len(uv) > 0
    out["uv_coverage"] = round(uv_coverage(uv, raw.faces), 4) if out["has_uv"] else None
    mat = getattr(vis, "material", None)
    out["material_type"] = type(mat).__name__ if mat is not None else None
    if mat is not None:
        out["textures"] = {
            "baseColor": tex_size(mat, "baseColorTexture"),
            "metallicRoughness": tex_size(mat, "metallicRoughnessTexture"),
            "normal": tex_size(mat, "normalTexture"),
            "occlusion": tex_size(mat, "occlusionTexture"),
            "emissive": tex_size(mat, "emissiveTexture"),
        }
        out["metallicFactor"] = getattr(mat, "metallicFactor", None)
        out["roughnessFactor"] = getattr(mat, "roughnessFactor", None)
    out["has_vertex_normals"] = bool(
        raw.vertex_normals is not None and len(raw.vertex_normals) == len(raw.vertices)
    )
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = validate(p)
        with open(p + ".validation.json", "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        print(json.dumps(r, indent=1))
