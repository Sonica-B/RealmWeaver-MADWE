"""Consolidate spike results (runner JSON + trimesh validation JSON) into markdown tables for the doc."""

import json
import sys
from pathlib import Path

GLB = Path(r"D:\WPI_Assignments\SideGigs\RealmWeaver-MADWE\models\mesh-spike")
RES = GLB / "results"

GEN = ["bg_mask", "cond", "ks_ss", "dec_ss", "ks_shape", "dec_shape", "ks_tex", "dec_tex"]
POST = ["remesh", "decim", "smooth1", "unwrap", "bake_tex", "bake_nrm", "bake_ao", "apply", "smooth2", "save"]
TURN = ["rot0", "view0", "rot1", "view1", "rot2", "view2", "rot3", "view3", "stitch1", "stitch2", "stitch3", "save_turn"]


def load(name, tag="cu128", glb_name=None):
    r = json.loads((RES / f"{name}_{tag}.json").read_text(encoding="utf-8"))
    vpath = GLB / f"{(glb_name or name)}.glb.validation.json"
    v = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    return r, v


def secs(r, keys):
    t = r["node_timings_s"]
    return sum(v for k, v in t.items() if k in keys and isinstance(v, (int, float)))


def main(rows):
    out = []
    out.append("| asset | input (seed) | mask | wall s (cold/warm) | gen s | post s | turntable s | torch peak alloc / reserved GB | nvidia-smi before -> peak MiB | peak stage |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, tag, glb_name, label, inp, cold in rows:
        r, v = load(name, tag, glb_name)
        tp, smi = r["torch_peak"], r["nvidia_smi_used_mib"]
        out.append(f"| {label} | {inp} | {r.get('mask','alpha')} | {r['wall_s']} ({cold}) | {secs(r, GEN):.1f} | {secs(r, POST):.1f} | {secs(r, TURN):.1f} | "
                   f"{tp.get('max_allocated_gb')} / {tp.get('max_reserved_gb')} | {smi.get('baseline')} -> {smi.get('peak')} | {smi.get('peak_node') or 'n/a (old runner)'} |")
    out.append("")
    out.append("| asset | GLB MB | triangles | verts stored / merged | bbox extents (x,y,z) | watertight | shells | boundary edges | non-manifold edges | UV coverage | base color / metal-rough / normal / AO px |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, tag, glb_name, label, inp, cold in rows:
        r, v = load(name, tag, glb_name)
        if not v:
            out.append(f"| {label} | {r.get('glb_mb')} | (no validation) |  |  |  |  |  |  |  |  |")
            continue
        tx = v["textures"]
        fmt = lambda s: f"{s[0]}" if s else "-"
        out.append(f"| {label} | {v['size_mb']} | {v['triangles']:,} | {v['vertices_as_stored']:,} / {v['vertices_merged']:,} | "
                   f"{' x '.join(f'{e:.2f}' for e in v['bbox_extents'])} | {v['watertight_merged']} | {v['connected_components']} | {v['boundary_edges']} | "
                   f"{v['nonmanifold_edges']} | {v['uv_coverage']*100:.1f} % | {fmt(tx['baseColor'])} / {fmt(tx['metallicRoughness'])} / {fmt(tx['normal'])} / {fmt(tx['occlusion'])} |")
    out.append("")
    out.append("Per-stage seconds (node timings from the ComfyUI websocket `executing` events):")
    out.append("")
    keys = GEN + POST
    out.append("| asset | " + " | ".join(keys) + " |")
    out.append("|---|" + "---|" * len(keys))
    for name, tag, glb_name, label, inp, cold in rows:
        r, v = load(name, tag, glb_name)
        t = r["node_timings_s"]
        out.append(f"| {label} | " + " | ".join(f"{t[k]:.1f}" if isinstance(t.get(k), (int, float)) else "-" for k in keys) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    rows = [
        ("mushroom", "cu128", "mushroom", "mushroom (final)", "forest_mushroom.png (s3)", "cold"),
        ("log", "cu128", "log", "log (final)", "forest_log.png (s3)", "warm"),
        ("fern", "cu128", "fern", "fern (final)", "forest_fern.png (s2)", "warm"),
        ("mushroom_budget", "cu128", "mushroom_budget", "mushroom, game budget", "forest_mushroom.png (s3)", "warm"),
    ]
    main(rows)
