"""TRELLIS.2 spike runner: build the ComfyUI API graph for one image, run it, collect measurements.

Usage (ComfyUI venv python):
  python spike_run.py --image forest_mushroom.png --name mushroom [--seed 42] [--res 512]
                      [--faces 700000] [--tex 4096] [--remesh-res 768] [--tag cu128]
Writes: <results_dir>/<name>_<tag>.json, copies the GLB to --glb-dir/<name>.glb and the turntable
PNG to --png-dir/spike-mesh-<name>.png.
"""

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import aiohttp

SERVER = "http://127.0.0.1:8188"
WS = "ws://127.0.0.1:8188/ws"
COMFY_OUT = Path(r"D:\tools\ComfyUI\output")


def build_graph(a, tag, probe_path):
    seed = a.seed
    g = {}

    def node(nid, ctype, **inputs):
        g[nid] = {"class_type": ctype, "inputs": inputs}
        return nid

    # ---- loaders
    node("unet", "UNETLoader", unet_name=a.unet, weight_dtype="default")
    node("clipv", "CLIPVisionLoader", clip_name=a.clip_vision)
    node("vae_shape", "VAELoader", vae_name="trellis_2_shape_vae_bf16.safetensors")
    node("vae_tex", "VAELoader", vae_name="trellis_2_texture_vae_bf16.safetensors")

    # ---- input image (alpha-keyed sprite): LoadImage mask is 1-alpha, so invert it for ImageCropToMask
    node("img", "LoadImage", image=a.image)
    node("probe0", "VRAMPeakReset", value=["img", 0], tag=tag)
    node("inv", "InvertMask", mask=["img", 1])
    if a.mask in ("birefnet", "both"):  # template path: BiRefNet salient-object matting on the RGB
        node("bg_model", "LoadBackgroundRemovalModel", bg_removal_name="birefnet.safetensors")
        node("bg_mask", "RemoveBackground", bg_removal_model=["bg_model", 0], image=["probe0", 0])
    if a.mask == "alpha":
        mask_ref = ["inv", 0]
    elif a.mask == "birefnet":
        mask_ref = ["bg_mask", 0]
    else:  # both: sprite alpha AND BiRefNet
        node(
            "mask_and",
            "MaskComposite",
            destination=["bg_mask", 0],
            source=["inv", 0],
            x=0,
            y=0,
            operation="multiply",
        )
        mask_ref = ["mask_and", 0]
    node("mask_img", "MaskToImage", mask=mask_ref)
    node("save_mask", "SaveImage", images=["mask_img", 0], filename_prefix=f"spike/{a.name}_mask")
    node(
        "crop",
        "ImageCropToMask",
        images=["probe0", 0],
        masks=mask_ref,
        width=1024,
        height=1024,
        pad_factor=a.pad_factor,
        grow_mask=0,
        background="#000000",
    )
    node("save_crop", "SaveImage", images=["crop", 0], filename_prefix=f"spike/{a.name}_input")
    if a.preview_only:
        return g
    node("cond", "Trellis2Conditioning", clip_vision_model=["clipv", 0], image=["crop", 0])

    # ---- model patches (template: CFGOverride + RescaleCFG + ModelSamplingSD3 reproduce upstream defaults)
    node("cfg_ss", "CFGOverride", model=["unet", 0], cfg=1.0, start_percent=0.667, end_percent=1.0)
    node("rescale_ss", "RescaleCFG", model=["cfg_ss", 0], multiplier=0.7)
    node("ms_ss", "ModelSamplingSD3", model=["rescale_ss", 0], shift=5.0)
    node("cfg_shape", "CFGOverride", model=["unet", 0], cfg=1.0, start_percent=0.769, end_percent=1.0)
    node("rescale_shape", "RescaleCFG", model=["cfg_shape", 0], multiplier=0.5)

    # ---- structure stage (sparse structure, 64^3 -> 32^3 for the 512 pipeline)
    node("lat_ss", "EmptyTrellis2LatentStructure", batch_size=1)
    node(
        "ks_ss",
        "KSampler",
        model=["ms_ss", 0],
        seed=seed,
        steps=12,
        cfg=7.5,
        sampler_name="euler",
        scheduler="normal",
        positive=["cond", 0],
        negative=["cond", 1],
        latent_image=["lat_ss", 0],
        denoise=1.0,
    )
    node("dec_ss", "VaeDecodeStructureTrellis2", samples=["ks_ss", 0], vae=["vae_shape", 0], resolution="32")

    # ---- shape stage at 512^3
    node("shape_stage", "Trellis2ShapeStage", positive=["cond", 0], negative=["cond", 1], voxel=["dec_ss", 0])
    node(
        "ks_shape",
        "KSampler",
        model=["rescale_shape", 0],
        seed=seed,
        steps=20,
        cfg=7.5,
        sampler_name="euler",
        scheduler="normal",
        positive=["shape_stage", 0],
        negative=["shape_stage", 1],
        latent_image=["shape_stage", 2],
        denoise=1.0,
    )
    stage, shape_latent = "shape_stage", "ks_shape"
    if a.res > 512:  # optional cascade (1024/1536): second shape pass on upsampled coords
        node(
            "ups",
            "Trellis2UpsampleStage",
            positive=["shape_stage", 0],
            negative=["shape_stage", 1],
            shape_latent=["ks_shape", 0],
            vae=["vae_shape", 0],
            target_resolution=a.res,
        )
        node(
            "ks_shape2",
            "KSampler",
            model=["rescale_shape", 0],
            seed=seed,
            steps=12,
            cfg=7.5,
            sampler_name="euler",
            scheduler="simple",
            positive=["ups", 0],
            negative=["ups", 1],
            latent_image=["ups", 2],
            denoise=1.0,
        )
        stage, shape_latent = "ups", "ks_shape2"
    node("dec_shape", "VaeDecodeShapeTrellis", samples=[shape_latent, 0], vae=["vae_shape", 0])
    node("info_raw", "GetMeshInfo", mesh=["dec_shape", 0])

    # ---- texture stage
    node(
        "tex_stage",
        "Trellis2TextureStage",
        positive=[stage, 0],
        negative=[stage, 1],
        shape_latent=[shape_latent, 0],
    )
    node(
        "ks_tex",
        "KSampler",
        model=["unet", 0],
        seed=seed,
        steps=12,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        positive=["tex_stage", 0],
        negative=["tex_stage", 1],
        latent_image=["tex_stage", 2],
        denoise=1.0,
    )
    node(
        "dec_tex",
        "VaeDecodeTextureTrellis",
        samples=["ks_tex", 0],
        vae=["vae_tex", 0],
        shape_subdivides=["dec_shape", 1],
    )

    # ---- post-processing (template defaults): remesh -> decimate -> smooth -> unwrap -> bake -> apply
    g["remesh"] = {
        "class_type": "RemeshMesh",
        "inputs": {
            "mesh": ["info_raw", 0],
            "resolution": a.remesh_res,
            "sign_mode": "udf",
            "sign_mode.qef": False,
            "sign_mode.drop_inverted_components": False,
            "sign_mode.drop_enclosed_components": False,
            "band": 1.0,
            "project_back": 0.0,
            "fix_poles": False,
            "smooth_iters": 20,
            "drop_small_components": 0.01,
            "precluster_max_verts": 20000000,
        },
    }
    node("decim", "DecimateMesh", mesh=["remesh", 0], target_face_count=a.faces, placement_mode="midpoint")
    node("smooth1", "MeshSmoothNormals", mesh=["decim", 0], crease_angle=180.0)
    node(
        "unwrap",
        "UnwrapMesh",
        mesh=["smooth1", 0],
        segmenter="pec",
        resolution=a.tex,
        padding=1,
        weld_distance=0.0002,
    )
    node(
        "bake_tex",
        "BakeTextureFromVoxel",
        mesh=["unwrap", 0],
        voxel_colors=["dec_tex", 0],
        texture_size=a.tex,
        reference_mesh=["dec_shape", 0],
    )
    node(
        "bake_nrm",
        "BakeNormalMapFromMesh",
        low_poly=["unwrap", 0],
        high_poly=["remesh", 0],
        resolution=a.normal_res,
        cage_distance=0.05,
        ignore_backfaces=True,
    )
    node(
        "bake_ao",
        "BakeAmbientOcclusion",
        low_poly=["unwrap", 0],
        high_poly=["remesh", 0],
        resolution=a.ao_res,
        samples=64,
        max_distance=0.71,
        strength=1.0,
        bias=0.01,
    )
    node(
        "apply",
        "ApplyTextureToMesh",
        mesh=["unwrap", 0],
        base_color=["bake_tex", 0],
        metallic=["bake_tex", 1],
        roughness=["bake_tex", 2],
        occlusion=["bake_ao", 0],
        normal_map=["bake_nrm", 0],
    )
    node("smooth2", "MeshSmoothNormals", mesh=["apply", 0], crease_angle=180.0)
    node("info_final", "GetMeshInfo", mesh=["smooth2", 0])
    node("probe1", "VRAMPeakReport", value=["info_final", 0], tag=tag, path=probe_path)
    node("save", "SaveGLB", mesh=["info_final", 0], filename_prefix=f"3d/spike/{a.name}")

    # ---- in-graph turntable: 4 views around Y (glTF up), slight tilt, stitched into one strip
    prev = None
    for i, ang in enumerate((0.0, 90.0, 180.0, 270.0)):
        g[f"rot{i}"] = {
            "class_type": "RotateMesh",
            "inputs": {
                "mesh": ["info_final", 0],
                "mode": "euler_xyz",
                "mode.angle_x": 20.0,
                "mode.angle_y": ang,
                "mode.angle_z": 0.0,
            },
        }
        node(
            f"view{i}",
            "RenderMesh",
            mesh=[f"rot{i}", 0],
            mode="texture",
            width=512,
            height=512,
            background="#303030",
        )
        if prev is None:
            prev = f"view{i}"
        else:
            node(
                f"stitch{i}",
                "ImageStitch",
                image1=[prev, 0],
                image2=[f"view{i}", 0],
                direction="right",
                match_image_size=True,
                spacing_width=4,
                spacing_color="black",
            )
            prev = f"stitch{i}"
    node("save_turn", "SaveImage", images=[prev, 0], filename_prefix=f"spike/{a.name}_turntable")
    return g


class SmiSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                self.samples.append((time.perf_counter(), int(out.splitlines()[0])))
            except Exception:
                pass
            self.stop.wait(1.0)


async def run(graph, client_id):
    timings, order, errors, starts = {}, [], [], {}
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"{WS}?clientId={client_id}", max_msg_size=0) as ws:
            async with s.post(f"{SERVER}/prompt", json={"prompt": graph, "client_id": client_id}) as r:
                resp = await r.json()
                if r.status != 200:
                    raise SystemExit(f"prompt rejected: {json.dumps(resp, indent=1)[:4000]}")
            pid = resp["prompt_id"]
            t_start, cur, t_cur = time.perf_counter(), None, None
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                d = m.get("data", {})
                if d.get("prompt_id") not in (None, pid):
                    continue
                now = time.perf_counter()
                if m["type"] == "executing":
                    if cur is not None:
                        timings[cur] = timings.get(cur, 0.0) + (now - t_cur)
                    cur, t_cur = d.get("node"), now
                    if cur is None:
                        break
                    order.append(cur)
                    starts[cur] = now
                elif m["type"] == "execution_error":
                    errors.append(d)
                    break
                elif m["type"] == "execution_cached":
                    for n in d.get("nodes", []):
                        timings.setdefault(n, "cached")
            wall = time.perf_counter() - t_start
        async with s.get(f"{SERVER}/history/{pid}") as r:
            hist = (await r.json()).get(pid, {})
    return pid, wall, timings, order, errors, hist, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--faces", type=int, default=700000)
    ap.add_argument("--tex", type=int, default=4096)
    ap.add_argument("--normal-res", type=int, default=2048)
    ap.add_argument("--ao-res", type=int, default=1024)
    ap.add_argument("--remesh-res", type=int, default=768)
    ap.add_argument("--pad-factor", type=float, default=1.0)
    ap.add_argument("--unet", default="trellis_2_int8_convrot.safetensors")
    ap.add_argument("--clip-vision", default="dino_v3_vit_l.safetensors")
    ap.add_argument("--tag", default="run")
    ap.add_argument(
        "--mask",
        choices=("alpha", "birefnet", "both"),
        default="birefnet",
        help="foreground mask: sprite alpha, BiRefNet matting (template default), or their product",
    )
    ap.add_argument(
        "--preview-only", action="store_true", help="only run mask + crop and save the preview images"
    )
    ap.add_argument(
        "--results-dir", default=r"D:\WPI_Assignments\SideGigs\RealmWeaver-MADWE\models\mesh-spike\results"
    )
    ap.add_argument("--glb-dir", default=r"D:\WPI_Assignments\SideGigs\RealmWeaver-MADWE\models\mesh-spike")
    ap.add_argument("--png-dir", default=r"D:\WPI_Assignments\SideGigs\RealmWeaver-MADWE\docs\images")
    ap.add_argument("--dump-graph", action="store_true")
    a = ap.parse_args()

    results = Path(a.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    run_tag = f"{a.name}-{a.tag}-{int(time.time())}"
    probe_path = str(results / f"{a.name}_{a.tag}_probe.json")
    graph = build_graph(a, run_tag, probe_path)
    (results / f"{a.name}_{a.tag}_graph_api.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
    if a.dump_graph:
        print(json.dumps(graph, indent=1))
        return

    smi = SmiSampler()
    base = None
    with contextlib.suppress(Exception):
        base = int(
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    smi.start()
    pid, wall, timings, order, errors, hist, starts = asyncio.run(run(graph, str(uuid.uuid4())))
    smi.stop.set()
    smi.join(timeout=3)
    # attribute each nvidia-smi sample to the node that was executing at that moment
    peak_node, node_peaks = None, {}
    if smi.samples:
        seq = sorted((starts[n], n) for n in order if n in starts)
        for t, used in smi.samples:
            active = None
            for t0, n in seq:
                if t0 <= t:
                    active = n
                else:
                    break
            if active is not None:
                node_peaks[active] = max(node_peaks.get(active, 0), used)
        peak_used = max(u for _, u in smi.samples)
        peak_node = max(node_peaks, key=node_peaks.get) if node_peaks else None
    else:
        peak_used = None

    probe = json.loads(Path(probe_path).read_text()) if Path(probe_path).exists() else {}
    outputs = hist.get("outputs", {})
    glb = None
    for item in outputs.get("save", {}).get("result", []) or outputs.get("save", {}).get("3d", []) or []:
        glb = COMFY_OUT / item.get("subfolder", "") / item["filename"]
    if glb is None:  # fall back to scanning the ui outputs
        for _key, val in outputs.get("save", {}).items():
            if isinstance(val, list) and val and isinstance(val[0], dict) and "filename" in val[0]:
                glb = COMFY_OUT / val[0].get("subfolder", "") / val[0]["filename"]
    turn = None
    for item in outputs.get("save_turn", {}).get("images", []):
        turn = COMFY_OUT / item.get("subfolder", "") / item["filename"]
    mesh_info = {k: outputs.get(k, {}).get("text") for k in ("info_raw", "info_final")}

    summary = {
        "name": a.name,
        "tag": a.tag,
        "prompt_id": pid,
        "image": a.image,
        "mask": a.mask,
        "seed": a.seed,
        "res": a.res,
        "faces_target": a.faces,
        "texture_size": a.tex,
        "remesh_res": a.remesh_res,
        "unet": a.unet,
        "wall_s": round(wall, 1),
        "errors": errors,
        "node_timings_s": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in timings.items()},
        "node_order": order,
        "torch_peak": probe,
        "nvidia_smi_used_mib": {
            "baseline": base,
            "peak": peak_used,
            "peak_node": peak_node,
            "per_node_peak": {k: v for k, v in sorted(node_peaks.items(), key=lambda kv: -kv[1])[:8]},
            "samples": len(smi.samples),
        },
        "mesh_info": mesh_info,
        "glb": str(glb) if glb else None,
        "turntable": str(turn) if turn else None,
    }
    if glb and glb.exists():
        Path(a.glb_dir).mkdir(parents=True, exist_ok=True)
        dst = Path(a.glb_dir) / f"{a.name}.glb"
        shutil.copyfile(glb, dst)
        summary["glb_copy"] = str(dst)
        summary["glb_mb"] = round(os.path.getsize(dst) / 2**20, 1)
    if turn and turn.exists():
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
        dst = Path(a.png_dir) / f"spike-mesh-{a.name}.png"
        shutil.copyfile(turn, dst)
        summary["turntable_copy"] = str(dst)
    (results / f"{a.name}_{a.tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("node_order",)}, indent=1))


if __name__ == "__main__":
    main()
