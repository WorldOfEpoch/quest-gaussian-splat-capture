"""Render two saved scan checkpoints from identical physical camera views.

Run with the configured Nerfstudio GPU Python. This command writes evidence;
it never trains, publishes, changes photo selection, or changes headset settings.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_msssim import ssim
from gsplat import rasterization
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup

from bridge.config import load_config
from bridge.preview import register_job
from bridge.reconstruct import _incremental_training_split, validate_gaussian_ply
from bridge.store import Store


def load_job(root: Path, project_id: str, job_id: str, session_id: str) -> dict:
    folder = (root / "projects" / project_id / "reconstructions" / job_id).resolve()
    if not folder.is_relative_to(root / "projects" / project_id / "reconstructions"):
        raise ValueError("Job is outside the selected project")
    job = json.loads((folder / "job.json").read_text())
    if (job["session_id"] != session_id or job["project_id"] != project_id or
            job["status"] not in {"ready_for_blender", "candidate_for_review"}):
        raise ValueError("Comparison needs completed jobs from the same session")
    _, digest, size = validate_gaussian_ply(Path(job["artifact"]["path"]))
    if digest != job["artifact"]["sha256"] or size != job["artifact"]["size_bytes"]:
        raise ValueError("Saved Gaussian artifact differs from its receipt")
    return job


def evaluation_ids(job: dict) -> list[str]:
    transform_path = Path(job["processed_dir"]) / "transforms.json"
    meta = json.loads(transform_path.read_text())
    # Reuse the reconstruction owner's verified Nerfstudio split rule. Explicit
    # lists preserve parent holdouts; legacy seeds derive the recorded fraction
    # only after their dataparser configuration has been verified.
    split = _incremental_training_split(job, meta, meta["frames"])["filenames"]
    saved_split = job.get("training_split")
    if saved_split and (saved_split.get("transforms_sha256") != Store._file_sha256(transform_path)
                        or saved_split.get("filenames") != split):
        raise ValueError("Saved evaluation split differs from its frozen reconstruction")
    names = job.get("frame_manifest") or {
        item["capture_id"]: f"frame_{i:05d}{Path(item['image_name']).suffix}"
        for i, item in enumerate(sorted(job["input_captures"], key=lambda x: x["image_name"]), 1)}
    inverse = {}
    for cid, name in names.items():
        basename = Path(name).name
        if basename in inverse:
            raise ValueError("Ambiguous saved frame-to-capture mapping")
        inverse[basename] = cid
    try:
        evaluation = [inverse[Path(path).name] for path in split["val_filenames"]]
    except KeyError as error:
        raise ValueError("Evaluation frame lacks its original capture identity") from error
    if len(evaluation) != len(set(evaluation)):
        raise ValueError("Evaluation frame maps to duplicate original captures")
    return evaluation


def camera_for_capture(metadata: dict, alignment: dict, width: int) -> Cameras:
    session_pose = np.asarray(metadata["camera_to_session"], dtype=np.float64).reshape(4, 4)
    similarity = np.asarray(alignment["unity_from_splat"], dtype=np.float64).reshape(4, 4)
    flip = np.diag([1., 1., -1., 1.])
    pose = flip @ np.linalg.inv(similarity) @ session_pose @ flip
    pose[:3, :3] *= alignment["scale"]
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError("Comparison camera is not a rigid rotation")
    ratio = width / metadata["image_width"]
    intrinsics = metadata["intrinsics"]
    return Cameras(camera_to_worlds=torch.tensor(pose[:3], dtype=torch.float32)[None],
                   fx=intrinsics["fx"] * ratio, fy=intrinsics["fy"] * ratio,
                   cx=intrinsics["cx"] * ratio, cy=intrinsics["cy"] * ratio,
                   width=width, height=round(metadata["image_height"] * ratio),
                   camera_type=CameraType.PERSPECTIVE).to("cuda")


def render_job(job: dict, alignment: dict, captures: dict, ids: list[str],
               destination: Path, label: str, width: int) -> dict:
    _, pipeline, checkpoint, step = eval_setup(Path(job["training_config"]), test_mode="inference")
    results = {}
    with torch.no_grad():
        for cid in ids:
            capture = captures[cid]
            camera = camera_for_capture(capture["metadata"], alignment, width)
            rgb = pipeline.model.get_outputs_for_camera(camera)["rgb"].clamp(0, 1)
            raw = np.asarray(Image.open(capture["image_path"]).convert("RGB").resize(
                (int(camera.width.item()), int(camera.height.item())), Image.Resampling.LANCZOS),
                dtype=np.float32) / 255.
            prediction = rgb.detach().cpu().numpy()
            mse = float(np.mean((prediction - raw) ** 2))
            truth_tensor = torch.from_numpy(raw).to(rgb.device)
            results[cid] = {"l1": float(np.mean(np.abs(prediction - raw))),
                            "psnr": float(-10 * np.log10(max(mse, 1e-12))),
                            "ssim": float(ssim(rgb.permute(2, 0, 1)[None],
                                              truth_tensor.permute(2, 0, 1)[None], data_range=1.)),
                            "step": step}
            Image.fromarray(np.uint8(prediction * 255)).save(destination / f"{label}-{cid}.png")
            Image.fromarray(np.uint8(raw * 255)).save(destination / f"photo-{cid}.png")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return results


def ply_tensors(path: Path):
    validate_gaussian_ply(path)
    data = path.read_bytes()
    offset = data.index(b"end_header\n") + len(b"end_header\n")
    lines = data[:offset].decode("ascii").splitlines()
    count = int(next(line.split()[2] for line in lines if line.startswith("element vertex ")))
    properties = [line.split() for line in lines if line.startswith("property ")]
    if any(len(p) != 3 or p[1] != "float" for p in properties):
        raise ValueError("Comparison requires float Gaussian records")
    rows = np.frombuffer(data, dtype=[(p[2], "<f4") for p in properties],
                         offset=offset, count=count)
    def columns(names):
        return torch.from_numpy(np.stack([rows[n] for n in names], axis=-1).copy()).cuda()
    means = columns(("x", "y", "z"))
    quats = torch.nn.functional.normalize(columns([f"rot_{i}" for i in range(4)]), dim=-1)
    scales = columns([f"scale_{i}" for i in range(3)]).exp()
    opacity = columns(["opacity"])[:, 0].sigmoid()
    dc = columns([f"f_dc_{i}" for i in range(3)])[:, None, :]
    rest_names = sorted((p[2] for p in properties if p[2].startswith("f_rest_")),
                        key=lambda n: int(n.rsplit("_", 1)[1]))
    if rest_names:
        rest = columns(rest_names).reshape(count, 3, -1).transpose(1, 2)
        colors = torch.cat([dc, rest], dim=1).contiguous()
    else:
        colors = dc
    return means, quats, scales, opacity, colors


def compare_preview(job: dict, store: Store, candidate: Path, out: Path, width: int):
    """Compare the actual reduced PLYs at the installed headset's SH degree 0.

    Spark quantization and the optical headset display remain device checks.
    Neither this measurement nor file existence publishes a preview.
    """
    alignment = register_job(store, job)
    if not alignment["headset_preview_allowed"]:
        raise ValueError("Preview comparison requires verified camera alignment")
    ids = evaluation_ids(job)
    captures = {c["receipt"]["capture_id"]: c for c in store.export_session(job["session_id"])["captures"]}
    files = {"full": Path(job["artifact"]["path"]),
             "subset": Path(job["quest_preview"]["path"]), "merged": candidate}
    tensors = {name: ply_tensors(path) for name, path in files.items()}
    results = []
    with torch.no_grad():
        for index, cid in enumerate(ids):
            cam = camera_for_capture(captures[cid]["metadata"], alignment, width)
            pose = torch.eye(4, device="cuda")
            pose[:3] = cam.camera_to_worlds[0]
            cv_flip = torch.diag(torch.tensor([1., -1., -1., 1.], device="cuda"))
            view = (cv_flip @ torch.linalg.inv(pose))[None].contiguous()
            k = cam.get_intrinsics_matrices().to("cuda").contiguous()
            rendered = {}
            for label, params in tensors.items():
                rgb, _, _ = rasterization(*params, viewmats=view, Ks=k,
                    width=int(cam.width.item()), height=int(cam.height.item()),
                    sh_degree=0, packed=False, rasterize_mode="classic")
                rendered[label] = rgb[0].clamp(0, 1)
            scores = {}
            for label in ("subset", "merged"):
                delta = rendered[label] - rendered["full"]
                scores[label] = {"l1_vs_full": float(delta.abs().mean()),
                    "psnr_vs_full": float(-10 * torch.log10(delta.square().mean().clamp_min(1e-12))),
                    "ssim_vs_full": float(ssim(rendered[label].permute(2, 0, 1)[None],
                        rendered["full"].permute(2, 0, 1)[None], data_range=1.))}
            results.append({"capture_id": cid, **scores})
            figure = Image.new("RGB", (1920, 520), (20, 25, 35))
            draw = ImageDraw.Draw(figure)
            for column, label in enumerate(("full", "subset", "merged")):
                frame = Image.fromarray(np.uint8(rendered[label].cpu().numpy() * 255))
                frame.save(out / f"{label}-{cid}.png")
                figure.paste(frame.resize((640, 480), Image.Resampling.LANCZOS), (640 * column, 35))
                draw.text((640 * column + 10, 10), label.title() + " " + cid[:8], fill="white")
            figure.save(out / f"preview-comparison-{index + 1}.png")
    aggregate = {label: {key: float(np.mean([row[label][key] for row in results]))
                        for key in ("l1_vs_full", "psnr_vs_full", "ssim_vs_full")}
                 for label in ("subset", "merged")}
    result = {"job_id": job["job_id"], "sh_degree": 0, "view_count": len(ids),
              "artifacts": {label: {"path": str(path), "sha256": validate_gaussian_ply(path)[1]}
                            for label, path in files.items()},
              "aggregate": aggregate, "views": results, "published": False}
    (out / "preview-comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"aggregate": aggregate, "view_count": len(ids)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--baseline-job-id")
    parser.add_argument("--candidate-job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--preview-ply", type=Path)
    args = parser.parse_args()
    if not 320 <= args.width <= 1920:
        parser.error("Comparison width must be 320 through 1920")
    cfg = load_config(args.data_dir)
    store = Store(cfg.data_dir)
    project_id = store.session(args.session_id)["project_id"]
    after = load_job(cfg.data_dir, project_id, args.candidate_job_id, args.session_id)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.preview_ply:
        compare_preview(after, store, args.preview_ply.resolve(), out, args.width)
        return
    if not args.baseline_job_id:
        parser.error("--baseline-job-id is required unless comparing a preview PLY")
    before = load_job(cfg.data_dir, project_id, args.baseline_job_id, args.session_id)
    alignments = [register_job(store, job) for job in (before, after)]
    if not all(a["headset_preview_allowed"] for a in alignments):
        raise ValueError("Camera registration failed: " + json.dumps(alignments))
    old_eval, new_eval = evaluation_ids(before), evaluation_ids(after)
    # Keep all held-out views rather than choosing the most flattering angle.
    ids = list(dict.fromkeys(old_eval + new_eval))
    captures = {c["receipt"]["capture_id"]: c for c in store.export_session(args.session_id)["captures"]}
    metrics = [render_job(job, alignment, captures, ids, out, label, args.width)
               for job, alignment, label in zip((before, after), alignments, ("before", "after"))]
    rows = [{"capture_id": cid, "held_out_before": cid in old_eval,
             "held_out_after": cid in new_eval, "before": metrics[0][cid],
             "after": metrics[1][cid]} for cid in ids]
    heldout = [row for row in rows if row["held_out_after"]]
    aggregate = {label: {key: float(np.mean([row[label][key] for row in heldout]))
                        for key in ("l1", "psnr", "ssim")} for label in ("before", "after")}
    result = {"schema_version": 1, "session_id": args.session_id,
              "baseline_job_id": before["job_id"], "candidate_job_id": after["job_id"],
              "alignment": alignments, "view_count": len(rows),
              "candidate_heldout_count": len(heldout), "aggregate_heldout": aggregate,
              "views": rows, "visual_acceptance": "pending", "published": False}
    (out / "comparison.json").write_text(json.dumps(result, indent=2))
    for group, start in enumerate(range(0, len(ids), 3), 1):
        selected = ids[start:start + 3]
        figure = Image.new("RGB", (1920, 520 * len(selected)), (20, 25, 35))
        draw = ImageDraw.Draw(figure)
        for row, cid in enumerate(selected):
            for column, label in enumerate(("photo", "before", "after")):
                frame = Image.open(out / f"{label}-{cid}.png").resize((640, 480), Image.Resampling.LANCZOS)
                figure.paste(frame, (640 * column, 520 * row + 35))
                score = "" if label == "photo" else f"  PSNR {metrics[column-1][cid]['psnr']:.2f}"
                draw.text((640 * column + 10, 520 * row + 10),
                          label.title() + " " + cid[:8] + score, fill="white")
        figure.save(out / f"comparison-{group}.png")
    print(json.dumps({"aggregate_heldout": aggregate, "view_count": len(rows),
                      "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
