"""Optional external Nerfstudio reconstruction runner.

The COLMAP result has its own scale and coordinate frame. Successful output is
therefore a Blender inspection artifact, never an aligned headset preview.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageStat, ImageOps

from .models import utc_now
from .nerfstudio_resume import RESUME_RUNNER_VERSION, RESUME_WARMUP_MARKER
from .store import Store, StoreError


class ReconstructionError(RuntimeError):
    pass


INCREMENTAL_SPLIT_VERSION = 1


def accept_reviewed_candidate(store: Store, session_id: str, job_id: str, *,
                              expected_sha256: str, review: str) -> dict:
    """Preserve a specifically selected trial, including its real optimizer state.

    Selection is an explicit caller action, never an automatic quality decision.
    The original manifest and artifacts stay available and continuation uses this
    job's checkpoint rather than its older ancestor. This does not claim device
    acceptance or relax camera-registration thresholds.
    """
    session = store.session(session_id)
    job_id = str(uuid.UUID(job_id))
    root = store.root / 'projects' / session['project_id'] / 'reconstructions' / job_id
    manifest = root / 'job.json'
    if root.is_symlink() or manifest.is_symlink():
        raise ReconstructionError('Reviewed candidate cannot follow linked paths')
    job = json.loads(manifest.read_text(encoding='utf-8'))
    if (job.get('session_id') != session_id or job.get('job_id') != job_id or
            job.get('project_id') != session['project_id'] or
            job.get('status') not in {'candidate_for_review', 'ready_for_blender'} or
            job.get('artifact', {}).get('sha256') != expected_sha256 or
            not review.strip()):
        raise ReconstructionError('Selection does not identify a completed reviewed candidate')
    checkpoint = Path(job['checkpoint_dir']) / f"step-{_checkpoint_step(Path(job['checkpoint_dir'])):09d}.ckpt"
    files = {'splat.ply': root / 'splat.ply', 'checkpoint.ckpt': checkpoint,
             'config.yml': Path(job['training_config']),
             'transforms.json': Path(job['processed_dir']) / 'transforms.json',
             'dataparser_transforms.json': Path(job['training_config']).with_name('dataparser_transforms.json')}
    if job.get('quest_preview'):
        files['quest-preview.ply'] = root / 'quest-preview.ply'
    for path in files.values():
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.parent.resolve()):
            raise ReconstructionError('Candidate recovery evidence escapes its immutable job')
    if (validate_gaussian_ply(files['splat.ply'])[1] != expected_sha256 or
            Store._file_sha256(checkpoint) != job.get('export_checkpoint_sha256')):
        raise ReconstructionError('Selected model or optimizer checkpoint changed')
    with files['splat.ply'].open('rb') as handle:
        header = handle.read(128*1024).split(b'end_header\n', 1)[0].decode('ascii')
    if (not all(f'property float f_rest_{index}\n' in header+'\n' for index in range(45)) or
            not re.search(r'^\s*sh_degree:\s*3\s*$', files['config.yml'].read_text(), re.M)):
        raise ReconstructionError('Selected baseline must retain its full SH3 coefficients')
    recovery = root / 'accepted-baseline'
    recovery.mkdir(exist_ok=True)
    proofs = {}
    for name, path in files.items():
        digest = Store._file_sha256(path)
        saved = recovery / name
        if saved.exists():
            if saved.is_symlink() or Store._file_sha256(saved) != digest:
                raise ReconstructionError('Preserved baseline differs; refusing replacement')
        else:
            shutil.copy2(path, saved)
        proofs[name] = {'sha256': digest, 'size_bytes': path.stat().st_size}
    preimage = recovery / 'job-before-acceptance.json'
    if not preimage.exists():
        shutil.copy2(manifest, preimage)
    acceptance = {'schema_version': 1, 'job_id': job_id, 'session_id': session_id,
                  'model_sha256': expected_sha256, 'checkpoint_step': _checkpoint_step(checkpoint.parent),
                  'files': proofs, 'review': review, 'accepted_utc': utc_now(),
                  'preserve_sh_degree': 3, 'device_quality_acceptance': 'not_claimed'}
    if job.get('accepted_review'):
        if job['accepted_review']['files'] != proofs:
            raise ReconstructionError('Accepted baseline evidence changed')
        return job
    job.update(status='ready_for_blender', processing_stage='ready_for_blender',
               publication='accepted_review', accepted_review=acceptance)
    temporary = manifest.with_suffix('.accept.tmp')
    temporary.write_text(json.dumps(job, indent=2), encoding='utf-8')
    os.replace(temporary, manifest)
    return job


@contextmanager
def _worker_lock(root: Path):
    """Serialize every reconstruction using the same local GPU and data root."""
    lock_path = root / "reconstruction.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 5
        while True:
            try:
                handle.seek(0)
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise ReconstructionError("A reconstruction worker is already active for this data root")
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _serialized_reconstruction(function):
    @wraps(function)
    def guarded(store: Store, session_id: str, *args, **kwargs):
        with _worker_lock(store.root):
            return function(store, session_id, *args, **kwargs)
    return guarded


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ReconstructionError(f"Required executable is unavailable: {name}")
    return path


def _resumed_train_command(args: list[str]) -> list[str]:
    """Keep Nerfstudio's CLI with the process-local Splatfacto repairs."""
    executable = Path(args[0])
    python = executable.with_name("python.exe" if os.name == "nt" else "python")
    return [str(python), str(Path(__file__).with_name("nerfstudio_resume.py")), *args[1:]]


def _analyzed_views(processed: Path) -> int | None:
    """Measured COLMAP feature rows, independently of received photo count."""
    database = processed / 'colmap' / 'database.db'
    if not database.is_file():
        return None
    try:
        with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as db:
            return db.execute('SELECT count(*) FROM keypoints WHERE rows>0').fetchone()[0]
    except sqlite3.DatabaseError:
        return None


def _normalize_phone_jpeg(source: Path, destination: Path):
    """Honor actual JPEG EXIF pixels in a new copy, retaining the raw original."""
    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image)
        exif = normalized.getexif()
        exif[274] = 1
        normalized.save(destination, 'JPEG', quality=98, exif=exif)


def _run(args: list[str], log_path: Path, env: dict[str, str], timeout_seconds: int | None = None):
    with log_path.open("ab") as log:
        log.write(("\nCOMMAND: " + " ".join(args) + "\n").encode("utf-8"))
        log.flush()
        try:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    check=False, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise ReconstructionError(f"External command timed out after {timeout_seconds}s: {args[0]}; see {log_path}") from error
    if result.returncode:
        raise ReconstructionError(f"External command failed ({result.returncode}): {args[0]}; see {log_path}")


def validate_gaussian_ply(path: Path) -> tuple[int, str, int]:
    """Return vertex count, SHA-256 and byte size after a conservative PLY check."""
    if not path.is_file() or path.stat().st_size < 256:
        raise ReconstructionError("Gaussian PLY is absent or too small")
    with path.open("rb") as handle:
        header = handle.read(128 * 1024)
    boundary = header.find(b"end_header\n")
    if boundary < 0:
        raise ReconstructionError("PLY header is missing or oversized")
    try:
        lines = header[:boundary].decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ReconstructionError("PLY header is not ASCII") from error
    if not lines or lines[0] != "ply":
        raise ReconstructionError("Not a PLY file")
    if "format binary_little_endian 1.0" not in lines:
        raise ReconstructionError("Only binary little-endian Gaussian PLY is supported")
    count = 0
    properties = set()
    in_vertex = False
    stride = 0
    sizes = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
             "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
             "int": 4, "uint": 4, "int32": 4, "uint32": 4, "float": 4, "float32": 4,
             "double": 8, "float64": 8}
    for line in lines:
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex ")
            if in_vertex:
                try:
                    count = int(line.split()[2])
                except (IndexError, ValueError) as error:
                    raise ReconstructionError("Invalid PLY vertex count") from error
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) != 3 or parts[1] not in sizes:
                raise ReconstructionError("Unsupported Gaussian PLY vertex property")
            properties.add(parts[2])
            stride += sizes[parts[1]]
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}
    if count <= 0 or not required <= properties:
        raise ReconstructionError("PLY does not contain Gaussian vertices and required attributes")
    body_start = boundary + len(b"end_header\n")
    if path.stat().st_size < body_start + count * stride:
        raise ReconstructionError("Gaussian PLY vertex data is truncated")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return count, digest.hexdigest(), path.stat().st_size


def make_quest_preview(source: Path, destination: Path,
                       max_vertices: int = 240_000) -> dict:
    """Merge a display copy within Quest's budget, preserving the full source.

    Dropping every nth Gaussian breaks opacity/coverage, especially after
    densification. Use the pinned PlayCanvas pairwise simplifier instead.
    Missing tooling is an explicit failure; never silently publish a subset.
    """
    count, source_digest, size = validate_gaussian_ply(source)
    if not 1 <= max_vertices <= 250_000 or source.resolve() == destination.resolve():
        raise ReconstructionError("Invalid Quest preview target")
    with source.open("rb") as handle:
        header_bytes = handle.read(128 * 1024)
    marker = b"end_header\n"
    body_start = header_bytes.index(marker) + len(marker)
    header = header_bytes[:body_start].decode("ascii")
    lines = header.splitlines(keepends=True)
    elements = [line for line in lines if line.startswith("element ")]
    if len(elements) != 1 or not elements[0].startswith("element vertex "):
        raise ReconstructionError("Quest preview requires a vertex-only Gaussian PLY")
    properties = [line for line in lines if line.startswith("property ")]
    if not properties or any(not line.startswith("property float ") for line in properties):
        raise ReconstructionError("Quest preview requires fixed float Gaussian records")
    stride = len(properties) * 4
    if size != body_start + count * stride:
        raise ReconstructionError("Quest preview source has unexpected trailing data")
    target = min(count, max_vertices)
    temporary = destination.with_name(destination.stem + ".tmp.ply")
    method = "source_copy"
    try:
        if count <= target:
            shutil.copyfile(source, temporary)
        else:
            tool_root = Path(__file__).resolve().parents[1] / "tools" / "splat-transform"
            package_root = tool_root / "node_modules" / "@playcanvas" / "splat-transform"
            cli = package_root / "bin" / "cli.mjs"
            try:
                package = json.loads((package_root / "package.json").read_text(encoding="utf-8"))
                if package["version"] != "3.9.2" or not cli.is_file():
                    raise ValueError("Unexpected simplifier installation")
            except (OSError, KeyError, ValueError, TypeError) as error:
                raise ReconstructionError(
                    "Quest simplifier is unavailable; run npm ci --prefix tools/splat-transform") from error
            _run([_tool("node"), str(cli), "--overwrite", "--gpu", "cpu", str(source),
                  "--decimate-adaptive", str(target), str(temporary)],
                 destination.with_suffix(".merge.log"), os.environ.copy(), timeout_seconds=180)
            method = "playcanvas_adaptive_pairwise_merge"
        vertices, digest, output_size = validate_gaussian_ply(temporary)
        if vertices != target or output_size > 64 * 1024 * 1024:
            raise ReconstructionError("Quest preview exceeds its vertex or byte budget")
        if validate_gaussian_ply(source)[1] != source_digest:
            raise ReconstructionError("Quest preview source changed during simplification")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(destination), "sha256": digest, "size_bytes": output_size,
            "vertex_count": vertices, "source_sha256": source_digest,
            "method": method, "simplifier_version": "3.9.2" if count > target else None,
            "content_type": "application/octet-stream"}


def make_clean_online_preview(source: Path, destination: Path) -> dict:
    """Remove obvious floaters from a display copy, preserving the trained PLY.

    Thresholds scale with the model's dense core. This is only a quick-view
    filter, not evidence that the retained geometry is correct or complete.
    """
    count, source_digest, size = validate_gaussian_ply(source)
    if source.resolve() == destination.resolve() or count > 1_000_000 or size > 128 * 1024 * 1024:
        raise ReconstructionError("Invalid online display preview target")
    data = source.read_bytes()
    marker = b"end_header\n"
    body_start = data.index(marker) + len(marker)
    header = data[:body_start].decode("ascii")
    lines = header.splitlines(keepends=True)
    elements = [line for line in lines if line.startswith("element ")]
    properties = [line.split() for line in lines if line.startswith("property ")]
    if (len(elements) != 1 or elements[0] != f"element vertex {count}\n" or
            not properties or any(len(parts) != 3 or parts[1] != "float" for parts in properties)):
        raise ReconstructionError("Online display preview requires float-only Gaussian vertices")
    names = [parts[2] for parts in properties]
    if len(set(names)) != len(names) or size != body_start + count * len(names) * 4:
        raise ReconstructionError("Online display preview has invalid records")
    vertices = np.frombuffer(data, dtype=np.dtype([(name, "<f4") for name in names]),
                             count=count, offset=body_start)
    xyz = np.stack([vertices[name] for name in ("x", "y", "z")], axis=1)
    log_scales = np.stack([vertices[name] for name in ("scale_0", "scale_1", "scale_2")], axis=1)
    logits = vertices["opacity"]
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(log_scales).all(axis=1) & np.isfinite(logits)
    if finite.sum() < 3:
        raise ReconstructionError("Online display preview has too few finite Gaussians")
    center = np.median(xyz[finite], axis=0)
    distance = np.linalg.norm(xyz - center, axis=1)
    max_scale = np.exp(np.clip(log_scales, -30, 30)).max(axis=1)
    opacity = 1. / (1. + np.exp(-np.clip(logits, -30, 30)))
    radius_limit = max(5., float(np.percentile(distance[finite], 90)) * 3.)
    scale_limit = max(.16, float(np.percentile(max_scale[finite], 90)) * 8.)
    keep = finite & (distance <= radius_limit) & (max_scale <= scale_limit) & (opacity >= .08)
    if keep.sum() < max(3, int(count * .7)):
        # A room-wide or outdoor scan can have legitimate distant geometry.
        # Never erase most of it because the scene is unlike the first room.
        keep = finite
    retained = int(keep.sum())
    output_header = header.replace(f"element vertex {count}\n",
                                   f"element vertex {retained}\n", 1).encode("ascii")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as output:
            output.write(output_header)
            output.write(vertices[keep].tobytes())
            output.flush()
            os.fsync(output.fileno())
        checked_count, digest, output_size = validate_gaussian_ply(temporary)
        if checked_count != retained:
            raise ReconstructionError("Clean display preview vertex count differs")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"sha256": digest, "size_bytes": output_size, "vertex_count": retained,
            "source_sha256": source_digest, "removed_gaussians": count - retained,
            "method": "display_only_spatial_scale_opacity_filter"}


def _capture_key(item: dict) -> tuple[str, str]:
    receipt = item["receipt"]
    return receipt["capture_id"], receipt["image_sha256"]


def _parent_inputs(job: dict) -> set[tuple[str, str]]:
    return {(item["capture_id"], item["image_sha256"])
            for item in job.get("input_captures", [])}


def _checkpoint_step(checkpoint_dir: Path) -> int:
    steps = [int(match.group(1)) for path in checkpoint_dir.glob("step-*.ckpt")
             if (match := re.fullmatch(r"step-(\d+)\.ckpt", path.name))]
    if not steps:
        raise ReconstructionError("Parent has no numbered Nerfstudio checkpoint")
    return max(steps)


def _bounded_candidate_profile(parent: dict) -> dict:
    """Preserve this quick seed's actual density controls during a short continuation.

    Parsing the saved config as text avoids executing Nerfstudio's Python-object
    YAML tags. Other/older profiles must be reviewed before they are resumed.
    """
    profile = parent.get("training_profile") or {}
    if parent.get('accepted_review'):
        accepted = parent['accepted_review']
        config = Path(parent['training_config'])
        checkpoint = Path(parent['checkpoint_dir']) / f"step-{accepted['checkpoint_step']:09d}.ckpt"
        if (Store._file_sha256(config) != accepted['files']['config.yml']['sha256'] or
                Store._file_sha256(checkpoint) != accepted['files']['checkpoint.ckpt']['sha256'] or
                parent['artifact']['sha256'] != accepted['model_sha256']):
            raise ReconstructionError('Accepted continuation baseline changed')
        if (profile.get('num_downscales') != 0 or profile.get('resolution_schedule') != 250 or
                profile.get('scale_regularization') is not True):
            raise ReconstructionError('Accepted source lacks the reviewed full-resolution profile')
        iterations = parent['target_training_step'] - parent['checkpoint_start_step'] - 1
        _verify_bounded_candidate_config(config, iterations, num_downscales=0, stop_split_at=15000)
        return {**profile, 'stop_split_at': 15000}
    if (profile.get("version") != 2 or profile.get("num_downscales") != 1 or
            profile.get("resolution_schedule") != 250 or
            profile.get("full_resolution_from_step") != 250 or
            profile.get("scale_regularization") is not True):
        raise ReconstructionError("Frozen source does not have the reviewed quick training profile")
    _verify_bounded_candidate_config(Path(parent["training_config"]), 5000)
    return {"version": 2, "num_downscales": 1, "resolution_schedule": 250,
            "full_resolution_from_step": 250, "stop_split_at": 3500,
            "scale_regularization": True}


def _verify_bounded_candidate_config(path: Path, iteration_count: int, *,
                                     num_downscales=1, stop_split_at=3500) -> None:
    config = path.read_text(encoding="utf-8")
    required = {"max_num_iterations": str(iteration_count), "num_downscales": str(num_downscales),
                "resolution_schedule": "250", "stop_split_at": str(stop_split_at),
                "use_scale_regularization": "true", "use_bilateral_grid": "false"}
    for key, expected in required.items():
        found = re.findall(rf"^\s*{re.escape(key)}:\s*([^\s#]+)", config, re.M)
        if len(found) != 1 or found[0].lower() != expected:
            raise ReconstructionError(f"Bounded continuation has an unreviewed {key} setting")
    if not re.search(r"camera_optimizer:.*?\n\s+mode:\s*['\"]?off", config, re.S):
        raise ReconstructionError("Bounded continuation has camera-dependent training state")


def _verified_resume_warmup(log: str, start_step: int, registered_views: int,
                            expected_train_views: int | None = None) -> dict:
    records = re.findall(r"^" + re.escape(RESUME_WARMUP_MARKER) + r"(\{[^\r\n]+\})\r?$", log, re.M)
    try:
        evidence = json.loads(records[0]) if len(records) == 1 else {}
        integer_fields = ("start_step", "train_camera_count", "refine_every", "warmup_steps",
                          "growth_hold_through_step", "effective_refine_start_iter")
        if (not isinstance(evidence, dict) or
                any(type(evidence.get(key)) is not int for key in integer_fields) or
                evidence["start_step"] != start_step or
                not 0 < evidence["train_camera_count"] <= registered_views or
                (expected_train_views is not None and evidence["train_camera_count"] != expected_train_views) or
                evidence["refine_every"] <= 0 or
                evidence["warmup_steps"] != max(evidence["train_camera_count"], evidence["refine_every"]) or
                evidence["growth_hold_through_step"] != start_step + evidence["warmup_steps"] - 1 or
                evidence["effective_refine_start_iter"] < evidence["growth_hold_through_step"] or
                type(evidence.get("refinement_available")) is not bool):
            raise ValueError("Invalid refinement warmup evidence")
    except (ValueError, KeyError, TypeError) as error:
        raise ReconstructionError("Resumed training did not verify its exact refinement warmup") from error
    return evidence


def _completed_bounded_candidate(reconstructions: Path, *, source_id: str,
                                 source_sha256: str, checkpoint_sha256: str,
                                 project_id: str, session_id: str,
                                 input_revision: str, target_step: int,
                                 source_training_config_sha256: str | None = None,
                                 source_transforms_sha256: str | None = None,
                                 processing_mode: str = "same_data_bounded_candidate") -> dict | None:
    """Reuse an intact isolated candidate; a retry must not retrain duplicates."""
    for manifest in reconstructions.glob("*/job.json"):
        try:
            job = json.loads(manifest.read_text(encoding="utf-8"))
            if (job.get("job_id") != manifest.parent.name or
                    job.get("project_id") != project_id or job.get("session_id") != session_id or
                    job.get("status") != "candidate_for_review" or
                    job.get("processing_mode") != processing_mode or
                    job.get("frozen_source_job_id") != source_id or
                    job.get("frozen_source_sha256") != source_sha256 or
                    job.get("frozen_source_checkpoint_sha256") != checkpoint_sha256 or
                    job.get("input_revision") != input_revision or
                    job.get("target_training_step") != target_step or
                    job.get("completed_training_step") != target_step - 1 or
                    job.get("resume_runner_version") != RESUME_RUNNER_VERSION or
                    job.get("optimizer_binding_verified") is not True or
                    job.get("opacity_reset_compatibility") not in
                    {"patched_faulty", "already_corrected"}):
                continue
            root = manifest.parent.resolve()
            if not root.is_relative_to(reconstructions.resolve()):
                continue
            if processing_mode == "new_capture_incremental_bounded_candidate":
                split = job.get("training_split") or {}
                processed = Path(job["processed_dir"]).resolve()
                if (split.get("version") != INCREMENTAL_SPLIT_VERSION or
                        split.get("method") != "preserve_parent_membership" or
                        split.get("source_training_config_sha256") != source_training_config_sha256 or
                        split.get("source_transforms_sha256") != source_transforms_sha256 or
                        not processed.is_relative_to(root) or
                        Store._file_sha256(processed / "transforms.json") !=
                        split.get("transforms_sha256")):
                    continue
            artifact = job["artifact"]
            path = Path(artifact["path"]).resolve()
            checkpoint = Path(job["checkpoint_dir"]).resolve()
            config = Path(job["training_config"]).resolve()
            if (path != root / "splat.ply" or not checkpoint.is_relative_to(root) or
                    not config.is_relative_to(root) or not config.is_file() or
                    _checkpoint_step(checkpoint) != target_step - 1 or
                    artifact.get("sha256") == source_sha256):
                continue
            saved_checkpoint = checkpoint / f"step-{target_step - 1:09d}.ckpt"
            if (not saved_checkpoint.is_file() or
                    Store._file_sha256(saved_checkpoint) != job.get("export_checkpoint_sha256")):
                continue
            count, digest, size = validate_gaussian_ply(path)
            if (count == artifact["vertex_count"] and digest == artifact["sha256"] and
                    size == artifact["size_bytes"]):
                return job
        except (OSError, ValueError, TypeError, KeyError, ReconstructionError):
            continue
    return None


def _valid_parent(job: dict, session_id: str, inputs: set[tuple[str, str]],
                  project_root: Path) -> bool:
    if job.get("session_id") != session_id or job.get("status") != "ready_for_blender":
        return False
    old = _parent_inputs(job)
    if not old or not old <= inputs:
        return False
    try:
        artifact = job["artifact"]
        paths = [Path(job[field]).resolve() for field in
                 ("processed_dir", "checkpoint_dir", "training_config")]
        artifact_path = Path(artifact["path"]).resolve()
        root = project_root.resolve()
        if any(not path.is_relative_to(root) for path in [*paths, artifact_path]):
            return False
        if not paths[0].is_dir() or not paths[1].is_dir() or not paths[2].is_file():
            return False
        if not artifact_path.is_file() or artifact_path.stat().st_size != artifact["size_bytes"]:
            return False
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != artifact["sha256"]:
            return False
        _checkpoint_step(paths[1])
    except (OSError, KeyError, TypeError, ValueError, ReconstructionError):
        return False
    return True


def _parent_frame_map(parent: dict) -> dict[str, str]:
    """Reconstruct Nerfstudio's sorted frame rename, checking real pixels.

    Older jobs did not record a frame map. They used ns-process-data's
    sorted list_images + frame_00001 sequence. JPEG recompression changes the
    byte hash, so compare decoded pixels instead of pretending hashes match.
    """
    saved = parent.get("frame_manifest")
    if saved:
        return dict(saved)
    source_dir = Path(parent["processed_dir"]).parent / "images"
    ordered = sorted(source_dir.glob("*.jpg"))
    captures = parent.get("input_captures", [])
    if len(ordered) != len(captures):
        raise ReconstructionError("Parent source image count differs from its receipt manifest")
    expected = {item["capture_id"]: item["image_sha256"] for item in captures}
    processed = Path(parent["processed_dir"]) / "images"
    mapping = {}
    for index, source in enumerate(ordered, 1):
        capture_id = source.stem
        target = processed / f"frame_{index:05d}.jpg"
        if expected.get(capture_id) != hashlib.sha256(source.read_bytes()).hexdigest() or not target.is_file():
            raise ReconstructionError("Parent frame mapping cannot be verified")
        with Image.open(source) as a, Image.open(target) as b:
            if a.size != b.size:
                raise ReconstructionError("Parent processed image dimensions changed")
            a.thumbnail((160, 120))
            b.thumbnail((160, 120))
            difference = sum(ImageStat.Stat(ImageChops.difference(a.convert("RGB"), b.convert("RGB"))).mean) / 3
            if difference > 12:
                raise ReconstructionError("Parent frame order does not match source photos")
        mapping[capture_id] = target.name
    return mapping


def _read_colmap_images(text_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read only pose lines; POINTS2D lines may be extremely large."""
    result = {}
    with text_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(maxsplit=9)
            if len(parts) != 10:
                raise ReconstructionError("Unexpected COLMAP images.txt pose record")
            q = np.asarray([float(value) for value in parts[1:5]], dtype=float)
            t = np.asarray([float(value) for value in parts[5:8]], dtype=float)
            if not np.isfinite(q).all() or not np.isfinite(t).all():
                raise ReconstructionError("COLMAP pose is nonfinite")
            result[parts[9]] = (q, t)
            next(handle, None)  # The observations line.
    return result


def _colmap_camera_to_saved(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Match Nerfstudio colmap_to_json's COLMAP/OpenGL axis conversion."""
    w, x, y, z = q / np.linalg.norm(q)
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -rotation.T @ t
    pose[:3, 1:3] *= -1
    pose = pose[[0, 2, 1, 3], :]
    pose[2, :] *= -1
    return pose


def _normalization(parent: dict) -> tuple[np.ndarray, float, str]:
    existing = parent.get("normalization")
    if existing:
        transform = np.asarray(existing["colmap_saved_to_training"], dtype=float)
        scale = float(existing["scale"])
        root_id = existing["root_job_id"]
    else:
        path = Path(parent["training_config"]).parent / "dataparser_transforms.json"
        if not path.is_file():
            raise ReconstructionError("Parent training normalization receipt is missing")
        record = json.loads(path.read_text(encoding="utf-8"))
        # Nerfstudio persists a transform from original COLMAP world space,
        # while transforms.json camera matrices are already in saved space.
        # Undo the processor's applied_transform before using it on those poses.
        original_to_training = np.eye(4)
        original_to_training[:3, :] = np.asarray(record["transform"], dtype=float)
        saved = json.loads((Path(parent["processed_dir"]) / "transforms.json").read_text(encoding="utf-8"))
        original_to_saved = np.eye(4)
        original_to_saved[:3, :] = np.asarray(saved["applied_transform"], dtype=float)
        transform = (original_to_training @ np.linalg.inv(original_to_saved))[:3, :]
        scale = float(record["scale"])
        root_id = parent["job_id"]
    if transform.shape != (3, 4) or not np.isfinite(transform).all() or not math.isfinite(scale) or scale <= 0:
        raise ReconstructionError("Parent normalization is invalid")
    return transform, scale, root_id


def _pose_in_training(q: np.ndarray, t: np.ndarray,
                      transform: np.ndarray, scale: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :] = transform
    pose = matrix @ _colmap_camera_to_saved(q, t)
    pose[:3, 3] *= scale
    return pose


def _check_quest_pose_consistency(snapshot: dict, frame_map: dict[str, str],
                                  model: dict[str, tuple[np.ndarray, np.ndarray]],
                                  new_names: set[str], transform: np.ndarray,
                                  scale: float) -> dict:
    """Guard a COLMAP extension with the real same-session Quest translations.

    This checks spatial agreement only. It does not establish the camera basis,
    metric overlay accuracy, or a headset preview alignment.
    """
    metadata = {item["receipt"]["capture_id"]: item["metadata"]
                for item in snapshot["captures"]}
    old_quest, old_colmap, new_quest, new_colmap = [], [], [], []
    for capture_id, frame_name in frame_map.items():
        if frame_name not in model or capture_id not in metadata:
            continue
        matrix = np.asarray(metadata[capture_id]["camera_to_session"], dtype=float).reshape(4, 4)
        if not np.isfinite(matrix).all():
            raise ReconstructionError("Quest pose for registered photo is nonfinite")
        q, t = model[frame_name]
        center = _pose_in_training(q, t, transform, scale)[:3, 3]
        if frame_name in new_names:
            new_quest.append(matrix[:3, 3])
            new_colmap.append(center)
        else:
            old_quest.append(matrix[:3, 3])
            old_colmap.append(center)
    if len(old_quest) < 3 or not new_quest:
        raise ReconstructionError("Need three previously registered poses and one new registered pose")
    source = np.asarray(old_quest)
    target = np.asarray(old_colmap)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered = source - source_center
    if np.linalg.matrix_rank(centered, tol=1e-4) < 2:
        raise ReconstructionError("Old Quest camera positions do not span a stable alignment")
    covariance = (target - target_center).T @ centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt  # Reflection is allowed: Unity and COLMAP differ in handedness.
    fit_scale = float(singular.sum() / np.mean(np.sum(centered * centered, axis=1)))
    if not math.isfinite(fit_scale) or fit_scale <= 0:
        raise ReconstructionError("Quest/COLMAP similarity scale is invalid")
    old_error = np.linalg.norm(fit_scale * (source - source_center) @ rotation.T +
                               target_center - target, axis=1)
    new_source = np.asarray(new_quest)
    new_target = np.asarray(new_colmap)
    new_error = np.linalg.norm(fit_scale * (new_source - source_center) @ rotation.T +
                               target_center - new_target, axis=1)
    old_rms = float(np.sqrt(np.mean(old_error ** 2)))
    new_max = float(new_error.max())
    # Nerfstudio normally normalizes the capture span to roughly unit scale.
    # Refuse a new registration that is inconsistent with the old tracked path.
    if old_rms > 0.3 or new_max > max(0.35, old_rms * 3):
        raise ReconstructionError(
            f"New COLMAP poses disagree with Quest tracking (old RMS {old_rms:.3f}, "
            f"new max {new_max:.3f} in training units)")
    return {"old_rms_training_units": old_rms,
            "new_max_training_units": new_max,
            "registered_new_views": len(new_quest),
            "headset_alignment_verified": False}


def _incremental_training_split(parent: dict, source_transforms: dict,
                                frames: list[dict]) -> dict:
    """Freeze actual parent memberships; only new registered frames get a split.

    Match the installed Nerfstudio fraction parser's sorted-frame/ceil/linspace
    rule without loading its Python-object YAML. Explicit parent split lists
    take precedence, exactly as in that parser. Unknown policies fail closed.
    """
    def paths(rows):
        if not isinstance(rows, list):
            raise ReconstructionError("Parent training split has invalid frames")
        values = []
        for row in rows:
            value = row.get("file_path") if isinstance(row, dict) else None
            if (not isinstance(value, str) or not value or "\\" in value or
                    Path(value).is_absolute() or Path(value).drive or ".." in Path(value).parts or
                    Path(value).as_posix() != value):
                raise ReconstructionError("Parent training split has an invalid frame path")
            values.append(value)
        if len(values) != len(set(values)):
            raise ReconstructionError("Parent training split has duplicate frame paths")
        return sorted(values)

    old_paths, current_paths = paths(source_transforms.get("frames", [])), paths(frames)
    if not old_paths or not set(old_paths) <= set(current_paths):
        raise ReconstructionError("Incremental training split lost a parent registered frame")
    config = Path(parent["training_config"]).read_text(encoding="utf-8")
    parser = re.findall(r"^([ \t]*)dataparser:\s*!!python/object:"
        r"nerfstudio\.data\.dataparsers\.nerfstudio_dataparser\.NerfstudioDataParserConfig\s*\n"
        r"\1[ \t]+_target:\s*!!python/name:"
        r"nerfstudio\.data\.dataparsers\.nerfstudio_dataparser\.Nerfstudio(?:\s|$)", config, re.M)
    if len(parser) != 1:
        raise ReconstructionError("Parent training split dataparser is unreviewed")
    matches = re.findall(r"^\s*train_split_fraction:\s*([^\s#]+)", config, re.M)
    try:
        fraction = float(matches[0]) if len(matches) == 1 else float("nan")
    except ValueError:
        fraction = float("nan")
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ReconstructionError("Parent training split fraction is unverified")

    def fraction_split(ordered):
        count = math.ceil(len(ordered) * fraction)
        train_indices = set(np.linspace(0, len(ordered) - 1, count, dtype=int)) if ordered else set()
        train = [value for index, value in enumerate(ordered) if index in train_indices]
        evaluation = [value for index, value in enumerate(ordered) if index not in train_indices]
        return {"train": train, "val": evaluation, "test": list(evaluation)}

    keys = {name: name + "_filenames" for name in ("train", "val", "test")}
    if any(key in source_transforms for key in keys.values()):
        old = {}
        for name, key in keys.items():
            values = source_transforms.get(key)
            if (not isinstance(values, list) or any(not isinstance(value, str) for value in values) or
                    len(values) != len(set(values)) or not set(values) <= set(old_paths)):
                raise ReconstructionError("Parent training split filenames are incomplete or invalid")
            old[name] = sorted(values)
        if (set(old["train"]) & (set(old["val"]) | set(old["test"])) or
                set(old["train"]) | set(old["val"]) | set(old["test"]) != set(old_paths)):
            raise ReconstructionError("Parent training split overlaps or omits registered frames")
    else:
        modes = re.findall(r"^\s*eval_mode:\s*([^\s#]+)", config, re.M)
        if len(modes) != 1 or modes[0].strip("'\"") != "fraction":
            raise ReconstructionError("Parent training split policy is unreviewed")
        old = fraction_split(old_paths)
    added = fraction_split(sorted(set(current_paths) - set(old_paths)))
    return {"version": INCREMENTAL_SPLIT_VERSION, "method": "preserve_parent_membership",
            "parent_job_id": parent["job_id"], "new_train_split_fraction": fraction,
            "source_training_config_sha256": Store._file_sha256(Path(parent["training_config"])),
            "parent_membership": old, "new_membership": added,
            "filenames": {keys[name]: sorted(old[name] + added[name]) for name in keys}}


def _prepare_incremental(parent: dict, snapshot: dict, root: Path,
                         tools: dict[str, str], log_path: Path,
                         env: dict[str, str], *, mesh_only: bool = False) -> dict:
    """Extend a copied COLMAP model without changing the parent job or its frame."""
    previous = Path(parent["processed_dir"])
    if not (previous / "colmap" / "database.db").is_file() or not (
        previous / "colmap" / "sparse" / "0" / "images.bin").is_file():
        raise ReconstructionError("Parent COLMAP sparse model/database is unavailable")
    config = Path(parent["training_config"]).read_text(encoding="utf-8")
    if not re.search(r"camera_optimizer:.*?\n\s+mode:\s*['\"]?off", config, re.S) or not re.search(
        r"use_bilateral_grid:\s*false", config
    ):
        raise ReconstructionError("Checkpoint has camera-dependent training state; incremental restore is unsafe")
    source_transforms = json.loads((previous / "transforms.json").read_text(encoding="utf-8"))
    _incremental_training_split(parent, source_transforms, source_transforms.get("frames", []))
    transform, scale, root_job_id = _normalization(parent)
    frame_map = _parent_frame_map(parent)
    old_keys = _parent_inputs(parent)
    new_items = [item for item in snapshot["captures"] if _capture_key(item) not in old_keys]
    if not new_items:
        raise ReconstructionError("Incremental build has no newly received photos")
    old_ids = {capture_id for capture_id, _ in old_keys}
    old_metadata = [item["metadata"] for item in snapshot["captures"]
                    if item["receipt"]["capture_id"] in old_ids]
    if len(old_metadata) != len(old_keys):
        raise ReconstructionError("Parent captures are missing from current session")
    # COLMAP's existing camera ID may only be reused for the same physical
    # camera calibration. A different eye/camera or crop needs a fresh model.
    calibration_fields = ("camera_id", "image_width", "image_height",
                          "intrinsics", "calibration_revision", "image_orientation")
    anchor = old_metadata[0]
    for item in [*old_metadata[1:], *(entry["metadata"] for entry in new_items)]:
        def camera_differs(field):
            if field == 'intrinsics' and mesh_only:
                # Photo-only MVS refits camera poses independently. Minor SDK
                # intrinsic drift below one pixel does not imply a changed eye
                # or crop. Live checkpoint continuation remains exactly strict.
                a, b = anchor.get(field), item.get(field)
                return (not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b) or
                        any(not isinstance(a[k], (int, float)) or not isinstance(b[k], (int, float)) or
                            not np.isfinite([a[k], b[k]]).all() or abs(a[k]-b[k]) > 1 for k in a))
            return item.get(field) != anchor.get(field)
        if any(camera_differs(field) for field in calibration_fields):
            raise ReconstructionError("Incremental photo camera calibration differs from existing COLMAP camera")

    processed = root / "processed"
    shutil.copytree(previous, processed)
    old_sparse = processed / "colmap" / "sparse" / "0"
    database = processed / "colmap" / "database.db"
    names = {path.name for path in (processed / "images").glob("frame_*.jpg")}
    if set(frame_map.values()) != names:
        raise ReconstructionError("Parent frame manifest differs from its processed images")
    next_index = max(int(re.fullmatch(r"frame_(\d+)\.jpg", name).group(1)) for name in names) + 1
    new_names = set()
    for item in new_items:
        source = Path(item["image_path"])
        if source.suffix.lower() != ".jpg" or hashlib.sha256(source.read_bytes()).hexdigest() != item["receipt"]["image_sha256"]:
            raise ReconstructionError("New image hash or JPEG encoding differs from its receipt")
        capture_id = item["receipt"]["capture_id"]
        name = f"frame_{next_index:05d}.jpg"
        next_index += 1
        with Image.open(source) as image:
            if image.size != (anchor["image_width"], anchor["image_height"]):
                raise ReconstructionError("Incremental photo dimensions differ from existing camera")
            shutil.copyfile(source, processed / "images" / name)
            for factor in (2, 4, 8):
                destination = processed / f"images_{factor}"
                if destination.is_dir():
                    reduced = image.resize((image.width // factor, image.height // factor), Image.Resampling.LANCZOS)
                    reduced.save(destination / name, format="JPEG", quality=95)
        frame_map[capture_id] = name
        new_names.add(name)

    with sqlite3.connect(database) as db:
        cameras = [row[0] for row in db.execute("SELECT camera_id FROM cameras")]
    if len(cameras) != 1:
        raise ReconstructionError("Incremental registration requires one unchanged COLMAP camera")
    new_list = root / "new-images.txt"
    new_list.write_text("\n".join(sorted(new_names)) + "\n", encoding="utf-8")
    _run([tools["colmap"], "feature_extractor", "--database_path", str(database),
          "--image_path", str(processed / "images"), "--image_list_path", str(new_list),
          "--ImageReader.existing_camera_id", str(cameras[0]),
          "--FeatureExtraction.use_gpu", "1"], log_path, env)

    parent_text = root / "parent-colmap-text"
    parent_text.mkdir()
    _run([tools["colmap"], "model_converter", "--input_path", str(old_sparse),
          "--output_path", str(parent_text), "--output_type", "TXT"], log_path, env)
    old_model = _read_colmap_images(parent_text / "images.txt")
    old_registered = set(old_model)
    pairs = set()
    for new_name in new_names:
        for old_name in old_registered:
            pairs.add(tuple(sorted((new_name, old_name))))
        for other in new_names:
            if other != new_name:
                pairs.add(tuple(sorted((new_name, other))))
    pair_file = root / "match-pairs.txt"
    pair_file.write_text("\n".join(" ".join(pair) for pair in sorted(pairs)) + "\n", encoding="utf-8")
    _run([tools["colmap"], "matches_importer", "--database_path", str(database),
          "--match_list_path", str(pair_file), "--match_type", "pairs",
          "--FeatureMatching.use_gpu", "1"], log_path, env)
    extended = root / "extended-sparse"
    extended.mkdir()
    _run([tools["colmap"], "image_registrator", "--database_path", str(database),
          "--input_path", str(old_sparse), "--output_path", str(extended),
          "--Mapper.fix_existing_frames", "1"], log_path, env)
    triangulated = root / "triangulated-sparse"
    triangulated.mkdir()
    try:
        _run([tools["colmap"], "point_triangulator", "--database_path", str(database),
              "--image_path", str(processed / "images"), "--input_path", str(extended),
              "--output_path", str(triangulated), "--clear_points", "0",
              "--Mapper.fix_existing_frames", "1"], log_path, env)
        candidate_sparse = triangulated
        triangulation = "completed"
    except ReconstructionError:
        candidate_sparse = extended
        triangulation = "failed; registered camera poses retained"
    model_text = root / "extended-colmap-text"
    model_text.mkdir()
    _run([tools["colmap"], "model_converter", "--input_path", str(candidate_sparse),
          "--output_path", str(model_text), "--output_type", "TXT"], log_path, env)
    model = _read_colmap_images(model_text / "images.txt")
    if not old_registered <= set(model):
        raise ReconstructionError("Existing registered COLMAP images disappeared")
    for name in old_registered:
        old_q, old_t = old_model[name]
        new_q, new_t = model[name]
        if not np.allclose(old_q, new_q, atol=1e-5) or not np.allclose(old_t, new_t, atol=1e-5):
            raise ReconstructionError("COLMAP moved an existing camera; parent frame cannot be reused")
    registered_new = new_names & set(model)
    if not registered_new:
        raise ReconstructionError("No new photo registered in the existing COLMAP frame")
    consistency = ({'status': 'not_checked_mesh_only',
                    'physical_headset_alignment': 'unverified', 'eligible_for_live_training': False}
                   if mesh_only else _check_quest_pose_consistency(snapshot, frame_map, model,
                                                 registered_new, transform, scale))
    # Preserve a sparse continuation for the next revision; the previous job
    # remains untouched. It is not itself proof that new points entered the splat.
    shutil.copytree(candidate_sparse, old_sparse, dirs_exist_ok=True)
    source_frames = source_transforms["frames"]
    frames = []
    for capture_id, name in frame_map.items():
        if name not in model:
            continue
        q, t = model[name]
        frames.append({"file_path": "images/" + name,
                       "transform_matrix": _pose_in_training(q, t, transform, scale).tolist()})
    split = _incremental_training_split(parent, {**source_transforms, "frames": source_frames}, frames)
    source_transforms["frames"] = frames
    source_transforms.update(split["filenames"])
    source_transforms["orientation_override"] = "none"
    source_transforms["applied_transform"] = np.eye(4)[:3, :].tolist()
    source_transforms.pop("applied_scale", None)
    source_transforms.pop("ply_file_path", None)
    (processed / "transforms.json").write_text(json.dumps(source_transforms, indent=2), encoding="utf-8")
    split["transforms_sha256"] = Store._file_sha256(processed / "transforms.json")
    split["source_transforms_sha256"] = Store._file_sha256(previous / "transforms.json")
    used_ids = {capture_id for capture_id, name in frame_map.items() if name in model}
    all_ids = {item["receipt"]["capture_id"] for item in snapshot["captures"]}
    return {"processed_dir": str(processed), "frame_manifest": frame_map,
            "registered_views": len(frames), "used_capture_ids": sorted(used_ids),
            "rejected_capture_ids": sorted(all_ids - used_ids),
            "new_registered_views": len(registered_new),
            "new_sparse_triangulation": triangulation,
            "training_split": split,
            "normalization": {"root_job_id": root_job_id,
                              "colmap_saved_to_training": transform.tolist(), "scale": scale},
            "pose_consistency": consistency}


@_serialized_reconstruction
def build_session(store: Store, session_id: str, native_orientation_reviewed: bool = False,
                  quality: str = "final", fresh_baseline: bool = False,
                  candidate: bool = False, capture_ids: list[str] | None = None,
                  house_source: bool = False, frozen_source_job_id: str | None = None,
                  target_training_step: int | None = None,
                  incremental_new_captures: bool = False) -> dict:
    """Build one immutable scan version from the currently received capture snapshot."""
    if quality not in {"quick", "final"}:
        raise ReconstructionError("Quality must be quick or final")
    if fresh_baseline and quality != "quick":
        raise ReconstructionError("Fresh baseline is an explicit quick-build operation")
    if candidate and quality == "quick" and not fresh_baseline:
        raise ReconstructionError("A quick quality candidate requires an explicit fresh baseline")
    if candidate and house_source:
        raise ReconstructionError("A House source cannot also be a review candidate")
    explicit_source = frozen_source_job_id is not None or target_training_step is not None
    bounded_incremental = incremental_new_captures
    bounded_candidate = explicit_source and not bounded_incremental
    if bounded_incremental and not explicit_source:
        raise ReconstructionError("Bounded incremental review needs a source job and target step")
    if explicit_source and (not candidate or quality != "final" or fresh_baseline or
                              house_source or capture_ids is not None or
                              frozen_source_job_id is None or type(target_training_step) is not int):
        raise ReconstructionError("Bounded continuation requires a final review candidate, a source job and a target step")
    # Check every prerequisite before creating a job folder or claiming work.
    frozen_parent = None
    if explicit_source:
        source_id = str(uuid.UUID(frozen_source_job_id))
        session = store.session(session_id)
        source_root = store.root / "projects" / session["project_id"] / "reconstructions" / source_id
        try:
            source_manifest = source_root / "job.json"
            if source_root.is_symlink() or source_manifest.is_symlink():
                raise ReconstructionError("Frozen source path is not a regular project job")
            frozen_parent = json.loads(source_manifest.read_text(encoding="utf-8"))
            if (frozen_parent.get("job_id") != source_id or
                    (frozen_parent.get("quality") != "quick" and not frozen_parent.get('accepted_review')) or
                    frozen_parent.get("project_id") != session["project_id"] or
                    frozen_parent.get("session_id") != session_id):
                raise ReconstructionError("Frozen quick source belongs to a different job or session")
            if frozen_parent.get("source_image_orientations") == ["unity_texture_native"] and (
                    frozen_parent.get("native_orientation_reviewed") is not True or
                    native_orientation_reviewed is not True):
                raise ReconstructionError("Frozen native Quest photos need the source's recorded review and the explicit continuation review flag")
            frozen_ids = [item["capture_id"] for item in frozen_parent["input_captures"]]
            if len(frozen_ids) < 6 or len(frozen_ids) != len(set(frozen_ids)):
                raise ReconstructionError("Frozen quick source has an invalid photo set")
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ReconstructionError("Frozen quick source is unavailable") from error
        snapshot = (store.export_session(session_id) if bounded_incremental else
                    store.export_session(session_id, capture_ids=frozen_ids))
    else:
        snapshot = store.export_session(session_id)
    if capture_ids is not None:
        selected = {str(uuid.UUID(value)) for value in capture_ids}
        if len(selected) != len(capture_ids):
            raise ReconstructionError("Frozen capture snapshot contains duplicate identities")
        snapshot["captures"] = [item for item in snapshot["captures"]
                                if item["receipt"]["capture_id"] in selected]
        if len(snapshot["captures"]) != len(selected):
            raise ReconstructionError("Frozen captures no longer belong to this session")
    minimum_views = 3 if quality == "quick" else 6
    if len(snapshot["captures"]) < minimum_views:
        raise ReconstructionError(f"At least {minimum_views} received views are required for a {quality} attempt; this does not establish sufficient overlap")
    sources = {item["metadata"]["capture_source"] for item in snapshot["captures"]}
    phone = sources == {'android_camera2'}
    if sources != {"quest_camera"} and not phone:
        raise ReconstructionError("Only physical Quest captures may enter an operational build")
    if bounded_incremental and sources != {"quest_camera"}:
        raise ReconstructionError("Bounded incremental review requires same-session physical Quest photos")
    orientations = {item["metadata"]["image_orientation"] for item in snapshot["captures"]}
    if not (phone and orientations == {'exif'}) and orientations != {"upright"} and not (
        orientations == {"unity_texture_native"} and native_orientation_reviewed
    ):
        raise ReconstructionError("Quest camera image orientation is unverified; inspect physical images and explicitly approve native orientation before reconstruction")
    revision_source = "\n".join(sorted(f"{item['receipt']['capture_id']}:{item['receipt']['image_sha256']}"
                                        for item in snapshot["captures"]))
    input_revision = hashlib.sha256(revision_source.encode("ascii")).hexdigest()
    project_id = snapshot["session"]["project_id"]
    reconstructions = store.root / "projects" / project_id / "reconstructions"
    current_inputs = {_capture_key(item) for item in snapshot["captures"]}
    candidate_profile = None
    if explicit_source:
        if not _valid_parent(frozen_parent, session_id, current_inputs, reconstructions) or \
                (bounded_candidate and _parent_inputs(frozen_parent) != current_inputs) or \
                (bounded_incremental and _parent_inputs(frozen_parent) == current_inputs):
            raise ReconstructionError("Frozen quick source or its original photo receipts changed")
        candidate_profile = _bounded_candidate_profile(frozen_parent)
        source_step = _checkpoint_step(Path(frozen_parent["checkpoint_dir"]))
        additional_steps = target_training_step - source_step - 1
        if bounded_incremental:
            if not 1000 <= additional_steps <= 15000:
                raise ReconstructionError("Incremental review needs 1000 to 15000 additional optimizer steps")
        elif not 500 <= additional_steps <= min(10000, 30000 - source_step - 1):
            raise ReconstructionError("Candidate target must be 500 to 10000 steps beyond its source checkpoint")
        source_checkpoint = next(path for path in Path(frozen_parent["checkpoint_dir"]).glob("step-*.ckpt")
                                 if re.fullmatch(rf"step-0*{source_step}\.ckpt", path.name))
        source_checkpoint_sha256 = Store._file_sha256(source_checkpoint)
        completed = _completed_bounded_candidate(reconstructions,
            source_id=frozen_parent["job_id"], source_sha256=frozen_parent["artifact"]["sha256"],
            checkpoint_sha256=source_checkpoint_sha256, project_id=project_id,
            session_id=session_id, input_revision=input_revision, target_step=target_training_step,
            source_training_config_sha256=Store._file_sha256(Path(frozen_parent["training_config"])),
            source_transforms_sha256=Store._file_sha256(Path(frozen_parent["processed_dir"]) / "transforms.json"),
            processing_mode=("new_capture_incremental_bounded_candidate" if bounded_incremental else
                             "same_data_bounded_candidate"))
        if completed:
            return completed
    same_data_parent = None
    revision_parent = None
    for existing_path in (() if explicit_source else
                          (reconstructions.glob("*/job.json") if reconstructions.exists() else ())):
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if not _valid_parent(existing, session_id, current_inputs, reconstructions):
                continue
            if (candidate and existing.get("publication") == "candidate_review" and
                    existing.get("resume_runner_version") != RESUME_RUNNER_VERSION):
                # A review candidate trained before the resume repairs cannot
                # satisfy a new private quality request for the same photos.
                continue
            if _parent_inputs(existing) == current_inputs:
                if not fresh_baseline and existing.get("quality", "final") == quality:
                    # A House source stays available for mesh/export, but a
                    # later ordinary request must create its own publishable
                    # job rather than silently reusing that offline source.
                    if house_source or existing.get("publication") != "offline_house_source":
                        return existing
                if quality == "final" and existing.get("quality") == "quick":
                    same_data_parent = existing
            elif (revision_parent is None or
                  len(_parent_inputs(existing)) > len(_parent_inputs(revision_parent)) or
                  (len(_parent_inputs(existing)) == len(_parent_inputs(revision_parent)) and
                   existing.get("created_utc", "") > revision_parent.get("created_utc", ""))):
                revision_parent = existing
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if bounded_candidate:
        same_data_parent = frozen_parent
    if bounded_incremental:
        revision_parent = frozen_parent
    if phone:
        # A phone has no tracked camera pose for the Quest incremental-frame fit.
        # A larger phone snapshot estimates its own frame; old revisions remain.
        revision_parent = None
    parent = same_data_parent or revision_parent
    inherit_checkpoint = bool(same_data_parent and not fresh_baseline)
    extend_colmap = bool(revision_parent and not same_data_parent and not fresh_baseline)
    processing_mode = ("new_capture_incremental_bounded_candidate" if bounded_incremental else
                       "same_data_bounded_candidate" if bounded_candidate else
                       "fresh_baseline" if fresh_baseline else
                       "same_data_checkpoint" if same_data_parent else
                       "new_capture_incremental" if revision_parent else "fresh")
    tools = {name: _tool(name) for name in ("colmap", "ffmpeg", "ns-process-data", "ns-train", "ns-export")}
    job_id = str(uuid.uuid4())
    root = store.root / "projects" / project_id / "reconstructions" / job_id
    root.mkdir(parents=True)
    images_dir = root / "images"
    processed_dir = Path(same_data_parent["processed_dir"]) if inherit_checkpoint else root / "processed"
    train_dir = root / "training"
    export_dir = root / "export"
    if not inherit_checkpoint:
        images_dir.mkdir(parents=True)
    log_path = root / "worker.log"
    input_receipts = []
    for item in snapshot["captures"]:
        source = Path(item["image_path"])
        image_name = f"{item['receipt']['capture_id']}{source.suffix}"
        if not inherit_checkpoint:
            if phone:
                _normalize_phone_jpeg(source, images_dir / image_name)
            else:
                shutil.copyfile(source, images_dir / image_name)
        input_receipts.append({"capture_id": item["receipt"]["capture_id"],
                               "image_sha256": item["receipt"]["image_sha256"],
                               "image_name": image_name})
    job = {"schema_version": 1, "job_id": job_id, "project_id": project_id,
           "session_id": session_id, "created_utc": utc_now(),
           "status": "running", "quality": quality, "input_revision": input_revision,
           "parent_job_id": parent["job_id"] if parent else None,
           "processing_mode": processing_mode,
           "inherited_parent_checkpoint": inherit_checkpoint or extend_colmap,
           "input_captures": input_receipts,
           "camera_pose_source": "colmap_reestimated_from_images",
           "capture_sources": sorted(sources),
           "processing_stage": "registering_images", "analyzed_views": None,
           "source_image_orientations": sorted({item["metadata"]["image_orientation"] for item in snapshot["captures"]}),
           "native_orientation_reviewed": native_orientation_reviewed,
           "training_profile": (candidate_profile if bounded_candidate else None) or {"version": 2,
                                "num_downscales": 1 if quality == "quick" else 0,
                                "resolution_schedule": 250,
                                "full_resolution_from_step": 250 if quality == "quick" else 0,
                                "quick_iterations": 5000,
                                "scale_regularization": True},
           "publication": ("offline_house_source" if house_source else
                           "candidate_review" if candidate else "automatic"),
           "alignment": {"status": "unregistered_colmap_frame", "headset_preview_allowed": False},
           "artifact": None}
    if explicit_source:
        job.update(checkpoint_start_step=source_step, target_training_step=target_training_step,
                   frozen_source_job_id=frozen_parent["job_id"],
                   frozen_source_sha256=frozen_parent["artifact"]["sha256"],
                   frozen_source_checkpoint_sha256=source_checkpoint_sha256)
    manifest_path = root / "job.json"
    manifest_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    env = dict(os.environ)
    # Nerfstudio/Rich prints Unicode progress markers; Windows detached
    # workers otherwise inherit a legacy code page that cannot encode them.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    paths = {str(Path(path).parent) for path in tools.values()}
    env["PATH"] = os.pathsep.join(paths) + os.pathsep + env.get("PATH", "")
    try:
        if extend_colmap:
            revision = _prepare_incremental(revision_parent, snapshot, root, tools, log_path, env)
            job.update(revision)
            processed_dir = Path(revision["processed_dir"])
        elif not inherit_checkpoint:
            _run([tools["ns-process-data"], "images", "--data", str(images_dir),
                  "--output-dir", str(processed_dir), "--matching-method", "exhaustive"], log_path, env,
                 timeout_seconds=900 if fresh_baseline else None)
        transforms = json.loads((processed_dir / "transforms.json").read_text(encoding="utf-8"))
        registered_views = len(transforms.get("frames", []))
        job['analyzed_views'] = _analyzed_views(processed_dir)
        job['processing_stage'] = 'training_splat'
        job["registered_views"] = registered_views
        manifest_path.write_text(json.dumps(job, indent=2), encoding='utf-8')
        if registered_views < 3:
            raise ReconstructionError(f"Only {registered_views} views registered together; capture more overlapping translated views")
        train_args = [tools["ns-train"], "splatfacto", "--data", str(processed_dir),
                      "--output-dir", str(train_dir), "--viewer.websocket-host", "127.0.0.1",
                      "--viewer.quit-on-train-completion", "True",
                      "--pipeline.model.num-downscales", str(candidate_profile['num_downscales']) if bounded_candidate else
                          ("1" if quality == "quick" else "0"),
                      "--pipeline.model.resolution-schedule", "250",
                      "--pipeline.model.use-scale-regularization", "True"]
        if extend_colmap:
            step = _checkpoint_step(Path(revision_parent["checkpoint_dir"]))
            target = target_training_step if bounded_incremental else step + (5000 if quality == "quick" else 15000)
            train_args += ["--load-dir", revision_parent["checkpoint_dir"],
                           "--max-num-iterations", str(target - step - 1), "--steps-per-save", "250",
                           "nerfstudio-data", "--orientation-method", "none",
                           "--center-method", "none", "--auto-scale-poses", "False",
                           "--load-3D-points", "False"]
            job["checkpoint_start_step"] = step
            job["target_training_step"] = target
            if bounded_incremental:
                job["frozen_source_job_id"] = frozen_parent["job_id"]
                job["frozen_source_sha256"] = frozen_parent["artifact"]["sha256"]
                job["frozen_source_checkpoint_sha256"] = source_checkpoint_sha256
            job["new_area_seeding"] = "splatfacto_densification_only; no new sparse points injected into checkpoint"
        elif bounded_candidate:
            continuation_iterations = target_training_step - source_step - 1
            train_args += ["--load-dir", same_data_parent["checkpoint_dir"],
                           "--max-num-iterations", str(continuation_iterations),
                           "--steps-per-save", "500", "--pipeline.model.stop-split-at",
                           str(candidate_profile["stop_split_at"])]
            job["checkpoint_start_step"] = source_step
            job["target_training_step"] = target_training_step
            job["frozen_source_job_id"] = frozen_parent["job_id"]
            job["frozen_source_sha256"] = frozen_parent["artifact"]["sha256"]
            job["frozen_source_checkpoint_sha256"] = source_checkpoint_sha256
        elif quality == "quick":
            # The old 2k-step run never crossed Splatfacto's default 3k
            # resolution schedule and stopped at 1/4 width and height.
            # Reserve a full-resolution fitting period after densification.
            train_args += ["--max-num-iterations", "5000", "--steps-per-save", "500",
                           "--pipeline.model.stop-split-at", "3500"]
        elif inherit_checkpoint:
            train_args += ["--load-dir", same_data_parent["checkpoint_dir"]]
        if explicit_source:
            # Persist frozen provenance before a long training call so progress
            # readers can identify the exact source and target while it runs.
            running_manifest = manifest_path.with_suffix(".tmp")
            running_manifest.write_text(json.dumps(job, indent=2), encoding="utf-8")
            os.replace(running_manifest, manifest_path)
        resumed_training = extend_colmap or inherit_checkpoint
        if bounded_candidate and frozen_parent.get('accepted_review'):
            # The selected incremental result already uses the frozen training
            # frame. Do not normalize it again or prompt to modify its dataset.
            parser = json.loads(Path(frozen_parent['training_config']).with_name('dataparser_transforms.json').read_text())
            if (not np.allclose(np.asarray(parser['transform']), np.eye(4)[:3], rtol=0, atol=1e-7) or
                    parser['scale'] != 1.):
                raise ReconstructionError('Accepted continuation needs its recorded normalized camera frame')
            train_args += ['nerfstudio-data', '--orientation-method', 'none',
                           '--center-method', 'none', '--auto-scale-poses', 'False',
                           '--load-3D-points', 'False']
        _run(_resumed_train_command(train_args), log_path, env,
             timeout_seconds=900 if fresh_baseline else None)
        training_log = log_path.read_text(encoding="utf-8", errors="replace")
        compatibility = re.findall(
            r"^Verified gsplat DefaultStrategy opacity reset compatibility: "
            r"(patched_faulty|already_corrected)\r?$", training_log, re.M)
        if len(compatibility) != 1:
            raise ReconstructionError("Trainer did not verify its exact gsplat opacity reset implementation")
        job["resume_runner_version"] = RESUME_RUNNER_VERSION
        job["opacity_reset_compatibility"] = compatibility[0]
        if resumed_training:
            if "Verified Splatfacto optimizer bindings for six restored Gaussian parameter groups" not in \
                    training_log:
                raise ReconstructionError("Resumed training did not verify restored Gaussian optimizer bindings")
            job["resume_refinement_warmup"] = _verified_resume_warmup(training_log,
                _checkpoint_step(Path(parent["checkpoint_dir"])) + 1, registered_views,
                len(transforms["train_filenames"]) if "train_filenames" in transforms else None)
            job["optimizer_binding_verified"] = True
        else:
            job["optimizer_binding_verified"] = False
        if explicit_source and Store._file_sha256(source_checkpoint) != source_checkpoint_sha256:
            raise ReconstructionError("Frozen source checkpoint changed during candidate continuation")
        if explicit_source and Store._file_sha256(Path(frozen_parent["artifact"]["path"])) != \
                frozen_parent["artifact"]["sha256"]:
            raise ReconstructionError("Frozen source Gaussian changed during candidate continuation")
        configs = list(train_dir.rglob("config.yml"))
        if len(configs) != 1:
            raise ReconstructionError(f"Expected one Nerfstudio training config; found {len(configs)}")
        if bounded_candidate:
            _verify_bounded_candidate_config(configs[0], continuation_iterations,
                num_downscales=candidate_profile['num_downscales'], stop_split_at=candidate_profile['stop_split_at'])
        checkpoint_dir = configs[0].parent / "nerfstudio_models"
        if not list(checkpoint_dir.glob("*.ckpt")):
            raise ReconstructionError("Training completed without a reusable checkpoint")
        completed_step = _checkpoint_step(checkpoint_dir)
        if explicit_source and completed_step != target_training_step - 1:
            raise ReconstructionError("Bounded candidate did not stop at its exact requested training step")
        if extend_colmap and not bounded_incremental and completed_step != target - 1:
            raise ReconstructionError("Incremental training did not stop at its requested step")
        job["completed_training_step"] = completed_step
        job["export_checkpoint_sha256"] = Store._file_sha256(
            next(path for path in checkpoint_dir.glob("step-*.ckpt")
                 if re.fullmatch(rf"step-0*{completed_step}\.ckpt", path.name)))
        job["processed_dir"] = str(processed_dir)
        job["checkpoint_dir"] = str(checkpoint_dir)
        job["training_config"] = str(configs[0])
        job['processing_stage'] = 'exporting_splat'
        manifest_path.write_text(json.dumps(job, indent=2), encoding='utf-8')
        _run([tools["ns-export"], "gaussian-splat", "--load-config", str(configs[0]), "--output-dir", str(export_dir)],
             log_path, env, timeout_seconds=180 if fresh_baseline else None)
        candidates = list(export_dir.rglob("*.ply"))
        if len(candidates) != 1:
            raise ReconstructionError(f"Expected one exported Gaussian PLY; found {len(candidates)}")
        vertices, digest, size = validate_gaussian_ply(candidates[0])
        if explicit_source and digest == frozen_parent["artifact"]["sha256"]:
            raise ReconstructionError("Bounded continuation exported the unchanged source Gaussian")
        artifact_path = root / "splat.ply"
        shutil.copyfile(candidates[0], artifact_path)
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != digest:
            raise ReconstructionError("Published artifact checksum differs from export")
        job["artifact"] = {"path": str(artifact_path), "sha256": digest, "size_bytes": size,
                           "vertex_count": vertices, "content_type": "application/octet-stream"}
        if vertices > 240_000 or size > 64 * 1024 * 1024:
            job["quest_preview"] = make_quest_preview(artifact_path, root / "quest-preview.ply")
        # A quality trial must not replace the last viewable model before its
        # fixed-camera comparisons have been inspected.
        job["status"] = "candidate_for_review" if candidate else "ready_for_blender"
        job['processing_stage'] = job['status']
    except Exception as error:
        job["status"] = "failed"
        job['processing_stage'] = 'failed'
        job["error"] = str(error)
        raise
    finally:
        job["updated_utc"] = utc_now()
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        os.replace(temporary, manifest_path)
    return job


def house_input_revision(captures: list[dict]) -> str:
    """Identity of the photos and independent spatial observations, not a label."""
    records = []
    for item in captures:
        receipt = item.get("receipt", item)
        records.append(f"{receipt['capture_id']}:{receipt['image_sha256']}:"
                       f"{receipt.get('spatial_evidence_sha256', '')}")
    return hashlib.sha256("\n".join(sorted(records)).encode("ascii")).hexdigest()


def house_capture_snapshot(store: Store, session_id: str,
                           capture_ids: list[str] | None = None) -> dict:
    """Freeze selected, verified originals in one unchanged anchored session."""
    snapshot = store.export_session(str(uuid.UUID(session_id)))
    with store.connection() as db:
        has_choices = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                 "AND name='capture_selections'").fetchone()
        excluded = {row[0] for row in db.execute(
            "SELECT capture_id FROM capture_selections WHERE included=0")} if has_choices else set()
    allowed = [item for item in snapshot["captures"]
               if item["receipt"]["capture_id"] not in excluded]
    if capture_ids is not None:
        wanted = {str(uuid.UUID(value)) for value in capture_ids}
        if len(wanted) != len(capture_ids):
            raise ReconstructionError("Duplicate capture in house snapshot")
        allowed = [item for item in allowed if item["receipt"]["capture_id"] in wanted]
        if len(allowed) != len(wanted):
            raise ReconstructionError("A frozen house photo was excluded or is no longer available")
    if len(allowed) < 6:
        raise ReconstructionError("A house attempt needs at least six selected overlapping photos")
    session = snapshot["session"]
    for item in allowed:
        metadata = item["metadata"]
        if (metadata.get("capture_source") != "quest_camera" or
                metadata.get("project_id") != session["project_id"] or
                metadata.get("session_id") != session["id"] or
                metadata.get("tracking_origin_id") != session["tracking_origin_id"]):
            raise ReconstructionError("House photos must share one verified Quest session and origin")
    return {**snapshot, "captures": allowed}


def read_house_build(store: Store, project_id: str, build_id: str) -> tuple[dict, Path]:
    """Verify an aggregate delivery without treating it as measured architecture."""
    from .mesh import read_mesh
    from .preview import read_job
    project_id, build_id = (str(uuid.UUID(value)) for value in (project_id, build_id))
    store.project(project_id)
    folder = store.root / "projects" / project_id / "house-builds" / build_id
    path = folder / "house.json"
    if any(parent.is_symlink() for parent in (folder, folder.parent, folder.parent.parent)) or path.is_symlink():
        raise ReconstructionError("House delivery path is a link")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("job_id") != build_id or
            record.get("project_id") != project_id or record.get("quality") != "house" or
            record.get("status") != "ready_for_blender"):
        raise ReconstructionError("House delivery is incomplete or belongs to another project")
    job, _ = read_job(store, project_id, record["source_job_id"])
    if (job["session_id"] != record["session_id"] or
            job["artifact"]["sha256"] != record["source_splat_sha256"]):
        raise ReconstructionError("House source splat changed")
    ids = record["input_capture_ids"]
    if (not isinstance(ids, list) or any(not isinstance(value, str) for value in ids) or
            len(ids) < 6 or len(set(ids)) != len(ids) or
            record["input_count"] != len(ids)):
        raise ReconstructionError("House capture inventory is invalid")
    inputs = job["input_captures"]
    if {item["capture_id"] for item in inputs} != set(ids) or len(inputs) != len(ids):
        raise ReconstructionError("House source splat has a different photo inventory")
    snapshot = store.export_session(record["session_id"])
    captures = [item for item in snapshot["captures"] if item["receipt"]["capture_id"] in ids]
    if (len(captures) != len(ids) or snapshot["session"]["project_id"] != project_id or
            snapshot["session"]["tracking_origin_id"] != record["tracking_origin_id"] or
            house_input_revision(captures) != record["input_revision"]):
        raise ReconstructionError("House input photographs or spatial evidence changed")
    hashes = {item["receipt"]["capture_id"]: item["receipt"]["image_sha256"] for item in captures}
    if any(hashes[item["capture_id"]] != item["image_sha256"] for item in inputs):
        raise ReconstructionError("House source splat used different photographs")
    raw, _ = read_mesh(store, project_id, record["source_job_id"], record["raw_revision_id"])
    depth, _ = read_mesh(store, project_id, record["source_job_id"], record["depth_revision_id"])
    mesh, artifact = read_mesh(store, project_id, record["source_job_id"], record["mesh_revision_id"])
    if mesh["artifact"]["sha256"] != record["artifact"]["sha256"]:
        raise ReconstructionError("House mesh changed")
    support = depth.get("spatial_support")
    evidence = support.get("evidence", {}) if isinstance(support, dict) else {}
    raw_path = (store.root / "projects" / project_id / "reconstructions" /
                record["source_job_id"] / "mesh-revisions" / record["raw_revision_id"] / "mesh.ply")
    if (raw.get("coordinate_frame") != "colmap_sparse" or raw.get("spatial_support") or
            depth.get("parent_mesh_revision_id") != raw["revision_id"] or
            depth.get("method") != "photo_mesh_with_independent_spatial_support" or
            mesh.get("parent_mesh_revision_id") != depth["revision_id"] or
            mesh.get("method") != "colmap_photo_mvs_cleaned_atlas" or
            not support or support != mesh.get("spatial_support") or
            support != record.get("spatial_support") or
            evidence.get("project_id") != project_id or evidence.get("session_id") != record["session_id"] or
            evidence.get("tracking_origin_id") != record["tracking_origin_id"] or
            raw_path.is_symlink() or Store._file_sha256(raw_path) != support.get("source_surface_sha256") or
            mesh.get("cleanup", {}).get("taubin_steps") != 0 or
            mesh.get("cleanup", {}).get("min_component_faces") != 1 or
            not mesh.get("cleanup", {}).get("geometry_preservation")):
        raise ReconstructionError("House mesh lacks the verified raw-to-depth-to-textured revision chain")
    return record, artifact


def build_house_session(store: Store, session_id: str,
                        native_orientation_reviewed: bool = False,
                        capture_ids: list[str] | None = None,
                        expected_input_revision: str | None = None) -> dict:
    """One command: final splat, photo surface, independent-depth check, texture.

    Interior and exterior must be captured into this same anchored session.
    Missing observations are never filled into a supposedly complete house.
    """
    from .mesh import (build_mesh, depth_assisted_mesh_revision, finish_mesh,
                       verified_spatial_observations)
    snapshot = house_capture_snapshot(store, session_id, capture_ids)
    orientations = {item["metadata"]["image_orientation"] for item in snapshot["captures"]}
    if orientations != {"upright"} and not (
            orientations == {"unity_texture_native"} and native_orientation_reviewed):
        raise ReconstructionError("Inspect the Quest camera photo orientation before a house build")
    ids = [item["receipt"]["capture_id"] for item in snapshot["captures"]]
    revision = house_input_revision(snapshot["captures"])
    if expected_input_revision is not None and revision != expected_input_revision:
        raise ReconstructionError("Frozen house inputs differ from the queued snapshot")
    observations = verified_spatial_observations(store, session_id, ids)
    session = snapshot["session"]
    parent = store.root / "projects" / session["project_id"] / "house-builds"
    parent.mkdir(parents=True, exist_ok=True)
    # Separate operation lock prevents duplicate orchestration. Each stage still
    # uses the original data-root GPU lock; no second GPU authority is introduced.
    with _worker_lock(parent):
        for previous in parent.glob("*/house.json"):
            try:
                data = json.loads(previous.read_text(encoding="utf-8"))
                if (isinstance(data, dict) and data.get("session_id") == session_id and data.get("input_revision") == revision
                        and data.get("status") == "ready_for_blender"):
                    return read_house_build(store, session["project_id"], previous.parent.name)[0]
            except (OSError, ValueError, KeyError, TypeError, StoreError, ReconstructionError):
                # Preserve damaged historical receipts for diagnosis without
                # allowing them to block a fresh attempt or authorize reuse.
                continue
        build_id = str(uuid.uuid4())
        folder = parent / build_id
        folder.mkdir()
        manifest = folder / "house.json"
        record = {"schema_version": 1, "job_id": build_id, "quality": "house",
                  "project_id": session["project_id"], "session_id": session_id,
                  "tracking_origin_id": session["tracking_origin_id"],
                  "input_revision": revision, "created_utc": utc_now(),
                  "input_count": len(ids), "input_capture_ids": ids,
                  "status": "running", "processing_stage": "final_splat",
                  "spatial_observations": observations["report"],
                  "accuracy_status": "unverified", "physical_acceptance": "pending",
                  "coverage": "Observed areas only; no inferred complete house",
                  "artifact": None}
        def save():
            record["updated_utc"] = utc_now()
            temporary = manifest.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest)
        save()
        try:
            job = build_session(store, session_id,
                                native_orientation_reviewed=native_orientation_reviewed,
                                quality="final", capture_ids=ids, house_source=True)
            record.update(source_job_id=job["job_id"], source_splat_sha256=job["artifact"]["sha256"],
                          processing_stage="photo_surface")
            save()
            raw = build_mesh(store, session["project_id"], job["job_id"])
            record.update(raw_revision_id=raw["revision_id"], processing_stage="depth_consistency")
            save()
            assisted = depth_assisted_mesh_revision(store, session["project_id"], job["job_id"],
                                                    raw["revision_id"])
            record.update(depth_revision_id=assisted["revision_id"], processing_stage="photo_textures")
            save()
            finished = finish_mesh(store, session["project_id"], job["job_id"], assisted["revision_id"],
                                   min_component_faces=1, smooth_steps=0)
            record.update(mesh_revision_id=finished["revision_id"], artifact=finished["artifact"],
                          spatial_support=assisted.get("spatial_support"),
                          status="ready_for_blender", processing_stage="ready_for_blender")
            save()
            read_house_build(store, session["project_id"], build_id)
        except Exception as error:
            record.update(status="failed", processing_stage="failed", error=str(error))
            save()
            raise
        return record
