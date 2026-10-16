"""Authenticated immutable Gaussian previews and measured session registration.

The PLY is preserved. The optional similarity transforms UnitySplats' RUB->RUF
import coordinates to the original camera session in meters. Poor fits remain
inspectable as a floating model, never as a claimed physical overlay.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .store import Store, StoreError
from .online import verified_registration, OnlineError
from .reconstruct import ReconstructionError, validate_gaussian_ply


def _within(path: Path, root: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path is outside its project")
    return path


def read_job(store: Store, project_id: str, job_id: str) -> tuple[dict, Path]:
    project_id, job_id = str(uuid.UUID(project_id)), str(uuid.UUID(job_id))
    store.project(project_id)
    root = store.root / "projects" / project_id / "reconstructions" / job_id
    try:
        job = json.loads((root / "job.json").read_text(encoding="utf-8"))
        if job["project_id"] != project_id or job["job_id"] != job_id or job["status"] != "ready_for_blender":
            raise ValueError("Job is not ready")
        path = _within(Path(job["artifact"]["path"]), root)
        if path != (root / "splat.ply").resolve():
            raise ValueError("Unexpected Gaussian artifact path")
        stat = path.stat()
        if stat.st_size != job["artifact"]["size_bytes"] or _digest(str(path), stat.st_mtime_ns, stat.st_size) != job["artifact"]["sha256"]:
            raise ValueError("Gaussian artifact checksum differs")
        return job, path
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise StoreError(404, "Verified Gaussian artifact is unavailable") from error


def read_quest_preview(store: Store, project_id: str, job_id: str) -> tuple[dict, Path]:
    job, _ = read_job(store, project_id, job_id)
    try:
        artifact = job["quest_preview"]
        if artifact["source_sha256"] != job["artifact"]["sha256"]:
            raise ValueError("Quest copy belongs to another source")
        root = store.root / "projects" / project_id / "reconstructions" / job_id
        path = _within(Path(artifact["path"]), root)
        if path != (root / "quest-preview.ply").resolve():
            raise ValueError("Unexpected Quest preview path")
        stat = path.stat()
        if (stat.st_size != artifact["size_bytes"] or
                _digest(str(path), stat.st_mtime_ns, stat.st_size) != artifact["sha256"]):
            raise ValueError("Quest preview checksum differs")
        return job, path
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise StoreError(404, "Verified Quest preview is unavailable") from error


@lru_cache(maxsize=24)
def _digest(path: str, mtime: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_session_similarity(source_poses: np.ndarray, target_poses: np.ndarray) -> dict:
    """Input source poses are in the final training frame, OpenGL camera basis.

    UnitySplats imports RUB by flipping global Z. Flip camera-local Z too for
    proper Unity camera rotations. Fit camera centers, then independently check
    their optical orientations so planar/degenerate fits cannot silently flip.
    """
    source = np.asarray(source_poses, dtype=float)
    target = np.asarray(target_poses, dtype=float)
    if source.shape != target.shape or source.ndim != 3 or source.shape[1:] != (4, 4) or len(source) < 3:
        raise ValueError("At least three matching camera poses are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Camera poses contain non-finite numbers")
    flip = np.diag([1., 1., -1.])
    x = source[:, :3, 3] @ flip
    y = target[:, :3, 3]
    xc, yc = x - x.mean(0), y - y.mean(0)
    baseline = float(np.max(np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)))
    variance = float(np.sum(xc * xc))
    if baseline < 0.10 or variance < 1e-8:
        raise ValueError("Move between photos; camera translation is insufficient")
    u, singular, vt = np.linalg.svd(yc.T @ xc)
    signs = np.eye(3)
    signs[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ signs @ vt
    scale = float(np.sum(singular * np.diag(signs)) / variance)
    if not 0.001 < scale < 1000:
        raise ValueError("Invalid alignment scale")
    translation = y.mean(0) - scale * rotation @ x.mean(0)
    residuals = np.linalg.norm(scale * (x @ rotation.T) + translation - y, axis=1)
    angles = []
    for s, t in zip(source, target):
        mapped = rotation @ flip @ s[:3, :3] @ flip
        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(mapped.T @ t[:3, :3]) - 1) / 2, -1, 1)))))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    allowed = rmse <= 0.12 and float(residuals.max()) <= 0.25 and max(angles) <= 20.
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {"status": "pose_fit" if allowed else "pose_fit_rejected", "headset_preview_allowed": allowed,
            "unity_from_splat": matrix.flatten().tolist() if allowed else None,
            "matched_views": len(source), "position_rmse_m": rmse,
            "max_position_error_m": float(residuals.max()), "max_rotation_error_deg": max(angles),
            "baseline_m": baseline, "scale": scale, "physical_acceptance": "pending"}


def _pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def _homogeneous(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0, 0, 0, 1]))
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid dataparser transform")
    return matrix


def register_job(store: Store, job: dict, *, fixed_unity_from_splat=None) -> dict:
    """Compute without mutating the original capture or reconstruction manifest."""
    if job.get('capture_sources') == ['android_camera2']:
        return {'status': 'phone_pose_unavailable', 'headset_preview_allowed': False,
                'unity_from_splat': None, 'reason': 'Phone photos have no tracked room pose or metric scale.'}
    root = store.root / "projects" / job["project_id"]
    try:
        processed = _within(Path(job["processed_dir"]), root)
        metadata = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
        config = _within(Path(job["training_config"]), root)
        parser = json.loads((config.parent / "dataparser_transforms.json").read_text(encoding="utf-8"))
        saved_from_original = _homogeneous(metadata.get("applied_transform", np.eye(4)))
        training_from_saved = _homogeneous(parser["transform"]) @ np.linalg.inv(saved_from_original)
        scale = float(parser["scale"]) / float(metadata.get("applied_scale", 1.0))
        # The reconstruction is a frozen input set. Later captures may be
        # incomplete or unrelated and must not poison its identity proof.
        snapshot = store.export_session(
            job["session_id"], capture_ids=[c["capture_id"] for c in job["input_captures"]])
        captures = snapshot["captures"]
        # A later duplicate need not be readable to make a frozen input's
        # identity ambiguous. Compare its database receipt without exporting
        # or validating every subsequent image in the session.
        session_digest_counts = {}
        for receipt in store.captures(job["session_id"]):
            digest = receipt["image_sha256"]
            session_digest_counts[digest] = session_digest_counts.get(digest, 0) + 1
        captures_by_id = {c["receipt"]["capture_id"]: c for c in captures}
        frame_manifest = job.get("frame_manifest") or {
            c["capture_id"]: f"frame_{i:05d}{Path(c['image_name']).suffix}"
            for i, c in enumerate(sorted(job["input_captures"], key=lambda c: c["image_name"]), 1)}
        by_name = {Path(name).name: captures_by_id[cid] for cid, name in frame_manifest.items() if cid in captures_by_id}
        # Keep the preview's existing pose lookup and fit unchanged. Geometry
        # needs an additional, unambiguous identity proof for each ray witness.
        pixel_captures, capture_counts, inputs_by_id, frames_by_name = {}, {}, {}, {}
        poses, original_pixels = {}, {}
        for capture in captures:
            capture_id = capture["receipt"]["capture_id"]
            pixels = _pixel_hash(Path(capture["image_path"]))
            poses[pixels] = np.asarray(capture["metadata"]["camera_to_session"]).reshape(4, 4)
            original_pixels[capture_id] = pixels
            pixel_captures.setdefault(pixels, set()).add(capture_id)
            capture_counts[capture_id] = capture_counts.get(capture_id, 0) + 1
        for item in job["input_captures"]:
            inputs_by_id.setdefault(item["capture_id"], []).append(item)
        for capture_id, name in frame_manifest.items():
            frames_by_name.setdefault(Path(name).name, []).append(capture_id)
        frame_counts = {}
        for frame in metadata["frames"]:
            name = Path(frame["file_path"]).name
            frame_counts[name] = frame_counts.get(name, 0) + 1
        source, target, proved = [], [], []
        for frame in metadata["frames"]:
            image = _within(processed / frame["file_path"], processed)
            pixels = _pixel_hash(image)
            pose = poses.get(pixels)
            if pose is None and image.name in by_name:
                original = by_name[image.name]
                # ns-process-data renames sorted sources and reencodes JPEGs.
                # Require the known name mapping AND matching image content.
                with Image.open(image) as a, Image.open(original["image_path"]) as b:
                    same_size = a.size == b.size
                    av = np.asarray(a.convert("RGB").resize((64, 48)), dtype=float)
                    bv = np.asarray(b.convert("RGB").resize((64, 48)), dtype=float)
                if same_size and float(np.sqrt(np.mean((av - bv) ** 2))) < 12:
                    pose = np.asarray(original["metadata"]["camera_to_session"]).reshape(4, 4)
            if pose is None:
                continue
            matrix = training_from_saved @ _homogeneous(frame["transform_matrix"])
            matrix[:3, 3] *= scale
            source.append(matrix)
            target.append(pose)
            identities = frames_by_name.get(image.name, [])
            if len(identities) != 1 or frame_counts[image.name] != 1:
                continue
            capture_id = identities[0]
            original = captures_by_id.get(capture_id)
            inputs = inputs_by_id.get(capture_id, [])
            if original is None or len(inputs) != 1 or capture_counts.get(capture_id) != 1:
                continue
            receipt, original_metadata = original["receipt"], original["metadata"]
            original_pose = np.asarray(original_metadata["camera_to_session"]).reshape(4, 4)
            if (not isinstance(receipt.get("image_sha256"), str) or
                    inputs[0].get("image_sha256") != receipt.get("image_sha256") or
                    original_metadata.get("image_sha256") != receipt.get("image_sha256") or
                    original_metadata.get("capture_id") != capture_id or
                    original_metadata.get("project_id") != job["project_id"] or
                    original_metadata.get("session_id") != job["session_id"] or
                    original_metadata.get("tracking_origin_id") is None or
                    original_metadata.get("tracking_origin_id") != snapshot.get("session", {}).get("tracking_origin_id") or
                    original_metadata.get("tracking_valid") is not True or
                    original_metadata.get("capture_source") != "quest_camera" or
                    len(pixel_captures[original_pixels[capture_id]]) != 1 or
                    session_digest_counts.get(receipt["image_sha256"]) != 1 or
                    not np.array_equal(pose, original_pose)):
                continue
            content_matches = pixels == original_pixels[capture_id]
            if not content_matches:
                with Image.open(image) as a, Image.open(original["image_path"]) as b:
                    same_size = a.size == b.size
                    av = np.asarray(a.convert("RGB").resize((64, 48)), dtype=float)
                    bv = np.asarray(b.convert("RGB").resize((64, 48)), dtype=float)
                content_matches = same_size and float(np.sqrt(np.mean((av - bv) ** 2))) < 12
            if content_matches:
                proved.append((capture_id, matrix, original_pose, image.name, receipt["image_sha256"]))
        result = fit_session_similarity(np.asarray(source), np.asarray(target))
        if fixed_unity_from_splat is not None:
            # A validated frozen continuation inherits the accepted room frame.
            # Recalculate each residual in that frame; never recenter it by a new fit.
            fixed = np.asarray(fixed_unity_from_splat, dtype=float).reshape(4, 4)
            fixed_scale = float(np.cbrt(np.linalg.det(fixed[:3, :3])))
            if (not np.isfinite(fixed).all() or fixed_scale <= 0 or
                    not np.allclose(fixed[3], [0, 0, 0, 1]) or
                    not np.allclose((fixed[:3, :3]/fixed_scale).T @
                                    (fixed[:3, :3]/fixed_scale), np.eye(3), atol=1e-5)):
                raise ValueError('Frozen continuation has an invalid room transform')
            result['unity_from_splat'] = fixed.flatten().tolist()
            result['scale'] = fixed_scale
        proof_counts = {}
        for capture_id, *_ in proved:
            proof_counts[capture_id] = proof_counts.get(capture_id, 0) + 1
        proved = [item for item in proved if proof_counts[item[0]] == 1]
        result["registered_capture_ids"] = sorted(item[0] for item in proved)
        result["capture_pose_residuals"] = {}
        if result["unity_from_splat"] is not None:
            transform = np.asarray(result["unity_from_splat"]).reshape(4, 4)
            rotation = transform[:3, :3] / result["scale"]
            flip = np.diag([1., 1., -1.])
            for capture_id, matrix, pose, frame_name, digest in proved:
                center = transform[:3, :3] @ flip @ matrix[:3, 3] + transform[:3, 3]
                mapped_rotation = rotation @ flip @ matrix[:3, :3] @ flip
                angle = float(np.degrees(np.arccos(np.clip(
                    (np.trace(mapped_rotation.T @ pose[:3, :3]) - 1) / 2, -1, 1))))
                result["capture_pose_residuals"][capture_id] = {
                    "frame_name": frame_name, "image_sha256": digest,
                    "position_error_m": float(np.linalg.norm(center - pose[:3, 3])),
                    "rotation_error_deg": angle}
        return result
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        return {"status": "unregistered", "headset_preview_allowed": False,
                "reason": str(error), "unity_from_splat": None}


@lru_cache(maxsize=32)
def _registration(root: str, job_id: str, manifest_mtime: int, encoded_job: str) -> dict:
    return register_job(Store(Path(root)), json.loads(encoded_job))


def read_online_preview(store: Store, project_id: str, session_id: str,
                        revision: str | None = None) -> tuple[dict, Path, dict]:
    """Verify the full optimizer snapshot and a bounded, immutable display copy.

    A requested revision may have been superseded while the headset downloaded
    its manifest. Its own saved receipt still binds those bytes to this session.
    """
    project_id, session_id = str(uuid.UUID(project_id)), str(uuid.UUID(session_id))
    session = store.session(session_id)
    if session["project_id"] != project_id:
        raise StoreError(404, "Online model belongs to another project")
    folder = store.root / "projects" / project_id / "online" / session_id
    path = folder / "current.ply"
    try:
        status_path = folder / "status.json"
        if folder.is_symlink() or path.is_symlink() or status_path.is_symlink():
            raise ValueError("Online model path is a link")
        _within(folder, store.root)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        artifact = status["artifact"]
        registration = status["registration"]
        if (status["project_id"] != project_id or status["session_id"] != session_id or
                status["seed_sha256"] != registration["seed_sha256"]):
            raise ValueError("Online model receipt does not match its registration")
        job_id = str(uuid.UUID(registration["evidence_job_id"]))
        job, _seed = read_job(store, project_id, job_id)
        if job["session_id"] != session_id:
            raise ValueError("Online model seed is unavailable")
        if status["seed_sha256"] != job["artifact"]["sha256"]:
            job, _seed = read_quest_preview(store, project_id, job_id)
            if status["seed_sha256"] != job["quest_preview"]["sha256"]:
                raise ValueError("Online model seed differs from its verified source job")
        model_from_session, _, _ = verified_registration(
            registration, session_id, status["seed_sha256"])
        count, digest, size = validate_gaussian_ply(path)
        if (count != artifact["vertex_count"] or size != artifact["size_bytes"] or
                digest != artifact["sha256"]):
            raise ValueError("Full online snapshot differs from its receipt")
        display = status.get("display_artifact")
        if display:
            if display.get("source_sha256") != digest:
                raise ValueError("Display preview belongs to another optimizer snapshot")
            display_sha = display["sha256"]
            if (not isinstance(display_sha, str) or len(display_sha) != 64 or
                    any(char not in "0123456789abcdef" for char in display_sha)):
                raise ValueError("Invalid online display digest")
            path = folder / "display-revisions" / (display_sha + ".ply")
            if not path.is_file():
                raise ValueError("Immutable online display revision is unavailable")
            if path.is_symlink():
                raise ValueError("Online display preview path is a link")
            _within(path, folder)
            count, digest, size = validate_gaussian_ply(path)
            if (count != display["vertex_count"] or size != display["size_bytes"] or
                    digest != display["sha256"] or count > 250000 or size > 64 * 1024 * 1024):
                raise ValueError("Online display preview differs from its receipt")
        elif count > 250000 or size > 64 * 1024 * 1024:
            raise ValueError("Online snapshot requires a bounded display copy")
        if revision is not None:
            if (not isinstance(revision, str) or len(revision) != 64 or
                    any(char not in "0123456789abcdef" for char in revision)):
                raise ValueError("Invalid online revision")
            if revision != digest:
                # Older downloads are served only from a separately saved
                # receipt; current status must never attest to older bytes.
                saved = folder / "display-revisions" / (revision + ".json")
                if saved.is_symlink() or not saved.is_file() or saved.stat().st_size > 32768:
                    raise ValueError("Requested online revision has no saved receipt")
                _within(saved, folder)
                record = json.loads(saved.read_text(encoding="utf-8"))
                source_sha = record.get("source_sha256")
                if (record.get("project_id") != project_id or record.get("session_id") != session_id or
                        record.get("sha256") != revision or not isinstance(source_sha, str) or
                        len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha)):
                    raise ValueError("Requested online revision belongs to another model")
                path = folder / "display-revisions" / (revision + ".ply")
                if path.is_symlink():
                    raise ValueError("Requested online revision is a link")
                _within(path, folder)
                count, digest, size = validate_gaussian_ply(path)
                if (count != record.get("vertex_count") or size != record.get("size_bytes") or
                        digest != revision or count > 250000 or size > 64 * 1024 * 1024):
                    raise ValueError("Requested online revision differs from its receipt")
        flip = np.diag([1., 1., -1., 1.])
        unity_from_splat = flip @ np.linalg.inv(model_from_session) @ flip
        alignment = {"status": "verified_online", "headset_preview_allowed": True,
                     "unity_from_splat": unity_from_splat.flatten().tolist(),
                     "physical_acceptance": registration.get("physical_acceptance", "pending")}
        return status, path, alignment
    except (OSError, ValueError, KeyError, TypeError, OnlineError, ReconstructionError,
            np.linalg.LinAlgError) as error:
        raise StoreError(404, "Verified online snapshot is unavailable") from error


def read_depth_preview(store: Store, project_id: str, session_id: str, revision=None):
    """Render observed depth samples as small markers using the existing XR renderer.

    This artifact has Gaussian fields solely for display. It is not a trained
    splat, inferred surface, or mesh, and never becomes the project's splat job.
    """
    from .mesh_spatial import read_session_spatial_map, _atomic_spatial_write
    project_id, session_id = str(uuid.UUID(project_id)), str(uuid.UUID(session_id))
    session = store.session(session_id)
    if session['project_id'] != project_id:
        raise StoreError(404, 'Depth preview belongs to another project')
    root = store.root/'projects'/project_id/'sessions'/session_id/'spatial-map'/'previews'
    try:
        if root.is_symlink() or not root.resolve().is_relative_to(store.root.resolve()):
            raise ValueError('Depth preview path escapes its owner')
        if revision is not None:
            if len(revision) != 64 or any(c not in '0123456789abcdef' for c in revision):
                raise ValueError('Invalid depth revision')
            saved = root/(revision+'.json')
            path = root/(revision+'.ply')
            if saved.is_file():
                if saved.is_symlink() or path.is_symlink() or saved.stat().st_size > 32768:
                    raise ValueError('Invalid saved depth preview')
                manifest = json.loads(saved.read_bytes())
                if (manifest['project_id'] != project_id or manifest['session_id'] != session_id or
                    manifest['tracking_origin_id'] != session['tracking_origin_id'] or
                    manifest['state_revision'] != revision or manifest['job_id'] != revision or
                    manifest['representation'] != 'depth' or
                    manifest['source_kind'] != 'provisional_anchor_frame_depth'):
                    raise ValueError('Saved depth preview identity mismatch')
                count, sha, size = validate_gaussian_ply(path)
                if (count != manifest['artifact']['vertex_count'] or
                    sha != manifest['artifact']['sha256'] or size != manifest['artifact']['size_bytes']):
                    raise ValueError('Saved depth preview bytes changed')
                return manifest, path
        state = read_session_spatial_map(store, session_id)
        digest = state['state_sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid depth revision')
        if revision is not None and revision != digest:
            raise ValueError('Depth map advanced; request its current manifest')
        palette = {'supported': (60, 200, 100), 'conflicting': (235, 70, 70),
            'tentative_conflict': (235, 140, 65), 'tentative_multi_view': (235, 210, 65),
            'tentative_single_view': (130, 145, 165), 'tentative_repeated_origin': (160, 130, 165)}
        surfaces = [surface for key in sorted(state['cells']) for surface in state['cells'][key]['surfaces']]
        if not surfaces:
            raise ValueError('No observed surface samples yet')
        # Fixed, deterministic display budget; source observations remain intact.
        if len(surfaces) > 160000:
            indices = np.linspace(0, len(surfaces)-1, 160000, dtype=np.int64)
            surfaces = [surfaces[i] for i in indices]
        names = ('x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')
        vertices = np.zeros(len(surfaces), dtype=[(name, '<f4') for name in names])
        points = np.asarray([surface['point'] for surface in surfaces], dtype=float)
        if points.shape != (len(surfaces), 3) or not np.isfinite(points).all():
            raise ValueError('Invalid observed depth positions')
        # Runtime loader's existing RUB import flips Z to Unity forward once.
        for index, name in enumerate(('x', 'y', 'z')):
            vertices[name] = points[:, index] * (-1 if index == 2 else 1)
        colors = np.asarray([palette[surface['state']] for surface in surfaces])/255.
        for index in range(3):
            vertices['f_dc_'+str(index)] = (colors[:, index]-.5)/.28209479177387814
        vertices['opacity'] = np.log(.8/.2)
        for index in range(3):
            vertices['scale_'+str(index)] = np.log(.008)
        vertices['rot_0'] = 1.
        header = ('ply\nformat binary_little_endian 1.0\n'
            'comment observed depth markers; not trained Gaussian geometry\n'
            f'element vertex {len(surfaces)}\n'+''.join(f'property float {name}\n' for name in names)+
            'end_header\n').encode('ascii')
        payload = header+vertices.tobytes()
        root.mkdir(parents=True, exist_ok=True)
        path = root/(digest+'.ply')
        if path.is_symlink():
            raise ValueError('Depth preview cannot be a symlink')
        if not path.exists():
            try:
                _atomic_spatial_write(path, payload, replace=False)
            except ReconstructionError:
                if not path.is_file():
                    raise
        if path.read_bytes() != payload:
            raise ValueError('Depth preview differs from the observed state')
        count, sha, size = validate_gaussian_ply(path)
        summary = state['summary']
        manifest = {'status': 'ready', 'representation': 'depth', 'project_id': project_id,
            'session_id': session_id, 'tracking_origin_id': state['tracking_origin_id'],
            'job_id': digest, 'state_revision': digest, 'source_kind': state['kind'],
            'revision_number': state.get('revision_number', 0),
            'accepted_views': summary['capture_count'], 'total_views': store.session(session_id)['capture_count'],
            'supported_samples': summary['surface_states'].get('supported', 0),
            'display_mode': 'world_overlay', 'accuracy_status': 'provisional_observed_samples',
            'alignment': {'status': 'anchored_depth_samples', 'headset_preview_allowed': True,
                          'unity_from_splat': np.eye(4).ravel().tolist()},
            'artifact': {'sha256': sha, 'size_bytes': size, 'vertex_count': count,
                'download_path': f'/api/v1/projects/{project_id}/spatial/{session_id}/{digest}'}}
        # Pin each requested publication until its download finishes, even if
        # another capture advances current.json in the meantime.
        saved = root/(digest+'.json')
        if saved.is_symlink():
            raise ValueError('Depth manifest cannot be a symlink')
        if not saved.exists():
            try:
                _atomic_spatial_write(saved, json.dumps(manifest, sort_keys=True).encode(), replace=False)
            except ReconstructionError:
                if not saved.is_file():
                    raise
        return manifest, path
    except (OSError, ValueError, KeyError, TypeError, ReconstructionError, StoreError) as error:
        raise StoreError(404, 'Verified depth preview is unavailable') from error


def preview_manifest(store: Store, session_id: str, representation: str) -> dict:
    session = store.session(session_id)
    pending = {"status": "not_ready", "representation": representation, "artifact": None,
               "reason": "Capture overlapping views and build a quick preview on the PC",
               "total_views": session["capture_count"]}
    if representation == 'depth':
        try:
            return read_depth_preview(store, session['project_id'], session_id)[0]
        except StoreError:
            return {**pending, 'reason': 'No shared depth map is ready on the PC yet'}
    if representation != "splat":
        return {**pending, "reason": "No mesh has been published for this scan"}
    try:
        status, _path, alignment = read_online_preview(store, session["project_id"], session_id)
        source_artifact = status.get("display_artifact", status["artifact"])
        artifact = {k: source_artifact[k] for k in ("sha256", "size_bytes", "vertex_count")}
        artifact["download_path"] = (
            f"/api/v1/projects/{session['project_id']}/online/{session_id}/splat/"
            f"{artifact['sha256']}" if status.get("display_artifact") else
            f"/api/v1/projects/{session['project_id']}/online/{session_id}/splat")
        # The seed batch may contain photos that never registered. Do not tell
        # the headset all seed inputs were processed into the model.
        baseline_views = status["registration"].get("pose_fit", {}).get("matched_views", 0)
        try:
            evidence, _ = read_job(store, session["project_id"],
                                   status["registration"]["evidence_job_id"])
            if isinstance(evidence.get("registered_views"), int):
                baseline_views = evidence["registered_views"]
        except (StoreError, ValueError):
            pass
        baseline_views = (min(baseline_views, len(status.get("seed_ids", [])))
                          if isinstance(baseline_views, int) and baseline_views >= 0 else 0)
        # accepted_ids includes processed photos that may still be awaiting a
        # published model revision. Prefer the published set in new worker
        # status; legacy snapshots did not record it.
        additional_views = status.get("published_capture_ids")
        if additional_views is None:
            additional_views = status.get("accepted_ids", [])
        # Area refinement can publish an already seeded photo again. Preserve
        # the registered baseline count and add only unique non-seed identities.
        additional_views = set(additional_views) - set(status.get("seed_ids", []))
        return {"status": "ready", "representation": "splat", "project_id": session["project_id"],
                "session_id": session_id, "tracking_origin_id": session["tracking_origin_id"],
                "job_id": status["registration"]["evidence_job_id"], "artifact": artifact,
                "alignment": alignment, "display_mode": "world_overlay",
                "revision_number": status.get("revision_number", 0),
                "accepted_views": baseline_views + len(additional_views),
                "total_views": session["capture_count"]}
    except StoreError:
        pass
    reconstructions = store.root / "projects" / session["project_id"] / "reconstructions"
    candidates = []
    for manifest in reconstructions.glob("*/job.json"):
        try:
            job, path = read_job(store, session["project_id"], manifest.parent.name)
            if job.get("publication") == "offline_house_source":
                continue
            artifact = job["artifact"]
            route = "splat"
            if artifact["size_bytes"] > 64 * 1024 * 1024 or artifact["vertex_count"] > 250000:
                job, path = read_quest_preview(store, session["project_id"], manifest.parent.name)
                artifact = job["quest_preview"]
                route = "quest-splat"
            if artifact["size_bytes"] <= 64 * 1024 * 1024 and artifact["vertex_count"] <= 250000:
                candidates.append((job, manifest, artifact, route))
        except (StoreError, ValueError):
            continue
    if not candidates:
        return pending
    job, manifest, artifact_source, route = max(candidates, key=lambda j: (j[0]["session_id"] == session_id, j[0].get("updated_utc", j[0]["created_utc"])))
    origin = store.session(job["session_id"])["tracking_origin_id"]
    alignment = _registration(str(store.root), job["job_id"], manifest.stat().st_mtime_ns, json.dumps(job, sort_keys=True))
    if origin != session["tracking_origin_id"] or job["session_id"] != session_id:
        alignment = {**alignment, "headset_preview_allowed": False, "status": "different_capture_session"}
    artifact = {k: artifact_source[k] for k in ("sha256", "size_bytes", "vertex_count")}
    artifact["download_path"] = f"/api/v1/projects/{job['project_id']}/reconstructions/{job['job_id']}/{route}"
    return {"status": "ready", "representation": "splat", "project_id": job["project_id"],
            "session_id": job["session_id"], "tracking_origin_id": origin, "job_id": job["job_id"],
            "artifact": artifact, "alignment": alignment,
            "accepted_views": job.get("registered_views") or len(job.get("input_captures", [])),
            "total_views": store.session(job["session_id"])["capture_count"],
            "display_mode": "world_overlay" if alignment["headset_preview_allowed"] else "inspection_model"}
