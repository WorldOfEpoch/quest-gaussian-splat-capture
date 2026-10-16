"""One persistent Gaussian model, seeded once from measured photo geometry.

Automatic launches are explicit, session-scoped and obey the scheduler's saved
controls. Original captures and reconstruction seeds are never overwritten.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
import os
import shutil
import sqlite3
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import utc_now
from .reconstruct import ReconstructionError, validate_gaussian_ply
from .store import Store, StoreError


class OnlineError(RuntimeError):
    pass


def ensure_refinement_allowed(status_path: Path) -> None:
    """Refuse automatic resume of a model paused after visible degradation."""
    if not status_path.is_file():
        return
    try:
        saved_status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OnlineError(f"Cannot verify online model status: {error}") from error
    if saved_status.get("status") in {"paused_quality_review", "paused_selection_review"}:
        raise OnlineError("Online refinement is paused for quality review; photo receiving remains available")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".online-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path, *, timeout: float = 30.) -> None:
    """Publish over a Windows reader that may briefly hold the old file open.

    The prior verified snapshot remains available throughout a retry.  Exhausting
    the bound raises without advancing the durable ingress cursor, so restarting
    the worker replays the uncommitted capture from its last checkpoint.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(.25, max(0., deadline - time.monotonic())))


def _selected_ids(store: Store, session_id: str) -> set[str]:
    """Read gallery choices; absent selection rows mean included."""
    with store.connection() as db:
        table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='capture_selections'").fetchone()
        if table:
            rows = db.execute("""SELECT c.capture_id FROM captures c
                LEFT JOIN capture_selections s ON s.capture_id=c.capture_id
                WHERE c.session_id=? AND COALESCE(s.included,1)=1""", (session_id,))
        else:
            rows = db.execute("SELECT capture_id FROM captures WHERE session_id=?", (session_id,))
        return {row[0] for row in rows}


def _scheduled_control(store: Store, session_id: str):
    """The queue owns automatic policy; capture storage is a separate database."""
    from .jobs import _db
    with _db(store.root) as db:
        row = db.execute("SELECT enabled,control_revision FROM splat_updates WHERE session_id=?",
                         (session_id,)).fetchone()
    if row is None or not row["enabled"]:
        raise OnlineError("Scheduled Gaussian refinement is disabled")
    return row["control_revision"]


def validated_camera(metadata: dict, model_from_session: np.ndarray,
                     calibration_revision: str | None,
                     image_orientation: str = "upright") -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV model-space c2w and intrinsic matrix for a verified image."""
    if (metadata.get("capture_source") != "quest_camera" or not metadata.get("tracking_valid")
            or metadata.get("image_orientation") != image_orientation
            or metadata.get("distortion") not in (None, {})
            or metadata.get("calibration_revision") != calibration_revision):
        raise OnlineError("Capture pose, upright image, distortion or calibration is unverified")
    if metadata.get("coordinate_convention") != "unity_lh_xright_yup_zforward":
        raise OnlineError("Unsupported Quest camera coordinate convention")
    raw = np.asarray(metadata["camera_to_session"], dtype=np.float64).reshape(4, 4)
    if not np.isfinite(raw).all() or not np.allclose(raw[3], [0, 0, 0, 1], atol=1e-4):
        raise OnlineError("Invalid camera pose")
    if not np.allclose(raw[:3, :3].T @ raw[:3, :3], np.eye(3), atol=1e-3):
        raise OnlineError("Camera rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(raw[:3, :3])), 1, abs_tol=1e-3):
        raise OnlineError("Camera rotation is reflected")
    # Unity world is left-handed Y-up; OpenCV camera is right-handed Y-down.
    world_flip = np.diag([1., 1., -1., 1.])
    camera_flip = np.diag([1., -1., 1., 1.])
    pose = model_from_session @ world_flip @ raw @ camera_flip
    if not np.isfinite(pose).all() or np.linalg.det(pose[:3, :3]) <= 0:
        raise OnlineError("Registered camera transform is invalid")
    scale = float(np.cbrt(np.linalg.det(pose[:3, :3])))
    pose[:3, :3] /= scale
    intr = metadata["intrinsics"]
    k = np.array([[intr["fx"], 0., intr["cx"]], [0., intr["fy"], intr["cy"]],
                  [0., 0., 1.]], dtype=np.float64)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise OnlineError("Invalid camera intrinsics")
    return pose, k


def verified_registration(registration: dict, session_id: str,
                          seed_sha256: str) -> tuple[np.ndarray, str | None, str]:
    """Reject unaligned/relabelled models before any GPU work can start."""
    measured = registration.get("verification_method") == "source_job_pose_fit"
    if (registration.get("status") != "verified" or
            registration.get("session_id") != session_id or
            registration.get("seed_sha256") != seed_sha256 or
            registration.get("basis") != "opencv_rh_from_unity" or
            not registration.get("evidence_job_id") or
            (registration.get("physical_acceptance") != "accepted" and not (
                measured and registration.get("physical_acceptance") == "pending"))):
        raise OnlineError("A verified, seed-specific Quest-to-model registration is required")
    fit = registration.get("pose_fit", {})
    limits = {"position_rmse_m": .04 if measured else .12,
              "max_position_error_m": .10 if measured else .25,
              "max_rotation_error_deg": 5. if measured else 20.}
    try:
        if (int(fit["matched_views"]) < (6 if measured else 3) or
                float(fit["baseline_m"]) < (.30 if measured else .10) or
                any(not math.isfinite(float(fit[key])) or float(fit[key]) > limit or float(fit[key]) < 0
                    for key, limit in limits.items())):
            raise OnlineError("Quest/model pose fit has not passed alignment thresholds")
    except (KeyError, TypeError, ValueError) as error:
        raise OnlineError("Quest/model pose fit evidence is incomplete") from error
    matrix = np.asarray(registration.get("model_from_session"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise OnlineError("Invalid registration matrix")
    rotation = matrix[:3, :3]
    scale = float(np.cbrt(np.linalg.det(rotation)))
    if not 0.01 <= scale <= 100 or not np.allclose(rotation.T @ rotation, np.eye(3) * scale * scale, atol=1e-3):
        raise OnlineError("Registration must be a positive similarity transform")
    orientation = registration.get("image_orientation", "upright")
    if orientation not in {"upright", "unity_texture_native"} or (
            orientation == "unity_texture_native" and
            registration.get("native_orientation_reviewed") is not True):
        raise OnlineError("Quest image orientation has not been reviewed")
    return matrix, registration.get("calibration_revision"), orientation


def _json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_identity(status: dict) -> str:
    # An error report may change after an interrupted write; the model may not.
    return _json_digest({key:value for key,value in status.items() if key not in
        {'status', 'last_error', 'error', 'updated_utc', 'worker_state', 'idle_yield'}})


def _publication_ledger(status: dict, registration: dict, store=None) -> dict:
    seed_ids, accepted = set(status.get('seed_ids', [])), set(status.get('accepted_ids', []))
    if ('seed_capture_ids' in registration and seed_ids != set(registration['seed_capture_ids'])):
        raise OnlineError('Online seed photo identities differ from verified registration')
    seed_proofs = registration.get('source_proof', {}).get('captures', {})
    proofs = {**seed_proofs,
              **status.get('photo_registration', {})}
    reviewed_missing = seed_ids-proofs.keys()
    if (reviewed_missing and store is not None and registration.get('physical_acceptance') == 'accepted' and
            registration.get('verification_method') != 'source_job_pose_fit'):
        for item in store.export_session(status['session_id'], capture_ids=sorted(reviewed_missing))['captures']:
            proofs[item['receipt']['capture_id']] = {'image_sha256':item['receipt']['image_sha256'],
                'metadata_sha256':_json_digest(item['metadata'])}
    ledger = {}
    for cid in sorted(seed_ids | accepted):
        proof = proofs.get(cid, {})
        if (not all(isinstance(proof.get(key), str) and len(proof[key]) == 64 and
                    all(char in '0123456789abcdef' for char in proof[key])
                    for key in ('image_sha256', 'metadata_sha256')) or
                (cid in accepted-seed_ids and proof.get('seed_sha256') != status.get('seed_sha256')) or
                (cid in seed_ids and cid in seed_proofs and any(
                    proof.get(key) != seed_proofs[cid].get(key)
                    for key in ('image_sha256', 'metadata_sha256')))):
            raise OnlineError('Online publication lacks exact original-photo provenance')
        ledger[cid] = {key:proof[key] for key in ('image_sha256', 'metadata_sha256')}
    return ledger


def update_photo_counts(status: dict, seed_ids, published, rejected, pending) -> None:
    """Keep checkpoint lineage while reporting each original photo only once."""
    processed, published, rejected = (list(ids) for ids in
        (status['accepted_ids'], published, rejected))
    pending = deepcopy(pending)
    pending_ids = pending.get('capture_ids', [])
    if (any(len(set(ids)) != len(ids) for ids in (processed, published, rejected, pending_ids)) or
            not set(published+rejected+pending_ids) <= set(processed) or
            set(published) & (set(rejected) | set(pending_ids))):
        raise OnlineError('Trainer publication lineage differs from processed photos')
    seed_ids = set(seed_ids)
    status.update(seed_view_count=len(seed_ids), processed_count=len(seed_ids | set(processed)),
        published_capture_ids=published, visible_count=len(seed_ids | set(published)),
        rejected_capture_ids=rejected, rejected_count=len(rejected),
        pending_count=len(pending_ids), pending_refinement=pending)


def _recover_online_publication(store, session_id, seed, registration, *, before_commit=None) -> bool:
    """Finish only a hashed publisher transaction, before loading its optimizer."""
    session = store.session(session_id)
    folder = store.root/'projects'/session['project_id']/'online'/session_id
    journal_path, status_path = folder/'pending-publication.json', folder/'status.json'
    if not journal_path.exists():
        return False
    if journal_path.is_symlink() or status_path.is_symlink():
        raise OnlineError('Online publication recovery cannot follow linked receipts')
    journal = json.loads(journal_path.read_text(encoding='utf-8'))
    next_status = journal['status']
    seed_sha = validate_gaussian_ply(seed)[1]
    if (journal.get('schema_version') != 1 or next_status.get('session_id') != session_id or
            next_status.get('project_id') != session['project_id'] or
            next_status.get('seed_sha256') != seed_sha or next_status.get('registration') != registration or
            journal.get('tracking_origin_id') != session['tracking_origin_id'] or
            journal.get('input_ledger') != _publication_ledger(next_status, registration, store)):
        raise OnlineError('Interrupted publication does not match the verified seed, frame and photo receipts')
    verified_registration(registration, session_id, seed_sha)
    ledger = journal['input_ledger']
    if not set(ledger) <= _selected_ids(store, session_id):
        raise OnlineError('Interrupted publication contains now-excluded photos')
    if ledger:
        snapshot = store.export_session(session_id, capture_ids=sorted(ledger))
        for item in snapshot['captures']:
            cid = item['receipt']['capture_id']
            if (item['receipt']['image_sha256'] != ledger[cid]['image_sha256'] or
                    _json_digest(item['metadata']) != ledger[cid]['metadata_sha256'] or
                    item['metadata'].get('tracking_origin_id') != session['tracking_origin_id']):
                raise OnlineError('Interrupted publication original photo or frame changed')
    current_bytes = status_path.read_bytes()
    current = json.loads(current_bytes)
    if _publication_identity(current) not in {journal['base_identity'], _publication_identity(next_status)}:
        raise OnlineError('A newer online generation supersedes the interrupted publication')
    ensure_refinement_allowed(status_path)
    sources = []
    for name, temporary in (('current.ply', '.current.tmp.ply'), ('current.ckpt', '.current.ckpt.tmp')):
        target = folder/name
        proof = journal['files'][name]
        source = next((path for path in (target, folder/temporary) if path.is_file() and
            not path.is_symlink() and path.stat().st_size == proof['size_bytes'] and
            _file_sha256(path) == proof['sha256']), None)
        if source is None or target.is_symlink():
            raise OnlineError('Interrupted publication checkpoint/model bytes cannot be verified')
        sources.append((source, target))
    count, sha, size = validate_gaussian_ply(sources[0][0])
    if next_status.get('artifact') != {'vertex_count':count, 'sha256':sha, 'size_bytes':size}:
        raise OnlineError('Interrupted publication model receipt differs from its exact bytes')
    display = next_status['display_artifact']
    if (not isinstance(display.get('sha256'), str) or len(display['sha256']) != 64 or
            any(char not in '0123456789abcdef' for char in display['sha256'])):
        raise OnlineError('Interrupted publication has an invalid immutable display identity')
    revision = folder/'display-revisions'/(display['sha256']+'.ply')
    receipt_path = revision.with_suffix('.json')
    expected = {'project_id':session['project_id'], 'session_id':session_id,
        **{key:display[key] for key in ('sha256','size_bytes','vertex_count','source_sha256')}}
    if revision.is_symlink() or receipt_path.is_symlink():
        raise OnlineError('Interrupted publication cannot follow a linked display receipt')
    display_receipt = json.loads(receipt_path.read_text())
    if (revision.is_symlink() or receipt_path.is_symlink() or
            {key:value for key,value in display_receipt.items() if key != 'revision_number'} != expected or
            type(display_receipt.get('revision_number')) is not int or display_receipt['revision_number'] < 0 or
            display['source_sha256'] != sha or
            validate_gaussian_ply(revision) != (display['vertex_count'], display['sha256'], display['size_bytes'])):
        raise OnlineError('Interrupted publication lacks its verified immutable display receipt')
    def check_recovery():
        ensure_refinement_allowed(status_path)
        if before_commit:
            before_commit()
        if status_path.read_bytes() != current_bytes:
            raise OnlineError('Online manifest changed during interrupted-publication recovery')
        if not set(ledger) <= _selected_ids(store, session_id):
            raise OnlineError('Photo selection changed during interrupted-publication recovery')
    for source, target in sources:
        check_recovery()
        if source != target:
            _replace_with_retry(source, target)
    display_path = folder/'display.ply'
    if display_path.is_symlink():
        raise OnlineError('Online publication recovery cannot replace a linked display')
    if not display_path.is_file() or _file_sha256(display_path) != display['sha256']:
        temporary = folder/'.display.recover.ply'
        if temporary.is_symlink():
            raise OnlineError('Online publication recovery cannot follow a linked temporary display')
        shutil.copyfile(revision, temporary)
        check_recovery()
        _replace_with_retry(temporary, display_path)
    check_recovery()
    next_status.pop('error', None); next_status.pop('last_error', None)
    next_status.pop('idle_yield', None); next_status.pop('worker_state', None)
    next_status.update(updated_utc=utc_now(), status='training' if next_status.get('pending_count') else 'waiting_for_new_views')
    _atomic_json(status_path, next_status)
    journal_path.unlink()
    return True


def registration_from_seed_job(store: Store, session_id: str, seed_job_id: str) -> tuple[Path, dict]:
    """Read and measure an authentic same-session seed; no human acceptance is invented."""
    from .preview import read_job, register_job
    session = store.session(session_id)
    job_id = str(uuid.UUID(seed_job_id))
    job, seed = read_job(store, session["project_id"], job_id)
    if (job["session_id"] != session_id or job.get("capture_sources") != ["quest_camera"] or
            job.get("publication") == "offline_house_source"):
        raise OnlineError("Online seed must be a ready Quest reconstruction from this session")
    inputs = job["input_captures"]
    ids = [item["capture_id"] for item in inputs]
    if len(set(ids)) != len(ids):
        raise OnlineError("Seed has duplicate input capture identities")
    snapshot = store.export_session(session_id, capture_ids=ids)
    captures = {item["receipt"]["capture_id"]: item for item in snapshot["captures"]}
    for item in inputs:
        original = captures[item["capture_id"]]
        metadata, receipt = original["metadata"], original["receipt"]
        if (receipt["image_sha256"] != item["image_sha256"] or
                metadata.get("image_sha256") != receipt["image_sha256"] or
                metadata.get("capture_id") != item["capture_id"] or
                metadata.get("project_id") != session["project_id"] or
                metadata.get("session_id") != session_id or
                metadata.get("tracking_origin_id") != session["tracking_origin_id"]):
            raise OnlineError("Seed original identity, tracking frame or photo hash differs")
    frozen_registration = None
    if job.get('accepted_review') and job.get('frozen_source_job_id'):
        from .preview import _homogeneous
        _, frozen_registration = registration_from_seed_job(store, session_id, job['frozen_source_job_id'])
        parent_job, _ = read_job(store, session['project_id'], job['frozen_source_job_id'])
        def cameras(source_job):
            data = json.loads((Path(source_job['processed_dir'])/'transforms.json').read_text())
            parser = json.loads(Path(source_job['training_config']).with_name('dataparser_transforms.json').read_text())
            frame = _homogeneous(parser['transform']) @ np.linalg.inv(_homogeneous(data.get('applied_transform', np.eye(4))))
            scale = float(parser['scale']) / float(data.get('applied_scale', 1.))
            result = {}
            for view in data['frames']:
                pose = frame @ _homogeneous(view['transform_matrix'])
                pose[:3, 3] *= scale
                result[Path(view['file_path']).name] = pose
            return result
        old, new = cameras(parent_job), cameras(job)
        if (job['frozen_source_sha256'] != parent_job['artifact']['sha256'] or
                not old.keys() <= new.keys() or
                any(not np.allclose(pose, new[name], rtol=0, atol=1e-5) for name, pose in old.items())):
            raise OnlineError('Accepted continuation changed its protected source camera frame')
        flip = np.diag([1., 1., -1., 1.])
        fixed = flip @ np.linalg.inv(frozen_registration['model_from_session']) @ flip
        fit = register_job(store, job, fixed_unity_from_splat=fixed)
        # The accepted visual result is preserved in full. Only independently
        # verified cameras prime the live worker; outlier poses remain untrusted.
        proved_all = fit.get('registered_capture_ids', [])
        valid = [cid for cid in proved_all if
                 fit['capture_pose_residuals'][cid]['position_error_m'] <= .10 and
                 fit['capture_pose_residuals'][cid]['rotation_error_deg'] <= 5.]
        fit['registered_capture_ids'] = valid
        fit['matched_views'] = len(valid)
        fit['position_rmse_m'] = float(np.sqrt(np.mean([
            fit['capture_pose_residuals'][cid]['position_error_m']**2 for cid in valid]))) if valid else float('inf')
        fit['max_position_error_m'] = max((fit['capture_pose_residuals'][cid]['position_error_m'] for cid in valid), default=float('inf'))
        fit['max_rotation_error_deg'] = max((fit['capture_pose_residuals'][cid]['rotation_error_deg'] for cid in valid), default=float('inf'))
    else:
        fit = register_job(store, job)
    proved = fit.get("registered_capture_ids", [])
    residuals = fit.get("capture_pose_residuals", {})
    if (not fit.get("headset_preview_allowed") or len(proved) < 6 or
            len(set(proved)) != len(proved) or set(proved)-captures.keys() or
            any(cid not in residuals or residuals[cid].get("image_sha256") !=
                captures[cid]["receipt"]["image_sha256"] or
                not np.isfinite([residuals[cid]["position_error_m"],
                                 residuals[cid]["rotation_error_deg"]]).all() or
                residuals[cid]["position_error_m"] > .10 or
                residuals[cid]["rotation_error_deg"] > 5. for cid in proved)):
        raise OnlineError("Seed lacks strict independent per-photo registration proof")
    orientation = {captures[cid]["metadata"].get("image_orientation") for cid in proved}
    revisions = {captures[cid]["metadata"].get("calibration_revision") for cid in proved}
    if len(orientation) != 1 or len(revisions) != 1:
        raise OnlineError("Seed mixes camera orientation or calibration revisions")
    flip = np.diag([1., 1., -1., 1.])
    matrix = np.asarray(fit["unity_from_splat"], dtype=float).reshape(4, 4)
    project_root = store.root/"projects"/session["project_id"]
    files = {"transforms": Path(job["processed_dir"])/"transforms.json",
             "training_config": Path(job["training_config"]),
             "dataparser": Path(job["training_config"]).parent/"dataparser_transforms.json"}
    source_files = {}
    for name, path in files.items():
        if path.is_symlink() or not path.resolve().is_relative_to(project_root.resolve()):
            raise OnlineError("Seed frame evidence escapes its project")
        source_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    anchor = captures[proved[0]]["metadata"]
    signature = {key: anchor.get(key) for key in ("camera_id", "image_width", "image_height", "intrinsics")}
    if any(any(captures[cid]["metadata"].get(key) != signature[key]
               for key in ("camera_id", "image_width", "image_height")) for cid in proved):
        raise OnlineError("Seed mixes physical cameras or image dimensions")
    registration = {"status": "verified", "verification_method": "source_job_pose_fit",
        "physical_acceptance": "pending", "session_id": session_id,
        "project_id": session["project_id"], "tracking_origin_id": session["tracking_origin_id"],
        "seed_sha256": validate_gaussian_ply(seed)[1], "evidence_job_id": job_id,
        "seed_capture_ids": sorted(proved), "basis": "opencv_rh_from_unity",
        "model_from_session": (frozen_registration['model_from_session'] if frozen_registration else
                               (flip @ np.linalg.inv(matrix) @ flip).tolist()),
        "image_orientation": next(iter(orientation)), "calibration_revision": next(iter(revisions)),
        "camera_signature": signature,
        "native_orientation_reviewed": job.get("native_orientation_reviewed") is True,
        "pose_fit": {key: fit[key] for key in ("matched_views", "baseline_m", "position_rmse_m",
            "max_position_error_m", "max_rotation_error_deg")},
        "capture_pose_residuals": {cid: residuals[cid] for cid in sorted(proved)},
        "source_proof": {"job_sha256": _json_digest(job), "frame_files": source_files,
            "captures": {cid: {"image_sha256": captures[cid]["receipt"]["image_sha256"],
                               "metadata_sha256": _json_digest(captures[cid]["metadata"])}
                         for cid in sorted(proved)}}}
    if frozen_registration:
        registration['frozen_frame_source_job_id'] = job['frozen_source_job_id']
        registration['model_trained_view_count'] = job['registered_views']
        registration['accepted_model_review'] = job['accepted_review']['model_sha256']
    verified_registration(registration, session_id, registration["seed_sha256"])
    return seed, registration


def adopt_reviewed_seed(store: Store, session_id: str, seed_job_id: str, *,
                       expected_current_sha256: str) -> dict:
    """Publish an explicitly accepted continuation through the existing journal.

    Keep the exact accepted room transform, full SH3 PLY and Nerfstudio
    optimizer checkpoint. The small online optimizer starts with every seed
    row protected; it does not impersonate the saved Nerfstudio optimizer.
    """
    from .reconstruct import _worker_lock
    from .online_torch import GsplatTrainer
    from .preview import read_job
    from .jobs import set_splat_updates
    session = store.session(session_id)
    job, seed = read_job(store, session['project_id'], seed_job_id)
    if not job.get('accepted_review'):
        raise OnlineError('Only an explicitly reviewed and preserved baseline can replace the current seed')
    seed, registration = registration_from_seed_job(store, session_id, seed_job_id)
    root = store.root/'projects'/session['project_id']/'online'/session_id
    with _worker_lock(store.root):
        status_path = root/'status.json'
        current = json.loads(status_path.read_text())
        if current.get('seed_job_id') == seed_job_id:
            if current['artifact']['sha256'] != job['artifact']['sha256']:
                raise OnlineError('Accepted baseline has already advanced; do not overwrite its newer state')
            return current
        if (current['artifact']['sha256'] != expected_current_sha256 or
                validate_gaussian_ply(root/'current.ply')[1] != expected_current_sha256 or
                registration['model_from_session'] != current['registration']['model_from_session']):
            raise OnlineError('Current scan changed or the accepted continuation would move its room frame')
        if current.get('model_metrics', {}).get('locked_regions'):
            raise OnlineError('Existing area locks require region-specific review before replacing a full baseline')
        if not set(registration['seed_capture_ids']) <= _selected_ids(store, session_id):
            raise OnlineError('Accepted baseline includes now-excluded verified reference photos')
        before = status_path.read_bytes()
        recovery = root/'accepted-baselines'/seed_job_id
        recovery.mkdir(parents=True, exist_ok=True)
        for name in ('current.ply', 'current.ckpt', 'display.ply', 'status.json'):
            original = root/name
            saved = recovery/name
            if original.is_symlink() or saved.is_symlink():
                raise OnlineError('Baseline recovery cannot follow linked files')
            if saved.exists():
                if _file_sha256(saved) != _file_sha256(original):
                    raise OnlineError('Baseline recovery already belongs to another generation')
            else:
                shutil.copy2(original, saved)
        trainer = GsplatTrainer(seed)
        metrics = trainer.save(root/'.current.ckpt.tmp', root/'.accepted-export.tmp.ply')
        shutil.copy2(seed, root/'.current.tmp.ply')  # Preserve exact accepted bytes, including every SH3 band.
        count, digest, size = validate_gaussian_ply(root/'.current.tmp.ply')
        replacement = {'schema_version': 1, 'session_id': session_id,
            'project_id': session['project_id'], 'seed_job_id': seed_job_id,
            'seed_sha256': digest, 'seed_ids': sorted(registration['seed_capture_ids']),
            'registration': registration, 'accepted_ids': [], 'photo_registration': {},
            'quarantined': {}, 'training_steps': 0, 'model_metrics': metrics,
            'revision_number': current.get('revision_number', 0)+1,
            'status': 'waiting_for_new_views', 'updated_utc': utc_now(),
            'artifact': {'vertex_count': count, 'sha256': digest, 'size_bytes': size},
            'accepted_baseline': job['accepted_review'],
            'offline_optimizer_checkpoint': {'directory': job['checkpoint_dir'],
                'step': job['accepted_review']['checkpoint_step'],
                'sha256': job['export_checkpoint_sha256']},
            'model_trained_view_count': job['registered_views'], 'protected_seed_rows': count}
        update_photo_counts(replacement, replacement['seed_ids'], [], [], {})
        def unchanged():
            if status_path.read_bytes() != before or _file_sha256(root/'current.ply') != expected_current_sha256:
                raise OnlineError('Live scan changed during accepted-baseline preparation')
        replacement['display_artifact'] = stage_display_revision(seed, root,
            session['project_id'], session_id, replacement['revision_number'],
            before_commit=unchanged, prepared_display=Path(job['quest_preview']['path']) if job.get('quest_preview') else None)
        unchanged()
        journal = {'schema_version': 1, 'base_identity': _publication_identity(current),
            'tracking_origin_id': session['tracking_origin_id'], 'status': replacement,
            'input_ledger': _publication_ledger(replacement, registration, store),
            'files': {name: {'size_bytes': (root/temp).stat().st_size,
                            'sha256': _file_sha256(root/temp)} for name, temp in
                (('current.ply', '.current.tmp.ply'), ('current.ckpt', '.current.ckpt.tmp'))}}
        # Save the new worker's seed identity while disabled, then commit using
        # the same recoverable transaction used for normal snapshot updates.
        set_splat_updates(store.root, session_id, False, seed_job_id=seed_job_id)
        _atomic_json(root/'pending-publication.json', journal)
        _recover_online_publication(store, session_id, seed, registration)
        return json.loads(status_path.read_text())


def bootstrap_seed_job(store: Store, session_id: str, *, native_orientation_reviewed=False) -> str:
    """Reuse a valid measured seed or build the first bounded frozen batch once."""
    from .reconstruct import build_session
    session = store.session(session_id)
    root = store.root/"projects"/session["project_id"]
    status_path = root/"online"/session_id/"status.json"
    ensure_refinement_allowed(status_path)
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        existing = status.get("seed_job_id") or status.get("registration", {}).get("evidence_job_id")
        if existing:
            registration_from_seed_job(store, session_id, existing)
            return existing
    candidates = []
    for path in (root/"reconstructions").glob("*/job.json"):
        try:
            job = json.loads(path.read_text())
            if job.get("session_id") == session_id and job.get("status") == "ready_for_blender":
                candidates.append((job.get("created_utc", ""), job["job_id"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for _, job_id in sorted(candidates, reverse=True):
        try:
            registration_from_seed_job(store, session_id, job_id)
            return job_id
        except (OnlineError, StoreError, OSError, ValueError, KeyError, TypeError):
            continue
    selected = _selected_ids(store, session_id)
    included = [row["capture_id"] for row in store.captures(session_id) if row["capture_id"] in selected]
    ids = [included[index] for index in np.linspace(0, len(included)-1, min(60, len(included)), dtype=int)]
    if len(ids) < 12:
        raise OnlineError("Initial Gaussian seed needs at least twelve selected overlapping photos")
    job = build_session(store, session_id, quality="quick", fresh_baseline=True,
                        native_orientation_reviewed=native_orientation_reviewed, capture_ids=ids)
    registration_from_seed_job(store, session_id, job["job_id"])
    return job["job_id"]


def seed_camera_views(store: Store, registration: dict) -> list[dict]:
    """Hydrate the exact optimized cameras in the seed's trained model frame.

    Tracking residuals prove these belong to the anchored session, but tracking
    poses are approximate and must not replace measured cameras for triangulation.
    """
    from .preview import read_job, _homogeneous
    job, _ = read_job(store, registration["project_id"], registration["evidence_job_id"])
    processed = Path(job["processed_dir"])
    if _json_digest(job) != registration["source_proof"]["job_sha256"]:
        raise OnlineError("Registered seed source job changed after registration")
    paths = {"transforms": processed/"transforms.json", "training_config": Path(job["training_config"]),
             "dataparser": Path(job["training_config"]).with_name("dataparser_transforms.json")}
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != registration["source_proof"]["frame_files"][name]:
            raise OnlineError("Registered seed camera evidence changed")
    source = json.loads(paths["transforms"].read_text())
    parser = json.loads(paths["dataparser"].read_text())
    transform = _homogeneous(parser["transform"]) @ np.linalg.inv(
        _homogeneous(source.get("applied_transform", np.eye(4))))
    scale = float(parser["scale"])/float(source.get("applied_scale", 1.))
    frames = {Path(frame["file_path"]).name: frame for frame in source["frames"]}
    snapshot = store.export_session(registration["session_id"], capture_ids=registration["seed_capture_ids"])
    views = []
    for item in snapshot["captures"]:
        cid = item["receipt"]["capture_id"]; metadata = item["metadata"]
        proof = registration["source_proof"]["captures"][cid]
        if (proof["image_sha256"] != item["receipt"]["image_sha256"] or
                proof["metadata_sha256"] != _json_digest(metadata)):
            raise OnlineError("Registered seed camera original differs from its receipt")
        frame = frames[registration["capture_pose_residuals"][cid]["frame_name"]]
        pose = transform @ _homogeneous(frame["transform_matrix"])
        pose[:3, 3] *= scale
        pose = pose @ np.diag([1., -1., -1., 1.])  # OpenGL camera -> OpenCV.
        calibration = {**source, **frame}
        intr = metadata["intrinsics"]
        k = np.array([[calibration.get("fl_x", intr["fx"]), 0., calibration.get("cx", intr["cx"])],
                      [0., calibration.get("fl_y", intr["fy"]), calibration.get("cy", intr["cy"])],
                      [0., 0., 1.]])
        from PIL import Image
        with Image.open(item["image_path"]) as image:
            dimensions = image.size
        if dimensions != (metadata["image_width"], metadata["image_height"]):
            raise OnlineError("Seed original dimensions differ from verified camera receipt")
        source_width, source_height = calibration.get("w", dimensions[0]), calibration.get("h", dimensions[1])
        if not np.isfinite([source_width, source_height]).all() or min(source_width, source_height) <= 0:
            raise OnlineError("Invalid optimized seed camera image dimensions")
        processed_image = processed/frame["file_path"]
        if not processed_image.resolve().is_relative_to(processed.resolve()):
            raise OnlineError("Registered seed camera image escapes its source")
        with Image.open(processed_image) as image:
            if image.size != (source_width, source_height):
                raise OnlineError("Optimized camera dimensions differ from its processed image")
        k[0] *= dimensions[0]/source_width; k[1] *= dimensions[1]/source_height
        distortion = [float(calibration.get(key, 0.)) for key in ("k1", "k2", "p1", "p2", "k3")]
        if (not np.isfinite(distortion).all() or (any(distortion) and
                calibration.get("camera_model") != "OPENCV")):
            raise OnlineError("Unsupported optimized seed lens-distortion model")
        if (not np.isfinite(pose).all() or not np.isfinite(k).all() or
                not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4) or min(k[0, 0], k[1, 1]) <= 0):
            raise OnlineError("Invalid optimized seed camera")
        views.append({"capture_id": cid, "image_path": Path(item["image_path"]), "pose": pose,
                      "intrinsics": k, "distortion_coefficients": distortion,
                      "camera_source": "verified_seed_frame"})
    return views


def triangulate_correspondences(points_a: np.ndarray, points_b: np.ndarray,
                                pose_a: np.ndarray, pose_b: np.ndarray,
                                intrinsics_a: np.ndarray, intrinsics_b: np.ndarray,
                                *, max_reprojection_px: float = 2.5,
                                min_parallax_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate matched pixels from two registered OpenCV cameras.

    Return model-space points and their source match indices. Invalid depth,
    weak parallax and inconsistent image projections never enter the model.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) < 2:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    baseline = np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])
    if not np.isfinite(baseline) or baseline < 0.08:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    projection_a = intrinsics_a @ np.linalg.inv(pose_a)[:3, :]
    projection_b = intrinsics_b @ np.linalg.inv(pose_b)[:3, :]
    equations = np.stack((a[:, 0, None] * projection_a[2] - projection_a[0],
                          a[:, 1, None] * projection_a[2] - projection_a[1],
                          b[:, 0, None] * projection_b[2] - projection_b[0],
                          b[:, 1, None] * projection_b[2] - projection_b[1]), axis=1)
    _, _, vt = np.linalg.svd(equations)
    homogeneous = vt[:, -1]
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-8)
    points = homogeneous[:, :3] / np.where(valid, homogeneous[:, 3], 1)[:, None]
    errors = []
    for pose, intrinsics, pixels in ((pose_a, intrinsics_a, a), (pose_b, intrinsics_b, b)):
        camera = (np.linalg.inv(pose) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
        valid &= np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.25) & (camera[:, 2] < 20)
        projected = (intrinsics @ camera.T).T
        uv = projected[:, :2] / np.where(np.abs(projected[:, 2:3]) > 1e-8,
                                        projected[:, 2:3], 1)
        errors.append(np.linalg.norm(uv - pixels, axis=1))
    valid &= (errors[0] <= max_reprojection_px) & (errors[1] <= max_reprojection_px)
    ray_a = points - pose_a[:3, 3]
    ray_b = points - pose_b[:3, 3]
    cosine = np.sum(ray_a * ray_b, axis=1) / np.maximum(
        np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1), 1e-8)
    parallax = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    valid &= np.isfinite(parallax) & (parallax >= min_parallax_deg)
    indices = np.flatnonzero(valid)
    return points[indices], indices


def dense_stereo_correspondences(pixels_a, pixels_b, pose_a, pose_b, intrinsics_a,
                                 intrinsics_b, *, max_matches=20000):
    """Infer a bounded dense patch from two calibrated, undistorted photographs.

    This is image stereo, not measured sensor depth or proof of empty space.
    Forward/backward disparity, texture, valid rectification coverage and the
    same physical triangulation gates as sparse growth must all agree. Callers
    still need independent cameras to review any proposed replacement.
    """
    import cv2
    empty = (np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)))
    a, b = np.asarray(pixels_a), np.asarray(pixels_b)
    if (a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3 or
            not np.isfinite(a).all() or not np.isfinite(b).all() or
            min(a.shape[:2]) < 96 or max(a.shape[:2]) > 1024 or
            not isinstance(max_matches, int) or not 1 <= max_matches <= 20000):
        return empty
    poses = [np.asarray(value, dtype=float) for value in (pose_a, pose_b)]
    ks = [np.asarray(value, dtype=float) for value in (intrinsics_a, intrinsics_b)]
    for pose, k in zip(poses, ks):
        if (pose.shape != (4, 4) or k.shape != (3, 3) or
                not np.isfinite(pose).all() or not np.isfinite(k).all() or
                not np.allclose(pose[3], [0, 0, 0, 1]) or
                not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4) or
                min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1])):
            return empty
    baseline = np.linalg.norm(poses[0][:3, 3] - poses[1][:3, 3])
    if not .08 <= baseline <= 2.:
        return empty
    relative = np.linalg.inv(poses[1]) @ poses[0]
    h, w = a.shape[:2]
    rect = cv2.stereoRectify(ks[0], np.zeros(5), ks[1], np.zeros(5), (w, h),
                           relative[:3, :3].copy(), relative[:3, 3:].copy(),
                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    r1, r2, p1, p2 = rect[:4]
    vertical = abs(p2[1, 3]) > abs(p2[0, 3])
    axis = 1 if vertical else 0
    if not np.isfinite(p1).all() or not np.isfinite(p2).all() or abs(p2[axis, 3]) < 1e-8:
        return empty
    maps, images, masks = [], [], []
    for pixels, k, rotation, projection in zip((a, b), ks, (r1, r2), (p1, p2)):
        mx, my = cv2.initUndistortRectifyMap(k, None, rotation, projection, (w, h), cv2.CV_32FC1)
        gray = cv2.cvtColor(np.uint8(np.clip(pixels, 0, 1) * 255), cv2.COLOR_RGB2GRAY)
        image = cv2.remap(gray, mx, my, cv2.INTER_LINEAR)
        mask = cv2.remap(np.ones((h, w), np.uint8), mx, my, cv2.INTER_NEAREST)
        if vertical:
            mx, my, image, mask = mx.T.copy(), my.T.copy(), image.T.copy(), mask.T.copy()
        maps.append((mx, my)); images.append(image); masks.append(mask)
    span = min(256, ((images[0].shape[1] // 2) // 16) * 16)
    if span < 32:
        return empty
    low = 0 if p2[axis, 3] < 0 else -span
    def disparity(left, right, minimum):
        matcher = cv2.StereoSGBM_create(minDisparity=minimum, numDisparities=span,
            blockSize=5, P1=8 * 25, P2=32 * 25, disp12MaxDiff=1,
            uniquenessRatio=12, speckleWindowSize=80, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return matcher.compute(left, right).astype(np.float32) / 16.
    forward = disparity(images[0], images[1], low)
    reverse_low = -low - span
    backward = disparity(images[1], images[0], reverse_low)
    yy, xx = np.mgrid[:forward.shape[0], :forward.shape[1]]
    other_x = xx.astype(np.float32) - forward
    other_y = yy.astype(np.float32)
    reverse = cv2.remap(backward, other_x, other_y, cv2.INTER_LINEAR, borderValue=reverse_low - 1)
    valid_b = cv2.remap(masks[1], other_x, other_y, cv2.INTER_NEAREST)
    float_a = images[0].astype(np.float32)
    mean = cv2.boxFilter(float_a, -1, (7, 7))
    variance = cv2.boxFilter(float_a * float_a, -1, (7, 7)) - mean * mean
    other = cv2.remap(images[1], other_x, other_y, cv2.INTER_LINEAR)
    photo_error = cv2.boxFilter(np.abs(float_a - other.astype(np.float32)), -1, (5, 5))
    ok = ((forward > low) & (reverse > reverse_low) &
          (np.abs(forward + reverse) <= 1.) & (np.abs(forward) >= 1.) &
          (masks[0] > 0) & (valid_b > 0) & (variance >= 16.) &
          (photo_error <= 20.) & (xx >= 4) & (yy >= 4) &
          (xx < forward.shape[1] - 4) & (yy < forward.shape[0] - 4) &
          (other_x >= 4) & (other_x < forward.shape[1] - 4) &
          ((xx % 2) == 0) & ((yy % 2) == 0))
    y, x = np.where(ok)
    if len(x) < 2:
        return empty
    # Fixed raster-order subsampling bounds SVD memory without cherry-picking
    # against either optimization loss or the held-out review photographs.
    if len(x) > max_matches:
        choose = np.linspace(0, len(x) - 1, max_matches, dtype=int)
        y, x = y[choose], x[choose]
    uv_a = np.column_stack((maps[0][0][y, x], maps[0][1][y, x]))
    bx = other_x[y, x].reshape(-1, 1)
    by = y.astype(np.float32).reshape(-1, 1)
    uv_b = np.column_stack([cv2.remap(m, bx, by, cv2.INTER_LINEAR).ravel() for m in maps[1]])
    points, indices = triangulate_correspondences(uv_a, uv_b, *poses, *ks,
                                                 max_reprojection_px=1., min_parallax_deg=1.5)
    return points, uv_a[indices], uv_b[indices]


class PhotoRegistrationVerifier:
    """Localize new RGB against two independently proved baseline views.

    Baseline rays define 3D correspondences without using the incoming pose.
    PnP then measures that photo's pose independently of its tracking metadata.
    Missing matches are unknown, not evidence of an aligned photo.
    """

    def __init__(self, views: list[dict], model_units_per_meter: float):
        indices = np.linspace(0, len(views)-1, min(64, len(views)), dtype=int) if views else []
        self.views = [views[index] for index in indices]
        self.scale = model_units_per_meter
        self.features = {}
        self.method_revision = "exact_seed_cameras_undistorted_pnp_v3"

    def _features(self, view):
        import cv2
        from PIL import Image
        key = str(view["image_path"])
        if key not in self.features:
            with Image.open(view["image_path"]) as source:
                image = source.convert("RGB"); width, height = image.size
                factor = min(1., 640/max(width, height))
                image = image.resize((max(1, round(width*factor)), max(1, round(height*factor))),
                                     Image.Resampling.LANCZOS)
                gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            k = np.asarray(view["intrinsics"], dtype=float).copy()
            k[0] *= image.width/width; k[1] *= image.height/height
            distortion = np.asarray(view.get("distortion_coefficients", [0.]*5), dtype=float)
            if distortion.shape != (5,) or not np.isfinite(distortion).all():
                raise OnlineError("Invalid measured lens-distortion coefficients")
            if distortion.any():
                gray = cv2.undistort(gray, k, distortion)
            points, descriptors = cv2.ORB_create(nfeatures=3500).detectAndCompute(gray, None)
            self.features[key] = (points, descriptors, k)
        return self.features[key]

    def verify(self, view: dict) -> dict:
        import cv2
        points, descriptors, k = self._features(view)
        if descriptors is None:
            raise OnlineError("Photo has insufficient features for independent localization")
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        def matches(first, second):
            if first is None or second is None or len(second) < 2:
                return []
            return [pair[0] for pair in matcher.knnMatch(first, second, k=2)
                    if len(pair) == 2 and pair[0].distance < .7*pair[1].distance]
        references = []
        for reference in self.views:
            keys, desc, intrinsics = self._features(reference)
            matched = matches(descriptors, desc)
            if len(matched) >= 12:
                references.append((len(matched), reference, keys, desc, intrinsics, matched))
        references.sort(key=lambda item: -item[0])
        world, pixels = [], []
        for first, second in itertools.combinations(references[:4], 2):
            _, a, ka, da, ia, ma = first
            _, b, kb, db, ib, _ = second
            paired = matches(da, db)
            if len(paired) < 12:
                continue
            pose_a, pose_b = np.asarray(a["pose"]).copy(), np.asarray(b["pose"]).copy()
            pose_a[:3, 3] /= self.scale; pose_b[:3, 3] /= self.scale
            cloud, indices = triangulate_correspondences(
                np.asarray([ka[m.queryIdx].pt for m in paired]),
                np.asarray([kb[m.trainIdx].pt for m in paired]), pose_a, pose_b, ia, ib)
            lookup = {paired[index].queryIdx: point for index, point in zip(indices, cloud)}
            for match in ma:
                if match.trainIdx in lookup:
                    world.append(lookup[match.trainIdx]); pixels.append(points[match.queryIdx].pt)
        if len(world) < 12:
            raise OnlineError("Photo lacks independently triangulated baseline correspondences")
        # Repeated pair matches must not inflate the support count.
        unique = {}
        for point, pixel in zip(world, pixels):
            unique.setdefault(tuple(np.round(pixel, 3)), (point, pixel))
        if len(unique) < 12:
            raise OnlineError("Photo has too few distinct localized image features")
        cloud = np.asarray([v[0] for v in unique.values()], dtype=np.float64)
        uv = np.asarray([v[1] for v in unique.values()], dtype=np.float64)
        success, rotation, translation, inliers = cv2.solvePnPRansac(
            cloud, uv, k, None, iterationsCount=100, reprojectionError=2.5,
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None or len(inliers) < 8 or len(inliers) < .5*len(cloud):
            raise OnlineError("Photo localization did not achieve independent geometric support")
        r, _ = cv2.Rodrigues(rotation)
        measured = np.eye(4); measured[:3, :3] = r.T; measured[:3, 3] = -r.T @ translation.ravel()
        expected = np.asarray(view["pose"]).copy(); expected[:3, 3] /= self.scale
        position = float(np.linalg.norm(measured[:3, 3]-expected[:3, 3]))
        angle = float(np.degrees(np.arccos(np.clip(
            (np.trace(measured[:3, :3].T @ expected[:3, :3])-1)/2, -1, 1))))
        if not np.isfinite([position, angle]).all() or position > .10 or angle > 5.:
            raise OnlineError("Photo tracking disagrees with independently localized RGB")
        measured[:3, 3] *= self.scale
        return {"method": "baseline_rgb_triangulation_pnp", "method_revision": self.method_revision,
                "position_error_m": position, "rotation_error_deg": angle,
                "inliers": len(inliers), "correspondences": len(cloud),
                "localized_pose": measured.tolist(), "camera_intrinsics": view["intrinsics"].tolist(),
                "distortion_coefficients": list(view.get("distortion_coefficients", [0.]*5))}

    def append_verified(self, view):
        if view["capture_id"] not in {item["capture_id"] for item in self.views}:
            self.views.append(view)
            # Retain original seed anchors while extending overlap references.
            if len(self.views) > 64:
                self.views.pop(min(32, len(self.views)-1))
        paths = {str(item["image_path"]) for item in self.views}
        self.features = {key: value for key, value in self.features.items() if key in paths}


def _colmap_localization_source(store: Store, registration: dict, views: list[dict]) -> dict:
    """Read a bounded sparse map, checking its cameras and indexed observation identities.

    The frozen trained cameras remain the frame authority. This does not extend
    COLMAP, change a camera, or manufacture a landmark from incoming tracking.
    """
    from .preview import read_job, _homogeneous
    job, _ = read_job(store, registration["project_id"], registration["evidence_job_id"])
    processed = Path(job["processed_dir"])
    if _json_digest(job) != registration["source_proof"]["job_sha256"]:
        raise OnlineError("Sparse landmark source job changed after registration")
    for name, path in {"transforms": processed/"transforms.json", "training_config": Path(job["training_config"]),
                       "dataparser": Path(job["training_config"]).with_name("dataparser_transforms.json")}.items():
        if Store._file_sha256(path) != registration["source_proof"]["frame_files"][name]:
            raise OnlineError("Sparse landmark seed frame proof changed")
    paths = {name: processed/"colmap"/relative for name, relative in {
        "database": "database.db", "images": "sparse/0/images.bin",
        "points": "sparse/0/points3D.bin"}.items()}
    for path in paths.values():
        if (not path.is_file() or path.is_symlink() or
                not path.resolve().is_relative_to(processed.resolve())):
            raise OnlineError("Measured COLMAP seed lacks safe sparse landmark evidence")
    hashes = {name: Store._file_sha256(path) for name, path in paths.items()}
    parser = json.loads(Path(job["training_config"]).with_name("dataparser_transforms.json").read_text())
    source = json.loads((processed/"transforms.json").read_text())
    transform = _homogeneous(parser["transform"])
    scale = float(parser["scale"])/float(source.get("applied_scale", 1.))
    if job.get('normalization'):
        # Incremental processing already wrote normalized training cameras;
        # its identity dataparser is not the raw sparse COLMAP world frame.
        # Reuse the recorded conversion, then still prove every camera below.
        from .reconstruct import _normalization, _colmap_camera_to_saved
        if (not np.allclose(transform, np.eye(4), atol=1e-8) or
                not math.isclose(scale, 1., abs_tol=1e-8)):
            raise OnlineError('Incremental sparse map has an unexpected second normalization')
        normalized, scale, _ = _normalization(job)
        raw_to_saved = _colmap_camera_to_saved(np.array([1.,0.,0.,0.]),np.zeros(3)) @ np.diag([1.,-1.,-1.,1.])
        transform = _homogeneous(normalized) @ raw_to_saved
    if (not np.isfinite(transform).all() or not math.isfinite(scale) or scale <= 0 or
            not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-4)):
        raise OnlineError("Sparse landmark normalization is invalid")
    refs = {registration["capture_pose_residuals"][view["capture_id"]]["frame_name"]: view for view in views}
    if len(refs) != len(views):
        raise OnlineError("Sparse landmark camera names are ambiguous")
    def read(stream, codec):
        size = struct.calcsize("<"+codec)
        data = stream.read(size)
        if len(data) != size:
            raise OnlineError("Sparse landmark evidence is truncated")
        return struct.unpack("<"+codec, data)
    def bounded_count(stream, limit):
        count = read(stream, "Q")[0]
        if count > limit:
            raise OnlineError("Sparse landmark evidence exceeds its budget")
        return count
    points = {}
    with paths["points"].open("rb") as stream:
        for _ in range(bounded_count(stream, 1_000_000)):
            pid, x, y, z, _r, _g, _b, error = read(stream, "QdddBBBd")
            count = bounded_count(stream, 10_000)
            track = [read(stream, "ii") for _ in range(count)]
            if pid in points or not np.isfinite([x, y, z, error]).all() or error < 0:
                raise OnlineError("Sparse landmark identity or coordinates are invalid")
            if len(track) != len(set(track)):
                raise OnlineError("Sparse landmark has duplicate observation identities")
            points[pid] = (np.asarray([x, y, z]), error, set(track))
        if stream.read(1):
            raise OnlineError("Sparse landmark evidence has trailing data")
    descriptor_rows, landmark_ids, observations = [], [], {}
    seen_names, seen_ids = set(), set()
    observation_count = 0
    with sqlite3.connect(paths["database"].as_uri()+"?mode=ro", uri=True) as db, paths["images"].open("rb") as stream:
        for _ in range(bounded_count(stream, 10_000)):
            values = read(stream, "idddddddi")
            image_id, q, translation = values[0], np.asarray(values[1:5]), np.asarray(values[5:8])
            name_bytes = bytearray()
            for _ in range(1024):
                value = stream.read(1)
                if value == b"\0":
                    break
                if not value:
                    raise OnlineError("Sparse landmark image name is truncated")
                name_bytes.extend(value)
            else:
                raise OnlineError("Sparse landmark image name exceeds its budget")
            name = name_bytes.decode("utf-8")
            count = bounded_count(stream, 100_000)
            observation_count += count
            if observation_count > 2_000_000:
                raise OnlineError("Sparse landmark observation budget exceeded")
            size = count*24
            data = stream.read(size)
            if len(data) != size:
                raise OnlineError("Sparse landmark image observations are truncated")
            image_points = np.frombuffer(data, dtype=[("xy", "<f8", (2,)), ("id", "<i8")])
            if name not in refs:
                continue  # Unproved cameras cannot contribute landmark votes.
            if name in seen_names or image_id in seen_ids or image_id <= 0:
                raise OnlineError("Sparse landmark camera identity is duplicated")
            seen_names.add(name); seen_ids.add(image_id)
            if not np.isfinite(q).all() or not np.isclose(q@q, 1., atol=1e-5):
                raise OnlineError("Sparse landmark camera rotation is invalid")
            w, x, y, z = q
            rotation = np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            raw_pose = np.eye(4); raw_pose[:3, :3] = rotation.T
            raw_pose[:3, 3] = -rotation.T@translation
            model_pose = transform@raw_pose; model_pose[:3, 3] *= scale
            if not np.allclose(model_pose, refs[name]["pose"], atol=1e-4):
                raise OnlineError("Sparse landmark cameras disagree with the exact trained seed frame")
            if db.execute("SELECT name FROM images WHERE image_id=?", (image_id,)).fetchone() != (name,):
                raise OnlineError("Sparse landmark database image identity differs")
            keys = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
            desc = db.execute("SELECT rows,cols,data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
            if "type" in {column[1] for column in db.execute("PRAGMA table_info(descriptors)")}:
                if db.execute("SELECT type FROM descriptors WHERE image_id=?", (image_id,)).fetchone() != (0,):
                    raise OnlineError("Sparse landmark descriptor type is unsupported")
            if (not keys or not desc or keys[0] != count or keys[1] not in (2, 4, 6) or
                    len(keys[2]) != count*keys[1]*4 or desc[:2] != (count, 128) or
                    len(desc[2]) != count*128):
                raise OnlineError("Sparse landmark descriptors or keypoint indices differ")
            keypoints = np.frombuffer(keys[2], dtype="<f4").reshape(count, keys[1])[:, :2]
            descriptors = np.frombuffer(desc[2], dtype=np.uint8).reshape(count, 128)
            if not np.isfinite(keypoints).all() or not np.allclose(keypoints, image_points["xy"], atol=1e-5):
                raise OnlineError("Sparse landmark feature coordinates differ from their indexed tracks")
            for index, pid in enumerate(image_points["id"]):
                if pid == -1:
                    continue
                if pid not in points or (image_id, index) not in points[pid][2]:
                    raise OnlineError("Sparse landmark track does not bind its descriptor observation")
                xyz, error, _ = points[pid]
                if error > 2.5:
                    continue
                observations.setdefault(pid, set()).add(image_id)
                descriptor_rows.append(descriptors[index]); landmark_ids.append(pid)
        if stream.read(1):
            raise OnlineError("Sparse camera evidence has trailing data")
    if seen_names != refs.keys():
        raise OnlineError("Sparse landmark evidence lacks a verified seed camera")
    valid = np.asarray([len(observations[pid]) >= 2 for pid in landmark_ids], dtype=bool)
    ids = np.asarray(landmark_ids, dtype=np.int64)[valid]
    descriptors = np.asarray(descriptor_rows, dtype=np.float32).reshape(-1, 128)[valid]
    if len(set(ids)) < 12:
        raise OnlineError("Sparse seed lacks twelve independently observed landmarks")
    xyz = {int(pid): (transform@np.r_[points[pid][0], 1.])[:3]*scale for pid in set(ids)}
    if hashes != {name: Store._file_sha256(path) for name, path in paths.items()}:
        raise OnlineError("Sparse landmark source changed while being read")
    return {"paths": paths, "hashes": hashes, "landmark_ids": ids,
            "descriptors": descriptors, "xyz": xyz}


class ColmapLandmarkVerifier(PhotoRegistrationVerifier):
    """Measure incoming RGB against frozen, indexed multi-view seed landmarks."""
    def __init__(self, store, registration, views, model_units_per_meter):
        super().__init__(views, model_units_per_meter)
        self.source = _colmap_localization_source(store, registration, views)
        self.source_proof = {"seed_sha256": registration["seed_sha256"],
            "frame_files": registration["source_proof"]["frame_files"],
            "sparse_files": self.source["hashes"]}
        self.method_revision = "seed_landmark_rootsift_pnp_v1"
        self.matcher = None

    def verify(self, view):
        import cv2
        if self.source["hashes"] != {name: Store._file_sha256(path)
                                    for name, path in self.source["paths"].items()}:
            raise OnlineError("Sparse landmark source changed after verification")
        if self.matcher is None:
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=128))
            self.matcher.add([self.source["descriptors"]]); self.matcher.train()
        image = cv2.imread(str(view["image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or max(image.shape) > 8192:
            raise OnlineError("Photo cannot be read for sparse landmark localization")
        keys, descriptors = cv2.SIFT_create(nfeatures=6000).detectAndCompute(image, None)
        if descriptors is None:
            raise OnlineError("Photo has insufficient sparse landmark features")
        # Compare raw-pixel descriptors with the database, then use the measured
        # lens model in PnP. Undistorting descriptors here would change their map.
        descriptors = np.sqrt(descriptors/np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12))*512
        selected = {}
        ids = self.source["landmark_ids"]
        for sequence in self.matcher.knnMatch(descriptors.astype(np.float32), k=min(16, len(ids))):
            if len(sequence) < 2:
                continue
            best = sequence[0]; pid = int(ids[best.trainIdx])
            alternate = next((match for match in sequence[1:] if ids[match.trainIdx] != pid), None)
            if alternate is None or best.distance >= .7*alternate.distance:
                continue
            previous = selected.get(pid)
            if previous is None or best.distance < previous.distance:
                selected[pid] = best  # Each incoming feature/landmark votes once.
        if len(selected) < 12:
            raise OnlineError("Photo lacks twelve independently observed sparse landmark correspondences")
        cloud = np.asarray([self.source["xyz"][pid]/self.scale for pid in selected], dtype=np.float64)
        pixels = np.asarray([keys[match.queryIdx].pt for match in selected.values()], dtype=np.float64)
        k = np.asarray(view["intrinsics"], dtype=float)
        distortion = np.asarray(view.get("distortion_coefficients", [0.]*5), dtype=float)
        if distortion.shape != (5,) or not np.isfinite(distortion).all():
            raise OnlineError("Invalid measured lens-distortion coefficients")
        success, rotation, translation, inliers = cv2.solvePnPRansac(cloud, pixels, k, distortion,
            iterationsCount=100, reprojectionError=2.5, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None or len(inliers) < 8 or len(inliers) < .5*len(cloud):
            raise OnlineError("Sparse photo localization did not achieve independent geometric support")
        indices = inliers.ravel()
        rotation, translation = cv2.solvePnPRefineLM(cloud[indices], pixels[indices], k, distortion,
                                                   rotation, translation)
        projected, _ = cv2.projectPoints(cloud[indices], rotation, translation, k, distortion)
        errors = np.linalg.norm(projected.reshape(-1, 2)-pixels[indices], axis=1)
        good = np.isfinite(errors) & (errors <= 2.5)
        indices, errors = indices[good], errors[good]
        if len(indices) < 8 or len(indices) < .5*len(cloud):
            raise OnlineError("Refined sparse camera lacks support within the reprojection bound")
        r, _ = cv2.Rodrigues(rotation)
        measured = np.eye(4); measured[:3, :3] = r.T; measured[:3, 3] = -r.T@translation.ravel()
        expected = np.asarray(view["pose"]).copy(); expected[:3, 3] /= self.scale
        position = float(np.linalg.norm(measured[:3, 3]-expected[:3, 3]))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(measured[:3, :3].T@expected[:3, :3])-1)/2, -1, 1))))
        if not np.isfinite([position, angle]).all() or position > .10 or angle > 5.:
            raise OnlineError("Photo tracking disagrees with independently localized sparse RGB")
        measured[:3, 3] *= self.scale
        return {"method": "seed_sparse_landmark_pnp", "method_revision": self.method_revision,
            "position_error_m": position, "rotation_error_deg": angle,
            "inliers": len(indices), "correspondences": len(cloud),
            "reprojection_rmse_px": float(np.sqrt(np.mean(errors**2))),
            "reprojection_max_px": float(errors.max()),
            "landmark_ids": [int(pid) for pid in np.asarray(list(selected))[indices]],
            "source_proof": self.source_proof, "localized_pose": measured.tolist(),
            "camera_intrinsics": k.tolist(), "distortion_coefficients": distortion.tolist()}


class Trainer(Protocol):
    @property
    def capture_ids(self) -> list[str]: ...
    def add_image(self, capture_id: str, image_path: Path, pose: np.ndarray,
                  intrinsics: np.ndarray) -> None: ...
    def train_steps(self, count: int) -> int: ...
    def save(self, checkpoint: Path, ply: Path) -> dict: ...


def stage_browser_revision(source: Path, folder: Path, project_id: str, session_id: str,
                           revision_number: int) -> dict:
    """Keep a byte-exact PC view; the headset copy has its own smaller budget."""
    count, digest, size = validate_gaussian_ply(source)
    if source.is_symlink() or size > 512 * 1024 * 1024:
        raise OnlineError('Full browser model exceeds its validated byte budget')
    revisions = folder / 'model-revisions'
    if revisions.is_symlink():
        raise OnlineError('Browser revision directory cannot follow a link')
    revisions.mkdir(exist_ok=True)
    target = revisions / (digest + '.ply')
    receipt = dict(project_id=project_id, session_id=session_id, sha256=digest,
                   source_sha256=digest, vertex_count=count, size_bytes=size,
                   revision_number=revision_number)
    if target.exists():
        if target.is_symlink() or validate_gaussian_ply(target) != (count, digest, size):
            raise OnlineError('Immutable full browser model changed')
    else:
        # Never link a reusable trainer staging file: later writes to that
        # filename must not mutate this earlier immutable model.
        with tempfile.NamedTemporaryFile(dir=revisions, prefix='.full-', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source, temporary)
            if validate_gaussian_ply(temporary) != (count, digest, size):
                raise OnlineError('Full model changed while preserving its browser view')
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    saved = target.with_suffix('.json')
    if saved.exists():
        if saved.is_symlink() or any(json.loads(saved.read_text()).get(key) != value
                for key, value in receipt.items() if key != 'revision_number'):
            raise OnlineError('Immutable full browser receipt changed')
    else:
        _atomic_json(saved, receipt)
    protected = {digest}
    status = folder / 'status.json'
    if status.is_file() and not status.is_symlink():
        previous = (json.loads(status.read_text()).get('artifact') or {}).get('sha256')
        if isinstance(previous, str):
            protected.add(previous)
    slots = max(0, 4 - len(protected))
    for path in sorted(revisions.glob('*.ply'), key=lambda p:p.stat().st_ctime_ns, reverse=True):
        if path.stem in protected or path.is_symlink():
            continue
        if slots:
            slots -= 1
        else:
            try:
                path.unlink(); path.with_suffix('.json').unlink(missing_ok=True)
            except PermissionError:
                pass
    return receipt


def stage_display_revision(source: Path, folder: Path, project_id: str, session_id: str,
                           revision_number: int, *, before_commit=None, publish_current=False,
                           prepared_display: Path | None = None) -> dict:
    """Use the same bounded display artifact and immutable receipt for every publisher."""
    revisions_path = folder / "display-revisions"
    temporary = folder / ".display.tmp.ply"
    if temporary.is_symlink() or source.resolve() == temporary.resolve():
        raise OnlineError('Display staging cannot follow a link or overwrite its own source')
    # The previous temporary may have been hard-linked into an immutable
    # revision by another publisher. Unlink it before any copy/conversion.
    temporary.unlink(missing_ok=True)
    # Preserve all verified model points for now. The earlier distance/size
    # heuristic hid legitimate room geometry, and a display filter must not
    # pretend to be a multi-view quality decision.
    count, digest, size = validate_gaussian_ply(source)
    source_digest = digest
    if count > 240000 or size > 64*1024*1024:
        stage_browser_revision(source, folder, project_id, session_id, revision_number)
    if prepared_display is not None:
        count, digest, size = validate_gaussian_ply(prepared_display)
        if count > 250000 or size > 64*1024*1024:
            raise OnlineError('Preserved display revision exceeds the headset budget')
        shutil.copyfile(prepared_display, temporary)
    elif count > 240000 or size > 64*1024*1024:
        from .reconstruct import make_quest_preview
        target = min(240000, max(1, int(count*(60*1024*1024)/size)))
        merged = make_quest_preview(source, temporary, max_vertices=target)
        count, digest, size = merged["vertex_count"], merged["sha256"], merged["size_bytes"]
    else:
        shutil.copyfile(source, temporary)
    if before_commit:
        before_commit()
    artifact = {"sha256": digest, "size_bytes": size, "vertex_count": count,
                "source_sha256": source_digest, "removed_gaussians": 0,
                "method": "quest_budget_pairwise_merge" if digest != source_digest else "unfiltered_model_review"}
    revisions_path.mkdir(exist_ok=True)
    revision = revisions_path / (artifact["sha256"] + ".ply")
    if revision.exists():
        count, digest, size = validate_gaussian_ply(revision)
        if (count != artifact["vertex_count"] or digest != artifact["sha256"] or
                size != artifact["size_bytes"]):
            raise OnlineError("Existing display revision has a mismatched checksum")
    else:
        os.link(temporary, revision)
    receipt_path = revision.with_suffix(".json")
    receipt = {**{key: artifact[key] for key in ("sha256", "size_bytes", "vertex_count", "source_sha256")},
        "project_id": project_id, "session_id": session_id,
        "revision_number": revision_number}
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if (type(existing.get('revision_number')) is not int or existing['revision_number'] < 0 or
                any(existing.get(key) != receipt[key] for key in receipt if key != "revision_number")):
            raise OnlineError("Immutable display receipt differs from its original source")
    else:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=revisions_path,
                                         prefix=".receipt-", delete=False) as handle:
            json.dump(receipt, handle); handle.flush(); os.fsync(handle.fileno())
            receipt_temp = Path(handle.name)
        try:
            os.link(receipt_temp, receipt_path)
        finally:
            receipt_temp.unlink(missing_ok=True)
    if publish_current:
        _replace_with_retry(temporary, folder / 'display.ply')
    # A bounded cache lets browser and Quest downloads finish even while
    # the optimizer publishes newer snapshots. These are display copies;
    # current.ckpt and current.ply remain the one authoritative model.
    old = sorted(revisions_path.glob("*.ply"),
                 key=lambda path: path.stat().st_ctime_ns, reverse=True)
    protected = {artifact['sha256']}
    status_path = folder / 'status.json'
    if status_path.is_file() and not status_path.is_symlink():
        try:
            previous = json.loads(status_path.read_text(encoding='utf-8'))
            previous_sha = (previous.get('display_artifact') or {}).get('sha256')
            if isinstance(previous_sha, str):
                protected.add(previous_sha)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    other_slots = max(0, 16-len(protected))
    for path in old:
        if path.stem in protected:
            continue
        if path.is_symlink():
            continue
        if other_slots:
            other_slots -= 1
            continue
        try:
            path.unlink()
            path.with_suffix(".json").unlink(missing_ok=True)
        except PermissionError:
            pass  # A Windows reader is still using this older revision.
    return artifact


class OnlineCoordinator:
    """Ingest each selected receipt once into one trainer instance and checkpoint.

    A failed camera or trainer update does not advance the durable cursor.  The
    caller may retry after correcting calibration; original capture files stay
    untouched.  Excluding a previously trained image cannot undo past gradient
    updates and is reported as a limitation in status.
    """

    def __init__(self, store: Store, session_id: str, seed_ply: Path,
                 registration: dict, trainer: Trainer, *, scheduled=False, photo_verifier=None):
        self.store = store
        self.session = store.session(session_id)
        self.session_id = session_id
        self.scheduled = scheduled
        initial_control = self._control()
        self.seed_ply = seed_ply
        self.registration = registration
        self.measured = registration.get("verification_method") == "source_job_pose_fit"
        if self.measured:
            actual_seed, actual = registration_from_seed_job(store, session_id, registration["evidence_job_id"])
            if actual_seed.resolve() != seed_ply.resolve() or actual != registration:
                raise OnlineError("Measured seed registration no longer matches source evidence")
        self.root = store.root / "projects" / self.session["project_id"] / "online" / session_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.checkpoint_path = self.root / "current.ckpt"
        self.ply_path = self.root / "current.ply"
        self.display_path = self.root / "display.ply"
        self.revisions_path = self.root / "display-revisions"
        _, seed_digest, _ = validate_gaussian_ply(seed_ply)
        self.model_from_session, self.calibration_revision, self.image_orientation = verified_registration(
            registration, session_id, seed_digest)
        seed_ids = registration.get("seed_capture_ids", [])
        if (not isinstance(seed_ids, list) or len(seed_ids) > 10000 or
                any(not isinstance(item, str) for item in seed_ids) or
                len(set(seed_ids)) != len(seed_ids)):
            raise OnlineError("Invalid baseline capture list")
        self.seed_ids = set(seed_ids)
        self.seed_sha256 = seed_digest
        self.trainer = trainer
        self.model_units_per_meter = float(np.cbrt(np.linalg.det(self.model_from_session[:3, :3])))
        if hasattr(trainer, "configure_geometry_scale"):
            trainer.configure_geometry_scale(self.model_units_per_meter)
        self.baseline_views = []
        seed_references = []
        self.localized_distortion = None
        if self.seed_ids:
            selected = _selected_ids(store, session_id)
            if not self.seed_ids <= selected:
                raise OnlineError("Seed includes excluded photos; explicit baseline review is required")
            if self.measured:
                seed_references = seed_camera_views(store, registration)
            else:
                snapshot = store.export_session(session_id, capture_ids=sorted(self.seed_ids))
                for item in snapshot["captures"]:
                    pose, k = self._camera(item)
                    seed_references.append({"capture_id": item["receipt"]["capture_id"],
                        "image_path": Path(item["image_path"]), "pose": pose, "intrinsics": k})
            # Keep spatially/time-distributed references, not only the tail of a
            # potentially long perimeter walk. Baseline ingress never grows rows.
            chosen = np.linspace(0, len(seed_references)-1, min(24, len(seed_references)), dtype=int)
            for index in chosen:
                self.baseline_views.append(seed_references[index])
            if not hasattr(trainer, "prime_baseline_views"):
                raise OnlineError("Trainer cannot prime independently verified baseline views")
            if self.measured:
                self.localized_distortion = seed_references[0]["distortion_coefficients"]
                if any(view["distortion_coefficients"] != self.localized_distortion for view in seed_references):
                    raise OnlineError("Seed mixes independently calibrated camera distortion")
                if hasattr(trainer, "configure_camera_distortion"):
                    trainer.configure_camera_distortion(self.localized_distortion)
                elif any(self.localized_distortion):
                    raise OnlineError("Trainer cannot apply measured seed lens-distortion correction")
            trainer.prime_baseline_views(self.baseline_views)
        self.photo_verifier = photo_verifier or (ColmapLandmarkVerifier(
            store, registration, seed_references, self.model_units_per_meter) if self.measured else None)
        self.localized_intrinsics = seed_references[0]["intrinsics"] if self.measured and seed_references else None
        def check_initial_controls():
            ensure_refinement_allowed(self.status_path)
            if self._control() != initial_control:
                raise OnlineError("Gaussian controls changed during initialization")
            if not self.seed_ids <= _selected_ids(store, session_id):
                raise OnlineError("Gaussian seed selection changed during initialization")
        check_initial_controls()
        _recover_online_publication(store, session_id, seed_ply, registration,
                                    before_commit=check_initial_controls)
        if self.status_path.exists():
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            if (status.get("session_id") != session_id or status.get("seed_sha256") != seed_digest
                    or status.get("registration") != registration):
                raise OnlineError("Existing online model belongs to a different seed or registration")
            if not self.checkpoint_path.is_file() and status["accepted_ids"]:
                raise OnlineError("Online checkpoint is missing; do not replay captures into a new model")
            self.status = status
            localizer_proof = getattr(self.photo_verifier, "source_proof", None)
            if self.status.get("localizer_source_proof") not in (None, localizer_proof):
                raise OnlineError("Saved localization source differs from the frozen sparse map")
            self.status.setdefault("quarantined", {})
            self.status.setdefault("revision_number", 0)
            checkpoint_ids = trainer.capture_ids
            if not set(self.status["accepted_ids"]) <= set(checkpoint_ids):
                raise OnlineError("Checkpoint lost images recorded in the online receipt")
            # A crash between checkpoint and status publication is recoverable
            # because the checkpoint carries the authoritative ingress cursor.
            if checkpoint_ids != self.status["accepted_ids"]:
                self.status["accepted_ids"] = list(checkpoint_ids)
                self.status["status"] = "recovered_checkpoint"
                _atomic_json(self.status_path, self.status)
            if self.status.get("artifact") and not self.status.get("display_artifact"):
                count, digest, size = validate_gaussian_ply(self.ply_path)
                if (count == self.status["artifact"]["vertex_count"] and
                        digest == self.status["artifact"]["sha256"] and
                        size == self.status["artifact"]["size_bytes"]):
                    self.status["display_artifact"] = self._publish_display(self.ply_path)
                    _atomic_json(self.status_path, self.status)
        else:
            self.status = {"schema_version": 1, "session_id": session_id,
                           "project_id": self.session["project_id"], "seed_sha256": seed_digest,
                           "registration": registration, "seed_ids": sorted(self.seed_ids),
                           "accepted_ids": [], "training_steps": 0, "quarantined": {},
                           "seed_job_id": registration.get("evidence_job_id"), "revision_number": 0,
                           "status": "initialized", "artifact": None, "updated_utc": utc_now()}
            _atomic_json(self.status_path, self.status)
        if getattr(self.photo_verifier, "source_proof", None) is not None:
            self.status["localizer_source_proof"] = self.photo_verifier.source_proof
        # Retain bounded independently localized references across a restart.
        # A saved pose proof is reusable only for the exact original receipt.
        if self.measured and hasattr(self.photo_verifier, "append_verified"):
            proofs = self.status.get("photo_registration", {})
            reference_ids = [cid for cid in self.status["accepted_ids"]
                             if proofs.get(cid, {}).get("image_sha256")][-12:]
            if reference_ids:
                snapshot = store.export_session(session_id, capture_ids=reference_ids)
                for item in snapshot["captures"]:
                    cid = item["receipt"]["capture_id"]
                    proof = proofs[cid]
                    if (proof.get("image_sha256") != item["receipt"]["image_sha256"] or
                            proof.get("metadata_sha256") != _json_digest(item["metadata"]) or
                            proof.get("seed_sha256") != self.seed_sha256):
                        raise OnlineError("Localized photo reference changed after its verified admission")
                    pose, k = self._camera(item)
                    view = {"capture_id": cid, "image_path": Path(item["image_path"]), "pose": pose, "intrinsics": k}
                    self.photo_verifier.append_verified(self._localized_view(view, proof))
        # Baseline is viewable before any delta arrives, and its camera cache
        # plus optimizer are restartable without retraining the original batch.
        if not self.status.get("artifact"):
            self._save_current(before_commit=check_initial_controls)

    def _control(self):
        if not self.scheduled:
            return None
        return _scheduled_control(self.store, self.session_id)

    def _camera(self, item):
        metadata, receipt = item["metadata"], item["receipt"]
        if (metadata.get("capture_id") != receipt["capture_id"] or
                metadata.get("image_sha256") != receipt["image_sha256"] or
                metadata.get("project_id") != self.session["project_id"] or
                metadata.get("session_id") != self.session_id or
                metadata.get("tracking_origin_id") != self.session["tracking_origin_id"]):
            raise OnlineError("Photo identity or anchored tracking origin differs")
        signature = self.registration.get("camera_signature")
        if signature:
            if any(metadata.get(key) != signature[key] for key in ("camera_id", "image_width", "image_height")):
                raise OnlineError("Photo physical camera or dimensions differ from the seed")
            intrinsics = metadata.get("intrinsics", {})
            if (set(intrinsics) != set(signature["intrinsics"]) or any(
                    not np.isfinite(float(intrinsics[key])) or
                    abs(float(intrinsics[key])-float(signature["intrinsics"][key])) > 1.
                    for key in intrinsics)):
                raise OnlineError("Photo camera calibration differs from measured seed")
        return validated_camera(metadata, self.model_from_session, self.calibration_revision,
                                self.image_orientation)

    def _localized_view(self, view, proof):
        if "localized_pose" not in proof:
            return view
        pose = np.asarray(proof["localized_pose"], dtype=float)
        k = np.asarray(proof["camera_intrinsics"], dtype=float)
        distortion = np.asarray(proof.get("distortion_coefficients", [0.]*5), dtype=float)
        if (pose.shape != (4, 4) or k.shape != (3, 3) or not np.isfinite(pose).all() or
                not np.isfinite(k).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5) or
                not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-4) or
                not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-4) or
                min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1])):
            raise OnlineError("Invalid independently localized camera proof")
        if (distortion.shape != (5,) or not np.isfinite(distortion).all() or
                (self.localized_distortion is not None and
                 not np.array_equal(distortion, self.localized_distortion))):
            raise OnlineError("Localized camera lens calibration differs from measured seed")
        position = np.linalg.norm(pose[:3, 3]-view["pose"][:3, 3])/self.model_units_per_meter
        angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T@view["pose"][:3, :3])-1)/2, -1, 1)))
        if position > .10 or angle > 5.:
            raise OnlineError("Saved localization differs from tracked camera proof")
        return {**view, "pose": pose, "intrinsics": k, "distortion_coefficients": distortion.tolist()}

    def _save_current(self, before_commit=None):
        journal_path = self.root/'pending-publication.json'
        if journal_path.exists():
            raise OnlineError('Recover the interrupted online publication before another GPU update')
        base = json.loads(self.status_path.read_text(encoding='utf-8'))
        if (base.get('artifact') != self.status.get('artifact') or
                base.get('revision_number', 0) != self.status.get('revision_number', 0)):
            raise OnlineError('A newer online generation supersedes this publisher')
        # The simplifier selects its reader from the final suffix, even for a
        # private checkpoint export. Keep the temporary Gaussian name as PLY.
        temporary_ckpt, temporary_ply = self.root/".current.ckpt.tmp", self.root/".current.tmp.ply"
        metrics = self.trainer.save(temporary_ckpt, temporary_ply)
        if not temporary_ckpt.is_file() or not temporary_ply.is_file():
            raise OnlineError("Trainer did not write a checkpoint and verified Gaussian artifact")
        count, digest, size = validate_gaussian_ply(temporary_ply)
        self._update_counts()
        if before_commit:
            before_commit()
        changed = digest != (self.status.get("artifact") or {}).get("sha256")
        revision = self.status.get("revision_number", 0)+int(changed)
        display = (self._publish_display(temporary_ply, revision, before_commit=before_commit,
                                        publish_current=False) if changed or
                   not self.status.get("display_artifact") else self.status["display_artifact"])
        if before_commit:
            before_commit()
        next_status = deepcopy(self.status)
        next_status.update(artifact={'sha256':digest, 'size_bytes':size, 'vertex_count':count},
                           display_artifact=display, revision_number=revision)
        if metrics:
            next_status['model_metrics'] = metrics
        next_status['pending_refinement'] = deepcopy(getattr(self.trainer, 'pending_refinement', {}))
        next_status.pop('last_error', None); next_status.pop('error', None)
        next_status.pop('idle_yield', None); next_status.pop('worker_state', None)
        next_status.update(updated_utc=utc_now(), status='training' if next_status.get('pending_count') else 'waiting_for_new_views')
        _atomic_json(journal_path, {'schema_version':1, 'base_identity':_publication_identity(base),
            'tracking_origin_id':self.session['tracking_origin_id'], 'status':next_status,
            'input_ledger':_publication_ledger(next_status, self.registration, self.store), 'files':{
                'current.ply':{'sha256':digest, 'size_bytes':size},
                'current.ckpt':{'sha256':_file_sha256(temporary_ckpt), 'size_bytes':temporary_ckpt.stat().st_size}}})
        if before_commit:
            before_commit()
        if _publication_identity(json.loads(self.status_path.read_text())) != _publication_identity(base):
            raise OnlineError('Online manifest changed before publication')
        if changed:
            _replace_with_retry(temporary_ply, self.ply_path)
        else:
            temporary_ply.unlink()
        _replace_with_retry(temporary_ckpt, self.checkpoint_path)
        if changed or not self.status.get('display_artifact'):
            _replace_with_retry(self.root/'.display.tmp.ply', self.display_path)
        self.status = next_status
        self._update_counts()
        self.status.pop("last_error", None)
        self.status.pop("error", None)
        _atomic_json(self.status_path, self.status)
        journal_path.unlink()

    def _update_counts(self):
        update_photo_counts(self.status, self.seed_ids,
            getattr(self.trainer, 'published_capture_ids', []),
            getattr(self.trainer, 'rejected_capture_ids', []),
            getattr(self.trainer, 'pending_refinement', {}))

    def _publish_display(self, source: Path, revision_number=None, before_commit=None,
                         publish_current=True) -> dict:
        number = self.status.get("revision_number", 0) if revision_number is None else revision_number
        return stage_display_revision(source, self.root, self.session["project_id"], self.session_id,
                                      number, before_commit=before_commit, publish_current=publish_current)

    def tick(self, *, steps: int = 4, maximum_new: int = 1) -> dict:
        if self.status.get("status") == "paused_quality_review":
            raise OnlineError("Online refinement is paused for quality review; photo receiving remains available")
        if steps < 0 or steps > 100 or maximum_new < 1 or maximum_new > 16:
            raise OnlineError("Online work per tick exceeds bounds")
        ensure_refinement_allowed(self.status_path)
        control = self._control()
        selected = _selected_ids(self.store, self.session_id)
        accepted = set(self.status["accepted_ids"])
        removed = (self.seed_ids | accepted)-selected
        if removed:
            self.status.update(status="paused_selection_review", excluded_after_training=sorted(removed))
            _atomic_json(self.status_path, self.status)
            raise OnlineError("Previously trained photos were excluded; explicit derived baseline review is required")
        ingress = []
        pending = bool(getattr(self.trainer, "has_pending_work", False))
        quarantined = self.status.setdefault("quarantined", {})
        if not pending:
            revision = getattr(self.trainer, "review_method_revision", None)
            retries = (set(getattr(self.trainer, "rejected_capture_ids", [])) & accepted & selected)-set(
                getattr(self.trainer, "published_capture_ids", []))
            if not revision or not hasattr(self.trainer, "retry_rejected_image"):
                retries = set()
            else:
                attempts = getattr(self.trainer, "review_attempts", {})
                reviews = getattr(self.trainer, "rejected_capture_reviews", {})
                retries = {cid for cid in retries if attempts.get(cid) != revision and
                           reviews.get(cid, {}).get("review_method_revision") != revision}
            rows = self.store.captures(self.session_id)
            candidates = ([row for row in rows if row["capture_id"] in retries] +
                          [row for row in rows if row["capture_id"] in selected-(self.seed_ids | accepted)])
            attempted = 0
            for row in candidates:
                cid = row["capture_id"]
                fingerprint = _json_digest({"image_sha256": row["image_sha256"], "metadata": row["metadata"],
                    "localizer_revision": getattr(self.photo_verifier, "method_revision", None),
                    "localizer_source_proof": getattr(self.photo_verifier, "source_proof", None),
                    "review_method_revision": revision if cid in retries else None,
                    "seed_frame_proof": self.registration.get("source_proof", {}).get("frame_files")})
                old = quarantined.get(cid)
                if (old and old.get("fingerprint") == fingerprint and
                    (not old.get("retry_with_new_reference") or old.get("reference_count") == len(accepted))):
                    continue
                if len(ingress) >= maximum_new or attempted >= 16:
                    break
                attempted += 1
                try:
                    item = (self.store.export_session(self.session_id, capture_ids=[cid])["captures"][0]
                            if cid in retries else self.store.export_pending_captures(
                                self.session_id, self.seed_ids | accepted, {cid}, 1)[0])
                    pose, k = self._camera(item)
                    if self.localized_intrinsics is not None:
                        k = self.localized_intrinsics.copy()
                    view = {"capture_id": cid, "image_path": Path(item["image_path"]), "pose": pose, "intrinsics": k}
                    if self.localized_distortion is not None:
                        view["distortion_coefficients"] = self.localized_distortion
                    proof = self.photo_verifier.verify(view) if self.photo_verifier else {"method": "reviewed_registration"}
                    proof = {**proof, "image_sha256": item["receipt"]["image_sha256"],
                        "metadata_sha256": _json_digest(item["metadata"]), "seed_sha256": self.seed_sha256}
                    view = self._localized_view(view, proof)
                except (OnlineError, StoreError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
                    reason = str(error) if not isinstance(error, StoreError) else error.detail
                    quarantined[cid] = {"fingerprint": fingerprint, "reason": reason[:320],
                        "reference_count": len(accepted), "retry_with_new_reference": any(
                            word in reason.lower() for word in ("insufficient", "lacks", "too few", "did not achieve")),
                        "updated_utc": utc_now()}
                    continue
                if cid in retries:
                    self.trainer.retry_rejected_image(cid, view["image_path"], view["pose"], view["intrinsics"])
                else:
                    self.trainer.add_image(cid, view["image_path"], view["pose"], view["intrinsics"])
                    self.status["accepted_ids"].append(cid)
                self.status.setdefault("photo_registration", {})[cid] = proof
                quarantined.pop(cid, None)
                ingress.append(cid)
                if hasattr(self.photo_verifier, "append_verified"):
                    self.photo_verifier.append_verified(view)
                if getattr(self.trainer, "has_pending_work", False):
                    break  # Refine one bounded candidate cohort before another admission.
        def check_controls():
            ensure_refinement_allowed(self.status_path)
            if self._control() != control:
                raise OnlineError("Gaussian update controls changed before publication")
            if not (self.seed_ids | set(self.status["accepted_ids"])) <= _selected_ids(self.store, self.session_id):
                raise OnlineError("Gaussian photo selection changed before publication")
        check_controls()
        trained = 0
        if steps and (ingress or pending):
            trained = self.trainer.train_steps(steps)
            self.status["training_steps"] += trained
        if ingress or trained:
            self._save_current(before_commit=check_controls)
        self.status["status"] = "training" if getattr(self.trainer, "has_pending_work", False) else "waiting_for_new_views"
        self.status["selected_count"] = len(selected)
        self.status["seed_view_count"] = len(self.seed_ids)
        self.status["total_count"] = len(self.store.captures(self.session_id))
        self.status["quarantined_count"] = len(quarantined)
        self._update_counts()
        self.status["excluded_after_training"] = []
        self.status["updated_utc"] = utc_now()
        self.status.pop("last_error", None)
        self.status.pop("error", None)
        self.status.pop("idle_yield", None)
        self.status.pop("worker_state", None)
        check_controls()
        _atomic_json(self.status_path, self.status)
        return dict(self.status)


    def yield_idle(self, input_revision: str, control_revision: str, *, run_id=None) -> bool:
        """Release scheduled GPU ownership only after all current work is durable.

        The receipt identifies the inventory actually drained, rather than the
        older launch inventory. A new capture racing this check remains pending
        for the scheduler; no checkpoint, optimizer or model is reset.
        """
        if (not self.scheduled or getattr(self.trainer, "has_pending_work", False) or
                self.status.get("status") != "waiting_for_new_views"):
            return False
        ensure_refinement_allowed(self.status_path)
        if self._control() != control_revision:
            raise OnlineError("Gaussian controls changed before idle yield")
        from .jobs import _depth_inventory
        if _depth_inventory(self.store, self.session_id)[2] != input_revision:
            return False
        if not self.checkpoint_path.is_file() or not self.status.get("artifact"):
            raise OnlineError("An idle Gaussian worker must retain its verified checkpoint and model")
        self.status.update(worker_state="idle", idle_yield={
            "input_revision": input_revision, "control_revision": control_revision,
            "process_id": os.getpid(), "run_id": run_id}, updated_utc=utc_now())
        _atomic_json(self.status_path, self.status)
        return True


def _run_online_worker(worker, *, steps=32, once=False, idle_seconds=15, run_id=None,
                       control_revision=None):
    """Drain bounded cohorts, then relinquish an unchanged idle scheduled scan."""
    idle_key, idle_since = None, None
    control_revision = (worker._control() if worker.scheduled and control_revision is None
                        else control_revision)
    while True:
        if worker.scheduled and worker._control() != control_revision:
            raise OnlineError("Gaussian controls changed after the scheduled launch")
        status = worker.tick(steps=steps, maximum_new=1)
        if once:
            return
        if (worker.scheduled and status.get("status") == "waiting_for_new_views" and
                not getattr(worker.trainer, "has_pending_work", False)):
            from .jobs import _depth_inventory
            revision = _depth_inventory(worker.store, worker.session_id)[2]
            control = worker._control()
            # updated_utc changes on every poll and is intentionally excluded.
            # Quarantine/refinement decisions are work too: drain them before
            # declaring the entire current inventory idle.
            key = _json_digest({"input_revision": revision, "control_revision": control,
                "accepted_ids": status.get("accepted_ids", []),
                "training_steps": status.get("training_steps", 0),
                "review_attempts": getattr(worker.trainer, "review_attempts", {}),
                "quarantined": {cid: {k: record.get(k) for k in
                    ("fingerprint", "reason", "reference_count")}
                    for cid, record in status.get("quarantined", {}).items()}})
            now = time.monotonic()
            if key != idle_key:
                idle_key, idle_since = key, now
            elif now-idle_since >= idle_seconds:
                if worker.yield_idle(revision, control, run_id=run_id):
                    return
                idle_key, idle_since = None, None
        else:
            idle_key, idle_since = None, None
        time.sleep(2)


def main() -> None:
    """Explicit standalone or scheduler-owned persistent Gaussian worker."""
    parser = argparse.ArgumentParser(description="Opt-in, single-state online Gaussian worker")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--seed-ply", type=Path)
    parser.add_argument("--registration", type=Path,
                        help="Reviewed seed-specific Quest/model similarity JSON")
    parser.add_argument("--seed-job", help="Measure and bootstrap this saved same-session reconstruction")
    parser.add_argument("--bootstrap", action="store_true", help="Reuse or create one initial bounded photo seed")
    parser.add_argument("--scheduled", action="store_true", help="Obey saved session splat_updates controls")
    parser.add_argument("--control-revision", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--reviewed-native-orientation", action="store_true")
    parser.add_argument("--steps-per-tick", type=int, default=32)
    parser.add_argument("--run", action="store_true", help="Required to enable GPU work")
    parser.add_argument("--once", action="store_true", help="Run one bounded ingress/training tick")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; online training is never enabled implicitly")
    if ((args.seed_job or args.bootstrap) and (args.seed_ply or args.registration) or
        (args.seed_job and args.bootstrap) or
        (not args.seed_job and not args.bootstrap and (not args.seed_ply or not args.registration))):
        parser.error("Use --seed-job, --bootstrap, or the --seed-ply/--registration pair")
    if not 1 <= args.steps_per_tick <= 100:
        parser.error("--steps-per-tick must be between 1 and 100")
    from .reconstruct import _worker_lock
    root = args.data_dir.resolve()
    store = Store(root)
    session = store.session(args.session_id)
    project_root = (root / "projects" / session["project_id"]).resolve()
    checkpoint = project_root / "online" / args.session_id / "current.ckpt"
    # A quality-review pause is durable, not merely a stopped process. A
    # routine service restart must not resume a model known to be degrading.
    def scheduled_control():
        if not args.scheduled:
            return None
        return _scheduled_control(store, args.session_id)
    try:
        ensure_refinement_allowed(checkpoint.parent / "status.json")
        control = scheduled_control()
        if args.control_revision is not None and control != args.control_revision:
            raise OnlineError("Gaussian controls changed before the scheduled launch")
        seed_job = args.seed_job
        if args.bootstrap:
            seed_job = bootstrap_seed_job(store, args.session_id,
                                         native_orientation_reviewed=args.reviewed_native_orientation)
        if seed_job:
            seed, registration = registration_from_seed_job(store, args.session_id, seed_job)
        else:
            seed = args.seed_ply.resolve()
            if not seed.is_relative_to(project_root / "reconstructions") or seed.name not in {"splat.ply", "quest-preview.ply"}:
                raise OnlineError("Seed must be a saved reconstruction from this project")
            registration = json.loads(args.registration.read_text(encoding="utf-8"))
        if scheduled_control() != control:
            raise OnlineError("Gaussian controls changed during baseline bootstrap")
        # Registration is measured before initializing any CUDA parameters.
        verified_registration(registration, args.session_id, validate_gaussian_ply(seed)[1])
        from .online_torch import GsplatTrainer
        with _worker_lock(root):
            def check_recovery_controls():
                if scheduled_control() != control:
                    raise OnlineError('Gaussian controls changed during publication recovery')
            _recover_online_publication(store, args.session_id, seed, registration,
                                        before_commit=check_recovery_controls)
            trainer = GsplatTrainer(seed, checkpoint if checkpoint.is_file() else None)
            worker = OnlineCoordinator(store, args.session_id, seed, registration, trainer, scheduled=args.scheduled)
            _run_online_worker(worker, steps=args.steps_per_tick, once=args.once,
                               run_id=args.run_id, control_revision=control)
    except (OnlineError, ReconstructionError, StoreError, OSError, ValueError, KeyError,
            TypeError, RuntimeError, ImportError) as error:
        status_path = checkpoint.parent/"status.json"
        try:
            status = json.loads(status_path.read_text()) if status_path.is_file() else {
                "schema_version": 1, "project_id": session["project_id"], "session_id": args.session_id,
                "seed_job_id": locals().get("seed_job"), "accepted_ids": [], "artifact": None}
            reason = error.detail if isinstance(error, StoreError) else str(error)
            if status.get("status") not in {"paused_quality_review", "paused_selection_review"}:
                status["status"] = "paused_scheduler" if "controls" in reason or "disabled" in reason else "blocked"
            status.update(last_error=reason[:640], updated_utc=utc_now())
            _atomic_json(status_path, status)
        except (OSError, ValueError, TypeError):
            pass
        if isinstance(error, OnlineError) and ("controls" in str(error) or "disabled" in str(error)):
            return
        parser.exit(2, f"Gaussian refinement unavailable: {error}\n")


if __name__ == "__main__":
    main()
