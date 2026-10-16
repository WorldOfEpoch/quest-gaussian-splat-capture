"""Small server-rendered project view for the PC's loopback bridge."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
import re
import sqlite3
import struct
import uuid
from pathlib import Path

import numpy as np

from .models import utc_now
from .store import Store, StoreError


CAPTURE_PAGE_SIZE = 24


def capture_gallery(store: Store, project_id: str, session_id: str | None = None,
                    page: int = 1, page_size: int = CAPTURE_PAGE_SIZE) -> tuple[list[dict], int]:
    """Bounded project or session photo listing; originals stay on disk."""
    store.project(project_id)
    if session_id is not None:
        session = store.session(session_id)
        if session["project_id"] != project_id:
            raise StoreError(404, "Capture batch not found in project")
    if page < 1 or page_size < 1 or page_size > CAPTURE_PAGE_SIZE:
        raise StoreError(422, "Invalid gallery page")
    with store.connection() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS capture_selections (
            capture_id TEXT PRIMARY KEY REFERENCES captures(capture_id),
            included INTEGER NOT NULL CHECK(included IN (0,1)), updated_utc TEXT NOT NULL)""")
        if session_id is None:
            count = db.execute("SELECT count(*) FROM captures WHERE project_id=?", (project_id,)).fetchone()[0]
            rows = db.execute("""SELECT c.capture_id,c.session_id,c.received_utc,c.metadata_json,
                COALESCE(s.included,1) AS included FROM captures c
                LEFT JOIN capture_selections s ON s.capture_id=c.capture_id
                WHERE c.project_id=? ORDER BY c.received_utc DESC,c.capture_id DESC LIMIT ? OFFSET ?""",
                (project_id, page_size, (page - 1) * page_size)).fetchall()
        else:
            count = db.execute("SELECT count(*) FROM captures WHERE project_id=? AND session_id=?",
                               (project_id, session_id)).fetchone()[0]
            rows = db.execute("""SELECT c.capture_id,c.session_id,c.received_utc,c.metadata_json,
                COALESCE(s.included,1) AS included FROM captures c
                LEFT JOIN capture_selections s ON s.capture_id=c.capture_id
                WHERE c.project_id=? AND c.session_id=?
                ORDER BY c.received_utc DESC,c.capture_id DESC LIMIT ? OFFSET ?""",
                (project_id, session_id, page_size, (page - 1) * page_size)).fetchall()
    records = []
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        records.append({"capture_id": row["capture_id"], "session_id": row["session_id"],
                        "received_utc": row["received_utc"], "captured_utc": metadata.get("image_timestamp_raw"),
                        "camera_id": metadata.get("camera_id"), "included": bool(row["included"])})
    return records, count


def set_capture_selection(store: Store, project_id: str, session_id: str,
                          capture_id: str, included: bool) -> None:
    store.project(project_id)
    session = store.session(session_id)
    if session["project_id"] != project_id:
        raise StoreError(404, "Capture batch not found in project")
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("""CREATE TABLE IF NOT EXISTS capture_selections (
            capture_id TEXT PRIMARY KEY REFERENCES captures(capture_id),
            included INTEGER NOT NULL CHECK(included IN (0,1)), updated_utc TEXT NOT NULL)""")
        exists = db.execute("SELECT 1 FROM captures WHERE project_id=? AND session_id=? AND capture_id=?",
                            (project_id, session_id, capture_id)).fetchone()
        if not exists:
            raise StoreError(404, "Photo not found in capture batch")
        db.execute("""INSERT INTO capture_selections VALUES (?,?,?)
            ON CONFLICT(capture_id) DO UPDATE SET included=excluded.included,updated_utc=excluded.updated_utc""",
            (capture_id, int(included), utc_now()))


def verified_capture_image(store: Store, project_id: str, capture_id: str) -> tuple[Path, str]:
    """Resolve a project's original image only after path and checksum checks."""
    store.project(project_id)
    with store.connection() as db:
        row = db.execute("""SELECT image_path,image_sha256,metadata_json FROM captures
            WHERE project_id=? AND capture_id=?""", (project_id, capture_id)).fetchone()
    if not row:
        raise StoreError(404, "Photo not found in project")
    metadata = json.loads(row["metadata_json"])
    extension = "jpg" if metadata.get("image_encoding") == "jpeg" else "png" if metadata.get("image_encoding") == "png" else None
    expected = Path("projects") / project_id / "captures" / f"{capture_id}.{extension}"
    if extension is None or Path(row["image_path"]) != expected:
        raise StoreError(404, "Original photo path is invalid")
    path = store.root / expected
    project_dir = store.root / "projects" / project_id
    capture_dir = project_dir / "captures"
    if (project_dir.is_symlink() or capture_dir.is_symlink() or path.is_symlink()
            or not path.resolve().is_relative_to(capture_dir.resolve()) or not path.is_file()):
        raise StoreError(404, "Original photo is unavailable")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise StoreError(404, "Original photo is unavailable") from error
    if digest.hexdigest() != row["image_sha256"]:
        raise StoreError(404, "Original photo checksum differs")
    return path, "image/jpeg" if extension == "jpg" else "image/png"


def verified_job(root: Path, project_id: str, job_id: str) -> tuple[dict, Path] | None:
    """Resolve only an immutable PLY inside the requested project's UUID job folder."""
    try:
        uuid.UUID(project_id)
        uuid.UUID(job_id)
    except ValueError:
        return None
    folder = (root / "projects" / project_id / "reconstructions" / job_id).resolve()
    manifest_path = folder / "job.json"
    try:
        job = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = job["artifact"]
        ply = folder / "splat.ply"
        if (job["project_id"] != project_id or job["job_id"] != job_id
                or job["status"] != "ready_for_blender" or not isinstance(artifact, dict)
                or Path(artifact["path"]).resolve() != ply.resolve()
                or not ply.is_file() or ply.stat().st_size != artifact["size_bytes"]
                or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])):
            return None
        digest = hashlib.sha256()
        with ply.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != artifact["sha256"]:
            return None
        return job, ply
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _ply_frame(path: Path, limit: int = 20000) -> dict | None:
    """Sample Gaussian centers to choose a usable initial browser camera."""
    with path.open("rb") as stream:
        header = stream.read(128 * 1024)
    marker = b"end_header\n"
    end = header.find(marker)
    if end < 0:
        return None
    lines = header[:end].decode("ascii", errors="strict").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        return None
    count = 0
    offset = 0
    offsets = {}
    in_vertex = False
    sizes = {"char": 1, "uchar": 1, "short": 2, "ushort": 2,
             "int": 4, "uint": 4, "float": 4, "double": 8}
    for line in lines:
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex ")
            if in_vertex:
                count = int(line.split()[2])
        elif in_vertex and line.startswith("property "):
            _, kind, name = line.split()
            if name in {"x", "y", "z"} and kind == "float":
                offsets[name] = offset
            offset += sizes[kind]
    if count < 1 or len(offsets) != 3 or offset < 12:
        return None
    body = end + len(marker)
    if path.stat().st_size < body + count * offset:
        return None
    values = [[], [], []]
    step = max(1, count // limit)
    with path.open("rb") as stream:
        for index in range(0, count, step):
            stream.seek(body + index * offset)
            record = stream.read(offset)
            xyz = [struct.unpack_from("<f", record, offsets[name])[0] for name in ("x", "y", "z")]
            if all(math.isfinite(value) for value in xyz):
                for axis, value in enumerate(xyz):
                    values[axis].append(value)
    if any(not axis for axis in values):
        return None
    bounds = []
    for axis in values:
        axis.sort()
        bounds.append((axis[int((len(axis) - 1) * .02)], axis[int((len(axis) - 1) * .98)]))
    center = [(low + high) / 2 for low, high in bounds]
    radius = max(0.5, math.sqrt(sum((high - low) ** 2 for low, high in bounds)) * .75)
    up_axis = "z" if "comment Vertical Axis: z" in lines else "y"
    return {"center": center, "radius": radius, "up_axis": up_axis}


def _person_view(job: dict, view: dict, job_dir: Path) -> dict:
    """Place the browser eye at a registered photograph, never at a splat centroid."""
    try:
        processed = Path(job["processed_dir"]).resolve()
        config = Path(job["training_config"]).resolve()
        if (not processed.is_relative_to(job_dir.resolve()) or
                not config.is_relative_to(job_dir.resolve())):
            return view
        transforms = json.loads((processed / "transforms.json").read_text(encoding="utf-8"))
        parser = json.loads((config.parent / "dataparser_transforms.json").read_text(encoding="utf-8"))

        def homogeneous(value):
            matrix = np.asarray(value, dtype=float)
            if matrix.shape == (3, 4):
                matrix = np.vstack((matrix, [0, 0, 0, 1]))
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("Invalid registered camera transform")
            return matrix

        from_saved = (homogeneous(parser["transform"]) @
                      np.linalg.inv(homogeneous(transforms.get("applied_transform", np.eye(4)))))
        scale = float(parser["scale"]) / float(transforms.get("applied_scale", 1.0))
        center = np.asarray(view["center"], dtype=float)
        up = np.array([0., 0., 1.]) if view["up_axis"] == "z" else np.array([0., 1., 0.])
        candidates = []
        for frame in transforms["frames"]:
            camera = from_saved @ homogeneous(frame["transform_matrix"])
            eye = camera[:3, 3] * scale
            forward = -camera[:3, 2]
            length = float(np.linalg.norm(forward))
            if not np.isfinite(eye).all() or length < 1e-6:
                continue
            forward /= length
            # Start from a level, registered capture near the scan center.
            if abs(float(forward @ up)) > 0.55:
                continue
            forward -= float(forward @ up) * up
            forward /= np.linalg.norm(forward)
            candidates.append((float(np.linalg.norm(eye - center)), eye, forward))
        if not candidates:
            return view
        _, eye, forward = min(candidates, key=lambda item: item[0])
        return {**view, "eye": eye.tolist(), "look_at": (eye + forward).tolist(),
                "start_view": "registered_capture"}
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError):
        return view


def dashboard_jobs(root: Path, project_id: str) -> list[dict]:
    reconstructions = root / "projects" / project_id / "reconstructions"
    jobs = []
    if not reconstructions.is_dir():
        return jobs
    for manifest in reconstructions.glob("*/job.json"):
        try:
            job_id = str(uuid.UUID(manifest.parent.name))
            job = json.loads(manifest.read_text(encoding="utf-8"))
            if job.get("job_id") != job_id or job.get("project_id") != project_id:
                continue
            verified = verified_job(root, project_id, job_id) if job.get("status") == "ready_for_blender" else None
            frame = _person_view(job, _ply_frame(verified[1]), manifest.parent) if verified else None
            jobs.append({"id": job_id, "session_id": job.get("session_id"),
                         "created_utc": job.get("created_utc"), "status": job.get("status") if verified or job.get("status") != "ready_for_blender" else "invalid_artifact",
                         "publication": job.get("publication", "automatic"),
                         "quality": job.get("quality"), "input_count": len(job.get("input_captures", [])),
                         "registered_views": job.get("registered_views"),
                         "vertex_count": job.get("artifact", {}).get("vertex_count") if verified else None,
                         "view": frame, "viewable": bool(verified and frame),
                         "url": f"/ui/jobs/{project_id}/{job_id}/splat.ply" if verified else None,
                         "alignment": "COLMAP frame; not aligned to the physical room"})
        except (OSError, ValueError, TypeError, KeyError, UnicodeDecodeError, struct.error):
            continue
    jobs.sort(key=lambda item: (item["created_utc"] or "", item["id"]), reverse=True)
    return jobs


def _session_photo_progress(root: Path, project_id: str, session_id: str,
                            seed_ids: set[str], accepted_ids: set[str]) -> dict | None:
    """Read current photo choices without treating a stored photo as trained."""
    database = root / "bridge.sqlite3"
    if not database.is_file():
        return None
    try:
        with sqlite3.connect(database) as db:
            table = db.execute("""SELECT 1 FROM sqlite_master
                WHERE type='table' AND name='capture_selections'""").fetchone()
            if table:
                rows = db.execute("""SELECT c.capture_id,COALESCE(s.included,1)
                    FROM captures c LEFT JOIN capture_selections s ON s.capture_id=c.capture_id
                    WHERE c.project_id=? AND c.session_id=?""", (project_id, session_id))
            else:
                rows = db.execute("""SELECT capture_id,1 FROM captures
                    WHERE project_id=? AND session_id=?""", (project_id, session_id))
            choices = {capture_id: bool(included) for capture_id, included in rows}
    except (OSError, sqlite3.DatabaseError):
        return None
    captured = set(choices)
    selected = {capture_id for capture_id, included in choices.items() if included}
    baseline = seed_ids & captured
    added = (accepted_ids - seed_ids) & captured
    return {"captured_count": len(captured), "selected_count": len(selected),
            "baseline_input_count": len(baseline), "added_count": len(added),
            "pending_selected_count": len(selected - baseline - added),
            "excluded_count": len(captured - selected),
            "excluded_after_admission_count": len((baseline | added) - selected),
            "rejected_count": None}


def dashboard_current_model(root: Path, project_id: str) -> dict | None:
    """One project-facing splat: newest verified online snapshot, else saved baseline."""
    try:
        uuid.UUID(project_id)
    except ValueError:
        return None
    base = root / "projects" / project_id / "online"
    candidates = []
    if base.is_dir() and not base.is_symlink():
        for folder in base.iterdir():
            try:
                session_id = str(uuid.UUID(folder.name))
                if folder.is_symlink():
                    continue
                manifest = folder / "status.json"
                ply = folder / "current.ply"
                status = json.loads(manifest.read_text(encoding="utf-8"))
                artifact = status.get("artifact")
                if (status.get("project_id") != project_id or status.get("session_id") != session_id
                        or not isinstance(artifact, dict) or ply.is_symlink()
                        or not ply.is_file() or ply.stat().st_size != artifact["size_bytes"]):
                    continue
                digest = hashlib.sha256()
                with ply.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != artifact["sha256"]:
                    continue
                display = status.get("display_artifact")
                if display:
                    if display.get("source_sha256") != artifact["sha256"]:
                        continue
                    view_ply = folder / "display.ply"
                    if (view_ply.is_symlink() or not view_ply.is_file() or
                            view_ply.stat().st_size != display["size_bytes"] or
                            hashlib.sha256(view_ply.read_bytes()).hexdigest() != display["sha256"]):
                        continue
                else:
                    display, view_ply = artifact, ply
                # The desktop can show all SH3 detail. Do not silently use the
                # lossy, smaller headset copy when the full revision exists.
                browser_ply = folder / 'model-revisions' / (artifact['sha256'] + '.ply')
                browser_receipt = browser_ply.with_suffix('.json')
                if browser_ply.is_file() and browser_receipt.is_file():
                    preserved = json.loads(browser_receipt.read_text())
                    if (browser_ply.is_symlink() or browser_receipt.is_symlink() or
                            any(preserved.get(key) != value for key,value in {
                                'project_id':project_id, 'session_id':session_id,
                                'sha256':artifact['sha256'], 'source_sha256':artifact['sha256'],
                                'size_bytes':artifact['size_bytes'], 'vertex_count':artifact['vertex_count']}.items()) or
                            browser_ply.stat().st_size != artifact['size_bytes'] or
                            Store._file_sha256(browser_ply) != artifact['sha256']):
                        continue
                    display, view_ply = artifact, browser_ply
                frame = _ply_frame(view_ply)
                if not frame:
                    continue
                seed_ids = set(status.get("seed_ids", []))
                represented_seed_ids = seed_ids
                accepted_ids = set(status.get("published_capture_ids", status.get("accepted_ids", [])))
                progress = _session_photo_progress(root, project_id, session_id, seed_ids, accepted_ids)
                evidence_id = status.get("registration", {}).get("evidence_job_id")
                evidence = verified_job(root, project_id, evidence_id) if isinstance(evidence_id, str) else None
                if (status.get('accepted_baseline') and evidence and
                        evidence[0].get('session_id') == session_id and
                        status.get('seed_sha256') == evidence[0].get('artifact', {}).get('sha256')):
                    # Strict live-reference cameras are a subset of the photos
                    # already represented in the explicitly accepted model.
                    # Do not confuse the 139 trusted references with 140
                    # trained views or with the 330 input originals.
                    source_ids = {row['capture_id'] for row in evidence[0]['input_captures']}
                    used_ids = set(evidence[0].get('used_capture_ids', []))
                    if used_ids <= source_ids and len(used_ids) == evidence[0].get('registered_views'):
                        represented_seed_ids = used_ids
                        progress = _session_photo_progress(root, project_id, session_id, source_ids, accepted_ids)
                        from .online import _selected_ids
                        progress['pending_selected_count'] = len(_selected_ids(Store(root), session_id) - used_ids - accepted_ids)
                if (evidence and evidence[0].get("session_id") == session_id and
                        status.get("seed_sha256") in {
                            evidence[0].get("artifact", {}).get("sha256"),
                            evidence[0].get("quest_preview", {}).get("sha256")}):
                    # Online exports omit the source PLY's vertical-axis
                    # comment. Use the verified seed's axis and capture pose;
                    # a guessed Y-up centroid points the browser at the floor.
                    seed_frame = _ply_frame(evidence[1])
                    if seed_frame:
                        frame["up_axis"] = seed_frame["up_axis"]
                        frame = _person_view(evidence[0], frame, evidence[1].parent)
                if progress is not None:
                    if "rejected_count" in status or "quarantined_count" in status:
                        progress["rejected_count"] = int(status.get("rejected_count", 0)) + int(status.get("quarantined_count", 0))
                    registered = None
                    if (evidence and evidence[0].get("session_id") == session_id and
                            status.get("seed_sha256") in {
                                evidence[0].get("artifact", {}).get("sha256"),
                                evidence[0].get("quest_preview", {}).get("sha256")}):
                        try:
                            registered = int(evidence[0]["registered_views"])
                        except (KeyError, TypeError, ValueError):
                            registered = None
                    if registered is not None and 0 <= registered <= progress["baseline_input_count"]:
                        progress["baseline_registered_count"] = registered
                        progress["baseline_unmatched_count"] = progress["baseline_input_count"] - registered
                    else:
                        progress["baseline_registered_count"] = None
                        progress["baseline_unmatched_count"] = None
                baseline_registered = (progress.get("baseline_registered_count")
                                       if progress else None)
                if baseline_registered is None:
                    baseline_registered = status.get("registration", {}).get("pose_fit", {}).get("matched_views")
                if not isinstance(baseline_registered, int) or baseline_registered < 0:
                    baseline_registered = 0
                baseline_registered = min(baseline_registered, len(represented_seed_ids))
                candidates.append((datetime.fromtimestamp(ply.stat().st_mtime, timezone.utc).isoformat(), {
                    "id": f"online:{session_id}:{display['sha256']}",
                    "session_id": session_id, "source": "online", "status": status.get("status"),
                    "source_sha256": artifact["sha256"],
                    "coordinate_frame": f"online:{session_id}:{status.get('seed_sha256', '')}",
                    "seed_job_id": evidence_id if evidence else None,
                    "accepted_views": baseline_registered + len(accepted_ids - represented_seed_ids),
                    "physical_acceptance": status.get("registration", {}).get("physical_acceptance", "pending"),
                    "photo_progress": progress,
                    "training_steps": status.get("training_steps", 0),
                    "worker_updated_utc": status.get("updated_utc"),
                    "model_updated_utc": datetime.fromtimestamp(ply.stat().st_mtime, timezone.utc).isoformat(),
                    "new_gaussians": status.get("model_metrics", {}).get("new_gaussians", 0),
                    "vertex_count": display["vertex_count"],
                    "filtered_out": display.get("removed_gaussians", 0), "view": frame,
                    "url": f"/ui/online/{project_id}/{session_id}/current.ply?revision={display['sha256']}",
                    "viewable": True,
                }))
            except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError, struct.error):
                continue
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    for job in dashboard_jobs(root, project_id):
        if job["viewable"] and job["publication"] != "offline_house_source":
            progress = _session_photo_progress(root, project_id, job["session_id"],
                                               set(job.get("input_capture_ids", [])), set())
            if progress is not None:
                progress["baseline_input_count"] = job["input_count"]
                progress["baseline_registered_count"] = job["registered_views"]
                progress["baseline_unmatched_count"] = (
                    job["input_count"] - job["registered_views"]
                    if isinstance(job["registered_views"], int) else None)
                progress["pending_selected_count"] = None  # No exact matched-ID map in an older job.
            return {**job, "source": "saved_baseline",
                    "accepted_views": job["registered_views"], "training_steps": None,
                    "photo_progress": progress,
                    "url": f"/ui/jobs/{project_id}/{job['id']}/splat.ply"}
    return None


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _entry_tree(entries: list[dict]) -> str:
    children: dict[str | None, list[dict]] = {}
    for entry in entries:
        children.setdefault(entry["parent_id"], []).append(entry)

    def branch(parent: str | None, seen: set[str]) -> str:
        rows = []
        for entry in children.get(parent, []):
            if entry["id"] in seen:
                continue
            kind = _text(entry["kind"].capitalize())
            label = _text(entry["name"])
            nested = branch(entry["id"], seen | {entry["id"]})
            rows.append(f"<li><span class='kind'>{kind}</span><span>{label}</span>{nested}</li>")
        return "<ul class='tree'>" + "".join(rows) + "</ul>" if rows else ""

    return branch(None, set()) or "<p class='muted'>No design categories yet.</p>"


def _photo_tile(project_id: str, record: dict, csrf_token: str | None = None, page: int = 1) -> str:
    capture_id = _text(record["capture_id"])
    photo_url = f"/ui/projects/{_text(project_id)}/captures/{capture_id}/image"
    selected = "Included" if record["included"] else "Excluded"
    toggle = "no" if record["included"] else "yes"
    action = "Exclude from future build" if record["included"] else "Include for future build"
    form = "" if csrf_token is None else (
        f"<form method='post' action='/ui/projects/{_text(project_id)}/sessions/{_text(record['session_id'])}/captures/{capture_id}/selection'>"
        f"<input type='hidden' name='csrf' value='{_text(csrf_token)}'>"
        f"<input type='hidden' name='included' value='{toggle}'>"
        f"<input type='hidden' name='page' value='{page}'>"
        f"<button type='submit'>{action}</button></form>")
    return (f"<article class='photo-card' id='photo-{capture_id}'>"
            f"<a href='{photo_url}' target='_blank' rel='noopener' aria-label='Open original camera photo'>"
            f"<img src='{photo_url}?thumbnail=true' loading='lazy' width='320' height='240' alt='Saved camera view'></a>"
            f"<div><strong>{selected}</strong> · {_text(record['camera_id'] or 'Camera')}</div>"
            f"<small>Captured {_text(record['captured_utc'] or 'unknown')}</small>"
            f"<small>PC received {_text(record['received_utc'])}</small>{form}</article>")


def render_capture_batch(project: dict, session: dict, records: list[dict], total: int,
                         page: int, csrf_token: str) -> str:
    project_id, session_id = project["id"], session["id"]
    pages = max(1, (total + CAPTURE_PAGE_SIZE - 1) // CAPTURE_PAGE_SIZE)
    route = f"/ui/projects/{_text(project_id)}/sessions/{_text(session_id)}/captures"
    prev_link = f"<a href='{route}?page={page - 1}'>Previous</a>" if page > 1 else ""
    next_link = f"<a href='{route}?page={page + 1}'>Next</a>" if page < pages else ""
    tiles = "".join(_photo_tile(project_id, item, csrf_token, page) for item in records)
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{_text(project['name'])} · Capture batch</title><style>
:root {{ color-scheme:dark; font-family:system-ui,Segoe UI,sans-serif; background:#101827; color:#f1f5f9; }}
* {{ box-sizing:border-box; }} body {{ margin:0; }} header {{ padding:24px clamp(20px,4vw,52px); border-bottom:1px solid #314056; }}
main {{ max-width:1500px; padding:26px clamp(20px,4vw,52px); margin:auto; }}
h1 {{ margin:12px 0 4px; }} a {{ color:#7dd3fc; }} .muted,small {{ color:#a8b6c9; }}
.notice {{ padding:12px 16px; border-left:3px solid #38bdf8; background:#19263a; line-height:1.5; }}
.photos {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; margin:22px 0; }}
.photo-card {{ min-width:0; background:#19263a; border:1px solid #314056; border-radius:12px; overflow:hidden; padding-bottom:14px; }}
.photo-card img {{ display:block; width:100%; height:220px; object-fit:contain; background:#080e19; }}
.photo-card div,.photo-card small {{ display:block; padding:5px 13px 0; overflow-wrap:anywhere; }}
.photo-card form {{ padding:9px 13px 0; }} button {{ padding:8px 11px; border:0; border-radius:7px; background:#253a52; color:#f1f5f9; cursor:pointer; }}
.pages {{ display:flex; gap:20px; align-items:center; }}
</style></head><body><header><a href='/ui?project_id={_text(project_id)}'>← {_text(project['name'])}</a></header>
<main><h1>Capture batch</h1><p class='muted'>Started {_text(session['created_utc'])} · {total} photos received on PC</p>
<p class='notice'>Photo selections never change or delete originals. The online worker skips excluded photos at ingress; the older offline build command still uses all received photos. Excluding a photo already trained into a model cannot undo its past updates.</p>
<nav class='pages' aria-label='Photo pages'>{prev_link}<span>Page {page} of {pages}</span>{next_link}</nav>
<section class='photos' aria-label='Saved camera photos'>{tiles or '<p>No photos in this batch yet.</p>'}</section>
<nav class='pages' aria-label='Photo pages'>{prev_link}<span>Page {page} of {pages}</span>{next_link}</nav>
</main></body></html>"""


def render_dashboard(projects: list[dict], selected: dict | None,
                     entries: list[dict], sessions: list[dict], jobs: list[dict],
                     csrf_token: str, policies: dict[str, dict] | None = None,
                     queue: list[dict] | None = None, models: list[dict] | None = None,
                     recent_captures: list[dict] | None = None,
                     current_model: dict | None = None, headset_notice: str = "",
                     *, builds_paused: bool = False, house_builds: list[dict] | None = None,
                     depth_updates: dict[str, dict] | None = None,
                     splat_updates: dict[str, dict] | None = None) -> str:
    policies = policies or {}
    queue = queue or []
    models = models or []
    recent_captures = recent_captures or []
    house_builds = house_builds or []
    depth_updates = depth_updates or {}
    splat_updates = splat_updates or {}
    processing_notice = ("Saved detailed builds are paused. Live splat and Depth have separate controls below. "
                         "Saved models remain viewable."
                         if builds_paused else "Build queue is available. Photo receiving does not mean the live model is training.")
    headset_forms = "".join(
        f"<form method='post' action='/ui/headset/{action}'>"
        f"<input type='hidden' name='csrf' value='{_text(csrf_token)}'>"
        + (f"<input type='hidden' name='project_id' value='{_text(selected['id'])}'>" if selected else "")
        + f"<button type='submit'>{label}</button></form>"
        for action, label in (("wake", "Wake headset"), ("keep-awake", "Keep awake for testing"),
                              ("wear-sensor", "Restore wear detection"))
    )
    project_cards = "".join(
        "<a class='project{}' href='/ui?project_id={}'><strong>{}</strong>"
        "<small>Created {}</small></a>".format(
            " active" if selected and project["id"] == selected["id"] else "",
            _text(project["id"]), _text(project["name"]),
            _text(project["created_utc"][:10]))
        for project in projects
    ) or "<p class='muted'>No projects yet. Create one below.</p>"
    def depth_label(session_id: str) -> str:
        depth = depth_updates.get(session_id)
        if not depth:
            return "Checking live observed Depth points…"
        state = str(depth.get("status", "unknown")).replace("_", " ")
        counts = (f"{depth.get('accepted_views', 0)} of {depth.get('total_views', 0)} photos admitted · "
                  f"{depth.get('pending_views', 0)} waiting · {depth.get('skipped_views', 0)} skipped · "
                  f"{depth.get('supported_samples', 0)} supported points")
        error = f" · {depth['error']}" if depth.get("error") else ""
        return f"{state}: {counts}{error}"

    session_rows = "".join(
        "<tr><td><a href='/ui/projects/{project}/sessions/{id}/captures'>View {count} photos</a>"
        "<br><code>{id}</code></td><td>{status}</td><td>{count}</td><td>{started}</td>"
        "<td data-depth-session='{id}'><span class='depth-summary' role='status'>{depth_label}</span>"
        "<form class='build depth-form' method='post' action='/ui/sessions/{id}/depth'>"
        "<input type='hidden' name='csrf' value='{csrf}'>"
        "<input class='depth-enabled' type='hidden' name='enabled' value='{depth_toggle}'>"
        "<button type='submit'>{depth_button}</button></form></td><td>"
        "<span class='splat-summary' data-splat-session='{id}' role='status'>{splat_label}</span>"
        "<form class='build splat-form' method='post' action='/ui/sessions/{id}/auto'>"
        "<input type='hidden' name='csrf' value='{csrf}'>"
        "<input class='splat-enabled' type='hidden' name='enabled' value='{toggle}'>"
        "{auto_review}"
        "<button type='submit'>{auto_label}</button></form>"
        "<form class='build' method='post' action='/ui/sessions/{id}/build'>"
        "<input type='hidden' name='csrf' value='{csrf}'>"
        "{review}"
        "<select name='quality' aria-label='Build quality'><option value='quick'>Quick splat</option><option value='final'>Detailed splat</option>"
        "<option value='house'>House: splat + depth-assisted textured mesh</option></select>"
        "<button type='submit'>Build</button></form></td></tr>".format(
            id=_text(session["id"]), status=_text(session["status"].capitalize()),
            project=_text(selected["id"] if selected else ""),
            count=_text(session["capture_count"]), started=_text(session["created_utc"][:16].replace("T", " ")),
            csrf=_text(csrf_token),
            depth_label=_text(depth_label(session["id"])),
            depth_toggle="no" if depth_updates.get(session["id"], {}).get("enabled") else "yes",
            depth_button="Pause live depth" if depth_updates.get(session["id"], {}).get("enabled") else "Start live depth",
            splat_label=_text("Live splat: " + str(splat_updates.get(session["id"], {}).get("status", "off")).replace("_", " ") + f" · {splat_updates.get(session['id'], {}).get('trained_views', 0)} / {session['capture_count']} photos"),
            toggle="no" if splat_updates.get(session["id"], {}).get("enabled") else "yes",
            auto_label="Pause live splat" if splat_updates.get(session["id"], {}).get("enabled") else "Start live splat",
            auto_review="" if policies.get(session["id"], {}).get("native_orientation_reviewed") else "<label><input type='checkbox' name='reviewed_native_orientation' value='yes' required> Camera orientation reviewed</label>",
            review="<span class='muted'>Camera orientation reviewed</span>" if policies.get(session["id"], {}).get("native_orientation_reviewed") else
                   "<label><input type='checkbox' name='reviewed_native_orientation' value='yes' required> Physical camera orientation reviewed</label>")
        for session in sessions
    ) or "<tr><td colspan='6' class='muted'>No capture sessions yet.</td></tr>"
    recent_tiles = "".join(_photo_tile(selected["id"], photo) for photo in recent_captures) if selected else ""
    job_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _text((job["created_utc"] or "")[:16].replace("T", " ")),
            _text(job["quality"] or "—"), _text(job["status"].replace("_", " ")),
            _text(f"{job['registered_views']} / {job['input_count']}" if job["registered_views"] is not None else job["input_count"]),
            _text(job["vertex_count"] if job["vertex_count"] is not None else "—"),
            ("<form method='post' action='/ui/projects/{}/jobs/{}/open-in-blender'>"
             "<input type='hidden' name='csrf' value='{}'>"
             "<button type='submit'>Open saved {} in Blender</button></form>".format(
                 _text(selected["id"]), _text(job["id"]), _text(csrf_token),
                 _text(job["quality"] or "build"))) if selected and job["viewable"] else "—")
        for job in jobs
    ) or "<tr><td colspan='6' class='muted'>No build jobs yet.</td></tr>"
    by_id = {entry["id"]: entry for entry in entries}

    def breadcrumb(entry_id: str) -> str:
        parts = []
        seen = set()
        while entry_id in by_id and entry_id not in seen:
            seen.add(entry_id)
            entry = by_id[entry_id]
            parts.append(entry["name"])
            entry_id = entry["parent_id"]
        return " › ".join(reversed(parts))

    model_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td><a class='refresh' href='/ui/models/{}/{}/artifact'>Download</a></td></tr>".format(
            _text(breadcrumb(model["entry_id"])), _text(model["display_name"] or model["format"].upper()),
            _text(model["created_utc"][:16].replace("T", " ")),
            _text(model["project_id"]), _text(model["revision_id"]))
        for model in models
    ) or "<tr><td colspan='4' class='muted'>No published model revisions yet.</td></tr>"
    if selected:
        count = sum(session["capture_count"] for session in sessions)
        house_rows = "".join(
            f"<li>{_text(build['input_count'])} photos · {_text(build['created_utc'][:16])} "
            f"<a href='/ui/projects/{_text(selected['id'])}/house-builds/{_text(build['job_id'])}/mesh.glb'>Download textured mesh</a>"
            " · Dimensions and coverage need review</li>" for build in house_builds)
        detail = f"""
          <section class='panel detail'>
            <div class='eyebrow'>Selected project</div>
            <h2>{_text(selected['name'])}</h2>
            <p id='processing-state' class='muted' role='status'>{processing_notice}</p>
            <div class='stats'><div><b>{count}</b><span>Images received</span></div>
              <div><b>{len(sessions)}</b><span>Capture sessions</span></div></div>
            <h3>Design library</h3>{_entry_tree(entries)}
            <div class='table-wrap'><table><thead><tr><th>Published under</th><th>Revision</th><th>Published (UTC)</th><th>File</th></tr></thead>
              <tbody>{model_rows}</tbody></table></div>
            <h3>Capture sessions</h3>
            <p class='muted'>Open a capture batch to see every saved camera photo. Originals remain on this PC.</p>
            <div class='recent-photos'>{recent_tiles or '<p class="muted">No camera photos received yet.</p>'}</div>
            <div class='table-wrap'><table><thead><tr><th>Session ID</th><th>Status</th>
              <th>Images</th><th>Started (UTC)</th><th>Live observed Depth points</th><th>Detailed Gaussian builds</th></tr></thead>
              <tbody>{session_rows}</tbody></table></div>
            <h3>Combined house build</h3>
            <p class='muted'>Capture interior and exterior in the same project and anchored session. Choose House in that batch's Build menu for a detailed splat and a textured mesh checked against Meta surface observations. It needs new depth-assisted photos from multiple positions. Unseen areas are not filled in.</p>
            <ul>{house_rows or '<li>No completed depth-assisted house build yet.</li>'}</ul>
            <h3>Gaussian splat viewer</h3>
            <p class='muted'>View the current model with its full detail on this PC. Compare it with the saved seed from the same camera viewpoint. Set before freezes the current revision for the next update. Saved final builds remain separate below; Open in Blender imports the current model.</p>
            <div class='viewer-toolbar'><span>Current model</span>
              <select id='splat-job' hidden aria-label='Current model'><option value=''>Waiting for a splat</option></select>
              <button id='reload-jobs' type='button'>Refresh</button>
              <button id='reset-view' type='button' aria-label='Return to scan camera'>Center view</button>
              <button id='freeze-before' type='button' title='Keep this model revision for comparison'>Set before</button>
              <button id='compare-models' type='button' disabled title='Waiting for a newer accepted model revision'>Show before</button>
              <form class='viewer-launch' method='post' action='/ui/projects/{_text(selected['id'])}/open-current-in-blender'>
                <input type='hidden' name='csrf' value='{_text(csrf_token)}'>
                <button type='submit'>Open current preview in Blender</button></form></div>
            <div id='splat-viewer' data-project-id='{_text(selected['id'])}' role='img' aria-label='Interactive Gaussian splat viewer'></div>
            <p id='photo-progress' class='muted' aria-live='polite'>Checking saved and admitted photo counts…</p>
            <a id='model-batch-link' hidden>View this model's source-photo batch</a>
            <p id='viewer-status' class='muted' aria-live='polite'>Waiting for the current project model. Hold right mouse to lock and hide the cursor while looking; release to restore it. Left drag shifts position; the wheel moves forward or back.</p>
            <details id='region-editor'><summary>Replace or protect an area</summary>
              <p class='muted'>Mark a box around an area and save it. Choose Protect this good area to keep it unchanged while scanning elsewhere. To repair an area, build a separate replacement from selected photos, inspect it, then lock and apply the draft.</p>
              <form id='region-form' data-csrf='{_text(csrf_token)}'>
                <label>Area name <input id='region-name' maxlength='100' required placeholder='e.g. Clear countertop'></label>
                <div class='viewer-toolbar'><label>Distance in front of view <input id='region-distance' type='number' min='0.01' max='10000' step='0.1' value='1'></label>
                  <button id='region-place' type='button'>Place box in front of view</button>
                  <button id='region-hide' type='button'>Hide box</button></div>
                <div class='region-coordinates'>
                  <label>Center X <input id='region-x' type='number' min='-10000' max='10000' step='0.01' value='0' required></label>
                  <label>Center Y <input id='region-y' type='number' min='-10000' max='10000' step='0.01' value='0' required></label>
                  <label>Center Z <input id='region-z' type='number' min='-10000' max='10000' step='0.01' value='0' required></label>
                  <label>Size X <input id='region-sx' type='number' min='0.01' max='10000' step='0.01' value='1' required></label>
                  <label>Size Y <input id='region-sy' type='number' min='0.01' max='10000' step='0.01' value='1' required></label>
                  <label>Size Z <input id='region-sz' type='number' min='0.01' max='10000' step='0.01' value='1' required></label>
                </div>
                <p class='muted'>Coordinates use the current model's frame; units may not be meters. The amber box is a volume, not an automatic object selection. Gaussians crossing its boundary are retained.</p>
                <h4>Replacement photos</h4><p class='muted'>Choose 2–128 overlapping views from this model's capture session for a replacement. Excluded photos cannot be used. After taking new photos, refresh this list and use them for the saved area below. Originals remain unchanged.</p>
                <div id='region-photos' class='region-photos'></div>
                <div class='viewer-toolbar'><button id='region-photos-prev' type='button'>Newer photos</button>
                  <span id='region-photo-page'></span><button id='region-photos-next' type='button'>Older photos</button>
                  <button id='region-photos-refresh' type='button'>Refresh photos</button>
                  <span id='region-photo-count'>0 photos selected</span></div>
                <button id='region-save' type='submit'>Save area</button>
              </form>
              <p id='region-status' class='muted' role='status' aria-live='polite'>Place a box on the current online model to begin. Nothing has been changed.</p>
              <div id='region-list'></div>
            </details>
            <details><summary>Past reconstruction attempts (diagnostics)</summary>
            <p class='muted'>These saved jobs are retained for diagnostics. They are not separate project models.</p>
            <p id='build-progress' class='muted' aria-live='polite'>Checking paused build queue…</p>
            <div class='table-wrap'><table><thead><tr><th>Started (UTC)</th><th>Quality</th><th>Status</th><th>Matched / received</th><th>Gaussians</th><th>Blender</th></tr></thead>
              <tbody id='build-job-rows'>{job_rows}</tbody></table></div></details>
          </section>"""
    else:
        detail = "<section class='panel detail'><h2>Choose a project</h2><p class='muted'>Your scans and designs will appear here.</p></section>"
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Quest Splat Bridge · Projects</title>
<style>
  :root {{ color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    background: #101827; color: #f1f5f9; }}
  * {{ box-sizing: border-box }} body {{ margin: 0; }}
  header {{ padding: 28px clamp(20px,4vw,52px); border-bottom: 1px solid #314056;
    display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }}
  h1 {{ font-size: 1.35rem; margin: 0; }} h2 {{ margin: 5px 0 18px; font-size: 1.75rem; }}
  h3 {{ margin: 30px 0 12px; font-size: 1.05rem; }}
  .eyebrow {{ color: #7dd3fc; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 700; }}
  .muted, small {{ color: #a8b6c9; }} .muted {{ line-height: 1.55; }}
  main {{ display: grid; grid-template-columns: minmax(260px,320px) minmax(0,1fr);
    gap: 22px; max-width: 1500px; margin: auto; padding: 26px clamp(20px,4vw,52px); }}
  .panel {{ background: #19263a; border: 1px solid #314056; border-radius: 16px; padding: 23px; }}
  .project {{ display: flex; flex-direction: column; gap: 5px; padding: 14px; margin-top: 9px;
    border: 1px solid #36475f; border-radius: 10px; color: inherit; text-decoration: none; }}
  .project:hover, .project.active {{ border-color: #38bdf8; background: #12354b; }}
  .project small {{ font-size: .78rem; }} form {{ display: flex; gap: 8px; margin-top: 14px; }}
  input {{ min-width: 0; flex: 1; padding: 11px; border: 1px solid #566b83; border-radius: 8px;
    background: #0d1726; color: inherit; font: inherit; }}
  button, .refresh {{ padding: 11px 14px; background: #0ea5e9; border: 0; border-radius: 8px;
    color: #061626; font: inherit; font-weight: 700; cursor: pointer; text-decoration: none; }}
  .refresh {{ background: #253a52; color: #e0f2fe; font-size: .85rem; }}
  .stats {{ display: flex; gap: 12px; flex-wrap: wrap; }} .stats div {{ min-width: 150px; padding: 15px;
    background: #112235; border-radius: 10px; display: flex; flex-direction: column; gap: 4px; }}
  .stats b {{ font-size: 1.8rem; }} .stats span {{ color: #a8b6c9; font-size: .8rem; }}
  .tree {{ list-style: none; margin: 0; padding-left: 16px; border-left: 1px solid #456079; }}
  .tree li {{ padding: 8px 0 2px 12px; }} .kind {{ color: #7dd3fc; font-size: .72rem; text-transform: uppercase;
    margin-right: 12px; }} .tree .tree {{ margin-top: 8px; }}
  .table-wrap {{ overflow-x: auto; }} table {{ width: 100%; border-collapse: collapse; text-align: left; }}
  th, td {{ padding: 12px 10px; border-bottom: 1px solid #35465b; white-space: nowrap; }}
  th {{ color: #a8b6c9; font-size: .76rem; text-transform: uppercase; }}
  code {{ font-size: .78rem; color: #c9e8f8; }} .note {{ margin-top: 28px; padding: 15px;
    border-left: 3px solid #38bdf8; background: #112235; color: #c9d6e4; line-height: 1.5; }}
  select {{ padding: 9px; background: #0d1726; color: #f1f5f9; border: 1px solid #566b83; border-radius: 8px; }}
  .build {{ margin: 0; align-items: center; flex-wrap: wrap; }} .build label {{ font-size: .75rem; max-width: 160px; white-space: normal; }}
  .build input {{ min-width: unset; padding: 0; }} .build button {{ padding: 7px 10px; }}
  .recent-photos {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }}
  .photo-card {{ min-width:0; border:1px solid #35465b; border-radius:9px; overflow:hidden; padding-bottom:7px; }}
  .photo-card img {{ display:block; width:100%; height:140px; object-fit:contain; background:#080e19; }}
  .photo-card div,.photo-card small {{ display:block; padding:3px 8px 0; overflow-wrap:anywhere; font-size:.74rem; }}
  .viewer-toolbar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }}
  .viewer-launch {{ margin: 0; }} .viewer-launch input {{ display: none; }}
  #splat-job {{ min-width: 260px; max-width: 100%; }} #splat-viewer {{ position: relative; width: 100%; height: min(62vh,640px); min-height: 340px; background: #070d17; border: 1px solid #456079; border-radius: 10px; overflow: hidden; }}
  #splat-viewer canvas {{ display: block; }}
  #region-editor {{ margin:24px 0; padding:16px; border:1px solid #566b83; border-radius:10px; }}
  #region-editor summary {{ cursor:pointer; font-weight:700; }}
  #region-form {{ display:block; }} #region-form label {{ display:flex; flex-direction:column; gap:5px; }}
  .region-coordinates {{ display:grid; grid-template-columns:repeat(3,minmax(80px,1fr)); gap:10px; }}
  .region-coordinates input {{ width:100%; }} #region-distance {{ max-width:160px; }}
  .region-photos {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:9px; max-height:400px; overflow:auto; }}
  .region-photos label {{ padding:6px; border:1px solid #456079; border-radius:6px; }}
  .region-photos img {{ width:100%; height:90px; object-fit:contain; background:#080e19; }}
  .region-photos input {{ flex:none; }} .region-card {{ padding:14px 0; border-top:1px solid #456079; }}
  .region-card label {{ display:block; margin:12px 0; }} .region-card input {{ margin-right:8px; }}
  .region-overlay {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
  button:disabled {{ opacity:.45; cursor:not-allowed; }}
  @media(max-width:760px) {{ main {{ grid-template-columns: 1fr; }} }}
</style><script type='module' src='/ui/static/viewer.js?v=preserved-sh3-20261013'></script></head><body>
<header><div><div class='eyebrow'>Local PC workspace</div><h1>Quest Splat Bridge</h1></div>
  <a class='refresh' href='/ui'>Refresh projects</a></header>
<main><aside class='panel'><div class='eyebrow'>Projects</div>{project_cards}
  <h3>New project</h3><form method='post' action='/ui/projects'>
    <input type='hidden' name='csrf' value='{_text(csrf_token)}'>
    <input name='name' maxlength='120' required placeholder='e.g. Home interior and exterior' aria-label='Project name'>
    <button type='submit'>Create</button></form>
  <section id='headset-controls'><h3>Connected headset</h3>
    <p class='muted'>USB debugging must be connected. Wake headset brings the screen back after sleep.</p>
    {headset_forms}
    <p class='muted'>Testing treats the headset as worn even when it is off your head. The power button can still put it to sleep.</p>
    <p role='status' aria-live='polite'>{_text(headset_notice)}</p>
  </section></aside>{detail}</main></body></html>"""
