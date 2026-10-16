"""Disk-backed reconstruction queue shared by the PC and headset listeners.

Only the loopback dashboard process consumes the queue. Both listeners can add
requests. The CLI/reconstruction runner owns the global GPU lock as well.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import utc_now
from .store import Store, StoreError


DEBOUNCE_SECONDS = 8
MINIMUM_QUICK = 3
MINIMUM_FINAL = 6


def _plus_seconds(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


@contextmanager
def _db(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "build-queue.sqlite3", timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS session_policy (
                session_id TEXT PRIMARY KEY, auto_enabled INTEGER NOT NULL,
                native_orientation_reviewed INTEGER NOT NULL, updated_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processing_control (
                id INTEGER PRIMARY KEY CHECK (id=1),
                paused INTEGER NOT NULL, updated_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, project_id TEXT NOT NULL,
                input_revision TEXT NOT NULL, input_count INTEGER NOT NULL,
                quality TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
                eligible_utc TEXT NOT NULL, requested_utc TEXT NOT NULL,
                started_utc TEXT, completed_utc TEXT, process_id INTEGER,
                output_job_id TEXT, error TEXT,
                UNIQUE(session_id, input_revision, quality)
            );
            CREATE INDEX IF NOT EXISTS requests_pending ON requests(status, eligible_utc);
            CREATE TABLE IF NOT EXISTS depth_updates (
                session_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                enabled INTEGER NOT NULL, input_revision TEXT,
                status_json TEXT NOT NULL, updated_utc TEXT NOT NULL,
                lease_token TEXT, lease_until TEXT, retry_after TEXT,
                control_revision TEXT
            );
            CREATE TABLE IF NOT EXISTS splat_updates (
                session_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                enabled INTEGER NOT NULL, control_revision TEXT NOT NULL,
                seed_job_id TEXT, status_json TEXT NOT NULL, input_revision TEXT,
                process_id INTEGER, started_utc TEXT, updated_utc TEXT NOT NULL,
                retry_after TEXT
            );
        """)
        if 'control_revision' not in {row[1] for row in db.execute('PRAGMA table_info(depth_updates)')}:
            db.execute('ALTER TABLE depth_updates ADD COLUMN control_revision TEXT')
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def policy(root: Path, session_id: str) -> dict:
    root = Path(root)
    session_id = str(uuid.UUID(session_id))
    with _db(root) as db:
        row = db.execute("SELECT * FROM session_policy WHERE session_id=?", (session_id,)).fetchone()
    if row:
        return {"session_id": session_id, "auto_enabled": bool(row["auto_enabled"]),
                "native_orientation_reviewed": bool(row["native_orientation_reviewed"]),
                "updated_utc": row["updated_utc"]}
    return {"session_id": session_id, "auto_enabled": False,
            "native_orientation_reviewed": False, "updated_utc": None}


def set_policy(root: Path, session_id: str, *, auto_enabled: bool,
               native_orientation_reviewed: bool) -> dict:
    root = Path(root)
    session_id = str(uuid.UUID(session_id))
    Store(root).session(session_id)
    if auto_enabled and not native_orientation_reviewed:
        raise ValueError("Auto preview requires reviewed camera image orientation")
    with _db(root) as db:
        db.execute("""INSERT INTO session_policy VALUES (?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET auto_enabled=excluded.auto_enabled,
                native_orientation_reviewed=excluded.native_orientation_reviewed,
                updated_utc=excluded.updated_utc""",
                   (session_id, int(auto_enabled), int(native_orientation_reviewed), utc_now()))
    return policy(root, session_id)


def processing_paused(root: Path) -> bool:
    with _db(Path(root)) as db:
        row = db.execute("SELECT paused FROM processing_control WHERE id=1").fetchone()
    return bool(row["paused"]) if row else False


def set_processing_paused(root: Path, paused: bool) -> dict:
    """Persist the operator's global build pause and cancel work not yet started."""
    with _db(Path(root)) as db:
        db.execute("BEGIN IMMEDIATE")
        now = utc_now()
        db.execute("""INSERT INTO processing_control (id,paused,updated_utc) VALUES (1,?,?)
            ON CONFLICT(id) DO UPDATE SET paused=excluded.paused,
                updated_utc=excluded.updated_utc""", (int(paused), now))
        cancelled = 0
        if paused:
            cancelled = db.execute("""UPDATE requests SET status='cancelled', completed_utc=?,
                error='Processing paused by operator' WHERE status='pending'""", (now,)).rowcount
    return {"paused": bool(paused), "cancelled_pending": cancelled}


def _revision(captures: list[dict]) -> str:
    lines = sorted(f"{item['capture_id']}:{item['image_sha256']}" for item in captures)
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def _existing_jobs(root: Path, project_id: str, session_id: str):
    reconstructions = root / "projects" / project_id / "reconstructions"
    if not reconstructions.is_dir():
        return
    import json
    for path in reconstructions.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("session_id") == session_id and job.get("project_id") == project_id:
                yield job
        except (OSError, ValueError):
            continue
    from .reconstruct import read_house_build, ReconstructionError
    for path in (root / "projects" / project_id / "house-builds").glob("*/house.json"):
        try:
            job, _ = read_house_build(Store(root), project_id, path.parent.name)
            if job["session_id"] == session_id:
                yield job
        except (OSError, ValueError, KeyError, ReconstructionError, StoreError):
            continue


def request_build(root: Path, session_id: str, *, reason: str = "preview",
                  quality: str = "quick") -> dict:
    """Queue a frozen capture revision; repeated failed/ready revisions are not retried."""
    root = Path(root).resolve()
    session_id = str(uuid.UUID(session_id))
    if quality not in {"quick", "final", "house"} or reason not in {"capture", "preview", "manual"}:
        raise ValueError("Invalid build quality or reason")
    if quality == "house" and reason != "manual":
        raise ValueError("House mesh builds are deliberate final operations")
    store = Store(root)
    session = store.session(session_id)
    captures = store.captures(session_id)
    if reason in {'capture', 'preview'} and quality == 'quick':
        with _db(root) as db:
            live = db.execute('SELECT enabled FROM splat_updates WHERE session_id=?', (session_id,)).fetchone()
        if live and live['enabled']:
            return {'status': 'live_splat', 'capture_count': len(captures)}
    minimum = MINIMUM_QUICK if quality == "quick" else MINIMUM_FINAL
    if len(captures) < minimum:
        return {"status": "need_more_views", "minimum": minimum, "capture_count": len(captures)}
    current_policy = policy(root, session_id)
    phone_exif = bool(captures) and all(c.get('metadata', {}).get('capture_source') == 'android_camera2'
                                      and c['metadata'].get('image_orientation') == 'exif' for c in captures)
    if not current_policy["native_orientation_reviewed"] and not phone_exif:
        return {"status": "orientation_review_needed", "capture_count": len(captures)}
    house_snapshot = None
    if quality == "house":
        from .reconstruct import house_capture_snapshot, house_input_revision
        from .mesh import verified_spatial_observations
        house_snapshot = house_capture_snapshot(store, session_id)
        verified_spatial_observations(store, session_id,
            [item["receipt"]["capture_id"] for item in house_snapshot["captures"]])
        captures = [item["receipt"] for item in house_snapshot["captures"]]
        revision = house_input_revision(captures)
    else:
        revision = _revision(captures)
    with _db(root) as db:
        db.execute("BEGIN IMMEDIATE")
        control = db.execute("SELECT paused FROM processing_control WHERE id=1").fetchone()
        if control and control["paused"]:
            return {"status": "paused", "capture_count": len(captures)}
        old = db.execute("SELECT * FROM requests WHERE session_id=? AND input_revision=? AND quality=?",
                         (session_id, revision, quality)).fetchone()
        if old:
            if old["status"] == "cancelled":
                now = utc_now()
                eligible = _plus_seconds(DEBOUNCE_SECONDS if reason == "capture" else 0)
                db.execute("""UPDATE requests SET status='pending', reason=?, eligible_utc=?,
                    requested_utc=?, started_utc=NULL, completed_utc=NULL, process_id=NULL,
                    error=NULL WHERE id=?""", (reason, eligible, now, old["id"]))
                return {"status": "pending", "request_id": old["id"],
                        "input_revision": revision, "capture_count": len(captures),
                        "eligible_utc": eligible}
            if old["status"] == "pending" and reason != "capture":
                db.execute("UPDATE requests SET eligible_utc=?, reason=? WHERE id=?",
                           (utc_now(), reason, old["id"]))
            return {"status": old["status"], "request_id": old["id"],
                    "input_revision": revision, "capture_count": len(captures),
                    "job_id": old["output_job_id"]}
        previous = db.execute("""SELECT MAX(input_count) AS count FROM requests WHERE session_id=?
            AND status IN ('running','succeeded','failed')""", (session_id,)).fetchone()["count"] or 0
        ready_job = None
        for job in _existing_jobs(root, session["project_id"], session_id):
            if job.get("status") != "ready_for_blender":
                continue
            previous = max(previous, len(job.get("input_captures", [])))
            if job.get("input_revision") == revision and job.get("quality") == quality:
                ready_job = job
        if ready_job:
            return {"status": "succeeded", "job_id": ready_job["job_id"],
                    "input_revision": revision, "capture_count": len(captures)}
        if reason == "capture" and previous and len(captures) < previous + 2:
            return {"status": "waiting_for_two_views", "capture_count": len(captures),
                    "next_capture_count": previous + 2}
        now = utc_now()
        # Only replace older automatic preview work. A manual frozen house/final
        # request must survive photos arriving while it waits for the GPU.
        if quality == "quick" and reason in {"capture", "preview"}:
            db.execute("UPDATE requests SET status='superseded', completed_utc=? "
                       "WHERE session_id=? AND status='pending' AND quality='quick' "
                       "AND reason IN ('capture','preview')", (now, session_id))
        request_id = str(uuid.uuid4())
        if house_snapshot is not None:
            import json
            destination = root / "build-queue-inputs" / f"{request_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                json.dump({"schema_version": 1, "session_id": session_id,
                           "project_id": session["project_id"], "input_revision": revision,
                           "capture_ids": [item["capture_id"] for item in captures]}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        eligible = _plus_seconds(DEBOUNCE_SECONDS if reason == "capture" else 0)
        db.execute("""INSERT INTO requests (id,session_id,project_id,input_revision,input_count,
            quality,reason,status,eligible_utc,requested_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
                   (request_id, session_id, session["project_id"], revision, len(captures),
                    quality, reason, "pending", eligible, now))
    return {"status": "pending", "request_id": request_id,
            "input_revision": revision, "capture_count": len(captures), "eligible_utc": eligible}


def on_capture_received(root: Path, session_id: str) -> dict | None:
    # The shared depth map is consumed by the same scheduler, independently of
    # deliberately paused expensive splat/mesh builds. Never train in upload.
    with _db(Path(root)) as db:
        db.execute("UPDATE depth_updates SET retry_after=NULL WHERE session_id=? AND enabled=1",
                   (session_id,))
        live = db.execute('SELECT enabled FROM splat_updates WHERE session_id=?', (session_id,)).fetchone()
    # A live model owns its ingress cursor. Never enqueue repeated full builds
    # for its new photographs, including when its worker is recovering.
    if live and live['enabled']:
        return {'status': 'live_splat', 'session_id': session_id}
    current = policy(root, session_id)
    if not current["auto_enabled"]:
        return None
    return request_build(root, session_id, reason="capture", quality="quick")


def splat_status(root: Path, session_id: str) -> dict:
    """Report the persistent Gaussian worker separately from observed Depth."""
    root = Path(root)
    session = Store(root).session(str(uuid.UUID(session_id)))
    with _db(root) as db:
        row = db.execute('SELECT * FROM splat_updates WHERE session_id=?', (session['id'],)).fetchone()
    operation = json.loads(row['status_json']) if row else {'status': 'off'}
    result = dict(operation)
    status_path = root/'projects'/session['project_id']/'online'/session['id']/'status.json'
    if row and status_path.is_file() and not status_path.is_symlink():
        try:
            model = json.loads(status_path.read_text(encoding='utf-8'))
            if model.get('session_id') == session['id'] and model.get('project_id') == session['project_id']:
                result.pop('error', None)
                result.update({key: model[key] for key in ('status', 'error', 'seed_view_count',
                    'selected_count', 'training_steps', 'model_metrics', 'revision_number',
                    'skipped', 'updated_utc', 'quarantined_count', 'pending_refinement',
                    'processed_count', 'visible_count', 'rejected_count') if key in model})
                if model.get('last_error') and model.get('status', '').startswith(('blocked', 'paused')):
                    result['error'] = model['last_error']
                published = model.get('published_capture_ids',
                                      (model.get('model_metrics') or {}).get('published_capture_ids', []))
                result['trained_views'] = (model.get('visible_count',
                    len(set(model.get('seed_ids', [])) | set(published))) if model.get('artifact') else 0)
                result['gaussian_count'] = (model.get('artifact') or {}).get('vertex_count', 0)
                run_started = row['started_utc'] or row['updated_utc']
                if (operation.get('status') in {'waiting_for_worker', 'starting', 'running'}
                        and str(model.get('updated_utc', '')) < run_started):
                    result['status'] = ('preparing_model' if operation.get('status') == 'running'
                                        else operation['status'])
                    result.pop('error', None)
        except (OSError, ValueError, TypeError):
            result['error'] = 'Unable to read the current splat progress; the last published model is retained.'
    result.update(session_id=session['id'], enabled=bool(row and row['enabled']),
                  total_views=session['capture_count'])
    result.setdefault('trained_views', 0)
    result.setdefault('gaussian_count', 0)
    if operation.get('status') in {'blocked', 'waiting_for_gpu', 'starting'}:
        if not str(result.get('status', '')).startswith(('blocked', 'paused')):
            result['status'] = operation['status']
        if operation.get('error') and not result.get('error'):
            result['error'] = operation['error']
    if row and not row['enabled']:
        result['status'] = 'paused'
        if result.get('error') == 'Scheduled Gaussian refinement is disabled':
            result.pop('error', None)
    return result


def set_splat_updates(root: Path, session_id: str, enabled: bool, *,
                      native_orientation_reviewed: bool = False, seed_job_id: str | None = None) -> dict:
    root = Path(root)
    session = Store(root).session(str(uuid.UUID(session_id)))
    if enabled and not (native_orientation_reviewed or policy(root, session['id'])['native_orientation_reviewed']):
        raise ValueError('Review camera image orientation before starting the live splat')
    if seed_job_id:
        seed_job_id = str(uuid.UUID(seed_job_id))
        from .preview import read_job
        seed, _ = read_job(Store(root), session['project_id'], seed_job_id)
        if seed['session_id'] != session['id']:
            raise ValueError('Seed belongs to another scan session')
    # Retain the existing orientation review but suppress the old repeated-build path.
    if enabled:
        set_policy(root, session['id'], auto_enabled=False, native_orientation_reviewed=True)
    now = utc_now()
    with _db(root) as db:
        db.execute('''INSERT INTO splat_updates
            (session_id,project_id,enabled,control_revision,seed_job_id,status_json,updated_utc)
            VALUES (?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET
            enabled=excluded.enabled,control_revision=excluded.control_revision,
            seed_job_id=COALESCE(excluded.seed_job_id,splat_updates.seed_job_id),
            status_json=excluded.status_json,input_revision=NULL,retry_after=NULL,updated_utc=excluded.updated_utc''',
            (session['id'], session['project_id'], int(enabled), str(uuid.uuid4()), seed_job_id,
             json.dumps({'status': 'waiting_for_worker' if enabled else 'paused'}), now))
        if enabled:
            db.execute("UPDATE requests SET status='superseded',completed_utc=? WHERE session_id=? "
                       "AND status='pending' AND quality='quick' AND reason IN ('capture','preview')",
                       (now, session['id']))
    return splat_status(root, session['id'])


def depth_status(root: Path, session_id: str) -> dict:
    session_id = str(uuid.UUID(session_id))
    with _db(Path(root)) as db:
        row = db.execute("SELECT * FROM depth_updates WHERE session_id=?", (session_id,)).fetchone()
    if row is None:
        return {"session_id": session_id, "enabled": False, "status": "off",
                "accepted_views": 0, "pending_views": 0, "skipped_views": 0}
    result = json.loads(row['status_json'])
    if not row['enabled']:
        result['status'] = 'paused'
        result.pop('error', None)
    return {**result, 'session_id': session_id, 'enabled': bool(row['enabled']),
            'updated_utc': row['updated_utc']}


def set_depth_updates(root: Path, session_id: str, enabled: bool) -> dict:
    """Opt one anchored session into observed depth updates, never full builds."""
    root = Path(root)
    session = Store(root).session(str(uuid.UUID(session_id)))
    with _db(root) as db:
        db.execute("""INSERT INTO depth_updates
            (session_id,project_id,enabled,status_json,updated_utc,control_revision) VALUES (?,?,?,?,?,?)
            ON CONFLICT(session_id) DO UPDATE SET enabled=excluded.enabled,
                input_revision=NULL, retry_after=NULL, updated_utc=excluded.updated_utc,
                control_revision=excluded.control_revision""",
            (session['id'], session['project_id'], int(enabled),
             json.dumps({'status': 'waiting', 'accepted_views': 0, 'pending_views': 0,
                         'skipped_views': 0}), utc_now(), str(uuid.uuid4())))
    return depth_status(root, session['id'])


def _depth_inventory(store, session_id):
    from .online import _selected_ids
    captures = store.captures(session_id)
    included = _selected_ids(store, session_id)
    revision = hashlib.sha256(json.dumps(
        {'captures': captures, 'included': sorted(included)}, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return captures, included, revision


class _DepthPublicationError(RuntimeError):
    def __init__(self, state):
        super().__init__('Depth map committed; preview publication is not verified')
        self.state = state


def _advance_depth(store, session_id, captures, included, *, batch_size=4, before_commit=None):
    from .mesh_spatial import read_session_spatial_map, update_session_spatial_map, verified_observations
    from .models import CaptureMetadata
    from .preview import read_depth_preview
    from .reconstruct import ReconstructionError
    session = store.session(session_id)
    path = store.root/'projects'/session['project_id']/'sessions'/session_id/'spatial-map'/'current.json'
    # A corrupt established source/map blocks publication; it must never be
    # silently discarded to manufacture a new apparent baseline.
    previous = read_session_spatial_map(store, session_id) if path.exists() else None
    accepted = set(previous['input_ledger']) if previous else set()
    selected = accepted & included
    candidates = [row for row in captures if row['capture_id'] in included - accepted]
    skipped, processed = [], 0
    for capture in candidates:
        if processed >= batch_size:
            break
        cid = capture['capture_id']
        try:
            metadata = CaptureMetadata.model_validate(capture['metadata'])
            if metadata.capture_source != 'quest_camera' or not metadata.tracking_valid:
                raise ValueError('Camera tracking is unavailable')
            if capture.get('spatial_availability') != 'ready' or not capture.get('spatial_hit_count'):
                skipped.append({'capture_id': cid, 'reason': 'No ready depth observations'})
                continue
            # Individual receipt/source validation precedes admission. Reuse the
            # ray validator; photo/COLMAP fits are not claimed by this map.
            evidence = verified_observations(store, session_id, [cid], require_surface_review=False)
            if not len(evidence['points']):
                skipped.append({'capture_id': cid, 'reason': 'No confident surface observations'})
                continue
            selected.add(cid)
            processed += 1
        except (OSError, ValueError, KeyError, TypeError, StoreError, ReconstructionError):
            skipped.append({'capture_id': cid, 'reason': 'Depth evidence could not be verified'})
    state = previous
    if selected != accepted or (previous is None and selected):
        kwargs = {}
        if previous:
            p = previous['parameters']
            kwargs = dict(voxel_size_m=p['voxel_size_m'], normal_agreement_deg=p['normal_agreement_deg'],
                          surface_consistency_m=p['surface_consistency_m'],
                          min_confidence=p['minimum_normal_confidence'],
                          minimum_baseline_m=p['minimum_view_baseline_m'])
        state = update_session_spatial_map(store, session_id, sorted(selected),
                                          before_commit=before_commit, **kwargs)
    if state and state['summary']['surface_sample_count']:
        try:
            manifest, _path = read_depth_preview(store, session['project_id'], session_id)
        except StoreError as error:
            raise _DepthPublicationError(state) from error
        revision_number = manifest.get('revision_number', 0)
    else:
        revision_number = state.get('revision_number', 0) if state else 0
    skipped_ids = {row['capture_id'] for row in skipped}
    pending = len(included - selected - skipped_ids)
    return {'status': 'updating' if pending else 'ready' if selected else 'waiting_for_depth',
            'accepted_views': len(selected), 'selected_views': len(included),
            'total_views': len(captures), 'pending_views': pending, 'skipped_views': len(skipped),
            'skipped': skipped[:16], 'supported_samples':
                state['summary']['surface_states'].get('supported', 0) if state else 0,
            'revision_number': revision_number,
            'state_revision': state['state_sha256'] if state else None,
            'protection': 'supported_surface_representatives_fixed',
            'source_kind': 'provisional_anchor_frame_depth'}


def request_status(root: Path, project_id: str) -> list[dict]:
    root = Path(root)
    with _db(root) as db:
        rows = db.execute("""SELECT session_id,input_count,quality,reason,status,requested_utc,
            started_utc,completed_utc,output_job_id,error FROM requests WHERE project_id=?
            AND status!='superseded' ORDER BY requested_utc DESC LIMIT 30""", (project_id,))
        return [dict(row) for row in rows]


def _pid_active(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        query = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(query, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _full_build_ready(db) -> bool:
    """Give already running or eligible explicit work the next GPU turn."""
    if db.execute("SELECT 1 FROM requests WHERE status='running' LIMIT 1").fetchone():
        return True
    control = db.execute("SELECT paused FROM processing_control WHERE id=1").fetchone()
    return not (control and control['paused']) and bool(db.execute(
        "SELECT 1 FROM requests WHERE status='pending' AND eligible_utc<=? LIMIT 1",
        (utc_now(),)).fetchone())


class Scheduler:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.processes: dict[str, subprocess.Popen] = {}
        self.splat_processes: dict[str, subprocess.Popen] = {}

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._loop, name="quest-splat-build-queue", daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:
                # Keep the queue available after transient filesystem/SQLite errors.
                logging.getLogger(__name__).exception('Reconstruction scheduler tick failed; retrying')
            self.stop_event.wait(2)

    def _finish(self, row: sqlite3.Row, exit_code: int | None):
        ready = next((job for job in _existing_jobs(self.root, row["project_id"], row["session_id"])
                      if job.get("status") == "ready_for_blender"
                      and job.get("input_revision") == row["input_revision"]
                      and job.get("quality") == row["quality"]), None)
        log = self.root / "build-queue-logs" / f"{row['id']}.log"
        try:
            tail = log.read_bytes()[-2048:].decode("utf-8", errors="replace")
        except OSError:
            tail = ""
        if ready:
            state, error = "succeeded", None
        elif "reconstruction worker is already active for this data root" in tail.lower():
            with _db(self.root) as db:
                db.execute("UPDATE requests SET status='pending', eligible_utc=?, started_utc=NULL, process_id=NULL WHERE id=?",
                           (_plus_seconds(12), row["id"]))
            return
        else:
            state = "failed"
            output = "depth-assisted textured mesh" if row["quality"] == "house" else "splat"
            error = f"Worker stopped without a verified {output}. See the local build log."
        with _db(self.root) as db:
            db.execute("""UPDATE requests SET status=?, completed_utc=?, process_id=NULL,
                output_job_id=?, error=? WHERE id=?""",
                       (state, utc_now(), ready["job_id"] if ready else None, error, row["id"]))

    def tick(self):
        self.tick_depth()
        splat_active = self.tick_splat()
        with _db(self.root) as db:
            running = db.execute("SELECT * FROM requests WHERE status='running' LIMIT 1").fetchone()
        if running:
            owned = self.processes.get(running["id"])
            if owned:
                code = owned.poll()
                if code is not None:
                    self.processes.pop(running["id"], None)
                    self._finish(running, code)
                return
            if _pid_active(running["process_id"]):
                return
            if running["process_id"] is None and running["started_utc"] > _plus_seconds(-60):
                return
            self._finish(running, None)
            return
        if splat_active:
            return  # Let the scheduled worker drain/yield before launching a competing GPU job.
        with _db(self.root) as db:
            db.execute("BEGIN IMMEDIATE")
            control = db.execute("SELECT paused FROM processing_control WHERE id=1").fetchone()
            if control and control["paused"]:
                return
            if db.execute("SELECT 1 FROM requests WHERE status='running' LIMIT 1").fetchone():
                return
            row = db.execute("SELECT * FROM requests WHERE status='pending' AND eligible_utc<=? "
                             "ORDER BY CASE WHEN quality='house' AND reason='manual' THEN 0 ELSE 1 END, requested_utc,id LIMIT 1",
                             (utc_now(),)).fetchone()
            if not row:
                return
            db.execute("UPDATE requests SET status='running', started_utc=? WHERE id=?", (utc_now(), row["id"]))
        helper = Path(__file__).resolve().parent.parent / "scripts" / "Build-QuestSplat.ps1"
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if not helper.is_file() or not powershell:
            with _db(self.root) as db:
                db.execute("UPDATE requests SET status='failed', completed_utc=?, error=? WHERE id=?",
                           (utc_now(), "Build helper or PowerShell is missing", row["id"]))
            return
        command = [powershell, "-NoProfile", "-NonInteractive", "-File", str(helper),
                   "-SessionId", row["session_id"], "-DataDir", str(self.root),
                   "-Quality", row["quality"]]
        if row["quality"] == "house":
            command += ["-CaptureSnapshot", str(self.root / "build-queue-inputs" / f"{row['id']}.json")]
        if policy(self.root, row["session_id"])["native_orientation_reviewed"]:
            command.append("-ReviewedNativeOrientation")
        log = self.root / "build-queue-logs" / f"{row['id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("ab") as output:
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                           stderr=subprocess.STDOUT, creationflags=flags, close_fds=True)
            self.processes[row["id"]] = process
            with _db(self.root) as db:
                db.execute("UPDATE requests SET process_id=? WHERE id=?", (process.pid, row["id"]))
        except OSError:
            with _db(self.root) as db:
                db.execute("UPDATE requests SET status='failed', completed_utc=?, error=? WHERE id=?",
                           (utc_now(), "Could not launch build helper", row["id"]))

    def tick_splat(self):
        """Resume one saved Gaussian model when its inventory/control changes.

        A clean idle worker exits with its exact drained inventory, releasing
        the shared GPU. It must not relaunch on every scheduler poll.
        """
        with _db(self.root) as db:
            rows = db.execute('SELECT * FROM splat_updates ORDER BY updated_utc,session_id').fetchall()
        for row in rows:
            owned = self.splat_processes.get(row['session_id'])
            if owned and owned.poll() is None:
                return True
            if not owned and _pid_active(row['process_id']):
                return True  # Includes graceful shutdown after a policy change.
            if owned or row['process_id']:
                self.splat_processes.pop(row['session_id'], None)
                operation = json.loads(row['status_json'])
                model = {}
                status_path = (self.root/'projects'/row['project_id']/'online'/row['session_id']/'status.json')
                try:
                    if not status_path.is_symlink():
                        model = json.loads(status_path.read_text(encoding='utf-8'))
                    if (model.get('project_id') != row['project_id'] or
                            model.get('session_id') != row['session_id']):
                        model = {}
                except (OSError, ValueError, TypeError):
                    model = {}
                idle = model.get('idle_yield') or {}
                drained = idle.get('input_revision')
                clean_idle = (model.get('worker_state') == 'idle' and
                    model.get('status') == 'waiting_for_new_views' and
                    isinstance(drained, str) and len(drained) == 64 and
                    all(char in '0123456789abcdef' for char in drained) and
                    # Windows venv's launcher and its base-Python child have
                    # different PIDs. The per-launch nonce binds this receipt;
                    # its process_id remains diagnostic, not worker identity.
                    isinstance(operation.get('run_id'), str) and
                    len(operation['run_id']) == 36 and
                    idle.get('run_id') == operation.get('run_id') and
                    idle.get('control_revision') == row['control_revision'] and
                    (owned is None or owned.poll() == 0))
                log = self.root/'build-queue-logs'/('live-splat-'+row['session_id']+'.log')
                try:
                    with log.open('rb') as handle:
                        handle.seek(max(0, int(operation.get('log_offset', 0))))
                        tail = handle.read()[-4096:].decode('utf-8', errors='replace')
                except OSError:
                    tail = ''
                current_error = (model.get('last_error', '') if
                    str(model.get('updated_utc', '')) >= (row['started_utc'] or '') else '')
                busy = 'reconstruction worker is already active' in (current_error or tail).lower()
                result = {'status': 'waiting_for_gpu' if busy else 'blocked',
                          'error': 'Waiting for the active reconstruction.' if busy else
                                   'The live splat worker stopped; see its local log and current model status.'}
                retry = _plus_seconds(15)
                if clean_idle:
                    result, retry = {'status':'waiting_for_new_views', 'worker_state':'idle'}, None
                elif model.get('status') == 'paused_scheduler':
                    result, retry = {'status':'waiting_for_worker' if row['enabled'] else 'paused'}, None
                if not row['enabled']:
                    result = {'status': 'paused'}
                with _db(self.root) as db:
                    # A concurrent pause/re-enable owns its new status. Clear
                    # only this dead process; never overwrite a newer command.
                    db.execute('''UPDATE splat_updates SET process_id=NULL,
                        retry_after=CASE WHEN control_revision=? THEN ? ELSE retry_after END,
                        status_json=CASE WHEN control_revision=? THEN ? ELSE status_json END,
                        input_revision=CASE WHEN control_revision=? AND ? THEN ? ELSE input_revision END,
                        updated_utc=CASE WHEN control_revision=? THEN ? ELSE updated_utc END
                        WHERE session_id=? AND process_id IS ?''',
                        (row['control_revision'], retry, row['control_revision'], json.dumps(result),
                         row['control_revision'], clean_idle, drained,
                         row['control_revision'], utc_now(), row['session_id'], row['process_id']))
        with _db(self.root) as db:
            if _full_build_ready(db):
                return False
            rows = db.execute('SELECT * FROM splat_updates ORDER BY updated_utc,session_id').fetchall()
        for row in rows:
            if not row['enabled'] or (row['retry_after'] and row['retry_after'] > utc_now()):
                continue
            store = Store(self.root)
            captures, included, revision = _depth_inventory(store, row['session_id'])
            if len(included) < 12 and not row['seed_job_id']:
                status = {'status': 'waiting_for_overlapping_views', 'minimum_initial_views': 12}
            elif (revision == row['input_revision'] and
                  json.loads(row['status_json']).get('status') in {'blocked', 'waiting_for_new_views'}):
                continue
            else:
                runtime = self.root/'tools'/'nerfstudio310-venv'/'Scripts'/'python.exe'
                if not runtime.is_file():
                    status = {'status': 'blocked', 'error': 'The installed Gaussian training runtime is unavailable.'}
                else:
                    now = utc_now()
                    run_id = str(uuid.uuid4())
                    with _db(self.root) as db:
                        db.execute('BEGIN IMMEDIATE')
                        if _full_build_ready(db) or db.execute('''SELECT 1 FROM splat_updates
                            WHERE process_id IS NOT NULL OR
                            (json_extract(status_json,'$.status')='starting' AND retry_after>?) LIMIT 1''',
                            (now,)).fetchone():
                            return False
                        claimed = db.execute('''UPDATE splat_updates SET started_utc=?,updated_utc=?,
                            input_revision=?,retry_after=?,status_json=? WHERE session_id=? AND enabled=1
                            AND control_revision=? AND process_id IS NULL
                            AND (retry_after IS NULL OR retry_after<=?)''',
                            (now, now, revision, _plus_seconds(60), json.dumps({'status':'starting','run_id':run_id}),
                             row['session_id'], row['control_revision'], now)).rowcount
                    if not claimed:
                        continue
                    command = [str(runtime), '-u', '-m', 'bridge.online', '--data-dir', str(self.root),
                               '--session-id', row['session_id'], '--run', '--scheduled',
                               '--control-revision', row['control_revision'], '--run-id', run_id]
                    if row['seed_job_id']:
                        command += ['--seed-job', row['seed_job_id']]
                    else:
                        command += ['--bootstrap', '--reviewed-native-orientation']
                    repo = Path(__file__).resolve().parent.parent
                    colmap = self.root/'tools'/'colmap-4.2.0'/'bin'
                    env = dict(os.environ)
                    env['QUEST_SPLAT_COLMAP_EXE'] = str(colmap/'colmap.exe')
                    env['PATH'] = os.pathsep.join(map(str, [repo/'tools'/'colmap-compat', colmap, runtime.parent])) + os.pathsep + env.get('PATH','')
                    env['PYTHONUTF8'] = '1'
                    env['PYTHONIOENCODING'] = 'utf-8'
                    log = self.root/'build-queue-logs'/('live-splat-'+row['session_id']+'.log')
                    log.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with log.open('ab') as output:
                            log_offset = output.tell()
                            process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                stdout=output, stderr=subprocess.STDOUT, close_fds=True,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        self.splat_processes[row['session_id']] = process
                        with _db(self.root) as db:
                            db.execute('''UPDATE splat_updates SET process_id=?,
                                status_json=CASE WHEN control_revision=? THEN ? ELSE status_json END
                                WHERE session_id=? AND started_utc=? AND process_id IS NULL''',
                                (process.pid,row['control_revision'],json.dumps({'status':'running',
                                 'run_id':run_id,'log_offset':log_offset}),row['session_id'],now))
                        return True
                    except OSError:
                        status = {'status':'blocked','error':'Could not launch the Gaussian worker; see the local build log.'}
            with _db(self.root) as db:
                db.execute('UPDATE splat_updates SET status_json=?,input_revision=?,updated_utc=? '
                           'WHERE session_id=? AND control_revision=?',
                           (json.dumps(status),revision,utc_now(),row['session_id'],row['control_revision']))
        return False

    def tick_depth(self):
        """Reconcile durable captures after upload, restart, or selection changes.

        One record per session replaces repeated build requests. A lease avoids
        duplicate consumers; the map itself also has its existing process lock.
        """
        with _db(self.root) as db:
            sessions = db.execute("SELECT session_id FROM depth_updates WHERE enabled=1 "
                "AND (retry_after IS NULL OR retry_after<=?) ORDER BY updated_utc", (utc_now(),)).fetchall()
        store = Store(self.root)
        for item in sessions:
            session_id, lease = item['session_id'], str(uuid.uuid4())
            with _db(self.root) as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT * FROM depth_updates WHERE session_id=?', (session_id,)).fetchone()
                if not row['enabled'] or (row['lease_until'] and row['lease_until'] > utc_now()):
                    continue
                db.execute('UPDATE depth_updates SET lease_token=?,lease_until=? WHERE session_id=?',
                           (lease, _plus_seconds(120), session_id))
            revision, state, retry = row['input_revision'], None, None
            try:
                captures, included, inventory_revision = _depth_inventory(store, session_id)
                if revision == inventory_revision and row['updated_utc'] > _plus_seconds(-30):
                    continue
                def check_current_selection():
                    from .online import _selected_ids
                    from .reconstruct import ReconstructionError
                    with _db(self.root) as guard_db:
                        control = guard_db.execute('SELECT enabled,control_revision,lease_token '
                            'FROM depth_updates WHERE session_id=?', (session_id,)).fetchone()
                    if (not control['enabled'] or control['control_revision'] != row['control_revision']
                            or control['lease_token'] != lease):
                        raise ReconstructionError('Live depth controls changed before publication')
                    # Newly arriving photos wait for the next delta. Changing
                    # membership of photos already considered cancels this one.
                    considered = {capture['capture_id'] for capture in captures}
                    if (_selected_ids(store, session_id) & considered) != included:
                        raise ReconstructionError('Capture selection changed before publication')
                state = _advance_depth(store, session_id, captures, included,
                                       before_commit=check_current_selection)
                if state['pending_views'] == 0:
                    revision = inventory_revision
            except _DepthPublicationError as error:
                logging.getLogger(__name__).warning('Live Depth preview publication failed for session %s', session_id)
                state = {**json.loads(row['status_json']), 'status': 'publication_blocked',
                         'committed_map_views': error.state['summary']['capture_count'],
                         'error': 'Depth observations saved; new preview unavailable. Last published counts retained'}
                retry = _plus_seconds(10)
            except Exception as error:
                # Keep both the previous map and its last known counts. Retry
                # transient I/O/locks; never swallow the failure in the UI.
                state = {**json.loads(row['status_json']), 'status': 'blocked',
                         'error': 'Depth evidence could not be verified; last published preview retained'}
                logging.getLogger(__name__).warning('Live Depth update blocked for session %s: %s',
                                                   session_id, str(error)[:512])
                retry = _plus_seconds(10)
            finally:
                with _db(self.root) as db:
                    if state is not None:
                        db.execute('UPDATE depth_updates SET input_revision=?,status_json=?,updated_utc=?, '
                                   'retry_after=? WHERE session_id=? AND lease_token=? '
                                   'AND control_revision IS ?',
                                   (revision, json.dumps(state), utc_now(), retry, session_id, lease,
                                    row['control_revision']))
                    db.execute('UPDATE depth_updates SET lease_token=NULL,lease_until=NULL '
                               'WHERE session_id=? AND lease_token=?', (session_id, lease))
