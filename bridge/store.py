from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from .models import SpatialEvidence, utc_now

MAX_SPATIAL_BYTES = 2 * 1024 * 1024


class StoreError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "bridge.sqlite3"
        self.lock = threading.RLock()
        with self.connection() as db:
            db.executescript("""
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    parent_id TEXT REFERENCES entries(id), kind TEXT NOT NULL,
                    name TEXT NOT NULL, sort_order INTEGER NOT NULL,
                    protected INTEGER NOT NULL,
                    created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    tracking_origin_id TEXT NOT NULL, status TEXT NOT NULL,
                    created_utc TEXT NOT NULL, finished_utc TEXT
                );
                CREATE TABLE IF NOT EXISTS captures (
                    capture_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    image_sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL,
                    image_path TEXT NOT NULL, received_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS captures_session ON captures(session_id);
                CREATE TABLE IF NOT EXISTS capture_spatial_evidence (
                    capture_id TEXT PRIMARY KEY REFERENCES captures(capture_id),
                    sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    source TEXT NOT NULL, availability TEXT NOT NULL,
                    ray_count INTEGER NOT NULL, hit_count INTEGER NOT NULL,
                    evidence_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS creation_requests (
                    request_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL, result_id TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def row(row):
        return dict(row) if row else None

    def project(self, project_id: str):
        with self.connection() as db:
            result = self.row(db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone())
        if not result:
            raise StoreError(404, "Project not found")
        return result

    @staticmethod
    def _creation_request(db, request_id, kind, payload):
        if request_id is None:
            return None
        request_id = str(uuid.UUID(request_id))
        row = db.execute("SELECT * FROM creation_requests WHERE request_id=?", (request_id,)).fetchone()
        if row:
            if row["kind"] != kind or row["payload_json"] != json.dumps(payload, sort_keys=True):
                raise StoreError(409, "Creation request was already used for different data")
            return row["result_id"]
        return None

    @staticmethod
    def _remember_creation(db, request_id, kind, payload, result_id):
        if request_id is not None:
            db.execute("INSERT INTO creation_requests VALUES (?,?,?,?)",
                       (str(uuid.UUID(request_id)), kind, json.dumps(payload, sort_keys=True), result_id))

    def create_project(self, name: str, request_id: str | None = None):
        project_id, original_id, now = str(uuid.uuid4()), str(uuid.uuid4()), utc_now()
        with self.lock, self.connection() as db:
            # USB and Wi-Fi use separate receiver processes. Serialize retry
            # lookup and creation in the database, not just this process lock.
            db.execute("BEGIN IMMEDIATE")
            existing = self._creation_request(db, request_id, "project", {"name": name})
            if existing:
                return self.row(db.execute("SELECT * FROM projects WHERE id=?", (existing,)).fetchone())
            db.execute("INSERT INTO projects VALUES (?,?,?,?)", (project_id, name, now, now))
            db.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?)",
                       (original_id, project_id, None, "category", "Original", 0, 1, now, now))
            self._remember_creation(db, request_id, "project", {"name": name}, project_id)
        return self.project(project_id)

    def projects(self):
        with self.connection() as db:
            return [self.row(row) for row in db.execute("SELECT * FROM projects ORDER BY created_utc, id")]

    def entries(self, project_id: str):
        self.project(project_id)
        with self.connection() as db:
            rows = db.execute("SELECT * FROM entries WHERE project_id=? ORDER BY sort_order, created_utc, id", (project_id,))
            return [self._entry(row) for row in rows]

    @staticmethod
    def _entry(row):
        value = dict(row)
        value["protected"] = bool(value["protected"])
        return value

    def _parent(self, db, project_id: str, kind: str, parent_id: str | None):
        parent = db.execute("SELECT * FROM entries WHERE id=?", (parent_id,)).fetchone() if parent_id else None
        if parent_id and (not parent or parent["project_id"] != project_id):
            raise StoreError(422, "Parent must belong to this project")
        allowed = {"category": {None}, "design": {"category"}, "variation": {"design", "variation"}}
        if (parent["kind"] if parent else None) not in allowed[kind]:
            raise StoreError(422, "Invalid parent kind")

    def create_entry(self, project_id: str, name: str, kind: str, parent_id: str | None, sort_order: int):
        self.project(project_id)
        entry_id, now = str(uuid.uuid4()), utc_now()
        with self.lock, self.connection() as db:
            self._parent(db, project_id, kind, parent_id)
            db.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?)",
                       (entry_id, project_id, parent_id, kind, name, sort_order, 0, now, now))
            return self._entry(db.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone())

    def patch_entry(self, project_id: str, entry_id: str, changes: dict):
        self.project(project_id)
        with self.lock, self.connection() as db:
            row = db.execute("SELECT * FROM entries WHERE id=? AND project_id=?", (entry_id, project_id)).fetchone()
            if not row:
                raise StoreError(404, "Entry not found")
            if row["protected"]:
                if "name" in changes or "parent_id" in changes:
                    raise StoreError(409, "Original category cannot be renamed or moved")
            name = changes.get("name", row["name"])
            parent_id = changes.get("parent_id", row["parent_id"])
            sort_order = changes.get("sort_order", row["sort_order"])
            self._parent(db, project_id, row["kind"], parent_id)
            cursor = parent_id
            while cursor:
                if cursor == entry_id:
                    raise StoreError(422, "Move would create a cycle")
                ancestor = db.execute("SELECT parent_id FROM entries WHERE id=?", (cursor,)).fetchone()
                cursor = ancestor["parent_id"] if ancestor else None
            now = utc_now()
            db.execute("UPDATE entries SET name=?,parent_id=?,sort_order=?,updated_utc=? WHERE id=?",
                       (name, parent_id, sort_order, now, entry_id))
            return self._entry(db.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone())

    def session(self, session_id: str):
        with self.connection() as db:
            row = db.execute("""SELECT s.*, (SELECT count(*) FROM captures c WHERE c.session_id=s.id) AS capture_count
                FROM sessions s WHERE s.id=?""", (session_id,)).fetchone()
        if not row:
            raise StoreError(404, "Session not found")
        return self.row(row)

    def create_session(self, project_id: str, tracking_origin_id: str, request_id: str | None = None):
        self.project(project_id)
        session_id, now = str(uuid.uuid4()), utc_now()
        with self.lock, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            payload = {"project_id": project_id, "tracking_origin_id": tracking_origin_id}
            existing = self._creation_request(db, request_id, "session", payload)
            if existing:
                session_id = existing
            else:
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                           (session_id, project_id, tracking_origin_id, "capturing", now, None))
                self._remember_creation(db, request_id, "session", payload, session_id)
        return self.session(session_id)

    def sessions(self, project_id: str):
        self.project(project_id)
        with self.connection() as db:
            rows = db.execute("""SELECT s.*, (SELECT count(*) FROM captures c WHERE c.session_id=s.id) AS capture_count
                FROM sessions s WHERE project_id=? ORDER BY created_utc, id""", (project_id,))
            return [self.row(row) for row in rows]

    def finish_session(self, session_id: str):
        self.session(session_id)
        with self.lock, self.connection() as db:
            db.execute("UPDATE sessions SET status='finished', finished_utc=COALESCE(finished_utc, ?) WHERE id=?",
                       (utc_now(), session_id))
        return self.session(session_id)

    def captures(self, session_id: str):
        self.session(session_id)
        with self.connection() as db:
            rows = db.execute("SELECT * FROM captures WHERE session_id=? ORDER BY received_utc, capture_id", (session_id,))
            result = []
            for row in rows:
                capture = self._capture(row, duplicate=False, include_metadata=True)
                spatial = db.execute("SELECT * FROM capture_spatial_evidence WHERE capture_id=?",
                                     (row["capture_id"],)).fetchone()
                if spatial:
                    capture.update(self._spatial_receipt(spatial))
                result.append(capture)
            return result

    def spatial_evidence(self, capture_id: str) -> dict | None:
        """Return a verified local worker path; never expose it through headset routes."""
        with self.connection() as db:
            row = db.execute("SELECT * FROM capture_spatial_evidence WHERE capture_id=?",
                             (capture_id,)).fetchone()
        if row is None:
            return None
        return {**self._spatial_receipt(row),
                "spatial_path": str(self._verified_spatial_path(row))}

    def export_session(self, session_id: str, capture_ids=None):
        session = self.session(session_id)
        with self.connection() as db:
            rows = list(db.execute("SELECT * FROM captures WHERE session_id=? ORDER BY received_utc, capture_id", (session_id,)))
        if capture_ids is not None:
            selected = {str(uuid.UUID(str(value))) for value in capture_ids}
            if selected - {row['capture_id'] for row in rows}:
                raise StoreError(422, "Selected capture is outside this session")
            rows = [row for row in rows if row['capture_id'] in selected]
        exported = []
        for row in rows:
            path = (self.root / row["image_path"]).resolve()
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
                raise StoreError(500, f"Capture {row['capture_id']} image missing or corrupted")
            spatial = self.spatial_evidence(row["capture_id"])
            exported.append({"receipt": {**self._capture(row, duplicate=False),
                                           **({k: v for k, v in spatial.items() if k != "spatial_path"}
                                              if spatial else {})},
                             "metadata": json.loads(row["metadata_json"]),
                             "image_path": str(path),
                             **({"spatial_path": spatial["spatial_path"]} if spatial else {})})
        return {"schema_version": 1, "session": session, "captures": exported}

    def export_pending_captures(self, session_id: str, seen_ids: set[str],
                                selected_ids: set[str], limit: int) -> list[dict]:
        """Hash only newly selected originals for a bounded online ingress tick."""
        self.session(session_id)
        if limit < 1 or limit > 16:
            raise ValueError("Pending capture limit is outside the worker bound")
        with self.connection() as db:
            rows = [row for row in db.execute(
                "SELECT * FROM captures WHERE session_id=? ORDER BY received_utc, capture_id",
                (session_id,)) if row["capture_id"] in selected_ids
                and row["capture_id"] not in seen_ids][:limit]
        exported = []
        root = self.root.resolve()
        for row in rows:
            path = self.root / row["image_path"]
            if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file() or \
                    self._file_sha256(path) != row["image_sha256"]:
                raise StoreError(500, f"Capture {row['capture_id']} image missing or corrupted")
            spatial = self.spatial_evidence(row["capture_id"])
            exported.append({"receipt": {**self._capture(row, duplicate=False),
                                           **({k: v for k, v in spatial.items() if k != "spatial_path"}
                                              if spatial else {})},
                             "metadata": json.loads(row["metadata_json"]),
                             "image_path": str(path.resolve()),
                             **({"spatial_path": spatial["spatial_path"]} if spatial else {})})
        return exported

    @staticmethod
    def _capture(row, duplicate: bool, include_metadata: bool = False):
        value = {"capture_id": row["capture_id"], "project_id": row["project_id"],
                 "session_id": row["session_id"], "image_sha256": row["image_sha256"],
                 "received_utc": row["received_utc"], "status": "received", "duplicate": duplicate}
        if include_metadata:
            value["metadata"] = json.loads(row["metadata_json"])
        return value

    @staticmethod
    def _validated_spatial(metadata: dict, spatial_json: bytes | None):
        declared = metadata.get("spatial_evidence_sha256")
        if (declared is None) != (spatial_json is None):
            raise StoreError(422, "Spatial evidence hash and sidecar must be supplied together")
        if spatial_json is None:
            return None
        if len(spatial_json) > MAX_SPATIAL_BYTES:
            raise StoreError(413, "Spatial evidence exceeds 2 MiB")
        if hashlib.sha256(spatial_json).hexdigest() != declared:
            raise StoreError(422, "Spatial evidence SHA-256 mismatch")
        try:
            def unique_keys(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate spatial evidence key: " + key)
                    result[key] = value
                return result

            def invalid_constant(value):
                raise ValueError("Non-finite spatial JSON number: " + value)

            data = json.loads(spatial_json.decode("utf-8"), object_pairs_hook=unique_keys,
                              parse_constant=invalid_constant)
            evidence = SpatialEvidence.model_validate(data)
        except (ValidationError, ValueError, UnicodeDecodeError) as error:
            raise StoreError(422, "Invalid spatial evidence: " + str(error)) from error
        for field in ("capture_id", "project_id", "session_id", "tracking_origin_id"):
            if str(getattr(evidence, field)) != metadata.get(field):
                raise StoreError(422, f"Spatial evidence {field} does not match capture")
        if (evidence.rgb_timestamp_raw != metadata.get("image_timestamp_raw") or
                evidence.rgb_timestamp_kind != metadata.get("image_timestamp_kind")):
            raise StoreError(422, "Spatial evidence RGB timestamp does not match capture")
        synthetic = metadata.get("capture_source") == "synthetic_test"
        if synthetic != (evidence.source == "synthetic_test"):
            raise StoreError(422, "Spatial evidence source does not match capture source")
        return evidence

    @staticmethod
    def _spatial_receipt(row) -> dict:
        return {"spatial_evidence_sha256": row["sha256"],
                "spatial_evidence_size_bytes": row["size_bytes"],
                "spatial_ray_count": row["ray_count"],
                "spatial_hit_count": row["hit_count"],
                "spatial_availability": row["availability"]}

    def _verified_spatial_path(self, row) -> Path:
        path = self.root / row["evidence_path"]
        if (path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve())
                or not path.is_file() or self._file_sha256(path) != row["sha256"]
                or path.stat().st_size != row["size_bytes"]):
            raise StoreError(500, "Previously received spatial evidence is missing or corrupted")
        return path.resolve()

    def save_capture(self, session_id: str, metadata: dict, image: bytes,
                     spatial_json: bytes | None = None):
        capture_id = metadata["capture_id"]
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(image).hexdigest() != metadata["image_sha256"]:
            raise StoreError(422, "Image SHA-256 mismatch")
        evidence = self._validated_spatial(metadata, spatial_json)
        with self.lock, self.connection() as db:
            # The thread lock does not cover the USB and LAN bridge processes.
            # Claim the SQLite write lock before checking for an existing ID.
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM captures WHERE capture_id=?", (capture_id,)).fetchone()
            if existing:
                if existing["session_id"] != session_id or existing["metadata_json"] != canonical or existing["image_sha256"] != metadata["image_sha256"]:
                    raise StoreError(409, "Capture ID already has different content")
                image_path = self.root / existing["image_path"]
                if (image_path.is_symlink() or not image_path.resolve().is_relative_to(self.root.resolve())
                        or not image_path.is_file() or self._file_sha256(image_path) != existing["image_sha256"]):
                    raise StoreError(500, "Previously received image is missing or corrupted")
                spatial_row = db.execute("SELECT * FROM capture_spatial_evidence WHERE capture_id=?",
                                         (capture_id,)).fetchone()
                if (evidence is None) != (spatial_row is None):
                    raise StoreError(409, "Capture ID already has different spatial evidence")
                if spatial_row is not None:
                    if spatial_row["sha256"] != metadata["spatial_evidence_sha256"]:
                        raise StoreError(409, "Capture ID already has different spatial evidence")
                    self._verified_spatial_path(spatial_row)
                return {**self._capture(existing, duplicate=True), "durable": True,
                        **(self._spatial_receipt(spatial_row) if spatial_row else {})}
            session = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not session:
                raise StoreError(404, "Session not found")
            if session["project_id"] != metadata["project_id"] or session["tracking_origin_id"] != metadata["tracking_origin_id"] or metadata["session_id"] != session_id:
                raise StoreError(422, "Capture project, session or tracking origin mismatch")
            if session["status"] != "capturing":
                raise StoreError(409, "Session has finished")
            extension = "jpg" if metadata["image_encoding"] == "jpeg" else "png"
            relative = Path("projects") / session["project_id"] / "captures" / f"{capture_id}.{extension}"
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if (destination.parent.is_symlink() or not destination.parent.resolve().is_relative_to(self.root.resolve())
                    or destination.is_symlink()):
                raise StoreError(500, "Capture storage path is invalid")
            temporary = destination.with_suffix(destination.suffix + f".{uuid.uuid4().hex}.tmp")
            spatial_relative = Path("projects") / session["project_id"] / "captures" / f"{capture_id}.spatial.json"
            spatial_destination = self.root / spatial_relative if evidence else None
            spatial_temporary = (spatial_destination.with_suffix(
                spatial_destination.suffix + f".{uuid.uuid4().hex}.tmp") if spatial_destination else None)
            try:
                if spatial_destination is not None:
                    if spatial_destination.is_symlink():
                        raise StoreError(500, "Spatial storage path is invalid")
                    if spatial_destination.exists():
                        if (self._file_sha256(spatial_destination) != metadata["spatial_evidence_sha256"]
                                or spatial_destination.stat().st_size != len(spatial_json)):
                            raise StoreError(409, "Unrecorded spatial evidence already has different content")
                    else:
                        with spatial_temporary.open("xb") as handle:
                            handle.write(spatial_json)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(spatial_temporary, spatial_destination)
                        self._sync_directory(spatial_destination.parent)
                if destination.exists():
                    # A previous attempt may have renamed the image before its
                    # database commit failed. Preserve those original bytes.
                    if self._file_sha256(destination) != metadata["image_sha256"]:
                        raise StoreError(409, "Unrecorded capture file already has different content")
                else:
                    with temporary.open("xb") as handle:
                        handle.write(image)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                    self._sync_directory(destination.parent)
                now = utc_now()
                db.execute("INSERT INTO captures VALUES (?,?,?,?,?,?,?)",
                           (capture_id, session["project_id"], session_id, metadata["image_sha256"], canonical,
                            str(relative), now))
                if evidence is not None:
                    db.execute("INSERT INTO capture_spatial_evidence VALUES (?,?,?,?,?,?,?,?)",
                               (capture_id, metadata["spatial_evidence_sha256"], len(spatial_json),
                                evidence.source, evidence.availability, len(evidence.rays),
                                sum(ray.status == "hit" for ray in evidence.rays), str(spatial_relative)))
            finally:
                temporary.unlink(missing_ok=True)
                if spatial_temporary is not None:
                    spatial_temporary.unlink(missing_ok=True)
            return {"capture_id": capture_id, "project_id": session["project_id"], "session_id": session_id,
                    "image_sha256": metadata["image_sha256"], "received_utc": now,
                    "status": "received", "duplicate": False, "durable": True,
                    **({"spatial_evidence_sha256": metadata["spatial_evidence_sha256"],
                        "spatial_evidence_size_bytes": len(spatial_json),
                        "spatial_ray_count": len(evidence.rays),
                        "spatial_hit_count": sum(ray.status == "hit" for ray in evidence.rays),
                        "spatial_availability": evidence.availability} if evidence else {})}

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _sync_directory(path: Path):
        # Windows cannot fsync directory handles through Python. On POSIX,
        # persist the rename itself before acknowledging the upload.
        if os.name != "nt":
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
