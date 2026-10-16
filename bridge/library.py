"""Immutable project model revisions attached to the existing entry tree."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import struct
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import NameBody, utc_now
from .preview import read_job
from .reconstruct import ReconstructionError, validate_gaussian_ply
from .store import Store, StoreError


MAX_MODEL_BYTES = 512 * 1024 * 1024
IDENTITY = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class ModelMetadata(BaseModel):
    schema_version: Literal[1]
    revision_id: uuid.UUID
    format: Literal["glb", "gaussian_ply"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_job_id: uuid.UUID | None = None
    transform: list[float] | None = None
    coordinate_frame: Literal["colmap", "quest_session"] = "colmap"
    display_name: str | None = None

    @field_validator("display_name")
    @classmethod
    def valid_label(cls, value: str | None) -> str | None:
        return NameBody(name=value).name if value is not None else None

    @model_validator(mode="after")
    def valid_transform(self):
        if self.format == "gaussian_ply" and self.transform is None:
            raise ValueError("Gaussian PLY requires a model-local to coordinate-frame transform")
        if self.transform is None:
            self.transform = IDENTITY.copy()
        m = self.transform
        if len(m) != 16 or not all(math.isfinite(value) for value in m):
            raise ValueError("Model transform must contain 16 finite numbers")
        if any(abs(m[i] - expected) > 1e-5 for i, expected in ((12, 0), (13, 0), (14, 0), (15, 1))):
            raise ValueError("Model transform has an invalid homogeneous row")
        return self


@contextmanager
def _db(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "library.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS revisions (
            revision_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, entry_id TEXT NOT NULL,
            format TEXT NOT NULL, sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL,
            size_bytes INTEGER NOT NULL, created_utc TEXT NOT NULL)""")
        db.execute("CREATE INDEX IF NOT EXISTS revisions_project ON revisions(project_id,entry_id,created_utc)")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _entry(store: Store, project_id: str, entry_id: str) -> dict:
    entry = next((entry for entry in store.entries(project_id) if entry["id"] == entry_id), None)
    if not entry:
        raise StoreError(404, "Model library entry not found in project")
    return entry


def _record(row: sqlite3.Row, duplicate: bool = False) -> dict:
    metadata = json.loads(row["metadata_json"])
    revision_id, project_id, entry_id = row["revision_id"], row["project_id"], row["entry_id"]
    return {"revision_id": revision_id, "project_id": project_id, "entry_id": entry_id,
            "format": row["format"], "artifact_sha256": row["sha256"],
            "size_bytes": row["size_bytes"], "created_utc": row["created_utc"],
            "source_job_id": metadata.get("source_job_id"), "transform": metadata["transform"],
            "coordinate_frame": metadata["coordinate_frame"],
            "display_name": metadata.get("display_name"), "placement_status": "unverified",
            "artifact_url": f"/api/v1/projects/{project_id}/models/{revision_id}/artifact",
            "duplicate": duplicate}


def _artifact_path(root: Path, row: sqlite3.Row) -> Path:
    extension = "glb" if row["format"] == "glb" else "ply"
    folder = root / "projects" / row["project_id"] / "models" / row["revision_id"]
    path = folder / f"model.{extension}"
    if not path.resolve().is_relative_to(root.resolve()) or folder.is_symlink() or path.is_symlink():
        raise StoreError(404, "Model artifact unavailable")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_artifact(path: Path, metadata: ModelMetadata) -> int:
    size = path.stat().st_size
    if size < 20 or size > MAX_MODEL_BYTES:
        raise StoreError(422, "Model file size is invalid")
    if _sha256(path) != metadata.artifact_sha256:
        raise StoreError(422, "Model SHA-256 mismatch")
    if metadata.format == "glb":
        with path.open("rb") as stream:
            header = stream.read(12)
            magic, version, declared_size = struct.unpack("<4sII", header)
            if magic != b"glTF" or version != 2 or declared_size != size:
                raise StoreError(422, "Model is not a complete GLB version 2 file")
            chunk_length, chunk_type = struct.unpack("<I4s", stream.read(8))
            if (chunk_type != b"JSON" or chunk_length % 4 != 0
                    or chunk_length > 16 * 1024 * 1024 or 20 + chunk_length > size):
                raise StoreError(422, "GLB JSON chunk is missing or too large")
            try:
                document = json.loads(stream.read(chunk_length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                raise StoreError(422, "GLB JSON chunk is invalid") from error
            if (not isinstance(document, dict) or not isinstance(document.get("asset"), dict)
                    or document["asset"].get("version") != "2.0"):
                raise StoreError(422, "GLB asset version is invalid")
            position = 20 + chunk_length
            while position < size:
                if size - position < 8:
                    raise StoreError(422, "GLB chunk header is truncated")
                length, kind = struct.unpack("<I4s", stream.read(8))
                if kind != b"BIN\x00" or length % 4 != 0 or position + 8 + length > size:
                    raise StoreError(422, "GLB binary chunk is invalid")
                stream.seek(length, 1)
                position += 8 + length
    else:
        try:
            validate_gaussian_ply(path)
        except ReconstructionError as error:
            raise StoreError(422, str(error)) from error
    return size


def publish_model(store: Store, project_id: str, entry_id: str,
                  metadata: ModelMetadata, staged_file: Path) -> dict:
    project_id, entry_id = str(uuid.UUID(project_id)), str(uuid.UUID(entry_id))
    _entry(store, project_id, entry_id)
    if metadata.source_job_id:
        read_job(store, project_id, str(metadata.source_job_id))
    size = _validate_artifact(staged_file, metadata)
    revision_id = str(metadata.revision_id)
    canonical = metadata.model_dump_json(exclude_none=False)
    with _db(store.root) as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT * FROM revisions WHERE revision_id=?", (revision_id,)).fetchone()
        if existing:
            if (existing["project_id"] != project_id or existing["entry_id"] != entry_id
                    or existing["sha256"] != metadata.artifact_sha256
                    or existing["metadata_json"] != canonical):
                raise StoreError(409, "Revision ID already has different content")
            path = _artifact_path(store.root, existing)
            if not path.is_file() or _sha256(path) != existing["sha256"]:
                raise StoreError(500, "Previously published model is missing or corrupt")
            return _record(existing, duplicate=True)
        base = store.root / "projects" / project_id / "models"
        base.mkdir(parents=True, exist_ok=True)
        if not base.resolve().is_relative_to(store.root.resolve()) or base.is_symlink():
            raise StoreError(500, "Model storage path is invalid")
        temporary = base / f".{revision_id}.{uuid.uuid4().hex}.tmp"
        final = base / revision_id
        try:
            temporary.mkdir()
            name = "model.glb" if metadata.format == "glb" else "model.ply"
            with staged_file.open("rb") as source, (temporary / name).open("xb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            (temporary / "transform.json").write_text(json.dumps({
                "transform": metadata.transform, "coordinate_frame": metadata.coordinate_frame,
                "placement_status": "unverified", "source_job_id": str(metadata.source_job_id) if metadata.source_job_id else None,
            }, indent=2), encoding="utf-8")
            temporary.rename(final)
            db.execute("""INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?)""",
                       (revision_id, project_id, entry_id, metadata.format,
                        metadata.artifact_sha256, canonical, size, utc_now()))
            row = db.execute("SELECT * FROM revisions WHERE revision_id=?", (revision_id,)).fetchone()
            return _record(row)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def list_models(store: Store, project_id: str, entry_id: str | None = None) -> list[dict]:
    project_id = str(uuid.UUID(project_id))
    store.project(project_id)
    if entry_id is not None:
        entry_id = str(uuid.UUID(entry_id))
        _entry(store, project_id, entry_id)
    with _db(store.root) as db:
        if entry_id is None:
            rows = db.execute("SELECT * FROM revisions WHERE project_id=? ORDER BY created_utc DESC,revision_id DESC",
                              (project_id,)).fetchall()
        else:
            rows = db.execute("SELECT * FROM revisions WHERE project_id=? AND entry_id=? ORDER BY created_utc DESC,revision_id DESC",
                              (project_id, entry_id)).fetchall()
    return [_record(row) for row in rows]


def model_artifact(store: Store, project_id: str, revision_id: str) -> tuple[dict, Path]:
    project_id, revision_id = str(uuid.UUID(project_id)), str(uuid.UUID(revision_id))
    store.project(project_id)
    with _db(store.root) as db:
        row = db.execute("SELECT * FROM revisions WHERE project_id=? AND revision_id=?",
                         (project_id, revision_id)).fetchone()
    if not row:
        raise StoreError(404, "Model revision not found")
    path = _artifact_path(store.root, row)
    if not path.is_file() or path.stat().st_size != row["size_bytes"] or _sha256(path) != row["sha256"]:
        raise StoreError(404, "Verified model artifact unavailable")
    return _record(row), path
