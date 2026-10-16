"""Phone transport in the existing homebase; never substitute phone poses."""
from __future__ import annotations
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .store import Store, StoreError
from .jobs import request_build, request_status, processing_paused
from .preview import read_job, read_quest_preview
from .mesh import read_mesh

MAX_PHONE_BYTES = 64 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class PhoneMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    schema_version: Literal[2]
    capture_source: Literal['android_camera2', 'synthetic_phone_test']
    capture_id: uuid.UUID
    project_id: uuid.UUID
    session_id: uuid.UUID
    tracking_origin_id: uuid.UUID  # session partition ID only, never a pose claim
    camera_id: str = Field(min_length=1, max_length=64)
    image_timestamp_raw: str = Field(max_length=64)
    timestamp_clock: Literal['android_wall_utc']
    sensor_timestamp_ns: int | None = Field(default=None, ge=0)
    sensor_timestamp_source: Literal['realtime', 'unknown']
    image_width: int = Field(gt=0, le=16384)
    image_height: int = Field(gt=0, le=16384)
    image_encoding: Literal['jpeg']
    image_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    image_size_bytes: int = Field(gt=0, le=MAX_PHONE_BYTES)
    image_orientation: Literal['exif']
    jpeg_orientation_degrees: Literal[0, 90, 180, 270]
    camera_to_session: None = None
    tracking_valid: Literal[False] = False
    intrinsics: None = None  # uncalibrated JPEG projection: COLMAP estimates it
    calibration: list[float] | None = Field(default=None, min_length=5, max_length=5)
    calibration_reference: Literal['sensor_pre_correction_active_array', 'unavailable']
    focal_length_mm: float | None = Field(default=None, gt=0, le=1000)
    aperture: float | None = Field(default=None, gt=0, le=100)
    exposure_duration_ns: int | None = Field(default=None, ge=0)
    iso: int | None = Field(default=None, gt=0, le=1000000)
    device_model: str = Field(max_length=120)

    @field_validator('image_timestamp_raw')
    @classmethod
    def timestamp(cls, v):
        if not v.endswith('Z'):
            raise ValueError('Phone wall timestamp must explicitly be UTC')
        datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


def _write(path: Path, value: dict):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    Store._sync_directory(path.parent)


def _durable_offset(root: Path, metadata: dict) -> int:
    """Recover only flushed bytes whose offset was atomically checkpointed.

    Call with the upload lock held. An interrupted append may have extended the
    file before its flush/checkpoint succeeded; that tail must be retransmitted.
    """
    part, progress = root / 'image.part', root / 'progress.json'
    size = part.stat().st_size if part.exists() else 0
    if not progress.exists():
        # Existing installations have no checkpoint. Flush their saved prefix
        # before adopting it; final commit still verifies the entire photo hash.
        if part.exists():
            with part.open('r+b') as f:
                os.fsync(f.fileno())
        _write(progress, {'offset': size})
    try:
        offset = json.loads(progress.read_text(encoding='utf-8'))['offset']
        if type(offset) is not int or not 0 <= offset <= metadata['image_size_bytes'] or size < offset:
            raise ValueError('Invalid durable offset')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StoreError(500, 'Upload progress is unavailable or corrupt') from error
    if size > offset:
        with part.open('r+b') as f:
            f.truncate(offset)
            f.flush()
            os.fsync(f.fileno())
    return offset


def phone_status(store: Store, session_id: str):
    session = store.session(session_id)
    jobs = []
    root = store.root / 'projects' / session['project_id'] / 'reconstructions'
    for p in root.glob('*/job.json'):
        try:
            j = json.loads(p.read_text())
            if j.get('session_id') == session_id:
                jobs.append(j)
        except (OSError, ValueError):
            continue
    newest = max(jobs, key=lambda j: j.get('updated_utc', j.get('created_utc', '')), default={})
    ready = sorted((j for j in jobs if j.get('status') == 'ready_for_blender'),
                   key=lambda j: j.get('updated_utc', j.get('created_utc', '')), reverse=True)
    completed = None
    for j in ready:
        try:
            verified, _ = read_job(store, session['project_id'], j['job_id'])
            a, suffix = verified['artifact'], 'splat'
            if 'quest_preview' in verified:
                try:
                    preview_job, _ = read_quest_preview(store, session['project_id'], j['job_id'])
                    preview = preview_job['quest_preview']
                    if preview['size_bytes'] <= MAX_PHONE_BYTES and preview['vertex_count'] <= 250000:
                        a, suffix = preview, 'quest-splat'
                except (StoreError, KeyError, ValueError, TypeError):
                    pass  # A verified, bounded original remains usable.
            if a['size_bytes'] > 64 * 1024 * 1024 or a['vertex_count'] > 250000:
                continue
            completed = {'job_id': j['job_id'], 'sha256': a['sha256'],
                         'size_bytes': a['size_bytes'], 'vertex_count': a['vertex_count'],
                         'download_path': f"/api/v1/projects/{session['project_id']}/reconstructions/{j['job_id']}/{suffix}",
                         'display_mode': 'inspection_model'}
            break
        except (StoreError, KeyError, ValueError, TypeError):
            continue
    queue = [r for r in request_status(store.root, session['project_id']) if r['session_id'] == session_id]
    meshes, ready_meshes = [], []
    for p in root.glob('*/mesh-revisions/*/mesh.json'):
        try:
            j=json.loads(p.read_text())
            source_job=next((job for job in jobs if job['job_id']==j.get('source_job_id')),None)
            if not source_job or j['project_id'] != session['project_id'] or j['revision_id'] != p.parent.name or j['source_job_id'] != p.parent.parent.parent.name:
                continue
            summary = {'revision_id':j['revision_id'],'source_job_id':j['source_job_id'],
                       'created_utc':j['created_utc'],'status':j['status'],'stage':j.get('stage'),
                       'error':j.get('error')}
            meshes.append(summary)
            if j['status']=='ready':
                verified, _ = read_mesh(store, session['project_id'], j['source_job_id'], j['revision_id'])
                a = verified['artifact']
                if a['size_bytes'] > MAX_PHONE_BYTES or a['vertex_count'] > 250000 or a['triangle_count'] > 500000:
                    continue
                ready_meshes.append({**summary, **{k:a[k] for k in ('sha256','size_bytes','vertex_count','triangle_count')},
                                     'method':verified.get('method'),'coordinate_frame':verified.get('coordinate_frame'),
                                     'uncertainty':verified.get('uncertainty'),
                                     'provenance':{k:verified.get(k) for k in ('input_captures','calibrated_photos')},
                                     'display_mode':'inspection_model',
                                     'download_path':f"/api/v1/projects/{session['project_id']}/reconstructions/{j['source_job_id']}/meshes/{j['revision_id']}"})
        except (OSError,ValueError,KeyError,TypeError,StoreError):
            continue
    mesh=max(ready_meshes,key=lambda j:j['created_utc'],default=None)
    mesh_processing=max(meshes,key=lambda j:j['created_utc'],default=None)
    return {'uploaded': session['capture_count'], 'analyzed': newest.get('analyzed_views'),
            'registered': newest.get('registered_views'),
            'processing_stage': newest.get('processing_stage', newest.get('status', 'not_requested')),
            'processing_paused': processing_paused(store.root), 'queue': queue,
            'preview': completed, 'mesh': mesh, 'mesh_processing': mesh_processing,
            'error': newest.get('error')}


def install_mobile(app, store: Store, authenticate):
    router = APIRouter(prefix='/api/v1/mobile', dependencies=[Depends(authenticate)])

    def staging(session_id, capture_id):
        session_id, capture_id = str(session_id), str(capture_id)
        store.session(session_id)
        path = store.root / 'phone-upload-staging' / session_id / capture_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def receipt_if_saved(metadata):
        with store.connection() as db:
            row = db.execute('SELECT * FROM captures WHERE capture_id=?', (metadata['capture_id'],)).fetchone()
        if not row:
            return None
        if row['metadata_json'] != json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False):
            raise StoreError(409, 'Capture ID already has different content')
        path = store.root / row['image_path']
        if (path.is_symlink() or not path.resolve().is_relative_to(store.root.resolve()) or
                not path.is_file() or store._file_sha256(path) != metadata['image_sha256']):
            raise StoreError(500, 'Previously received original is corrupt')
        return {**store._capture(row, True), 'durable': True}

    @router.post('/sessions/{session_id}/uploads')
    def begin(session_id: uuid.UUID, body: PhoneMetadata):
        m = body.model_dump(mode='json')
        session = store.session(str(session_id))
        if (m['session_id'] != str(session_id) or m['project_id'] != session['project_id'] or
                m['tracking_origin_id'] != session['tracking_origin_id']):
            raise StoreError(422, 'Phone session partition mismatch')
        done = receipt_if_saved(m)
        if done:
            return {'offset': m['image_size_bytes'], 'receipt': done}
        if session['status'] != 'capturing':
            raise StoreError(409, 'Session has finished')
        root = staging(session_id, body.capture_id)
        # Store-wide OS lock serializes writers across LAN/loopback listeners.
        with _upload_lock(root):
            # Another listener may commit while this request waits for the lock.
            done = receipt_if_saved(m)
            if done:
                return {'offset': m['image_size_bytes'], 'receipt': done}
            mp = root / 'metadata.json'
            if mp.exists() and json.loads(mp.read_text()) != m:
                raise StoreError(409, 'Upload ID already has different metadata')
            if not mp.exists():
                _write(mp, m)
            return {'offset': _durable_offset(root, m)}

    @router.get('/sessions/{session_id}/uploads/{capture_id}')
    def offset(session_id: uuid.UUID, capture_id: uuid.UUID):
        root = staging(session_id, capture_id)
        with _upload_lock(root):
            if not (root / 'metadata.json').exists():
                raise StoreError(404, 'Upload not initialized')
            m = json.loads((root / 'metadata.json').read_text())
            receipt = receipt_if_saved(m)
            return {'offset': m['image_size_bytes'] if receipt else _durable_offset(root, m),
                    'receipt': receipt}

    @router.put('/sessions/{session_id}/uploads/{capture_id}')
    async def chunk(session_id: uuid.UUID, capture_id: uuid.UUID, request: Request):
        root = staging(session_id, capture_id)
        if not (root / 'metadata.json').exists():
            raise StoreError(404, 'Upload not initialized')
        try:
            start = int(request.headers['Upload-Offset'])
        except (KeyError, ValueError):
            raise StoreError(422, 'Upload-Offset must be an integer')
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > CHUNK_BYTES:
                raise StoreError(413, 'Chunk exceeds 1 MiB')
            data.extend(part)
        if not data:
            raise StoreError(422, 'Empty chunk')
        digest = request.headers.get('Upload-Chunk-SHA256', '')
        if hashlib.sha256(data).hexdigest() != digest:
            raise StoreError(422, 'Chunk integrity mismatch')
        with _upload_lock(root):
            m = json.loads((root / 'metadata.json').read_text())
            if receipt_if_saved(m):
                raise StoreError(409, 'Upload already completed')
            size = _durable_offset(root, m)
            if start != size:
                raise StoreError(409, f'Offset mismatch; durable offset is {size}')
            if start < 0 or size + len(data) > m['image_size_bytes']:
                raise StoreError(413, 'Upload exceeds declared photo size')
            with (root / 'image.part').open('ab') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            _write(root / 'progress.json', {'offset': size + len(data)})
            return {'offset': size + len(data), 'chunk_sha256': digest}

    @router.post('/sessions/{session_id}/uploads/{capture_id}/commit')
    def commit(session_id: uuid.UUID, capture_id: uuid.UUID):
        root = staging(session_id, capture_id)
        with _upload_lock(root):
            if not (root / 'metadata.json').exists():
                raise StoreError(404, 'Upload not initialized')
            m = json.loads((root / 'metadata.json').read_text())
            done = receipt_if_saved(m)
            if done:
                (root / 'image.part').unlink(missing_ok=True)
                return done
            part = root / 'image.part'
            if _durable_offset(root, m) != m['image_size_bytes']:
                raise StoreError(409, 'Photo upload is incomplete')
            if store._file_sha256(part) != m['image_sha256']:
                raise StoreError(422, 'Photo integrity mismatch; original retained')
            try:
                with Image.open(part) as im:
                    if im.format != 'JPEG' or im.size != (m['image_width'], m['image_height']):
                        raise ValueError('Photo dimensions mismatch')
                    im.load()  # JPEG verify() alone accepts truncated scan data.
            except (OSError, ValueError, Image.DecompressionBombError):
                raise StoreError(422, 'Invalid JPEG')
            result = store.save_capture(str(session_id), m, part.read_bytes())
            # Only staged transport copy is discarded; local and home originals stay.
            part.unlink()
            return result

    @router.get('/sessions/{session_id}/status')
    def status(session_id: uuid.UUID):
        return phone_status(store, str(session_id))

    @router.post('/sessions/{session_id}/update')
    def update(session_id: uuid.UUID):
        return request_build(store.root, str(session_id), reason='preview', quality='quick')

    app.include_router(router)


from contextlib import contextmanager

@contextmanager
def _upload_lock(root):
    import sys
    with (root / 'upload.lock').open('a+b') as f:
        if f.tell() == 0:
            f.write(b'\0'); f.flush()
        f.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise StoreError(409, 'Upload writer busy; retry')
        try:
            yield
        finally:
            f.seek(0)
            if sys.platform == 'win32':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
