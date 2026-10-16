"""Durable optional Quest spatial evidence beside original camera captures."""

import hashlib
import io
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bridge.app import create_app
from bridge.config import Config
from bridge.models import CaptureMetadata
from bridge.store import Store, StoreError


def scene(tmp_path):
    store = Store(tmp_path)
    project = store.create_project("Home")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    photo = io.BytesIO()
    Image.new("RGB", (16, 12), (10, 20, 30)).save(photo, format="JPEG")
    image = photo.getvalue()
    metadata = {
        "schema_version": 1, "capture_id": str(uuid.uuid4()),
        "project_id": project["id"], "session_id": session["id"],
        "tracking_origin_id": session["tracking_origin_id"],
        "camera_id": "left", "image_timestamp_raw": "2026-10-12T12:00:00.0000000Z",
        "image_timestamp_kind": "Utc", "timestamp_clock": "meta_datetime",
        "image_width": 16, "image_height": 12, "image_encoding": "jpeg",
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "intrinsics": {"fx": 10., "fy": 10., "cx": 8., "cy": 6.},
        "camera_to_session": [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
        "coordinate_convention": "unity_lh_xright_yup_zforward",
        "position_units": "meters", "tracking_valid": True,
        "capture_source": "synthetic_test",
    }
    return store, project, session, metadata, image


def sidecar(metadata, **changes):
    evidence = {
        "schema_version": 1, "capture_id": metadata["capture_id"],
        "project_id": metadata["project_id"], "session_id": metadata["session_id"],
        "tracking_origin_id": metadata["tracking_origin_id"],
        "source": "synthetic_test", "association": "near_capture_unaligned",
        "coordinate_frame": "quest_session",
        "coordinate_convention": "unity_lh_xright_yup_zforward", "position_units": "meters",
        "trigger_realtime_s": 100., "sample_started_realtime_s": 100.1,
        "sample_completed_realtime_s": 100.2,
        "rgb_timestamp_raw": metadata["image_timestamp_raw"],
        "rgb_timestamp_kind": metadata["image_timestamp_kind"],
        "availability": "ready",
        "head_to_session": metadata["camera_to_session"],
        "rays": [{"origin": [0., 0., 0.], "direction": [0., 0., 1.],
                  "status": "hit", "hit": [0., 0., 2.],
                  "normal": [0., 0., -1.], "normal_confidence": .8},
                 {"origin": [0., 0., 0.], "direction": [1., 0., 0.],
                  "status": "no_hit"}],
    }
    evidence.update(changes)
    raw = json.dumps(evidence, separators=(",", ":"), allow_nan=False).encode()
    return raw


def with_hash(metadata, raw):
    return {**metadata, "spatial_evidence_sha256": hashlib.sha256(raw).hexdigest()}


def test_spatial_upload_receipt_retry_export_and_originals(tmp_path):
    store, project, session, metadata, image = scene(tmp_path)
    raw = sidecar(metadata)
    with_spatial = with_hash(metadata, raw)
    first = store.save_capture(session["id"], with_spatial, image, raw)
    assert first["durable"] is True and first["duplicate"] is False
    assert first["spatial_evidence_sha256"] == hashlib.sha256(raw).hexdigest()
    assert first["spatial_evidence_size_bytes"] == len(raw)
    assert (first["spatial_ray_count"], first["spatial_hit_count"]) == (2, 1)
    assert store.save_capture(session["id"], with_spatial, image, raw)["duplicate"] is True
    listed = store.captures(session["id"])[0]
    assert listed["spatial_availability"] == "ready" and "spatial_path" not in listed
    evidence = store.spatial_evidence(metadata["capture_id"])
    assert evidence["spatial_path"] and open(evidence["spatial_path"], "rb").read() == raw
    exported = store.export_session(session["id"])["captures"][0]
    assert exported["spatial_path"] == evidence["spatial_path"]
    assert exported["receipt"]["spatial_ray_count"] == 2
    pending = store.export_pending_captures(session["id"], set(), {metadata["capture_id"]}, 1)
    assert pending[0]["spatial_path"] == evidence["spatial_path"]
    assert (tmp_path / "projects" / project["id"] / "captures" /
            f"{metadata['capture_id']}.jpg").read_bytes() == image
    store.finish_session(session["id"])
    assert store.save_capture(session["id"], with_spatial, image, raw)["duplicate"] is True


def test_bad_spatial_upload_cannot_create_capture_or_change_retry(tmp_path):
    store, project, session, metadata, image = scene(tmp_path)
    raw = sidecar(metadata)
    paired = with_hash(metadata, raw)
    for invalid in (
        sidecar(metadata, capture_id=str(uuid.uuid4())),
        sidecar(metadata, sample_started_realtime_s=101.),
        sidecar(metadata, rays=[{"origin": [0, 0, 0], "direction": [0, 0, 1],
                                 "status": "hit", "hit": [1, 0, 2]}]),
        sidecar(metadata, untrusted_extra=True),
        sidecar(metadata, head_to_session=[0.] * 16),
        sidecar(metadata, source="meta_environment_raycast_openxr"),
        sidecar(metadata, availability="ready", rays=[]),
        sidecar(metadata).replace(b'"availability":"ready"',
                                  b'"availability":"ready","availability":"ready"'),
    ):
        with pytest.raises(StoreError) as error:
            store.save_capture(session["id"], with_hash(metadata, invalid), image, invalid)
        assert error.value.status == 422
        assert store.captures(session["id"]) == []
        assert not (tmp_path / "projects" / project["id"] / "captures" /
                    f"{metadata['capture_id']}.jpg").exists()
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], paired, image)
    assert error.value.status == 422
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], metadata, image, raw)
    assert error.value.status == 422
    oversized = b"x" * (2 * 1024 * 1024 + 1)
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], with_hash(metadata, oversized), image, oversized)
    assert error.value.status == 413
    store.save_capture(session["id"], paired, image, raw)
    changed = sidecar(metadata, availability="not_ready", rays=[])
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], with_hash(metadata, changed), image, changed)
    assert error.value.status == 409
    path = tmp_path / "projects" / project["id"] / "captures" / f"{metadata['capture_id']}.spatial.json"
    path.write_bytes(b"corrupt")
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], paired, image, raw)
    assert error.value.status == 500


def test_spatial_unavailable_empty_and_concurrent_exact_retry(tmp_path):
    store, project, session, metadata, image = scene(tmp_path)
    raw = sidecar(metadata, availability="not_supported", rays=[])
    paired = with_hash(metadata, raw)
    other = Store(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = [future.result(timeout=10) for future in (
            pool.submit(store.save_capture, session["id"], paired, image, raw),
            pool.submit(other.save_capture, session["id"], paired, image, raw))]
    assert {item["duplicate"] for item in receipts} == {False, True}
    assert all(item["durable"] for item in receipts)
    assert store.captures(session["id"])[0]["spatial_ray_count"] == 0


def test_spatial_storage_failure_has_no_receipt_and_exact_retry_recovers(tmp_path, monkeypatch):
    store, project, session, metadata, image = scene(tmp_path)
    raw = sidecar(metadata)
    paired = with_hash(metadata, raw)
    original_sync = Store._sync_directory

    def fail_once(_path):
        raise OSError("simulated spatial rename durability failure")

    monkeypatch.setattr(Store, "_sync_directory", staticmethod(fail_once))
    with pytest.raises(OSError):
        store.save_capture(session["id"], paired, image, raw)
    assert store.captures(session["id"]) == []
    sidecar_path = tmp_path / "projects" / project["id"] / "captures" / f"{metadata['capture_id']}.spatial.json"
    assert sidecar_path.read_bytes() == raw  # Unacknowledged original is retained.
    changed = sidecar(metadata, availability="not_ready", rays=[])
    monkeypatch.setattr(Store, "_sync_directory", staticmethod(original_sync))
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], with_hash(metadata, changed), image, changed)
    assert error.value.status == 409 and sidecar_path.read_bytes() == raw
    assert store.save_capture(session["id"], paired, image, raw)["durable"] is True


def test_authenticated_multipart_and_image_only_backward_compatibility(tmp_path):
    store, project, session, metadata, image = scene(tmp_path)
    token = "t" * 48
    app = create_app(Config(tmp_path, token))
    client = TestClient(app)
    assert "spatial_evidence" in client.get("/health").json()["capabilities"]
    endpoint = f"/api/v1/sessions/{session['id']}/captures"
    raw = sidecar(metadata)
    paired = with_hash(metadata, raw)
    files = {"metadata_json": (None, json.dumps(paired)),
             "image": ("image.jpg", image, "image/jpeg"),
             "spatial_json": ("evidence.json", raw, "application/json")}
    assert client.post(endpoint, files=files).status_code == 401
    response = client.post(endpoint, files=files, headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200, response.text
    assert response.json()["spatial_evidence_sha256"] == paired["spatial_evidence_sha256"]
    assert "spatial_path" not in client.get(endpoint, headers={"Authorization": "Bearer " + token}).text
    legacy = {**metadata, "capture_id": str(uuid.uuid4())}
    legacy_files = {"metadata_json": (None, json.dumps(legacy)),
                    "image": ("image.jpg", image, "image/jpeg")}
    for duplicate in (False, True):
        response = client.post(endpoint, files=legacy_files,
                               headers={"Authorization": "Bearer " + token})
        assert response.status_code == 200, response.text
        assert response.json()["duplicate"] is duplicate
        assert "spatial_evidence_sha256" not in response.json()
    assert CaptureMetadata.model_validate(legacy).spatial_evidence_sha256 is None
    # An image-only row committed by the pre-sidecar bridge has no new null
    # field in canonical metadata. A current HTTP retry must still match it.
    historical = {**metadata, "capture_id": str(uuid.uuid4())}
    old_canonical = CaptureMetadata.model_validate(historical).model_dump(mode="json")
    old_canonical.pop("spatial_evidence_sha256")
    store.save_capture(session["id"], old_canonical, image)
    response = client.post(endpoint, files={"metadata_json": (None, json.dumps(historical)),
                                            "image": ("image.jpg", image, "image/jpeg")},
                           headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200 and response.json()["duplicate"] is True
