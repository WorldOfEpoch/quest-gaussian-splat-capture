import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from bridge.store import Store, StoreError


def _capture(store, project, session, *, capture_id=None, image=b"camera-image"):
    metadata = {"capture_id": capture_id or str(uuid.uuid4()),
                "project_id": project["id"], "session_id": session["id"],
                "tracking_origin_id": session["tracking_origin_id"],
                "image_encoding": "jpeg", "image_sha256": hashlib.sha256(image).hexdigest()}
    return metadata, image


def _seed(tmp_path):
    store = Store(tmp_path)
    project = store.create_project("Home")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    return store, project, session


def test_exact_retry_receipt_requires_original_hash_even_after_session_finished(tmp_path):
    store, project, session = _seed(tmp_path)
    metadata, image = _capture(store, project, session)
    first = store.save_capture(session["id"], metadata, image)
    assert first["status"] == "received" and first["durable"] is True and first["duplicate"] is False
    path = tmp_path / "projects" / project["id"] / "captures" / f"{metadata['capture_id']}.jpg"
    assert path.read_bytes() == image
    store.finish_session(session["id"])
    retry = store.save_capture(session["id"], metadata, image)
    assert retry["durable"] is True and retry["duplicate"] is True
    assert retry["received_utc"] == first["received_utc"]
    path.write_bytes(b"corrupt")
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], metadata, image)
    assert error.value.status == 500
    assert path.read_bytes() == b"corrupt"  # No PC original pruning or quiet repair.


def test_mismatched_upload_bytes_never_acknowledged(tmp_path):
    store, project, session = _seed(tmp_path)
    metadata, image = _capture(store, project, session)
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], metadata, b"different-image")
    assert error.value.status == 422
    assert store.captures(session["id"]) == []
    assert store.save_capture(session["id"], metadata, image)["durable"] is True


def test_storage_failure_leaves_unrecorded_original_for_exact_retry(tmp_path, monkeypatch):
    store, project, session = _seed(tmp_path)
    metadata, image = _capture(store, project, session)
    original_sync = Store._sync_directory

    def fail_once(_path):
        raise OSError("simulated durable rename failure")

    monkeypatch.setattr(Store, "_sync_directory", staticmethod(fail_once))
    with pytest.raises(OSError):
        store.save_capture(session["id"], metadata, image)
    assert store.captures(session["id"]) == []
    path = tmp_path / "projects" / project["id"] / "captures" / f"{metadata['capture_id']}.jpg"
    assert path.read_bytes() == image
    monkeypatch.setattr(Store, "_sync_directory", staticmethod(original_sync))
    changed = b"another-image"
    changed_meta = dict(metadata, image_sha256=hashlib.sha256(changed).hexdigest())
    with pytest.raises(StoreError) as error:
        store.save_capture(session["id"], changed_meta, changed)
    assert error.value.status == 409 and path.read_bytes() == image
    receipt = store.save_capture(session["id"], metadata, image)
    assert receipt["durable"] is True and path.read_bytes() == image


def test_two_bridge_store_instances_serialize_same_capture_id(tmp_path):
    store, project, session = _seed(tmp_path)
    metadata, image = _capture(store, project, session)
    other = Store(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(instance.save_capture, session["id"], metadata, image)
                   for instance in (store, other)]
        results = [future.result(timeout=10) for future in pending]
    assert {receipt["duplicate"] for receipt in results} == {False, True}
    assert all(receipt["durable"] is True for receipt in results)
    assert results[0]["received_utc"] == results[1]["received_utc"]
    assert len(store.captures(session["id"])) == 1
