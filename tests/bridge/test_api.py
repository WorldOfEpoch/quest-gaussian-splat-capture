import hashlib
import io
import json
import uuid

from fastapi.testclient import TestClient
from PIL import Image

from bridge.app import create_app
from bridge.config import Config


def client(tmp_path):
    token = "t" * 48
    return TestClient(create_app(Config(tmp_path, token))), {"Authorization": f"Bearer {token}"}


def project_session(api, headers):
    project = api.post("/api/v1/projects", headers=headers, json={"name": " House "}).json()
    origin = str(uuid.uuid4())
    session = api.post("/api/v1/sessions", headers=headers,
                       json={"project_id": project["id"], "tracking_origin_id": origin}).json()
    return project, session


def capture_payload(project, session):
    output = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(output, format="JPEG")
    data = output.getvalue()
    metadata = {
        "schema_version": 1, "capture_id": str(uuid.uuid4()), "project_id": project["id"],
        "session_id": session["id"], "tracking_origin_id": session["tracking_origin_id"],
        "camera_id": "left", "image_timestamp_raw": "2026-09-28T12:34:56.1234567Z",
        "image_timestamp_kind": "Utc", "timestamp_clock": "meta_datetime",
        "image_width": 16, "image_height": 12, "image_encoding": "jpeg",
        "image_sha256": hashlib.sha256(data).hexdigest(),
        "intrinsics": {"fx": 10, "fy": 10, "cx": 8, "cy": 6},
        "camera_to_session": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "coordinate_convention": "unity_lh_xright_yup_zforward", "position_units": "meters",
        "tracking_valid": True, "capture_source": "synthetic_test",
    }
    return metadata, data


def upload(api, headers, session, metadata, data):
    return api.post(f"/api/v1/sessions/{session['id']}/captures", headers=headers,
                    data={"metadata_json": json.dumps(metadata)}, files={"image": ("ignored.jpg", data, "image/jpeg")})


def test_auth_and_persistent_project_hierarchy(tmp_path):
    api, headers = client(tmp_path)
    assert api.get("/health").status_code == 200
    assert api.get("/api/v1/projects").status_code == 401
    project = api.post("/api/v1/projects", headers=headers, json={"name": " House "}).json()
    assert project["name"] == "House"
    base = f"/api/v1/projects/{project['id']}/entries"
    original = api.get(base, headers=headers).json()["entries"][0]
    assert original["name"] == "Original" and original["protected"]
    assert api.patch(f"{base}/{original['id']}", headers=headers, json={"name": "Changed"}).status_code == 409
    category = api.post(base, headers=headers, json={"name": "Additions", "kind": "category"}).json()
    design = api.post(base, headers=headers, json={"name": "Rear room", "kind": "design", "parent_id": category["id"]}).json()
    variation = api.post(base, headers=headers, json={"name": "Pitched roof", "kind": "variation", "parent_id": design["id"]}).json()
    assert variation["parent_id"] == design["id"]
    assert api.patch(f"{base}/{design['id']}", headers=headers, json={"parent_id": variation["id"]}).status_code == 422
    nested = api.post(base, headers=headers, json={"name": "Alternate roof", "kind": "variation", "parent_id": variation["id"]}).json()
    assert api.patch(f"{base}/{variation['id']}", headers=headers, json={"parent_id": nested["id"]}).status_code == 422
    assert api.patch(f"{base}/{design['id']}", headers=headers, json={"name": None}).status_code == 422
    second = api.post("/api/v1/projects", headers=headers, json={"name": "Garden"}).json()
    assert api.post(f"/api/v1/projects/{second['id']}/entries", headers=headers,
                    json={"name": "Bad", "kind": "design", "parent_id": category["id"]}).status_code == 422
    restarted, restarted_headers = client(tmp_path)
    assert len(restarted.get(base, headers=restarted_headers).json()["entries"]) == 5


def test_capture_receipt_retry_conflict_and_finish(tmp_path):
    api, headers = client(tmp_path)
    project, session = project_session(api, headers)
    metadata, data = capture_payload(project, session)
    first = upload(api, headers, session, metadata, data)
    assert first.status_code == 200 and first.json()["duplicate"] is False
    assert upload(api, headers, session, metadata, data).json()["duplicate"] is True
    changed = dict(metadata, image_timestamp_raw="2026-09-28T12:34:57.1234567Z")
    assert upload(api, headers, session, changed, data).status_code == 409
    assert api.get(f"/api/v1/sessions/{session['id']}", headers=headers).json()["capture_count"] == 1
    saved = tmp_path / "projects" / project["id"] / "captures" / f"{metadata['capture_id']}.jpg"
    assert saved.read_bytes() == data
    exported = api.app.state.store.export_session(session["id"])
    assert exported["captures"][0]["image_path"] == str(saved.resolve())
    assert exported["captures"][0]["metadata"]["capture_source"] == "synthetic_test"
    preview = api.get(f"/api/v1/sessions/{session['id']}/preview?representation=splat", headers=headers).json()
    assert preview["status"] == "not_ready" and preview["artifact"] is None
    finished = api.post(f"/api/v1/sessions/{session['id']}/finish", headers=headers).json()
    assert finished["status"] == "finished"
    assert api.post(f"/api/v1/sessions/{session['id']}/finish", headers=headers).json()["finished_utc"] == finished["finished_utc"]
    assert upload(api, headers, session, metadata, data).json()["duplicate"] is True
    new_metadata = dict(metadata, capture_id=str(uuid.uuid4()))
    assert upload(api, headers, session, new_metadata, data).status_code == 409


def test_reject_corrupt_or_misaligned_capture(tmp_path):
    api, headers = client(tmp_path)
    project, session = project_session(api, headers)
    metadata, data = capture_payload(project, session)
    assert upload(api, headers, session, metadata, b"not an image").status_code == 422
    assert upload(api, headers, session, dict(metadata, image_width=17), data).status_code == 422
    assert upload(api, headers, session, dict(metadata, tracking_origin_id=str(uuid.uuid4())), data).status_code == 422
    bad_pose = dict(metadata, camera_to_session=[2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    assert upload(api, headers, session, bad_pose, data).status_code == 422
    assert api.get(f"/api/v1/sessions/{session['id']}", headers=headers).json()["capture_count"] == 0
