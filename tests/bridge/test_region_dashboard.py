"""Area UI boundaries use synthetic photos and never launch the GPU worker."""
import hashlib
import io
import json
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bridge.app import create_app
from bridge.config import Config
from bridge.dashboard import set_capture_selection
from bridge.online import OnlineError


@pytest.fixture
def area_ui(tmp_path, monkeypatch):
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    store = app.state.store
    project = store.create_project("Synthetic area test")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    image = io.BytesIO()
    Image.new("RGB", (16, 16), (20, 30, 40)).save(image, format="JPEG")
    pixels = image.getvalue()
    photos = []
    for _ in range(3):
        capture_id = str(uuid.uuid4())
        store.save_capture(session["id"], {
            "capture_id": capture_id, "project_id": project["id"], "session_id": session["id"],
            "tracking_origin_id": session["tracking_origin_id"], "image_encoding": "jpeg",
            "image_sha256": hashlib.sha256(pixels).hexdigest(), "camera_id": "left",
        }, pixels)
        photos.append(capture_id)
    current = {"id": f"online:{session['id']}:{'a' * 64}", "source": "online",
               "source_sha256": "b" * 64, "coordinate_frame": f"online:{session['id']}:{'c' * 64}",
               "session_id": session["id"], "view": {"center": [0, 0, 0], "radius": 3, "up_axis": "z"}}
    monkeypatch.setattr("bridge.app.dashboard_current_model", lambda root, project_id: current)
    local = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    html = local.get(f"/ui?project_id={project['id']}").text
    csrf = re.search(r"name='csrf' value='([^']+)'", html).group(1)
    route = f"/ui/projects/{project['id']}/regions"
    plan = {"name": "Counter <corner>", "min": [0, 0, 0], "max": [1, 1, 1],
            "source_model_id": current["id"], "source_sha256": current["source_sha256"],
            "coordinate_frame": current["coordinate_frame"], "session_id": session["id"],
            "replacement_capture_ids": photos[:2]}
    return app, local, project, session, photos, csrf, route, plan, html


def test_area_editor_and_photo_picker_are_bounded_local_and_project_scoped(area_ui):
    app, local, project, session, photos, csrf, route, plan, html = area_ui
    assert "Replace or protect an area" in html
    assert "Save area" in html and "region-place" in html
    assert "region-distance" in html and "region-photos" in html
    assert "Gaussians crossing its boundary are retained" in html
    photo_route = f"/ui/projects/{project['id']}/region-photos?session_id={session['id']}"
    reply = local.get(photo_route)
    assert reply.status_code == 200 and reply.headers["cache-control"] == "no-store"
    assert reply.json()["count"] == 3 and reply.json()["page_size"] == 24
    assert {item["capture_id"] for item in reply.json()["photos"]} == set(photos)
    assert local.get(photo_route + "&page=0").status_code == 422
    other = app.state.store.create_project("Other")
    assert local.get(photo_route.replace(project["id"], other["id"])).status_code == 404
    remote = TestClient(app, base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert remote.get(photo_route).status_code == 403
    assert remote.get(route).status_code == 403


def test_area_plan_requires_exact_model_csrf_and_included_same_session_photos(area_ui, monkeypatch):
    app, local, project, session, photos, csrf, route, plan, html = area_ui
    saved = []
    def create(root, project_id, proposal):
        saved.append(proposal)
        return {**proposal, "id": str(uuid.uuid4()), "status": "planned", "private_path": str(root)}
    monkeypatch.setattr("bridge.app.create_region", create)
    assert local.post(route, data={"csrf": "bad", "plan": json.dumps(plan)}).status_code == 403
    assert local.post(route, data={"csrf": csrf, "plan": "[]"}).status_code == 422
    for key in ("source_model_id", "source_sha256", "coordinate_frame", "session_id"):
        stale = {**plan, key: "changed"}
        assert local.post(route, data={"csrf": csrf, "plan": json.dumps(stale)}).status_code == 409
    invalid = {**plan, "replacement_capture_ids": [str(uuid.uuid4())]}
    assert local.post(route, data={"csrf": csrf, "plan": json.dumps(invalid)}).status_code == 422
    set_capture_selection(app.state.store, project["id"], session["id"], photos[0], False)
    assert local.post(route, data={"csrf": csrf, "plan": json.dumps(plan)}).status_code == 422
    assert not saved
    set_capture_selection(app.state.store, project["id"], session["id"], photos[0], True)
    response = local.post(route, data={"csrf": csrf, "plan": json.dumps({**plan, "shell": "do not run"})})
    assert response.status_code == 201
    assert "shell" not in saved[0] and saved[0]["view"]["up_axis"] == "z"
    assert "private_path" not in response.json()
    assert response.json()["status"] == "planned"
    remote = TestClient(app, base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert remote.post(route, data={"csrf": csrf, "plan": json.dumps(plan)}).status_code == 403
    assert len(saved) == 1


def test_area_actions_require_explicit_review_and_fixed_commands(area_ui, monkeypatch):
    app, local, project, session, photos, csrf, route, plan, html = area_ui
    identifier = str(uuid.uuid4())
    called = []
    def act(root, project_id, region_id, action):
        called.append(action)
        return {"id": region_id, "name": "Area", "status": "queued", "message": "Preparing draft"}
    monkeypatch.setattr("bridge.app.request_region_action", act)
    path = route + f"/{identifier}"
    assert local.post(path + "/publish", data={"csrf": csrf}).status_code == 422
    assert local.post(path + "/shell", data={"csrf": csrf}).status_code == 404
    assert local.post(path + "/clear", data={"csrf": "bad"}).status_code == 403
    assert called == []
    for action in ("clear", "build", "lock", "protect", "publish", "restore"):
        assert local.post(path + "/" + action, data={"csrf": csrf, "reviewed": "yes"}).status_code == 200
    assert called == ["clear", "build", "lock", "protect", "publish", "restore"]
    updated = []
    monkeypatch.setattr("bridge.app.update_region_photos", lambda root, project, region, photos:
                        updated.append(photos) or {"id": region, "status": "planned"})
    response = local.post(path + "/set-photos", data={"csrf": csrf, "capture_ids": json.dumps(photos[:2])})
    assert response.status_code == 200 and updated == [photos[:2]]
    assert local.post(path + "/set-photos", data={"csrf": "bad", "capture_ids": "[]"}).status_code == 403
    def blocked(*args):
        raise OnlineError("Newer model work exists")
    monkeypatch.setattr("bridge.app.request_region_action", blocked)
    response = local.post(path + "/publish", data={"csrf": csrf, "reviewed": "yes"})
    assert response.status_code == 409 and response.json()["detail"] == "Newer model work exists"


def test_region_artifact_requires_exact_verified_revision(area_ui, monkeypatch, tmp_path):
    app, local, project, session, photos, csrf, route, plan, html = area_ui
    identifier = str(uuid.uuid4())
    data = b"synthetic artifact bytes"
    path = tmp_path / "candidate.ply"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    record = {"id": identifier, "name": "Area", "status": "draft_ready", "source_path": str(path),
              "artifact": {"sha256": digest, "size_bytes": len(data), "vertex_count": 1, "path": str(path)}}
    monkeypatch.setattr("bridge.app.list_regions", lambda *args: [record])
    listed = local.get(route)
    assert str(tmp_path) not in listed.text
    assert "path" not in listed.json()["regions"][0]["artifact"]
    monkeypatch.setattr("bridge.app.verified_region_artifact", lambda *args: (record, path))
    artifact_route = route + f"/{identifier}/draft.ply"
    assert local.get(artifact_route + "?revision=" + "f" * 64).status_code == 409
    response = local.get(artifact_route + "?revision=" + digest)
    assert response.status_code == 200 and response.content == data
    assert response.headers["cache-control"] == "private, no-store"
    def missing(*args):
        raise OnlineError("Missing candidate")
    monkeypatch.setattr("bridge.app.verified_region_artifact", missing)
    assert local.get(artifact_route + "?revision=" + digest).status_code == 404
