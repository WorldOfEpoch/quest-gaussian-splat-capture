import re
import hashlib
import json
import struct
import uuid
import numpy as np

from fastapi.testclient import TestClient

from bridge.app import create_app
from bridge.config import Config
from bridge.dashboard import _person_view, render_dashboard
from bridge.store import Store


def test_live_depth_session_control_is_separate_from_gaussian_build(tmp_path):
    store = Store(tmp_path)
    project = store.create_project("Mesh Test")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    args = ([project], project, store.entries(project["id"]),
            store.sessions(project["id"]), [], "private-csrf")
    depth = {session["id"]: {"enabled": True, "status": "running", "accepted_views": 4,
             "total_views": 7, "pending_views": 2, "skipped_views": 1,
             "supported_samples": 123, "revision_number": 9, "error": "<retry>"}}
    html = render_dashboard(*args, depth_updates=depth)
    assert f"data-depth-session='{session['id']}'" in html
    assert "4 of 7 photos admitted" in html and "123 supported points" in html
    assert "&lt;retry&gt;" in html and "<retry>" not in html
    assert f"action='/ui/sessions/{session['id']}/depth'" in html
    assert "name='enabled' value='no'" in html and "Pause live depth" in html
    assert "name='csrf' value='private-csrf'" in html
    assert "Detailed Gaussian builds" in html and "Quick splat" in html
    idle = render_dashboard(*args, depth_updates={session["id"]: {"enabled": False,
        "status": "paused", "accepted_views": 4, "total_views": 7, "pending_views": 2,
        "skipped_views": 1, "supported_samples": 123}})
    assert "Start live depth" in idle and "name='enabled' value='yes'" in idle


def test_person_view_uses_registered_eye_and_levels_camera(tmp_path):
    job_dir = tmp_path / "job"
    processed = job_dir / "processed"
    config_dir = job_dir / "config"
    processed.mkdir(parents=True)
    config_dir.mkdir()
    angle = np.deg2rad(80.)
    camera = np.eye(4)
    camera[:3, :3] = [[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                      [0., np.sin(angle), np.cos(angle)]]
    camera[:3, 3] = [1., 2., 1.6]
    (processed / "transforms.json").write_text(json.dumps({
        "frames": [{"transform_matrix": camera.tolist()}]}))
    (config_dir / "dataparser_transforms.json").write_text(json.dumps({
        "transform": np.eye(4).tolist(), "scale": 1.}))
    job = {"processed_dir": str(processed), "training_config": str(config_dir / "config.yml")}
    view = _person_view(job, {"center": [0., 0., 0.], "radius": 4., "up_axis": "z"}, job_dir)
    assert view["start_view"] == "registered_capture"
    assert view["eye"] == [1., 2., 1.6]
    assert view["look_at"][2] == 1.6
    assert view["look_at"][1] > view["eye"][1]


def test_open_in_blender_is_local_verified_and_csrf_protected(tmp_path, monkeypatch):
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    project = app.state.store.create_project("House")
    local = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    page = local.get(f"/ui?project_id={project['id']}")
    assert "Open current preview in Blender" in page.text
    token = re.search(r"name='csrf' value='([^']+)'", page.text).group(1)
    route = f"/ui/projects/{project['id']}/open-current-in-blender"
    assert local.post(route, data={"csrf": "invalid"}).status_code == 403
    assert local.post(route, data={"csrf": token}).status_code == 409
    launched = []
    monkeypatch.setattr("bridge.app.open_current_splat_in_blender",
                        lambda root, project_id: launched.append((root, project_id)))
    assert local.post(route, data={"csrf": token}, follow_redirects=False).status_code == 303
    assert launched == [(tmp_path, project["id"])]
    lan = TestClient(app, base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert lan.post(route, data={"csrf": token}).status_code == 403
    assert len(launched) == 1


def test_local_project_dashboard_and_form(tmp_path):
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    page = client.get("/ui")
    assert page.status_code == 200
    assert "New project" in page.text
    assert page.headers["cache-control"] == "no-store"
    token = re.search(r"name='csrf' value='([^']+)'", page.text).group(1)
    assert client.post("/ui/projects", data={"name": "House", "csrf": "wrong"}).status_code == 403
    created = client.post("/ui/projects", data={"name": "Home <outside>", "csrf": token}, follow_redirects=True)
    assert created.status_code == 200
    assert "Home &lt;outside&gt;" in created.text
    assert "<outside>" not in created.text
    assert "Original" in created.text
    assert "Gaussian splat viewer" in created.text
    assert "wasm-unsafe-eval" in created.headers["content-security-policy"]
    assert len(app.state.store.projects()) == 1


def test_dashboard_is_loopback_only(tmp_path):
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    lan = TestClient(app, base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert lan.get("/ui").status_code == 403
    spoofed_host = TestClient(app, base_url="http://untrusted.example:8765", client=("127.0.0.1", 50000))
    assert spoofed_host.get("/ui").status_code == 403
    disabled = TestClient(create_app(Config(tmp_path / "disabled", "t" * 48)))
    assert disabled.get("/ui").status_code == 404


def test_local_splat_routes_verify_artifact_and_hide_file_path(tmp_path, monkeypatch):
    project_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    folder = tmp_path / "projects" / project_id / "reconstructions" / job_id
    folder.mkdir(parents=True)
    properties = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
    header = ("ply\nformat binary_little_endian 1.0\ncomment Vertical Axis: z\n"
              "element vertex 2\n" + "".join(f"property float {name}\n" for name in properties)
              + "end_header\n").encode("ascii")
    data = header + struct.pack("<14f", *([0.0] * 14)) + struct.pack("<14f", *([1.0] * 14))
    ply = folder / "splat.ply"
    ply.write_bytes(data)
    (folder / "job.json").write_text(json.dumps({
        "job_id": job_id, "project_id": project_id, "session_id": str(uuid.uuid4()),
        "created_utc": "2026-09-28T12:00:00Z", "status": "ready_for_blender",
        "quality": "quick", "input_captures": [], "registered_views": 2,
        "artifact": {"path": str(ply), "sha256": hashlib.sha256(data).hexdigest(),
                     "size_bytes": len(data), "vertex_count": 2},
    }), encoding="utf-8")
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    api = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    # A project row is required for the dashboard's job selection.
    with app.state.store.connection() as db:
        db.execute("INSERT INTO projects VALUES (?,?,?,?)",
                   (project_id, "Synthetic test project", "2026-09-28T12:00:00Z", "2026-09-28T12:00:00Z"))
    jobs = api.get(f"/ui/jobs?project_id={project_id}")
    assert jobs.status_code == 200
    assert jobs.json()["jobs"][0]["viewable"] is True
    assert jobs.json()["jobs"][0]["view"]["up_axis"] == "z"
    assert str(tmp_path) not in jobs.text
    assert api.get(f"/ui/jobs/{project_id}/{job_id}/splat.ply").content == data
    page = api.get(f"/ui?project_id={project_id}")
    assert "Open saved quick in Blender" in page.text
    token = re.search(r"name='csrf' value='([^']+)'", page.text).group(1)
    saved_route = f"/ui/projects/{project_id}/jobs/{job_id}/open-in-blender"
    assert api.post(saved_route, data={"csrf": "wrong"}).status_code == 403
    launched = []
    monkeypatch.setattr("bridge.app.open_saved_splat_in_blender",
                        lambda root, project, job: launched.append((root, project, job)))
    assert api.post(saved_route, data={"csrf": token}, follow_redirects=False).status_code == 303
    assert launched == [(tmp_path, project_id, job_id)]
    assert api.get("/ui/static/viewer.js").status_code == 200
    lan = TestClient(app, base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert lan.get(f"/ui/jobs/{project_id}/{job_id}/splat.ply").status_code == 403
    assert api.get(f"/ui/jobs/{uuid.uuid4()}/{job_id}/splat.ply").status_code == 404
    ply.write_bytes(data + b"tampered")
    assert api.get(f"/ui/jobs/{project_id}/{job_id}/splat.ply").status_code == 404
