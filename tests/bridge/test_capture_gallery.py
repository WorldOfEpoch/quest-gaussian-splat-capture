import hashlib
import io
import re
import uuid

from fastapi.testclient import TestClient
from PIL import Image

from bridge.app import create_app
from bridge.config import Config


def _seed(tmp_path, count=3):
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    store = app.state.store
    project = store.create_project("Home interior and exterior")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    originals = {}
    for index in range(count):
        data = io.BytesIO()
        Image.new("RGB", (640, 480), (index % 256, 20, 70)).save(data, format="JPEG", quality=95)
        image = data.getvalue()
        capture_id = str(uuid.uuid4())
        store.save_capture(session["id"], {
            "capture_id": capture_id, "project_id": project["id"],
            "session_id": session["id"], "tracking_origin_id": session["tracking_origin_id"],
            "image_encoding": "jpeg", "image_sha256": hashlib.sha256(image).hexdigest(),
            "image_timestamp_raw": f"2026-09-28T12:00:{index % 60:02d}.0000000Z",
            "camera_id": "left",
        }, image)
        originals[capture_id] = image
    local = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    return app, local, project, session, originals


def test_project_panel_and_paged_batch_show_real_original_photos(tmp_path):
    _app, client, project, session, originals = _seed(tmp_path, 53)
    home = client.get(f"/ui?project_id={project['id']}")
    assert home.status_code == 200
    assert "Home interior and exterior" in home.text
    assert "View 53 photos" in home.text
    assert home.text.count("Saved camera view") == 6  # Bounded latest-photo preview.
    path = f"/ui/projects/{project['id']}/sessions/{session['id']}/captures"
    first = client.get(path)
    assert first.status_code == 200
    assert "Page 1 of 3" in first.text
    assert first.text.count("Saved camera view") == 24
    assert "older offline build command still uses all received photos" in first.text
    last = client.get(path + "?page=3")
    assert last.status_code == 200 and last.text.count("Saved camera view") == 5
    capture_id, data = next(iter(originals.items()))
    image_url = f"/ui/projects/{project['id']}/captures/{capture_id}/image"
    full = client.get(image_url)
    assert full.status_code == 200 and full.content == data
    assert full.headers["content-type"] == "image/jpeg"
    thumb = client.get(image_url + "?thumbnail=true")
    assert thumb.status_code == 200 and thumb.content != data
    with Image.open(io.BytesIO(thumb.content)) as image:
        assert image.size == (320, 240)
    assert (tmp_path / "projects" / project["id"] / "captures" / f"{capture_id}.jpg").read_bytes() == data


def test_reversible_selection_is_local_and_does_not_alter_original(tmp_path):
    app, client, project, session, originals = _seed(tmp_path)
    capture_id, data = next(iter(originals.items()))
    batch_url = f"/ui/projects/{project['id']}/sessions/{session['id']}/captures"
    page = client.get(batch_url)
    token = re.search(r"name='csrf' value='([^']+)'", page.text).group(1)
    action = f"{batch_url}/{capture_id}/selection"
    assert client.post(action, data={"csrf": "wrong", "included": "no"}).status_code == 403
    excluded = client.post(action, data={"csrf": token, "included": "no", "page": "1"},
                           follow_redirects=True)
    assert excluded.status_code == 200 and "Excluded" in excluded.text
    with app.state.store.connection() as db:
        assert db.execute("SELECT included FROM capture_selections WHERE capture_id=?", (capture_id,)).fetchone()[0] == 0
    included = client.post(action, data={"csrf": token, "included": "yes", "page": "1"},
                           follow_redirects=True)
    assert included.status_code == 200
    with app.state.store.connection() as db:
        assert db.execute("SELECT included FROM capture_selections WHERE capture_id=?", (capture_id,)).fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM captures WHERE session_id=?", (session["id"],)).fetchone()[0] == 3
    assert app.state.store.export_session(session["id"])["captures"]  # Current builder still sees all.
    assert (tmp_path / "projects" / project["id"] / "captures" / f"{capture_id}.jpg").read_bytes() == data


def test_gallery_rejects_remote_cross_project_corrupt_and_escaped_images(tmp_path):
    app, client, project, session, originals = _seed(tmp_path)
    capture_id, data = next(iter(originals.items()))
    image_url = f"/ui/projects/{project['id']}/captures/{capture_id}/image"
    other = app.state.store.create_project("Other")
    assert client.get(image_url.replace(project["id"], other["id"])).status_code == 404
    assert client.get(f"/ui/projects/{other['id']}/sessions/{session['id']}/captures").status_code == 404
    lan = TestClient(app, base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert lan.get(image_url).status_code == 403
    assert lan.get(f"/ui/projects/{project['id']}/sessions/{session['id']}/captures").status_code == 403
    path = tmp_path / "projects" / project["id"] / "captures" / f"{capture_id}.jpg"
    path.write_bytes(data + b"tampered")
    assert client.get(image_url).status_code == 404
    path.write_bytes(data)
    with app.state.store.connection() as db:
        db.execute("UPDATE captures SET image_path=? WHERE capture_id=?", ("../outside.jpg", capture_id))
    assert client.get(image_url).status_code == 404
