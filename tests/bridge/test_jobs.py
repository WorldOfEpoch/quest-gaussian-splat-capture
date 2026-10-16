import hashlib
import re
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from bridge.app import create_app
from bridge.config import Config
from bridge.jobs import (Scheduler, on_capture_received, policy, processing_paused,
                         request_build, request_status, set_policy, set_processing_paused)
from bridge.store import Store


def _seed(tmp_path):
    store = Store(tmp_path)
    project = store.create_project("Home")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    return store, project, session


def _capture(store, project, session, index):
    data = f"image-{index}".encode()
    return store.save_capture(session["id"], {
        "capture_id": str(uuid.uuid4()), "project_id": project["id"],
        "session_id": session["id"], "tracking_origin_id": session["tracking_origin_id"],
        "image_sha256": hashlib.sha256(data).hexdigest(), "image_encoding": "jpeg",
    }, data)


def test_auto_policy_threshold_debounce_and_idempotent_preview(tmp_path):
    store, project, session = _seed(tmp_path)
    assert policy(tmp_path, session["id"])["auto_enabled"] is False
    assert on_capture_received(tmp_path, session["id"]) is None
    set_policy(tmp_path, session["id"], auto_enabled=True, native_orientation_reviewed=True)
    for index in range(2):
        _capture(store, project, session, index)
        assert on_capture_received(tmp_path, session["id"])["status"] == "need_more_views"
    _capture(store, project, session, 2)
    pending = on_capture_received(tmp_path, session["id"])
    assert pending["status"] == "pending" and pending["capture_count"] == 3
    immediate = request_build(tmp_path, session["id"], reason="preview", quality="quick")
    assert immediate["request_id"] == pending["request_id"]
    with __import__("sqlite3").connect(tmp_path / "build-queue.sqlite3") as db:
        row = db.execute("SELECT reason,eligible_utc,requested_utc FROM requests WHERE id=?",
                         (pending["request_id"],)).fetchone()
    assert row[0] == "preview" and row[1] >= row[2]
    assert len(request_status(tmp_path, project["id"])) == 1


def test_auto_waits_for_two_new_views_after_failed_attempt(tmp_path):
    store, project, session = _seed(tmp_path)
    set_policy(tmp_path, session["id"], auto_enabled=True, native_orientation_reviewed=True)
    for index in range(3):
        _capture(store, project, session, index)
    first = on_capture_received(tmp_path, session["id"])
    with __import__("sqlite3").connect(tmp_path / "build-queue.sqlite3") as db:
        db.execute("UPDATE requests SET status='failed' WHERE id=?", (first["request_id"],))
    assert on_capture_received(tmp_path, session["id"])["status"] == "failed"
    _capture(store, project, session, 3)
    assert on_capture_received(tmp_path, session["id"])["status"] == "waiting_for_two_views"
    _capture(store, project, session, 4)
    assert on_capture_received(tmp_path, session["id"])["status"] == "pending"
    assert len(request_status(tmp_path, project["id"])) == 2


def test_global_pause_cancels_pending_and_blocks_preview_and_scheduler(tmp_path):
    store, project, session = _seed(tmp_path)
    set_policy(tmp_path, session["id"], auto_enabled=True, native_orientation_reviewed=True)
    for index in range(3):
        _capture(store, project, session, index)
    pending = on_capture_received(tmp_path, session["id"])
    assert pending["status"] == "pending"

    pause = set_processing_paused(tmp_path, True)
    assert pause == {"paused": True, "cancelled_pending": 1}
    assert processing_paused(tmp_path)
    assert request_status(tmp_path, project["id"])[0]["status"] == "cancelled"
    assert request_build(tmp_path, session["id"], reason="preview")["status"] == "paused"
    assert request_build(tmp_path, session["id"], reason="manual")["status"] == "paused"
    assert on_capture_received(tmp_path, session["id"])["status"] == "paused"
    with patch("bridge.jobs.subprocess.Popen") as launch:
        Scheduler(tmp_path).tick()
    launch.assert_not_called()

    assert set_processing_paused(tmp_path, False)["paused"] is False
    assert request_build(tmp_path, session["id"], reason="manual")["status"] == "pending"


def test_dashboard_auto_setting_and_scheduler_uses_validated_helper(tmp_path):
    store, project, session = _seed(tmp_path)
    for index in range(3):
        _capture(store, project, session, index)
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    client = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    page = client.get(f"/ui?project_id={project['id']}")
    csrf = re.search(r"name='csrf' value='([^']+)'", page.text).group(1)
    assert client.post(f"/ui/sessions/{session['id']}/auto", data={
        "csrf": csrf, "enabled": "yes"}).status_code == 422
    changed = client.post(f"/ui/sessions/{session['id']}/auto", data={
        "csrf": csrf, "enabled": "yes", "reviewed_native_orientation": "yes"}, follow_redirects=True)
    assert changed.status_code == 200
    from bridge.jobs import splat_status, set_splat_updates
    assert policy(tmp_path, session["id"])["auto_enabled"] is False
    assert splat_status(tmp_path, session["id"])["enabled"] is True
    assert "Pause live splat" in changed.text
    assert request_status(tmp_path, project["id"]) == []
    # Deliberate saved builds still use their existing validated helper.
    set_splat_updates(tmp_path, session["id"], False)
    request_build(tmp_path, session["id"], reason="manual")

    commands = []

    class FakeProcess:
        pid = 21212

        def poll(self):
            return None

    def launch(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    with patch("bridge.jobs.shutil.which", return_value="powershell.exe"), patch(
        "bridge.jobs.subprocess.Popen", side_effect=launch
    ):
        Scheduler(tmp_path).tick()
    assert commands and commands[0][0] == "powershell.exe"
    assert commands[0][commands[0].index("-Quality") + 1] == "quick"
    assert commands[0][commands[0].index("-SessionId") + 1] == session["id"]
    assert "-ReviewedNativeOrientation" in commands[0]
    assert request_status(tmp_path, project["id"])[0]["status"] == "running"
