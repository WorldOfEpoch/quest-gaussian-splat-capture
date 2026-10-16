"""Automatic observed-depth intake uses one protected room state and durable receipts."""
import json
import re
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from bridge.app import create_app
from bridge.config import Config
from bridge.dashboard import set_capture_selection
from bridge.jobs import (Scheduler, depth_status, on_capture_received, processing_paused,
                         request_status, set_depth_updates, set_processing_paused)
from bridge.preview import read_depth_preview
from bridge.store import StoreError
from test_session_spatial_map import room, capture, state_path, surfaces, originals


def start(room):
    store, _, session = room
    set_processing_paused(store.root, True)
    set_depth_updates(store.root, session['id'], True)
    return Scheduler(store.root)


def test_same_room_advances_after_upload_and_protects_supported_geometry(room):
    store, project, session = room
    ids = [capture(room, i) for i in range(3)]
    worker = start(room)
    with patch('bridge.jobs.subprocess.Popen') as launch:
        worker.tick()
        first = json.loads(state_path(room).read_text())
        old = surfaces(first)[0]
        before = originals(room)
        new = capture(room, 3, point=(1.019, .232, .232))
        on_capture_received(store.root, session['id'])
        worker.tick()
        launch.assert_not_called()
    current = json.loads(state_path(room).read_text())
    preserved = next(s for s in surfaces(current) if s['id'] == old['id'])
    assert preserved['point'] == old['point']
    assert preserved['normal'] == old['normal']
    assert current['revision_number'] > first['revision_number']
    assert set(current['input_ledger']) == set(ids + [new])
    status = depth_status(store.root, session['id'])
    assert (status['status'], status['accepted_views'], status['total_views']) == ('ready', 4, 4)
    assert processing_paused(store.root) and request_status(store.root, project['id']) == []
    assert all(originals(room)[p] == digest for p, digest in before.items())


def test_bounded_batches_restart_reconciliation_and_byte_identical_idle(room):
    store, _, session = room
    [capture(room, i) for i in range(9)]
    worker = start(room)
    for admitted, pending in ((4, 5), (8, 1), (9, 0)):
        worker.tick_depth()
        status = depth_status(store.root, session['id'])
        assert (status['accepted_views'], status['pending_views']) == (admitted, pending)
        worker = Scheduler(store.root)  # no receipt callback needed after a restart
    content = state_path(room).read_bytes()
    worker.tick_depth()
    assert state_path(room).read_bytes() == content
    assert len(json.loads(content)['input_ledger']) == 9


def test_disabled_by_default_and_pause_preserves_current_map(room):
    store, _, session = room
    capture(room, 0)
    Scheduler(store.root).tick_depth()
    assert not state_path(room).exists()
    worker = start(room); worker.tick_depth()
    content = state_path(room).read_bytes()
    set_depth_updates(store.root, session['id'], False)
    capture(room, 1); worker.tick_depth()
    assert state_path(room).read_bytes() == content
    set_depth_updates(store.root, session['id'], True)
    worker.tick_depth()
    assert depth_status(store.root, session['id'])['accepted_views'] == 2


def test_unusable_new_depth_is_skipped_without_blocking_valid_neighbor(room):
    store, _, session = room
    capture(room, 0, sidecar=False)
    capture(room, 1, confidence=.1)
    capture(room, 2, status='no_hit')
    good = capture(room, 3)
    worker = start(room); worker.tick_depth()
    status = depth_status(store.root, session['id'])
    assert (status['accepted_views'], status['skipped_views'], status['pending_views']) == (1, 3, 0)
    assert set(json.loads(state_path(room).read_text())['input_ledger']) == {good}


def test_corrupt_excluded_file_does_not_block_but_changed_accepted_source_does(room):
    store, project, session = room
    good = capture(room, 0)
    excluded = capture(room, 1)
    set_capture_selection(store, project['id'], session['id'], excluded, False)
    paths = {row['metadata']['capture_id']: Path(row['image_path'])
             for row in store.export_session(session['id'])['captures']}
    paths[excluded].write_bytes(b'corrupt excluded original')
    worker = start(room); worker.tick_depth()
    assert depth_status(store.root, session['id'])['accepted_views'] == 1
    content = state_path(room).read_bytes()
    paths[good].write_bytes(b'corrupt accepted original')
    capture(room, 2)
    worker.tick_depth()
    assert depth_status(store.root, session['id'])['status'] == 'blocked'
    assert state_path(room).read_bytes() == content


def test_immutable_publication_can_finish_downloading_after_next_update(room):
    store, project, session = room
    capture(room, 0)
    worker = start(room); worker.tick_depth()
    first, path = read_depth_preview(store, project['id'], session['id'])
    old = path.read_bytes()
    capture(room, 1, point=(2.012, .232, .232)); worker.tick_depth()
    current, _ = read_depth_preview(store, project['id'], session['id'])
    saved, saved_path = read_depth_preview(store, project['id'], session['id'], first['state_revision'])
    assert saved['artifact'] == first['artifact']
    assert saved_path.read_bytes() == old
    assert current['revision_number'] > first['revision_number']
    assert current['accepted_views'] == 2
    saved_path.write_bytes(b'changed publication')
    with pytest.raises(StoreError):
        read_depth_preview(store, project['id'], session['id'], first['state_revision'])


def test_filtered_export_rejects_foreign_capture(room):
    store, project, session = room
    other = store.create_session(project['id'], __import__('uuid').uuid4().__str__())
    foreign = capture((store, project, other), 0)
    with pytest.raises(StoreError):
        store.export_session(session['id'], [foreign])


def test_idle_source_integrity_is_rechecked_without_another_upload(room):
    from bridge.jobs import _db
    store, _, session = room
    capture(room, 0)
    worker = start(room); worker.tick_depth()
    content = state_path(room).read_bytes()
    Path(store.export_session(session['id'])['captures'][0]['image_path']).write_bytes(b'changed')
    with _db(store.root) as db:
        db.execute("UPDATE depth_updates SET updated_utc='2000-01-01T00:00:00Z'")
    worker.tick_depth()
    assert depth_status(store.root, session['id'])['status'] == 'blocked'
    assert state_path(room).read_bytes() == content


def test_export_failure_reports_committed_map_separately_from_published_counts(room):
    store, project, session = room
    capture(room, 0)
    worker = start(room); worker.tick_depth()
    first, path = read_depth_preview(store, project['id'], session['id'])
    old = path.read_bytes()
    capture(room, 1)
    with patch('bridge.preview.read_depth_preview', side_effect=StoreError(404, 'test export failure')):
        worker.tick_depth()
    status = depth_status(store.root, session['id'])
    assert status['status'] == 'publication_blocked'
    assert status['committed_map_views'] == 2 and status['accepted_views'] == 1
    assert path.read_bytes() == old
    on_capture_received(store.root, session['id'])
    worker.tick_depth()
    assert depth_status(store.root, session['id'])['accepted_views'] == 2


def test_dashboard_controls_are_local_csrf_guarded_and_report_counts(room):
    store, project, session = room
    capture(room, 0)
    app = create_app(Config(store.root, 't' * 48), dashboard_enabled=True)
    client = TestClient(app, base_url='http://127.0.0.1:8765', client=('127.0.0.1', 50000))
    route = f"/ui/sessions/{session['id']}/depth"
    assert client.post(route, data={'enabled': 'yes'}).status_code in (403, 422)
    page = client.get(f"/ui?project_id={project['id']}")
    csrf = re.search(r"name='csrf' value='([^']+)'", page.text).group(1)
    assert client.post(route, data={'enabled': 'yes', 'csrf': csrf}, follow_redirects=False).status_code == 303
    Scheduler(store.root).tick_depth()
    status = client.get(f"/ui/projects/{project['id']}/depth-updates").json()['sessions'][0]
    assert status['enabled'] and status['accepted_views'] == status['total_views'] == 1
    remote = TestClient(app, base_url='http://127.0.0.1:8765', client=('192.168.0.10', 50000))
    assert remote.post(route, data={'enabled': 'no', 'csrf': csrf}).status_code == 403
    assert remote.get(f"/ui/projects/{project['id']}/depth-updates").status_code == 403


@pytest.mark.parametrize('change', ['pause', 'exclude'])
def test_controls_changed_during_fusion_preserve_previous_publication(room, change):
    from bridge import mesh_spatial
    store, project, session = room
    capture(room, 0)
    worker = start(room); worker.tick_depth()
    content = state_path(room).read_bytes()
    new = capture(room, 1)
    original = mesh_spatial.update_session_spatial_map
    def intercepted(*args, **kwargs):
        if change == 'pause':
            set_depth_updates(store.root, session['id'], False)
        else:
            set_capture_selection(store, project['id'], session['id'], new, False)
        return original(*args, **kwargs)
    with patch.object(mesh_spatial, 'update_session_spatial_map', intercepted):
        worker.tick_depth()
    assert state_path(room).read_bytes() == content
    status = depth_status(store.root, session['id'])
    assert status['accepted_views'] == 1
    if change == 'pause':
        assert status['status'] == 'paused' and not status['enabled']
        assert status.get('error') is None
