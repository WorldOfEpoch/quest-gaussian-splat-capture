"""Headset new-room retries cannot duplicate or mix capture destinations."""
from concurrent.futures import ThreadPoolExecutor
import uuid

from bridge.store import Store
from test_api import client, project_session, capture_payload, upload


def test_new_room_retries_preserve_original_and_keep_capture_routes_separate(tmp_path):
    api, headers = client(tmp_path)
    assert 'idempotent_creation' in api.get('/health').json()['capabilities']
    home, home_session = project_session(api, headers)
    original_photo, data = capture_payload(home, home_session)
    assert upload(api, headers, home_session, original_photo, data).status_code == 200
    before = api.get(f"/api/v1/projects/{home['id']}/sessions", headers=headers).json()
    project_body = {"name": "Room 2026-09-29 08:30", "request_id": str(uuid.uuid4())}
    room = api.post('/api/v1/projects', headers=headers, json=project_body).json()
    assert room['id'] != home['id']
    session_body = {'project_id': room['id'], 'tracking_origin_id': str(uuid.uuid4()),
                    'request_id': str(uuid.uuid4())}
    room_session = api.post('/api/v1/sessions', headers=headers, json=session_body).json()
    # A timeout/transport switch or app restart reuses the same request IDs.
    restarted, _ = client(tmp_path)
    assert restarted.post('/api/v1/projects', headers=headers, json=project_body).json() == room
    assert restarted.post('/api/v1/sessions', headers=headers, json=session_body).json() == room_session
    assert len(restarted.get('/api/v1/projects', headers=headers).json()['projects']) == 2
    assert len(restarted.get(f"/api/v1/projects/{room['id']}/entries", headers=headers).json()['entries']) == 1
    assert restarted.get(f"/api/v1/projects/{home['id']}/sessions", headers=headers).json() == before
    # Old queued metadata must be refused if misrouted to the new destination.
    assert upload(restarted, headers, room_session, original_photo, data).status_code == 409
    new_photo, new_data = capture_payload(room, room_session)
    assert upload(restarted, headers, room_session, new_photo, new_data).status_code == 200
    assert restarted.get(f"/api/v1/sessions/{home_session['id']}", headers=headers).json()['capture_count'] == 1
    assert restarted.get(f"/api/v1/sessions/{room_session['id']}", headers=headers).json()['capture_count'] == 1
    assert (tmp_path/'projects'/home['id']/'captures'/f"{original_photo['capture_id']}.jpg").read_bytes() == data


def test_creation_request_conflicts_are_explicit(tmp_path):
    api, headers = client(tmp_path)
    request_id = str(uuid.uuid4())
    project = api.post('/api/v1/projects', headers=headers,
                       json={'name':'Room', 'request_id':request_id}).json()
    assert api.post('/api/v1/projects', headers=headers,
                    json={'name':'Different room', 'request_id':request_id}).status_code == 409
    session_body = {'project_id':project['id'], 'tracking_origin_id':str(uuid.uuid4()),
                    'request_id':request_id}
    assert api.post('/api/v1/sessions', headers=headers, json=session_body).status_code == 409
    session_body['request_id'] = str(uuid.uuid4())
    assert api.post('/api/v1/sessions', headers=headers, json=session_body).status_code == 200
    session_body['tracking_origin_id'] = str(uuid.uuid4())
    assert api.post('/api/v1/sessions', headers=headers, json=session_body).status_code == 409
    assert len(api.get(f"/api/v1/projects/{project['id']}/sessions", headers=headers).json()['sessions']) == 1


def test_simultaneous_receiver_retries_create_one_project_and_session(tmp_path):
    # Independent Store objects mimic independent USB/LAN receiver processes.
    stores = [Store(tmp_path), Store(tmp_path)]
    request_id = str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=2) as pool:
        projects = list(pool.map(lambda store: store.create_project('New room',request_id), stores))
    assert projects[0] == projects[1]
    assert len(stores[0].projects()) == 1
    assert len(stores[0].entries(projects[0]['id'])) == 1
    session_request, origin = str(uuid.uuid4()), str(uuid.uuid4())
    with ThreadPoolExecutor(max_workers=2) as pool:
        sessions = list(pool.map(lambda store: store.create_session(projects[0]['id'],origin,session_request), stores))
    assert sessions[0] == sessions[1]
    assert len(stores[0].sessions(projects[0]['id'])) == 1


def test_first_scan_session_in_existing_project_preserves_all_six_projects(tmp_path):
    api, headers = client(tmp_path)
    originals = [api.post('/api/v1/projects', headers=headers,
                         json={'name': name}).json()
                 for name in ['Home', 'Interior', 'Exterior', 'Object', 'Other', 'Room October']]
    room = originals[-1]
    before = api.get('/api/v1/projects', headers=headers).json()
    # Confirmation can be cancelled before mutation: project/session GETs are read-only.
    assert api.get(f"/api/v1/projects/{room['id']}", headers=headers).json() == room
    assert api.get(f"/api/v1/projects/{room['id']}/sessions", headers=headers).json()['sessions'] == []
    assert api.get('/api/v1/projects', headers=headers).json() == before
    body = {'project_id': room['id'], 'tracking_origin_id': str(uuid.uuid4()),
            'request_id': str(uuid.uuid4())}
    session = api.post('/api/v1/sessions', headers=headers, json=body).json()
    assert session['project_id'] == room['id'] and session['capture_count'] == 0
    assert session['tracking_origin_id'] == body['tracking_origin_id']
    # Transport failover and reopening reuse the persisted session request.
    restarted, _ = client(tmp_path)
    assert restarted.post('/api/v1/sessions', headers=headers, json=body).json() == session
    assert restarted.get('/api/v1/projects', headers=headers).json() == before
    assert len(restarted.get('/api/v1/projects', headers=headers).json()['projects']) == 6
    assert restarted.get(f"/api/v1/projects/{room['id']}", headers=headers).json() == room
    for untouched in originals[:-1]:
        assert restarted.get(f"/api/v1/projects/{untouched['id']}/sessions", headers=headers).json()['sessions'] == []
    sessions = restarted.get(f"/api/v1/projects/{room['id']}/sessions", headers=headers).json()['sessions']
    assert sessions == [session]
    assert restarted.get(f"/api/v1/sessions/{session['id']}/captures", headers=headers).json()['captures'] == []
