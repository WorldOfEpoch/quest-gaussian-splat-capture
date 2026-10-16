import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from bridge.app import create_app
from bridge.config import load_config
from bridge.jobs import set_processing_paused
from bridge.runtime import data_root_id, probe


TEST_TOKEN = 'synthetic-runtime-test-' + 'x' * 32


@pytest.fixture(autouse=True)
def synthetic_runtime_token(monkeypatch):
    # probe() loads config internally; never let an ambient pairing token enter
    # the local HTTP fixture or its assertion output.
    monkeypatch.setenv('QUEST_SPLAT_TOKEN', TEST_TOKEN)


def test_authenticated_runtime_reports_workspace_and_preserves_pause(tmp_path):
    config = load_config(tmp_path, token=TEST_TOKEN)
    set_processing_paused(tmp_path, True)
    client = TestClient(create_app(config, dashboard_enabled=True))
    assert client.get('/api/v1/runtime').status_code == 401
    response = client.get('/api/v1/runtime', headers={'Authorization': f'Bearer {config.token}'})
    assert response.json() == {'status': 'ok', 'protocol_version': '0.1',
                               'data_root_id': data_root_id(tmp_path),
                               'dashboard_enabled': True, 'processing_paused': True}
    assert str(tmp_path) not in response.text
    assert config.token not in response.text


def test_launcher_refuses_plaintext_lan(tmp_path):
    with pytest.raises(ValueError, match='HTTPS'):
        probe(tmp_path, 'http://192.168.1.10:8765', dashboard=False, wait=0)


def test_dashboard_exposes_persisted_build_pause_without_resuming(tmp_path):
    config = load_config(tmp_path, token=TEST_TOKEN)
    app = create_app(config, dashboard_enabled=True)
    project = app.state.store.create_project('Review')
    client = TestClient(app, base_url='http://127.0.0.1:8765', client=('127.0.0.1', 50000))
    set_processing_paused(tmp_path, True)
    assert 'Saved detailed builds are paused.' in client.get('/ui').text
    assert "id='model-batch-link'" in client.get('/ui').text
    route = f"/ui/build-status?project_id={project['id']}"
    assert client.get(route).json()['processing_paused'] is True
    set_processing_paused(tmp_path, False)
    assert 'Build queue is available.' in client.get('/ui').text
    assert client.get(route).json()['processing_paused'] is False


@pytest.mark.parametrize('case', ['correct', 'wrong-workspace', 'wrong-role', 'redirect'])
def test_launcher_probe_does_not_accept_wrong_services_or_forward_credentials(tmp_path, case):
    config = load_config(tmp_path, token=TEST_TOKEN)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get('Authorization')))
            if case == 'redirect':
                self.send_response(302)
                self.send_header('Location', '/should-not-receive-token')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'protocol_version': '0.1',
                'data_root_id': 'other' if case == 'wrong-workspace' else data_root_id(tmp_path),
                'dashboard_enabled': case != 'wrong-role', 'processing_paused': True}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_port}'
        if case == 'correct':
            assert probe(tmp_path, url, dashboard=True, wait=0)['processing_paused'] is True
        else:
            with pytest.raises(RuntimeError):
                probe(tmp_path, url, dashboard=True, wait=0)
        assert requests == [('/api/v1/runtime', f'Bearer {config.token}')]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
