import io
import json
import threading
import unittest
import uuid
import urllib.error
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import re

import pytest

from blender_addon.quest_splat_bridge.client import BridgeClient, BridgeError, catalog_view, reorder_plan, token_file_path


class _Response:
    def __init__(self, data):
        self._data = io.BytesIO(json.dumps(data).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, *args):
        return self._data.read(*args)


class BridgeClientTests(unittest.TestCase):
    def setUp(self):
        self.client = BridgeClient("http://127.0.0.1:8765", "test-token")
        self.pid = str(uuid.UUID(int=1))
        self.eid = str(uuid.UUID(int=2))

    def test_create_project_sends_authenticated_trimmed_name(self):
        with patch.object(self.client._opener, "open", return_value=_Response({"id": self.pid})) as open_url:
            self.assertEqual(self.client.create_project("  Backyard  ")["id"], self.pid)
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8765/api/v1/projects")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
        self.assertEqual(json.loads(request.data), {"name": "Backyard"})

    def test_preview_not_ready_remains_not_ready(self):
        reply = {"status": "not_ready", "representation": "splat", "reason": "Worker unavailable", "artifact": None}
        with patch.object(self.client._opener, "open", return_value=_Response(reply)):
            self.assertEqual(self.client.get_preview(self.pid), reply)

    def test_ready_without_artifact_is_rejected(self):
        with patch.object(self.client._opener, "open", return_value=_Response({"status": "ready", "artifact": None})):
            with self.assertRaises(BridgeError):
                self.client.get_preview(self.pid)

    def test_nonlocal_cleartext_and_path_injection_rejected(self):
        with self.assertRaises(ValueError):
            BridgeClient("http://192.0.2.10:8765", "token")
        with self.assertRaises(ValueError):
            self.client.get_project("../other")
        with self.assertRaises(ValueError):
            self.client.create_entry(self.pid, "Name", "design", "../other")

    def test_http_error_surfaces_bridge_detail(self):
        error = urllib.error.HTTPError(
            "http://127.0.0.1", 409, "Conflict", {}, io.BytesIO(b'{"detail":"Name already exists"}')
        )
        with patch.object(self.client._opener, "open", side_effect=error):
            with self.assertRaisesRegex(BridgeError, "Name already exists"):
                self.client.create_project("Backyard")

    def test_entry_update_only_allows_safe_fields(self):
        with self.assertRaises(ValueError):
            self.client.update_entry(self.pid, self.eid, kind="category")

    def test_token_path_matches_bridge_windows_default_and_override(self):
        with TemporaryDirectory() as directory:
            local_app_data = Path(directory) / "AppData" / "Local"
            configured_data = Path(directory) / "configured-data"
            environment = {"LOCALAPPDATA": str(local_app_data)}
            self.assertEqual(
                token_file_path(environment, platform_name="nt"),
                local_app_data / "QuestSplatBridge" / "bridge.token",
            )
            environment["QUEST_SPLAT_DATA_DIR"] = str(configured_data)
            self.assertEqual(
                token_file_path(environment, platform_name="nt"),
                configured_data / "bridge.token",
            )

    def test_publish_model_is_authenticated_binary_and_idempotent(self):
        with TemporaryDirectory() as folder:
            model = Path(folder) / "edited.ply"
            model.write_bytes(b"ply\nGaussian test content\0\xff")
            observed = []
            def fake_open(request, timeout):
                self.assertGreaterEqual(timeout, 120)
                self.assertEqual(request.get_method(), "POST")
                self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
                self.assertIn(f"/projects/{self.pid}/entries/{self.eid}/models", request.full_url)
                body = request.data
                self.assertIn(model.read_bytes(), body)
                match = re.search(rb'\r\n\r\n(\{.*?\})\r\n--', body)
                metadata = json.loads(match.group(1))
                self.assertEqual(metadata["artifact_sha256"], hashlib.sha256(model.read_bytes()).hexdigest())
                self.assertEqual(metadata["source_job_id"], self.pid)
                observed.append(metadata)
                return _Response({"revision_id": metadata["revision_id"], "duplicate": len(observed) > 1})
            with patch.object(self.client._opener, "open", side_effect=fake_open):
                arguments = dict(format="gaussian_ply", transform=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
                                 coordinate_frame="colmap", source_job_id=self.pid, display_name="Edited room test")
                first = self.client.publish_model(self.pid, self.eid, model, **arguments)
                second = self.client.publish_model(self.pid, self.eid, model, **arguments)
            self.assertEqual(first["revision_id"], second["revision_id"])
            self.assertEqual(observed[0], observed[1])

    def test_publish_model_rejects_remote_bridge(self):
        with self.assertRaisesRegex(BridgeError, "local PC bridge"):
            BridgeClient("https://example.com", "token").publish_model(
                self.pid, self.eid, "missing.ply", format="gaussian_ply",
                transform=[1]*16, coordinate_frame="colmap", source_job_id=None,
                display_name="Test")


@contextmanager
def _local_http_fixture(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("operation", ["get", "json", "multipart"])
def test_redirect_never_transmits_pairing_token_to_another_origin(tmp_path, monkeypatch, status, operation):
    # Different loopback ports are different origins. Neither server is a real
    # bridge, and both the upload and token are disposable synthetic fixtures.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    synthetic_token = "synthetic-redirect-test-" + "x" * 32
    origin_requests = []
    target_requests = []

    class QuietHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def discard_body(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))

    class TargetHandler(QuietHandler):
        def do_GET(self):
            self.discard_body()
            target_requests.append((self.command, self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"projects":[]}')

        do_POST = do_GET

    with _local_http_fixture(TargetHandler) as target:
        class OriginHandler(QuietHandler):
            def do_GET(self):
                self.discard_body()
                origin_requests.append((self.command, self.headers.get("Authorization")))
                self.send_response(status)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/redirect-target")
                self.end_headers()

            do_POST = do_GET

        with _local_http_fixture(OriginHandler) as origin:
            client = BridgeClient(f"http://127.0.0.1:{origin.server_port}", synthetic_token, timeout=2)
            model = tmp_path / "synthetic.ply"
            model.write_bytes(b"ply\nSynthetic model bytes\n")
            with pytest.raises(BridgeError, match=f"Bridge HTTP {status}:"):
                if operation == "get":
                    client.list_projects()
                elif operation == "json":
                    client.create_project("Synthetic project")
                else:
                    client.publish_model(
                        str(uuid.UUID(int=1)), str(uuid.UUID(int=2)), model,
                        format="gaussian_ply", transform=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
                        coordinate_frame="colmap", source_job_id=None,
                        display_name="Synthetic model",
                    )

            expected_method = "GET" if operation == "get" else "POST"
            assert origin_requests == [(expected_method, f"Bearer {synthetic_token}")]
            assert target_requests == []


class CatalogViewTests(unittest.TestCase):
    def setUp(self):
        self.pid = "project"
        self.project = {"id": self.pid, "name": "House outside makeover"}
        self.entries = [
            {"id": "original", "parent_id": None, "kind": "category", "name": "Original", "sort_order": 0},
            {"id": "additions", "parent_id": None, "kind": "category", "name": "Additions", "sort_order": 1},
            {"id": "rear", "parent_id": "additions", "kind": "design", "name": "Rear family room", "sort_order": 0},
            {"id": "flat", "parent_id": "rear", "kind": "variation", "name": "Flat roof", "sort_order": 0},
            {"id": "pitched", "parent_id": "rear", "kind": "variation", "name": "Pitched roof", "sort_order": 1},
        ]

    def test_current_level_and_breadcrumb(self):
        view = catalog_view(self.project, self.entries, "rear")
        self.assertEqual(view.breadcrumb, ("House outside makeover", "Additions", "Rear family room"))
        self.assertEqual([child["name"] for child in view.children], ["Flat roof", "Pitched roof"])
        root = catalog_view(self.project, self.entries, None)
        self.assertEqual([child["name"] for child in root.children], ["Original", "Additions"])

    def test_broken_parent_or_cycle_is_rejected(self):
        broken = [dict(entry) for entry in self.entries]
        broken[2]["parent_id"] = "missing"
        with self.assertRaises(ValueError):
            catalog_view(self.project, broken, "rear")
        cycle = [dict(entry) for entry in self.entries]
        cycle[1]["parent_id"] = "rear"
        with self.assertRaises(ValueError):
            catalog_view(self.project, cycle, "rear")

    def test_reorder_moves_one_sibling_only(self):
        plan = reorder_plan(self.entries, "pitched", -1)
        self.assertEqual(plan, [("pitched", 0), ("flat", 10)])
        self.assertEqual(reorder_plan(self.entries, "original", -1), [])


if __name__ == "__main__":
    unittest.main()
