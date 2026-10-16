"""Pure protocol client and human-readable catalog navigation.

No bpy import, network daemon, or global Blender state belongs in this module.
"""

from __future__ import annotations

import json
import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid


class BridgeError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Pairing credentials and model files belong only to the configured
        # bridge. Let HTTPError report every redirect without following it.
        return None


def token_file_path(environment: dict[str, str] | None = None, platform_name: str | None = None, home: Path | None = None) -> Path:
    """Match the bridge's default data directory without storing secrets in .blend."""
    env = os.environ if environment is None else environment
    system = os.name if platform_name is None else platform_name
    user_home = Path.home() if home is None else home
    configured = env.get("QUEST_SPLAT_DATA_DIR")
    if configured:
        data_dir = Path(configured).expanduser()
    elif system == "nt":
        local_app_data = env.get("LOCALAPPDATA")
        data_dir = Path(local_app_data) / "QuestSplatBridge" if local_app_data else user_home / "AppData" / "Local" / "QuestSplatBridge"
    else:
        data_dir = user_home / "QuestSplatBridge"
    return data_dir / "bridge.token"


def _uuid_path(value: str) -> str:
    # A path segment must never permit traversal or query insertion. The API
    # identity is UUID, which has a deliberately narrow alphabet.
    import uuid

    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Expected a UUID identity") from exc


def _name(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 120:
        raise ValueError("Name must contain 1 to 120 characters")
    return normalized


class BridgeClient:
    """Synchronous client; caller must run methods outside Blender's UI thread."""

    def __init__(self, base_url: str, token: str, timeout: float = 10.0):
        parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Bridge URL must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Bridge URL cannot contain credentials, query, or fragment")
        host = parsed.hostname or ""
        if parsed.scheme == "http" and host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Remote bridge connections require HTTPS")
        if not token.strip():
            raise ValueError("Set QUEST_SPLAT_TOKEN in Blender's environment")
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect())

    def _request(self, method: str, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if data is None else json.dumps(data, separators=(",", ":")).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + "/api/v1" + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except (UnicodeError, ValueError, AttributeError):
                detail = None
            raise BridgeError(f"Bridge HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BridgeError(f"Bridge connection failed: {exc}") from exc
        except (UnicodeError, ValueError) as exc:
            raise BridgeError("Bridge returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise BridgeError("Bridge returned an unexpected response")
        return payload

    def list_projects(self) -> list[dict[str, Any]]:
        return self._request("GET", "/projects")["projects"]

    def create_project(self, name: str) -> dict[str, Any]:
        return self._request("POST", "/projects", {"name": _name(name)})

    def get_project(self, project_id: str) -> dict[str, Any]:
        return self._request("GET", f"/projects/{_uuid_path(project_id)}")

    def list_entries(self, project_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/projects/{_uuid_path(project_id)}/entries")["entries"]

    def list_models(self, project_id: str, entry_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/projects/{_uuid_path(project_id)}/entries/{_uuid_path(entry_id)}/models")["models"]

    def create_entry(self, project_id: str, name: str, kind: str, parent_id: str | None, sort_order: int = 0) -> dict[str, Any]:
        if kind not in {"category", "design", "variation"}:
            raise ValueError("Unknown library entry kind")
        return self._request(
            "POST",
            f"/projects/{_uuid_path(project_id)}/entries",
            {"name": _name(name), "kind": kind, "parent_id": _uuid_path(parent_id) if parent_id else None, "sort_order": int(sort_order)},
        )

    def update_entry(self, project_id: str, entry_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {"name", "parent_id", "sort_order"}
        if not changes or not set(changes).issubset(allowed):
            raise ValueError("Only entry name, parent, and sort order may change")
        if "name" in changes:
            changes["name"] = _name(changes["name"])
        if "parent_id" in changes and changes["parent_id"] is not None:
            changes["parent_id"] = _uuid_path(changes["parent_id"])
        return self._request("PATCH", f"/projects/{_uuid_path(project_id)}/entries/{_uuid_path(entry_id)}", changes)

    def list_sessions(self, project_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/projects/{_uuid_path(project_id)}/sessions")["sessions"]

    def create_session(self, project_id: str, tracking_origin_id: str) -> dict[str, Any]:
        return self._request("POST", "/sessions", {"project_id": _uuid_path(project_id), "tracking_origin_id": _uuid_path(tracking_origin_id)})

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sessions/{_uuid_path(session_id)}")

    def finish_session(self, session_id: str) -> dict[str, Any]:
        return self._request("POST", f"/sessions/{_uuid_path(session_id)}/finish", {})

    def get_preview(self, session_id: str, representation: str = "splat") -> dict[str, Any]:
        if representation not in {"splat", "mesh"}:
            raise ValueError("Unknown preview representation")
        result = self._request("GET", f"/sessions/{_uuid_path(session_id)}/preview?representation={representation}")
        if result.get("status") not in {"not_ready", "ready"}:
            raise BridgeError("Bridge returned an unknown preview state")
        if result["status"] == "ready" and not result.get("artifact"):
            raise BridgeError("Ready preview has no artifact manifest")
        return result

    def publish_model(
        self, project_id: str, entry_id: str, model_path: str | Path,
        *, format: str, transform: list[float], coordinate_frame: str,
        source_job_id: str | None, display_name: str,
    ) -> dict[str, Any]:
        """Attach one immutable local artifact; safe retries use a stable revision ID.

        This method is called from Blender's worker only. Publication is scoped
        to loopback even if the configured bridge permits remote HTTPS reads.
        """
        if urllib.parse.urlsplit(self.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise BridgeError("Model publication is restricted to the local PC bridge")
        project_id, entry_id = _uuid_path(project_id), _uuid_path(entry_id)
        if format not in {"gaussian_ply", "glb"} or coordinate_frame not in {"colmap", "quest_session"}:
            raise ValueError("Unsupported model format or coordinate frame")
        import math
        if len(transform) != 16 or not all(math.isfinite(float(value)) for value in transform):
            raise ValueError("Model transform must have 16 finite values")
        if source_job_id:
            source_job_id = _uuid_path(source_job_id)
        source = Path(model_path).expanduser().resolve(strict=True)
        if source.suffix.lower() != (".ply" if format == "gaussian_ply" else ".glb"):
            raise ValueError("Model file suffix does not match format")
        if not source.is_file() or source.stat().st_size < 1:
            raise ValueError("Model file is empty")
        artifact = source.read_bytes()
        sha = hashlib.sha256(artifact).hexdigest()
        metadata = {
            "schema_version": 1,
            "format": format,
            "artifact_sha256": sha,
            "source_job_id": source_job_id,
            "transform": [float(value) for value in transform],
            "coordinate_frame": coordinate_frame,
            "display_name": _name(display_name),
        }
        identity = json.dumps([project_id, entry_id, metadata], sort_keys=True, separators=(",", ":"))
        metadata["revision_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "quest-splat-model:" + identity))
        boundary = "quest-splat-" + uuid.uuid4().hex
        body = b"".join((
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"metadata_json\"\r\nContent-Type: application/json\r\n\r\n".encode("ascii"),
            json.dumps(metadata, separators=(",", ":")).encode("utf-8"), b"\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"; filename=\"model{source.suffix.lower()}\"\r\nContent-Type: {'application/octet-stream' if format == 'gaussian_ply' else 'model/gltf-binary'}\r\n\r\n".encode("ascii"),
            artifact, b"\r\n", f"--{boundary}--\r\n".encode("ascii"),
        ))
        request = urllib.request.Request(
            self.base_url + f"/api/v1/projects/{project_id}/entries/{entry_id}/models",
            data=body,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json",
                     "Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=max(self.timeout, 120.0)) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except (UnicodeError, ValueError, AttributeError):
                detail = None
            raise BridgeError(f"Bridge HTTP {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BridgeError(f"Bridge upload failed: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("revision_id") != metadata["revision_id"]:
            raise BridgeError("Bridge returned an unexpected model revision")
        return payload


@dataclass(frozen=True)
class CatalogView:
    project_name: str
    parent_id: str | None
    breadcrumb: tuple[str, ...]
    children: tuple[dict[str, Any], ...]


def catalog_view(project: dict[str, Any], entries: list[dict[str, Any]], parent_id: str | None) -> CatalogView:
    """Return one human-readable level, validating the breadcrumb shape."""
    by_id = {entry["id"]: entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("Duplicate catalog identities")
    if parent_id is not None and parent_id not in by_id:
        raise ValueError("Catalog entry no longer exists")
    names = [project["name"]]
    cursor = parent_id
    seen: set[str] = set()
    while cursor is not None:
        if cursor in seen:
            raise ValueError("Catalog cycle")
        seen.add(cursor)
        entry = by_id[cursor]
        names.append(entry["name"])
        cursor = entry["parent_id"]
        if cursor is not None and cursor not in by_id:
            raise ValueError("Catalog parent is missing")
    children = tuple(sorted(
        (entry for entry in entries if entry["parent_id"] == parent_id),
        key=lambda item: (item["sort_order"], item["name"].casefold(), item["id"]),
    ))
    return CatalogView(project["name"], parent_id, tuple([names[0], *reversed(names[1:])]), children)


def reorder_plan(entries: list[dict[str, Any]], entry_id: str, direction: int) -> list[tuple[str, int]]:
    """Return sibling sort updates that move one entry a single position."""
    if direction not in {-1, 1}:
        raise ValueError("Direction must be -1 or 1")
    target = next((entry for entry in entries if entry["id"] == entry_id), None)
    if target is None:
        raise ValueError("Catalog entry no longer exists")
    siblings = sorted(
        (entry for entry in entries if entry["parent_id"] == target["parent_id"]),
        key=lambda item: (item["sort_order"], item["name"].casefold(), item["id"]),
    )
    position = next(index for index, entry in enumerate(siblings) if entry["id"] == entry_id)
    destination = position + direction
    if destination < 0 or destination >= len(siblings):
        return []
    siblings[position], siblings[destination] = siblings[destination], siblings[position]
    return [(entry["id"], index * 10) for index, entry in enumerate(siblings)]
