import hashlib
import json
import struct
import uuid

from fastapi.testclient import TestClient

from bridge.app import create_app
from bridge.config import Config


def _client(tmp_path):
    token = "t" * 48
    api = TestClient(create_app(Config(tmp_path, token)))
    headers = {"Authorization": f"Bearer {token}"}
    project = api.post("/api/v1/projects", headers=headers, json={"name": "House"}).json()
    category = api.post(f"/api/v1/projects/{project['id']}/entries", headers=headers,
                        json={"name": "Additions", "kind": "category"}).json()
    design = api.post(f"/api/v1/projects/{project['id']}/entries", headers=headers,
                      json={"name": "Rear room", "kind": "design", "parent_id": category["id"]}).json()
    return api, headers, project, design


def _glb():
    document = json.dumps({"asset": {"version": "2.0"}, "scenes": [{"nodes": []}], "scene": 0},
                          separators=(",", ":")).encode()
    document += b" " * ((-len(document)) % 4)
    return struct.pack("<4sII", b"glTF", 2, 20 + len(document)) + struct.pack("<I4s", len(document), b"JSON") + document


def _ply():
    properties = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]
    header = ("ply\nformat binary_little_endian 1.0\ncomment Vertical Axis: z\n"
              "element vertex 2\n" + "".join(f"property float {name}\n" for name in properties)
              + "end_header\n").encode("ascii")
    return header + struct.pack("<28f", *([0.0] * 28))


def _upload(api, headers, project_id, entry_id, metadata, data):
    return api.post(f"/api/v1/projects/{project_id}/entries/{entry_id}/models", headers=headers,
                    data={"metadata_json": json.dumps(metadata)},
                    files={"model": ("ignored-name.bin", data, "application/octet-stream")})


def test_model_revision_auth_list_download_and_immutable_retry(tmp_path):
    api, headers, project, design = _client(tmp_path)
    data = _glb()
    revision = str(uuid.uuid4())
    metadata = {"schema_version": 1, "revision_id": revision, "format": "glb",
                "artifact_sha256": hashlib.sha256(data).hexdigest(), "coordinate_frame": "colmap"}
    route = f"/api/v1/projects/{project['id']}/entries/{design['id']}/models"
    assert _upload(api, {}, project["id"], design["id"], metadata, data).status_code == 401
    first = _upload(api, headers, project["id"], design["id"], metadata, data)
    assert first.status_code == 200, first.text
    record = first.json()
    assert record["entry_id"] == design["id"] and record["duplicate"] is False
    assert record["placement_status"] == "unverified"
    assert record["transform"] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert str(tmp_path) not in first.text
    assert _upload(api, headers, project["id"], design["id"], metadata, data).json()["duplicate"] is True
    assert len(api.get(route, headers=headers).json()["models"]) == 1
    assert len(api.get(f"/api/v1/projects/{project['id']}/models", headers=headers).json()["models"]) == 1
    assert api.get(record["artifact_url"]).status_code == 401
    downloaded = api.get(record["artifact_url"], headers=headers)
    assert downloaded.content == data and downloaded.headers["content-type"] == "model/gltf-binary"
    assert _upload(api, headers, project["id"], design["id"], dict(metadata, display_name="Changed"), data).status_code == 409
    second = api.post("/api/v1/projects", headers=headers, json={"name": "Other"}).json()
    assert _upload(api, headers, second["id"], design["id"], metadata, data).status_code == 404
    assert api.get(record["artifact_url"].replace(project["id"], second["id"]), headers=headers).status_code == 404
    stored = tmp_path / "projects" / project["id"] / "models" / revision / "model.glb"
    stored.write_bytes(data + b"tampered")
    assert api.get(record["artifact_url"], headers=headers).status_code == 404


def test_gaussian_sidecar_and_source_validation(tmp_path):
    api, headers, project, design = _client(tmp_path)
    data = _ply()
    metadata = {"schema_version": 1, "revision_id": str(uuid.uuid4()), "format": "gaussian_ply",
                "artifact_sha256": hashlib.sha256(data).hexdigest(), "coordinate_frame": "colmap"}
    assert _upload(api, headers, project["id"], design["id"], metadata, data).status_code == 422
    matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    metadata["transform"] = matrix
    metadata["source_job_id"] = str(uuid.uuid4())
    assert _upload(api, headers, project["id"], design["id"], metadata, data).status_code == 404
    metadata["source_job_id"] = None
    published = _upload(api, headers, project["id"], design["id"], metadata, data)
    assert published.status_code == 200, published.text
    revision = published.json()["revision_id"]
    sidecar = json.loads((tmp_path / "projects" / project["id"] / "models" / revision / "transform.json").read_text())
    assert sidecar["coordinate_frame"] == "colmap" and sidecar["transform"] == matrix
    assert _upload(api, headers, project["id"], design["id"], dict(metadata, revision_id=str(uuid.uuid4()),
        transform=[float("nan")] * 16), data).status_code == 422
    assert _upload(api, headers, project["id"], design["id"], dict(metadata, revision_id=str(uuid.uuid4()),
        artifact_sha256="0" * 64), data).status_code == 422


def test_model_source_job_is_project_scoped_and_dashboard_download_local(tmp_path):
    api, headers, project, design = _client(tmp_path)
    other = api.post("/api/v1/projects", headers=headers, json={"name": "Other house"}).json()
    job_id = str(uuid.uuid4())
    other_job = tmp_path / "projects" / other["id"] / "reconstructions" / job_id
    other_job.mkdir(parents=True)
    ply = other_job / "splat.ply"
    ply.write_bytes(_ply())
    (other_job / "job.json").write_text(json.dumps({
        "project_id": other["id"], "job_id": job_id, "status": "ready_for_blender",
        "artifact": {"path": str(ply), "sha256": hashlib.sha256(ply.read_bytes()).hexdigest(),
                     "size_bytes": ply.stat().st_size},
    }))
    data = _glb()
    metadata = {"schema_version": 1, "revision_id": str(uuid.uuid4()), "format": "glb",
                "artifact_sha256": hashlib.sha256(data).hexdigest(), "source_job_id": job_id}
    assert _upload(api, headers, project["id"], design["id"], metadata, data).status_code == 404
    metadata["source_job_id"] = None
    record = _upload(api, headers, project["id"], design["id"], metadata, data).json()
    local = TestClient(create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True),
                       base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    page = local.get(f"/ui?project_id={project['id']}")
    assert "Additions › Rear room" in page.text and "Download" in page.text
    assert local.get(f"/ui/models/{project['id']}/{record['revision_id']}/artifact").content == data
    lan = TestClient(create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True),
                     base_url="http://192.168.0.136:8765", client=("192.168.0.38", 50000))
    assert lan.get(f"/ui/models/{project['id']}/{record['revision_id']}/artifact").status_code == 403
