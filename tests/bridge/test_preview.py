import hashlib
import json
import struct
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bridge.app import create_app
from bridge.config import Config
from bridge.preview import fit_session_similarity, preview_manifest, read_job, read_online_preview
from bridge.reconstruct import make_quest_preview
from bridge.store import Store, StoreError


def poses():
    source = np.repeat(np.eye(4)[None], 4, axis=0)
    source[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, .2], [1, 1, .1]]
    return source


def test_similarity_preserves_meters_and_handedness():
    source = poses()
    flip = np.diag([1., 1., -1.])
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    target = poses()
    target[:, :3, :3] = rotation @ flip @ source[:, :3, :3] @ flip
    target[:, :3, 3] = 2.5 * (source[:, :3, 3] @ flip @ rotation.T) + [4, 2, 1]
    result = fit_session_similarity(source, target)
    assert result["headset_preview_allowed"]
    matrix = np.array(result["unity_from_splat"]).reshape(4, 4)
    transformed = (matrix @ np.column_stack((source[:, :3, 3] @ flip, np.ones(4))).T).T[:, :3]
    np.testing.assert_allclose(transformed, target[:, :3, 3], atol=1e-10)
    assert result["scale"] == pytest.approx(2.5)


def test_rotation_disagreement_and_stationary_photos_rejected():
    source = poses()
    target = poses()
    target[:, 2, 3] *= -1
    target[:, :3, :3] = np.diag([-1., -1., 1.])
    assert not fit_session_similarity(source, target)["headset_preview_allowed"]
    target[:, :3, 3] = 0
    with pytest.raises(ValueError, match="translation"):
        fit_session_similarity(source, target)


def make_job(store, session):
    job_id = str(uuid.uuid4())
    root = store.root / "projects" / session["project_id"] / "reconstructions" / job_id
    root.mkdir(parents=True)
    # Explicit synthetic fixture; never an operational build or physical receipt.
    payload = b"synthetic Gaussian test artifact" * 20
    ply = root / "splat.ply"
    ply.write_bytes(payload)
    job = {"project_id": session["project_id"], "session_id": session["id"], "job_id": job_id,
           "status": "ready_for_blender", "created_utc": "2026-09-28T00:00:00Z",
           "artifact": {"path": str(ply), "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload), "vertex_count": 4}}
    (root / "job.json").write_text(json.dumps(job))
    return job, ply


def test_artifact_digest_streams_on_python_without_file_digest(tmp_path, monkeypatch):
    store = Store(tmp_path)
    project = store.create_project("Portable checksum")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    job, path = make_job(store, session)
    payload = b"A" * (1024 * 1024 + 17)
    path.write_bytes(payload)
    job["artifact"].update(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    (path.parent / "job.json").write_text(json.dumps(job))
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert read_job(store, project["id"], job["job_id"])[1] == path
    path.write_bytes(b"B" + payload[1:])
    from bridge.preview import _digest
    _digest.cache_clear()  # Same-length corruption must be rehashed, even on coarse filesystems.
    with pytest.raises(StoreError):
        read_job(store, project["id"], job["job_id"])


def test_download_requires_auth_and_rejects_corrupted_artifact(tmp_path):
    token = "test-token-" * 5
    app = create_app(Config(tmp_path, token))
    store = app.state.store
    p = store.create_project("Synthetic test")
    s = store.create_session(p["id"], str(uuid.uuid4()))
    job, path = make_job(store, s)
    route = f"/api/v1/projects/{p['id']}/reconstructions/{job['job_id']}/splat"
    client = TestClient(app)
    assert client.get(route).status_code == 401
    headers = {"Authorization": "Bearer " + token}
    assert client.get(route, headers=headers).content == path.read_bytes()
    path.write_bytes(b"damaged")
    assert client.get(route, headers=headers).status_code == 404


def test_quest_copy_is_authenticated_and_bound_to_untouched_full_artifact(tmp_path):
    token = "test-token-" * 5
    app = create_app(Config(tmp_path, token))
    store = app.state.store
    project = store.create_project("Room test")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    job, full = make_job(store, session)
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 6\n" +
              "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    full.write_bytes(header + b"".join(struct.pack("<14f", float(i), *([0.] * 13)) for i in range(6)))
    job["artifact"].update(sha256=hashlib.sha256(full.read_bytes()).hexdigest(), size_bytes=full.stat().st_size,
                           vertex_count=6)
    small = full.with_name("quest-preview.ply")
    job["quest_preview"] = make_quest_preview(full, small, max_vertices=3)
    (full.parent / "job.json").write_text(json.dumps(job))
    route = f"/api/v1/projects/{project['id']}/reconstructions/{job['job_id']}/quest-splat"
    client = TestClient(app)
    assert client.get(route).status_code == 401
    headers = {"Authorization": "Bearer " + token}
    assert client.get(route, headers=headers).content == small.read_bytes()
    small.write_bytes(small.read_bytes() + b"tampered")
    assert client.get(route, headers=headers).status_code == 404
    assert client.get(route.replace(project["id"], str(uuid.uuid4())), headers=headers).status_code == 404


def test_other_session_is_only_an_inspection_model(tmp_path, monkeypatch):
    store = Store(tmp_path)
    p = store.create_project("Synthetic test")
    old = store.create_session(p["id"], str(uuid.uuid4()))
    current = store.create_session(p["id"], str(uuid.uuid4()))
    job, _ = make_job(store, old)
    monkeypatch.setattr("bridge.preview._registration", lambda *a: {
        "headset_preview_allowed": True, "unity_from_splat": np.eye(4).flatten().tolist()})
    result = preview_manifest(store, current["id"], "splat")
    assert result["status"] == "ready"
    assert result["display_mode"] == "inspection_model"
    assert result["total_views"] == store.session(old["id"])["capture_count"]
    assert not result["alignment"]["headset_preview_allowed"]
    assert "path" not in result["artifact"]
    unavailable = preview_manifest(store, current["id"], "mesh")
    assert unavailable["status"] == "not_ready"
    assert unavailable["total_views"] == store.session(current["id"])["capture_count"]
    other = store.create_project("Another test")
    with pytest.raises(StoreError):
        read_job(store, other["id"], job["job_id"])


def test_online_snapshot_replaces_seed_in_headset_and_rejects_tampering(tmp_path, monkeypatch):
    token = "test-token-" * 5
    app = create_app(Config(tmp_path, token))
    store = app.state.store
    project = store.create_project("Online room")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    job, full = make_job(store, session)
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n" +
              "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    full.write_bytes(header + struct.pack("<42f", *([0.] * 42)))
    job["artifact"].update(sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
                           size_bytes=full.stat().st_size, vertex_count=3)
    seed = full.with_name("quest-preview.ply")
    job["quest_preview"] = make_quest_preview(full, seed, max_vertices=3)
    (full.parent / "job.json").write_text(json.dumps(job))
    folder = tmp_path / "projects" / project["id"] / "online" / session["id"]
    folder.mkdir(parents=True)
    current = folder / "current.ply"
    current.write_bytes(seed.read_bytes())
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    registration = {"status": "verified", "session_id": session["id"],
                    "seed_sha256": job["quest_preview"]["sha256"],
                    "basis": "opencv_rh_from_unity", "evidence_job_id": job["job_id"],
                    "physical_acceptance": "accepted", "model_from_session": np.eye(4).tolist(),
                    "pose_fit": {"matched_views": 3, "baseline_m": .5,
                                 "position_rmse_m": .01, "max_position_error_m": .02,
                                 "max_rotation_error_deg": 2.}}
    status = {"project_id": project["id"], "session_id": session["id"],
              "seed_sha256": job["quest_preview"]["sha256"],
              "registration": registration, "seed_ids": ["seed-a", "seed-b", "seed-c", "unmatched-seed"],
              "accepted_ids": ["new-a"],
              "artifact": {"sha256": digest, "size_bytes": current.stat().st_size,
                           "vertex_count": 3}}
    (folder / "status.json").write_text(json.dumps(status))
    manifest = preview_manifest(store, session["id"], "splat")
    assert manifest["display_mode"] == "world_overlay"
    assert manifest["accepted_views"] == 4
    assert manifest["total_views"] == store.session(session["id"])["capture_count"]
    assert manifest["artifact"]["sha256"] == digest
    assert "/online/" in manifest["artifact"]["download_path"]
    client = TestClient(app)
    route = manifest["artifact"]["download_path"]
    assert client.get(route).status_code == 401
    headers = {"Authorization": "Bearer " + token}
    monkeypatch.setattr("bridge.app.request_build", lambda *args, **kwargs: pytest.fail(
        "B preview must not queue a full rebuild when the online model is ready"))
    live_manifest = client.get(
        f"/api/v1/sessions/{session['id']}/preview?representation=splat", headers=headers)
    assert live_manifest.json()["artifact"]["sha256"] == digest
    assert client.get(route, headers=headers).content == current.read_bytes()
    current.write_bytes(current.read_bytes() + b"tampered")
    assert client.get(route, headers=headers).status_code == 404
    fallback = preview_manifest(store, session["id"], "splat")
    assert fallback["artifact"]["download_path"].endswith("/splat")
    assert "/online/" not in fallback["artifact"]["download_path"]


def test_online_manifest_counts_only_published_delta(tmp_path, monkeypatch):
    store = Store(tmp_path)
    project = store.create_project("Pending online photos")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    job, seed = make_job(store, session)
    status = {"registration": {"evidence_job_id": job["job_id"],
                               "pose_fit": {"matched_views": 2}},
              "seed_ids": ["seed-a", "seed-b"], "accepted_ids": ["visible", "pending"],
              "published_capture_ids": ["visible"], "revision_number": 2,
              "artifact": {"sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
                           "size_bytes": seed.stat().st_size, "vertex_count": 4}}
    monkeypatch.setattr("bridge.preview.read_online_preview", lambda *args: (
        status, seed, {"status": "verified", "headset_preview_allowed": True}))
    manifest = preview_manifest(store, session["id"], "splat")
    assert manifest["accepted_views"] == 3
    assert manifest["revision_number"] == 2


@pytest.mark.parametrize("published,accepted,registered,expected", [
    (["seed-a", "visible", "visible"], ["seed-a", "visible", "pending"], 2, 3),
    (None, ["seed-a", "seed-b", "visible", "visible"], 2, 3),
    ([], ["seed-a", "visible"], 2, 2),
    (["seed-a", "seed-b"], ["seed-a", "seed-b"], 2, 2),
    (["seed-b", "visible"], ["seed-b", "visible"], 1, 2),
])
def test_online_manifest_counts_unique_non_seed_additions(
        tmp_path, monkeypatch, published, accepted, registered, expected):
    store = Store(tmp_path)
    project = store.create_project("Refined baseline photo counts")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    job, seed = make_job(store, session)
    job["registered_views"] = registered
    (seed.parent / "job.json").write_text(json.dumps(job))
    status = {"registration": {"evidence_job_id": job["job_id"],
                               "pose_fit": {"matched_views": 2}},
              "seed_ids": ["seed-a", "seed-b"], "accepted_ids": accepted,
              "revision_number": 2,
              "artifact": {"sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
                           "size_bytes": seed.stat().st_size, "vertex_count": 4}}
    if published is not None:
        status["published_capture_ids"] = published
    monkeypatch.setattr("bridge.preview.read_online_preview", lambda *args: (
        status, seed, {"status": "verified", "headset_preview_allowed": True}))

    manifest = preview_manifest(store, session["id"], "splat")

    assert manifest["accepted_views"] == expected
    assert manifest["revision_number"] == 2
    assert manifest["artifact"]["sha256"] == status["artifact"]["sha256"]


def _small_splat(x: float, count: int = 3) -> bytes:
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n" +
              "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    record = struct.pack("<14f", x, *([0.] * 13))
    return header + record * count


def test_online_full_seed_and_immutable_revisions_keep_newer_splat(tmp_path):
    store = Store(tmp_path)
    project = store.create_project("Evolving room")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    job, seed = make_job(store, session)
    seed.write_bytes(_small_splat(0))
    job["artifact"].update(sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                           size_bytes=seed.stat().st_size, vertex_count=3)
    (seed.parent / "job.json").write_text(json.dumps(job))
    folder = tmp_path / "projects" / project["id"] / "online" / session["id"]
    revisions = folder / "display-revisions"
    revisions.mkdir(parents=True)
    registration = {"status": "verified", "session_id": session["id"],
                    "seed_sha256": job["artifact"]["sha256"],
                    "basis": "opencv_rh_from_unity", "evidence_job_id": job["job_id"],
                    "physical_acceptance": "accepted", "model_from_session": np.eye(4).tolist(),
                    "pose_fit": {"matched_views": 3, "baseline_m": .5,
                                 "position_rmse_m": .01, "max_position_error_m": .02,
                                 "max_rotation_error_deg": 2.}}

    def publish(number: int, x: float):
        current = folder / "current.ply"
        current.write_bytes(_small_splat(x))
        full = {"sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
                "size_bytes": current.stat().st_size, "vertex_count": 3}
        display_bytes = _small_splat(x + 10)
        display_sha = hashlib.sha256(display_bytes).hexdigest()
        (revisions / (display_sha + ".ply")).write_bytes(display_bytes)
        display = {"sha256": display_sha, "size_bytes": len(display_bytes),
                   "vertex_count": 3, "source_sha256": full["sha256"]}
        (revisions / (display_sha + ".json")).write_text(json.dumps({
            "project_id": project["id"], "session_id": session["id"],
            **display, "revision_number": number}))
        status = {"project_id": project["id"], "session_id": session["id"],
                  "seed_sha256": job["artifact"]["sha256"], "registration": registration,
                  "seed_ids": [], "accepted_ids": [], "revision_number": number,
                  "artifact": full, "display_artifact": display}
        (folder / "status.json").write_text(json.dumps(status))
        return display_sha, display_bytes

    first_sha, first_bytes = publish(1, 1)
    first = preview_manifest(store, session["id"], "splat")
    assert first["artifact"]["sha256"] == first_sha
    assert first["artifact"]["download_path"].endswith("/splat/" + first_sha)
    assert first["revision_number"] == 1
    second_sha, second_bytes = publish(2, 2)
    second = preview_manifest(store, session["id"], "splat")
    assert second["job_id"] == first["job_id"]
    assert second["artifact"]["sha256"] == second_sha != first_sha
    assert second["revision_number"] == 2
    assert read_online_preview(store, project["id"], session["id"], first_sha)[1].read_bytes() == first_bytes
    assert read_online_preview(store, project["id"], session["id"], second_sha)[1].read_bytes() == second_bytes
    token = "test-token-" * 5
    client = TestClient(create_app(Config(tmp_path, token)))
    assert client.get(first["artifact"]["download_path"]).status_code == 401
    headers = {"Authorization": "Bearer " + token}
    assert client.get(first["artifact"]["download_path"], headers=headers).content == first_bytes
    assert client.get(second["artifact"]["download_path"], headers=headers).content == second_bytes
    (revisions / (first_sha + ".json")).write_text("{}")
    with pytest.raises(StoreError):
        read_online_preview(store, project["id"], session["id"], first_sha)
    assert client.get(first["artifact"]["download_path"], headers=headers).status_code == 404
    assert preview_manifest(store, session["id"], "splat")["artifact"]["sha256"] == second_sha


def test_online_full_model_can_exceed_quest_budget_with_verified_small_copy(tmp_path):
    store = Store(tmp_path)
    project = store.create_project("Large training state")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    job, seed = make_job(store, session)
    seed.write_bytes(_small_splat(0))
    job["artifact"].update(sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                           size_bytes=seed.stat().st_size, vertex_count=3)
    (seed.parent / "job.json").write_text(json.dumps(job))
    folder = tmp_path / "projects" / project["id"] / "online" / session["id"]
    revisions = folder / "display-revisions"
    revisions.mkdir(parents=True)
    current = folder / "current.ply"
    current.write_bytes(_small_splat(1, 250001))
    full = {"sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
            "size_bytes": current.stat().st_size, "vertex_count": 250001}
    display_bytes = _small_splat(2)
    display_sha = hashlib.sha256(display_bytes).hexdigest()
    (revisions / (display_sha + ".ply")).write_bytes(display_bytes)
    display = {"sha256": display_sha, "size_bytes": len(display_bytes),
               "vertex_count": 3, "source_sha256": full["sha256"]}
    registration = {"status": "verified", "session_id": session["id"],
                    "seed_sha256": job["artifact"]["sha256"],
                    "basis": "opencv_rh_from_unity", "evidence_job_id": job["job_id"],
                    "physical_acceptance": "accepted", "model_from_session": np.eye(4).tolist(),
                    "pose_fit": {"matched_views": 3, "baseline_m": .5,
                                 "position_rmse_m": .01, "max_position_error_m": .02,
                                 "max_rotation_error_deg": 2.}}
    status = {"project_id": project["id"], "session_id": session["id"],
              "seed_sha256": job["artifact"]["sha256"], "registration": registration,
              "seed_ids": [], "accepted_ids": [], "artifact": full,
              "display_artifact": display}
    (folder / "status.json").write_text(json.dumps(status))
    assert read_online_preview(store, project["id"], session["id"])[1].read_bytes() == display_bytes
    assert preview_manifest(store, session["id"], "splat")["artifact"]["sha256"] == display_sha
    status["display_artifact"] = {**display, "source_sha256": "0" * 64}
    (folder / "status.json").write_text(json.dumps(status))
    with pytest.raises(StoreError):
        read_online_preview(store, project["id"], session["id"])
    status["display_artifact"] = display
    (folder / "status.json").write_text(json.dumps(status))
    current.write_bytes(current.read_bytes() + b"tampered")
    with pytest.raises(StoreError):
        read_online_preview(store, project["id"], session["id"])
