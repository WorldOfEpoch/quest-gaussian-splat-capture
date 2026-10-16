"""House intermediates remain explicit inputs, never automatic current models."""

import hashlib
import json
import struct
import uuid

import pytest

from bridge.dashboard import dashboard_current_model, verified_job
from bridge.preview import preview_manifest, read_job
from bridge.reconstruct import ReconstructionError, build_session
from bridge.store import Store, StoreError


def _job(store, session, *, publication, status, created):
    job_id = str(uuid.uuid4())
    folder = store.root / "projects" / session["project_id"] / "reconstructions" / job_id
    folder.mkdir(parents=True)
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 1\n" +
              "".join(f"property float {name}\n" for name in names) +
              "end_header\n").encode("ascii")
    ply = folder / "splat.ply"
    ply.write_bytes(header + struct.pack("<14f", *([0.0] * 14)))
    job = {"schema_version": 1, "job_id": job_id, "project_id": session["project_id"],
           "session_id": session["id"], "created_utc": created, "status": status,
           "quality": "final", "publication": publication, "registered_views": 6,
           "input_captures": [], "artifact": {
               "path": str(ply), "sha256": hashlib.sha256(ply.read_bytes()).hexdigest(),
               "size_bytes": ply.stat().st_size, "vertex_count": 1}}
    (folder / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return job


@pytest.mark.parametrize("house_status", ["ready_for_blender", "failed"])
def test_house_intermediate_cannot_replace_previous_current_model(tmp_path, monkeypatch, house_status):
    store = Store(tmp_path)
    project = store.create_project("House fixture")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    previous = _job(store, session, publication="automatic",
                    status="ready_for_blender", created="2026-10-12T10:00:00Z")
    house = _job(store, session, publication="offline_house_source",
                 status=house_status, created="2026-10-12T11:00:00Z")
    monkeypatch.setattr("bridge.preview._registration", lambda *_: {
        "status": "unregistered_colmap_frame", "headset_preview_allowed": False})

    browser = dashboard_current_model(store.root, project["id"])
    headset = preview_manifest(store, session["id"], "splat")
    assert browser is not None and browser["id"] == previous["job_id"]
    assert headset["status"] == "ready" and headset["job_id"] == previous["job_id"]
    assert headset["artifact"]["download_path"].endswith(f"/{previous['job_id']}/splat")

    if house_status == "ready_for_blender":
        # The House pipeline and explicit exports retain their original access.
        assert read_job(store, project["id"], house["job_id"])[0]["job_id"] == house["job_id"]
        assert verified_job(store.root, project["id"], house["job_id"])[0]["job_id"] == house["job_id"]
    else:
        with pytest.raises(StoreError):
            read_job(store, project["id"], house["job_id"])


def test_new_house_source_is_marked_before_external_work(tmp_path, monkeypatch):
    image = tmp_path / "input.jpg"
    image.write_bytes(b"synthetic image fixture")
    project_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
    captures = [{"receipt": {"capture_id": str(uuid.uuid4()),
                             "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()},
                 "metadata": {"capture_source": "quest_camera", "image_orientation": "upright"},
                 "image_path": str(image)} for _ in range(6)]

    class SnapshotStore:
        root = tmp_path

        def export_session(self, _session_id):
            return {"session": {"project_id": project_id}, "captures": captures}

    monkeypatch.setattr("bridge.reconstruct._tool", lambda name: name)
    monkeypatch.setattr("bridge.reconstruct._run", lambda *args, **kwargs: (
        _ for _ in ()).throw(ReconstructionError("External build intentionally stopped")))
    with pytest.raises(ReconstructionError, match="intentionally stopped"):
        build_session(SnapshotStore(), session_id, house_source=True)
    manifests = list((tmp_path / "projects" / project_id / "reconstructions").glob("*/job.json"))
    assert len(manifests) == 1
    house = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert house["publication"] == "offline_house_source" and house["status"] == "failed"


def test_house_reuses_existing_ready_job_without_relabelling_it(tmp_path, monkeypatch):
    project_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
    captures = [{"receipt": {"capture_id": str(uuid.uuid4()),
                             "image_sha256": str(index) * 64},
                 "metadata": {"capture_source": "quest_camera", "image_orientation": "upright"}}
                for index in range(6)]
    folder = tmp_path / "projects" / project_id / "reconstructions" / str(uuid.uuid4())
    processed, checkpoint = folder / "processed", folder / "training" / "nerfstudio_models"
    processed.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (checkpoint / "step-000000001.ckpt").write_bytes(b"fixture checkpoint")
    config = checkpoint.parent / "config.yml"
    config.write_text("fixture", encoding="utf-8")
    ply = folder / "splat.ply"
    ply.write_bytes(b"fixture artifact")
    job = {"job_id": folder.name, "session_id": session_id, "status": "ready_for_blender",
           "publication": "automatic", "quality": "final", "processed_dir": str(processed),
           "checkpoint_dir": str(checkpoint), "training_config": str(config),
           "input_captures": [dict(item["receipt"]) for item in captures],
           "artifact": {"path": str(ply), "size_bytes": ply.stat().st_size,
                        "sha256": hashlib.sha256(ply.read_bytes()).hexdigest()}}
    manifest = folder / "job.json"
    manifest.write_text(json.dumps(job), encoding="utf-8")
    original = manifest.read_bytes()

    class SnapshotStore:
        root = tmp_path

        def export_session(self, _session_id):
            return {"session": {"project_id": project_id}, "captures": captures}

    monkeypatch.setattr("bridge.reconstruct._tool", lambda name: (
        _ for _ in ()).throw(AssertionError("Reusable job must not launch tools")))
    assert build_session(SnapshotStore(), session_id, house_source=True)["job_id"] == job["job_id"]
    assert manifest.read_bytes() == original
