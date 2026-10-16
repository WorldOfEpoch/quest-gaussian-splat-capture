"""Frozen house deliveries use isolated stores and never launch native/GPU tools."""

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path
import uuid

import numpy as np
import pytest
from PIL import Image

from bridge import mesh, reconstruct
from bridge.dashboard import set_capture_selection
from bridge.jobs import on_capture_received, request_build, request_status, set_policy
from bridge.reconstruct import ReconstructionError, house_capture_snapshot, house_input_revision
from bridge.store import Store, StoreError


def _capture(store, project, session, index, *, depth=True):
    image = io.BytesIO()
    Image.new("RGB", (16, 12), (20 + index, 40, 60)).save(image, format="JPEG")
    pixels = image.getvalue()
    capture_id = str(uuid.uuid4())
    position = index * .1
    metadata = {
        "schema_version": 1, "capture_id": capture_id,
        "project_id": project["id"], "session_id": session["id"],
        "tracking_origin_id": session["tracking_origin_id"],
        "camera_id": "left", "image_timestamp_raw": "2026-10-12T15:00:00.0000000Z",
        "image_timestamp_kind": "Utc", "timestamp_clock": "meta_datetime",
        "image_width": 16, "image_height": 12, "image_encoding": "jpeg",
        "image_sha256": hashlib.sha256(pixels).hexdigest(),
        "intrinsics": {"fx": 10., "fy": 10., "cx": 8., "cy": 6.},
        "camera_to_session": [1., 0., 0., position, 0., 1., 0., 0.,
                              0., 0., 1., 0., 0., 0., 0., 1.],
        "coordinate_convention": "unity_lh_xright_yup_zforward",
        "position_units": "meters", "tracking_valid": True,
        "capture_source": "quest_camera", "image_orientation": "upright",
    }
    payload = None
    if depth:
        rays = []
        for y, z in ((.2, .3), (.4, .3), (.2, .5), (.4, .5)):
            origin = np.array([position, 0., 0.])
            hit = np.array([2., y, z])
            direction = hit - origin
            direction /= np.linalg.norm(direction)
            rays.append({"origin": origin.tolist(), "direction": direction.tolist(),
                         "status": "hit", "hit": hit.tolist(),
                         "normal": [-1., 0., 0.], "normal_confidence": .95})
        evidence = {
            "schema_version": 1, "capture_id": capture_id,
            "project_id": project["id"], "session_id": session["id"],
            "tracking_origin_id": session["tracking_origin_id"],
            "source": "meta_environment_raycast_openxr", "association": "near_capture_unaligned",
            "coordinate_frame": "quest_session", "coordinate_convention": "unity_lh_xright_yup_zforward",
            "position_units": "meters", "trigger_realtime_s": 100. + index,
            "sample_started_realtime_s": 100.05 + index,
            "sample_completed_realtime_s": 100.1 + index,
            "rgb_timestamp_raw": metadata["image_timestamp_raw"], "rgb_timestamp_kind": "Utc",
            "availability": "ready", "rays": rays,
        }
        payload = json.dumps(evidence, separators=(",", ":"), allow_nan=False).encode()
        metadata["spatial_evidence_sha256"] = hashlib.sha256(payload).hexdigest()
    return store.save_capture(session["id"], metadata, pixels, payload)


def _scene(tmp_path, *, count=8, depth=True):
    store = Store(tmp_path)
    project = store.create_project("House fixture")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    captures = [_capture(store, project, session, index, depth=depth) for index in range(count)]
    return store, project, session, captures


def _stub_native_pipeline(monkeypatch, store, project, session, *, failure=None,
                          before_splat=None, bad_historical_id=None):
    """Replace expensive stages only; snapshot/preflight/receipts run normally."""
    source_id, raw_id, depth_id, finished_id = [str(uuid.uuid4()) for _ in range(4)]
    stages = []
    source = {"job_id": source_id, "artifact": {"sha256": "a" * 64}}
    raw = {"revision_id": raw_id}
    assisted = {"revision_id": depth_id, "spatial_support": {"state": "fixture_validation"}}
    finished = {"revision_id": finished_id, "artifact": {
        "sha256": "b" * 64, "path": str(store.root / "fixture.glb")}}

    def stage(name, result):
        def run(*args, **kwargs):
            stages.append((name, args, kwargs))
            if name == "splat" and before_splat:
                before_splat()
            if failure == name:
                raise ReconstructionError("Fixture failure: " + name)
            return result
        return run

    def verify(_store, project_id, build_id):
        if build_id == bad_historical_id:
            raise StoreError(404, "Corrupted historical fixture")
        stages.append(("verify", (), {}))
        if failure == "verify":
            raise ReconstructionError("Fixture failure: verify")
        path = store.root / "projects" / project_id / "house-builds" / build_id / "house.json"
        return json.loads(path.read_text()), store.root / "fixture.glb"

    monkeypatch.setattr(reconstruct, "build_session", stage("splat", source))
    monkeypatch.setattr(mesh, "build_mesh", stage("raw", raw))
    monkeypatch.setattr(mesh, "depth_assisted_mesh_revision", stage("depth", assisted))
    monkeypatch.setattr(mesh, "finish_mesh", stage("texture", finished))
    monkeypatch.setattr(reconstruct, "read_house_build", verify)
    return stages


def test_house_snapshot_excludes_review_choices_and_rejects_missing_or_duplicate_frozen_ids(tmp_path):
    store, project, session, captures = _scene(tmp_path)
    omitted = captures[0]["capture_id"]
    set_capture_selection(store, project["id"], session["id"], omitted, False)
    snapshot = house_capture_snapshot(store, session["id"])
    selected = [item["receipt"]["capture_id"] for item in snapshot["captures"]]
    assert len(selected) == 7 and omitted not in selected
    assert len(store.captures(session["id"])) == 8
    with pytest.raises(ReconstructionError, match="excluded or is no longer available"):
        house_capture_snapshot(store, session["id"], [item["capture_id"] for item in captures])
    with pytest.raises(ReconstructionError, match="Duplicate capture"):
        house_capture_snapshot(store, session["id"], selected + [selected[0]])


def test_queued_house_inventory_stays_frozen_when_new_photos_arrive(tmp_path, monkeypatch):
    store, project, session, captures = _scene(tmp_path)
    set_policy(tmp_path, session["id"], auto_enabled=True, native_orientation_reviewed=True)
    request = request_build(tmp_path, session["id"], reason="manual", quality="house")
    inventory_path = tmp_path / "build-queue-inputs" / (request["request_id"] + ".json")
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    original_ids = {capture["capture_id"] for capture in captures}
    _capture(store, project, session, 8)
    stages = _stub_native_pipeline(monkeypatch, store, project, session,
                                  before_splat=lambda: _capture(store, project, session, 9))
    result = reconstruct.build_house_session(store, session["id"], capture_ids=inventory["capture_ids"],
        expected_input_revision=inventory["input_revision"], native_orientation_reviewed=True)
    assert set(stages[0][2]["capture_ids"]) == original_ids
    assert set(result["input_capture_ids"]) == original_ids and result["input_count"] == 8
    assert stages[0][2]["quality"] == "final"
    assert stages[-2][2] == {"min_component_faces": 1, "smooth_steps": 0}
    assert len(store.captures(session["id"])) == 10
    assert inventory_path.read_bytes() == inventory_bytes


def test_wrong_frozen_revision_fails_before_native_work_or_output(tmp_path, monkeypatch):
    store, project, session, captures = _scene(tmp_path)
    stages = _stub_native_pipeline(monkeypatch, store, project, session)
    with pytest.raises(ReconstructionError, match="differ from the queued snapshot"):
        reconstruct.build_house_session(store, session["id"],
            capture_ids=[item["capture_id"] for item in captures], expected_input_revision="0" * 64)
    assert stages == []
    assert not (tmp_path / "projects" / project["id"] / "house-builds").exists()


def test_image_only_house_attempt_stops_before_gpu_and_house_output(tmp_path, monkeypatch):
    store, project, session, captures = _scene(tmp_path, depth=False)
    stages = _stub_native_pipeline(monkeypatch, store, project, session)
    originals = {item["image_path"]: hashlib.sha256(Path(item["image_path"]).read_bytes()).hexdigest()
                 for item in store.export_session(session["id"])["captures"]}
    with pytest.raises(ReconstructionError, match="Need new depth-assisted captures"):
        reconstruct.build_house_session(store, session["id"], native_orientation_reviewed=True)
    assert stages == []
    assert not (tmp_path / "projects" / project["id"] / "house-builds").exists()
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in originals.items())


def test_manual_house_request_survives_automatic_quick_arrival(tmp_path):
    store, project, session, _ = _scene(tmp_path)
    set_policy(tmp_path, session["id"], auto_enabled=True, native_orientation_reviewed=True)
    house = request_build(tmp_path, session["id"], reason="manual", quality="house")
    _capture(store, project, session, 8)
    quick = on_capture_received(tmp_path, session["id"])
    assert quick["status"] == "pending" and quick["request_id"] != house["request_id"]
    rows = request_status(tmp_path, project["id"])
    assert {(row["quality"], row["reason"], row["status"]) for row in rows} == {
        ("house", "manual", "pending"), ("quick", "capture", "pending")}


@pytest.mark.parametrize("failed_stage", ["splat", "raw", "depth", "texture", "verify"])
def test_house_stage_failure_never_leaves_a_ready_delivery(tmp_path, monkeypatch, failed_stage):
    store, project, session, _ = _scene(tmp_path)
    stages = _stub_native_pipeline(monkeypatch, store, project, session, failure=failed_stage)
    with pytest.raises(ReconstructionError, match="Fixture failure: " + failed_stage):
        reconstruct.build_house_session(store, session["id"], native_orientation_reviewed=True)
    receipts = list((tmp_path / "projects" / project["id"] / "house-builds").glob("*/house.json"))
    assert len(receipts) == 1
    result = json.loads(receipts[0].read_text())
    assert result["status"] == "failed" and result["processing_stage"] == "failed"
    assert result["error"] == "Fixture failure: " + failed_stage
    assert [stage[0] for stage in stages][-1] == failed_stage
    assert len(store.captures(session["id"])) == 8


@pytest.mark.parametrize("corruption", ["invalid_json", "non_object_json", "invalid_verified_delivery"])
def test_corrupted_historical_house_receipt_cannot_block_a_new_attempt(tmp_path, monkeypatch, corruption):
    store, project, session, _ = _scene(tmp_path)
    snapshot = house_capture_snapshot(store, session["id"])
    old_id = str(uuid.uuid4())
    old = tmp_path / "projects" / project["id"] / "house-builds" / old_id / "house.json"
    old.parent.mkdir(parents=True)
    if corruption == "invalid_json":
        old.write_text('{"interrupted":')
    elif corruption == "non_object_json":
        old.write_text('[]')
    else:
        old.write_text(json.dumps({"session_id": session["id"], "status": "ready_for_blender",
            "input_revision": house_input_revision(snapshot["captures"])}))
    previous_bytes = old.read_bytes()
    stages = _stub_native_pipeline(monkeypatch, store, project, session, bad_historical_id=old_id)
    result = reconstruct.build_house_session(store, session["id"], native_orientation_reviewed=True)
    assert result["status"] == "ready_for_blender" and result["job_id"] != old_id
    assert [stage[0] for stage in stages] == ["splat", "raw", "depth", "texture", "verify"]
    assert old.read_bytes() == previous_bytes


def _house_delivery(tmp_path):
    """Write small immutable fixture artifacts for the real aggregate validators."""
    store, project, session, captures = _scene(tmp_path)
    source_id, raw_id, depth_id, finished_id, build_id = [str(uuid.uuid4()) for _ in range(5)]
    job_root = tmp_path / "projects" / project["id"] / "reconstructions" / source_id
    job_root.mkdir(parents=True)
    splat = job_root / "splat.ply"
    splat.write_bytes(b"Isolated source artifact; this test validates identity and checksums, not native splat rendering.")
    source_artifact = {"sha256": Store._file_sha256(splat), "size_bytes": splat.stat().st_size,
                       "path": str(splat)}
    inputs = [{"capture_id": item["capture_id"], "image_sha256": item["image_sha256"]}
              for item in captures]
    job = {"schema_version": 1, "job_id": source_id, "project_id": project["id"],
           "session_id": session["id"], "status": "ready_for_blender", "quality": "final",
           "input_captures": inputs, "artifact": source_artifact}
    job_path = job_root / "job.json"
    job_path.write_text(json.dumps(job))
    ply = ("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
           "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
           "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    mesh_records, mesh_paths = {}, {}
    for revision_id in (raw_id, depth_id, finished_id):
        folder = job_root / "mesh-revisions" / revision_id
        folder.mkdir(parents=True)
        (folder / "mesh.ply").write_text(ply)
        artifact = mesh.mesh_to_glb(folder / "mesh.ply", folder / "mesh.glb", {"scope": "test_fixture"})
        mesh_records[revision_id] = {"schema_version": 1, "revision_id": revision_id,
            "project_id": project["id"], "source_job_id": source_id, "status": "ready",
            "coordinate_frame": "colmap_sparse", "artifact": artifact}
        mesh_paths[revision_id] = folder / "mesh.json"
    support = {"state": "limited_surface_validation",
        "source_surface_sha256": Store._file_sha256(mesh_paths[raw_id].parent / "mesh.ply"),
        "evidence": mesh.verified_spatial_observations(store, session["id"])["report"]}
    mesh_records[depth_id].update(parent_mesh_revision_id=raw_id,
        method="photo_mesh_with_independent_spatial_support", spatial_support=deepcopy(support))
    mesh_records[finished_id].update(parent_mesh_revision_id=depth_id,
        method="colmap_photo_mvs_cleaned_atlas", coordinate_frame="nerfstudio_training_zup",
        spatial_support=deepcopy(support), cleanup={"taubin_steps": 0, "min_component_faces": 1,
            "geometry_preservation": {"valid_triangles_unchanged": True}})
    for revision_id, record in mesh_records.items():
        mesh_paths[revision_id].write_text(json.dumps(record))
    snapshot = house_capture_snapshot(store, session["id"])
    record = {"schema_version": 1, "job_id": build_id, "quality": "house",
        "project_id": project["id"], "session_id": session["id"],
        "tracking_origin_id": session["tracking_origin_id"], "status": "ready_for_blender",
        "source_job_id": source_id, "source_splat_sha256": source_artifact["sha256"],
        "input_capture_ids": [item["capture_id"] for item in captures], "input_count": len(captures),
        "input_revision": house_input_revision(snapshot["captures"]),
        "raw_revision_id": raw_id, "depth_revision_id": depth_id, "mesh_revision_id": finished_id,
        "artifact": mesh_records[finished_id]["artifact"], "spatial_support": deepcopy(support)}
    manifest = tmp_path / "projects" / project["id"] / "house-builds" / build_id / "house.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(record))
    return {"store": store, "project": project, "session": session, "build_id": build_id,
            "record": record, "manifest": manifest, "job": job, "job_path": job_path,
            "meshes": mesh_records, "mesh_paths": mesh_paths, "raw_id": raw_id,
            "depth_id": depth_id, "finished_id": finished_id, "splat": splat}


def test_house_reader_requires_a_verified_frozen_revision_chain(tmp_path):
    delivery = _house_delivery(tmp_path)
    record, artifact = reconstruct.read_house_build(delivery["store"], delivery["project"]["id"],
                                                   delivery["build_id"])
    assert record["job_id"] == delivery["build_id"]
    assert artifact == delivery["mesh_paths"][delivery["finished_id"]].parent / "mesh.glb"
    assert Store._file_sha256(artifact) == record["artifact"]["sha256"]


@pytest.mark.parametrize("corruption", ["depth_parent", "texture_parent", "missing_support",
    "wrong_support_origin", "smoothing", "component_deletion", "missing_geometry_proof",
    "frozen_inventory", "frozen_revision", "source_photo_hash", "raw_surface_hash",
    "mesh_bytes", "splat_bytes"])
def test_house_reader_rejects_broken_chain_inventory_and_artifacts(tmp_path, corruption):
    delivery = _house_delivery(tmp_path)
    record, job = delivery["record"], delivery["job"]
    raw = delivery["mesh_paths"][delivery["raw_id"]].parent / "mesh.ply"
    depth = delivery["meshes"][delivery["depth_id"]]
    finished = delivery["meshes"][delivery["finished_id"]]
    if corruption == "depth_parent": depth["parent_mesh_revision_id"] = delivery["finished_id"]
    elif corruption == "texture_parent": finished["parent_mesh_revision_id"] = delivery["raw_id"]
    elif corruption == "missing_support": depth.pop("spatial_support")
    elif corruption == "wrong_support_origin":
        for support in (record["spatial_support"], depth["spatial_support"], finished["spatial_support"]):
            support["evidence"]["tracking_origin_id"] = "00000000-0000-0000-0000-000000000001"
    elif corruption == "smoothing": finished["cleanup"]["taubin_steps"] = 2
    elif corruption == "component_deletion": finished["cleanup"]["min_component_faces"] = 25
    elif corruption == "missing_geometry_proof": finished["cleanup"].pop("geometry_preservation")
    elif corruption == "frozen_inventory": record["input_capture_ids"][-1] = str(uuid.uuid4())
    elif corruption == "frozen_revision": record["input_revision"] = "0" * 64
    elif corruption == "source_photo_hash": job["input_captures"][0]["image_sha256"] = "0" * 64
    elif corruption == "raw_surface_hash": raw.write_bytes(raw.read_bytes() + b"\n")
    elif corruption == "mesh_bytes":
        artifact = delivery["mesh_paths"][delivery["finished_id"]].parent / "mesh.glb"
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
    elif corruption == "splat_bytes": delivery["splat"].write_bytes(b"tampered")
    delivery["manifest"].write_text(json.dumps(record))
    delivery["job_path"].write_text(json.dumps(job))
    for revision_id, mesh_record in delivery["meshes"].items():
        delivery["mesh_paths"][revision_id].write_text(json.dumps(mesh_record))
    with pytest.raises((ReconstructionError, StoreError)):
        reconstruct.read_house_build(delivery["store"], delivery["project"]["id"], delivery["build_id"])


@pytest.mark.parametrize("body", [[], None, {"schema_version": 1}])
def test_house_download_reports_unavailable_for_malformed_receipts(tmp_path, body):
    from fastapi.testclient import TestClient
    from bridge.app import create_app
    from bridge.config import Config
    delivery = _house_delivery(tmp_path)
    delivery["manifest"].write_text(json.dumps(body))
    api = TestClient(create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True),
                     base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    response = api.get(f"/ui/projects/{delivery['project']['id']}/house-builds/{delivery['build_id']}/mesh.glb")
    assert response.status_code == 404
    assert response.json()["detail"] == "Verified house mesh is unavailable"
