"""An old global fit cannot authorize a new, unlocalized depth witness."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import uuid

import numpy as np
import pytest
from PIL import Image

from bridge import mesh, preview
from bridge.reconstruct import ReconstructionError
from bridge.store import Store


def _save_capture(store, project, session, pose, index, *, depth=False, pixels=None):
    capture_id = str(uuid.uuid4())
    if pixels is None:
        output = io.BytesIO()
        Image.new("RGB", (32, 24), (20 + index * 22, 30 + index * 17, 40 + index * 13)).save(output, "JPEG")
        pixels = output.getvalue()
    timestamp = "2026-10-12T15:00:00.0000000Z"
    metadata = {
        "schema_version": 1, "capture_id": capture_id, "project_id": project["id"],
        "session_id": session["id"], "tracking_origin_id": session["tracking_origin_id"],
        "camera_id": "left", "image_timestamp_raw": timestamp, "image_timestamp_kind": "Utc",
        "timestamp_clock": "meta_datetime", "image_width": 32, "image_height": 24,
        "image_encoding": "jpeg", "image_sha256": hashlib.sha256(pixels).hexdigest(),
        "intrinsics": {"fx": 20., "fy": 20., "cx": 16., "cy": 12.},
        "camera_to_session": pose.ravel().tolist(), "coordinate_convention": "unity_lh_xright_yup_zforward",
        "position_units": "meters", "tracking_valid": True,
        "image_orientation": "upright", "capture_source": "quest_camera",
    }
    sidecar = None
    if depth:
        origin = pose[:3, 3]
        rays = []
        for y, z in ((.2, .3), (.4, .3), (.2, .5), (.4, .5)):
            hit = np.array([1., y, z]); direction = hit - origin
            direction /= np.linalg.norm(direction)
            rays.append({"origin": origin.tolist(), "direction": direction.tolist(), "status": "hit",
                         "hit": hit.tolist(), "normal": [-1., 0., 0.], "normal_confidence": .95})
        evidence = {
            "schema_version": 1, "capture_id": capture_id, "project_id": project["id"],
            "session_id": session["id"], "tracking_origin_id": session["tracking_origin_id"],
            "source": "meta_environment_raycast_openxr", "association": "near_capture_unaligned",
            "coordinate_frame": "quest_session", "coordinate_convention": "unity_lh_xright_yup_zforward",
            "position_units": "meters", "trigger_realtime_s": 100. + index,
            "sample_started_realtime_s": 100.05 + index, "sample_completed_realtime_s": 100.1 + index,
            "rgb_timestamp_raw": timestamp, "rgb_timestamp_kind": "Utc", "availability": "ready", "rays": rays,
        }
        sidecar = json.dumps(evidence).encode()
        metadata["spatial_evidence_sha256"] = hashlib.sha256(sidecar).hexdigest()
    receipt = store.save_capture(session["id"], metadata, pixels, sidecar)
    capture = next(item for item in store.export_session(session["id"])["captures"]
                   if item["receipt"]["capture_id"] == capture_id)
    return receipt, capture


def _scene(tmp_path, *, depth=False):
    store = Store(tmp_path)
    project = store.create_project("Photo registration fixture")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    source = np.repeat(np.eye(4)[None], 8, axis=0)
    source[:, :3, 3] = [[0., 0., 0.], [.2, 0., 0.], [0., .2, .03], [.2, .2, .05],
                         [.4, 0., .04], [0., .4, .08], [.4, .4, .02], [.2, .4, .06]]
    target = source.copy(); target[:, 2, 3] *= -1
    captures = [_save_capture(store, project, session, pose, index, depth=depth and index < 3)[1]
                for index, pose in enumerate(target)]
    job_id = str(uuid.uuid4())
    root = tmp_path / "projects" / project["id"] / "reconstructions" / job_id
    processed = root / "processed"; images = processed / "images"
    images.mkdir(parents=True)
    config = root / "training" / "config.yml"; config.parent.mkdir()
    config.write_text("synthetic fixture; no trainer invocation")
    (config.parent / "dataparser_transforms.json").write_text(json.dumps(
        {"transform": np.eye(4)[:3].tolist(), "scale": 1.}))
    frames, inputs, mapping = [], [], {}
    for index, capture in enumerate(captures):
        receipt = capture["receipt"]; frame_name = f"frame_{index + 1:05d}.jpg"
        (images / frame_name).write_bytes(Path(capture["image_path"]).read_bytes())
        frames.append({"file_path": "images/" + frame_name, "transform_matrix": source[index].tolist()})
        inputs.append({"capture_id": receipt["capture_id"], "image_sha256": receipt["image_sha256"],
                       "image_name": receipt["capture_id"] + ".jpg"})
        mapping[receipt["capture_id"]] = frame_name
    transforms = {"frames": frames}
    (processed / "transforms.json").write_text(json.dumps(transforms))
    job = {"job_id": job_id, "project_id": project["id"], "session_id": session["id"],
           "status": "ready_for_blender", "processed_dir": str(processed), "training_config": str(config),
           "input_captures": inputs, "frame_manifest": mapping}
    (root / "job.json").write_text(json.dumps(job))
    (root / "splat.ply").write_bytes(b"synthetic identity fixture; never rendered")
    return store, project, session, job, captures, source, target


def _raw_surface(store, job, inputs=None):
    revision_id = str(uuid.uuid4())
    root = store.root / "projects" / job["project_id"] / "reconstructions" / job["job_id"] / "mesh-revisions" / revision_id
    root.mkdir(parents=True)
    (root / "mesh.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "1 .2 .3\n1 .4 .3\n1 .2 .5\n3 0 1 2\n")
    artifact = mesh.mesh_to_glb(root / "mesh.ply", root / "mesh.glb", {"fixture": True})
    parent = {"revision_id": revision_id, "project_id": job["project_id"], "source_job_id": job["job_id"],
              "status": "ready", "method": "colmap_photo_mvs_delaunay", "coordinate_frame": "colmap_sparse",
              "input_captures": inputs or job["input_captures"], "uncertainty": {}, "artifact": artifact}
    (root / "mesh.json").write_text(json.dumps(parent))
    return parent, root


def _protected_files(tmp_path):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp_path.rglob("*") if path.is_file() and path.name != "bridge.sqlite3"}


def test_registered_photo_ids_and_individual_errors_preserve_the_existing_preview_fit(tmp_path):
    store, _, _, job, captures, source, target = _scene(tmp_path)
    before = _protected_files(tmp_path)
    expected = preview.fit_session_similarity(source, target)
    result = preview.register_job(store, job)
    assert {key: result[key] for key in expected} == expected
    ids = {capture["receipt"]["capture_id"] for capture in captures}
    assert set(result["registered_capture_ids"]) == set(result["capture_pose_residuals"]) == ids
    for capture_id, proof in result["capture_pose_residuals"].items():
        receipt = next(item["receipt"] for item in captures if item["receipt"]["capture_id"] == capture_id)
        assert proof["image_sha256"] == receipt["image_sha256"]
        assert proof["frame_name"] == job["frame_manifest"][capture_id]
        assert proof["position_error_m"] < 1e-10
        assert proof["rotation_error_deg"] < 1e-5
    assert _protected_files(tmp_path) == before


def test_unregistered_photo_has_no_proof_despite_good_global_fit(tmp_path):
    store, _, _, job, captures, _, _ = _scene(tmp_path)
    path = Path(job["processed_dir"]) / "transforms.json"
    transforms = json.loads(path.read_text()); transforms["frames"].pop(0)
    path.write_text(json.dumps(transforms))
    result = preview.register_job(store, job)
    assert result["headset_preview_allowed"] and result["matched_views"] == 7
    missing = captures[0]["receipt"]["capture_id"]
    assert missing not in result["registered_capture_ids"]
    assert missing not in result["capture_pose_residuals"]


@pytest.mark.parametrize("ambiguity", ["wrong_identity", "duplicate_frame", "wrong_hash"])
def test_proof_requires_unique_original_identity_and_verified_content(tmp_path, ambiguity):
    store, _, _, job, captures, _, _ = _scene(tmp_path)
    first, second = [item["receipt"]["capture_id"] for item in captures[:2]]
    if ambiguity == "wrong_identity":
        job["frame_manifest"][first], job["frame_manifest"][second] = (
            job["frame_manifest"][second], job["frame_manifest"][first])
    elif ambiguity == "duplicate_frame":
        job["frame_manifest"][second] = job["frame_manifest"][first]
    else:
        job["input_captures"][0]["image_sha256"] = "0" * 64
    result = preview.register_job(store, job)
    assert result["headset_preview_allowed"] and result["matched_views"] == 8
    assert first not in result["registered_capture_ids"]
    assert first not in result["capture_pose_residuals"]


def test_identical_rgb_captures_are_ambiguous_even_if_their_target_poses_are_equal(tmp_path):
    store, project, session, job, captures, source, target = _scene(tmp_path)
    original = captures[0]
    _, duplicate = _save_capture(store, project, session, target[0], 9,
                                 pixels=Path(original["image_path"]).read_bytes())
    result = preview.register_job(store, job)
    expected = preview.fit_session_similarity(source, target)
    assert {key: result[key] for key in expected} == expected
    assert original["receipt"]["capture_id"] not in result["registered_capture_ids"]
    assert duplicate["receipt"]["capture_id"] not in result["registered_capture_ids"]
    assert len(result["registered_capture_ids"]) == 7


def test_known_unique_reencoded_photo_retains_identity_and_residuals(tmp_path):
    store, _, _, job, captures, _, _ = _scene(tmp_path)
    capture_id = captures[0]["receipt"]["capture_id"]
    path = Path(job["processed_dir"]) / "images" / job["frame_manifest"][capture_id]
    with Image.open(path) as image:
        output = io.BytesIO(); image.save(output, "JPEG", quality=35)
    path.write_bytes(output.getvalue())
    result = preview.register_job(store, job)
    assert result["headset_preview_allowed"]
    assert capture_id in result["registered_capture_ids"]
    assert result["capture_pose_residuals"][capture_id]["position_error_m"] < 1e-10


@pytest.mark.parametrize("condition", ["missing_proof", "unregistered", "missing_residual", "wrong_hash",
                                      "wrong_position", "wrong_rotation", "nonfinite", "global_rmse",
                                      "global_views", "global_baseline"])
def test_good_global_fit_never_bypasses_independent_witness_proof_before_mesh_io(tmp_path, monkeypatch, condition):
    store, _, _, job, captures, _, _ = _scene(tmp_path, depth=True)
    parent, raw_root = _raw_surface(store, job)
    registration = preview.register_job(store, job)
    witness = captures[0]["receipt"]["capture_id"]
    if condition == "missing_proof":
        registration.pop("registered_capture_ids"); registration.pop("capture_pose_residuals")
    elif condition == "unregistered": registration["registered_capture_ids"].remove(witness)
    elif condition == "missing_residual": registration["capture_pose_residuals"].pop(witness)
    elif condition == "wrong_hash": registration["capture_pose_residuals"][witness]["image_sha256"] = "0" * 64
    elif condition == "wrong_position": registration["capture_pose_residuals"][witness]["position_error_m"] = .10001
    elif condition == "wrong_rotation": registration["capture_pose_residuals"][witness]["rotation_error_deg"] = 5.001
    elif condition == "nonfinite": registration["capture_pose_residuals"][witness]["position_error_m"] = float("nan")
    elif condition == "global_rmse": registration["position_rmse_m"] = .04001
    elif condition == "global_views": registration["matched_views"] = 5
    else: registration["baseline_m"] = .299
    monkeypatch.setattr(preview, "register_job", lambda *_: deepcopy(registration))
    monkeypatch.setattr(mesh, "read_mesh", lambda *_: pytest.fail("Rejected rays must not read any mesh binary"))
    monkeypatch.setattr(mesh, "mesh_to_glb", lambda *_: pytest.fail("Rejected rays must not export geometry"))
    before = _protected_files(tmp_path)
    with pytest.raises(ReconstructionError):
        mesh.depth_assisted_mesh_revision(store, job["project_id"], job["job_id"], parent["revision_id"])
    assert _protected_files(tmp_path) == before
    assert list(raw_root.parent.iterdir()) == [raw_root]


def test_new_ready_ray_capture_outside_source_registration_cannot_use_old_global_fit(tmp_path, monkeypatch):
    store, project, session, job, captures, _, target = _scene(tmp_path, depth=True)
    # Replace one original ray-bearing witness with a new photo, as happens
    # when a mesh-only photo inventory grows beyond the trained source job.
    original = captures[0]
    new_pose = target[0].copy(); new_pose[0, 3] = .6
    _, new_capture = _save_capture(store, project, session, new_pose, 9, depth=True)
    inputs = [item for item in job["input_captures"] if item["capture_id"] != original["receipt"]["capture_id"]]
    inputs.append({"capture_id": new_capture["receipt"]["capture_id"],
                   "image_sha256": new_capture["receipt"]["image_sha256"], "image_name": "new.jpg"})
    parent, _ = _raw_surface(store, job, inputs)
    result = preview.register_job(store, job)
    assert result["headset_preview_allowed"] and result["position_rmse_m"] < .04
    assert new_capture["receipt"]["capture_id"] not in result["registered_capture_ids"]
    monkeypatch.setattr(mesh, "read_mesh", lambda *_: pytest.fail("New unlocalized ray witness must fail before mesh I/O"))
    before = _protected_files(tmp_path)
    with pytest.raises(ReconstructionError, match="not registered in the source photo frame"):
        mesh.depth_assisted_mesh_revision(store, job["project_id"], job["job_id"], parent["revision_id"])
    assert _protected_files(tmp_path) == before


def test_actual_unregistered_ray_photo_is_rejected_with_good_remaining_fit(tmp_path, monkeypatch):
    store, _, _, job, _, _, _ = _scene(tmp_path, depth=True)
    parent, _ = _raw_surface(store, job)
    path = Path(job["processed_dir"]) / "transforms.json"
    transforms = json.loads(path.read_text()); transforms["frames"].pop(0)
    path.write_text(json.dumps(transforms))
    result = preview.register_job(store, job)
    assert result["headset_preview_allowed"] and result["matched_views"] == 7
    assert result["position_rmse_m"] < .04
    monkeypatch.setattr(mesh, "read_mesh", lambda *_: pytest.fail("Unregistered photo must fail before mesh I/O"))
    with pytest.raises(ReconstructionError, match="not registered in the source photo frame"):
        mesh.depth_assisted_mesh_revision(store, job["project_id"], job["job_id"], parent["revision_id"])


def test_valid_registered_ray_witnesses_reach_surface_stage_without_changing_originals(tmp_path, monkeypatch):
    pytest.importorskip("plyfile")
    store, _, _, job, captures, _, _ = _scene(tmp_path, depth=True)
    parent, _ = _raw_surface(store, job)
    before = _protected_files(tmp_path)
    calls = []
    def validate(vertices, faces, observations):
        calls.append(observations["report"]["independent_capture_ids"])
        return vertices.copy(), np.full(len(vertices), 3), {"moved_vertices": 0, "faces_removed": 0}
    monkeypatch.setattr("bridge.mesh_spatial.assist_observed_surface", validate)
    result = mesh.depth_assisted_mesh_revision(store, job["project_id"], job["job_id"], parent["revision_id"])
    assert result["status"] == "ready"
    witness_ids = {item["receipt"]["capture_id"] for item in captures[:3]}
    assert set(calls[0]) == witness_ids
    proofs = result["spatial_support"]["frame_registration"]["ray_capture_registration"]
    assert set(proofs) == witness_ids
    assert all(proof["position_error_m"] <= .10 and proof["rotation_error_deg"] <= 5. for proof in proofs.values())
    after = _protected_files(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
