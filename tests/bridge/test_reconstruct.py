import json
import hashlib
import struct
import uuid
import ast
from pathlib import Path

import numpy as np
import pytest

from bridge.reconstruct import (ReconstructionError, _check_quest_pose_consistency,
                                _incremental_training_split, _normalization,
                                _pose_in_training, _verified_resume_warmup, build_session,
                                make_clean_online_preview, make_quest_preview,
                                validate_gaussian_ply)
from bridge.store import Store


def _reviewable_sh3_fixture(tmp_path):
    store, job, captures, _ = _bounded_candidate_fixture(tmp_path)
    folder = Path(job['artifact']['path']).parent
    names = ('x','y','z','opacity','scale_0','scale_1','scale_2',
             'rot_0','rot_1','rot_2','rot_3','f_dc_0','f_dc_1','f_dc_2') + tuple(f'f_rest_{i}' for i in range(45))
    ply = ('ply\nformat binary_little_endian 1.0\nelement vertex 6\n' +
           ''.join(f'property float {name}\n' for name in names) + 'end_header\n').encode()
    ply += struct.pack('<59f', *([0.]*59))*6
    Path(job['artifact']['path']).write_bytes(ply)
    checkpoint = Path(job['checkpoint_dir'])/'step-000007999.ckpt'
    checkpoint.write_bytes(b'selected SH3 optimizer momentum')
    config = Path(job['training_config'])
    config.write_text(config.read_text().replace('5000','3000').replace('num_downscales: 1','num_downscales: 0')
        .replace('stop_split_at: 3500','stop_split_at: 15000') + '  sh_degree: 3\n')
    config.with_name('dataparser_transforms.json').write_text(json.dumps({'transform':np.eye(4)[:3].tolist(),'scale':1.}))
    job.update(status='candidate_for_review', quality='final',checkpoint_start_step=4999,target_training_step=8000,
        export_checkpoint_sha256=Store._file_sha256(checkpoint))
    job['training_profile']['num_downscales']=0
    job['artifact'].update(sha256=hashlib.sha256(ply).hexdigest(),size_bytes=len(ply))
    (folder/'job.json').write_text(json.dumps(job))
    return store,job,captures,ply


def test_selected_sh3_preserves_model_optimizer_and_continuation_profile(tmp_path):
    from bridge.reconstruct import accept_reviewed_candidate, _bounded_candidate_profile
    store,job,captures,original = _reviewable_sh3_fixture(tmp_path)
    selected = accept_reviewed_candidate(store,job['session_id'],job['job_id'],
        expected_sha256=job['artifact']['sha256'],review='User selected SH3; continue from it')
    recovery = Path(job['artifact']['path']).parent/'accepted-baseline'
    assert (recovery/'splat.ply').read_bytes()==original
    assert (recovery/'checkpoint.ckpt').read_bytes()==b'selected SH3 optimizer momentum'
    assert selected['accepted_review']['checkpoint_step']==7999
    assert _bounded_candidate_profile(selected)['num_downscales']==0
    assert _bounded_candidate_profile(selected)['stop_split_at']==15000
    assert Path(job['artifact']['path']).read_bytes()==original
    assert all(Path(row['image_path']).read_bytes()==f'photo-{i}'.encode() for i,row in enumerate(captures))
    assert accept_reviewed_candidate(store,job['session_id'],job['job_id'],
        expected_sha256=job['artifact']['sha256'],review='Repeat')==selected
    (recovery/'checkpoint.ckpt').write_bytes(b'wrong preserved checkpoint')
    with pytest.raises(ReconstructionError,match='Preserved baseline differs'):
        accept_reviewed_candidate(store,job['session_id'],job['job_id'],
            expected_sha256=job['artifact']['sha256'],review='Repeat')
    assert Path(job['artifact']['path']).read_bytes()==original


def test_selected_sh3_refuses_wrong_model_or_changed_optimizer(tmp_path):
    from bridge.reconstruct import accept_reviewed_candidate
    store,job,_,original = _reviewable_sh3_fixture(tmp_path)
    with pytest.raises(ReconstructionError,match='identify'):
        accept_reviewed_candidate(store,job['session_id'],job['job_id'],expected_sha256='0'*64,review='Wrong scan')
    (Path(job['checkpoint_dir'])/'step-000007999.ckpt').write_bytes(b'changed momentum')
    with pytest.raises(ReconstructionError,match='checkpoint changed'):
        accept_reviewed_candidate(store,job['session_id'],job['job_id'],expected_sha256=job['artifact']['sha256'],review='Changed')
    assert Path(job['artifact']['path']).read_bytes()==original
    assert not (Path(job['artifact']['path']).parent/'accepted-baseline').exists()


def test_clean_online_preview_removes_floaters_without_changing_source(tmp_path):
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    records = []
    for index in range(100):
        records.append((index % 10 * .04, index // 10 * .04, 0., 2., -4., -4., -4.,
                        1., 0., 0., 0., 0., 0., 0.))
    records += [(30., 0., 0., 2., -4., -4., -4., 1., 0., 0., 0., 0., 0., 0.),
                (0., 0., 0., 2., 2., -4., -4., 1., 0., 0., 0., 0., 0., 0.),
                (0., 0., 0., -8., -4., -4., -4., 1., 0., 0., 0., 0., 0., 0.)]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(records)}\n" +
              "".join(f"property float {name}\n" for name in names) +
              "end_header\n").encode()
    source, display = tmp_path / "full.ply", tmp_path / "display.ply"
    source.write_bytes(header + b"".join(struct.pack("<14f", *row) for row in records))
    original = source.read_bytes()
    artifact = make_clean_online_preview(source, display)
    assert source.read_bytes() == original
    assert artifact["vertex_count"] == 100 and artifact["removed_gaussians"] == 3
    assert validate_gaussian_ply(display)[1] == artifact["sha256"]


@pytest.mark.parametrize("count,quality,passes_count_check", [(2, "quick", False),
    (3, "quick", True), (3, "final", False), (6, "final", True)])
def test_small_capture_set_only_enters_quick_attempt(tmp_path, count, quality, passes_count_check):
    class SnapshotStore:
        root = tmp_path

        def export_session(self, _session_id):
            return {"captures": [{"metadata": {"capture_source": "synthetic"}} for _ in range(count)]}

    # Once count passes, the physical-camera guard must still reject synthetic data.
    expected = "physical Quest" if passes_count_check else "received views"
    with pytest.raises(ReconstructionError, match=expected):
        build_session(SnapshotStore(), "session", quality=quality)


def test_reject_non_gaussian_and_truncated_ply(tmp_path):
    path = tmp_path / "not-a-splat.ply"
    path.write_text("not a PLY", encoding="utf-8")
    with pytest.raises(ReconstructionError):
        validate_gaussian_ply(path)
    props = "".join(f"property float {name}\n" for name in
                    ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                     "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"))
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 100\n"
              + props + "end_header\n")
    path.write_bytes(header.encode("ascii") + b"\x00" * 256)
    with pytest.raises(ReconstructionError, match="truncated"):
        validate_gaussian_ply(path)


def test_quest_copy_merges_coverage_without_changing_original(tmp_path):
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 6\n" +
              "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    records = [struct.pack("<14f", index * .002, 0., 0., 2., -4., -4., -4.,
                           1., 0., 0., 0., .1, .2, .3) for index in range(6)]
    source = tmp_path / "splat.ply"
    source.write_bytes(header + b"".join(records))
    original = source.read_bytes()
    reduced = tmp_path / "quest-preview.ply"
    artifact = make_quest_preview(source, reduced, max_vertices=3)
    assert source.read_bytes() == original
    assert artifact["vertex_count"] == 3
    assert artifact["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert validate_gaussian_ply(reduced)[0] == 3
    body = reduced.read_bytes().split(b"end_header\n", 1)[1]
    # Merged opacities/footprints must represent discarded neighbors; copying
    # three unchanged rows cannot preserve the original surface coverage.
    assert body != records[1] + records[3] + records[5]
    assert artifact["method"] == "playcanvas_adaptive_pairwise_merge"


def test_failed_quest_merge_preserves_source_and_existing_preview(tmp_path, monkeypatch):
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    data = ("ply\nformat binary_little_endian 1.0\nelement vertex 6\n" +
            "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    data += struct.pack("<14f", *([0.] * 14)) * 6
    source, destination = tmp_path / "source.ply", tmp_path / "preview.ply"
    source.write_bytes(data)
    destination.write_bytes(b"previous accepted preview")
    def fail(*_args, **_kwargs):
        raise ReconstructionError("Simplifier failed")
    monkeypatch.setattr("bridge.reconstruct._run", fail)
    with pytest.raises(ReconstructionError, match="Simplifier failed"):
        make_quest_preview(source, destination, max_vertices=3)
    assert source.read_bytes() == data
    assert destination.read_bytes() == b"previous accepted preview"


def test_small_quest_copy_is_lossless(tmp_path):
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    data = ("ply\nformat binary_little_endian 1.0\nelement vertex 6\n" +
            "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    data += struct.pack("<14f", *([0.] * 14)) * 6
    source, destination = tmp_path / "source.ply", tmp_path / "preview.ply"
    source.write_bytes(data)
    result = make_quest_preview(source, destination)
    assert destination.read_bytes() == data
    assert result["method"] == "source_copy"


def test_native_orientation_requires_explicit_review(tmp_path, monkeypatch):
    class SnapshotStore:
        root = tmp_path

        def export_session(self, _session_id):
            return {
                "session": {"project_id": "project"},
                "captures": [
                    {"metadata": {"capture_source": "quest_camera", "image_orientation": "unity_texture_native"},
                     "receipt": {"capture_id": str(i), "image_sha256": "a" * 64}}
                    for i in range(6)
                ],
            }

    store = SnapshotStore()
    with pytest.raises(ReconstructionError, match="explicitly approve"):
        build_session(store, "session")
    def stop_at_tool(_name):
        raise ReconstructionError("Reached dependency check")
    monkeypatch.setattr("bridge.reconstruct._tool", stop_at_tool)
    with pytest.raises(ReconstructionError, match="Reached dependency check"):
        build_session(store, "session", native_orientation_reviewed=True)


def test_normalization_undoes_colmap_applied_transform(tmp_path):
    """Saved camera frames already include COLMAP's world-axis conversion."""
    processed = tmp_path / "processed"
    processed.mkdir()
    training = tmp_path / "training"
    training.mkdir()
    applied = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=float)
    saved_to_training = np.eye(4)
    saved_to_training[:3, 3] = [1, 2, 3]
    original_to_training = saved_to_training @ np.vstack([applied, [0, 0, 0, 1]])
    (processed / "transforms.json").write_text(json.dumps({"applied_transform": applied.tolist()}))
    (training / "dataparser_transforms.json").write_text(json.dumps({
        "transform": original_to_training[:3, :].tolist(), "scale": 0.25}))
    transform, scale, root_id = _normalization({"processed_dir": str(processed),
                                                "training_config": str(training / "config.yml"),
                                                "job_id": "root"})
    assert np.allclose(transform, saved_to_training[:3, :])
    assert scale == 0.25 and root_id == "root"
    pose = _pose_in_training(np.array([1., 0., 0., 0.]),
                             np.array([0., 0., 0.]), transform, scale)
    assert np.allclose(pose[:3, 3], [0.25, 0.5, 0.75])


def test_new_registered_pose_must_agree_with_old_quest_path():
    identity = np.eye(4)
    captures = []
    model = {}
    frame_map = {}
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    for index, position in enumerate(positions):
        matrix = np.eye(4)
        matrix[:3, 3] = position
        capture_id = f"capture-{index}"
        frame_name = f"frame_{index:05d}.jpg"
        captures.append({"receipt": {"capture_id": capture_id},
                         "metadata": {"camera_to_session": matrix.reshape(-1).tolist()}})
        frame_map[capture_id] = frame_name
        # Identity COLMAP qvec, tvec=-camera center; saved axes swap Y/Z.
        model[frame_name] = (np.array([1., 0., 0., 0.]), -np.asarray(position, dtype=float))
    snapshot = {"captures": captures}
    result = _check_quest_pose_consistency(snapshot, frame_map, model,
                                           {"frame_00003.jpg"}, identity[:3, :], 1.0)
    assert result["registered_new_views"] == 1
    assert result["headset_alignment_verified"] is False
    model["frame_00003.jpg"] = (np.array([1., 0., 0., 0.]), np.array([-30., 0., 0.]))
    with pytest.raises(ReconstructionError, match="disagree with Quest tracking"):
        _check_quest_pose_consistency(snapshot, frame_map, model,
                                      {"frame_00003.jpg"}, identity[:3, :], 1.0)


def test_exact_ready_revision_is_reused_without_starting_tools(tmp_path, monkeypatch):
    project_id = "project"
    session_id = "session"
    reconstruction = tmp_path / "projects" / project_id / "reconstructions" / "old-job"
    processed = reconstruction / "processed"
    checkpoint = reconstruction / "training" / "nerfstudio_models"
    processed.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    config = checkpoint.parent / "config.yml"
    config.write_text("camera_optimizer: off\n")
    (checkpoint / "step-000000001.ckpt").write_bytes(b"checkpoint")
    artifact = reconstruction / "splat.ply"
    artifact.write_bytes(b"existing artifact")
    captures = []
    for index in range(3):
        captures.append({"receipt": {"capture_id": str(index), "image_sha256": f"{index}" * 64},
                         "metadata": {"capture_source": "quest_camera", "image_orientation": "upright"}})
    job = {"job_id": "old-job", "session_id": session_id, "status": "ready_for_blender",
           "quality": "quick", "processed_dir": str(processed),
           "checkpoint_dir": str(checkpoint), "training_config": str(config),
           "input_captures": [{"capture_id": item["receipt"]["capture_id"],
                               "image_sha256": item["receipt"]["image_sha256"]} for item in captures],
           "artifact": {"path": str(artifact), "size_bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    (reconstruction / "job.json").write_text(json.dumps(job))

    class SnapshotStore:
        root = tmp_path

        def export_session(self, _session_id):
            return {"session": {"project_id": project_id}, "captures": captures}

    def forbidden_tool(_name):
        raise AssertionError("Existing immutable result should not start external tools")

    monkeypatch.setattr("bridge.reconstruct._tool", forbidden_tool)
    assert build_session(SnapshotStore(), session_id, quality="quick")["job_id"] == "old-job"


@pytest.mark.parametrize("candidate", [False, True])
def test_explicit_fresh_baseline_keeps_old_ready_artifact_and_never_loads_checkpoint(tmp_path, monkeypatch, candidate):
    project_id, session_id = "project", "session"
    old = tmp_path / "projects" / project_id / "reconstructions" / "old-job"
    processed = old / "processed"
    checkpoint = old / "training" / "nerfstudio_models"
    processed.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    config = checkpoint.parent / "config.yml"
    config.write_text("camera_optimizer:\n  mode: off\n")
    (checkpoint / "step-000000001.ckpt").write_bytes(b"checkpoint")
    artifact = old / "splat.ply"
    artifact.write_bytes(b"previous ready preview")
    captures = []
    for index in range(4):
        capture_id = str(index)
        source = tmp_path / f"{capture_id}.jpg"
        source.write_bytes(f"photo-{index}".encode())
        captures.append({"receipt": {"capture_id": capture_id,
                                     "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                         "metadata": {"capture_source": "quest_camera", "image_orientation": "upright"},
                         "image_path": str(source)})
    parent = {"job_id": "old-job", "session_id": session_id, "status": "ready_for_blender",
              "quality": "quick", "processed_dir": str(processed),
              "checkpoint_dir": str(checkpoint), "training_config": str(config),
              "input_captures": [{"capture_id": item["receipt"]["capture_id"],
                                  "image_sha256": item["receipt"]["image_sha256"]} for item in captures[:3]],
              "artifact": {"path": str(artifact), "size_bytes": artifact.stat().st_size,
                           "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    (old / "job.json").write_text(json.dumps(parent))

    class SnapshotStore:
        root = tmp_path

        def export_session(self, _session_id):
            return {"session": {"project_id": project_id}, "captures": captures}

    properties = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 4\n" +
              "".join(f"property float {name}\n" for name in properties) + "end_header\n").encode()
    fake_ply = header + struct.pack("<56f", *([0.0] * 56))
    commands = []

    def fake_run(args, _log, _env, timeout_seconds=None):
        commands.append((args, timeout_seconds))
        if args[0] == "ns-process-data":
            folder = __import__("pathlib").Path(args[args.index("--output-dir") + 1])
            folder.mkdir(parents=True)
            (folder / "transforms.json").write_text(json.dumps({"frames": [{} for _ in captures]}))
        elif len(args) > 1 and args[1].endswith("nerfstudio_resume.py"):
            with _log.open("ab") as log:
                log.write(b"Verified gsplat DefaultStrategy opacity reset compatibility: patched_faulty\n")
            folder = __import__("pathlib").Path(args[args.index("--output-dir") + 1]) / "run"
            (folder / "nerfstudio_models").mkdir(parents=True)
            (folder / "config.yml").write_text("config")
            (folder / "nerfstudio_models" / "step-000002000.ckpt").write_bytes(b"checkpoint")
        else:
            folder = __import__("pathlib").Path(args[args.index("--output-dir") + 1])
            folder.mkdir(parents=True)
            (folder / "splat.ply").write_bytes(fake_ply)

    monkeypatch.setattr("bridge.reconstruct._tool", lambda name: name)
    monkeypatch.setattr("bridge.reconstruct._run", fake_run)
    monkeypatch.setattr("bridge.reconstruct._prepare_incremental", lambda *_: pytest.fail("Inherited parent used"))
    result = build_session(SnapshotStore(), session_id, quality="quick", fresh_baseline=True,
                           candidate=candidate)
    assert result["status"] == ("candidate_for_review" if candidate else "ready_for_blender")
    assert result["processing_mode"] == "fresh_baseline"
    assert result["parent_job_id"] == "old-job"
    assert result["inherited_parent_checkpoint"] is False
    assert result["registered_views"] == 4
    assert all("--load-dir" not in args for args, _ in commands)
    train = next(args for args, _ in commands if len(args) > 1 and
                 args[1].endswith("nerfstudio_resume.py"))
    steps = int(train[train.index("--max-num-iterations") + 1])
    schedule = int(train[train.index("--pipeline.model.resolution-schedule") + 1])
    downscales = int(train[train.index("--pipeline.model.num-downscales") + 1])
    split_stop = int(train[train.index("--pipeline.model.stop-split-at") + 1])
    assert steps - schedule * downscales >= 4000
    assert steps - split_stop >= 1000
    assert result["training_profile"]["full_resolution_from_step"] == schedule * downscales
    assert result["opacity_reset_compatibility"] == "patched_faulty"
    assert result["optimizer_binding_verified"] is False
    assert [timeout for _, timeout in commands] == [900, 900, 180]
    assert artifact.read_bytes() == b"previous ready preview"
    assert json.loads((old / "job.json").read_text())["status"] == "ready_for_blender"


def test_fresh_baseline_is_quick_only(tmp_path):
    class SnapshotStore:
        root = tmp_path

    with pytest.raises(ReconstructionError, match="explicit quick"):
        build_session(SnapshotStore(), "session", quality="final", fresh_baseline=True)


def _bounded_candidate_fixture(tmp_path):
    project_id, session_id, job_id = (str(uuid.uuid4()) for _ in range(3))
    root = tmp_path / "projects" / project_id / "reconstructions" / job_id
    processed = root / "processed"
    checkpoint = root / "training" / "run" / "nerfstudio_models"
    processed.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (processed / "transforms.json").write_text(json.dumps({"frames": [{}] * 6}))
    (checkpoint / "step-000004999.ckpt").write_bytes(b"original optimizer checkpoint")
    config = checkpoint.parent / "config.yml"
    config.write_text("max_num_iterations: 5000\n  num_downscales: 1\n"
                      "  resolution_schedule: 250\n  stop_split_at: 3500\n"
                      "  use_scale_regularization: true\n  use_bilateral_grid: false\n"
                      "  camera_optimizer:\n    mode: 'off'\n")
    names = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    ply = ("ply\nformat binary_little_endian 1.0\nelement vertex 6\n" +
           "".join(f"property float {name}\n" for name in names) + "end_header\n").encode()
    ply += struct.pack("<14f", *([0.] * 14)) * 6
    artifact = root / "splat.ply"
    artifact.write_bytes(ply)
    captures = []
    for index in range(7):
        cid = str(uuid.uuid4())
        source = tmp_path / f"{cid}.jpg"
        source.write_bytes(f"photo-{index}".encode())
        captures.append({"receipt": {"capture_id": cid,
                         "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
                         "metadata": {"capture_source": "quest_camera", "image_orientation": "upright"},
                         "image_path": str(source)})
    parent = {"job_id": job_id, "project_id": project_id, "session_id": session_id,
              "status": "ready_for_blender", "quality": "quick", "processed_dir": str(processed),
              "checkpoint_dir": str(checkpoint), "training_config": str(config),
              "training_profile": {"version": 2, "num_downscales": 1,
                                   "resolution_schedule": 250, "full_resolution_from_step": 250,
                                   "scale_regularization": True},
              "input_captures": [{"capture_id": item["receipt"]["capture_id"],
                                  "image_sha256": item["receipt"]["image_sha256"]}
                                 for item in captures[:6]],
              "artifact": {"path": str(artifact), "size_bytes": artifact.stat().st_size,
                           "sha256": hashlib.sha256(ply).hexdigest()}}
    (root / "job.json").write_text(json.dumps(parent))

    class SnapshotStore:
        root = tmp_path
        selected = None

        def session(self, value):
            assert value == session_id
            return {"project_id": project_id}

        def export_session(self, value, capture_ids=None):
            assert value == session_id
            self.selected = capture_ids
            chosen = set(capture_ids or [item["receipt"]["capture_id"] for item in captures])
            return {"session": {"project_id": project_id},
                    "captures": [item for item in captures if item["receipt"]["capture_id"] in chosen]}

    return SnapshotStore(), parent, captures, ply


def _warmup_log(count):
    evidence = {"start_step": 5000, "train_camera_count": count, "refine_every": 100,
                "warmup_steps": max(count, 100), "growth_hold_through_step": 5000 + max(count, 100) - 1,
                "effective_refine_start_iter": 5000 + max(count, 100) - 1,
                "refinement_available": True}
    return ("Verified resumed Splatfacto refinement warmup: " + json.dumps(evidence) + "\n").encode()


def test_bounded_candidate_resumes_exact_seed_photos_and_keeps_original(tmp_path, monkeypatch):
    store, parent, captures, original_ply = _bounded_candidate_fixture(tmp_path)
    commands = []

    def fake_run(args, _log, _env, timeout_seconds=None):
        commands.append(args)
        if args[0] == "ns-process-data":
            pytest.fail("Same-data continuation unexpectedly re-ran COLMAP/photo processing")
        if len(args) > 1 and args[1].endswith("nerfstudio_resume.py"):
            with _log.open("ab") as log:
                log.write(b"Verified gsplat DefaultStrategy opacity reset compatibility: patched_faulty\n"
                          b"Verified Splatfacto optimizer bindings for six restored Gaussian parameter groups\n" +
                          _warmup_log(6))
            training_root = __import__("pathlib").Path(args[args.index("--output-dir") + 1])
            running = json.loads((training_root.parent / "job.json").read_text())
            assert running["checkpoint_start_step"] == 4999
            assert running["target_training_step"] == 10000
            assert running["frozen_source_job_id"] == parent["job_id"]
            assert running["frozen_source_sha256"] == parent["artifact"]["sha256"]
            assert running["status"] == "running"
            folder = training_root / "run"
            (folder / "nerfstudio_models").mkdir(parents=True)
            (folder / "config.yml").write_text(
                __import__("pathlib").Path(parent["training_config"]).read_text())
            (folder / "nerfstudio_models" / "step-000009999.ckpt").write_bytes(b"candidate")
        else:
            folder = __import__("pathlib").Path(args[args.index("--output-dir") + 1])
            folder.mkdir(parents=True)
            candidate_ply = bytearray(original_ply)
            data_start = original_ply.index(b"end_header\n") + len(b"end_header\n")
            candidate_ply[data_start:data_start + 4] = struct.pack("<f", 1.)
            (folder / "candidate.ply").write_bytes(candidate_ply)

    monkeypatch.setattr("bridge.reconstruct._tool", lambda name: name)
    monkeypatch.setattr("bridge.reconstruct._run", fake_run)
    result = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=10000)
    assert set(store.selected) == {item["receipt"]["capture_id"] for item in captures[:6]}
    assert len(result["input_captures"]) == 6 and captures[6]["receipt"]["capture_id"] not in str(result)
    assert result["status"] == "candidate_for_review"
    assert result["processing_mode"] == "same_data_bounded_candidate"
    assert result["processed_dir"] == parent["processed_dir"]
    assert result["checkpoint_start_step"] == 4999 and result["target_training_step"] == 10000
    assert result["completed_training_step"] == 9999
    assert result["resume_runner_version"] == 3 and result["optimizer_binding_verified"] is True
    assert result["resume_refinement_warmup"]["growth_hold_through_step"] == 5099
    assert result["opacity_reset_compatibility"] == "patched_faulty"
    assert result["artifact"]["sha256"] != parent["artifact"]["sha256"]
    assert result["export_checkpoint_sha256"] == hashlib.sha256(b"candidate").hexdigest()
    assert result["publication"] == "candidate_review"
    train = next(args for args in commands if len(args) > 1 and args[1].endswith("nerfstudio_resume.py"))
    for flag, value in (("--load-dir", parent["checkpoint_dir"]),
                        ("--max-num-iterations", "5000"),
                        ("--pipeline.model.num-downscales", "1"),
                        ("--pipeline.model.stop-split-at", "3500"),
                        ("--pipeline.model.use-scale-regularization", "True")):
        assert train[train.index(flag) + 1] == value
    assert (tmp_path / "projects" / parent["project_id"] / "reconstructions" /
            parent["job_id"] / "splat.ply").read_bytes() == original_ply
    assert (tmp_path / "projects" / parent["project_id"] / "reconstructions" /
            parent["job_id"] / "training" / "run" / "nerfstudio_models" /
            "step-000004999.ckpt").read_bytes() == b"original optimizer checkpoint"
    before = len(commands)
    repeated = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=10000)
    assert repeated["job_id"] == result["job_id"] and len(commands) == before
    old_manifest = (tmp_path / "projects" / parent["project_id"] / "reconstructions" /
                    result["job_id"] / "job.json")
    old = json.loads(old_manifest.read_text())
    old["resume_runner_version"] = 1
    old.pop("opacity_reset_compatibility")
    old_manifest.write_text(json.dumps(old))
    rebuilt = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=10000)
    assert rebuilt["job_id"] != result["job_id"] and len(commands) > before
    (__import__("pathlib").Path(rebuilt["checkpoint_dir"]) / "step-000009999.ckpt").write_bytes(b"tampered")
    newest = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=10000)
    assert newest["job_id"] not in {result["job_id"], rebuilt["job_id"]}


@pytest.mark.parametrize("target", [6500, 20000])
def test_incremental_review_registers_new_photos_then_trains_only_requested_steps(tmp_path, monkeypatch, target):
    store, parent, captures, original_ply = _bounded_candidate_fixture(tmp_path)
    commands = []
    preparations = []

    def prepare(source, snapshot, root, *_args):
        preparations.append((source["job_id"], [x["receipt"]["capture_id"] for x in snapshot["captures"]]))
        processed = root / "processed"
        processed.mkdir()
        (processed / "transforms.json").write_text(json.dumps({"frames": [{} for _ in captures]}))
        return {"processed_dir": str(processed), "registered_views": len(captures),
                "training_split": {"version": 1, "method": "preserve_parent_membership",
                    "source_training_config_sha256": hashlib.sha256(__import__("pathlib").Path(source["training_config"]).read_bytes()).hexdigest(),
                    "source_transforms_sha256": hashlib.sha256((__import__("pathlib").Path(source["processed_dir"]) / "transforms.json").read_bytes()).hexdigest(),
                    "transforms_sha256": hashlib.sha256((processed / "transforms.json").read_bytes()).hexdigest()},
                "new_registered_views": 1, "pose_consistency": {"status": "checked"}}

    def fake_run(args, log, _env, timeout_seconds=None):
        commands.append(args)
        if len(args) > 1 and args[1].endswith("nerfstudio_resume.py"):
            with log.open("ab") as stream:
                stream.write(b"Verified gsplat DefaultStrategy opacity reset compatibility: patched_faulty\n"
                             b"Verified Splatfacto optimizer bindings for six restored Gaussian parameter groups\n" +
                             _warmup_log(7))
            training = __import__("pathlib").Path(args[args.index("--output-dir") + 1]) / "run"
            (training / "nerfstudio_models").mkdir(parents=True)
            (training / "config.yml").write_text(f"max_num_iterations: {target - 5000}\n")
            (training / "nerfstudio_models" / f"step-{target - 1:09d}.ckpt").write_bytes(b"new views")
        elif args[0] == "ns-export":
            folder = __import__("pathlib").Path(args[args.index("--output-dir") + 1])
            folder.mkdir()
            ply = bytearray(original_ply)
            offset = original_ply.index(b"end_header\n") + len(b"end_header\n")
            ply[offset:offset + 4] = struct.pack("<f", 2.)
            (folder / "splat.ply").write_bytes(ply)
        else:
            pytest.fail(f"Unexpected fresh reconstruction command: {args[0]}")

    monkeypatch.setattr("bridge.reconstruct._tool", lambda name: name)
    monkeypatch.setattr("bridge.reconstruct._prepare_incremental", prepare)
    monkeypatch.setattr("bridge.reconstruct._run", fake_run)
    result = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=target,
        incremental_new_captures=True)
    assert result["status"] == "candidate_for_review"
    assert result["publication"] == "candidate_review"
    assert result["processing_mode"] == "new_capture_incremental_bounded_candidate"
    assert result["parent_job_id"] == parent["job_id"]
    assert len(result["input_captures"]) == 7 and result["new_registered_views"] == 1
    assert preparations == [(parent["job_id"], [x["receipt"]["capture_id"] for x in captures])]
    assert result["checkpoint_start_step"] == 4999
    assert result["target_training_step"] == target and result["completed_training_step"] == target - 1
    assert result["resume_runner_version"] == 3 and result["optimizer_binding_verified"] is True
    assert result["opacity_reset_compatibility"] == "patched_faulty"
    train = commands[0]
    assert train[train.index("--load-dir") + 1] == parent["checkpoint_dir"]
    assert train[train.index("--max-num-iterations") + 1] == str(target - 5000)
    assert (tmp_path / "projects" / parent["project_id"] / "reconstructions" /
            parent["job_id"] / "training" / "run" / "nerfstudio_models" /
            "step-000004999.ckpt").read_bytes() == b"original optimizer checkpoint"
    again = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=target,
        incremental_new_captures=True)
    assert again["job_id"] == result["job_id"] and len(preparations) == 1 and len(commands) == 2
    # Pre-fix candidates cannot satisfy a new request even with intact PLY and
    # checkpoint: their fraction split may have trained reserved source views.
    from pathlib import Path
    manifest = Path(result["artifact"]["path"]).with_name("job.json")
    stale = json.loads(manifest.read_text())
    stale.pop("training_split")
    manifest.write_text(json.dumps(stale))
    rebuilt = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=target,
        incremental_new_captures=True)
    assert rebuilt["job_id"] != result["job_id"] and len(preparations) == 2
    # The frozen source split config is part of the candidate's reuse identity.
    config = Path(parent["training_config"])
    config.write_text(config.read_text() + "train_split_fraction: 0.8\n")
    changed_source = build_session(store, parent["session_id"], quality="final", candidate=True,
        frozen_source_job_id=parent["job_id"], target_training_step=target,
        incremental_new_captures=True)
    assert changed_source["job_id"] != rebuilt["job_id"] and len(preparations) == 3


def _split_fixture(tmp_path, count=53):
    config = tmp_path / "config.yml"
    config.write_text("dataparser: !!python/object:nerfstudio.data.dataparsers.nerfstudio_dataparser.NerfstudioDataParserConfig\n"
        "  _target: !!python/name:nerfstudio.data.dataparsers.nerfstudio_dataparser.Nerfstudio ''\n"
        "  eval_mode: fraction\n  train_split_fraction: 0.9\n")
    parent = {"job_id": str(uuid.uuid4()), "training_config": str(config)}
    frames = [{"file_path": f"images/frame_{index:05d}.jpg"} for index in range(1, count + 1)]
    return parent, {"frames": frames}


def test_incremental_split_preserves_all_five_seed_holdouts_and_old_training(tmp_path):
    parent, source = _split_fixture(tmp_path)
    frames = source["frames"] + [{"file_path": f"images/frame_{index:05d}.jpg"}
                                  for index in range(54, 118)]
    source_bytes = json.dumps(source, sort_keys=True)
    result = _incremental_training_split(parent, source, list(reversed(frames)))
    old = result["parent_membership"]
    assert len(old["train"]) == 48 and len(old["val"]) == 5
    legacy_train = {frames[index]["file_path"] for index in np.linspace(0, 116, 106, dtype=int)}
    # Reproduce the real 53 -> 117 defect, with three original holdouts leaked.
    assert len(legacy_train & set(old["val"])) == 3
    assert set(old["val"]).isdisjoint(result["filenames"]["train_filenames"])
    assert set(old["train"]) <= set(result["filenames"]["train_filenames"])
    assert set(old["val"]) <= set(result["filenames"]["val_filenames"])
    assert old["test"] == old["val"]
    assert len(result["new_membership"]["train"]) == 58
    assert len(result["new_membership"]["val"]) == 6
    assert json.dumps(source, sort_keys=True) == source_bytes
    assert result == _incremental_training_split(parent, source, frames)


def test_saved_comparison_uses_explicit_heldouts_instead_of_global_fraction(tmp_path):
    # Load only the comparison script's pure split reader: its rendering imports
    # require the optional GPU environment, while this regression is CPU-only.
    script = Path(__file__).resolve().parents[2] / "scripts" / "Compare-QuestSplat.py"
    module = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    function = next(row for row in module.body if isinstance(row, ast.FunctionDef)
                    and row.name == "evaluation_ids")
    scope = {"json": json, "Path": Path, "Store": Store,
             "_incremental_training_split": _incremental_training_split}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"), scope)
    evaluation_ids = scope["evaluation_ids"]

    parent, seed = _split_fixture(tmp_path)
    old_frames = seed["frames"]
    frames = old_frames + [{"file_path": f"images/frame_{i:05d}.jpg"}
                           for i in range(54, 118)]
    split = _incremental_training_split(parent, seed, frames)
    saved = {"frames": frames, **split["filenames"]}
    processed = tmp_path / "processed"
    processed.mkdir()
    transforms = processed / "transforms.json"
    transforms.write_text(json.dumps(saved))
    split["transforms_sha256"] = Store._file_sha256(transforms)
    job = {**parent, "processed_dir": str(processed), "training_split": split,
           "frame_manifest": {f"capture-{i}": f"frame_{i:05d}.jpg"
                              for i in range(1, 118)}}
    evaluation = evaluation_ids(job)
    old_holdouts = {f"capture-{int(Path(path).stem[-5:])}"
                    for path in split["parent_membership"]["val"]}
    assert len(evaluation) == 11
    assert old_holdouts <= set(evaluation)
    assert not old_holdouts & {f"capture-{int(Path(path).stem[-5:])}"
                               for path in saved["train_filenames"]}
    # A global 90% split of 117 would leak three of the five old holdouts.
    global_train = set(np.linspace(0, 116, 106, dtype=int))
    assert len({int(Path(path).stem[-5:])-1 for path in split["parent_membership"]["val"]}
               & global_train) == 3
    saved["val_filenames"].pop()
    transforms.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="frozen reconstruction"):
        evaluation_ids(job)


def test_later_incremental_split_keeps_every_existing_explicit_membership(tmp_path):
    parent, source = _split_fixture(tmp_path, 20)
    first_frames = source["frames"] + [{"file_path": f"images/frame_{index:05d}.jpg"}
                                       for index in range(21, 41)]
    first = _incremental_training_split(parent, source, first_frames)
    saved = {"frames": first_frames, **first["filenames"]}
    second_frames = first_frames + [{"file_path": f"images/frame_{index:05d}.jpg"}
                                    for index in range(41, 60)]
    second = _incremental_training_split(parent, saved, second_frames)
    for name in ("train", "val", "test"):
        assert second["parent_membership"][name] == saved[name + "_filenames"]
        assert set(saved[name + "_filenames"]) <= set(second["filenames"][name + "_filenames"])
    assert not (set(second["filenames"]["train_filenames"]) &
                set(second["filenames"]["val_filenames"]))


@pytest.mark.parametrize("change", ["missing_list", "overlap", "foreign_path", "omitted_frame",
                                    "duplicate_frame", "unknown_policy", "bad_fraction",
                                    "aliased_path", "unknown_parser"])
def test_incremental_split_rejects_unverifiable_parent_membership(tmp_path, change):
    parent, source = _split_fixture(tmp_path, 20)
    initial = _incremental_training_split(parent, source, source["frames"])
    source.update(initial["filenames"])
    if change == "missing_list":
        source.pop("test_filenames")
    elif change == "overlap":
        source["train_filenames"].append(source["val_filenames"][0])
    elif change == "foreign_path":
        source["test_filenames"].append("../other/photo.jpg")
    elif change == "omitted_frame":
        source["train_filenames"].pop()
    elif change == "duplicate_frame":
        source["frames"].append(source["frames"][0])
    elif change == "unknown_policy":
        for key in initial["filenames"]:
            source.pop(key)
        config = __import__("pathlib").Path(parent["training_config"])
        config.write_text(config.read_text().replace("eval_mode: fraction", "eval_mode: all"))
    elif change == "bad_fraction":
        config = __import__("pathlib").Path(parent["training_config"])
        config.write_text(config.read_text().replace("train_split_fraction: 0.9", "train_split_fraction: nan"))
    elif change == "aliased_path":
        source["frames"].append({"file_path": "./" + source["frames"][0]["file_path"]})
    elif change == "unknown_parser":
        config = __import__("pathlib").Path(parent["training_config"])
        config.write_text(config.read_text().replace("nerfstudio_dataparser.Nerfstudio ''", "custom.NativeLikeParser ''"))
    before = json.dumps(source, sort_keys=True)
    with pytest.raises(ReconstructionError, match="split"):
        _incremental_training_split(parent, source, source["frames"])
    assert json.dumps(source, sort_keys=True) == before


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_start", "wrong_count", "short_hold"])
def test_resume_receipt_rejects_unverified_warmup(change):
    log = _warmup_log(6).decode()
    if change == "missing":
        log = ""
    elif change == "duplicate":
        log += log
    else:
        evidence = json.loads(log.split(": ", 1)[1])
        if change == "wrong_start":
            evidence["start_step"] = 4999
        elif change == "wrong_count":
            evidence["train_camera_count"] = 7
        elif change == "short_hold":
            evidence["growth_hold_through_step"] = 5000
        log = "Verified resumed Splatfacto refinement warmup: " + json.dumps(evidence) + "\n"
    with pytest.raises(ReconstructionError, match="refinement warmup"):
        _verified_resume_warmup(log, 5000, 6, 6)


@pytest.mark.parametrize("target", [5999, 20001])
def test_incremental_review_rejects_out_of_bound_target_before_tools(tmp_path, monkeypatch, target):
    store, parent, _, _ = _bounded_candidate_fixture(tmp_path)
    monkeypatch.setattr("bridge.reconstruct._tool", lambda name: pytest.fail("Tools started before preflight"))
    with pytest.raises(ReconstructionError, match="1000 to 15000"):
        build_session(store, parent["session_id"], quality="final", candidate=True,
            frozen_source_job_id=parent["job_id"], target_training_step=target,
            incremental_new_captures=True)


@pytest.mark.parametrize("change", ["too_short", "too_long", "unreviewed_profile",
                                    "wrong_source_hash", "native_without_review_flag"])
def test_bounded_candidate_rejects_unsafe_source_before_start(tmp_path, monkeypatch, change):
    store, parent, _, _ = _bounded_candidate_fixture(tmp_path)
    if change == "unreviewed_profile":
        __import__("pathlib").Path(parent["training_config"]).write_text(
            __import__("pathlib").Path(parent["training_config"]).read_text().replace(
                "stop_split_at: 3500", "stop_split_at: 15000"))
    if change == "wrong_source_hash":
        __import__("pathlib").Path(parent["artifact"]["path"]).write_bytes(b"changed")
    if change == "native_without_review_flag":
        manifest = tmp_path / "projects" / parent["project_id"] / "reconstructions" / parent["job_id"] / "job.json"
        parent.update(source_image_orientations=["unity_texture_native"], native_orientation_reviewed=True)
        manifest.write_text(json.dumps(parent))
    monkeypatch.setattr("bridge.reconstruct._tool", lambda name: pytest.fail("Tools started before preflight"))
    target = 5200 if change == "too_short" else 16000 if change == "too_long" else 10000
    with pytest.raises(ReconstructionError):
        build_session(store, parent["session_id"], quality="final", candidate=True,
            frozen_source_job_id=parent["job_id"], target_training_step=target)
    assert len(list((tmp_path / "projects" / parent["project_id"] / "reconstructions").iterdir())) == 1
