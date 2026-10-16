import hashlib
import io
import json
import sqlite3
import struct
import uuid
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bridge.app import create_app
from bridge.config import Config
from bridge.dashboard import dashboard_current_model
from bridge.online import (OnlineCoordinator, OnlineError, _replace_with_retry,
                           triangulate_correspondences, validated_camera,
                           registration_from_seed_job, bootstrap_seed_job,
                           PhotoRegistrationVerifier, ColmapLandmarkVerifier,
                           _colmap_localization_source, seed_camera_views, _run_online_worker,
                           _recover_online_publication)
from bridge.store import Store


PROPS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")


def test_snapshot_publish_survives_transient_windows_reader_lock(tmp_path, monkeypatch):
    source, destination = tmp_path / "new.ply", tmp_path / "current.ply"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")
    original = __import__("os").replace
    attempts = []

    def locked_once(src, dst):
        attempts.append(1)
        if len(attempts) == 1:
            assert destination.read_bytes() == b"old"
            raise PermissionError("temporary viewer lock")
        return original(src, dst)

    monkeypatch.setattr("bridge.online.os.replace", locked_once)
    monkeypatch.setattr("bridge.online.time.sleep", lambda _seconds: None)
    _replace_with_retry(source, destination)
    assert len(attempts) == 2
    assert destination.read_bytes() == b"new"


def splat_bytes(count=3):
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n" +
              "".join(f"property float {name}\n" for name in PROPS) + "end_header\n")
    return header.encode() + struct.pack("<14f", *([0.] * 14))*count


class FakeTrainer:
    def __init__(self, ids=()):
        self.ids = list(ids)
        self.steps = 0
        self.identity = id(self)
        self.baseline = []

    def configure_geometry_scale(self, scale):
        self.scale = scale

    def configure_camera_distortion(self, distortion):
        self.distortion = list(distortion)

    def prime_baseline_views(self, views):
        self.baseline = list(views)
        return len(self.baseline)

    @property
    def capture_ids(self):
        return list(self.ids)

    @property
    def published_capture_ids(self):
        return list(self.ids)

    def add_image(self, capture_id, image_path, pose, intrinsics):
        assert image_path.is_file()
        assert pose.shape == (4, 4) and intrinsics.shape == (3, 3)
        self.ids.append(capture_id)

    def train_steps(self, count):
        self.steps += count
        return count

    def save(self, checkpoint, ply):
        checkpoint.write_text(json.dumps(self.ids))
        ply.write_bytes(splat_bytes())
        return {}


def scene(tmp_path):
    store = Store(tmp_path)
    project = store.create_project("House")
    session = store.create_session(project["id"], "origin")
    seed = tmp_path / "real-seed.ply"
    seed.write_bytes(splat_bytes())
    registration = {"status": "verified", "session_id": session["id"],
                    "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
                    "basis": "opencv_rh_from_unity", "evidence_job_id": "reviewed-seed",
                    "calibration_revision": "camera-1", "model_from_session": np.eye(4).tolist(),
                    "physical_acceptance": "accepted",
                    "pose_fit": {"matched_views": 4, "baseline_m": 0.5,
                                 "position_rmse_m": 0.01, "max_position_error_m": 0.02,
                                 "max_rotation_error_deg": 2.0}}
    return store, project, session, seed, registration


def capture(store, project, session, capture_id):
    image = b"original-quest-jpeg-" + capture_id.encode()
    metadata = {"capture_id": capture_id, "project_id": project["id"],
                "session_id": session["id"], "tracking_origin_id": "origin",
                "image_sha256": hashlib.sha256(image).hexdigest(), "image_encoding": "jpeg",
                "capture_source": "quest_camera", "tracking_valid": True,
                "image_orientation": "upright", "calibration_revision": "camera-1",
                "coordinate_convention": "unity_lh_xright_yup_zforward",
                "camera_to_session": np.eye(4).reshape(-1).tolist(),
                "intrinsics": {"fx": 100., "fy": 100., "cx": 50., "cy": 50.}}
    store.save_capture(session["id"], metadata, image)


def test_one_trainer_accepts_new_receipts_once_and_resumes(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    first, second = "first", "second"
    capture(store, project, session, first)
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer)
    status = worker.tick(steps=3)
    assert status["accepted_ids"] == [first] and status["training_steps"] == 3
    assert trainer.identity == id(worker.trainer)
    assert worker.ply_path.read_bytes() == splat_bytes()
    capture(store, project, session, second)
    status = worker.tick(steps=2)
    assert status["accepted_ids"] == [first, second]
    assert trainer.steps == 5 and trainer.capture_ids == [first, second]
    worker.tick(steps=1)
    assert trainer.capture_ids == [first, second]
    resumed = OnlineCoordinator(store, session["id"], seed, registration,
                                FakeTrainer(json.loads(worker.checkpoint_path.read_text())))
    resumed.tick(steps=0)
    assert resumed.status["accepted_ids"] == [first, second]


def test_quality_review_pause_survives_worker_restart(tmp_path):
    from bridge.online import ensure_refinement_allowed
    status = tmp_path / "status.json"
    ensure_refinement_allowed(status)
    status.write_text(json.dumps({"status": "paused_quality_review"}))
    with pytest.raises(OnlineError, match="paused for quality review"):
        ensure_refinement_allowed(status)
    status.write_text("not json")
    with pytest.raises(OnlineError, match="Cannot verify"):
        ensure_refinement_allowed(status)


def test_area_baseline_reprocessing_counts_original_ids_once_without_changing_lineage():
    worker = object.__new__(OnlineCoordinator)
    worker.seed_ids = {'seed-1', 'seed-2'}
    worker.status = {'accepted_ids': ['seed-1', 'delta', 'pending', 'rejected']}
    class Lineage:
        published_capture_ids = ['seed-1', 'delta']
        rejected_capture_ids = ['rejected']
        pending_refinement = {'capture_ids': ['pending'], 'steps': 32}
    worker.trainer = Lineage()
    worker._update_counts()
    assert worker.status['seed_view_count'] == 2
    assert worker.status['processed_count'] == 5 and worker.status['visible_count'] == 3
    assert worker.status['rejected_count'] == worker.status['pending_count'] == 1
    assert worker.status['accepted_ids'] == ['seed-1', 'delta', 'pending', 'rejected']
    assert worker.status['published_capture_ids'] == ['seed-1', 'delta']


@pytest.mark.parametrize('changed', [None, 'seed_override', 'missing_delta', 'claimed_seed'])
def test_reprocessed_seed_publication_keeps_authoritative_original_proof(changed):
    from bridge.online import _publication_ledger
    registration = {'seed_capture_ids': ['seed'], 'source_proof': {'captures': {
        'seed': {'image_sha256': 'a'*64, 'metadata_sha256': 'b'*64}}}}
    status = {'seed_ids': ['seed'], 'accepted_ids': ['seed', 'delta'], 'seed_sha256': 'c'*64,
        'photo_registration': {'delta': {'image_sha256': 'd'*64, 'metadata_sha256': 'e'*64,
            'seed_sha256': 'c'*64}}}
    if changed == 'seed_override':
        status['photo_registration']['seed'] = {'image_sha256': 'f'*64, 'metadata_sha256': 'b'*64}
    elif changed == 'missing_delta':
        status['photo_registration'].clear()
    elif changed == 'claimed_seed':
        status['seed_ids'].append('delta')
    if changed:
        with pytest.raises(OnlineError):
            _publication_ledger(status, registration)
    else:
        assert _publication_ledger(status, registration) == {
            'seed': {'image_sha256': 'a'*64, 'metadata_sha256': 'b'*64},
            'delta': {'image_sha256': 'd'*64, 'metadata_sha256': 'e'*64}}


def test_full_browser_revision_preserves_exact_rows_without_aliasing_trainer_file(tmp_path):
    from bridge.online import stage_browser_revision
    project,session=str(uuid.uuid4()),str(uuid.uuid4())
    folder=tmp_path/'online';folder.mkdir()
    source=tmp_path/'reused-trainer-stage.ply';original=splat_bytes(5)
    source.write_bytes(original)
    receipt=stage_browser_revision(source,folder,project,session,1)
    preserved=folder/'model-revisions'/(receipt['sha256']+'.ply')
    assert preserved.read_bytes()==original
    source.write_bytes(splat_bytes(6))
    assert preserved.read_bytes()==original
    source.write_bytes(original)
    assert stage_browser_revision(source,folder,project,session,2)['sha256']==receipt['sha256']
    preserved.write_bytes(splat_bytes(7))
    with pytest.raises(OnlineError,match='full browser model changed'):
        stage_browser_revision(source,folder,project,session,3)


def test_old_restored_and_current_displays_survive_bounded_cache_eviction(tmp_path, monkeypatch):
    from bridge.online import stage_display_revision
    folder = tmp_path/'online'; revisions = folder/'display-revisions'
    revisions.mkdir(parents=True)
    project, session = str(uuid.uuid4()), str(uuid.uuid4())
    receipts, protected = {}, []
    for number, count in enumerate(range(3, 23), 1):
        payload = splat_bytes(count)
        sha = hashlib.sha256(payload).hexdigest()
        path = revisions/(sha+'.ply'); path.write_bytes(payload)
        receipt = {'project_id': project, 'session_id': session, 'sha256': sha,
            'source_sha256': sha, 'size_bytes': len(payload), 'vertex_count': count,
            'revision_number': number}
        receipts[sha] = json.dumps(receipt).encode()
        path.with_suffix('.json').write_bytes(receipts[sha])
        if count in (3, 4): protected.append(sha)
    (folder/'status.json').write_text(json.dumps({'display_artifact': {'sha256': protected[0]}}))
    source = tmp_path/'restored.ply'; source.write_bytes(splat_bytes(4))
    original_stat = Path.stat
    class OldStat:
        st_ctime_ns = 0
        def __init__(self, actual): self.actual = actual
        def __getattr__(self, name): return getattr(self.actual, name)
    def old_creation_time(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        return OldStat(result) if path.parent == revisions and path.stem in protected else result
    monkeypatch.setattr(Path, 'stat', old_creation_time)
    display = stage_display_revision(source, folder, project, session, 21)
    assert display['sha256'] == protected[1]
    for sha in protected:
        assert (revisions/(sha+'.ply')).is_file()
        assert (revisions/(sha+'.json')).read_bytes() == receipts[sha]
    assert len(list(revisions.glob('*.ply'))) == 16


def test_quality_review_pause_blocks_in_memory_tick(tmp_path):
    store, _project, session, seed, registration = scene(tmp_path)
    worker = OnlineCoordinator(store, session["id"], seed, registration, FakeTrainer())
    baseline = worker.checkpoint_path.read_bytes()
    worker.status["status"] = "paused_quality_review"
    with pytest.raises(OnlineError, match="paused for quality review"):
        worker.tick()
    assert worker.checkpoint_path.read_bytes() == baseline


def test_poll_verifies_only_new_originals(tmp_path, monkeypatch):
    store, project, session, seed, registration = scene(tmp_path)
    capture(store, project, session, "first")
    worker = OnlineCoordinator(store, session["id"], seed, registration, FakeTrainer())
    worker.tick(steps=0)
    capture(store, project, session, "second")
    checked = []
    original = store._file_sha256

    def check(path):
        checked.append(path.name)
        return original(path)

    monkeypatch.setattr(store, "_file_sha256", check)
    worker.tick(steps=0)
    assert checked == ["second.jpg"]
    worker.tick(steps=0)
    assert checked == ["second.jpg"]


def test_gallery_exclusion_prevents_ingress_without_deleting_original(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    capture(store, project, session, "included")
    capture(store, project, session, "excluded")
    with store.connection() as db:
        db.execute("CREATE TABLE capture_selections (capture_id TEXT PRIMARY KEY, included INTEGER)")
        db.execute("INSERT INTO capture_selections VALUES ('excluded', 0)")
    original = (tmp_path / "projects" / project["id"] / "captures" / "excluded.jpg").read_bytes()
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer)
    assert worker.tick(steps=0, maximum_new=8)["accepted_ids"] == ["included"]
    assert (tmp_path / "projects" / project["id"] / "captures" / "excluded.jpg").read_bytes() == original


def test_unverified_registration_and_orientation_are_rejected(tmp_path):
    store, _project, session, seed, registration = scene(tmp_path)
    with pytest.raises(OnlineError, match="verified"):
        OnlineCoordinator(store, session["id"], seed, {**registration, "status": "unregistered"}, FakeTrainer())
    metadata = {"capture_source": "quest_camera", "tracking_valid": True,
                "image_orientation": "unity_texture_native", "distortion": None,
                "calibration_revision": "camera-1",
                "coordinate_convention": "unity_lh_xright_yup_zforward",
                "camera_to_session": np.eye(4).reshape(-1).tolist(),
                "intrinsics": {"fx": 100., "fy": 100., "cx": 50., "cy": 50.}}
    with pytest.raises(OnlineError, match="unverified"):
        validated_camera(metadata, np.eye(4), "camera-1")
    pose, intrinsics = validated_camera({**metadata, "image_orientation": "upright"},
                                        np.eye(4), "camera-1")
    assert np.allclose(pose[:3, :3], np.diag([1., -1., -1.]))
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1.)
    assert intrinsics[0, 0] == 100.


def test_triangulation_keeps_consistent_depth_and_rejects_bad_pairs():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = .5
    k = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    world = np.array([[.1, -.2, 2.], [-.3, .1, 3.], [.2, .2, 4.]])

    def project(points, pose):
        camera = (np.linalg.inv(pose) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
        pixels = (k @ camera.T).T
        return pixels[:, :2] / pixels[:, 2:3]

    uv_a = project(world, a)
    uv_b = project(world, b)
    uv_b[2, 1] += 40  # inconsistent correspondence
    points, indices = triangulate_correspondences(uv_a, uv_b, a, b, k, k)
    assert indices.tolist() == [0, 1]
    assert np.allclose(points, world[:2], atol=1e-5)
    b[0, 3] = .01
    assert len(triangulate_correspondences(uv_a, uv_b, a, b, k, k)[0]) == 0


def test_checkpoint_cursor_recovers_after_status_write_interruption(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    capture(store, project, session, "a")
    capture(store, project, session, "b")
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer)
    worker.tick(steps=1, maximum_new=1)
    # Simulate a crash after a complete checkpoint but before its JSON receipt.
    worker.checkpoint_path.write_text(json.dumps(["a", "b"]))
    resumed_trainer = FakeTrainer(["a", "b"])
    resumed = OnlineCoordinator(store, session["id"], seed, registration, resumed_trainer)
    assert resumed.status["accepted_ids"] == ["a", "b"]
    resumed.tick(steps=0)
    assert resumed_trainer.capture_ids == ["a", "b"]


def test_missing_seed_is_not_replaced_with_synthetic_geometry(tmp_path):
    store, _project, session, seed, registration = scene(tmp_path)
    seed.unlink()
    with pytest.raises(Exception, match="absent"):
        OnlineCoordinator(store, session["id"], seed, registration, FakeTrainer())


def test_loopback_current_model_is_one_verified_atomic_snapshot(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    capture(store, project, session, "a")
    worker = OnlineCoordinator(store, session["id"], seed, registration, FakeTrainer())
    worker.tick(steps=1)
    app = create_app(Config(tmp_path, "t" * 48), dashboard_enabled=True)
    local = TestClient(app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    model = local.get(f"/ui/current-model?project_id={project['id']}").json()["model"]
    assert model["source"] == "online" and model["accepted_views"] == 1
    assert model["training_steps"] == 1 and model["vertex_count"] == 3
    assert local.get(model["url"]).content == splat_bytes()
    assert worker.status["display_artifact"]["source_sha256"] == worker.status["artifact"]["sha256"]
    assert (worker.revisions_path / (worker.status["display_artifact"]["sha256"] + ".ply")).is_file()
    assert local.get(f"/ui?project_id={project['id']}").text.count("Current model") >= 1
    remote = TestClient(app, base_url="http://192.168.0.4:8765", client=("192.168.0.3", 50000))
    assert remote.get(model["url"]).status_code == 403
    other = store.create_project("Other")
    assert local.get(model["url"].replace(project["id"], other["id"])).status_code == 404
    worker.ply_path.write_bytes(splat_bytes() + b"tampered")
    assert local.get(f"/ui/current-model?project_id={project['id']}").json()["model"] is None
    # The immutable display revision remains loadable after the mutable model
    # changes; this is what keeps browser downloads from racing the worker.
    assert local.get(model["url"]).content == splat_bytes()


def test_browser_uses_full_revision_while_headset_copy_remains_separate(tmp_path):
    from bridge.online import stage_browser_revision
    store,project,session,seed,registration=scene(tmp_path)
    capture(store,project,session,'a')
    worker=OnlineCoordinator(store,session['id'],seed,registration,FakeTrainer())
    worker.tick(steps=1)
    original=worker.ply_path.read_bytes()
    stage_browser_revision(worker.ply_path,worker.root,project['id'],session['id'],worker.status['revision_number'])
    app=create_app(Config(tmp_path,'t'*48),dashboard_enabled=True)
    local=TestClient(app,base_url='http://127.0.0.1:8765',client=('127.0.0.1',50000))
    model=local.get(f"/ui/current-model?project_id={project['id']}").json()['model']
    assert model['source_sha256']==hashlib.sha256(original).hexdigest()
    assert local.get(model['url']).content==original
    assert worker.display_path.is_file()
    assert worker.ply_path.read_bytes()==original
    remote=TestClient(app,base_url='http://192.168.0.4:8765',client=('192.168.0.3',50000))
    assert remote.get(model['url']).status_code==403
    immutable=worker.root/'model-revisions'/(model['source_sha256']+'.ply')
    immutable.write_bytes(original+b'tampered')
    assert local.get(model['url']).status_code==404
    assert local.get(f"/ui/current-model?project_id={project['id']}").json()['model'] is None


def test_online_view_inherits_verified_seed_axis_and_camera(tmp_path, monkeypatch):
    store, project, session, seed, registration = scene(tmp_path)
    registration["evidence_job_id"] = str(uuid.uuid4())
    capture(store, project, session, "one")
    worker = OnlineCoordinator(store, session["id"], seed, registration, FakeTrainer())
    worker.tick(steps=1)
    z_seed = tmp_path / "z-seed.ply"
    z_seed.write_bytes(splat_bytes().replace(
        b"format binary_little_endian 1.0\n",
        b"format binary_little_endian 1.0\ncomment Vertical Axis: z\n", 1))
    evidence = {"session_id": session["id"], "registered_views": 3,
                "artifact": {"sha256": registration["seed_sha256"]}}
    monkeypatch.setattr("bridge.dashboard.verified_job",
                        lambda _root, _project, job_id:
                        (evidence, z_seed) if job_id == registration["evidence_job_id"] else None)
    monkeypatch.setattr("bridge.dashboard._person_view",
                        lambda _job, view, _folder: {**view, "eye": [1., 2., 3.],
                                                      "look_at": [1., 3., 3.],
                                                      "start_view": "registered_capture"})
    current = dashboard_current_model(tmp_path, project["id"])
    assert current["view"]["up_axis"] == "z"
    assert current["view"]["start_view"] == "registered_capture"
    assert current["view"]["look_at"] == [1., 3., 3.]


def measured_scene(tmp_path, *, omit_frame=False):
    """Authentic original receipts, frozen frame identities and measured poses."""
    store = Store(tmp_path)
    project = store.create_project("Measured room")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    sources = np.repeat(np.eye(4)[None], 8, axis=0)
    sources[:, :3, 3] = [[0., 0., 0.], [.2, 0., 0.], [0., .2, .03], [.2, .2, .05],
                         [.4, 0., .04], [0., .4, .08], [.4, .4, .02], [.2, .4, .06]]
    targets = sources.copy(); targets[:, 2, 3] *= -1
    job_id = str(uuid.uuid4())
    folder = tmp_path/"projects"/project["id"]/"reconstructions"/job_id
    images = folder/"processed"/"images"; images.mkdir(parents=True)
    config = folder/"training"/"config.yml"; config.parent.mkdir()
    config.write_text("Fixture only; no trainer invoked")
    config.with_name("dataparser_transforms.json").write_text(json.dumps(
        {"transform": np.eye(4)[:3].tolist(), "scale": 1.}))
    frames, inputs, mapping = [], [], {}
    for index, pose in enumerate(targets):
        cid = str(uuid.UUID(int=index+1))
        output = io.BytesIO()
        Image.new("RGB", (64, 48), (20+index*22, 30+index*17, 40+index*13)).save(output, "JPEG")
        pixels = output.getvalue()
        metadata = {"capture_id": cid, "project_id": project["id"], "session_id": session["id"],
            "tracking_origin_id": session["tracking_origin_id"], "image_sha256": hashlib.sha256(pixels).hexdigest(),
            "image_encoding": "jpeg", "capture_source": "quest_camera", "tracking_valid": True,
            "image_orientation": "upright", "calibration_revision": "camera-1", "camera_id": "left",
            "image_width": 64, "image_height": 48, "coordinate_convention": "unity_lh_xright_yup_zforward",
            "camera_to_session": pose.ravel().tolist(), "intrinsics": {"fx": 50., "fy": 50., "cx": 32., "cy": 24.}}
        store.save_capture(session["id"], metadata, pixels)
        frame_name = f"frame_{index+1:05d}.jpg"
        (images/frame_name).write_bytes(pixels)
        frames.append({"file_path": "images/"+frame_name, "transform_matrix": sources[index].tolist()})
        inputs.append({"capture_id": cid, "image_sha256": metadata["image_sha256"], "image_name": cid+".jpg"})
        mapping[cid] = frame_name
    (images.parent/"transforms.json").write_text(json.dumps({"frames": frames[1:] if omit_frame else frames}))
    _fixture_sparse_map(images.parent, sources, frames)
    seed = folder/"splat.ply"; seed.write_bytes(splat_bytes())
    job = {"job_id": job_id, "project_id": project["id"], "session_id": session["id"],
        "status": "ready_for_blender", "capture_sources": ["quest_camera"], "processed_dir": str(images.parent),
        "training_config": str(config), "input_captures": inputs, "frame_manifest": mapping,
        "artifact": {"path": str(seed), "sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
                     "size_bytes": seed.stat().st_size, "vertex_count": 3}}
    (folder/"job.json").write_text(json.dumps(job))
    return store, project, session, job, seed


def _fixture_sparse_map(processed, sources, frames):
    """Exact tiny indexed COLMAP map; never involves a real photo or device."""
    sparse = processed/"colmap"/"sparse"/"0"; sparse.mkdir(parents=True)
    rng = np.random.default_rng(27)
    world = rng.uniform([-.3, -.3, -4.], [.6, .6, -2.], (32, 3))
    descriptors = rng.integers(0, 128, (len(world), 128), dtype=np.uint8)
    with sqlite3.connect(processed/"colmap"/"database.db") as db, (sparse/"images.bin").open("wb") as stream:
        db.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY,name TEXT)")
        db.execute("CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB)")
        db.execute("CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB)")
        stream.write(struct.pack("<Q", len(sources)))
        for i, (pose, frame) in enumerate(zip(sources, frames), 1):
            raw = pose@np.diag([1., -1., -1., 1.]); rotation = raw[:3, :3].T
            translation = -rotation@raw[:3, 3]
            assert np.allclose(rotation, np.diag([1., -1., -1.]))
            xy_camera = (rotation@(world-raw[:3, 3]).T).T
            xy = xy_camera[:, :2]/xy_camera[:, 2:]*50.+[32., 24.]
            name = Path(frame["file_path"]).name
            stream.write(struct.pack("<idddddddi", i, 0., 1., 0., 0., *translation, 1))
            stream.write(name.encode()+b"\0"+struct.pack("<Q", len(world)))
            for index, (x, y) in enumerate(xy):
                stream.write(struct.pack("<ddq", x, y, index+1))
            db.execute("INSERT INTO images VALUES (?,?)", (i, name))
            db.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (i, len(world), 2, xy.astype('<f4').tobytes()))
            db.execute("INSERT INTO descriptors VALUES (?,?,?,?)", (i, len(world), 128, descriptors.tobytes()))
    with (sparse/"points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(world)))
        for index, xyz in enumerate(world):
            stream.write(struct.pack("<QdddBBBdQ", index+1, *xyz, 100, 100, 100, .1, len(sources)))
            for i in range(1, len(sources)+1):
                stream.write(struct.pack("<ii", i, index))


@pytest.mark.parametrize("fault", ["missing", "truncated", "frame", "camera", "feature_identity", "track_identity"])
def test_sparse_localizer_rejects_unbound_or_corrupt_seed_map(tmp_path, fault):
    store, _project, session, job, _seed = measured_scene(tmp_path)
    _, reg = registration_from_seed_job(store, session["id"], job["job_id"])
    views = seed_camera_views(store, reg)
    processed = Path(job["processed_dir"])
    images = processed/"colmap"/"sparse"/"0"/"images.bin"
    if fault == "missing":
        images.unlink()
    elif fault == "truncated":
        images.write_bytes(images.read_bytes()[:-1])
    elif fault == "frame":
        parser = Path(job["training_config"]).with_name("dataparser_transforms.json")
        parser.write_text(parser.read_text()+" ")
    elif fault == "camera":
        data = bytearray(images.read_bytes())
        struct.pack_into("<d", data, 8+4+4*8, 20.)  # Wrong raw COLMAP translation.
        images.write_bytes(data)
    elif fault == "feature_identity":
        with sqlite3.connect(processed/"colmap"/"database.db") as db:
            row = db.execute("SELECT data FROM keypoints WHERE image_id=1").fetchone()[0]
            keys = np.frombuffer(row, dtype='<f4').copy(); keys[0] += 10.
            db.execute("UPDATE keypoints SET data=? WHERE image_id=1", (keys.tobytes(),))
    else:
        data = bytearray(images.read_bytes())
        name_end = data.index(0, 8+64)+1
        struct.pack_into("<q", data, name_end+8+16, 9999)
        images.write_bytes(data)
    with pytest.raises(OnlineError, match="landmark|Sparse"):
        _colmap_localization_source(store, reg, views)


@pytest.mark.parametrize('wrong_receipt',[False,True])
def test_incremental_sparse_landmarks_keep_the_original_trained_world_frame(tmp_path,wrong_receipt):
    from bridge.reconstruct import _colmap_camera_to_saved
    store,_project,session,job,_seed=measured_scene(tmp_path)
    processed=Path(job['processed_dir']);transforms=processed/'transforms.json'
    data=json.loads(transforms.read_text())
    matrix=np.array([[0.,-1.,0.,-.6],[1.,0.,0.,.8],[0.,0.,1.,1.],[0.,0.,0.,1.]])
    for frame in data['frames']:
        pose=matrix@np.array(frame['transform_matrix']);pose[:3,3]*=.37
        frame['transform_matrix']=pose.tolist()
    data['applied_transform']=np.eye(4)[:3].tolist()
    transforms.write_text(json.dumps(data))
    axis=_colmap_camera_to_saved(np.array([1.,0.,0.,0.]),np.zeros(3))@np.diag([1.,-1.,-1.,1.])
    normalized=matrix@np.linalg.inv(axis)
    if wrong_receipt:normalized[0,3]+=.1
    job['normalization']={'root_job_id':job['job_id'],'colmap_saved_to_training':normalized[:3].tolist(),'scale':.37}
    (processed.parent/'job.json').write_text(json.dumps(job))
    _,reg=registration_from_seed_job(store,session['id'],job['job_id'])
    views=seed_camera_views(store,reg)
    if wrong_receipt:
        with pytest.raises(OnlineError,match='exact trained seed frame'):
            _colmap_localization_source(store,reg,views)
    else:
        source=_colmap_localization_source(store,reg,views)
        raw=np.array(struct.unpack_from('<ddd',source['paths']['points'].read_bytes(),16))
        assert np.allclose(source['xyz'][1],(matrix@np.r_[raw,1])[:3]*.37)


def test_sparse_landmark_points_follow_exact_seed_normalization_and_resume_binding(tmp_path):
    store, _project, session, job, seed = measured_scene(tmp_path)
    parser = Path(job["training_config"]).with_name("dataparser_transforms.json")
    matrix = np.asarray([[0., -1., 0., -.6], [1., 0., 0., .8], [0., 0., 1., 1.], [0., 0., 0., 1.]])
    parser.write_text(json.dumps({"transform": matrix[:3].tolist(), "scale": .37}))
    _, reg = registration_from_seed_job(store, session["id"], job["job_id"])
    views = seed_camera_views(store, reg)
    source = _colmap_localization_source(store, reg, views)
    raw = np.asarray(struct.unpack_from("<ddd", source["paths"]["points"].read_bytes(), 16))
    assert np.allclose(source["xyz"][1], (matrix@np.r_[raw, 1])[:3]*.37)
    assert len(source["xyz"]) == 32 and len(source["landmark_ids"]) == 8*32
    worker = OnlineCoordinator(store, session["id"], seed, reg, FakeTrainer())
    protected = worker.checkpoint_path.read_bytes(), worker.ply_path.read_bytes()
    with sqlite3.connect(Path(job["processed_dir"])/"colmap"/"database.db") as db:
        # A descriptor change retains valid camera/track geometry but changes its frozen evidence.
        db.execute("UPDATE descriptors SET data=? WHERE image_id=1", (bytes(32*128),))
    with pytest.raises(OnlineError, match="Saved localization source"):
        OnlineCoordinator(store, session["id"], seed, reg, FakeTrainer())
    assert (worker.checkpoint_path.read_bytes(), worker.ply_path.read_bytes()) == protected


def test_failed_first_sixty_seed_can_use_newer_time_distributed_photos(tmp_path, monkeypatch):
    store, project, session, seed, _reg = scene(tmp_path)
    ids = [str(uuid.UUID(int=i+1)) for i in range(65)]
    for cid in ids:
        capture(store, project, session, cid)
    old = str(uuid.uuid4()); new = str(uuid.uuid4())
    folder = tmp_path/"projects"/project["id"]/"reconstructions"/old
    folder.mkdir(parents=True)
    (folder/"job.json").write_text(json.dumps({"job_id": old, "session_id": session["id"],
                                               "status": "ready_for_blender", "created_utc": "old"}))
    measured, built = [], []
    def register(_store, _sid, jid):
        measured.append(jid)
        if jid == old:
            raise OnlineError("Seed lacks strict independent per-photo registration proof")
        return seed, {}
    def build(_store, _sid, **kwargs):
        built.append(kwargs)
        return {"job_id": new}
    monkeypatch.setattr("bridge.online.registration_from_seed_job", register)
    monkeypatch.setattr("bridge.reconstruct.build_session", build)
    assert bootstrap_seed_job(store, session["id"]) == new
    assert measured == [old, new] and len(built) == 1
    assert built[0]["fresh_baseline"] is True
    assert len(set(built[0]["capture_ids"])) == 60
    assert built[0]["capture_ids"][0] == ids[0] and built[0]["capture_ids"][-1] == ids[-1]
    assert set(built[0]["capture_ids"]) & set(ids[60:])


def test_rejected_unpublished_photo_retries_once_per_method_without_duplicate_counts(tmp_path):
    store, project, session, seed, reg = scene(tmp_path)
    cid = str(uuid.UUID(int=99))
    capture(store, project, session, cid)
    class RetryTrainer(FakeTrainer):
        review_method_revision = "patch-v2"
        review_attempts = {}
        retries = []
        @property
        def rejected_capture_ids(self):
            return self.ids
        @property
        def published_capture_ids(self):
            return []
        def retry_rejected_image(self, cid, path, pose, k):
            assert cid in self.ids and path.is_file()
            self.retries.append(cid)
            self.review_attempts[cid] = self.review_method_revision
    trainer = RetryTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, reg, trainer)
    worker.tick(steps=0)  # First ingress remains a unique processed receipt.
    worker.tick(steps=0)
    worker.tick(steps=0)
    assert trainer.retries == [cid] and trainer.capture_ids == [cid]
    assert worker.status["accepted_ids"] == [cid]
    assert worker.status["processed_count"] == 1 and worker.status["visible_count"] == 0
    trainer.review_method_revision = "patch-v3"
    worker.tick(steps=0); worker.tick(steps=0)
    assert trainer.retries == [cid, cid] and trainer.capture_ids == [cid]
    trainer.review_method_revision = "patch-v4"
    trainer.rejected_capture_reviews = {cid: {"review_method_revision": "patch-v4"}}
    worker.tick(steps=0)
    assert trainer.retries == [cid, cid]  # Current-method rejection is not a retry candidate.


def test_sparse_pnp_uses_metric_seed_points_and_rejects_wrong_tracked_pose(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    store, _project, session, job, _seed = measured_scene(tmp_path)
    parser = Path(job["training_config"]).with_name("dataparser_transforms.json")
    parser.write_text(json.dumps({"transform": np.eye(4)[:3].tolist(), "scale": .37}))
    _, reg = registration_from_seed_job(store, session["id"], job["job_id"])
    views = seed_camera_views(store, reg)
    verifier = ColmapLandmarkVerifier(store, reg, views, .37)
    view = dict(views[0])
    ids = sorted(verifier.source["xyz"])
    points = np.asarray([verifier.source["xyz"][pid]/.37 for pid in ids])
    raw_pose = view["pose"].copy(); raw_pose[:3, 3] /= .37
    camera = (raw_pose[:3, :3].T@(points-raw_pose[:3, 3]).T).T
    pixels = (view["intrinsics"]@camera.T).T
    pixels = pixels[:, :2]/pixels[:, 2:]
    keys = [cv2.KeyPoint(float(x), float(y), 1.) for x, y in pixels]
    class Detector:
        def detectAndCompute(self, _image, _mask):
            return keys, np.ones((len(keys), 128), np.float32)
    class Matcher:
        def knnMatch(self, _descriptors, k):
            result = []
            for query, pid in enumerate(ids):
                first = int(np.flatnonzero(verifier.source["landmark_ids"] == pid)[0])
                other = int(np.flatnonzero(verifier.source["landmark_ids"] != pid)[0])
                result.append([cv2.DMatch(query, first, 1.), cv2.DMatch(query, other, 10.)])
            return result
    monkeypatch.setattr(cv2, "SIFT_create", lambda **_kwargs: Detector())
    verifier.matcher = Matcher()
    proof = verifier.verify(view)
    assert proof["inliers"] == 32 and proof["reprojection_max_px"] < 1e-4
    assert np.allclose(proof["localized_pose"], view["pose"], atol=1e-5)
    wrong = {**view, "pose": view["pose"].copy()}
    wrong["pose"][0, 3] += .30*.37
    with pytest.raises(OnlineError, match="tracking disagrees"):
        verifier.verify(wrong)
    path = verifier.source["paths"]["images"]
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(OnlineError, match="source changed"):
        verifier.verify(view)


def test_measured_seed_preserves_exact_registered_ids_and_publishes_baseline(tmp_path):
    store, _project, session, job, seed = measured_scene(tmp_path, omit_frame=True)
    original = seed.read_bytes()
    path, registration = registration_from_seed_job(store, session["id"], job["job_id"])
    assert path == seed
    assert registration["physical_acceptance"] == "pending"
    assert registration["pose_fit"]["matched_views"] == 7
    assert registration["seed_capture_ids"] == [item["capture_id"] for item in job["input_captures"][1:]]
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer)
    assert len(trainer.baseline) == 7 and trainer.capture_ids == []
    assert worker.status["visible_count"] == worker.status["processed_count"] == 7
    assert worker.ply_path.read_bytes() == seed.read_bytes() == original
    sha = worker.status["display_artifact"]["sha256"]
    receipt = json.loads((worker.revisions_path/(sha+".json")).read_text())
    assert receipt == {"project_id": session["project_id"], "session_id": session["id"],
        "sha256": sha, "size_bytes": len(original), "vertex_count": 3,
        "source_sha256": registration["seed_sha256"], "revision_number": 1}
    assert bootstrap_seed_job(store, session["id"]) == job["job_id"]
    assert seed.read_bytes() == original


def test_baseline_uses_exact_seed_cameras_instead_of_tracking_approximation(tmp_path):
    store, _project, session, job, seed = measured_scene(tmp_path)
    first = job["input_captures"][0]["capture_id"]
    metadata = next(row["metadata"] for row in store.captures(session["id"]) if row["capture_id"] == first)
    metadata["camera_to_session"][3] += .015
    with store.connection() as db:
        db.execute("UPDATE captures SET metadata_json=? WHERE capture_id=?", (json.dumps(metadata), first))
    path = Path(job["processed_dir"])/"transforms.json"
    transforms = json.loads(path.read_text()); transforms.update(fl_x=50.25, fl_y=50.25, cx=32., cy=24.)
    path.write_text(json.dumps(transforms))
    _, registration = registration_from_seed_job(store, session["id"], job["job_id"])
    views = seed_camera_views(store, registration)
    assert np.allclose(views[0]["pose"], np.diag([1., -1., -1., 1.]))
    assert views[0]["intrinsics"][0, 0] == 50.25
    tracked, _ = validated_camera(metadata, np.asarray(registration["model_from_session"]), "camera-1")
    assert not np.allclose(views[0]["pose"], tracked)
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer)
    assert np.array_equal(trainer.baseline[0]["pose"], views[0]["pose"])
    assert worker.photo_verifier.views[0]["camera_source"] == "verified_seed_frame"
    assert trainer.capture_ids == []


def test_corrected_localizer_revision_retries_quarantine_without_duplicate_admission(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    capture(store, project, session, "new")
    class Verifier:
        method_revision = "old"
        calls = 0
        def verify(self, view):
            self.calls += 1
            if self.method_revision == "old":
                raise OnlineError("Photo lacks independently triangulated baseline correspondences")
            return {"method": "measured_rgb", "method_revision": self.method_revision}
    verifier = Verifier(); trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer, photo_verifier=verifier)
    worker.tick(steps=0); worker.tick(steps=0)
    assert verifier.calls == 1 and trainer.capture_ids == []
    verifier.method_revision = "corrected"
    worker.tick(steps=0); worker.tick(steps=0)
    assert verifier.calls == 2 and trainer.capture_ids == ["new"]
    assert worker.status["quarantined"] == {}


def test_training_receives_measured_refined_pose_after_strict_tracking_comparison(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    capture(store, project, session, "new")
    class Verifier:
        def verify(self, view):
            pose = view["pose"].copy(); pose[0, 3] += .04
            return {"method": "measured_rgb", "localized_pose": pose.tolist(),
                    "camera_intrinsics": view["intrinsics"].tolist()}
    class CapturingTrainer(FakeTrainer):
        def add_image(self, cid, path, pose, k):
            self.last_pose = pose.copy()
            return super().add_image(cid, path, pose, k)
    trainer = CapturingTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer, photo_verifier=Verifier())
    worker.tick(steps=0)
    assert trainer.last_pose[0, 3] == pytest.approx(.04)
    assert worker.status["photo_registration"]["new"]["localized_pose"][0][3] == pytest.approx(.04)
    assert store.captures(session["id"])[0]["metadata"]["camera_to_session"][3] == 0.


def test_measured_seed_lens_calibration_is_forwarded_without_touching_originals(tmp_path):
    store, _project, session, job, seed = measured_scene(tmp_path)
    original = seed.read_bytes()
    path = Path(job["processed_dir"])/"transforms.json"
    source = json.loads(path.read_text()); source.update(w=64, h=48, camera_model="OPENCV",
                                                       k1=.003, k2=-.002, p1=.0004, p2=.0006)
    path.write_text(json.dumps(source))
    _, registration = registration_from_seed_job(store, session["id"], job["job_id"])
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer)
    assert trainer.distortion == [.003, -.002, .0004, .0006, 0.]
    assert worker.photo_verifier.views[0]["distortion_coefficients"] == trainer.distortion
    assert seed.read_bytes() == original


@pytest.mark.parametrize("fault", ["dimensions", "distortion_model"])
def test_seed_calibration_rejects_unbound_dimensions_or_unsupported_lens_model(tmp_path, fault):
    store, _project, session, job, _seed = measured_scene(tmp_path)
    path = Path(job["processed_dir"])/"transforms.json"
    source = json.loads(path.read_text())
    source.update({"w": 32, "h": 24} if fault == "dimensions" else {"camera_model": "OPENCV_FISHEYE", "k1": .003})
    path.write_text(json.dumps(source))
    _, registration = registration_from_seed_job(store, session["id"], job["job_id"])
    with pytest.raises(OnlineError, match="dimensions|lens-distortion"):
        seed_camera_views(store, registration)


@pytest.mark.parametrize("fault", ["image_hash", "tracking_origin", "camera", "fit", "source_changed"])
def test_measured_seed_rejects_corrupt_or_changed_source(tmp_path, fault):
    store, project, session, job, seed = measured_scene(tmp_path)
    _, registration = registration_from_seed_job(store, session["id"], job["job_id"])
    if fault in {"image_hash", "tracking_origin", "camera"}:
        cid = job["input_captures"][0]["capture_id"]
        metadata = next(item["metadata"] for item in store.captures(session["id"]) if item["capture_id"] == cid)
        key = {"image_hash": "image_sha256", "tracking_origin": "tracking_origin_id", "camera": "camera_id"}[fault]
        metadata[key] = "wrong"
        with store.connection() as db:
            db.execute("UPDATE captures SET metadata_json=? WHERE capture_id=?", (json.dumps(metadata), cid))
    elif fault == "fit":
        path = Path(job["processed_dir"])/"transforms.json"
        data = json.loads(path.read_text()); data["frames"][0]["transform_matrix"][0][3] += .4
        path.write_text(json.dumps(data))
    else:
        Path(job["training_config"]).write_text("Different frame-owner receipt")
    with pytest.raises(Exception):
        OnlineCoordinator(store, session["id"], seed, registration, FakeTrainer())
    assert not (tmp_path/"projects"/project["id"]/"online"/session["id"]/"current.ply").exists()


def test_scheduled_worker_reads_queue_database_and_stops_before_commit(tmp_path):
    from bridge.jobs import set_splat_updates
    store, project, session, seed, registration = scene(tmp_path)
    set_splat_updates(tmp_path, session["id"], True, native_orientation_reviewed=True)
    class PausingTrainer(FakeTrainer):
        def train_steps(self, count):
            set_splat_updates(tmp_path, session["id"], False)
            return super().train_steps(count)
    worker = OnlineCoordinator(store, session["id"], seed, registration, PausingTrainer(), scheduled=True)
    baseline = {p.name: p.read_bytes() for p in (worker.ply_path, worker.checkpoint_path)}
    capture(store, project, session, "new")
    with pytest.raises(OnlineError, match="disabled"):
        worker.tick(steps=32)
    assert {p.name: p.read_bytes() for p in (worker.ply_path, worker.checkpoint_path)} == baseline
    assert json.loads(worker.status_path.read_text())["accepted_ids"] == []


def test_scheduled_constructor_cannot_publish_when_priming_changes_control(tmp_path):
    from bridge.jobs import set_splat_updates
    store, _project, session, job, seed = measured_scene(tmp_path)
    _, registration = registration_from_seed_job(store, session["id"], job["job_id"])
    set_splat_updates(tmp_path, session["id"], True, native_orientation_reviewed=True)
    class PausingTrainer(FakeTrainer):
        def prime_baseline_views(self, views):
            set_splat_updates(tmp_path, session["id"], False)
            return super().prime_baseline_views(views)
    with pytest.raises(OnlineError, match="disabled"):
        OnlineCoordinator(store, session["id"], seed, registration, PausingTrainer(), scheduled=True)
    assert not (tmp_path/"projects"/session["project_id"]/"online"/session["id"]/"current.ply").exists()


def test_bad_photo_quarantined_once_does_not_block_next_good_photo(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    capture(store, project, session, "bad")
    capture(store, project, session, "good")
    class Verifier:
        checked = []
        def verify(self, view):
            self.checked.append(view["capture_id"])
            if view["capture_id"] == "bad":
                raise OnlineError("Photo metadata disagrees with independently localized RGB pose")
            return {"method": "independent_rgb", "position_error_m": .02, "rotation_error_deg": 1.}
    verifier = Verifier(); trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session["id"], seed, registration, trainer, photo_verifier=verifier)
    status = worker.tick(steps=1)
    assert status["accepted_ids"] == ["good"] and status["quarantined_count"] == 1
    assert "disagrees" in status["quarantined"]["bad"]["reason"]
    worker.tick(steps=1)
    assert verifier.checked == ["bad", "good"] and trainer.capture_ids == ["good"]


def test_pending_refinement_keeps_visible_count_and_revision_until_approved(tmp_path):
    store, project, session, seed, registration = scene(tmp_path)
    class PendingTrainer(FakeTrainer):
        published = []
        @property
        def published_capture_ids(self): return list(self.published)
        @property
        def has_pending_work(self): return bool(set(self.ids)-set(self.published))
        @property
        def pending_refinement(self):
            return {"capture_ids": sorted(set(self.ids)-set(self.published))}
        def save(self, checkpoint, ply):
            if self.steps >= 64: self.published = list(self.ids)
            checkpoint.write_text(json.dumps(self.ids))
            data = splat_bytes()
            if self.published: data = data[:-4]+struct.pack("<f", .25)
            ply.write_bytes(data)
            return {"published_capture_ids": self.published}
    trainer = PendingTrainer(); worker = OnlineCoordinator(store, session["id"], seed, registration, trainer)
    baseline = worker.ply_path.read_bytes(); capture(store, project, session, "first")
    status = worker.tick(steps=32)
    assert status["processed_count"] == 1 and status["visible_count"] == 0 and status["pending_count"] == 1
    assert status["revision_number"] == 1 and worker.ply_path.read_bytes() == baseline
    capture(store, project, session, "second")
    status = worker.tick(steps=32)
    assert trainer.capture_ids == ["first"]  # Pending quality work has priority over more ingress.
    assert status["visible_count"] == 1 and status["pending_count"] == 0 and status["revision_number"] == 2
    assert worker.ply_path.read_bytes() != baseline


def test_gpu_runtime_failure_records_actionable_status_without_publishing(tmp_path, monkeypatch):
    import sys
    import types
    from bridge.online import main
    store, _project, session, job, _seed = measured_scene(tmp_path)
    module = types.ModuleType("bridge.online_torch")
    def unavailable(*_args): raise RuntimeError("CUDA allocation unavailable")
    module.GsplatTrainer = unavailable
    monkeypatch.setitem(sys.modules, "bridge.online_torch", module)
    monkeypatch.setattr(sys, "argv", ["online", "--data-dir", str(tmp_path), "--session-id", session["id"],
                                    "--seed-job", job["job_id"], "--run", "--once"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    root = tmp_path/"projects"/session["project_id"]/"online"/session["id"]
    status = json.loads((root/"status.json").read_text())
    assert status["status"] == "blocked" and status["last_error"] == "CUDA allocation unavailable"
    assert not (root/"current.ply").exists()


def test_pause_during_display_copy_keeps_last_good_snapshot_and_checkpoint(tmp_path, monkeypatch):
    import shutil
    from bridge.jobs import set_splat_updates
    store, project, session, seed, registration = scene(tmp_path)
    set_splat_updates(tmp_path, session["id"], True, native_orientation_reviewed=True)
    class ChangedTrainer(FakeTrainer):
        def save(self, checkpoint, ply):
            result = super().save(checkpoint, ply)
            if self.ids: ply.write_bytes(splat_bytes()[:-4]+struct.pack("<f", .25))
            return result
    worker = OnlineCoordinator(store, session["id"], seed, registration, ChangedTrainer(), scheduled=True)
    before = {p.name: p.read_bytes() for p in (worker.ply_path, worker.checkpoint_path, worker.status_path)}
    copy = shutil.copyfile
    def pausing_copy(source, destination, **kwargs):
        result = copy(source, destination, **kwargs)
        if Path(destination).name == ".display.tmp.ply":
            set_splat_updates(tmp_path, session["id"], False)
        return result
    monkeypatch.setattr("bridge.online.shutil.copyfile", pausing_copy)
    capture(store, project, session, "next")
    with pytest.raises(OnlineError, match="disabled"): worker.tick(steps=32)
    assert {p.name: p.read_bytes() for p in (worker.ply_path, worker.checkpoint_path, worker.status_path)} == before


def test_large_seed_converter_receives_supported_ply_suffix_and_preserves_source(tmp_path, monkeypatch):
    from bridge.reconstruct import validate_gaussian_ply
    store, _project, session, seed, registration = scene(tmp_path)
    original = seed.read_bytes()
    class LargeTrainer(FakeTrainer):
        def save(self, checkpoint, ply):
            checkpoint.write_text(json.dumps(self.ids))
            ply.write_bytes(splat_bytes(240001))
            return {}
    calls = []
    def converter(source, destination, max_vertices):
        assert source.suffix == destination.suffix == ".ply"
        assert validate_gaussian_ply(source)[0] == 240001
        assert max_vertices == 240000
        calls.append(source.name)
        destination.write_bytes(splat_bytes())
        count, sha, size = validate_gaussian_ply(destination)
        return {"sha256": sha, "size_bytes": size, "vertex_count": count}
    monkeypatch.setattr("bridge.reconstruct.make_quest_preview", converter)
    worker = OnlineCoordinator(store, session["id"], seed, registration, LargeTrainer())
    assert calls == [".current.tmp.ply"]
    assert worker.status["artifact"]["vertex_count"] == 240001
    assert worker.status["display_artifact"]["vertex_count"] == 3
    assert seed.read_bytes() == original
    assert not (worker.root/".current.tmp.ply").exists()


def test_successful_retry_clears_stale_worker_error(tmp_path):
    store, _project, session, seed, registration = scene(tmp_path)
    worker = OnlineCoordinator(store, session["id"], seed, registration, FakeTrainer())
    worker.status.update(last_error="old converter failure", error="old converter failure", status="blocked")
    worker.status_path.write_text(json.dumps(worker.status))
    status = worker.tick(steps=0)
    assert status["status"] == "waiting_for_new_views"
    assert "last_error" not in status and "error" not in json.loads(worker.status_path.read_text())


def test_independent_photo_localization_measures_pose_instead_of_trusting_metadata(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    random = np.random.default_rng(3)
    cloud = np.column_stack((random.uniform(-.4, .7, 40), random.uniform(-.3, .4, 40),
                             random.uniform(2., 4., 40)))
    descriptors = random.integers(0, 256, (40, 32), dtype=np.uint8)
    k = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    a, b, actual = np.eye(4), np.eye(4), np.eye(4)
    b[0, 3] = .5; actual[:3, 3] = [.2, .05, 0.]
    views = [{"capture_id": name, "image_path": Path(name), "pose": pose, "intrinsics": k}
             for name, pose in (("a", a), ("b", b))]
    verifier = PhotoRegistrationVerifier(views, 1.)
    def features(view):
        pose = actual if view["capture_id"] == "new" else view["pose"]
        camera = (np.linalg.inv(pose) @ np.column_stack((cloud, np.ones(len(cloud)))).T).T[:, :3]
        projected = (k @ camera.T).T; uv = projected[:, :2]/projected[:, 2:3]
        return [cv2.KeyPoint(float(x), float(y), 1.) for x, y in uv], descriptors, k
    monkeypatch.setattr(verifier, "_features", features)
    view = {"capture_id": "new", "image_path": Path("new"), "pose": actual, "intrinsics": k}
    result = verifier.verify(view)
    assert result["position_error_m"] < 1e-4 and result["rotation_error_deg"] < .01
    wrong = actual.copy(); wrong[0, 3] += .3
    with pytest.raises(OnlineError, match="disagrees"):
        verifier.verify({**view, "pose": wrong})


def _scheduled_fixture(tmp_path):
    from bridge.jobs import set_splat_updates
    store, project, session, seed, registration = scene(tmp_path)
    set_splat_updates(tmp_path, session['id'], True, native_orientation_reviewed=True)
    return store, project, session, seed, registration


def _fast_worker_clock(monkeypatch, on_sleep=None):
    clock = [0.]
    monkeypatch.setattr('bridge.online.time.monotonic', lambda: clock[0])
    def advance(seconds):
        clock[0] += seconds
        assert clock[0] < 60, 'Scheduled fixture failed to relinquish the GPU'
        if on_sleep:
            on_sleep(clock[0])
    monkeypatch.setattr('bridge.online.time.sleep', advance)
    return clock


def test_scheduled_idle_yield_retains_checkpoint_and_resumes_same_cursor(tmp_path, monkeypatch):
    from bridge.jobs import _depth_inventory
    store, project, session, seed, registration = _scheduled_fixture(tmp_path)
    capture(store, project, session, 'first')
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session['id'], seed, registration, trainer, scheduled=True)
    _fast_worker_clock(monkeypatch)
    _run_online_worker(worker, steps=32, idle_seconds=4, run_id='this-run')
    saved = json.loads(worker.status_path.read_text())
    assert saved['worker_state'] == 'idle' and saved['status'] == 'waiting_for_new_views'
    assert saved['idle_yield']['input_revision'] == _depth_inventory(store, session['id'])[2]
    assert saved['idle_yield']['control_revision'] == worker._control()
    assert saved['idle_yield']['run_id'] == 'this-run'
    assert trainer.steps == 32 and saved['accepted_ids'] == ['first']
    protected = worker.checkpoint_path.read_bytes(), worker.ply_path.read_bytes(), seed.read_bytes()
    resumed = OnlineCoordinator(store, session['id'], seed, registration,
                                FakeTrainer(json.loads(worker.checkpoint_path.read_text())), scheduled=True)
    assert (worker.checkpoint_path.read_bytes(), worker.ply_path.read_bytes(), seed.read_bytes()) == protected
    capture(store, project, session, 'second')
    resumed.tick(steps=32)
    assert resumed.status['accepted_ids'] == ['first', 'second']
    assert 'idle_yield' not in resumed.status and 'worker_state' not in resumed.status
    assert seed.read_bytes() == protected[2]


def test_scheduled_idle_clock_does_not_interrupt_pending_refinement(tmp_path, monkeypatch):
    store, project, session, seed, registration = _scheduled_fixture(tmp_path)
    class PendingTrainer(FakeTrainer):
        @property
        def has_pending_work(self): return bool(self.ids) and self.steps < 96
        @property
        def pending_refinement(self): return {'capture_ids': list(self.ids) if self.has_pending_work else []}
        @property
        def published_capture_ids(self): return [] if self.has_pending_work else list(self.ids)
    trainer = PendingTrainer()
    worker = OnlineCoordinator(store, session['id'], seed, registration, trainer, scheduled=True)
    capture(store, project, session, 'cohort')
    clock = _fast_worker_clock(monkeypatch)
    _run_online_worker(worker, steps=32, idle_seconds=4)
    assert trainer.steps == 96 and clock[0] == 8
    assert worker.status['pending_count'] == 0 and worker.status['visible_count'] == 1
    assert json.loads(worker.checkpoint_path.read_text()) == ['cohort']


def test_capture_racing_idle_yield_is_not_lost(tmp_path, monkeypatch):
    store, project, session, seed, registration = _scheduled_fixture(tmp_path)
    worker = OnlineCoordinator(store, session['id'], seed, registration, FakeTrainer(), scheduled=True)
    original_yield = worker.yield_idle
    attempts = []
    def racing_yield(revision, control, **kwargs):
        if not attempts:
            capture(store, project, session, 'racing-photo')
        result = original_yield(revision, control, **kwargs)
        attempts.append(result)
        return result
    monkeypatch.setattr(worker, 'yield_idle', racing_yield)
    _fast_worker_clock(monkeypatch)
    _run_online_worker(worker, steps=32, idle_seconds=4)
    assert attempts == [False, True]
    assert worker.status['accepted_ids'] == ['racing-photo']


def test_scheduler_control_change_ends_existing_run_without_retraining(tmp_path, monkeypatch):
    from bridge.jobs import set_splat_updates
    store, _project, session, seed, registration = _scheduled_fixture(tmp_path)
    trainer = FakeTrainer()
    worker = OnlineCoordinator(store, session['id'], seed, registration, trainer, scheduled=True)
    protected = worker.checkpoint_path.read_bytes(), worker.ply_path.read_bytes(), seed.read_bytes()
    def reenable(_seconds):
        set_splat_updates(tmp_path, session['id'], True)
    _fast_worker_clock(monkeypatch, reenable)
    with pytest.raises(OnlineError, match='controls changed after'):
        _run_online_worker(worker, idle_seconds=4)
    assert trainer.steps == 0
    assert (worker.checkpoint_path.read_bytes(), worker.ply_path.read_bytes(), seed.read_bytes()) == protected
    assert 'idle_yield' not in json.loads(worker.status_path.read_text())


def test_pending_candidate_checkpoint_survives_pause_and_reentry(tmp_path, monkeypatch):
    from bridge.jobs import set_splat_updates
    store, project, session, seed, registration = _scheduled_fixture(tmp_path)
    class StatefulPendingTrainer(FakeTrainer):
        def __init__(self, saved=None):
            super().__init__((saved or {}).get('ids', []))
            self.steps = (saved or {}).get('steps', 0)
        @property
        def has_pending_work(self): return bool(self.ids) and self.steps < 96
        @property
        def pending_refinement(self): return {'capture_ids': list(self.ids) if self.has_pending_work else []}
        @property
        def published_capture_ids(self): return [] if self.has_pending_work else list(self.ids)
        def save(self, checkpoint, ply):
            checkpoint.write_text(json.dumps({'ids':self.ids, 'steps':self.steps}))
            data = splat_bytes()
            if self.ids and not self.has_pending_work: data = data[:-4]+struct.pack('<f', .25)
            ply.write_bytes(data)
            return {}
    trainer = StatefulPendingTrainer()
    worker = OnlineCoordinator(store, session['id'], seed, registration, trainer, scheduled=True)
    original = seed.read_bytes(), worker.ply_path.read_bytes()
    capture(store, project, session, 'pending')
    _fast_worker_clock(monkeypatch, lambda _seconds: set_splat_updates(tmp_path, session['id'], False))
    with pytest.raises(OnlineError, match='disabled'):
        _run_online_worker(worker, idle_seconds=4)
    checkpoint = json.loads(worker.checkpoint_path.read_text())
    assert checkpoint == {'ids':['pending'], 'steps':32}
    assert worker.status['pending_count'] == 1 and worker.status['visible_count'] == 0
    assert (seed.read_bytes(), worker.ply_path.read_bytes()) == original
    set_splat_updates(tmp_path, session['id'], True)
    resumed_trainer = StatefulPendingTrainer(checkpoint)
    resumed = OnlineCoordinator(store, session['id'], seed, registration, resumed_trainer, scheduled=True)
    _fast_worker_clock(monkeypatch)
    _run_online_worker(resumed, idle_seconds=4)
    assert resumed_trainer.steps == 96 and resumed.status['accepted_ids'] == ['pending']
    assert resumed.status['pending_count'] == 0 and resumed.status['visible_count'] == 1
    assert seed.read_bytes() == original[0] and worker.ply_path.read_bytes() != original[1]


class _ChangedPublicationTrainer(FakeTrainer):
    def save(self, checkpoint, ply):
        checkpoint.write_text(json.dumps(self.ids))
        data = splat_bytes()
        if self.ids: data = data[:-4]+struct.pack('<f', .25)
        ply.write_bytes(data)
        return {}


def _interrupted_publication(tmp_path, monkeypatch, phase='status'):
    import bridge.online as online
    store, project, session, seed, registration = scene(tmp_path)
    trainer = _ChangedPublicationTrainer()
    worker = OnlineCoordinator(store, session['id'], seed, registration, trainer)
    original = seed.read_bytes()
    capture(store, project, session, str(uuid.uuid4()))
    replace, atomic = online._replace_with_retry, online._atomic_json
    def fail_replace(source, destination, **kwargs):
        name = {'model':'current.ply', 'checkpoint':'current.ckpt', 'display':'display.ply'}.get(phase)
        if Path(destination).name == name:
            raise OSError('injected publication interruption')
        return replace(source, destination, **kwargs)
    def fail_status(path, value):
        if phase == 'status' and path == worker.status_path and value.get('revision_number') == 2:
            raise OSError('injected publication interruption')
        return atomic(path, value)
    with monkeypatch.context() as scoped:
        scoped.setattr(online, '_replace_with_retry', fail_replace)
        scoped.setattr(online, '_atomic_json', fail_status)
        with pytest.raises(OSError, match='injected'):
            worker.tick(steps=32)
    assert (worker.root/'pending-publication.json').is_file()
    assert seed.read_bytes() == original
    return store, project, session, seed, registration, worker


@pytest.mark.parametrize('phase', ['model', 'checkpoint', 'display', 'status'])
def test_interrupted_publication_recovers_exact_pair_manifest_and_counts(tmp_path, monkeypatch, phase):
    from bridge.reconstruct import validate_gaussian_ply
    store, project, session, seed, registration, worker = _interrupted_publication(tmp_path, monkeypatch, phase)
    original_items = store.export_session(session['id'])['captures']
    cid = original_items[0]['receipt']['capture_id']
    originals = {item['receipt']['capture_id']:Path(item['image_path']).read_bytes() for item in original_items}
    # The CLI may have recorded its error after the interrupted write. That
    # transient report is not a new model generation or a quality-review pause.
    old_status = json.loads(worker.status_path.read_text())
    old_status.update(status='blocked', last_error='injected publication interruption')
    worker.status_path.write_text(json.dumps(old_status))
    assert _recover_online_publication(store, session['id'], seed, registration)
    recovered = json.loads(worker.status_path.read_text())
    count, sha, size = validate_gaussian_ply(worker.ply_path)
    assert recovered['artifact'] == {'vertex_count':count, 'sha256':sha, 'size_bytes':size}
    assert recovered['revision_number'] == 2 and recovered['visible_count'] == recovered['processed_count'] == 1
    assert recovered['accepted_ids'] == json.loads(worker.checkpoint_path.read_text()) == [cid]
    assert recovered['display_artifact']['sha256'] == hashlib.sha256(worker.display_path.read_bytes()).hexdigest()
    assert not (worker.root/'pending-publication.json').exists() and 'last_error' not in recovered
    assert _recover_online_publication(store, session['id'], seed, registration) is False
    resumed = OnlineCoordinator(store, session['id'], seed, registration,
                                _ChangedPublicationTrainer(json.loads(worker.checkpoint_path.read_text())))
    resumed.tick(steps=0)
    assert resumed.status['visible_count'] == 1 and resumed.status['artifact']['sha256'] == sha
    assert {item['receipt']['capture_id']:Path(item['image_path']).read_bytes()
            for item in store.export_session(session['id'])['captures']} == originals


@pytest.mark.parametrize('changed', ['newer-generation', 'corrupt-checkpoint', 'changed-original', 'excluded-original'])
def test_recovery_cannot_override_newer_or_unverified_source(tmp_path, monkeypatch, changed):
    store, project, session, seed, registration, worker = _interrupted_publication(tmp_path, monkeypatch)
    cid = store.captures(session['id'])[0]['capture_id']
    if changed == 'newer-generation':
        status = json.loads(worker.status_path.read_text()); status['revision_number'] = 9
        worker.status_path.write_text(json.dumps(status))
    elif changed == 'corrupt-checkpoint':
        worker.checkpoint_path.write_bytes(b'corrupted checkpoint')
    elif changed == 'changed-original':
        with store.connection() as db:
            row = db.execute('SELECT metadata_json FROM captures WHERE capture_id=?', (cid,)).fetchone()
            metadata = json.loads(row[0]); metadata['calibration_revision'] = 'changed-camera'
            db.execute('UPDATE captures SET metadata_json=? WHERE capture_id=?', (json.dumps(metadata), cid))
    else:
        from bridge.dashboard import set_capture_selection
        set_capture_selection(store, project['id'], session['id'], cid, False)
    paths = [worker.status_path, worker.checkpoint_path, worker.ply_path, worker.root/'pending-publication.json']
    protected = {path:path.read_bytes() for path in paths}
    with pytest.raises((OnlineError, __import__('bridge.store', fromlist=['StoreError']).StoreError)):
        _recover_online_publication(store, session['id'], seed, registration)
    assert {path:path.read_bytes() for path in paths} == protected


def test_recovery_rechecks_manifest_before_any_authoritative_write(tmp_path, monkeypatch):
    store, _project, session, seed, registration, worker = _interrupted_publication(tmp_path, monkeypatch, 'model')
    pair = worker.ply_path.read_bytes(), worker.checkpoint_path.read_bytes()
    def newer_manifest():
        status = json.loads(worker.status_path.read_text()); status['revision_number'] = 9
        worker.status_path.write_text(json.dumps(status))
    with pytest.raises(OnlineError, match='manifest changed'):
        _recover_online_publication(store, session['id'], seed, registration, before_commit=newer_manifest)
    assert (worker.ply_path.read_bytes(), worker.checkpoint_path.read_bytes()) == pair
    assert json.loads(worker.status_path.read_text())['revision_number'] == 9


@pytest.mark.parametrize('saved_revision', [0, -1, True, '1'])
def test_recovery_binds_reused_display_identity_without_rewriting_immutable_revision(
        tmp_path, monkeypatch, saved_revision):
    store, _project, session, seed, registration, worker = _interrupted_publication(
        tmp_path, monkeypatch, 'model')
    journal = json.loads((worker.root/'pending-publication.json').read_text())
    sha = journal['status']['display_artifact']['sha256']
    receipt_path = worker.revisions_path/(sha+'.json')
    receipt = json.loads(receipt_path.read_text()); receipt['revision_number'] = saved_revision
    receipt_path.write_text(json.dumps(receipt))
    original_receipt = receipt_path.read_bytes()
    if type(saved_revision) is int and saved_revision >= 0:
        assert _recover_online_publication(store, session['id'], seed, registration)
        assert json.loads(worker.status_path.read_text())['revision_number'] == 2
    else:
        with pytest.raises(OnlineError, match='immutable display receipt'):
            _recover_online_publication(store, session['id'], seed, registration)
    assert receipt_path.read_bytes() == original_receipt
