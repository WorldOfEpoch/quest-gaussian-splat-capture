"""Area-worker admission/count regressions with isolated originals and no CUDA."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import uuid

import numpy as np
import pytest
from PIL import Image

from bridge import online, online_torch, reconstruct, region_edit
from bridge.online import OnlineError
from bridge.store import Store


PROPS = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")


def splat_bytes(count):
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n" +
              "".join(f"property float {name}\n" for name in PROPS) + "end_header\n")
    return header.encode() + struct.pack("<14f", *([0.] * 14)) * count


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def area(tmp_path, monkeypatch):
    store = Store(tmp_path)
    project = store.create_project("Synthetic repair")
    origin = str(uuid.uuid4())
    session = store.create_session(project["id"], origin)
    ids = [str(uuid.uuid4()) for _ in range(7)]
    matrix = np.diag([2., 2., 2., 1.])
    metadata = {}
    for index, cid in enumerate(ids):
        image = ("unchanged-camera-original-" + cid).encode()
        pose = np.eye(4); pose[0, 3] = index * .1
        row = {"capture_id": cid, "project_id": project["id"], "session_id": session["id"],
               "tracking_origin_id": origin, "image_sha256": hashlib.sha256(image).hexdigest(),
               "image_encoding": "jpeg", "capture_source": "quest_camera", "tracking_valid": True,
               "image_orientation": "upright", "calibration_revision": "camera-1", "camera_id": "left",
               "image_width": 1280, "image_height": 960,
               "coordinate_convention": "unity_lh_xright_yup_zforward",
               "camera_to_session": pose.reshape(-1).tolist(),
               "intrinsics": {"fx": 100., "fy": 101., "cx": 640., "cy": 480.}}
        store.save_capture(session["id"], row, image)
        metadata[cid] = row
    seed = tmp_path / "measured-seed.ply"; seed.write_bytes(splat_bytes(3))
    registration = {"status": "verified", "verification_method": "source_job_pose_fit",
                    "project_id": project["id"], "session_id": session["id"], "tracking_origin_id": origin,
                    "seed_sha256": digest(seed), "evidence_job_id": str(uuid.uuid4()),
                    "basis": "opencv_rh_from_unity", "model_from_session": matrix.tolist(),
                    "physical_acceptance": "pending", "calibration_revision": "camera-1", "image_orientation": "upright",
                    "seed_capture_ids": ids[:6], "camera_signature": {key: metadata[ids[0]][key]
                        for key in ("camera_id", "image_width", "image_height", "intrinsics")},
                    "pose_fit": {"matched_views": 6, "baseline_m": .5, "position_rmse_m": .01,
                                 "max_position_error_m": .02, "max_rotation_error_deg": 1.},
                    "source_proof": {"captures": {}}, "capture_pose_residuals": {}}
    distortion = [.01, -.001, .0002, .0003, 0.]
    calibrated_k = np.array([[102.5, 0., 639.], [0., 103., 479.], [0., 0., 1.]])
    snapshot = store.export_session(session["id"])
    items = {item["receipt"]["capture_id"]: item for item in snapshot["captures"]}
    references = []
    for cid in ids[:6]:
        pose, _ = online.validated_camera(metadata[cid], matrix, "camera-1")
        pose[0, 3] += .008  # Optimized camera, not the approximate tracked one.
        references.append({"capture_id": cid, "image_path": Path(items[cid]["image_path"]),
                           "pose": pose, "intrinsics": calibrated_k.copy(),
                           "distortion_coefficients": distortion, "camera_source": "verified_seed_frame"})
        registration["source_proof"]["captures"][cid] = {
            "image_sha256": metadata[cid]["image_sha256"], "metadata_sha256": online._json_digest(metadata[cid])}
        registration["capture_pose_residuals"][cid] = {"image_sha256": metadata[cid]["image_sha256"],
            "position_error_m": .004, "rotation_error_deg": 0., "frame_name": f"{cid}.jpg"}
    online_folder = tmp_path / "projects" / project["id"] / "online" / session["id"]
    online_folder.mkdir(parents=True)
    original_ids = [str(uuid.uuid4())]
    (online_folder / "current.ckpt").write_text(json.dumps({"ids": original_ids}))
    (online_folder / "current.ply").write_bytes(splat_bytes(5))
    (online_folder / "display.ply").write_bytes(splat_bytes(4))
    source = {"registration": registration, "seed_sha256": digest(seed), "seed_ids": ids[:6],
              "artifact": {"sha256": digest(online_folder / "current.ply")}}
    (online_folder / "status.json").write_text(json.dumps(source))
    region_id = str(uuid.uuid4())
    folder = tmp_path / "projects" / project["id"] / "area-edits" / region_id
    folder.mkdir(parents=True)
    record = {"id": region_id, "name": "Small observed patch", "project_id": project["id"],
              "session_id": session["id"], "min": [-.2, -.2, 1.], "max": [.4, .4, 2.],
              "source_sha256": digest(online_folder / "current.ply"),
              "source_checkpoint_sha256": digest(online_folder / "current.ckpt"),
              "replacement_capture_ids": [ids[0], ids[6]], "status": "queued"}
    (folder / "region.json").write_text(json.dumps(record))
    events, trainers, verifiers = [], [], []
    validated_view = online_torch.GsplatTrainer._validated_view

    class FakeTrainer:
        _validated_view = staticmethod(validated_view)

        def __init__(self, seed_path, checkpoint, *, published_ply):
            self.params = {"means": [0.] * 5}
            self.accepted_ids = json.loads(Path(checkpoint).read_text())["ids"]
            self.added_total = 0
            self.initial_rows = 5
            self.has_pending_work = False
            self.images, self.admissions = [], []
            self.growth = 2
            self.after_train = None
            trainers.append(self); events.append("trainer")

        @property
        def capture_ids(self):
            return list(self.accepted_ids)

        def configure_geometry_scale(self, scale):
            self.scale = scale; events.append("scale")

        def configure_camera_distortion(self, values):
            self.distortion = list(values); events.append("distortion")

        def prime_baseline_views(self, views):
            self.references = list(views); self.images = list(views); events.append("prime")

        def begin_area_replacement(self, region, captures):
            self.images = []; events.append("clear")
            return {"removed_gaussians": 1, "preserved_boundary_gaussians": 1}

        def add_image(self, cid, image_path, pose, k):
            self.admissions.append((cid, Path(image_path), pose.copy(), k.copy()))
            self.params["means"].extend([0.] * self.growth)
            if cid not in self.accepted_ids:
                self.accepted_ids.append(cid)
            self.images.append(cid); events.append("add")

        def train_steps(self, count):
            if self.after_train:
                callback, self.after_train = self.after_train, None
                callback()
            events.append("train")

        def save(self, checkpoint, ply):
            # Reproduces the actual trainer's committed-only counter.
            assert self.added_total == 0
            self.added_total = len(self.params["means"]) - self.initial_rows
            checkpoint.write_text(json.dumps({"ids": self.accepted_ids}))
            ply.write_bytes(splat_bytes(len(self.params["means"])))
            events.append("save")
            return {"new_gaussians": self.added_total, "training_steps": 48}

        def lock_replacement_area(self):
            events.append("lock")

        def protect_existing_area(self, region):
            events.append("protect")

    class FakeVerifier:
        def __init__(self, owner_store, proof, views, scale):
            assert owner_store.root == store.root and proof == registration and scale == pytest.approx(2.)
            self.references = list(views); self.calls = []; self.error = None
            self.change = None; verifiers.append(self); events.append("verifier")

        def verify(self, view):
            self.calls.append(deepcopy(view))
            if self.error:
                raise OnlineError(self.error)
            pose = view["pose"].copy(); pose[0, 3] += .02
            result = {"method": "seed_sparse_landmark_pnp", "method_revision": "existing_owner",
                      "position_error_m": .01, "rotation_error_deg": 0., "inliers": 12,
                      "correspondences": 15, "reprojection_rmse_px": .3, "reprojection_max_px": .8,
                      "localized_pose": pose.tolist(), "camera_intrinsics": view["intrinsics"].tolist(),
                      "distortion_coefficients": distortion, "source_proof": {"seed_sha256": digest(seed)}}
            if self.change:
                self.change(result)
            return result

    monkeypatch.setattr(online_torch, "GsplatTrainer", FakeTrainer)
    # Keep the worker tests focused on identity, counts and publication state;
    # the real measured-camera gate is exercised separately below.
    monkeypatch.setattr(region_edit, "_prepare_area_review", lambda *_args: {
        "status": "prepared", "views": [{"capture_id": ids[2]}, {"capture_id": ids[3]}]})
    monkeypatch.setattr(region_edit, "_finish_area_review", lambda *_args: {
        "status": "ready_for_visual_review", "views": [{"capture_id": ids[2]}, {"capture_id": ids[3]}]})
    monkeypatch.setattr(online, "registration_from_seed_job", lambda *_args: (seed, registration))
    monkeypatch.setattr(online, "seed_camera_views", lambda *_args: references)
    monkeypatch.setattr(online, "ColmapLandmarkVerifier", FakeVerifier)
    monkeypatch.setattr(reconstruct, "_worker_lock", lambda *_args: nullcontext())

    def source_owner(_root, current_record):
        if (digest(online_folder / "current.ply") != current_record["source_sha256"] or
                digest(online_folder / "current.ckpt") != current_record["source_checkpoint_sha256"]):
            raise OnlineError("The source model changed; create a new area edit from the current model")
        return online_folder, source, seed

    monkeypatch.setattr(region_edit, "_source", source_owner)
    protected = {str(p): p.read_bytes() for p in
        [seed, *online_folder.iterdir(), *(Path(item["image_path"]) for item in items.values())]}
    return {"store": store, "root": tmp_path, "project": project["id"], "session": session["id"],
            "region": region_id, "record": record, "folder": folder, "source": source, "seed": seed,
            "ids": ids, "items": items, "metadata": metadata, "matrix": matrix,
            "references": references, "distortion": distortion, "k": calibrated_k,
            "events": events, "trainers": trainers, "verifiers": verifiers,
            "Trainer": FakeTrainer, "Verifier": FakeVerifier, "protected": protected}


def run(area):
    return region_edit.run_region_action(area["root"], area["project"], area["region"], "build")


def test_protect_existing_area_never_reads_replacement_photos_or_changes_ply(area):
    online_folder = area['root'] / 'projects' / area['project'] / 'online' / area['session']
    area['source']['accepted_ids'] = json.loads((online_folder / 'current.ckpt').read_text())['ids']
    result = region_edit.run_region_action(area['root'], area['project'], area['region'], 'protect')
    assert result['status'] == 'locked' and result['draft_action'] == 'protect'
    assert result['replacement_capture_ids'] == []
    assert result['artifact']['sha256'] == result['source_sha256']
    assert result['accepted_ids'] == area['source']['accepted_ids']
    assert result['quality_review']['method'] == 'unchanged_geometry_protection_v1'
    assert result['clear_metrics']['removed_gaussians'] == 0
    assert area['events'] == ['trainer', 'protect', 'save']
    assert not area['verifiers']
    assert all(Path(p).read_bytes() == content for p, content in area['protected'].items())


def test_protect_refuses_a_trainer_that_changes_geometry(area, monkeypatch):
    def bad_protect(self, region):
        self.params['means'].append(0.)
    monkeypatch.setattr(area['Trainer'], 'protect_existing_area', bad_protect)
    with pytest.raises(OnlineError, match='changed the source model'):
        region_edit.run_region_action(area['root'], area['project'], area['region'], 'protect')
    assert not (area['folder'] / 'candidate.ply').exists()
    assert all(Path(p).read_bytes() == content for p, content in area['protected'].items())


def change_metadata(area, cid, mutation):
    metadata = deepcopy(area["metadata"][cid]); mutation(metadata)
    with area["store"].connection() as db:
        db.execute("UPDATE captures SET metadata_json=? WHERE capture_id=?", (json.dumps(metadata), cid))


def exclude(area, cid):
    from bridge.dashboard import set_capture_selection
    set_capture_selection(area["store"], area["project"], area["session"], cid, False)


def test_build_uses_measured_cameras_and_committed_export_growth(area):
    result = run(area)
    trainer = area["trainers"][0]
    assert trainer.scale == pytest.approx(2.) and trainer.distortion == area["distortion"]
    assert area["events"].index("prime") < area["events"].index("clear") < area["events"].index("add")
    assert trainer.images == area["record"]["replacement_capture_ids"]
    baseline, delta = trainer.admissions
    np.testing.assert_array_equal(baseline[2], area["references"][0]["pose"])
    np.testing.assert_array_equal(baseline[3], area["k"])
    tracked, _ = online.validated_camera(area["metadata"][delta[0]], area["matrix"], "camera-1")
    assert delta[2][0, 3] == pytest.approx(tracked[0, 3] + .02)
    np.testing.assert_array_equal(delta[3], area["k"])
    assert result["status"] == "draft_ready" and result["replacement_gaussians"] == 4
    assert result["appended_gaussians"] == 4
    assert result["quality_review"]["status"] == "ready_for_visual_review"
    assert result["quality_review"]["candidate_sha256"] == result["artifact"]["sha256"]
    assert result["quality_review"]["candidate_checkpoint_sha256"] == result["checkpoint_sha256"]
    assert result["model_metrics"]["new_gaussians"] == 4
    assert area["ids"][0] in result["accepted_ids"]  # Keep baseline delta lineage internally consistent.
    proof = result["replacement_registration"]
    assert proof["model_units_per_meter"] == pytest.approx(2.) and proof["distortion_coefficients"] == area["distortion"]
    assert proof["captures"][area["ids"][0]]["proof"]["method"] == "verified_seed_frame"
    assert proof["captures"][area["ids"][6]]["proof"]["method"] == "seed_sparse_landmark_pnp"
    assert all(Path(p).read_bytes() == content for p, content in area["protected"].items())
    for name in ("current.ckpt", "current.ply", "display.ply", "status.json"):
        assert (area["folder"] / "original" / name).read_bytes() == area["protected"][str(
            area["root"] / "projects" / area["project"] / "online" / area["session"] / name)]


def test_seed_only_replacement_does_not_require_new_photo_localization(area):
    area["record"]["replacement_capture_ids"] = area["ids"][:2]
    (area["folder"] / "region.json").write_text(json.dumps(area["record"]))
    result = run(area)
    assert not area["verifiers"] and result["replacement_gaussians"] == 4
    assert all(cid in result["accepted_ids"] for cid in area["ids"][:2])


@pytest.mark.parametrize("kind", ["unmeasured", "changed_source", "excluded_seed", "excluded_replacement",
                                  "foreign_origin", "camera_dimensions", "camera_intrinsics", "invalid_tracking"])
def test_untrusted_inputs_fail_before_trainer_or_area_clear(area, monkeypatch, kind):
    registration = area["source"]["registration"]
    if kind == "unmeasured":
        registration.pop("verification_method")
    elif kind == "changed_source":
        changed = deepcopy(registration); changed["calibration_revision"] = "different"
        monkeypatch.setattr(online, "registration_from_seed_job", lambda *_: (area["seed"], changed))
    elif kind.startswith("excluded"):
        exclude(area, area["ids"][5 if kind == "excluded_seed" else 6])
    elif kind == "foreign_origin":
        change_metadata(area, area["ids"][6], lambda m: m.update(tracking_origin_id=str(uuid.uuid4())))
    elif kind == "camera_dimensions":
        change_metadata(area, area["ids"][6], lambda m: m.update(image_width=640))
    elif kind == "camera_intrinsics":
        change_metadata(area, area["ids"][6], lambda m: m["intrinsics"].update(fx=104.))
    elif kind == "invalid_tracking":
        change_metadata(area, area["ids"][6], lambda m: m.update(tracking_valid=False))
    with pytest.raises(OnlineError):
        run(area)
    assert not area["trainers"] and "clear" not in area["events"]
    assert not (area["folder"] / "candidate.ply").exists()


@pytest.mark.parametrize("kind", ["no_support", "missing_pose", "wrong_position", "wrong_rotation", "wrong_distortion"])
def test_new_photo_requires_independent_valid_localization(area, monkeypatch, kind):
    original = area["Verifier"].verify
    def altered(self, view):
        if kind == "no_support":
            raise OnlineError("Photo lacks independently observed sparse landmark correspondences")
        result = original(self, view)
        if kind == "missing_pose": result.pop("localized_pose")
        elif kind == "wrong_position": result["localized_pose"][0][3] += .3
        elif kind == "wrong_rotation":
            theta = np.deg2rad(6.)
            rotation = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
            pose = np.asarray(result["localized_pose"]); pose[:3, :3] = rotation @ pose[:3, :3]
            result["localized_pose"] = pose.tolist()
        elif kind == "wrong_distortion": result["distortion_coefficients"] = [0.] * 5
        return result
    monkeypatch.setattr(area["Verifier"], "verify", altered)
    with pytest.raises(OnlineError): run(area)
    assert not area["trainers"] and "clear" not in area["events"]


def test_zero_growth_is_rejected_without_exporting_a_cleared_draft(area, monkeypatch):
    original = area["Trainer"].__init__
    def no_growth(self, *args, **kwargs):
        original(self, *args, **kwargs); self.growth = 0
    monkeypatch.setattr(area["Trainer"], "__init__", no_growth)
    with pytest.raises(OnlineError, match="no supported replacement geometry"):
        run(area)
    assert "save" not in area["events"] and not (area["folder"] / "candidate.ply").exists()
    assert all(Path(p).read_bytes() == content for p, content in area["protected"].items())


def test_independent_camera_rejection_precedes_clearing_and_training(area, monkeypatch):
    monkeypatch.setattr(region_edit, "_prepare_area_review", lambda *_args: {
        "status": "quality_rejected", "reason": "insufficient independent area cameras"})
    result = run(area)
    assert result["status"] == "quality_rejected"
    assert result["quality_review"]["status"] == "quality_rejected"
    assert "clear" not in area["events"] and "save" not in area["events"]
    assert not (area["folder"] / "candidate.ply").exists()


def test_rejected_rebuild_does_not_advertise_prior_draft_as_current(area, monkeypatch):
    first = run(area)
    previous = (area["folder"] / "candidate.ply").read_bytes()
    assert first["artifact"]["sha256"] == digest(area["folder"] / "candidate.ply")
    monkeypatch.setattr(region_edit, "_prepare_area_review", lambda *_args: {
        "status": "quality_rejected", "reason": "independent cameras do not overlap this area"})
    second = run(area)
    assert second["status"] == "quality_rejected"
    assert "artifact" not in second and "checkpoint_sha256" not in second
    assert (area["folder"] / "candidate.ply").read_bytes() == previous  # recoverable private old draft


def test_bad_replacement_stays_private_after_export(area, monkeypatch):
    monkeypatch.setattr(region_edit, "_finish_area_review", lambda *_args: {
        "status": "quality_rejected", "independent_validation_gain": False, "views": []})
    result = run(area)
    assert result["status"] == "quality_rejected"
    assert result["artifact"]["sha256"] == digest(area["folder"] / "candidate.ply")
    assert result["quality_review"]["candidate_checkpoint_sha256"] == digest(area["folder"] / "candidate.ckpt")
    assert all(Path(p).read_bytes() == content for p, content in area["protected"].items())


@pytest.mark.parametrize("kind", ["replacement_metadata", "excluded_seed", "source_model"])
def test_change_during_heldout_render_cannot_commit_candidate(area, monkeypatch, kind):
    def changed_during_review(*_args):
        if kind == "replacement_metadata":
            change_metadata(area, area["ids"][6], lambda m: m.update(calibration_revision="changed"))
        elif kind == "excluded_seed":
            exclude(area, area["ids"][5])
        else:
            current = area["root"] / "projects" / area["project"] / "online" / area["session"] / "current.ply"
            current.write_bytes(splat_bytes(6))
        return {"status": "ready_for_visual_review", "views": []}

    monkeypatch.setattr(region_edit, "_finish_area_review", changed_during_review)
    with pytest.raises(OnlineError):
        run(area)
    assert "save" in area["events"]
    assert not (area["folder"] / "candidate.ply").exists()
    assert not (area["folder"] / "candidate.ckpt").exists()


def test_worker_failure_keeps_latest_progress_and_releases_action_lock(area, monkeypatch):
    lock = area["folder"] / "action.lock"
    lock.write_text("worker owns this operation")
    area["record"].update(artifact={"sha256": "previous"}, checkpoint_sha256="previous")
    (area["folder"] / "region.json").write_text(json.dumps(area["record"]))

    def progress_then_fail(root, project_id, region_id, action):
        folder, latest = region_edit._record(root, project_id, region_id)
        latest.update(status="building", processed_photos=2,
                      clear_metrics={"removed_gaussians": 574, "preserved_boundary_gaussians": 554})
        latest.pop("artifact", None)
        latest.pop("checkpoint_sha256", None)
        region_edit._write(folder, latest)
        raise OnlineError("No supported replacement geometry inside the box")

    monkeypatch.setattr(region_edit, "run_region_action", progress_then_fail)
    monkeypatch.setattr(sys, "argv", ["region_edit", "--data-dir", str(area["root"]),
        "--project-id", area["project"], "--region-id", area["region"], "--action", "build"])
    with pytest.raises(OnlineError, match="No supported replacement geometry"):
        region_edit.main()
    saved = json.loads((area["folder"] / "region.json").read_text())
    assert saved["status"] == "failed" and saved["processed_photos"] == 2
    assert saved["clear_metrics"] == {"removed_gaussians": 574, "preserved_boundary_gaussians": 554}
    assert "artifact" not in saved and "checkpoint_sha256" not in saved
    assert saved["message"] == "No supported replacement geometry inside the box"
    assert not lock.exists()


def test_fixed_area_projection_is_geometric_and_outside_view_fails_closed():
    box = {"id": str(uuid.uuid4()), "name": "patch", "min": [-.2, -.2, 1.7], "max": [.2, .2, 2.3]}
    k = np.array([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]])
    mask = region_edit._area_review_mask(np.eye(4), k, (128, 128), box, 1.)
    assert mask.sum() > 16 and mask[64, 64] and not mask[0, 0]
    backward = np.eye(4); backward[2, 3] = 3.
    assert not region_edit._area_review_mask(backward, k, (128, 128), box, 1.).any()
    for axis, offset in ((0, 10.), (0, -10.), (1, 10.), (1, -10.)):
        offscreen = np.eye(4); offscreen[axis, 3] = offset
        assert not region_edit._area_review_mask(offscreen, k, (128, 128), box, 1.).any()
    nonfinite = k.copy(); nonfinite[0, 0] = np.nan
    assert not region_edit._area_review_mask(np.eye(4), nonfinite, (128, 128), box, 1.).any()


def test_replacement_review_excludes_training_photos_and_requires_independent_gain(tmp_path, monkeypatch):
    box = {"id": str(uuid.uuid4()), "name": "patch", "min": [-.2, -.2, 1.7], "max": [.2, .2, 2.3]}
    k = np.array([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]])
    refs = []
    for index in range(3):
        path = tmp_path / f"camera-{index}.png"
        Image.new("RGB", (128, 128), (127, 127, 127)).save(path)
        pose = np.eye(4); pose[0, 3] = index * .1
        refs.append({"capture_id": f"camera-{index}", "image_path": path,
                     "pose": pose, "intrinsics": k})

    class CameraOnlyTrainer:
        max_image_edge = 640
        _validated_view = staticmethod(online_torch.GsplatTrainer._validated_view)

        def _scaled_image(self, path, intrinsics):
            return np.full((128, 128, 3), .5, dtype=np.float32), intrinsics.copy()

    trainer = CameraOnlyTrainer()
    monkeypatch.setattr(region_edit, "_area_review_rgb", lambda *_args: np.zeros((128, 128, 3), np.float32))
    prepared = region_edit._prepare_area_review(trainer, refs, box, ["camera-0"], 1.)
    assert prepared["status"] == "prepared"
    assert {v["capture_id"] for v in prepared["views"]} == {"camera-1", "camera-2"}
    assert all(np.linalg.norm(np.asarray(v["camera_center_m"])) >= .1 for v in prepared["views"])

    def improved(_trainer, pose, intrinsics, shape):
        result = np.zeros((*shape, 3), np.float32)
        mask = region_edit._area_review_mask(pose, intrinsics, shape, box, 1.)
        result[mask] = .5
        return result

    monkeypatch.setattr(region_edit, "_area_review_rgb", improved)
    report = region_edit._finish_area_review(trainer, prepared, ["camera-0"])
    assert report["status"] == "ready_for_visual_review"
    assert report["independent_validation_gain"]
    assert all(row["held_out"] and row["outside_support_max_change"] == 0 for row in report["views"])


@pytest.mark.parametrize("kind", ["excluded", "metadata_changed"])
def test_changes_during_training_cannot_save_a_new_draft(area, monkeypatch, kind):
    original = area["Trainer"].__init__
    def changed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.after_train = (lambda: exclude(area, area["ids"][6])) if kind == "excluded" else (
            lambda: change_metadata(area, area["ids"][6], lambda m: m.update(calibration_revision="changed")))
    monkeypatch.setattr(area["Trainer"], "__init__", changed)
    with pytest.raises(OnlineError, match="changed during"):
        run(area)
    assert "save" not in area["events"] and not (area["folder"] / "candidate.ply").exists()


def test_pending_current_cohort_cannot_be_repurposed_as_area_geometry(area, monkeypatch):
    original = area["Trainer"].__init__
    def pending(self, *args, **kwargs):
        original(self, *args, **kwargs); self.has_pending_work = True
    monkeypatch.setattr(area["Trainer"], "__init__", pending)
    with pytest.raises(OnlineError, match="pending model refinement"):
        run(area)
    assert "clear" not in area["events"] and "save" not in area["events"]


@pytest.mark.parametrize("kind", ["unselected_seed_excluded", "source_calibration_changed"])
def test_complete_seed_evidence_is_revalidated_before_committing_draft(area, monkeypatch, kind):
    source_file = area["root"] / "source-calibration.json"
    source_file.write_text("frozen calibration")
    registration = area["source"]["registration"]
    registration["source_proof"]["frame_files"] = {"training_config": digest(source_file)}
    def measured_again(*_args):
        current = deepcopy(registration)
        current["source_proof"]["frame_files"]["training_config"] = digest(source_file)
        return area["seed"], current
    monkeypatch.setattr(online, "registration_from_seed_job", measured_again)
    original = area["Trainer"].__init__
    def changed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.after_train = ((lambda: exclude(area, area["ids"][5])) if kind == "unselected_seed_excluded"
                            else lambda: source_file.write_text("changed calibration"))
    monkeypatch.setattr(area["Trainer"], "__init__", changed)
    with pytest.raises(OnlineError, match="Measured seed .* changed during"):
        run(area)
    assert "save" in area["events"]  # Only private next-files may exist.
    assert not (area["folder"] / "candidate.ply").exists()
    assert not (area["folder"] / "candidate.ckpt").exists()


def test_tampered_candidate_checkpoint_cannot_be_locked(area):
    record = run(area)
    (area["folder"] / "candidate.ckpt").write_text("changed draft checkpoint")
    with pytest.raises(OnlineError, match="draft checkpoint differs"):
        region_edit.run_region_action(area["root"], area["project"], area["region"], "lock")
    assert len(area["trainers"]) == 1 and "lock" not in area["events"]


def prepare_locked_apply(area, monkeypatch):
    """Prepare synthetic measured draft state; no actual CUDA or wearer acceptance."""
    current = area['root']/'projects'/area['project']/'online'/area['session']
    existing_id = json.loads((current/'current.ckpt').read_text())['ids'][0]
    metadata = deepcopy(area['metadata'][area['ids'][6]])
    image = b'unchanged-existing-delta'
    metadata.update(capture_id=existing_id, image_sha256=hashlib.sha256(image).hexdigest())
    area['store'].save_capture(area['session'], metadata, image)
    tracked, _ = online.validated_camera(metadata, area['matrix'], 'camera-1')
    existing_proof = {'method': 'seed_sparse_landmark_pnp', 'localized_pose': tracked.tolist(),
        'camera_intrinsics': area['k'].tolist(), 'distortion_coefficients': area['distortion'],
        'image_sha256': metadata['image_sha256'], 'metadata_sha256': online._json_digest(metadata),
        'seed_sha256': digest(area['seed'])}
    source = area['source']
    source.update(project_id=area['project'], session_id=area['session'], status='paused_quality_review',
        revision_number=4, accepted_ids=[existing_id], published_capture_ids=[existing_id],
        photo_registration={existing_id: existing_proof}, training_steps=7)
    source['artifact'].update(size_bytes=(current/'current.ply').stat().st_size, vertex_count=5)
    source['display_artifact'] = {'sha256': digest(current/'display.ply'),
        'size_bytes': (current/'display.ply').stat().st_size, 'vertex_count': 4,
        'source_sha256': digest(current/'current.ply')}
    (current/'status.json').write_text(json.dumps(source))
    original_save = area['Trainer'].save
    def lineage_save(self, checkpoint, ply):
        return {**original_save(self, checkpoint, ply), 'published_capture_ids': self.capture_ids,
            'rejected_capture_ids': [], 'pending_refinement': {}}
    monkeypatch.setattr(area['Trainer'], 'save', lineage_save)
    record = run(area)
    record['status'] = 'locked'  # This fixture tests apply, independently of GPU lock export.
    (area['folder']/'region.json').write_text(json.dumps(record))
    originals = {name: (current/name).read_bytes() for name in
        ('current.ckpt', 'current.ply', 'display.ply', 'status.json')}
    return current, record, originals, existing_id


def test_apply_merges_verified_provenance_and_unique_counts_then_undo(area, monkeypatch):
    current, record, originals, existing_id = prepare_locked_apply(area, monkeypatch)
    previous = deepcopy(area['source']['photo_registration'])
    result = region_edit.request_region_action(area['root'], area['project'], area['region'], 'publish')
    assert result['status'] == 'published'
    state = json.loads((current/'status.json').read_text())
    assert state['status'] == 'paused_quality_review' and state['revision_number'] == 5
    assert state['processed_count'] == state['visible_count'] == 8
    assert state['photo_registration'][existing_id] == previous[existing_id]
    for cid in record['replacement_capture_ids']:
        frozen = record['replacement_registration']['captures'][cid]
        assert state['photo_registration'][cid] == {**frozen['proof'],
            'image_sha256': frozen['image_sha256'], 'metadata_sha256': frozen['metadata_sha256'],
            'seed_sha256': state['seed_sha256']}
    ledger = online._publication_ledger(state, state['registration'])
    assert len(ledger) == 8 and area['ids'][0] in ledger and area['ids'][6] in ledger
    assert state['published_capture_ids'] == record['model_metrics']['published_capture_ids']
    assert state['pending_count'] == state['rejected_count'] == 0
    region_edit.request_region_action(area['root'], area['project'], area['region'], 'restore')
    restored = json.loads((current/'status.json').read_text())
    assert restored['revision_number'] == 6
    assert restored['photo_registration'] == previous
    for name in ('current.ckpt', 'current.ply', 'display.ply'):
        assert (current/name).read_bytes() == originals[name]


@pytest.mark.parametrize('kind', ['missing_proof', 'wrong_seed', 'wrong_frame', 'wrong_photo',
    'wrong_pose', 'changed_metadata', 'excluded_delta', 'excluded_seed', 'bad_quality',
    'wrong_quality_hash', 'unproved_ingress', 'old_proof_conflict'])
def test_apply_rejects_changed_or_unproved_area_without_touching_current(area, monkeypatch, kind):
    current, record, originals, _existing = prepare_locked_apply(area, monkeypatch)
    proof = record['replacement_registration']
    cid = area['ids'][6]
    if kind == 'missing_proof': proof['captures'].pop(cid)
    elif kind == 'wrong_seed': proof['seed_sha256'] = 'f'*64
    elif kind == 'wrong_frame': proof['source_proof'] = {'frame_files': {'transforms': 'f'*64}}
    elif kind == 'wrong_photo': proof['captures'][cid]['image_sha256'] = 'f'*64
    elif kind == 'wrong_pose': proof['captures'][cid]['proof']['localized_pose'][0][3] += .05
    elif kind == 'changed_metadata': change_metadata(area, cid, lambda m: m.update(image_width=640))
    elif kind == 'excluded_delta': exclude(area, cid)
    elif kind == 'excluded_seed': exclude(area, area['ids'][5])
    elif kind == 'bad_quality': record['quality_review']['status'] = 'quality_rejected'
    elif kind == 'wrong_quality_hash': record['quality_review']['candidate_sha256'] = 'f'*64
    elif kind == 'unproved_ingress': record['accepted_ids'].append(str(uuid.uuid4()))
    elif kind == 'old_proof_conflict':
        area['source']['photo_registration'][cid] = {'image_sha256': 'f'*64,
            'metadata_sha256': proof['captures'][cid]['metadata_sha256']}
        (current/'status.json').write_text(json.dumps(area['source']))
        originals['status.json'] = (current/'status.json').read_bytes()
    (area['folder']/'region.json').write_text(json.dumps(record))
    with pytest.raises(OnlineError):
        region_edit.request_region_action(area['root'], area['project'], area['region'], 'publish')
    assert {name: (current/name).read_bytes() for name in originals} == originals
    assert not (current/'.display.tmp.ply').exists()


def test_apply_keeps_frozen_camera_when_repeat_pnp_inlier_diagnostics_vary(area, monkeypatch):
    current, record, _originals, _existing = prepare_locked_apply(area, monkeypatch)
    frozen = deepcopy(record['replacement_registration']['captures'][area['ids'][6]])
    verify = area['Verifier'].verify
    def repeat(self, view):
        proof = verify(self, view)
        proof['localized_pose'][0][3] += .0001
        proof['inliers'] = 13
        proof['reprojection_max_px'] += .01
        return proof
    monkeypatch.setattr(area['Verifier'], 'verify', repeat)
    region_edit.request_region_action(area['root'], area['project'], area['region'], 'publish')
    saved = json.loads((current/'status.json').read_text())['photo_registration'][area['ids'][6]]
    assert saved == {**frozen['proof'], 'image_sha256': frozen['image_sha256'],
        'metadata_sha256': frozen['metadata_sha256'], 'seed_sha256': area['source']['seed_sha256']}
