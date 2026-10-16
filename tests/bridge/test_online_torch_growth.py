"""Focused checks for bounded live growth and side-effect-free evaluation."""

import numpy as np
import pytest
from PIL import Image


def test_unchanged_reviewed_sh3_export_preserves_exact_file_but_real_changes_publish(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    from bridge.online_torch import GsplatTrainer
    names = ("x", "y", "z") + tuple(f"f_rest_{i}" for i in range(45))
    vertices = np.zeros(3, dtype=[(name, "<f4") for name in names])
    vertices["f_rest_44"] = .37
    source, output = tmp_path / "reviewed.ply", tmp_path / "staged.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(vertices, "vertex")], text=False,
                    comments=["accepted SH3 optimizer step 7999"],
                    obj_info=["preserve original source identity"]).write(str(source))
    original = source.read_bytes()
    trainer = GsplatTrainer.__new__(GsplatTrainer)
    trainer._published_ply_path = source
    trainer._write_published_ply(vertices.copy(), output)
    assert output.read_bytes() == original
    # A real change in the last SH3 property must never masquerade as unchanged.
    changed = vertices.copy()
    changed["f_rest_44"][1] += .01
    trainer._write_published_ply(changed, output)
    assert output.read_bytes() != original
    np.testing.assert_array_equal(plyfile.PlyData.read(str(output))["vertex"].data, changed)
    assert source.read_bytes() == original


@pytest.mark.parametrize("units", [1., .025, 4.])
def test_growth_can_use_older_overlapping_view_with_geometric_checks(units):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from bridge.online_torch import GsplatTrainer

    class ArrayValue:
        def __init__(self, array):
            self.array = np.asarray(array)

        def numpy(self):
            return self.array

        def __len__(self):
            return len(self.array)

        def detach(self):
            return self

        def cpu(self):
            return self

    rng = np.random.default_rng(41)
    # A planar textured wall at z=2.56. A camera translated +0.2 along X
    # observes its features 20 pixels left at fx=256.
    gray = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    current = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.
    shifted = np.zeros_like(current)
    shifted[:, :-20] = current[:, 20:]
    blank = np.zeros_like(current)
    old_pose = np.eye(4)
    old_pose[0, 3] = .2 * units
    intrinsics = np.array([[256., 0., 128.], [0., 256., 128.], [0., 0., 1.]])
    trainer = GsplatTrainer.__new__(GsplatTrainer)
    trainer.model_units_per_meter = units
    trainer.images = [{"pixels": ArrayValue(shifted), "pose": ArrayValue(old_pose),
                       "intrinsics": ArrayValue(intrinsics)}]
    for _ in range(4):
        trainer.images.append({"pixels": ArrayValue(blank), "pose": ArrayValue(old_pose),
                               "intrinsics": ArrayValue(intrinsics)})
    trainer.params = {"means": ArrayValue(np.array([[100., 100., 100.]]) * units)}
    trainer.removed_rows = ArrayValue(np.array([False]))

    points, colors, scales = trainer._new_points(current, np.eye(4), intrinsics)
    assert len(points) >= 6
    assert len(points) == len(colors) == len(scales)
    assert trainer.last_growth_diagnostics["triangulated_pairs"] >= 1
    assert trainer.last_growth_diagnostics["added_points"] == len(points)
    assert np.isfinite(points).all()
    assert np.median(points[:, 2]) / units == pytest.approx(2.56, abs=.03)
    assert np.min(scales) >= .003 * units and np.max(scales) <= .08 * units


def test_render_l1_does_not_change_model_optimizer_or_receipts(tmp_path):
    torch = pytest.importorskip("torch")
    plyfile = pytest.importorskip("plyfile")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA gsplat worker is unavailable")
    from bridge.online_torch import GsplatTrainer

    names = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    vertex = np.zeros(3, dtype=[(name, "<f4") for name in names])
    vertex["z"] = 2.
    vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = -2.
    vertex["rot_0"] = 1.
    seed = tmp_path / "seed.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(vertex, "vertex")], text=False).write(str(seed))
    photo = tmp_path / "held-out.png"
    Image.new("RGB", (256, 128), (128, 64, 32)).save(photo)
    trainer = GsplatTrainer(seed, max_image_edge=128)
    before = {name: parameter.detach().clone() for name, parameter in trainer.params.items()}
    optimizer_before = len(trainer.optimizer.state)

    def constant_rasterizer(**kwargs):
        rendered = torch.full((1, kwargs["height"], kwargs["width"], 3), .5,
                              device=trainer.device)
        return rendered, None, {}

    trainer.rasterization = constant_rasterizer
    loss = trainer.render_l1(photo, np.eye(4),
                             np.array([[100., 0., 128.], [0., 100., 64.], [0., 0., 1.]]))
    assert loss == pytest.approx(np.mean(np.abs(np.array([128, 64, 32]) / 255. - .5)), abs=1e-6)
    assert trainer.step == 0 and trainer.accepted_ids == []
    assert len(trainer.optimizer.state) == optimizer_before
    for name, parameter in trainer.params.items():
        assert torch.equal(parameter, before[name])


def _cuda_seed(tmp_path, *, degree=0, units=.25):
    torch = pytest.importorskip("torch")
    plyfile = pytest.importorskip("plyfile")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA gsplat worker is unavailable")
    from bridge.online_torch import GsplatTrainer
    fields = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    extras = [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]
    vertices = np.zeros(4, dtype=[(name, "<f4") for name in (*fields, *extras)])
    vertices["x"] = np.array([-.2, .2, -.2, .2]) * units
    vertices["y"] = np.array([-.2, -.2, .2, .2]) * units
    vertices["z"] = 2. * units
    vertices["rot_0"] = 1.
    for axis in range(3):
        vertices[f"scale_{axis}"] = np.log(.02 * units)
    for i, name in enumerate(extras):
        vertices[name] = (i + 1) * .002
    seed = tmp_path / "seed.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(vertices, "vertex")], text=False).write(str(seed))
    trainer = GsplatTrainer(seed, max_image_edge=128)
    trainer.configure_geometry_scale(units)
    photo = tmp_path / "view.png"
    Image.new("RGB", (32, 32), (80, 100, 120)).save(photo)
    k = np.array([[28., 0., 16.], [0., 28., 16.], [0., 0., 1.]])
    views = []
    for i in range(3):
        pose = np.eye(4)
        pose[0, 3] = i * .2 * units
        views.append({"capture_id": f"baseline-{i}", "image_path": photo,
                      "pose": pose, "intrinsics": k})
    return trainer, vertices, seed, photo, k, views


def test_baseline_priming_preserves_seed_and_ingress_cursor(tmp_path):
    trainer, source, seed, photo, k, views = _cuda_seed(tmp_path, degree=3)
    torch = trainer.torch
    before = {name: value.detach().clone() for name, value in trainer.params.items()}
    assert trainer.prime_baseline_views(views) == 3
    assert sum(v["validation_only"] for v in trainer.images) == 2
    assert trainer.capture_ids == [] and trainer.added_total == 0
    assert not trainer.has_pending_work and trainer.step == 0
    for name, value in trainer.params.items():
        assert torch.equal(value, before[name])
    checkpoint, ply = tmp_path / "prime.ckpt", tmp_path / "current.ply"
    metrics = trainer.save(checkpoint, ply)
    plyfile = pytest.importorskip("plyfile")
    actual = plyfile.PlyData.read(str(ply))["vertex"].data
    assert actual.tobytes() == source.tobytes()
    assert metrics["accepted_views"] == 0
    colors, appearance = trainer._appearance(len(source))
    assert appearance == {"sh_degree": 3}
    assert colors.shape == (len(source), 16, 3)
    assert colors[0, 1, 1].item() == pytest.approx(source["f_rest_15"][0])
    assert torch.equal(colors, trainer._published_sh)


def test_baseline_cache_rejects_untranslated_or_reflected_cameras(tmp_path):
    trainer, _, _, _, _, views = _cuda_seed(tmp_path)
    for view in views:
        view["pose"] = np.eye(4)
    from bridge.online import OnlineError
    with pytest.raises(OnlineError, match="independent"):
        trainer.prime_baseline_views(views)
    assert trainer.images == [] and trainer.capture_ids == []
    views[0]["pose"][0, 0] = -1
    with pytest.raises(OnlineError, match="camera"):
        trainer.prime_baseline_views(views)


def test_resumed_baseline_hydration_preserves_guard_ids_and_refreshes_measured_cameras(tmp_path):
    trainer, _, seed, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_steps = 7
    checkpoint, current = tmp_path / "state.ckpt", tmp_path / "current.ply"
    trainer.save(checkpoint, current)
    from bridge.online_torch import GsplatTrainer
    resumed = GsplatTrainer(seed, checkpoint, max_image_edge=128)
    guards = [v["capture_id"] for v in resumed.images if v.get("validation_only")]
    assert guards == ["baseline-0", "baseline-2"]
    updated = []
    for i in range(53):
        pose = np.eye(4)
        pose[0, 3] = (i / 52. - .5) * resumed.model_units_per_meter
        updated.append({"capture_id": f"measured-{i}", "image_path": photo,
                        "pose": pose, "intrinsics": k.copy()})
    for index, capture_id, position in ((1, guards[0], .02), (3, guards[1], .5)):
        updated[index]["capture_id"] = capture_id
        updated[index]["pose"][0, 3] = position * resumed.model_units_per_meter
        updated[index]["intrinsics"][0, 0] = 29. + index
    updated[2]["capture_id"] = "baseline-1"
    # Neither old guard survives the previous uniform 53 -> 24 sampling.
    assert not {1, 3}.intersection(np.linspace(0, 52, 24).astype(int))
    original = {name: value.detach().clone() for name, value in resumed.params.items()}
    optimizer = resumed.optimizer
    assert resumed.prime_baseline_views(updated) == 24
    assert len(resumed.images) == 24
    refreshed = [v for v in resumed.images if v.get("validation_only")]
    assert [v["capture_id"] for v in refreshed] == guards
    for row, index in zip(refreshed, (1, 3)):
        np.testing.assert_array_equal(row["pose"].numpy(), updated[index]["pose"].astype(np.float32))
        np.testing.assert_array_equal(row["intrinsics"].numpy(), updated[index]["intrinsics"].astype(np.float32))
    assert resumed.images[0] is refreshed[0] and resumed.images[1] is refreshed[1]
    assert resumed.capture_ids == ["new-photo"] and resumed.published_capture_ids == []
    assert resumed.candidate_capture_ids == ["new-photo"] and resumed.candidate_steps == 7
    assert resumed.has_pending_work and resumed.added_total == 0
    assert resumed.optimizer is optimizer and resumed.step == 0
    assert resumed.images[-1]["capture_id"] == "new-photo"
    for name, value in resumed.params.items():
        assert resumed.torch.equal(value, original[name])


@pytest.mark.parametrize("failure", ["missing", "collapsed"])
def test_repriming_refuses_missing_or_untranslated_original_guards_without_mutation(tmp_path, failure):
    trainer, _, _, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    original_cache = trainer.images
    updated = [{**v, "pose": v["pose"].copy(), "intrinsics": v["intrinsics"].copy()} for v in views]
    if failure == "missing":
        updated[0]["capture_id"] = "replacement-camera"
    else:
        updated[2]["pose"] = updated[0]["pose"].copy()
        # Other independent cameras cannot replace the established guard pair.
        updated[1]["pose"][0, 3] = 3. * trainer.model_units_per_meter
    from bridge.online import OnlineError
    with pytest.raises(OnlineError, match="held-out|independent"):
        trainer.prime_baseline_views(updated)
    assert trainer.images is original_cache
    assert trainer.capture_ids == [] and trainer.published_capture_ids == []
    assert trainer.step == 0 and not trainer.has_pending_work


def test_measured_distortion_is_shared_by_baseline_new_photo_evaluation_and_resume(tmp_path):
    cv2 = pytest.importorskip("cv2")
    trainer, source, seed, photo, _, views = _cuda_seed(tmp_path, degree=3)
    rng = np.random.default_rng(53)
    raw = rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8)
    Image.fromarray(raw).save(photo)
    original_bytes = photo.read_bytes()
    k = np.array([[166., 0., 128.], [0., 170., 64.], [0., 0., 1.]])
    d = np.array([.00363725, -.00203717, .000435996, .000625099, 0.])
    for view in views:
        view["intrinsics"] = k
    # Simulate the step-zero legacy seed checkpoint's uncorrected cache.
    trainer.prime_baseline_views(views)
    guards = [v["capture_id"] for v in trainer.images if v.get("validation_only")]
    original = {name: value.detach().clone() for name, value in trainer.params.items()}
    trainer.configure_camera_distortion(d)
    from bridge.online import OnlineError
    with pytest.raises(OnlineError, match="Re-prime"):
        trainer.add_image("new-photo", photo, np.eye(4), k)
    with pytest.raises(OnlineError, match="Re-prime"):
        trainer.train_steps(1)
    with pytest.raises(OnlineError, match="Re-prime"):
        trainer.save(tmp_path / "unsafe.ckpt", tmp_path / "unsafe.ply")
    trainer.prime_baseline_views(views)
    assert [v["capture_id"] for v in trainer.images if v.get("validation_only")] == guards
    expected = np.asarray(Image.fromarray(cv2.undistort(raw, k, d)).resize(
        (128, 64), Image.Resampling.LANCZOS), dtype=np.float32) / 255.
    uncorrected = np.asarray(Image.fromarray(raw).resize(
        (128, 64), Image.Resampling.LANCZOS), dtype=np.float32) / 255.
    assert not np.array_equal(expected, uncorrected)
    for row in trainer.images:
        np.testing.assert_array_equal(row["pixels"].numpy(), expected)
        np.testing.assert_array_equal(row["intrinsics"].numpy(), (np.diag([.5, .5, 1.]) @ k).astype(np.float32))
    trainer._new_points = lambda *args: tuple(np.empty((0, 3), dtype=np.float32) for _ in range(3))
    trainer.add_image("new-photo", photo, np.eye(4), k)
    np.testing.assert_array_equal(trainer.images[-1]["pixels"].numpy(), expected)
    evaluation = []
    trainer._render_cached_l1 = lambda view: evaluation.append(view) or .1
    assert trainer.render_l1(photo, np.eye(4), k) == .1
    np.testing.assert_array_equal(evaluation[0]["pixels"].numpy(), expected)
    checkpoint, current = tmp_path / "state.ckpt", tmp_path / "current.ply"
    trainer.save(checkpoint, current)
    from bridge.online_torch import GsplatTrainer
    resumed = GsplatTrainer(seed, checkpoint, max_image_edge=128)
    np.testing.assert_array_equal(resumed.camera_distortion, d)
    resumed.configure_camera_distortion(d)
    with pytest.raises(OnlineError, match="established camera distortion"):
        resumed.configure_camera_distortion(d + np.array([.0001, 0., 0., 0., 0.]))
    np.testing.assert_array_equal(resumed._scaled_image(photo, k)[0], expected)
    assert resumed.capture_ids == ["new-photo"] and resumed.published_capture_ids == []
    assert not resumed.has_pending_work and resumed.step == 0
    for name, value in resumed.params.items():
        assert resumed.torch.equal(value, original[name])
    plyfile = pytest.importorskip("plyfile")
    assert plyfile.PlyData.read(str(current))["vertex"].data.tobytes() == source.tobytes()
    assert photo.read_bytes() == original_bytes


@pytest.mark.parametrize("invalid", [[0.] * 4, [[0.] * 5], [np.nan, 0., 0., 0., 0.], [3., 0., 0., 0., 0.]])
def test_camera_distortion_rejects_invalid_calibration(invalid):
    from bridge.online import OnlineError
    from bridge.online_torch import GsplatTrainer
    trainer = GsplatTrainer.__new__(GsplatTrainer)
    with pytest.raises(OnlineError, match="Invalid measured camera distortion"):
        trainer.configure_camera_distortion(invalid)


@pytest.mark.parametrize("legacy_state", ["optimized", "admitted"])
def test_legacy_distortion_adoption_refuses_existing_optimization_or_ingress(tmp_path, legacy_state):
    trainer, _, _, _, _, _ = _cuda_seed(tmp_path)
    if legacy_state == "optimized":
        trainer.step = 1
    else:
        trainer.accepted_ids.append("already-admitted")
    from bridge.online import OnlineError
    with pytest.raises(OnlineError, match="must precede"):
        trainer.configure_camera_distortion([.003, 0., 0., 0., 0.])
    assert trainer.camera_distortion is None


def _append_candidate(trainer, photo, k):
    # Synthetic direct geometry isolates optimizer/publication behavior from
    # ORB/PnP, which is exercised by the coordinator and scale-growth tests.
    def candidate(*args):
        trainer.last_candidate_support = {"baseline-1": [0]}
        return (np.array([[0., 0., .5]], dtype=np.float32),
                  np.array([[.4, .4, .4]], dtype=np.float32),
                  np.array([[.02, .02, .02]], dtype=np.float32))
    trainer._new_points = candidate
    trainer.add_image("new-photo", photo, np.eye(4), k)


def test_pending_candidates_resume_without_early_publication_or_freezing(tmp_path):
    trainer, source, seed, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_steps = 7
    checkpoint, current = tmp_path / "state.ckpt", tmp_path / "current.ply"
    metrics = trainer.save(checkpoint, current)
    plyfile = pytest.importorskip("plyfile")
    assert plyfile.PlyData.read(str(current))["vertex"].data.tobytes() == source.tobytes()
    assert metrics["gaussians"] == len(source)
    assert metrics["optimizer_gaussians"] == len(source) + 1
    from bridge.online_torch import GsplatTrainer
    resumed = GsplatTrainer(seed, checkpoint, max_image_edge=128)
    assert resumed.has_pending_work and resumed.candidate_steps == 7
    assert resumed.protected_count == len(source)
    assert resumed.model_units_per_meter == .25
    assert resumed.capture_ids == ["new-photo"]
    assert resumed.published_capture_ids == []
    assert resumed.candidate_capture_ids == ["new-photo"]
    assert resumed.added_total == 0
    assert len(resumed.images) == 4
    assert trainer.torch.equal(resumed.params["means"], trainer.params["means"])
    from bridge.online import OnlineError
    with pytest.raises(OnlineError, match="pending candidate"):
        resumed.add_image("next-photo", photo, np.eye(4), k)
    assert resumed.capture_ids == ["new-photo"] and resumed.candidate_steps == 7
    # Supplying the same frozen baseline again is safe on resume and must not
    # reset pending optimization or change either admission/visible cursor.
    resumed.prime_baseline_views(views)
    assert resumed.candidate_steps == 7 and resumed.capture_ids == ["new-photo"]
    assert resumed.published_capture_ids == []


def test_candidate_gate_rejects_held_out_regression_despite_target_improvement(tmp_path):
    trainer, source, _, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_steps = trainer.maximum_candidate_steps
    # A candidate can explain its source photo while wrongly occluding an
    # independently held-out view. It must be discarded, not frozen/published.
    def score(view, *, published_only=False):
        if published_only:
            return .1
        return .13 if view.get("validation_only") else .07
    trainer._render_cached_l1 = score
    trainer._render_cached_patch_l1 = lambda view, mask, **kwargs: score(view, **kwargs)
    original = {name: value[:len(source)].detach().clone() for name, value in trainer.params.items()}
    metrics = trainer.save(tmp_path / "reject.ckpt", tmp_path / "reject.ply")
    assert not trainer.has_pending_work
    assert metrics["pending_refinement"]["publication_review"]["status"] == "discarded"
    assert metrics["pending_refinement"]["publication_review"]["non_regression"] is False
    assert len(trainer.params["means"]) == len(source)
    assert trainer.capture_ids == ["new-photo"]
    assert trainer.published_capture_ids == [] and trainer.rejected_capture_ids == ["new-photo"]
    assert trainer.added_total == 0
    for name, value in trainer.params.items():
        assert trainer.torch.equal(value, original[name])
    plyfile = pytest.importorskip("plyfile")
    assert plyfile.PlyData.read(str(tmp_path / "reject.ply"))["vertex"].data.tobytes() == source.tobytes()


def test_validated_candidate_freezes_only_after_training_and_improvement(tmp_path):
    trainer, source, _, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer._render_cached_l1 = lambda view, published_only=False: .1 if published_only else .09
    trainer._render_cached_patch_l1 = lambda view, mask, published_only=False: .1 if published_only else .09
    early = trainer.save(tmp_path / "early.ckpt", tmp_path / "early.ply")
    assert early["gaussians"] == len(source) and trainer.has_pending_work
    trainer.candidate_steps = trainer.minimum_candidate_steps
    accepted = trainer.save(tmp_path / "accepted.ckpt", tmp_path / "accepted.ply")
    assert accepted["gaussians"] == len(source) + 1
    assert trainer.published_capture_ids == ["new-photo"] and trainer.added_total == 1
    assert not trainer.has_pending_work and trainer.protected_count == len(source) + 1
    assert accepted["pending_refinement"]["publication_review"]["status"] == "accepted"
    plyfile = pytest.importorskip("plyfile")
    published = plyfile.PlyData.read(str(tmp_path / "accepted.ply"))["vertex"].data
    assert published[:len(source)].tobytes() == source.tobytes()


def test_sparse_patch_gain_cannot_hide_overlapping_validation_regression(tmp_path):
    trainer, source, _, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_steps = trainer.minimum_candidate_steps
    # Whole-frame change is small and every original guard passes. The local
    # triangulating witness regresses like the independently reproduced room.
    trainer._render_cached_l1 = lambda view, published_only=False: .05 if published_only else .04998
    def patch(view, mask, published_only=False):
        if published_only:
            return .05
        return .054 if view["capture_id"] == "baseline-1" else .044
    trainer._render_cached_patch_l1 = patch
    review = trainer._review_candidates()
    assert review["non_regression"] and review["useful_patch"]
    assert review["relevant_validation"] and not review["local_non_regression"]
    assert review["status"] == "quality_rejected"
    assert trainer.candidate_validation_ids == ["baseline-1"]
    metrics = trainer.save(tmp_path/"patch.ckpt", tmp_path/"patch.ply")
    assert metrics["gaussians"] == len(source) and trainer.published_capture_ids == []


def test_sparse_local_gain_can_publish_below_old_whole_frame_threshold(tmp_path):
    trainer, source, _, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_steps = trainer.minimum_candidate_steps
    trainer._render_cached_l1 = lambda view, published_only=False: .05 if published_only else .04998
    trainer._render_cached_patch_l1 = lambda view, mask, published_only=False: .05 if published_only else .044
    review = trainer._review_candidates()
    assert review["new_view_improvement_l1"] < .0001
    assert review["status"] == "accepted" and review["local_non_regression"]
    metrics = trainer.save(tmp_path/"safe.ckpt", tmp_path/"safe.ply")
    assert metrics["gaussians"] == len(source)+1 and trainer.published_capture_ids == ["new-photo"]


def test_unseen_patch_without_independent_observation_stays_unpublished(tmp_path):
    trainer, source, _, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_validation_ids = []
    trainer.candidate_steps = trainer.maximum_candidate_steps
    assert trainer._review_candidates()["status"] == "needs_candidate_support"
    metrics = trainer.save(tmp_path/"unsupported.ckpt", tmp_path/"unsupported.ply")
    assert metrics["gaussians"] == len(source) and not trainer.has_pending_work
    assert trainer.rejected_capture_ids == ["new-photo"]
    assert trainer.published_capture_ids == []


def test_candidate_footprint_and_validation_identity_survive_resume_and_parameter_drift(tmp_path):
    trainer, _, seed, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    target = trainer.images[-1]
    original_mask, original_visible = trainer._candidate_support_mask(target)
    with trainer.torch.no_grad():
        trainer.params["means"][-1, 0] += .001
        trainer.params["log_scales"][-1] -= .1
    checkpoint, current = tmp_path/"fixed.ckpt", tmp_path/"current.ply"
    trainer.save(checkpoint, current)
    from bridge.online_torch import GsplatTrainer
    resumed = GsplatTrainer(seed, checkpoint, max_image_edge=128)
    assert resumed.candidate_validation_ids == ["baseline-1"]
    assert resumed.candidate_observations == {"baseline-1": [0]}
    restored_mask, restored_visible = resumed._candidate_support_mask(resumed.images[-1])
    np.testing.assert_array_equal(restored_mask, original_mask)
    assert restored_visible == original_visible
    resumed.prime_baseline_views(views)
    assert resumed.candidate_validation_ids == ["baseline-1"]
    assert any(v["capture_id"] == "baseline-1" for v in resumed.images)


def test_versioned_rejected_retry_preserves_ingress_and_is_bounded_across_resume(tmp_path):
    trainer, source, seed, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_steps = trainer.maximum_candidate_steps
    trainer._render_cached_l1 = lambda view, published_only=False: .1 if published_only else .2
    trainer._render_cached_patch_l1 = lambda view, mask, published_only=False: .1 if published_only else .2
    checkpoint, current = tmp_path/"retry.ckpt", tmp_path/"current.ply"
    trainer.save(checkpoint, current)
    # This fixture represents the older whole-frame review being upgraded;
    # current-method rejections must never loop automatically.
    trainer.rejected_capture_reviews["new-photo"]["review_method_revision"] = "whole_frame_v0"
    original = {name: value.detach().clone() for name, value in trainer.params.items()}
    trainer.retry_rejected_image("new-photo", photo, np.eye(4), k)
    assert trainer.capture_ids == ["new-photo"] and trainer.published_capture_ids == []
    assert trainer.review_attempts == {"new-photo": trainer.review_method_revision}
    assert len([v for v in trainer.images if v["capture_id"] == "new-photo"]) == 1
    assert trainer.has_pending_work
    trainer.save(checkpoint, current)
    from bridge.online_torch import GsplatTrainer
    from bridge.online import OnlineError
    resumed = GsplatTrainer(seed, checkpoint, max_image_edge=128)
    with pytest.raises(OnlineError, match="bounded review retry"):
        resumed.retry_rejected_image("new-photo", photo, np.eye(4), k)
    assert resumed.capture_ids == ["new-photo"]
    for name, value in resumed.params.items():
        assert trainer.torch.equal(value[:len(source)], original[name])


def test_resume_refuses_changed_published_rotation_or_nonfinite_candidate(tmp_path):
    trainer, _, seed, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    checkpoint, ply = tmp_path / "state.ckpt", tmp_path / "current.ply"
    trainer.save(checkpoint, ply)
    torch = trainer.torch
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original["parameters"]["quats"][0] = torch.tensor([0., 1., 0., 0.])
    torch.save(original, checkpoint)
    from bridge.online import OnlineError
    from bridge.online_torch import GsplatTrainer
    with pytest.raises(OnlineError, match="rotations.*disagree"):
        GsplatTrainer(seed, checkpoint, max_image_edge=128)
    original["parameters"]["quats"][0] = torch.tensor([1., 0., 0., 0.])
    original["parameters"]["means"][-1, 0] = float("nan")
    torch.save(original, checkpoint)
    with pytest.raises(OnlineError, match="invalid Gaussian"):
        GsplatTrainer(seed, checkpoint, max_image_edge=128)


def test_unsupported_seed_sh_layout_cannot_approve_new_geometry(tmp_path):
    trainer, source, seed, photo, k, views = _cuda_seed(tmp_path)
    changed = np.zeros(len(source), dtype=source.dtype.descr + [("f_rest_0", "<f4")])
    for name in source.dtype.names:
        changed[name] = source[name]
    changed["f_rest_0"] = .37
    plyfile = pytest.importorskip("plyfile")
    plyfile.PlyData([plyfile.PlyElement.describe(changed, "vertex")], text=False).write(str(seed))
    from bridge.online_torch import GsplatTrainer
    trainer = GsplatTrainer(seed, max_image_edge=128)
    trainer.configure_geometry_scale(.25)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    trainer.candidate_steps = trainer.maximum_candidate_steps
    trainer._render_cached_l1 = lambda *args, **kwargs: pytest.fail("Unsupported SH must not reach quality rendering")
    result = trainer.save(tmp_path / "unsupported.ckpt", tmp_path / "unsupported.ply")
    assert result["pending_refinement"]["publication_review"]["status"] == "discarded"
    assert "faithfully" in result["pending_refinement"]["publication_review"]["reason"]
    assert trainer.published_capture_ids == []
    assert plyfile.PlyData.read(str(tmp_path / "unsupported.ply"))["vertex"].data.tobytes() == changed.tobytes()


def test_training_excludes_held_out_views_and_remains_bounded_without_new_photos(tmp_path):
    trainer, source, _, photo, k, views = _cuda_seed(tmp_path)
    trainer.prime_baseline_views(views)
    _append_candidate(trainer, photo, k)
    target_positions = []
    torch = trainer.torch
    def differentiable_rasterizer(**kwargs):
        # Camera centers reconstructed from supplied inverse view matrices.
        target_positions.append(float(torch.linalg.inv(kwargs["viewmats"])[0, 0, 3]))
        color = kwargs["colors"].mean(dim=(0, 1)) * .28209479177387814 + .5
        value = color + sum(p.mean() * .001 for p in trainer.params.values())
        return value.reshape(1, 1, 1, 3).expand(1, kwargs["height"], kwargs["width"], 3), None, {}
    trainer.rasterization = differentiable_rasterizer
    original = {name: value[:len(source)].detach().clone() for name, value in trainer.params.items()}
    assert trainer.train_steps(64) == 64
    # Reserved baseline centers are 0 and .1; new camera 0 is permitted, but
    # baseline-2 (.1) must never be selected as an optimization target.
    assert all(not np.isclose(position, .1) and not np.isclose(position, .05) for position in target_positions)
    assert trainer.candidate_steps == 64 and trainer.has_pending_work
    for name, value in trainer.params.items():
        assert torch.equal(value[:len(source)], original[name])
    trainer.candidate_steps = trainer.maximum_candidate_steps - 1
    assert trainer.train_steps(100) == 1
    assert trainer.train_steps(1) == 0
    from bridge.online import OnlineError
    with pytest.raises(OnlineError, match="budget"):
        trainer.train_steps(101)


@pytest.mark.parametrize("degree", [0, 3])
def test_native_cuda_candidate_refines_and_passes_real_held_out_views(tmp_path, degree):
    """Exercise actual gsplat rendering/backpropagation, not a mock renderer."""
    trainer, source, _, photo, k, views = _cuda_seed(tmp_path, degree=degree)
    torch = trainer.torch
    candidate = torch.tensor([[0., 0., .5]], device=trainer.device)
    with torch.no_grad():
        means = torch.cat((trainer.params["means"], candidate), dim=0)
        quats = torch.cat((trainer.params["quats"], torch.tensor([[1., 0., 0., 0.]], device=trainer.device)))
        scales = torch.cat((torch.exp(trainer.params["log_scales"]),
                            torch.full((1, 3), .018, device=trainer.device)))
        new_sh = torch.zeros((1, (degree + 1) ** 2, 3), device=trainer.device)
        new_sh[:, 0] = (torch.tensor([[.8, .7, .6]], device=trainer.device) - .5) / .28209479177387814
        colors = torch.cat((trainer._published_sh, new_sh))
        for index, view in enumerate(views):
            rendered, _, _ = trainer.rasterization(means=means, quats=quats, scales=scales,
                opacities=torch.full((len(source) + 1,), .7, device=trainer.device),
                colors=colors, sh_degree=degree,
                viewmats=torch.linalg.inv(torch.tensor(view["pose"], dtype=torch.float32,
                    device=trainer.device))[None], Ks=torch.tensor(k, dtype=torch.float32,
                    device=trainer.device)[None], width=32, height=32)
            path = tmp_path / f"native-{index}.png"
            Image.fromarray(np.rint(rendered[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(path)
            view["image_path"] = path
    trainer.prime_baseline_views(views)
    def supported_candidate(*args):
        trainer.last_candidate_support = {"baseline-1": [0]}
        return (candidate.cpu().numpy(), np.array([[.4, .4, .4]], dtype=np.float32),
                np.full((1, 3), .018, dtype=np.float32))
    trainer._new_points = supported_candidate
    trainer.add_image("new-native-view", views[0]["image_path"], np.eye(4), k)
    before = {name: value[:len(source)].detach().clone() for name, value in trainer.params.items()}
    old_candidate = trainer.params["color_logits"][-1].detach().clone()
    for _ in range(3):
        trainer.train_steps(64)
        metrics = trainer.save(tmp_path / "native.ckpt", tmp_path / "native.ply")
        if not trainer.has_pending_work:
            break
    assert metrics["pending_refinement"]["publication_review"]["status"] == "accepted"
    assert metrics["gaussians"] == len(source) + 1
    assert not torch.equal(trainer.params["color_logits"][-1], old_candidate)
    for name, value in trainer.params.items():
        assert torch.equal(value[:len(source)], before[name])
    review = metrics["pending_refinement"]["publication_review"]
    assert review["non_regression"] and review["new_view_improvement_l1"] > .0001
