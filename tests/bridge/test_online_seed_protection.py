"""CUDA-only regression for immutable original-scan Gaussians."""

import numpy as np
import pytest
from PIL import Image


def test_new_view_cannot_move_seed_or_retain_seed_adam_momentum(tmp_path):
    torch = pytest.importorskip("torch")
    plyfile = pytest.importorskip("plyfile")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA gsplat worker is unavailable in this test environment")
    from bridge.online_torch import GsplatTrainer

    names = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
             "f_rest_0")
    vertex = np.zeros(4, dtype=[(name, "<f4") for name in names])
    vertex["x"] = [-.1, .1, -.1, .1]
    vertex["y"] = [-.1, -.1, .1, .1]
    vertex["z"] = 1.
    vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = -2.
    vertex["rot_0"] = 1.
    vertex["f_dc_0"] = 2.  # Outside display RGB; export must keep source coefficient.
    vertex["f_rest_0"] = .37
    seed = tmp_path / "seed.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(vertex, "vertex")], text=False).write(str(seed))
    photo = tmp_path / "new.png"
    Image.new("RGB", (32, 32), (230, 40, 120)).save(photo)
    trainer = GsplatTrainer(seed, max_image_edge=128)
    trainer.add_image("new-view", photo, np.eye(4), np.array([[28., 0., 16.],
                                                               [0., 28., 16.], [0., 0., 1.]]))
    trainer._grow(np.array([[0., 0., 1.]], dtype=np.float32),
                  np.array([[.1, .1, .1]], dtype=np.float32),
                  np.array([[.1, .1, .1]], dtype=np.float32))
    originals = {name: value[:trainer.seed_count].detach().clone()
                 for name, value in trainer.params.items()}
    new_color = trainer.params["color_logits"][-1].detach().clone()

    # Controlled differentiable rasterizer: both seed and new rows receive
    # nonzero gradients. This isolates the seed/Adam barrier from GPU kernels.
    def differentiable_rasterizer(**kwargs):
        all_params = sum(parameter.mean() * .01 for parameter in trainer.params.values())
        color = torch.sigmoid(trainer.params["color_logits"].mean(dim=0) + all_params)
        image = color.reshape(1, 1, 1, 3).expand(1, kwargs["height"], kwargs["width"], 3)
        return image, None, {}

    trainer.rasterization = differentiable_rasterizer
    trainer.train_steps(1)
    for parameter in trainer.params.values():
        with torch.no_grad():
            parameter[:trainer.seed_count].add_(0.5)
        state = trainer.optimizer.state[parameter]
        state["exp_avg"][:trainer.seed_count].fill_(1.)
        state["exp_avg_sq"][:trainer.seed_count].fill_(1.)
    trainer.train_steps(1)
    for name, parameter in trainer.params.items():
        assert torch.equal(parameter[:trainer.seed_count], originals[name]), name
        state = trainer.optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][:trainer.seed_count]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][:trainer.seed_count]) == 0
    assert not torch.equal(trainer.params["color_logits"][-1], new_color)

    # A subsequent photo may train its newly triangulated points, but cannot
    # rewrite a Gaussian published by the previous photo.
    published_new = {name: value[-1].detach().clone()
                     for name, value in trainer.params.items()}
    trainer._protect_existing()
    trainer._grow(np.array([[.2, 0., 1.]], dtype=np.float32),
                  np.array([[.8, .2, .2]], dtype=np.float32),
                  np.array([[.1, .1, .1]], dtype=np.float32))
    trainer.train_steps(2)
    for name, parameter in trainer.params.items():
        assert torch.equal(parameter[trainer.seed_count], published_new[name]), name

    checkpoint, current = tmp_path / "state.ckpt", tmp_path / "current.ply"
    trainer.save(checkpoint, current)
    from bridge.reconstruct import validate_gaussian_ply
    # Two steps without held-out baseline evidence must not publish either
    # addition. The checkpoint retains their mutable state for later review.
    assert validate_gaussian_ply(current)[0] == trainer.seed_count
    saved = plyfile.PlyData.read(str(current))["vertex"].data
    for name in names:
        assert np.array_equal(saved[name][:trainer.seed_count], vertex[name]), name
    resumed = GsplatTrainer(seed, checkpoint, max_image_edge=128)
    assert resumed.capture_ids == ["new-view"]
    assert resumed.has_pending_work
    assert resumed.protected_count == resumed.seed_count
    assert len(resumed.params["means"]) == resumed.seed_count + 2
    assert resumed.candidate_steps == 2
    for name, parameter in resumed.params.items():
        assert torch.equal(parameter[:resumed.seed_count], originals[name]), name
        state = resumed.optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][:resumed.seed_count]) == 0


def test_resume_preserves_published_trained_rows_and_ply_schema(tmp_path):
    """Legacy Home checkpoints must resume from the displayed revision, not the older seed."""
    torch = pytest.importorskip("torch")
    plyfile = pytest.importorskip("plyfile")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA gsplat worker is unavailable in this test environment")
    from bridge.online_torch import GsplatTrainer

    fields = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    source = np.zeros(4, dtype=[(name, "<f4") for name in (*fields, "f_rest_0")])
    source["x"] = [-.1, .1, -.1, .1]
    source["z"] = 1.
    source["rot_0"] = 1.
    source["f_rest_0"] = .4
    seed = tmp_path / "seed.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(source, "vertex")], text=False).write(str(seed))
    trainer = GsplatTrainer(seed, max_image_edge=128)
    checkpoint, initial = tmp_path / "state.ckpt", tmp_path / "initial.ply"
    trainer.save(checkpoint, initial)

    # Simulate the already-published pre-protection worker: it optimized seed
    # geometry, then exported a reduced PLY without the source's SH extras.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["parameters"]["means"][0, 0] = .35
    payload["parameters"]["opacity_logits"][0] = .7
    torch.save(payload, checkpoint)
    published = np.zeros(4, dtype=[(name, "<f4") for name in fields])
    for name in fields:
        published[name] = source[name]
    published["x"][0] = .35
    published["opacity"][0] = .7
    current = tmp_path / "current.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(published, "vertex")], text=False).write(str(current))

    resumed = GsplatTrainer(seed, checkpoint, published_ply=current, max_image_edge=128)
    assert resumed.params["means"][0, 0].item() == pytest.approx(.35)
    assert resumed.params["opacity_logits"][0].item() == pytest.approx(.7)
    rewritten = tmp_path / "rewritten.ply"
    resumed.save(tmp_path / "rewritten.ckpt", rewritten)
    actual = plyfile.PlyData.read(str(rewritten))["vertex"].data
    assert actual.dtype.names == published.dtype.names
    for name in fields:
        assert np.array_equal(actual[name], published[name]), name

    # If a crash left a newer PLY beside an older checkpoint, fail before
    # training rather than silently merging two different model revisions.
    from bridge.online import OnlineError
    published["x"][1] += .2
    plyfile.PlyData([plyfile.PlyElement.describe(published, "vertex")], text=False).write(str(current))
    try:
        GsplatTrainer(seed, checkpoint, published_ply=current, max_image_edge=128)
    except OnlineError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("Mismatched checkpoint and published PLY were accepted")


def _cpu_reviewed_seed(units):
    """Exercise the real optimizer/ROI owner on CPU without a CUDA allocation."""
    torch = pytest.importorskip("torch")
    from bridge.online_torch import GsplatTrainer
    trainer = GsplatTrainer.__new__(GsplatTrainer)
    trainer.torch, trainer.device = torch, torch.device("cpu")
    trainer.seed_count = trainer.protected_count = 4
    trainer.model_units_per_meter = units
    trainer.params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor([[0., 0., 2.], [1., 0., 2.],
            [-1., 0., 2.], [1., 1., 2.]])*units),
        "log_scales": torch.nn.Parameter(torch.full((4, 3), np.log(.012*units))),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]*4)),
        "opacity_logits": torch.nn.Parameter(torch.zeros(4)),
        "color_logits": torch.nn.Parameter(torch.zeros(4, 3))})
    trainer.optimizer = torch.optim.Adam(trainer.params.parameters(), lr=1.)
    trainer.candidate_anchors = trainer.params["means"].detach().clone()
    trainer.roi_editable_rows = torch.zeros(4, dtype=torch.bool)
    trainer.removed_rows = torch.zeros(4, dtype=torch.bool)
    trainer.roi_gradient_sum = torch.zeros(4)
    trainer.roi_gradient_count = torch.zeros(4)
    trainer.roi_children_total = 0
    trainer.locked_regions, trainer.active_replacement = [], None
    trainer._seed_parameters = {name: value.detach().clone() for name, value in trainer.params.items()}
    trainer._protect_existing()
    trainer.images = [{"capture_id": "training", "pixels": torch.zeros(8, 8, 3),
        "pose": torch.eye(4), "intrinsics": torch.eye(3)}]
    trainer.pending_newest, trainer.step = False, 0
    trainer._camera_cache_requires_prime = False
    trainer.candidate_steps = 0
    trainer.minimum_candidate_steps, trainer.maximum_candidate_steps = 64, 192
    trainer.candidate_validation_ids, trainer.candidate_training_ids = [], []
    trainer.candidate_observations, trainer.candidate_capture_ids = {}, []
    trainer.candidate_initial_scales = np.empty((0, 3), dtype=np.float32)
    trainer.sh_degree = None

    def differentiable_rasterizer(**kwargs):
        projected = trainer.params["means"][:, :2].unsqueeze(0)
        value = torch.sigmoid(projected.mean()+trainer.params["color_logits"].mean())
        rendered = value.expand(1, kwargs["height"], kwargs["width"], 3)
        return rendered, None, {"means2d": projected,
            "radii": torch.ones(1, len(trainer.params["means"]), 2)}

    trainer.rasterization = differentiable_rasterizer
    k = np.array([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]])
    second = np.eye(4); second[0, 3] = .1*units
    views = [{"pose": pose, "intrinsics": k, "size": (128, 128),
              "rect": (40., 40., 85., 85.), "role": "edit"} for pose in (np.eye(4), second)]
    views.append({"pose": np.diag([-1., 1., -1., 1.]), "intrinsics": k,
                  "size": (128, 128), "role": "protect"})
    return trainer, views


@pytest.mark.parametrize("units", [.025, 1., 4.])
def test_cpu_reviewed_seed_optimizer_cannot_walk_beyond_physical_bound(units):
    trainer, views = _cpu_reviewed_seed(units)
    torch = trainer.torch
    # A previous approved position must become this review's fixed origin;
    # an older seed coordinate cannot be silently reused as the new anchor.
    with torch.no_grad():
        trainer.params["means"][0, 0] = .05*units
    assert trainer.configure_reviewed_roi(views, max_seed_rows=1) == 1
    anchors = trainer.candidate_anchors.clone()
    original = {name: value.detach().clone() for name, value in trainer.params.items()}
    trainer.train_steps(8)
    drift = torch.linalg.vector_norm(trainer.params["means"][0]-anchors[0])/units
    assert drift.item() == pytest.approx(.03, abs=1e-6)
    assert torch.equal(trainer.candidate_anchors, anchors)
    for name, value in trainer.params.items():
        assert torch.equal(value[1:], original[name][1:]), name
    state = trainer.optimizer.state[trainer.params["means"]]
    assert torch.count_nonzero(state["exp_avg"][0]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"][0]) == 0
    trainer._validate_reviewed_anchor_bound()
    # The old unrestricted optimizer would have moved this row by >1m.
    with torch.no_grad():
        trainer.params["means"][0, 0] += .1*units
    from bridge.online import OnlineError
    with pytest.raises(OnlineError, match="fixed physical refinement bound"):
        trainer._validate_reviewed_anchor_bound()


def test_cpu_reviewed_seed_bound_runs_before_stricter_region_rollback():
    trainer, views = _cpu_reviewed_seed(1.)
    torch = trainer.torch
    trainer.configure_reviewed_roi(views, max_seed_rows=1)
    original = trainer.params["means"].detach().clone()
    # The explicit replacement/lock contract may be tighter than 3cm. Its
    # existing rollback must still reject a clamped step outside that volume.
    import uuid
    trainer.locked_regions = [{"id": str(uuid.uuid4()), "name": "Far protected area",
                              "min": [9., 9., 9.], "max": [10., 10., 10.]}]
    trainer._allowed_growth = lambda points, scales: np.zeros(len(points), dtype=bool)
    trainer.train_steps(2)
    assert torch.equal(trainer.params["means"], original)
    assert torch.equal(trainer.candidate_anchors, original)


def test_cpu_reviewed_split_inherits_fixed_origin_instead_of_resetting_bound():
    pytest.importorskip("gsplat")
    trainer, views = _cpu_reviewed_seed(1.)
    torch = trainer.torch
    trainer.configure_reviewed_roi(views, max_seed_rows=1)
    anchors = trainer.candidate_anchors.clone()
    original = {name: value.detach().clone() for name, value in trainer.params.items()}
    # A parent near its total movement limit must not give its child a fresh
    # three centimeter allowance at the already displaced position.
    with torch.no_grad():
        trainer.params["means"][0, 0] = .029
        trainer.roi_gradient_sum[0] = .01
        trainer.roi_gradient_count[0] = 4
    assert trainer.split_reviewed_roi(max_children=1, min_observations=2) == 1
    assert torch.equal(trainer.candidate_anchors[:4], anchors)
    assert torch.equal(trainer.candidate_anchors[4], anchors[0])
    trainer.train_steps(4)
    for row in (0, 4):
        drift = torch.linalg.vector_norm(trainer.params["means"][row]-anchors[0])
        assert drift.item() <= .030001
    for name, value in trainer.params.items():
        assert torch.equal(value[1:4], original[name][1:]), name
    trainer._validate_reviewed_anchor_bound()


def test_reviewed_seed_anchor_is_persisted_and_not_reset_on_cuda_resume(tmp_path):
    torch = pytest.importorskip("torch")
    plyfile = pytest.importorskip("plyfile")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA trainer constructor is unavailable")
    from bridge.online_torch import GsplatTrainer
    from bridge.online import OnlineError
    cpu, views = _cpu_reviewed_seed(.25)
    names = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    vertices = np.zeros(4, dtype=[(name, "<f4") for name in names])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = cpu.params["means"].detach().numpy()[:, axis]
    for axis in range(3):
        vertices[f"scale_{axis}"] = np.log(.003)
    vertices["rot_0"] = 1.
    seed = tmp_path/"seed.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(vertices, "vertex")], text=False).write(str(seed))
    trainer = GsplatTrainer(seed, max_image_edge=128)
    trainer.configure_geometry_scale(.25)
    trainer.configure_reviewed_roi(views, max_seed_rows=1)
    anchor = trainer.candidate_anchors.detach().cpu().clone()
    with torch.no_grad():
        trainer.params["means"][0, 0] += .006
    ckpt, current = tmp_path/"state.ckpt", tmp_path/"current.ply"
    trainer.save(ckpt, current)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["roi_anchor_guard_version"] == 1
    assert torch.equal(payload["candidate_anchors"], anchor)
    resumed = GsplatTrainer(seed, ckpt, published_ply=current, max_image_edge=128)
    assert torch.equal(resumed.candidate_anchors.cpu(), anchor)
    with torch.no_grad():
        resumed.params["means"][0, 0] += .1
    resumed._bound_reviewed_positions()
    assert torch.linalg.vector_norm(resumed.params["means"][0].cpu()-anchor[0]).item() == pytest.approx(.0075, abs=1e-6)
    # A legacy unbounded area edit has no trustworthy guard proof. Do not
    # launder its already moved published point into a new resume origin.
    payload.pop("roi_anchor_guard_version")
    torch.save(payload, ckpt)
    with pytest.raises(OnlineError, match="fixed refinement anchors"):
        GsplatTrainer(seed, ckpt, published_ply=current, max_image_edge=128)
