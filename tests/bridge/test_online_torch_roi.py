"""CUDA regression for explicit, bounded reviewed-region seed refinement."""

import numpy as np
import pytest
from PIL import Image


def test_reviewed_roi_keeps_other_seed_rows_and_source_extras(tmp_path):
    torch = pytest.importorskip("torch")
    plyfile = pytest.importorskip("plyfile")
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA gsplat worker is unavailable")
    from bridge.online_torch import GsplatTrainer

    names = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
             "f_rest_0")
    vertex = np.zeros(4, dtype=[(name, "<f4") for name in names])
    vertex["x"] = [0., 1., -1., 1.]
    vertex["y"] = [0., 0., 0., 1.]
    vertex["z"] = 2.
    vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = -4.
    vertex["rot_0"] = 1.
    vertex["f_rest_0"] = [.37, .21, .13, .07]
    seed = tmp_path / "seed.ply"
    plyfile.PlyData([plyfile.PlyElement.describe(vertex, "vertex")], text=False).write(str(seed))
    trainer = GsplatTrainer(seed, max_image_edge=128)
    k = np.array([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]])
    first, second = np.eye(4), np.eye(4)
    second[0, 3] = .1
    away = np.diag([-1., 1., -1., 1.])
    views = [
        {"pose": pose, "intrinsics": k, "size": (128, 128),
         "rect": (40., 40., 85., 85.), "role": "edit"}
        for pose in (first, second)
    ] + [{"pose": away, "intrinsics": k, "size": (128, 128), "role": "protect"}]
    assert trainer.configure_reviewed_roi(views, max_seed_rows=2) == 1
    assert torch.where(trainer.roi_editable_rows)[0].tolist() == [0]
    originals = {name: value.detach().clone() for name, value in trainer.params.items()}

    photo = tmp_path / "corner.png"
    Image.new("RGB", (128, 128), (220, 30, 20)).save(photo)
    trainer.add_image("corner", photo, first, k)

    def differentiable_rasterizer(**kwargs):
        means2d = trainer.params["means"][:, :2].unsqueeze(0)
        rgb = torch.sigmoid(trainer.params["color_logits"].mean(dim=0) +
                            means2d.mean() * .5)
        rendered = rgb.reshape(1, 1, 1, 3).expand(1, kwargs["height"], kwargs["width"], 3)
        count = len(trainer.params["means"])
        return rendered, None, {"means2d": means2d,
                                "gaussian_ids": torch.arange(count, device=trainer.device),
                                "radii": torch.ones(count, device=trainer.device)}

    trainer.rasterization = differentiable_rasterizer
    trainer.train_steps(4)
    for name, parameter in trainer.params.items():
        assert torch.equal(parameter[1:trainer.seed_count], originals[name][1:]), name
    assert not torch.equal(trainer.params["color_logits"][0], originals["color_logits"][0])
    assert trainer.split_reviewed_roi(max_children=1, min_observations=2) == 1
    assert len(trainer.params["means"]) == trainer.seed_count + 1
    assert trainer.roi_editable_rows.tolist() == [True, False, False, False, True]
    trainer.train_steps(2)
    ckpt, ply = tmp_path / "state.ckpt", tmp_path / "current.ply"
    trainer.save(ckpt, ply)
    saved = plyfile.PlyData.read(str(ply))["vertex"].data
    for name in names:
        assert np.array_equal(saved[name][1:trainer.seed_count], vertex[name][1:]), name
    assert saved["f_rest_0"][0] == vertex["f_rest_0"][0]
    assert saved["f_rest_0"][-1] == 0.
    resumed = GsplatTrainer(seed, ckpt, max_image_edge=128)
    assert resumed.roi_editable_rows.tolist() == [True, False, False, False, True]
    assert resumed.roi_children_total == 1
    for name, parameter in resumed.params.items():
        assert torch.equal(parameter[1:resumed.seed_count], originals[name][1:]), name
        assert torch.equal(parameter[0], trainer.params[name][0]), name
