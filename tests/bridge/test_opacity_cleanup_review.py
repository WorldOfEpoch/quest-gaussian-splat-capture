import uuid

import numpy as np
import pytest

from bridge.online import OnlineError
from bridge.region_edit import review_opacity_cleanup


def example():
    names = ("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "f_dc_0")
    original = np.zeros(3, dtype=[(name, "f4") for name in names])
    for i in range(3):
        original[f"scale_{i}"] = -5
    candidate = original.copy()
    candidate["opacity"][0] = -1
    frames = []
    for i in range(2):
        truth = np.full((64, 64, 3), .5, dtype=np.float32)
        before = truth.copy()
        before[:32, :32] += .15
        after = truth.copy()
        after[:32, :32] += .1
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32, :32] = True
        frames.append(dict(capture_id=f"heldout-{i}", camera_center_m=[i*.1, 0, 0],
                           reference=truth, before=before, after=after, support=mask))
    return original, candidate, frames


def review(original, candidate, frames, **kwargs):
    return review_opacity_cleanup(original, candidate, frames,
                                  selection_capture_ids=["training-a", "training-b"], **kwargs)


def test_bounded_improvement_is_only_ready_for_visual_review_and_never_mutates():
    original, candidate, frames = example()
    snapshots = [x.tobytes() for x in (original, candidate)]
    result = review(original, candidate, frames)
    assert result["status"] == "ready_for_visual_review"
    assert result["changed_rows"] == 1
    assert not result["published"]
    assert snapshots == [x.tobytes() for x in (original, candidate)]


@pytest.mark.parametrize("field", ["x", "scale_1", "rot_0", "f_dc_0"])
def test_reject_geometry_and_appearance_edits(field):
    original, candidate, frames = example()
    candidate[field][1] += .01
    with pytest.raises(OnlineError, match="geometry"):
        review(original, candidate, frames)


def test_reject_added_reordered_and_increased_rows():
    original, candidate, frames = example()
    with pytest.raises(OnlineError, match="rows"):
        review(original, candidate[:2], frames)
    candidate["opacity"][1] = 1
    with pytest.raises(OnlineError, match="reduce"):
        review(original, candidate, frames)


def test_lock_protects_footprint_even_when_center_is_outside():
    original, candidate, frames = example()
    lock = dict(id=str(uuid.uuid4()), name="Good trim", min=[.01, -.01, -.01], max=[.1, .1, .1])
    with pytest.raises(OnlineError, match="locked"):
        review(original, candidate, frames, locked_regions=[lock])


def test_training_frames_cannot_be_called_independent_evidence():
    original, candidate, frames = example()
    frames[0]["capture_id"] = "training-a"
    assert review(original, candidate, frames)["status"] == "quality_rejected"


def test_duplicate_camera_and_untranslated_rotation_do_not_count_as_new_evidence():
    original, candidate, frames = example()
    frames[1]["camera_center_m"] = [0, 0, 0]
    assert review(original, candidate, frames)["status"] == "quality_rejected"
    frames[1]["capture_id"] = frames[0]["capture_id"]
    with pytest.raises(OnlineError, match="distinct"):
        review(original, candidate, frames)


def test_global_gain_cannot_hide_local_damage():
    original, candidate, frames = example()
    for f in frames:
        f["support"][:] = True
        f["after"][32:, 32:] += .01
    result = review(original, candidate, frames)
    assert all(v["after_l1"] < v["before_l1"] for v in result["views"])
    assert result["status"] == "quality_rejected"
    assert result["views"][0]["regressing_tiles"]


def test_patch_mask_cannot_hide_damage_elsewhere():
    original, candidate, frames = example()
    frames[0]["after"][48, 48] = .7
    assert review(original, candidate, frames)["status"] == "quality_rejected"


def test_good_pixels_inside_bad_patch_remain_protected():
    original, candidate, frames = example()
    for f in frames:
        f["before"][:4, :4] = .5
        f["after"][:4, :4] = .51
    result = review(original, candidate, frames)
    assert result["status"] == "quality_rejected"
    assert result["views"][0]["good_pixel_l1_increase"] > .001


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1., 2.])
def test_invalid_render_cannot_pass(invalid):
    original, candidate, frames = example()
    frames[0]["after"][0, 0] = invalid
    with pytest.raises(OnlineError, match="finite"):
        review(original, candidate, frames)
