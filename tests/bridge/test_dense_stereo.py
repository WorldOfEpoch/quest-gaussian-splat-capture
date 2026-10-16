"""Physical stereo checks for private area replacement; no live device state."""
import numpy as np
import pytest

from bridge.online import dense_stereo_correspondences


@pytest.mark.parametrize('axis, sign', [(0, 1), (0, -1), (1, 1), (1, -1)])
def test_dense_patch_preserves_metric_depth_and_camera_orientation(axis, sign):
    pytest.importorskip('cv2')
    rng = np.random.default_rng(15)
    image = np.repeat(rng.uniform(0, 1, (320, 320, 1)).astype(np.float32), 3, axis=2)
    other = np.zeros_like(image)
    # A plane at 2 m, fx=320, 0.2 m translation: exactly 32 pixel disparity.
    d = sign * 32
    if axis == 0:
        if d > 0:
            other[:, :-d] = image[:, d:]
        else:
            other[:, -d:] = image[:, :d]
    elif d > 0:
        other[:-d] = image[d:]
    else:
        other[-d:] = image[:d]
    a, b = np.eye(4), np.eye(4)
    b[axis, 3] = sign * .2
    k = np.array([[320., 0., 160.], [0., 320., 160.], [0., 0., 1.]])
    points, uv, other_uv = dense_stereo_correspondences(image, other, a, b, k, k)
    assert len(points) > 100
    assert len(points) == len(uv) == len(other_uv)
    assert np.median(points[:, 2]) == pytest.approx(2., abs=.02)
    assert np.max(np.abs(uv[:, axis] - other_uv[:, axis] - d)) <= 1.
    assert np.isfinite(points).all()


def test_blank_or_unrelated_photos_are_not_replacement_surfaces():
    pytest.importorskip('cv2')
    rng = np.random.default_rng(18)
    a, b = np.eye(4), np.eye(4)
    b[0, 3] = .2
    k = np.array([[320., 0., 160.], [0., 320., 160.], [0., 0., 1.]])
    blank = np.full((320, 320, 3), .5, dtype=np.float32)
    noise = rng.uniform(0, 1, blank.shape).astype(np.float32)
    for left, right in ((blank, blank), (blank, noise), (noise, blank)):
        assert not len(dense_stereo_correspondences(left, right, a, b, k, k)[0])


def test_baseline_invalid_calibration_and_match_budget_are_rejected():
    pytest.importorskip('cv2')
    image = np.ones((128, 128, 3), dtype=np.float32)
    a, b = np.eye(4), np.eye(4)
    b[0, 3] = .01
    k = np.array([[128., 0., 64.], [0., 128., 64.], [0., 0., 1.]])
    assert not len(dense_stereo_correspondences(image, image, a, b, k, k)[0])
    b[0, 3] = .2
    for bad_k in (k * np.nan, np.zeros((3, 3))):
        assert not len(dense_stereo_correspondences(image, image, a, b, bad_k, k)[0])
    assert not len(dense_stereo_correspondences(image, image, a, b, k, k, max_matches=50000)[0])
