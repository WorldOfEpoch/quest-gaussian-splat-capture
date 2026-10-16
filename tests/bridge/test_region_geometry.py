import uuid
import numpy as np
import pytest
from bridge.online import OnlineError
from bridge.region_edit import validate_region, region_membership, regions_overlap


def region():
    return {"id": str(uuid.uuid4()), "name": "Countertop", "min": [-1., -1., -1.], "max": [1., 1., 1.]}


def test_region_support_protects_boundary_and_locks_new_intrusions():
    box = region()
    points = np.array([[0., 0., 0.], [.99, 0., 0.], [1.1, 0., 0.], [3., 0., 0.]])
    scales = np.full((4, 3), .1)
    assert region_membership(points, scales, box, contained=True).tolist() == [True, False, False, False]
    assert region_membership(points, scales, box).tolist() == [True, True, True, False]
    other = region()
    other.update(min=[1., -1., -1.], max=[2., 1., 1.])
    assert regions_overlap(box, other)


def test_bad_region_or_geometry_fails_closed():
    for bounds in ([float('nan'), 0., 0.], [2., 0., 0.], [0., 0.]):
        with pytest.raises(OnlineError):
            validate_region({**region(), "min": bounds})
    with pytest.raises(OnlineError):
        region_membership(np.zeros((1, 3)), np.full((1, 3), np.inf), region())
