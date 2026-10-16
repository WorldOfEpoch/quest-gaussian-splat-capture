"""Texture and cleanup must not undo preserved door/object geometry."""
import importlib.util
import numpy as np
import pytest
from bridge import mesh
from bridge.reconstruct import ReconstructionError

pytest.importorskip('scipy')


def test_preservation_accepts_uv_splits_reindexing_and_existing_degenerate_face():
    vertices = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.6, .2, .4]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 0, 1]])
    split = vertices[faces[:2].ravel()]
    result = mesh.verify_surface_geometry(vertices, faces, split, np.arange(6).reshape(2, 3))
    assert result['valid_geometric_triangles_preserved'] == 2
    assert result['max_coordinate_difference'] == 0


@pytest.mark.parametrize('change', ['flatten_door', 'remove_door', 'add_wall', 'flip_door'])
def test_preservation_rejects_changed_or_missing_object_surface(change):
    vertices = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.6, .2, .4]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    changed, triangles = vertices.copy(), faces.copy()
    if change == 'flatten_door': changed[3, 0] = 0
    elif change == 'remove_door': triangles = triangles[:1]
    elif change == 'add_wall': triangles = np.concatenate((triangles, [[0, 1, 3]]))
    else: triangles[1] = triangles[1, ::-1]
    with pytest.raises(ReconstructionError, match='protected'):
        mesh.verify_surface_geometry(vertices, faces, changed, triangles)


@pytest.mark.skipif(not importlib.util.find_spec('pymeshlab'), reason='Optional mesh cleanup dependency')
@pytest.mark.parametrize('settings', [{'smooth_steps': 1, 'min_component_faces': 1},
                                     {'smooth_steps': 0, 'min_component_faces': 25}])
def test_finish_rejects_settings_that_undo_detail_protection(monkeypatch, settings):
    monkeypatch.setattr(mesh, 'read_mesh', lambda *args: ({'detail_preserving_room': {'protected_vertices': 5}}, None))
    identity = '00000000-0000-4000-8000-000000000001'
    with pytest.raises(ReconstructionError, match='zero smoothing'):
        mesh.finish_mesh(None, identity, identity, identity, **settings)
