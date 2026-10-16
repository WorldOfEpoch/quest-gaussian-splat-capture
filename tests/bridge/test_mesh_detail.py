"""Room planes must not turn an open door or foreground object into a decal."""
import json

import numpy as np
import pytest

pytest.importorskip('scipy', reason='Optional architectural mesh dependency is unavailable')
from bridge.mesh_room import constrain_room_surfaces


BOUNDS = np.array([[0., 0., 0.], [4., 4., 3.]])


def grid(u, v, location):
    uu, vv = np.meshgrid(u, v, indexing='ij')
    vertices = location(uu.ravel(), vv.ravel())
    a, b = np.meshgrid(np.arange(len(u)-1), np.arange(len(v)-1), indexing='ij')
    first = (a*len(v)+b).ravel()
    faces = np.concatenate((np.column_stack((first, first+len(v), first+1)),
                            np.column_stack((first+1, first+len(v), first+len(v)+1))))
    return vertices, faces


def wall():
    yy = np.linspace(.2, 3.8, 31)
    zz = np.linspace(.2, 2.8, 27)
    vertices, faces = grid(yy, zz, lambda y, z: np.column_stack((
        .014*np.sin(y*7)*np.sin(z*5), y, z)))
    return vertices, faces, yy, zz


def constrain(vertices, faces, points=None, normals=None, **kwargs):
    points = vertices if points is None else points
    normals = np.tile([1., 0., 0.], (len(points), 1)) if normals is None else normals
    return constrain_room_surfaces(vertices, faces, points, normals, np.eye(3), BOUNDS,
                                  max_displacement=.08, support_radius=.09, **kwargs)


def test_noisy_wall_interior_improves_without_new_geometry_or_input_mutation():
    vertices, faces, _, _ = wall()
    saved_vertices, saved_faces = vertices.copy(), faces.copy()
    out, labels, report = constrain(vertices, faces)
    chosen = labels >= 0
    assert chosen.sum() > len(vertices)*.5
    assert np.sqrt(np.mean(out[chosen, 0]**2)) < np.sqrt(np.mean(vertices[chosen, 0]**2))*.1
    assert report['moved_vertices'] > 300
    assert report['faces_added'] == report['faces_removed'] == 0
    assert report['input_faces'] == report['output_faces'] == len(faces)
    assert np.array_equal(vertices, saved_vertices)
    assert np.array_equal(faces, saved_faces)
    assert np.array_equal(out[labels < 0], vertices[labels < 0])
    assert report['max_displacement'] <= .08
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('angle', [10., 25., 45., 90.])
def test_open_door_and_wall_shared_hinge_keep_exact_3d_positions(angle):
    vertices, faces, yy, zz = wall()
    hinge_row = 12
    hinge = vertices[hinge_row*len(zz):(hinge_row+1)*len(zz)]
    opening = np.linspace(0., .7, 13)
    radians = np.deg2rad(angle)
    door, door_faces = grid(opening, zz, lambda u, z: np.column_stack((
        np.interp(z, zz, hinge[:, 0])+np.sin(radians)*u,
        np.full(len(u), yy[hinge_row])+np.cos(radians)*u, z)))
    # The scan is one connected surface: the door shares its hinge vertices
    # with the wall.  Protecting just the door's private vertices is insufficient.
    door_index = np.arange(len(door))+len(vertices)-len(zz)
    door_index[:len(zz)] = np.arange(hinge_row*len(zz), (hinge_row+1)*len(zz))
    combined = np.concatenate((vertices, door[len(zz):]))
    combined_faces = np.concatenate((faces, door_index[door_faces]))
    points = np.concatenate((vertices, door))
    normals = np.concatenate((np.tile([1., 0., 0.], (len(vertices), 1)),
        np.tile([np.cos(radians), -np.sin(radians), 0.], (len(door), 1))))
    out, labels, report = constrain(combined, combined_faces, points, normals)
    assert report['moved_vertices'] > 100  # The wall is improved in the same pass.
    assert np.array_equal(out[door_index], combined[door_index])
    assert np.all(labels[door_index] == -1)
    assert np.ptp(out[door_index, 0]) > .7*np.sin(radians)*.8
    assert np.array_equal(combined_faces[len(faces):], door_index[door_faces])


def test_parallel_raised_object_is_not_flattened_by_a_normal_gate():
    vertices, faces, _, _ = wall()
    raised, raised_faces = grid(np.linspace(.9, 1.9, 29), np.linspace(.8, 1.8, 29),
        lambda y, z: np.column_stack((np.full(len(y), .06), y, z)))
    combined = np.concatenate((vertices, raised))
    combined_faces = np.concatenate((faces, raised_faces+len(vertices)))
    # The underlying wall is occluded here; use the observed front surface,
    # not invented observations behind it, for the local depth test.
    visible_wall = ~((vertices[:, 1] >= .8) & (vertices[:, 1] <= 2.) &
                     (vertices[:, 2] >= .7) & (vertices[:, 2] <= 1.9))
    points = np.concatenate((vertices[visible_wall], raised))
    out, labels, report = constrain(combined, combined_faces, points)
    assert report['coherent_offset_detail_vertices'] >= len(raised)*.8
    assert np.array_equal(out[len(vertices):], raised)
    assert np.all(labels[len(vertices):] == -1)
    assert report['moved_vertices'] > 100


def test_no_observed_depth_and_hole_boundary_are_unchanged():
    vertices, faces, _, _ = wall()
    # Remove a central wall patch from the actual scan; no helper may fill it.
    centroids = vertices[faces].mean(axis=1)
    hole = ((centroids[:, 1] > 1.4) & (centroids[:, 1] < 2.) &
            (centroids[:, 2] > 1.) & (centroids[:, 2] < 1.8))
    faces = faces[~hole]
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundaries = np.unique(unique_edges[counts == 1])
    supported_points = vertices[vertices[:, 1] < 3.]
    out, labels, report = constrain(vertices, faces, supported_points)
    assert np.array_equal(out[boundaries], vertices[boundaries])
    unsupported = vertices[:, 1] > 3.2
    assert np.array_equal(out[unsupported], vertices[unsupported])
    assert np.all(labels[unsupported] == -1)
    assert report['inferred_surfaces'].startswith('none')
    assert report['output_faces'] == len(faces)


def test_reflected_frame_preserves_detail_and_matches_world_frame_result():
    vertices, faces, _, _ = wall()
    expected, expected_labels, _ = constrain(vertices, faces)
    basis = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    world = vertices@basis.T
    normals = np.tile([1., 0., 0.], (len(vertices), 1))@basis.T
    out, labels, report = constrain_room_surfaces(world, faces, world, normals, basis, BOUNDS,
                                                max_displacement=.08, support_radius=.09)
    assert np.allclose(out@basis, expected, atol=1e-12)
    assert np.array_equal(labels, expected_labels)
    assert report['reverted_vertices'] == 0


def test_incompatible_normals_and_unbounded_changes_are_rejected_or_preserved():
    vertices, faces, _, _ = wall()
    out, labels, report = constrain(vertices, faces, normals=np.tile([0., 1., 0.], (len(vertices), 1)))
    assert np.array_equal(out, vertices)
    assert np.all(labels == -1)
    assert report['moved_vertices'] == 0
    with pytest.raises(ValueError, match='budget'):
        constrain_room_surfaces(vertices, faces, vertices, np.ones_like(vertices), np.eye(3), BOUNDS,
                                max_displacement=1., support_radius=.09)
    with pytest.raises(ValueError, match='inputs'):
        constrain_room_surfaces(vertices, faces, vertices, np.ones_like(vertices), np.eye(3)*2, BOUNDS)
