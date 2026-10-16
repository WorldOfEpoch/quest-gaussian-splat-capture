import importlib.util
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import json
import uuid

import numpy as np
from PIL import Image

from bridge.mesh import (ReconstructionError, constrain_supported_planes, fit_room_axes, _reuse_photo_depth,
                         _native_map_valid, _inspect_depth_maps, _require_mesh_space, resume_mesh)


@unittest.skipUnless(importlib.util.find_spec('scipy'), 'Optional plane neighbor search unavailable')
class SupportedPlaneTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        xy = np.array(np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))).reshape(2, -1).T
        self.points = np.column_stack((xy, rng.normal(0, .003, len(xy))))
        self.normals = np.tile([0., 0., 1.], (len(xy), 1))
        grid = np.array(np.meshgrid(np.linspace(-.9, .9, 40), np.linspace(-.9, .9, 40))).reshape(2, -1).T
        self.vertices = np.column_stack((grid, .02*np.sin(grid[:, 0]*10)*np.sin(grid[:, 1]*10)))
        self.faces = np.array([[y*40+x, y*40+x+1, (y+1)*40+x] for y in range(39) for x in range(39)] +
                              [[y*40+x+1, (y+1)*40+x+1, (y+1)*40+x] for y in range(39) for x in range(39)])
        self.boxes = [{'center': [0, 0, 0], 'rotation': np.eye(3).tolist(), 'extent': [2, 2, .02]}]

    def apply(self, v=None, t=None, **kwargs):
        return constrain_supported_planes(self.vertices if v is None else v, self.faces if t is None else t,
                    self.points, self.normals, self.boxes, .05, .04, min_support=500, **kwargs)

    def test_noisy_supported_wall_becomes_flat_without_changing_faces_or_input(self):
        original = self.vertices.copy()
        result, labels, report = self.apply()
        np.testing.assert_array_equal(self.vertices, original)
        self.assertLess(np.std(result[:, 2]), np.std(original[:, 2])*.05)
        self.assertGreater(report['moved_vertices'], 1500)
        self.assertEqual(report['faces_added'], 0)
        self.assertEqual(report['faces_removed'], 0)
        self.assertEqual(report['patches'][0]['held_out_points'], len(self.points)//2)

    def test_raised_object_and_unobserved_patch_are_unchanged(self):
        object_points = np.array([[.2, .2, .15], [.3, .2, .15], [.2, .3, .15]])
        distant_points = np.array([[2.2, 0, .02], [2.3, 0, .02], [2.2, .1, .02]])
        v = np.vstack((self.vertices, object_points, distant_points))
        t = np.vstack((self.faces, [[1600, 1601, 1602], [1603, 1604, 1605]]))
        result, labels, report = self.apply(v, t)
        np.testing.assert_array_equal(result[-6:], v[-6:])
        np.testing.assert_array_equal(labels[-6:], [-1]*6)

    def test_normal_conflict_prevents_flattening_curved_or_perpendicular_detail(self):
        self.normals[:] = [1, 0, 0]
        result, labels, report = self.apply()
        np.testing.assert_array_equal(result, self.vertices)
        self.assertEqual(report['moved_vertices'], 0)

    def test_missing_local_observations_do_not_flatten_a_void(self):
        keep = np.linalg.norm(self.points[:, :2], axis=1) > .45
        self.points, self.normals = self.points[keep], self.normals[keep]
        result, labels, report = self.apply()
        void = np.linalg.norm(self.vertices[:, :2], axis=1) < .35
        np.testing.assert_array_equal(result[void], self.vertices[void])

    def test_invalid_or_unbounded_plane_request_is_rejected(self):
        self.boxes[0]['rotation'][0][0] = 2
        with self.assertRaises(ReconstructionError): self.apply()

    def test_square_room_axes_are_perpendicular_and_opposing_normals_parallel(self):
        boxes = []
        for n, area in [([1, .02, 0], 12), ([.01, 1, .015], 10), ([.02, 0, 1], 9), ([-1, 0, 0], 8)]:
            n = np.asarray(n, float); n /= np.linalg.norm(n)
            tangent = np.cross(n, [0, 0, 1] if abs(n[2]) < .8 else [0, 1, 0])
            tangent /= np.linalg.norm(tangent)
            boxes.append({'rotation': np.column_stack((tangent, np.cross(n, tangent), n)).tolist(),
                          'extent': [area, 1, .02]})
        axes, report = fit_room_axes(boxes)
        np.testing.assert_allclose(axes.T@axes, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(axes), 1)
        self.assertEqual(report['corner_angle_degrees'], 90)
        self.assertIn('not constrained', report['equal_wall_lengths'])

    def test_no_room_box_is_invented_from_only_one_observed_wall(self):
        with self.assertRaises(ReconstructionError): fit_room_axes(self.boxes)

    def test_known_room_prior_flattens_a_supported_patch_to_the_shared_axes(self):
        result, labels, report = self.apply(room_axes=np.eye(3))
        self.assertEqual(report['patches'][0]['room_axis'], 2)
        self.assertLess(np.std(result[:, 2]), .001)


class RetainedPhotoDepthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root/'old/dense'; self.new = self.root/'new/dense'
        for directory in (self.old, self.new):
            for child in ('sparse', 'images', 'stereo/depth_maps', 'stereo/normal_maps'):
                (directory/child).mkdir(parents=True)
            (directory/'sparse/cameras.bin').write_bytes(b'calibrated-camera')
            Image.new('RGB', (2, 2), (100, 80, 60)).save(directory/'images/frame.jpg')
        for category, channels in [('depth_maps', 1), ('normal_maps', 3)]:
            (self.old/'stereo'/category/'frame.jpg.photometric.bin').write_bytes(
                f'2&2&{channels}&'.encode()+np.ones(4*channels, dtype='<f4').tobytes())
            (self.old/'stereo'/category/'frame.jpg.geometric.bin').write_bytes(b'do-not-reuse')
        self.pose = {'frame.jpg': (np.array([1., 0, 0, 0]), np.zeros(3))}

    def reuse(self, new_pose=None):
        with patch('bridge.mesh._run'), patch('bridge.reconstruct._read_colmap_images',
                 side_effect=[self.pose, self.pose if new_pose is None else new_pose]):
            return _reuse_photo_depth({'private_work_directory': str(self.old.parent),
                                      'revision_id': 'retained'}, self.new, 'colmap', self.root, {})

    def test_verified_photometric_maps_reused_and_geometric_maps_recomputed(self):
        report = self.reuse()
        self.assertEqual(report['reused_photometric_images'], 1)
        self.assertEqual(report['geometric_maps_reused'], 0)
        for category in ('depth_maps', 'normal_maps'):
            target = self.new/'stereo'/category/'frame.jpg.photometric.bin'
            self.assertEqual(target.read_bytes(), (self.old/'stereo'/category/target.name).read_bytes())
            self.assertFalse((target.parent/'frame.jpg.geometric.bin').exists())
            self.assertEqual(len(report['verified_maps'][0]['maps'][category]), 64)

    def test_changed_camera_calibration_rejects_reuse(self):
        (self.new/'sparse/cameras.bin').write_bytes(b'changed-camera')
        with self.assertRaisesRegex(ReconstructionError, 'calibration differs'): self.reuse()

    def test_changed_photo_pixels_reject_reuse(self):
        Image.new('RGB', (2, 2), 'white').save(self.new/'images/frame.jpg')
        with self.assertRaisesRegex(ReconstructionError, 'pixels differ'): self.reuse()

    def test_moved_camera_rejects_reuse(self):
        changed = {'frame.jpg': (self.pose['frame.jpg'][0], np.array([.01, 0, 0]))}
        with self.assertRaisesRegex(ReconstructionError, 'pose moved'): self.reuse(changed)

    def test_truncated_map_rejects_reuse(self):
        (self.old/'stereo/depth_maps/frame.jpg.photometric.bin').write_bytes(b'2&2&1&short')
        with self.assertRaisesRegex(ReconstructionError, 'map is invalid'): self.reuse()

    def test_nonfinite_depth_rejects_reuse(self):
        (self.old/'stereo/depth_maps/frame.jpg.photometric.bin').write_bytes(
            b'2&2&1&'+np.array([1, 2, float('nan'), 3], dtype='<f4').tobytes())
        with self.assertRaisesRegex(ReconstructionError, 'map is invalid'): self.reuse()

    def test_pre_fusion_check_finds_bad_geometric_maps_without_rejecting_valid_photometric_maps(self):
        invalid = _inspect_depth_maps(self.old)
        self.assertEqual({p.name for p in invalid}, {'frame.jpg.geometric.bin'})
        self.assertEqual(len(invalid), 2)
        self.assertTrue(_native_map_valid(self.old/'stereo/normal_maps/frame.jpg.photometric.bin', 2, 2, 3))
        self.assertFalse(_native_map_valid(self.old/'stereo/normal_maps/frame.jpg.photometric.bin', 2, 2, 1))

    def test_space_shortage_fails_before_native_processing(self):
        with patch('bridge.mesh.shutil.disk_usage', return_value=SimpleNamespace(free=2**30)):
            with self.assertRaisesRegex(ReconstructionError, 'needs 2.0 GiB free'):
                _require_mesh_space(self.root, 2*2**30)

    def test_resume_cannot_overwrite_a_completed_mesh(self):
        project, job, revision = [str(uuid.uuid4()) for _ in range(3)]
        directory = self.root/'projects'/project/'reconstructions'/job/'mesh-revisions'/revision
        directory.mkdir(parents=True)
        (directory/'mesh.json').write_text(json.dumps({'project_id': project, 'source_job_id': job,
                                                     'status': 'ready', 'artifact': {'sha256': 'retained'}}))
        with self.assertRaisesRegex(ReconstructionError, 'unfinished failed mesh'):
            resume_mesh(SimpleNamespace(root=self.root), project, job, revision)


if __name__ == '__main__':
    unittest.main()
