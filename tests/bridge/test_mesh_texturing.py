"""Portable texture packaging invariants; run in the bridge's mesh-extra environment.

python -m unittest discover -s tests/bridge -p test_mesh_texturing.py
"""
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from bridge.mesh import ReconstructionError, textured_mesh_to_glb


@unittest.skipUnless(importlib.util.find_spec('plyfile'), 'Optional mesh texture PLY reader is unavailable')
class TexturePackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'textured.ply'
        self.source.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n'
                               'property float y\nproperty float z\nelement face 1\n'
                               'property list uchar int vertex_indices\nproperty list uchar float texcoord\n'
                               'end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2 6 0 0 1 0 0 1\n')
        self.texture = self.root/'texture.png'
        Image.new('RGB', (2, 2), (10, 120, 220)).save(self.texture)

    def test_texture_uvs_and_coordinate_conversion_are_preserved(self):
        before = self.source.read_bytes()
        transform = np.eye(4); transform[:3, 3] = [2, 3, 4]
        result = textured_mesh_to_glb(self.source, self.texture, self.root/'mesh.glb', {'test': True}, transform)
        data = (self.root/'mesh.glb').read_bytes()
        js_size = struct.unpack_from('<I', data, 12)[0]
        doc = json.loads(data[20:20+js_size]); binary = data[28+js_size:]
        def buffer(index, dtype, shape):
            v = doc['bufferViews'][index]
            return np.frombuffer(binary, dtype=dtype, count=np.prod(shape), offset=v['byteOffset']).reshape(shape)
        np.testing.assert_allclose(buffer(0, '<f4', (3, 3)), [[2, 4, -3], [3, 4, -3], [2, 4, -4]])
        np.testing.assert_allclose(buffer(2, '<f4', (3, 2)), [[0, 1], [1, 1], [0, 0]])
        np.testing.assert_allclose(buffer(1, '<f4', (3, 3)), [[0, 1, 0]]*3)
        png_view = doc['bufferViews'][6]
        self.assertEqual(binary[png_view['byteOffset']:png_view['byteOffset']+png_view['byteLength']], self.texture.read_bytes())
        self.assertEqual(result['triangle_count'], 1)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertTrue(result['embedded_texture'])

    def test_bad_uv_is_rejected_without_a_deliverable(self):
        self.source.write_text(self.source.read_text().replace('2 6 0 0', '2 6 2 0'))
        with self.assertRaises(ReconstructionError):
            textured_mesh_to_glb(self.source, self.texture, self.root/'bad.glb', {}, np.eye(4))
        self.assertFalse((self.root/'bad.glb').exists())

    def test_nonfinite_transform_is_rejected(self):
        transform = np.eye(4); transform[0, 0] = np.nan
        with self.assertRaises(ReconstructionError):
            textured_mesh_to_glb(self.source, self.texture, self.root/'bad.glb', {}, transform)

    def test_previous_delivery_is_not_overwritten(self):
        target = self.root/'mesh.glb'; target.write_bytes(b'previous accepted mesh')
        with self.assertRaises(FileExistsError):
            textured_mesh_to_glb(self.source, self.texture, target, {}, np.eye(4))
        self.assertEqual(target.read_bytes(), b'previous accepted mesh')

    def test_faces_without_atlas_uvs_keep_observed_geometry_and_photo_colors(self):
        self.source.write_text(self.source.read_text().replace('6 0 0 1 0 0 1', '6 0 0 0 0 0 0'))
        colors = self.root/'colored.ply'
        colors.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n'
                          'property float y\nproperty float z\nproperty uchar red\n'
                          'property uchar green\nproperty uchar blue\nend_header\n'
                          '0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n')
        result = textured_mesh_to_glb(self.source, self.texture, self.root/'mesh.glb', {}, np.eye(4), colors)
        data = (self.root/'mesh.glb').read_bytes(); size = struct.unpack_from('<I', data, 12)[0]
        doc = json.loads(data[20:20+size]); primitive = doc['meshes'][0]['primitives'][0]
        self.assertEqual(primitive['material'], 1)
        self.assertIn('COLOR_0', primitive['attributes'])
        self.assertEqual(result['photo_color_fallback_faces'], 1)
        self.assertEqual(result['triangle_count'], 1)

    def test_generated_white_colors_are_not_presented_as_observed_photo_colors(self):
        self.source.write_text(self.source.read_text().replace('6 0 0 1 0 0 1', '6 0 0 0 0 0 0'))
        colors = self.root/'default_white.ply'
        colors.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\n'
                          'property float y\nproperty float z\nproperty uchar red\n'
                          'property uchar green\nproperty uchar blue\nend_header\n'
                          '0 0 0 255 255 255\n1 0 0 255 255 255\n0 1 0 255 255 255\n')
        result = textured_mesh_to_glb(self.source, self.texture, self.root/'mesh.glb', {},
                                     np.eye(4), colors, fallback_photo_colors=False)
        data = (self.root/'mesh.glb').read_bytes(); size = struct.unpack_from('<I', data, 12)[0]
        doc = json.loads(data[20:20+size]); view = doc['bufferViews'][5]
        actual = np.frombuffer(data[28+size:], '<f4', count=9, offset=view['byteOffset'])
        np.testing.assert_allclose(actual, [0.25]*9)
        self.assertEqual(result['photo_color_fallback_faces'], 0)
        self.assertEqual(result['neutral_fallback_faces'], 1)
        self.assertIn('Neutral surface', doc['materials'][1]['name'])


if __name__ == '__main__':
    unittest.main()
