from pathlib import Path
import hashlib
import json
from tempfile import TemporaryDirectory
import unittest
import uuid

from blender_addon.quest_splat_bridge.import_adapter import ImportUnavailable, inspect_gaussian_ply, _paired_transform, _source_identity


PROPERTIES = (
    "x y z f_dc_0 f_dc_1 f_dc_2 f_rest_0 opacity "
    "scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3"
).split()


def sample_ply(count=2, payload=True):
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {count}",
        *(f"property float {name}" for name in PROPERTIES), "end_header", "",
    ]).encode("ascii")
    return header + (b"\0" * (count * len(PROPERTIES) * 4) if payload else b"")


class GaussianPlyValidationTests(unittest.TestCase):
    def test_accepts_complete_gaussian_payload(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.ply"
            path.write_bytes(sample_ply())
            self.assertEqual(inspect_gaussian_ply(path), 2)

    def test_rejects_truncated_payload(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.ply"
            path.write_bytes(sample_ply(payload=False))
            with self.assertRaisesRegex(ImportUnavailable, "truncated"):
                inspect_gaussian_ply(path)

    def test_rejects_plain_point_cloud(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "points.ply"
            path.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n" + b"\0" * 12)
            with self.assertRaisesRegex(ImportUnavailable, "lacks Gaussian properties"):
                inspect_gaussian_ply(path)

    def test_transform_sidecar_is_bound_to_exported_ply_hash(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "edited.ply"
            path.write_bytes(sample_ply())
            transform = {
                "schema": "quest-splat-transform", "schema_version": 1,
                "coordinate_convention": "BlenderRH_Zup", "ply_file": path.name,
                "exported_ply_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "matrix_world_row_major": [[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]],
            }
            path.with_suffix(".ply.transform.json").write_text(json.dumps(transform))
            self.assertEqual(_paired_transform(path, transform["exported_ply_sha256"])["matrix_world_row_major"][0][3], 1)
            path.write_bytes(path.read_bytes() + b"modified")
            with self.assertRaisesRegex(ImportUnavailable, "SHA-256"):
                _paired_transform(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_source_job_is_inferred_only_from_reconstruction_layout(self):
        pid = str(uuid.UUID(int=1))
        jid = str(uuid.UUID(int=2))
        self.assertEqual(_source_identity(Path("projects") / pid / "reconstructions" / jid / "splat.ply"), (pid, jid))
        self.assertEqual(_source_identity(Path("other") / pid / "reconstructions" / jid / "splat.ply"), ("", ""))


if __name__ == "__main__":
    unittest.main()
