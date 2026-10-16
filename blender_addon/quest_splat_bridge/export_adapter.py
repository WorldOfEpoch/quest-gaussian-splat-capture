"""Export edited native Blender Gaussian PointCloud data without losing splats.

Blender 5.3 alpha's ordinary PLY exporter writes zero vertices for PointCloud.
This serializer emits the common 3DGS binary PLY layout. Coordinates remain in
the PointCloud's local frame; a required sidecar records its Blender transform.
Keeping the transform separate avoids silently corrupting view-dependent SH.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile


class ExportUnavailable(RuntimeError):
    pass


PROPERTY_NAMES = (
    "x", "y", "z", "nx", "ny", "nz",
    *(f"f_dc_{i}" for i in range(3)),
    *(f"f_rest_{i}" for i in range(45)),
    "opacity", "scale_0", "scale_1", "scale_2",
    *(f"rot_{i}" for i in range(4)),
)


def gaussian_rows(obj):
    """Copy all native attributes on Blender's main thread into PLY float rows."""
    import numpy as np

    if obj.type != "POINTCLOUD" or getattr(obj.data, "type", None) != "GAUSSIAN_SPLAT":
        raise ExportUnavailable("Select a native Blender Gaussian PointCloud")
    count = len(obj.data.points)
    if count < 1:
        raise ExportUnavailable("PointCloud contains no splats")

    def read(name, width, field):
        attr = obj.data.attributes.get(name)
        if attr is None or len(attr.data) != count:
            raise ExportUnavailable(f"Required Gaussian attribute is missing: {name}")
        values = np.empty(count * width, dtype=np.float32)
        attr.data.foreach_get(field, values)
        return values.reshape(count, width)

    position = read("position", 3, "vector")
    base = read("radiance:base", 4, "vector")
    scales = read("scale", 3, "vector")
    rotation = read("rotation", 4, "value")
    sh = [read(f"radiance:sh_{i}", 3, "vector") for i in range(15)]
    if not all(np.isfinite(array).all() for array in (position, base, scales, rotation, *sh)):
        raise ExportUnavailable("Gaussian attributes contain NaN or infinity")
    if np.any(scales <= 0):
        raise ExportUnavailable("Gaussian scales must be positive")
    if np.any((base[:, 3] < 0) | (base[:, 3] > 1)):
        raise ExportUnavailable("Gaussian alpha must be between zero and one")
    if np.any(np.linalg.norm(rotation, axis=1) < 1e-8):
        raise ExportUnavailable("Gaussian rotations must be nonzero quaternions")

    rows = np.zeros((count, len(PROPERTY_NAMES)), dtype="<f4")
    rows[:, :3] = position
    rows[:, 6:9] = base[:, :3]
    for channel in range(3):
        for basis in range(15):
            rows[:, 9 + channel * 15 + basis] = sh[basis][:, channel]
    # Native import may round extreme sigmoid values to exactly 0 or 1.
    alpha = np.clip(base[:, 3].astype(np.float64), 1e-7, 1.0 - 1e-7)
    rows[:, 54] = np.log(alpha / (1.0 - alpha)).astype(np.float32)
    rows[:, 55:58] = np.log(scales).astype(np.float32)
    rows[:, 58:62] = rotation / np.linalg.norm(rotation, axis=1)[:, None]
    return rows


def export_gaussian_ply(obj, path: str | Path) -> tuple[Path, Path, int]:
    """Write local-frame splats and transform sidecar; never replace source scan."""
    import numpy as np

    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".ply":
        raise ExportUnavailable("Choose a .ply export path")
    source = obj.get("quest_splat_source_path")
    if source and target == Path(source).expanduser().resolve():
        raise ExportUnavailable("Export path would overwrite the original scan")
    if not target.parent.is_dir():
        raise ExportUnavailable("Export folder does not exist")
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ExportUnavailable("Object transform is invalid")
    # Even shear/nonuniform scale is preserved exactly in the sidecar; it is
    # deliberately not baked into Gaussian scales and view-dependent SH.
    rows = gaussian_rows(obj)
    inferred_job_id = ""
    if source and not obj.get("quest_splat_source_job_id"):
        from .import_adapter import _source_identity
        _, inferred_job_id = _source_identity(Path(source))
    sidecar = target.with_suffix(target.suffix + ".transform.json")
    header = "\n".join((
        "ply", "format binary_little_endian 1.0",
        "comment Quest Splat Bridge local-coordinate Gaussian export; apply paired transform JSON",
        f"element vertex {len(rows)}",
        *(f"property float {name}" for name in PROPERTY_NAMES),
        "end_header", "",
    )).encode("ascii")
    transform = {
        "schema": "quest-splat-transform",
        "schema_version": 1,
        "coordinate_frame": "blender_object_local",
        "coordinate_convention": "BlenderRH_Zup",
        "matrix_world_row_major": matrix.tolist(),
        "ply_file": target.name,
        "project_id": obj.get("quest_splat_project_id", ""),
        "source_sha256": obj.get("quest_splat_source_sha256", ""),
        "source_job_id": obj.get("quest_splat_source_job_id", "") or inferred_job_id,
        "alignment": obj.get("quest_splat_alignment", ""),
        "splat_count": len(rows),
    }
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=".qsb-", delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(header)
            handle.write(rows.tobytes(order="C"))
        except Exception:
            pending.unlink(missing_ok=True)
            raise
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=".qsb-", delete=False) as handle:
        pending_sidecar = Path(handle.name)
        try:
            json.dump(transform, handle, indent=2)
        except Exception:
            pending_sidecar.unlink(missing_ok=True)
            pending.unlink(missing_ok=True)
            raise
    os.replace(pending, target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    transform["exported_ply_sha256"] = digest.hexdigest()
    with pending_sidecar.open("w", encoding="utf-8") as handle:
        json.dump(transform, handle, indent=2)
    os.replace(pending_sidecar, sidecar)
    return target, sidecar, len(rows)
