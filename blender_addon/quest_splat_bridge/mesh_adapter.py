"""Explicitly approximate, editable isosurface from a Gaussian opacity field.

This is a preview mesh, not photogrammetric surface reconstruction. Thin splats
are broadened to the voxel width; percentile cropping omits extreme floaters;
the result may have holes, fused surfaces and no texture. The original native
Gaussian PointCloud remains untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


class MeshUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class MeshResult:
    vertices: object
    faces: object
    voxel_m: float
    splats_used: int
    grid_shape: tuple[int, int, int]
    level: float


def read_native_splats(obj):
    """Copy Blender data on the main thread before background computation."""
    import numpy as np

    if obj.type != "POINTCLOUD" or getattr(obj.data, "type", None) != "GAUSSIAN_SPLAT":
        raise MeshUnavailable("Select a native Gaussian PointCloud")
    count = len(obj.data.points)
    def read(name, width, field):
        attr = obj.data.attributes.get(name)
        if attr is None:
            raise MeshUnavailable(f"Gaussian attribute {name} is missing")
        values = np.empty(count * width, dtype=np.float32)
        attr.data.foreach_get(field, values)
        return values.reshape(count, width)
    return (
        read("position", 3, "vector"),
        read("scale", 3, "vector"),
        read("rotation", 4, "value"),
        read("radiance:base", 4, "vector")[:, 3].copy(),
    )


def _rotation_matrices(quaternions):
    import numpy as np

    q = quaternions.astype(np.float32, copy=True)
    norms = np.linalg.norm(q, axis=1)
    q /= np.maximum(norms[:, None], 1e-8)
    w, x, y, z = q.T
    result = np.empty((len(q), 3, 3), dtype=np.float32)
    result[:, 0, 0] = 1 - 2 * (y*y + z*z)
    result[:, 0, 1] = 2 * (x*y - z*w)
    result[:, 0, 2] = 2 * (x*z + y*w)
    result[:, 1, 0] = 2 * (x*y + z*w)
    result[:, 1, 1] = 1 - 2 * (x*x + z*z)
    result[:, 1, 2] = 2 * (y*z - x*w)
    result[:, 2, 0] = 2 * (x*z - y*w)
    result[:, 2, 1] = 2 * (y*z + x*w)
    result[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def compute_approximate_mesh(
    positions, scales, quaternions, opacities,
    requested_voxel_m: float = 0.15, max_voxels: int = 1_500_000,
    level: float = 0.65, smooth_sigma_voxels: float = 0.0,
) -> MeshResult:
    """Sample rotated Gaussian opacity and extract an approximate isosurface.

    Only the 12 nearest eligible splats contribute per voxel. This bounds work
    but deliberately means the output is a rough occupancy envelope.
    """
    import numpy as np
    try:
        from scipy.spatial import cKDTree
        from scipy.ndimage import gaussian_filter
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise MeshUnavailable("Approximate mesh needs scipy and scikit-image in Blender's Python") from exc

    if not 0.03 <= requested_voxel_m <= 1.0 or max_voxels < 1000 or not 0.05 <= level <= 3.0 or not 0 <= smooth_sigma_voxels <= 3:
        raise MeshUnavailable("Mesh sampling settings are outside supported bounds")
    if not (len(positions) == len(scales) == len(quaternions) == len(opacities)):
        raise MeshUnavailable("Gaussian attribute arrays have inconsistent lengths")
    valid = (
        np.isfinite(positions).all(axis=1) & np.isfinite(scales).all(axis=1) &
        np.isfinite(quaternions).all(axis=1) & np.isfinite(opacities) &
        (opacities >= 0.25) & (scales > 0).all(axis=1) &
        (scales.max(axis=1) <= 1.5)
    )
    pos, scale, quat, alpha = positions[valid], scales[valid], quaternions[valid], opacities[valid]
    if len(pos) < 100:
        raise MeshUnavailable("Too few usable splats for an opacity-field mesh")
    lower, upper = np.percentile(pos, [1, 99], axis=0)
    keep = ((pos >= lower) & (pos <= upper)).all(axis=1)
    pos, scale, quat, alpha = pos[keep], scale[keep], quat[keep], alpha[keep]
    if len(pos) < 100:
        raise MeshUnavailable("Too few splats remain after outlier cropping")
    margin = 0.45
    lower = pos.min(axis=0) - margin
    upper = pos.max(axis=0) + margin
    extent = upper - lower
    voxel = max(float(requested_voxel_m), float((np.prod(extent) / max_voxels) ** (1/3)))
    shape = tuple(int(np.ceil(width / voxel)) + 1 for width in extent)
    while np.prod(shape) > max_voxels:
        voxel *= 1.05
        shape = tuple(int(np.ceil(width / voxel)) + 1 for width in extent)
    if min(shape) < 3:
        raise MeshUnavailable("Scan is too thin for a 3D isosurface")

    scale = np.clip(scale, voxel * 0.6, 0.6)
    rotation = _rotation_matrices(quat)
    tree = cKDTree(pos)
    field = np.zeros(shape, dtype=np.float32)
    flat = field.ravel()
    total = flat.size
    # At most 12.8 MB for each 100k x 12 float32 temporary, plus neighbors.
    for start in range(0, total, 50_000):
        stop = min(start + 50_000, total)
        ix, iy, iz = np.unravel_index(np.arange(start, stop), shape)
        query = lower + np.column_stack((ix, iy, iz)) * voxel
        distances, indices = tree.query(query, k=12, distance_upper_bound=1.8, workers=-1)
        present = indices < len(pos)
        safe = np.minimum(indices, len(pos) - 1)
        offset = query[:, None, :] - pos[safe]
        local = np.einsum("nkj,nkji->nki", offset, rotation[safe], optimize=True)
        sigma = scale[safe]
        squared = np.sum((local / sigma) ** 2, axis=2)
        weights = np.where(present, alpha[safe] * np.exp(-0.5 * squared), 0.0)
        flat[start:stop] = weights.sum(axis=1)
    if smooth_sigma_voxels:
        field = gaussian_filter(field, sigma=smooth_sigma_voxels)
    if not np.any(field >= level) or not np.any(field < level):
        raise MeshUnavailable("Opacity field never crossed the selected surface level")
    vertices, faces, _normals, _values = marching_cubes(
        field, level=level, spacing=(voxel, voxel, voxel), step_size=2,
        allow_degenerate=False,
    )
    vertices += lower.astype(np.float32)
    if not len(faces):
        raise MeshUnavailable("Opacity field produced no editable faces")
    return MeshResult(vertices, faces, voxel, len(pos), shape, level)


def create_mesh_object(context, scan, result: MeshResult):
    """Create editable Blender mesh on main thread and leave scan untouched."""
    import bpy

    project_id = scan.get("quest_splat_project_id", "")
    parent = next(
        (collection for collection in scan.users_collection if collection.get("quest_splat_role") == "scan_reference"),
        None,
    )
    root = next((collection for collection in bpy.data.collections if collection.get("quest_splat_role") == "project" and collection.get("quest_splat_project_id") == project_id), None)
    if root is None or parent is None:
        raise MeshUnavailable("Scan must be in the managed ScanReference collection")
    base = next((collection for collection in root.children if collection.get("quest_splat_role") == "base_mesh"), None)
    if base is None:
        base = bpy.data.collections.new("BaseMesh")
        base["quest_splat_role"] = "base_mesh"
        root.children.link(base)
    mesh = bpy.data.meshes.new("Approximate Gaussian Field Surface")
    mesh.from_pydata(result.vertices.tolist(), [], result.faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("Approximate Scan Mesh", mesh)
    base.objects.link(obj)
    obj.matrix_world = scan.matrix_world.copy()
    obj["quest_splat_project_id"] = project_id
    obj["quest_splat_source_sha256"] = scan["quest_splat_source_sha256"]
    if scan.get("quest_splat_source_job_id"):
        obj["quest_splat_source_job_id"] = scan["quest_splat_source_job_id"]
    obj["quest_splat_mesh_method"] = "sampled_rotated_gaussian_opacity_isosurface"
    obj["quest_splat_mesh_quality"] = "approximate_visual_preview_not_architectural_measurement"
    obj["quest_splat_voxel_m"] = result.voxel_m
    obj["quest_splat_splats_used"] = result.splats_used
    obj["quest_splat_iso_level"] = result.level
    return obj
