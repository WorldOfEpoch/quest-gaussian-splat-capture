"""Native Blender 5.3 Gaussian PLY import boundary.

The bridge currently publishes no artifact download manifest. This adapter
imports an explicitly selected, existing local reconstruction file. It accepts
the result only when Blender's native PLY operator makes a Gaussian PointCloud.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid


class ImportUnavailable(RuntimeError):
    pass


_REQUIRED_PROPERTIES = {
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0",
    "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
}


def inspect_gaussian_ply(path: str | Path) -> int:
    """Validate the compact PLY header and return its advertised splat count."""
    source = Path(path)
    if not source.is_file() or source.suffix.lower() != ".ply":
        raise ImportUnavailable("Choose an existing Gaussian .ply file")
    with source.open("rb") as handle:
        header = handle.read(32768)
    terminator = header.find(b"end_header\n")
    if terminator < 0:
        raise ImportUnavailable("PLY header is incomplete or exceeds 32 KiB")
    try:
        lines = header[:terminator].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ImportUnavailable("PLY header is not ASCII") from exc
    if not lines or lines[0] != "ply" or "format binary_little_endian 1.0" not in lines:
        raise ImportUnavailable("Expected a binary little-endian Gaussian PLY")
    vertex_line = next((line for line in lines if line.startswith("element vertex ")), None)
    try:
        count = int(vertex_line.split()[2]) if vertex_line else 0
    except (ValueError, IndexError) as exc:
        raise ImportUnavailable("PLY vertex count is invalid") from exc
    if count < 1:
        raise ImportUnavailable("Gaussian PLY has no splats")
    property_lines = [line.split() for line in lines if line.startswith("property ")]
    if any(len(parts) != 3 or parts[1] != "float" for parts in property_lines):
        raise ImportUnavailable("Expected float-valued Gaussian PLY vertex properties")
    properties = {parts[2] for parts in property_lines}
    missing = sorted(_REQUIRED_PROPERTIES - properties)
    if missing:
        raise ImportUnavailable("PLY lacks Gaussian properties: " + ", ".join(missing))
    payload_start = terminator + len(b"end_header\n")
    expected_bytes = payload_start + count * len(property_lines) * 4
    if source.stat().st_size < expected_bytes:
        raise ImportUnavailable("Gaussian PLY payload is truncated")
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_transform(source: Path, source_hash: str):
    sidecar = source.with_suffix(source.suffix + ".transform.json")
    if not sidecar.is_file():
        return None
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        if (document.get("schema"), document.get("schema_version")) != ("quest-splat-transform", 1):
            raise ValueError("unsupported schema")
        if document.get("coordinate_convention") != "BlenderRH_Zup":
            raise ValueError("unsupported coordinate convention")
        if document.get("ply_file") != source.name or document.get("exported_ply_sha256") != source_hash:
            raise ValueError("PLY filename or SHA-256 does not match")
        matrix = document["matrix_world_row_major"]
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError("expected a 4 by 4 object matrix")
        import math
        if not all(math.isfinite(float(value)) for row in matrix for value in row):
            raise ValueError("object matrix contains invalid values")
        return document
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as exc:
        raise ImportUnavailable(f"Paired transform sidecar is invalid: {exc}") from exc


def _source_identity(source: Path) -> tuple[str, str]:
    """Infer immutable reconstruction IDs from the bridge's on-disk layout."""
    parts = source.parts
    if source.name != "splat.ply" or len(parts) < 5 or parts[-3] != "reconstructions" or parts[-5] != "projects":
        return "", ""
    try:
        return str(uuid.UUID(parts[-4])), str(uuid.UUID(parts[-2]))
    except ValueError:
        return "", ""


def _project_collection(bpy, context, project_id: str):
    for collection in bpy.data.collections:
        if collection.get("quest_splat_project_id") == project_id and collection.get("quest_splat_role") == "project":
            root = collection
            break
    else:
        root = bpy.data.collections.new("Quest Scan Project")
        root["quest_splat_project_id"] = project_id
        root["quest_splat_role"] = "project"
        context.scene.collection.children.link(root)
    for collection in root.children:
        if collection.get("quest_splat_role") == "scan_reference":
            return collection
    reference = bpy.data.collections.new("ScanReference")
    reference["quest_splat_role"] = "scan_reference"
    root.children.link(reference)
    return reference


def import_splat_file(context, path: str | Path, project_id: str = ""):
    """Import one immutable local PLY as a native 5.3 Gaussian PointCloud.

    User collections and prior scan references remain intact. A failed native
    import is rolled back without touching pre-existing objects.
    """
    import bpy

    if bpy.app.version < (5, 3, 0):
        raise ImportUnavailable("Native Gaussian PLY import requires Blender 5.3 or later")
    source = Path(path).expanduser().resolve(strict=True)
    expected_count = inspect_gaussian_ply(source)
    source_hash = _sha256(source)
    transform = _paired_transform(source, source_hash)
    source_project_id, source_job_id = _source_identity(source)
    if source_project_id and project_id and source_project_id != project_id:
        raise ImportUnavailable("Gaussian source belongs to a different project")
    if transform and project_id and transform.get("project_id") and project_id != transform["project_id"]:
        raise ImportUnavailable("Paired transform belongs to a different project")
    before = {obj.as_pointer() for obj in bpy.data.objects}
    try:
        outcome = bpy.ops.wm.ply_import(
            filepath=str(source),
            forward_axis="Y",
            up_axis="Z",
            import_attributes=True,
        )
    except Exception as exc:
        raise ImportUnavailable(f"Blender PLY importer failed: {exc}") from exc
    created = [obj for obj in bpy.data.objects if obj.as_pointer() not in before]
    recognized = [
        obj for obj in created
        if obj.type == "POINTCLOUD" and getattr(obj.data, "type", None) == "GAUSSIAN_SPLAT"
    ]
    if "FINISHED" not in outcome or len(recognized) != 1:
        for obj in created:
            bpy.data.objects.remove(obj, do_unlink=True)
        raise ImportUnavailable("Native importer did not create exactly one Gaussian PointCloud; imported objects were removed")
    obj = recognized[0]
    actual_count = len(obj.data.points)
    if actual_count != expected_count:
        for item in created:
            bpy.data.objects.remove(item, do_unlink=True)
        raise ImportUnavailable(f"Native importer produced {actual_count} splats; PLY declares {expected_count}")
    effective_project_id = project_id or (transform.get("project_id", "") if transform else "") or source_project_id
    reference = _project_collection(bpy, context, effective_project_id)
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    reference.objects.link(obj)
    obj.name = f"Gaussian Scan {source.stem}"
    obj["quest_splat_source_path"] = str(source)
    obj["quest_splat_source_sha256"] = source_hash
    obj["quest_splat_project_id"] = effective_project_id
    obj["quest_splat_alignment"] = "blender_object_local_with_sidecar" if transform else "unregistered_colmap_frame"
    obj["quest_splat_splat_count"] = actual_count
    if source_job_id:
        obj["quest_splat_source_job_id"] = source_job_id
    if transform:
        from mathutils import Matrix
        obj.matrix_world = Matrix(transform["matrix_world_row_major"])
        obj["quest_splat_parent_source_sha256"] = transform.get("source_sha256", "")
        if transform.get("source_job_id"):
            obj["quest_splat_source_job_id"] = transform["source_job_id"]
    for prior in reference.objects:
        if prior != obj and prior.get("quest_splat_source_sha256"):
            prior.hide_viewport = True
            prior.hide_render = True
    obj.hide_viewport = False
    obj.hide_render = False
    return obj


def import_splat_manifest(_context, _manifest):
    raise ImportUnavailable(
        "The bridge has no pinned artifact download contract yet. Use the "
        "local Gaussian PLY import control for a completed reconstruction."
    )
