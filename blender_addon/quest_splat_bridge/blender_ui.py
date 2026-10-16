"""Blender UI for the protocol 0.1 bridge.

Workers perform HTTP requests only. The timer consumes their messages and is
the sole place where results change Blender scene/UI state.
"""

from __future__ import annotations

import os
import json
import queue
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import bpy

from .client import BridgeClient, BridgeError, catalog_view, reorder_plan, token_file_path
from .import_adapter import ImportUnavailable, import_splat_file, import_splat_manifest, _source_identity
from .export_adapter import ExportUnavailable, export_gaussian_ply
from .mesh_adapter import MeshUnavailable, compute_approximate_mesh, create_mesh_object, read_native_splats


_executor: ThreadPoolExecutor | None = None
_results: queue.SimpleQueue[tuple[str, dict[str, Any] | Exception]] = queue.SimpleQueue()


@dataclass
class _Runtime:
    projects: list[dict[str, Any]] = field(default_factory=list)
    project: dict[str, Any] | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    models_entry_id: str = ""
    sessions: list[dict[str, Any]] = field(default_factory=list)
    session: dict[str, Any] | None = None
    preview: dict[str, Any] | None = None
    message: str = "Connect to the PC bridge."
    busy: int = 0


_state = _Runtime()


def _client(context):
    token = os.environ.get("QUEST_SPLAT_TOKEN", "")
    if not token:
        token_path = token_file_path()
        if token_path.is_file():
            token = token_path.read_text(encoding="ascii").strip()
    return BridgeClient(context.scene.quest_splat_bridge_url, token)


def _submit(context, operation: str, work):
    global _executor
    try:
        client = _client(context)
    except (ValueError, BridgeError, OSError, UnicodeError) as exc:
        _state.message = str(exc)
        return
    _state.busy += 1
    _state.message = "Contacting bridge…"

    def task():
        try:
            _results.put((operation, work(client)))
        except Exception as exc:  # Report service failures in the panel.
            _results.put((operation, exc))

    if _executor is None:
        _state.busy = max(0, _state.busy - 1)
        _state.message = "Add-on worker is not running. Re-enable the add-on."
        return
    _executor.submit(task)


def _submit_local(operation: str, work):
    if _executor is None:
        _state.message = "Add-on worker is not running. Re-enable the add-on."
        return
    _state.busy += 1
    _state.message = "Computing experimental rough mesh…"
    def task():
        try:
            _results.put((operation, work()))
        except Exception as exc:
            _results.put((operation, exc))
    _executor.submit(task)


def _set_project_scene(scene, project_id: str):
    scene.quest_splat_project_id = project_id
    scene.quest_splat_session_id = ""
    scene.quest_splat_entry_id = ""
    scene.quest_splat_parent_id = ""


def _accept_result(operation: str, payload: dict[str, Any] | Exception):
    _state.busy = max(0, _state.busy - 1)
    if isinstance(payload, Exception):
        _state.message = str(payload)
        return
    # A user can switch scenes while a request is running. Update the scene
    # identified at submission, rather than assuming the currently active one.
    scene = bpy.data.scenes.get(payload.get("scene_name", ""))
    if scene is None:
        _state.message = "Bridge replied, but the originating scene was closed."
        return
    if operation == "projects":
        _state.projects = payload["projects"]
        _state.message = f"{len(_state.projects)} project(s)."
    elif operation == "create_project":
        _state.projects = payload["projects"]
        _state.project = payload["project"]
        _state.entries = payload["entries"]
        _state.sessions = []
        _state.session = None
        _state.preview = None
        _set_project_scene(scene, _state.project["id"])
        _state.message = f"Project created: {_state.project['name']}"
    elif operation == "open_project":
        if scene.quest_splat_project_id != payload["project"]["id"]:
            return  # Selection changed after request was issued.
        _state.project = payload["project"]
        _state.entries = payload["entries"]
        _state.sessions = payload["sessions"]
        _state.session = None
        _state.preview = None
        _state.message = f"Project open: {_state.project['name']}"
    elif operation == "entries":
        if scene.quest_splat_project_id != payload["project_id"]:
            return
        _state.entries = payload["entries"]
        _state.message = payload["message"]
    elif operation == "models":
        if scene.quest_splat_project_id != payload["project_id"] or scene.quest_splat_entry_id != payload["entry_id"]:
            return
        _state.models = payload["models"]
        _state.models_entry_id = payload["entry_id"]
        _state.message = f"{len(_state.models)} saved revision(s) for selected entry."
    elif operation == "sessions":
        if scene.quest_splat_project_id != payload["project_id"]:
            return
        _state.sessions = payload["sessions"]
        if payload.get("session"):
            _state.session = payload["session"]
            scene.quest_splat_session_id = _state.session["id"]
        _state.message = payload["message"]
    elif operation == "session":
        if scene.quest_splat_session_id != payload["session"]["id"]:
            return
        _state.session = payload["session"]
        _state.message = f"Session {_state.session['status']}; {_state.session['capture_count']} received capture(s)."
    elif operation == "preview":
        if scene.quest_splat_session_id != payload["session_id"]:
            return
        _state.preview = payload["preview"]
        result = _state.preview
        _state.message = result.get("reason", "Preview ready; import adapter pending.") if result["status"] == "not_ready" else "Preview artifact reported ready."
    elif operation == "approx_mesh":
        scan = bpy.data.objects.get(payload["scan_name"])
        if scan is None:
            _state.message = "Scan was removed before the rough mesh finished."
        else:
            try:
                mesh = create_mesh_object(bpy.context, scan, payload["result"])
                _state.message = f"Experimental rough mesh: {len(mesh.data.vertices)} vertices; inspect before use."
            except MeshUnavailable as exc:
                _state.message = str(exc)
    elif operation == "publish_model":
        revision = payload["revision"]
        if scene.quest_splat_project_id == payload["project_id"] and scene.quest_splat_entry_id == payload["entry_id"]:
            _state.models = payload["models"]
            _state.models_entry_id = payload["entry_id"]
        _state.message = (f"Saved local model revision {revision['revision_id'][:8]} under "
                          f"{payload['entry_name']}; Quest placement remains unverified.")
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _poll_results():
    while True:
        try:
            operation, payload = _results.get_nowait()
        except queue.Empty:
            break
        _accept_result(operation, payload)
    return 0.2


def _scene_name(context):
    return context.scene.name


class QSB_OT_refresh_projects(bpy.types.Operator):
    bl_idname = "qsb.refresh_projects"
    bl_label = "Refresh Projects"

    def execute(self, context):
        scene_name = _scene_name(context)
        _submit(context, "projects", lambda c: {"scene_name": scene_name, "projects": c.list_projects()})
        return {'FINISHED'}


class QSB_OT_create_project(bpy.types.Operator):
    bl_idname = "qsb.create_project"
    bl_label = "New Project"

    def execute(self, context):
        scene_name = _scene_name(context)
        name = context.scene.quest_splat_name
        def work(client):
            project = client.create_project(name)
            return {"scene_name": scene_name, "project": project, "projects": client.list_projects(), "entries": client.list_entries(project["id"])}
        _submit(context, "create_project", work)
        return {'FINISHED'}


class QSB_OT_open_project(bpy.types.Operator):
    bl_idname = "qsb.open_project"
    bl_label = "Open Project"

    project_id: bpy.props.StringProperty()

    def execute(self, context):
        project_id = self.project_id
        _set_project_scene(context.scene, project_id)
        scene_name = _scene_name(context)
        def work(client):
            return {"scene_name": scene_name, "project": client.get_project(project_id), "entries": client.list_entries(project_id), "sessions": client.list_sessions(project_id)}
        _submit(context, "open_project", work)
        return {'FINISHED'}


class QSB_OT_refresh_entries(bpy.types.Operator):
    bl_idname = "qsb.refresh_entries"
    bl_label = "Refresh Library"

    def execute(self, context):
        project_id = context.scene.quest_splat_project_id
        if not project_id:
            return {'CANCELLED'}
        scene_name = _scene_name(context)
        _submit(context, "entries", lambda c: {"scene_name": scene_name, "project_id": project_id, "entries": c.list_entries(project_id), "message": "Library refreshed."})
        return {'FINISHED'}


class QSB_OT_create_entry(bpy.types.Operator):
    bl_idname = "qsb.create_entry"
    bl_label = "Create Library Entry"

    kind: bpy.props.StringProperty()

    def execute(self, context):
        scene = context.scene
        project_id, parent_id = scene.quest_splat_project_id, scene.quest_splat_parent_id or None
        name, kind, scene_name = scene.quest_splat_name, self.kind, _scene_name(context)
        def work(client):
            client.create_entry(project_id, name, kind, parent_id)
            return {"scene_name": scene_name, "project_id": project_id, "entries": client.list_entries(project_id), "message": f"{kind.title()} created: {name.strip()}"}
        _submit(context, "entries", work)
        return {'FINISHED'}


class QSB_OT_rename_entry(bpy.types.Operator):
    bl_idname = "qsb.rename_entry"
    bl_label = "Rename Selected Entry"

    def execute(self, context):
        scene = context.scene
        entry_id, project_id, name = scene.quest_splat_entry_id, scene.quest_splat_project_id, scene.quest_splat_name
        if not entry_id:
            return {'CANCELLED'}
        chosen = next((e for e in _state.entries if e["id"] == entry_id), None)
        if not chosen or chosen.get("protected"):
            _state.message = "Original is protected."
            return {'CANCELLED'}
        scene_name = _scene_name(context)
        def work(client):
            client.update_entry(project_id, entry_id, name=name)
            return {"scene_name": scene_name, "project_id": project_id, "entries": client.list_entries(project_id), "message": f"Renamed to {name.strip()}"}
        _submit(context, "entries", work)
        return {'FINISHED'}


class QSB_OT_navigate(bpy.types.Operator):
    bl_idname = "qsb.navigate"
    bl_label = "Browse Library"

    entry_id: bpy.props.StringProperty()

    def execute(self, context):
        scene = context.scene
        scene.quest_splat_parent_id = self.entry_id
        return {'FINISHED'}


class QSB_OT_move_entry_here(bpy.types.Operator):
    bl_idname = "qsb.move_entry_here"
    bl_label = "Move Selected Here"

    def execute(self, context):
        scene = context.scene
        entry_id, project_id = scene.quest_splat_entry_id, scene.quest_splat_project_id
        destination = scene.quest_splat_parent_id or None
        selected = next((e for e in _state.entries if e["id"] == entry_id), None)
        if not selected or selected.get("protected"):
            _state.message = "Choose an unprotected library entry."
            return {'CANCELLED'}
        scene_name = _scene_name(context)
        def work(client):
            client.update_entry(project_id, entry_id, parent_id=destination)
            return {"scene_name": scene_name, "project_id": project_id, "entries": client.list_entries(project_id), "message": f"Moved {selected['name']}."}
        _submit(context, "entries", work)
        return {'FINISHED'}


class QSB_OT_reorder_entry(bpy.types.Operator):
    bl_idname = "qsb.reorder_entry"
    bl_label = "Reorder Entry"

    entry_id: bpy.props.StringProperty()
    direction: bpy.props.IntProperty()

    def execute(self, context):
        scene = context.scene
        project_id, scene_name = scene.quest_splat_project_id, _scene_name(context)
        plan = reorder_plan(_state.entries, self.entry_id, self.direction)
        if not plan:
            return {'CANCELLED'}
        def work(client):
            for entry_id, sort_order in plan:
                client.update_entry(project_id, entry_id, sort_order=sort_order)
            return {"scene_name": scene_name, "project_id": project_id, "entries": client.list_entries(project_id), "message": "Library order updated."}
        _submit(context, "entries", work)
        return {'FINISHED'}


class QSB_OT_select_entry(bpy.types.Operator):
    bl_idname = "qsb.select_entry"
    bl_label = "Select Entry"

    entry_id: bpy.props.StringProperty()

    def execute(self, context):
        context.scene.quest_splat_entry_id = self.entry_id
        _state.models = []
        _state.models_entry_id = ""
        return {'FINISHED'}


class QSB_OT_refresh_models(bpy.types.Operator):
    bl_idname = "qsb.refresh_models"
    bl_label = "Refresh Saved Revisions"

    def execute(self, context):
        project_id = context.scene.quest_splat_project_id
        entry_id = context.scene.quest_splat_entry_id
        if not project_id or not entry_id:
            return {'CANCELLED'}
        scene_name = _scene_name(context)
        _submit(context, "models", lambda c: {"scene_name": scene_name, "project_id": project_id,
                "entry_id": entry_id, "models": c.list_models(project_id, entry_id)})
        return {'FINISHED'}


class QSB_OT_start_session(bpy.types.Operator):
    bl_idname = "qsb.start_session"
    bl_label = "Start Capture Session"

    def execute(self, context):
        scene = context.scene
        project_id, origin_id = scene.quest_splat_project_id, scene.quest_splat_tracking_origin_id
        try:
            str(uuid.UUID(origin_id))
        except ValueError:
            _state.message = "Enter the Quest's current tracking-origin UUID first."
            return {'CANCELLED'}
        scene_name = _scene_name(context)
        def work(client):
            session = client.create_session(project_id, origin_id)
            return {"scene_name": scene_name, "project_id": project_id, "session": session, "sessions": client.list_sessions(project_id), "message": "Capture session started."}
        _submit(context, "sessions", work)
        return {'FINISHED'}


class QSB_OT_open_session(bpy.types.Operator):
    bl_idname = "qsb.open_session"
    bl_label = "Open Capture Session"

    session_id: bpy.props.StringProperty()

    def execute(self, context):
        session_id, scene_name = self.session_id, _scene_name(context)
        context.scene.quest_splat_session_id = session_id
        _submit(context, "session", lambda c: {"scene_name": scene_name, "session": c.get_session(session_id)})
        return {'FINISHED'}


class QSB_OT_refresh_session(bpy.types.Operator):
    bl_idname = "qsb.refresh_session"
    bl_label = "Refresh Session"

    def execute(self, context):
        session_id, scene_name = context.scene.quest_splat_session_id, _scene_name(context)
        if not session_id:
            return {'CANCELLED'}
        _submit(context, "session", lambda c: {"scene_name": scene_name, "session": c.get_session(session_id)})
        return {'FINISHED'}


class QSB_OT_finish_session(bpy.types.Operator):
    bl_idname = "qsb.finish_session"
    bl_label = "Finish Session"

    def execute(self, context):
        session_id, scene_name = context.scene.quest_splat_session_id, _scene_name(context)
        if not session_id:
            return {'CANCELLED'}
        _submit(context, "session", lambda c: {"scene_name": scene_name, "session": c.finish_session(session_id)})
        return {'FINISHED'}


class QSB_OT_check_preview(bpy.types.Operator):
    bl_idname = "qsb.check_preview"
    bl_label = "Check Latest Scan"

    def execute(self, context):
        session_id, scene_name = context.scene.quest_splat_session_id, _scene_name(context)
        if not session_id:
            return {'CANCELLED'}
        _submit(context, "preview", lambda c: {"scene_name": scene_name, "session_id": session_id, "preview": c.get_preview(session_id)})
        return {'FINISHED'}


class QSB_OT_load_scan(bpy.types.Operator):
    bl_idname = "qsb.load_scan"
    bl_label = "Load Latest Scan"

    def execute(self, context):
        preview = _state.preview
        if not preview or preview["status"] != "ready":
            self.report({'WARNING'}, "No complete Gaussian scan is available")
            return {'CANCELLED'}
        try:
            import_splat_manifest(context, preview["artifact"])
        except ImportUnavailable as exc:
            self.report({'WARNING'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class QSB_OT_import_local_ply(bpy.types.Operator):
    bl_idname = "qsb.import_local_ply"
    bl_label = "Import Gaussian PLY"

    def execute(self, context):
        scene = context.scene
        try:
            obj = import_splat_file(context, scene.quest_splat_local_ply, scene.quest_splat_project_id)
        except (ImportUnavailable, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            _state.message = str(exc)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Imported {obj['quest_splat_splat_count']} native Gaussian splats")
        _state.message = f"Imported native Gaussian PointCloud: {obj.name}"
        return {'FINISHED'}


class QSB_OT_export_gaussian_ply(bpy.types.Operator):
    bl_idname = "qsb.export_gaussian_ply"
    bl_label = "Export Edited Gaussian PLY"

    def execute(self, context):
        if not context.active_object:
            self.report({'ERROR'}, "Select a native Gaussian PointCloud")
            return {'CANCELLED'}
        context.view_layer.update()
        try:
            ply, sidecar, count = export_gaussian_ply(context.active_object, context.scene.quest_splat_export_ply)
        except (ExportUnavailable, OSError, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            _state.message = str(exc)
            return {'CANCELLED'}
        _state.message = f"Exported {count} edited splats and paired transform JSON."
        self.report({'INFO'}, f"Exported {count} splats to {ply.name} with {sidecar.name}")
        return {'FINISHED'}


class QSB_OT_rough_mesh(bpy.types.Operator):
    bl_idname = "qsb.rough_mesh"
    bl_label = "Create Experimental Rough Mesh"

    def execute(self, context):
        scan = context.active_object
        if scan is None:
            self.report({'ERROR'}, "Select a native Gaussian PointCloud")
            return {'CANCELLED'}
        try:
            arrays = read_native_splats(scan)
        except (MeshUnavailable, ValueError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        scene_name = _scene_name(context)
        scan_name = scan.name
        def work():
            result = compute_approximate_mesh(*arrays, requested_voxel_m=0.2, smooth_sigma_voxels=0.7)
            return {"scene_name": scene_name, "scan_name": scan_name, "result": result}
        _submit_local("approx_mesh", work)
        return {'FINISHED'}


class QSB_OT_export_mesh_glb(bpy.types.Operator):
    bl_idname = "qsb.export_mesh_glb"
    bl_label = "Export Selected Mesh GLB"

    def execute(self, context):
        obj = context.active_object
        target = Path(context.scene.quest_splat_mesh_glb).expanduser().resolve()
        if obj is None or obj.type != 'MESH' or not len(obj.data.polygons):
            self.report({'ERROR'}, "Select an editable mesh with faces")
            return {'CANCELLED'}
        if target.suffix.lower() != '.glb' or not target.parent.is_dir():
            self.report({'ERROR'}, "Choose a .glb path in an existing folder")
            return {'CANCELLED'}
        try:
            _export_mesh_glb(context, obj, target)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        _state.message = f"Exported selected mesh as {target.name}; rough scan meshes need inspection."
        return {'FINISHED'}


def _export_mesh_glb(context, obj, target: Path):
    prior = tuple(context.selected_objects)
    try:
        for item in prior:
            item.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        result = bpy.ops.export_scene.gltf(
            filepath=str(target), export_format='GLB', use_selection=True,
            export_apply=False, export_yup=True,
        )
        if 'FINISHED' not in result or not target.is_file():
            raise RuntimeError("Blender GLB exporter did not write a file")
    finally:
        obj.select_set(False)
        for item in prior:
            item.select_set(True)


def _source_job_for_object(obj):
    job_id = obj.get("quest_splat_source_job_id", "")
    if job_id:
        return job_id
    source_path = obj.get("quest_splat_source_path", "")
    if source_path:
        _, job_id = _source_identity(Path(source_path))
        if job_id:
            return job_id
    source_hash = obj.get("quest_splat_source_sha256", "")
    if not source_hash:
        return None
    for scan in bpy.data.objects:
        if scan.get("quest_splat_source_sha256", "") == source_hash and scan.get("quest_splat_source_path"):
            _, job_id = _source_identity(Path(scan["quest_splat_source_path"]))
            if job_id:
                return job_id
    return None


class QSB_OT_publish_model(bpy.types.Operator):
    bl_idname = "qsb.publish_model"
    bl_label = "Save Model Revision to Project"

    def execute(self, context):
        scene = context.scene
        entry = next((item for item in _state.entries if item["id"] == scene.quest_splat_entry_id), None)
        obj = context.active_object
        if not _state.project or scene.quest_splat_project_id != _state.project["id"]:
            self.report({'ERROR'}, "Open a project before saving a revision")
            return {'CANCELLED'}
        if not entry or entry["kind"] not in {"design", "variation"} or entry.get("protected"):
            self.report({'ERROR'}, "Select an existing design or variation in the project library")
            return {'CANCELLED'}
        if obj is None or obj.type not in {'POINTCLOUD', 'MESH'}:
            self.report({'ERROR'}, "Select a native Gaussian splat or editable mesh")
            return {'CANCELLED'}
        source_project = obj.get("quest_splat_project_id", "")
        if source_project and source_project != scene.quest_splat_project_id:
            self.report({'ERROR'}, "Selected model belongs to a different project")
            return {'CANCELLED'}
        try:
            if obj.type == 'POINTCLOUD':
                target = Path(scene.quest_splat_export_ply).expanduser().resolve()
                context.view_layer.update()
                ply, sidecar, _count = export_gaussian_ply(obj, target)
                transform = [value for row in json.loads(sidecar.read_text(encoding="utf-8"))["matrix_world_row_major"] for value in row]
                format = "gaussian_ply"
                artifact = ply
            else:
                target = Path(scene.quest_splat_mesh_glb).expanduser().resolve()
                if target.suffix.lower() != '.glb' or not target.parent.is_dir():
                    raise ValueError("Choose a .glb path in an existing folder")
                _export_mesh_glb(context, obj, target)
                # GLB already embeds the object's placement in its scene node.
                transform = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
                format = "glb"
                artifact = target
            name = scene.quest_splat_revision_name.strip()
            if not name:
                raise ValueError("Enter a human-readable revision name")
        except (ExportUnavailable, OSError, ValueError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            _state.message = str(exc)
            return {'CANCELLED'}
        project_id, entry_id, entry_name = scene.quest_splat_project_id, entry["id"], entry["name"]
        source_job_id = _source_job_for_object(obj)
        scene_name = _scene_name(context)
        def work(client):
            revision = client.publish_model(
                project_id, entry_id, artifact, format=format, transform=transform,
                coordinate_frame="colmap", source_job_id=source_job_id, display_name=name)
            return {"scene_name": scene_name, "revision": revision, "entry_name": entry_name,
                    "project_id": project_id, "entry_id": entry_id,
                    "models": client.list_models(project_id, entry_id)}
        _submit(context, "publish_model", work)
        return {'FINISHED'}


class QSB_PT_panel(bpy.types.Panel):
    bl_label = "Quest Scan"
    bl_idname = "QSB_PT_quest_scan"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Quest Scan"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "quest_splat_bridge_url", text="PC Bridge")
        layout.label(text="Token: environment variable", icon='LOCKED')
        layout.label(text=_state.message[:110])
        if _state.busy:
            layout.label(text=f"{_state.busy} request(s) running", icon='TIME')
        layout.operator("qsb.refresh_projects", text="Connect / Refresh Projects", icon='FILE_REFRESH')
        layout.prop(scene, "quest_splat_name", text="Name")
        layout.operator("qsb.create_project", text="New Project", icon='ADD')

        projects_box = layout.box()
        projects_box.label(text="Projects")
        for project in _state.projects:
            row = projects_box.row()
            row.label(text=project["name"])
            op = row.operator("qsb.open_project", text="Open")
            op.project_id = project["id"]
        local = layout.box()
        local.label(text="Local Gaussian Scan (Blender 5.3)")
        local.prop(scene, "quest_splat_local_ply", text="Import PLY")
        local.operator("qsb.import_local_ply", text="Import Native Splat")
        local.prop(scene, "quest_splat_export_ply", text="Export PLY")
        local.operator("qsb.export_gaussian_ply", text="Export Edited Native Splat")
        local.label(text="Export includes required transform JSON.")
        local.operator("qsb.rough_mesh", text="Experimental Rough Mesh")
        local.label(text="Rough, untextured; not architectural geometry.")
        local.prop(scene, "quest_splat_mesh_glb", text="Mesh GLB")
        local.operator("qsb.export_mesh_glb", text="Export Selected Mesh GLB")
        if not _state.project or scene.quest_splat_project_id != _state.project["id"]:
            if scene.quest_splat_project_id:
                layout.label(text="Open saved project to resume.")
            return

        library = layout.box()
        library.label(text="Project Library", icon='OUTLINER_COLLECTION')
        try:
            view = catalog_view(_state.project, _state.entries, scene.quest_splat_parent_id or None)
        except ValueError:
            scene.quest_splat_parent_id = ""
            view = catalog_view(_state.project, _state.entries, None)
        library.label(text=" › ".join(view.breadcrumb)[:110])
        if view.parent_id:
            parent = next((e for e in _state.entries if e["id"] == view.parent_id), None)
            op = library.operator("qsb.navigate", text="Back", icon='BACK')
            op.entry_id = parent["parent_id"] if parent and parent["parent_id"] else ""
        for entry in view.children:
            row = library.row()
            row.label(text=f"{entry['name']}  ({entry['kind']})")
            op = row.operator("qsb.select_entry", text="Select")
            op.entry_id = entry["id"]
            if entry["kind"] in {"category", "design", "variation"}:
                op = row.operator("qsb.navigate", text="Open")
                op.entry_id = entry["id"]
            op = row.operator("qsb.reorder_entry", text="", icon='TRIA_UP')
            op.entry_id, op.direction = entry["id"], -1
            op = row.operator("qsb.reorder_entry", text="", icon='TRIA_DOWN')
            op.entry_id, op.direction = entry["id"], 1
        parent = next((e for e in _state.entries if e["id"] == view.parent_id), None)
        selected = next((e for e in _state.entries if e["id"] == scene.quest_splat_entry_id), None)
        if selected:
            library.label(text=f"Selected: {selected['name']}")
            if not selected.get("protected"):
                library.operator("qsb.rename_entry", text="Rename Selected")
                destination_kind = "root" if parent is None else parent["kind"]
                valid_parent = (
                    (selected["kind"] == "category" and destination_kind == "root") or
                    (selected["kind"] == "design" and destination_kind == "category") or
                    (selected["kind"] == "variation" and destination_kind in {"design", "variation"})
                )
                if valid_parent and selected["parent_id"] != view.parent_id:
                    library.operator("qsb.move_entry_here", text="Move Selected Here")
                if selected["kind"] in {"design", "variation"}:
                    library.prop(scene, "quest_splat_revision_name", text="Revision Name")
                    library.operator("qsb.publish_model", text="Save Selected Model Here")
                    library.label(text="Unverified placement; Colmap frame only.")
                    library.operator("qsb.refresh_models", text="Refresh Saved Revisions")
                    if _state.models_entry_id == selected["id"]:
                        for model in _state.models[:12]:
                            library.label(text=f"{model.get('display_name') or model['format']} ({model['format']})")
        kind = "category" if parent is None else "design" if parent["kind"] == "category" else "variation"
        op = library.operator("qsb.create_entry", text=f"New {kind.title()}", icon='ADD')
        op.kind = kind
        library.operator("qsb.refresh_entries", text="Refresh Library", icon='FILE_REFRESH')
        library.label(text="Entries are organization; no mesh is implied.")

        sessions = layout.box()
        sessions.label(text="Capture Sessions", icon='CAMERA_DATA')
        sessions.prop(scene, "quest_splat_tracking_origin_id", text="Quest Origin UUID")
        sessions.operator("qsb.start_session", text="Start Session")
        for session in _state.sessions:
            row = sessions.row()
            row.label(text=f"{session['created_utc'][:16]}  {session['status']}  ({session['capture_count']})")
            op = row.operator("qsb.open_session", text="Open")
            op.session_id = session["id"]
        if scene.quest_splat_session_id:
            sessions.operator("qsb.refresh_session", text="Refresh Count")
            if _state.session and _state.session["id"] == scene.quest_splat_session_id:
                sessions.label(text=f"Received: {_state.session['capture_count']}")
                if _state.session["status"] == "capturing":
                    sessions.operator("qsb.finish_session", text="Finish Session")
            sessions.operator("qsb.check_preview", text="Check Latest Scan")
            if _state.preview:
                sessions.label(text=f"Splat: {_state.preview['status']}")
                if _state.preview["status"] == "not_ready":
                    sessions.label(text=_state.preview.get("reason", "Reconstruction pending.")[:90])
                else:
                    sessions.operator("qsb.load_scan", text="Load Latest Scan")
        layout.label(text="Bridge publication awaits an artifact endpoint.")


_CLASSES = (
    QSB_OT_refresh_projects, QSB_OT_create_project, QSB_OT_open_project,
    QSB_OT_refresh_entries, QSB_OT_create_entry, QSB_OT_rename_entry,
    QSB_OT_navigate, QSB_OT_move_entry_here, QSB_OT_reorder_entry,
    QSB_OT_select_entry, QSB_OT_refresh_models, QSB_OT_start_session,
    QSB_OT_open_session, QSB_OT_refresh_session, QSB_OT_finish_session,
    QSB_OT_check_preview, QSB_OT_load_scan, QSB_OT_import_local_ply,
    QSB_OT_export_gaussian_ply, QSB_OT_rough_mesh, QSB_OT_export_mesh_glb, QSB_OT_publish_model,
    QSB_PT_panel,
)


def register():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quest-splat-http")
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.quest_splat_bridge_url = bpy.props.StringProperty(name="PC Bridge", default="http://127.0.0.1:8765")
    bpy.types.Scene.quest_splat_name = bpy.props.StringProperty(name="Name", default="")
    bpy.types.Scene.quest_splat_project_id = bpy.props.StringProperty(name="Project ID", default="")
    bpy.types.Scene.quest_splat_session_id = bpy.props.StringProperty(name="Session ID", default="")
    bpy.types.Scene.quest_splat_parent_id = bpy.props.StringProperty(name="Library Parent ID", default="")
    bpy.types.Scene.quest_splat_entry_id = bpy.props.StringProperty(name="Selected Entry ID", default="")
    bpy.types.Scene.quest_splat_tracking_origin_id = bpy.props.StringProperty(name="Quest Origin UUID", default="")
    bpy.types.Scene.quest_splat_local_ply = bpy.props.StringProperty(name="Gaussian PLY", subtype='FILE_PATH', default="")
    bpy.types.Scene.quest_splat_export_ply = bpy.props.StringProperty(name="Edited Gaussian PLY", subtype='FILE_PATH', default="")
    bpy.types.Scene.quest_splat_mesh_glb = bpy.props.StringProperty(name="Editable Mesh GLB", subtype='FILE_PATH', default="")
    bpy.types.Scene.quest_splat_revision_name = bpy.props.StringProperty(name="Revision Name", default="")
    if not bpy.app.timers.is_registered(_poll_results):
        bpy.app.timers.register(_poll_results, persistent=True)


def unregister():
    global _executor
    if bpy.app.timers.is_registered(_poll_results):
        bpy.app.timers.unregister(_poll_results)
    for attr in (
        "quest_splat_bridge_url", "quest_splat_name", "quest_splat_project_id",
        "quest_splat_session_id", "quest_splat_parent_id", "quest_splat_entry_id",
        "quest_splat_tracking_origin_id",
        "quest_splat_local_ply",
        "quest_splat_export_ply",
        "quest_splat_mesh_glb",
        "quest_splat_revision_name",
    ):
        delattr(bpy.types.Scene, attr)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
