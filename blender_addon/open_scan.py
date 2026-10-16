"""Blender startup script used by the local dashboard's Open in Blender button."""

from __future__ import annotations

import sys
import argparse
import json
import shutil
from pathlib import Path


def render_geometry_reviews(bpy, scene, cameras, output_directory):
    """Render texture-free surfaces without altering the saved mesh or materials."""
    render_dir = Path(output_directory); render_dir.mkdir(parents=True, exist_ok=True)
    engine, camera, filepath = scene.render.engine, scene.camera, scene.render.filepath
    rendered = []
    try:
        scene.render.engine = 'BLENDER_WORKBENCH'
        shading = scene.display.shading
        shading.light = 'STUDIO'; shading.color_type = 'SINGLE'
        shading.single_color = (0.6, 0.6, 0.6)
        shading.show_shadows = True; shading.show_cavity = True
        shading.cavity_type = 'BOTH'; shading.show_specular_highlight = False
        for view in cameras:
            scene.camera = view
            path = render_dir/(view['capture_id']+'-geometry.png')
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            rendered.append({'capture_id': view['capture_id'], 'render': str(path)})
    finally:
        scene.render.engine = engine; scene.camera = camera; scene.render.filepath = filepath
    return rendered


def open_textured_mesh(bpy, scan, mesh_path, project_id, output, review_captures, mesh_title='Room scan', review_geometry=False, detail_mesh=None):
    """Assemble a separate editable delivery, keeping the scan and raw surface."""
    import bmesh
    from mathutils import Matrix, Vector
    mesh_path = Path(mesh_path).resolve(strict=True)
    manifest = json.loads((mesh_path.parent/'mesh.json').read_text())
    from quest_splat_bridge.import_adapter import _sha256
    if manifest['project_id'] != project_id or _sha256(mesh_path) != manifest['artifact']['sha256']:
        raise ValueError('The textured mesh does not match its verified project receipt')
    source_splat = mesh_path.parents[2]/'splat.ply'
    if (scan.get('quest_splat_project_id') != project_id or
            scan.get('quest_splat_source_job_id') != manifest['source_job_id'] or
            scan.get('quest_splat_source_sha256') != _sha256(source_splat)):
        raise ValueError('The retained Gaussian must match the mesh project, source job and file hash')
    delivery_root = Path(output).resolve().parent
    delivery_root.mkdir(parents=True, exist_ok=True)
    portable_glb = delivery_root/(Path(output).stem+'.glb')
    if not portable_glb.exists(): shutil.copyfile(mesh_path, portable_glb)
    if _sha256(portable_glb) != manifest['artifact']['sha256']:
        raise ValueError('The portable mesh copy differs from the immutable revision')
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(portable_glb))
    created = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
    if len(created) != 1:
        raise ValueError('Expected one textured room surface')
    mesh = created[0]
    mesh.name = mesh_title+(' - Detail-preserving comparison' if manifest.get('detail_preserving_room')
                           else ' - Cleaned Textured Mesh')
    root = next(c for c in bpy.data.collections if c.get('quest_splat_role') == 'project'
                and c.get('quest_splat_project_id') == project_id)
    root.name = mesh_title+' - Room Scan'
    base = bpy.data.collections.new('BaseMesh - Detail-preserving comparison' if manifest.get('detail_preserving_room')
                                    else 'BaseMesh - Cleaned and Textured')
    base['quest_splat_role'] = 'base_mesh'; root.children.link(base)
    for c in list(mesh.users_collection): c.objects.unlink(mesh)
    base.objects.link(mesh)
    # Restore shared geometric vertices after glTF's UV-seam splitting. Loop UVs remain independent.
    bm = bmesh.new(); bm.from_mesh(mesh.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    bm.to_mesh(mesh.data); bm.free(); mesh.data.update()
    mesh['quest_splat_project_id'] = project_id
    mesh['quest_splat_source_job_id'] = manifest['source_job_id']
    mesh['quest_splat_mesh_revision_id'] = manifest['revision_id']
    mesh['quest_splat_coordinate_frame'] = manifest['coordinate_frame']
    mesh['quest_splat_physical_measurements'] = 'unverified'
    retained_detail = None
    if detail_mesh:
        detail_path = Path(detail_mesh).resolve(strict=True)
        detail_manifest = json.loads((detail_path.parent/'mesh.json').read_text())
        if (detail_manifest['project_id'] != project_id or
            detail_manifest['source_job_id'] != manifest['source_job_id'] or
            detail_manifest['coordinate_frame'] != manifest['coordinate_frame'] or
            _sha256(detail_path) != detail_manifest['artifact']['sha256']):
            raise ValueError('Detailed scan must be verified in the same project, job and coordinate frame')
        portable_detail = delivery_root/'Retained-Detailed-Photo-Scan.glb'
        if not portable_detail.exists(): shutil.copyfile(detail_path, portable_detail)
        if _sha256(portable_detail) != detail_manifest['artifact']['sha256']:
            raise ValueError('Retained detail copy differs from its immutable revision')
        previous = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(portable_detail))
        imported = [o for o in bpy.data.objects if o not in previous and o.type == 'MESH']
        if len(imported) != 1: raise ValueError('Expected one retained detailed scan')
        detail = imported[0]; detail.name = 'Detailed Photo Reconstruction - Retained'
        detail_collection = bpy.data.collections.new('DetailedScan - Retained Photo Reconstruction')
        root.children.link(detail_collection)
        for collection in list(detail.users_collection): collection.objects.unlink(detail)
        detail_collection.objects.link(detail)
        detail_collection.hide_viewport = True; detail_collection.hide_render = True
        detail.hide_render = True
        detail['quest_splat_mesh_revision_id'] = detail_manifest['revision_id']
        detail['quest_splat_original_preserved'] = True
        retained_detail = {'name': detail.name, 'sha256': detail_manifest['artifact']['sha256'],
                           'triangles': len(detail.data.polygons), 'revision_id': detail_manifest['revision_id']}
    raw_path = Path(manifest['cleanup']['raw_surface_retained'])
    portable_raw = delivery_root/'Raw-Room-Surface.ply'
    if not portable_raw.exists(): shutil.copyfile(raw_path, portable_raw)
    if _sha256(portable_raw) != _sha256(raw_path):
        raise ValueError('The retained raw surface copy differs from its immutable source')
    raw_before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=str(portable_raw), forward_axis='Y', up_axis='Z')
    raw = next(o for o in bpy.data.objects if o not in raw_before and o.type == 'MESH')
    raw.name = 'Raw reconstructed surface - retained'
    raw.matrix_world = Matrix(manifest['original_colmap_to_training'])
    archive = bpy.data.collections.new('RawSurface - Retained Original')
    root.children.link(archive)
    for c in list(raw.users_collection): c.objects.unlink(raw)
    archive.objects.link(raw); archive.hide_viewport = True; archive.hide_render = True
    raw.hide_render = True
    scan.name = 'Original Gaussian Splat - Retained'
    scan.hide_set(True); scan.hide_render = True
    scan['quest_splat_original_preserved'] = True
    bpy.ops.object.select_all(action='DESELECT'); mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    if not mesh.data.uv_layers or not mesh.data.materials:
        raise ValueError('The mesh did not retain its UVs and photo material')
    for image in bpy.data.images:
        if image.source == 'FILE' and image.has_data: image.pack()

    job_root = mesh_path.parents[2]
    job = json.loads((job_root/'job.json').read_text())
    processed = job_root/'processed'
    cameras = json.loads((processed/'transforms.json').read_text())
    # Use the verified receipt instead of a stale or Windows-overlength trainer path.
    # The receipt includes scale in both its linear part and translation.
    original_to_training = Matrix(manifest['original_colmap_to_training'])
    reconstruction_scale = original_to_training.to_3x3().col[0].length
    original_to_saved = Matrix([*cameras['applied_transform'], [0, 0, 0, 1]])
    saved_to_training = original_to_training @ original_to_saved.inverted()
    frame_map = job.get('frame_manifest') or {
        c['capture_id']: f"frame_{i:05d}{Path(c['image_name']).suffix}"
        for i, c in enumerate(sorted(job['input_captures'], key=lambda c: c['image_name']), 1)}
    by_name = {Path(f['file_path']).name: f for f in cameras['frames']}
    chosen = [v for v in review_captures.split(',') if v in frame_map]
    if not chosen: chosen = list(frame_map)[::max(1, len(frame_map)//3)][:3]
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.cycles.samples = 8
    scene.render.resolution_x = 1200; scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'Standard'
    scene.world.color = (0.015, 0.015, 0.015)
    scene.unit_settings.system = 'NONE'
    scene['quest_splat_project_id'] = project_id
    scene['quest_splat_source_job_id'] = job['job_id']
    scene['quest_splat_mesh_revision_id'] = manifest['revision_id']
    scene['quest_splat_geometry_limits'] = 'Captured surfaces; metric accuracy and unobserved geometry unverified'
    if manifest.get('detail_preserving_room'):
        scene['quest_splat_geometry_limits'] = 'Supported wall/floor interiors constrained; scanned objects and gaps retained; dimensions unmeasured'
        mesh['quest_splat_detail_preserving_room'] = json.dumps(manifest['detail_preserving_room'])
        notes = bpy.data.texts.new('Detail Preservation - Read Me')
        notes.write('The visible mesh retains the full photo-reconstructed scene.\n'
                    'Only locally supported wall/floor interiors are constrained to the fitted room planes.\n'
                    'Non-wall triangles, open doors, raised objects and their shared boundary vertices are protected.\n'
                    'No replacement box, hidden wall infill or ceiling is generated.\n'
                    'Existing reconstruction noise, uncertain edges and unseen surfaces can remain.\n'
                    'This is a preservation comparison, not an accepted clean architectural mesh.\n'
                    'The original Gaussian and source mesh are retained.\n')
    if manifest.get('architectural_structure'):
        scene['quest_splat_geometry_limits'] = 'Photo-fitted structural prior: hidden wall/floor surfaces inferred, dimensions unmeasured, no ceiling'
        mesh['quest_splat_architectural_structure'] = json.dumps(manifest['architectural_structure'])
        notes = bpy.data.texts.new('Room Structure - Read Me')
        notes.write('The visible layer fits straight perpendicular walls and floor to observed photo depth.\n'
                    'It infers structure behind occluders; this is not a measured architectural survey.\n'
                    'Room width and length are estimated independently, not forced equal.\n'
                    'No ceiling is generated. Photo textures can retain foreground objects and lighting.\n'
                    'The separate DetailedScan layer retains the full photo reconstruction and object geometry.\n'
                    'The original Gaussian and raw surface are retained in their own layers.\n')
    review = bpy.data.collections.new('Review Cameras'); scene.collection.children.link(review)
    first = None
    for capture in chosen:
        frame = by_name[frame_map[capture]]
        pose = saved_to_training @ Matrix(frame['transform_matrix'])
        for row in range(3):
            for column in range(3): pose[row][column] /= reconstruction_scale
        data = bpy.data.cameras.new('Captured viewpoint')
        camera = bpy.data.objects.new('View '+capture[:8], data); review.objects.link(camera)
        camera.matrix_world = pose
        width = frame.get('w', cameras['w']); height = frame.get('h', cameras['h'])
        data.type = 'PERSP'; data.sensor_fit = 'HORIZONTAL'; data.sensor_width = 36
        data.lens = frame.get('fl_x', cameras['fl_x']) / width * 36
        data.shift_x = -(frame.get('cx', cameras['cx']) - width/2) / width
        data.shift_y = (frame.get('cy', cameras['cy']) - height/2) / width
        data.clip_start = 0.005; data.clip_end = 1000
        camera['capture_id'] = capture
        first = first or camera
    overview_target = None
    if manifest.get('architectural_structure') or manifest.get('detail_preserving_room'):
        structure = manifest.get('detail_preserving_room') or manifest['architectural_structure']
        basis = Matrix(structure['room_basis'])
        low, high = [Vector(row) for row in structure['room_bounds']]
        overview_target = original_to_training @ (basis @ ((low+high)*.5))
        directions = original_to_training.to_3x3() @ basis
        span = high-low
        offset = (directions.col[0]*span.x*.85 + directions.col[1]*span.y*.85 +
                  directions.col[2]*max(span.x,span.y)*1.15)
        data = bpy.data.cameras.new('Room overview - derived inspection view')
        data.type = 'ORTHO'; data.ortho_scale = max(span.x,span.y)*reconstruction_scale*1.8
        data.clip_start = .005; data.clip_end = 1000
        overview = bpy.data.objects.new('Room Overview',data)
        review.objects.link(overview)
        overview.location = overview_target+offset
        overview.rotation_euler = (-offset).to_track_quat('-Z','Y').to_euler()
        overview['capture_id'] = 'architectural-overview'
        overview['view_provenance'] = 'Derived overview, not a captured photograph'
        if manifest.get('architectural_structure'):
            first = overview
        else:
            # A full scan can contain uncertain outer shells. Start inside from
            # an actual captured camera; keep the external overview for review.
            overview_target = None
    bpy.context.view_layer.update()
    scene.camera = first
    # Fresh files have several workspaces; initialize each saved 3D view so
    # changing workspace cannot return to the factory camera or orbit center.
    for area in [a for screen in bpy.data.screens for a in screen.areas]:
        if area.type == 'VIEW_3D':
            space = area.spaces.active
            space.lens = first.data.lens
            space.shading.type = 'MATERIAL'; space.overlay.show_overlays = False
            region = space.region_3d
            region.view_perspective = 'PERSP'
            region.view_rotation = first.matrix_world.to_quaternion()
            if overview_target is not None:
                region.view_location = overview_target
                region.view_distance = (first.location-overview_target).length
            else:
                region.view_distance = 1
                region.view_location = first.location - first.matrix_world.to_3x3().col[2]
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    render_dir = output.parent/'review'; render_dir.mkdir(exist_ok=True)
    rendered = []
    for camera in review.objects:
        scene.camera = camera
        path = render_dir/(camera['capture_id']+'.png')
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append({'capture_id': camera['capture_id'], 'render': str(path),
                         'source_photo': (str(processed/'images'/frame_map[camera['capture_id']])
                                          if camera['capture_id'] in frame_map else None)})
    geometry_renders = []
    if review_geometry:
        # Inspect the actual surface independently of photo textures. Keep this
        # diagnostic shading out of the saved editable/textured deliverable.
        geometry_renders = render_geometry_reviews(bpy, scene, review.objects, render_dir)
    scene.camera = first
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    audit = {'blend': str(output), 'mesh': mesh.name, 'vertices': len(mesh.data.vertices),
             'triangles': len(mesh.data.polygons), 'uv_layers': len(mesh.data.uv_layers),
             'materials': len(mesh.data.materials), 'gaussians_retained': len(scan.data.points),
             'source_splat_sha256': scan['quest_splat_source_sha256'],
             'raw_surface_retained': raw.name, 'packed_textures': [i.name for i in bpy.data.images if i.packed_file],
             'detailed_scan_retained': retained_detail,
             'renders': rendered, 'geometry_renders': geometry_renders,
             'geometry_visual_acceptance': 'pending review of rendered surfaces'}
    (output.parent/'blender-review.json').write_text(json.dumps(audit, indent=2))
    print('QUEST_MESH_BLENDER_REVIEW='+json.dumps(audit), flush=True)


def main() -> None:
    import bpy
    from mathutils import Vector

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from quest_splat_bridge.import_adapter import import_splat_file

    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('project_id')
    parser.add_argument('--mesh'); parser.add_argument('--output')
    parser.add_argument('--review-captures', default='')
    parser.add_argument('--mesh-title', default='Room scan')
    parser.add_argument('--review-geometry', action='store_true')
    parser.add_argument('--detail-mesh', help='Retain a verified detailed photo mesh as a hidden layer')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source, project_id = args.source, args.project_id
    obj = import_splat_file(bpy.context, source, project_id)
    for previous in list(bpy.data.objects):
        if previous != obj and previous.name in {"Cube", "Camera", "Light"}:
            bpy.data.objects.remove(previous, do_unlink=True)
    if args.mesh:
        if not args.output: raise ValueError('A mesh delivery needs its own .blend output path')
        open_textured_mesh(bpy, obj, args.mesh, project_id, args.output, args.review_captures, args.mesh_title, args.review_geometry, args.detail_mesh)
        return
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    radius = max((corner - center).length for corner in corners)
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_location = center
            area.spaces.active.region_3d.view_distance = max(1.0, radius * 2.5)
    print(f"Quest scan opened in Blender: {len(obj.data.points)} Gaussians", flush=True)


if __name__ == '__main__':
    main()
