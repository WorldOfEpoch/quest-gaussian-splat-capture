"""Photo-derived COLMAP MVS mesh, sharing the existing GPU lock and source jobs.

Dense stereo estimates depth from calibrated photos; no Gaussian center surface.
All outputs are new immutable mesh revisions with independently verified faces.
"""
from __future__ import annotations
import hashlib
import json
import os
import struct
import uuid
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from fastapi import Depends
from fastapi.responses import FileResponse
from .store import Store, StoreError
from .models import utc_now
from .reconstruct import ReconstructionError, _worker_lock, _run, _tool


def ply_triangles(path: Path):
    """Read COLMAP ASCII/binary PLY; reject nonfinite, degenerate, invalid faces."""
    sizes = {'char':'b','uchar':'B','short':'h','ushort':'H','int':'i','uint':'I',
             'float':'f','double':'d','int8':'b','uint8':'B','int32':'i','uint32':'I','float32':'f','float64':'d'}
    with path.open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ReconstructionError('Mesh is not PLY')
        elements, fmt = [], None
        for _ in range(4096):
            line = f.readline().decode('ascii').strip().split()
            if not line:
                raise ReconstructionError('Truncated mesh header')
            if line[0] == 'format': fmt = line[1]
            elif line[0] == 'element':
                n = int(line[2])
                if n < 0 or n > 10000000: raise ReconstructionError('Mesh budget exceeded')
                elements.append((line[1], n, []))
            elif line[0] == 'property': elements[-1][2].append(line[1:])
            elif line[0] == 'end_header': break
        else: raise ReconstructionError('Mesh header exceeds budget')
        if fmt not in {'ascii', 'binary_little_endian'}: raise ReconstructionError('Unsupported PLY encoding')
        vertices, colors, faces = [], [], []
        def scalar(kind, tokens):
            if kind not in sizes: raise ReconstructionError('Unsupported PLY scalar')
            if tokens is not None: return float(tokens.pop(0)) if sizes[kind] in 'fd' else int(tokens.pop(0))
            codec = '<' + sizes[kind]
            buf = f.read(struct.calcsize(codec))
            if len(buf) != struct.calcsize(codec): raise ReconstructionError('Mesh data truncated')
            return struct.unpack(codec, buf)[0]
        for name, count, props in elements:
            for _ in range(count):
                tokens = f.readline().decode('ascii').split() if fmt == 'ascii' else None
                row = {}
                for prop in props:
                    if prop[0] == 'list':
                        n = int(scalar(prop[1], tokens))
                        if n < 0 or n > 1024: raise ReconstructionError('Mesh face list invalid')
                        row[prop[3]] = [scalar(prop[2], tokens) for _ in range(n)]
                    else: row[prop[1]] = scalar(prop[0], tokens)
                if name == 'vertex':
                    vertices.append([row[k] for k in ('x','y','z')])
                    colors.append([row.get(k, 180) / 255 for k in ('red','green','blue')])
                elif name == 'face':
                    indices = row.get('vertex_indices', row.get('vertex_index'))
                    if indices is None or len(indices) != 3: raise ReconstructionError('Mesh must contain triangles')
                    faces.append(indices)
    v, c, t = np.asarray(vertices,dtype='<f4'), np.asarray(colors,dtype='<f4'), np.asarray(faces,dtype='<u4')
    if len(v) < 3 or len(t) < 1 or not np.isfinite(v).all() or not np.isfinite(c).all():
        raise ReconstructionError('Mesh has no valid surface')
    if t.max() >= len(v): raise ReconstructionError('Mesh face index out of bounds')
    area = np.linalg.norm(np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]),axis=1)
    keep = area > 1e-12
    t = t[keep]
    if not len(t): raise ReconstructionError('All mesh triangles are degenerate')
    return v, c, t


def mesh_to_glb(source: Path, destination: Path, provenance: dict):
    v, c, t = ply_triangles(source)
    buffers = [v.tobytes(), c.tobytes(), t.tobytes()]
    binary = b''.join(buffers)
    offsets = [0,len(buffers[0]),len(buffers[0])+len(buffers[1])]
    doc = {'asset':{'version':'2.0','generator':'QuestSplat photo mesh'},
           'buffers':[{'byteLength':len(binary)}],
           'bufferViews':[{'buffer':0,'byteOffset':o,'byteLength':len(b),'target':34963 if i==2 else 34962}
                          for i,(o,b) in enumerate(zip(offsets,buffers))],
           'accessors':[{'bufferView':0,'componentType':5126,'count':len(v),'type':'VEC3','min':v.min(0).tolist(),'max':v.max(0).tolist()},
                        {'bufferView':1,'componentType':5126,'count':len(c),'type':'VEC3'},
                        {'bufferView':2,'componentType':5125,'count':int(t.size),'type':'SCALAR'}],
           'meshes':[{'primitives':[{'attributes':{'POSITION':0,'COLOR_0':1},'indices':2,'mode':4}]}],
           'nodes':[{'mesh':0}], 'scenes':[{'nodes':[0]}], 'scene':0, 'extras':provenance}
    js = json.dumps(doc,separators=(',',':'),allow_nan=False).encode()
    js += b' ' * (-len(js)%4); binary += b'\0' * (-len(binary)%4)
    with destination.open('xb') as f:
        f.write(struct.pack('<III',0x46546c67,2,28+len(js)+len(binary)))
        f.write(struct.pack('<II',len(js),0x4e4f534a)); f.write(js)
        f.write(struct.pack('<II',len(binary),0x004e4942)); f.write(binary)
        f.flush(); os.fsync(f.fileno())
    return {'sha256':Store._file_sha256(destination),'size_bytes':destination.stat().st_size,
            'vertex_count':len(v),'triangle_count':len(t),'path':str(destination)}


def extend_photo_mesh_calibration(store: Store, project_id: str, source_job_id: str):
    """Register extra session photos on private copies, solely for offline meshing.

    Do not train/publish a Gaussian, change parent poses, or claim headset tracking.
    The live continuation's strict calibration and tracking checks stay enabled.
    """
    from .reconstruct import _prepare_incremental, _read_colmap_images, _colmap_camera_to_saved, _parent_frame_map
    project_id, source_job_id = str(uuid.UUID(project_id)), str(uuid.UUID(source_job_id))
    store.project(project_id)
    job_root = store.root/'projects'/project_id/'reconstructions'/source_job_id
    job = json.loads((job_root/'job.json').read_text())
    if job['project_id'] != project_id or job['status'] != 'ready_for_blender':
        raise ReconstructionError('A verified source job is required for extra photo registration')
    snapshot = store.export_session(job['session_id'])
    previous = job_root/'processed'
    protected = {p: Store._file_sha256(p) for p in previous.rglob('*') if p.is_file()}
    with _worker_lock(store.root):
        revision = str(uuid.uuid4()); root = job_root/'mesh-calibrations'/revision; root.mkdir(parents=True)
        work = Path(tempfile.mkdtemp(prefix='qs-calib-'))
        copied = work/'parent'; shutil.copytree(previous, copied)
        parent = {**job, 'processed_dir': str(copied), 'frame_manifest': _parent_frame_map(job)}
        manifest = {'revision_id': revision, 'source_job_id': source_job_id, 'project_id': project_id,
                    'created_utc': utc_now(), 'status': 'running', 'scope': 'offline_mesh_only',
                    'private_work_directory': str(work), 'original_calibrated_views': job['registered_views']}
        (root/'calibration.json').write_text(json.dumps(manifest, indent=2))
        try:
            result = _prepare_incremental(parent, snapshot, work,
                {'colmap': os.environ.get('QUEST_SPLAT_COLMAP_EXE') or _tool('colmap')},
                root/'worker.log', dict(os.environ), mesh_only=True)
            processed = Path(result['processed_dir'])
            # Mesh rendering expects raw saved camera poses; the checkpoint
            # helper's normalized poses are not used to train or publish here.
            cameras = json.loads((previous/'transforms.json').read_text())
            model = _read_colmap_images(work/'extended-colmap-text/images.txt')
            cameras['frames'] = [{'file_path': 'images/'+name,
                                'transform_matrix': _colmap_camera_to_saved(*model[name]).tolist()}
                                for name in result['frame_manifest'].values() if name in model]
            (processed/'transforms.json').write_text(json.dumps(cameras, indent=2))
            canonical = root/'processed'; shutil.copytree(processed, canonical)
            manifest.update(result); manifest['processed_dir'] = str(canonical)
            used = set(result['used_capture_ids'])
            manifest['input_captures'] = [{**{k:c['receipt'][k] for k in ('capture_id','image_sha256')},
                                          'image_name': Path(c['image_path']).name}
                                         for c in snapshot['captures'] if c['receipt']['capture_id'] in used]
            manifest['calibrated_hashes'] = {str(p.relative_to(canonical)): Store._file_sha256(p)
                                           for p in canonical.rglob('*') if p.is_file()}
            if any(Store._file_sha256(p) != digest for p, digest in protected.items()):
                raise ReconstructionError('Original calibration changed during registration')
            manifest['status'] = 'ready'
        except Exception as error:
            manifest['status'] = 'failed'; manifest['error'] = str(error)
            (root/'calibration.json').write_text(json.dumps(manifest, indent=2)); raise
        (root/'calibration.json').write_text(json.dumps(manifest, indent=2))
        return manifest


def _reuse_photo_depth(seed, dense, colmap, root, env):
    """Reuse photometric estimates only when pixels, cameras and poses match.

    Geometric depth must be recomputed against the expanded view set. File sizes
    and finite native map payloads are checked before copying retained estimates.
    """
    from .reconstruct import _read_colmap_images
    previous = Path(seed['private_work_directory'])/'dense'
    camera = dense/'sparse/cameras.bin'
    if Store._file_sha256(camera) != Store._file_sha256(previous/'sparse/cameras.bin'):
        raise ReconstructionError('Retained depth camera calibration differs')
    texts = []
    for label, folder in [('old', previous), ('new', dense)]:
        output = dense.parent/('depth-camera-'+label); output.mkdir()
        _run([colmap, 'model_converter', '--input_path', str(folder/'sparse'),
              '--output_path', str(output), '--output_type', 'TXT'], root/'worker.log', env, 120)
        texts.append(_read_colmap_images(output/'images.txt'))
    records = []
    for image in sorted((previous/'images').glob('*.jpg')):
        name = image.name
        if name not in texts[1] or Store._file_sha256(image) != Store._file_sha256(dense/'images'/name):
            raise ReconstructionError('Retained depth pixels differ from expanded calibration')
        if not all(np.allclose(a,b,atol=1e-6,rtol=0) for a,b in zip(texts[0][name],texts[1][name])):
            raise ReconstructionError('Retained depth camera pose moved')
        with Image.open(image) as im: width, height = im.size
        hashes = {}
        for category, channels in [('depth_maps',1), ('normal_maps',3)]:
            p = previous/'stereo'/category/(name+'.photometric.bin')
            payload = p.read_bytes(); header = payload.split(b'&',3)
            if (len(header) != 4 or [int(x) for x in header[:3]] != [width,height,channels] or
                    len(header[3]) != width*height*channels*4 or
                    not np.isfinite(np.frombuffer(header[3],dtype='<f4')).all()):
                raise ReconstructionError('Retained photometric map is invalid')
            target = dense/'stereo'/category/p.name
            shutil.copyfile(p,target); hashes[category] = Store._file_sha256(target)
        records.append({'image':name,'maps':hashes})
    return {'source_revision_id':seed['revision_id'],'reused_photometric_images':len(records),
            'geometric_maps_reused':0,'camera_sha256':Store._file_sha256(camera),'verified_maps':records}


def _native_map_valid(path, width, height, channels):
    try:
        with path.open('rb') as handle:
            start = handle.read(64)
            parts = start.split(b'&', 3)
            if len(parts) != 4 or [int(x) for x in parts[:3]] != [width, height, channels]: return False
            offset = sum(len(x)+1 for x in parts[:3])
            if path.stat().st_size != offset+width*height*channels*4: return False
            handle.seek(offset)
            return bool(np.isfinite(np.frombuffer(handle.read(), dtype='<f4')).all())
    except (OSError, ValueError): return False


def _inspect_depth_maps(dense):
    invalid = []
    for image in sorted((dense/'images').glob('*.jpg')):
        with Image.open(image) as im: width, height = im.size
        for kind in ('photometric', 'geometric'):
            for category, channels in (('depth_maps', 1), ('normal_maps', 3)):
                p = dense/'stereo'/category/(image.name+'.'+kind+'.bin')
                if not _native_map_valid(p, width, height, channels): invalid.append(p)
    return invalid


def _require_mesh_space(path, bytes_required):
    free = shutil.disk_usage(path).free
    if free < bytes_required:
        raise ReconstructionError(f'Mesh workspace needs {bytes_required/2**30:.1f} GiB free; '
                                  f'{free/2**30:.1f} GiB available. Use a larger scratch directory.')


def _complete_dense_mesh(manifest, work, root, colmap, env, stage):
    dense, log = work/'dense', root/'worker.log'
    settings = manifest['dense_settings']
    stage('estimating_depth')
    _run([colmap,'patch_match_stereo','--workspace_path',str(dense),'--workspace_format','COLMAP',
          '--PatchMatchStereo.geom_consistency','true',
          '--PatchMatchStereo.gpu_index', ','.join(['0']*settings.get('gpu_workers', 1)),
          '--PatchMatchStereo.num_iterations',str(settings['iterations']),
          '--PatchMatchStereo.window_step',str(settings['window_step'])],log,env,14400)
    invalid = _inspect_depth_maps(dense)
    if invalid:
        raise ReconstructionError(f'{len(invalid)} depth/normal maps are incomplete or invalid; '
                                  'preserve the workspace and resume after repairing its storage')
    _require_mesh_space(root, 2*2**30)
    stage('fusing_depth')
    _run([colmap,'stereo_fusion','--workspace_path',str(dense),'--workspace_format','COLMAP',
          '--input_type','geometric','--output_path',str(dense/'fused.ply')],log,env,3600)
    stage('triangulating_surface')
    _run([colmap,'delaunay_mesher','--input_path',str(dense),'--input_type','dense',
          '--output_path',str(work/'mesh.ply')],log,env,3600)
    shutil.copyfile(work/'mesh.ply',root/'mesh.ply')
    stage('exporting_triangles')
    provenance = {k:manifest[k] for k in ('method','source_job_id','input_captures','coordinate_frame','uncertainty')}
    manifest['artifact'] = mesh_to_glb(root/'mesh.ply',root/'mesh.glb',provenance)
    manifest['status']='ready'; stage('ready')


def resume_mesh(store, project_id, source_job_id, revision_id):
    """Repair interrupted generated maps in one unfinished mesh workspace."""
    project_id, source_job_id, revision_id = map(lambda x: str(uuid.UUID(x)),
                                               (project_id, source_job_id, revision_id))
    root = store.root/'projects'/project_id/'reconstructions'/source_job_id/'mesh-revisions'/revision_id
    m = json.loads((root/'mesh.json').read_text())
    if (m['status'] != 'failed' or m.get('artifact') or (root/'mesh.glb').exists() or
            m['project_id'] != project_id or m['source_job_id'] != source_job_id):
        raise ReconstructionError('Only an unfinished failed mesh can resume')
    private_path = Path(m['private_work_directory'])
    work = private_path.resolve(strict=True)
    if private_path.is_symlink() or not work.name.startswith('qs-mvs-'):
        raise ReconstructionError('Unexpected private mesh workspace')
    # Original session hashes and every staged calibrated photograph must agree.
    job = json.loads((root.parents[1]/'job.json').read_text())
    store.export_session(job['session_id'])
    for photo in m['calibrated_photos']:
        path = work/'images'/Path(photo['image_name']).name
        if Store._file_sha256(path) != photo['sha256']:
            raise ReconstructionError('Staged calibrated photo changed before resume')
    for relative, digest in m.get('workspace_input_hashes', {}).items():
        path = (work/relative).resolve()
        if not path.is_relative_to(work) or Store._file_sha256(path) != digest:
            raise ReconstructionError('Native mesh camera/image inputs changed before resume')
    retained_camera = m.get('retained_photo_depth', {}).get('camera_sha256')
    if retained_camera and Store._file_sha256(work/'dense/sparse/cameras.bin') != retained_camera:
        raise ReconstructionError('Retained depth camera calibration changed before resume')
    with _worker_lock(store.root):
        invalid = _inspect_depth_maps(work/'dense')
        _require_mesh_space(work, 4*2**30)
        _require_mesh_space(root, 2*2**30)
        quarantine = work/'incomplete-maps'/str(uuid.uuid4())
        records = []
        for path in invalid:
            if not path.resolve().is_relative_to(work/'dense/stereo'):
                raise ReconstructionError('Invalid map escaped the generated workspace')
            record = {'path': str(path.relative_to(work)), 'missing': not path.exists()}
            if path.exists():
                record.update(size_bytes=path.stat().st_size, sha256=Store._file_sha256(path))
                retained = quarantine/path.parent.name/path.name; retained.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(retained)); record['retained_at'] = str(retained)
            records.append(record)
        m.setdefault('resume_events', []).append({'utc': utc_now(), 'previous_error': m.get('error'),
                                                 'invalid_maps': records})
        m.pop('error', None); m['status'] = 'running'
        def stage(value):
            m['stage'] = value
            temporary = root/'mesh.tmp'; temporary.write_text(json.dumps(m, indent=2))
            os.replace(temporary, root/'mesh.json')
        try:
            _complete_dense_mesh(m, work, root, os.environ.get('QUEST_SPLAT_COLMAP_EXE') or _tool('colmap'),
                                 dict(os.environ), stage)
        except Exception as error:
            m['status'] = 'failed'; m['error'] = str(error); stage('failed'); raise
        return m


def build_mesh(store: Store, project_id: str, source_job_id: str, max_image_size=1600,
               stereo_sources=20, stereo_iterations=5, stereo_window_step=1, calibration_revision_id=None,
               stereo_gpu_workers=1, reuse_photo_depth_revision_id=None):
    project_id, source_job_id = str(uuid.UUID(project_id)), str(uuid.UUID(source_job_id))
    if not 640 <= max_image_size <= 3200: raise ReconstructionError('Invalid dense photo budget')
    if (not 3 <= stereo_sources <= 30 or not 2 <= stereo_iterations <= 10 or
            stereo_window_step not in (1, 2) or stereo_gpu_workers not in (1, 2)):
        raise ReconstructionError('Invalid stereo processing budget')
    store.project(project_id)
    seed = None
    if reuse_photo_depth_revision_id:
        seed, _ = read_mesh(store,project_id,source_job_id,reuse_photo_depth_revision_id)
        expected={'max_image_size':max_image_size,'source_views':stereo_sources,
                  'iterations':stereo_iterations,'window_step':stereo_window_step}
        if any(seed.get('dense_settings',{}).get(k) != v for k,v in expected.items()):
            raise ReconstructionError('Retained depth processing settings differ')
    source_root = store.root / 'projects' / project_id / 'reconstructions' / source_job_id
    job = json.loads((source_root / 'job.json').read_text())
    if job['project_id'] != project_id or job['status'] not in {'ready_for_blender','candidate_for_review'}:
        raise ReconstructionError('A calibrated source reconstruction is required')
    # Use canonical sibling directories, avoiding stale installation absolute paths.
    processed = source_root / 'processed'
    if not processed.exists():
        processed = Path(job['processed_dir']).resolve()
    if not processed.resolve().is_relative_to((store.root / 'projects' / project_id).resolve()):
        raise ReconstructionError('Source calibration is outside this project')
    calibration = None
    if calibration_revision_id:
        calibration_revision_id = str(uuid.UUID(calibration_revision_id))
        croot = source_root/'mesh-calibrations'/calibration_revision_id
        calibration = json.loads((croot/'calibration.json').read_text())
        if (calibration['status'] != 'ready' or calibration['project_id'] != project_id or
                calibration['source_job_id'] != source_job_id or calibration['scope'] != 'offline_mesh_only'):
            raise ReconstructionError('Unverified mesh photo calibration')
        processed = croot/'processed'
        for relative, digest in calibration['calibrated_hashes'].items():
            p = processed/relative
            if not p.resolve().is_relative_to(processed.resolve()) or Store._file_sha256(p) != digest:
                raise ReconstructionError('Extended mesh calibration changed')
        job = {**job, 'input_captures': calibration['input_captures'], 'frame_manifest': calibration['frame_manifest']}
    sparse = processed / 'colmap' / 'sparse' / '0'
    images = processed / 'images'
    if not sparse.is_dir() or not images.is_dir(): raise ReconstructionError('COLMAP calibrated photos are missing')
    snapshot = store.export_session(job['session_id'])
    receipts = {c['receipt']['capture_id']:c['receipt']['image_sha256'] for c in snapshot['captures']}
    originals = {c['receipt']['capture_id']:c for c in snapshot['captures']}
    frame_manifest = job.get('frame_manifest') or {item['capture_id']:f"frame_{i:05d}{Path(item['image_name']).suffix}"
                         for i,item in enumerate(sorted(job['input_captures'],key=lambda c:c['image_name']),1)}
    processed_provenance=[]
    for item in job['input_captures']:
        if receipts.get(item['capture_id']) != item['image_sha256']: raise ReconstructionError('Source provenance changed')
        image = images / Path(frame_manifest[item['capture_id']]).name
        with Image.open(originals[item['capture_id']]['image_path']) as a, Image.open(image) as b:
            av=np.asarray(ImageOps.exif_transpose(a).convert('RGB').resize((64,48)),dtype=float)
            bv=np.asarray(b.convert('RGB').resize((64,48)),dtype=float)
            if float(np.sqrt(np.mean((av-bv)**2)))>14:
                raise ReconstructionError('Calibrated photo differs from original source')
        processed_provenance.append({'capture_id':item['capture_id'],'image_name':image.name,'sha256':Store._file_sha256(image)})
    colmap = os.environ.get('QUEST_SPLAT_COLMAP_EXE') or _tool('colmap')
    with _worker_lock(store.root):
        revision_id = str(uuid.uuid4())
        root = source_root / 'mesh-revisions' / revision_id
        root.mkdir(parents=True)
        # COLMAP's native Windows image I/O cannot reliably open long project
        # paths. Work on verified private copies in a short temporary directory.
        scratch = Path(os.environ.get('QUEST_SPLAT_MESH_WORK_ROOT') or tempfile.gettempdir()).resolve()
        if str(scratch) == scratch.anchor: raise ReconstructionError('Use a project scratch folder, not a drive root')
        scratch.mkdir(parents=True, exist_ok=True)
        _require_mesh_space(scratch, len(job['input_captures'])*max_image_size**2*4*8+4*2**30)
        _require_mesh_space(root, 2*2**30)
        work = Path(tempfile.mkdtemp(prefix='qs-mvs-', dir=scratch))
        work_images = work / 'images'
        work_sparse = work / 'sparse'
        shutil.copytree(images, work_images)
        shutil.copytree(sparse, work_sparse)
        dense = work / 'dense'
        log = root / 'worker.log'
        manifest = {'schema_version':1,'revision_id':revision_id,'project_id':project_id,'source_job_id':source_job_id,
                    'created_utc':utc_now(),'status':'running','method':'colmap_photo_mvs_delaunay',
                    'input_captures':job['input_captures'], 'coordinate_frame':'colmap_sparse',
                    'calibrated_photos':processed_provenance,
                    'calibration_revision_id':calibration_revision_id,
                    'dense_settings':{'max_image_size':max_image_size, 'source_views':stereo_sources,
                                      'iterations':stereo_iterations, 'window_step':stereo_window_step,
                                      'gpu_workers': stereo_gpu_workers},
                    'private_work_directory':str(work),
                    'uncertainty':{'metric_scale':'unverified','unseen_surfaces':'unknown',
                                   'accuracy':'not measured; reflective and textureless areas may fail',
                                   'splat_frame_alignment':'not established'},'artifact':None}
        def stage(s):
            manifest['stage']=s
            temp = root / 'mesh.tmp'
            temp.write_text(json.dumps(manifest,indent=2))
            os.replace(temp,root/'mesh.json')
        env = dict(os.environ)
        try:
            stage('undistorting_photos')
            _run([colmap,'image_undistorter','--image_path',str(work_images),'--input_path',str(work_sparse),
                  '--output_path',str(dense),'--output_type','COLMAP','--max_image_size',str(max_image_size)],log,env,1800)
            manifest['workspace_input_hashes'] = {str(p.relative_to(work)): Store._file_sha256(p)
                for folder in (dense/'images', dense/'sparse') for p in folder.rglob('*') if p.is_file()}
            if seed:
                manifest['retained_photo_depth'] = _reuse_photo_depth(seed,dense,colmap,root,env)
            # Bound supporting views without changing the registered reference cameras.
            stereo_config = dense/'stereo'/'patch-match.cfg'
            stereo_config.write_text(stereo_config.read_text().replace('__auto__, 20', f'__auto__, {stereo_sources}'))
            _complete_dense_mesh(manifest, work, root, colmap, env, stage)
        except Exception as e:
            manifest['status']='failed'; manifest['error']=str(e); stage('failed'); raise
        return manifest


def read_mesh(store: Store, project_id: str, job_id: str, revision_id: str):
    """Verify the immutable revision before advertising or serving its bytes."""
    project_id, job_id, revision_id = (str(uuid.UUID(value)) for value in (project_id, job_id, revision_id))
    store.project(project_id)
    project_root = store.root/'projects'/project_id
    root = project_root/'reconstructions'/job_id/'mesh-revisions'/revision_id
    try:
        j = json.loads((root/'mesh.json').read_text(encoding='utf-8'))
        path = root/'mesh.glb'
        a = j['artifact']
        if (j['status'] != 'ready' or j['project_id'] != project_id or
                j['source_job_id'] != job_id or j['revision_id'] != revision_id or
                path.is_symlink() or not path.resolve().is_relative_to(project_root.resolve()) or
                Path(a['path']).resolve() != path.resolve() or
                type(a['vertex_count']) is not int or a['vertex_count'] < 3 or
                type(a['triangle_count']) is not int or a['triangle_count'] < 1 or
                path.stat().st_size != a['size_bytes'] or Store._file_sha256(path) != a['sha256']):
            raise ValueError('Mesh revision verification failed')
        return j, path
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StoreError(404, 'Verified mesh unavailable') from error


def textured_mesh_to_glb(source: Path, texture: Path, destination: Path,
                         provenance: dict, original_to_training: np.ndarray, fallback_mesh: Path | None = None,
                         fallback_photo_colors=True):
    """Preserve per-corner atlas UVs and pack the photo texture into a portable GLB.

    COLMAP uses bottom-origin UVs. glTF uses top-origin UVs and Y-up coordinates.
    Blender's glTF importer reverses both conversions on import.
    """
    from plyfile import PlyData
    ply = PlyData.read(source)
    vertices = np.column_stack([ply['vertex'][k] for k in ('x', 'y', 'z')]).astype(np.float64)
    face_records = ply['face']
    indices = np.stack(face_records['vertex_indices']).astype(np.int64)
    uv = np.stack(face_records['texcoord']).reshape(-1, 3, 2).astype(np.float32)
    if (indices.ndim != 2 or indices.shape[1] != 3 or not len(indices) or
            indices.min() < 0 or indices.max() >= len(vertices) or
            not np.isfinite(vertices).all() or not np.isfinite(uv).all() or
            np.any(uv < -1e-5) or np.any(uv > 1.00001)):
        raise ReconstructionError('Textured surface has invalid triangles or UVs')
    transform = np.asarray(original_to_training, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ReconstructionError('Invalid mesh normalization')
    vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    normals = np.zeros_like(vertices)
    face_normal = np.cross(vertices[indices[:, 1]] - vertices[indices[:, 0]],
                           vertices[indices[:, 2]] - vertices[indices[:, 0]])
    for i in range(3):
        np.add.at(normals, indices[:, i], face_normal)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # Z-up reconstruction -> glTF Y-up. A proper rotation, not a reflection.
    z_to_y = np.asarray([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    positions = (vertices[indices.reshape(-1)] @ z_to_y.T).astype('<f4')
    normal_data = (normals[indices.reshape(-1)] @ z_to_y.T).astype('<f4')
    texcoords = uv.reshape(-1, 2).copy(); texcoords[:, 1] = 1 - texcoords[:, 1]
    triangles = np.arange(len(positions), dtype='<u4').reshape(-1, 3)
    unknown = np.all(uv == 0, axis=(1, 2)) if fallback_mesh else np.zeros(len(indices), dtype=bool)
    fallback_colors = np.full((len(vertices), 3), 0.25, dtype=np.float32)
    if fallback_mesh and np.any(unknown):
        fallback = PlyData.read(fallback_mesh)['vertex']
        fallback_positions = np.column_stack([fallback[k] for k in ('x', 'y', 'z')])
        original_positions = np.column_stack([ply['vertex'][k] for k in ('x', 'y', 'z')])
        if not np.allclose(fallback_positions, original_positions, atol=1e-6):
            raise ReconstructionError('Vertex-color fallback does not match the textured surface')
        if fallback_photo_colors and all(k in fallback.data.dtype.names for k in ('red', 'green', 'blue')):
            rgb = np.column_stack([fallback[k] for k in ('red', 'green', 'blue')]).astype(np.float32)/255
            fallback_colors = np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
    with Image.open(texture) as image:
        image.verify()
    png = texture.read_bytes()
    chunks = [positions.tobytes(), normal_data.tobytes(), texcoords.astype('<f4').tobytes(),
              triangles[~unknown].astype('<u4').tobytes(),
              triangles[unknown].astype('<u4').tobytes(),
              fallback_colors[indices.reshape(-1)].astype('<f4').tobytes(), png]
    views, binary = [], bytearray()
    for i, chunk in enumerate(chunks):
        binary.extend(b'\0' * (-len(binary) % 4))
        view = {'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(chunk)}
        if i < 6: view['target'] = 34963 if i in (3, 4) else 34962
        views.append(view); binary.extend(chunk)
    doc = {'asset': {'version': '2.0', 'generator': 'QuestSplat calibrated photo mesh'},
           'buffers': [{'byteLength': len(binary)}], 'bufferViews': views,
           'accessors': [
               {'bufferView': 0, 'componentType': 5126, 'count': len(positions), 'type': 'VEC3',
                'min': positions.min(0).tolist(), 'max': positions.max(0).tolist()},
               {'bufferView': 1, 'componentType': 5126, 'count': len(positions), 'type': 'VEC3'},
               {'bufferView': 2, 'componentType': 5126, 'count': len(positions), 'type': 'VEC2'},
               {'bufferView': 3, 'componentType': 5125, 'count': int(np.sum(~unknown)*3), 'type': 'SCALAR'},
               {'bufferView': 4, 'componentType': 5125, 'count': int(np.sum(unknown)*3), 'type': 'SCALAR'},
               {'bufferView': 5, 'componentType': 5126, 'count': len(positions), 'type': 'VEC3'}],
           'images': [{'bufferView': 6, 'mimeType': 'image/png'}],
           'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}],
           'textures': [{'source': 0, 'sampler': 0}],
           'extensionsUsed': ['KHR_materials_unlit'],
           'materials': [{'name': 'Captured room photographs', 'doubleSided': True,
                          'extensions': {'KHR_materials_unlit': {}},
                          'pbrMetallicRoughness': {'baseColorTexture': {'index': 0},
                                                 'metallicFactor': 0, 'roughnessFactor': 1}},
                         {'name': ('Photo-fused color' if fallback_photo_colors else 'Neutral surface')+' - atlas unavailable', 'doubleSided': True,
                          'extensions': {'KHR_materials_unlit': {}},
                          'pbrMetallicRoughness': {'metallicFactor': 0, 'roughnessFactor': 1}}],
           'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2},
                                      'indices': 3, 'material': 0, 'mode': 4}]}],
           'nodes': [{'mesh': 0, 'name': 'Cleaned room mesh'}],
           'scenes': [{'nodes': [0]}], 'scene': 0, 'extras': provenance}
    if not np.any(~unknown): doc['meshes'][0]['primitives'] = []
    if np.any(unknown):
        doc['meshes'][0]['primitives'].append({'attributes': {'POSITION': 0, 'NORMAL': 1, 'COLOR_0': 5},
                                             'indices': 4, 'material': 1, 'mode': 4})
    # glTF disallows empty bufferViews/accessors, even when unused. Retain a
    # non-empty padding record for the unused alternate index stream.
    for index in (3, 4):
        if not views[index]['byteLength']:
            views[index]['byteOffset'] = views[3 if index == 4 else 4]['byteOffset']
            views[index]['byteLength'] = 12
            doc['accessors'][index]['count'] = 3
    js = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    js += b' ' * (-len(js) % 4); binary.extend(b'\0' * (-len(binary) % 4))
    with destination.open('xb') as handle:
        handle.write(struct.pack('<III', 0x46546c67, 2, 28 + len(js) + len(binary)))
        handle.write(struct.pack('<II', len(js), 0x4e4f534a)); handle.write(js)
        handle.write(struct.pack('<II', len(binary), 0x004e4942)); handle.write(binary)
        handle.flush(); os.fsync(handle.fileno())
    return {'sha256': Store._file_sha256(destination), 'size_bytes': destination.stat().st_size,
            'vertex_count': len(positions), 'unique_surface_vertices': len(vertices),
            'triangle_count': len(indices), 'texture_sha256': Store._file_sha256(texture),
            'path': str(destination), 'embedded_texture': True,
            'atlas_textured_faces': int(np.sum(~unknown)),
            'photo_color_fallback_faces': int(np.sum(unknown)) if fallback_photo_colors else 0,
            'neutral_fallback_faces': 0 if fallback_photo_colors else int(np.sum(unknown))}


def remesh_surface(store: Store, project_id: str, source_job_id: str, source_revision_id: str,
                   poisson_depth=10, density_trim=7.0):
    """Review an alternative surface from retained depth fusion, without rerunning photos."""
    project_id, source_job_id, source_revision_id = (str(uuid.UUID(value)) for value in
                                                   (project_id, source_job_id, source_revision_id))
    if not 8 <= poisson_depth <= 12 or not 0 <= density_trim <= 12:
        raise ReconstructionError('Invalid Poisson reconstruction settings')
    parent, _ = read_mesh(store, project_id, source_job_id, source_revision_id)
    job_root = store.root/'projects'/project_id/'reconstructions'/source_job_id
    fused = Path(parent['private_work_directory'])/'dense/fused.ply'
    if not fused.is_file(): raise ReconstructionError('Retained fused depth is unavailable')
    colmap = os.environ.get('QUEST_SPLAT_COLMAP_EXE') or _tool('colmap')
    with _worker_lock(store.root):
        revision = str(uuid.uuid4()); root = job_root/'mesh-revisions'/revision
        root.mkdir(parents=True)
        manifest = {**parent, 'revision_id': revision, 'parent_mesh_revision_id': source_revision_id,
                    'created_utc': utc_now(), 'status': 'running', 'artifact': None,
                    'stage': 'poisson_surface', 'method': 'colmap_photo_mvs_poisson',
                    'surface_settings': {'depth': poisson_depth, 'density_trim': density_trim,
                                         'fused_depth_sha256': Store._file_sha256(fused)}}
        def save(): (root/'mesh.json').write_text(json.dumps(manifest, indent=2))
        save()
        short = Path(parent['private_work_directory'])/'remesh'/revision
        short.mkdir(parents=True)
        try:
            _run([colmap, 'poisson_mesher', '--input_path', str(fused), '--output_path', str(short/'mesh.ply'),
                  '--PoissonMeshing.depth', str(poisson_depth), '--PoissonMeshing.trim', str(density_trim),
                  '--PoissonMeshing.point_weight', '2'], root/'worker.log', dict(os.environ), 1800)
            shutil.copyfile(short/'mesh.ply', root/'mesh.ply')
            provenance = {k: manifest[k] for k in ('method', 'source_job_id', 'coordinate_frame', 'uncertainty')}
            manifest['artifact'] = mesh_to_glb(root/'mesh.ply', root/'mesh.glb', provenance)
            manifest['status'] = 'ready'; manifest['stage'] = 'ready'; save()
        except Exception as error:
            manifest['status'] = 'failed'; manifest['stage'] = 'failed'; manifest['error'] = str(error); save(); raise
        return manifest


def fit_room_axes(boxes):
    """Infer three dominant directions, then enforce the known right-angle room.

    This is a declared room prior, not evidence of equal wall lengths or scale.
    Smaller/nonparallel object patches need not follow the architectural axes.
    """
    ordered = sorted(boxes, key=lambda b: b['extent'][0]*b['extent'][1], reverse=True)
    chosen, weights = [], []
    for box in ordered:
        n = np.asarray(box['rotation'], dtype=float)[:, 2]
        if all(abs(n@old) < .25 for old in chosen):
            chosen.append(n); weights.append(box['extent'][0]*box['extent'][1])
        if len(chosen) == 3: break
    if len(chosen) != 3: raise ReconstructionError('Three supported room directions were not detected')
    frame = np.column_stack(chosen)
    if np.linalg.det(frame) < 0: frame[:, 2] *= -1
    u, _, vt = np.linalg.svd(frame@np.diag(weights))
    axes = u@vt
    deviations = np.rad2deg(np.arccos(np.clip(np.abs(np.sum(frame*axes, axis=0)), 0, 1)))
    if deviations.max() > 12: raise ReconstructionError('Observed planes do not support a right-angle room')
    return axes, {'axes': axes.tolist(), 'observed_axis_corrections_degrees': deviations.tolist(),
                  'corner_angle_degrees': 90, 'basis': 'user-confirmed square/right-angle room',
                  'equal_wall_lengths': 'not constrained without dimensions'}


def constrain_supported_planes(vertices, faces, points, point_normals, boxes,
                               max_displacement, support_radius, min_support=500, room_axes=None):
    """Project only locally observed, normal-compatible mesh patches onto planes.

    Boxes are detected from photo-fused depth, not the interpolated mesh. Fit on
    alternating points and check independent held-out support. Never create faces
    across voids or move distant/curved objects merely because a plane is nearby.
    All distances are reconstruction units, not claimed physical measurements.
    """
    from scipy.spatial import cKDTree
    v = np.asarray(vertices, dtype=float)
    t = np.asarray(faces, dtype=np.int64)
    p = np.asarray(points, dtype=float)
    pn = np.asarray(point_normals, dtype=float)
    if (v.ndim != 2 or v.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 3 or
            p.shape != pn.shape or p.ndim != 2 or p.shape[1] != 3 or
            not all(np.isfinite(a).all() for a in (v, p, pn)) or
            not np.isfinite(max_displacement) or not np.isfinite(support_radius) or
            max_displacement <= 0 or support_radius <= 0 or len(p) < min_support or
            len(t) == 0 or t.min() < 0 or t.max() >= len(v)):
        raise ReconstructionError('Invalid supported-plane inputs')
    pn = pn / np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-12)
    cross = np.cross(v[t[:, 1]]-v[t[:, 0]], v[t[:, 2]]-v[t[:, 0]])
    vn = np.zeros_like(v)
    for i in range(3): np.add.at(vn, t[:, i], cross)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    nearest_distance, nearest_index = cKDTree(p).query(v, workers=2)
    out = v.copy(); labels = np.full(len(v), -1, dtype=np.int32)
    score = np.full(len(v), np.inf); reports = []
    for box in sorted(boxes, key=lambda b: float(b['extent'][0]*b['extent'][1]), reverse=True):
        center, rotation, extent = (np.asarray(box[k], dtype=float) for k in ('center', 'rotation', 'extent'))
        if (center.shape != (3,) or rotation.shape != (3, 3) or extent.shape != (3,) or
                not all(np.isfinite(a).all() for a in (center, rotation, extent)) or
                np.any(extent <= 0) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)):
            raise ReconstructionError('Invalid planar patch bounds')
        normal = rotation[:, 2]
        local = (p-center)@rotation
        mask = ((np.abs(local[:, :2]) <= extent[:2]/2).all(axis=1) &
                (np.abs(local[:, 2]) <= min(max_displacement, extent[2]/2 + support_radius)) &
                (np.abs(pn@normal) >= np.cos(np.deg2rad(25))))
        support = p[mask]
        if len(support) < min_support: continue
        fit, held = support[::2], support[1::2]
        for _ in range(3):
            origin = np.median(fit, axis=0)
            _, _, axes = np.linalg.svd(fit-origin, full_matrices=False)
            n = axes[-1]
            residual = np.abs((fit-origin)@n)
            fit = fit[residual <= max(np.quantile(residual, .85), support_radius/4)]
        room_axis = None
        if room_axes is not None:
            match = np.abs(n@room_axes)
            if match.max() >= np.cos(np.deg2rad(12)):
                room_axis = int(np.argmax(match)); n = room_axes[:, room_axis].copy()
                origin = np.median(fit, axis=0)
        if np.abs(n@normal) < np.cos(np.deg2rad(10)): continue
        held_error = np.abs((held-origin)@n)
        if np.quantile(held_error, .9) > max_displacement*.6: continue
        local_v = (v-center)@rotation
        signed = (v-origin)@n
        distance = np.abs(signed)
        # Normal gates protect silhouettes, raised detail and curved objects.
        eligible = ((np.abs(local_v[:, :2]) <= extent[:2]/2).all(axis=1) &
                    (distance <= max_displacement) & (nearest_distance <= support_radius) &
                    (np.abs(vn@n) >= np.cos(np.deg2rad(45))) &
                    (np.abs(pn[nearest_index]@n) >= np.cos(np.deg2rad(25))))
        choose = eligible & (distance < score)
        if np.count_nonzero(choose) < min_support//2: continue
        index = len(reports)
        # Ease only the perimeter/depth transition; the patch interior is flat.
        edge = np.min(1-2*np.abs(local_v[:, :2])/extent[:2], axis=1)
        blend = np.minimum(np.clip(edge/.08, 0, 1),
                           np.clip((max_displacement-distance)/(max_displacement*.3), 0, 1))
        out[choose] = v[choose] - (signed[choose]*blend[choose])[:, None]*n
        score[choose] = distance[choose]; labels[choose] = index
        reports.append({'normal': n.tolist(), 'origin': origin.tolist(), 'bounds': box,
                        'room_axis': room_axis,
                        'fit_points': len(fit), 'held_out_points': len(held),
                        'held_out_p90_residual': float(np.quantile(held_error, .9))})
    # Projection must not collapse or reverse existing triangles. Undo affected
    # vertices instead of deleting genuine geometry or inventing replacement faces.
    reverted = np.zeros(len(v), dtype=bool)
    original_valid = np.linalg.norm(cross, axis=1) >= 1e-12
    for _ in range(24):
        new_cross = np.cross(out[t[:, 1]]-out[t[:, 0]], out[t[:, 2]]-out[t[:, 0]])
        bad = original_valid & ((np.linalg.norm(new_cross, axis=1) < 1e-12) |
                                (np.sum(cross*new_cross, axis=1) <= 0))
        if not bad.any(): break
        undo = np.unique(t[bad]); out[undo] = v[undo]; labels[undo] = -1; reverted[undo] = True
    else: raise ReconstructionError('Plane constraints could not retain valid topology')
    move = np.linalg.norm(out-v, axis=1)
    if move.max(initial=0) > max_displacement*(1+1e-8):
        raise ReconstructionError('Plane constraint exceeded its displacement bound')
    for index, report in enumerate(reports):
        chosen = labels == index
        report['constrained_vertices'] = int(chosen.sum())
        if chosen.any():
            n, origin = np.asarray(report['normal']), np.asarray(report['origin'])
            report['mesh_rms_before'] = float(np.sqrt(np.mean(((v[chosen]-origin)@n)**2)))
            report['mesh_rms_after'] = float(np.sqrt(np.mean(((out[chosen]-origin)@n)**2)))
    return out, labels, {'patches': reports, 'moved_vertices': int((move > 1e-8).sum()),
                         'reverted_vertices': int(reverted.sum()),
                         'max_displacement': float(move.max(initial=0)), 'faces_added': 0,
                         'faces_removed': 0, 'units': 'COLMAP reconstruction units; metric scale unverified'}


def regularize_planes(store: Store, project_id: str, source_job_id: str, source_revision_id: str,
                      displacement_fraction=.008, support_fraction=.006, right_angle_room=False):
    """New photo-supported plane-constrained revision; original mesh stays intact."""
    import open3d as o3d
    from plyfile import PlyData
    project_id, source_job_id, source_revision_id = (str(uuid.UUID(value)) for value in
                                                   (project_id, source_job_id, source_revision_id))
    if not .001 <= displacement_fraction <= .015 or not .001 <= support_fraction <= .01:
        raise ReconstructionError('Plane constraint budget is outside conservative bounds')
    parent, _ = read_mesh(store, project_id, source_job_id, source_revision_id)
    if parent['coordinate_frame'] != 'colmap_sparse':
        raise ReconstructionError('Plane constraints require the raw COLMAP surface frame')
    job_root = store.root/'projects'/project_id/'reconstructions'/source_job_id
    source = job_root/'mesh-revisions'/source_revision_id/'mesh.ply'
    work = Path(parent['private_work_directory'])
    fused = work/'dense/fused.ply'
    if not fused.is_file(): raise ReconstructionError('Retained photo-fused depth is unavailable')
    protected = {p: Store._file_sha256(p) for p in (source, fused, job_root/'splat.ply')}
    with _worker_lock(store.root):
        cloud = o3d.io.read_point_cloud(str(fused))
        points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
        if not cloud.has_normals(): raise ReconstructionError('Depth support needs measured surface normals')
        diagonal = float(np.linalg.norm(np.quantile(points, .99, axis=0)-np.quantile(points, .01, axis=0)))
        if not np.isfinite(diagonal) or diagonal <= 0: raise ReconstructionError('Invalid depth extent')
        sample = cloud.voxel_down_sample(diagonal/500)
        boxes = sample.detect_planar_patches(normal_variance_threshold_deg=35, coplanarity_deg=75,
                    outlier_ratio=.5, min_plane_edge_length=diagonal*.05, min_num_points=150,
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
        bounds = [{'center': b.center.tolist(), 'rotation': b.R.tolist(), 'extent': b.extent.tolist()}
                  for b in boxes if min(b.extent[:2]) >= diagonal*.025]
        room_axes, room_report = fit_room_axes(bounds) if right_angle_room else (None, None)
        ply = PlyData.read(source)
        vertices = np.column_stack([ply['vertex'][k] for k in ('x', 'y', 'z')])
        faces = np.stack(ply['face']['vertex_indices'])
        constrained, labels, report = constrain_supported_planes(vertices, faces, points, normals, bounds,
                                        diagonal*displacement_fraction, diagonal*support_fraction, room_axes=room_axes)
        report['room_geometry_constraint'] = room_report
        if report['moved_vertices'] < 500: raise ReconstructionError('No sufficiently supported plane improvement')
        revision = str(uuid.uuid4()); root = job_root/'mesh-revisions'/revision; root.mkdir(parents=True)
        for i, field in enumerate(('x', 'y', 'z')): ply['vertex'][field] = constrained[:, i]
        ply.write(root/'mesh.ply')
        np.savez_compressed(root/'plane-membership.npz', plane_id=labels)
        (root/'plane-review.json').write_text(json.dumps(report, indent=2))
        manifest = {**parent, 'revision_id': revision, 'parent_mesh_revision_id': source_revision_id,
                    'created_utc': utc_now(), 'method': 'photo_depth_supported_plane_constraints',
                    'plane_constraints': {**report, 'detected_patches': len(boxes),
                        'open3d_version': o3d.__version__, 'depth_sha256': protected[fused],
                        'displacement_fraction': displacement_fraction, 'support_fraction': support_fraction},
                    'visual_acceptance': 'pending; geometric checks do not establish architectural accuracy'}
        provenance = {k: manifest[k] for k in ('method', 'source_job_id', 'coordinate_frame', 'uncertainty')}
        manifest['artifact'] = mesh_to_glb(root/'mesh.ply', root/'mesh.glb', provenance)
        if any(Store._file_sha256(p) != digest for p, digest in protected.items()):
            raise ReconstructionError('Protected depth, raw mesh or splat changed')
        manifest['status'] = 'ready'; manifest['stage'] = 'ready'
        (root/'mesh.json').write_text(json.dumps(manifest, indent=2))
        return manifest


def structural_room_revision(store: Store, project_id: str, source_job_id: str, source_revision_id: str):
    """Fit the user-confirmed straight/right-angle room, retaining dense detail.

    The result explicitly infers structural planes behind occluders. It is not
    a measured replacement for the retained photo reconstruction or Gaussian.
    """
    import open3d as o3d
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation
    from .reconstruct import _read_colmap_images
    from .mesh_room import select_room_planes, plane_grids, add_opening_evidence, build_room_surface
    project_id, source_job_id, source_revision_id = (str(uuid.UUID(v)) for v in
                                                   (project_id, source_job_id, source_revision_id))
    parent, _ = read_mesh(store, project_id, source_job_id, source_revision_id)
    planes=parent.get('plane_constraints', {})
    prior=planes.get('room_geometry_constraint')
    if parent['coordinate_frame']!='colmap_sparse' or not prior or not prior.get('axes'):
        raise ReconstructionError('An explicitly constrained raw room frame is required')
    job_root=store.root/'projects'/project_id/'reconstructions'/source_job_id
    job=json.loads((job_root/'job.json').read_text())
    work=Path(parent['private_work_directory']); dense=work/'dense'; fused=dense/'fused.ply'
    source=job_root/'mesh-revisions'/source_revision_id/'mesh.ply'
    protected={p:Store._file_sha256(p) for p in (source,fused,job_root/'splat.ply',dense/'sparse/cameras.bin',dense/'sparse/images.bin')}
    revision=str(uuid.uuid4()); root=job_root/'mesh-revisions'/revision
    _require_mesh_space(store.root,2*2**30)
    with _worker_lock(store.root):
        root.mkdir(parents=True)
        manifest={**parent,'revision_id':revision,'parent_mesh_revision_id':source_revision_id,
                  'created_utc':utc_now(),'status':'running','stage':'fitting_room_structure','artifact':None,
                  'method':'photo_fitted_architectural_room',
                  'uncertainty':{**parent['uncertainty'],'architectural_prior':'user-confirmed straight walls and right-angle corners',
                      'hidden_surfaces':'inferred structural planes behind occluders, not measured hidden geometry'}}
        def save(): (root/'mesh.json').write_text(json.dumps(manifest,indent=2))
        save()
        try:
            text=work/('room-cameras-'+revision);text.mkdir()
            colmap=os.environ.get('QUEST_SPLAT_COLMAP_EXE') or _tool('colmap')
            _run([colmap,'model_converter','--input_path',str(dense/'sparse'),'--output_path',str(text),'--output_type','TXT'],
                 root/'worker.log',dict(os.environ),120)
            rows=[line.split() for line in (text/'cameras.txt').read_text().splitlines() if line and not line.startswith('#')]
            if len(rows)!=1 or rows[0][1]!='PINHOLE': raise ReconstructionError('One undistorted calibrated camera is required')
            _,_,width,height,fx,fy,cx,cy=rows[0]
            width,height=int(width),int(height);fx,fy,cx,cy=map(float,(fx,fy,cx,cy))
            cameras=_read_colmap_images(text/'images.txt')
            rotations={name:Rotation.from_quat(q[[1,2,3,0]]).as_matrix() for name,(q,t) in cameras.items()}
            centers=np.array([-rotations[name].T@t for name,(q,t) in cameras.items()])
            cloud=o3d.io.read_point_cloud(str(fused))
            if not cloud.has_normals() or not cloud.has_colors(): raise ReconstructionError('Measured depth normals and photo colors are required')
            points,normals,colors=np.asarray(cloud.points),np.asarray(cloud.normals),np.asarray(cloud.colors)
            axes=np.asarray(prior['axes'])
            normalization=json.loads((Path(job['training_config']).parent/'dataparser_transforms.json').read_text())
            transformed=np.asarray(normalization['transform'])[:,:3]@axes
            vertical=int(np.argmax(abs(transformed[2])));up=int(np.sign(transformed[2,vertical]))
            basis,bounds,diagonal,evidence=select_room_planes(points,normals,planes['patches'],axes,centers,vertical,up)
            grids=plane_grids(bounds,diagonal*.004)
            vv,uu=np.meshgrid(np.arange(4,height,8),np.arange(4,width,8),indexing='ij')
            camera_rays=np.column_stack(((uu.ravel()-cx)/fx,(vv.ravel()-cy)/fy,np.ones(uu.size)))
            used=[]; accepted_centers=[]; rejected={'outside_room':0,'near_duplicate_position':0}
            manifest['stage']='verifying_openings_from_depth';save()
            for name,(q,t) in cameras.items():
                r=rotations[name];origin=(-r.T@t)@basis
                if np.any(origin<=bounds[0]) or np.any(origin>=bounds[1]):
                    rejected['outside_room']+=1;continue
                if any(np.linalg.norm(origin-c)<diagonal*.003 for c in accepted_centers):
                    rejected['near_duplicate_position']+=1;continue
                path=dense/'stereo/depth_maps'/(name+'.geometric.bin')
                if not _native_map_valid(path,width,height,1): raise ReconstructionError('Invalid geometric depth for an opening camera')
                with path.open('rb') as f:
                    ampersands=0
                    while ampersands<3:
                        byte=f.read(1)
                        if not byte:raise ReconstructionError('Truncated native depth header')
                        ampersands+=byte==b'&'
                    depth=np.fromfile(f,dtype='<f4').reshape(height,width)
                add_opening_evidence(grids,bounds,origin,camera_rays@r@basis,depth[vv,uu].ravel(),diagonal*.02)
                accepted_centers.append(origin);used.append(name)
            if len(used)<3:raise ReconstructionError('Insufficient distinct in-room cameras for opening review')
            v,f,c,report=build_room_surface(points,colors,basis,bounds,grids)
            report.update({'observed_plane_support':evidence,'distinct_depth_cameras':len(used),
                'opening_minimum_camera_votes':3,'depth_camera_spacing_fraction':.003,
                'depth_beyond_wall_tolerance_fraction':.02,'excluded_cameras':rejected,
                'detailed_scan_revision_retained':source_revision_id,'depth_sha256':protected[fused]})
            vertices=np.empty(len(v),dtype=[(k,'<f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
            for i,k in enumerate(('x','y','z')):vertices[k]=v[:,i]
            for i,k in enumerate(('red','green','blue')):vertices[k]=np.rint(np.clip(c[:,i],0,1)*255).astype('u1')
            triangles=np.empty(len(f),dtype=[('vertex_indices','<i4',(3,))]);triangles['vertex_indices']=f
            PlyData([PlyElement.describe(vertices,'vertex'),PlyElement.describe(triangles,'face')],text=False).write(root/'mesh.ply')
            (root/'plane-review.json').write_text(json.dumps({**planes,'architectural_structure':report},indent=2))
            manifest['architectural_structure']=report
            manifest['visual_acceptance']='pending review of actual structure and photo textures'
            manifest['artifact']=mesh_to_glb(root/'mesh.ply',root/'mesh.glb',
                {k:manifest[k] for k in ('method','source_job_id','coordinate_frame','uncertainty')})
            if any(Store._file_sha256(p)!=digest for p,digest in protected.items()):raise ReconstructionError('A protected source changed')
            manifest['status']='ready';manifest['stage']='ready';save()
            return manifest
        except Exception as error:
            manifest['status']='failed';manifest['stage']='failed';manifest['error']=str(error);save();raise


def detail_preserving_room_revision(store: Store, project_id: str, source_job_id: str,
                                   source_revision_id: str, room_reference_revision_id: str):
    """Constrain supported architecture without replacing the scanned 3D scene.

    The reference provides fitted plane coordinates only. Its inferred faces are
    never copied. The source must be the measured dense surface, before local
    projection, so open doors and object volumes remain available to protect.
    """
    import open3d as o3d
    from plyfile import PlyData
    from .mesh_room import constrain_room_surfaces
    project_id, source_job_id, source_revision_id, room_reference_revision_id = (
        str(uuid.UUID(value)) for value in
        (project_id, source_job_id, source_revision_id, room_reference_revision_id))
    parent, _ = read_mesh(store, project_id, source_job_id, source_revision_id)
    reference, _ = read_mesh(store, project_id, source_job_id, room_reference_revision_id)
    structure = reference.get('architectural_structure')
    if (parent['coordinate_frame'] != 'colmap_sparse' or not structure or
            reference['coordinate_frame'] != parent['coordinate_frame'] or
            parent.get('architectural_structure') or parent.get('plane_constraints') or
            parent.get('detail_preserving_room')):
        raise ReconstructionError('Use an unprojected raw surface and a verified same-frame room fit')
    if parent.get('calibration_revision_id') != reference.get('calibration_revision_id'):
        raise ReconstructionError('Room fit and detailed surface need the same photo calibration')
    job_root = store.root/'projects'/project_id/'reconstructions'/source_job_id
    work = Path(parent['private_work_directory'])
    fused = work/'dense/fused.ply'
    source = job_root/'mesh-revisions'/source_revision_id/'mesh.ply'
    fitted = job_root/'mesh-revisions'/room_reference_revision_id/'mesh.ply'
    if not fused.is_file():
        raise ReconstructionError('Retained measured depth support is unavailable')
    protected = {path: Store._file_sha256(path) for path in (source, fitted, fused, job_root/'splat.ply')}
    if structure.get('depth_sha256') != protected[fused]:
        raise ReconstructionError('The room fit was not derived from this verified depth cloud')
    _require_mesh_space(store.root, 2*2**30)
    with _worker_lock(store.root):
        cloud = o3d.io.read_point_cloud(str(fused))
        if not cloud.has_normals():
            raise ReconstructionError('Measured normals are required for detail protection')
        ply = PlyData.read(source)
        vertices = np.column_stack([ply['vertex'][key] for key in ('x', 'y', 'z')])
        faces = np.stack(ply['face']['vertex_indices'])
        changed, labels, report = constrain_room_surfaces(
            vertices, faces, np.asarray(cloud.points), np.asarray(cloud.normals),
            np.asarray(structure['room_basis']), np.asarray(structure['room_bounds']))
        if report['moved_vertices'] == 0:
            raise ReconstructionError('No supported wall/floor interiors could be safely straightened')
        revision = str(uuid.uuid4())
        root = job_root/'mesh-revisions'/revision
        root.mkdir(parents=True)
        report.update({'room_basis': structure['room_basis'], 'room_bounds': structure['room_bounds'],
                       'room_reference_revision_id': room_reference_revision_id,
                       'depth_sha256': protected[fused], 'source_surface_sha256': protected[source],
                       'hidden_surfaces': 'not generated; existing source gaps and objects retained',
                       'units': 'COLMAP reconstruction units; metric scale unverified'})
        manifest = {**parent, 'revision_id': revision, 'parent_mesh_revision_id': source_revision_id,
                    'created_utc': utc_now(), 'method': 'detail_preserving_architectural_constraints',
                    'detail_preserving_room': report,
                    'status': 'running', 'stage': 'exporting_preserved_scene', 'artifact': None,
                    'visual_acceptance': 'pending actual object and architectural geometry review'}
        (root/'mesh.json').write_text(json.dumps(manifest, indent=2))
        try:
            for index, key in enumerate(('x', 'y', 'z')):
                ply['vertex'][key] = changed[:, index]
            ply.write(root/'mesh.ply')
            np.savez_compressed(root/'plane-membership.npz', plane_id=labels)
            (root/'plane-review.json').write_text(json.dumps(report, indent=2))
            manifest['artifact'] = mesh_to_glb(root/'mesh.ply', root/'mesh.glb',
                {key: manifest[key] for key in ('method', 'source_job_id', 'coordinate_frame', 'uncertainty')})
            if any(Store._file_sha256(path) != digest for path, digest in protected.items()):
                raise ReconstructionError('A protected raw surface, depth cloud or splat changed')
            manifest['status'] = 'ready'; manifest['stage'] = 'ready'
        except Exception as error:
            manifest['status'] = 'failed'; manifest['stage'] = 'failed'; manifest['error'] = str(error)
            (root/'mesh.json').write_text(json.dumps(manifest, indent=2))
            raise
        (root/'mesh.json').write_text(json.dumps(manifest, indent=2))
        return manifest


def verified_spatial_observations(store: Store, session_id: str, capture_ids=None):
    """Preflight same-session depth support before running a house reconstruction.

    Real raycast hits must have valid normals/confidence from three translated
    views.  Image-only older photos stay usable through the ordinary mesh path.
    """
    from .mesh_spatial import verified_observations
    return verified_observations(store, session_id, capture_ids)


def depth_assisted_mesh_revision(store: Store, project_id: str, source_job_id: str,
                                 source_revision_id: str):
    """Create one immutable locally supported surface in its original raw frame.

    One anchored capture session can include indoor and outdoor photos.  Mixing
    independently recentered sessions is refused; no shared origin is assumed.
    The existing photo texturer can texture this new surface without changing
    its cameras.  Geometry outside trustworthy independent ray support stays
    exactly as it was; no rectangular room or missing surface is generated.
    """
    from .preview import register_job
    from .mesh_spatial import source_to_session_transform, assist_observed_surface
    project_id, source_job_id, source_revision_id = (str(uuid.UUID(value)) for value in
                                                   (project_id, source_job_id, source_revision_id))
    job_root = store.root/'projects'/project_id/'reconstructions'/source_job_id
    parent_path = job_root/'mesh-revisions'/source_revision_id/'mesh.json'
    # Read only receipt metadata until every independent ray witness has its
    # own photo-frame proof. A good old global fit cannot localize a new photo.
    try:
        parent = json.loads(parent_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise StoreError(404, 'Verified mesh unavailable') from error
    if (parent.get('status') != 'ready' or parent.get('project_id') != project_id or
            parent.get('source_job_id') != source_job_id or parent.get('revision_id') != source_revision_id):
        raise ReconstructionError('A ready same-source raw surface receipt is required')
    if (parent.get('coordinate_frame') != 'colmap_sparse' or
        any(parent.get(key) for key in ('architectural_structure', 'plane_constraints',
                                        'detail_preserving_room', 'spatial_support'))):
        raise ReconstructionError('Depth assistance requires an unprojected raw photo-derived mesh')
    job_path = job_root/'job.json'
    job = json.loads(job_path.read_text(encoding='utf-8'))
    if job.get('project_id') != project_id or job.get('job_id') != source_job_id or job.get('status') != 'ready_for_blender':
        raise ReconstructionError('A ready calibrated source job is required for depth assistance')
    source = job_root/'mesh-revisions'/source_revision_id/'mesh.ply'
    normalization_path = Path(job['training_config']).parent/'dataparser_transforms.json'
    if normalization_path.is_symlink() or not normalization_path.resolve().is_relative_to(job_root.resolve()):
        raise ReconstructionError('Source frame calibration is outside the owning reconstruction')
    observations = verified_spatial_observations(store, job['session_id'],
                                                 [item['capture_id'] for item in parent['input_captures']])
    if observations['report']['project_id'] != project_id:
        raise ReconstructionError('Spatial evidence is from a different project')
    registration = register_job(store, job)
    normalization = json.loads(normalization_path.read_text(encoding='utf-8'))
    raw_to_session, frame_review = source_to_session_transform(job, registration, normalization)
    registered = registration.get('registered_capture_ids')
    residuals = registration.get('capture_pose_residuals')
    if (not isinstance(registered, list) or any(not isinstance(value, str) for value in registered) or
            len(registered) != len(set(registered)) or not isinstance(residuals, dict)):
        raise ReconstructionError('Independent ray captures lack verified per-photo frame registration')
    input_hashes = {item['capture_id']: item['image_sha256'] for item in job['input_captures']}
    witness_proofs = {}
    for capture_id in observations['report']['independent_capture_ids']:
        proof = residuals.get(capture_id)
        if (capture_id not in registered or capture_id not in input_hashes or not isinstance(proof, dict) or
                proof.get('image_sha256') != input_hashes[capture_id] or
                not isinstance(proof.get('frame_name'), str) or not proof['frame_name']):
            raise ReconstructionError('Independent ray capture is not registered in the source photo frame: '+capture_id)
        for field, limit in (('position_error_m', .10), ('rotation_error_deg', 5.)):
            value = proof.get(field)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= limit:
                raise ReconstructionError('Independent ray capture photo pose is too uncertain: '+capture_id+' '+field)
        witness_proofs[capture_id] = proof
    frame_review['ray_capture_registration'] = witness_proofs
    verified_parent, _ = read_mesh(store, project_id, source_job_id, source_revision_id)
    if verified_parent != parent:
        raise ReconstructionError('Raw surface receipt changed during photo-frame preflight')
    protected = {path: Store._file_sha256(path) for path in
                 (source, job_path, job_root/'splat.ply', normalization_path)}
    from plyfile import PlyData
    ply = PlyData.read(source)
    vertices = np.column_stack([ply['vertex'][key] for key in ('x', 'y', 'z')]).astype(float)
    faces = np.stack(ply['face']['vertex_indices'])
    session_vertices = vertices@raw_to_session[:3, :3].T+raw_to_session[:3, 3]
    assisted, support, review = assist_observed_surface(session_vertices, faces, observations)
    if np.count_nonzero(support) < 3:
        raise ReconstructionError('The depth samples do not overlap a safely supported mesh patch; '
                                  'take overlapping translated captures before claiming improved geometry')
    inverse = np.linalg.inv(raw_to_session)
    changed = vertices.copy()
    moved = np.linalg.norm(assisted-session_vertices, axis=1) > 1e-8
    changed[moved] = assisted[moved]@inverse[:3, :3].T+inverse[:3, 3]
    if not np.array_equal(changed[~moved], vertices[~moved]):
        raise ReconstructionError('Independent ray assistance modified unsupported source geometry')
    # Record exact evidence sources so a later scene can reproduce its review.
    review.update({'frame_registration': frame_review, 'evidence': observations['report'],
                   'source_surface_sha256': protected[source],
                   'scope': 'one anchored session containing its indoor/outdoor captures',
                   'cross_session_merging': 'not performed; recentered origins require separate verified registration',
                   'state': 'limited_surface_corrections' if moved.any() else 'limited_surface_validation'})
    _require_mesh_space(store.root, 2*2**30)
    with _worker_lock(store.root):
        if any(Store._file_sha256(path) != digest for path, digest in protected.items()):
            raise ReconstructionError('A protected raw surface, source job, normalization or splat changed')
        revision = str(uuid.uuid4()); root = job_root/'mesh-revisions'/revision
        root.mkdir(parents=True)
        manifest = {**parent, 'revision_id': revision, 'parent_mesh_revision_id': source_revision_id,
                    'created_utc': utc_now(), 'method': 'photo_mesh_with_independent_spatial_support',
                    'spatial_support': review, 'status': 'running', 'stage': 'exporting_supported_surface',
                    'artifact': None, 'visual_acceptance': 'pending real geometry and textured scene inspection',
                    'uncertainty': {**parent.get('uncertainty', {}),
                        'spatial_depth': 'near-capture OpenXR raycasts; sensor-frame age and RGB synchronization unknown',
                        'house_completeness': 'not established; only photo surfaces and limited ray hits are represented'}}
        def save():
            (root/'mesh.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
        save()
        try:
            for index, key in enumerate(('x', 'y', 'z')):
                ply['vertex'][key] = changed[:, index]
            ply.write(root/'mesh.ply')
            np.savez_compressed(root/'spatial-membership.npz', supporting_views=support)
            (root/'spatial-support.json').write_text(json.dumps(review, indent=2, allow_nan=False))
            manifest['artifact'] = mesh_to_glb(root/'mesh.ply', root/'mesh.glb',
                {key: manifest[key] for key in ('method', 'source_job_id', 'coordinate_frame', 'uncertainty')})
            if any(Store._file_sha256(path) != digest for path, digest in protected.items()):
                raise ReconstructionError('A protected source changed during depth-assisted export')
            manifest['status'] = 'ready'; manifest['stage'] = 'ready'
        except Exception as error:
            manifest['status'] = 'failed'; manifest['stage'] = 'failed'; manifest['error'] = str(error)
            save(); raise
        save()
        return manifest


def verify_surface_geometry(source_vertices, source_faces, target_vertices, target_faces):
    """Verify texture/cleanup preserved every valid geometric triangle.

    UV seam splits, coincident duplicates and pre-existing zero-area faces are
    storage differences. A displaced point, missing valid face or new face is not.
    """
    from scipy.spatial import cKDTree
    source_vertices, target_vertices = (np.asarray(value, dtype=float) for value in
                                        (source_vertices, target_vertices))
    source_faces, target_faces = (np.asarray(value, dtype=np.int64) for value in
                                  (source_faces, target_faces))
    for vertices, faces in ((source_vertices, source_faces), (target_vertices, target_faces)):
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or
                faces.shape[1] != 3 or not np.isfinite(vertices).all() or not len(faces) or
                faces.min() < 0 or faces.max() >= len(vertices)):
            raise ReconstructionError('Invalid geometry preservation input')
    unique, source_ids = np.unique(source_vertices, axis=0, return_inverse=True)
    distance, target_ids = cKDTree(unique).query(target_vertices, workers=2)
    tolerance = max(float(np.linalg.norm(np.ptp(source_vertices, axis=0)))*1e-7, 1e-8)
    if distance.max(initial=0) > tolerance:
        raise ReconstructionError('Texturing or cleanup displaced protected scene geometry')
    def triangles(vertices, faces, ids):
        cross = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                         vertices[faces[:, 2]]-vertices[faces[:, 0]])
        valid = np.linalg.norm(cross, axis=1) > 1e-12
        corners = ids[faces[valid]]
        # Cyclic reindexing preserves winding; a reversed face does not.
        first = np.argmin(corners, axis=1)
        canonical = np.take_along_axis(corners, (first[:, None]+np.arange(3)) % 3, axis=1)
        return np.unique(canonical, axis=0)
    before = triangles(source_vertices, source_faces, source_ids)
    after = triangles(target_vertices, target_faces, target_ids)
    if not np.array_equal(before, after):
        raise ReconstructionError('Texturing or cleanup changed protected valid scene triangles')
    return {'valid_geometric_triangles_preserved': len(before),
            'max_coordinate_difference': float(distance.max(initial=0)), 'tolerance': tolerance,
            'allowed_storage_changes': 'UV splits, duplicate coordinates/faces, pre-existing zero-area faces only'}


def finish_mesh(store: Store, project_id: str, source_job_id: str, source_revision_id: str,
                min_component_faces=25, smooth_steps=2, texture_visible_vertices=3,
                texture_color_correction=True, texture_padding_radius=4, texture_image_size=None,
                texture_min_cos_angle=.1, texture_smoothing_iterations=3, texture_reference_revision_id=None):
    """Clean and photograph-texture a new revision, preserving its raw parent."""
    import pymeshlab
    from plyfile import PlyData
    project_id, source_job_id, source_revision_id = (str(uuid.UUID(value)) for value in
                                                   (project_id, source_job_id, source_revision_id))
    if not 1 <= min_component_faces <= 200 or not 0 <= smooth_steps <= 4 or texture_visible_vertices not in (2, 3):
        raise ReconstructionError('Cleanup settings are outside conservative bounds')
    if type(texture_color_correction) is not bool or not 0 <= texture_padding_radius <= 16:
        raise ReconstructionError('Texture correction settings are outside conservative bounds')
    if texture_image_size is not None and not 640 <= texture_image_size <= 3200:
        raise ReconstructionError('Invalid texture photograph budget')
    if not .1 <= texture_min_cos_angle <= .8 or not 0 <= texture_smoothing_iterations <= 32:
        raise ReconstructionError('Invalid texture view selection budget')
    parent, _ = read_mesh(store, project_id, source_job_id, source_revision_id)
    if (parent.get('detail_preserving_room') or parent.get('spatial_support')) and (smooth_steps != 0 or min_component_faces != 1):
        raise ReconstructionError('Detail-preserving or spatially supported surfaces require zero smoothing and no component pruning')
    job_root = store.root/'projects'/project_id/'reconstructions'/source_job_id
    job = json.loads((job_root/'job.json').read_text())
    parent_root = job_root/'mesh-revisions'/source_revision_id
    source = parent_root/'mesh.ply'
    # MeshLab may add default white vertex colors when saving an uncolored mesh.
    # Only colors present on the actual raw surface can be called photo-derived.
    raw_color_fields = PlyData.read(source)['vertex'].data.dtype.names
    has_photo_colors = all(k in raw_color_fields for k in ('red', 'green', 'blue'))
    protected = {p: Store._file_sha256(p) for p in (source, job_root/'splat.ply')}
    dense = Path(parent['private_work_directory'])/'dense'
    texture_work = Path(parent['private_work_directory'])
    texture_reference = None
    if texture_reference_revision_id:
        from .reconstruct import _read_colmap_images
        texture_reference, _ = read_mesh(store, project_id, source_job_id, texture_reference_revision_id)
        texture_work = Path(texture_reference['private_work_directory'])
        if texture_reference['coordinate_frame'] != 'colmap_sparse' or not texture_image_size:
            raise ReconstructionError('A raw, calibrated texture reference and explicit photo resolution are required')
        if Store._file_sha256(texture_work/'sparse/cameras.bin') != Store._file_sha256(Path(parent['private_work_directory'])/'sparse/cameras.bin'):
            raise ReconstructionError('Texture reference camera calibration differs')
    if not (dense/'sparse').is_dir() or not (dense/'images').is_dir():
        raise ReconstructionError('The calibrated dense workspace is unavailable')
    normalization_path = Path(job['training_config']).parent/'dataparser_transforms.json'
    normalization = json.loads(normalization_path.read_text())
    transform = np.eye(4); transform[:3] = np.asarray(normalization['transform'])
    scale = float(normalization['scale'])
    if not np.isfinite(transform).all() or not np.isfinite(scale) or scale <= 0:
        raise ReconstructionError('The source splat normalization is invalid')
    transform[:3] *= scale
    colmap = os.environ.get('QUEST_SPLAT_COLMAP_EXE') or _tool('colmap')
    with _worker_lock(store.root):
        revision = str(uuid.uuid4()); root = job_root/'mesh-revisions'/revision
        root.mkdir(parents=True)
        manifest = {**parent, 'revision_id': revision, 'parent_mesh_revision_id': source_revision_id,
                    'created_utc': utc_now(), 'status': 'running', 'artifact': None,
                    'method': 'colmap_photo_mvs_cleaned_atlas',
                    'coordinate_frame': 'nerfstudio_training_zup',
                    'uncertainty': {**parent['uncertainty'],
                                    'splat_frame_alignment': 'shared saved training frame; physical placement unverified'},
                    'original_colmap_to_training': transform.tolist()}
        def stage(value):
            manifest['stage'] = value
            temporary = root/'mesh.tmp'; temporary.write_text(json.dumps(manifest, indent=2))
            os.replace(temporary, root/'mesh.json')
        try:
            if texture_image_size:
                stage('preparing_full_resolution_photo_textures')
                work = Path(parent['private_work_directory'])
                dense = work/('texture-photos-'+revision)
                if texture_reference:
                    models=[]
                    for index,folder in enumerate((texture_work,work)):
                        text=work/('texture-pose-check-'+revision+'-'+str(index));text.mkdir()
                        _run([colmap,'model_converter','--input_path',str(folder/'sparse'),'--output_path',str(text),'--output_type','TXT'],
                             root/'worker.log',dict(os.environ),120)
                        models.append(_read_colmap_images(text/'images.txt'))
                    for name,(q,t) in models[0].items():
                        if name not in models[1] or not np.allclose(q,models[1][name][0],atol=1e-6,rtol=0) or not np.allclose(t,models[1][name][1],atol=1e-6,rtol=0):
                            raise ReconstructionError('Texture reference camera pose differs')
                        if Store._file_sha256(texture_work/'images'/name) != Store._file_sha256(work/'images'/name):
                            raise ReconstructionError('Texture reference photograph differs')
                    manifest['texture_reference']={'revision_id':texture_reference_revision_id,
                        'verified_same_frame_photographs':len(models[0]),'source':'restricted verified photo calibration'}
                _run([colmap,'image_undistorter','--image_path',str(texture_work/'images'),'--input_path',str(texture_work/'sparse'),
                      '--output_path',str(dense),'--output_type','COLMAP','--max_image_size',str(texture_image_size)],
                     root/'worker.log',dict(os.environ),1800)
            stage('cleaning_surface')
            ms = pymeshlab.MeshSet(); ms.load_new_mesh(str(source))
            before = {'vertices': ms.current_mesh().vertex_number(), 'faces': ms.current_mesh().face_number()}
            for name in ('meshing_remove_duplicate_vertices', 'meshing_remove_duplicate_faces',
                         'meshing_remove_null_faces', 'meshing_remove_unreferenced_vertices'):
                ms.apply_filter(name)
            ms.apply_filter('meshing_remove_connected_component_by_face_number',
                            mincomponentsize=min_component_faces, removeunref=True)
            if smooth_steps:
                ms.apply_filter('apply_coord_taubin_smoothing', stepsmoothnum=smooth_steps,
                                lambda_=0.5, mu=-0.53)
            after = {'vertices': ms.current_mesh().vertex_number(), 'faces': ms.current_mesh().face_number()}
            if after['faces'] < before['faces'] * 0.8:
                raise ReconstructionError('Cleanup removed more than 20% of faces; candidate rejected')
            clean = root/'mesh.ply'
            ms.save_current_mesh(str(clean), binary=True)
            ply_triangles(clean)
            manifest['cleanup'] = {'before': before, 'after': after, 'min_component_faces': min_component_faces,
                                   'taubin_steps': smooth_steps, 'holes_filled': False,
                                   'raw_vertex_colors_present': has_photo_colors,
                                   'raw_surface_retained': str(source)}
            if parent.get('detail_preserving_room') or parent.get('spatial_support'):
                source_ply = PlyData.read(source); clean_ply = PlyData.read(clean)
                sv = np.column_stack([source_ply['vertex'][key] for key in ('x', 'y', 'z')])
                sf = np.stack(source_ply['face']['vertex_indices'])
                cv = np.column_stack([clean_ply['vertex'][key] for key in ('x', 'y', 'z')])
                cf = np.stack(clean_ply['face']['vertex_indices'])
                manifest['cleanup']['geometry_preservation'] = verify_surface_geometry(sv, sf, cv, cf)
            stage('texturing_from_photographs')
            # Keep native COLMAP's Windows paths short, then retain all texture products.
            short = Path(parent['private_work_directory'])/'finish'/revision
            short.mkdir(parents=True); shutil.copyfile(clean, short/'clean.ply')
            _run([colmap, 'mesh_texturer', '--workspace_path', str(dense), '--input_path', str(short/'clean.ply'),
                  '--output_path', str(short/'textured'), '--MeshTextureMapping.atlas_patch_padding', '4',
                  '--MeshTextureMapping.min_visible_vertices', str(texture_visible_vertices),
                  '--MeshTextureMapping.min_cos_normal_angle', str(texture_min_cos_angle),
                  '--MeshTextureMapping.view_selection_smoothing_iterations', str(texture_smoothing_iterations),
                  '--MeshTextureMapping.apply_color_correction', str(int(texture_color_correction)),
                  '--MeshTextureMapping.inpaint_radius', str(texture_padding_radius)],
                 root/'worker.log', dict(os.environ), 7200)
            shutil.copytree(short/'textured', root/'textured')
            with Image.open(root/'textured/texture.png') as image:
                manifest['texture'] = {'source': 'calibrated captured photographs', 'size': list(image.size),
                                       'sha256': Store._file_sha256(root/'textured/texture.png'),
                                       'min_visible_vertices': texture_visible_vertices,
                                       'min_cos_normal_angle': texture_min_cos_angle,
                                       'view_selection_smoothing_iterations': texture_smoothing_iterations,
                                       'color_correction': texture_color_correction,
                                       'atlas_padding_radius': texture_padding_radius,
                                       'image_size_bound': texture_image_size or parent['dense_settings']['max_image_size'],
                                       'includes_captured_lighting': True}
            stage('packing_textured_surface')
            if parent.get('detail_preserving_room') or parent.get('spatial_support'):
                textured_ply = PlyData.read(root/'textured/mesh.ply')
                tv = np.column_stack([textured_ply['vertex'][key] for key in ('x', 'y', 'z')])
                tf = np.stack(textured_ply['face']['vertex_indices'])
                manifest['texture']['geometry_preservation'] = verify_surface_geometry(cv, cf, tv, tf)
            provenance = {k: manifest[k] for k in ('method', 'source_job_id', 'parent_mesh_revision_id',
                                                   'coordinate_frame', 'original_colmap_to_training', 'uncertainty')}
            manifest['artifact'] = textured_mesh_to_glb(root/'textured/mesh.ply', root/'textured/texture.png',
                                                       root/'mesh.glb', provenance, transform, clean, has_photo_colors)
            if any(Store._file_sha256(p) != digest for p, digest in protected.items()):
                raise ReconstructionError('A protected reconstruction source changed')
            manifest['status'] = 'ready'; stage('ready')
        except Exception as error:
            manifest['status'] = 'failed'; manifest['error'] = str(error); stage('failed'); raise
        return manifest


def install_mesh(app, store, authenticate):
    @app.get('/api/v1/projects/{project_id}/reconstructions/{job_id}/meshes',dependencies=[Depends(authenticate)])
    def revisions(project_id:uuid.UUID,job_id:uuid.UUID):
        store.project(str(project_id))
        root=store.root/'projects'/str(project_id)/'reconstructions'/str(job_id)/'mesh-revisions'
        result=[]
        for p in root.glob('*/mesh.json'):
            try:
                j,_=read_mesh(store,str(project_id),str(job_id),p.parent.name)
                j.pop('input_captures',None)
                j.pop('private_work_directory',None)
                j['artifact'].pop('path',None)
                j['artifact']['download_path']=f'/api/v1/projects/{project_id}/reconstructions/{job_id}/meshes/{j["revision_id"]}'
                result.append(j)
            except (StoreError, ValueError):
                continue
        return {'meshes':result}

    @app.get('/api/v1/projects/{project_id}/reconstructions/{job_id}/meshes/{revision_id}',dependencies=[Depends(authenticate)])
    def download(project_id:uuid.UUID,job_id:uuid.UUID,revision_id:uuid.UUID):
        j,path=read_mesh(store,str(project_id),str(job_id),str(revision_id))
        return FileResponse(path,media_type='model/gltf-binary',headers={'Cache-Control':'private, immutable','ETag':'"'+j['artifact']['sha256']+'"'})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--project-id',required=True); p.add_argument('--source-job-id',required=True)
    p.add_argument('--max-image-size',type=int,default=1600)
    p.add_argument('--stereo-sources', type=int, default=20)
    p.add_argument('--stereo-iterations', type=int, default=5)
    p.add_argument('--stereo-window-step', type=int, default=1)
    p.add_argument('--stereo-gpu-workers', type=int, default=1)
    p.add_argument('--reuse-photo-depth-revision', help='Reuse hash/pose-verified photometric depth; recompute geometric consistency')
    p.add_argument('--resume-revision', help='Repair invalid generated maps and resume an unfinished failed mesh')
    p.add_argument('--extend-calibration', action='store_true', help='Register additional session photos on isolated mesh-only copies')
    p.add_argument('--calibration-revision', help='Use a verified extended mesh-only photo calibration')
    p.add_argument('--finish-revision', help='Clean/texture an existing raw mesh into a new immutable revision')
    p.add_argument('--remesh-revision', help='Review Poisson surface reconstruction from retained depth fusion')
    p.add_argument('--plane-revision', help='Constrain supported flat patches of a raw mesh into a separate revision')
    p.add_argument('--structure-revision', help='Fit a separate architectural base to a confirmed right-angle room')
    p.add_argument('--detail-preserving-revision', help='Straighten supported wall/floor interiors while retaining the raw 3D scene')
    p.add_argument('--depth-evidence-revision', help='Locally validate/refine a raw photo mesh using verified same-session Quest raycasts')
    p.add_argument('--room-reference-revision', help='Verified same-frame room plane fit used only for architectural coordinates')
    p.add_argument('--plane-displacement-fraction', type=float, default=.008)
    p.add_argument('--plane-support-fraction', type=float, default=.006)
    p.add_argument('--right-angle-room', action='store_true', help='Use the user-confirmed rectangular room constraint')
    p.add_argument('--poisson-depth', type=int, default=10)
    p.add_argument('--density-trim', type=float, default=7.0)
    p.add_argument('--min-component-faces', type=int, default=25)
    p.add_argument('--smooth-steps', type=int, default=2)
    p.add_argument('--texture-visible-vertices', type=int, default=3)
    p.add_argument('--texture-color-correction', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--texture-padding-radius', type=int, default=4)
    p.add_argument('--texture-image-size', type=int, help='Independently use original-resolution photos for texturing')
    p.add_argument('--texture-min-cos-angle', type=float, default=.1)
    p.add_argument('--texture-smoothing-iterations', type=int, default=3)
    p.add_argument('--texture-reference-revision', help='Use a verified same-frame raw photo calibration solely for textures')
    a=p.parse_args()
    store=Store(a.data_dir)
    if sum(bool(value) for value in (a.finish_revision, a.remesh_revision, a.plane_revision, a.structure_revision, a.detail_preserving_revision, a.depth_evidence_revision, a.extend_calibration, a.resume_revision)) > 1:
        p.error('Choose one revision operation')
    if bool(a.detail_preserving_revision) != bool(a.room_reference_revision):
        p.error('Detail-preserving constraints require both raw surface and room reference revisions')
    if a.resume_revision:
        result=resume_mesh(store,a.project_id,a.source_job_id,a.resume_revision)
    elif a.extend_calibration:
        result=extend_photo_mesh_calibration(store,a.project_id,a.source_job_id)
    elif a.depth_evidence_revision:
        result=depth_assisted_mesh_revision(store,a.project_id,a.source_job_id,a.depth_evidence_revision)
    elif a.detail_preserving_revision:
        result=detail_preserving_room_revision(store,a.project_id,a.source_job_id,
                                               a.detail_preserving_revision,a.room_reference_revision)
    elif a.structure_revision:
        result=structural_room_revision(store,a.project_id,a.source_job_id,a.structure_revision)
    elif a.plane_revision:
        result=regularize_planes(store,a.project_id,a.source_job_id,a.plane_revision,
                                 a.plane_displacement_fraction,a.plane_support_fraction,a.right_angle_room)
    elif a.remesh_revision:
        result=remesh_surface(store,a.project_id,a.source_job_id,a.remesh_revision,a.poisson_depth,a.density_trim)
    elif a.finish_revision:
        result=finish_mesh(store,a.project_id,a.source_job_id,a.finish_revision,a.min_component_faces,a.smooth_steps,
                           a.texture_visible_vertices,a.texture_color_correction,a.texture_padding_radius,a.texture_image_size,
                           a.texture_min_cos_angle,a.texture_smoothing_iterations,a.texture_reference_revision)
    else:
        result=build_mesh(store,a.project_id,a.source_job_id,a.max_image_size,a.stereo_sources,
                          a.stereo_iterations,a.stereo_window_step,a.calibration_revision,a.stereo_gpu_workers,
                          a.reuse_photo_depth_revision)
    print(json.dumps(result,indent=2))
