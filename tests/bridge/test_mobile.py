import hashlib
import io
import json
import uuid
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from bridge.app import create_app
from bridge.config import Config
from bridge.jobs import set_processing_paused
from bridge.preview import register_job
from bridge.mesh import ply_triangles,mesh_to_glb,ReconstructionError
import pytest


def phone(tmp_path,large=False):
    token='test-private-phone-token-'+'x'*32
    client=TestClient(create_app(Config(tmp_path,token)));h={'Authorization':'Bearer '+token}
    p=client.post('/api/v1/projects',headers=h,json={'name':'Synthetic phone fixture'}).json()
    s=client.post('/api/v1/sessions',headers=h,json={'project_id':p['id'],'tracking_origin_id':str(uuid.uuid4())}).json()
    b=io.BytesIO();im=Image.fromarray(np.random.default_rng(22).integers(0,255,(1536,2048,3),dtype=np.uint8)) if large else Image.new('RGB',(32,24),'blue')
    im.save(b,'JPEG',quality=95);data=b.getvalue()
    m={'schema_version':2,'capture_source':'synthetic_phone_test','capture_id':str(uuid.uuid4()),'project_id':p['id'],'session_id':s['id'],'tracking_origin_id':s['tracking_origin_id'],
       'camera_id':'fixture-rear','image_timestamp_raw':'2026-10-09T10:01:00Z','timestamp_clock':'android_wall_utc','sensor_timestamp_source':'unknown','image_width':im.width,'image_height':im.height,
       'image_encoding':'jpeg','image_sha256':hashlib.sha256(data).hexdigest(),'image_size_bytes':len(data),'image_orientation':'exif','jpeg_orientation_degrees':90,
       'camera_to_session':None,'tracking_valid':False,'intrinsics':None,'calibration_reference':'unavailable','device_model':'SYNTHETIC FIXTURE'}
    return client,h,p,s,m,data


def test_interruption_restart_duplicates_and_integrity(tmp_path):
    c,h,p,s,m,data=phone(tmp_path,large=True)
    assert len(data)>1024*1024
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    assert c.post(base,json=m).status_code==401
    assert c.post(base,headers=h,json=m).json()['offset']==0
    part=data[:1024*1024];chunk_headers={**h,'Upload-Offset':'0','Upload-Chunk-SHA256':hashlib.sha256(part).hexdigest()}
    assert c.put(path,headers=chunk_headers,content=part).json()['offset']==len(part)
    # Lost acknowledgement retries discover the offset; repeating old offset cannot append twice.
    assert c.put(path,headers=chunk_headers,content=part).status_code==409
    assert c.post(path+'/commit',headers=h).status_code==409
    c=TestClient(create_app(Config(tmp_path,h['Authorization'][7:])))
    assert c.get(path,headers=h).json()['offset']==len(part)
    offset=len(part)
    while offset<len(data):
        part=data[offset:offset+1024*1024];hh={**h,'Upload-Offset':str(offset),'Upload-Chunk-SHA256':hashlib.sha256(part).hexdigest()}
        assert c.put(path,headers={**hh,'Upload-Chunk-SHA256':'0'*64},content=part).status_code==422
        r=c.put(path,headers=hh,content=part);assert r.status_code==200;offset=r.json()['offset']
    receipt=c.post(path+'/commit',headers=h).json();assert receipt['durable'] and not receipt['duplicate']
    assert c.post(path+'/commit',headers=h).json()['duplicate']
    assert c.post(base,headers=h,json=m).json()['receipt']['duplicate']
    assert c.post(base,headers=h,json={**m,'jpeg_orientation_degrees':180}).status_code==409
    stored=tmp_path/'projects'/p['id']/'captures'/(m['capture_id']+'.jpg')
    assert stored.read_bytes()==data
    assert c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()['uploaded']==1
    with c.app.state.store.connection() as db:
        saved=json.loads(db.execute('select metadata_json from captures').fetchone()[0])
    assert saved['camera_to_session'] is None and saved['tracking_valid'] is False


def test_large_invalid_phone_metadata_and_chunk(tmp_path):
    c,h,p,s,m,data=phone(tmp_path);base=f'/api/v1/mobile/sessions/{s["id"]}/uploads'
    for change in ({'image_size_bytes':64*1024*1024+1},{'camera_to_session':[1]*16},{'tracking_valid':True},{'sensor_timestamp_ns':-1},{'image_timestamp_raw':'2026-10-09T10:00:00'}):
        assert c.post(base,headers=h,json={**m,**change}).status_code==422
    assert c.post(base,headers=h,json=m).status_code==200
    huge=b'0'*(1024*1024+1)
    assert c.put(base+'/'+m['capture_id'],headers={**h,'Upload-Offset':'0','Upload-Chunk-SHA256':hashlib.sha256(huge).hexdigest()},content=huge).status_code==413


def test_exact_64_mib_photo_limit(tmp_path):
    c,h,p,s,m,data=phone(tmp_path)
    # Clearly synthetic JPEG padding exercises the transport byte limit without
    # expensive image generation or any physical capture claim.
    data=data+b'\0'*(64*1024*1024-len(data));m['image_size_bytes']=len(data);m['image_sha256']=hashlib.sha256(data).hexdigest()
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    assert c.post(base,headers=h,json=m).status_code==200
    for offset in range(0,len(data),1024*1024):
        part=data[offset:offset+1024*1024]
        assert c.put(path,headers={**h,'Upload-Offset':str(offset),'Upload-Chunk-SHA256':hashlib.sha256(part).hexdigest()},content=part).status_code==200
    r=c.post(path+'/commit',headers=h);assert r.status_code==200 and r.json()['durable']


def test_analyzed_count_uses_features_not_receipts(tmp_path):
    import sqlite3
    from bridge.reconstruct import _analyzed_views
    processed=tmp_path/'processed';(processed/'colmap').mkdir(parents=True)
    assert _analyzed_views(processed) is None
    with sqlite3.connect(processed/'colmap/database.db') as db:
        db.execute('CREATE TABLE keypoints (image_id INTEGER,rows INTEGER)')
        db.executemany('INSERT INTO keypoints VALUES (?,?)',[(1,35),(2,0),(3,72)])
    assert _analyzed_views(processed)==2


@pytest.mark.parametrize('orientation,expected',[(1,(32,24)),(3,(32,24)),(6,(24,32)),(8,(24,32))])
def test_mobile_portrait_landscape_exif_preserves_original(tmp_path,orientation,expected):
    from bridge.reconstruct import _normalize_phone_jpeg
    source=tmp_path/'camera-original.jpg';destination=tmp_path/'training-copy.jpg'
    im=Image.new('RGB',(32,24),'blue');exif=im.getexif();exif[274]=orientation;exif[271]='Synthetic camera fixture';im.save(source,exif=exif)
    original=source.read_bytes();_normalize_phone_jpeg(source,destination)
    assert source.read_bytes()==original
    with Image.open(destination) as normalized:
        assert normalized.size==expected and normalized.getexif()[274]==1
        assert normalized.getexif()[271]=='Synthetic camera fixture'


def test_unknown_stage_and_pause_preserved(tmp_path):
    c,h,p,s,m,data=phone(tmp_path);set_processing_paused(tmp_path,True)
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    assert st['preview'] is None and st['analyzed'] is None and st['registered'] is None and st['processing_paused']
    assert register_job(c.app.state.store,{'capture_sources':['android_camera2']})['headset_preview_allowed'] is False


def test_mesh_triangle_export_rejects_point_cloud(tmp_path):
    p=tmp_path/'surface.ply';p.write_text('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n3 0 1 2\n3 0 2 3\n')
    v,c,t=ply_triangles(p);assert len(t)==2
    artifact=mesh_to_glb(p,tmp_path/'mesh.glb',{'source':'synthetic fixture','accuracy':'unknown'})
    assert artifact['triangle_count']==2
    import struct
    b=(tmp_path/'mesh.glb').read_bytes();magic,version,size=struct.unpack('<III',b[:12]);assert magic==0x46546c67 and version==2 and size==len(b)
    n=struct.unpack('<I',b[12:16])[0];j=json.loads(b[20:20+n]);assert j['meshes'][0]['primitives'][0]['mode']==4
    p.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n')
    with pytest.raises(ReconstructionError,match='surface'):mesh_to_glb(p,tmp_path/'bad.glb',{})


def test_completed_preview_survives_failed_new_revision(tmp_path):
    c,h,p,s,m,data=phone(tmp_path)
    from bridge.reconstruct import validate_gaussian_ply
    import struct
    names=['x','y','z','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','f_dc_0','f_dc_1','f_dc_2']
    jid=str(uuid.uuid4());root=tmp_path/'projects'/p['id']/'reconstructions'/jid;root.mkdir(parents=True)
    ply=root/'splat.ply';ply.write_bytes(('ply\nformat binary_little_endian 1.0\nelement vertex 2\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n').encode()+struct.pack('<14f',*([0.0]*14))*2)
    vertices,digest,size=validate_gaussian_ply(ply)
    j={'project_id':p['id'],'session_id':s['id'],'job_id':jid,'status':'ready_for_blender','created_utc':'2026-10-09T00:00:00Z','artifact':{'path':str(ply),'sha256':digest,'size_bytes':size,'vertex_count':vertices}}
    (root/'job.json').write_text(json.dumps(j))
    other=root.parent/str(uuid.uuid4());other.mkdir();(other/'job.json').write_text(json.dumps({**j,'status':'failed','created_utc':'2026-10-10T00:00:00Z','processing_stage':'failed'}))
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    assert st['processing_stage']=='failed' and st['preview']['job_id']==jid
    assert c.get(st['preview']['download_path'],headers=h).content==ply.read_bytes()


def upload_chunk(client, headers, path, data, offset=0):
    return client.put(path, headers={**headers, 'Upload-Offset': str(offset),
                      'Upload-Chunk-SHA256': hashlib.sha256(data).hexdigest()}, content=data)


def test_completed_offset_and_receipt_survive_restart_and_session_finish(tmp_path):
    c,h,p,s,m,data=phone(tmp_path)
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    assert c.post(base,headers=h,json=m).status_code==200
    assert upload_chunk(c,h,path,data).status_code==200
    assert c.post(path+'/commit',headers=h).json()['durable']
    assert c.post(f'/api/v1/sessions/{s["id"]}/finish',headers=h).status_code==200
    c=TestClient(create_app(Config(tmp_path,h['Authorization'][7:])))
    result=c.get(path,headers=h).json()
    assert result['offset']==len(data)
    assert result['receipt']['durable'] and result['receipt']['duplicate']
    assert c.post(base,headers=h,json=m).json()['offset']==len(data)
    assert c.post(path+'/commit',headers=h).json()['capture_id']==m['capture_id']
    assert upload_chunk(c,h,path,data).status_code==409
    assert not (tmp_path/'phone-upload-staging'/s['id']/m['capture_id']/'image.part').exists()


def test_failed_chunk_fsync_does_not_publish_unacknowledged_tail(tmp_path,monkeypatch):
    import bridge.mobile as mobile
    c,h,p,s,m,data=phone(tmp_path)
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    assert c.post(base,headers=h,json=m).status_code==200
    assert upload_chunk(c,h,path,data[:100]).json()['offset']==100
    def disk_failure(_):
        raise OSError('Synthetic interrupted flush')
    with monkeypatch.context() as patch:
        patch.setattr(mobile.os,'fsync',disk_failure)
        with pytest.raises(OSError,match='Synthetic interrupted flush'):
            upload_chunk(c,h,path,data[100:200],100)
    c=TestClient(create_app(Config(tmp_path,h['Authorization'][7:])))
    assert c.get(path,headers=h).json()['offset']==100
    assert c.post(base,headers=h,json=m).json()['offset']==100
    assert upload_chunk(c,h,path,data[100:],100).json()['offset']==len(data)
    assert c.post(path+'/commit',headers=h).json()['durable']
    assert (tmp_path/'projects'/p['id']/'captures'/(m['capture_id']+'.jpg')).read_bytes()==data


def test_lost_commit_reply_recovers_original_and_single_receipt(tmp_path,monkeypatch):
    from bridge.store import StoreError
    c,h,p,s,m,data=phone(tmp_path)
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    assert c.post(base,headers=h,json=m).status_code==200
    assert upload_chunk(c,h,path,data).status_code==200
    save=c.app.state.store.save_capture
    def lost_reply(*args):
        save(*args)
        raise StoreError(503,'Synthetic lost commit response')
    monkeypatch.setattr(c.app.state.store,'save_capture',lost_reply)
    assert c.post(path+'/commit',headers=h).status_code==503
    c=TestClient(create_app(Config(tmp_path,h['Authorization'][7:])))
    receipt=c.post(path+'/commit',headers=h).json()
    assert receipt['durable'] and receipt['duplicate']
    assert c.get(path,headers=h).json()['offset']==len(data)
    assert c.get(f'/api/v1/sessions/{s["id"]}',headers=h).json()['capture_count']==1
    assert (tmp_path/'projects'/p['id']/'captures'/(m['capture_id']+'.jpg')).read_bytes()==data


def test_truncated_jpeg_with_matching_declared_hash_is_rejected(tmp_path):
    c,h,p,s,m,data=phone(tmp_path)
    data=data[:-20];m={**m,'image_size_bytes':len(data),'image_sha256':hashlib.sha256(data).hexdigest()}
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    assert c.post(base,headers=h,json=m).status_code==200
    assert upload_chunk(c,h,path,data).status_code==200
    assert c.post(path+'/commit',headers=h).status_code==422
    assert c.get(f'/api/v1/sessions/{s["id"]}',headers=h).json()['capture_count']==0


def synthetic_job(tmp_path,project,session,created,**fields):
    jid=str(uuid.uuid4());root=tmp_path/'projects'/project['id']/'reconstructions'/jid
    root.mkdir(parents=True)
    job={'project_id':project['id'],'session_id':session['id'],'job_id':jid,
         'status':'running','created_utc':created,**fields}
    (root/'job.json').write_text(json.dumps(job))
    return root,job


def synthetic_mesh(job_root,job,created,status='ready'):
    rid=str(uuid.uuid4());root=job_root/'mesh-revisions'/rid;root.mkdir(parents=True)
    mesh={'revision_id':rid,'source_job_id':job['job_id'],'project_id':job['project_id'],
          'created_utc':created,'status':status,'stage':status,'artifact':None,
          'method':'synthetic_triangle_fixture','coordinate_frame':'colmap_sparse',
          'input_captures':[],'calibrated_photos':[],
          'uncertainty':{'metric_scale':'unverified','splat_frame_alignment':'not established'}}
    if status=='ready':
        ply=root/'mesh.ply'
        ply.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
        mesh['artifact']=mesh_to_glb(ply,root/'mesh.glb',{'source':'synthetic fixture'})
    (root/'mesh.json').write_text(json.dumps(mesh))
    return root,mesh


@pytest.mark.parametrize('new_status',['running','failed'])
def test_ready_mesh_retained_through_newer_attempts(tmp_path,new_status):
    c,h,p,s,m,data=phone(tmp_path)
    first,job=synthetic_job(tmp_path,p,s,'2026-10-09T00:00:00Z')
    mesh_root,mesh=synthetic_mesh(first,job,'2026-10-09T00:01:00Z')
    second,new_job=synthetic_job(tmp_path,p,s,'2026-10-10T00:00:00Z',status=new_status)
    _,attempt=synthetic_mesh(second,new_job,'2026-10-10T00:01:00Z',new_status)
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    ready=st['mesh']
    assert ready['revision_id']==mesh['revision_id'] and ready['status']=='ready'
    for key in ('sha256','size_bytes','vertex_count','triangle_count'):
        assert ready[key]==mesh['artifact'][key]
    assert ready['uncertainty']==mesh['uncertainty']
    assert ready['provenance']=={'input_captures':[],'calibrated_photos':[]}
    assert c.get(ready['download_path'],headers=h).content==(mesh_root/'mesh.glb').read_bytes()
    assert st['mesh_processing']['revision_id']==attempt['revision_id']
    assert st['mesh_processing']['status']==new_status


@pytest.mark.parametrize('defect',['bytes','size','identity','path'])
def test_unverified_mesh_never_replaces_previous_ready_revision(tmp_path,defect):
    c,h,p,s,m,data=phone(tmp_path)
    job_root,job=synthetic_job(tmp_path,p,s,'2026-10-09T00:00:00Z')
    _,good=synthetic_mesh(job_root,job,'2026-10-09T00:01:00Z')
    broken_root,broken=synthetic_mesh(job_root,job,'2026-10-10T00:01:00Z')
    if defect=='bytes': (broken_root/'mesh.glb').write_bytes(b'corrupt synthetic mesh')
    elif defect=='size': broken['artifact']['size_bytes']+=1
    elif defect=='identity': broken['project_id']=str(uuid.uuid4())
    else: broken['artifact']['path']=str(tmp_path/'unexpected.glb')
    (broken_root/'mesh.json').write_text(json.dumps(broken))
    route=f'/api/v1/projects/{p["id"]}/reconstructions/{job["job_id"]}/meshes'
    assert c.get(route+'/'+broken['revision_id'],headers=h).status_code==404
    assert [x['revision_id'] for x in c.get(route,headers=h).json()['meshes']]==[good['revision_id']]
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    assert st['mesh']['revision_id']==good['revision_id']


@pytest.mark.parametrize('analyzed,registered',[(4,3),(None,3),(4,None),(0,0)])
def test_status_counts_remain_independent_measurements(tmp_path,analyzed,registered):
    c,h,p,s,m,data=phone(tmp_path)
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    c.post(base,headers=h,json=m);upload_chunk(c,h,path,data);c.post(path+'/commit',headers=h)
    synthetic_job(tmp_path,p,s,'2026-10-09T00:00:00Z',analyzed_views=analyzed,registered_views=registered)
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    assert st['uploaded']==1 and st['analyzed']==analyzed and st['registered']==registered
    synthetic_job(tmp_path,p,s,'2026-10-10T00:00:00Z')
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    assert st['uploaded']==1 and st['analyzed'] is None and st['registered'] is None


def test_begin_recovers_commit_that_finishes_before_upload_lock(tmp_path,monkeypatch):
    from contextlib import contextmanager
    import bridge.mobile as mobile
    c,h,p,s,m,data=phone(tmp_path)
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    c.post(base,headers=h,json=m);assert upload_chunk(c,h,path,data).status_code==200
    lock=mobile._upload_lock
    armed=True
    @contextmanager
    def commit_before_lock(root):
        nonlocal armed
        if armed:
            armed=False
            assert c.post(path+'/commit',headers=h).json()['durable']
        with lock(root):
            yield
    monkeypatch.setattr(mobile,'_upload_lock',commit_before_lock)
    result=c.post(base,headers=h,json=m).json()
    assert result['offset']==len(data) and result['receipt']['durable']


@pytest.mark.parametrize('defect',['full_hash','declared_size','dimensions'])
def test_invalid_photo_never_publishes_durable_receipt(tmp_path,defect):
    c,h,p,s,m,data=phone(tmp_path)
    if defect=='full_hash': m['image_sha256']='0'*64
    elif defect=='declared_size': m['image_size_bytes']-=1
    else: m['image_width']+=1
    base=f'/api/v1/mobile/sessions/{s["id"]}/uploads';path=base+'/'+m['capture_id']
    assert c.post(base,headers=h,json=m).status_code==200
    response=upload_chunk(c,h,path,data)
    if defect=='declared_size':
        assert response.status_code==413
        assert c.get(path,headers=h).json()['offset']==0
    else:
        assert response.status_code==200
        assert c.post(path+'/commit',headers=h).status_code==422
    assert c.get(path,headers=h).json()['receipt'] is None
    assert c.get(f'/api/v1/sessions/{s["id"]}',headers=h).json()['capture_count']==0


def test_all_mobile_and_mesh_routes_require_matching_auth(tmp_path):
    c,h,p,s,m,data=phone(tmp_path)
    base=f'/api/v1/mobile/sessions/{s["id"]}';path=base+'/uploads/'+m['capture_id']
    mesh=f'/api/v1/projects/{p["id"]}/reconstructions/{uuid.uuid4()}/meshes'
    requests=[('post',base+'/uploads',{'json':m}),('get',path,{}),
              ('put',path,{'content':data}),('post',path+'/commit',{}),
              ('get',base+'/status',{}),('post',base+'/update',{}),
              ('get',mesh,{}),('get',mesh+'/'+str(uuid.uuid4()),{})]
    for headers in ({},{'Authorization':'Bearer wrong-synthetic-token'}):
        for method,url,kwargs in requests:
            assert getattr(c,method)(url,headers=headers,**kwargs).status_code==401
    assert not (tmp_path/'phone-upload-staging').exists()


def test_invalid_optional_preview_falls_back_to_verified_bounded_original(tmp_path):
    import struct
    from bridge.reconstruct import validate_gaussian_ply
    c,h,p,s,m,data=phone(tmp_path)
    root,job=synthetic_job(tmp_path,p,s,'2026-10-09T00:00:00Z',status='ready_for_blender')
    names=['x','y','z','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','f_dc_0','f_dc_1','f_dc_2']
    ply=root/'splat.ply'
    ply.write_bytes(('ply\nformat binary_little_endian 1.0\nelement vertex 2\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n').encode()+struct.pack('<14f',*([0.0]*14))*2)
    vertices,digest,size=validate_gaussian_ply(ply)
    job['artifact']={'path':str(ply),'sha256':digest,'size_bytes':size,'vertex_count':vertices}
    job['quest_preview']={**job['artifact'],'path':str(root/'quest-preview.ply'),'source_sha256':digest}
    (root/'job.json').write_text(json.dumps(job))
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    assert st['preview']['job_id']==job['job_id']
    assert st['preview']['download_path'].endswith('/splat')
    assert c.get(st['preview']['download_path'],headers=h).content==ply.read_bytes()


@pytest.mark.parametrize('budget',['size_bytes','vertex_count','triangle_count'])
def test_mobile_mesh_budget_retains_older_supported_revision(tmp_path,monkeypatch,budget):
    import bridge.mobile as mobile
    c,h,p,s,m,data=phone(tmp_path)
    root,job=synthetic_job(tmp_path,p,s,'2026-10-09T00:00:00Z')
    _,good=synthetic_mesh(root,job,'2026-10-09T00:01:00Z')
    large_root,large=synthetic_mesh(root,job,'2026-10-10T00:01:00Z')
    if budget=='size_bytes':
        # Valid synthetic GLB with larger JSON exercises the byte budget cheaply.
        assert mobile.MAX_PHONE_BYTES==64*1024*1024
        monkeypatch.setattr(mobile,'MAX_PHONE_BYTES',good['artifact']['size_bytes'])
        (large_root/'mesh.glb').unlink()
        large['artifact']=mesh_to_glb(large_root/'mesh.ply',large_root/'mesh.glb',{'source':'synthetic fixture','padding':'x'*1024})
    else:
        large['artifact'][budget]={'vertex_count':250001,'triangle_count':500001}[budget]
    (large_root/'mesh.json').write_text(json.dumps(large))
    st=c.get(f'/api/v1/mobile/sessions/{s["id"]}/status',headers=h).json()
    assert st['mesh']['revision_id']==good['revision_id']
    assert st['mesh_processing']['revision_id']==large['revision_id']
