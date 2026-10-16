"""A provisional shared depth map preserves identity, uncertainty and open doors."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import uuid

import numpy as np
import pytest
from PIL import Image

from bridge import mesh_spatial
from bridge.mesh_spatial import update_session_spatial_map, export_session_spatial_map_ply
from bridge.reconstruct import ReconstructionError
from bridge.store import Store


@pytest.fixture
def room(tmp_path):
    store = Store(tmp_path)
    project = store.create_project('Provisional depth-map fixture')
    session = store.create_session(project['id'], str(uuid.uuid4()))
    return store, project, session


def capture(room, index, *, point=(1.012, .232, .232), normal=(-1., 0., 0.),
            repeats=1, origin=None, status='hit', confidence=.95, sidecar=True):
    store, project, session = room
    capture_id = str(uuid.uuid4())
    pose = np.eye(4); pose[:3, 3] = origin if origin is not None else [.1*index, 0., 0.]
    out = io.BytesIO(); Image.new('RGB', (32, 24), (20+index, 35, 50)).save(out, 'JPEG')
    pixels = out.getvalue(); timestamp = '2026-10-12T15:00:00.0000000Z'
    metadata = {'schema_version': 1, 'capture_id': capture_id, 'project_id': project['id'],
        'session_id': session['id'], 'tracking_origin_id': session['tracking_origin_id'],
        'camera_id': 'left', 'image_timestamp_raw': timestamp, 'image_timestamp_kind': 'Utc',
        'timestamp_clock': 'meta_datetime', 'image_width': 32, 'image_height': 24,
        'image_encoding': 'jpeg', 'image_sha256': hashlib.sha256(pixels).hexdigest(),
        'intrinsics': {'fx': 20., 'fy': 20., 'cx': 16., 'cy': 12.},
        'camera_to_session': pose.ravel().tolist(), 'coordinate_convention': 'unity_lh_xright_yup_zforward',
        'position_units': 'meters', 'tracking_valid': True,
        'image_orientation': 'upright', 'capture_source': 'quest_camera'}
    payload = None
    if sidecar:
        direction = np.asarray(point)-pose[:3, 3]; direction /= np.linalg.norm(direction)
        ray = {'origin': pose[:3, 3].tolist(), 'direction': direction.tolist(), 'status': status}
        if status == 'hit':
            ray.update(hit=list(point), normal=None if normal is None else list(normal),
                       normal_confidence=confidence if normal is not None else None)
        evidence = {'schema_version': 1, 'capture_id': capture_id, 'project_id': project['id'],
            'session_id': session['id'], 'tracking_origin_id': session['tracking_origin_id'],
            'source': 'meta_environment_raycast_openxr', 'association': 'near_capture_unaligned',
            'coordinate_frame': 'quest_session', 'coordinate_convention': 'unity_lh_xright_yup_zforward',
            'position_units': 'meters', 'trigger_realtime_s': 100.+index,
            'sample_started_realtime_s': 100.02+index, 'sample_completed_realtime_s': 100.04+index,
            'rgb_timestamp_raw': timestamp, 'rgb_timestamp_kind': 'Utc',
            'availability': 'ready', 'rays': [deepcopy(ray) for _ in range(repeats)]}
        payload = json.dumps(evidence).encode()
        metadata['spatial_evidence_sha256'] = hashlib.sha256(payload).hexdigest()
    store.save_capture(session['id'], metadata, pixels, payload)
    return capture_id


def build(room, ids, **kwargs):
    return update_session_spatial_map(room[0], room[2]['id'], ids, **kwargs)


def surfaces(state):
    return [value for cell in state['cells'].values() for value in cell['surfaces']]


def state_path(room):
    store, project, session = room
    return store.root/'projects'/project['id']/'sessions'/session['id']/'spatial-map'/'current.json'


def originals(room):
    result = {}
    for value in room[0].export_session(room[2]['id'])['captures']:
        for key in ('image_path', 'spatial_path'):
            if key in value:
                path = Path(value[key]); result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_single_view_stays_tentative_and_unknowns_never_become_free_space(room):
    ids = [capture(room, 0), capture(room, 1, status='no_hit'),
           capture(room, 2, normal=None), capture(room, 3, confidence=.5),
           capture(room, 4, sidecar=False)]
    before = originals(room)
    state = build(room, ids)
    assert state['summary']['accepted_hits'] == 1
    assert state['summary']['unknown_capture_count'] == 4
    assert state['summary']['unknown_ray_count'] == 3
    assert [value['state'] for value in surfaces(state)] == ['tentative_single_view']
    assert 'not verified' in state['limitations']['photo_geometry']
    assert 'no free-space carving' in state['limitations']['unknown_space']
    assert state['kind'] == 'provisional_anchor_frame_depth'
    assert originals(room) == before


def test_three_translated_captures_support_one_observed_surface(room):
    ids = [capture(room, index) for index in range(3)]
    state = build(room, ids)
    assert len(surfaces(state)) == 1
    surface = surfaces(state)[0]
    assert surface['state'] == surface['support_level'] == 'supported'
    assert surface['independent_view_count'] == 3
    assert set(surface['independent_capture_ids']) == set(ids)
    assert surface['point'] == [1.012, .232, .232]
    assert state['summary']['capture_count'] == 3
    for value in state['input_ledger'].values():
        assert value['receipt']['image_sha256']
        assert value['receipt']['spatial_evidence_sha256']
        assert value['receipt']['metadata_sha256']


def test_spam_and_repeated_origins_cannot_create_independent_support(room):
    ids = [capture(room, index, repeats=100, origin=[0., 0., 0.]) for index in range(3)]
    state = build(room, ids)
    surface = surfaces(state)[0]
    assert surface['observation_count'] == 300
    assert surface['independent_view_count'] == 1
    assert surface['state'] == 'tentative_repeated_origin'


def test_added_capture_merges_only_delta_and_preserves_unaffected_mature_voxel(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids)
    old_cell = deepcopy(next(iter(first['cells'].values())))
    before = originals(room)
    new = capture(room, 3, point=(2.012, .232, .232))
    second = build(room, ids+[new])
    old_key = next(iter(first['cells']))
    assert second['cells'][old_key] == old_cell
    assert second['update']['processed_capture_ids'] == [new]
    assert second['update']['changed_voxel_count'] == 1
    assert second['summary']['accepted_hits'] == 4
    assert all(originals(room)[key] == digest for key, digest in before.items())


def test_retry_is_byte_identical_and_cannot_double_count(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids); content = state_path(room).read_bytes()
    second = build(room, list(reversed(ids)))
    assert second == first
    assert state_path(room).read_bytes() == content
    assert surfaces(second)[0]['observation_count'] == 3


def test_unprotected_evidence_is_deterministic_independent_of_arrival_order(tmp_path):
    store = Store(tmp_path); project = store.create_project('Arrival order fixture')
    session = store.create_session(project['id'], str(uuid.uuid4())); room = store, project, session
    ids = [capture(room, index, point=(1.011+index*.001, .232, .232)) for index in range(2)]
    all_at_once = build(room, ids)
    state_path(room).unlink()
    for count in range(1, 3):
        incremental = build(room, list(reversed(ids))[:count])
    assert incremental['cells'] == all_at_once['cells']
    assert incremental['input_ledger'] == all_at_once['input_ledger']
    assert incremental['input_revision'] == all_at_once['input_revision']


@pytest.mark.parametrize('normal', [(1., 0., 0.), (0., 0., 1.)])
def test_opposite_or_angled_door_normals_never_blend_into_wall(room, normal):
    wall = [capture(room, index) for index in range(3)]
    door = [capture(room, index+3, normal=normal) for index in range(3)]
    state = build(room, wall+door)
    assert len(surfaces(state)) == 2
    assert {tuple(value['normal']) for value in surfaces(state)} == {(-1., 0., 0.), normal}
    assert all(value['state'] == 'supported' for value in surfaces(state))
    assert all(value['conflicting_with'] == [] for value in surfaces(state))


def test_single_dense_conflict_does_not_move_or_erase_supported_surface(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids); original_point = surfaces(first)[0]['point']
    bad = capture(room, 3, point=(1.035, .232, .232), repeats=300)
    state = build(room, ids+[bad])
    assert len(surfaces(state)) == 2
    good = next(value for value in surfaces(state) if value['support_level'] == 'supported')
    assert good['state'] == 'supported' and good['point'] == original_point
    assert good['independent_view_count'] == 3
    assert next(value for value in surfaces(state) if value is not good)['state'] == 'tentative_conflict'


def test_independent_parallel_disagreement_is_retained_and_marked_conflicting(room):
    ids = [capture(room, index) for index in range(3)]
    other = [capture(room, index+3, point=(1.035, .232, .232)) for index in range(3)]
    state = build(room, ids+other)
    assert len(surfaces(state)) == 2
    assert all(value['state'] == 'conflicting' for value in surfaces(state))
    assert {value['point'][0] for value in surfaces(state)} == {1.012, 1.035}
    assert state['summary']['surface_states']['conflicting'] == 2


def test_exclusion_rebuilds_derived_map_without_changing_originals(room):
    ids = [capture(room, index) for index in range(3)]
    build(room, ids); before = originals(room)
    state = build(room, ids[:2])
    assert state['update']['mode'] == 'rebuild_after_exclusion'
    assert state['update']['removed_capture_ids'] == ids[2:]
    assert set(state['input_ledger']) == set(ids[:2])
    assert surfaces(state)[0]['state'] == 'tentative_multi_view'
    assert 'not inversion' in state['limitations']['exclusion']
    assert originals(room) == before
    empty = build(room, [])
    assert empty['cells'] == empty['input_ledger'] == {}
    assert empty['summary']['surface_sample_count'] == 0
    assert originals(room) == before


@pytest.mark.parametrize('condition', ['missing_origin', 'mixed_origin', 'wrong_project', 'wrong_receipt',
                                      'missing_id', 'duplicate_id', 'changed_old_hash'])
def test_mixed_missing_and_changed_identity_cannot_replace_current_map(room, condition):
    ids = [capture(room, index) for index in range(3)]
    build(room, ids); before = state_path(room).read_bytes(); original_hashes = originals(room)
    store, _, session = room; snapshot = store.export_session(session['id'])
    if condition == 'missing_origin': snapshot['session'].pop('tracking_origin_id')
    elif condition == 'mixed_origin': snapshot['captures'][0]['metadata']['tracking_origin_id'] = str(uuid.uuid4())
    elif condition == 'wrong_project': snapshot['captures'][0]['metadata']['project_id'] = str(uuid.uuid4())
    elif condition == 'wrong_receipt': snapshot['captures'][0]['receipt']['session_id'] = str(uuid.uuid4())
    elif condition == 'changed_old_hash': snapshot['captures'][0]['receipt']['received_utc'] = 'changed immutable receipt'
    elif condition == 'missing_id': ids.append(str(uuid.uuid4()))
    elif condition == 'duplicate_id': ids.append(ids[0])
    class SnapshotStore:
        root = store.root
        def export_session(self, _, capture_ids=None): return deepcopy(snapshot)
    with pytest.raises(ReconstructionError):
        update_session_spatial_map(SnapshotStore(), session['id'], ids)
    assert state_path(room).read_bytes() == before
    assert originals(room) == original_hashes


def test_atomic_failure_leaves_current_state_and_originals_unchanged(room, monkeypatch):
    ids = [capture(room, index) for index in range(3)]
    build(room, ids); before = state_path(room).read_bytes()
    new = capture(room, 3); original_hashes = originals(room)
    def refuse(*_): raise OSError('test replace failure')
    monkeypatch.setattr(mesh_spatial.os, 'replace', refuse)
    with pytest.raises(OSError, match='test replace failure'):
        build(room, ids+[new])
    assert state_path(room).read_bytes() == before
    assert not list(state_path(room).parent.glob('*.tmp'))
    assert originals(room) == original_hashes


def test_corrupted_saved_state_is_not_accepted_as_a_resume(room):
    ids = [capture(room, index) for index in range(3)]
    state = build(room, ids); state['summary']['accepted_hits'] += 1
    state_path(room).write_text(json.dumps(state)); corrupted = state_path(room).read_bytes()
    with pytest.raises(ReconstructionError, match='checksum'):
        build(room, ids)
    assert state_path(room).read_bytes() == corrupted


def test_export_contains_actual_hit_samples_and_no_faces_or_overwriting_originals(room):
    ids = [capture(room, index) for index in range(3)]
    tentative = capture(room, 3, point=(2.012, .232, .232))
    state = build(room, ids+[tentative]); before = originals(room)
    path = room[0].root/'observed.ply'
    receipt = export_session_spatial_map_ply(state, path)
    assert receipt['vertex_count'] == 2 and receipt['face_count'] == 0
    assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    header, rows = path.read_text().split('end_header\n')
    assert 'element vertex 2' in header and 'element face' not in header
    assert {tuple(map(float, line.split()[:3])) for line in rows.strip().splitlines()} == {
        (1.012, .232, .232), (2.012, .232, .232)}
    strict = export_session_spatial_map_ply(state, room[0].root/'supported.ply', include_tentative=False)
    assert strict['vertex_count'] == 1
    with pytest.raises(ReconstructionError, match='new local'):
        export_session_spatial_map_ply(state, path)
    assert originals(room) == before


def test_configuration_and_implicit_all_capture_updates_are_rejected(room):
    ids = [capture(room, 0)]
    build(room, ids); before = state_path(room).read_bytes()
    with pytest.raises(ReconstructionError, match='explicit allowed'):
        build(room, None)
    with pytest.raises(ReconstructionError, match='cannot change'):
        build(room, ids, voxel_size_m=.06)
    for value in (False, float('nan'), 'large'):
        with pytest.raises(ReconstructionError, match='Invalid bounded'):
            build(room, ids, voxel_size_m=value)
    assert state_path(room).read_bytes() == before


def test_symlink_lock_cannot_write_to_file_outside_derived_owner(room):
    ids = [capture(room, 0)]
    build(room, ids); before = state_path(room).read_bytes()
    lock = state_path(room).parent/'current.lock'
    lock.unlink(); outside = room[0].root/'unrelated-empty.txt'; outside.write_bytes(b'')
    try:
        lock.symlink_to(outside)
    except OSError:
        pytest.skip('Filesystem does not permit fixture symlinks')
    with pytest.raises(ReconstructionError, match='symlink'):
        build(room, ids)
    assert outside.read_bytes() == b''
    assert state_path(room).read_bytes() == before


def test_export_race_cannot_replace_an_existing_file(room, monkeypatch):
    state = build(room, [capture(room, 0)])
    destination = room[0].root/'concurrent-export.ply'
    def competitor(temporary, path):
        Path(path).write_bytes(b'other writer owns this file')
        raise FileExistsError('test competing publication')
    monkeypatch.setattr(mesh_spatial.os, 'link', competitor)
    with pytest.raises(ReconstructionError, match='destination already exists'):
        export_session_spatial_map_ply(state, destination)
    assert destination.read_bytes() == b'other writer owns this file'
    assert not list(destination.parent.glob('concurrent-export.ply.*.tmp'))


def test_close_angle_ambiguity_retains_an_actual_hit_and_is_not_claimed_resolved(room):
    theta = np.deg2rad(10.)
    tilted = (-float(np.cos(theta)), 0., float(np.sin(theta)))
    ids = [capture(room, 0), capture(room, 1, normal=tilted), capture(room, 2)]
    state = build(room, ids)
    assert len(surfaces(state)) == 1
    # Unit normalization can alter the final float bit; the representative must
    # still equal one actual observed normal rather than an averaged surface.
    assert any(np.allclose(surfaces(state)[0]['normal'], normal, atol=1e-14, rtol=0)
               for normal in [(-1., 0., 0.), tilted])
    assert 'close-angle surface ambiguity remains unresolved' in state['limitations']['normal_resolution']


def test_depth_display_uses_observed_positions_in_the_existing_loader_frame(room):
    from bridge.preview import read_depth_preview
    ids = [capture(room, index, point=(1.012, .232, 2.232)) for index in range(3)]
    state = build(room, ids); before = originals(room)
    manifest, path = read_depth_preview(room[0], room[1]['id'], room[2]['id'])
    payload = path.read_bytes(); header, body = payload.split(b'end_header\n', 1)
    names = [line.split()[-1].decode() for line in header.splitlines() if line.startswith(b'property ')]
    values = np.frombuffer(body, dtype=[(name, '<f4') for name in names])
    # Existing RUB loader performs exactly one Z flip into Unity's anchored frame.
    np.testing.assert_allclose([values['x'][0], values['y'][0], -values['z'][0]],
                               [1.012, .232, 2.232], atol=1e-6)
    assert manifest['representation'] == 'depth'
    assert manifest['accepted_views'] == manifest['total_views'] == 3
    assert manifest['supported_samples'] == 1
    assert manifest['artifact']['vertex_count'] == 1
    assert manifest['state_revision'] == state['state_sha256']
    assert manifest['accuracy_status'] == 'provisional_observed_samples'
    assert 'element face' not in header.decode()
    np.testing.assert_array_equal(manifest['alignment']['unity_from_splat'], np.eye(4).ravel())
    assert read_depth_preview(room[0], room[1]['id'], room[2]['id'])[1].read_bytes() == payload
    assert originals(room) == before
    assert not (room[0].root/'projects'/room[1]['id']/'reconstructions').exists()


def test_depth_display_routes_require_auth_and_cannot_start_reconstruction(room, monkeypatch):
    from fastapi.testclient import TestClient
    from bridge.app import create_app
    from bridge.config import Config
    def no_build(*_, **__): raise AssertionError('Depth display must not start Gaussian training')
    monkeypatch.setattr('bridge.app.request_build', no_build)
    client = TestClient(create_app(Config(room[0].root, 'fixture-token-'*4)))
    route = f"/api/v1/sessions/{room[2]['id']}/preview?representation=depth"
    assert client.get(route).status_code == 401
    auth = {'Authorization': 'Bearer '+'fixture-token-'*4}
    assert client.get(route, headers=auth).json()['status'] == 'not_ready'
    build(room, [capture(room, 0)])
    response = client.get(route, headers=auth)
    assert response.status_code == 200 and response.json()['status'] == 'ready'
    artifact = response.json()['artifact']
    assert client.get(artifact['download_path']).status_code == 401
    downloaded = client.get(artifact['download_path'], headers=auth)
    assert downloaded.status_code == 200
    assert hashlib.sha256(downloaded.content).hexdigest() == artifact['sha256']
    wrong = artifact['download_path'].replace(room[1]['id'], str(uuid.uuid4()))
    assert client.get(wrong, headers=auth).status_code == 404
    stale = artifact['download_path'].rsplit('/', 1)[0]+'/'+('0'*64)
    assert client.get(stale, headers=auth).status_code == 404


@pytest.mark.parametrize('condition', ['wrong_origin', 'corrupt_artifact', 'changed_image'])
def test_invalid_depth_display_cannot_publish_or_change_originals(room, condition):
    from bridge.preview import read_depth_preview, preview_manifest
    from bridge.store import StoreError
    ids = [capture(room, index) for index in range(3)]
    state = build(room, ids)
    _, artifact = read_depth_preview(room[0], room[1]['id'], room[2]['id'])
    original_paths = [Path(value) for value in originals(room)]
    if condition == 'wrong_origin':
        state['tracking_origin_id'] = str(uuid.uuid4())
        state['state_sha256'] = hashlib.sha256(mesh_spatial._spatial_json_bytes(
            {key: value for key, value in state.items() if key != 'state_sha256'})).hexdigest()
        state_path(room).write_text(json.dumps(state))
    elif condition == 'corrupt_artifact': artifact.write_bytes(b'fixture deliberately corrupt display copy')
    else:
        original = Path(room[0].export_session(room[2]['id'])['captures'][0]['image_path'])
        original.write_bytes(b'fixture changed original')
    protected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths}
    before = state_path(room).read_bytes()
    with pytest.raises(StoreError, match='Verified depth preview'):
        read_depth_preview(room[0], room[1]['id'], room[2]['id'])
    assert preview_manifest(room[0], room[2]['id'], 'depth')['status'] == 'not_ready'
    assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths} == protected
    assert state_path(room).read_bytes() == before


def test_compatible_dense_additions_extend_evidence_without_moving_supported_hit(room):
    ids = [capture(room, index, point=(1.012+index*.001, .232, .232)) for index in range(3)]
    first = build(room, ids); established = deepcopy(surfaces(first)[0]); before = originals(room)
    theta = np.deg2rad(19.)
    tilted = (-float(np.cos(theta)), 0., float(np.sin(theta)))
    new = [capture(room, index, point=(1.022, .232, .232), normal=tilted, repeats=150)
           for index in range(3, 7)]
    second = build(room, ids+new)
    current = next(value for value in surfaces(second) if value['id'] == established['id'])
    for field in ('id', 'point', 'normal', 'confidence', 'state', 'protection'):
        assert current[field] == established[field]
    assert set(current['capture_ids']) == set(ids+new)
    assert set(established['independent_capture_ids']).issubset(current['independent_capture_ids'])
    assert current['observation_count'] == 603
    for capture_id in ids:
        assert current['observation_membership'][capture_id] == established['observation_membership'][capture_id]
    assert second['update']['preserved_protected_surface_count'] == 1
    assert second['update']['changed_protected_geometry_count'] == 0
    assert second['summary']['protected_surface_count'] == 1
    assert all(originals(room)[key] == digest for key, digest in before.items())
    committed = state_path(room).read_bytes()
    assert build(room, ids+new) == second
    assert state_path(room).read_bytes() == committed


def test_supported_geometry_survives_cumulative_drift_and_competing_independent_surface(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids); established = deepcopy(surfaces(first)[0])
    # First within-tolerance feedback must not move the reference plane and let
    # the next measurement gradually drag it onto a different surface.
    near = capture(room, 3, point=(1.022, .232, .232), repeats=100)
    second = build(room, ids+[near])
    assert next(value for value in surfaces(second) if value['id'] == established['id'])['point'] == established['point']
    contrary = [capture(room, index, point=(1.032, .232, .232), repeats=100)
                for index in range(4, 7)]
    third = build(room, ids+[near]+contrary)
    retained = next(value for value in surfaces(third) if value['id'] == established['id'])
    for field in ('id', 'point', 'normal', 'confidence', 'state', 'protection'):
        assert retained[field] == established[field]
    assert set(retained['capture_ids']) == set(ids+[near])
    competitors = [value for value in surfaces(third) if value['id'] != established['id']]
    assert len(competitors) == 1 and competitors[0]['state'] == 'conflicting'
    assert set(competitors[0]['capture_ids']) == set(contrary)
    assert retained['conflicting_with'] == [competitors[0]['id']]
    assert third['summary']['protected_conflict_surface_count'] == 2
    assert third['update']['preserved_protected_surface_count'] == 1
    assert third['update']['newly_supported_surface_count'] == 1


def test_angled_and_opposite_additions_keep_original_supported_door_and_wall(room):
    wall = [capture(room, index) for index in range(3)]
    door = [capture(room, index+3, normal=(0., 0., 1.)) for index in range(3)]
    first = build(room, wall+door)
    established = {value['id']: deepcopy(value) for value in surfaces(first)}
    contrary = [capture(room, index+6, normal=(1., 0., 0.)) for index in range(3)]
    second = build(room, wall+door+contrary)
    assert len(surfaces(second)) == 3
    for old_id, old in established.items():
        current = next(value for value in surfaces(second) if value['id'] == old_id)
        for field in ('point', 'normal', 'confidence', 'state', 'protection'):
            assert current[field] == old[field]
    assert second['update']['preserved_protected_surface_count'] == 2


def test_no_hit_addition_keeps_entire_protected_cell_exact(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids)
    unknown = capture(room, 3, status='no_hit', repeats=100)
    second = build(room, ids+[unknown])
    assert second['cells'] == first['cells']
    assert second['update']['changed_voxel_count'] == 0
    assert second['update']['preserved_protected_surface_count'] == 1
    assert second['summary']['unknown_ray_count'] == 100


def test_explicit_exclusion_rebuild_can_replace_supported_geometry_from_retained_evidence(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids); established = surfaces(first)[0]['point']
    replacement = [capture(room, index, point=(1.022, .232, .232)) for index in range(3, 6)]
    incremental = build(room, ids+replacement)
    assert surfaces(incremental)[0]['point'] == established
    before = originals(room)
    rebuilt = build(room, replacement)
    assert surfaces(rebuilt)[0]['point'] == [1.022, .232, .232]
    assert rebuilt['update']['mode'] == 'rebuild_after_exclusion'
    assert set(rebuilt['update']['removed_capture_ids']) == set(ids)
    assert rebuilt['update']['preserved_protected_surface_count'] == 0
    assert originals(room) == before


def test_monotonic_revision_advances_on_commits_but_not_duplicate_retries(room):
    ids = [capture(room, 0)]
    first = build(room, ids)
    assert first['revision_number'] == 1
    before = state_path(room).read_bytes()
    assert build(room, ids)['revision_number'] == 1
    assert state_path(room).read_bytes() == before
    ids.append(capture(room, 1))
    assert build(room, ids)['revision_number'] == 2
    assert build(room, ids[:1])['revision_number'] == 3
    assert build(room, [])['revision_number'] == 4
    assert build(room, [])['revision_number'] == 4


def test_legacy_schema_one_retry_is_exact_and_next_delta_preserves_supported_geometry(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids); first.pop('revision_number')
    for value in surfaces(first):
        for field in ('observation_membership', 'geometry_protected', 'protection', 'ambiguous_protected_with'):
            value.pop(field)
    first['state_sha256'] = hashlib.sha256(mesh_spatial._spatial_json_bytes(
        {key: value for key, value in first.items() if key != 'state_sha256'})).hexdigest()
    state_path(room).write_bytes(mesh_spatial._spatial_json_bytes(first)); before = state_path(room).read_bytes()
    assert build(room, ids) == first and state_path(room).read_bytes() == before
    new = capture(room, 3, point=(1.022, .232, .232))
    upgraded = build(room, ids+[new])
    assert upgraded['schema_version'] == 1 and upgraded['revision_number'] == 1
    for field in ('id', 'point', 'normal', 'confidence'):
        assert surfaces(upgraded)[0][field] == surfaces(first)[0][field]
    assert surfaces(upgraded)[0]['geometry_protected'] is True
    assert surfaces(upgraded)[0]['protection']['established_capture_ids'] == sorted(ids)


def ambiguous_legacy_map(room, first_compatible):
    """Saved receipts from the two shipped schema-1 grouping policies.

    The middle observations fit both planes but are closer to the second. Build
    the historical receipts explicitly so replaying the current grouping rule
    alone cannot accidentally satisfy this regression.
    """
    normal2 = (-float(np.cos(np.deg2rad(2))), 0., float(np.sin(np.deg2rad(2))))
    normal4 = (-float(np.cos(np.deg2rad(4))), 0., float(np.sin(np.deg2rad(4))))
    near = [capture(room, index) for index in range(3)]
    far = [capture(room, index, point=(1.032, .232, .232), normal=normal2)
           for index in range(3, 6)]
    middle = [capture(room, index, point=(1.023, .232, .232), normal=normal4)
              for index in range(6, 8)]
    ids = near+far+middle
    saved = build(room, ids)
    assert len(saved['cells']) == 1
    key, cell = next(iter(saved['cells'].items()))
    old_surfaces = []
    for seed_ids, members in ((near, near+middle if first_compatible else near),
                              (far, far if first_compatible else far+middle)):
        seed_id = min(seed_ids)
        row = cell['samples'][seed_id][0]
        identifier = hashlib.sha256(mesh_spatial._spatial_json_bytes(
            [key, seed_id, 0, row[:6]])).hexdigest()[:24]
        old_surfaces.append({'id': identifier, 'point': row[:3], 'normal': row[3:6],
            'confidence': row[9], 'capture_ids': sorted(members),
            'independent_capture_ids': sorted(members), 'independent_view_count': len(members),
            'observation_count': len(members), 'support_level': 'supported',
            'state': 'conflicting', 'conflicting_with': []})
    for surface in old_surfaces:
        surface['conflicting_with'] = [value['id'] for value in old_surfaces
                                       if value['id'] != surface['id']]
    cell['surfaces'] = sorted(old_surfaces, key=lambda value: value['id'])
    saved['state_sha256'] = hashlib.sha256(mesh_spatial._spatial_json_bytes(
        {key: value for key, value in saved.items() if key != 'state_sha256'})).hexdigest()
    state_path(room).write_bytes(mesh_spatial._spatial_json_bytes(saved))
    return saved, ids


@pytest.mark.parametrize('first_compatible', [True, False], ids=['original-first-match', 'later-nearest-match'])
def test_ambiguous_legacy_groups_upgrade_using_exact_historical_receipts(room, first_compatible):
    saved, ids = ambiguous_legacy_map(room, first_compatible)
    before = state_path(room).read_bytes()
    assert build(room, ids) == saved and state_path(room).read_bytes() == before
    new = capture(room, 8, point=(1.045, .232, .232))
    source_hashes = originals(room)
    updated = build(room, ids+[new])
    assert updated['revision_number'] == saved['revision_number']+1
    current = {value['id']: value for value in surfaces(updated)}
    for old in surfaces(saved):
        restored = current[old['id']]
        for field in ('id', 'point', 'normal', 'confidence', 'state'):
            assert restored[field] == old[field]
        assert set(old['capture_ids']).issubset(restored['capture_ids'])
        assert restored['protection']['established_capture_ids'] == old['capture_ids']
        assert restored['protection']['established_independent_capture_ids'] == old['independent_capture_ids']
        assert restored['protection']['established_observation_count'] == old['observation_count']
        assert restored['protection']['legacy_replay'] == (
            'first_compatible_v1' if first_compatible else 'nearest_compatible_v1')
    assert updated['update']['preserved_protected_surface_count'] == 2
    assert updated['update']['changed_protected_geometry_count'] == 0
    assert originals(room) == source_hashes
    committed = state_path(room).read_bytes()
    assert build(room, ids+[new]) == updated and state_path(room).read_bytes() == committed


def test_ambiguous_legacy_receipt_mismatch_is_not_replaced_by_a_different_group(room):
    saved, ids = ambiguous_legacy_map(room, True)
    surfaces(saved)[0]['observation_count'] += 1
    saved['state_sha256'] = hashlib.sha256(mesh_spatial._spatial_json_bytes(
        {key: value for key, value in saved.items() if key != 'state_sha256'})).hexdigest()
    state_path(room).write_bytes(mesh_spatial._spatial_json_bytes(saved))
    new = capture(room, 8, point=(1.045, .232, .232))
    before = state_path(room).read_bytes(); source_hashes = originals(room)
    with pytest.raises(ReconstructionError, match='saved surface receipts'):
        build(room, ids+[new])
    assert state_path(room).read_bytes() == before and originals(room) == source_hashes


def test_bad_unselected_capture_does_not_poison_existing_map_or_selected_update(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids)
    bad = capture(room, 3)
    bad_capture = room[0].export_session(room[2]['id'], capture_ids=[bad])['captures'][0]
    bad_image = Path(bad_capture['image_path']); bad_image.write_bytes(b'deliberately corrupt excluded fixture')
    assert mesh_spatial.read_session_spatial_map(room[0], room[2]['id']) == first
    assert build(room, ids) == first
    good = capture(room, 4, point=(2.012, .232, .232))
    second = build(room, ids+[good])
    assert second['revision_number'] == 2 and set(second['input_ledger']) == set(ids+[good])
    assert bad_image.read_bytes() == b'deliberately corrupt excluded fixture'


def test_ambiguous_vote_cannot_reinforce_either_of_two_protected_planes(room):
    first_ids = [capture(room, index) for index in range(3)]
    second_ids = [capture(room, index, point=(1.034, .232, .232)) for index in range(3, 6)]
    first = build(room, first_ids+second_ids)
    established = {value['id']: deepcopy(value) for value in surfaces(first)}
    ambiguous = capture(room, 6, point=(1.023, .232, .232), repeats=200)
    updated = build(room, first_ids+second_ids+[ambiguous])
    for old_id, old in established.items():
        current = next(value for value in surfaces(updated) if value['id'] == old_id)
        for field in ('point', 'normal', 'confidence', 'capture_ids', 'independent_capture_ids', 'protection'):
            assert current[field] == old[field]
    alternative = next(value for value in surfaces(updated) if value['id'] not in established)
    assert alternative['capture_ids'] == [ambiguous]
    assert set(alternative['ambiguous_protected_with']) == set(established)
    assert alternative['state'] == 'tentative_conflict'
    assert alternative['point'] == [1.023, .232, .232]
    assert updated['update']['changed_protected_geometry_count'] == 0


def test_rejected_precommit_policy_leaves_exact_last_good_bytes_and_revision(room):
    ids = [capture(room, index) for index in range(3)]
    first = build(room, ids); before = state_path(room).read_bytes()
    new = capture(room, 3, point=(2.012, .232, .232)); hashes = originals(room)
    calls = []
    def reject():
        calls.append('checked')
        raise ReconstructionError('Capture selection changed during fusion')
    with pytest.raises(ReconstructionError, match='selection changed'):
        build(room, ids+[new], before_commit=reject)
    assert calls == ['checked']
    assert state_path(room).read_bytes() == before
    assert mesh_spatial.read_session_spatial_map(room[0], room[2]['id'])['revision_number'] == first['revision_number']
    assert originals(room) == hashes
    assert not list(state_path(room).parent.glob('*.tmp'))
    assert build(room, ids+[new])['revision_number'] == first['revision_number']+1


def test_precommit_runs_only_on_actual_publication_not_duplicate_retry(room):
    ids = [capture(room, 0)]; calls = []
    first = build(room, ids, before_commit=lambda: calls.append('first write'))
    assert calls == ['first write']
    def no_retry_callback(): raise AssertionError('Unchanged retry must not publish or run callback')
    before = state_path(room).read_bytes()
    assert build(room, ids, before_commit=no_retry_callback) == first
    assert state_path(room).read_bytes() == before
    ids.append(capture(room, 1))
    second = build(room, ids, before_commit=lambda: calls.append('second write'))
    assert calls == ['first write', 'second write'] and second['revision_number'] == 2
