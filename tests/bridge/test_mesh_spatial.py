"""Optional depth support cannot blend frames, invent geometry, or erase doors."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid

import numpy as np
import pytest

from bridge import mesh
from bridge.mesh_spatial import verified_observations, source_to_session_transform
from bridge.reconstruct import ReconstructionError


class EvidenceStore:
    def __init__(self, root, snapshot):
        self.root, self.snapshot = root, snapshot

    def export_session(self, session_id):
        assert self.snapshot['session']['id'] == session_id
        return deepcopy(self.snapshot)


def evidence_store(tmp_path, confidence=.95, mutate=None, origins=(0., .1, .2)):
    project_id, session_id, origin_id = [str(uuid.uuid4()) for _ in range(3)]
    session = {'id': session_id, 'project_id': project_id, 'tracking_origin_id': origin_id}
    captures = []
    for index, position in enumerate(origins):
        capture_id = str(uuid.uuid4())
        rays = []
        for y, z in ((.2, .3), (.4, .3), (.2, .5), (.4, .5)):
            origin = np.array([position, 0., 0.])
            hit = np.array([1., y, z]); direction = hit-origin
            direction /= np.linalg.norm(direction)
            rays.append({'origin': origin.tolist(), 'direction': direction.tolist(), 'status': 'hit',
                         'hit': hit.tolist(), 'normal': [-1., 0., 0.], 'normal_confidence': confidence})
        sidecar = {'schema_version': 1, 'capture_id': capture_id, 'project_id': project_id,
            'session_id': session_id, 'tracking_origin_id': origin_id,
            'source': 'meta_environment_raycast_openxr', 'association': 'near_capture_unaligned',
            'coordinate_frame': 'quest_session', 'coordinate_convention': 'unity_lh_xright_yup_zforward',
            'position_units': 'meters', 'trigger_realtime_s': 100.+index,
            'sample_started_realtime_s': 100.05+index, 'sample_completed_realtime_s': 100.1+index,
            'rgb_timestamp_raw': '2026-10-12T15:00:00.0000000Z', 'rgb_timestamp_kind': 'Utc',
            'availability': 'ready', 'rays': rays}
        if mutate:
            mutate(sidecar, index)
        payload = json.dumps(sidecar).encode()
        root = tmp_path/'projects'/project_id/'sessions'/session_id
        root.mkdir(parents=True, exist_ok=True)
        path = root/(capture_id+'.spatial.json'); path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        metadata = {'capture_id': capture_id, 'project_id': project_id, 'session_id': session_id,
            'tracking_origin_id': origin_id, 'tracking_valid': True, 'capture_source': 'quest_camera',
            'image_timestamp_raw': '2026-10-12T15:00:00.0000000Z', 'image_timestamp_kind': 'Utc',
            'spatial_evidence_sha256': digest}
        captures.append({'metadata': metadata, 'spatial_path': str(path), 'receipt': {
            'capture_id': capture_id, 'spatial_evidence_sha256': digest,
            'spatial_evidence_size_bytes': len(payload)}})
    return EvidenceStore(tmp_path, {'session': session, 'captures': captures})


def test_preflight_requires_real_confident_translated_views_and_records_clock_limits(tmp_path):
    store = evidence_store(tmp_path)
    result = verified_observations(store, store.snapshot['session']['id'])
    assert result['points'].shape == (12, 3)
    assert result['report']['independent_view_count'] == 3
    assert 'not RGB pixel depth' in result['report']['rgb_depth_association']
    assert result['report']['sensor_frame_freshness'].startswith('unknown')
    json.dumps(result['report'], allow_nan=False)


@pytest.mark.parametrize('condition', ['wrong_origin', 'wrong_session', 'stale_attempt',
                                      'wrong_rgb_frame', 'synthetic', 'missing_sidecar'])
def test_preflight_does_not_accept_untrusted_or_stale_evidence(tmp_path, condition):
    def mutate(sidecar, index):
        if index != 0:
            return
        if condition == 'wrong_origin': sidecar['tracking_origin_id'] = str(uuid.uuid4())
        elif condition == 'wrong_session': sidecar['session_id'] = str(uuid.uuid4())
        elif condition == 'stale_attempt': sidecar['sample_started_realtime_s'] = 101.; sidecar['sample_completed_realtime_s'] = 101.05
        elif condition == 'wrong_rgb_frame': sidecar['rgb_timestamp_raw'] = '2026-10-12T15:00:01.0000000Z'
        elif condition == 'synthetic': sidecar['source'] = 'synthetic_test'
    store = evidence_store(tmp_path, mutate=mutate)
    if condition == 'missing_sidecar':
        for capture in store.snapshot['captures']:
            capture.pop('spatial_path')
    with pytest.raises(ReconstructionError):
        verified_observations(store, store.snapshot['session']['id'])


def test_low_confidence_unknown_normals_and_rotations_in_place_do_not_supply_geometry(tmp_path):
    low = evidence_store(tmp_path/'low', confidence=.7)
    stationary = evidence_store(tmp_path/'stationary', origins=(0., .01, .015))
    unknown = evidence_store(tmp_path/'unknown', mutate=lambda evidence, _: [
        ray.update(normal=None, normal_confidence=None) for ray in evidence['rays']])
    for store in (low, stationary, unknown):
        with pytest.raises(ReconstructionError, match='three distinct Quest views'):
            verified_observations(store, store.snapshot['session']['id'])


def test_sidecar_hash_is_rechecked_and_no_hit_is_unknown(tmp_path):
    store = evidence_store(tmp_path)
    Path(store.snapshot['captures'][0]['spatial_path']).write_bytes(b'changed')
    with pytest.raises(ReconstructionError, match='hash/size'):
        verified_observations(store, store.snapshot['session']['id'])
    def no_hits(evidence, _):
        evidence['rays'] = [{key: ray[key] for key in ('origin', 'direction')} | {'status': 'no_hit'}
                            for ray in evidence['rays']]
    store = evidence_store(tmp_path/'unknown', mutate=no_hits)
    with pytest.raises(ReconstructionError, match='three distinct Quest views'):
        verified_observations(store, store.snapshot['session']['id'])


def pose_fit():
    return {'status': 'pose_fit', 'headset_preview_allowed': True, 'matched_views': 10,
            'baseline_m': 1.2, 'position_rmse_m': .01, 'max_position_error_m': .025,
            'max_rotation_error_deg': 1., 'unity_from_splat': np.eye(4).ravel().tolist()}


def test_frame_conversion_is_explicit_and_loose_overlay_fit_cannot_move_geometry():
    job = {'job_id': str(uuid.uuid4()), 'session_id': str(uuid.uuid4())}
    normalization = {'transform': np.eye(4)[:3].tolist(), 'scale': 2.}
    matrix, review = source_to_session_transform(job, pose_fit(), normalization)
    assert np.allclose(matrix[:3, :3], np.diag([2., 2., -2.]))
    assert 'explicit training global-Z flip' in review['handedness']
    for field, value in (('position_rmse_m', .05), ('max_position_error_m', .11),
                         ('max_rotation_error_deg', 6.), ('matched_views', 3), ('baseline_m', .1)):
        fit = pose_fit(); fit[field] = value
        with pytest.raises(ReconstructionError):
            source_to_session_transform(job, fit, normalization)
    fit = pose_fit(); fit['unity_from_splat'] = np.diag([1., 1., -1., 1.]).ravel().tolist()
    with pytest.raises(ReconstructionError, match='positive similarity'):
        source_to_session_transform(job, fit, normalization)


def grid(y, z, x):
    yy, zz = np.meshgrid(y, z, indexing='ij')
    vertices = np.column_stack((x(yy.ravel(), zz.ravel()), yy.ravel(), zz.ravel()))
    rows, columns = np.meshgrid(np.arange(len(y)-1), np.arange(len(z)-1), indexing='ij')
    first = (rows*len(z)+columns).ravel()
    faces = np.concatenate((np.column_stack((first, first+len(z), first+1)),
                            np.column_stack((first+1, first+len(z), first+len(z)+1))))
    return vertices, faces


def supported_wall_observations(extra_outlier=False):
    y = np.linspace(.2, 1.2, 31); z = np.linspace(.2, 1.2, 31)
    points, _ = grid(y, z, lambda y, z: np.ones(len(y)))
    p = np.concatenate([points]*3)
    origins = np.concatenate([np.tile([.1*i, 0., 0.], (len(points), 1)) for i in range(3)])
    codes = np.repeat(np.arange(3), len(points))
    if extra_outlier:
        bad = points[:40].copy(); bad[:, 0] = 1.15
        p = np.concatenate((p, bad)); origins = np.concatenate((origins, np.tile([.4, 0., 0.], (len(bad), 1))))
        codes = np.concatenate((codes, np.full(len(bad), 3)))
    directions = p-origins; directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return {'points': p, 'normals': np.tile([-1., 0., 0.], (len(p), 1)), 'origins': origins,
            'directions': directions, 'confidence': np.full(len(p), .95), 'capture_index': codes}


def test_local_depth_improves_supported_surface_preserving_door_and_unsupported_object():
    pytest.importorskip('scipy')
    from bridge.mesh_spatial import assist_observed_surface
    vertices, faces = grid(np.linspace(.2, 1.2, 21), np.linspace(.2, 1.2, 21),
                          lambda y, z: 1.+.008*np.sin(y*7)*np.sin(z*5))
    door = np.array([[1., .7, .3], [1.45, .7, .3], [1.45, .7, 1.]])
    floating = np.array([[2., .2, .3], [2., .4, .3], [2., .2, .5]])
    combined = np.concatenate((vertices, door, floating))
    triangles = np.concatenate((faces, np.array([[0, 1, 2], [3, 4, 5]])+len(vertices)))
    out, supporting_views, report = assist_observed_surface(combined, triangles, supported_wall_observations())
    moved = supporting_views > 0
    assert report['moved_vertices'] > 100
    assert np.mean(np.abs(out[moved, 0]-1.)) < np.mean(np.abs(combined[moved, 0]-1.))*.1
    assert np.array_equal(out[len(vertices):], combined[len(vertices):])
    assert report['faces_added'] == report['faces_removed'] == 0
    assert report['max_displacement_m'] <= .02
    assert 'no rectangular room' in report['architectural_prior']
    assert np.array_equal(out[~moved], combined[~moved])


def test_far_depth_outlier_and_single_view_spam_cannot_replace_a_surface():
    pytest.importorskip('scipy')
    from bridge.mesh_spatial import assist_observed_surface
    vertices, faces = grid(np.linspace(.2, 1.2, 21), np.linspace(.2, 1.2, 21),
                          lambda y, z: np.full(len(y), 1.008))
    out, supporting_views, report = assist_observed_surface(vertices, faces, supported_wall_observations(True))
    assert report['moved_vertices'] > 100
    assert np.allclose(out[supporting_views > 0, 0], 1.)
    observations = supported_wall_observations()
    observations['capture_index'][:] = 0
    with pytest.raises(ReconstructionError, match='independent'):
        assist_observed_surface(vertices, faces, observations)
    observations = supported_wall_observations()
    observations['origins'][:] = [0., 0., 0.]
    with pytest.raises(ReconstructionError, match='translated view origins'):
        assist_observed_surface(vertices, faces, observations)


@pytest.mark.skipif(not importlib.util.find_spec('pymeshlab'), reason='Optional mesh cleanup dependency')
def test_depth_supported_mesh_cannot_be_smoothed_or_have_object_components_pruned(monkeypatch):
    monkeypatch.setattr(mesh, 'read_mesh', lambda *args: ({'spatial_support': {'supported_vertices': 10}}, None))
    identity = str(uuid.uuid4())
    for settings in ({'smooth_steps': 1, 'min_component_faces': 1},
                     {'smooth_steps': 0, 'min_component_faces': 25}):
        with pytest.raises(ReconstructionError, match='zero smoothing'):
            mesh.finish_mesh(None, identity, identity, identity, **settings)
