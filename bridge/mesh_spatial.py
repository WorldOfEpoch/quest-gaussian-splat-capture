"""Conservative, optional surface assistance from verified headset raycast evidence.

These rays are near a camera capture, not RGB pixel depth or measured LiDAR.
They can support local geometry after a sufficiently accurate frame fit.  A
no-hit is unknown, and this module never deletes faces or generates surfaces.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from .models import SpatialEvidence
from .reconstruct import ReconstructionError


def verified_observations(store, session_id, capture_ids=None, min_confidence=.8,
                          minimum_baseline_m=.05, *, require_surface_review=True):
    """Read immutable sidecars and reject wrong frames before expensive work.

    The same-clock trigger bounds the *raycast attempt* age.  The SDK has not
    provided the underlying depth sensor frame age, and the RGB and depth clocks
    have not been synchronized.  This limitation is retained in the receipt.
    """
    session_id = str(uuid.UUID(session_id))
    if (not np.isfinite(min_confidence) or not .8 <= min_confidence <= 1. or
        not np.isfinite(minimum_baseline_m) or not .03 <= minimum_baseline_m <= .25):
        raise ReconstructionError('Invalid spatial support confidence or view baseline')
    selected = None if capture_ids is None else {str(uuid.UUID(value)) for value in capture_ids}
    snapshot = (store.export_session(session_id, capture_ids=sorted(selected))
                if selected is not None else store.export_session(session_id))
    session = snapshot['session']
    expected = {field: str(session[field]) for field in ('project_id', 'tracking_origin_id')}
    expected['session_id'] = session_id
    if selected is not None and selected-{str(value['receipt']['capture_id']) for value in snapshot['captures']}:
        raise ReconstructionError('Selected spatial evidence is missing from its session')
    points, normals, origins, directions, confidence, codes = [], [], [], [], [], []
    records, capture_origins, captures_with_hits = [], [], []
    rejected = {'no_sidecar': 0, 'unavailable': 0, 'not_a_hit': 0,
                'normal_or_confidence_unavailable': 0, 'low_confidence': 0}
    for capture in snapshot['captures']:
        metadata, receipt = capture['metadata'], capture['receipt']
        capture_id = str(receipt['capture_id'])
        if selected is not None and capture_id not in selected:
            continue
        for field, value in expected.items():
            if str(metadata.get(field)) != value:
                raise ReconstructionError('Spatial capture belongs to a different project/session/tracking origin')
        if metadata.get('tracking_valid') is not True:
            raise ReconstructionError('Spatial support requires valid capture tracking')
        if not capture.get('spatial_path'):
            rejected['no_sidecar'] += 1
            continue
        path = Path(capture['spatial_path'])
        project_root = (store.root/'projects'/expected['project_id']).resolve()
        if path.is_symlink() or not path.resolve().is_relative_to(project_root) or not path.is_file():
            raise ReconstructionError('Spatial evidence is outside its owning project')
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if (len(payload) > 2*1024*1024 or digest != metadata.get('spatial_evidence_sha256') or
            digest != receipt.get('spatial_evidence_sha256') or
            len(payload) != receipt.get('spatial_evidence_size_bytes')):
            raise ReconstructionError('Spatial evidence hash/size differs from the capture receipt')
        try:
            evidence = SpatialEvidence.model_validate_json(payload)
        except ValueError as error:
            raise ReconstructionError('Invalid or stale spatial raycast attempt: '+str(error)) from error
        for field, value in {**expected, 'capture_id': capture_id}.items():
            if str(getattr(evidence, field)) != value:
                raise ReconstructionError('Spatial evidence belongs to a different project/session/tracking origin/capture')
        if (evidence.rgb_timestamp_raw != metadata.get('image_timestamp_raw') or
            evidence.rgb_timestamp_kind != metadata.get('image_timestamp_kind')):
            raise ReconstructionError('Spatial evidence is associated with a different camera frame')
        if evidence.source != 'meta_environment_raycast_openxr' or metadata.get('capture_source') != 'quest_camera':
            raise ReconstructionError('Real geometry assistance requires actual Quest environment raycasts')
        record = {'capture_id': capture_id, 'sha256': digest, 'size_bytes': len(payload),
                  'availability': evidence.availability, 'ray_count': len(evidence.rays),
                  'trigger_realtime_s': evidence.trigger_realtime_s,
                  'sampling_lag_s': evidence.sample_started_realtime_s-evidence.trigger_realtime_s,
                  'sampling_duration_s': evidence.sample_completed_realtime_s-evidence.sample_started_realtime_s}
        records.append(record)
        if evidence.availability != 'ready':
            rejected['unavailable'] += 1
            continue
        current_origins = []
        code = len(captures_with_hits)
        for ray in evidence.rays:
            if ray.status != 'hit':
                rejected['not_a_hit'] += 1
                continue
            if ray.normal is None or ray.normal_confidence is None:
                rejected['normal_or_confidence_unavailable'] += 1
                continue
            if ray.normal_confidence < min_confidence:
                rejected['low_confidence'] += 1
                continue
            points.append(ray.hit); normals.append(ray.normal); origins.append(ray.origin)
            directions.append(ray.direction); confidence.append(ray.normal_confidence); codes.append(code)
            current_origins.append(ray.origin)
        record['accepted_hits'] = len(current_origins)
        if current_origins:
            captures_with_hits.append(capture_id)
            capture_origins.append(np.median(current_origins, axis=0))
    independent = []
    for code, origin in enumerate(capture_origins):
        if all(np.linalg.norm(origin-capture_origins[previous]) >= minimum_baseline_m for previous in independent):
            independent.append(code)
    if require_surface_review and (len(independent) < 3 or len(points) < 12):
        raise ReconstructionError('Need new depth-assisted captures: at least three distinct Quest views with '
                                  'confident hit normals and a 5 cm translation baseline; image-only photos remain usable')
    codes = np.asarray(codes, dtype=np.int32)
    keep = np.isin(codes, independent) if require_surface_review else np.ones(len(codes), dtype=bool)
    if require_surface_review and keep.sum() < 12:
        raise ReconstructionError('Insufficient independent high-confidence ray hits for a surface review')
    arrays = {name: np.asarray(value, dtype=float).reshape((-1, 3))[keep] for name, value in (
        ('points', points), ('normals', normals), ('origins', origins), ('directions', directions))}
    arrays['confidence'] = np.asarray(confidence, dtype=float)[keep]
    arrays['normals'] /= np.linalg.norm(arrays['normals'], axis=1, keepdims=True)
    arrays['directions'] /= np.linalg.norm(arrays['directions'], axis=1, keepdims=True)
    arrays['capture_index'] = codes[keep]
    arrays['report'] = {'schema_version': 1, **expected, 'source': 'meta_environment_raycast_openxr',
        'coordinate_frame': 'quest_session', 'coordinate_convention': 'unity_lh_xright_yup_zforward',
        'position_units': 'meters', 'accepted_hits': int(keep.sum()),
        'captures_with_confident_hits': len(captures_with_hits),
        'independent_view_count': len(independent),
        'independent_capture_ids': [captures_with_hits[i] for i in independent],
        'capture_ids_by_index': captures_with_hits,
        'minimum_view_baseline_m': minimum_baseline_m, 'minimum_normal_confidence': min_confidence,
        'evidence_records': records, 'rejected': rejected,
        'rgb_depth_association': 'near_capture_unaligned; not RGB pixel depth',
        'sensor_frame_freshness': 'unknown; trigger bounds the attempt age only',
        'no_hit_interpretation': 'unknown; never used as proof of empty space or openings',
        'coverage': 'limited observed samples; not a complete house or measured architectural survey'}
    return arrays


def _spatial_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _atomic_spatial_write(destination, payload, *, replace=True):
    """Publish a whole derived file; a failed replace leaves the previous state."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name+'.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise ReconstructionError('Session surface-map export destination already exists') from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _spatial_map_identity(snapshot, session_id):
    try:
        session = snapshot['session']
        identity = {key: str(uuid.UUID(str(session[key]))) for key in
                    ('project_id', 'tracking_origin_id')}
        identity['session_id'] = str(uuid.UUID(session_id))
        if str(session['id']) != identity['session_id']:
            raise ValueError('Session ID differs')
        return identity
    except (KeyError, ValueError, TypeError) as error:
        raise ReconstructionError('Session surface map requires a single identified tracking origin') from error


def _spatial_map_ledger(store, capture, identity):
    metadata, receipt = capture['metadata'], capture['receipt']
    capture_id = str(uuid.UUID(receipt['capture_id']))
    if (str(metadata.get('capture_id')) != capture_id or
        any(str(metadata.get(key)) != value or
            (key != 'tracking_origin_id' and str(receipt.get(key)) != value)
            for key, value in identity.items()) or
        metadata.get('tracking_valid') is not True or metadata.get('capture_source') != 'quest_camera'):
        raise ReconstructionError('Session surface map capture has a mixed identity, origin, or invalid tracking')
    project_root = (store.root/'projects'/identity['project_id']).resolve()
    image = Path(capture['image_path'])
    digest = receipt.get('image_sha256')
    if (not isinstance(digest, str) or len(digest) != 64 or
        digest != metadata.get('image_sha256') or image.is_symlink() or
        not image.resolve().is_relative_to(project_root) or not image.is_file() or
        hashlib.sha256(image.read_bytes()).hexdigest() != digest):
        raise ReconstructionError('Session surface map original image differs from its receipt')
    spatial_sha = receipt.get('spatial_evidence_sha256')
    spatial_size = receipt.get('spatial_evidence_size_bytes')
    if capture.get('spatial_path'):
        path = Path(capture['spatial_path'])
        if (path.is_symlink() or not path.resolve().is_relative_to(project_root) or not path.is_file()):
            raise ReconstructionError('Session surface map evidence is outside its owning project')
        payload = path.read_bytes()
        if (len(payload) > 2*1024*1024 or len(payload) != spatial_size or
            hashlib.sha256(payload).hexdigest() != spatial_sha or
            spatial_sha != metadata.get('spatial_evidence_sha256')):
            raise ReconstructionError('Session surface map spatial evidence differs from its receipt')
    elif any(value is not None for value in
             (spatial_sha, spatial_size, metadata.get('spatial_evidence_sha256'))):
        raise ReconstructionError('Session surface map is missing its declared spatial evidence')
    return {'capture_id': capture_id, 'image_sha256': digest,
            'metadata_sha256': hashlib.sha256(_spatial_json_bytes(metadata)).hexdigest(),
            'receipt_sha256': hashlib.sha256(_spatial_json_bytes(receipt)).hexdigest(),
            'spatial_evidence_sha256': spatial_sha, 'spatial_evidence_size_bytes': spatial_size}


def _independent_surface_views(votes, baseline):
    """Capture density never creates extra views; deterministic translated votes."""
    keys = sorted(votes)
    best = []
    for start in range(len(keys)):
        selected = []
        for key in keys[start:]+keys[:start]:
            if all(np.linalg.norm(votes[key]['origin']-votes[other]['origin']) >= baseline
                   for other in selected):
                selected.append(key)
        selected.sort()
        if len(selected) > len(best) or (len(selected) == len(best) and selected < best):
            best = selected
    return best


def _legacy_spatial_cell_membership(key, samples, parameters, *, first_compatible=True):
    """Recover schema-1 first-compatible grouping and its original medoid ties.

    This replays only the old evidence. It supplies membership receipts, never a
    replacement point/normal; callers require exact saved geometry/provenance.
    """
    ordered = [(capture_id, index, np.asarray(row, dtype=float))
               for capture_id in sorted(samples) for index, row in enumerate(samples[capture_id])]
    ordered.sort(key=lambda item: (*item[2][3:6], *item[2][:3], item[0], item[1]))
    angle = np.cos(np.deg2rad(parameters['normal_agreement_deg']))
    tolerance = parameters['surface_consistency_m']
    groups = []
    for capture_id, index, row in ordered:
        compatible = [candidate for candidate in groups if
                      np.dot(row[3:6], candidate['normal']) >= angle and
                      abs(np.dot(row[:3]-candidate['point'], candidate['normal'])) <= tolerance]
        group = (compatible[0] if first_compatible else min(compatible, key=lambda candidate:
                 (abs(np.dot(row[:3]-candidate['point'], candidate['normal'])), candidate['id']))) if compatible else None
        if group is None:
            group = {'id': hashlib.sha256(_spatial_json_bytes(
                [key, capture_id, index, row[:6].tolist()])).hexdigest()[:24],
                'point': row[:3], 'normal': row[3:6], 'rows': {}}
            groups.append(group)
        group['rows'].setdefault(capture_id, []).append((index, row))
    recovered = {}
    for group in groups:
        votes = {}
        for capture_id, rows in group['rows'].items():
            data = np.asarray([row for _, row in rows])
            votes[capture_id] = {'point': np.median(data[:, :3], axis=0),
                                 'origin': np.median(data[:, 6:9], axis=0)}
        independent = _independent_surface_views(votes, parameters['minimum_view_baseline_m'])
        center = np.median([votes[value]['point'] for value in independent], axis=0)
        # Original code broke medoid ties by the position within each sorted
        # group, not by the ray's source-array index. Retain that exact choice.
        candidates = [(capture_id, grouped_index, source_index, row)
                      for capture_id, rows in group['rows'].items() if capture_id in independent
                      for grouped_index, (source_index, row) in enumerate(rows)]
        capture_id, _, index, row = min(candidates, key=lambda value:
            (float(np.linalg.norm(value[3][:3]-center)), value[0], value[1]))
        count = sum(map(len, group['rows'].values()))
        recovered[group['id']] = {'id': group['id'], 'point': row[:3].tolist(),
            'normal': row[3:6].tolist(), 'confidence': float(row[9]),
            'capture_ids': sorted(votes), 'independent_capture_ids': independent,
            'independent_view_count': len(independent), 'observation_count': count,
            'observation_membership': {value: [source_index for source_index, _ in rows]
                                      for value, rows in group['rows'].items()},
            'protection': {'representative_capture_id': capture_id, 'representative_sample_index': index,
                'established_capture_ids': sorted(votes),
                'established_independent_capture_ids': list(independent),
                'established_observation_count': count,
                'legacy_replay': 'first_compatible_v1' if first_compatible else 'nearest_compatible_v1'}}
    return recovered


def _spatial_cell_surfaces(key, samples, parameters, previous_surfaces=()):
    """Freeze established hits; later contrary evidence stays a separate hypothesis."""
    ordered = [(capture_id, index, np.asarray(row, dtype=float))
               for capture_id in sorted(samples) for index, row in enumerate(samples[capture_id])]
    ordered.sort(key=lambda item: (*item[2][3:6], *item[2][:3], item[0], item[1]))
    angle = np.cos(np.deg2rad(parameters['normal_agreement_deg']))
    tolerance = parameters['surface_consistency_m']
    groups, fixed_membership = [], {}
    protected = [surface for surface in previous_surfaces if
                 surface.get('geometry_protected') is True or surface.get('support_level') == 'supported']
    # Schema-1 maps predating membership receipts are upgraded only on an actual
    # delta. Reproduce their original grouping to recover exact old provenance.
    legacy = {}
    if any('observation_membership' not in surface for surface in protected):
        old_ids = {value for surface in previous_surfaces for value in surface['capture_ids']}
        old_samples = {value: rows for value, rows in samples.items() if value in old_ids}
        # Both legacy implementations were shipped before membership receipts.
        # Choose one only when every old surface receipt matches exactly. Prefer
        # the original first-match replay when both reproduce the saved cell.
        fields = ('id', 'point', 'normal', 'confidence', 'capture_ids',
                  'independent_capture_ids', 'independent_view_count', 'observation_count')
        for first_compatible in (True, False):
            replay = _legacy_spatial_cell_membership(key, old_samples, parameters,
                                                      first_compatible=first_compatible)
            if all(surface['id'] in replay and all(replay[surface['id']][field] == surface[field]
                   for field in fields) for surface in previous_surfaces):
                legacy = replay
                break
        if not legacy:
            raise ReconstructionError('Cannot recover protected legacy grouping without changing saved surface receipts')
    for surface in protected:
        recovered = legacy.get(surface['id'], surface)
        if any(recovered[field] != surface[field] for field in
               ('id', 'point', 'normal', 'confidence', 'capture_ids',
                'independent_capture_ids', 'independent_view_count', 'observation_count')):
            raise ReconstructionError('Cannot recover protected surface provenance; explicit derived rebuild is required')
        membership = recovered.get('observation_membership')
        if not membership:
            raise ReconstructionError('Protected session surface lacks observed membership')
        seed = {**surface, 'observation_membership': membership,
                'geometry_protected': True, 'protection': surface.get('protection', recovered.get('protection'))}
        group = {'id': surface['id'], 'point': np.asarray(surface['point']),
                 'normal': np.asarray(surface['normal']), 'rows': {}, 'protected': seed,
                 'ambiguous_protected_with': set()}
        for capture_id, indices in membership.items():
            for index in indices:
                member = (capture_id, index)
                if (capture_id not in samples or type(index) is not int or
                    not 0 <= index < len(samples[capture_id]) or member in fixed_membership):
                    raise ReconstructionError('Protected surface has invalid or overlapping observed membership')
                fixed_membership[member] = group
        groups.append(group)
    for capture_id, index, row in ordered:
        point, normal = row[:3], row[3:6]
        compatible = [group for group in groups if
                      np.dot(normal, group['normal']) >= angle and
                      abs(np.dot(point-group['point'], group['normal'])) <= tolerance]
        compatible_protected = [group for group in compatible if group.get('protected')]
        ambiguous = set()
        if (capture_id, index) in fixed_membership:
            group = fixed_membership[capture_id, index]
        elif len(compatible_protected) == 1:
            group = compatible_protected[0]
        else:
            # A vote that fits two protected surfaces cannot choose which old
            # surface to reinforce. Keep it in an explicitly ambiguous group.
            ambiguous = {group['id'] for group in compatible_protected}
            candidates = [group for group in compatible if not group.get('protected')]
            group = (min(candidates, key=lambda value:
                        (abs(np.dot(point-value['point'], value['normal'])), value['id']))
                     if candidates else None)
        if group is None:
            identifier = hashlib.sha256(_spatial_json_bytes([key, capture_id, index,
                                                            row[:6].tolist()])).hexdigest()[:24]
            group = {'id': identifier, 'point': point, 'normal': normal, 'rows': {},
                     'ambiguous_protected_with': set()}
            groups.append(group)
        group['ambiguous_protected_with'].update(ambiguous)
        group['rows'].setdefault(capture_id, []).append((index, row))
    surfaces = []
    for group in groups:
        votes = {}
        for capture_id, rows in group['rows'].items():
            rows = np.asarray([row for _, row in rows])
            votes[capture_id] = {'point': np.median(rows[:, :3], axis=0),
                                 'origin': np.median(rows[:, 6:9], axis=0)}
        independent = _independent_surface_views(votes, parameters['minimum_view_baseline_m'])
        prior = group.get('protected')
        if prior:
            established = list(prior['independent_capture_ids'])
            if any(value not in votes for value in established):
                raise ReconstructionError('Protected surface lost established independent-view provenance')
            independent = established
            for capture_id in sorted(votes):
                if capture_id not in independent and all(
                        np.linalg.norm(votes[capture_id]['origin']-votes[value]['origin']) >=
                        parameters['minimum_view_baseline_m'] for value in independent):
                    independent.append(capture_id)
        # Use one vote per independent view, then select an actual positive hit
        # near their median. This exports no imagined point between surfaces.
        center = np.median([votes[value]['point'] for value in independent], axis=0)
        supported_rows = [(capture_id, index, row) for capture_id, rows in group['rows'].items()
                          if capture_id in independent for index, row in rows]
        representative_id, representative_index, representative = min(supported_rows, key=lambda item:
                                   (float(np.linalg.norm(item[2][:3]-center)), item[0], item[1]))
        count = len(independent)
        level = ('supported' if count >= 3 else 'tentative_multi_view' if count == 2 else
                 'tentative_repeated_origin' if len(votes) > 1 else 'tentative_single_view')
        surface = {'id': group['id'], 'point': representative[:3].tolist(),
            'normal': representative[3:6].tolist(), 'confidence': float(representative[9]),
            'capture_ids': sorted(votes), 'independent_capture_ids': independent,
            'independent_view_count': count, 'observation_count': sum(map(len, group['rows'].values())),
            'support_level': level, 'state': level, 'conflicting_with': [],
            'observation_membership': {value: [index for index, _ in rows]
                                      for value, rows in group['rows'].items()},
            'ambiguous_protected_with': sorted(group['ambiguous_protected_with']),
            'geometry_protected': count >= 3}
        if prior:
            for field in ('point', 'normal', 'confidence', 'state'):
                surface[field] = prior[field]
            surface['protection'] = prior['protection']
        elif count >= 3:
            surface['protection'] = {'representative_capture_id': representative_id,
                'representative_sample_index': representative_index,
                'established_capture_ids': sorted(votes),
                'established_independent_capture_ids': list(independent),
                'established_observation_count': surface['observation_count']}
        if surface['ambiguous_protected_with']:
            surface['state'] = 'conflicting' if count >= 2 else 'tentative_conflict'
        surfaces.append(surface)
    surfaces.sort(key=lambda value: value['id'])
    prior_by_id = {group['id']: group.get('protected') for group in groups}
    for i, first in enumerate(surfaces):
        for second in surfaces[i+1:]:
            dot = np.dot(first['normal'], second['normal'])
            offset = abs(np.dot(np.subtract(first['point'], second['point']), first['normal']))
            if dot >= angle and offset > tolerance:
                first['conflicting_with'].append(second['id'])
                second['conflicting_with'].append(first['id'])
                for surface, competitor in ((first, second), (second, first)):
                    prior = prior_by_id[surface['id']]
                    if prior:
                        # Preserve the accepted support state; expose contrary
                        # hypotheses separately instead of replacing good hits.
                        surface['state'] = prior['state']
                    elif surface['independent_view_count'] >= 2 and competitor['independent_view_count'] >= 2:
                        surface['state'] = 'conflicting'
                    elif surface['independent_view_count'] < 2:
                        surface['state'] = 'tentative_conflict'
    for surface in surfaces:
        if surface.get('geometry_protected') and 'established_state' not in surface['protection']:
            surface['protection']['established_state'] = surface['state']
    return surfaces


def _check_session_spatial_map(state):
    if not isinstance(state, dict) or state.get('schema_version') != 1 or \
            state.get('kind') != 'provisional_anchor_frame_depth':
        raise ReconstructionError('Invalid session surface-map state')
    if (type(state.get('revision_number', 0)) is not int or state.get('revision_number', 0) < 0):
        raise ReconstructionError('Invalid session surface-map revision number')
    core = {key: value for key, value in state.items() if key != 'state_sha256'}
    if hashlib.sha256(_spatial_json_bytes(core)).hexdigest() != state.get('state_sha256'):
        raise ReconstructionError('Session surface-map state checksum differs')
    return state


def read_session_spatial_map(store, session_id):
    """Read the current derived map with exact live-session and source receipts."""
    session_id = str(uuid.UUID(session_id))
    identity = _spatial_map_identity({'session': store.session(session_id)}, session_id)
    root = store.root/'projects'/identity['project_id']/'sessions'/session_id/'spatial-map'
    path = root/'current.json'
    if (path.is_symlink() or not path.resolve().is_relative_to(store.root.resolve()) or
        not path.is_file() or path.stat().st_size > 256*1024*1024):
        raise ReconstructionError('Verified session depth map is unavailable')
    try:
        state = _check_session_spatial_map(json.loads(path.read_bytes()))
        if any(state.get(key) != value for key, value in identity.items()):
            raise ReconstructionError('Depth map belongs to another project/session/origin')
        snapshot = store.export_session(session_id, capture_ids=sorted(state['input_ledger']))
        captures = {str(value['receipt']['capture_id']): value for value in snapshot['captures']}
        if set(state['input_ledger'])-captures.keys():
            raise ReconstructionError('Depth map refers to a capture outside this session')
        for capture_id, value in state['input_ledger'].items():
            if value['receipt'] != _spatial_map_ledger(store, captures[capture_id], identity):
                raise ReconstructionError('Depth map original evidence or receipt changed')
        return state
    except (ValueError, TypeError, KeyError) as error:
        raise ReconstructionError('Invalid session depth-map state') from error


def update_session_spatial_map(store, session_id, allowed_capture_ids, *, voxel_size_m=.05,
                               normal_agreement_deg=20., surface_consistency_m=.012,
                               min_confidence=.8, minimum_baseline_m=.05, before_commit=None):
    """Fuse a selected anchored-session ray subset into one provisional map.

    This is positive normal-bearing surface sampling, not RGB-D TSDF, verified
    photo geometry, an architectural survey, or a complete mesh. New captures
    affect only their voxels. Exclusion rebuilds a derived map from retained
    originals; it does not invert training. The caller owns the photo/anchor
    localization decision and must supply an explicit allowed-ID subset.
    An optional caller policy check runs immediately before an actual publication
    and may reject stale selection/control state; unchanged retries never call it.
    """
    session_id = str(uuid.UUID(session_id))
    if allowed_capture_ids is None:
        raise ReconstructionError('Session surface map requires explicit allowed capture IDs')
    if before_commit is not None and not callable(before_commit):
        raise ReconstructionError('Session surface-map pre-commit check must be callable')
    selected = [str(uuid.UUID(str(value))) for value in allowed_capture_ids]
    if len(set(selected)) != len(selected):
        raise ReconstructionError('Session surface map contains duplicate capture identities')
    parameters = {'voxel_size_m': voxel_size_m, 'normal_agreement_deg': normal_agreement_deg,
                  'surface_consistency_m': surface_consistency_m,
                  'minimum_normal_confidence': min_confidence,
                  'minimum_view_baseline_m': minimum_baseline_m}
    if type(voxel_size_m) not in (int, float) or not np.isfinite(voxel_size_m):
        raise ReconstructionError('Invalid bounded session surface-map parameters')
    bounds = {'voxel_size_m': (.015, .10), 'normal_agreement_deg': (5., 30.),
              'surface_consistency_m': (.003, min(.025, voxel_size_m)),
              'minimum_normal_confidence': (.8, 1.), 'minimum_view_baseline_m': (.03, .25)}
    if any(type(value) not in (int, float) or not np.isfinite(value) or
           not bounds[key][0] <= value <= bounds[key][1] for key, value in parameters.items()):
        raise ReconstructionError('Invalid bounded session surface-map parameters')
    snapshot = store.export_session(session_id, capture_ids=selected)
    identity = _spatial_map_identity(snapshot, session_id)
    captures = {}
    for capture in snapshot['captures']:
        capture_id = str(capture['receipt']['capture_id'])
        if capture_id in captures:
            raise ReconstructionError('Session surface map received duplicate capture receipts')
        captures[capture_id] = capture
    if set(selected)-captures.keys():
        raise ReconstructionError('Session surface map selected capture is absent from this session')
    selected.sort()
    ledger = {value: _spatial_map_ledger(store, captures[value], identity) for value in selected}
    root = store.root.resolve()/'projects'/identity['project_id']/'sessions'/session_id/'spatial-map'
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ReconstructionError('Session surface-map path escapes its store')
    root.mkdir(parents=True, exist_ok=True)
    destination = root/'current.json'
    if destination.is_symlink() or (root/'current.lock').is_symlink():
        raise ReconstructionError('Session surface-map state cannot be a symlink')
    # A native advisory lock is released by the OS on process exit. Atomic
    # replacement alone would not prevent two writers losing each other's delta.
    with (root/'current.lock').open('a+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            if lock.read(1) == b'':
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ReconstructionError('Session surface map is being updated by another writer') from error
        else:
            import fcntl
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ReconstructionError('Session surface map is being updated by another writer') from error
        previous = None
        if destination.exists():
            if destination.stat().st_size > 256*1024*1024:
                raise ReconstructionError('Session surface-map state exceeds its bounded size')
            try:
                previous = _check_session_spatial_map(json.loads(destination.read_bytes()))
            except (ValueError, TypeError, KeyError) as error:
                raise ReconstructionError('Invalid session surface-map state') from error
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ReconstructionError('Session surface map belongs to a different project/session/origin')
            if previous['parameters'] != parameters:
                raise ReconstructionError('Session surface-map parameters cannot change during accumulation')
            for capture_id in set(previous['input_ledger']) & set(ledger):
                if previous['input_ledger'][capture_id]['receipt'] != ledger[capture_id]:
                    raise ReconstructionError('Previously fused capture receipt or original content changed')
        removed = set(previous['input_ledger'])-set(selected) if previous else set()
        new_ids = (set(selected)-set(previous['input_ledger'])) if previous and not removed else set(selected)
        if previous and not removed and not new_ids:
            return previous
        cells = previous['cells'] if previous and not removed else {}
        inputs = previous['input_ledger'] if previous and not removed else {}
        protected_before = {(key, surface['id']): surface for key, cell in cells.items()
                            for surface in cell['surfaces'] if surface.get('geometry_protected') is True or
                            surface.get('support_level') == 'supported'}
        delta_snapshot = {**snapshot, 'captures': [captures[value] for value in sorted(new_ids)]}
        class SnapshotStore:
            root = store.root
            def export_session(self, value, capture_ids=None):
                assert value == session_id
                return delta_snapshot
        observations = verified_observations(SnapshotStore(), session_id, sorted(new_ids),
            min_confidence, minimum_baseline_m, require_surface_review=False)
        report = observations['report']
        records = {record['capture_id']: record for record in report['evidence_records']}
        for capture_id in sorted(new_ids):
            record = records.get(capture_id, {})
            inputs[capture_id] = {'receipt': ledger[capture_id],
                'availability': record.get('availability', 'no_sidecar'),
                'accepted_hits': record.get('accepted_hits', 0),
                'unknown_rays': record.get('ray_count', 0)-record.get('accepted_hits', 0)}
        if sum(value['accepted_hits'] for value in inputs.values()) > 1_000_000:
            raise ReconstructionError('Session surface map exceeds one million bounded observed samples')
        changed_cells = set()
        for point, normal, origin, confidence, code in zip(observations['points'], observations['normals'],
                observations['origins'], observations['confidence'], observations['capture_index']):
            capture_id = report['capture_ids_by_index'][code]
            key = ','.join(map(str, np.floor(point/voxel_size_m).astype(np.int64)))
            cell = cells.setdefault(key, {'samples': {}, 'surfaces': []})
            cell['samples'].setdefault(capture_id, []).append(
                [*point.tolist(), *normal.tolist(), *origin.tolist(), float(confidence)])
            changed_cells.add(key)
        for key in sorted(changed_cells):
            cells[key]['surfaces'] = _spatial_cell_surfaces(key, cells[key]['samples'], parameters,
                                                         cells[key]['surfaces'])
        protected_after = {(key, surface['id']): surface for key, cell in cells.items()
                           for surface in cell['surfaces'] if surface.get('geometry_protected') is True or
                           surface.get('support_level') == 'supported'}
        for identity_key, old in protected_before.items():
            current = protected_after.get(identity_key)
            if (current is None or any(current[field] != old[field] for field in ('point', 'normal', 'confidence')) or
                not set(old['capture_ids']).issubset(current['capture_ids']) or
                not set(old['independent_capture_ids']).issubset(current['independent_capture_ids']) or
                (old.get('protection') is not None and current.get('protection') != old['protection'])):
                raise ReconstructionError('Incremental depth map changed protected geometry or established provenance')
        surface_states = {}
        for cell in cells.values():
            for surface in cell['surfaces']:
                surface_states[surface['state']] = surface_states.get(surface['state'], 0)+1
        state = {'schema_version': 1, 'kind': 'provisional_anchor_frame_depth', **identity,
            'revision_number': previous.get('revision_number', 0)+1 if previous else 1,
            'coordinate_frame': 'quest_session', 'coordinate_convention': 'unity_lh_xright_yup_zforward',
            'position_units': 'meters', 'source': 'meta_environment_raycast_openxr',
            'parameters': parameters, 'input_ledger': inputs, 'cells': cells,
            'input_revision': hashlib.sha256(_spatial_json_bytes(ledger)).hexdigest(),
            'update': {'mode': 'rebuild_after_exclusion' if removed else 'merge_delta',
                       'processed_capture_ids': sorted(new_ids), 'removed_capture_ids': sorted(removed),
                       'changed_voxel_count': len(changed_cells),
                       'preserved_protected_surface_count': len(protected_before),
                       'changed_protected_geometry_count': 0,
                       'newly_supported_surface_count': len(set(protected_after)-set(protected_before))},
            'summary': {'capture_count': len(inputs), 'accepted_hits': sum(v['accepted_hits'] for v in inputs.values()),
                'unknown_capture_count': sum(v['accepted_hits'] == 0 for v in inputs.values()),
                'unknown_ray_count': sum(v['unknown_rays'] for v in inputs.values()),
                'voxel_count': len(cells), 'surface_sample_count': sum(surface_states.values()),
                'protected_surface_count': len(protected_after),
                'protected_conflict_surface_count': sum(bool(surface['conflicting_with'] or
                    surface.get('ambiguous_protected_with')) for surface in protected_after.values()),
                'surface_states': surface_states},
            'limitations': {'photo_geometry': 'not verified; caller must explicitly allow localized capture IDs',
                'anchor_continuity': 'session origin identity only; independent relocalization verification is still required',
                'rgb_depth_association': report['rgb_depth_association'],
                'sensor_frame_freshness': report['sensor_frame_freshness'],
                'normal_resolution': f"Signed normal hypotheses separate above {normal_agreement_deg:g} degrees; "
                    'near-hinge and close-angle surface ambiguity remains unresolved',
                'unknown_space': 'no-hit, low-confidence and missing normals remain unknown; no free-space carving',
                'coverage': 'observed positive surface samples only; no gap filling, faces, or architectural planes',
                'geometry_protection': 'once supported, observed representative and established evidence remain fixed '
                    'during additions; contrary hypotheses are separate; explicit exclusion rebuild can change geometry',
                'exclusion': 'rebuild selected derived evidence; not inversion of an optimizer'}}
        state['state_sha256'] = hashlib.sha256(_spatial_json_bytes(state)).hexdigest()
        payload = _spatial_json_bytes(state)
        if len(payload) > 256*1024*1024:
            raise ReconstructionError('Session surface-map state exceeds its bounded size')
        if before_commit is not None:
            before_commit()
        _atomic_spatial_write(destination, payload)
        return state


def export_session_spatial_map_ply(state, destination, *, include_tentative=True):
    """Export actual positive-hit representatives; normals/color, no faces."""
    _check_session_spatial_map(state)
    destination = Path(destination)
    if destination.suffix.lower() != '.ply' or destination.is_symlink() or destination.exists():
        raise ReconstructionError('Session surface-map export needs a new local .ply path')
    if not destination.parent.is_dir():
        raise ReconstructionError('Session surface-map export directory is missing')
    surfaces = [surface for key in sorted(state['cells']) for surface in state['cells'][key]['surfaces']
                if include_tentative or surface['state'] == 'supported']
    palette = {'supported': (60, 200, 100), 'conflicting': (235, 70, 70),
               'tentative_conflict': (235, 140, 65), 'tentative_multi_view': (235, 210, 65),
               'tentative_single_view': (130, 145, 165), 'tentative_repeated_origin': (160, 130, 165)}
    lines = ['ply', 'format ascii 1.0', 'comment provisional anchored-session positive depth samples',
             'comment Unity left-handed x-right y-up z-forward meters; not verified photo geometry',
             f'element vertex {len(surfaces)}', 'property double x', 'property double y', 'property double z',
             'property float nx', 'property float ny', 'property float nz',
             'property uchar red', 'property uchar green', 'property uchar blue',
             'property uint independent_views', 'property float confidence', 'end_header']
    for surface in surfaces:
        row = [*surface['point'], *surface['normal'], *palette[surface['state']],
               surface['independent_view_count'], surface['confidence']]
        lines.append(' '.join(str(value) for value in row))
    payload = ('\n'.join(lines)+'\n').encode('ascii')
    _atomic_spatial_write(destination, payload, replace=False)
    return {'path': str(destination.resolve()), 'sha256': hashlib.sha256(payload).hexdigest(),
            'size_bytes': len(payload), 'vertex_count': len(surfaces), 'face_count': 0,
            'kind': state['kind'], 'source_state_sha256': state['state_sha256'],
            'input_revision': state['input_revision'], 'includes_tentative': include_tentative}


def source_to_session_transform(job, registration, normalization):
    """Compose the existing reviewed pose fit with explicit handedness conversion."""
    if (registration.get('status') != 'pose_fit' or registration.get('headset_preview_allowed') is not True or
        registration.get('matched_views', 0) < 6 or registration.get('baseline_m', 0) < .3):
        raise ReconstructionError('Geometry assistance needs a verified six-view Quest/photo pose fit')
    limits = {'position_rmse_m': .04, 'max_position_error_m': .10, 'max_rotation_error_deg': 5.}
    for key, limit in limits.items():
        value = registration.get(key)
        if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= limit:
            raise ReconstructionError('Quest/photo calibration is too uncertain for geometry assistance: '+key)
    transform = np.asarray(registration.get('unity_from_splat'), dtype=float)
    if transform.size != 16:
        raise ReconstructionError('Missing verified Quest/photo frame transform')
    transform = transform.reshape(4, 4)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    if (not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8) or
        not .001 <= scale <= 1000 or
        not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3)*scale**2, atol=1e-6)):
        raise ReconstructionError('Quest/photo registration is not a verified positive similarity')
    original = np.asarray(normalization.get('transform'), dtype=float)
    raw_to_training = np.eye(4)
    if original.shape == (4, 4):
        raw_to_training = original.copy()
    elif original.shape == (3, 4):
        raw_to_training[:3] = original
    else:
        raise ReconstructionError('Source photo normalization has an invalid frame transform')
    training_scale = normalization.get('scale')
    if (not isinstance(training_scale, (float, int)) or not np.isfinite(training_scale) or
        training_scale <= 0 or not np.isfinite(raw_to_training).all() or
        not np.allclose(raw_to_training[3], [0, 0, 0, 1], atol=1e-8)):
        raise ReconstructionError('Source photo normalization is invalid')
    raw_to_training[:3] *= training_scale
    # register_job fits UnitySplats' imported training coordinates: global Z is
    # flipped once, deliberately, before the positive session similarity.
    imported_from_training = np.diag([1., 1., -1., 1.])
    raw_to_session = transform@imported_from_training@raw_to_training
    singular = np.linalg.svd(raw_to_session[:3, :3], compute_uv=False)
    if not np.allclose(singular, singular[0], rtol=1e-5) or singular[-1] <= 0:
        raise ReconstructionError('Source-to-session transform contains a non-uniform warp')
    return raw_to_session, {'status': 'verified_photo_pose_fit_for_spatial_support',
        'session_id': job['session_id'], 'source_job_id': job['job_id'],
        'matched_views': registration['matched_views'], 'baseline_m': registration['baseline_m'],
        **{key: float(registration[key]) for key in limits}, 'limits': limits,
        'raw_colmap_to_session': raw_to_session.tolist(),
        'handedness': 'explicit training global-Z flip followed by verified positive Quest-session similarity',
        'physical_accuracy': 'pose residual check; independent physical measurement is still unperformed'}


def assist_observed_surface(vertices, faces, observations, max_displacement_m=.02,
                            support_radius_m=.06):
    """Bounded local correction; preserve topology and mismatched door/object faces."""
    from scipy.spatial import cKDTree
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or f.ndim != 2 or f.shape[1] != 3 or
        not np.issubdtype(f.dtype, np.integer) or not len(f) or f.min() < 0 or f.max() >= len(v) or
        not np.isfinite(v).all() or not .005 <= max_displacement_m <= .03 or
        not .03 <= support_radius_m <= .10):
        raise ReconstructionError('Invalid bounded spatial surface correction inputs')
    f = f.astype(np.int64, copy=False)
    p, n, origins, directions = (np.asarray(observations[key], dtype=float) for key in
                                  ('points', 'normals', 'origins', 'directions'))
    confidence = np.asarray(observations['confidence'], dtype=float)
    original_codes = np.asarray(observations['capture_index'])
    if not np.issubdtype(original_codes.dtype, np.integer):
        raise ReconstructionError('Spatial capture membership must use integer identities')
    codes = original_codes.astype(np.int64, copy=False)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < 12 or
        any(value.shape != p.shape for value in (n, origins, directions)) or
        confidence.shape != (len(p),) or codes.shape != (len(p),) or
        not all(np.isfinite(value).all() for value in (p, n, origins, directions, confidence)) or
        np.any(confidence < .8) or np.any(confidence > 1.) or np.any(codes < 0) or len(np.unique(codes)) < 3 or
        not np.allclose(np.linalg.norm(n, axis=1), 1., atol=.01) or
        not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=.01)):
        raise ReconstructionError('Spatial observations lack independent confident unit-normal support')
    view_origins = [np.median(origins[codes == code], axis=0) for code in np.unique(codes)]
    distinct_origins = []
    for origin in view_origins:
        if all(np.linalg.norm(origin-previous) >= .05 for previous in distinct_origins):
            distinct_origins.append(origin)
    if len(distinct_origins) < 3:
        raise ReconstructionError('Spatial observations lack three independently translated view origins')
    tree = cKDTree(p)
    nearest_distance, nearest = tree.query(v, workers=2)
    cross = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    areas = np.linalg.norm(cross, axis=1)
    face_normals = cross/np.maximum(areas[:, None], 1e-12)
    near_normal = n[nearest]
    angle = np.cos(np.deg2rad(20.))
    face_compatible = (np.abs(np.sum(face_normals[:, None]*near_normal[f], axis=2)) >= angle).all(axis=1)
    face_compatible &= (areas > 1e-12)
    eligible = nearest_distance <= support_radius_m
    eligible[np.unique(f[~face_compatible])] = False
    out = v.copy()
    correction = np.zeros_like(v)
    support_count = np.zeros(len(v), dtype=np.int16)
    candidate_indices = np.flatnonzero(eligible)
    rejected = {'insufficient_views': 0, 'inconsistent_surface': 0, 'outside_displacement': 0,
                'conflicting_positive_rays': 0}
    for start in range(0, len(candidate_indices), 8192):
        index = candidate_indices[start:start+8192]
        distance, neighbor = tree.query(v[index], k=24, distance_upper_bound=support_radius_m, workers=2)
        valid = (neighbor < len(p)) & np.isfinite(distance)
        safe = np.minimum(neighbor, len(p)-1)
        reference = near_normal[index]
        dot = np.sum(n[safe]*reference[:, None], axis=2)
        valid &= np.abs(dot) >= np.cos(np.deg2rad(15.))
        current_codes = np.where(valid, codes[safe], -1)
        repeated = current_codes[:, :, None] == current_codes[:, None, :]
        repeated &= valid[:, None, :]
        per_capture_count = np.maximum(repeated.sum(axis=2), 1)
        weights = np.where(valid, confidence[safe]/per_capture_count, 0.)
        ordered_codes = np.sort(current_codes, axis=1)
        distinct = np.sum((ordered_codes[:, 1:] != ordered_codes[:, :-1]) &
                          (ordered_codes[:, 1:] >= 0), axis=1)+(ordered_codes[:, 0] >= 0)
        displacement = np.sum((p[safe]-v[index, None])*reference[:, None], axis=2)
        order = np.argsort(np.where(valid, displacement, np.inf), axis=1)
        ordered_weight = np.take_along_axis(weights, order, axis=1)
        cumulative = np.cumsum(ordered_weight, axis=1)
        middle = np.argmax(cumulative >= weights.sum(axis=1, keepdims=True)*.5, axis=1)
        median = np.take_along_axis(displacement, np.take_along_axis(order, middle[:, None], axis=1), axis=1)[:, 0]
        deviation = np.abs(displacement-median[:, None])
        consistent_weight = np.sum(np.where(deviation <= .012, weights, 0.), axis=1)
        consistent = consistent_weight >= weights.sum(axis=1)*.8
        in_budget = np.abs(median) <= max_displacement_m
        proposed = v[index]+median[:, None]*reference
        # Only confident positive hits can contradict a proposed surface.
        # A no-hit never supplies free-space votes and never removes geometry.
        delta = proposed[:, None]-origins[safe]
        along = np.sum(delta*directions[safe], axis=2)
        hit_range = np.sum((p[safe]-origins[safe])*directions[safe], axis=2)
        sideways = np.linalg.norm(delta-along[:, :, None]*directions[safe], axis=2)
        conflict = valid & (along > 0) & (along < hit_range-.025) & (sideways < .012)
        conflict_codes = np.sort(np.where(conflict, codes[safe], -1), axis=1)
        conflicts = np.sum((conflict_codes[:, 1:] != conflict_codes[:, :-1]) &
                           (conflict_codes[:, 1:] >= 0), axis=1)+(conflict_codes[:, 0] >= 0)
        accept = (distinct >= 3) & consistent & in_budget & (conflicts < 2)
        rejected['insufficient_views'] += int((distinct < 3).sum())
        rejected['inconsistent_surface'] += int(((distinct >= 3) & ~consistent).sum())
        rejected['outside_displacement'] += int(((distinct >= 3) & consistent & ~in_budget).sum())
        rejected['conflicting_positive_rays'] += int(((distinct >= 3) & consistent & in_budget & (conflicts >= 2)).sum())
        support_count[index[accept]] = distinct[accept]
        correction[index[accept]] = median[accept, None]*reference[accept]
    eligible &= support_count >= 3
    # Every vertex of an unsupported or angularly conflicting face is pinned,
    # including shared door hinges.  A collar protects the transition geometry.
    protected = ~eligible
    protected[np.unique(f[~eligible[f].all(axis=1)])] = True
    edges = np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    unique_edges, incidence = np.unique(edges, axis=0, return_counts=True)
    protected[np.unique(unique_edges[incidence != 2])] = True
    protected[np.unique(f[protected[f].any(axis=1)])] = True
    out[~protected] += correction[~protected]
    reverted = np.zeros(len(v), dtype=bool)
    for _ in range(24):
        new_cross = np.cross(out[f[:, 1]]-out[f[:, 0]], out[f[:, 2]]-out[f[:, 0]])
        bad = (areas > 1e-12) & ((np.linalg.norm(new_cross, axis=1) < areas*.1) |
                                 (np.sum(cross*new_cross, axis=1) <= 0))
        if not bad.any():
            break
        undo = np.unique(f[bad]); undo = undo[~protected[undo]]
        if len(undo) == 0:
            raise ReconstructionError('Spatial correction could not preserve source triangle orientation')
        out[undo] = v[undo]; protected[undo] = True; reverted[undo] = True
    else:
        raise ReconstructionError('Spatial correction topology guard did not converge')
    movement = np.linalg.norm(out-v, axis=1)
    if not np.array_equal(out[protected], v[protected]) or movement.max(initial=0) > max_displacement_m*(1+1e-8):
        raise ReconstructionError('Spatial correction changed protected geometry or exceeded its budget')
    moved = movement > 1e-8
    before = np.abs(np.sum((p[nearest[moved]]-v[moved])*near_normal[moved], axis=1))
    after = np.abs(np.sum((p[nearest[moved]]-out[moved])*near_normal[moved], axis=1))
    report = {'method': 'local_multiview_environment_raycast_surface_assistance',
        'input_vertices': len(v), 'output_vertices': len(out), 'input_faces': len(f), 'output_faces': len(f),
        'faces_added': 0, 'faces_removed': 0, 'moved_vertices': int(moved.sum()),
        'protected_vertices': int(protected.sum()), 'reverted_vertices': int(reverted.sum()),
        'max_displacement_m': float(movement.max(initial=0)), 'rejected': rejected,
        'limits': {'max_displacement_m': max_displacement_m, 'support_radius_m': support_radius_m,
            'minimum_distinct_views': 3, 'normal_agreement_degrees': 20,
            'local_surface_consistency_m': .012, 'positive_ray_free_space_margin_m': .025},
        'unsupported_geometry': 'retained exactly; no deletion or completion claims',
        'architectural_prior': 'none; no rectangular room, plane box, or hidden surface inferred',
        'coverage': 'local supported surfaces only; full house completeness is not established'}
    if moved.any():
        report['nearest_supported_normal_rms_before_m'] = float(np.sqrt(np.mean(before**2)))
        report['nearest_supported_normal_rms_after_m'] = float(np.sqrt(np.mean(after**2)))
    return out, np.where(protected, 0, support_count), report
