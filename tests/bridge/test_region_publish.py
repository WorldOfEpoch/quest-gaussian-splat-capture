"""Exercise actual reversible publication with tiny synthetic artifacts only."""
import hashlib
import json
import struct
import uuid

import pytest

from bridge.online import OnlineError
from bridge.region_edit import request_region_action
from bridge.store import Store
from bridge.preview import read_online_preview


def splat_bytes(count, x=0.):
    props = ('x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3', 'f_dc_0', 'f_dc_1', 'f_dc_2')
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {count}\n' +
              ''.join(f'property float {name}\n' for name in props) + 'end_header\n')
    return header.encode()+struct.pack('<14f', x, *([0.]*13))*count


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def ready_area(tmp_path):
    store = Store(tmp_path)
    project = store.create_project('Synthetic area apply')['id']
    session = store.create_session(project, str(uuid.uuid4()))['id']
    region = str(uuid.uuid4())
    online = tmp_path / "projects" / project / "online" / session
    folder = tmp_path / "projects" / project / "area-edits" / region
    original = folder / "original"
    online.mkdir(parents=True)
    original.mkdir(parents=True)
    originals = {"current.ply": splat_bytes(4), "display.ply": splat_bytes(3, .1),
                 "current.ckpt": b"original optimizer state"}
    job_id = str(uuid.uuid4())
    job_folder = tmp_path/'projects'/project/'reconstructions'/job_id
    job_folder.mkdir(parents=True)
    seed = job_folder/'splat.ply'; seed.write_bytes(splat_bytes(4))
    seed_sha = digest(seed.read_bytes())
    artifact = {'path': str(seed), 'sha256': seed_sha, 'size_bytes': seed.stat().st_size, 'vertex_count': 4}
    (job_folder/'job.json').write_text(json.dumps({'job_id': job_id, 'project_id': project,
        'session_id': session, 'status': 'ready_for_blender', 'artifact': artifact}))
    source_state = {"status": "paused_quality_review", "preserved_marker": "synthetic only",
                    'project_id': project, 'session_id': session, 'seed_sha256': seed_sha,
                    'seed_ids': [], 'accepted_ids': ['synthetic-photo'], 'revision_number': 4,
                    'published_capture_ids': ['synthetic-photo'],
                    'photo_registration': {'synthetic-photo': {'image_sha256': 'a'*64,
                        'metadata_sha256': 'b'*64, 'seed_sha256': seed_sha}},
                    'registration': {'status': 'verified', 'session_id': session, 'seed_sha256': seed_sha,
                        'basis': 'opencv_rh_from_unity', 'evidence_job_id': job_id,
                        'physical_acceptance': 'accepted',
                        'model_from_session': [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
                        'pose_fit': {'matched_views': 3, 'baseline_m': .5, 'position_rmse_m': .01,
                            'max_position_error_m': .02, 'max_rotation_error_deg': 2.}},
                    "artifact": {"sha256": digest(originals["current.ply"]), "size_bytes": len(originals["current.ply"]), "vertex_count": 4},
                    "display_artifact": {"sha256": digest(originals["display.ply"]), "size_bytes": len(originals["display.ply"]),
                        "vertex_count": 3, 'source_sha256': digest(originals['current.ply'])}}
    originals["status.json"] = json.dumps(source_state).encode()
    for name, content in originals.items():
        (online / name).write_bytes(content)
        (original / name).write_bytes(content)
    candidate = splat_bytes(2, 2.)
    candidate_checkpoint = b"candidate optimizer and locked region"
    (folder / "candidate.ply").write_bytes(candidate)
    (folder / "candidate.ckpt").write_bytes(candidate_checkpoint)
    record = {"id": region, "project_id": project, "session_id": session, "name": "Synthetic corner",
              "min": [0, 0, 0], "max": [1, 1, 1], "status": "locked",
              "source_sha256": digest(originals["current.ply"]), "source_checkpoint_sha256": digest(originals["current.ckpt"]),
              "artifact": {"sha256": digest(candidate), "size_bytes": len(candidate), "vertex_count": 2},
              "checkpoint_sha256": digest(candidate_checkpoint), "accepted_ids": ["synthetic-photo"],
              "model_metrics": {"training_steps": 12, 'published_capture_ids': ['synthetic-photo']},
              "replacement_capture_ids": [], 'draft_action': 'clear'}
    (folder / "region.json").write_text(json.dumps(record))
    return tmp_path, project, region, online, folder, originals, candidate, candidate_checkpoint


def test_publish_then_restore_preserves_exact_originals_and_pause(ready_area):
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    applied = request_region_action(root, project, region, "publish")
    assert applied["status"] == "published"
    assert (online / "current.ply").read_bytes() == candidate
    assert (online / "display.ply").read_bytes() == candidate
    assert (online / "current.ckpt").read_bytes() == checkpoint
    assert (online / "display-revisions" / (digest(candidate) + ".ply")).read_bytes() == candidate
    state = json.loads((online / "status.json").read_text())
    assert state["status"] == "paused_quality_review" and state["preserved_marker"] == "synthetic only"
    assert state["last_area_edit_id"] == region
    for name, content in originals.items():
        assert (folder / "original" / name).read_bytes() == content
    restored = request_region_action(root, project, region, "restore")
    assert restored["status"] == "restored"
    for name in ("current.ply", "display.ply", "current.ckpt"):
        assert (online / name).read_bytes() == originals[name]
    restored_state = json.loads((online / 'status.json').read_text())
    assert restored_state['revision_number'] == 6
    original_state = json.loads(originals['status.json'])
    for key, value in original_state.items():
        if key not in ('revision_number', 'updated_utc', 'display_artifact'):
            assert restored_state[key] == value
    assert restored_state['display_artifact']['sha256'] == original_state['display_artifact']['sha256']


@pytest.mark.parametrize('changed_model', [False, True])
def test_protection_applies_only_unchanged_geometry_and_keeps_live_updates_running(ready_area, changed_model):
    root, project, region, online, folder, originals, _candidate, checkpoint = ready_area
    source = json.loads((online / 'status.json').read_text())
    source['status'] = 'waiting_for_new_views'
    (online / 'status.json').write_text(json.dumps(source))
    candidate = splat_bytes(4, 9.) if changed_model else originals['current.ply']
    (folder / 'candidate.ply').write_bytes(candidate)
    record = json.loads((folder / 'region.json').read_text())
    record.update(draft_action='protect', artifact={'sha256': digest(candidate),
        'size_bytes': len(candidate), 'vertex_count': 4})
    record['quality_review'] = {'method': 'unchanged_geometry_protection_v1',
        'status': 'ready_for_visual_review', 'source_sha256': record['source_sha256'],
        'candidate_sha256': digest(candidate), 'candidate_checkpoint_sha256': record['checkpoint_sha256'],
        'selection_capture_ids': [], 'removed_gaussians': 0, 'appended_gaussians': 0}
    (folder / 'region.json').write_text(json.dumps(record))
    if changed_model:
        with pytest.raises(OnlineError, match='every source model byte'):
            request_region_action(root, project, region, 'publish')
        assert (online / 'current.ply').read_bytes() == originals['current.ply']
        assert (online / 'current.ckpt').read_bytes() == originals['current.ckpt']
    else:
        result = request_region_action(root, project, region, 'publish')
        state = json.loads((online / 'status.json').read_text())
        assert result['status'] == 'published'
        assert (online / 'current.ply').read_bytes() == originals['current.ply']
        assert (online / 'current.ckpt').read_bytes() == checkpoint
        assert state['status'] == 'waiting_for_new_views'
        assert state['accepted_ids'] == source['accepted_ids']
        assert state['visible_count'] == 1
        restored = request_region_action(root, project, region, 'restore')
        assert restored['status'] == 'restored'
        # The preimage is the original state captured before this test's active
        # runtime update. Preserve that exact pre-edit control state on restore.
        assert json.loads((online / 'status.json').read_text())['status'] == json.loads(originals['status.json'])['status']


@pytest.mark.parametrize("changed", ["current.ply", "current.ckpt"])
def test_stale_source_refuses_publication_without_overwriting_newer_work(ready_area, changed):
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    (online / changed).write_bytes(b"newer work")
    before = {name: (online / name).read_bytes() for name in originals}
    with pytest.raises(OnlineError, match="Newer model work"):
        request_region_action(root, project, region, "publish")
    assert {name: (online / name).read_bytes() for name in originals} == before


@pytest.mark.parametrize("changed", ["candidate.ply", "candidate.ckpt"])
def test_tampered_candidate_cannot_replace_current_model(ready_area, changed):
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    (folder / changed).write_bytes(b"tampered")
    with pytest.raises(OnlineError):
        request_region_action(root, project, region, "publish")
    assert {name: (online / name).read_bytes() for name in originals} == originals


def test_restore_refuses_to_overwrite_post_publish_work(ready_area):
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    request_region_action(root, project, region, "publish")
    (online / "current.ckpt").write_bytes(b"newer checkpoint after region publication")
    before = {name: (online / name).read_bytes() for name in originals}
    with pytest.raises(OnlineError, match="Newer model work"):
        request_region_action(root, project, region, "restore")
    assert {name: (online / name).read_bytes() for name in originals} == before


def test_publish_failure_rolls_back_already_replaced_artifacts(ready_area, monkeypatch):
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    from bridge import online as online_module
    original_replace = online_module._replace_with_retry
    failed = False
    def fail_once(source, destination):
        nonlocal failed
        if destination == online / "display.ply" and not failed:
            failed = True
            raise OSError("synthetic publication interruption")
        return original_replace(source, destination)
    monkeypatch.setattr(online_module, "_replace_with_retry", fail_once)
    with pytest.raises((OnlineError, OSError)):
        request_region_action(root, project, region, "publish")
    assert failed
    for name in ("current.ply", "display.ply", "current.ckpt"):
        assert (online / name).read_bytes() == originals[name]
    assert json.loads((online / "status.json").read_text()) == json.loads(originals["status.json"])
    assert json.loads((folder / "region.json").read_text())["status"] == "locked"


def test_applied_and_restored_revisions_use_existing_verified_reader(ready_area):
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    request_region_action(root, project, region, 'publish')
    store = Store(root)
    state, path, _alignment = read_online_preview(store, project, json.loads(
        (folder/'region.json').read_text())['session_id'])
    session = state['session_id']
    assert state['revision_number'] == 5 and path.read_bytes() == candidate
    sha = state['display_artifact']['sha256']
    receipt = json.loads((online/'display-revisions'/(sha+'.json')).read_text())
    assert receipt == {'project_id': project, 'session_id': session, 'revision_number': 5,
        **{key: state['display_artifact'][key] for key in
           ('sha256', 'size_bytes', 'vertex_count', 'source_sha256')}}
    request_region_action(root, project, region, 'restore')
    restored, path, _alignment = read_online_preview(store, project, session)
    assert restored['revision_number'] == 6 and path.read_bytes() == originals['display.ply']
    assert read_online_preview(store, project, session, sha)[1].read_bytes() == candidate


def test_area_apply_uses_headset_budget_and_rollback_preserves_bounded_display(ready_area, monkeypatch):
    from bridge import reconstruct, online as online_module
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    full = splat_bytes(240001, 2.)
    bounded = splat_bytes(2, 3.)
    (folder/'candidate.ply').write_bytes(full)
    record = json.loads((folder/'region.json').read_text())
    record['artifact'] = {'sha256': digest(full), 'size_bytes': len(full), 'vertex_count': 240001}
    (folder/'region.json').write_text(json.dumps(record))
    def convert(source, destination, *, max_vertices):
        assert source == folder/'candidate.ply' and max_vertices <= 240000
        destination.write_bytes(bounded)
        return {'vertex_count': 2, 'sha256': digest(bounded), 'size_bytes': len(bounded)}
    monkeypatch.setattr(reconstruct, 'make_quest_preview', convert)
    request_region_action(root, project, region, 'publish')
    assert (online/'current.ply').read_bytes() == full
    assert (online/'display.ply').read_bytes() == bounded
    preimages = {name: (online/name).read_bytes() for name in originals}
    replace = online_module._replace_with_retry
    interrupted = False
    def fail_once(source, destination):
        nonlocal interrupted
        if destination == online/'display.ply' and not interrupted:
            interrupted = True
            raise OSError('synthetic restore interruption')
        return replace(source, destination)
    monkeypatch.setattr(online_module, '_replace_with_retry', fail_once)
    with pytest.raises(OSError, match='restore interruption'):
        request_region_action(root, project, region, 'restore')
    assert {name: (online/name).read_bytes() for name in originals} == preimages


def test_clear_draft_cannot_introduce_unproved_photo_lineage(ready_area):
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    record = json.loads((folder/'region.json').read_text())
    record['accepted_ids'].append('unproved-new-photo')
    (folder/'region.json').write_text(json.dumps(record))
    with pytest.raises(OnlineError, match='lineage'):
        request_region_action(root, project, region, 'publish')
    assert {name: (online/name).read_bytes() for name in originals} == originals


def test_publication_failure_keeps_preimages_if_rollback_is_also_blocked(ready_area, monkeypatch):
    from bridge import online as online_module
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    replace = online_module._replace_with_retry
    def blocked(source, destination):
        if destination == online/'display.ply' or '.area-rollback-' in str(source):
            raise OSError('synthetic persistent file lock')
        return replace(source, destination)
    monkeypatch.setattr(online_module, '_replace_with_retry', blocked)
    with pytest.raises(OSError, match='persistent file lock'):
        request_region_action(root, project, region, 'publish')
    for name in ('current.ckpt', 'current.ply', 'display.ply'):
        assert (online/('.area-preimage-'+name)).read_bytes() == originals[name]
    assert (online/'status.json').read_bytes() == originals['status.json']


@pytest.mark.parametrize('after_write', [False, True])
def test_region_receipt_failure_rolls_back_entire_apply(ready_area, monkeypatch, after_write):
    from bridge import region_edit
    root, project, region, online, folder, originals, candidate, checkpoint = ready_area
    original_record = (folder/'region.json').read_bytes()
    write = region_edit._write
    def fail_receipt(destination, record):
        if after_write:
            write(destination, record)
        raise OSError('synthetic area receipt failure')
    monkeypatch.setattr(region_edit, '_write', fail_receipt)
    with pytest.raises(OSError, match='receipt failure'):
        request_region_action(root, project, region, 'publish')
    assert {name: (online/name).read_bytes() for name in originals} == originals
    assert (folder/'region.json').read_bytes() == original_record
