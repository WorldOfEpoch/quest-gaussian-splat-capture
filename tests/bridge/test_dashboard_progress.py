import hashlib
import json
import struct
import uuid
import pytest

from bridge.dashboard import dashboard_current_model, dashboard_jobs, render_dashboard
from bridge.store import Store


def _ply():
    props = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n" +
              "".join(f"property float {name}\n" for name in props) + "end_header\n")
    return header.encode() + struct.pack("<42f", *([0.] * 42))


@pytest.mark.parametrize("ledger", ["accepted_ids", "published_capture_ids"])
@pytest.mark.parametrize('reviewed', [False,True])
def test_current_model_progress_separates_baseline_match_live_admission_and_pending(tmp_path, ledger,reviewed):
    store = Store(tmp_path)
    project = store.create_project("Home")
    session = store.create_session(project["id"], str(uuid.uuid4()))
    ids = [str(uuid.uuid4()) for _ in range(6)]
    for capture_id in ids:
        image = b"saved-original-" + capture_id.encode()
        store.save_capture(session["id"], {
            "capture_id": capture_id, "project_id": project["id"],
            "session_id": session["id"], "tracking_origin_id": session["tracking_origin_id"],
            "image_encoding": "jpeg", "image_sha256": hashlib.sha256(image).hexdigest(),
        }, image)
    with store.connection() as db:
        db.execute("CREATE TABLE capture_selections (capture_id TEXT PRIMARY KEY, included INTEGER)")
        db.execute("INSERT INTO capture_selections VALUES (?,0)", (ids[0],))
    seed = _ply()
    seed_sha = hashlib.sha256(seed).hexdigest()
    job_id = str(uuid.uuid4())
    job_dir = tmp_path / "projects" / project["id"] / "reconstructions" / job_id
    job_dir.mkdir(parents=True)
    seed_path = job_dir / "splat.ply"
    seed_path.write_bytes(seed)
    (job_dir / "job.json").write_text(json.dumps({
        "job_id": job_id, "project_id": project["id"], "session_id": session["id"],
        "status": "ready_for_blender", "registered_views": 3,
        'used_capture_ids': ids[:3],
        "input_captures": [{"capture_id": item} for item in ids[:4]],
        "artifact": {"path": str(seed_path), "sha256": seed_sha, "size_bytes": len(seed),
                     "vertex_count": 3},
    }))
    online = tmp_path / "projects" / project["id"] / "online" / session["id"]
    online.mkdir(parents=True)
    (online / "current.ply").write_bytes(seed)
    (online / "status.json").write_text(json.dumps({
        "project_id": project["id"], "session_id": session["id"],
        # The accepted model can include more trained cameras than passed the
        # stricter live-reference test. Those counts must remain separate.
        "seed_sha256": seed_sha, "seed_ids": ids[:2] if reviewed else ids[:4], "accepted_ids": [ids[4]],
        "status": "paused_quality_review", "updated_utc": "2026-09-28T12:00:00Z",
        "visible_count": 5,  # Legacy worker counted one unmatched seed input.
        "registration": {"evidence_job_id": job_id},
        'accepted_baseline': {'job_id':job_id} if reviewed else None,
        "artifact": {"sha256": seed_sha, "size_bytes": len(seed), "vertex_count": 3},
        ledger: [ids[0], ids[4], ids[4]],  # Reused seed and repeated delta count once.
    }))
    model = dashboard_current_model(tmp_path, project["id"])
    assert model["source"] == "online" and model["status"] == "paused_quality_review"
    assert model["seed_job_id"] == job_id
    assert model["model_updated_utc"]
    reference = next(job for job in dashboard_jobs(tmp_path, project["id"]) if job["id"] == job_id)
    assert reference["viewable"] and reference["url"] == f"/ui/jobs/{project['id']}/{job_id}/splat.ply"
    assert model["accepted_views"] == 4  # Three registered baseline views plus one live view.
    assert model["physical_acceptance"] == "pending"
    assert model["photo_progress"] == {
        "captured_count": 6, "selected_count": 5, "baseline_input_count": 4,
        "baseline_registered_count": 3, "baseline_unmatched_count": 1,
        "added_count": 1, "pending_selected_count": 2 if reviewed else 1, "excluded_count": 1,
        "excluded_after_admission_count": 1, "rejected_count": None,
    }
    # Counts are freshly read from the gallery's DB, not stale worker status.
    with store.connection() as db:
        db.execute("UPDATE capture_selections SET included=1 WHERE capture_id=?", (ids[0],))
    updated = dashboard_current_model(tmp_path, project["id"])
    assert updated["photo_progress"]["selected_count"] == 6
    html = render_dashboard([project], project, store.entries(project["id"]),
                            store.sessions(project["id"]), [], "csrf")
    assert "id='photo-progress'" in html
