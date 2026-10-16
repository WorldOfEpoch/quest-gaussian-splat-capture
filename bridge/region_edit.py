"""Model-space area edit contracts; GPU work remains in GsplatTrainer.

A region is a volume, not a list of mutable row indexes. Bounds are in the
verified seed model frame, never screen pixels or assumed physical meters.
"""
from __future__ import annotations

import uuid
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .online import OnlineError


def validate_region(region: dict) -> dict:
    try:
        identifier = str(uuid.UUID(region["id"]))
        name = region["name"].strip()
        low = np.asarray(region["min"], dtype=np.float64)
        high = np.asarray(region["max"], dtype=np.float64)
        if (not 1 <= len(name) <= 120 or low.shape != (3,) or high.shape != (3,)
                or not np.isfinite([low, high]).all() or not (high > low).all()
                or np.abs([low, high]).max() > 1e6):
            raise ValueError()
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise OnlineError("Area needs a name, UUID, and finite ordered 3D bounds") from error
    return {"id": identifier, "name": name, "min": low.tolist(), "max": high.tolist()}


def region_membership(points: np.ndarray, scales: np.ndarray, region: dict,
                      *, contained: bool = False) -> np.ndarray:
    """Conservative three-sigma sphere covers each anisotropic footprint.

    Containment protects the outside boundary during deletion; intersection
    rejects later growth that could overlap a locked area even if its center
    is just outside. Gaussian tails are infinite; three sigma is the explicit
    operational support bound, not a claim of zero mathematical influence.
    """
    points, scales = np.asarray(points), np.asarray(scales)
    if points.ndim != 2 or points.shape[1:] != (3,) or scales.shape != points.shape:
        raise OnlineError("Invalid area geometry dimensions")
    if not np.isfinite(points).all() or not np.isfinite(scales).all() or (scales < 0).any():
        raise OnlineError("Invalid area geometry values")
    region = validate_region(region)
    low, high = np.asarray(region["min"]), np.asarray(region["max"])
    radius = 3. * scales.max(axis=1)
    if contained:
        return ((points - radius[:, None] >= low) & (points + radius[:, None] <= high)).all(axis=1)
    closest = np.clip(points, low, high)
    return ((points - closest) ** 2).sum(axis=1) <= radius ** 2


def regions_overlap(left: dict, right: dict) -> bool:
    left, right = validate_region(left), validate_region(right)
    return bool((np.asarray(left["max"]) >= right["min"]).all() and
                (np.asarray(right["max"]) >= left["min"]).all())


def review_depth_conflicts(points, scales, rotations, views, *, clearance,
                           minimum_views=3, minimum_baseline_m=.10, world_units_per_meter=1.):
    """Find private cleanup candidates in *observed* multi-view free space.

    Inputs share a calibrated world frame: means, positive standard deviations,
    orthonormal rotation matrices and positive camera-z depth images. Each view
    supplies capture_id, world_to_camera, intrinsics and camera_center_m. Depth
    holes, off-image footprints and near-plane crossings cast no free vote.
    The whole projected 3-sigma oriented box must be in front of the smallest
    valid depth across its conservative image square. Any center-ray surface
    agreement vetoes cleanup. This function neither changes nor publishes rows;
    candidates still need locked-area exclusion and independent RGB review.
    Three sigma bounds operational support, not infinite Gaussian tails.
    """
    from scipy.ndimage import minimum_filter
    p, s, r = (np.asarray(v, dtype=np.float64) for v in (points, scales, rotations))
    if (p.ndim != 2 or p.shape[1:] != (3,) or s.shape != p.shape or
            r.shape != (len(p), 3, 3) or not all(np.isfinite(v).all() for v in (p,s,r)) or
            (s <= 0).any() or not np.allclose(r.transpose(0,2,1)@r, np.eye(3), atol=1e-5) or
            not np.allclose(np.linalg.det(r), 1, atol=1e-5) or
            not np.isfinite(clearance) or clearance <= 0 or
            type(minimum_views) is not int or not 3 <= minimum_views <= 32 or
            not np.isfinite(minimum_baseline_m) or minimum_baseline_m < .05 or
            not np.isfinite(world_units_per_meter) or world_units_per_meter<=0):
        raise OnlineError('Invalid calibrated depth cleanup geometry or evidence limits')
    if len(views)<minimum_views or len(views)>128:
        raise OnlineError('Depth cleanup needs independent overlapping views')
    ids = [v['capture_id'] for v in views]
    if any(not isinstance(cid,str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise OnlineError('Depth cleanup capture identities must be distinct')
    free = np.zeros((len(p),len(views)), dtype=bool)
    supported = np.zeros(len(p),dtype=bool)
    origins=[]
    actual_origins=[]
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for vi,v in enumerate(views):
        d=np.asarray(v['depth'],dtype=np.float64)
        w=np.asarray(v['world_to_camera'],dtype=np.float64)
        k=np.asarray(v['intrinsics'],dtype=np.float64)
        origin=np.asarray(v['camera_center_m'],dtype=np.float64)
        if (d.ndim!=2 or min(d.shape)<3 or w.shape!=(4,4) or k.shape!=(3,3) or origin.shape!=(3,) or
                not all(np.isfinite(a).all() for a in (w,k,origin)) or
                not np.allclose(w[3],[0,0,0,1]) or not np.allclose(k[2],[0,0,1]) or
                not np.allclose(w[:3,:3].T@w[:3,:3],np.eye(3),atol=1e-5) or
                not np.isclose(np.linalg.det(w[:3,:3]),1,atol=1e-5) or k[0,0]<=0 or k[1,1]<=0):
            raise OnlineError('Invalid depth cleanup camera')
        origins.append(origin)
        actual_origin=-w[:3,:3].T@w[:3,3]/world_units_per_meter
        for declared, actual in zip(origins[:-1], actual_origins):
            if not np.isclose(np.linalg.norm(origin-declared),np.linalg.norm(actual_origin-actual),
                              atol=.001,rtol=.001):
                raise OnlineError('Declared depth camera baselines differ from calibrated transforms')
        actual_origins.append(actual_origin)
        # Invalid depth is unknown. A single unknown pixel in the conservative
        # support square prevents that view from voting to erase geometry.
        d=np.where(np.isfinite(d)&(d>0),d,0.)
        filters={radius:minimum_filter(d,size=2*radius+1,mode='constant',cval=0.)
                 for radius in (1,2,4,8,16,32,64)}
        for start in range(0,len(p),16384):
            stop=min(start+16384,len(p))
            c=p[start:stop]@w[:3,:3].T+w[:3,3]
            axes=np.einsum('ij,njk->nik',w[:3,:3],r[start:stop])*s[start:stop,None,:]*3.
            zextent=np.abs(axes[:,2,:]).sum(axis=1)
            safe=c[:,2]>zextent+1e-6
            center_pixels=c@k.T
            uv=center_pixels[:,:2]/np.maximum(center_pixels[:,2:3],1e-8)
            # Clip before integer conversion even for out-of-frame huge models.
            ij=np.rint(np.clip(uv,-1e8,1e8)).astype(np.int64)
            in_image=safe&(ij[:,0]>=0)&(ij[:,0]<d.shape[1])&(ij[:,1]>=0)&(ij[:,1]<d.shape[0])
            idx=np.flatnonzero(in_image)
            if not len(idx):continue
            sampled=d[ij[idx,1],ij[idx,0]]
            agreement=(sampled>0)&(np.abs(sampled-c[idx,2])<=zextent[idx]+clearance)
            supported[start+idx[agreement]]=True
            possible=(sampled>0)&(c[idx,2]+zextent[idx]+clearance<sampled)
            idx=idx[possible]
            if not len(idx):continue
            corners=c[idx,None,:]+np.einsum('nij,kj->nki',axes[idx],signs)
            # Box corners also must be in front. This is stricter than the
            # ellipsoid's near-plane bound and avoids singular projected boxes.
            ok=(corners[:,:,2]>1e-6).all(axis=1)
            projected=corners@k.T
            px=projected[:,:,:2]/np.maximum(projected[:,:,2:3],1e-8)
            radius=np.ceil(np.max(np.abs(px-ij[idx,None,:]),axis=(1,2)))+1
            bucket=np.searchsorted([1,2,4,8,16,32,64],radius)
            for bi,rad in enumerate((1,2,4,8,16,32,64)):
                local=np.flatnonzero(ok&(bucket==bi))
                ii=idx[local]
                minimum=filters[rad][ij[ii,1],ij[ii,0]]
                yes=(minimum>0)&(c[ii,2]+zextent[ii]+clearance<minimum)
                free[start+ii[yes],vi]=True
    counts=free.sum(axis=1)
    eligible=np.flatnonzero((counts>=minimum_views)&~supported)
    centers=np.array(origins)
    selected=np.zeros(len(p),dtype=bool)
    # Usually only a small fraction qualify. Explicit independent centers
    # avoid treating repeated captures from one position as multiple evidence.
    for row in eligible:
        chosen=[]
        for view_index in np.flatnonzero(free[row]):
            if all(np.linalg.norm(centers[view_index]-centers[old])>=minimum_baseline_m for old in chosen):
                chosen.append(view_index)
        selected[row]=len(chosen)>=minimum_views
    return dict(candidate_mask=selected, surface_supported=supported, free_view_mask=free,
                free_view_count=counts, capture_ids=ids,
                method='conservative_full_footprint_multi_view_depth_v1')


def review_opacity_cleanup(before: np.ndarray, after: np.ndarray, views: list[dict], *,
                           selection_capture_ids: list[str], locked_regions=()) -> dict:
    """Review a private attenuation draft; never mutate or publish a model.

    Each view supplies calibrated, identically rendered float RGB arrays named
    reference/before/after, a fixed geometric support mask from the original
    selected rows, a capture_id, and a camera_center_m in the verified frame.
    Validation photos must not have selected or optimized the changed rows.
    This is a photometric check, not proof that a point occupies free space.
    Caller retains the source/pose/image hashes and inspects the output before
    using the existing area publication owner. No normal capture invokes it.
    """
    fields = set(before.dtype.names or ())
    needed = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"}
    if (before.ndim != 1 or before.dtype != after.dtype or before.shape != after.shape
            or not needed <= fields or not len(before)):
        raise OnlineError("Cleanup must retain the original Gaussian rows and schema")
    # Byte comparison also protects SH coefficients, rotations and signed zero.
    if any(before[name].tobytes() != after[name].tobytes() for name in fields - {"opacity"}):
        raise OnlineError("Opacity cleanup cannot change geometry, color, or row identity")
    old_alpha, new_alpha = before["opacity"], after["opacity"]
    if (not np.isfinite(old_alpha).all() or not np.isfinite(new_alpha).all()
            or (new_alpha > old_alpha).any()):
        raise OnlineError("Cleanup may only reduce finite opacity values")
    changed = new_alpha < old_alpha
    if not changed.any():
        raise OnlineError("Cleanup contains no changed rows")
    points = np.column_stack([before[key][changed] for key in ("x", "y", "z")])
    scales = np.exp(np.column_stack([before[f"scale_{i}"][changed] for i in range(3)]).astype(np.float64))
    for region in locked_regions:
        if region_membership(points, scales, region).any():
            raise OnlineError("Cleanup intersects a locked Gaussian footprint")
    return {**review_cleanup_views(views, selection_capture_ids=selection_capture_ids),
            "method": "fixed_geometry_opacity_review_v1",
            "changed_rows": int(changed.sum()), "total_rows": len(before),
            "geometry_and_color_exact": True, "locked_regions_preserved": True}


def review_cleanup_views(views: list[dict], *, selection_capture_ids: list[str]) -> dict:
    """Shared fixed-camera quality gate, separate from each edit's geometry proof.

    Opacity and depth-supported covariance drafts use the same image criteria.
    An image pass alone never authorizes editing geometry or publishing a draft.
    Inputs and calibration evidence are the caller's responsibility as described
    in review_opacity_cleanup. This routine has no model or filesystem access.
    """
    selection = set(selection_capture_ids)
    if not selection or len(selection) != len(selection_capture_ids):
        raise OnlineError("Record the distinct photos used to select the cleanup")
    if not isinstance(views, list) or len(views) < 2:
        raise OnlineError("Cleanup needs independent validation cameras")
    ids, centers, rows = set(), [], []
    tolerance = 0.00025  # Less than one tenth of an 8-bit channel level.
    for view in views:
        cid = view["capture_id"]
        center = np.asarray(view["camera_center_m"], dtype=np.float64)
        rgb = [np.asarray(view[key], dtype=np.float32) for key in ("reference", "before", "after")]
        mask = np.asarray(view["support"])
        if (not isinstance(cid, str) or not cid or cid in ids or center.shape != (3,)
                or not np.isfinite(center).all() or rgb[0].ndim != 3 or rgb[0].shape[-1] != 3
                or any(x.shape != rgb[0].shape or not np.isfinite(x).all()
                       or (x < 0).any() or (x > 1).any() for x in rgb)
                or mask.dtype != np.bool_ or mask.shape != rgb[0].shape[:2]):
            raise OnlineError("Cleanup comparison needs distinct calibrated finite RGB views and fixed support")
        ids.add(cid)
        reference, old, new = rgb
        old_error, new_error = np.abs(old-reference).mean(axis=-1), np.abs(new-reference).mean(axis=-1)
        delta = np.abs(old-new).max(axis=-1)
        # Do not let a caller omit damaged pixels from its patch review.
        outside_change = float(delta[~mask].max()) if (~mask).any() else 0.
        support_count = int(mask.sum())
        old_patch = float(old_error[mask].mean()) if support_count else 0.
        new_patch = float(new_error[mask].mean()) if support_count else 0.
        tile_regressions = []
        for y in range(0, mask.shape[0], 32):
            for x in range(0, mask.shape[1], 32):
                tile = np.s_[y:y+32, x:x+32]
                local = mask[tile]
                if local.any():
                    loss = float((new_error[tile][local]-old_error[tile][local]).mean())
                    if loss > tolerance:
                        tile_regressions.append({"x": x, "y": y, "l1_increase": loss})
        good = mask & (old_error < .03)
        good_increase = float((new_error[good]-old_error[good]).mean()) if good.any() else 0.
        held_out = cid not in selection
        changed_pixels = int((mask & (delta > 1e-5)).sum())
        row = {"capture_id": cid, "held_out": held_out, "support_pixels": support_count,
               "changed_pixels": changed_pixels, "before_l1": float(old_error.mean()),
               "after_l1": float(new_error.mean()), "patch_before_l1": old_patch,
               "patch_after_l1": new_patch, "good_pixel_l1_increase": good_increase,
               "outside_support_max_change": outside_change, "regressing_tiles": tile_regressions,
               "useful_patch": changed_pixels >= 16 and old_patch-new_patch >= .0005
                   and (old_patch-new_patch)/max(old_patch, 1e-8) >= .01}
        row["non_regression"] = (row["after_l1"] <= row["before_l1"]+tolerance
            and new_patch <= old_patch+tolerance and not tile_regressions
            and good_increase <= tolerance and outside_change <= 1e-5)
        rows.append(row)
        if held_out and row["useful_patch"]:
            centers.append(center)
    independent = any(np.linalg.norm(a-b) >= .05 for i, a in enumerate(centers) for b in centers[i+1:])
    accepted = independent and all(row["non_regression"] for row in rows)
    return {"method": "fixed_camera_cleanup_review_v1",
            "status": "ready_for_visual_review" if accepted else "quality_rejected",
            "independent_validation_gain": independent, "allowed_l1_regression": tolerance,
            "views": rows, "published": False,
            "limitation": "Photometric evidence does not establish empty space or unobserved-view quality."}


def _folder(root: Path, project_id: str, region_id: str) -> Path:
    project_id, region_id = str(uuid.UUID(project_id)), str(uuid.UUID(region_id))
    base = root.resolve() / "projects" / project_id
    path = base / "area-edits" / region_id
    if any(p.is_symlink() for p in (base, path.parent, path)) or not path.resolve().is_relative_to(base.resolve()):
        raise OnlineError("Area edit path is invalid")
    return path


def _record(root: Path, project_id: str, region_id: str) -> tuple[Path, dict]:
    folder = _folder(root, project_id, region_id)
    try:
        record = json.loads((folder / "region.json").read_text(encoding="utf-8"))
        if record["project_id"] != project_id or record["id"] != region_id:
            raise ValueError()
        validate_region(record)
        return folder, record
    except (OSError, ValueError, KeyError) as error:
        raise OnlineError("Area edit was not found") from error


def _write(folder: Path, record: dict) -> dict:
    from .online import _atomic_json
    from .models import utc_now
    record["updated_utc"] = utc_now()
    _atomic_json(folder / "region.json", record)
    return record


def list_regions(root: Path, project_id: str) -> list[dict]:
    project_id = str(uuid.UUID(project_id))
    base = root / "projects" / project_id / "area-edits"
    records = []
    if base.is_dir() and not base.is_symlink():
        for path in base.glob("*/region.json"):
            try:
                _, record = _record(root, project_id, str(uuid.UUID(path.parent.name)))
                records.append(record)
            except (ValueError, OnlineError):
                continue
    return sorted(records, key=lambda r: r.get("updated_utc", ""), reverse=True)


def create_region(root: Path, project_id: str, plan: dict) -> dict:
    from .dashboard import dashboard_current_model
    from .store import Store
    store = Store(root)
    store.project(project_id)
    current = dashboard_current_model(root, project_id)
    if not current or current.get("source") != "online":
        raise OnlineError("Area edits require an existing online model checkpoint")
    keys = ("source_model_id", "source_sha256", "coordinate_frame", "session_id")
    expected = dict(zip(keys, (current["id"], current["source_sha256"],
                              current["coordinate_frame"], current["session_id"])))
    if any(plan.get(key) != value for key, value in expected.items()):
        raise OnlineError("The model changed; refresh and select the area again")
    region = validate_region({"id": str(uuid.uuid4()), "name": plan.get("name"),
                              "min": plan.get("min"), "max": plan.get("max")})
    capture_ids = plan.get("replacement_capture_ids", [])
    if (not isinstance(capture_ids, list) or len(capture_ids) > 128 or
            not all(isinstance(x, str) for x in capture_ids) or len(set(capture_ids)) != len(capture_ids)):
        raise OnlineError("Choose at most 128 distinct replacement photos")
    try:
        capture_ids = [str(uuid.UUID(x)) for x in capture_ids]
    except (ValueError, TypeError, AttributeError) as error:
        raise OnlineError("Invalid replacement photo selection") from error
    with store.connection() as db:
        rows = {r[0] for r in db.execute("SELECT capture_id FROM captures WHERE project_id=? AND session_id=?",
                                         (project_id, expected["session_id"]))}
        selections = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='capture_selections'").fetchone()
        excluded = {r[0] for r in db.execute("SELECT capture_id FROM capture_selections WHERE included=0")} if selections else set()
    if not set(capture_ids) <= rows or set(capture_ids) & excluded:
        raise OnlineError("Replacement photos must be included photos from the model's capture session")
    folder = _folder(root, project_id, region["id"])
    checkpoint = root / "projects" / project_id / "online" / expected["session_id"] / "current.ckpt"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise OnlineError("The source checkpoint is unavailable")
    folder.mkdir(parents=True, exist_ok=False)
    return _write(folder, {**region, **expected, "schema_version": 1, "project_id": project_id,
                          "replacement_capture_ids": capture_ids, "view": current["view"],
                          "source_checkpoint_sha256": _hash(checkpoint),
                          "status": "planned", "message": "Area saved; the displayed model is unchanged."})


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
        if digest is not None:
            return digest.hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def update_region_photos(root: Path, project_id: str, region_id: str, capture_ids: list[str]) -> dict:
    from .store import Store
    folder, record = _record(root, project_id, region_id)
    if record["status"] not in {"planned", "failed", "draft_ready", "quality_rejected"} or (folder / "action.lock").exists():
        raise OnlineError("Photo selection can change only before the area is locked or while no worker is active")
    if (not isinstance(capture_ids, list) or len(capture_ids) > 128 or
            not all(isinstance(x, str) for x in capture_ids) or len(set(capture_ids)) != len(capture_ids)):
        raise OnlineError("Choose at most 128 distinct replacement photos")
    store = Store(root)
    with store.connection() as db:
        choices = {r[0] for r in db.execute("SELECT capture_id FROM captures WHERE project_id=? AND session_id=?",
                                            (project_id, record["session_id"]))}
        has_selections = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='capture_selections'").fetchone()
        excluded = {r[0] for r in db.execute("SELECT capture_id FROM capture_selections WHERE included=0")} if has_selections else set()
    if not set(capture_ids) <= choices or set(capture_ids) & excluded:
        raise OnlineError("Replacement photos must be included photos from the source session")
    _source(root, record)
    record.update(replacement_capture_ids=list(capture_ids), status="planned",
                  message="Replacement photos updated. Rebuild the draft before locking it.")
    record.pop("artifact", None)
    record.pop("checkpoint_sha256", None)
    record.pop("quality_review", None)
    return _write(folder, record)


def verified_region_artifact(root: Path, project_id: str, region_id: str) -> tuple[dict, Path]:
    folder, record = _record(root, project_id, region_id)
    path = folder / "candidate.ply"
    artifact = record.get("artifact") or {}
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != artifact.get("size_bytes") or
            _hash(path) != artifact.get("sha256")):
        raise OnlineError("A verified area draft is not available yet")
    return record, path


def request_region_action(root: Path, project_id: str, region_id: str, action: str) -> dict:
    folder, record = _record(root, project_id, region_id)
    if action in {"publish", "restore"}:
        return _apply_candidate(root, project_id, region_id, restore=action == "restore")
    required = {"build": {"planned", "failed", "draft_ready", "quality_rejected"},
                "clear": {"planned", "failed", "draft_ready", "quality_rejected"},
                "protect": {"planned", "failed", "draft_ready", "quality_rejected"}, "lock": {"draft_ready"}}
    if action not in required or record["status"] not in required[action]:
        raise OnlineError("That action is unavailable in the area's current state")
    if action == "build" and len(record["replacement_capture_ids"]) < 2:
        raise OnlineError("Select at least two overlapping replacement photos before building")
    # Reuse the reconstruction installation. The web form cannot choose an
    # executable, command string or output path.
    python = root / "tools" / "nerfstudio310-venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise OnlineError("The configured reconstruction Python is unavailable")
    lock_path = folder / "action.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError as error:
        raise OnlineError("An area operation is already active") from error
    try:
        record.update(status="queued", requested_action=action, message="Area operation queued; current model is unchanged.")
        _write(folder, record)
        with (folder / "worker.log").open("ab") as log:
            subprocess.Popen([str(python), "-m", "bridge.region_edit", "--data-dir", str(root),
                              "--project-id", project_id, "--region-id", region_id, "--action", action],
                             cwd=str(Path(__file__).resolve().parent.parent), stdout=log, stderr=log,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except OSError as error:
        lock_path.unlink(missing_ok=True)
        record.update(status="failed", message="The area worker could not start.")
        _write(folder, record)
        raise OnlineError(record["message"]) from error
    return record


def _source(root: Path, record: dict):
    from .dashboard import verified_job
    from .reconstruct import validate_gaussian_ply
    session = str(uuid.UUID(record["session_id"]))
    base = root / "projects" / record["project_id"]
    online = base / "online" / session
    state = json.loads((online / "status.json").read_text(encoding="utf-8"))
    if (state["artifact"]["sha256"] != record["source_sha256"] or
            _hash(online / "current.ply") != record["source_sha256"] or
            _hash(online / "current.ckpt") != record["source_checkpoint_sha256"]):
        raise OnlineError("The source model changed; create a new area edit from the current model")
    evidence = verified_job(root, record["project_id"], state["registration"]["evidence_job_id"])
    if not evidence:
        raise OnlineError("The source model registration is no longer verifiable")
    seed = evidence[1]
    if validate_gaussian_ply(seed)[1] != state["seed_sha256"]:
        seed = seed.with_name("quest-preview.ply")
    if validate_gaussian_ply(seed)[1] != state["seed_sha256"]:
        raise OnlineError("The source seed no longer matches the model")
    return online, state, seed


def _replacement_views(store, record: dict, source: dict, seed: Path):
    """Preflight every selected original against the unchanged measured seed.

    Registered seed photos retain their optimized cameras. Other photos must
    independently localize against that same sparse map before any draft rows
    are cleared or a CUDA trainer is created.
    """
    from .online import (_json_digest, _selected_ids, registration_from_seed_job,
                         seed_camera_views, verified_registration, validated_camera,
                         ColmapLandmarkVerifier)
    from .online_torch import GsplatTrainer
    registration = source["registration"]
    if registration.get("verification_method") != "source_job_pose_fit":
        raise OnlineError("Area repair requires independently measured seed cameras")
    actual_seed, actual_registration = registration_from_seed_job(
        store, record["session_id"], registration["evidence_job_id"])
    if actual_seed.resolve() != seed.resolve() or actual_registration != registration:
        raise OnlineError("Area repair seed registration differs from its saved source")
    matrix, calibration, orientation = verified_registration(
        registration, record["session_id"], source["seed_sha256"])
    units = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    session = store.session(record["session_id"])
    if (session["project_id"] != record["project_id"] or
            registration["project_id"] != record["project_id"] or
            registration["tracking_origin_id"] != session["tracking_origin_id"]):
        raise OnlineError("Area repair belongs to a different anchored session")
    captures = record["replacement_capture_ids"]
    if (not isinstance(captures, list) or not 2 <= len(captures) <= 128 or
            len(set(captures)) != len(captures)):
        raise OnlineError("Select two to 128 distinct replacement photos")
    selected = _selected_ids(store, record["session_id"])
    if not set(captures + registration["seed_capture_ids"]) <= selected:
        raise OnlineError("Area repair includes excluded or unavailable source photos")
    references = seed_camera_views(store, registration)
    by_id = {view["capture_id"]: view for view in references}
    if not references or len(by_id) != len(references):
        raise OnlineError("Area repair has no distinct measured seed cameras")
    distortion = references[0]["distortion_coefficients"]
    if any(view["distortion_coefficients"] != distortion for view in references):
        raise OnlineError("Area repair seed mixes measured lens calibration")
    verifier = (ColmapLandmarkVerifier(store, registration, references, units)
                if set(captures) - by_id.keys() else None)
    items, seen = [], set()
    while len(seen) < len(captures):
        batch = store.export_pending_captures(record["session_id"], seen, set(captures), 16)
        if not batch:
            raise OnlineError("Selected replacement photos are no longer available")
        for item in batch:
            receipt, metadata = item["receipt"], item["metadata"]
            cid = receipt["capture_id"]
            if (cid in seen or cid not in captures or
                    receipt["project_id"] != record["project_id"] or
                    receipt["session_id"] != record["session_id"] or
                    metadata.get("capture_id") != cid or
                    metadata.get("image_sha256") != receipt["image_sha256"] or
                    metadata.get("project_id") != record["project_id"] or
                    metadata.get("session_id") != record["session_id"] or
                    metadata.get("tracking_origin_id") != session["tracking_origin_id"]):
                raise OnlineError("Replacement photo identity or anchored origin differs")
            signature = registration["camera_signature"]
            intrinsics = metadata.get("intrinsics", {})
            if (any(metadata.get(key) != signature[key]
                    for key in ("camera_id", "image_width", "image_height")) or
                    set(intrinsics) != set(signature["intrinsics"]) or any(
                        not np.isfinite(float(intrinsics[key])) or
                        abs(float(intrinsics[key])-float(signature["intrinsics"][key])) > 1.
                        for key in intrinsics)):
                raise OnlineError("Replacement photo camera calibration differs from the seed")
            tracked, _ = validated_camera(metadata, matrix, calibration, orientation)
            if cid in by_id:
                view = by_id[cid]
                original = registration["source_proof"]["captures"][cid]
                if (original["image_sha256"] != receipt["image_sha256"] or
                        original["metadata_sha256"] != _json_digest(metadata) or
                        Path(view["image_path"]).resolve() != Path(item["image_path"]).resolve()):
                    raise OnlineError("Replacement baseline original differs from measured seed evidence")
                proof = {"method": "verified_seed_frame", "source_proof":
                         original,
                         **registration["capture_pose_residuals"][cid]}
            else:
                view = {"capture_id": cid, "image_path": Path(item["image_path"]),
                        "pose": tracked, "intrinsics": references[0]["intrinsics"],
                        "distortion_coefficients": distortion}
                proof = verifier.verify(view)
                if "localized_pose" not in proof or "camera_intrinsics" not in proof:
                    raise OnlineError("Replacement photo lacks independently localized camera proof")
                pose, k = GsplatTrainer._validated_view(proof["localized_pose"], proof["camera_intrinsics"])
                d = np.asarray(proof.get("distortion_coefficients"), dtype=float)
                position = float(np.linalg.norm(pose[:3, 3]-tracked[:3, 3])/units)
                angle = float(np.degrees(np.arccos(np.clip(
                    (np.trace(pose[:3, :3].T@tracked[:3, :3])-1)/2, -1, 1))))
                if (d.shape != (5,) or not np.array_equal(d, distortion) or
                        not np.isfinite([position, angle]).all() or position > .10 or angle > 5.):
                    raise OnlineError("Replacement localization disagrees with its tracked camera or lens calibration")
                view = {**view, "pose": pose, "intrinsics": k}
            # Store reads verify image bytes. The seed camera owner additionally
            # verifies its original metadata hash; retain both for later checks.
            items.append({**view, "capture_id": cid, "image_path": Path(item["image_path"]),
                          "proof": proof, "image_sha256": receipt["image_sha256"],
                          "metadata_sha256": _json_digest(metadata)})
            seen.add(cid)
    return items, units, distortion, references


def _check_replacement_inputs(store, record: dict, views, *, source=None, seed=None) -> None:
    """Refuse changed selections, original bytes or metadata before saving a draft."""
    from .online import _json_digest, _selected_ids
    expected = {view["capture_id"]: view for view in views}
    if not expected.keys() <= _selected_ids(store, record["session_id"]):
        raise OnlineError("Replacement photo selection changed during the area operation")
    snapshot = store.export_session(record["session_id"], capture_ids=list(expected))
    for item in snapshot["captures"]:
        proof = expected[item["receipt"]["capture_id"]]
        if (item["receipt"]["image_sha256"] != proof["image_sha256"] or
                _json_digest(item["metadata"]) != proof["metadata_sha256"]):
            raise OnlineError("Replacement original or camera metadata changed during the area operation")
    if source is not None:
        from .online import registration_from_seed_job
        registration = source["registration"]
        if not set(registration["seed_capture_ids"]) <= _selected_ids(store, record["session_id"]):
            raise OnlineError("Measured seed photo selection changed during the area operation")
        actual_seed, actual_registration = registration_from_seed_job(
            store, record["session_id"], registration["evidence_job_id"])
        if actual_seed.resolve() != seed.resolve() or actual_registration != registration:
            raise OnlineError("Measured seed camera evidence changed during the area operation")


def _area_review_mask(pose, intrinsics, shape, region, units):
    """Project the *predeclared box*, never pixels chosen from image errors.

    Every allowed replacement Gaussian has its three-sigma sphere contained in
    this box. A small pixel margin covers rasterization rounding; effects
    outside it are independently checked by review_cleanup_views.
    """
    import itertools
    height, width = shape
    low, high = np.asarray(region["min"], dtype=float), np.asarray(region["max"], dtype=float)
    corners = np.asarray(list(itertools.product(*zip(low, high))), dtype=float)
    camera = (np.linalg.inv(pose) @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    mask = np.zeros((height, width), dtype=bool)
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= .25 * units):
        return mask
    projection = (intrinsics @ camera.T).T
    uv = projection[:, :2] / projection[:, 2, None]
    if not np.isfinite(uv).all():
        return mask
    left, top = np.floor(uv.min(axis=0) - 4).astype(int)
    right, bottom = np.ceil(uv.max(axis=0) + 4).astype(int)
    x0, x1 = max(0, min(width, left)), max(0, min(width, right + 1))
    y0, y1 = max(0, min(height, top)), max(0, min(height, bottom + 1))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _area_review_rgb(trainer, pose, intrinsics, shape):
    """Use the trainer's actual Gaussian renderer and SH appearance."""
    torch = trainer.torch
    height, width = shape
    with torch.no_grad():
        colors, appearance = trainer._appearance(len(trainer.params["means"]))
        rendered, _alpha, _info = trainer.rasterization(
            means=trainer.params["means"],
            quats=torch.nn.functional.normalize(trainer.params["quats"], dim=-1),
            scales=torch.exp(trainer.params["log_scales"]).clamp(min=1e-5, max=100.),
            opacities=torch.sigmoid(trainer.params["opacity_logits"]),
            colors=colors, **appearance,
            viewmats=torch.from_numpy(np.linalg.inv(pose).astype(np.float32)).to(trainer.device)[None],
            Ks=torch.from_numpy(intrinsics.astype(np.float32)).to(trainer.device)[None],
            width=width, height=height)
        return rendered[0].clamp(0, 1).detach().cpu().numpy()


def _prepare_area_review(trainer, references, region, capture_ids, units):
    """Choose fixed, independent measured seed cameras before optimization."""
    from PIL import Image
    choices = []
    for view in references:
        if view["capture_id"] in capture_ids:
            continue
        pose, k = trainer._validated_view(view["pose"], view["intrinsics"])
        with Image.open(view["image_path"]) as image:
            width, height = image.size
        shrink = min(1., trainer.max_image_edge / max(width, height))
        target = (max(1, round(width * shrink)), max(1, round(height * shrink)))
        scaled_k = k.copy()
        scaled_k[0] *= target[0] / width
        scaled_k[1] *= target[1] / height
        mask = _area_review_mask(pose, scaled_k, (target[1], target[0]), region, units)
        if mask.sum() >= 16:
            choices.append((view, pose, int(mask.sum())))
    if len(choices) < 2:
        return {"status": "quality_rejected", "reason": "Fewer than two measured held-out seed cameras see the selected area"}
    # Geometry-only selection, fixed before gradients. Cover distinct viewing
    # positions rather than choosing cameras with favorable image error.
    choices.sort(key=lambda row: (-row[2], row[0]["capture_id"]))
    chosen = [choices.pop(0)]
    while choices and len(chosen) < 8:
        choice = max(choices, key=lambda row: (min(np.linalg.norm(
            row[1][:3, 3] - previous[1][:3, 3]) for previous in chosen),
            row[2], row[0]["capture_id"]))
        chosen.append(choice)
        choices = [row for row in choices if row[0]["capture_id"] != choice[0]["capture_id"]]
    centers = [row[1][:3, 3] / units for row in chosen]
    if not any(np.linalg.norm(left - right) >= .05 for i, left in enumerate(centers)
               for right in centers[i + 1:]):
        return {"status": "quality_rejected", "reason": "Held-out area cameras lack a five-centimeter translated baseline"}
    fixed = []
    for view, pose, _support in chosen:
        pixels, scaled_k = trainer._scaled_image(view["image_path"], np.asarray(view["intrinsics"], dtype=float))
        mask = _area_review_mask(pose, scaled_k, pixels.shape[:2], region, units)
        fixed.append({"capture_id": view["capture_id"], "camera_center_m": (pose[:3, 3] / units).tolist(),
                      "pose": pose, "intrinsics": scaled_k, "reference": pixels,
                      "support": mask, "before": _area_review_rgb(trainer, pose, scaled_k, pixels.shape[:2])})
    return {"status": "prepared", "views": fixed}


def _finish_area_review(trainer, prepared, capture_ids):
    if prepared["status"] != "prepared":
        return prepared
    comparisons = []
    for view in prepared["views"]:
        comparisons.append({key: view[key] for key in
                            ("capture_id", "camera_center_m", "reference", "before", "support")})
        comparisons[-1]["after"] = _area_review_rgb(
            trainer, view["pose"], view["intrinsics"], view["reference"].shape[:2])
    return review_cleanup_views(comparisons, selection_capture_ids=capture_ids)


def review_area_candidate_files(before_ply: Path, candidate_ply: Path, references: list[dict],
                                region: dict, capture_ids: list[str], *, units: float,
                                distortion: list[float]) -> dict:
    """Review existing private PLYs with the same camera gate, without a build/apply."""
    import gc
    from .online_torch import GsplatTrainer
    before = GsplatTrainer(before_ply)
    before.configure_geometry_scale(units)
    before.configure_camera_distortion(distortion)
    prepared = _prepare_area_review(before, references, validate_region(region), capture_ids, units)
    del before
    gc.collect()
    if prepared["status"] != "prepared":
        return prepared
    after = GsplatTrainer(candidate_ply)
    after.configure_geometry_scale(units)
    after.configure_camera_distortion(distortion)
    try:
        return _finish_area_review(after, prepared, capture_ids)
    finally:
        del after
        gc.collect()


def run_region_action(root: Path, project_id: str, region_id: str, action: str) -> dict:
    from .online_torch import GsplatTrainer
    from .store import Store
    from .reconstruct import _worker_lock, validate_gaussian_ply
    folder, record = _record(root, project_id, region_id)
    with _worker_lock(root):
        online, source, seed = _source(root, record)
        record.update(status="locking" if action == "lock" else "building",
                      message="Preparing an isolated area draft; current model is unchanged.")
        _write(folder, record)
        if action == "lock":
            verified_region_artifact(root, project_id, region_id)
            if (folder / "candidate.ckpt").is_symlink() or _hash(folder / "candidate.ckpt") != record["checkpoint_sha256"]:
                raise OnlineError("The area draft checkpoint differs from its saved receipt")
            trainer = GsplatTrainer(seed, folder / "candidate.ckpt", published_ply=folder / "candidate.ply")
            trainer.lock_replacement_area()
        else:
            if action not in {"build", "clear", "protect"}:
                raise OnlineError("Unsupported area operation")
            record["draft_action"] = action
            record.pop("quality_review", None)
            # An older private candidate may remain on disk for recovery, but
            # its receipt must not be displayed as this new attempt's result.
            record.pop("artifact", None)
            record.pop("checkpoint_sha256", None)
            captures = record["replacement_capture_ids"] if action == "build" else []
            store = Store(root)
            prepared = (_replacement_views(store, record, source, seed) if action == "build" else None)
            backup = folder / "original"
            if not backup.exists():
                backup.mkdir()
                for filename in ("current.ckpt", "current.ply", "display.ply", "status.json"):
                    shutil.copy2(online / filename, backup / filename)
            if (_hash(backup / "current.ply") != record["source_sha256"] or
                    _hash(backup / "current.ckpt") != record["source_checkpoint_sha256"]):
                raise OnlineError("The preserved original does not match this area's source model")
            trainer = GsplatTrainer(seed, backup / "current.ckpt", published_ply=backup / "current.ply")
            if trainer.has_pending_work:
                raise OnlineError("Finish or explicitly reject pending model refinement before starting an area draft")
            initial_rows = len(trainer.params["means"])
            if prepared:
                views, units, distortion, references = prepared
                trainer.configure_geometry_scale(units)
                trainer.configure_camera_distortion(distortion)
                # A legacy cache may require rehydration after adopting measured
                # distortion. begin_area_replacement then clears old images so
                # only explicitly selected replacement photos can reseed it.
                trainer.prime_baseline_views(references)
                area_review = _prepare_area_review(trainer, references, record, captures, units)
                if area_review["status"] != "prepared":
                    record.update(status="quality_rejected", quality_review={
                        **area_review, "method": "fixed_area_seed_heldout_v1",
                        "source_sha256": record["source_sha256"],
                        "selection_capture_ids": list(captures), "review_capture_ids": []},
                        message=area_review["reason"])
                    return _write(folder, record)
            if action == "protect":
                record["replacement_capture_ids"] = []
                record.pop("replacement_registration", None)
                record.pop("appended_gaussians", None)
                record.pop("replacement_gaussians", None)
                record.pop("processed_photos", None)
                trainer.protect_existing_area(record)
                record["clear_metrics"] = {"removed_gaussians": 0, "preserved_boundary_gaussians": 0}
            else:
                record["clear_metrics"] = trainer.begin_area_replacement(record, captures)
            if action == "build":
                record["replacement_registration"] = {
                    "method": "measured_seed_and_sparse_pnp", "seed_sha256": source["seed_sha256"],
                    "evidence_job_id": source["registration"]["evidence_job_id"],
                    "source_proof": source["registration"]["source_proof"],
                    "model_units_per_meter": units, "distortion_coefficients": distortion,
                    "captures": {view["capture_id"]: {key: view[key] for key in
                        ("image_sha256", "metadata_sha256", "proof")} for view in views}}
                for index, view in enumerate(views, 1):
                    trainer.add_image(view["capture_id"], view["image_path"], view["pose"], view["intrinsics"])
                    trainer.train_steps(12)
                    record.update(processed_photos=index, message=f"Replacement photos: {index} / {len(captures)}")
                    _write(folder, record)
                if len(trainer.params["means"]) <= initial_rows:
                    raise OnlineError("These photos produced no supported replacement geometry inside the box; keep the current model and choose better overlapping views")
                trainer.train_steps(24)
                _check_replacement_inputs(store, record, views)
                _source(root, record)
        temp_ckpt, temp_ply = folder / "candidate.next.ckpt", folder / "candidate.next.ply"
        record["model_metrics"] = trainer.save(temp_ckpt, temp_ply)
        record["accepted_ids"] = trainer.capture_ids
        count, digest, size = validate_gaussian_ply(temp_ply)
        if action == "protect":
            # A lock-only operation has a stronger gate than image scoring:
            # every original PLY byte and photo identity must be unchanged.
            if digest != record["source_sha256"] or trainer.capture_ids != source.get("accepted_ids", []):
                raise OnlineError("Protecting an area changed the source model or photo lineage")
            _source(root, record)
            record["quality_review"] = {"status": "ready_for_visual_review",
                "method": "unchanged_geometry_protection_v1", "source_sha256": record["source_sha256"],
                "candidate_sha256": digest, "candidate_checkpoint_sha256": _hash(temp_ckpt),
                "selection_capture_ids": [], "removed_gaussians": 0, "appended_gaussians": 0}
        previous_artifact = record.get("artifact") if action == "lock" else None
        if action == "lock" and (not previous_artifact or digest != previous_artifact["sha256"]):
            raise OnlineError("Locking changed the reviewed area geometry; rebuild and review the draft")
        if action == "build":
            if count <= initial_rows:
                raise OnlineError("The area draft did not export its supported replacement geometry")
            # Cleared rows remain invisible tombstones, so this is appended
            # row count, not net visible geometry or a quality result.
            record["appended_gaussians"] = count - initial_rows
            record["replacement_gaussians"] = record["appended_gaussians"]  # legacy UI field
            _check_replacement_inputs(store, record, views, source=source, seed=seed)
            _source(root, record)
            review = _finish_area_review(trainer, area_review, captures)
            # Rendering can take time. Revalidate after it so the quality
            # receipt never binds changed originals or seed calibration.
            _check_replacement_inputs(store, record, views, source=source, seed=seed)
            _source(root, record)
            record["quality_review"] = {**review, "method": "fixed_area_seed_heldout_v1",
                "source_sha256": record["source_sha256"], "candidate_sha256": digest,
                "candidate_checkpoint_sha256": _hash(temp_ckpt),
                "selection_capture_ids": list(captures),
                "review_capture_ids": [view["capture_id"] for view in area_review["views"]]}
        temp_ckpt.replace(folder / "candidate.ckpt")
        temp_ply.replace(folder / "candidate.ply")
        if action == "lock" and record.get("quality_review"):
            # Locking changes optimizer policy in the checkpoint, but must not
            # change any reviewed PLY bytes or source photo selection.
            record["quality_review"]["candidate_checkpoint_sha256"] = _hash(folder / "candidate.ckpt")
        accepted = action != "build" or record["quality_review"]["status"] == "ready_for_visual_review"
        record.update(artifact={"sha256": digest, "size_bytes": size, "vertex_count": count},
                      checkpoint_sha256=_hash(folder / "candidate.ckpt"),
                      status="locked" if action in {"lock", "protect"} else "draft_ready" if accepted else "quality_rejected",
                      message="Accepted area protected without changing any model bytes." if action == "protect" else
                              "Draft locked; review before applying." if action == "lock" else
                              "Draft ready for visual review. The original model is still displayed." if accepted else
                              "Private draft failed independent camera review; original model is still displayed.")
        return _write(folder, record)


def _merged_area_provenance(root: Path, record: dict, state: dict) -> dict:
    """Bind reviewed replacement views to the unchanged source before apply."""
    from copy import deepcopy
    from .store import Store
    from .online import _publication_ledger, validated_camera, verified_registration
    from .online_torch import GsplatTrainer
    merged = deepcopy(state.get("photo_registration", {}))
    accepted = record["accepted_ids"]
    old_ids = set(state.get("accepted_ids", []))
    no_photo_edit = record.get("draft_action") in {"clear", "protect"}
    selected = [] if no_photo_edit else record["replacement_capture_ids"]
    seed_ids = set(state.get("seed_ids", []))
    if (len(set(accepted)) != len(accepted) or not old_ids <= set(accepted) or
            (no_photo_edit and accepted != state.get("accepted_ids", [])) or
            not set(selected) <= set(accepted) or
            set(accepted)-old_ids-seed_ids-set(selected)):
        raise OnlineError("Area checkpoint photo lineage differs from its reviewed source")
    if selected:
        store = Store(root)
        _folder, source, seed = _source(root, record)
        views, units, distortion, _references = _replacement_views(store, record, source, seed)
        proof = record.get("replacement_registration", {})
        expected = {"method": "measured_seed_and_sparse_pnp", "seed_sha256": state["seed_sha256"],
            "evidence_job_id": state["registration"]["evidence_job_id"],
            "source_proof": state["registration"]["source_proof"],
            "model_units_per_meter": units, "distortion_coefficients": distortion}
        if (any(proof.get(key) != value for key, value in expected.items()) or
                set(proof.get("captures", {})) != set(selected)):
            raise OnlineError("Area replacement registration differs from its measured source")
        matrix, calibration, orientation = verified_registration(
            state['registration'], record['session_id'], state['seed_sha256'])
        metadata = {item['receipt']['capture_id']: item['metadata'] for item in
                    store.export_session(record['session_id'], capture_ids=selected)['captures']}
        for view in views:
            cid = view["capture_id"]
            frozen = proof["captures"][cid]
            if any(frozen.get(key) != view[key] for key in ("image_sha256", "metadata_sha256")):
                raise OnlineError("Area replacement camera or original differs from its reviewed proof")
            old_camera, measured = frozen.get('proof', {}), view['proof']
            if cid in seed_ids:
                if old_camera != measured:
                    raise OnlineError('Area baseline camera differs from measured seed evidence')
            else:
                # FLANN/RANSAC may pick different valid inlier subsets on an
                # identical source. Preserve the original candidate camera,
                # require strict fresh support and a tight camera agreement,
                # rather than compare randomized diagnostic dictionaries.
                if (any(old_camera.get(key) != measured.get(key) for key in
                        ('method', 'method_revision', 'source_proof')) or
                        old_camera.get('method') != 'seed_sparse_landmark_pnp'):
                    raise OnlineError('Area replacement localization source changed')
                pose, k = GsplatTrainer._validated_view(
                    old_camera['localized_pose'], old_camera['camera_intrinsics'])
                tracked, _k = validated_camera(metadata[cid], matrix, calibration, orientation)
                def residual(other):
                    position = float(np.linalg.norm(pose[:3, 3]-other[:3, 3])/units)
                    angle = float(np.degrees(np.arccos(np.clip(
                        (np.trace(pose[:3, :3].T@other[:3, :3])-1)/2, -1, 1))))
                    return position, angle
                position, angle = residual(tracked)
                repeat_position, repeat_angle = residual(view['pose'])
                errors = [old_camera['reprojection_rmse_px'], old_camera['reprojection_max_px']]
                inliers, total = old_camera.get('inliers'), old_camera.get('correspondences')
                if (not np.isfinite([position, angle, repeat_position, repeat_angle, *errors]).all() or
                        position > .10 or angle > 5. or repeat_position > .01 or repeat_angle > .25 or
                        type(inliers) is not int or type(total) is not int or
                        inliers < 8 or total < inliers or inliers < .5*total or
                        min(errors) < 0 or max(errors) > 2.5 or
                        not np.allclose(k, view['intrinsics'], rtol=0., atol=1e-8) or
                        not np.array_equal(old_camera.get('distortion_coefficients'), distortion)):
                    raise OnlineError('Area replacement camera differs from its reviewed localization')
            old = merged.get(cid)
            if old and any(old.get(key) != frozen[key] for key in ("image_sha256", "metadata_sha256")):
                raise OnlineError("Area replacement conflicts with existing original-photo provenance")
            merged[cid] = {**deepcopy(frozen["proof"]), "image_sha256": frozen["image_sha256"],
                "metadata_sha256": frozen["metadata_sha256"], "seed_sha256": state["seed_sha256"]}
        _check_replacement_inputs(store, record, views, source=source, seed=seed)
    if state.get("registration"):
        _publication_ledger({**state, "accepted_ids": accepted, "photo_registration": merged},
                            state["registration"], Store(root))
    return merged


def _apply_candidate(root: Path, project_id: str, region_id: str, *, restore: bool) -> dict:
    from .reconstruct import _worker_lock
    from .online import (_atomic_json, _replace_with_retry, stage_display_revision,
                         update_photo_counts, utc_now)
    folder, record = _record(root, project_id, region_id)
    if record["status"] != ("published" if restore else "locked"):
        raise OnlineError("Review and lock the draft before applying; only an applied area can be restored")
    with _worker_lock(root):
        online = root / "projects" / project_id / "online" / str(uuid.UUID(record["session_id"]))
        state = json.loads((online / "status.json").read_text(encoding="utf-8"))
        expected = record["artifact"]["sha256"] if restore else record["source_sha256"]
        expected_checkpoint = record["checkpoint_sha256"] if restore else record["source_checkpoint_sha256"]
        if (state["artifact"]["sha256"] != expected or _hash(online / "current.ply") != expected or
                _hash(online / "current.ckpt") != expected_checkpoint):
            raise OnlineError("Newer model work exists; refusing to overwrite it with this area edit")
        if restore:
            source = folder / "original"
            replacement = json.loads((source / "status.json").read_text(encoding="utf-8"))
            files = {name: source / name for name in ("current.ckpt", "current.ply", "display.ply")}
            if (_hash(files["current.ply"]) != record["source_sha256"] or
                    _hash(files["current.ckpt"]) != record["source_checkpoint_sha256"] or
                    _hash(files["display.ply"]) != replacement["display_artifact"]["sha256"]):
                raise OnlineError("The preserved original failed verification")
        else:
            verified_region_artifact(root, project_id, region_id)
            if _hash(folder / "candidate.ckpt") != record["checkpoint_sha256"]:
                raise OnlineError("The candidate checkpoint failed verification")
            review = record.get("quality_review", {})
            protection = record.get("draft_action") == "protect"
            expected_review = "unchanged_geometry_protection_v1" if protection else "fixed_area_seed_heldout_v1"
            if protection and (record["artifact"]["sha256"] != record["source_sha256"] or
                    record.get("replacement_capture_ids") or review.get("removed_gaussians") != 0 or
                    review.get("appended_gaussians") != 0):
                raise OnlineError("Area protection must preserve every source model byte")
            if record.get("draft_action") != "clear" and (review.get("status") != "ready_for_visual_review" or
                    review.get("method") != expected_review or
                    review.get("source_sha256") != record["source_sha256"] or
                    review.get("candidate_sha256") != record["artifact"]["sha256"] or
                    review.get("candidate_checkpoint_sha256") != record["checkpoint_sha256"] or
                    review.get("selection_capture_ids") != record["replacement_capture_ids"]):
                raise OnlineError("The area candidate lacks its matching independent quality review")
            files = {"current.ckpt": folder / "candidate.ckpt", "current.ply": folder / "candidate.ply",
                     "display.ply": folder / "candidate.ply"}
            replacement = dict(state)
            replacement["artifact"] = dict(record["artifact"])
            replacement["last_area_edit_id"] = region_id
            replacement["accepted_ids"] = record["accepted_ids"]
            replacement["photo_registration"] = _merged_area_provenance(root, record, state)
            replacement["model_metrics"] = record["model_metrics"]
            replacement["training_steps"] = record["model_metrics"]["training_steps"]
            metrics = record["model_metrics"]
            update_photo_counts(replacement, state.get("seed_ids", []),
                metrics.get("published_capture_ids", state.get("published_capture_ids", [])),
                metrics.get("rejected_capture_ids", state.get("rejected_capture_ids", [])),
                metrics.get("pending_refinement", {}))
            if replacement["pending_count"]:
                raise OnlineError("Finish pending candidate work before applying the reviewed area")
        replacement["revision_number"] = state.get("revision_number", 0) + int(
            replacement["artifact"]["sha256"] != state["artifact"]["sha256"])
        original_status = (online / "status.json").read_bytes()
        original_record = (folder / "region.json").read_bytes()
        def check_source():
            if ((online / "status.json").read_bytes() != original_status or
                    (folder / "region.json").read_bytes() != original_record or
                    _hash(online / "current.ply") != expected or
                    _hash(online / "current.ckpt") != expected_checkpoint):
                raise OnlineError("The online model changed during area display preparation")
            if (_hash(files["current.ply"]) != replacement["artifact"]["sha256"] or
                    _hash(files["current.ckpt"]) != (record["source_checkpoint_sha256"] if restore
                                                   else record["checkpoint_sha256"])):
                raise OnlineError("The reviewed area files changed during display preparation")
            if not restore:
                _merged_area_provenance(root, record, state)
        replacement["display_artifact"] = stage_display_revision(
            files["current.ply"], online, project_id, record["session_id"],
            replacement["revision_number"], before_commit=check_source,
            prepared_display=files["display.ply"] if restore else None)
        files["display.ply"] = online / ".display.tmp.ply"
        # Stage every file before replacing any published artifact. Originals
        # and the candidate remain available to recover an interrupted apply.
        for name, source in files.items():
            shutil.copy2(online / name, online / (".area-preimage-" + name))
            shutil.copy2(source, online / (".area-" + name))
        check_source()
        rollback_complete = False
        try:
            for name in ("current.ckpt", "current.ply", "display.ply"):
                _replace_with_retry(online / (".area-" + name), online / name)
            replacement["status"] = (replacement["status"] if record.get("draft_action") == "protect"
                                     else "paused_quality_review")
            replacement["updated_utc"] = utc_now()
            _atomic_json(online / "status.json", replacement)
            record.update(status="restored" if restore else "published",
                          message="Original model restored." if restore else
                                  "Area protected without changing the scan. Automatic updates keep their previous state." if record.get("draft_action") == "protect" else
                                  "Reviewed locked area applied. Automatic refinement remains paused.")
            result = _write(folder, record)
            rollback_complete = True
        except Exception:
            rollback = {name: online / (".area-preimage-" + name)
                        for name in ("current.ckpt", "current.ply", "display.ply")}
            for name, source in rollback.items():
                shutil.copy2(source, online / (".area-rollback-" + name))
                _replace_with_retry(online / (".area-rollback-" + name), online / name)
            status_temp = online/'.area-rollback-status.json'
            status_temp.write_bytes(original_status)
            _replace_with_retry(status_temp, online/'status.json')
            if (folder/'region.json').read_bytes() != original_record:
                region_temp = online/'.area-rollback-region.json'
                region_temp.write_bytes(original_record)
                _replace_with_retry(region_temp, folder/'region.json')
            rollback_complete = True
            raise
        finally:
            if rollback_complete:
                for name in ("current.ckpt", "current.ply", "display.ply"):
                    (online / (".area-preimage-" + name)).unlink(missing_ok=True)
        return result


def main():
    parser = argparse.ArgumentParser(description="Build an isolated area replacement through the existing Gaussian trainer")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--action", choices=("build", "clear", "lock", "protect"), required=True)
    args = parser.parse_args()
    folder, record = _record(args.data_dir, args.project_id, args.region_id)
    try:
        run_region_action(args.data_dir, args.project_id, args.region_id, args.action)
        if args.action == "protect":
            # The Protect button explicitly authorizes this metadata-only lock.
            # Apply revalidates the source again after releasing the GPU lock.
            _apply_candidate(args.data_dir, args.project_id, args.region_id, restore=False)
    except Exception as error:
        # The worker writes progress and may clear a superseded artifact
        # receipt before it fails. Reload that verified state rather than
        # overwriting it with the record read before work began.
        folder, record = _record(args.data_dir, args.project_id, args.region_id)
        record.update(status="failed", message=str(error)[:300] if isinstance(error, OnlineError) else
                      "Area processing failed; the original model is unchanged. See the local worker log.")
        _write(folder, record)
        raise
    finally:
        (folder / "action.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
