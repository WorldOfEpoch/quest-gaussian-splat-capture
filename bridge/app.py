from __future__ import annotations

import hashlib
import hmac
import ipaddress
import io
import json
import os
import secrets
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .config import Config, load_config
from .blender_launcher import (BlenderLaunchError, open_current_splat_in_blender,
                               open_saved_splat_in_blender)
from .dashboard import (capture_gallery, dashboard_current_model, dashboard_jobs, render_capture_batch,
                        render_dashboard, set_capture_selection, verified_capture_image, verified_job)
from .jobs import (Scheduler, on_capture_received, policy, processing_paused, request_build,
                   request_status, set_policy, depth_status, set_depth_updates, splat_status, set_splat_updates)
from .library import MAX_MODEL_BYTES, ModelMetadata, list_models, model_artifact, publish_model
from .models import CaptureMetadata, EntryCreate, EntryPatch, NameBody, ProjectCreate, SessionCreate
from .online import OnlineError
from .preview import preview_manifest, read_job, read_online_preview, read_quest_preview, read_depth_preview
from .quest_device import QuestDeviceError, perform_device_action
from .region_edit import (create_region, list_regions, request_region_action,
                          update_region_photos, verified_region_artifact)
from .reconstruct import ReconstructionError, read_house_build
from .store import MAX_SPATIAL_BYTES, Store, StoreError
from .runtime import data_root_id

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def create_app(config: Config | None = None, dashboard_enabled: bool = False) -> FastAPI:
    config = config or load_config()
    store = Store(config.data_dir)
    scheduler = Scheduler(config.data_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if dashboard_enabled:
            scheduler.start()
        try:
            yield
        finally:
            if dashboard_enabled:
                scheduler.stop()

    app = FastAPI(title="Quest Splat Bridge", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.scheduler_error = None
    dashboard_token = secrets.token_urlsafe(32)

    @app.exception_handler(StoreError)
    async def store_error(_request: Request, error: StoreError):
        return JSONResponse(status_code=error.status, content={"detail": error.detail})

    def authenticate(request: Request):
        supplied = request.headers.get("Authorization", "")
        if not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:], config.token):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    def local_dashboard_only(request: Request):
        if not dashboard_enabled:
            raise HTTPException(status_code=404, detail="Dashboard is not enabled on this listener")
        try:
            remote_is_local = ipaddress.ip_address(request.client.host).is_loopback
        except (ValueError, AttributeError):
            remote_is_local = False
        if not remote_is_local or request.url.hostname not in {"127.0.0.1", "localhost"}:
            raise HTTPException(status_code=403, detail="PC dashboard is loopback-only")

    @app.get("/", include_in_schema=False)
    def dashboard_root(request: Request):
        local_dashboard_only(request)
        return RedirectResponse("/ui", status_code=303)

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request, project_id: uuid.UUID | None = None,
                  headset_notice: str = Query(default="", max_length=300)):
        local_dashboard_only(request)
        projects = store.projects()
        selected = store.project(str(project_id)) if project_id else (projects[0] if projects else None)
        entries = store.entries(selected["id"]) if selected else []
        sessions = store.sessions(selected["id"]) if selected else []
        jobs = dashboard_jobs(config.data_dir, selected["id"]) if selected else []
        models = list_models(store, selected["id"]) if selected else []
        recent_captures = capture_gallery(store, selected["id"], page_size=6)[0] if selected else []
        policies = {session["id"]: policy(config.data_dir, session["id"]) for session in sessions}
        queue = request_status(config.data_dir, selected["id"]) if selected else []
        current_model = dashboard_current_model(config.data_dir, selected["id"]) if selected else None
        house_builds = []
        if selected:
            for manifest in (config.data_dir / "projects" / selected["id"] / "house-builds").glob("*/house.json"):
                try:
                    delivery, _ = read_house_build(store, selected["id"], manifest.parent.name)
                    house_builds.append(delivery)
                except (OSError, ValueError, KeyError, TypeError, ReconstructionError, StoreError):
                    continue
        response = HTMLResponse(render_dashboard(projects, selected, entries, sessions, jobs,
                                                 dashboard_token, policies, queue, models, recent_captures,
                                                 current_model, headset_notice,
                                                 builds_paused=processing_paused(config.data_dir), house_builds=house_builds,
                                                 depth_updates={s['id']: depth_status(config.data_dir, s['id']) for s in sessions},
                                                 splat_updates={s['id']: splat_status(config.data_dir, s['id']) for s in sessions}))
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'unsafe-inline'; "
            "connect-src 'self'; worker-src 'self' blob:; img-src 'self' blob: data:; "
            "form-action 'self'; base-uri 'none'"
        )
        return response

    @app.post("/ui/headset/{action}", include_in_schema=False)
    def dashboard_headset_action(request: Request, action: str, csrf: str = Form(...),
                                  project_id: uuid.UUID | None = Form(None)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        if action not in {"wake", "keep-awake", "wear-sensor"}:
            raise HTTPException(status_code=422, detail="Unsupported headset action")
        try:
            notice = perform_device_action(config.data_dir, action)["message"]
        except QuestDeviceError as error:
            notice = str(error)
        query = {"headset_notice": notice}
        if project_id:
            query["project_id"] = str(project_id)
        return RedirectResponse("/ui?" + urlencode(query) + "#headset-controls", status_code=303)

    @app.get("/ui/projects/{project_id}/sessions/{session_id}/captures", include_in_schema=False)
    def dashboard_capture_batch(request: Request, project_id: uuid.UUID, session_id: uuid.UUID,
                                page: int = Query(default=1, ge=1, le=100000)):
        local_dashboard_only(request)
        project = store.project(str(project_id))
        session = store.session(str(session_id))
        records, total = capture_gallery(store, str(project_id), str(session_id), page)
        response = HTMLResponse(render_capture_batch(project, session, records, total, page, dashboard_token))
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; form-action 'self'; base-uri 'none'"
        )
        return response

    @app.post("/ui/projects/{project_id}/sessions/{session_id}/captures/{capture_id}/selection",
              include_in_schema=False)
    def dashboard_capture_selection(request: Request, project_id: uuid.UUID, session_id: uuid.UUID,
                                    capture_id: uuid.UUID, csrf: str = Form(...),
                                    included: str = Form(...), page: int = Form(1)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        if included not in {"yes", "no"} or page < 1 or page > 100000:
            raise HTTPException(status_code=422, detail="Invalid photo selection")
        set_capture_selection(store, str(project_id), str(session_id), str(capture_id), included == "yes")
        return RedirectResponse(
            f"/ui/projects/{project_id}/sessions/{session_id}/captures?page={page}#photo-{capture_id}",
            status_code=303)

    @app.get("/ui/projects/{project_id}/captures/{capture_id}/image", include_in_schema=False)
    def dashboard_capture_image(request: Request, project_id: uuid.UUID, capture_id: uuid.UUID,
                                thumbnail: bool = False):
        local_dashboard_only(request)
        path, media_type = verified_capture_image(store, str(project_id), str(capture_id))
        if thumbnail:
            try:
                with Image.open(path) as original:
                    thumb = original.convert("RGB")
                    thumb.thumbnail((320, 240), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    thumb.save(output, format="JPEG", quality=85)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                raise HTTPException(status_code=404, detail="Original photo cannot be decoded") from error
            response = Response(output.getvalue(), media_type="image/jpeg")
        else:
            response = FileResponse(path, media_type=media_type)
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    static_root = Path(__file__).parent / "static"
    static_files = {"viewer.js", "vendor/gaussian-splats-3d.module.js", "vendor/three.module.js"}

    @app.get("/ui/static/{asset_path:path}", include_in_schema=False)
    def dashboard_static(request: Request, asset_path: str):
        local_dashboard_only(request)
        if asset_path not in static_files:
            raise HTTPException(status_code=404, detail="Static asset not found")
        response = FileResponse(static_root / asset_path, media_type="text/javascript")
        response.headers["Cache-Control"] = "public, max-age=3600"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/ui/jobs", include_in_schema=False)
    def dashboard_list_jobs(request: Request, project_id: uuid.UUID):
        local_dashboard_only(request)
        store.project(str(project_id))
        response = JSONResponse({"jobs": dashboard_jobs(config.data_dir, str(project_id))})
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/ui/current-model", include_in_schema=False)
    def dashboard_current(request: Request, project_id: uuid.UUID):
        local_dashboard_only(request)
        store.project(str(project_id))
        response = JSONResponse({"model": dashboard_current_model(config.data_dir, str(project_id))})
        response.headers["Cache-Control"] = "no-store"
        return response

    def public_region(record: dict) -> dict:
        # The UI needs model-space data, never local checkpoint or photo paths.
        keys = ("id", "name", "min", "max", "status", "message", "replacement_capture_ids",
                "source_model_id", "source_sha256", "coordinate_frame", "session_id", "view")
        result = {key: record[key] for key in keys if key in record}
        artifact = record.get("artifact")
        if isinstance(artifact, dict):
            result["artifact"] = {key: artifact[key] for key in ("sha256", "size_bytes", "vertex_count")
                                  if key in artifact}
        return result

    @app.get("/ui/projects/{project_id}/regions", include_in_schema=False)
    def dashboard_regions(request: Request, project_id: uuid.UUID):
        local_dashboard_only(request)
        store.project(str(project_id))
        response = JSONResponse({"regions": [public_region(item) for item in list_regions(config.data_dir, str(project_id))]})
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/ui/projects/{project_id}/region-photos", include_in_schema=False)
    def dashboard_region_photos(request: Request, project_id: uuid.UUID, session_id: uuid.UUID,
                                page: int = Query(default=1, ge=1)):
        local_dashboard_only(request)
        photos, count = capture_gallery(store, str(project_id), str(session_id), page=page)
        response = JSONResponse({"photos": photos, "count": count, "page": page, "page_size": 24})
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/ui/projects/{project_id}/regions", include_in_schema=False)
    def dashboard_create_region(request: Request, project_id: uuid.UUID,
                                csrf: str = Form(...), plan: str = Form(..., max_length=60000)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        store.project(str(project_id))
        try:
            proposal = json.loads(plan)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail="Invalid area selection") from error
        if not isinstance(proposal, dict):
            raise HTTPException(status_code=422, detail="Invalid area selection")
        current = dashboard_current_model(config.data_dir, str(project_id))
        if not current or current.get("source") != "online":
            raise HTTPException(status_code=409, detail="Area editing needs a saved online checkpoint for this project")
        if any(proposal.get(key) != current.get(source_key) for key, source_key in (
                ("source_model_id", "id"), ("source_sha256", "source_sha256"),
                ("coordinate_frame", "coordinate_frame"), ("session_id", "session_id"))):
            raise HTTPException(status_code=409, detail="The current model changed. Place the box again before saving")
        capture_ids = proposal.get("replacement_capture_ids", [])
        if not isinstance(capture_ids, list) or len(capture_ids) > 128:
            raise HTTPException(status_code=422, detail="Select up to 128 replacement photos")
        try:
            capture_ids = list(dict.fromkeys(str(uuid.UUID(item)) for item in capture_ids))
        except (ValueError, TypeError, AttributeError) as error:
            raise HTTPException(status_code=422, detail="Invalid replacement photo selection") from error
        # The shared gallery owns include/exclude state; materialize that table
        # through it before verifying all chosen photos in a single read.
        capture_gallery(store, str(project_id), current["session_id"], page_size=1)
        with store.connection() as db:
            allowed = {row[0] for row in db.execute("""SELECT c.capture_id FROM captures c
                LEFT JOIN capture_selections s ON s.capture_id=c.capture_id
                WHERE c.project_id=? AND c.session_id=? AND COALESCE(s.included,1)=1""",
                (str(project_id), current["session_id"]))}
        if not set(capture_ids).issubset(allowed):
            raise HTTPException(status_code=422, detail="Replacement photos must be included photos from this model's session")
        clean_plan = {key: proposal.get(key) for key in (
            "name", "min", "max", "source_model_id", "source_sha256", "coordinate_frame", "session_id")}
        clean_plan.update(replacement_capture_ids=capture_ids, view=current.get("view"))
        try:
            record = create_region(config.data_dir, str(project_id), clean_plan)
        except (OnlineError, ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error) if isinstance(error, (OnlineError, ValueError))
                                else "The area draft could not be saved") from error
        return JSONResponse(public_region(record), status_code=201, headers={"Cache-Control": "no-store"})

    @app.post("/ui/projects/{project_id}/regions/{region_id}/{action}", include_in_schema=False)
    def dashboard_region_action(request: Request, project_id: uuid.UUID, region_id: uuid.UUID,
                                action: str, csrf: str = Form(...), reviewed: str | None = Form(None),
                                capture_ids: str = Form(default="[]", max_length=6000)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        store.project(str(project_id))
        if action not in {"build", "clear", "lock", "protect", "publish", "restore", "set-photos"}:
            raise HTTPException(status_code=404, detail="Unknown area action")
        if action == "publish" and reviewed != "yes":
            raise HTTPException(status_code=422, detail="Inspect and approve the locked draft before applying it")
        try:
            if action == "set-photos":
                record = update_region_photos(config.data_dir, str(project_id), str(region_id), json.loads(capture_ids))
            else:
                record = request_region_action(config.data_dir, str(project_id), str(region_id), action)
        except (OnlineError, ValueError, OSError) as error:
            raise HTTPException(status_code=409, detail=str(error) if isinstance(error, (OnlineError, ValueError))
                                else "The area action could not be completed") from error
        return JSONResponse(public_region(record), headers={"Cache-Control": "no-store"})

    @app.get("/ui/projects/{project_id}/regions/{region_id}/draft.ply", include_in_schema=False)
    def dashboard_region_splat(request: Request, project_id: uuid.UUID, region_id: uuid.UUID,
                               revision: str = Query(..., pattern="^[0-9a-f]{64}$")):
        local_dashboard_only(request)
        store.project(str(project_id))
        try:
            record, path = verified_region_artifact(config.data_dir, str(project_id), str(region_id))
        except (OnlineError, ValueError, OSError) as error:
            raise HTTPException(status_code=404, detail="Verified area draft is unavailable") from error
        if record.get("artifact", {}).get("sha256") != revision:
            raise HTTPException(status_code=409, detail="This draft changed; reload it before reviewing")
        # Read and verify these exact bytes so a concurrently replaced draft
        # cannot be served under the earlier review revision.
        if path.stat().st_size > 64 * 1024 * 1024:
            raise HTTPException(status_code=409, detail="This draft is too large for the browser viewer")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != revision:
            raise HTTPException(status_code=409, detail="This draft changed; reload it before reviewing")
        response = Response(content, media_type="application/octet-stream")
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.post("/ui/projects/{project_id}/open-current-in-blender", include_in_schema=False)
    def dashboard_open_current_in_blender(request: Request, project_id: uuid.UUID,
                                          csrf: str = Form(...)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        store.project(str(project_id))
        try:
            open_current_splat_in_blender(config.data_dir, str(project_id))
        except BlenderLaunchError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return RedirectResponse(f"/ui?project_id={project_id}", status_code=303)

    @app.post("/ui/projects/{project_id}/jobs/{job_id}/open-in-blender", include_in_schema=False)
    def dashboard_open_saved_in_blender(request: Request, project_id: uuid.UUID,
                                        job_id: uuid.UUID, csrf: str = Form(...)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        store.project(str(project_id))
        try:
            open_saved_splat_in_blender(config.data_dir, str(project_id), str(job_id))
        except BlenderLaunchError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return RedirectResponse(f"/ui?project_id={project_id}", status_code=303)

    @app.get("/ui/online/{project_id}/{session_id}/current.ply", include_in_schema=False)
    def dashboard_online_splat(request: Request, project_id: uuid.UUID, session_id: uuid.UUID,
                               revision: str = Query(..., pattern="^[0-9a-f]{64}$")):
        local_dashboard_only(request)
        session = store.session(str(session_id))
        if session["project_id"] != str(project_id):
            raise HTTPException(status_code=404, detail="Online model belongs to another project")
        folder = config.data_dir / "projects" / str(project_id) / "online" / str(session_id)
        ply = folder / "display-revisions" / (revision + ".ply")
        full = folder / 'model-revisions' / (revision + '.ply')
        full_model = full.is_file()
        if full_model:
            receipt_path = full.with_suffix('.json')
            if full.is_symlink() or receipt_path.is_symlink() or not receipt_path.is_file():
                raise HTTPException(status_code=404, detail='Full model revision is invalid')
            try:
                receipt = json.loads(receipt_path.read_text())
            except (ValueError, OSError):
                raise HTTPException(status_code=404, detail='Full model receipt is invalid')
            if (receipt.get('project_id') != str(project_id) or receipt.get('session_id') != str(session_id)
                    or receipt.get('sha256') != revision or receipt.get('source_sha256') != revision
                    or receipt.get('size_bytes') != full.stat().st_size):
                raise HTTPException(status_code=404, detail='Full model identity differs')
            ply = full
        if not ply.is_file():
            # Older workers published only one mutable file. Keep that path
            # readable until they restart and create immutable display copies.
            model = dashboard_current_model(config.data_dir, str(project_id))
            if (not model or model["source"] != "online" or model["session_id"] != str(session_id)
                    or not model["id"].endswith(":" + revision)):
                raise HTTPException(status_code=404, detail="Online revision not available")
            ply = folder / ("display.ply" if (folder / "display.ply").is_file() else "current.ply")
        if ply.is_symlink():
            raise HTTPException(status_code=404, detail="Online revision is invalid")
        content = ply.read_bytes()
        if len(content) > (512 if full_model else 64) * 1024 * 1024 or hashlib.sha256(content).hexdigest() != revision:
            raise HTTPException(status_code=404, detail="Online revision changed")
        response = Response(content, media_type="application/octet-stream")
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/ui/build-status", include_in_schema=False)
    def dashboard_build_status(request: Request, project_id: uuid.UUID):
        local_dashboard_only(request)
        store.project(str(project_id))
        response = JSONResponse({"requests": request_status(config.data_dir, str(project_id)),
                                 "jobs": dashboard_jobs(config.data_dir, str(project_id)),
                                 "processing_paused": processing_paused(config.data_dir)})
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/ui/jobs/{project_id}/{job_id}/splat.ply", include_in_schema=False)
    def dashboard_splat(request: Request, project_id: uuid.UUID, job_id: uuid.UUID):
        local_dashboard_only(request)
        found = verified_job(config.data_dir, str(project_id), str(job_id))
        if not found:
            raise HTTPException(status_code=404, detail="Verified splat not found")
        response = FileResponse(found[1], media_type="application/octet-stream", filename="splat.ply")
        response.headers["Cache-Control"] = "private, max-age=3600, immutable"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/ui/models/{project_id}/{revision_id}/artifact", include_in_schema=False)
    def dashboard_model_download(request: Request, project_id: uuid.UUID, revision_id: uuid.UUID):
        local_dashboard_only(request)
        record, path = model_artifact(store, str(project_id), str(revision_id))
        extension = "glb" if record["format"] == "glb" else "ply"
        media_type = "model/gltf-binary" if extension == "glb" else "application/octet-stream"
        response = FileResponse(path, media_type=media_type, filename=f"model-{revision_id}.{extension}")
        response.headers["Cache-Control"] = "private, max-age=3600, immutable"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.post("/ui/sessions/{session_id}/build", include_in_schema=False)
    def dashboard_build(request: Request, session_id: uuid.UUID, csrf: str = Form(...),
                        reviewed_native_orientation: str | None = Form(None), quality: str = Form(...)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        if quality not in {"quick", "final", "house"}:
            raise HTTPException(status_code=422, detail="Choose a valid build quality")
        session = store.session(str(session_id))
        current = policy(config.data_dir, str(session_id))
        reviewed = current["native_orientation_reviewed"] or reviewed_native_orientation == "yes"
        if not reviewed:
            raise HTTPException(status_code=422, detail="Review physical camera image orientation before building")
        set_policy(config.data_dir, str(session_id), auto_enabled=current["auto_enabled"],
                   native_orientation_reviewed=reviewed)
        try:
            result = request_build(config.data_dir, str(session_id), reason="manual", quality=quality)
        except ReconstructionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        if result["status"] == "need_more_views":
            raise HTTPException(status_code=409, detail=f"At least {result['minimum']} received images are needed")
        if quality == "house" and result["status"] == "paused":
            raise HTTPException(status_code=409, detail="Build queue is paused. Your photos remain saved.")
        return RedirectResponse(f"/ui?project_id={session['project_id']}", status_code=303)

    @app.get("/ui/projects/{project_id}/house-builds/{build_id}/mesh.glb", include_in_schema=False)
    def dashboard_house_mesh(request: Request, project_id: uuid.UUID, build_id: uuid.UUID):
        local_dashboard_only(request)
        try:
            _, artifact = read_house_build(store, str(project_id), str(build_id))
        except (OSError, ValueError, KeyError, TypeError, ReconstructionError, StoreError) as error:
            raise HTTPException(status_code=404, detail="Verified house mesh is unavailable") from error
        return FileResponse(artifact, media_type="model/gltf-binary", filename="House-Textured-Mesh.glb")

    @app.post("/ui/sessions/{session_id}/auto", include_in_schema=False)
    def dashboard_auto(request: Request, session_id: uuid.UUID, csrf: str = Form(...),
                       enabled: str = Form(...), reviewed_native_orientation: str | None = Form(None)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        if enabled not in {"yes", "no"}:
            raise HTTPException(status_code=422, detail="Invalid auto preview setting")
        session = store.session(str(session_id))
        current = policy(config.data_dir, str(session_id))
        reviewed = current["native_orientation_reviewed"] or reviewed_native_orientation == "yes"
        try:
            set_splat_updates(config.data_dir, str(session_id), enabled == "yes",
                              native_orientation_reviewed=reviewed)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return RedirectResponse(f"/ui?project_id={session['project_id']}", status_code=303)

    @app.post("/ui/projects", include_in_schema=False)
    def dashboard_create_project(request: Request, name: str = Form(...), csrf: str = Form(...)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail="Invalid dashboard form token")
        try:
            valid_name = NameBody(name=name).name
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="Invalid project name") from error
        project = store.create_project(valid_name)
        return RedirectResponse(f"/ui?project_id={project['id']}", status_code=303)

    @app.get("/health")
    def health():
        return {"status": "ok", "protocol_version": "0.1", "capabilities": ["projects", "entries", "sessions", "captures", "spatial_evidence", "gaussian_preview", "idempotent_creation"]}

    @app.get('/ui/projects/{project_id}/depth-updates', include_in_schema=False)
    def dashboard_depth_updates(request: Request, project_id: uuid.UUID):
        local_dashboard_only(request)
        sessions = store.sessions(str(project_id))
        result = {'sessions': [{**depth_status(config.data_dir, s['id']),
                                'total_views': s['capture_count']} for s in sessions]}
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})

    @app.post('/ui/sessions/{session_id}/depth', include_in_schema=False)
    def dashboard_depth_policy(request: Request, session_id: uuid.UUID,
                               csrf: str = Form(...), enabled: str = Form(...)):
        local_dashboard_only(request)
        if not hmac.compare_digest(csrf, dashboard_token):
            raise HTTPException(status_code=403, detail='Invalid dashboard form token')
        if enabled not in {'yes', 'no'}:
            raise HTTPException(status_code=422, detail='Choose start or pause')
        session = store.session(str(session_id))
        set_depth_updates(config.data_dir, session['id'], enabled == 'yes')
        return RedirectResponse('/ui?project_id='+session['project_id'], status_code=303)

    @app.get('/ui/projects/{project_id}/splat-updates', include_in_schema=False)
    def dashboard_splat_updates(request: Request, project_id: uuid.UUID):
        local_dashboard_only(request)
        result = {'sessions': [splat_status(config.data_dir, s['id'])
                               for s in store.sessions(str(project_id))]}
        return JSONResponse(result, headers={'Cache-Control': 'no-store'})

    @app.get("/api/v1/runtime", dependencies=[Depends(authenticate)])
    def runtime_status():
        return {"status": "ok", "protocol_version": "0.1",
                "data_root_id": data_root_id(config.data_dir),
                "dashboard_enabled": dashboard_enabled,
                "processing_paused": processing_paused(config.data_dir)}

    @app.post("/api/v1/projects", dependencies=[Depends(authenticate)])
    def create_project(body: ProjectCreate):
        return store.create_project(body.name, str(body.request_id) if body.request_id else None)

    @app.get("/api/v1/projects", dependencies=[Depends(authenticate)])
    def list_projects():
        return {"projects": store.projects()}

    @app.get("/api/v1/projects/{project_id}", dependencies=[Depends(authenticate)])
    def get_project(project_id: uuid.UUID):
        return store.project(str(project_id))

    @app.get("/api/v1/projects/{project_id}/entries", dependencies=[Depends(authenticate)])
    def list_entries(project_id: uuid.UUID):
        return {"entries": store.entries(str(project_id))}

    @app.post("/api/v1/projects/{project_id}/entries", dependencies=[Depends(authenticate)])
    def create_entry(project_id: uuid.UUID, body: EntryCreate):
        return store.create_entry(str(project_id), body.name, body.kind,
                                  str(body.parent_id) if body.parent_id else None, body.sort_order)

    @app.patch("/api/v1/projects/{project_id}/entries/{entry_id}", dependencies=[Depends(authenticate)])
    def patch_entry(project_id: uuid.UUID, entry_id: uuid.UUID, body: EntryPatch):
        changes = body.model_dump(exclude_unset=True)
        if "parent_id" in changes and changes["parent_id"] is not None:
            changes["parent_id"] = str(changes["parent_id"])
        return store.patch_entry(str(project_id), str(entry_id), changes)

    @app.get("/api/v1/projects/{project_id}/models", dependencies=[Depends(authenticate)])
    def project_models(project_id: uuid.UUID):
        return {"models": list_models(store, str(project_id))}

    @app.get("/api/v1/projects/{project_id}/entries/{entry_id}/models", dependencies=[Depends(authenticate)])
    def entry_models(project_id: uuid.UUID, entry_id: uuid.UUID):
        return {"models": list_models(store, str(project_id), str(entry_id))}

    @app.post("/api/v1/projects/{project_id}/entries/{entry_id}/models", dependencies=[Depends(authenticate)])
    async def publish_entry_model(project_id: uuid.UUID, entry_id: uuid.UUID,
                                  metadata_json: str = Form(...), model: UploadFile = File(...)):
        if len(metadata_json) > 16384:
            raise HTTPException(status_code=413, detail="Model metadata exceeds 16 KiB")
        try:
            metadata = ModelMetadata.model_validate_json(metadata_json)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_context=False,
                                                                      include_input=False)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        staging = config.data_dir / "model-upload-staging"
        staging.mkdir(parents=True, exist_ok=True)
        temporary = staging / f"{uuid.uuid4()}.upload"
        total = 0
        try:
            with temporary.open("xb") as output:
                while chunk := await model.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_MODEL_BYTES:
                        raise HTTPException(status_code=413, detail="Model exceeds 512 MiB")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            return publish_model(store, str(project_id), str(entry_id), metadata, temporary)
        finally:
            await model.close()
            temporary.unlink(missing_ok=True)

    @app.get("/api/v1/projects/{project_id}/models/{revision_id}/artifact", dependencies=[Depends(authenticate)])
    def download_model(project_id: uuid.UUID, revision_id: uuid.UUID):
        record, path = model_artifact(store, str(project_id), str(revision_id))
        media_type = "model/gltf-binary" if record["format"] == "glb" else "application/octet-stream"
        response = FileResponse(path, media_type=media_type)
        response.headers["ETag"] = '"' + record["artifact_sha256"] + '"'
        response.headers["Cache-Control"] = "private, max-age=3600, immutable"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.post("/api/v1/sessions", dependencies=[Depends(authenticate)])
    def create_session(body: SessionCreate):
        return store.create_session(str(body.project_id), str(body.tracking_origin_id),
                                    str(body.request_id) if body.request_id else None)

    @app.get("/api/v1/projects/{project_id}/sessions", dependencies=[Depends(authenticate)])
    def list_sessions(project_id: uuid.UUID):
        return {"sessions": store.sessions(str(project_id))}

    @app.get("/api/v1/sessions/{session_id}", dependencies=[Depends(authenticate)])
    def get_session(session_id: uuid.UUID):
        return store.session(str(session_id))

    @app.post("/api/v1/sessions/{session_id}/finish", dependencies=[Depends(authenticate)])
    def finish_session(session_id: uuid.UUID):
        return store.finish_session(str(session_id))

    @app.post("/api/v1/sessions/{session_id}/captures", dependencies=[Depends(authenticate)])
    async def upload_capture(session_id: uuid.UUID, metadata_json: str = Form(...),
                             image: UploadFile = File(...),
                             spatial_json: UploadFile | None = File(None)):
        try:
            metadata = CaptureMetadata.model_validate_json(metadata_json)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_context=False)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        data = await image.read(MAX_IMAGE_BYTES + 1)
        await image.close()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds 20 MiB")
        if hashlib.sha256(data).hexdigest() != metadata.image_sha256:
            raise HTTPException(status_code=422, detail="Image SHA-256 mismatch")
        try:
            with Image.open(io.BytesIO(data)) as decoded:
                expected_format = "JPEG" if metadata.image_encoding == "jpeg" else "PNG"
                if decoded.format != expected_format or decoded.size != (metadata.image_width, metadata.image_height):
                    raise HTTPException(status_code=422, detail="Image format or dimensions mismatch")
                decoded.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
            raise HTTPException(status_code=422, detail="Invalid image") from error
        spatial_bytes = None
        if spatial_json is not None:
            spatial_bytes = await spatial_json.read(MAX_SPATIAL_BYTES + 1)
            await spatial_json.close()
            if len(spatial_bytes) > MAX_SPATIAL_BYTES:
                raise HTTPException(status_code=413, detail="Spatial evidence exceeds 2 MiB")
        stored_metadata = metadata.model_dump(mode="json")
        if metadata.spatial_evidence_sha256 is None:
            # Keep the legacy canonical metadata byte-identical for retries of
            # captures stored before the optional field was introduced.
            stored_metadata.pop("spatial_evidence_sha256")
        receipt = store.save_capture(str(session_id), stored_metadata, data, spatial_bytes)
        if not receipt["duplicate"]:
            try:
                on_capture_received(config.data_dir, str(session_id))
            except (OSError, sqlite3.DatabaseError) as error:
                # The image is durably received; a queue failure must not force
                # the headset to resend it as though capture persistence failed.
                app.state.scheduler_error = str(error)
        return receipt

    @app.get("/api/v1/sessions/{session_id}/captures", dependencies=[Depends(authenticate)])
    def list_captures(session_id: uuid.UUID):
        return {"captures": store.captures(str(session_id))}

    @app.get("/api/v1/sessions/{session_id}/preview", dependencies=[Depends(authenticate)])
    def preview(session_id: uuid.UUID, representation: str = Query(..., pattern="^(splat|mesh|depth)$")):
        manifest = preview_manifest(store, str(session_id), representation)
        if representation == "splat" and manifest["status"] != "ready":
            try:
                request_build(config.data_dir, str(session_id), reason="preview", quality="quick")
            except (OSError, sqlite3.DatabaseError) as error:
                app.state.scheduler_error = str(error)
        return manifest

    @app.get('/api/v1/projects/{project_id}/spatial/{session_id}/{revision}', dependencies=[Depends(authenticate)])
    def download_depth_preview(project_id: uuid.UUID, session_id: uuid.UUID, revision: str):
        manifest, path = read_depth_preview(store, str(project_id), str(session_id), revision)
        response = FileResponse(path, media_type='application/octet-stream')
        response.headers['ETag'] = '"'+manifest['artifact']['sha256']+'"'
        response.headers['Cache-Control'] = 'private, no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get("/api/v1/projects/{project_id}/reconstructions/{job_id}/splat", dependencies=[Depends(authenticate)])
    def download_preview(project_id: uuid.UUID, job_id: uuid.UUID):
        job, path = read_job(store, str(project_id), str(job_id))
        response = FileResponse(path, media_type="application/octet-stream")
        response.headers["ETag"] = '"' + job["artifact"]["sha256"] + '"'
        response.headers["Cache-Control"] = "private, max-age=3600, immutable"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/v1/projects/{project_id}/reconstructions/{job_id}/quest-splat", dependencies=[Depends(authenticate)])
    def download_quest_preview(project_id: uuid.UUID, job_id: uuid.UUID):
        job, path = read_quest_preview(store, str(project_id), str(job_id))
        response = FileResponse(path, media_type="application/octet-stream")
        response.headers["ETag"] = '"' + job["quest_preview"]["sha256"] + '"'
        response.headers["Cache-Control"] = "private, max-age=3600, immutable"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/v1/projects/{project_id}/online/{session_id}/splat", dependencies=[Depends(authenticate)])
    def download_online_preview(project_id: uuid.UUID, session_id: uuid.UUID):
        status, path, _alignment = read_online_preview(store, str(project_id), str(session_id))
        response = FileResponse(path, media_type="application/octet-stream")
        response.headers["ETag"] = '"' + status.get("display_artifact", status["artifact"])["sha256"] + '"'
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get('/api/v1/projects/{project_id}/online/{session_id}/splat/{revision}', dependencies=[Depends(authenticate)])
    def download_online_revision(project_id: uuid.UUID, session_id: uuid.UUID, revision: str):
        _status, path, _alignment = read_online_preview(store, str(project_id), str(session_id), revision=revision)
        response = FileResponse(path, media_type='application/octet-stream')
        response.headers['ETag'] = '"' + revision + '"'
        response.headers['Cache-Control'] = 'private, max-age=3600, immutable'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    from .mobile import install_mobile
    install_mobile(app, store, authenticate)
    from .mesh import install_mesh
    install_mesh(app, store, authenticate)
    return app
