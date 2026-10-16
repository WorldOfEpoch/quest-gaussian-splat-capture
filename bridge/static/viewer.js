import * as GaussianSplats3D from './vendor/gaussian-splats-3d.module.js?v=sh3-20261013';

const viewport = document.getElementById('splat-viewer');
const selector = document.getElementById('splat-job');
const refresh = document.getElementById('reload-jobs');
const resetView = document.getElementById('reset-view');
const freezeBefore = document.getElementById('freeze-before');
const compareModels = document.getElementById('compare-models');
const status = document.getElementById('viewer-status');
const photoProgress = document.getElementById('photo-progress');
const modelBatchLink = document.getElementById('model-batch-link');
const processingState = document.getElementById('processing-state');
const buildProgress = document.getElementById('build-progress');
const buildRows = document.getElementById('build-job-rows');
const depthRows = document.querySelectorAll('[data-depth-session]');

if (viewport && selector && refresh && status) {
  const projectId = viewport.dataset.projectId;
  let depthRefreshRunning = false;
  let jobs = [];
  let viewer = null;
  let viewerStarted = false;
  let currentId = null;
  let desiredId = null;
  let loading = false;
  let fetchingJobs = false;
  let failedLoadId = null;
  let retryDelayMs = 0;
  let retryAfter = 0;
  let navigationBound = false;
  let navigationView = null;
  let stopWheelNavigation = () => {};
  let beforeJob = null;
  let showingBefore = false;
  let regionJob = null;
  let showingRegion = false;
  const regionEditor = document.getElementById('region-editor');
  const regionForm = document.getElementById('region-form');
  const regionStatus = document.getElementById('region-status');
  let regionSource = null;
  let boxVisible = false;
  let regionOverlay = null;
  let regionPhotoPage = 1;
  let regionPhotoSession = null;
  const replacementPhotos = new Set();
  const reviewedDrafts = new Map();
  const approvedDrafts = new Map();

  function updateComparisonControl(current) {
    if (!compareModels) return;
    const ready = Boolean(beforeJob && current && beforeJob.id !== current.id);
    compareModels.disabled = !ready;
    compareModels.textContent = ready && currentId === beforeJob.id ? 'Show current' : 'Show before';
    compareModels.title = ready
      ? 'Switch model revisions without moving the inspection camera'
      : 'Waiting for a newer accepted model revision';
  }

  const setStatus = (message) => { status.textContent = message; };

  function workerState(job) {
    if (job.status === 'paused_quality_review') return 'Paused for quality review; photo receiving continues.';
    if (job.status === 'blocked') return 'Live splat update stopped; the last published model is still visible. See Capture sessions for the reason.';
    if (String(job.status || '').startsWith('paused')) return 'Live splat updates are paused; the last published model is still visible.';
    if (job.status === 'training') return 'Refining new detail; the last published model stays visible.';
    if (job.status === 'waiting_for_new_views') return 'Waiting for new usable views.';
    const updated = Date.parse(job.worker_updated_utc);
    if (Number.isFinite(updated) && Date.now() - updated < 30000) return 'Online worker checking for new photos.';
    return 'Online worker has not reported recently.';
  }

  function renderPhotoProgress(job) {
    if (modelBatchLink) {
      modelBatchLink.hidden = !job?.session_id;
      if (job?.session_id) modelBatchLink.href = `/ui/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(job.session_id)}/captures`;
      else modelBatchLink.removeAttribute('href');
    }
    if (!photoProgress) return;
    if (!job) {
      photoProgress.textContent = 'No current model yet. Saved camera photos remain available in Capture sessions.';
      return;
    }
    const progress = job.photo_progress;
    if (!progress) {
      photoProgress.textContent = 'Photo progress is unavailable for this model.';
      return;
    }
    const selected = `This model's batch: ${progress.selected_count} selected of ${progress.captured_count} saved photos`;
    const baseline = progress.baseline_registered_count == null
      ? `${progress.baseline_input_count} baseline inputs; matched count unavailable`
      : `${progress.baseline_registered_count} of ${progress.baseline_input_count} baseline photos matched`;
    const unmatched = progress.baseline_unmatched_count > 0
      ? `; ${progress.baseline_unmatched_count} baseline photos did not match` : '';
    const excluded = progress.excluded_count ? `; ${progress.excluded_count} excluded` : '';
    if (job.source === 'online') {
      const modelTime = Date.parse(job.model_updated_utc);
      const published = Number.isFinite(modelTime)
        ? `Visible model published ${new Date(modelTime).toLocaleString()}. ` : '';
      const represented = progress.baseline_registered_count == null
        ? 'Total photos admitted to this model are not yet verified'
        : `${(progress.baseline_registered_count + progress.added_count).toLocaleString()} photos registered or admitted to the current model (not a detail-quality score)`;
      const pending = progress.pending_selected_count == null
        ? 'pending count unavailable'
        : `${progress.pending_selected_count} selected photos not yet admitted`;
      const review = progress.rejected_count > 0
        ? ` ${progress.rejected_count} photos need alignment or quality review; their originals remain saved.` : '';
      photoProgress.textContent = `${represented}. ${published}${selected}. ${baseline}${unmatched}. ` +
        `${progress.added_count} new photos admitted to the live model; ${pending}${excluded}.${review} ${workerState(job)}`;
    } else {
      photoProgress.textContent = `${selected}. Saved model: ${baseline}${unmatched}${excluded}. ` +
        'No online additions are shown for this saved build.';
    }
  }

  function frameCamera(view, initialize = false) {
    const [x, y, z] = view.center;
    const zUp = view.up_axis === 'z';
    const position = view.eye || [x, y, z];
    const lookAt = view.look_at || (zUp ? [x, y - 1, z] : [x, y, z - 1]);
    const up = zUp ? [0, 0, 1] : [0, 1, 0];
    if (initialize) {
      viewer = new GaussianSplats3D.Viewer({
        rootElement: viewport,
        cameraUp: up,
        initialCameraPosition: position,
        initialCameraLookAt: lookAt,
        useBuiltInControls: false,
        sharedMemoryForWorkers: false,
        gpuAcceleratedSort: false,
        integerBasedSort: false,
        sphericalHarmonicsDegree: 3,
        showLoadingUI: false,
      });
    }
  }

  function centerCamera(view) {
    if (!viewer || !view) return;
    stopWheelNavigation();
    const camera = viewer.camera;
    const zUp = view.up_axis === 'z';
    const eye = view.eye || view.center;
    const target = view.look_at || (zUp
      ? [eye[0], eye[1] - 1, eye[2]] : [eye[0], eye[1], eye[2] - 1]);
    camera.up.set(...(zUp ? [0, 0, 1] : [0, 1, 0]));
    camera.position.set(...eye);
    camera.lookAt(...target);
    camera.updateMatrixWorld();
  }

  function bindNavigation() {
    if (navigationBound) return;
    navigationBound = true;
    const canvas = viewer.renderer.domElement;
    let dragging = null;
    let lastX = 0;
    let lastY = 0;
    const pendingTravel = viewer.camera.position.clone().set(0, 0, 0);
    let wheelFrame = null;
    let lastWheelFrame = 0;
    stopWheelNavigation = () => {
      if (wheelFrame !== null) cancelAnimationFrame(wheelFrame);
      wheelFrame = null;
      pendingTravel.set(0, 0, 0);
    };
    const advanceWheel = (now) => {
      wheelFrame = null;
      // Ease by elapsed time, with a bounded step after a stalled frame. Do
      // not teleport through the room when the tab/render thread resumes.
      const elapsed = Math.min(32, Math.max(0, now - lastWheelFrame));
      lastWheelFrame = now;
      const fraction = -Math.expm1(-elapsed / 80);
      viewer.camera.position.addScaledVector(pendingTravel, fraction);
      pendingTravel.multiplyScalar(1 - fraction);
      viewer.camera.updateMatrixWorld();
      if (pendingTravel.lengthSq() > 1e-12) {
        wheelFrame = requestAnimationFrame(advanceWheel);
      } else {
        stopWheelNavigation();
      }
    };
    window.addEventListener('blur', stopWheelNavigation);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stopWheelNavigation();
    });
    canvas.addEventListener('pointerleave', stopWheelNavigation);
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 && event.button !== 2) return;
      stopWheelNavigation();
      dragging = {button: event.button, pointerId: event.pointerId};
      lastX = event.clientX;
      lastY = event.clientY;
      if (event.button === 2) {
        // Pointer lock keeps the cursor fixed and hidden while looking. The
        // lock is released on button-up, so the cursor returns to the page.
        if (canvas.requestPointerLock) {
          const request = canvas.requestPointerLock();
          if (request && typeof request.catch === 'function') {
            request.catch(() => { /* Drag still works if the browser refuses lock. */ });
          }
        }
      } else {
        canvas.setPointerCapture(event.pointerId);
      }
      event.preventDefault();
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!dragging || dragging.pointerId !== event.pointerId) return;
      const locked = document.pointerLockElement === canvas;
      const dx = locked ? event.movementX : event.clientX - lastX;
      const dy = locked ? event.movementY : event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      const camera = viewer.camera;
      const view = navigationView;
      if (!view) return;
      if (dragging.button === 2) {
        const before = camera.quaternion.clone();
        camera.rotateOnWorldAxis(camera.up, (view.up_axis === 'z' ? dx : -dx) * 0.004);
        camera.rotateX(-dy * 0.004);
        const forward = camera.getWorldDirection(camera.up.clone());
        if (Math.abs(forward.dot(camera.up)) > 0.985) camera.quaternion.copy(before);
      } else {
        const right = camera.up.clone().set(1, 0, 0).applyQuaternion(camera.quaternion);
        const distance = Math.max(view.radius * 0.002, 0.002);
        camera.position.addScaledVector(right, -dx * distance);
        camera.position.addScaledVector(camera.up, dy * distance);
      }
      camera.updateMatrixWorld();
      event.preventDefault();
    });
    const finishDrag = (event) => {
      if (!dragging || dragging.pointerId !== event.pointerId) return;
      dragging = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (document.pointerLockElement === canvas) document.exitPointerLock();
    };
    canvas.addEventListener('pointerup', finishDrag);
    canvas.addEventListener('pointercancel', finishDrag);
    document.addEventListener('pointerup', finishDrag);
    document.addEventListener('pointerlockchange', () => {
      if (document.pointerLockElement === canvas && !dragging) {
        document.exitPointerLock();
      } else if (dragging?.button === 2 && document.pointerLockElement !== canvas) {
        dragging = null;
      }
    });
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const camera = viewer.camera;
      const view = navigationView;
      if (!view) return;
      // Wheel deltas can be pixels, lines, or pages. Keep small trackpad
      // gestures small; there must be no minimum travel per event.
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1;
      const pixels = event.deltaY * unit;
      if (!Number.isFinite(pixels) || pixels === 0) return;
      const step = Math.max(view.radius * 0.02, 0.01);
      const travel = camera.getWorldDirection(camera.up.clone())
        .multiplyScalar(-Math.max(-200, Math.min(200, pixels)) * step / 100);
      // A reversal should respond immediately, not fight the previous glide.
      if (pendingTravel.dot(travel) < 0) pendingTravel.set(0, 0, 0);
      pendingTravel.add(travel).clampLength(0, step * 4);
      if (wheelFrame === null) {
        lastWheelFrame = performance.now();
        wheelFrame = requestAnimationFrame(advanceWheel);
      }
    }, {passive: false});
  }

  async function loadDesired() {
    if (loading) return;
    loading = true;
    try {
      while (desiredId && desiredId !== currentId) {
        const id = desiredId;
        if (id === failedLoadId && Date.now() < retryAfter) break;
        const job = jobs.find((item) => item.id === id && item.viewable);
        if (!job) {
          setStatus('That scan is not available for viewing.');
          desiredId = currentId;
          break;
        }
        if (!viewer) frameCamera(job.view, true);
        const previousId = currentId;
        stopWheelNavigation();
        const path = job.url;
        setStatus(`Loading ${job.vertex_count.toLocaleString()} Gaussians. The previous scan remains visible until this one is ready…`);
        try {
          await viewer.addSplatScene(path, {
            // Revision URLs have a query suffix; the renderer otherwise tries
            // to infer the format with path.endsWith('.ply') and rejects them.
            format: GaussianSplats3D.SceneFormat.Ply,
            showLoadingUI: false,
            progressiveLoad: false,
            splatAlphaRemovalThreshold: 1,
          });
          if (desiredId !== id) {
            await viewer.removeSplatScene(previousId ? 1 : 0, false);
            continue;
          }
          if (previousId) await viewer.removeSplatScene(0, false);
          currentId = id;
          failedLoadId = null;
          retryDelayMs = 0;
          retryAfter = 0;
          navigationView = job.view;
          if (job.source === 'area_draft') reviewedDrafts.set(job.regionId, job.artifactHash);
          updateComparisonControl(jobs[0]);
          if (!viewerStarted) {
            bindNavigation();
            viewer.start();
            viewerStarted = true;
            startRegionOverlay();
          }
          const detail = job.source === 'area_draft'
            ? `; area draft: ${job.name}; not applied to the current scan`
            : beforeJob && job.id === beforeJob.id && job.id !== jobs[0]?.id
            ? job.comparisonRole === 'saved_reference'
              ? `; saved ${job.quality || 'quick'} reference from ${job.registered_views ?? '?'} matched photos`
              : '; frozen Before revision for same-angle comparison'
            : job.source === 'online'
            ? `; live quick preview; ${job.training_steps} optimization steps; ` +
              (job.filtered_out > 0 ? `${job.filtered_out.toLocaleString()} display-only floaters hidden; ` : '') +
              workerState(job)
            : `; saved ${job.quality || 'unknown-quality'} build; ${job.registered_views ?? '?'} of ${job.input_count} images matched`;
          const placement = job.source === 'area_draft'
            ? 'Inspect this same-angle draft before locking and applying it.'
            : job.source === 'online'
            ? job.physical_acceptance === 'accepted'
              ? 'The room overlay was visually accepted in the headset; newly added areas still need inspection.'
              : 'Check physical placement in the headset before trusting the room overlay; new areas still need inspection.'
            : 'Physical room overlay requires alignment acceptance.';
          setStatus(`${job.vertex_count.toLocaleString()} Gaussians${detail}. ${placement} Hold right mouse to lock and hide the cursor while looking; release to restore it. Left drag shifts position; wheel moves forward or back.`);
        } catch (error) {
          console.error('Local splat load failed', error);
          retryDelayMs = failedLoadId === id ? Math.min(retryDelayMs * 2, 60000) : 5000;
          failedLoadId = id;
          retryAfter = Date.now() + retryDelayMs;
          setStatus('Unable to load this splat. The previous scan is still visible; retrying shortly.');
          desiredId = currentId;
          break;
        }
      }
    } finally {
      loading = false;
    }
  }

  function regionMessage(message) {
    if (regionStatus) regionStatus.textContent = message;
  }

  function selectedBounds() {
    const center = ['x', 'y', 'z'].map((axis) => Number(document.getElementById(`region-${axis}`).value));
    const size = ['x', 'y', 'z'].map((axis) => Number(document.getElementById(`region-s${axis}`).value));
    if (center.some((value) => !Number.isFinite(value) || Math.abs(value) > 10000) ||
        size.some((value) => !Number.isFinite(value) || value < 0.01 || value > 10000)) return null;
    return {min: center.map((value, i) => value - size[i] / 2),
      max: center.map((value, i) => value + size[i] / 2)};
  }

  function setBox(bounds) {
    ['x', 'y', 'z'].forEach((axis, i) => {
      document.getElementById(`region-${axis}`).value = ((bounds.min[i] + bounds.max[i]) / 2).toFixed(4);
      document.getElementById(`region-s${axis}`).value = (bounds.max[i] - bounds.min[i]).toFixed(4);
    });
    boxVisible = true;
    document.getElementById('region-hide').textContent = 'Hide box';
  }

  function startRegionOverlay() {
    if (!regionEditor || regionOverlay) return;
    regionOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    regionOverlay.classList.add('region-overlay');
    regionOverlay.setAttribute('aria-hidden', 'true');
    const edges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
    const lines = edges.map(() => {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('stroke', '#ffcf67'); line.setAttribute('stroke-width', '2');
      regionOverlay.appendChild(line); return line;
    });
    viewport.appendChild(regionOverlay);
    const draw = () => {
      regionOverlay.style.display = boxVisible && regionEditor.open ? '' : 'none';
      const bounds = selectedBounds();
      if (boxVisible && regionEditor.open && bounds && viewer?.camera) {
        const width = viewport.clientWidth, height = viewport.clientHeight;
        regionOverlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
        const camera = viewer.camera;
        camera.updateMatrixWorld();
        const corners = Array.from({length: 8}, (_, i) => camera.position.clone().set(
          i & 1 ? bounds.max[0] : bounds.min[0], i & 2 ? bounds.max[1] : bounds.min[1],
          i & 4 ? bounds.max[2] : bounds.min[2]));
        edges.forEach(([a,b], i) => {
          const start = corners[a].clone().applyMatrix4(camera.matrixWorldInverse);
          const end = corners[b].clone().applyMatrix4(camera.matrixWorldInverse);
          // Clip at the near plane; projecting behind-eye corners flips lines.
          const near = -camera.near * 1.01;
          const visible = start.z < near || end.z < near;
          lines[i].style.display = visible ? '' : 'none';
          if (!visible) return;
          if (start.z >= near) start.lerp(end, (near - start.z) / (end.z - start.z));
          if (end.z >= near) end.lerp(start, (near - end.z) / (start.z - end.z));
          start.applyMatrix4(camera.projectionMatrix); end.applyMatrix4(camera.projectionMatrix);
          lines[i].setAttribute('x1', (start.x + 1) * width / 2);
          lines[i].setAttribute('y1', (1 - start.y) * height / 2);
          lines[i].setAttribute('x2', (end.x + 1) * width / 2);
          lines[i].setAttribute('y2', (1 - end.y) * height / 2);
        });
      }
      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  }

  async function regionRequest(route, data = null) {
    const response = await fetch(`/ui/projects/${encodeURIComponent(projectId)}/${route}`, data
      ? {method: 'POST', body: new URLSearchParams({csrf: regionForm.dataset.csrf, ...data})}
      : {cache: 'no-store'});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The area request could not be completed.');
    return result;
  }

  async function loadRegionPhotos() {
    if (!regionEditor) return;
    const current = jobs[0];
    const grid = document.getElementById('region-photos');
    if (!current || current.source !== 'online') {
      grid.textContent = 'Area editing needs an online model with a saved checkpoint.'; return;
    }
    if (regionPhotoSession !== current.session_id) {
      regionPhotoSession = current.session_id; regionPhotoPage = 1; replacementPhotos.clear();
    }
    const result = await regionRequest(`region-photos?session_id=${encodeURIComponent(regionPhotoSession)}&page=${regionPhotoPage}`);
    grid.replaceChildren();
    for (const photo of result.photos) {
      const label = document.createElement('label');
      const thumb = document.createElement('img'); thumb.loading = 'lazy';
      thumb.src = `/ui/projects/${encodeURIComponent(projectId)}/captures/${photo.capture_id}/image?thumbnail=true`;
      thumb.alt = `Photo received ${photo.received_utc}`;
      const checkbox = document.createElement('input'); checkbox.type = 'checkbox';
      checkbox.checked = replacementPhotos.has(photo.capture_id); checkbox.disabled = !photo.included;
      checkbox.setAttribute('aria-label', `Use photo received ${photo.received_utc}`);
      checkbox.addEventListener('change', () => {
        if (checkbox.checked && replacementPhotos.size >= 128) {
          checkbox.checked = false; regionMessage('Choose up to 128 photos for one area.'); return;
        }
        if (checkbox.checked) replacementPhotos.add(photo.capture_id); else replacementPhotos.delete(photo.capture_id);
        document.getElementById('region-photo-count').textContent = `${replacementPhotos.size} photos selected`;
      });
      const text = document.createElement('small');
      text.textContent = photo.included ? photo.received_utc.slice(0,19).replace('T',' ') : 'Excluded from processing';
      const link = document.createElement('a'); link.href = thumb.src.replace('?thumbnail=true','');
      link.target = '_blank'; link.rel = 'noopener'; link.appendChild(thumb);
      label.append(link, checkbox, text); grid.appendChild(label);
    }
    const pages = Math.max(1, Math.ceil(result.count / result.page_size));
    document.getElementById('region-photo-page').textContent = `Page ${regionPhotoPage} of ${pages}`;
    document.getElementById('region-photo-count').textContent = `${replacementPhotos.size} photos selected`;
    document.getElementById('region-photos-prev').disabled = regionPhotoPage <= 1;
    document.getElementById('region-photos-next').disabled = regionPhotoPage >= pages;
  }

  async function previewRegion(record) {
    const artifact = record.artifact;
    if (!artifact || !record.view) throw new Error('This area has no verified viewable draft yet.');
    regionJob = {id: `area:${record.id}:${artifact.sha256}`, source: 'area_draft', regionId: record.id,
      name: record.name, artifactHash: artifact.sha256, view: record.view, vertex_count: artifact.vertex_count,
      viewable: true, url: `/ui/projects/${encodeURIComponent(projectId)}/regions/${record.id}/draft.ply?revision=${artifact.sha256}`};
    showingRegion = true; showingBefore = false;
    jobs = jobs.filter((job) => job.source !== 'area_draft'); jobs.push(regionJob);
    desiredId = regionJob.id; setBox(record); await loadDesired();
    if (currentId === regionJob.id) regionMessage(`Viewing draft: ${record.name}. The current project model has not changed.`);
    await refreshRegions();
  }

  async function refreshRegions() {
    if (!regionEditor) return;
    const result = await regionRequest('regions');
    const list = document.getElementById('region-list'); list.replaceChildren();
    for (const record of result.regions) {
      const card = document.createElement('section'); card.className = 'region-card';
      const title = document.createElement('strong'); title.textContent = record.name;
      const info = document.createElement('p'); info.className = 'muted';
      info.textContent = `${record.status.replaceAll('_', ' ')} · ${record.replacement_capture_ids?.length || 0} replacement photos. ${record.message || ''}`;
      const toolbar = document.createElement('div'); toolbar.className = 'viewer-toolbar';
      const button = (label, callback, disabled = false) => {
        const element = document.createElement('button'); element.type = 'button'; element.textContent = label;
        element.disabled = disabled; element.addEventListener('click', async () => {
          element.disabled = true;
          try { await callback(); } catch (error) { regionMessage(error.message); }
          finally { element.disabled = disabled; }
        }); toolbar.appendChild(element); return element;
      };
      const action = async (name, extra = {}) => {
        const next = await regionRequest(`regions/${record.id}/${name}`, extra);
        regionMessage(`${next.name}: ${next.status.replaceAll('_', ' ')}. ${next.message || ''}`);
        if (name === 'publish' || name === 'restore') { showingRegion = false; regionJob = null; await fetchJobs(); }
        await refreshRegions();
      };
      const busy = ['queued', 'building', 'locking'].includes(record.status);
      const mutable = ['planned', 'draft_ready', 'quality_rejected', 'failed'].includes(record.status);
      button('Show area box', () => { setBox(record); });
      button('Protect this good area', () => action('protect'), busy || !mutable);
      button('Use selected photos for this area', () => action('set-photos', {capture_ids: JSON.stringify([...replacementPhotos])}),
        busy || !mutable);
      button('Clear area in draft', () => action('clear'), busy || !mutable);
      button('Build replacement', () => action('build'), busy || !mutable || (record.replacement_capture_ids?.length || 0) < 2);
      button('Preview draft', () => previewRegion(record), busy || !record.artifact);
      button('Lock draft', () => action('lock'), record.status !== 'draft_ready');
      if (record.status === 'locked') {
        const label = document.createElement('label'); const reviewed = document.createElement('input');
        reviewed.type = 'checkbox';
        reviewed.checked = approvedDrafts.get(record.id) === record.artifact?.sha256;
        label.append(reviewed, 'I inspected this locked draft and want to apply it to this project.');
        card.appendChild(label);
        const apply = button('Apply reviewed locked area', () => action('publish', {reviewed: 'yes'}),
          !reviewed.checked || reviewedDrafts.get(record.id) !== record.artifact?.sha256);
        reviewed.addEventListener('change', () => {
          apply.disabled = !reviewed.checked || reviewedDrafts.get(record.id) !== record.artifact?.sha256;
          if (reviewed.checked) approvedDrafts.set(record.id, record.artifact?.sha256); else approvedDrafts.delete(record.id);
          if (reviewed.checked && apply.disabled) regionMessage('Preview this exact locked draft first, then apply it after inspection.');
        });
      }
      if (record.status === 'published') button('Restore pre-edit scan', () => action('restore'));
      card.prepend(title, info); card.appendChild(toolbar); list.appendChild(card);
    }
    if (!result.regions.length) list.textContent = 'No saved areas yet.';
  }

  if (regionEditor && regionForm) {
    document.getElementById('region-place').addEventListener('click', () => {
      const current = jobs[0];
      if (!viewer || current?.source !== 'online' || currentId !== current.id) {
        regionMessage('Show the current online model before placing an area.'); return;
      }
      const distance = Number(document.getElementById('region-distance').value);
      if (!Number.isFinite(distance) || distance < 0.01 || distance > 10000) return;
      stopWheelNavigation();
      const center = viewer.camera.getWorldDirection(viewer.camera.position.clone()).multiplyScalar(distance).add(viewer.camera.position);
      ['x','y','z'].forEach((axis) => { document.getElementById(`region-${axis}`).value = center[axis].toFixed(4); });
      regionSource = {source_model_id: current.id, source_sha256: current.source_sha256,
        coordinate_frame: current.coordinate_frame, session_id: current.session_id};
      boxVisible = true; document.getElementById('region-hide').textContent = 'Hide box';
      regionMessage('Adjust the amber box to enclose only the area you intend to replace. Save area creates a plan; it does not delete the current scan.');
    });
    document.getElementById('region-hide').addEventListener('click', (event) => {
      boxVisible = !boxVisible; event.target.textContent = boxVisible ? 'Hide box' : 'Show box';
    });
    regionForm.addEventListener('submit', async (event) => {
      event.preventDefault(); const bounds = selectedBounds();
      if (!bounds || !regionSource || currentId !== jobs[0]?.id) {
        regionMessage('Place a valid box on the current model before saving.'); return;
      }
      const save = document.getElementById('region-save'); save.disabled = true;
      try {
        const record = await regionRequest('regions', {plan: JSON.stringify({...regionSource, ...bounds,
          name: document.getElementById('region-name').value, replacement_capture_ids: [...replacementPhotos]})});
        regionSource = null;
        regionMessage(`${record.name}: area saved. Protect this good area keeps its geometry unchanged, or build a private replacement.`);
        await refreshRegions();
      } catch (error) { regionMessage(error.message); }
      finally { save.disabled = false; }
    });
    regionEditor.addEventListener('toggle', () => {
      if (regionEditor.open) Promise.all([loadRegionPhotos(), refreshRegions()]).catch((error) => regionMessage(error.message));
    });
    document.getElementById('region-photos-prev').addEventListener('click', () => {
      regionPhotoPage = Math.max(1, regionPhotoPage - 1); loadRegionPhotos().catch((error) => regionMessage(error.message));
    });
    document.getElementById('region-photos-next').addEventListener('click', () => {
      regionPhotoPage++; loadRegionPhotos().catch((error) => regionMessage(error.message));
    });
    document.getElementById('region-photos-refresh').addEventListener('click', () => {
      regionPhotoPage = 1; loadRegionPhotos().catch((error) => regionMessage(error.message));
    });
  }

  async function fetchJobs() {
    if (fetchingJobs) return;
    fetchingJobs = true;
    try {
      const [response, savedResponse] = await Promise.all([
        fetch(`/ui/current-model?project_id=${encodeURIComponent(projectId)}`, {cache: 'no-store'}),
        fetch(`/ui/jobs?project_id=${encodeURIComponent(projectId)}`, {cache: 'no-store'}),
      ]);
      if (!response.ok) throw new Error(`Current model returned ${response.status}`);
      if (!savedResponse.ok) throw new Error(`Saved models returned ${savedResponse.status}`);
      const current = (await response.json()).model;
      const savedJobs = (await savedResponse.json()).jobs || [];
      if (current && !beforeJob) {
        const reference = current.source === 'online'
          ? savedJobs.find((job) => job.id === current.seed_job_id &&
              job.session_id === current.session_id && job.viewable && job.url)
          : null;
        beforeJob = reference ? {...reference, comparisonRole: 'saved_reference'} : {...current};
      }
      jobs = current ? [current] : [];
      if (beforeJob && beforeJob.id !== current?.id) jobs.push(beforeJob);
      if (regionJob) jobs.push(regionJob);
      if (current && current.id === currentId) navigationView = current.view;
      renderPhotoProgress(current);
      const selected = showingRegion ? regionJob?.id : showingBefore ? beforeJob?.id : current?.id;
      selector.replaceChildren(new Option('Current model', ''));
      for (const job of jobs) {
        if (!job.viewable) continue;
        const label = job.source === 'area_draft' ? `Area draft · ${job.name}`
          : job.id === beforeJob?.id && job.id !== current?.id
          ? `${job.comparisonRole === 'saved_reference' ? 'Saved reference' : 'Before'} · ${job.vertex_count.toLocaleString()} Gaussians`
          : `${job.source === 'online' ? 'Current preview' : `Saved ${job.quality || 'build'}`} · ${job.vertex_count.toLocaleString()} Gaussians`;
        selector.add(new Option(label, job.id));
      }
      selector.hidden = jobs.length < 2;
      const available = jobs.some((job) => job.id === selected && job.viewable);
      selector.value = available ? selected : (jobs[0]?.id || '');
      if (selector.options.length > 1 && selector.value !== currentId) {
        desiredId = selector.value;
        await loadDesired();
      } else if (selector.options.length === 1) {
        setStatus('No verified Gaussian model is available for this project yet.');
      }
      updateComparisonControl(current);
    } finally {
      fetchingJobs = false;
    }
  }

  async function refreshBuildStatus() {
    if (!buildProgress || !buildRows) return;
    const response = await fetch(`/ui/build-status?project_id=${encodeURIComponent(projectId)}`, {cache: 'no-store'});
    if (!response.ok) throw new Error(`Build status returned ${response.status}`);
    const data = await response.json();
    if (processingState) processingState.textContent = data.processing_paused
      ? 'Build processing is paused. New photos can still be received; saved models remain viewable. The separate live model may also be paused for quality review.'
      : 'Build queue is available. Photo receiving does not mean the live model is training.';
    const active = data.requests.find((item) => item.status === 'running' || item.status === 'pending');
    const latest = active || data.requests[0];
    buildProgress.textContent = latest
      ? `${latest.quality} build: ${latest.status.replaceAll('_', ' ')} (${latest.input_count} captured views)` +
        (latest.error ? `. ${latest.error}` : '')
      : 'No builds queued. Capture at least three overlapping views for a quick attempt.';
    if (data.processing_paused) buildProgress.textContent = 'Build queue paused. ' + buildProgress.textContent;
    buildRows.replaceChildren();
    for (const job of data.jobs) {
      const row = document.createElement('tr');
      const matched = job.registered_views == null ? String(job.input_count) : `${job.registered_views} / ${job.input_count}`;
      for (const value of [job.created_utc?.slice(0, 16).replace('T', ' ') || '—', job.quality || '—',
                           job.status.replaceAll('_', ' '), matched, job.vertex_count ?? '—']) {
        const cell = document.createElement('td');
        cell.textContent = String(value);
        row.appendChild(cell);
      }
      buildRows.appendChild(row);
    }
    if (!data.jobs.length) {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 5;
      cell.textContent = 'No build jobs yet.';
      row.appendChild(cell);
      buildRows.appendChild(row);
    }
    await fetchJobs();
  }

  async function refreshDepthUpdates() {
    if (!depthRows.length || depthRefreshRunning) return;
    depthRefreshRunning = true;
    try {
      const response = await fetch(`/ui/projects/${encodeURIComponent(projectId)}/depth-updates`, {cache: 'no-store'});
      if (!response.ok) throw new Error(`Live depth status returned ${response.status}`);
      const data = await response.json();
      const sessions = new Map((data.sessions || []).map((item) => [item.session_id, item]));
      for (const row of depthRows) {
        const item = sessions.get(row.dataset.depthSession);
        if (!item) continue;
        const summary = row.querySelector('.depth-summary');
        const button = row.querySelector('.depth-form button');
        const toggle = row.querySelector('.depth-enabled');
        if (!summary || !button || !toggle) continue;
        const count = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString() : '0';
        const state = String(item.status || 'unknown').replaceAll('_', ' ');
        summary.textContent = `${state}: ${count(item.accepted_views)} of ${count(item.total_views)} photos admitted · ` +
          `${count(item.pending_views)} waiting · ${count(item.skipped_views)} skipped · ` +
          `${count(item.supported_samples)} supported points` + (item.error ? ` · ${item.error}` : '');
        toggle.value = item.enabled ? 'no' : 'yes';
        button.textContent = item.enabled ? 'Pause live depth' : 'Start live depth';
      }
    } catch (error) {
      console.error('Live depth status refresh failed', error);
      for (const row of depthRows) {
        const summary = row.querySelector('.depth-summary');
        if (summary) summary.textContent = 'Live depth status unavailable; refresh the page to retry.';
      }
    } finally {
      depthRefreshRunning = false;
    }
  }

  let splatRefreshRunning = false;
  async function refreshSplatUpdates() {
    if (splatRefreshRunning) return;
    splatRefreshRunning = true;
    try {
      const response = await fetch(`/ui/projects/${encodeURIComponent(projectId)}/splat-updates`, {cache: 'no-store'});
      if (!response.ok) throw new Error(`Live splat status returned ${response.status}`);
      const data = await response.json();
      for (const item of data.sessions || []) {
        const label = [...document.querySelectorAll('[data-splat-session]')].find(el => el.dataset.splatSession === item.session_id);
        if (!label) continue;
        const pending = item.pending_refinement?.pending ? ' · refining new detail' : '';
        const skipped = Number(item.quarantined_count || 0) + Number(item.rejected_count || 0);
        const processed = item.processed_count == null ? '' : ` · ${Number(item.processed_count)} processed`;
        label.textContent = `Live splat: ${String(item.status || 'off').replaceAll('_', ' ')} · ${Number(item.trained_views || 0)} / ${Number(item.total_views || 0)} photos in visible model` + processed + ` · ${Number(item.gaussian_count || 0).toLocaleString()} Gaussians` + pending + (skipped ? ` · ${skipped} need review` : '') + (item.error ? ` · ${item.error}` : '');
        const form = label.parentElement.querySelector('.splat-form');
        if (form) {
          form.querySelector('.splat-enabled').value = item.enabled ? 'no' : 'yes';
          form.querySelector('button').textContent = item.enabled ? 'Pause live splat' : 'Start live splat';
        }
      }
    } catch (error) { console.error('Live splat progress unavailable', error); }
    finally { splatRefreshRunning = false; }
  }
  refreshSplatUpdates();
  setInterval(refreshSplatUpdates, 3000);

  selector.addEventListener('change', () => {
    desiredId = selector.value;
    showingRegion = Boolean(regionJob && desiredId === regionJob.id);
    showingBefore = Boolean(beforeJob && desiredId === beforeJob.id && desiredId !== jobs[0]?.id);
    if (desiredId) loadDesired();
  });
  refresh.addEventListener('click', () => {
    retryAfter = 0;
    fetchJobs().catch((error) => {
      console.error('Local job refresh failed', error);
      setStatus('Could not refresh the local job list.');
    });
  });
  resetView?.addEventListener('click', () => {
    const view = jobs.find((job) => job.id === currentId)?.view;
    if (view) centerCamera(view);
  });
  freezeBefore?.addEventListener('click', () => {
    const current = jobs[0];
    if (!current?.viewable) return;
    beforeJob = {...current};
    showingBefore = false;
    updateComparisonControl(current);
    setStatus('Current model marked as Before. When a quality-checked revision arrives, Show before will switch between the two at this same camera position.');
  });
  compareModels?.addEventListener('click', () => {
    const current = jobs[0];
    if (!beforeJob || !current || beforeJob.id === current.id) return;
    desiredId = currentId === beforeJob.id ? current.id : beforeJob.id;
    showingRegion = false;
    showingBefore = desiredId === beforeJob.id;
    selector.value = desiredId;
    loadDesired();
  });
  fetchJobs().catch((error) => {
    console.error('Local job list failed', error);
    setStatus('Could not load the local job list.');
  });
  setInterval(() => fetchJobs().catch((error) => {
    console.error('Local job refresh failed', error);
    if (!currentId) setStatus('Could not refresh the local job list.');
  }), 5000);
  refreshBuildStatus().catch((error) => {
    console.error('Local build status failed', error);
    if (buildProgress) buildProgress.textContent = 'Could not load build queue status.';
  });
  setInterval(() => refreshBuildStatus().catch((error) => {
    console.error('Local build status refresh failed', error);
    if (buildProgress) buildProgress.textContent = 'Could not refresh build queue status.';
  }), 10000);
  refreshDepthUpdates();
  setInterval(refreshDepthUpdates, 3000);
  if (regionEditor) setInterval(() => {
    if (regionEditor.open) refreshRegions().catch((error) => regionMessage(error.message));
  }, 5000);
}
