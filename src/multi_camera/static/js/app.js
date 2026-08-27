/**
 * ARGUS Multi-Camera Surveillance Operations Center Orchestrator
 */

class SurveillanceApp {
  constructor() {
    this.mode = 'matrix'; // 'matrix' | 'topology'
    this.cameras = [];
    this.activeCameraId = null;
    this.targetState = 'UNSELECTED';
    this.searchProgress = null;
    this.statusPollTimer = null;
    this.galleryPollTimer = null;

    // Topology Graph Canvas & Managers
    this.inspector = new Inspector(this);
    this.cameraManager = new CameraManager(this);
    this.graphCanvas = null;

    this.initElements();
    this.initEvents();
    this.loadLiveMatrix();
    this.startPolling();
  }

  initElements() {
    // Mode Views
    this.viewMatrix = document.getElementById('view-matrix');
    this.viewTopology = document.getElementById('view-topology');
    this.matrixGrid = document.getElementById('camera-matrix-grid');
    this.matrixCountBadge = document.getElementById('matrix-cam-count');

    // Header Badges
    this.hdrActiveCam = document.getElementById('header-active-cam');
    this.hdrTargetState = document.getElementById('header-target-state');
    this.hdrSearchRadius = document.getElementById('header-search-radius');
    this.hdrGalleryStats = document.getElementById('header-gallery-stats');

    // Target Summary Card Elements
    this.cardTargetId = document.getElementById('target-card-id');
    this.cardTargetState = document.getElementById('target-card-state');
    this.cardTargetCam = document.getElementById('target-card-cam');
    this.cardTargetSamples = document.getElementById('target-card-samples');
    this.galleryCountBadge = document.getElementById('gallery-count-badge');
    this.galleryCardsList = document.getElementById('gallery-cards-list');

    // Forensic & Radius Dock
    this.transitTrailEl = document.getElementById('transit-trail-display');
  }

  initEvents() {
    // Mode Switch
    document.getElementById('btn-mode-matrix').addEventListener('click', () => this.setMode('matrix'));
    document.getElementById('btn-mode-topology').addEventListener('click', () => this.setMode('topology'));

    // Grid Layout Buttons
    document.querySelectorAll('.grid-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.grid-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const layout = btn.dataset.layout;
        this.matrixGrid.className = `camera-matrix-grid layout-${layout}`;
      });
    });

    // Target Action Buttons (Header + Card)
    const addSampleFn = async () => {
      try {
        const res = await API.addSample(this.activeCameraId);
        if (res.success) {
          this.showToast(`Target appearance angle captured! (${res.size} in gallery)`, 'success');
          this.refreshGallery();
        } else {
          this.showToast('No active target locked to capture angle', 'error');
        }
      } catch (err) {
        this.showToast(`Failed to capture angle: ${err.message}`, 'error');
      }
    };
    this.addSampleFn = addSampleFn;

    const clearTargetFn = async () => {
      try {
        await API.clearTarget();
        this.showToast('Focus target cleared and gallery purged', 'info');
        this.refreshStatus();
        this.refreshGallery();
      } catch (err) {
        this.showToast(`Failed to clear target: ${err.message}`, 'error');
      }
    };
    this.clearTargetFn = clearTargetFn;

    document.getElementById('btn-add-sample-global').addEventListener('click', addSampleFn);
    document.getElementById('btn-card-add-sample').addEventListener('click', addSampleFn);

    document.getElementById('btn-clear-target-global').addEventListener('click', clearTargetFn);
    document.getElementById('btn-card-clear').addEventListener('click', clearTargetFn);

    const btnQuit = document.getElementById('btn-quit-global');
    if (btnQuit) {
      btnQuit.addEventListener('click', () => this.safeQuit());
    }

    // Global Keyboard Shortcuts (Issue 2)
    window.addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const key = e.key.toLowerCase();
      if (key === 'a') {
        e.preventDefault();
        addSampleFn();
      } else if (key === 'c') {
        e.preventDefault();
        clearTargetFn();
      } else if (key === 'q') {
        e.preventDefault();
        this.safeQuit();
      }
    });
  }

  async safeQuit() {
    if (!confirm('Are you sure you want to cleanly shut down Argus Surveillance? All camera connections and models will be released.')) {
      return;
    }
    if (this.statusPollTimer) clearInterval(this.statusPollTimer);
    if (this.galleryPollTimer) clearInterval(this.galleryPollTimer);
    try {
      this.showToast('Shutting down surveillance pipeline and releasing camera resources...', 'info');
      await API.quit();
      setTimeout(() => {
        document.body.innerHTML = `
          <div style="height:100vh;display:flex;align-items:center;justify-content:center;background:#0b0f17;color:#94a3b8;font-family:system-ui,-apple-system,sans-serif;flex-direction:column;gap:14px;text-align:center;">
            <div style="width:48px;height:48px;border-radius:50%;background:rgba(0,242,254,0.1);border:1px solid #00f2fe;display:flex;align-items:center;justify-content:center;color:#00f2fe;font-size:20px;">&#10003;</div>
            <h2 style="color:#f8fafc;margin:0;font-weight:600;">Argus Surveillance Operations Center Closed</h2>
            <p style="margin:0;max-width:400px;font-size:13px;line-height:1.5;">All camera capture streams, AI models, and background workers have been released safely.</p>
            <p style="font-size:11px;color:#64748b;">You can safely close this browser window.</p>
          </div>
        `;
      }, 400);
    } catch (err) {
      this.showToast(`Shutdown: ${err.message}`, 'info');
    }
  }

  setMode(mode) {
    this.mode = mode;
    document.getElementById('btn-mode-matrix').classList.toggle('active', mode === 'matrix');
    document.getElementById('btn-mode-topology').classList.toggle('active', mode === 'topology');

    if (mode === 'matrix') {
      this.viewMatrix.style.display = 'flex';
      this.viewTopology.style.display = 'none';
      this.loadLiveMatrix();
    } else {
      this.viewMatrix.style.display = 'none';
      this.viewTopology.style.display = 'flex';
      if (!this.graphCanvas) {
        this.graphCanvas = new GraphCanvas(this, 'graph-canvas');
        this.loadGraphTopology();
      }
    }
  }

  async loadGraphTopology() {
    try {
      const data = await API.getGraph();
      if (this.graphCanvas) {
        this.graphCanvas.loadGraph(data);
        this.graphCanvas.fitToScreen();
      }
    } catch (e) {
      this.showToast(`Topology load error: ${e.message}`, 'error');
    }
  }

  /* ==========================================================================
     MULTI-CAMERA LIVE MATRIX RENDERING
     ========================================================================== */

  async loadLiveMatrix() {
    try {
      const data = await API.getLiveCameras();
      this.cameras = data.cameras || [];
      this.activeCameraId = data.active_camera;

      this.renderMatrixGrid();
      this.matrixCountBadge.textContent = `${this.cameras.length} Cameras Online`;
    } catch (err) {
      this.matrixGrid.innerHTML = `
        <div class="matrix-loading">
          <span>Failed to connect to surveillance stream backend. Retrying...</span>
        </div>
      `;
    }
  }

  renderMatrixGrid() {
    if (this.cameras.length === 0) {
      this.matrixGrid.innerHTML = `
        <div class="matrix-loading">
          <span>No camera sources configured. Switch to Topology Map to add cameras.</span>
        </div>
      `;
      return;
    }

    this.matrixGrid.innerHTML = '';

    this.cameras.forEach((cam) => {
      const isAct = (cam.camera_id === this.activeCameraId);
      const isSearch = cam.is_searching;

      let tileClass = 'camera-tile';
      let badgeClass = 'badge-standby';
      let badgeText = cam.status || 'STANDBY';

      if (isAct) {
        tileClass += ' tile-active';
        badgeClass = 'badge-active';
        badgeText = 'ACTIVE FOCUS';
      } else if (isSearch) {
        tileClass += ' tile-searching';
        badgeClass = 'badge-searching';
        badgeText = 'SEARCHING';
      } else if (!cam.enabled) {
        badgeClass = 'badge-offline';
        badgeText = 'DISABLED';
      }

      const streamUrl = API.getCameraStreamUrl(cam.camera_id);
      const fallbackUrl = API.getCameraFrameUrl(cam.camera_id);

      const tile = document.createElement('div');
      tile.className = tileClass;
      tile.id = `tile-${cam.camera_id}`;
      tile.dataset.cameraId = cam.camera_id;

      tile.innerHTML = `
        <div class="camera-tile-header">
          <div class="tile-cam-info">
            <span class="tile-cam-name">${cam.name || cam.camera_id}</span>
            <span class="tile-cam-id">[${cam.camera_id}]</span>
          </div>
          <span class="tile-cam-badge ${badgeClass}" id="badge-${cam.camera_id}">${badgeText}</span>
        </div>

        <div class="camera-tile-video" id="stage-${cam.camera_id}">
          <img class="camera-feed-img" 
               id="img-${cam.camera_id}"
               src="${streamUrl}" 
               alt="${cam.name}" 
               onerror="this.onerror=null; this.src='${fallbackUrl}'">
          
          <div class="camera-tile-overlay">
            <button class="tile-overlay-btn btn-focus-cam" data-cam="${cam.camera_id}">Set Active</button>
            <span style="font-size:10px; color:#94a3b8;">Click feed to lock target</span>
            <button class="tile-overlay-btn btn-snap-cam" data-cam="${cam.camera_id}">+ Angle</button>
          </div>
        </div>
      `;

      // Click on tile video stage: Set active camera + select target at coordinates
      const stage = tile.querySelector('.camera-tile-video');
      const img = tile.querySelector('.camera-feed-img');

      stage.addEventListener('click', async (e) => {
        if (e.target.classList.contains('tile-overlay-btn')) return;

        const rect = img.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Scale to image actual pixel dimensions
        const scaleX = (img.naturalWidth || 640) / rect.width;
        const scaleY = (img.naturalHeight || 480) / rect.height;

        const targetX = clickX * scaleX;
        const targetY = clickY * scaleY;

        try {
          const res = await API.selectTarget(cam.camera_id, targetX, targetY);
          if (res && res.selected_id !== null && res.selected_id !== undefined) {
            this.showToast(`Target locked! Tracker ID: ${res.selected_id} on ${cam.name || cam.camera_id}`, 'success');
            this.activeCameraId = cam.camera_id;
            this.refreshStatus();
            this.refreshGallery();
          } else {
            // Simply switch active camera focus
            await API.setActiveCamera(cam.camera_id);
            this.activeCameraId = cam.camera_id;
            this.updateActiveTileVisuals();
            this.refreshStatus();
            this.showToast(`Active focus set to '${cam.name || cam.camera_id}'`, 'info');
          }
        } catch (err) {
          console.error(err);
          this.showToast(`Camera activation error: ${err.message}`, 'error');
        }
      });

      // Right-click on feed: Capture manual angle (Shortcut action)
      stage.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        this.addSampleFn();
      });

      // Overlay button handlers
      tile.querySelector('.btn-focus-cam').addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          await API.setActiveCamera(cam.camera_id);
          this.activeCameraId = cam.camera_id;
          this.updateActiveTileVisuals();
          this.refreshStatus();
          this.showToast(`Focused on '${cam.name || cam.camera_id}'`, 'info');
        } catch (err) {
          this.showToast(`Focus error: ${err.message}`, 'error');
        }
      });


      tile.querySelector('.btn-snap-cam').addEventListener('click', async (e) => {
        e.stopPropagation();
        const res = await API.addSample(cam.camera_id);
        if (res.success) {
          this.showToast(`Angle captured on ${cam.name}!`, 'success');
          this.refreshGallery();
        } else {
          this.showToast('No active target locked to add angle', 'error');
        }
      });

      this.matrixGrid.appendChild(tile);
    });
  }

  updateActiveTileVisuals() {
    document.querySelectorAll('.camera-tile').forEach((t) => {
      const cid = t.dataset.cameraId;
      const isAct = (cid === this.activeCameraId);
      t.classList.toggle('tile-active', isAct);
      const badge = document.getElementById(`badge-${cid}`);
      if (badge && isAct) {
        badge.className = 'tile-cam-badge badge-active';
        badge.textContent = 'ACTIVE FOCUS';
      }
    });
  }

  /* ==========================================================================
     PERIODIC STATUS & GALLERY POLLING
     ========================================================================== */

  startPolling() {
    this.refreshStatus();
    this.refreshGallery();

    this.statusPollTimer = setInterval(() => this.refreshStatus(), 700);
    this.galleryPollTimer = setInterval(() => this.refreshGallery(), 750);
  }

  async refreshStatus() {
    try {
      const st = await API.getStatus();
      if (!st) return;

      this.activeCameraId = st.active_camera;
      this.targetState = st.target_state || 'UNSELECTED';
      this.searchProgress = st.search_progress;

      // Update Header HUD
      this.hdrActiveCam.textContent = st.active_camera || 'None';

      this.hdrTargetState.textContent = this.targetState;
      this.hdrTargetState.className = `chip-val chip-badge state-${this.targetState.toLowerCase()}`;

      const rad = st.search_progress ? st.search_progress.search_radius : 0;
      const searchSt = st.search_progress ? st.search_progress.state.toUpperCase() : 'IDLE';
      this.hdrSearchRadius.textContent = `R = ${rad} (${searchSt})`;

      this.hdrGalleryStats.textContent = `${st.gallery_size || 0} / ${st.gallery_max || 25}`;

      // Update Target Summary Card
      this.cardTargetId.textContent = st.target_track_id ? `Tracker #${st.target_track_id}` : (st.target_state !== 'UNSELECTED' ? 'TARGET_0' : 'UNSELECTED');
      this.cardTargetState.textContent = this.targetState;
      this.cardTargetState.className = `state-tag state-${this.targetState.toLowerCase()}`;
      this.cardTargetCam.textContent = st.active_camera || 'None';

      const scores = st.candidate_scores || {};
      const scoreKeys = Object.keys(scores);
      if (scoreKeys.length > 0) {
        const scoreStr = scoreKeys.map(k => `#${k}: ${(scores[k]).toFixed(2)}`).join(' | ');
        this.cardTargetSamples.innerHTML = `<span style="color:#00f2fe;font-weight:600;">Sim: ${scoreStr}</span> (${st.gallery_manual || 0}m/${st.gallery_auto || 0}a)`;
      } else {
        this.cardTargetSamples.textContent = `${st.gallery_manual || 0} manual / ${st.gallery_auto || 0} auto`;
      }

      // Update Bottom Radius Stepper
      this.updateRadiusStepper(rad, searchSt);

      // Update Forensic Trail
      const trail = st.transit_history || [];
      if (trail.length === 0) {
        this.transitTrailEl.innerHTML = '<span class="trail-empty">No cross-camera movement recorded yet</span>';
      } else {
        this.transitTrailEl.innerHTML = trail.map((t, idx) => `
          <span class="step-pill step-active">${t.camera_id || t}</span>
          ${idx < trail.length - 1 ? '<span class="step-arrow">&rarr;</span>' : ''}
        `).join('');
      }

      // Dynamic topology sync: if backend cameras list changed, re-render matrix grid
      const backendCamIds = Object.keys(st.camera_statuses || {});
      const localCamIds = this.cameras.map(c => c.camera_id);
      if (backendCamIds.length !== localCamIds.length || backendCamIds.some(id => !localCamIds.includes(id))) {
        this.loadLiveMatrix();
      }

      // Update Active/Searching tile indicators in the grid
      if (this.cameras.length > 0) {
        const searchingCams = new Set(st.search_progress ? st.search_progress.active_cameras : []);
        this.cameras.forEach((cam) => {
          const tile = document.getElementById(`tile-${cam.camera_id}`);
          const badge = document.getElementById(`badge-${cam.camera_id}`);
          if (!tile || !badge) return;

          const isAct = (cam.camera_id === st.active_camera);
          const isSearch = searchingCams.has(cam.camera_id);

          tile.classList.toggle('tile-active', isAct);
          tile.classList.toggle('tile-searching', !isAct && isSearch);

          if (isAct) {
            badge.className = 'tile-cam-badge badge-active';
            badge.textContent = 'ACTIVE FOCUS';
          } else if (isSearch) {
            badge.className = 'tile-cam-badge badge-searching';
            badge.textContent = `SEARCHING (R=${rad})`;
          } else {
            badge.className = 'tile-cam-badge badge-standby';
            badge.textContent = 'STANDBY';
          }
        });
      }
    } catch (e) {
      console.debug('Status poll error', e);
    }
  }


  updateRadiusStepper(radius, stateStr) {
    for (let r = 0; r <= 3; r++) {
      const stepEl = document.getElementById(`radius-step-${r}`);
      if (!stepEl) continue;
      if (r === 0) {
        stepEl.className = 'step-pill step-active';
      } else if (r <= radius && stateStr !== 'IDLE') {
        stepEl.className = 'step-pill step-searching';
      } else {
        stepEl.className = 'step-pill';
      }
    }
  }

  /* ==========================================================================
     RIGHT VERTICAL TARGET GALLERY COLUMN RENDERING (REQUIREMENT #2)
     ========================================================================== */

  async refreshGallery() {
    try {
      const g = await API.getGallery();
      if (!g) return;

      this.galleryCountBadge.textContent = `${g.size} / ${g.max_size}`;

      if (!g.thumbnails || g.thumbnails.length === 0) {
        this.galleryCardsList.innerHTML = `
          <div class="gallery-empty-state">
            <div class="empty-icon">&#128100;</div>
            <div class="empty-text">No target locked</div>
            <div class="empty-hint">Click on any tracked person in a live camera feed to lock focus and seed their appearance gallery.</div>
          </div>
        `;
        return;
      }

      // Reverse list to show newest appearances at top
      const items = [...g.thumbnails].reverse();

      this.galleryCardsList.innerHTML = items.map((thumb) => {
        const isMan = thumb.is_manual;
        const tagClass = isMan ? 'badge-manual' : 'badge-auto';
        const tagLabel = isMan ? 'MANUAL' : 'AUTO';
        const qualityPct = Math.round((thumb.quality_score || thumb.confidence || 0.9) * 100);
        const imgSrc = thumb.image_b64 ? `data:image/jpeg;base64,${thumb.image_b64}` : '';

        return `
          <div class="gallery-crop-card" title="Entry: ${thumb.entry_id} | Camera: ${thumb.camera_id}">
            <div class="crop-thumb-box">
              ${imgSrc ? `<img src="${imgSrc}" class="crop-thumb-img" alt="Target Crop">` : '<div style="color:#64748b;font-size:10px;">No image</div>'}
            </div>
            <div class="crop-meta-box">
              <div class="crop-meta-top">
                <span class="crop-type-badge ${tagClass}">${tagLabel}</span>
                <span class="crop-quality-text">Q: ${qualityPct}%</span>
                <button class="crop-delete-btn" data-entry="${thumb.entry_id}" title="Remove this angle">&times;</button>
              </div>
              <div class="crop-cam-text">${thumb.camera_id || 'cam_0'}</div>
              <div class="crop-time-text">${thumb.timestamp_ms ? `${(thumb.timestamp_ms / 1000).toFixed(1)}s` : 'Captured'}</div>
            </div>
          </div>
        `;
      }).join('');

      // Wire up crop delete buttons (Hypothesis A fix)
      this.galleryCardsList.querySelectorAll('.crop-delete-btn').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
          e.stopPropagation();
          const entryId = btn.dataset.entry;
          try {
            await API.deleteGalleryEntry(entryId);
            this.showToast('Removed angle crop from gallery', 'info');
            this.refreshGallery();
            this.refreshStatus();
          } catch (err) {
            this.showToast(`Failed to delete crop: ${err.message}`, 'error');
          }
        });
      });
    } catch (e) {
      console.debug('Gallery poll error', e);
    }
  }

  showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
}

// Bootstrap Application on Load
window.addEventListener('DOMContentLoaded', () => {
  window.app = new SurveillanceApp();
});
