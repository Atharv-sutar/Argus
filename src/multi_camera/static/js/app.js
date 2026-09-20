/**
 * ARGUS Multi-Camera Surveillance Operations Center Orchestrator
 */

class SurveillanceApp {
  constructor() {
    this.mode = 'matrix'; // 'matrix' | 'topology'
    this.cameras = [];
    this.activeCameraId = null;
    this._pendingActiveCameraSwitch = null;
    this._pendingActiveCameraSwitchTime = 0;
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

    // Annotations

    // Cases
    this.caseOverlay = document.getElementById('case-overlay');
    this.activeCaseDisplay = document.getElementById('active-case-display');
    this.btnCaseClose = document.getElementById('btn-case-close');
    this.caseTableBody = document.getElementById('case-table-body');
    this.caseIdInput = document.getElementById('case-id-input');
    this.caseOpInput = document.getElementById('case-operator-input');
    this.btnCaseCreate = document.getElementById('btn-case-create');
    this.activeCaseId = null;

    this.annoOverlay = document.getElementById('annotation-overlay');
    this.annoCamId = document.getElementById('anno-cam-id');
    this.annoTimeMs = document.getElementById('anno-time-ms');
    this.annoInput = document.getElementById('anno-text-input');
    this.btnAnnoCancel = document.getElementById('btn-anno-cancel');
    this.btnAnnoSave = document.getElementById('btn-anno-save');
    this.annotations = [];
    this.annoPollTimer = null;
    this.currentPlaybackMs = 0; // fallback tracking

  }

  initEvents() {
    // Mode Switch

    const handoffOverlay = document.getElementById('handoff-overlay');
    const btnHandoffConfirm = document.getElementById('btn-handoff-confirm');
    const btnHandoffReject = document.getElementById('btn-handoff-reject');
    
    if (btnHandoffConfirm) {
      btnHandoffConfirm.addEventListener('click', async () => {
        try {
          await fetch('/api/handoff/confirm', { method: 'POST' });
          if(handoffOverlay) handoffOverlay.style.display = 'none';
        } catch (err) { console.error('Handoff confirm error', err); }
      });
    }

    if (btnHandoffReject) {
      btnHandoffReject.addEventListener('click', async () => {
        try {
          await fetch('/api/handoff/reject', { method: 'POST' });
          if(handoffOverlay) handoffOverlay.style.display = 'none';
        } catch (err) { console.error('Handoff reject error', err); }
      });
    }

    document.getElementById('btn-mode-matrix').addEventListener('click', () => this.setMode('matrix'));
    document.getElementById('btn-mode-topology').addEventListener('click', () => this.setMode('topology'));

    // Bulk Import RTSP
    const btnBulkImportOpen = document.getElementById('btn-bulk-import-open');
    const bulkImportOverlay = document.getElementById('bulk-import-overlay');
    const btnBulkImportCancel = document.getElementById('btn-bulk-import-cancel');
    const btnBulkImportSubmit = document.getElementById('btn-bulk-import-submit');
    const bulkImportData = document.getElementById('bulk-import-data');

    if (btnBulkImportOpen) {
      btnBulkImportOpen.addEventListener('click', () => {
        if (bulkImportOverlay) bulkImportOverlay.style.display = 'flex';
      });
    }
    if (btnBulkImportCancel) {
      btnBulkImportCancel.addEventListener('click', () => {
        if (bulkImportOverlay) bulkImportOverlay.style.display = 'none';
      });
    }
    if (btnBulkImportSubmit) {
      btnBulkImportSubmit.addEventListener('click', async () => {
        btnBulkImportSubmit.disabled = true;
        try {
          const res = await API.bulkImportCameras(bulkImportData.value);
          if (res.success) {
            this.showToast(`Imported ${res.imported} cameras successfully!`, 'success');
            bulkImportData.value = '';
            if (bulkImportOverlay) bulkImportOverlay.style.display = 'none';
            // Refresh graph
            const graphData = await API.getGraph();
            if (this.graphCanvas) {
              this.graphCanvas.loadGraph(graphData);
            }
            this.refreshListView(graphData);
          }
        } catch (err) {
          this.showToast(`Bulk import error: ${err.message}`, 'error');
        } finally {
          btnBulkImportSubmit.disabled = false;
        }
      });
    }

    // Map/List View Toggle
    const btnMap = document.getElementById('btn-topo-map');
    const btnList = document.getElementById('btn-topo-list');
    const canvasEl = document.getElementById('graph-canvas');
    const listEl = document.getElementById('topo-list-view');

    if (btnMap && btnList) {
      btnMap.addEventListener('click', () => {
        btnMap.classList.add('active');
        btnList.classList.remove('active');
        if (canvasEl) canvasEl.style.display = 'block';
        if (listEl) listEl.style.display = 'none';
      });
      btnList.addEventListener('click', async () => {
        btnList.classList.add('active');
        btnMap.classList.remove('active');
        if (canvasEl) canvasEl.style.display = 'none';
        if (listEl) listEl.style.display = 'block';
        
        const graphData = await API.getGraph();
        this.refreshListView(graphData);
      });
    }
    
    // Search/Filter for List View
    const topoSearch = document.getElementById('topo-search-input');
    const topoFilterZone = document.getElementById('topo-filter-zone');
    const topoFilterFloor = document.getElementById('topo-filter-floor');
    const updateList = async () => {
        const graphData = await API.getGraph();
        this.refreshListView(graphData);
    };
    if (topoSearch) topoSearch.addEventListener('input', updateList);
    if (topoFilterZone) topoFilterZone.addEventListener('change', updateList);
    if (topoFilterFloor) topoFilterFloor.addEventListener('change', updateList);

    // List batch actions
    const btnEnableSel = document.getElementById('btn-list-enable-selected');
    const btnDisableSel = document.getElementById('btn-list-disable-selected');
    const btnDelSel = document.getElementById('btn-list-delete-selected');
    const selectAllCheckbox = document.getElementById('list-select-all');

    if (selectAllCheckbox) {
      selectAllCheckbox.addEventListener('change', (e) => {
        const checkboxes = document.querySelectorAll('.list-item-checkbox');
        checkboxes.forEach(cb => cb.checked = e.target.checked);
      });
    }

    const performBatchAction = async (action) => {
      if (!this.graphCanvas) return;
      const checkboxes = document.querySelectorAll('.list-item-checkbox:checked');
      if (checkboxes.length === 0) return;
      
      const ids = Array.from(checkboxes).map(cb => cb.dataset.id);
      
      this.graphCanvas.nodes.forEach(node => {
        if (ids.includes(node.camera_id)) {
          if (action === 'enable') node.enabled = true;
          if (action === 'disable') node.enabled = false;
        }
      });
      if (action === 'delete') {
        this.graphCanvas.nodes = this.graphCanvas.nodes.filter(n => !ids.includes(n.camera_id));
        this.graphCanvas.edges = this.graphCanvas.edges.filter(e => !ids.includes(e.source) && !ids.includes(e.target));
      }
      
      try {
        await API.saveGraph(this.graphCanvas.toJSON());
        this.showToast(`Batch ${action} completed`, 'success');
        this.refreshListView(this.graphCanvas.toJSON());
      } catch (err) {
        this.showToast(`Batch error: ${err.message}`, 'error');
      }
    };

    if (btnEnableSel) btnEnableSel.addEventListener('click', () => performBatchAction('enable'));
    if (btnDisableSel) btnDisableSel.addEventListener('click', () => performBatchAction('disable'));
    if (btnDelSel) btnDelSel.addEventListener('click', () => performBatchAction('delete'));


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
    const addSampleFn = async (e) => {
      if (e && typeof e.stopPropagation === 'function') {
        e.preventDefault();
        e.stopPropagation();
      }
      try {
        const res = await API.addSample(this.activeCameraId);
        if (res.success) {
          this.showToast(`Target appearance angle captured! (${res.size} in gallery)`, 'success');
          await this.refreshGallery();
          await this.refreshStatus();
        } else {
          this.showToast('No active target locked to capture angle', 'error');
        }
      } catch (err) {
        this.showToast(`Failed to capture angle: ${err.message}`, 'error');
      }
    };
    this.addSampleFn = addSampleFn;

    const clearTargetFn = async (e) => {
      if (e && typeof e.stopPropagation === 'function') {
        e.preventDefault();
        e.stopPropagation();
      }
      try {
        await API.clearTarget();
        this.showToast('Focus target cleared and gallery purged', 'info');
        await this.refreshStatus();
        await this.refreshGallery();
      } catch (err) {
        this.showToast(`Failed to clear target: ${err.message}`, 'error');
      }
    };
    this.clearTargetFn = clearTargetFn;

    const undoFn = async () => {
      try {
        const res = await API.undoAction();
        this.showToast(`Undid action: ${res.action.action_type}`, 'success');
        await this.refreshStatus();
        await this.refreshGallery();
        this.pollUndoStack();
      } catch (err) {
        this.showToast(err.message, 'error');
      }
    };
    this.undoFn = undoFn;

    const redoFn = async () => {
      try {
        const res = await API.redoAction();
        this.showToast(`Redid action: ${res.action.action_type}`, 'success');
        await this.refreshStatus();
        await this.refreshGallery();
        this.pollUndoStack();
      } catch (err) {
        this.showToast(err.message, 'error');
      }
    };
    this.redoFn = redoFn;

    document.getElementById('btn-undo-global')?.addEventListener('click', undoFn);
    document.getElementById('btn-redo-global')?.addEventListener('click', redoFn);


    document.getElementById('btn-add-sample-global').addEventListener('click', addSampleFn);
    document.getElementById('btn-card-add-sample').addEventListener('click', addSampleFn);

    document.getElementById('btn-clear-target-global').addEventListener('click', clearTargetFn);
    document.getElementById('btn-card-clear').addEventListener('click', clearTargetFn);

    // Refresh / Restart Feeds Button
    const refreshCamerasFn = async () => {
      const btnRefresh = document.getElementById('btn-refresh-cameras');
      if (btnRefresh) {
        btnRefresh.disabled = true;
        btnRefresh.innerHTML = `
          <svg class="spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
          </svg>
          <span>Restarting...</span>
        `;
      }
      this.showToast('Shutting down and restarting all camera feeds...', 'info');

      // 1. Drop active MJPEG connections so sockets close on client side
      document.querySelectorAll('.camera-feed-img').forEach((img) => {
        img.onerror = null;
        img.src = '';
      });

      try {
        await API.restartCameras();
        await new Promise((r) => setTimeout(r, 300));
        await this.loadLiveMatrix();

        // 2. Force re-attach streams with new cache-busting timestamp
        document.querySelectorAll('.camera-feed-img').forEach((img) => {
          const camId = img.id.replace('img-', '');
          if (camId) {
            img.src = `${API.getCameraStreamUrl(camId)}?t=${Date.now()}`;
          }
        });

        this.showToast('All camera streams restarted successfully', 'success');
      } catch (err) {
        this.showToast(`Failed to restart cameras: ${err.message}`, 'error');
        await this.loadLiveMatrix();
      } finally {
        if (btnRefresh) {
          btnRefresh.disabled = false;
          btnRefresh.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
              <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
            </svg>
            <span>Restart Feeds</span>
          `;
        }
      }
    };
    this.refreshCamerasFn = refreshCamerasFn;

    const btnRefresh = document.getElementById('btn-refresh-cameras');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', refreshCamerasFn);
    }



    if (this.activeCaseDisplay) {
        this.activeCaseDisplay.addEventListener('click', () => this.showCaseManager());
    }
    if (this.btnCaseClose) {
        this.btnCaseClose.addEventListener('click', () => this.caseOverlay.style.display = 'none');
    }
    if (this.btnCaseCreate) {
        this.btnCaseCreate.addEventListener('click', async () => {
            const cid = this.caseIdInput.value.trim();
            const op = this.caseOpInput.value.trim();
            if (!cid) {
                this.showToast('Case ID is required', 'error');
                return;
            }
            try {
                await API.createCase(cid, 'live', op, '');
                await this.openCase(cid);
                this.caseIdInput.value = '';
                this.caseOpInput.value = '';
                this.caseOverlay.style.display = 'none';
            } catch (err) {
                this.showToast(`Failed to create case: ${err.message}`, 'error');
            }
        });
    }

    this.showCaseManager = async () => {
        this.caseOverlay.style.display = 'flex';
        await this.loadCaseList();
    };

    this.loadCaseList = async () => {
        if (!this.caseTableBody) return;
        try {
            const data = await API.getCases();
            this.activeCaseId = data.active_case;
            if (this.activeCaseDisplay) {
                this.activeCaseDisplay.textContent = this.activeCaseId || '[No Active Case]';
            }
            
            this.caseTableBody.innerHTML = '';
            data.cases.forEach(c => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid var(--border-color)';
                const isActive = (c.case_id === this.activeCaseId);
                
                tr.innerHTML = `
                    <td style="padding: 8px;">${c.case_id} ${isActive ? '<span style="color:var(--cyan-bright)">(Active)</span>' : ''}</td>
                    <td style="padding: 8px;">${c.status}</td>
                    <td style="padding: 8px;">${new Date(c.created_at).toLocaleString()}</td>
                    <td style="padding: 8px;">
                        ${!isActive ? `<button class="btn btn-primary" onclick="window.appInstance.openCase('${c.case_id}')" style="padding: 2px 6px; font-size: 12px;">Open</button>` : ''}
                        <button class="btn btn-secondary" onclick="window.appInstance.deleteCase('${c.case_id}')" style="padding: 2px 6px; font-size: 12px;">Del</button>
                    </td>
                `;
                this.caseTableBody.appendChild(tr);
            });
        } catch (err) {
            console.error('Failed to load cases', err);
        }
    };

    this.openCase = async (cid) => {
        try {
            await API.openCase(cid);
            this.showToast(`Opened case ${cid}`, 'success');
            if (this.caseOverlay) this.caseOverlay.style.display = 'none';
            // Clear current state and refresh UI
            this.galleryItems = [];
            this.annotations = [];
            this.renderGallery();
            this.loadCaseList();
            this.pollGallery(); // immediate refresh
        } catch (err) {
            this.showToast(`Failed to open case: ${err.message}`, 'error');
        }
    };

    this.deleteCase = async (cid) => {
        if (!confirm(`Are you sure you want to delete case ${cid}?`)) return;
        try {
            await API.deleteCase(cid);
            this.showToast(`Deleted case ${cid}`, 'success');
            await this.loadCaseList();
        } catch (err) {
            this.showToast(`Failed to delete case: ${err.message}`, 'error');
        }
    };

    if (this.btnAnnoCancel) {
      this.btnAnnoCancel.addEventListener('click', () => {
        this.annoOverlay.style.display = 'none';
        this.annoInput.value = '';
      });
    }

    if (this.btnAnnoSave) {
      this.btnAnnoSave.addEventListener('click', async () => {
        const text = this.annoInput.value.trim();
        const camId = this.annoCamId.textContent;
        const timeMs = parseFloat(this.annoTimeMs.dataset.ms || "0");
        if (!text) {
           this.showToast('Note cannot be empty', 'error');
           return;
        }
        
        try {
          await API.createAnnotation(camId, timeMs, text, null, 'note');
          this.showToast('Annotation saved', 'success');
          this.annoOverlay.style.display = 'none';
          this.annoInput.value = '';
          this.pollAnnotations();
        } catch (err) {
          this.showToast(`Failed to save: ${err.message}`, 'error');
        }
      });
    }
    
    this.openAnnotationModal = () => {
        if (!this.activeCameraId || this.activeCameraId === 'None') {
            this.showToast('No active camera selected for annotation', 'error');
            return;
        }
        if (this.annoOverlay) {
            this.annoCamId.textContent = this.activeCameraId;
            // In live mode, use date.now relative to some start or just 0 for now.
            // A more precise app would sync with pipeline timestamps.
            const ms = this.currentPlaybackMs > 0 ? this.currentPlaybackMs : 0.0;
            this.annoTimeMs.textContent = (ms / 1000).toFixed(1);
            this.annoTimeMs.dataset.ms = ms;
            this.annoOverlay.style.display = 'flex';
            this.annoInput.focus();
        }
    };

    const btnQuit = document.getElementById('btn-quit-global');
    if (btnQuit) {
      btnQuit.addEventListener('click', () => this.safeQuit());
    }

    // Global Keyboard Shortcuts (Issue 2)
    window.addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
      const key = e.key.toLowerCase();
      if (key === 'z' && e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        redoFn();
      } else if (key === 'z' && e.ctrlKey) {
        e.preventDefault();
        undoFn();
      } else if (key === 'n' && !e.ctrlKey) {
        e.preventDefault();
        this.openAnnotationModal();
      } else if (key === 'a') {
        e.preventDefault();
        addSampleFn();
      } else if (key === 'c') {
        e.preventDefault();
        clearTargetFn();
      } else if (key === 'r') {
        e.preventDefault();
        refreshCamerasFn();
      } else if (key === 'q') {
        e.preventDefault();
        this.safeQuit();
      }
    });
  }


  refreshListView(graphData) {
    const tbody = document.getElementById('topo-list-tbody');
    const searchInput = document.getElementById('topo-search-input');
    const filterZone = document.getElementById('topo-filter-zone');
    const filterFloor = document.getElementById('topo-filter-floor');
    
    if (!tbody || !graphData || !graphData.cameras) return;
    
    const query = (searchInput ? searchInput.value.toLowerCase() : '');
    const selZone = (filterZone ? filterZone.value : '');
    const selFloor = (filterFloor ? filterFloor.value : '');
    
    let zones = new Set();
    let floors = new Set();
    
    let cameras = graphData.cameras.filter(c => {
      if (c.zone) zones.add(c.zone);
      if (c.floor) floors.add(c.floor);
      
      if (selZone && c.zone !== selZone) return false;
      if (selFloor && c.floor !== selFloor) return false;
      if (query && !c.name.toLowerCase().includes(query) && !c.camera_id.toLowerCase().includes(query)) return false;
      return true;
    });
    
    // Populate dropdowns if empty
    if (filterZone && filterZone.options.length <= 1) {
      zones.forEach(z => filterZone.add(new Option(z, z)));
    }
    if (filterFloor && filterFloor.options.length <= 1) {
      floors.forEach(f => filterFloor.add(new Option(f, f)));
    }
    
    tbody.innerHTML = cameras.map(c => `
      <tr style="border-bottom: 1px solid var(--border-color); background: ${c.enabled ? 'transparent' : 'rgba(255,0,0,0.05)'}">
        <td style="padding: 8px;"><input type="checkbox" class="list-item-checkbox" data-id="${c.camera_id}"></td>
        <td style="padding: 8px; font-family: monospace;">${c.camera_id}</td>
        <td style="padding: 8px;">${c.name} ${c.enabled ? '' : '<span style="color:var(--red-glow); font-size:10px;">(DISABLED)</span>'}</td>
        <td style="padding: 8px;">${c.source_type}</td>
        <td style="padding: 8px; font-size: 11px; max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${c.source}">${c.source}</td>
        <td style="padding: 8px;">${c.floor || '-'} / ${c.zone || '-'}</td>
        <td style="padding: 8px;">
          <button class="btn btn-secondary btn-xs" onclick="window.app.toggleCameraState('${c.camera_id}')">${c.enabled ? 'Disable' : 'Enable'}</button>
        </td>
      </tr>
    `).join('');
  }
  
  async toggleCameraState(cameraId) {
    if (!this.graphCanvas) return;
    const node = this.graphCanvas.nodes.find(n => n.camera_id === cameraId);
    if (node) {
      node.enabled = !node.enabled;
      try {
        await API.saveGraph(this.graphCanvas.toJSON());
        this.refreshListView(this.graphCanvas.toJSON());
      } catch (err) {
        this.showToast(`Failed to toggle: ${err.message}`, 'error');
      }
    }
  }

  async safeQuit() {
    if (this._isShuttingDown) return;
    this._isShuttingDown = true;

    // 1. Immediately clear all polling timers
    if (this.statusPollTimer) {
      clearInterval(this.statusPollTimer);
      this.statusPollTimer = null;
    }
    if (this.galleryPollTimer) {
      clearInterval(this.galleryPollTimer);
      this.galleryPollTimer = null;
    }
    if (this.annoPollTimer) {
      clearInterval(this.annoPollTimer);
      this.annoPollTimer = null;
    }
    if (this.auditLogPollTimer) {
      clearInterval(this.auditLogPollTimer);
      this.auditLogPollTimer = null;
    }

    // 2. Disconnect all live MJPEG video streams immediately to release sockets & server threads
    document.querySelectorAll('.camera-feed-img').forEach(img => {
      img.onerror = null;
      img.src = '';
    });

    // 3. Immediately render clean shutdown screen for instant user feedback
    document.body.innerHTML = `
      <div style="height:100vh;display:flex;align-items:center;justify-content:center;background:#0b0f17;color:#94a3b8;font-family:system-ui,-apple-system,sans-serif;flex-direction:column;gap:14px;text-align:center;">
        <div style="width:52px;height:52px;border-radius:50%;background:rgba(0,242,254,0.1);border:1px solid #00f2fe;display:flex;align-items:center;justify-content:center;color:#00f2fe;font-size:24px;">&#10003;</div>
        <h2 style="color:#f8fafc;margin:0;font-weight:600;font-size:20px;">Argus Surveillance Operations Center Closed</h2>
        <p style="margin:0;max-width:420px;font-size:13px;line-height:1.5;color:#94a3b8;">All camera capture streams, AI models, and background workers have been terminated cleanly.</p>
        <p style="font-size:11px;color:#64748b;margin-top:8px;">You can safely close this browser window or tab.</p>
      </div>
    `;

    // 4. Send quit signal to server
    try {
      await API.quit();
    } catch (err) {
      console.log('Shutdown signal dispatched');
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
      }
      setTimeout(() => {
        if (this.graphCanvas) {
          this.graphCanvas.resize();
          this.loadGraphTopology();
        }
      }, 50);
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
      let badgeText = '⏸️ STANDBY';

      const camStatus = cam.status ? cam.status.toUpperCase() : 'UNKNOWN';

      if (!cam.enabled || camStatus === 'OFFLINE' || camStatus === 'ERROR') {
        badgeClass = 'badge-offline';
        badgeText = '🔴 OFFLINE';
      } else if (isAct) {
        tileClass += ' tile-active';
        badgeClass = 'badge-active';
        badgeText = '🟢 ACTIVE FOCUS';
      } else if (isSearch) {
        tileClass += ' tile-searching';
        badgeClass = 'badge-searching';
        badgeText = '🟡 SEARCHING';
      } else if (camStatus === 'ONLINE') {
        badgeClass = 'badge-standby';
        badgeText = '🟢 LIVE (STANDBY)';
      } else if (camStatus === 'CONNECTING') {
        badgeClass = 'badge-searching';
        badgeText = '🟡 CONNECTING';
      }

      const streamUrl = `${API.getCameraStreamUrl(cam.camera_id)}?t=${Date.now()}`;
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
               alt="${cam.name}">
          
          <div class="camera-tile-overlay">
            <button class="tile-overlay-btn btn-focus-cam" data-cam="${cam.camera_id}">Set Active</button>
            <span style="font-size:10px; color:#94a3b8;">Click: Correct Target | Shift+Click: New Target</span>
            <button class="tile-overlay-btn btn-snap-cam" data-cam="${cam.camera_id}">+ Angle</button>
          </div>
        </div>
      `;

      const stage = tile.querySelector('.camera-tile-video');
      const img = tile.querySelector('.camera-feed-img');
      const header = tile.querySelector('.camera-tile-header');

      // Click on tile header: Instantly focus active camera
      header.addEventListener('click', async () => {
        try {
          await API.setActiveCamera(cam.camera_id);
          this.activeCameraId = cam.camera_id;
          this._pendingActiveCameraSwitch = cam.camera_id;
          this._pendingActiveCameraSwitchTime = Date.now();
          this.updateActiveTileVisuals();
          this.refreshStatus();
          this.showToast(`Focused on '${cam.name || cam.camera_id}'`, 'info');
        } catch (err) {
          this.showToast(`Focus error: ${err.message}`, 'error');
        }
      });

      // Auto-reconnect stream if disconnected
      img.onerror = () => {
        setTimeout(() => {
          if (tile.isConnected) {
            img.src = `${API.baseUrl}/api/camera/${encodeURIComponent(cam.camera_id)}/stream?t=${Date.now()}`;
          }
        }, 1000);
      };

      // Click on tile video stage: Set active camera + select target at coordinates
      stage.addEventListener('click', async (e) => {
        if (e.target.classList.contains('tile-overlay-btn')) return;

        const rect = img.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        // Scale to image actual pixel dimensions
        let imgW = img.naturalWidth;
        let imgH = img.naturalHeight;
        if (!imgW || imgW === 0) {
          imgW = (cam.width && cam.width > 0) ? cam.width : 640;
        }
        if (!imgH || imgH === 0) {
          imgH = (cam.height && cam.height > 0) ? cam.height : 480;
        }
        // Fix P-14: Account for object-fit: contain letterboxing offset
        const imgRatio = imgW / imgH;
        const rectRatio = rect.width / rect.height;
        let renderW = rect.width;
        let renderH = rect.height;
        let offsetX = 0;
        let offsetY = 0;

        if (imgRatio > rectRatio) {
          renderH = rect.width / imgRatio;
          offsetY = (rect.height - renderH) / 2;
        } else {
          renderW = rect.height * imgRatio;
          offsetX = (rect.width - renderW) / 2;
        }

        const clickOnImgX = clickX - offsetX;
        const clickOnImgY = clickY - offsetY;

        // Ignore clicks on black bars
        if (clickOnImgX < 0 || clickOnImgX > renderW || clickOnImgY < 0 || clickOnImgY > renderH) {
          return;
        }

        const scaleX = imgW / renderW;
        const scaleY = imgH / renderH;

        const targetX = clickOnImgX * scaleX;
        const targetY = clickOnImgY * scaleY;
        const startNew = e.shiftKey;

        try {
          const res = await API.selectTarget(cam.camera_id, targetX, targetY, null, startNew);
          if (res && res.selected_id !== null && res.selected_id !== undefined) {
            this.showToast(`Target ${startNew ? 'selected (New)' : 'corrected'}! Tracker ID: ${res.selected_id} on ${cam.name || cam.camera_id}`, 'success');
            this.activeCameraId = cam.camera_id;
            this._pendingActiveCameraSwitch = cam.camera_id;
            this._pendingActiveCameraSwitchTime = Date.now();
            this.updateActiveTileVisuals();
            this.refreshStatus();
            this.refreshGallery();
          } else {
            // Simply switch active camera focus
            await API.setActiveCamera(cam.camera_id);
            this.activeCameraId = cam.camera_id;
            this._pendingActiveCameraSwitch = cam.camera_id;
            this._pendingActiveCameraSwitchTime = Date.now();
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
          this._pendingActiveCameraSwitch = cam.camera_id;
          this._pendingActiveCameraSwitchTime = Date.now();
          this.updateActiveTileVisuals();
          this.refreshStatus();
          this.showToast(`Focused on '${cam.name || cam.camera_id}'`, 'info');
        } catch (err) {
          this.showToast(`Focus error: ${err.message}`, 'error');
        }
      });

      tile.querySelector('.btn-snap-cam').addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
          const res = await API.addSample(cam.camera_id);
          if (res.success) {
            this.showToast(`Angle captured on ${cam.name || cam.camera_id}!`, 'success');
            this.refreshGallery();
          } else {
            this.showToast('No active target locked to add angle', 'error');
          }
        } catch (err) {
          this.showToast(`Failed to capture angle: ${err.message}`, 'error');
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
      t.classList.remove('tile-searching');
      const badge = document.getElementById(`badge-${cid}`);
      if (badge) {
        if (isAct) {
          badge.className = 'tile-cam-badge badge-active';
          badge.textContent = 'ACTIVE FOCUS';
        } else {
          badge.className = 'tile-cam-badge badge-standby';
          badge.textContent = 'STANDBY';
        }
      }
    });
  }

  /* ==========================================================================
     PERIODIC STATUS & GALLERY POLLING
     ========================================================================== */



  async pollAnnotations() {
    try {
      this.annotations = await API.getAnnotations();
    } catch(err) {}
  }

  async pollUndoStack() {
    const stack = await API.getUndoStack();
    const btnUndo = document.getElementById('btn-undo-global');
    const btnRedo = document.getElementById('btn-redo-global');
    if (btnUndo) btnUndo.disabled = !stack.can_undo;
    if (btnRedo) btnRedo.disabled = !stack.can_redo;
  }

  startPolling() {
    console.log('[SSE] Initializing Server-Sent Events for unified telemetry...');
    if (this.telemetrySource) {
      this.telemetrySource.close();
    }
    
    this.telemetrySource = new EventSource('/api/stream/events');
    
    this.telemetrySource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.refreshStatus(data);
        if (data.gallery) {
          this.refreshGallery(data.gallery);
        }
        if (data.topology_version !== undefined && data.topology_version !== this._lastTopologyVersion) {
          if (this._lastTopologyVersion !== undefined) {
            console.log(`[SSE] Topology updated (v${data.topology_version}), reloading grids...`);
            this.loadLiveMatrix();
            if (this.currentMode === 'topology') {
              this.loadGraphTopology();
            }
          }
          this._lastTopologyVersion = data.topology_version;
        }
      } catch (err) {
        console.error('[SSE] Parse error:', err);
      }
    };
    
    this.telemetrySource.onerror = (err) => {
      console.warn('[SSE] Connection lost, retrying automatically...', err);
    };

    if (this.auditLogPollTimer) clearInterval(this.auditLogPollTimer);
    this.updateAuditLog();
    this.auditLogPollTimer = setInterval(() => this.updateAuditLog(), 3000);
  }

  async updateAuditLog() {
    try {
      const res = await API.getAuditLogs(50, 0);
      if (res && res.success && res.logs) {
        const container = document.getElementById('event-log-container');
        if (!container) return;
        
        if (res.logs.length === 0) {
          container.innerHTML = '<div style="color: var(--text-dim);">System initialized. Waiting for events...</div>';
          return;
        }

        const logHtml = res.logs.map(log => {
          const timeStr = new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
          let color = '#94a3b8'; // default
          if (log.event_type.includes('START') || log.event_type.includes('MATCH')) color = '#00f2fe';
          if (log.event_type.includes('LOST') || log.event_type.includes('REJECT') || log.event_type.includes('REMOVE') || log.event_type.includes('CLEAR')) color = 'var(--red-glow)';
          if (log.event_type.includes('ADD') || log.event_type.includes('ACCEPT') || log.event_type.includes('LOCKED')) color = '#10b981';
          if (log.event_type.includes('CORRECTION') || log.event_type.includes('EXPAND')) color = '#f59e0b';
          
          let detStr = log.details;
          if (detStr.length > 60) {
             detStr = detStr.substring(0, 57) + '...';
          }
          
          return `<div style="margin-bottom: 4px; display: flex; gap: 8px;">
            <span style="color: #64748b;">[${timeStr}]</span>
            <span style="color: ${color}; min-width: 130px; font-weight: 600;">${log.event_type}</span>
            <span style="color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title='${log.details}'>${detStr}</span>
          </div>`;
        }).join('');
        
        container.innerHTML = logHtml;
      }
    } catch(err) {
      // ignore
    }
  }

  async refreshStatus(st = null) {
    try {
      if (!st) {
        st = await API.getStatus();
      }
      if (!st) return;

      const pendingGraceMs = 2000;
      if (this._pendingActiveCameraSwitch && (Date.now() - this._pendingActiveCameraSwitchTime) < pendingGraceMs) {
        if (st.active_camera === this._pendingActiveCameraSwitch) {
          this._pendingActiveCameraSwitch = null;
        }
      } else {
        this._pendingActiveCameraSwitch = null;
        this.activeCameraId = st.active_camera;
      }

      this.targetState = st.target_state || 'UNSELECTED';
      this.searchProgress = st.search_progress;

      // Update Header HUD
      this.hdrActiveCam.textContent = st.active_camera || 'None';

      if (this.targetState === 'LOST_PERMANENTLY') {
        this.hdrTargetState.textContent = 'TARGET LOST';
        this.hdrTargetState.className = `chip-val chip-badge state-lost-perm`;
        this.hdrTargetState.style.background = 'var(--red-glow)';
        this.hdrTargetState.style.color = '#fff';
      } else {
        this.hdrTargetState.textContent = this.targetState;
        this.hdrTargetState.className = `chip-val chip-badge state-${this.targetState.toLowerCase()}`;
        this.hdrTargetState.style.background = '';
        this.hdrTargetState.style.color = '';
      }

      // Handle Handoff Confirmation Overlay
      const handoffOverlay = document.getElementById('handoff-overlay');
      const handoffCamId = document.getElementById('handoff-cam-id');
      const handoffSim = document.getElementById('handoff-sim');
      if (this.targetState === 'UNCERTAIN' && st.pending_handoff) {
        if (handoffOverlay) {
          handoffOverlay.style.display = 'flex';
          if (handoffCamId) handoffCamId.textContent = st.pending_handoff.camera_id;
          if (handoffSim) handoffSim.textContent = st.pending_handoff.similarity.toFixed(2);
        }
      } else {
        if (handoffOverlay) {
          handoffOverlay.style.display = 'none';
        }
      }

      const rad = st.search_progress ? st.search_progress.search_radius : 0;
      const searchSt = st.search_progress ? st.search_progress.state.toUpperCase() : 'IDLE';
      this.hdrSearchRadius.textContent = `R = ${rad} (${searchSt})`;

      const gSize = st.gallery ? st.gallery.size : (st.gallery_size || 0);
      const gMax = st.gallery ? st.gallery.max_size : (st.gallery_max || 25);
      this.hdrGalleryStats.textContent = `${gSize} / ${gMax}`;

      // Update Target Summary Card
      this.cardTargetId.textContent = st.target_track_id ? `Tracker #${st.target_track_id}` : (st.target_state !== 'UNSELECTED' ? 'TARGET_0' : 'UNSELECTED');
      this.cardTargetState.textContent = this.targetState;
      this.cardTargetState.className = `state-tag state-${this.targetState.toLowerCase()}`;
      this.cardTargetCam.textContent = st.active_camera || 'None';

      const scores = st.candidate_scores || {};
      const scoreKeys = Object.keys(scores);
      const gMan = st.gallery ? st.gallery.manual_count : (st.gallery_manual || 0);
      const gAuto = st.gallery ? st.gallery.auto_count : (st.gallery_auto || 0);

      if (scoreKeys.length > 0) {
        const scoreStr = scoreKeys.map(k => `#${k}: ${(scores[k]).toFixed(2)}`).join(' | ');
        this.cardTargetSamples.innerHTML = `<span style="color:#00f2fe;font-weight:600;">Sim: ${scoreStr}</span> (${gMan}m/${gAuto}a)`;
      } else {
        this.cardTargetSamples.textContent = `${gMan} manual / ${gAuto} auto`;
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

      // Update Active/Searching tile indicators in the grid
      if (this.cameras.length > 0) {
        const searchingCams = new Set(st.search_progress ? st.search_progress.active_cameras : []);
        this.cameras.forEach((cam) => {
          const tile = document.getElementById(`tile-${cam.camera_id}`);
          const badge = document.getElementById(`badge-${cam.camera_id}`);
          if (!tile || !badge) return;

          const isAct = (cam.camera_id === st.active_camera);
          const isSearch = searchingCams.has(cam.camera_id);

          const camStatus = (st.camera_statuses && st.camera_statuses[cam.camera_id]) ? st.camera_statuses[cam.camera_id].toUpperCase() : 'UNKNOWN';

          tile.classList.toggle('tile-active', isAct);
          tile.classList.toggle('tile-searching', !isAct && isSearch);

          if (!cam.enabled || camStatus === 'OFFLINE' || camStatus === 'ERROR') {
            badge.className = 'tile-cam-badge badge-offline';
            badge.textContent = '🔴 OFFLINE';
          } else if (isAct) {
            badge.className = 'tile-cam-badge badge-active';
            badge.textContent = '🟢 ACTIVE FOCUS';
          } else if (isSearch) {
            badge.className = 'tile-cam-badge badge-searching';
            badge.textContent = `🟡 SEARCHING (R=${rad})`;
          } else if (camStatus === 'ONLINE') {
            badge.className = 'tile-cam-badge badge-standby';
            badge.textContent = '🟢 LIVE (STANDBY)';
          } else if (camStatus === 'CONNECTING') {
            badge.className = 'tile-cam-badge badge-searching';
            badge.textContent = '🟡 CONNECTING';
          } else {
            badge.className = 'tile-cam-badge badge-standby';
            badge.textContent = '⏸️ STANDBY';
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

  async refreshGallery(g = null) {
    try {
      if (!g) {
        g = await API.getGallery();
      }
      if (!g) return;

      this.galleryCountBadge.textContent = `${g.size} / ${g.max_size}`;

      if (!g.thumbnails || g.thumbnails.length === 0) {
        this.galleryCardsList.innerHTML = `
          <div class="gallery-empty-state">
            <div class="empty-icon">&#128100;</div>
            <div class="empty-text">No target locked</div>
            <div class="empty-hint">Shift+Click on any person in a feed to start tracking.</div>
          </div>
        `;
        return;
      }

      if (this.targetState === 'UNSELECTED') {
        const items = [...g.thumbnails].reverse();
        this.galleryCardsList.innerHTML = `
          <div class="gallery-empty-state" style="border: 1px solid rgba(0, 242, 254, 0.3); background: rgba(0, 242, 254, 0.05); margin-bottom: 12px; height: auto; padding: 16px;">
            <div class="empty-icon" style="color: #00f2fe;">&#128100;</div>
            <div class="empty-text" style="color: #00f2fe; margin-bottom: 8px;">Previous Gallery Loaded</div>
            <div class="empty-hint" style="color: #94a3b8; font-size: 11px;">Click on a person in any feed to resume tracking, or clear gallery.</div>
            <button class="btn btn-danger btn-sm" style="margin-top: 12px;" onclick="window.clearTargetFn && window.clearTargetFn()">Clear Gallery</button>
          </div>
          <div style="opacity: 0.5; pointer-events: none;">
            ${items.map(t => this.renderGalleryCard(t)).join('')}
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
