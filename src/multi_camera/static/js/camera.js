/**
 * Camera Discovery & Preview Management
 */
class CameraManager {
  constructor(app) {
    this.app = app;
    this.discoveredCameras = [];
    this.selectedPreviewSource = null;

    this.listEl = document.getElementById('discovered-list');
    this.previewImg = document.getElementById('preview-image');
    this.previewPlaceholder = document.querySelector('#pane-topo-discover .preview-placeholder');
    this.previewInfo = document.getElementById('preview-info');

    this.initTabs();
    this.initPresets();
    this.initStreamProbe();
    this.initEvents();
  }

  initTabs() {
    const tabs = [
      { btnId: 'tab-topo-provision', paneId: 'pane-topo-provision' },
      { btnId: 'tab-topo-inspect', paneId: 'pane-topo-inspect' },
      { btnId: 'tab-topo-discover', paneId: 'pane-topo-discover' },
    ];

    tabs.forEach(({ btnId, paneId }) => {
      const btn = document.getElementById(btnId);
      if (btn) {
        btn.addEventListener('click', () => {
          tabs.forEach(t => {
            const b = document.getElementById(t.btnId);
            const p = document.getElementById(t.paneId);
            if (b) b.classList.remove('active');
            if (p) p.style.display = 'none';
          });
          btn.classList.add('active');
          const pane = document.getElementById(paneId);
          if (pane) pane.style.display = 'block';

          if (btnId === 'tab-topo-provision') {
            this.updateConnectDropdown();
          }
        });
      }
    });

    // Populate connect dropdown initially
    setTimeout(() => this.updateConnectDropdown(), 200);
  }

  switchTab(tabKey) {
    const map = {
      'provision': { btnId: 'tab-topo-provision', paneId: 'pane-topo-provision' },
      'inspect': { btnId: 'tab-topo-inspect', paneId: 'pane-topo-inspect' },
      'discover': { btnId: 'tab-topo-discover', paneId: 'pane-topo-discover' },
    };
    const target = map[tabKey];
    if (!target) return;

    ['tab-topo-provision', 'tab-topo-inspect', 'tab-topo-discover'].forEach(id => {
      const b = document.getElementById(id);
      if (b) b.classList.remove('active');
    });
    ['pane-topo-provision', 'pane-topo-inspect', 'pane-topo-discover'].forEach(id => {
      const p = document.getElementById(id);
      if (p) p.style.display = 'none';
    });

    const activeBtn = document.getElementById(target.btnId);
    const activePane = document.getElementById(target.paneId);
    if (activeBtn) activeBtn.classList.add('active');
    if (activePane) activePane.style.display = 'block';

    if (tabKey === 'provision') {
      this.updateConnectDropdown();
    }
  }

  initPresets() {
    const presetButtons = document.querySelectorAll('.btn-preset[data-preset]');
    const idInput = document.getElementById('custom-cam-id');
    const nameInput = document.getElementById('custom-cam-name');
    const typeSelect = document.getElementById('custom-cam-type');
    const sourceInput = document.getElementById('custom-cam-source');

    const presets = {
      webcam: {
        idSuffix: 'webcam_0',
        name: 'USB Webcam 0',
        type: 'webcam',
        source: '0',
      },
      rtsp: {
        idSuffix: 'rtsp_1',
        name: 'RTSP Corridor North',
        type: 'rtsp',
        source: 'rtsp://192.168.1.100:554/live',
      },
      android: {
        idSuffix: 'mobile_1',
        name: 'Android IP Cam',
        type: 'rtsp',
        source: 'http://192.168.1.50:8080/video',
      },
      video: {
        idSuffix: 'video_1',
        name: 'Recorded Test Feed',
        type: 'video_file',
        source: 'test_footage/cam_0.mp4',
      },
      synthetic: {
        idSuffix: 'sim_1',
        name: 'Synthetic Corridor',
        type: 'synthetic',
        source: 'synthetic:hallway',
      },
    };

    presetButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        presetButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const presetKey = btn.dataset.preset;
        const config = presets[presetKey];
        if (!config) return;

        const count = this.app.graphCanvas ? this.app.graphCanvas.nodes.length : 0;
        if (idInput) idInput.value = `cam_${config.idSuffix}`;
        if (nameInput) nameInput.value = config.name;
        if (typeSelect) typeSelect.value = config.type;
        if (sourceInput) sourceInput.value = config.source;

        // Reset stream probe status
        const statusEl = document.getElementById('stream-probe-status');
        if (statusEl) statusEl.textContent = `Preset loaded: ${config.name}. Ready to probe.`;
      });
    });
  }

  initStreamProbe() {
    const btnTest = document.getElementById('btn-test-stream');
    const previewBox = document.getElementById('add-cam-preview-box');
    const previewImg = document.getElementById('add-cam-preview-img');
    const previewPlaceholder = document.getElementById('add-cam-preview-placeholder');
    const statusEl = document.getElementById('stream-probe-status');

    if (!btnTest) return;

    btnTest.addEventListener('click', async () => {
      const sourceInput = document.getElementById('custom-cam-source');
      const typeSelect = document.getElementById('custom-cam-type');
      if (!sourceInput || !typeSelect) return;

      let source = sourceInput.value.trim();
      let type = typeSelect.value;
      if (!source) {
        if (statusEl) statusEl.textContent = 'Please specify a source URI or index first.';
        return;
      }

      // Normalization
      if (/^https?:\/\//i.test(source)) {
        if (type === 'webcam') type = 'rtsp';
      }

      btnTest.disabled = true;
      btnTest.textContent = 'Probing...';
      if (statusEl) statusEl.textContent = 'Connecting to stream endpoint...';

      if (previewBox) previewBox.style.display = 'block';
      if (previewPlaceholder) {
        previewPlaceholder.style.display = 'block';
        previewPlaceholder.textContent = `Testing feed: ${source}...`;
      }
      if (previewImg) previewImg.style.display = 'none';

      const url = `${API.getPreviewUrl(source, type)}&t=${Date.now()}`;

      if (previewImg) {
        previewImg.onload = () => {
          previewImg.style.display = 'block';
          if (previewPlaceholder) previewPlaceholder.style.display = 'none';
          if (statusEl) {
            statusEl.innerHTML = '<span style="color: var(--emerald-bright); font-weight: 600;">🟢 Stream Active // Feed Verified Online</span>';
          }
          btnTest.disabled = false;
          btnTest.textContent = '⚡ Probe Feed';
        };

        previewImg.onerror = () => {
          previewImg.style.display = 'none';
          if (previewPlaceholder) {
            previewPlaceholder.style.display = 'block';
            previewPlaceholder.textContent = 'Feed Offline or Unreachable';
          }
          if (statusEl) {
            statusEl.innerHTML = '<span style="color: var(--border-danger); font-weight: 600;">🔴 Stream Inactive // Device Offline</span>';
          }
          btnTest.disabled = false;
          btnTest.textContent = '⚡ Probe Feed';
        };

        previewImg.src = url;
      }
    });
  }

  updateConnectDropdown() {
    const targetSelect = document.getElementById('custom-cam-connect-target');
    const linkProps = document.getElementById('custom-cam-link-props');
    if (!targetSelect) return;

    const currentVal = targetSelect.value;
    targetSelect.innerHTML = '<option value="">-- None (Isolated Node) --</option>';

    const nodes = (this.app.graphCanvas && this.app.graphCanvas.nodes && this.app.graphCanvas.nodes.length > 0)
      ? this.app.graphCanvas.nodes
      : (this.app.cameras || []);

    nodes.forEach(node => {
      const opt = document.createElement('option');
      opt.value = node.camera_id;
      opt.textContent = `${node.name || node.camera_id} (${node.camera_id})`;
      targetSelect.appendChild(opt);
    });

    targetSelect.value = currentVal;
    if (linkProps) {
      linkProps.style.display = targetSelect.value ? 'block' : 'none';
    }

    targetSelect.onchange = () => {
      if (linkProps) {
        linkProps.style.display = targetSelect.value ? 'block' : 'none';
      }
    };
  }

  initEvents() {
    const btnDiscover = document.getElementById('btn-discover');
    if (btnDiscover) {
      btnDiscover.addEventListener('click', () => this.discover());
    }

    const btnDiscoverToolbar = document.getElementById('btn-discover-toolbar');
    if (btnDiscoverToolbar) {
      btnDiscoverToolbar.addEventListener('click', () => this.discover());
    }

    // Auto-discover webcams after initial page load (deferred to avoid blocking
    // the HTTP server during critical early API requests like topology graph load)
    // DISABLED: Users prefer the webcam is not accessed automatically on startup.
    // setTimeout(() => this.discover(), 5000);

    const customForm = document.getElementById('form-custom-camera');
    customForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const id = document.getElementById('custom-cam-id').value.trim();
      const name = document.getElementById('custom-cam-name').value.trim();
      let type = document.getElementById('custom-cam-type').value;
      let source = document.getElementById('custom-cam-source').value.trim();

      if (!id || !name || !source) return;

      // Auto-detect and normalize HTTP/HTTPS network streams (e.g. Android IP Webcam)
      if (/^https?:\/\//i.test(source)) {
        if (type === 'webcam') {
          type = 'rtsp';
        }
        try {
          const parsed = new URL(source);
          if (parsed.pathname === '/' || parsed.pathname === '') {
            parsed.pathname = '/video';
            source = parsed.toString();
          }
        } catch (_) {
          if (source.endsWith(':8080') || source.endsWith(':8080/')) {
            source = source.replace(/\/?$/, '/video');
          }
        }
      } else if (/^rtsp:\/\//i.test(source)) {
        if (type === 'webcam') {
          type = 'rtsp';
        }
      }

      if (!this.app.graphCanvas) {
        this.app.graphCanvas = new GraphCanvas(this.app, 'graph-canvas');
      }

      // Ensure graph is loaded if canvas was newly created
      if (this.app.graphCanvas.nodes.length === 0) {
        try {
          const existingData = await API.getGraph();
          if (existingData && existingData.cameras && existingData.cameras.length > 0) {
            this.app.graphCanvas.loadGraph(existingData);
          }
        } catch (_) {}
      }

      const connectTargetId = document.getElementById('custom-cam-connect-target')?.value;
      const edgeType = document.getElementById('custom-cam-edge-type')?.value || 'adjacent';
      const transitS = parseFloat(document.getElementById('custom-cam-transit-s')?.value) || 4.5;

      // Intelligent Positioning in current visible viewport
      let posX = undefined;
      let posY = undefined;

      let targetNode = null;
      if (connectTargetId && this.app.graphCanvas.nodes) {
        targetNode = this.app.graphCanvas.nodes.find(n => n.camera_id === connectTargetId);
        if (targetNode) {
          const angle = Math.random() * Math.PI * 2;
          posX = Math.round(targetNode.position_x + Math.cos(angle) * 160);
          posY = Math.round(targetNode.position_y + Math.sin(angle) * 160);
        }
      }

      if (posX === undefined || posY === undefined) {
        const rect = this.app.graphCanvas.canvas.getBoundingClientRect();
        const center = this.app.graphCanvas.screenToWorld(
          (rect.width > 0 ? rect.width : this.app.graphCanvas.canvas.width) / 2,
          (rect.height > 0 ? rect.height : this.app.graphCanvas.canvas.height) / 2
        );
        posX = Math.round(center.x + (Math.random() - 0.5) * 60);
        posY = Math.round(center.y + (Math.random() - 0.5) * 60);
      }

      const newNode = {
        camera_id: id,
        name: name,
        source: (source !== '' && !isNaN(source) && String(parseInt(source, 10)) === source) ? parseInt(source, 10) : source,
        source_type: type,
        enabled: true,
        position_x: posX,
        position_y: posY,
      };

      this.app.graphCanvas.addNode(newNode);

      // Add connecting edge if target selected
      if (targetNode) {
        this.app.graphCanvas.addEdge({
          source_camera_id: targetNode.camera_id,
          target_camera_id: id,
          source: targetNode.camera_id,
          target: id,
          edge_type: edgeType,
          direction: 'bidirectional',
          enabled: true,
          expected_typical_transition_s: transitS,
          expected_min_transition_s: Math.max(0.5, transitS - 1.5),
          expected_max_transition_s: transitS + 2.5,
        });
        this.app.showToast(`Added '${name}' and connected to '${targetNode.name}' (⏱ ${transitS}s transit)`, 'success');
      } else {
        this.app.showToast(`Added camera '${name}' to topology map`, 'success');
      }

      // Automatically persist to backend so the camera node is permanently saved
      try {
        await API.saveGraph(this.app.graphCanvas.toJSON());
      } catch (saveErr) {
        console.warn('Auto-save graph warning:', saveErr);
      }

      this.app.graphCanvas.panToNode(newNode);
      this.updateConnectDropdown();
      if (typeof this.app.refreshListView === 'function') {
        this.app.refreshListView(this.app.graphCanvas.toJSON());
      }
      if (typeof this.app.loadLiveMatrix === 'function') {
        this.app.loadLiveMatrix();
      }
      customForm.reset();
    });
  }

  async discover() {
    if (this.app && this.app._isShuttingDown) return;
    const btn1 = document.getElementById('btn-discover');
    const btn2 = document.getElementById('btn-discover-toolbar');
    
    let startIdx = 0;
    let endIdx = 4;
    const startInput = document.getElementById('discover-start');
    const endInput = document.getElementById('discover-end');
    if (startInput) {
      const parsedStart = parseInt(startInput.value, 10);
      if (!isNaN(parsedStart) && parsedStart >= 0) startIdx = parsedStart;
    }
    if (endInput) {
      const parsedEnd = parseInt(endInput.value, 10);
      if (!isNaN(parsedEnd) && parsedEnd > startIdx) endIdx = parsedEnd;
    }

    if (btn1) { btn1.disabled = true; btn1.textContent = 'Scanning...'; }
    if (btn2) { btn2.disabled = true; }

    this.listEl.innerHTML = '<div class="empty-state"><span class="spinner"></span> Scanning video capture devices...</div>';
    try {
      const data = await API.discoverCameras(startIdx, endIdx);
      if (this.app && this.app._isShuttingDown) return;
      this.discoveredCameras = data.cameras || [];
      this.renderList();
      try {
        if (this.app && typeof this.app.showToast === 'function') {
          this.app.showToast(`Detected ${this.discoveredCameras.length} camera source(s)`, 'info');
        }
      } catch (toastErr) {
        console.warn('Toast failed:', toastErr);
      }
    } catch (err) {
      if (this.app && this.app._isShuttingDown) return;
      const msg = err.name === 'AbortError' ? 'Scan timed out after 15s' : err.message;
      this.listEl.innerHTML = `<div class="empty-state">Discovery failed: ${msg}</div>`;
      try {
        if (this.app && typeof this.app.showToast === 'function') {
          this.app.showToast(`Discovery failed: ${msg}`, 'error');
        }
      } catch (_) {}
    } finally {
      if (btn1 && (!this.app || !this.app._isShuttingDown)) { btn1.disabled = false; btn1.textContent = 'Probe'; }
      if (btn2 && (!this.app || !this.app._isShuttingDown)) { btn2.disabled = false; }
    }
  }

  renderList() {
    if (this.discoveredCameras.length === 0) {
      this.listEl.innerHTML = '<div class="empty-state">No local webcams detected. You can add RTSP or synthetic sources below.</div>';
      return;
    }

    this.listEl.innerHTML = '';

    // Quick Action Bar: Apply all discovered cameras to live surveillance
    const headerAction = document.createElement('div');
    headerAction.className = 'discovered-actions-bar';
    headerAction.style.cssText = 'padding: 0 0 10px 0; display: flex; flex-direction: column; gap: 6px;';
    headerAction.innerHTML = `
      <button class="btn-apply-all-cams" style="width: 100%; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 6px; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #020617; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; transition: opacity 0.2s;">
        <span>⚡</span> Apply Discovered (${this.discoveredCameras.length}) to Live Matrix
      </button>
    `;
    headerAction.querySelector('.btn-apply-all-cams').addEventListener('click', async (e) => {
      e.stopPropagation();
      const btn = e.currentTarget;
      btn.disabled = true;
      btn.textContent = 'Applying to Live Matrix...';
      try {
        const res = await API.applyDiscoveredCameras(this.discoveredCameras);
        if (res && res.success) {
          if (this.app && typeof this.app.showToast === 'function') {
            this.app.showToast(`Applied ${res.applied || this.discoveredCameras.length} hardware camera(s) to Live Matrix!`, 'success');
          }
          if (this.app && typeof this.app.setMode === 'function') {
            this.app.setMode('matrix');
          }
          if (this.app && typeof this.app.loadLiveMatrix === 'function') {
            await this.app.loadLiveMatrix();
          }
          if (this.app && this.app.graphCanvas) {
            const graphData = await API.getGraph();
            this.app.graphCanvas.loadGraph(graphData);
          }
        }
      } catch (err) {
        if (this.app && typeof this.app.showToast === 'function') {
          this.app.showToast(`Failed to apply cameras: ${err.message}`, 'error');
        }
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<span>⚡</span> Apply Discovered (${this.discoveredCameras.length}) to Live Matrix`;
      }
    });
    this.listEl.appendChild(headerAction);
    this.discoveredCameras.forEach((cam) => {
      const card = document.createElement('div');
      card.className = 'camera-card';
      card.innerHTML = `
        <div class="cam-meta">
          <span class="cam-name">${cam.name}</span>
          <span class="cam-desc">${cam.width}x${cam.height} @ ${cam.fps}fps</span>
        </div>
        <button class="cam-add-btn">+ Add</button>
      `;

      card.addEventListener('click', (e) => {
        if (!e.target.classList.contains('cam-add-btn')) {
          this.preview(cam.source, cam.source_type, cam.name);
          document.querySelectorAll('.camera-card').forEach(c => c.classList.remove('selected'));
          card.classList.add('selected');
        }
      });

      card.querySelector('.cam-add-btn').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!this.app.graphCanvas) {
          this.app.graphCanvas = new GraphCanvas(this.app, 'graph-canvas');
        }

        // Ensure graph is loaded if canvas was newly created
        if (this.app.graphCanvas.nodes.length === 0) {
          try {
            const existingData = await API.getGraph();
            if (existingData && existingData.cameras && existingData.cameras.length > 0) {
              this.app.graphCanvas.loadGraph(existingData);
            }
          } catch (_) {}
        }

        const rect = this.app.graphCanvas.canvas.getBoundingClientRect();
        const center = this.app.graphCanvas.screenToWorld(
          (rect.width > 0 ? rect.width : this.app.graphCanvas.canvas.width) / 2,
          (rect.height > 0 ? rect.height : this.app.graphCanvas.canvas.height) / 2
        );

        const camId = `cam_${cam.source}`;
        const newNode = {
          camera_id: camId,
          name: cam.name,
          source: cam.source,
          source_type: cam.source_type,
          enabled: true,
          position_x: Math.round(center.x + (Math.random() - 0.5) * 60),
          position_y: Math.round(center.y + (Math.random() - 0.5) * 60),
        };

        this.app.graphCanvas.addNode(newNode);

        // Auto-save graph to backend
        try {
          await API.saveGraph(this.app.graphCanvas.toJSON());
        } catch (saveErr) {
          console.warn('Auto-save graph warning:', saveErr);
        }

        this.app.graphCanvas.panToNode(newNode);
        this.updateConnectDropdown();
        if (typeof this.app.refreshListView === 'function') {
          this.app.refreshListView(this.app.graphCanvas.toJSON());
        }
        if (typeof this.app.loadLiveMatrix === 'function') {
          this.app.loadLiveMatrix();
        }
        this.app.showToast(`Added ${cam.name} to topology map`, 'success');
      });

      this.listEl.appendChild(card);
    });
  }

  preview(source, sourceType, name) {
    this.selectedPreviewSource = { source, sourceType, name };
    const url = `${API.getPreviewUrl(source, sourceType)}&t=${Date.now()}`;

    this.previewImg.onload = () => {
      this.previewImg.style.display = 'block';
      this.previewPlaceholder.style.display = 'none';
      this.previewInfo.textContent = `${name} (${sourceType}: ${source})`;
    };

    this.previewImg.onerror = () => {
      this.previewImg.style.display = 'none';
      this.previewPlaceholder.style.display = 'block';
      this.previewPlaceholder.textContent = 'Preview unavailable';
      this.previewInfo.textContent = `${name} (Offline / In Use)`;
    };

    this.previewImg.src = url;
  }
}
