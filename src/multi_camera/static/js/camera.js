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
    this.previewPlaceholder = document.querySelector('.preview-placeholder');
    this.previewInfo = document.getElementById('preview-info');

    this.initEvents();
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
    customForm.addEventListener('submit', (e) => {
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

      this.app.graphCanvas.addNode({
        camera_id: id,
        name: name,
        source: (source !== '' && !isNaN(source) && String(parseInt(source, 10)) === source) ? parseInt(source, 10) : source,
        source_type: type,
        enabled: true,
        position_x: 200 + Math.random() * 100,
        position_y: 150 + Math.random() * 100,
      });

      this.app.showToast(`Added camera '${name}' to canvas`, 'success');
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

      card.querySelector('.cam-add-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        if (!this.app.graphCanvas) {
          this.app.graphCanvas = new GraphCanvas(this.app, 'graph-canvas');
        }
        const camId = `cam_${cam.source}`;
        this.app.graphCanvas.addNode({
          camera_id: camId,
          name: cam.name,
          source: cam.source,
          source_type: cam.source_type,
          enabled: true,
          position_x: 200 + Math.random() * 100,
          position_y: 150 + Math.random() * 100,
        });
        this.app.showToast(`Added ${cam.name} to canvas`, 'success');
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
