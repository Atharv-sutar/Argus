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

    // Auto-discover webcams on initial load
    setTimeout(() => this.discover(), 300);

    const customForm = document.getElementById('form-custom-camera');
    customForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const id = document.getElementById('custom-cam-id').value.trim();
      const name = document.getElementById('custom-cam-name').value.trim();
      const type = document.getElementById('custom-cam-type').value;
      const source = document.getElementById('custom-cam-source').value.trim();

      if (!id || !name || !source) return;

      if (!this.app.graphCanvas) {
        this.app.graphCanvas = new GraphCanvas(this.app, 'graph-canvas');
      }

      this.app.graphCanvas.addNode({
        camera_id: id,
        name: name,
        source: isNaN(source) ? source : parseInt(source, 10),
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
    const btn1 = document.getElementById('btn-discover');
    const btn2 = document.getElementById('btn-discover-toolbar');
    if (btn1) { btn1.disabled = true; btn1.textContent = 'Scanning...'; }
    if (btn2) { btn2.disabled = true; }

    this.listEl.innerHTML = '<div class="empty-state"><span class="spinner"></span> Scanning video capture devices...</div>';
    try {
      const data = await API.discoverCameras();
      this.discoveredCameras = data.cameras || [];
      this.renderList();
      this.app.showToast(`Detected ${this.discoveredCameras.length} camera source(s)`, 'info');
    } catch (err) {
      this.listEl.innerHTML = `<div class="empty-state">Discovery failed: ${err.message}</div>`;
      this.app.showToast('Discovery failed', 'error');
    } finally {
      if (btn1) { btn1.disabled = false; btn1.textContent = 'Auto-Detect'; }
      if (btn2) { btn2.disabled = false; }
    }
  }

  renderList() {
    if (this.discoveredCameras.length === 0) {
      this.listEl.innerHTML = '<div class="empty-state">No local webcams detected. You can add RTSP or synthetic sources below.</div>';
      return;
    }

    this.listEl.innerHTML = '';
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
