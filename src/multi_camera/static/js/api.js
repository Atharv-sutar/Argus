/**
 * Argus Surveillance REST API & Stream Client
 */
const API = {
  baseUrl: '',

  async getGraph() {
    const res = await fetch(`${this.baseUrl}/api/graph`);
    if (!res.ok) throw new Error(`Failed to load graph: ${res.statusText}`);
    return await res.json();
  },

  async saveGraph(graphData) {
    const res = await fetch(`${this.baseUrl}/api/graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graphData),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.errors ? data.errors.join(', ') : (data.error || 'Failed to save graph'));
    return data;
  },

  async validateGraph(graphData) {
    const res = await fetch(`${this.baseUrl}/api/graph/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graphData),
    });
    return await res.json();
  },

  async discoverCameras() {
    const res = await fetch(`${this.baseUrl}/api/cameras/discover`);
    if (!res.ok) throw new Error('Failed to discover cameras');
    return await res.json();
  },

  async getLiveCameras() {
    const res = await fetch(`${this.baseUrl}/api/cameras/live`);
    if (!res.ok) return { cameras: [], active_camera: null };
    return await res.json();
  },

  async restartCameras() {
    const res = await fetch(`${this.baseUrl}/api/cameras/restart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to restart cameras');
    return data;
  },

  async setActiveCamera(cameraId) {
    const res = await fetch(`${this.baseUrl}/api/camera/select_active`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: cameraId }),
    });
    return await res.json();
  },

  async selectTarget(cameraId, x = null, y = null, trackId = null) {
    const payload = { camera_id: cameraId };
    if (x !== null && y !== null) {
      payload.x = x;
      payload.y = y;
    }
    if (trackId !== null) {
      payload.track_id = trackId;
    }
    const res = await fetch(`${this.baseUrl}/api/target/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return await res.json();
  },

  async getStatus() {
    const res = await fetch(`${this.baseUrl}/api/status`);
    if (!res.ok) return null;
    return await res.json();
  },

  async getGallery() {
    const res = await fetch(`${this.baseUrl}/api/target/gallery`);
    if (!res.ok) return null;
    return await res.json();
  },

  async addSample(cameraId = null) {
    const res = await fetch(`${this.baseUrl}/api/target/add_sample`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: cameraId }),
    });
    return await res.json();
  },

  async clearTarget() {
    const res = await fetch(`${this.baseUrl}/api/target/clear`, {
      method: 'POST',
    });
    return await res.json();
  },

  async deleteGalleryEntry(entryId) {
    const res = await fetch(`${this.baseUrl}/api/target/gallery/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entry_id: entryId }),
    });
    return await res.json();
  },

  async quit() {
    const res = await fetch(`${this.baseUrl}/api/system/quit`, {
      method: 'POST',
    });
    return await res.json();
  },

  getCameraStreamUrl(cameraId) {
    return `${this.baseUrl}/api/camera/${encodeURIComponent(cameraId)}/stream`;
  },

  getCameraFrameUrl(cameraId) {
    return `${this.baseUrl}/api/camera/${encodeURIComponent(cameraId)}/frame.jpg?t=${Date.now()}`;
  },

  getPreviewUrl(source, sourceType) {
    return `${this.baseUrl}/api/preview?source=${encodeURIComponent(source)}&type=${encodeURIComponent(sourceType)}&t=${Date.now()}`;
  }
};
