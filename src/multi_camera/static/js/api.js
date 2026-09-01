/**
 * Argus Surveillance REST API & Stream Client
 */
const API = {
  baseUrl: '',

  async getGraph() {
    try {
      const res = await fetch(`${this.baseUrl}/api/graph`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Failed to load graph (${res.status} ${res.statusText})`);
      }
      return await res.json();
    } catch (err) {
      console.error('[API] getGraph error:', err);
      throw err;
    }
  },

  async saveGraph(graphData) {
    try {
      const res = await fetch(`${this.baseUrl}/api/graph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graphData),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.errors ? data.errors.join(', ') : (data.error || `Failed to save graph (${res.status})`));
      return data;
    } catch (err) {
      console.error('[API] saveGraph error:', err);
      throw err;
    }
  },

  async validateGraph(graphData) {
    try {
      const res = await fetch(`${this.baseUrl}/api/graph/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graphData),
      });
      return await res.json();
    } catch (err) {
      console.error('[API] validateGraph error:', err);
      throw err;
    }
  },

  async discoverCameras() {
    try {
      const res = await fetch(`${this.baseUrl}/api/cameras/discover`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Discovery failed (${res.status} ${res.statusText})`);
      }
      return await res.json();
    } catch (err) {
      console.error('[API] discoverCameras error:', err);
      throw err;
    }
  },

  async getLiveCameras() {
    try {
      const res = await fetch(`${this.baseUrl}/api/cameras/live`);
      if (!res.ok) return { cameras: [], active_camera: null };
      return await res.json();
    } catch (err) {
      console.debug('[API] getLiveCameras error:', err);
      return { cameras: [], active_camera: null };
    }
  },

  async restartCameras() {
    try {
      const res = await fetch(`${this.baseUrl}/api/cameras/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `Failed to restart cameras (${res.status})`);
      return data;
    } catch (err) {
      console.error('[API] restartCameras error:', err);
      throw err;
    }
  },

  async setActiveCamera(cameraId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/camera/select_active`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: cameraId }),
      });
      return await res.json();
    } catch (err) {
      console.error('[API] setActiveCamera error:', err);
      throw err;
    }
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
    try {
      const res = await fetch(`${this.baseUrl}/api/target/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      return await res.json();
    } catch (err) {
      console.error('[API] selectTarget error:', err);
      throw err;
    }
  },

  async getStatus() {
    try {
      const res = await fetch(`${this.baseUrl}/api/status`);
      if (!res.ok) return null;
      return await res.json();
    } catch (err) {
      return null;
    }
  },

  async getGallery() {
    try {
      const res = await fetch(`${this.baseUrl}/api/target/gallery`);
      if (!res.ok) return null;
      return await res.json();
    } catch (err) {
      return null;
    }
  },

  async addSample(cameraId = null) {
    try {
      const res = await fetch(`${this.baseUrl}/api/target/add_sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: cameraId }),
      });
      return await res.json();
    } catch (err) {
      console.error('[API] addSample error:', err);
      throw err;
    }
  },

  async clearTarget() {
    try {
      const res = await fetch(`${this.baseUrl}/api/target/clear`, {
        method: 'POST',
      });
      return await res.json();
    } catch (err) {
      console.error('[API] clearTarget error:', err);
      throw err;
    }
  },

  async deleteGalleryEntry(entryId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/target/gallery/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entry_id: entryId }),
      });
      return await res.json();
    } catch (err) {
      console.error('[API] deleteGalleryEntry error:', err);
      throw err;
    }
  },

  async quit() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200);
      const res = await fetch(`${this.baseUrl}/api/system/quit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      return await res.json();
    } catch (err) {
      console.log('[API] Quit command dispatched:', err);
      return { success: true };
    }
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
