/**
 * Argus Surveillance REST API & Stream Client
 */
const API = {
  baseUrl: '',

  async getGraph() {
    const attempt = async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      try {
        const res = await fetch(`${this.baseUrl}/api/graph`, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          throw new Error(data.error || `Failed to load graph (${res.status} ${res.statusText})`);
        }
        return await res.json();
      } catch (err) {
        clearTimeout(timeoutId);
        throw err;
      }
    };

    try {
      return await attempt();
    } catch (firstErr) {
      // Retry once after 1s delay (server may be busy with camera probe)
      console.warn('[API] getGraph first attempt failed, retrying in 1s...', firstErr.message);
      await new Promise(r => setTimeout(r, 1000));
      try {
        return await attempt();
      } catch (retryErr) {
        console.error('[API] getGraph retry also failed:', retryErr);
        throw retryErr;
      }
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

  async selectTarget(cameraId, x = null, y = null, trackId = null, startNew = false) {
    const payload = { camera_id: cameraId, start_new: startNew };
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

  async getAuditLogs(limit = 100, offset = 0) {
    try {
      const res = await fetch(`${this.baseUrl}/api/audit/log?limit=${limit}&offset=${offset}`);
      if (!res.ok) return { success: false, logs: [] };
      return await res.json();
    } catch (err) {
      console.error('[API] getAuditLogs error:', err);
      return { success: false, logs: [] };
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



  async getAnnotations(cameraId = null) {
    let url = `${this.baseUrl}/api/annotations`;
    if (cameraId) {
      url += `?camera_id=${encodeURIComponent(cameraId)}`;
    }
    try {
      const res = await fetch(url);
      if (!res.ok) return [];
      return await res.json();
    } catch (err) {
      return [];
    }
  },

  async createAnnotation(cameraId, timestampMs, text, bbox = null, annotationType = 'note') {
    try {
      const res = await fetch(`${this.baseUrl}/api/annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: cameraId,
          timestamp_ms: timestampMs,
          text: text,
          bbox: bbox,
          annotation_type: annotationType
        })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to create annotation');
      return data;
    } catch (err) {
      throw err;
    }
  },

  async updateAnnotation(annotationId, text, annotationType = null) {
    try {
      const payload = { text };
      if (annotationType) payload.annotation_type = annotationType;
      
      const res = await fetch(`${this.baseUrl}/api/annotations/${annotationId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to update annotation');
      return data;
    } catch (err) {
      throw err;
    }
  },

  async deleteAnnotation(annotationId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/annotations/${annotationId}`, {
        method: 'DELETE'
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to delete annotation');
      return data;
    } catch (err) {
      throw err;
    }
  },


  async getCases() {
    try {
      const res = await fetch(`${this.baseUrl}/api/cases`);
      if (!res.ok) return {cases: [], active_case: null};
      return await res.json();
    } catch (err) {
      return {cases: [], active_case: null};
    }
  },

  async createCase(caseId, mode, operator, notes) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cases`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: caseId,
          mode: mode,
          operator: operator,
          notes: notes
        })
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to create case');
      return data;
    } catch (err) {
      throw err;
    }
  },

  async openCase(caseId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cases/${caseId}/open`, {
        method: 'POST'
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to open case');
      return data;
    } catch (err) {
      throw err;
    }
  },

  async closeCase(caseId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cases/${caseId}/close`, {
        method: 'POST'
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to close case');
      return data;
    } catch (err) {
      throw err;
    }
  },

  async deleteCase(caseId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cases/${caseId}`, {
        method: 'DELETE'
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to delete case');
      return data;
    } catch (err) {
      throw err;
    }
  },


  async startExport(caseId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cases/${caseId}/export`, {
        method: 'POST'
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) throw new Error(data.error || 'Failed to start export');
      return data;
    } catch (err) {
      throw err;
    }
  },

  async getExportStatus(caseId) {
    try {
      const res = await fetch(`${this.baseUrl}/api/cases/${caseId}/export/status`);
      if (!res.ok) throw new Error('Failed to fetch status');
      return await res.json();
    } catch (err) {
      throw err;
    }
  },


  async startFastScan(skipZones = []) {
    try {
      const res = await fetch(`${this.baseUrl}/api/playback/fast-scan/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ skip_zones: skipZones })
      });
      return await res.json();
    } catch (err) {
      throw err;
    }
  },

  async pauseFastScan() {
    try {
      const res = await fetch(`${this.baseUrl}/api/playback/fast-scan/pause`, {
        method: 'POST'
      });
      return await res.json();
    } catch (err) {
      throw err;
    }
  },

  async getPlaybackStats() {
    try {
      const res = await fetch(`${this.baseUrl}/api/playback/stats`);
      return await res.json();
    } catch (err) {
      throw err;
    }
  },

  async switchSource(cameraId, mode) {
    try {
      const res = await fetch(`${this.baseUrl}/api/system/switch_source`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_id: cameraId, mode: mode })
      });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('[API] switchSource error:', err);
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
  },

    // Settings API
    async getSettings() {
        try {
            const res = await fetch('/api/settings');
            if (!res.ok) throw new Error('Settings fetch failed');
            return await res.json();
        } catch (e) {
            console.error('getSettings error:', e);
            return null;
        }
    },

    async updateSettings(settings) {
        try {
            const res = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            return await res.json();
        } catch (e) {
            console.error('updateSettings error:', e);
            return { success: false, error: e.message };
        }
    }

};
