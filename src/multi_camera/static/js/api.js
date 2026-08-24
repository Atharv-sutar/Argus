/**
 * Argus Mapping REST API Client
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

  async getStatus() {
    const res = await fetch(`${this.baseUrl}/api/status`);
    if (!res.ok) return null;
    return await res.json();
  },

  getPreviewUrl(source, sourceType) {
    return `${this.baseUrl}/api/preview?source=${encodeURIComponent(source)}&type=${encodeURIComponent(sourceType)}&t=${Date.now()}`;
  }
};
