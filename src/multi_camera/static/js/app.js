/**
 * Argus Topology & Camera Mapping Application Orchestrator
 */
class InspectorManager {
  constructor(app) {
    this.app = app;
    this.emptySec = document.getElementById('inspector-empty');
    this.nodeSec = document.getElementById('inspector-node');
    this.edgeSec = document.getElementById('inspector-edge');

    this.currentNode = null;
    this.currentEdge = null;

    this.initNodeInputs();
    this.initEdgeInputs();
  }

  clear() {
    this.emptySec.style.display = 'block';
    this.nodeSec.style.display = 'none';
    this.edgeSec.style.display = 'none';
    this.currentNode = null;
    this.currentEdge = null;
  }

  inspectNode(node) {
    this.currentNode = node;
    this.currentEdge = null;

    this.emptySec.style.display = 'none';
    this.nodeSec.style.display = 'block';
    this.edgeSec.style.display = 'none';

    document.getElementById('prop-node-id').value = node.camera_id;
    document.getElementById('prop-node-name').value = node.name || '';
    document.getElementById('prop-node-source').value = node.source !== undefined ? node.source : '';
    document.getElementById('prop-node-type').value = node.source_type || 'webcam';
    document.getElementById('prop-node-floor').value = node.floor || '';
    document.getElementById('prop-node-zone').value = node.zone || '';
    document.getElementById('prop-node-desc').value = node.description || '';
    document.getElementById('prop-node-enabled').checked = node.enabled !== false;
  }

  initNodeInputs() {
    const bind = (id, prop, isNum, isBool) => {
      document.getElementById(id).addEventListener('input', (e) => {
        if (!this.currentNode) return;
        let val = e.target.value;
        if (isBool) val = e.target.checked;
        else if (isNum) val = isNaN(val) ? val : parseInt(val, 10);
        this.currentNode[prop] = val;
      });
    };

    bind('prop-node-name', 'name');
    bind('prop-node-source', 'source', true);
    bind('prop-node-type', 'source_type');
    bind('prop-node-floor', 'floor');
    bind('prop-node-zone', 'zone');
    bind('prop-node-desc', 'description');
    bind('prop-node-enabled', 'enabled', false, true);

    document.getElementById('btn-delete-node').addEventListener('click', () => {
      if (this.currentNode) {
        const id = this.currentNode.camera_id;
        this.app.graphCanvas.removeNode(id);
        this.app.showToast(`Deleted camera '${id}'`, 'info');
      }
    });
  }

  inspectEdge(edge) {
    this.currentEdge = edge;
    this.currentNode = null;

    this.emptySec.style.display = 'none';
    this.nodeSec.style.display = 'none';
    this.edgeSec.style.display = 'block';

    const srcNode = this.app.graphCanvas.nodes.find(n => n.camera_id === edge.source_camera_id);
    const tgtNode = this.app.graphCanvas.nodes.find(n => n.camera_id === edge.target_camera_id);

    const srcName = srcNode ? srcNode.name : edge.source_camera_id;
    const tgtName = tgtNode ? tgtNode.name : edge.target_camera_id;

    document.getElementById('prop-edge-endpoints').textContent = `${srcName} \u2194 ${tgtName}`;
    document.getElementById('prop-edge-type').value = edge.edge_type || 'adjacent';
    document.getElementById('prop-edge-direction').value = edge.direction || 'bidirectional';
    document.getElementById('prop-edge-min-t').value = edge.expected_min_transition_s || '';
    document.getElementById('prop-edge-typ-t').value = edge.expected_typical_transition_s || '';
    document.getElementById('prop-edge-max-t').value = edge.expected_max_transition_s || '';
    document.getElementById('prop-edge-enabled').checked = edge.enabled !== false;
  }

  initEdgeInputs() {
    const bind = (id, prop, isFloat, isBool) => {
      document.getElementById(id).addEventListener('input', (e) => {
        if (!this.currentEdge) return;
        let val = e.target.value;
        if (isBool) val = e.target.checked;
        else if (isFloat) val = val === '' ? null : parseFloat(val);
        this.currentEdge[prop] = val;
      });
    };

    bind('prop-edge-type', 'edge_type');
    bind('prop-edge-direction', 'direction');
    bind('prop-edge-min-t', 'expected_min_transition_s', true);
    bind('prop-edge-typ-t', 'expected_typical_transition_s', true);
    bind('prop-edge-max-t', 'expected_max_transition_s', true);
    bind('prop-edge-enabled', 'enabled', false, true);

    document.getElementById('btn-delete-edge').addEventListener('click', () => {
      if (this.currentEdge) {
        this.app.graphCanvas.removeEdge(this.currentEdge.source_camera_id, this.currentEdge.target_camera_id);
        this.app.showToast('Deleted connection', 'info');
      }
    });
  }
}

class App {
  constructor() {
    this.mode = 'edit'; // 'edit' | 'live'
    this.statusPollInterval = null;

    this.graphCanvas = new GraphCanvas(this, 'graph-canvas');
    this.cameraManager = new CameraManager(this);
    this.inspector = new InspectorManager(this);

    this.initEvents();
    this.loadGraph();
  }

  initEvents() {
    // Mode Switching
    document.getElementById('btn-mode-edit').addEventListener('click', () => this.setMode('edit'));
    document.getElementById('btn-mode-live').addEventListener('click', () => this.setMode('live'));

    // Save Graph
    document.getElementById('btn-save').addEventListener('click', () => this.saveGraph());

    // Validate Graph
    document.getElementById('btn-validate').addEventListener('click', () => this.validateGraph());
  }

  setMode(mode) {
    this.mode = mode;
    document.getElementById('btn-mode-edit').classList.toggle('active', mode === 'edit');
    document.getElementById('btn-mode-live').classList.toggle('active', mode === 'live');

    const liveHud = document.getElementById('live-hud');
    if (mode === 'live') {
      liveHud.style.display = 'flex';
      this.startStatusPolling();
      this.showToast('Switched to Live Surveillance Monitor', 'info');
    } else {
      liveHud.style.display = 'none';
      this.stopStatusPolling();
      this.showToast('Switched to Topology Editor', 'info');
    }
  }

  startStatusPolling() {
    this.stopStatusPolling();
    const poll = async () => {
      if (this.mode !== 'live') return;
      const statusData = await API.getStatus();
      if (statusData) {
        this.graphCanvas.cameraStatuses = statusData.camera_statuses || {};
        this.graphCanvas.activeCameraId = statusData.active_camera;
        this.graphCanvas.searchProgress = statusData.search_progress;

        document.getElementById('hud-active-cam').textContent = statusData.active_camera || 'None';
        if (statusData.search_progress) {
          document.getElementById('hud-search-state').textContent = statusData.search_progress.state.toUpperCase();
          document.getElementById('hud-search-radius').textContent = statusData.search_progress.search_radius;
        }
      }
    };
    poll();
    this.statusPollInterval = setInterval(poll, 1000);
  }

  stopStatusPolling() {
    if (this.statusPollInterval) {
      clearInterval(this.statusPollInterval);
      this.statusPollInterval = null;
    }
  }

  async loadGraph() {
    try {
      const data = await API.getGraph();
      this.graphCanvas.loadGraph(data);
      this.graphCanvas.fitToScreen();
      this.showToast('Graph topology loaded', 'info');
    } catch (err) {
      this.showToast(`Error loading graph: ${err.message}`, 'error');
    }
  }

  async saveGraph() {
    const graphData = this.graphCanvas.toJSON();
    try {
      const res = await API.saveGraph(graphData);
      this.showToast(res.message || 'Graph saved successfully!', 'success');
    } catch (err) {
      this.showToast(`Failed to save: ${err.message}`, 'error');
    }
  }

  async validateGraph() {
    const graphData = this.graphCanvas.toJSON();
    try {
      const res = await API.validateGraph(graphData);
      if (res.valid) {
        this.showToast('Topology is valid and ready!', 'success');
      } else {
        this.showToast(`Validation issues:\n${res.errors.join('\n')}`, 'error');
      }
    } catch (err) {
      this.showToast(`Validation failed: ${err.message}`, 'error');
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

// Bootstrap Application
window.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
