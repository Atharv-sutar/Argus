/**
 * Node & Edge Inspector / Property Editor
 */
class Inspector {
  constructor(app) {
    this.app = app;
    this.currentNode = null;
    this.currentEdge = null;

    this.panelEmpty = document.getElementById('inspector-empty');
    this.panelNode = document.getElementById('inspector-node');
    this.panelEdge = document.getElementById('inspector-edge');

    this.initEvents();
  }

  initEvents() {
    // Node Property Save
    const formNode = document.getElementById('form-edit-node');
    if (formNode) {
      formNode.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!this.currentNode) return;

        this.currentNode.name = document.getElementById('prop-cam-name').value.trim();
        this.currentNode.source_type = document.getElementById('prop-cam-type').value;
        const src = document.getElementById('prop-cam-source').value.trim();
        this.currentNode.source = (src !== '' && !isNaN(src) && String(parseInt(src, 10)) === src) ? parseInt(src, 10) : src;
        this.currentNode.enabled = document.getElementById('prop-cam-enabled').checked;

        this.app.graphCanvas.draw();
        this.app.showToast(`Updated camera '${this.currentNode.name}'`, 'info');
      });
    }

    // Node Delete
    const btnDeleteNode = document.getElementById('btn-delete-node');
    if (btnDeleteNode) {
      btnDeleteNode.addEventListener('click', () => {
        if (!this.currentNode) return;
        const name = this.currentNode.name;
        this.app.graphCanvas.removeNode(this.currentNode.camera_id);
        this.clear();
        this.app.showToast(`Deleted camera '${name}' from graph`, 'info');
      });
    }

    // Edge Property Save
    const formEdge = document.getElementById('form-edit-edge');
    if (formEdge) {
      formEdge.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!this.currentEdge) return;

        this.currentEdge.edge_type = document.getElementById('prop-edge-type').value;
        this.currentEdge.weight = parseFloat(document.getElementById('prop-edge-weight').value) || 1.0;

        this.app.graphCanvas.draw();
        this.app.showToast('Updated connection properties', 'info');
      });
    }

    // Edge Delete
    const btnDeleteEdge = document.getElementById('btn-delete-edge');
    if (btnDeleteEdge) {
      btnDeleteEdge.addEventListener('click', () => {
        if (!this.currentEdge) return;
        this.app.graphCanvas.removeEdge(this.currentEdge.source_camera_id, this.currentEdge.target_camera_id);
        this.clear();
        this.app.showToast('Connection removed', 'info');
      });
    }
  }

  inspectNode(node) {
    this.currentNode = node;
    this.currentEdge = null;

    if (this.panelEmpty) this.panelEmpty.style.display = 'none';
    if (this.panelEdge) this.panelEdge.style.display = 'none';
    if (this.panelNode) this.panelNode.style.display = 'block';

    const elId = document.getElementById('prop-cam-id');
    const elName = document.getElementById('prop-cam-name');
    const elType = document.getElementById('prop-cam-type');
    const elSource = document.getElementById('prop-cam-source');
    const elEnabled = document.getElementById('prop-cam-enabled');

    if (elId) elId.value = node.camera_id;
    if (elName) elName.value = node.name || '';
    if (elType) elType.value = node.source_type || 'webcam';
    if (elSource) elSource.value = node.source !== undefined ? node.source : '';
    if (elEnabled) elEnabled.checked = node.enabled !== false;
  }

  inspectEdge(edge) {
    this.currentEdge = edge;
    this.currentNode = null;

    if (this.panelEmpty) this.panelEmpty.style.display = 'none';
    if (this.panelNode) this.panelNode.style.display = 'none';
    if (this.panelEdge) this.panelEdge.style.display = 'block';

    const elSource = document.getElementById('prop-edge-source');
    const elTarget = document.getElementById('prop-edge-target');
    const elType = document.getElementById('prop-edge-type');
    const elWeight = document.getElementById('prop-edge-weight');

    if (elSource) elSource.textContent = edge.source_camera_id;
    if (elTarget) elTarget.textContent = edge.target_camera_id;
    if (elType) elType.value = edge.edge_type || 'adjacent';
    if (elWeight) elWeight.value = edge.weight !== undefined ? edge.weight : 1.0;
  }

  clear() {
    this.currentNode = null;
    this.currentEdge = null;

    if (this.panelNode) this.panelNode.style.display = 'none';
    if (this.panelEdge) this.panelEdge.style.display = 'none';
    if (this.panelEmpty) this.panelEmpty.style.display = 'block';
  }
}
