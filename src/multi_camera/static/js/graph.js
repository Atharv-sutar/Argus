/**
 * Interactive Camera Graph Canvas Engine
 */
class GraphCanvas {
  constructor(app, canvasId) {
    this.app = app;
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    // Graph Data Model
    this.nodes = [];  // List of CameraNodeConfig
    this.edges = [];  // List of CameraEdgeConfig
    this.backgroundMap = null;

    // Viewport State
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isPanning = false;
    this.panStartX = 0;
    this.panStartY = 0;

    // Interaction State
    this.tool = 'select'; // 'select' | 'connect'
    this.selectedNode = null;
    this.selectedEdge = null;
    this.draggedNode = null;
    this.dragOffset = { x: 0, y: 0 };
    this.connectStartNode = null;
    this.connectCurrentPos = null;

    // Runtime Status (Live Mode)
    this.cameraStatuses = {};
    this.activeCameraId = null;
    this.searchProgress = null;

    this.nodeRadius = 24;

    this.initCanvasSize();
    this.initEvents();
    this.startRenderLoop();
  }

  initCanvasSize() {
    const container = this.canvas.parentElement;
    const resize = () => {
      this.canvas.width = container.clientWidth * window.devicePixelRatio;
      this.canvas.height = container.clientHeight * window.devicePixelRatio;
      this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();
    window.addEventListener('resize', resize);
  }

  initEvents() {
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    window.addEventListener('mouseup', (e) => this.onMouseUp(e));
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });

    // Toolbar Tool Selection
    document.getElementById('tool-select').addEventListener('click', () => this.setTool('select'));
    document.getElementById('tool-connect').addEventListener('click', () => this.setTool('connect'));

    document.getElementById('btn-zoom-in').addEventListener('click', () => this.zoom(1.2));
    document.getElementById('btn-zoom-out').addEventListener('click', () => this.zoom(0.8));
    document.getElementById('btn-zoom-reset').addEventListener('click', () => this.fitToScreen());

    const btnSave = document.getElementById('btn-save');
    if (btnSave) {
      btnSave.addEventListener('click', async () => {
        try {
          const res = await API.saveGraph(this.toJSON());
          this.app.showToast('Camera topology graph saved successfully!', 'success');
          // Reload live matrix
          this.app.loadLiveMatrix();
        } catch (err) {
          this.app.showToast(`Save failed: ${err.message}`, 'error');
        }
      });
    }

    const btnValidate = document.getElementById('btn-validate');
    if (btnValidate) {
      btnValidate.addEventListener('click', async () => {
        try {
          const res = await API.validateGraph(this.toJSON());
          if (res.valid) {
            this.app.showToast('Topology graph is valid and well-connected!', 'success');
          } else {
            this.app.showToast(`Topology warnings: ${(res.errors || []).join('; ')}`, 'error');
          }
        } catch (err) {
          this.app.showToast(`Validation error: ${err.message}`, 'error');
        }
      });
    }
  }

  setTool(tool) {
    this.tool = tool;
    document.getElementById('tool-select').classList.toggle('active', tool === 'select');
    document.getElementById('tool-connect').classList.toggle('active', tool === 'connect');
    this.connectStartNode = null;
    this.connectCurrentPos = null;
  }

  loadGraph(data) {
    this.nodes = (data.cameras || []).map(c => ({
      ...c,
      position_x: c.position_x || 200 + Math.random() * 300,
      position_y: c.position_y || 150 + Math.random() * 200,
    }));
    this.edges = data.edges || [];
    this.backgroundMap = data.background_map || null;
    this.selectedNode = null;
    this.selectedEdge = null;
    this.app.inspector.clear();
  }

  toJSON() {
    return {
      version: 1,
      cameras: this.nodes,
      edges: this.edges,
      background_map: this.backgroundMap,
    };
  }

  addNode(nodeConfig) {
    // Ensure unique ID
    const baseId = nodeConfig.camera_id;
    let id = baseId;
    let counter = 1;
    while (this.nodes.some(n => n.camera_id === id)) {
      id = `${baseId}_${counter++}`;
    }
    nodeConfig.camera_id = id;
    this.nodes.push(nodeConfig);
    this.selectNode(nodeConfig);
  }

  removeNode(cameraId) {
    this.nodes = this.nodes.filter(n => n.camera_id !== cameraId);
    this.edges = this.edges.filter(e => e.source_camera_id !== cameraId && e.target_camera_id !== cameraId);
    if (this.selectedNode && this.selectedNode.camera_id === cameraId) {
      this.selectedNode = null;
      this.app.inspector.clear();
    }
  }

  addEdge(edgeConfig) {
    // Check if edge already exists
    const exists = this.edges.some(e =>
      (e.source_camera_id === edgeConfig.source_camera_id && e.target_camera_id === edgeConfig.target_camera_id) ||
      (e.source_camera_id === edgeConfig.target_camera_id && e.target_camera_id === edgeConfig.source_camera_id)
    );
    if (!exists && edgeConfig.source_camera_id !== edgeConfig.target_camera_id) {
      this.edges.push(edgeConfig);
      this.selectEdge(edgeConfig);
      return true;
    }
    return false;
  }

  removeEdge(sourceId, targetId) {
    this.edges = this.edges.filter(e =>
      !(e.source_camera_id === sourceId && e.target_camera_id === targetId) &&
      !(e.source_camera_id === targetId && e.target_camera_id === sourceId)
    );
    this.selectedEdge = null;
    this.app.inspector.clear();
  }

  screenToWorld(screenX, screenY) {
    const rect = this.canvas.getBoundingClientRect();
    const x = (screenX - rect.left - this.panX) / this.scale;
    const y = (screenY - rect.top - this.panY) / this.scale;
    return { x, y };
  }

  getNodeAt(worldX, worldY) {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      const dx = node.position_x - worldX;
      const dy = node.position_y - worldY;
      if (Math.sqrt(dx * dx + dy * dy) <= this.nodeRadius + 6) {
        return node;
      }
    }
    return null;
  }

  getEdgeAt(worldX, worldY) {
    for (const edge of this.edges) {
      const src = this.nodes.find(n => n.camera_id === edge.source_camera_id);
      const tgt = this.nodes.find(n => n.camera_id === edge.target_camera_id);
      if (!src || !tgt) continue;

      const dist = this.distToSegment(worldX, worldY, src.position_x, src.position_y, tgt.position_x, tgt.position_y);
      if (dist < 10) {
        return edge;
      }
    }
    return null;
  }

  distToSegment(px, py, x1, y1, x2, y2) {
    const l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    if (l2 === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
  }

  onMouseDown(e) {
    if (e.button !== 0) return; // Left click only
    const { x, y } = this.screenToWorld(e.clientX, e.clientY);
    const clickedNode = this.getNodeAt(x, y);

    if (this.tool === 'connect') {
      if (clickedNode) {
        this.connectStartNode = clickedNode;
        this.connectCurrentPos = { x, y };
      }
      return;
    }

    if (clickedNode) {
      this.draggedNode = clickedNode;
      this.dragOffset = { x: clickedNode.position_x - x, y: clickedNode.position_y - y };
      this.selectNode(clickedNode);
    } else {
      const clickedEdge = this.getEdgeAt(x, y);
      if (clickedEdge) {
        this.selectEdge(clickedEdge);
      } else {
        // Start Canvas Panning
        this.isPanning = true;
        this.panStartX = e.clientX - this.panX;
        this.panStartY = e.clientY - this.panY;
        this.selectedNode = null;
        this.selectedEdge = null;
        this.app.inspector.clear();
      }
    }
  }

  onMouseMove(e) {
    const { x, y } = this.screenToWorld(e.clientX, e.clientY);

    if (this.connectStartNode) {
      this.connectCurrentPos = { x, y };
      return;
    }

    if (this.draggedNode) {
      this.draggedNode.position_x = x + this.dragOffset.x;
      this.draggedNode.position_y = y + this.dragOffset.y;
      return;
    }

    if (this.isPanning) {
      this.panX = e.clientX - this.panStartX;
      this.panY = e.clientY - this.panStartY;
    }
  }

  onMouseUp(e) {
    if (this.connectStartNode) {
      const { x, y } = this.screenToWorld(e.clientX, e.clientY);
      const targetNode = this.getNodeAt(x, y);
      if (targetNode && targetNode.camera_id !== this.connectStartNode.camera_id) {
        const added = this.addEdge({
          source_camera_id: this.connectStartNode.camera_id,
          target_camera_id: targetNode.camera_id,
          edge_type: 'adjacent',
          direction: 'bidirectional',
          enabled: true,
          expected_min_transition_s: null,
          expected_typical_transition_s: null,
          expected_max_transition_s: null,
        });
        if (added) {
          this.app.showToast(`Connected ${this.connectStartNode.name} &harr; ${targetNode.name}`, 'success');
        }
      }
      this.connectStartNode = null;
      this.connectCurrentPos = null;
    }

    this.draggedNode = null;
    this.isPanning = false;
  }

  onWheel(e) {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    this.zoom(zoomFactor, e.clientX, e.clientY);
  }

  zoom(factor, clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = clientX !== undefined ? clientX - rect.left : rect.width / 2;
    const cy = clientY !== undefined ? clientY - rect.top : rect.height / 2;

    const newScale = Math.max(0.3, Math.min(3.0, this.scale * factor));
    this.panX = cx - (cx - this.panX) * (newScale / this.scale);
    this.panY = cy - (cy - this.panY) * (newScale / this.scale);
    this.scale = newScale;
  }

  fitToScreen() {
    if (this.nodes.length === 0) {
      this.scale = 1.0;
      this.panX = 0;
      this.panY = 0;
      return;
    }
    const xs = this.nodes.map(n => n.position_x);
    const ys = this.nodes.map(n => n.position_y);
    const minX = Math.min(...xs) - 80;
    const maxX = Math.max(...xs) + 80;
    const minY = Math.min(...ys) - 80;
    const maxY = Math.max(...ys) + 80;

    const w = maxX - minX;
    const h = maxY - minY;
    const rect = this.canvas.getBoundingClientRect();

    const scaleX = rect.width / w;
    const scaleY = rect.height / h;
    this.scale = Math.max(0.4, Math.min(1.5, Math.min(scaleX, scaleY)));
    this.panX = rect.width / 2 - ((minX + maxX) / 2) * this.scale;
    this.panY = rect.height / 2 - ((minY + maxY) / 2) * this.scale;
  }

  selectNode(node) {
    this.selectedNode = node;
    this.selectedEdge = null;
    this.app.inspector.inspectNode(node);
  }

  selectEdge(edge) {
    this.selectedEdge = edge;
    this.selectedNode = null;
    this.app.inspector.inspectEdge(edge);
  }

  startRenderLoop() {
    const render = () => {
      this.draw();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  draw() {
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    this.ctx.clearRect(0, 0, w, h);
    this.ctx.save();

    // Apply Viewport Transform
    this.ctx.translate(this.panX, this.panY);
    this.ctx.scale(this.scale, this.scale);

    // 1. Draw Grid
    this.drawGrid(w, h);

    // 2. Draw Search Radius Visualizer (Live mode)
    if (this.app.mode === 'live' && this.activeCameraId && this.searchProgress && this.searchProgress.state === 'searching') {
      this.drawSearchRings();
    }

    // 3. Draw Edges
    for (const edge of this.edges) {
      this.drawEdge(edge);
    }

    // 4. Draw Connecting Line in Progress
    if (this.connectStartNode && this.connectCurrentPos) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.connectStartNode.position_x, this.connectStartNode.position_y);
      this.ctx.lineTo(this.connectCurrentPos.x, this.connectCurrentPos.y);
      this.ctx.strokeStyle = '#00f0ff';
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([6, 6]);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    // 5. Draw Nodes
    for (const node of this.nodes) {
      this.drawNode(node);
    }

    this.ctx.restore();
  }

  drawGrid(screenW, screenH) {
    const gridSize = 40;
    const minX = -this.panX / this.scale;
    const maxX = (screenW - this.panX) / this.scale;
    const minY = -this.panY / this.scale;
    const maxY = (screenH - this.panY) / this.scale;

    this.ctx.beginPath();
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    this.ctx.lineWidth = 1;

    for (let x = Math.floor(minX / gridSize) * gridSize; x <= maxX; x += gridSize) {
      this.ctx.moveTo(x, minY);
      this.ctx.lineTo(x, maxY);
    }
    for (let y = Math.floor(minY / gridSize) * gridSize; y <= maxY; y += gridSize) {
      this.ctx.moveTo(minX, y);
      this.ctx.lineTo(maxX, y);
    }
    this.ctx.stroke();
  }

  drawSearchRings() {
    const activeNode = this.nodes.find(n => n.camera_id === this.activeCameraId);
    if (!activeNode) return;

    const radiusHops = this.searchProgress.search_radius || 1;
    const pixelRadius = radiusHops * 110;

    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.arc(activeNode.position_x, activeNode.position_y, pixelRadius, 0, Math.PI * 2);
    this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.4)';
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([8, 8]);
    this.ctx.stroke();

    this.ctx.fillStyle = 'rgba(255, 215, 0, 0.04)';
    this.ctx.fill();
    this.ctx.restore();
  }

  drawEdge(edge) {
    const src = this.nodes.find(n => n.camera_id === edge.source_camera_id);
    const tgt = this.nodes.find(n => n.camera_id === edge.target_camera_id);
    if (!src || !tgt) return;

    const isSelected = this.selectedEdge === edge;
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.moveTo(src.position_x, src.position_y);
    this.ctx.lineTo(tgt.position_x, tgt.position_y);

    let color = '#00f0ff'; // Cyan default
    if (edge.edge_type === 'overlap') color = '#00ff88'; // Green
    if (edge.edge_type === 'travel') color = '#ff9900';  // Orange

    if (!edge.enabled) color = '#64748b';

    this.ctx.strokeStyle = isSelected ? '#ffffff' : color;
    this.ctx.lineWidth = isSelected ? 4 : 2;

    if (edge.edge_type === 'travel') {
      this.ctx.setLineDash([6, 6]);
    } else {
      this.ctx.setLineDash([]);
    }

    if (edge.edge_type === 'overlap') {
      // Glow effect for overlap
      this.ctx.shadowColor = color;
      this.ctx.shadowBlur = 8;
    }

    this.ctx.stroke();

    // Draw directional indicator arrow
    if (edge.direction !== 'bidirectional') {
      this.drawArrowOnEdge(src, tgt, edge.direction === 'b_to_a');
    }

    this.ctx.restore();
  }

  drawArrowOnEdge(src, tgt, reverse) {
    const x1 = reverse ? tgt.position_x : src.position_x;
    const y1 = reverse ? tgt.position_y : src.position_y;
    const x2 = reverse ? src.position_x : tgt.position_x;
    const y2 = reverse ? src.position_y : tgt.position_y;

    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const arrowLen = 10;

    this.ctx.beginPath();
    this.ctx.moveTo(midX, midY);
    this.ctx.lineTo(midX - arrowLen * Math.cos(angle - Math.PI / 6), midY - arrowLen * Math.sin(angle - Math.PI / 6));
    this.ctx.lineTo(midX - arrowLen * Math.cos(angle + Math.PI / 6), midY - arrowLen * Math.sin(angle + Math.PI / 6));
    this.ctx.closePath();
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fill();
  }

  drawNode(node) {
    const isSelected = this.selectedNode === node;
    const x = node.position_x;
    const y = node.position_y;
    const r = this.nodeRadius;

    const status = this.cameraStatuses[node.camera_id] || (node.enabled ? 'online' : 'disabled');
    const isActiveTarget = (this.activeCameraId === node.camera_id);

    this.ctx.save();

    // 1. Status Ring / Glow
    if (isActiveTarget) {
      const tState = (this.targetState || 'TRACKING').toUpperCase();
      let targetGlowColor = '#10b981'; // Green default for TRACKING
      if (tState === 'OCCLUDED') targetGlowColor = '#c084fc';
      else if (tState === 'UNCERTAIN') targetGlowColor = '#f59e0b';
      else if (tState === 'LOST') targetGlowColor = '#f43f5e';
      else if (tState === 'LOCKED' || tState === 'ACQUIRING_REFERENCE') targetGlowColor = '#00f0ff';

      this.ctx.beginPath();
      this.ctx.arc(x, y, r + 8, 0, Math.PI * 2);
      this.ctx.strokeStyle = targetGlowColor;
      this.ctx.lineWidth = 3;
      this.ctx.stroke();

      // Corner Brackets for active target lock
      const bLen = 10;
      const bPad = r + 12;
      this.ctx.strokeStyle = targetGlowColor;
      this.ctx.lineWidth = 2;
      // Top-Left
      this.ctx.strokeRect(x - bPad, y - bPad, bLen, 0);
      this.ctx.strokeRect(x - bPad, y - bPad, 0, bLen);
      // Top-Right
      this.ctx.strokeRect(x + bPad - bLen, y - bPad, bLen, 0);
      this.ctx.strokeRect(x + bPad, y - bPad, 0, bLen);
      // Bottom-Left
      this.ctx.strokeRect(x - bPad, y + bPad, bLen, 0);
      this.ctx.strokeRect(x - bPad, y + bPad - bLen, 0, bLen);
      // Bottom-Right
      this.ctx.strokeRect(x + bPad - bLen, y + bPad, bLen, 0);
      this.ctx.strokeRect(x + bPad, y + bPad - bLen, 0, bLen);
    } else if (status === 'searching') {
      this.ctx.beginPath();
      this.ctx.arc(x, y, r + 6, 0, Math.PI * 2);
      this.ctx.strokeStyle = '#ff9900';
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([4, 4]);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    // 2. Node Body Background
    this.ctx.beginPath();
    this.ctx.arc(x, y, r, 0, Math.PI * 2);
    this.ctx.fillStyle = isSelected ? '#1e293b' : '#141a29';
    this.ctx.fill();

    let borderColor = '#00f0ff';
    if (!node.enabled || status === 'offline') borderColor = '#64748b';
    if (status === 'searching') borderColor = '#ff9900';
    if (isActiveTarget) borderColor = '#10b981';
    if (isSelected) borderColor = '#ffffff';

    this.ctx.strokeStyle = borderColor;
    this.ctx.lineWidth = isSelected ? 3 : 2;
    this.ctx.stroke();

    // 3. Camera Icon (Lens circle)
    this.ctx.beginPath();
    this.ctx.arc(x, y, 7, 0, Math.PI * 2);
    this.ctx.fillStyle = borderColor;
    this.ctx.fill();

    // 4. Label Badge Below Node
    this.ctx.font = '11px Inter, sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'top';

    const text = node.name || node.camera_id;
    const tw = this.ctx.measureText(text).width;

    this.ctx.fillStyle = 'rgba(16, 22, 38, 0.9)';
    this.ctx.fillRect(x - tw / 2 - 6, y + r + 6, tw + 12, 18);
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    this.ctx.strokeRect(x - tw / 2 - 6, y + r + 6, tw + 12, 18);

    this.ctx.fillStyle = '#ffffff';
    this.ctx.fillText(text, x, y + r + 9);

    this.ctx.restore();
  }
}
