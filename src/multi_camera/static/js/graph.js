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

    // Runtime Status (Live Mode Telemetry)
    this.cameraStatuses = {};
    this.activeCameraId = null;
    this.targetState = 'UNSELECTED';
    this.searchProgress = null;
    this.candidateScores = {};
    this.transitHistory = [];

    // Visual Display Toggles
    this.filterReidOnly = false;
    this.showFovCones = true;
    this.showMinimap = true;

    this.nodeRadius = 26;

    this.initCanvasSize();
    this.initEvents();
    this.startRenderLoop();
  }

  initCanvasSize() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const container = this.canvas.parentElement;
    if (container) {
      const rect = container.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
      }
    }
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

    const btnReid = document.getElementById('btn-filter-reid');
    if (btnReid) {
      btnReid.addEventListener('click', () => {
        this.filterReidOnly = !this.filterReidOnly;
        btnReid.classList.toggle('active', this.filterReidOnly);
        this.app.showToast(this.filterReidOnly ? 'Re-ID Focus Filter: Enabled' : 'Re-ID Focus Filter: Disabled', 'info');
      });
    }

    const btnFov = document.getElementById('btn-toggle-fov');
    if (btnFov) {
      btnFov.addEventListener('click', () => {
        this.showFovCones = !this.showFovCones;
        btnFov.classList.toggle('active', this.showFovCones);
      });
    }

    const btnMinimap = document.getElementById('btn-toggle-minimap');
    if (btnMinimap) {
      btnMinimap.addEventListener('click', () => {
        this.showMinimap = !this.showMinimap;
        btnMinimap.classList.toggle('active', this.showMinimap);
      });
    }

    const btnSave = document.getElementById('btn-save');
    if (btnSave) {
      btnSave.addEventListener('click', async () => {
        btnSave.disabled = true;
        try {
          const res = await API.saveGraph(this.toJSON());
          this.app.showToast('Camera topology graph saved successfully!', 'success');
          // Reload live matrix
          await this.app.loadLiveMatrix();
        } catch (err) {
          this.app.showToast(`Save failed: ${err.message}`, 'error');
        } finally {
          btnSave.disabled = false;
        }
      });
    }

    const btnValidate = document.getElementById('btn-validate');
    if (btnValidate) {
      btnValidate.addEventListener('click', async () => {
        btnValidate.disabled = true;
        try {
          const res = await API.validateGraph(this.toJSON());
          if (res.valid) {
            this.app.showToast('Topology graph is valid and well-connected!', 'success');
          } else {
            this.app.showToast(`Topology warnings: ${(res.errors || []).join('; ')}`, 'error');
          }
        } catch (err) {
          this.app.showToast(`Validation error: ${err.message}`, 'error');
        } finally {
          btnValidate.disabled = false;
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
    this.nodes = (data.cameras || []).map((c, idx) => ({
      ...c,
      position_x: (c.position_x !== undefined && c.position_x !== null && !isNaN(c.position_x))
        ? Number(c.position_x)
        : 240 + (idx % 3) * 220,
      position_y: (c.position_y !== undefined && c.position_y !== null && !isNaN(c.position_y))
        ? Number(c.position_y)
        : 180 + Math.floor(idx / 3) * 180,
    }));
    this.edges = (data.edges || []).map(e => ({
      ...e,
      source_camera_id: e.source_camera_id || e.source,
      target_camera_id: e.target_camera_id || e.target,
      source: e.source_camera_id || e.source,
      target: e.target_camera_id || e.target,
    }));
    this.backgroundMap = data.background_map || null;
    this.selectedNode = null;
    this.selectedEdge = null;
    this._hasInitialFit = false;
    if (this.app && this.app.inspector) {
      this.app.inspector.clear();
    }
  }

  toJSON() {
    return {
      version: 1,
      cameras: this.nodes.map(n => ({
        ...n,
        position_x: Number(n.position_x) || 0,
        position_y: Number(n.position_y) || 0,
      })),
      edges: this.edges.map(e => ({
        ...e,
        source_camera_id: e.source_camera_id || e.source,
        target_camera_id: e.target_camera_id || e.target,
        source: e.source_camera_id || e.source,
        target: e.target_camera_id || e.target,
      })),
      background_map: this.backgroundMap,
    };
  }

  addNode(nodeConfig) {
    // Ensure unique ID
    const baseId = nodeConfig.camera_id || 'cam';
    let id = baseId;
    let counter = 1;
    while (this.nodes.some(n => n.camera_id === id)) {
      id = `${baseId}_${counter++}`;
    }
    nodeConfig.camera_id = id;

    // Validate coordinates - place near current screen center if unset
    if (isNaN(nodeConfig.position_x) || nodeConfig.position_x === null || nodeConfig.position_x === undefined) {
      const rect = this.canvas.getBoundingClientRect();
      const center = this.screenToWorld(
        (rect.width > 0 ? rect.width : this.canvas.width) / 2,
        (rect.height > 0 ? rect.height : this.canvas.height) / 2
      );
      nodeConfig.position_x = Math.round(center.x + (Math.random() - 0.5) * 80);
    }
    if (isNaN(nodeConfig.position_y) || nodeConfig.position_y === null || nodeConfig.position_y === undefined) {
      const rect = this.canvas.getBoundingClientRect();
      const center = this.screenToWorld(
        (rect.width > 0 ? rect.width : this.canvas.width) / 2,
        (rect.height > 0 ? rect.height : this.canvas.height) / 2
      );
      nodeConfig.position_y = Math.round(center.y + (Math.random() - 0.5) * 80);
    }

    this.nodes.push(nodeConfig);
    this.selectNode(nodeConfig);
    this.panToNode(nodeConfig);
  }

  panToNode(node) {
    if (!node) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = (rect.width > 0 ? rect.width : this.canvas.width) || 800;
    const h = (rect.height > 0 ? rect.height : this.canvas.height) || 600;
    this.panX = w / 2 - node.position_x * this.scale;
    this.panY = h / 2 - node.position_y * this.scale;
  }

  removeNode(cameraId) {
    this.nodes = this.nodes.filter(n => n.camera_id !== cameraId);
    this.edges = this.edges.filter(e =>
      (e.source_camera_id || e.source) !== cameraId &&
      (e.target_camera_id || e.target) !== cameraId
    );
    if (this.selectedNode && this.selectedNode.camera_id === cameraId) {
      this.selectedNode = null;
      if (this.app && this.app.inspector) {
        this.app.inspector.clear();
      }
    }
  }

  addEdge(edgeConfig) {
    const srcId = edgeConfig.source_camera_id || edgeConfig.source;
    const tgtId = edgeConfig.target_camera_id || edgeConfig.target;
    if (!srcId || !tgtId || srcId === tgtId) return false;

    // Check if edge already exists
    const exists = this.edges.some(e => {
      const eSrc = e.source_camera_id || e.source;
      const eTgt = e.target_camera_id || e.target;
      return (eSrc === srcId && eTgt === tgtId) || (eSrc === tgtId && eTgt === srcId);
    });

    if (!exists) {
      const normalizedEdge = {
        ...edgeConfig,
        source_camera_id: srcId,
        target_camera_id: tgtId,
        source: srcId,
        target: tgtId,
      };
      this.edges.push(normalizedEdge);
      this.selectEdge(normalizedEdge);
      return true;
    }
    return false;
  }

  removeEdge(sourceId, targetId) {
    this.edges = this.edges.filter(e => {
      const eSrc = e.source_camera_id || e.source;
      const eTgt = e.target_camera_id || e.target;
      return !(eSrc === sourceId && eTgt === targetId) && !(eSrc === targetId && eTgt === sourceId);
    });
    this.selectedEdge = null;
    if (this.app && this.app.inspector) {
      this.app.inspector.clear();
    }
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
      const srcId = edge.source_camera_id || edge.source;
      const tgtId = edge.target_camera_id || edge.target;
      const src = this.nodes.find(n => n.camera_id === srcId);
      const tgt = this.nodes.find(n => n.camera_id === tgtId);
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
    const rect = this.canvas.getBoundingClientRect();
    const w = (rect.width > 0 ? rect.width : this.canvas.width) || 800;
    const h = (rect.height > 0 ? rect.height : this.canvas.height) || 600;

    if (this.nodes.length === 0) {
      this.scale = 1.0;
      this.panX = w / 2;
      this.panY = h / 2;
      return;
    }

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    this.nodes.forEach((n, idx) => {
      if (isNaN(n.position_x) || n.position_x === null || n.position_x === undefined) {
        n.position_x = 240 + (idx % 3) * 220;
      }
      if (isNaN(n.position_y) || n.position_y === null || n.position_y === undefined) {
        n.position_y = 180 + Math.floor(idx / 3) * 180;
      }
      minX = Math.min(minX, n.position_x);
      maxX = Math.max(maxX, n.position_x);
      minY = Math.min(minY, n.position_y);
      maxY = Math.max(maxY, n.position_y);
    });

    const boxW = Math.max(160, maxX - minX + 220);
    const boxH = Math.max(160, maxY - minY + 220);

    const scaleX = (w - 120) / boxW;
    const scaleY = (h - 120) / boxH;
    this.scale = Math.max(0.4, Math.min(1.2, Math.min(scaleX, scaleY)));

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    this.panX = w / 2 - centerX * this.scale;
    this.panY = h / 2 - centerY * this.scale;
    this._hasInitialFit = true;
  }

  selectNode(node) {
    this.selectedNode = node;
    this.selectedEdge = null;
    if (this.app && this.app.inspector) {
      this.app.inspector.inspectNode(node);
    }
  }

  selectEdge(edge) {
    this.selectedEdge = edge;
    this.selectedNode = null;
    if (this.app && this.app.inspector) {
      this.app.inspector.inspectEdge(edge);
    }
  }

  startRenderLoop() {
    const render = () => {
      this.draw();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  isSearchingNode(cameraId) {
    if (!cameraId) return false;
    const status = this.cameraStatuses ? this.cameraStatuses[cameraId] : null;
    if (status && String(status).toLowerCase() === 'searching') return true;
    if (this.searchProgress) {
      const activeCams = this.searchProgress.active_cameras || this.searchProgress.active_search_camera_ids;
      if (Array.isArray(activeCams) && activeCams.includes(cameraId)) return true;
    }
    return false;
  }

  draw() {
    const rect = this.canvas.getBoundingClientRect();
    const w = (rect.width > 0 ? rect.width : this.canvas.width) || 800;
    const h = (rect.height > 0 ? rect.height : this.canvas.height) || 600;
    if (w <= 0 || h <= 0) return;

    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    // Auto-fit preloaded nodes on first visible render
    if (this.nodes.length > 0 && !this._hasInitialFit && (this.panX === 0 && this.panY === 0)) {
      this.fitToScreen();
    }

    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.save();

    // Apply Viewport Transform safely
    const panX = isNaN(this.panX) ? 0 : this.panX;
    const panY = isNaN(this.panY) ? 0 : this.panY;
    const scale = isNaN(this.scale) || this.scale <= 0 ? 1.0 : this.scale;

    this.ctx.translate(panX, panY);
    this.ctx.scale(scale, scale);

    // 1. Draw Tactical Coordinates & Grid
    this.drawGrid(w, h);
    this.drawTacticalCoordinates(w, h);

    // 2. Draw Dynamic Search Radius Circles
    if (this.activeCameraId && this.searchProgress && this.searchProgress.state === 'searching') {
      this.drawSearchRings();
    }

    // 3. Draw FOV Sight Cones
    if (this.showFovCones) {
      this.drawFovCones();
    }

    // 4. Draw Edges with Animated Energy Pulses
    for (const edge of this.edges) {
      this.drawEdge(edge);
    }

    // 5. Draw Connecting Line in Progress
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

    // 6. Draw Nodes (Active target lock rings, searching sweeps, standby sentries)
    for (const node of this.nodes) {
      this.drawNode(node);
    }

    this.ctx.restore();

    // 7. Screen-space HUD & Minimap Overlay (rendered in screen coordinates)
    if (this.showMinimap && this.nodes.length > 0) {
      this.drawMinimap(w, h);
    }
  }

  drawGrid(screenW, screenH) {
    const gridSize = 40;
    const majorGrid = 160;
    const minX = -this.panX / this.scale;
    const maxX = (screenW - this.panX) / this.scale;
    const minY = -this.panY / this.scale;
    const maxY = (screenH - this.panY) / this.scale;

    // Minor grid lines
    this.ctx.beginPath();
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.035)';
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

    // Major grid lines & Crosshairs
    this.ctx.beginPath();
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.08)';
    this.ctx.lineWidth = 1.2;
    for (let x = Math.floor(minX / majorGrid) * majorGrid; x <= maxX; x += majorGrid) {
      this.ctx.moveTo(x, minY);
      this.ctx.lineTo(x, maxY);
    }
    for (let y = Math.floor(minY / majorGrid) * majorGrid; y <= maxY; y += majorGrid) {
      this.ctx.moveTo(minX, y);
      this.ctx.lineTo(maxX, y);
    }
    this.ctx.stroke();

    // Major Intersection Crosshairs (+)
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    this.ctx.lineWidth = 1;
    const cSize = 4;
    for (let x = Math.floor(minX / majorGrid) * majorGrid; x <= maxX; x += majorGrid) {
      for (let y = Math.floor(minY / majorGrid) * majorGrid; y <= maxY; y += majorGrid) {
        this.ctx.beginPath();
        this.ctx.moveTo(x - cSize, y);
        this.ctx.lineTo(x + cSize, y);
        this.ctx.moveTo(x, y - cSize);
        this.ctx.lineTo(x, y + cSize);
        this.ctx.stroke();
      }
    }
  }

  drawTacticalCoordinates(screenW, screenH) {
    const minX = -this.panX / this.scale;
    const minY = -this.panY / this.scale;

    this.ctx.save();
    this.ctx.font = '9px "JetBrains Mono", monospace';
    this.ctx.fillStyle = 'rgba(0, 240, 255, 0.35)';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';
    this.ctx.fillText('GRID SEC-07A // SECTOR VECTOR MESH // PROJ: EPSG:3857', minX + 16, minY + 16);
    this.ctx.fillStyle = 'rgba(148, 163, 184, 0.35)';
    this.ctx.fillText('LAT: 33°49\'12.4"N // LON: 118°14\'08.2"W // ELEV: +14.2M MSL', minX + 16, minY + 30);
    this.ctx.restore();
  }

  drawSearchRings() {
    const activeNode = this.nodes.find(n => n.camera_id === this.activeCameraId);
    if (!activeNode) return;

    const radiusHops = (this.searchProgress && this.searchProgress.search_radius) ? this.searchProgress.search_radius : 1;
    const pixelRadius = radiusHops * 120;
    const now = Date.now();

    this.ctx.save();

    // Multi-tier animated dashed search perimeter
    for (let h = 1; h <= radiusHops; h++) {
      const r = h * 120;
      this.ctx.beginPath();
      this.ctx.arc(activeNode.position_x, activeNode.position_y, r, 0, Math.PI * 2);
      this.ctx.strokeStyle = h === radiusHops ? 'rgba(245, 158, 11, 0.6)' : 'rgba(245, 158, 11, 0.25)';
      this.ctx.lineWidth = h === radiusHops ? 2 : 1;
      this.ctx.setLineDash([8, 8]);
      this.ctx.lineDashOffset = -(now / 80) % 16;
      this.ctx.stroke();

      this.ctx.fillStyle = 'rgba(245, 158, 11, 0.025)';
      this.ctx.fill();
    }

    this.ctx.restore();
  }

  drawFovCones() {
    const now = Date.now();
    for (const node of this.nodes) {
      const isActive = (this.activeCameraId === node.camera_id);
      const isSearching = this.isSearchingNode(node.camera_id);
      if (!isActive && !isSearching) continue;

      // Find average vector direction to connected neighbors
      let dirAngle = 0;
      const connectedEdges = this.edges.filter(e => {
        const s = e.source_camera_id || e.source;
        const t = e.target_camera_id || e.target;
        return s === node.camera_id || t === node.camera_id;
      });
      if (connectedEdges.length > 0) {
        let sumX = 0, sumY = 0;
        connectedEdges.forEach(e => {
          const s = e.source_camera_id || e.source;
          const t = e.target_camera_id || e.target;
          const otherId = (s === node.camera_id) ? t : s;
          const otherNode = this.nodes.find(n => n.camera_id === otherId);
          if (otherNode) {
            sumX += (otherNode.position_x - node.position_x);
            sumY += (otherNode.position_y - node.position_y);
          }
        });
        if (sumX !== 0 || sumY !== 0) {
          dirAngle = Math.atan2(sumY, sumX);
        }
      }

      const coneLen = isActive ? 120 : 95;
      const coneFov = isActive ? 0.75 : 0.65; // ~40 degrees
      const startAngle = dirAngle - coneFov / 2;
      const endAngle = dirAngle + coneFov / 2;

      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.moveTo(node.position_x, node.position_y);
      this.ctx.arc(node.position_x, node.position_y, coneLen, startAngle, endAngle);
      this.ctx.closePath();

      const grad = this.ctx.createRadialGradient(
        node.position_x, node.position_y, 10,
        node.position_x, node.position_y, coneLen
      );
      if (isActive) {
        grad.addColorStop(0, 'rgba(0, 240, 255, 0.35)');
        grad.addColorStop(0.7, 'rgba(16, 185, 129, 0.12)');
        grad.addColorStop(1, 'rgba(0, 240, 255, 0)');
      } else {
        grad.addColorStop(0, 'rgba(245, 158, 11, 0.35)');
        grad.addColorStop(0.7, 'rgba(245, 158, 11, 0.1)');
        grad.addColorStop(1, 'rgba(245, 158, 11, 0)');
      }

      this.ctx.fillStyle = grad;
      this.ctx.fill();

      // Outer dashed arc
      this.ctx.beginPath();
      this.ctx.arc(node.position_x, node.position_y, coneLen, startAngle, endAngle);
      this.ctx.strokeStyle = isActive ? 'rgba(0, 240, 255, 0.5)' : 'rgba(245, 158, 11, 0.5)';
      this.ctx.lineWidth = 1.5;
      this.ctx.setLineDash([4, 4]);
      this.ctx.stroke();

      this.ctx.restore();
    }
  }

  drawEdge(edge) {
    const srcId = edge.source_camera_id || edge.source;
    const tgtId = edge.target_camera_id || edge.target;
    const src = this.nodes.find(n => n.camera_id === srcId);
    const tgt = this.nodes.find(n => n.camera_id === tgtId);
    if (!src || !tgt) return;

    if (this.filterReidOnly) {
      const srcActive = (this.activeCameraId === src.camera_id) || this.isSearchingNode(src.camera_id);
      const tgtActive = (this.activeCameraId === tgt.camera_id) || this.isSearchingNode(tgt.camera_id);
      if (!srcActive && !tgtActive) return;
    }

    const isSelected = this.selectedEdge === edge;
    const isActiveTargetSrc = (this.activeCameraId === src.camera_id);
    const isActiveTargetTgt = (this.activeCameraId === tgt.camera_id);
    const isSearchingSrc = this.isSearchingNode(src.camera_id);
    const isSearchingTgt = this.isSearchingNode(tgt.camera_id);
    const isHotEdge = (isActiveTargetSrc || isActiveTargetTgt) && (isSearchingSrc || isSearchingTgt || isActiveTargetSrc || isActiveTargetTgt);

    this.ctx.save();

    // 1. Base Edge Line
    this.ctx.beginPath();
    this.ctx.moveTo(src.position_x, src.position_y);
    this.ctx.lineTo(tgt.position_x, tgt.position_y);

    let color = '#00f0ff'; // Cyan default
    if (edge.edge_type === 'overlap') color = '#10b981'; // Green
    if (edge.edge_type === 'travel') color = '#f59e0b';  // Amber
    if (!edge.enabled) color = '#475569';

    if (isHotEdge) {
      color = '#00f0ff';
      this.ctx.shadowColor = '#00f0ff';
      this.ctx.shadowBlur = 10;
    }

    this.ctx.strokeStyle = isSelected ? '#ffffff' : color;
    this.ctx.lineWidth = isSelected ? 4 : (isHotEdge ? 3 : 2);

    if (edge.edge_type === 'travel') {
      this.ctx.setLineDash([6, 6]);
    } else {
      this.ctx.setLineDash([]);
    }

    this.ctx.stroke();

    // 2. Animated Energy Particles along Hot / Searching Transit Edges
    if (isHotEdge || isSelected) {
      const now = Date.now();
      const numParticles = 3;
      const dx = tgt.position_x - src.position_x;
      const dy = tgt.position_y - src.position_y;

      for (let p = 0; p < numParticles; p++) {
        const t = ((now / 1400) + (p / numParticles)) % 1.0;
        const px = src.position_x + dx * t;
        const py = src.position_y + dy * t;

        this.ctx.beginPath();
        this.ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        this.ctx.fillStyle = isSearchingTgt || isSearchingSrc ? '#fef08a' : '#a5f3fc';
        this.ctx.shadowColor = '#00f0ff';
        this.ctx.shadowBlur = 8;
        this.ctx.fill();
      }
    }

    // 3. Directional Arrow
    if (edge.direction !== 'bidirectional') {
      this.drawArrowOnEdge(src, tgt, edge.direction === 'b_to_a');
    }

    // 4. Transit Time Badge Pill
    const midX = (src.position_x + tgt.position_x) / 2;
    const midY = (src.position_y + tgt.position_y) / 2;
    const transitSec = edge.expected_typical_transition_s || edge.expected_min_transition_s || (edge.edge_type === 'overlap' ? '0.5' : '4.5');
    const badgeText = edge.edge_type === 'overlap' ? 'OVERLAP' : `⏱ ${transitSec}s`;

    this.ctx.font = '10px "JetBrains Mono", monospace';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    const bW = this.ctx.measureText(badgeText).width + 12;
    const bH = 16;

    this.ctx.fillStyle = 'rgba(10, 14, 22, 0.92)';
    this.ctx.fillRect(midX - bW / 2, midY - bH / 2, bW, bH);
    this.ctx.strokeStyle = isHotEdge ? 'rgba(0, 240, 255, 0.6)' : 'rgba(255, 255, 255, 0.15)';
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(midX - bW / 2, midY - bH / 2, bW, bH);

    this.ctx.fillStyle = isHotEdge ? '#00f0ff' : '#94a3b8';
    this.ctx.fillText(badgeText, midX, midY);

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
    const now = Date.now();

    const status = this.cameraStatuses[node.camera_id] || (node.enabled ? 'online' : 'disabled');
    const isActiveTarget = (this.activeCameraId === node.camera_id);
    const isSearching = this.isSearchingNode(node.camera_id);

    if (this.filterReidOnly && !isActiveTarget && !isSearching) {
      return; // Filtered out
    }

    this.ctx.save();

    // 1. ACTIVE TARGET LOCK: Animated Concentric Sonar Rings & Precision Reticles
    if (isActiveTarget) {
      const tState = (this.targetState || 'TRACKING').toUpperCase();
      let targetGlowColor = '#10b981'; // Surveillance Emerald default
      if (tState === 'OCCLUDED') targetGlowColor = '#c084fc';
      else if (tState === 'UNCERTAIN') targetGlowColor = '#f59e0b';
      else if (tState === 'LOST' || tState === 'LOST_PERMANENTLY') targetGlowColor = '#ef4444';
      else if (tState === 'LOCKED' || tState === 'ACQUIRING_REFERENCE') targetGlowColor = '#00f0ff';

      // 3-Tier Expanding Sonar Radar Rings
      for (let ring = 0; ring < 3; ring++) {
        const phase = ((now + ring * 500) % 1500) / 1500;
        const ringRadius = r + phase * 44;
        const alpha = Math.max(0, (1 - phase) * 0.7);

        this.ctx.beginPath();
        this.ctx.arc(x, y, ringRadius, 0, Math.PI * 2);
        this.ctx.strokeStyle = `rgba(0, 240, 255, ${alpha})`;
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
      }

      // Static Lock Ring
      this.ctx.beginPath();
      this.ctx.arc(x, y, r + 8, 0, Math.PI * 2);
      this.ctx.strokeStyle = targetGlowColor;
      this.ctx.lineWidth = 3;
      this.ctx.stroke();

      // Precision Corner L-Brackets [ + ]
      const bLen = 12;
      const bPad = r + 14;
      this.ctx.strokeStyle = targetGlowColor;
      this.ctx.lineWidth = 2.5;

      // Top-Left
      this.ctx.beginPath();
      this.ctx.moveTo(x - bPad, y - bPad + bLen);
      this.ctx.lineTo(x - bPad, y - bPad);
      this.ctx.lineTo(x - bPad + bLen, y - bPad);
      this.ctx.stroke();

      // Top-Right
      this.ctx.beginPath();
      this.ctx.moveTo(x + bPad - bLen, y - bPad);
      this.ctx.lineTo(x + bPad, y - bPad);
      this.ctx.lineTo(x + bPad, y - bPad + bLen);
      this.ctx.stroke();

      // Bottom-Left
      this.ctx.beginPath();
      this.ctx.moveTo(x - bPad, y + bPad - bLen);
      this.ctx.lineTo(x - bPad, y + bPad);
      this.ctx.lineTo(x - bPad + bLen, y + bPad);
      this.ctx.stroke();

      // Bottom-Right
      this.ctx.beginPath();
      this.ctx.moveTo(x + bPad - bLen, y + bPad);
      this.ctx.lineTo(x + bPad, y + bPad);
      this.ctx.lineTo(x + bPad, y + bPad - bLen);
      this.ctx.stroke();

      // Slow Rotating Crosshair Reticle Ticks
      const rotAngle = (now / 2400) % (Math.PI * 2);
      this.ctx.save();
      this.ctx.translate(x, y);
      this.ctx.rotate(rotAngle);
      this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
      this.ctx.lineWidth = 1.5;
      const tDist = r + 4;
      for (let a = 0; a < 4; a++) {
        this.ctx.rotate(Math.PI / 2);
        this.ctx.beginPath();
        this.ctx.moveTo(0, tDist);
        this.ctx.lineTo(0, tDist + 4);
        this.ctx.stroke();
      }
      this.ctx.restore();

      // Floating Tactical Lock Telemetry Tag (Above Node)
      const lockTag = `TARGET LOCKED // ${tState}`;
      this.ctx.font = 'bold 9px "JetBrains Mono", monospace';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'bottom';
      const tagW = this.ctx.measureText(lockTag).width + 12;
      this.ctx.fillStyle = 'rgba(10, 14, 22, 0.95)';
      this.ctx.fillRect(x - tagW / 2, y - bPad - 16, tagW, 16);
      this.ctx.strokeStyle = targetGlowColor;
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(x - tagW / 2, y - bPad - 16, tagW, 16);
      this.ctx.fillStyle = targetGlowColor;
      this.ctx.fillText(lockTag, x, y - bPad - 4);

    } else if (isSearching) {
      // 2. SEARCHING / LOOKING CAMERAS: Animated Amber Radar Sweep Fan & Dashed Pulse
      const sweepAngle = (now / 700) % (Math.PI * 2);

      // Rotating Radar Sector Fan
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.moveTo(x, y);
      this.ctx.arc(x, y, r + 24, sweepAngle, sweepAngle + 0.6);
      this.ctx.closePath();
      this.ctx.fillStyle = 'rgba(245, 158, 11, 0.25)';
      this.ctx.fill();
      this.ctx.restore();

      // Pulsing Dashed Amber Ring
      const searchPulseR = r + 6 + ((now % 1200) / 1200) * 16;
      this.ctx.beginPath();
      this.ctx.arc(x, y, searchPulseR, 0, Math.PI * 2);
      this.ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([4, 4]);
      this.ctx.stroke();
      this.ctx.setLineDash([]);

      // Floating Tactical Search Badge (Above Node)
      const candScore = this.candidateScores[node.camera_id];
      const rad = (this.searchProgress && this.searchProgress.search_radius) ? this.searchProgress.search_radius : 1;
      const searchTag = candScore != null ? `SEARCHING R=${rad} | Sim: ${candScore.toFixed(2)}` : `SEARCHING R=${rad}`;

      this.ctx.font = 'bold 9px "JetBrains Mono", monospace';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'bottom';
      const sTagW = this.ctx.measureText(searchTag).width + 12;
      this.ctx.fillStyle = 'rgba(10, 14, 22, 0.95)';
      this.ctx.fillRect(x - sTagW / 2, y - r - 22, sTagW, 16);
      this.ctx.strokeStyle = '#f59e0b';
      this.ctx.lineWidth = 1;
      this.ctx.strokeRect(x - sTagW / 2, y - r - 22, sTagW, 16);
      this.ctx.fillStyle = '#f59e0b';
      this.ctx.fillText(searchTag, x, y - r - 10);
    }

    // 3. Node Main Body Background
    this.ctx.beginPath();
    this.ctx.arc(x, y, r, 0, Math.PI * 2);
    this.ctx.fillStyle = isSelected ? '#1e293b' : (isActiveTarget ? '#09252a' : (isSearching ? '#271d0e' : '#0c1322'));
    this.ctx.fill();

    let borderColor = '#00f0ff';
    if (!node.enabled || status === 'offline') borderColor = '#475569';
    if (isSearching) borderColor = '#f59e0b';
    if (isActiveTarget) borderColor = '#10b981';
    if (isSelected) borderColor = '#ffffff';

    this.ctx.strokeStyle = borderColor;
    this.ctx.lineWidth = isSelected ? 3.5 : (isActiveTarget ? 3 : 2);
    this.ctx.stroke();

    // 4. Center Camera Optic Icon / Reticle
    this.ctx.beginPath();
    this.ctx.arc(x, y, 8, 0, Math.PI * 2);
    this.ctx.fillStyle = borderColor;
    this.ctx.fill();

    // Blinking Live Recording Dot in Center
    if (isActiveTarget || isSearching) {
      const blink = Math.floor(now / 500) % 2 === 0;
      if (blink) {
        this.ctx.beginPath();
        this.ctx.arc(x, y, 3, 0, Math.PI * 2);
        this.ctx.fillStyle = isActiveTarget ? '#ef4444' : '#fbbf24';
        this.ctx.fill();
      }
    }

    // 5. Label Badge Below Node
    this.ctx.font = '11px "Inter", sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'top';

    const text = node.name || node.camera_id;
    const tw = this.ctx.measureText(text).width;

    this.ctx.fillStyle = 'rgba(10, 14, 22, 0.95)';
    this.ctx.fillRect(x - tw / 2 - 8, y + r + 7, tw + 16, 20);
    this.ctx.strokeStyle = isSelected ? '#ffffff' : (isActiveTarget ? 'rgba(16, 185, 129, 0.6)' : (isSearching ? 'rgba(245, 158, 11, 0.6)' : 'rgba(255, 255, 255, 0.12)'));
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(x - tw / 2 - 8, y + r + 7, tw + 16, 20);

    this.ctx.fillStyle = isSelected ? '#ffffff' : (isActiveTarget ? '#4edea3' : (isSearching ? '#fcd34d' : '#e2e8f0'));
    this.ctx.fillText(text, x, y + r + 11);

    this.ctx.restore();
  }

  drawMinimap(screenW, screenH) {
    const mapW = 160;
    const mapH = 100;
    const mapX = screenW - mapW - 14;
    const mapY = screenH - mapH - 14;

    this.ctx.save();

    // Background & Frame
    this.ctx.fillStyle = 'rgba(8, 12, 20, 0.88)';
    this.ctx.fillRect(mapX, mapY, mapW, mapH);
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(mapX, mapY, mapW, mapH);

    // Title
    this.ctx.font = 'bold 8px "JetBrains Mono", monospace';
    this.ctx.fillStyle = 'rgba(0, 240, 255, 0.6)';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';
    this.ctx.fillText('RADAR MINIMAP // SEC-07', mapX + 8, mapY + 6);

    // Find bounding box of all nodes
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    this.nodes.forEach(n => {
      minX = Math.min(minX, n.position_x);
      maxX = Math.max(maxX, n.position_x);
      minY = Math.min(minY, n.position_y);
      maxY = Math.max(maxY, n.position_y);
    });

    const boxW = Math.max(200, maxX - minX + 100);
    const boxH = Math.max(200, maxY - minY + 100);
    const scaleX = (mapW - 24) / boxW;
    const scaleY = (mapH - 28) / boxH;
    const mScale = Math.min(scaleX, scaleY);

    const offsetX = mapX + 12 + (mapW - 24 - (maxX - minX) * mScale) / 2 - minX * mScale;
    const offsetY = mapY + 20 + (mapH - 28 - (maxY - minY) * mScale) / 2 - minY * mScale;

    // Mini Edges
    this.ctx.beginPath();
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
    this.ctx.lineWidth = 1;
    for (const edge of this.edges) {
      const srcId = edge.source_camera_id || edge.source;
      const tgtId = edge.target_camera_id || edge.target;
      const src = this.nodes.find(n => n.camera_id === srcId);
      const tgt = this.nodes.find(n => n.camera_id === tgtId);
      if (src && tgt) {
        this.ctx.moveTo(src.position_x * mScale + offsetX, src.position_y * mScale + offsetY);
        this.ctx.lineTo(tgt.position_x * mScale + offsetX, tgt.position_y * mScale + offsetY);
      }
    }
    this.ctx.stroke();

    // Mini Nodes
    const now = Date.now();
    for (const node of this.nodes) {
      const nx = node.position_x * mScale + offsetX;
      const ny = node.position_y * mScale + offsetY;
      const isAct = (this.activeCameraId === node.camera_id);
      const isSrch = this.isSearchingNode(node.camera_id);

      this.ctx.beginPath();
      this.ctx.arc(nx, ny, isAct ? 4 : 2.5, 0, Math.PI * 2);
      this.ctx.fillStyle = isAct ? '#00f0ff' : (isSrch ? '#f59e0b' : '#64748b');
      this.ctx.fill();

      if (isAct) {
        // Active node ping in minimap
        const mPing = ((now % 1000) / 1000) * 8;
        this.ctx.beginPath();
        this.ctx.arc(nx, ny, mPing, 0, Math.PI * 2);
        this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
        this.ctx.stroke();
      }
    }

    // Viewport Frustum Box in Minimap
    const vpWorldX = -this.panX / this.scale;
    const vpWorldY = -this.panY / this.scale;
    const vpWorldW = screenW / this.scale;
    const vpWorldH = screenH / this.scale;

    const vpMapX = vpWorldX * mScale + offsetX;
    const vpMapY = vpWorldY * mScale + offsetY;
    const vpMapW = vpWorldW * mScale;
    const vpMapH = vpWorldH * mScale;

    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(vpMapX, vpMapY, vpMapW, vpMapH);

    this.ctx.restore();
  }
}
}
