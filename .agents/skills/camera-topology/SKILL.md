---
name: camera-topology
description: Procedures for defining, editing, validating, and updating multi-camera network graphs (nodes, edges, overlap/adjacent transitions, and physical transit times).
---

# Camera Topology Management Runbook

Use this skill when adding cameras, establishing spatial relationships, or calibrating transition times between surveillance nodes.

## 1. Graph File Location
- Primary configuration: `configs/camera_graph.json`
- Backup configuration: `configs/camera_graph.json.bak`

## 2. Node Schema & Properties

Every camera node requires:
- `camera_id`: Unique identifier (e.g. `cam_0`, `cam_1`, `cam_entrance`).
- `name`: Human-readable display label (e.g. `Main Entrance Lobby`).
- `source`: Device index (e.g. `0`, `1`), RTSP URL (`rtsp://...`), HTTP stream, or video file path.
- `source_type`: One of `"webcam"`, `"rtsp"`, `"video_file"`, `"synthetic"`.
- `enabled`: Boolean (`true` / `false`).
- `floor` / `zone` / `group`: Spatial tags for filtering in the UI.

## 3. Edge Schema & Transit Times

Edges represent physical travel corridors between cameras:
- `source`: Starting camera ID.
- `target`: Destination camera ID.
- `edge_type`:
  - `"overlap"`: Shared field of view (instantaneous transition, 0.0s).
  - `"adjacent"`: Immediate adjacent room or doorway (typical transit: 1–5s).
  - `"travel"`: Longer hallway, stairwell, or corridor (typical transit: 5–60s).
- `direction`: `"bidirectional"` or `"unidirectional"`.
- `expected_min_transition_s`: Minimum possible physical transit time (>= 0).
- `expected_typical_transition_s`: Expected transit time.
- `expected_max_transition_s`: Maximum threshold before target is considered lost.

> **Constraint Rule**: `expected_min <= expected_typical <= expected_max`. Negative transition times are invalid.

## 4. Topology Validation

Validate graph consistency via automated tests:
```powershell
.\.venv\Scripts\python -m pytest tests/unit/multi_camera/test_camera_graph.py
```
Or programmatically in Python:
```python
from pathlib import Path
from src.multi_camera.camera_graph import CameraGraph

graph = CameraGraph.load(Path("configs/camera_graph.json"))
errors = graph.validate()
if errors:
    print(f"Topology errors found: {errors}")
else:
    print(f"Topology is valid! {len(graph.cameras)} cameras, {len(graph.edges)} edges.")
```

## 5. UI Topology Editor
Operators can also interactively manage nodes and draw edges visually via the **Topology Map** tab in the Argus Web Command Center (`http://127.0.0.1:8798`), with live webcam auto-probing and bulk RTSP import.
