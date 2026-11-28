# Argus — Multi-Camera Surveillance & Target Tracking System

**Argus** is a real-time AI-powered multi-camera surveillance and person re-identification operations center featuring hardware-adaptive GPU inference, graph-driven cross-camera search, forensic clip export, and an interactive web dashboard.

---

## Core Capabilities

- **Hardware-Adaptive Detection**: Ultralytics YOLO11 with runtime CPU/GPU capability profiling (`detect_hardware_profile()`), dynamically scaling batch sizes, FP16/TF32 precision, and TensorRT/CUDA/ONNX backends across entry GPUs, mid/high-VRAM RTX GPUs, and CPU hosts.
- **Motion + Appearance Tracking**: BoT-SORT (Kalman filter `[xc, yc, w, h, dx, dy, dw, dh]`, sparse optical-flow GMC, SciPy Hungarian assignment).
- **Protected Person Re-Identification (ReID)**: `person_reid_youtu` (768D ONNX/CUDA default) with modular backbone registry (`osnet_x1_0`, `osnet_x0_25`, `mobilenet_v3_small`), crop sharpness gating, multi-frame evidence accumulation, and anti-scoop gallery protection (`max_gallery_size: 100`).
- **Topology-Driven Multi-Camera Search**: Connected camera graph ([configs/camera_graph.json](file:///e:/MAJOR_PROJECT/Project/configs/camera_graph.json)) with progressive hop-radius expansion (`R1 -> R2 -> R3`), minimum transit-time spatial gating, and cross-camera handoff confirmation.
- **Forensics, Cases & Continuous NVR**: SQLite-backed FAISS identity persistence, continuous live-camera segment recording, target forensic clip export, timeline playback, and case management.
- **Web Operations Center**: Dependency-free Python HTTP server + Vanilla JS / HTML5 Canvas dashboard at **`http://127.0.0.1:8765`**.

---

## Quick Start

### Windows Launchers (Recommended)

From PowerShell or Command Prompt (auto-creates/uses `.venv`, checks dependencies, and opens the dashboard):

```powershell
# PowerShell
.\run.ps1
```

```cmd
:: Windows Command Prompt (CMD)
run.bat
```

Common launch flags work with both `.\run.ps1` and `run.bat`:

| Mode | Command Example |
| :--- | :--- |
| **Default Web Dashboard** | `.\run.ps1` |
| **Specific Webcam** | `.\run.ps1 --source 0` |
| **Video File / RTSP Stream** | `.\run.ps1 --source "rtsp://admin:pass@192.168.1.100:554/stream1"` |
| **Multi-Camera Video Directory** | `.\run.ps1 --video-dir "test_footage/EPFL campus"` |
| **Synthetic Simulation Feed** | `.\run.ps1 --synthetic` |
| **Force CPU / CUDA Device** | `.\run.ps1 --device cpu` / `.\run.ps1 --device cuda` |
| **Topology Map Editor Only** | `.\run.ps1 --map-ui` |

### Direct Python Execution

```powershell
.\.venv\Scripts\python -m src.app.main
```

### Automated Test Suite

```powershell
.\.venv\Scripts\python -m pytest tests/unit tests/integration -q -r a
```

---

## Documentation Directory

Documentation is strictly split between **human/developer docs** (`README.md`, `docs/`) and **AI coding-agent instructions/memory** (`AGENTS.md`, `.agents/`):

### For Users & Engineers
- [**User & Operations Guide**](file:///e:/MAJOR_PROJECT/Project/docs/USER_GUIDE.md) — Detailed launch modes, CLI options, Web UI controls, case workflows, and troubleshooting.
- [**Manual Testing Guide**](file:///e:/MAJOR_PROJECT/Project/docs/TESTING_GUIDE.md) — Physical webcam/RTSP verification protocols for discovery, 360° ReID, occlusion, and cross-camera handoffs.
- [**System Architecture**](file:///e:/MAJOR_PROJECT/Project/docs/ARCHITECTURE.md) — Module layout, processing pipeline data flow, ReID design rationale, and multi-camera search architecture.
- [**Migration & Performance Roadmap**](file:///e:/MAJOR_PROJECT/Project/docs/MIGRATION_PERFORMANCE.md) — Measured bottlenecks and verified guidance for future native C++/CUDA/Rust optimizations.

### For AI Coding Agents
- [**AGENTS.md**](file:///e:/MAJOR_PROJECT/Project/AGENTS.md) — Entry guide for AI agents (environment, rules, skills, and workflows).
- **`.agents/`** — Agent-only operational files:
  - `.agents/rules/` — Coding, GPU/performance, testing, UI, and git workflow rules.
  - `.agents/skills/` — Task-specific agent playbooks (`pipeline-runner`, `performance-profiler`, `camera-topology`, etc.).
  - `.agents/workflows/` — Repeatable development/release procedures.
  - `.agents/CONTEXT.md` — Persistent compact engineering memory, empirical EPFL benchmarks, and legacy audit inventory.
  - `.agents/BUGS.md` — Volatile open-bug tracker used during debugging workflows.
