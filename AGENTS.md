# Argus Multi-Camera Surveillance System // Agent Guide

## System Overview
Argus is an AI-powered real-time multi-camera surveillance and person tracking operations center.
- **Detector**: YOLO11n (`yolo11n.onnx` / `yolo11n.pt`)
- **Tracker**: BoT-SORT (Kalman filter + GMC + Hungarian association)
- **ReID**: OSNet (`models/osnet_x0_25.onnx` default for low VRAM, `models/osnet_x1_0.onnx` high compute)
- **Multi-Camera Graph**: Dynamic topological search with distance & timeout radius expansion
- **Identity & Storage**: FAISS vector search with SQLite persistence
- **Frontend UI**: Vanilla JavaScript + CSS variables + HTML5 Canvas web operations center (`http://127.0.0.1:8798`)

## Python Environment
Always execute tools and tests through the project virtual environment:
```powershell
.\.venv\Scripts\python -m pytest <test_path>
.\.venv\Scripts\python -m src.app.main
```

## Agent Customizations Directory (`.agents/`)
- **Rules (`.agents/rules/`)**:
  - `project-core.md`: Architectural principles and priorities.
  - `git-workflow.md`: Mandatory automatic commit & push instructions.
  - `testing-rules.md`: Virtual env and ReID dataset testing guidelines (no synthetic ReID).
  - `bug-tracking.md`: Volatile issue tracking in `BUGS.md`.
  - `code-quality.md`: Typing, dataclass contracts, and resource management.
  - `gpu-performance.md`: Constraints for 4GB GTX 1650 class hardware.
  - `frontend-architecture.md`: Vanilla JS, unique DOM IDs, and telemetry standards.
- **Skills (`.agents/skills/`)**:
  - `karpathy-guidelines`: Caution-biased engineering principles (unmodified).
  - `pipeline-runner`: Runbook for launching live, directory, and synthetic modes.
  - `performance-profiler`: Profiling FPS, latency, and GPU memory benchmarks.
  - `camera-topology`: Multi-camera graph topology and transition time guidelines.
  - `plan-and-execute`: Planning & model handoff workflow.
  - `deep-cleaning`: Systematic codebase cleanup process.
- **Workflows (`.agents/workflows/`)**:
  - `feature-development.md`: Five-step feature development pipeline.
  - `bug-fix.md`: Step-by-step bug reproduction and resolution workflow.
  - `release-check.md`: Pre-release quality assurance checklist.
