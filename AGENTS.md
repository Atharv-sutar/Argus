# Argus Multi-Camera Surveillance System // Agent Guide

> This file and the `.agents/` directory are **for AI coding agents**. Human users and operators should start at [`README.md`](file:///e:/MAJOR_PROJECT/Project/README.md) and [`docs/`](file:///e:/MAJOR_PROJECT/Project/docs/).

## System Overview
Argus is an AI-powered real-time multi-camera surveillance and person tracking operations center.
- **Detector**: YOLO11 (`auto` hardware-adaptive model/engine selection)
- **Tracker**: BoT-SORT (Kalman filter + GMC + Hungarian association)
- **ReID**: `person_reid_youtu` (768D ONNX/CUDA/TensorRT default) with modular registry (`osnet_x1_0`, `osnet_x0_25`, `mobilenet_v3_small`)
- **Multi-Camera Graph**: Dynamic topological search with distance & timeout radius expansion
- **Identity & Storage**: FAISS vector search with SQLite persistence
- **Frontend UI**: Vanilla JavaScript + CSS variables + HTML5 Canvas web operations center (`http://127.0.0.1:8765`)

## Python Environment
Always execute tools and tests through the project virtual environment:
```powershell
.\.venv\Scripts\python -m pytest <test_path>
.\.venv\Scripts\python -m src.app.main
```

## Agent Customizations Directory (`.agents/`)
- **Context & Bug Register**:
  - `.agents/CONTEXT.md`: Persistent engineering memory, EPFL ReID benchmarks, runtime architecture invariants, and legacy code inventory.
  - `.agents/BUGS.md`: Active volatile bug register.
- **Rules (`.agents/rules/`)**:
  - `project-core.md`: Architectural principles and priorities.
  - `git-workflow.md`: Mandatory automatic commit & push instructions.
  - `testing-rules.md`: Virtual env and ReID dataset testing guidelines (no synthetic ReID).
  - `bug-tracking.md`: Volatile issue tracking in `.agents/BUGS.md`.
  - `code-quality.md`: Typing, dataclass contracts, and resource management.
  - `gpu-performance.md`: Hardware-adaptive GPU profiling, VRAM scaling, and inference constraints.
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

## Persistent Context Reference
- Read [`.agents/CONTEXT.md`](file:///e:/MAJOR_PROJECT/Project/.agents/CONTEXT.md) whenever debugging ReID, tracking, target selection/rendering, multi-camera UI wiring, or reviewing removed/redundant legacy code and empirical EPFL benchmarks.
- Consult [`docs/MIGRATION_PERFORMANCE.md`](file:///e:/MAJOR_PROJECT/Project/docs/MIGRATION_PERFORMANCE.md) before proposing native C++/CUDA/Rust optimizations or runtime dependency migrations.
