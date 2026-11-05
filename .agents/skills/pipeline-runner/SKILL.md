---
name: pipeline-runner
description: Step-by-step runbook for launching and operating the Argus surveillance pipeline across live webcam, recorded video directory, synthetic simulation, and headless test modes.
---

# Pipeline Runner Runbook

Use this skill whenever you need to execute, test, or demonstrate the Argus surveillance application.

## 1. Environment Verification

Before running, ensure you use the project virtual environment:
```powershell
.\.venv\Scripts\python --version
```

Verify model files exist in the project root:
- `yolo11n.onnx` or `yolo11n.pt` (Object Detection)
- `models/osnet_x0_25.onnx` or `models/osnet_x1_0.onnx` (ReID)

## 2. Execution Modes

### A. Live Interactive GUI Server (Default Mode)
Runs the backend surveillance pipeline and opens the Web Command Center at `http://127.0.0.1:8798`:
```powershell
.\.venv\Scripts\python -m src.app.main
```
- Interactive prompt allows selecting between default local webcams, synthetic feeds, or custom RTSP/video files.
- Automatically launches browser UI.

### B. Pre-recorded Multi-Camera Video Directory Mode
Runs against a multi-camera recorded video folder (e.g. EPFL campus footage):
```powershell
.\.venv\Scripts\python -m src.app.main --dir "test_footage/EPFL campus"
```
Or use the **"Load Videos"** button in the Web UI Live Matrix toolbar.

### C. Synthetic Simulation Mode (Headless / Camera-less Testing)
Useful for automated testing and CI where no physical camera hardware is connected:
```powershell
.\.venv\Scripts\python -m src.app.main --source synthetic
```

### D. Specific Camera Index / RTSP URL
```powershell
.\.venv\Scripts\python -m src.app.main --source 0
.\.venv\Scripts\python -m src.app.main --source "rtsp://user:pass@192.168.1.100:554/stream"
```

## 3. Keyboard & Operator Shortcuts in UI
- `A` / Right-Click: Capture current angle crop to Target Gallery.
- `C`: Clear target focus and reset search state.
- `Space`: Toggle play/pause (in recorded video playback mode).
- `Ctrl+Z`: Undo last target selection or gallery action.
- `Ctrl+Y`: Redo last target action.
- `S`: Open Advanced Settings (ReID model, search radius, thresholds).
- `Q`: Safe shutdown of camera workers and clean exit.

## 4. Verification Checkpoints
- Web UI opens at `http://127.0.0.1:8798` without port collision.
- Telemetry SSE connection is established (`status-uptime` ticks upward).
- Camera tiles stream video or show `STANDBY` when idle.
