# Argus AI — User & Operations Guide

Welcome to the **Argus Real-Time Multi-Camera Surveillance Operations Center**.

---

## 1. Launching Argus

### One-Click / Self-Healing Launchers
Both launchers detect `.venv` (creating it automatically if absent), verify required packages, log detected GPU/CPU capabilities, and start the Web Operations Center at **`http://127.0.0.1:8765`**:

- **PowerShell**: `.\run.ps1`
- **Command Prompt (CMD)**: `run.bat`
- **Direct Python**: `.\.venv\Scripts\python -m src.app.main`

Environment variables recognized by launchers/runtime:
- `ARGUS_NO_BROWSER=1`: Starts the web server without automatically opening a browser tab.
- `ARGUS_NO_PAUSE=1`: Skips interactive `pause` on error exit in `run.bat` (useful for CI/scripts).

---

## 2. Command-Line Modes & Flags

All CLI flags pass directly through `.\run.ps1`, `run.bat`, or `python -m src.app.main`:

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--config` | `configs/default.yaml` | Path to YAML configuration |
| `--graph` | `configs/camera_graph.json` | Path to camera topology graph JSON |
| `--ui-port` | `8765` (or YAML `multi_camera.ui_port`) | HTTP port for Web Operations Center |
| `--source` | `None` | Single camera override: webcam index (`0`, `1`), video file path, RTSP/HTTP stream URL, or `synthetic` |
| `--video-dir` | `None` | Auto-load folder of multi-camera video files (e.g. `"test_footage/EPFL campus"`) |
| `--device` | `auto` | Force execution device (`auto`, `cuda`, `cpu`) |
| `--target-id` | `None` | Pre-select a specific tracker ID at startup |
| `--synthetic` | `False` | Run with built-in synthetic frame simulator |
| `--map-ui` | `False` | Launch only the interactive Camera Mapping & Topology UI |
| `--legacy-desktop` | `False` | Deprecated OpenCV desktop window mode (debug fallback only) |

### Examples

```powershell
# Default multi-camera web operations center (reads configs/camera_graph.json)
.\run.ps1

# Load recorded multi-camera footage folder for synced playback
.\run.ps1 --video-dir "test_footage/EPFL campus"

# Local USB webcam #0 forced on CPU
.\run.ps1 --source 0 --device cpu

# Synthetic simulation feed on custom UI port
.\run.ps1 --synthetic --ui-port 8770

# Topology Map configuration editor only
.\run.ps1 --map-ui
```

---

## 3. Web Operations Center Controls & Workflows

Open **`http://127.0.0.1:8765`** in your browser:

| Action | Control / Hotkey | Behavior |
| :--- | :--- | :--- |
| **Select / Lock Target** | **Click any tracked person** (`Shift + Click` to lock/switch cleanly) | Seeds the ground-truth appearance anchor, initializes a fresh forensic trail, and locks onto that target. |
| **Add Manual Viewpoint** | **`A` key** or **"Add Angle"** button | Enrolls the target's current angle (front/back/side) as a trusted manual gallery reference. |
| **Clear / Release Target** | **`C` key** or **"Clear Target"** button | Clears active target lock, discards unexported clip buffer, resets search radius and trail. |
| **Undo / Redo Action** | **`Ctrl + Z`** / **Undo / Redo** buttons | Reverts accidental target corrections or gallery removals. |
| **Topology Map** | **Topology tab** | Auto-detect local webcams, test previews, add RTSP/file/webcam nodes, drag edges & transit times, and save live to `configs/camera_graph.json`. |
| **Source Switching** | **Live View / Video File** buttons | Switch between multi-camera recorded playback directories/files and probed live webcams cleanly at runtime. |
| **Cases & Forensic Export** | **Cases modal / Export** buttons | Create, open, close (restoring default gallery/annotation DB paths), and export forensic ZIP evidence + stitched MP4 journeys. |
| **Stop Server** | **Quit button** in UI or **`Ctrl + C`** in terminal | Cleanly stops workers, flushes database state, releases capture handles, and shuts down the HTTP server. |

*(In deprecated `--legacy-desktop` mode, press `q` or `Esc` inside the OpenCV window to exit.)*

---

## 4. Hardware Adaptation & Troubleshooting

- **Hardware-Adaptive Inference**: Argus automatically profiles your hardware via `detect_hardware_profile()` ([src/inference/device.py](file:///e:/MAJOR_PROJECT/Project/src/inference/device.py)) on every launch:
  - **Compact GPUs (`<6 GB`, e.g., GTX 1650)**: Conservative VRAM caps, `yolo11n`, smaller detection/ReID batches, FP16 where supported.
  - **Mid / High-VRAM RTX GPUs (`>=6 GB`)**: Unlocks higher batch sizes, exhaustive cuDNN autotuning, TensorRT/TF32 on Ampere+, and lower ReID interval cadence.
  - **CPU Hosts**: Prefers exported `yolo11n.onnx` + ONNX Runtime `CPUExecutionProvider`.
- **Missing Python on Windows**: `run.bat` and `run.ps1` automatically attempt `winget install Python.Python.3.12` if Python is absent from `PATH`.
- **Automated Tests**: Run all unit and integration tests via:
  ```powershell
  .\.venv\Scripts\python -m pytest tests/unit tests/integration -q -r a
  ```
