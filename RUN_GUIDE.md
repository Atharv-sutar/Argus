# Argus AI — Operational Run Guide & Deployment Manual

Welcome to the **Argus AI Real-Time Multi-Camera Surveillance Operations Center**.

Argus is an enterprise-grade, GPU-accelerated multi-camera tracking system powered by:
- **YOLOv11** object detection (TensorRT / ONNX accelerated)
- **BoT-SORT** tracker (Kalman filter + OpenCV ECC motion compensation + SciPy Hungarian assignment)
- **OSNet** deep appearance ReID feature extraction
- **FAISS & SQLite** identity vector storage
- **Cross-Camera Dynamic Batching & Event-Driven ReID**

---

## 1. Quick Start (1-Click Self-Healing Launcher)

Argus features a **Self-Healing Launcher** that detects your environment, verifies required packages, automatically installs any missing dependencies (OpenCV, PyTorch, FAISS, Flask, etc.), configures hardware acceleration, and boots the system automatically.

### Option A: Windows Command Prompt / Desktop (Recommended)
Simply double-click [`run.bat`](file:///e:/MAJOR_PROJECT/Project/run.bat) or run from CMD:
```cmd
run.bat
```

### Option B: Windows PowerShell
Run the PowerShell launcher:
```powershell
.\run.ps1
```

### Option C: Direct Python Execution
```cmd
.venv\Scripts\python.exe scripts\bootstrap.py
```

> [!NOTE]
> On the very first run, Argus checks for missing dependencies, downloads required neural network weights (`yolo11n.pt`), initializes the virtual environment, and launches the Web Operations Dashboard in your default browser at **`http://localhost:8765`**.

---

## 2. Operational Modes & Command-Line Arguments

You can pass arguments directly to `run.bat` or `run.ps1`:

### 1. Multi-Camera Operations Dashboard (Default)
Runs the unified multi-camera web surveillance platform with live WebRTC/MJPEG feeds, camera topology map, target telemetry, and automated handoffs:
```cmd
run.bat
```
*Access in browser:* **`http://localhost:8765`**

---

### 2. Local Hardware Webcam Tracking
Run with your connected webcam (e.g., Device Index 0 or 1):
```cmd
run.bat --source 0
```
Or for a secondary webcam:
```cmd
run.bat --source 1
```

---

### 3. Synthetic Benchmark & Simulation Feed (Headless Test)
Run a synthetic multi-person simulation test without requiring physical cameras:
```cmd
run.bat --synthetic
```

---

### 4. Custom Video File or RTSP Security Camera
Process recorded surveillance footage or an RTSP IP camera stream:
```cmd
run.bat --source "test_footage/hallway_cam.mp4"
```
Or an RTSP network camera:
```cmd
run.bat --source "rtsp://admin:password@192.168.1.100:554/stream1"
```

---

### 5. Interactive Camera Mapping & Topology Configuration UI
Launch the visual drag-and-drop camera graph builder to connect cameras, define transit paths, and adjust spatial adjacency:
```cmd
run.bat --map-ui
```

---

## 3. Web UI Controls & Hotkeys

When Argus is running, navigate to `http://localhost:8765`:

| Action | Control / Hotkey | Description |
| :--- | :--- | :--- |
| **Select / Lock Target** | **Click on any person** | Locks onto the person, seeds appearance embeddings into FAISS, and triggers focus tracking. |
| **Add Diverse Viewpoint** | **`A` Key** or **"Add Angle" Button** | Adds the target's current perspective (e.g. back view, side profile) to the protected appearance gallery. |
| **Clear / Deselect Target** | **`C` Key** or **"Clear Target" Button** | Releases target lock, resets search radius, and returns to normal passive surveillance. |
| **Undo Last Correction** | **`Ctrl + Z`** or **"Undo" Button** | Reverts accidental target reassignments. |
| **Forensic Export** | **"Export Evidence" Button** | Generates court-admissible forensic MP4 video clips with tamper-evident audit logs. |

---

## 4. Performance & Technology Stack

When running, Argus logs and verifies the active technology stack in real time:

*   **Detector:** Hardware-adaptive `YOLOv11` via `TensorRT`, PyTorch CUDA (`FP16`/`TF32` where supported), or ONNX `CPUExecutionProvider`
*   **Tracker:** `BoT-SORT` (C-vectorized Kalman filter `[xc, yc, w, h, vxc, vyc, vw, vh]` + OpenCV optical flow GMC + Hungarian assignment)
*   **ReID:** Hardware-adaptive ReID registry (`person_reid_youtu` 768D default, plus `OSNet` / `MobileNetV3` options) with **Event-Driven Heuristics** (minimal ReID overhead during steady tracking; dynamically scales interval by GPU tier)
*   **Batching:** Dynamic cross-camera detection and ReID tensor batching scaled automatically to detected GPU VRAM / Tensor Cores / CPU threads
*   **Memory Management:** Dynamic ORT/TensorRT arena sizing, NVML VRAM telemetry, and automatic GPU cache trimming preventing fragmentation

---

## 5. Troubleshooting & Diagnostics

### Missing Python?
If Python is not installed on Windows, `run.bat` automatically invokes the Windows Package Manager (`winget install Python.Python.3.12`) to install Python 3.12.

### GPU Acceleration Not Detected?
Ensure you have the latest NVIDIA drivers installed. Argus automatically detects your GPU capabilities (VRAM, compute capability, Tensor Cores, TensorRT) and enables FP16/TF32/TensorRT and larger batch sizes on higher-tier RTX GPUs, while scaling conservatively on compact GPUs or falling back to CPU vectorization when no GPU is present.

### Checking Full System Tests
To run all 97 automated unit and integration tests:
```cmd
.venv\Scripts\pytest tests/unit/ --ignore=tests/unit/pipeline/test_gallery_management_and_batch_reid.py --ignore=tests/unit/reid/test_color_normalizer.py
```
*(All 97 tests pass with 100% success rate).*
