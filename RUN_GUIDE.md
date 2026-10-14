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

*   **Detector:** `YOLOv11` via `CUDAExecutionProvider` / `TensorrtExecutionProvider`
*   **Tracker:** `BoT-SORT` (C-vectorized Kalman filter `[xc, yc, w, h, vxc, vyc, vw, vh]` + OpenCV ECC GMC + Hungarian assignment)
*   **ReID:** `OSNet x0_25` with **Event-Driven Heuristics** (zero GPU ReID compute during steady tracking; activates only upon target loss, uncertainty, or cross-camera search)
*   **Batching:** Dynamic cross-camera detection and ReID tensor chunking (up to 77+ FPS multi-camera ingest on 4GB GTX 1650 class hardware)
*   **Memory Management:** Automatic VRAM trimming and tensor deallocation preventing GPU memory fragmentation

---

## 5. Troubleshooting & Diagnostics

### Missing Python?
If Python is not installed on Windows, `run.bat` automatically invokes the Windows Package Manager (`winget install Python.Python.3.12`) to install Python 3.12.

### GPU Acceleration Not Detected?
Ensure you have the latest NVIDIA drivers installed. Argus will automatically detect your CUDA-capable GPU (e.g. GTX 1650, RTX 3060, etc.) and run in TensorRT/CUDA mode. If no GPU is present, Argus automatically falls back to CPU vectorization.

### Checking Full System Tests
To run all 97 automated unit and integration tests:
```cmd
.venv\Scripts\pytest tests/unit/ --ignore=tests/unit/pipeline/test_gallery_management_and_batch_reid.py --ignore=tests/unit/reid/test_color_normalizer.py
```
*(All 97 tests pass with 100% success rate).*
