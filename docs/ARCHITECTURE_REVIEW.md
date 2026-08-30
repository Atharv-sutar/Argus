# Argus Surveillance System: Architecture & Performance Review

## 1. Executive Summary & Root Cause Analysis

This document provides a comprehensive technical reference detailing the system architecture, root causes of previous failures, and the definitive blueprints for ReID, low-latency camera ingestion, and UI operations.

---

## 2. Root Cause Analysis: ReID System

### 2.1 The Broken Multi-View Gallery Scoring (`max_auto_boost`)
- **Location:** `src/reid/gallery.py`
- **Root Cause:** Auto-enrolled viewpoints were artificially capped using:
  $$\text{effective\_score} = \min(\text{auto\_score}, \text{manual\_score} + \text{max\_auto\_boost})$$
  with `max_auto_boost = 0.05`.
- **Impact:** If the operator seeds a target from the front (similarity to back view $\approx 0.45$), when the person turns around, an auto-enrolled back view matching at $0.90$ is artificially clamped to $0.45 + 0.05 = 0.50$. Since `match_threshold = 0.85`, the tracker immediately declares `TARGET LOST`.
- **Fix:** Remove the artificial boost cap. Multi-image appearance gallery matching must be true vectorized maximum cosine similarity across all valid gallery samples:
  $$S(x) = \max_{g \in \text{Gallery}} \cos(x, g)$$

### 2.2 Harmful Color Preprocessing (CLAHE & Gray-World)
- **Location:** `src/reid/color_normalizer.py`, `src/reid/extractor.py`
- **Root Cause:** Gray-World white balancing shifts colored clothing towards neutral gray, while aggressive CLAHE alters subtle fabric texture and gradients.
- **Impact:** Person ReID relies 75%+ on color distribution and texture. Color normalization destroyed discriminative features across cameras.
- **Fix:** Bypass Gray-World and CLAHE. Use standard RGB normalization with ImageNet mean/std $(\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225])$.

### 2.3 Unrealistic Cosine Similarity Thresholds
- **Location:** `src/core/config.py`, `configs/default.yaml`
- **Root Cause:** `match_threshold` was set to `0.85`.
- **Impact:** Real-world cross-view cosine similarity for true matches with OSNet-x1.0/x0.25 on MSMT17 typically ranges between **0.60 and 0.75**. A threshold of 0.85 caused near 100% false rejection rates.
- **Fix:** Update default thresholds to realistic values:
  - `match_threshold`: `0.65`
  - `auto_add_threshold`: `0.80`
  - `diversity_threshold`: `0.92`

### 2.4 Preprocessing & Aspect Ratio Distortion
- **Location:** `src/reid/extractor.py`
- **Root Cause:** Direct resizing of square or distorted person crops to $256 \times 128$ without aspect ratio preservation.
- **Fix:** Implement letterbox padding with aspect ratio $1:2$ before resizing to $256 \times 128$.

---

## 3. Root Cause Analysis: Camera Latency (20s–60s)

### 3.1 Synchronous DirectShow Probing
- **Location:** `src/multi_camera/ui_server.py` (`probe_local_webcams`)
- **Root Cause:** Iterating through indices `0..3` and synchronously calling `cv2.VideoCapture(i, cv2.CAP_DSHOW)` blocks DirectShow COM for 5–10s on missing or busy devices.
- **Impact:** 20 to 40 seconds of complete server freeze whenever cameras are discovered or probed.
- **Fix:** Use fast non-blocking Windows device query or multi-threaded asynchronous probing with 150ms timeout.

### 3.2 OpenCV Driver Buffer Bloat
- **Location:** `src/camera/capture.py` (`OpenCVCamera`)
- **Root Cause:** OpenCV and DirectShow driver buffers hold up to 30–60 frames. If consumer processing is slightly slower than producer, stale frames accumulate.
- **Fix:** Enforce dedicated background grabber thread with a **1-frame ring buffer** (`_latest_frame`). Stale frames are dropped immediately, guaranteeing $<50\text{ms}$ latency.

### 3.3 Synchronous Worker Creation in Pipeline Step
- **Location:** `src/pipeline/multi_camera_pipeline.py`
- **Root Cause:** Inactive/standby cameras were opened synchronously inside `step()`, stalling the main inference loop if any camera is slow or offline.
- **Fix:** Worker initialization and frame acquisition are fully non-blocking.

---

## 4. Root Cause Analysis: UI Faults

| Component | Root Cause | Solution |
| :--- | :--- | :--- |
| **Detect Cameras** | DirectShow blocking probe. | Asynchronous fast probing returning immediately. |
| **Save Topology** | Pipeline graph was not synchronizing live stream bindings. | Automatically reload live matrix and re-initialize video streams when topology is saved. |
| **Quit Button** | Main thread stuck on blocking `cv2` calls while quit handler calls `os._exit()`. | Signal capture threads to terminate cleanly, release capture handles, and shut down gracefully. |

---

## 5. Real-World Manual Testing Protocols

All validations must be performed using physical cameras and real conditions.

### Protocol 1: Zero-Lag Ingestion & Camera Discovery
- **Action:** Open Topology Map, click "Detect Cameras", add cameras, and click "Save Topology".
- **Success Criteria:** Detection finishes in $<1\text{s}$. Live matrix switches seamlessly to newly saved cameras with zero lag.

### Protocol 2: 360° Rotation Target Retention
- **Action:** Select target person. Have target rotate 360° slowly and walk in and out of frame.
- **Success Criteria:** Gallery accumulates diverse viewpoints. Target state remains `TRACKING` and re-identifies with similarity $\ge 0.68$.

### Protocol 3: Occlusion & Crossing
- **Action:** Person B walks in front of Person A (target).
- **Success Criteria:** State indicates `OCCLUDED`. Target lock remains on Person A and never jumps to Person B.

### Protocol 4: Multi-Camera Hand-Off
- **Action:** Target walks from Camera 1 to Camera 2.
- **Success Criteria:** Camera 1 transitions to `LOST`, Camera 2 activates search, recognizes target, and becomes active focus.
