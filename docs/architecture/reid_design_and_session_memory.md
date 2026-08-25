# ReID Architecture, Deep-Dive Analysis & Session Memory

## 1. Executive Summary & Session Log

This document serves as the persistent repository session memory and architectural blueprint for the Argus Multi-Camera Person-ReID subsystem.

### Historical Bug & Fix Registry
| Issue ID | Root Cause | Implemented Solution | Verification Status |
| :--- | :--- | :--- | :--- |
| **BUG-01: Zero Discrimination** | MobileNetV3 classification backbone + ReLU clamping yielded identical cosine similarity ($0.602$ vs $0.607$) for true targets and impostors. | Integrated standalone **OSNet-x0.25** person-ReID backbone; eliminated terminal ReLU clamping; transitioned to score-level consensus. | **FIXED** (Sep margin: $+0.223$). |
| **BUG-02: Multi-Person Memory Contamination** | Unconstrained background auto-acquisition (`add_reference_sample` & `verified_update`) continuously ingested surrounding people and blurred crops into the prototype gallery. | **Ground-Truth Target Immutability**: Frozen target representation strictly to operator-clicked ground-truth anchor. | **FIXED** (Impostors rejected $100\%$). |
| **BUG-03: Distant Camera False Locking** | Non-search, faraway cameras (`cam_0` when target in `cam_2`) independently ran single-camera candidate recovery. | **Multi-Camera Graph Spatial Gating**: Restricted candidate evaluation strictly to the active camera and authorized radius search set. | **FIXED** (No simultaneous multi-camera locks). |
| **BUG-04: Cross-Camera Quality/Color Drop** | Different camera sensors exhibit distinct auto-white-balance, exposure, and color temperature curves, penalizing raw HSV color histograms. | Color constancy preprocessing & multi-component balancing (Analyzed in Section 2). | **ANALYZED / PLANNED**. |
| **BUG-05: Slow-Crossing Tracker Swap** | When Person B crosses slowly in front of Person A, spatial Kalman tracking drifts and swaps track IDs across the occlusion boundary. | Trajectory-ReID velocity gating & spatial overlap lock-freezing (Analyzed in Section 3). | **ANALYZED / PLANNED**. |

---

## 2. End-to-End ReID Pipeline Analysis

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│ 1. CROP & FRAME │ ──► │ 2. PREPROCESSING    │ ──► │ 3. MULTI-MODAL REID  │ ──► │ 4. MATCHING &      │
│    ACQUISITION  │     │    & COLOR CONSTANCY│     │    REPRESENTATION    │     │    CONSENSUS       │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘     └────────────────────┘
```

### Stage 1: Framing & Bounding Box Extraction
* **Current Approach**:
  - Crops are sliced directly via bounding box coordinates: `frame[y1:y2, x1:x2]`.
* **Identified Weaknesses**:
  - **Background Pixel Contamination**: Rectangular bounding boxes contain $25-40\%$ background pixels (walls, floors, foliage). When a target moves from a light background in Camera 0 to a dark background in Camera 1, the background color pollutes the color histogram.
  - **Perspective & Pose Distortion**: Fixed vertical slicing (`[0.15h:0.55h]` for upper body) fails when the camera has a steep top-down angle or when a person bends or turns diagonally.

### Stage 2: Preprocessing & Cross-Camera Color Constancy
* **Current Approach**:
  - Standard RGB normalization for OSNet; raw BGR-to-HSV for handcrafted color histograms.
* **Identified Weaknesses**:
  - **Sensor White-Balance Discrepancy**: Webcam 0 (warm tint) vs Webcam 1 (cool/green tint) vs Webcam 2 (low-light saturation) causes identical clothing to occupy different HSV bins across cameras.
* **Technological Improvement**:
  - Apply **Gray-World Color Constancy** or **CLAHE (Contrast Limited Adaptive Histogram Equalization)** across the L-channel in Lab space before extracting color features. This normalizes illumination and color temperature across disparate camera hardware.

### Stage 3: Feature Representation (Deep vs Handcrafted)
* **Current Approach**:
  - **OSNet-x0.25 (512D)**: Deep structural/semantic features.
  - **Upper Body (192D)** & **Lower Body (192D)**: HSV + Lab histograms.
  - **Spatial Color & Texture (832D)**: 4 horizontal body stripes.
* **Evaluation**:
  - The combination of **Deep Semantics + Spatial Color Stripes** is mathematically optimal for real-time edge hardware (GTX 1650 class), requiring $< 4\text{ms}$ inference latency.

### Stage 4: Feature Storage & Memory Representation
* **Current Approach**:
  - Single ground-truth crop stored on manual click.
* **Identified Weaknesses**:
  - **Single-View Blindspot**: If the target was selected facing the camera (front view) and enters Camera 1 walking away (back view), appearance details (e.g. front graphic vs plain back) diverge.
* **Technological Improvement**:
  - **Multi-View Keyframe Gallery (Front, Back, Profile)**:
    When the target is actively tracked with high confidence ($S \ge 0.90$) and zero occlusion, save up to 3 distinct view anchors based on aspect ratio or significant orientation shifts.

---

## 3. The Slow-Crossing Tracking Swap: Root Cause & Solution

### Root Cause
When Person B walks slowly in front of Person A:
1. Person B's bounding box occludes Person A ($IoU > 0.30$).
2. ByteTrack's Kalman filter merges detections, then splits them with swapped tracker IDs upon separation.
3. If the pipeline attempts to re-associate or evaluate the swapped track without verifying against the immutable reference prototype, the system transfers the target label to Person B.

### Proposed Architectural Safeguard:
1. **Occlusion Trajectory Freeze**:
   - When bounding box overlap is detected ($IoU > 0.08$ or center distance $< 1.2 \times \text{width}$), the target state immediately enters `OCCLUDED_TRACKING`.
   - ReID matching and track reassignment are **strictly suspended**.
2. **Post-Crossing Identity Re-Verification**:
   - Once the tracks cleanly separate ($IoU = 0.0$), evaluate both departing tracks against the **Immutable Clicked Prototype**.
   - The track with $S \ge 0.78$ and $S_{\text{upper}} \ge 0.60$ is assigned the target.
   - The track with $S \le 0.65$ is rejected as the distractor.

---

## 4. Solid Optimization Plan (For Future Implementation)

### Phase 1: Camera Color Constancy & Preprocessing
* Implement `ColorNormalizer` in `src/reid/` using Gray-World / Lab CLAHE normalization to remove webcam sensor tint differences before feature extraction.
* Implement center-weighted Gaussian masking on person crops to suppress background edge pixels.

### Phase 2: Multi-View Anchor Modeling
* Implement a 3-slot `MultiViewGallery` (Front, Back, Angle) populated strictly during clean, high-confidence ($S \ge 0.90$), unoccluded tracking.
* Candidate matching computes similarity against $\max(S_{\text{front}}, S_{\text{back}}, S_{\text{profile}})$.

### Phase 3: Occlusion Trajectory State Machine
* Add explicit `TargetState.OCCLUDED` in `src/core/types.py`.
* Freeze tracker re-assignment during occlusion; enforce post-separation ReID re-verification against the immutable prototype before resuming active tracking.
