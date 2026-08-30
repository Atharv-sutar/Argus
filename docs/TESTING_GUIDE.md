# Argus Surveillance System: Real-World Manual Testing Guide

## Purpose
This document provides standard operating procedures and test scripts for manually verifying the Argus Surveillance operations center under **physical conditions** (USB webcams, RTSP streams, real persons, and multi-camera handoffs).

---

## Pre-Flight Checklist

1. **Option A (Direct Execution via Virtual Environment):**
   ```powershell
   .\.venv\Scripts\python.exe src/app/main.py
   ```

2. **Option B (Activate venv first):**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python src/app/main.py
   ```

3. **Option C (Launcher Script):**
   ```powershell
   .\run.ps1
   # or
   .\run.bat
   ```
   *The web UI will automatically open at `http://127.0.0.1:8765`.*

---

## Test Suite 1: Camera Discovery & Topology Configuration

### Test 1.1: Instant Camera Detection
- **Goal:** Verify non-blocking hardware camera probing.
- **Steps:**
  1. Plug in 1 or 2 USB webcams.
  2. Switch to **Topology Map** mode (top right button).
  3. Click **Auto-Detect** under the Camera Devices pane.
- **Expected Result:**
  - Status changes to "Scanning video capture devices...".
  - Probing completes in **$< 1.5$ seconds**.
  - All available hardware devices appear in the list with resolution (e.g., `640x480 @ 30fps`).
  - Clicking a detected camera displays an instant live preview.

### Test 1.2: Topology Graph Creation & Live Matrix Synchronization
- **Goal:** Verify that saving camera nodes in the graph instantly updates the live surveillance matrix.
- **Steps:**
  1. Add 2 cameras to the topology canvas.
  2. Click the **Connect Tool**, drag an edge from Camera 1 to Camera 2.
  3. Click **Save Topology**.
  4. Switch back to **Live Matrix** mode.
- **Expected Result:**
  - Toast shows `"Camera topology graph saved successfully!"`.
  - Live Matrix immediately displays both camera tiles with active live video feeds.
  - No application restart or page refresh is required.

---

## Test Suite 2: Single-Camera Target Tracking & 360° ReID

### Test 2.1: Target Selection & Seed Gallery Enrollment
- **Goal:** Verify manual target locking from live video feed.
- **Steps:**
  1. Have a person walk into the field of view of Camera 1.
  2. In the Live Matrix view, click directly on the person's bounding box.
- **Expected Result:**
  - Target bounding box turns **Cyan/Green** with label `TARGET [LOCKED] (ID: #X)`.
  - Target card in the right sidebar displays Tracker ID, Camera ID, and state `LOCKED`.
  - Seed crop is captured and enrolled in the Appearance Gallery (`Gallery: 1/25`).

### Test 2.2: 360° Multi-Angle Viewpoint Accumulation
- **Goal:** Verify target lock continuity across full body rotations (front, profile, back).
- **Steps:**
  1. With target locked, have the person stand 2-3 meters from the camera.
  2. Have the person slowly turn around 360° over 10 seconds (Front $\rightarrow$ Left Profile $\rightarrow$ Back $\rightarrow$ Right Profile $\rightarrow$ Front).
- **Expected Result:**
  - Target lock **never breaks** and never drops to `LOST`.
  - Appearance Gallery automatically enrolls diverse viewpoints (Gallery count increases to `4-8/25`).
  - Real-time similarity scores remain strong ($\ge 0.70$).

### Test 2.3: Re-Identification After Temporary Exit
- **Goal:** Verify rapid target re-acquisition when the person leaves and re-enters the scene.
- **Steps:**
  1. Have the locked target walk completely out of frame.
  2. Wait 5 seconds (Target state transitions to `LOST`).
  3. Have the target walk back into frame wearing the same clothes.
- **Expected Result:**
  - Within 1-2 frames of re-entry, the tracker re-identifies the target.
  - State transitions immediately from `LOST` $\rightarrow$ `LOCKED`.
  - Similarity score reads $\ge 0.68$.

---

## Test Suite 3: Crowd Occlusion & Distractor Resistance

### Test 3.1: Path Crossing (Distractor Test)
- **Goal:** Ensure target lock does not jump to another person crossing paths.
- **Steps:**
  1. Lock Target (Person A, e.g., wearing a black shirt).
  2. Have Person B (wearing a different color shirt) walk directly across Person A's path, fully occluding Person A for 1-2 seconds.
- **Expected Result:**
  - During the crossing, Person A's state briefly shows `OCCLUDED`.
  - Lock **never transfers** to Person B.
  - As Person A emerges, target lock remains firmly on Person A.

---

## Test Suite 4: Cross-Camera Recovery & Handoff

### Test 4.1: Multi-Camera Hand-Off
- **Goal:** Verify graph-driven multi-camera search and seamless handoff.
- **Steps:**
  1. Configure Camera 1 and Camera 2 as connected nodes.
  2. Lock target on Camera 1.
  3. Have target walk out of Camera 1's view and enter Camera 2's view.
- **Expected Result:**
  - When target leaves Camera 1, Camera 1 indicates `LOST` and Camera 2 activates search (status shows `SEARCHING R1`).
  - As target enters Camera 2, ReID confirms appearance match against the gallery ($\ge 0.65$).
  - Handoff triggers: Camera 2 becomes `ACTIVE FOCUS` with a green highlight.
  - Transit Trail logs: `TARGET_HANDOFF | camera_0 -> camera_1`.

---

## Test Suite 5: System Control & Graceful Termination

### Test 5.1: Clean Quit
- **Goal:** Verify safe release of camera handles and clean shutdown.
- **Steps:**
  1. Click the red **Quit** button in the UI header.
  2. Confirm the shutdown dialog.
- **Expected Result:**
  - Toast indicates `"Shutting down surveillance pipeline and releasing camera resources..."`.
  - Browser displays the clean shutdown confirmation page.
  - The Python backend terminates cleanly without leaving zombie processes or locking the webcam driver.
