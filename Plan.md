# Argus — Human-Assisted ReID Refactoring Plan

## Executive Summary

This plan transforms Argus from a fully autonomous ReID system into a **Human-Assisted ReID** system where an operator works in tandem with AI. The AI tracks and re-identifies autonomously, but a human monitor can correct mistakes instantly by clicking the correct target. The most significant architectural addition is **Recorded Video Mode**: the ability to run the entire pipeline against pre-recorded synchronized footage, generate a target's route, and produce a stitched journey video.

This plan is organized into **13 phases**, ordered by dependency and priority. Phases 0–6 cover the core functional transformation. Phases 7–12 add forensic-grade reliability, operator ergonomics, and evidence-chain integrity required for the forensic evidence use case.

---

## Table of Contents

1. [Phase 0 — Topology Reliability (Critical Fix)](#phase-0--topology-reliability-critical-fix)
2. [Phase 1 — Camera Detection & Stream Stability](#phase-1--camera-detection--stream-stability)
3. [Phase 2 — Gallery Persistence & Human-Assisted Target Correction](#phase-2--gallery-persistence--human-assisted-target-correction)
4. [Phase 3 — Recorded Video Mode (Core Feature)](#phase-3--recorded-video-mode-core-feature)
5. [Phase 4 — Camera Handoff Hardening](#phase-4--camera-handoff-hardening)
6. [Phase 5 — Scalable Camera Management UI](#phase-5--scalable-camera-management-ui)
7. [Phase 6 — UI/UX Overhaul & Advanced Settings](#phase-6--uiux-overhaul--advanced-settings)
8. [Phase 7 — Forensic Audit Trail / Chain of Custody](#phase-7--forensic-audit-trail--chain-of-custody)
9. [Phase 8 — Undo/Redo for Operator Corrections](#phase-8--undoredo-for-operator-corrections)
10. [Phase 9 — Operator Annotations](#phase-9--operator-annotations)
11. [Phase 10 — Session / Case Management](#phase-10--session--case-management)
12. [Phase 11 — Evidence Export Package](#phase-11--evidence-export-package)
13. [Phase 12 — Batch Processing for Recorded Mode](#phase-12--batch-processing-for-recorded-mode)

---

## Phase 0 — Topology Reliability (Critical Fix)

> **Priority: HIGHEST.** The topology map fails to load ~90% of the time in real-world tests. Nothing else matters if the graph cannot be reliably saved and loaded.

### Root Cause Analysis

Based on the audit, topology is managed by:
- `CameraGraph.from_dict()` / `CameraGraph.save()` in `src/multi_camera/camera_graph.py`
- `MappingAPIHandler` in `src/multi_camera/ui_server.py` reading/writing `configs/camera_graph.json`
- The SPA frontend (`src/multi_camera/static/`) making `GET /api/graph` and `POST /api/graph` calls

Likely failure points:
1. **JSON schema drift** — The frontend sends a shape that `CameraGraph.from_dict()` doesn't expect, or vice versa. No schema version enforcement exists.
2. **Silent parse errors** — `from_dict()` may swallow malformed edge/node data without surfacing the error to the UI.
3. **File I/O race** — The UI server reads/writes the JSON file from HTTP handler threads with no file-level locking. Concurrent `GET` during a `POST` could read a partial write.
4. **Empty initial state** — When `camera_graph.json` doesn't exist or is empty, the frontend may fail to initialize its internal state correctly.
5. **Validation gaps** — `graph.validate()` may pass graphs that later crash during pipeline startup (e.g., edges referencing nonexistent camera IDs).

### Plan

#### 0.1 — Define a strict, versioned JSON schema

Create a formal schema definition for `camera_graph.json`. Every save and load must validate against this schema. Include a `schema_version` integer field at the top level so that future changes can be migrated.

```
{
  "schema_version": 2,
  "cameras": [ ... ],
  "edges": [ ... ],
  "background_map": null | { ... },
  "metadata": {
    "created_at": "ISO8601",
    "last_modified": "ISO8601"
  }
}
```

Every camera node must have at minimum: `camera_id` (unique string), `name`, `source`, `source_type`, `enabled`. Every edge must reference two existing `camera_id` values. Validate this structurally on both save AND load.

#### 0.2 — Atomic file writes with backup

Replace the current raw `json.dump()` write with an atomic write pattern:
1. Write to `camera_graph.json.tmp`
2. Validate the written file by reading it back and parsing it
3. Rename `camera_graph.json` → `camera_graph.json.bak`
4. Rename `camera_graph.json.tmp` → `camera_graph.json`

This guarantees that `camera_graph.json` is never in a half-written state. If the process crashes mid-write, the `.bak` file preserves the last known good state.

#### 0.3 — Automatic recovery on corrupt load

When `CameraGraph.load()` fails:
1. Log the error with full detail
2. Attempt to load `camera_graph.json.bak`
3. If backup also fails, start with an empty graph and surface a clear warning to the UI
4. Never crash the application due to a corrupt topology file

#### 0.4 — Strict bidirectional validation

Add validation that catches:
- Duplicate `camera_id` values
- Edges referencing nonexistent camera IDs
- Self-loop edges
- Missing required fields
- Source type values not in the allowed enum

Return all validation errors as a list to the UI so the operator sees exactly what's wrong.

#### 0.5 — File locking for concurrent access

Add a `threading.Lock` around all reads and writes to `camera_graph.json` within the UI server to prevent partial reads during concurrent HTTP requests.

### Files Affected

| File | Change |
|------|--------|
| `src/multi_camera/camera_graph.py` | Atomic write, schema validation, backup/recovery |
| `src/multi_camera/ui_server.py` | File lock around graph reads/writes |
| `src/core/multi_camera_types.py` | Add `schema_version` to config types |
| `src/multi_camera/static/` (frontend JS) | Display validation errors, handle empty/error states gracefully |

### Success Criteria

- Topology saves and loads correctly 100% of the time across process restarts
- Corrupt files are recovered automatically from backup
- All validation errors are surfaced to the UI operator
- No silent failures

---

## Phase 1 — Camera Detection & Stream Stability

> **Priority: HIGH.** Cameras turn on/off multiple times before displaying footage, sometimes never rendering properly.

### Root Cause Analysis

Based on the audit:
- `probe_local_webcams()` in `ui_server.py` uses `ThreadPoolExecutor` with per-index timeout of 1.5s
- On Windows, it probes via DirectShow (`cv2.CAP_DSHOW`) which requires exclusive device access
- The pipeline is paused during probing (`pipeline.pause_processing()`), but the 150ms sleep may not be enough for Windows to release DirectShow handles
- `OpenCVCamera._capture_loop` has aggressive retry logic (30 retries before reconnection), which combined with the probe's device-stealing behavior causes the on/off flickering
- MJPEG stream staleness detection uses `len(frame_bytes)` as a hash — two different frames with the same JPEG byte count would be falsely considered "stale"

### Plan

#### 1.1 — Decouple camera probing from live pipeline

Camera discovery should never touch cameras that are already managed by the pipeline. Change `probe_local_webcams()` to:
1. Accept a set of "already-in-use source indices" from the pipeline
2. Skip those indices entirely during probing
3. Remove the pause/resume pipeline dance — it's the primary cause of flickering

If the operator wants to re-probe an already-active camera, they should explicitly stop that camera first from the UI.

#### 1.2 — Sequential probing with proper cleanup

Replace the `ThreadPoolExecutor` parallel probe with sequential probing:
- Probe one camera at a time
- After each probe, explicitly `cap.release()` and add a 200ms delay on Windows to let DirectShow fully release the handle
- Show probe progress in the UI (e.g., "Probing camera 2 of 4...")
- Add a "Cancel probe" button so the operator can abort a long probe

#### 1.3 — Fix MJPEG stream staleness detection

Replace the `len(frame_bytes)` hash with a proper hash or a monotonic frame counter:
- Have each camera worker increment a frame sequence number on every new frame
- The MJPEG stream compares sequence numbers, not byte lengths
- If the sequence number hasn't changed in 2 seconds, show a "STALE — RECONNECTING" overlay instead of silently evicting the cache

#### 1.4 — Camera health indicator in UI

Add a per-camera health badge visible in the live matrix:
- 🟢 LIVE — receiving new frames
- 🟡 CONNECTING — cap opened but no frames yet
- 🔴 OFFLINE — cap failed to open or no frames for >5 seconds
- ⏸️ PAUSED — camera is in standby (not in active search set)

### Files Affected

| File | Change |
|------|--------|
| `src/multi_camera/ui_server.py` | Rewrite probe logic, remove pause/resume, sequential probe |
| `src/camera/capture.py` | Add frame sequence counter, improve reconnection backoff |
| `src/multi_camera/static/` (frontend) | Probe progress UI, health badges, cancel button |

### Success Criteria

- Camera probe never causes flickering on already-active cameras
- Streams render within 2 seconds of camera activation
- Stale streams are visually indicated, never silently frozen

---

## Phase 2 — Gallery Persistence & Human-Assisted Target Correction

> **Priority: HIGH.** Core paradigm shift from autonomous to human-assisted.

### Current Behavior (from audit)

- When a new target is selected, the gallery is cleared and rebuilt from scratch
- Gallery is split into `trusted_gallery` (manual/anchor) and `provisional_gallery` (auto)
- Manual sample capture works via Right-Click or 'A' key, calling `IdentityManager.add_reference_sample(force=True)`
- The `TargetManager` state machine governs transitions: UNSELECTED → LOCKED → TRACKING → LOST → SEARCHING, etc.

### Desired Behavior

1. **Gallery never auto-clears.** When the human clicks a different person (to correct a ReID mistake), the system re-locks to that person WITHOUT clearing the existing gallery. The new target's crops are added to the same gallery.
2. **Manual gallery management.** The operator can:
   - View all gallery thumbnails in the UI
   - Delete individual entries (e.g., remove a crop that was from the wrong person)
   - Clear the entire gallery explicitly
3. **Target correction flow.** When the operator clicks a different track ID on the active camera:
   - The system does NOT treat this as "select new target" (which clears gallery)
   - Instead, it treats this as "correct the current target lock" — the new track becomes the active track, the old track's recent crops may optionally be flagged for review, and the gallery is preserved
4. **Distinction between "New Investigation" and "Correct Target".** Add two explicit actions:
   - **"New Investigation"** — clears gallery, resets all state, starts fresh (used when switching to a completely different person of interest)
   - **"Correct Target" (click on person)** — re-locks to clicked person, preserves gallery (used when ReID made a mistake)

### Plan

#### 2.1 — Separate "target correction" from "new target selection" in TargetManager

Add a new method `correct_target(track_id)` that:
- Updates the active track ID to the clicked person
- Does NOT clear the gallery
- Marks the transition as a "human correction" event in the route log
- Extracts the new person's crop and adds it to the gallery immediately

Keep the existing `select_target(track_id)` but rename it conceptually to `start_new_investigation(track_id)` which does the full clear-and-restart.

#### 2.2 — Gallery persistence across sessions

The gallery should survive application restarts:
- On every gallery add/remove, persist the gallery state to SQLite (the `SQLiteVectorStore` already exists)
- On startup, if a saved gallery exists, offer to "Resume previous investigation" or "Start new"
- Store gallery entries with metadata: `source` (manual/auto), `camera_id`, `timestamp`, `quality_score`

#### 2.3 — Gallery management UI endpoints

Add REST endpoints:
- `GET /api/target/gallery` — already exists, returns thumbnails
- `DELETE /api/target/gallery/entry/{entry_id}` — delete one entry
- `POST /api/target/gallery/clear` — clear entire gallery (explicit "New Investigation")
- `POST /api/target/correct` — correct target to a different track (preserves gallery)

#### 2.4 — UI for gallery management

In the web dashboard, add a gallery panel:
- Grid of thumbnail images with source badges (🔵 manual, 🟢 auto)
- Click-to-delete on individual thumbnails (with confirmation)
- "Clear All" button (with confirmation dialog: "This will start a new investigation")
- Gallery count indicator (e.g., "12/25 samples")

### Files Affected

| File | Change |
|------|--------|
| `src/target/manager.py` | Add `correct_target()`, separate from `select_target()` |
| `src/identity/manager.py` | Remove auto-clear on target switch, add persistence hooks |
| `src/identity/sqlite_store.py` | Gallery persistence methods |
| `src/multi_camera/ui_server.py` | New endpoints for correction and gallery management |
| `src/multi_camera/static/` (frontend) | Gallery panel, correction vs. new investigation UX |

### Success Criteria

- Clicking a different person preserves the gallery and re-locks tracking
- Gallery thumbnails are visible and individually deletable in the UI
- Gallery survives application restarts
- "New Investigation" is an explicit, deliberate action

---

## Phase 3 — Recorded Video Mode (Core Feature)

> **Priority: HIGHEST (feature).** The most important new capability.

### Concept

The operator loads pre-recorded footage from all cameras, selects a target in one camera's video, and the system tracks that target across all cameras using the same AI pipeline — but against video files instead of live streams. The operator monitors and corrects as needed. When satisfied, they press "Complete Route" and the system generates:
1. A **route map** showing the target's path through the camera topology
2. A **journey video** — clips from each camera where the target was visible, stitched chronologically into a single continuous video

### Architecture

#### 3.1 — Video folder convention

Define a standard folder structure:

```
footage/
├── session_metadata.json       (optional: date, case ID, notes)
├── cam_entrance/
│   └── video.mp4              (or .avi, .mkv)
├── cam_lobby/
│   └── video.mp4
├── cam_hallway_1/
│   └── video.mp4
└── cam_parking/
    └── video.mp4
```

Each subfolder name matches a `camera_id` from the loaded topology. The system validates that every subfolder maps to a known camera in the graph. Videos within each folder are assumed to start at the same wall-clock time (time-synchronized).

`session_metadata.json` (optional):
```json
{
  "case_id": "CASE-2026-0042",
  "recording_date": "2026-09-15",
  "notes": "Suspect entered via main entrance at ~14:32",
  "start_time_utc": "2026-09-15T14:00:00Z"
}
```

#### 3.2 — VideoFileCamera: new BaseCamera implementation

Create `VideoFileCamera` in `src/camera/capture.py` (or a new `src/camera/video_file.py`):

- Takes a file path instead of a device index or RTSP URL
- Reads frames sequentially using `cv2.VideoCapture(file_path)`
- Exposes frame position as a timestamp relative to video start (using `CAP_PROP_POS_MSEC`)
- Supports **playback controls**: play, pause, seek to timestamp, playback speed (0.5x, 1x, 2x, 4x)
- Does NOT use a background polling thread (unlike live cameras) — instead, reads frames on-demand at the requested playback speed, advancing all cameras synchronously
- Implements the same `BaseCamera` interface so the rest of the pipeline doesn't care whether it's live or recorded

#### 3.3 — Synchronized multi-video playback controller

Create `src/playback/controller.py`:

This is the orchestrator for recorded mode. It manages:
- A clock that advances at the chosen playback speed
- Synchronization: all `VideoFileCamera` instances advance to the same timestamp on each tick
- Pause/resume: freezes the clock
- Seek: jumps all cameras to a given timestamp
- End detection: signals when all videos have reached their end

The playback controller replaces the "real-time" loop of the live pipeline. Instead of cameras pushing frames as fast as hardware allows, the controller pulls frames at the playback rate.

Key design decision: **The pipeline's processing loop must be abstracted** so that it can be driven either by:
- Live camera push (current behavior) — frames arrive as fast as the camera produces them
- Playback controller pull (new behavior) — frames are requested at a controlled rate

This means the `MultiCameraPipeline` (or a new `RecordedPipeline`) needs a pluggable "frame source strategy."

#### 3.4 — Route recorder

Create `src/playback/route_recorder.py`:

Every time the target is confirmed on a camera, log:
```python
@dataclass
class RouteEvent:
    camera_id: str
    track_id: int
    timestamp_ms: float          # video timestamp
    event_type: str              # "enter" | "exit" | "correction" | "lost" | "handoff"
    bbox: Tuple[int,int,int,int] # bounding box at this moment
    confidence: float            # ReID confidence at handoff
    was_human_corrected: bool    # True if human clicked to correct
```

The route is a chronological list of `RouteEvent` entries. It records:
- When the target entered each camera's view
- When they left (last frame where they were tracked)
- Camera handoffs (which camera they moved to, with ReID confidence)
- Human corrections (when the operator clicked to fix a mismatch)

#### 3.5 — Journey video generator

Create `src/playback/journey_stitcher.py`:

Given a completed route and the source video files:

1. For each camera segment in the route (ordered chronologically):
   a. Open the source video for that camera
   b. Seek to the "enter" timestamp
   c. Extract frames from "enter" to "exit"
   d. Optionally crop/zoom to the target's bounding box region (with padding) OR keep full frame with a highlight box around the target
   e. Overlay metadata: camera name, timestamp, track ID
2. Concatenate all segments chronologically
3. Handle gaps: if there's a period where the target is between cameras (not visible anywhere), insert a brief "IN TRANSIT" card showing which cameras the target is moving between and the elapsed time
4. Output as a single MP4 file using OpenCV's `VideoWriter` or `ffmpeg`

Cross-camera ReID error handling during stitching:
- Use the corrected route log (which includes human corrections) — the stitcher only uses segments that the operator confirmed
- If a segment was human-corrected, mark it in the overlay (e.g., "⚠ Operator Corrected")

#### 3.6 — Recorded mode UI

Add a "Mode" selector in the UI:
- **Live Mode** — current behavior (connect to cameras, real-time tracking)
- **Recorded Mode** — load footage folder, playback controls, route generation

Recorded mode UI additions:
- **Folder selector** — browse/enter path to footage folder
- **Timeline scrubber** — horizontal bar showing video progress with markers for route events
- **Playback controls** — play/pause, speed selector (0.5x/1x/2x/4x), seek
- **Camera grid** — same as live mode, but showing video frames instead of live streams
- **Target selection** — click on a person in any camera to start tracking
- **"Complete Route" button** — finalizes the route, generates the journey video
- **Route summary panel** — shows the chronological list of camera transitions with timestamps
- **Download links** — download the generated journey video and route report

#### 3.7 — Route report export

Generate a downloadable report alongside the journey video:

```
Route Report — Case CASE-2026-0042
Generated: 2026-09-18 23:45:00

Target: First identified on cam_entrance at 14:32:15

Timeline:
  14:32:15 - 14:33:42  cam_entrance     (1m 27s)  AUTO
  14:33:42 - 14:33:50  IN TRANSIT        (8s)
  14:33:50 - 14:35:12  cam_lobby         (1m 22s)  AUTO
  14:35:12 - 14:35:20  IN TRANSIT        (8s)
  14:35:20 - 14:36:45  cam_hallway_1     (1m 25s)  CORRECTED ⚠
  14:36:45 - 14:38:00  cam_parking       (1m 15s)  AUTO

Total tracked time: 5m 45s
Cameras visited: 4
Human corrections: 1
Journey video: journey_CASE-2026-0042.mp4
```

### Files Affected

| File | Change |
|------|--------|
| `src/camera/video_file.py` | **[NEW]** `VideoFileCamera` implementing `BaseCamera` |
| `src/playback/__init__.py` | **[NEW]** Playback module |
| `src/playback/controller.py` | **[NEW]** Synchronized multi-video playback controller |
| `src/playback/route_recorder.py` | **[NEW]** Route event logging |
| `src/playback/journey_stitcher.py` | **[NEW]** Video stitching and export |
| `src/playback/report.py` | **[NEW]** Route report generation |
| `src/core/config.py` | Add `PlaybackConfig` dataclass |
| `src/pipeline/` or equivalent | Abstract frame source (live vs recorded) |
| `src/multi_camera/ui_server.py` | Playback endpoints, folder loading, route export |
| `src/multi_camera/static/` (frontend) | Mode selector, timeline, playback controls, route panel |

### Success Criteria

- Operator can load a folder of synchronized videos and select a target
- System tracks the target across cameras using the same AI pipeline
- Human can correct mismatches by clicking
- "Complete Route" generates a chronological journey video
- Route report documents every camera transition including human corrections
- All footage segments stitch seamlessly with no gaps or overlaps

---

## Phase 4 — Camera Handoff Hardening

> **Priority: MEDIUM.** Handoff works in principle, but must be robust for both live and recorded modes.

### Current State (from audit)

- `SearchManager` expands radius from 1-hop to `max_radius` (3) with `per_radius_timeout_s: 5.0`
- `TransitionValidator.is_plausible()` scores temporal plausibility per edge
- `confirmation_frames: 3` required before accepting a handoff
- `handoff_confirm_delay_s: 2.0`
- Total recovery timeout: 30s

### Plan

#### 4.1 — Deterministic handoff for recorded mode

In recorded mode, the system has a unique advantage: it can look ahead. When the target is lost on camera A:
1. The system knows the exact timestamp of loss
2. It can immediately check neighboring cameras at that exact timestamp (no need to wait for real-time timeout)
3. If a matching person appears within the edge's `expected_min_transition_s` to `expected_max_transition_s` window, propose the handoff immediately

This means the `per_radius_timeout_s` can be much shorter in recorded mode (or eliminated entirely in favor of frame-based scanning).

#### 4.2 — Handoff confidence feedback

When a handoff candidate is found, show the operator:
- Side-by-side comparison: last crop from camera A vs. candidate crop from camera B
- ReID confidence score
- Temporal plausibility score
- A brief "Confirm / Reject" prompt (with auto-accept after a configurable delay if confidence is high enough)

For high-confidence matches (`> auto_add_threshold`), auto-accept after 2 seconds. For uncertain matches (`uncertain_band_min` to `uncertain_band_max`), pause and wait for human confirmation.

#### 4.3 — Handoff failure recovery

If the target is not found within `total_recovery_timeout_s`:
- In live mode: show a "TARGET LOST" alert and keep searching at reduced intensity
- In recorded mode: pause playback and ask the operator to manually locate the target in any camera, or skip ahead

### Files Affected

| File | Change |
|------|--------|
| `src/multi_camera/search_manager.py` | Recorded-mode fast search, confidence-based auto/manual accept |
| `src/target/manager.py` | Handoff confirmation states |
| `src/multi_camera/ui_server.py` | Handoff confirmation endpoints |
| `src/multi_camera/static/` (frontend) | Side-by-side comparison UI, confirm/reject buttons |

### Success Criteria

- Handoffs in recorded mode happen within 1-2 seconds (no real-time waiting)
- Uncertain handoffs pause for human confirmation
- High-confidence handoffs auto-accept with visual feedback
- Lost targets produce a clear UI prompt instead of silently failing

---

## Phase 5 — Scalable Camera Management UI

> **Priority: MEDIUM.** Current topology editing is congested for 50+ cameras.

### Plan

#### 5.1 — Batch camera detection

Instead of probing one device at a time:
- Add "Scan Range" controls: start index, end index, max concurrent probes
- Show results in a paginated table, not a flat list
- Allow the operator to select which detected cameras to add (checkbox multi-select)
- For RTSP cameras: add a "Bulk Import" that accepts a CSV or text list of RTSP URLs with labels

#### 5.2 — Camera grouping by floor/zone

Extend the `CameraNodeConfig` to support:
- `floor: str` (e.g., "Ground Floor", "Floor 2")
- `zone: str` (e.g., "East Wing", "Parking")
- `group: str` (optional, for arbitrary grouping)

The topology map should allow filtering by floor/zone/group so that the operator only sees a subset at a time.

#### 5.3 — Search and filter in topology editor

Add a search bar in the topology editor that filters cameras by:
- Name (partial match)
- Camera ID
- Floor/zone
- Source type (webcam/RTSP/file)
- Status (online/offline/standby)

#### 5.4 — Camera list view (alternative to map)

For large deployments, the spatial map becomes unmanageable. Add an alternative **list/table view**:
- Sortable columns: Name, ID, Floor, Zone, Source, Status, FPS
- Quick-action buttons: Enable/Disable, Preview, Edit, Delete
- Bulk operations: select multiple cameras → enable/disable/delete

The map view remains available for spatial layout, but the list view is the default for >10 cameras.

#### 5.5 — Camera preview during topology editing

When adding a camera to the topology, show a live preview thumbnail:
- Small 320×180 snapshot from the camera
- Updates every 2 seconds
- Helps the operator identify which physical camera they're adding without memorizing device indices

### Files Affected

| File | Change |
|------|--------|
| `src/multi_camera/ui_server.py` | Batch probe endpoint, bulk import, filter/search API |
| `src/core/multi_camera_types.py` | Add floor/zone/group to `CameraNodeConfig` |
| `src/multi_camera/camera_graph.py` | Support for grouping and filtering queries |
| `src/multi_camera/static/` (frontend) | List view, search bar, batch selection, preview thumbnails |

### Success Criteria

- 50+ cameras are manageable without scrolling through a congested map
- Cameras can be filtered, searched, and bulk-managed
- RTSP cameras can be imported in bulk
- Camera preview is available during topology editing

---

## Phase 6 — UI/UX Overhaul & Advanced Settings

> **Priority: MEDIUM.** Make the dashboard operator-friendly with clear separation of common and advanced controls.

### Plan

#### 6.1 — Two-tier settings architecture

Split settings into two panels:

**Main Panel (always visible):**
- Mode selector (Live / Recorded)
- Active camera selector
- Target status indicator (LOCKED / LOST / SEARCHING / UNSELECTED)
- Gallery thumbnail strip
- Playback controls (recorded mode only)
- "New Investigation" / "Complete Route" buttons

**Advanced Settings (collapsible panel or modal, opened via ⚙️ icon):**

| Setting | Config Key | Current Default | Description |
|---------|-----------|----------------|-------------|
| Max Search Radius | `search.max_radius` | 3 | How many hops to expand when searching |
| Per-Radius Timeout | `search.per_radius_timeout_s` | 5.0s | How long to search at each radius before expanding |
| Total Recovery Timeout | `search.total_recovery_timeout_s` | 30.0s | Maximum time to search before declaring target lost |
| Confirmation Frames | `search.confirmation_frames` | 3 | Frames needed to confirm a handoff |
| ReID Match Threshold | `reid.match_threshold` | 0.78 | Minimum similarity to consider a match |
| Auto-Add Threshold | `reid.auto_add_threshold` | 0.82 | Minimum similarity to auto-add to gallery |
| Max Gallery Size | `reid.max_gallery_size` | 25 | Maximum number of gallery entries |
| Detection Confidence | `detection.confidence_threshold` | 0.4 | YOLO detection confidence threshold |
| Min Crop Sharpness | `reid.min_sharpness` | 60.0 | Laplacian variance threshold for blur rejection |
| Device | `inference.device` | auto | GPU/CPU selection |
| Half Precision | `inference.half_precision` | false | FP16 inference |

Changes to advanced settings should:
- Take effect immediately (no restart required where possible)
- Be persisted to `configs/default.yaml` on explicit "Save Settings"
- Show current vs. default values so the operator knows what they changed
- Include a "Reset to Defaults" button

#### 6.2 — Operator status bar

A persistent status bar at the bottom of the dashboard showing:
- FPS (detection pipeline throughput)
- GPU memory usage
- Active cameras count
- Target state
- Gallery size
- Current mode (Live/Recorded)
- Uptime

#### 6.3 — Event log panel

A scrollable log panel showing recent system events:
- Target locked/lost/found
- Camera handoffs (with confidence)
- Human corrections
- Errors and warnings

Each event is timestamped and color-coded by severity. The log is filterable by event type.

#### 6.4 — Keyboard shortcuts

Document and display keyboard shortcuts for common operations:

| Key | Action |
|-----|--------|
| Space | Pause/resume (recorded mode) |
| A | Add manual gallery sample |
| C | Correct target (then click person) |
| N | New investigation |
| R | Complete route (recorded mode) |
| F | Toggle fullscreen on active camera |
| 1-9 | Switch to camera 1-9 |
| +/- | Zoom in/out on topology map |
| ? | Show keyboard shortcut overlay |

#### 6.5 — Decouple UI state from pipeline internals

Based on the audit, `ui_server.py` directly accesses `pipeline._nodes`, `pipeline._frame_lock`, `pipeline._latest_jpegs`, and uses `getattr(pipeline, 'target_state')`. This must be cleaned up:

Create a `PipelineTelemetry` dataclass that the pipeline publishes at a fixed interval (10Hz):
```python
@dataclass
class PipelineTelemetry:
    active_camera_id: Optional[str]
    target_state: str
    target_track_id: Optional[int]
    camera_statuses: Dict[str, str]
    search_progress: Optional[dict]
    gallery_size: int
    gallery_max: int
    fps: float
    gpu_memory_mb: float
    candidate_scores: Dict[str, float]
```

The UI server reads only from this telemetry object, never from pipeline internals. This decouples the UI completely from the pipeline's internal data structures.

### Files Affected

| File | Change |
|------|--------|
| `src/multi_camera/ui_server.py` | Telemetry-based state access, settings endpoints |
| `src/core/config.py` | Runtime settings update support |
| `src/core/types.py` | `PipelineTelemetry` dataclass |
| `src/multi_camera/static/` (frontend) | Settings panel, status bar, event log, keyboard shortcuts |
| `src/app/main.py` | Telemetry publishing, remove direct UI state logic |

### Success Criteria

- Common operations are one-click accessible
- Advanced settings are discoverable but not overwhelming
- Pipeline internals are never accessed directly from UI code
- Status bar gives immediate system health awareness

---

## Phase 7 — Forensic Audit Trail / Chain of Custody

> **Priority: HIGH.** If this system's output is used as forensic evidence, the defense will challenge every AI decision and every operator action. An immutable, verifiable audit trail is non-negotiable.

### What Needs to Change

Currently, the system has no persistent record of what happened during an investigation. If the operator corrects a ReID mismatch, there is no log of when that happened, what the AI thought, or why the operator overrode it. The `TargetManager` state transitions, `IdentityManager` gallery modifications, and `SearchManager` camera activations all happen in-memory with only debug-level Python logging to `stdout`.

### Plan

#### 7.1 — Create `src/audit/logger.py`: Append-Only Audit Log

Create a new module `src/audit/` with a single-responsibility `AuditLogger` class. This logger writes to an append-only SQLite table — every row is a chronological event that cannot be modified or deleted after insertion.

**SQLite schema** (new table in the case database, or a dedicated `audit.db`):

```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,           -- ISO 8601, always UTC
    event_type TEXT NOT NULL,              -- e.g. "TARGET_LOCKED", "REID_MATCH", "HUMAN_CORRECTION"
    camera_id TEXT,                        -- which camera this happened on
    track_id INTEGER,                      -- track ID involved (if any)
    detail_json TEXT NOT NULL,             -- structured JSON payload with event-specific data
    prev_hash TEXT NOT NULL,               -- SHA-256 hash of the previous row (chain)
    row_hash TEXT NOT NULL                 -- SHA-256(timestamp + event_type + detail_json + prev_hash)
);
```

The `prev_hash` / `row_hash` columns form a **hash chain**: each row's hash depends on the previous row's hash. If any row is retroactively modified or deleted, the chain breaks and the tampering is detectable. The first row uses a known seed (e.g., `SHA-256("AUDIT_LOG_GENESIS")`).

#### 7.2 — Define the event taxonomy

Every auditable event gets a specific `event_type` and a structured `detail_json`. Here is the complete list:

| event_type | When it fires | detail_json contents |
|------------|--------------|---------------------|
| `INVESTIGATION_START` | Operator starts a new investigation | `{mode: "live"|"recorded", topology_file, config_snapshot}` |
| `INVESTIGATION_END` | Operator completes or closes | `{reason: "completed"|"abandoned", duration_s}` |
| `TARGET_SELECTED` | Operator clicks a person to start tracking | `{camera_id, track_id, bbox, frame_timestamp_ms}` |
| `TARGET_LOCKED` | TargetManager transitions to LOCKED | `{camera_id, track_id, reid_confidence}` |
| `TARGET_LOST` | TargetManager transitions to LOST | `{camera_id, last_track_id, last_seen_timestamp_ms}` |
| `SEARCH_START` | SearchManager begins expanding radius | `{from_camera_id, initial_radius, search_cameras}` |
| `SEARCH_EXPAND` | SearchManager expands to next radius | `{new_radius, added_cameras}` |
| `REID_MATCH` | IdentityManager finds a candidate match | `{camera_id, track_id, similarity_score, threshold, decision: "accept"|"reject"|"uncertain"}` |
| `HANDOFF_PROPOSED` | System proposes a cross-camera handoff | `{from_camera, to_camera, candidate_track_id, confidence, plausibility_score}` |
| `HANDOFF_ACCEPTED` | Handoff confirmed (auto or human) | `{from_camera, to_camera, track_id, method: "auto"|"human"}` |
| `HANDOFF_REJECTED` | Handoff rejected (auto or human) | `{from_camera, to_camera, track_id, method: "auto"|"human", reason}` |
| `HUMAN_CORRECTION` | Operator clicks to correct a mismatch | `{camera_id, old_track_id, new_track_id, gallery_preserved: true}` |
| `GALLERY_ADD` | New crop added to gallery | `{source: "manual"|"auto", camera_id, quality_score, gallery_size_after}` |
| `GALLERY_REMOVE` | Operator removes a gallery entry | `{entry_id, source: "manual"|"auto", gallery_size_after}` |
| `GALLERY_CLEAR` | Operator clears entire gallery | `{cleared_count, reason: "new_investigation"|"manual_clear"}` |
| `CONFIG_CHANGE` | Operator changes an advanced setting | `{setting_key, old_value, new_value}` |
| `ANNOTATION_ADD` | Operator adds a text annotation | `{camera_id, timestamp_ms, text}` |
| `UNDO` | Operator undoes a previous action | `{undone_event_id, undone_event_type}` |
| `REDO` | Operator redoes a previously undone action | `{redone_event_id, redone_event_type}` |

#### 7.3 — Integrate audit logging into existing modules

This requires inserting `audit_logger.log(event_type, detail)` calls at specific points in existing code. The key locations:

- **`src/target/manager.py`** — `select_target()`, `correct_target()` (new from Phase 2), every state transition (`mark_locked`, `mark_lost`, `mark_searching`, etc.)
- **`src/identity/manager.py`** — `add_reference_sample()`, `remove_entry()`, `clear()`, `evaluate_candidate_crop()` (log the match decision)
- **`src/multi_camera/search_manager.py`** — `start_search()`, `expand_radius()`
- **`src/multi_camera/ui_server.py`** — `POST /api/target/select`, `POST /api/target/correct`, `POST /api/target/gallery/delete`, `POST /api/target/gallery/clear`

The audit logger must be injected as a dependency (not imported globally) so it can be mocked in tests.

#### 7.4 — Audit log viewer in UI

Add a `GET /api/audit/log` endpoint that returns the last N audit events (paginated). The frontend displays these in the event log panel from Phase 6.3. Each event should show:
- Timestamp (relative to investigation start)
- Icon by event type (🎯 target, 🔄 handoff, ✋ human correction, 📷 gallery, ⚙️ config)
- One-line summary
- Expandable detail JSON

#### 7.5 — Hash chain integrity verification

Add a `verify_chain()` method to `AuditLogger` that reads all rows and verifies each `row_hash` matches `SHA-256(timestamp + event_type + detail_json + prev_hash)`. This can be run:
- On investigation load (to detect tampered logs)
- On evidence export (to certify the log)
- Via a CLI command for manual verification

### Files Affected

| File | Change |
|------|--------|
| `src/audit/__init__.py` | **[NEW]** Module init |
| `src/audit/logger.py` | **[NEW]** `AuditLogger` with hash-chained SQLite, event taxonomy |
| `src/target/manager.py` | Add audit logging at state transitions |
| `src/identity/manager.py` | Add audit logging at gallery mutations and match decisions |
| `src/multi_camera/search_manager.py` | Add audit logging at search start/expand |
| `src/multi_camera/ui_server.py` | Audit log viewer endpoint, log on POST actions |

### Success Criteria

- Every operator action and AI decision is recorded with structured data
- Hash chain detects any retroactive tampering
- Audit log is viewable in the UI
- Log survives application restarts (persisted to SQLite)
- Audit logger can be mocked for unit tests

---

## Phase 8 — Undo/Redo for Operator Corrections

> **Priority: MEDIUM.** The operator must be able to recover from mis-clicks without restarting the investigation.

### What Needs to Change

Currently, once the operator clicks to correct a target, there is no way to reverse that action. In a high-pressure monitoring scenario, mis-clicks are inevitable. The system needs a bounded undo stack.

### Plan

#### 8.1 — Create `src/core/undo_stack.py`: Action History

Implement a simple undo stack with the following mechanics:

```python
@dataclass
class UndoableAction:
    action_id: str                       # UUID
    action_type: str                     # "target_correction", "gallery_add", "gallery_remove", etc.
    timestamp: float                     # time.time()
    forward_data: dict                   # data needed to redo the action
    reverse_data: dict                   # data needed to undo the action (snapshot of state before)

class UndoStack:
    def __init__(self, max_depth: int = 10):
        self._undo: List[UndoableAction] = []   # completed actions
        self._redo: List[UndoableAction] = []   # undone actions available for redo
        self.max_depth = max_depth

    def push(self, action: UndoableAction) -> None:
        """Push a new action. Clears the redo stack (new action invalidates redo history)."""
        self._undo.append(action)
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)       # evict oldest
        self._redo.clear()

    def undo(self) -> Optional[UndoableAction]:
        """Pop and return the most recent action for reversal. Moves it to redo stack."""
        if not self._undo:
            return None
        action = self._undo.pop()
        self._redo.append(action)
        return action

    def redo(self) -> Optional[UndoableAction]:
        """Pop and return the most recently undone action for re-application."""
        if not self._redo:
            return None
        action = self._redo.pop()
        self._undo.append(action)
        return action
```

#### 8.2 — Define undoable actions and their reversal logic

Not every action is undoable. Specifically, these actions are undoable:

| Action | Forward (do) | Reverse (undo) |
|--------|-------------|----------------|
| **Target correction** | Set active track to new track ID | Restore previous track ID as active |
| **Gallery add (manual)** | Add crop entry to gallery | Remove that specific entry |
| **Gallery remove** | Remove entry from gallery | Re-insert the entry (requires snapshotting the crop + embedding before deletion) |
| **Gallery clear** | Remove all entries | Restore all entries from snapshot |

Actions that are **NOT undoable** (because they're either irreversible by nature or too dangerous to silently reverse):
- New Investigation (destructive, confirmation-gated)
- Config changes (take effect immediately, operator can manually revert)
- Handoff accept/reject (already committed to the route)

#### 8.3 — Integrate undo stack into TargetManager and IdentityManager

When `TargetManager.correct_target()` is called:
1. Snapshot `reverse_data = {old_track_id, old_camera_id}`
2. Execute the correction
3. Push `UndoableAction(action_type="target_correction", forward_data={new_track_id}, reverse_data=snapshot)`

When `IdentityManager.add_reference_sample()` is called:
1. Record the entry ID that will be created
2. Execute the add
3. Push `UndoableAction(action_type="gallery_add", forward_data={entry_id}, reverse_data={entry_data_snapshot})`

When an undo is requested:
1. Pop from undo stack
2. Dispatch based on `action_type`: call the appropriate reversal method
3. Log the undo to the audit trail (Phase 7)

#### 8.4 — UI controls and keyboard shortcuts

- **Ctrl+Z** → Undo last action
- **Ctrl+Shift+Z** → Redo
- Small undo/redo buttons in the toolbar
- Toast notification showing what was undone (e.g., "Undone: Target correction on cam_lobby")
- Undo button grays out when stack is empty

#### 8.5 — REST endpoints

- `POST /api/undo` — undo last action, returns the undone action summary
- `POST /api/redo` — redo last undone action
- `GET /api/undo/stack` — returns the current undo stack (for UI display)

### Files Affected

| File | Change |
|------|--------|
| `src/core/undo_stack.py` | **[NEW]** `UndoStack`, `UndoableAction` |
| `src/target/manager.py` | Push undo entries on target correction |
| `src/identity/manager.py` | Push undo entries on gallery add/remove/clear, snapshot before delete |
| `src/multi_camera/ui_server.py` | Undo/redo endpoints |
| `src/multi_camera/static/` (frontend) | Undo/redo buttons, keyboard shortcuts, toast notifications |

### Success Criteria

- Operator can undo the last 10 actions with Ctrl+Z
- Redo is available after undo
- Gallery entries are fully restorable after undo of a delete
- Undo actions are recorded in the audit trail
- Stack depth is bounded (no unbounded memory growth)

---

## Phase 9 — Operator Annotations

> **Priority: MEDIUM.** Allows the operator to attach context to specific moments in the investigation.

### What Needs to Change

Currently the operator has no way to record observations. In a forensic investigation, notes like "suspect drops bag at elevator" or "meets second individual" are critical context that must travel with the evidence.

### Plan

#### 9.1 — Annotation data model

Create `src/playback/annotations.py`:

```python
@dataclass
class Annotation:
    annotation_id: str                    # UUID
    timestamp_ms: float                   # video timestamp (or wall-clock for live mode)
    camera_id: str                        # which camera the annotation is anchored to
    text: str                             # operator's note
    created_at: str                       # ISO 8601 wall-clock time of creation
    bbox: Optional[Tuple[int,int,int,int]] # optional: bounding box region of interest
    annotation_type: str                  # "note" | "flag" | "bookmark"

class AnnotationStore:
    """Manages annotations for the current investigation. Persisted to SQLite."""

    def add(self, annotation: Annotation) -> None: ...
    def remove(self, annotation_id: str) -> bool: ...
    def get_all(self) -> List[Annotation]: ...
    def get_by_camera(self, camera_id: str) -> List[Annotation]: ...
    def get_in_range(self, start_ms: float, end_ms: float) -> List[Annotation]: ...
```

#### 9.2 — SQLite persistence

Add an `annotations` table to the case database:

```sql
CREATE TABLE annotations (
    annotation_id TEXT PRIMARY KEY,
    timestamp_ms REAL NOT NULL,
    camera_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    bbox_json TEXT,                        -- nullable, JSON "[x1, y1, x2, y2]"
    annotation_type TEXT NOT NULL DEFAULT 'note'
);
```

#### 9.3 — How the operator creates annotations

Two methods:

**Quick note (keyboard):** Press `N` while watching a camera → a text input overlay appears at the bottom of the screen → type the note → press Enter. The annotation is anchored to the current camera and current timestamp.

**Context-click note:** Right-click on a specific region in a camera frame → select "Add Note" from context menu → text input appears → the annotation is anchored to the camera, timestamp, AND the clicked bounding box region.

#### 9.4 — Annotation display

- **In recorded mode timeline:** Annotations appear as colored markers on the timeline scrubber. Hovering over a marker shows the note text. Clicking a marker seeks to that timestamp.
- **In camera view:** When playing through a timestamp that has an annotation, show a brief overlay banner at the bottom of the camera frame: `📝 "suspect drops bag at elevator"`. The banner fades after 3 seconds or until the next annotation.
- **In route report:** Annotations are included chronologically between route events:
  ```
  14:33:50 - 14:35:12  cam_lobby    (1m 22s)  AUTO
      📝 14:34:22  "Suspect interacts with reception desk"
  14:35:12 - 14:35:20  IN TRANSIT   (8s)
  ```
- **In journey video:** Annotations are burned into the video as text overlays at their respective timestamps.

#### 9.5 — REST endpoints

- `POST /api/annotations` — create annotation `{camera_id, timestamp_ms, text, bbox?, type?}`
- `GET /api/annotations` — list all, with optional `?camera_id=` and `?start_ms=&end_ms=` filters
- `DELETE /api/annotations/{annotation_id}` — remove annotation
- `PUT /api/annotations/{annotation_id}` — edit annotation text

### Files Affected

| File | Change |
|------|--------|
| `src/playback/annotations.py` | **[NEW]** `Annotation`, `AnnotationStore` |
| `src/playback/route_recorder.py` | Include annotations in route output |
| `src/playback/journey_stitcher.py` | Burn annotation text into journey video at timestamps |
| `src/playback/report.py` | Include annotations in text report |
| `src/multi_camera/ui_server.py` | Annotation CRUD endpoints |
| `src/multi_camera/static/` (frontend) | Annotation input overlay, timeline markers, camera overlays |

### Success Criteria

- Operator can add a text note in under 3 seconds
- Annotations appear on the timeline, in the route report, and in the journey video
- Annotations are persisted and survive restarts
- Annotations can be edited and deleted

---

## Phase 10 — Session / Case Management

> **Priority: MEDIUM.** Each forensic investigation must be a self-contained, portable unit.

### What Needs to Change

Currently, all data (gallery, SQLite store, topology) lives in fixed global paths (`data/identities.db`, `configs/camera_graph.json`). There is no concept of separate "cases." If the operator switches to a different investigation, they must manually clear the database and gallery. There is no way to return to a previous case.

### Plan

#### 10.1 — Case folder structure

Each investigation is a self-contained directory:

```
cases/
├── CASE-2026-0042/
│   ├── case.json                         # case metadata
│   ├── topology.json                     # frozen copy of the camera graph used
│   ├── config_snapshot.yaml              # frozen copy of default.yaml at case start
│   ├── gallery/
│   │   ├── entries.db                    # gallery SQLite (embeddings + thumbnails)
│   │   └── thumbnails/                   # JPEG crops, named by entry_id
│   │       ├── e1a2b3c4.jpg
│   │       └── ...
│   ├── audit.db                          # audit log SQLite (hash-chained)
│   ├── annotations.db                    # annotations SQLite
│   ├── route.json                        # completed route events
│   ├── report.txt                        # generated text report
│   └── journey_video.mp4                 # generated journey video (if completed)
│
├── CASE-2026-0043/
│   └── ...
```

`case.json`:
```json
{
  "case_id": "CASE-2026-0042",
  "created_at": "2026-09-15T14:00:00Z",
  "status": "active",                     // "active" | "completed" | "archived"
  "mode": "recorded",                     // "live" | "recorded"
  "footage_path": "D:/evidence/sept15/",  // only for recorded mode
  "operator": "Officer Singh",
  "notes": "Suspect entered via main entrance at ~14:32",
  "last_opened_at": "2026-09-18T23:45:00Z"
}
```

#### 10.2 — Create `src/case/manager.py`: Case Lifecycle

```python
class CaseManager:
    """Manages creation, loading, saving, and switching of investigation cases."""

    def __init__(self, cases_dir: str = "cases/"):
        self.cases_dir = Path(cases_dir)
        self.active_case: Optional[Case] = None

    def create_case(self, case_id: str, mode: str, operator: str = "",
                    notes: str = "", footage_path: str = None) -> Case:
        """Create a new case folder with all sub-files initialized."""
        ...

    def open_case(self, case_id: str) -> Case:
        """Load an existing case. Restores gallery, audit log, annotations."""
        ...

    def close_case(self) -> None:
        """Close the active case, flush all buffers to disk."""
        ...

    def list_cases(self) -> List[CaseSummary]:
        """List all cases with ID, status, date, mode."""
        ...

    def delete_case(self, case_id: str) -> None:
        """Permanently delete a case folder (requires confirmation)."""
        ...

    def export_case(self, case_id: str, output_path: str) -> str:
        """Package a case as a ZIP for transfer (see Phase 11)."""
        ...
```

#### 10.3 — What happens when the operator opens a case

1. Load `case.json` and validate
2. Load `topology.json` into `CameraGraph` — this is the frozen topology for this case, NOT `configs/camera_graph.json`
3. Initialize `IdentityManager` with `gallery/entries.db`
4. Initialize `AuditLogger` with `audit.db`
5. Initialize `AnnotationStore` with `annotations.db`
6. Load `route.json` if it exists (to resume a partially completed investigation)
7. If recorded mode: validate that `footage_path` exists and all expected camera folders are present
8. Surface the case details in the UI header bar

#### 10.4 — What happens when the operator creates a new case

1. Generate a case ID (auto-incremented or user-specified)
2. Create the case folder structure
3. Snapshot the current `configs/camera_graph.json` → `topology.json`
4. Snapshot the current `configs/default.yaml` → `config_snapshot.yaml`
5. Initialize empty databases (gallery, audit, annotations)
6. Log `INVESTIGATION_START` to audit trail
7. Open the case

#### 10.5 — Case switcher in UI

A sidebar or modal that shows all cases:
- Table with columns: Case ID, Status (🟢 Active, ✅ Completed, 📦 Archived), Date, Mode, Operator
- "New Case" button
- "Open" button for each case
- "Archive" / "Delete" actions
- Search/filter by status or date range

When switching cases, the system:
1. Closes the current case (flushes to disk)
2. Opens the new case (loads from disk)
3. Reinitializes all pipeline state from the loaded case data

#### 10.6 — REST endpoints

- `GET /api/cases` — list all cases
- `POST /api/cases` — create new case `{case_id, mode, operator, notes, footage_path?}`
- `POST /api/cases/{case_id}/open` — open/switch to a case
- `POST /api/cases/{case_id}/close` — close active case
- `DELETE /api/cases/{case_id}` — delete case (requires confirmation token)
- `GET /api/cases/{case_id}` — get case details

### Files Affected

| File | Change |
|------|--------|
| `src/case/__init__.py` | **[NEW]** Module init |
| `src/case/manager.py` | **[NEW]** `CaseManager`, `Case`, `CaseSummary` |
| `src/case/types.py` | **[NEW]** Case-related data types |
| `src/identity/manager.py` | Accept case-specific DB path instead of global path |
| `src/audit/logger.py` | Accept case-specific DB path |
| `src/playback/annotations.py` | Accept case-specific DB path |
| `src/multi_camera/ui_server.py` | Case management endpoints, case switcher |
| `src/app/main.py` | Initialize `CaseManager`, route pipeline through active case |
| `src/multi_camera/static/` (frontend) | Case switcher UI, case header bar |

### Success Criteria

- Each case is a self-contained folder that can be moved/copied independently
- Switching cases preserves all state of both the old and new case
- Opening a case from a previous session restores gallery, route, and annotations
- Case list is searchable and filterable
- Deleting a case requires explicit confirmation

---

## Phase 11 — Evidence Export Package

> **Priority: LOW-MEDIUM.** Needed before the system is used in actual legal proceedings.

### What Needs to Change

The journey video alone is not forensically admissible. A complete evidence package must prove:
1. What footage was analyzed
2. What the AI decided at each step
3. What the operator did at each step
4. That nothing was retroactively altered
5. What system configuration was in effect

### Plan

#### 11.1 — Evidence package structure

When the operator clicks "Export Evidence," the system creates a ZIP containing:

```
EVIDENCE_CASE-2026-0042.zip
├── manifest.json                         # lists all files with SHA-256 hashes
├── case.json                             # case metadata
├── topology.json                         # camera graph used
├── config_snapshot.yaml                  # system config used
├── audit_log.json                        # full audit trail exported as JSON (human-readable)
├── audit_chain_verification.txt          # output of hash chain verification
├── route.json                            # complete route events
├── route_report.txt                      # human-readable route report
├── annotations.json                      # all operator annotations
├── gallery/
│   ├── gallery_manifest.json             # list of all gallery entries with metadata
│   └── thumbnails/
│       ├── e1a2b3c4.jpg
│       └── ...
├── journey_video.mp4                     # stitched journey video
└── source_footage_hashes.json            # SHA-256 hashes of original source videos (NOT the videos themselves — they're too large)
```

#### 11.2 — Create `src/case/evidence_exporter.py`

```python
class EvidenceExporter:
    """Generates a forensically complete evidence package from a case."""

    def export(self, case: Case, output_path: str) -> str:
        """
        1. Verify audit log hash chain integrity
        2. Export audit log as human-readable JSON
        3. Compute SHA-256 hashes of all source footage files
        4. Copy gallery thumbnails
        5. Include route report and journey video
        6. Generate manifest.json with SHA-256 of every file in the package
        7. Create ZIP archive
        8. Return path to the ZIP
        """
        ...
```

#### 11.3 — `manifest.json` format

```json
{
  "package_version": 1,
  "case_id": "CASE-2026-0042",
  "exported_at": "2026-09-18T23:45:00Z",
  "exported_by": "Argus v2.0.0",
  "audit_chain_valid": true,
  "files": [
    {"path": "case.json", "sha256": "a1b2c3..."},
    {"path": "audit_log.json", "sha256": "d4e5f6..."},
    {"path": "journey_video.mp4", "sha256": "g7h8i9..."},
    ...
  ],
  "source_footage": [
    {"camera_id": "cam_entrance", "file": "D:/evidence/sept15/cam_entrance/video.mp4", "sha256": "j0k1l2..."},
    ...
  ]
}
```

The manifest allows a third party to verify:
- That the package contents haven't been modified (re-compute SHA-256 of each file)
- That the original footage hasn't been modified (compare SHA-256 with the originals)
- That the audit trail is intact (check `audit_chain_valid`)

#### 11.4 — UI integration

- "Export Evidence" button in the case management panel (only enabled for completed cases)
- Progress bar during export (hashing large video files takes time)
- Download link when export is complete
- Option to include or exclude the journey video (reduces file size significantly)

#### 11.5 — REST endpoints

- `POST /api/cases/{case_id}/export` — start evidence export, returns a task ID
- `GET /api/cases/{case_id}/export/status` — check export progress
- `GET /api/cases/{case_id}/export/download` — download the ZIP when ready

### Files Affected

| File | Change |
|------|--------|
| `src/case/evidence_exporter.py` | **[NEW]** Export logic, hashing, ZIP generation |
| `src/audit/logger.py` | Add `export_as_json()` and `verify_chain()` methods |
| `src/multi_camera/ui_server.py` | Export endpoints |
| `src/multi_camera/static/` (frontend) | Export button, progress bar, download link |

### Success Criteria

- Evidence package is a single self-contained ZIP
- Every file in the package has a verifiable SHA-256 hash
- Audit log hash chain is verified before export
- Source footage hashes are included for integrity verification
- A third party with the ZIP and the original footage can independently verify nothing was altered

---

## Phase 12 — Batch Processing for Recorded Mode

> **Priority: LOW-MEDIUM.** Turns multi-hour footage review into a focused human review session.

### What Needs to Change

Phase 3 describes recorded mode as playback-speed processing: the operator watches the video play and intervenes when needed. But for hours of footage across many cameras, this means the operator sits through the entire duration. If the AI is highly confident for 95% of the footage, the operator is wasting time watching segments that don't need intervention.

### Plan

#### 12.1 — Add "Fast Scan" mode to PlaybackController

Extend the `PlaybackController` from Phase 3 with a third processing strategy:

| Mode | Speed | When to use |
|------|-------|------------|
| **Playback** | 0.5x–4x real-time | Detailed review, watching footage |
| **Fast Scan** | As fast as GPU allows | Initial processing, confidence-gated |

In Fast Scan mode:
1. Process frames at maximum GPU throughput (no display rendering, no sleep between frames)
2. Track the target autonomously using the same AI pipeline
3. When a decision point occurs (handoff, ReID match, target lost), evaluate confidence:
   - **High confidence** (`> auto_add_threshold`): auto-accept and continue scanning
   - **Low confidence** (`< match_threshold`): auto-reject and continue scanning
   - **Uncertain** (between thresholds): **PAUSE** and switch to playback mode for human review
4. The operator reviews the uncertain segment, makes a decision, and can either:
   - Resume fast scan (press "Continue Scan")
   - Stay in playback mode to watch more carefully

#### 12.2 — Confidence-gated pause mechanics

The `PlaybackController` needs a `check_confidence_gate()` method called after every handoff decision:

```python
def check_confidence_gate(self, decision_type: str, confidence: float) -> None:
    """
    In fast-scan mode, pause processing if confidence falls in the uncertain band.
    Has no effect in playback mode.
    """
    if self.mode != "fast_scan":
        return

    uncertain_min = self.config.reid.uncertain_band_min  # 0.73
    uncertain_max = self.config.reid.uncertain_band_max  # 0.78

    if uncertain_min <= confidence <= uncertain_max:
        self.pause()
        self.switch_to_playback_mode()
        self.notify_operator(
            f"⚠️ Uncertain {decision_type} (confidence: {confidence:.2f}). "
            f"Please review and confirm or correct."
        )
```

#### 12.3 — Processing progress and statistics

During fast scan, show a progress panel:
- Overall progress bar (percentage of total footage processed)
- Processing speed (e.g., "12.5x real-time")
- Segments processed without intervention
- Number of auto-accepted handoffs
- Number of pauses requiring human review
- Estimated time remaining

After fast scan completes, show a summary:
- Total footage duration vs. actual processing time
- Number of uncertain segments reviewed by human
- Number of corrections made
- Final route preview

#### 12.4 — Rewind from pause point

When fast scan pauses at an uncertain moment, the operator needs context. Auto-seek backwards 5 seconds from the pause point so the operator sees the lead-up to the uncertain event, not just the frozen uncertain frame.

#### 12.5 — Skip-ahead for known dead zones

Allow the operator to mark time ranges as "skip" (e.g., the target is known to be inside a building with no cameras for 30 minutes). During fast scan, these ranges are skipped entirely. This is implemented as annotation type "skip_zone" with a start and end timestamp.

#### 12.6 — Configurable confidence gate thresholds

Add to `Advanced Settings` (Phase 6):
- "Fast Scan Auto-Accept Threshold" (default: `auto_add_threshold` = 0.82)
- "Fast Scan Auto-Reject Threshold" (default: `match_threshold` = 0.78)
- "Fast Scan Rewind Seconds" (default: 5)

The tighter the band, the more often the scan pauses for human input. The wider the band, the more autonomy the AI has (but with higher risk of unreviewed errors).

### Files Affected

| File | Change |
|------|--------|
| `src/playback/controller.py` | Add fast-scan mode, confidence gate, progress tracking |
| `src/playback/annotations.py` | Add "skip_zone" annotation type |
| `src/core/config.py` | Add fast-scan thresholds to `SearchConfig` or new `PlaybackConfig` |
| `src/multi_camera/ui_server.py` | Fast-scan start/pause/resume endpoints, progress endpoint |
| `src/multi_camera/static/` (frontend) | Fast-scan controls, progress panel, summary screen |

### Success Criteria

- 4 hours of footage can be processed in under 30 minutes with fast scan
- Uncertain moments pause automatically for human review
- Operator sees 5 seconds of context before each pause point
- Skip zones are respected during scanning
- Processing statistics are visible throughout

---

## Dependency Order & Implementation Sequence

```
Phase 0 (Topology Reliability)     ← MUST be first, everything depends on topology
    │
    ▼
Phase 1 (Camera Detection Fix)     ← Unblocks reliable camera usage
    │
    ├──────────────┐
    ▼              ▼
Phase 2          Phase 5
(Gallery &       (Scalable Camera
 Correction)      Management)
    │              │
    ▼              │
Phase 3            │
(Recorded Mode)    │
    │              │
    ├──────────────┘
    ▼
Phase 4 (Handoff Hardening)        ← Benefits from both recorded mode and gallery work
    │
    ▼
Phase 6 (UI/UX Overhaul)          ← Final polish, integrates everything
    │
    ├──────────────────────────────┐
    ▼                              ▼
Phase 7 (Audit Trail)          Phase 8 (Undo/Redo)
    │                              │
    ▼                              │
Phase 9 (Annotations)             │
    │                              │
    ├──────────────────────────────┘
    ▼
Phase 10 (Case Management)     ← Needs audit trail, annotations, gallery persistence
    │
    ├──────────────┐
    ▼              ▼
Phase 11         Phase 12
(Evidence         (Batch
 Export)           Processing)
```

Phases 2 and 5 can be worked in parallel since they affect different parts of the codebase. Phase 3 depends on Phase 2 (gallery persistence is needed for recorded mode corrections). Phase 6 should be last of the core phases since it integrates and polishes previous work.

Phases 7 and 8 can be parallelized (audit trail and undo/redo affect different layers). Phase 10 depends on 7 and 9 (case folders need to include audit logs and annotations). Phases 11 and 12 can be parallelized as the final tier.

---

## Estimated Scope per Phase

| Phase | New Files | Modified Files | Complexity | Risk |
|-------|-----------|---------------|------------|------|
| 0 — Topology | 0 | 4 | Low | Low (isolated fix) |
| 1 — Camera Detection | 0 | 3 | Medium | Medium (OS-specific DirectShow behavior) |
| 2 — Gallery & Correction | 0 | 5 | Medium | Low (well-isolated changes) |
| 3 — Recorded Mode | 5-6 | 4-5 | **High** | **High** (new subsystem, sync complexity) |
| 4 — Handoff Hardening | 0 | 4 | Medium | Medium (state machine complexity) |
| 5 — Camera Management | 0 | 4 | Medium | Low (mostly frontend) |
| 6 — UI/UX Overhaul | 1 | 5 | Medium | Low (progressive enhancement) |
| 7 — Audit Trail | 2 | 4 | Medium | Low (append-only, isolated) |
| 8 — Undo/Redo | 1 | 3 | Medium | Medium (state rollback complexity) |
| 9 — Annotations | 1-2 | 3 | Low | Low (additive data model) |
| 10 — Case Management | 2-3 | 4 | Medium | Medium (file I/O, session lifecycle) |
| 11 — Evidence Export | 2 | 2 | Medium | Low (packaging, no core changes) |
| 12 — Batch Processing | 1 | 3 | Medium | Medium (playback controller changes) |

---

## Architectural Principles for This Refactor

1. **Never break live mode.** Every change must keep live camera mode fully functional. Recorded mode is additive.
2. **The human is always right.** When the operator clicks to correct, the system accepts immediately. No "are you sure?" friction on target correction — only on destructive actions (clear gallery, new investigation).
3. **Gallery is sacred.** The gallery is the operator's curated evidence. It should never be modified without explicit human action (auto-add is the exception, and even that should be reviewable).
4. **Fail visible, not silent.** Every error (topology load failure, camera disconnect, ReID uncertainty) must be surfaced in the UI. No `except: pass` in critical paths.
5. **Recorded and live share the same AI pipeline.** Detection, tracking, ReID, and identity management must not have separate code paths for live vs. recorded. Only the frame source and timing differ.
6. **Configuration has one source of truth.** Every threshold should be defined in exactly one place (`configs/default.yaml`), loaded via `AppConfig`, and injected into modules. No hardcoded fallback defaults that differ from the config file.
7. **Every action is auditable.** Every operator action and every AI decision must be logged with enough detail to reconstruct the full sequence of events after the fact. This is non-negotiable for forensic use.
8. **Mistakes are reversible.** The operator must be able to undo recent corrections without restarting the investigation. Destructive actions (clear gallery, delete case) require explicit confirmation.
9. **Cases are self-contained.** A case folder must contain everything needed to reproduce or review the analysis: topology, gallery, route, annotations, audit log, and configuration snapshot. Moving or copying the folder should "just work."
