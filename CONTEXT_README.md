# Argus Compact Context README

> Persistent engineering context, verified ReID benchmarks, bugfix history, and redundant-code inventory for **Argus**. Consult this file (referenced in `AGENTS.md`) before debugging ReID, tracking, target selection, UI topology/controls, or cleaning legacy fields.

---

## 1. Environment & Quick Commands

- **Python Virtual Environment** (required for all execution):
  ```powershell
  .\.venv\Scripts\python -m src.app.main
  .\.venv\Scripts\python -m pytest tests/unit tests/integration -q -r a
  .\.venv\Scripts\python scripts/evaluate_reid.py --frames 360 --seed-cam cam_2 --seed-frame 120
  ```
- **Web Operations Center**: `http://127.0.0.1:8798`
- **Hardware-Adaptive Inference**: Auto-detects host CPU/GPU capabilities (`src/inference/device.py:HardwareProfile`) to scale batch size, FP16/TF32 precision, ORT/TensorRT memory budgets, detector variant, and ReID cadence dynamically across CPU, entry GPUs (`<6 GB`), mid-tier Tensor/RTX GPUs (`6–10 GB`), and high-VRAM RTX/workstation GPUs (`>=10 GB`).
- **Git Baseline**: Originally documented at `6cac660` (`https://github.com/Atharv-sutar/Argus.git`).

---

## 2. Core Architecture Summary

- **Single-camera flow**: `camera -> detection -> tracking -> target -> reid/identity -> visualization`
- **Multi-camera recovery flow**: `active camera -> target loss -> search manager -> candidate cameras -> detection/ReID -> target recovery -> active camera handoff`

### Key Subsystems & Entrypoints
- **Hardware Profiler (`src/inference/device.py`, `src/inference/tensorrt_engine.py`)**: `detect_hardware_profile()` probes GPU VRAM, compute capability, Tensor Cores, TensorRT, and CPU threads to build a `HardwareProfile` (`cpu`, `gpu_entry`, `gpu_mid`, `gpu_high`) and scale FP16/TF32, batch sizes, ORT/TRT arenas, and `pynvml` memory reporting.
- **Detector (`src/detection/yolo_detector.py`)**: Modular `DETECTOR_REGISTRY` + `create_detector()` supporting `"auto"` hardware-tiered YOLO11 selection (`.engine` / `.pt` CUDA FP16/TF32 / `.onnx` CPU) with `YOLO_AUTOINSTALL=false`.
- **Tracker (`src/tracking/bot_sort.py`, `src/tracking/__init__.py`)**: Modular `TRACKER_REGISTRY` + `create_tracker()` wrapping BoT-SORT (Kalman filter + GMC + Hungarian association) and ByteTracker.
- **ReID (`src/reid/extractor.py`)**: Modular `REID_MODEL_REGISTRY` + `create_reid_extractor()` + `get_reid_model_catalog()`; `person_reid_youtu` (`src/reid/backbones/person_reid_youtu_2021nov.onnx`, 768D) default with `osnet_x1_0`, `osnet_x0_25`, and `mobilenet_v3_small`.
- **Identity & Storage (`src/identity/manager.py`, `src/identity/sqlite_store.py`)**: `IdentityManager` (`reid_extractor` getter+setter for live switching) with FAISS vector search + SQLite persistence, disk gallery verification/rebuild (`verify_gallery_from_disk`, `rebuild_gallery_from_disk`), and dynamic case DB switching (`set_db_path`).
- **Pipeline Orchestrator (`src/pipeline/multi_camera_pipeline.py`)**: Coordinates 4-phase acquisition, batched detection, BoT-SORT tracking, batched ReID, target evaluation, cross-camera search/handoff, annotated JPEG streaming, and live hardware telemetry (`get_telemetry().hardware_info`).
- **Multi-Camera Graph (`src/multi_camera/`)**: `camera_graph.py`, `camera_node.py`, `transition.py`, `search_manager.py`, `search_state.py`, `camera_priority.py`.
- **NVR Recording (`src/recording/`)**: `manager.py`, `retention_manager.py`, `segment_index.py`, `segment_writer.py` — continuous segment recording for live `webcam`/`rtsp` feeds tracked via SQLite `segments.db` (excludes offline files).
- **Dynamic Dependencies (`src/core/dependencies.py`)**: Checks/installs `ultralytics`, `torch`/`torchvision`, `scipy`, `faiss-cpu`, `onnx`, `onnxruntime-gpu`, and `nvidia-ml-py`.
- **Web UI (`src/multi_camera/ui_server.py`, `static/index.html`, `static/js/app.js`, `static/js/graph.js`)**: Vanilla JS + CSS variables + HTML5 Canvas with dynamic hardware capability telemetry and ReID catalog settings.

---

## 3. ReID Empirical Benchmarks & Active Configuration

### Backbone Evaluation (EPFL Campus4 Ground Truth — `118` crops, `14` identities)

| Backbone | Runtime | AUC | Rank-1 | Genuine Sim | Impostor Sim | Quality Gate |
|---|---|---|---|---|---|---|
| `osnet_x0_25` | ONNX CUDA | `0.8185` | `79.3%` | `0.972` (`p5=0.918`) | `0.925` (`p95=0.984`) | `False` |
| `osnet_x1_0` | ONNX CUDA | `0.8159` | `75.9%` | `0.998` | `0.995` | `False` |
| **`person_reid_youtu`** | **ONNX CUDA (768D)** | **`0.9546`** | **`96.6%`** | **`0.806` (`p5=0.532`)** | **`0.362` (`p95=0.801`)** | **`True`** |

- Prior `osnet_x0_25` default scored arbitrary bystanders at `0.92–0.98` similarity.
- With `person_reid_youtu`, `seed45` vs `cam_2` target similarity measured `0.693–0.837` (mean `0.797`) while bystanders peaked at `0.196`.

### Active Defaults (`configs/default.yaml` & `src/core/config.py`)
- `inference`: `device: auto`, `engine_type: auto`, `precision: auto`, `batch_size: auto`
- `detection`: `backend: yolo`, `model_name: auto` (YAML) / `yolo11n.pt` (dataclass default)
- `reid`: `model_name: auto` (resolves via `HardwareProfile` -> `person_reid_youtu`), `extract_interval_frames: auto` (YAML), `batch_size: auto`
- Thresholds: `match_threshold: 0.60`, `reacquisition_threshold: 0.65`, `auto_add_threshold: 0.74`, `uncertain_band_min: 0.52`, `uncertain_band_max: 0.65`, `min_margin: 0.10`, `min_sharpness: 12.0`
- Gallery: `gallery.max_size: 100` (artificial UI/server `25`-crop cap removed)
- Enabled runtime flags (all 9 fix flags were previously `False`, causing BoT-SORT ID fragmentation and HTTP vs `step()` race conditions):
  - `tracking.fix_association_cost: True`
  - `target.switch_when_better: True`
  - `target.score_all_tracks: True`
  - `target.blur_gate_scope: "all"`
  - `fixes.*: True`

---

## 4. Major Bugfixes Implemented (Through Commit `6cac660`)

1. **Sticky / Wrong-Target Yellow Bounding Box (`src/visualization/annotator.py`)**:
   - `FrameAnnotator.annotate()` draws the solid yellow `(0, 215, 255)` box **only** when `target.state` is in `(LOCKED, TRACKING, CONFIRMED, ACQUIRING_REFERENCE)` **and** `target.track_id` matches a live track in `track_result.tracks`.
   - `OCCLUDED` renders magenta corner brackets.
   - `UNCERTAIN` / `LOST` / `SEARCHING` render amber corner brackets only.
   - `REJECTED` / `LOST_PERMANENTLY` render no target box over `last_known_box` or stale tracks.
2. **Continuous Phase-4 ReID & Selection Safety (`src/pipeline/multi_camera_pipeline.py`)**:
   - Included `TargetState.SEARCHING` and `TargetState.REJECTED` in `target_is_lost` / `target_lost_or_uncertain`, and `LOCKED` / `SEARCHING` / `REJECTED` in `_evaluate_active_camera_target` `should_reid`, ensuring ReID runs every frame during recovery instead of skipping non-interval frames.
   - Replaced hardcoded `synth_id = 1` fallback in `select_target_on_camera` and `correct_target_on_camera` with collision-free `max(existing_ids, default=0) + 1`, preserving existing tracks.
   - Passed explicit `frame_ids=[self._frame_count]` in `detect_batch(...)` and guarded `track_res is None` inside `_evaluate_active_camera_target`.
3. **Identity & Case Switching (`src/identity/manager.py`, `src/multi_camera/ui_server.py`)**:
   - Added `IdentityManager.set_db_path(db_path)` (called by `ui_server.py:1142` for case switching).
   - Wired `self.w_deep`, `self.w_upper`, `self.w_color`, and `self.w_lower` inside `evaluate_candidate_crop`.
4. **UI Selection Interaction (`src/multi_camera/static/js/app.js`, `index.html`, `ui_server.py`)**:
   - Prevented plain clicks while `target.state` is `REJECTED` or `LOST_PERMANENTLY` from triggering `is_correction`.
   - Removed the `25`-crop limit in favor of `gallery.max_size` (`100`).
5. **Offline Evaluation Harness (`src/benchmark/evaluator.py`, `scripts/evaluate_reid.py`)**:
   - Added async worker readiness synchronization in `ReIDEvaluationHarness.run()` (fixes `End of footage reached at frame 0`) and disabled storage/recording persistence during offline evaluation.

---

## 5. Dead & Redundant Code Audit Inventory

### Removed
- `src/pipeline/multi_camera_pipeline.py`: duplicate standby loop (old lines `1361–1365`).
- `src/identity/manager.py`: four empty `pass` stub methods `undo_gallery_add`, `redo_gallery_add`, `undo_gallery_clear`, `redo_gallery_clear` (old lines `377–387`).

### Kept & Marked Redundant
| Location | Lines | Symbol / Code | Reason Retained |
|---|---|---|---|
| `src/tracking/bot_sort.py` | `480`, `493` | `appearance_thresh: float = 0.35`, `self.appearance_thresh = appearance_thresh` | Retired key kept solely for kwargs backwards compatibility. |
| `src/core/types.py` | `324–325`, `331` | `rejected_gallery: List[Embedding]` (serialized at `src/identity/serialization.py:83,107`) | Unpopulated/unread field retained for SQLite/JSON schema compatibility. |
| `src/reid/color_normalizer.py` | `16–115` | `class ColorNormalizer` (tested in `tests/unit/reid/test_color_normalizer.py`) | Kept standalone as reference; excluded before `person_reid_youtu` because enabling it dropped EPFL Rank-1 from `96.6%` to `93.1%`. |

---

## 6. Known Limitations & Local State Notes

- **Border-Clipped Initial Seeding**: Seeding on a heavily border-clipped partial silhouette entering the frame edge (observed on EPFL `cam_0` frame 35 `[2.7, 147.5, 39.0, 287.6]`) can cause cosine similarity to fall below `uncertain_band_min` once the full body enters if no full-body crop is enrolled yet.
- **Uncommitted Local Topology Files**: Live UI runs may modify `configs/camera_graph.json` and `configs/camera_graph.json.bak` with machine-specific absolute paths (`campus4-c0..c2`). Leave these local files out of git commits unless intentionally updating shared repository defaults.

---

## 7. Baseline Verification Results (Post-`6cac660`)

- **Test Suite (`.\.venv\Scripts\python -m pytest tests/unit tests/integration -q -r a`)**:
  - `207 passed in 90.40s` (`0 failed`, `0 skipped`), including `tests/unit/pipeline/test_yellow_box_and_selection_fixes.py`.
- **EPFL ReID Evaluation (`.\.venv\Scripts\python scripts/evaluate_reid.py --frames 360 --seed-cam cam_2 --seed-frame 120`)**:
  - `Precision: 100.00%`, `Recall: 67.05%`, `F1: 80.27%`, `FP Rate: 0.00%`
  - `TP=177, FP=0, FN=87, TN=96, ID-Swaps=0, Recovery Events=2`
  - `Mean Target Sim: 0.915`, `Mean Bystander Sim: 0.151`, `Gap: +0.765` (`diagnostics/reid_eval_report.json`)
