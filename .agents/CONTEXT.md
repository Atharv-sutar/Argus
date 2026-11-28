# Argus Compact Context (`.agents/CONTEXT.md`)

> Persistent engineering context, verified ReID benchmarks, runtime bugfix history, and redundant-code inventory for AI agents working on **Argus**. Consult this file (referenced in [`AGENTS.md`](file:///e:/MAJOR_PROJECT/Project/AGENTS.md)) before debugging ReID, tracking, target selection/rendering, multi-camera switching, or reviewing legacy shims.

---

## 1. Environment & Quick Commands

- **Python Virtual Environment** (required for all execution):
  ```powershell
  .\.venv\Scripts\python -m pytest tests/unit tests/integration -q -r a
  .\.venv\Scripts\python -m src.app.main
  .\.venv\Scripts\python scripts/evaluate_reid.py --frames 360 --seed-cam cam_2 --seed-frame 120
  ```
- **Web Operations Center**: `http://127.0.0.1:8765` (config `multi_camera.ui_port: 8765`, overridable via `--ui-port`).
- **Windows Launchers**: `run.bat` / `run.ps1` (CRLF enforced via `.gitattributes`, invoke `scripts/bootstrap.py`, pass `--device`, `--synthetic`, `--target-id`, `--source`, `--video-dir`, `--ui-port`, `--map-ui`). `src/core/dependencies.py` checks installed versions via stdlib tuple comparison before invoking `pip`.
- **Hardware-Adaptive Inference**: `detect_hardware_profile()` (`src/inference/device.py`) scales batch size, FP16/TF32 precision, ORT/TensorRT memory budgets, detector variant, and ReID interval dynamically across CPU, compact CUDA GPUs (`<6 GB`), mid-tier RTX GPUs (`6–10 GB`), and high-VRAM GPUs (`>=10 GB`).
- **Future Optimization Guide**: See [`docs/MIGRATION_PERFORMANCE.md`](file:///e:/MAJOR_PROJECT/Project/docs/MIGRATION_PERFORMANCE.md) for empirically measured bottlenecks and native C++/CUDA/Rust compatibility caveats (no migration performed yet).

---

## 2. ReID Empirical Benchmarks & Active Configuration

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
- `reid`: `model_name: auto` (resolves via `HardwareProfile` -> `person_reid_youtu`), `extract_interval_frames: auto`, `batch_size: auto`
- Thresholds: `match_threshold: 0.60`, `reacquisition_threshold: 0.65`, `auto_add_threshold: 0.74`, `uncertain_band_min: 0.52`, `uncertain_band_max: 0.65`, `min_margin: 0.10`, `min_sharpness: 12.0`
- Gallery: `max_gallery_size: 100`
- Active runtime fixes (`fixes.*: True`, `tracking.fix_association_cost: True`, `target.switch_when_better: True`, `target.score_all_tracks: True`, `target.blur_gate_scope: "all"`).

---

## 3. Critical Architectural & Bugfix Notes

1. **Sticky / Wrong-Target Yellow Box (`src/visualization/annotator.py`)**:
   - Solid yellow `(0, 215, 255)` box draws **only** when `target.state` is in `(LOCKED, TRACKING, CONFIRMED, ACQUIRING_REFERENCE)` **and** `target.track_id` matches a live track.
   - `OCCLUDED` draws magenta brackets; `UNCERTAIN`/`LOST`/`SEARCHING` draw amber corner brackets only; `REJECTED`/`LOST_PERMANENTLY` draw no target box.
2. **Authorization & State Sync (`src/target/manager.py`, `src/identity/manager.py`, `src/pipeline/multi_camera_pipeline.py`)**:
   - `TargetManager.reassociate_target` and `IdentityManager.enroll_cross_camera_viewpoint` enforce `VerifiedIdentityDecision.is_authorized_for(...)`.
   - `MultiCameraPipeline._apply_node_config_update` unifies node disable/enable transitions (`CameraNode.mark_disabled()`) across `_sync_nodes_with_graph()` and `update_graph()`, and `_deactivate_search_cameras()` notifies `SearchManager.on_search_complete()`.
   - `update_graph()` releases and clears `playback_controller = None` whenever no `VIDEO_FILE` cameras remain so recorded-to-live view switches resume streaming cleanly.
3. **UI Server & Cases (`src/multi_camera/ui_server.py`, `static/js/app.js`)**:
   - `/api/cases/<id>/close` restores `default_annotation_db_path` and `default_identity_db_path`.
   - `/api/cameras/apply_discovered` and `/api/system/switch_source` (`mode == "live"`) construct valid `CameraNodeConfig` and `CameraEdgeConfig(direction=EdgeDirection.BIDIRECTIONAL)`.
4. **Offline Evaluation (`src/benchmark/evaluator.py`, `scripts/evaluate_reid.py`)**:
   - `ReIDEvaluationHarness.run()` synchronizes async worker initialization before stepping frames, with storage/recording persistence disabled.

---

## 4. Redundant & Unused Markers Inventory

Retained backward-compatibility shims, extension hooks, and diagnostic/public APIs are explicitly annotated with `REDUNDANT` or `UNUSED` comments (exact line ranges) at the top of each source file (`src/core/types.py`, `src/tracking/bot_sort.py`, `src/reid/color_normalizer.py`, `src/camera/video_file.py`, `src/playback/annotations.py`, `src/playback/controller.py`, `src/multi_camera/camera_graph.py`, `src/inference/device.py`, `src/inference/tensorrt_engine.py`, `src/detection/yolo_detector.py`, `src/reid/extractor.py`, `src/tracking/__init__.py`, `src/benchmark/evaluator.py`, `src/multi_camera/ui_server.py`, `src/multi_camera/static/js/api.js`).

---

## 5. Known Limitations & Local State Notes

- **Border-Clipped Initial Seeding**: Seeding on a heavily clipped silhouette entering frame edge (observed on EPFL `cam_0` frame 35) can cause similarity to drop once full-body enters unless additional viewpoints are enrolled.
- **Uncommitted Local Topology Files**: Live UI sessions modify `configs/camera_graph.json` and `configs/camera_graph.json.bak` with machine-local paths; keep local edits out of commits unless intentionally updating repo defaults.
- **EPFL Offline Baseline (`scripts/evaluate_reid.py --frames 360 --seed-cam cam_2 --seed-frame 120`)**: `Precision=100.00%`, `Recall=67.05%`, `F1=80.27%`, `FP Rate=0.00%` (`TP=177, FP=0, ID-Swaps=0, Recovery Events=2`).
