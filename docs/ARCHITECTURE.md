# Argus — System Architecture Overview

This document describes the modular software architecture, processing flow, and core design invariants implemented in **Argus**.

---

## 1. Processing Flow

### Single-Camera Loop
```text
Camera -> Detection (YOLO11) -> Tracking (BoT-SORT) -> Target State -> ReID / Identity -> Visualization
```

### Multi-Camera Recovery Loop
```text
Active Camera -> Target Lost -> Search Manager (Graph Radius R1..R3 + Transit Time Gating)
    -> Neighbor Camera Activation -> Batched Detection & ReID -> Verified Cross-Camera Handoff
```

---

## 2. Package Responsibilities (`src/`)

Dependency direction flows downward (`app -> pipeline -> domain modules -> core interfaces/types`) so AI components can be swapped via registries without rewriting orchestration logic:

| Package | Key Modules | Responsibility |
| :--- | :--- | :--- |
| **`app/`** | `main.py` | Unified CLI parsing, config loading, launcher integration |
| **`core/`** | `types.py`, `interfaces.py`, `config.py`, `multi_camera_types.py`, `dependencies.py`, `undo_stack.py` | Domain dataclasses, abstract interfaces (`BaseDetector`, `BaseTracker`, `BaseReID`, `BaseVectorStore`, `BaseCamera`), YAML config |
| **`inference/`** | `device.py`, `tensorrt_engine.py` | Hardware capability profiling (`HardwareProfile`), TensorRT/ONNX CUDA provider setup, VRAM trimming |
| **`camera/`** | `capture.py`, `video_file.py`, `directory_loader.py` | Threaded OpenCV webcam/RTSP reader (`OpenCVCamera` 1-frame freshness buffer), looped `VideoFileCamera`, multi-cam folder discovery |
| **`detection/`** | `yolo_detector.py` | YOLO11 detector (`DETECTOR_REGISTRY`, batched frame inference) returning `DetectionResult` |
| **`tracking/`** | `bot_sort.py`, `byte_tracker.py` | `TRACKER_REGISTRY`; BoT-SORT Kalman `[xc,yc,w,h]`, GMC (`sparseOptFlow`), SciPy Hungarian association |
| **`target/`** | `manager.py` | Manual target locking, track correction, token-authorized reassociation (`VerifiedIdentityDecision.is_authorized_for`) |
| **`reid/`** | `extractor.py`, `quality.py`, `decision_logger.py`, `color_normalizer.py` | `REID_MODEL_REGISTRY` (`person_reid_youtu` 768D ONNX/CUDA default, OSNet, MobileNetV3), Laplacian blur/size quality gating |
| **`identity/`** | `manager.py`, `evidence.py`, `faiss_store.py`, `sqlite_store.py`, `serialization.py` | Trusted vs provisional galleries, multi-frame `EvidenceEngine` hysteresis, FAISS + SQLite persistence, anti-scoop rollback |
| **`multi_camera/`** | `camera_graph.py`, `camera_node.py`, `search_manager.py`, `search_state.py`, `camera_priority.py`, `ui_server.py` | Topology graph & shortest transit time queries, priority ranking, search state machine, REST/MJPEG/SSE server + Vanilla JS UI |
| **`pipeline/`** | `multi_camera_pipeline.py`, `camera_worker.py`, `single_camera.py` | 4-phase orchestrated acquisition -> batched YOLO -> parallel tracking -> batched ReID -> annotation/JPEG encoding |
| **`playback/`** | `controller.py`, `annotations.py`, `route_recorder.py`, `journey_stitcher.py`, `report.py` | Multi-video synchronized playback clock, SQLite annotations, route logging, journey MP4 stitching |
| **`forensic/` & `case/` & `recording/` & `audit/`** | `clip_recorder.py`, `manager.py`, `evidence_exporter.py`, `segment_writer.py`, `retention_manager.py`, `logger.py` | Target forensic clip capture, ZIP evidence export, continuous live NVR segment rotation/retention, tamper-evident audit logs |
| **`visualization/`** | `annotator.py` | Pure display-only overlays (gold/yellow locked box + corner brackets, magenta occluded brackets, amber lost/uncertain brackets) |
| **`benchmark/`** | `evaluator.py`, `types.py` | Offline multi-camera ReID evaluation harness and ground-truth metrics |

---

## 3. Key ReID & Tracking Design Decisions

1. **Aspect-Ratio Preserved Letterbox Preprocessing**: Person crops in `src/reid/extractor.py` are padded with `BORDER_REPLICATE` to `1:2` (`256x128`) and normalized with ImageNet RGB mean/std without CLAHE or Gray-World color constancy (which empirical EPFL evaluation showed degraded discriminative clothing color cues).
2. **Event-Driven & Batched Execution**: During steady `LOCKED`/`TRACKING`, only the active target track runs periodic ReID unless occlusion, uncertainty, loss, or `score_all_tracks` triggers full candidate evaluation across active/searching cameras.
3. **Immutable Seed Anchor + Anti-Scoop Protection**:
   - Manual clicks seed protected `trusted_gallery` entries (`max_reference_samples: 50`) stored as real crop images + embeddings.
   - `provisional_gallery` auto-enrollment (`auto_add_threshold: 0.74`) is rolled back on track switches, and cross-camera handoffs require multi-frame `EvidenceEngine` confirmation tokens (`VerifiedIdentityDecision`).
4. **Hardware-Adaptive Execution**: All batch sizes, precision (`FP16`/`TF32`), model defaults, and execution provider selection scale automatically via `HardwareProfile` without hardcoded GPU specs.
