# Argus Migration & Performance Optimization Guide

This document records **only empirically verified performance bottlenecks** measured in the Argus codebase where migrating specific components to a native compiled language (C++ / CUDA / Rust / Cython) genuinely improves startup time, FPS/latency, multi-camera scalability, or memory footprint.

> [!NOTE]
> Components that already execute in fast native libraries or consume negligible per-frame time (for example, [`IdentityManager.match_batch_details`](file:///e:/MAJOR_PROJECT/Project/src/identity/manager.py#L436-L476) at **0.351 ms** for 25 candidates, `faiss-cpu` vector lookup, `scipy.optimize.linear_sum_assignment`, SQLite persistence, and the Vanilla JS frontend) are **intentionally excluded**.

---

## Measured Host Baseline

Measured on the local development host (`NVIDIA GeForce GTX 1650`, 4.0 GB VRAM, `tier=gpu_entry`, Python `3.12.1`, `torch 2.5.1+cu121`, `ultralytics 8.4.126`, `onnxruntime-gpu 1.19.2`, `opencv-python 5.0.0.93`):

| Operation | Measured Cost | Source Reference |
| :--- | :--- | :--- |
| `import torch` (cold) | **2.086 s** | PyTorch runtime initialization |
| `create_detector` (`yolo11n.pt`) | **1.807 s** | [yolo_detector.py](file:///e:/MAJOR_PROJECT/Project/src/detection/yolo_detector.py) |
| `create_reid_extractor` (ONNX CUDA) | **0.458 s** | [extractor.py](file:///e:/MAJOR_PROJECT/Project/src/reid/extractor.py), [tensorrt_engine.py](file:///e:/MAJOR_PROJECT/Project/src/inference/tensorrt_engine.py) |
| 4-camera `MultiCameraPipeline` cold init | **6.349 s** | [multi_camera_pipeline.py](file:///e:/MAJOR_PROJECT/Project/src/pipeline/multi_camera_pipeline.py) |
| 4-camera 720p `step()` end-to-end | **39.21 ms** mean (**25.5 FPS**), **63.26 ms** p95 | [`MultiCameraPipeline.step`](file:///e:/MAJOR_PROJECT/Project/src/pipeline/multi_camera_pipeline.py#L1182) |
| `YOLO detect_batch` (4 x 720p frames) | **46.62 ms / batch** | [`YOLODetector.detect_batch`](file:///e:/MAJOR_PROJECT/Project/src/detection/yolo_detector.py#L226) |
| `ReID _preprocess` (12 crops on CPU) | **10.45 ms** (~15.1% of **69.06 ms** `extract_batch`) | [`PyTorchReIDExtractor._preprocess`](file:///e:/MAJOR_PROJECT/Project/src/reid/extractor.py#L310) |
| 720p resize + `cv2.imencode('.jpg')` (x1 / x4 cams) | **3.78 ms** / **15.11 ms** CPU | [`MultiCameraPipeline.step` Phase 2](file:///e:/MAJOR_PROJECT/Project/src/pipeline/multi_camera_pipeline.py#L1688) |
| `BoTSORTTracker.update` (25 tracks, `gmc=sparseOptFlow`) | **12.085 ms / cam** (**48.34 ms** x4 cams) | [`BoTSORTTracker.update`](file:///e:/MAJOR_PROJECT/Project/src/tracking/bot_sort.py#L517) |
| `BoTSORTTracker.update` (25 tracks, `gmc=none`) | **3.250 ms / cam** (**13.00 ms** x4 cams) | [`BoTSORTTracker.update`](file:///e:/MAJOR_PROJECT/Project/src/tracking/bot_sort.py#L517) |
| `ForensicClipRecorder` RAM buffer limit | **9000 x 1280x720x3 BGR** (**~24.88 GB** ceiling before disk spill) | [clip_recorder.py](file:///e:/MAJOR_PROJECT/Project/src/forensic/clip_recorder.py#L73-L196) |

---

## 1. Unified C++ / TensorRT Inference Runtime (Eliminate Dual PyTorch + ONNX Runtime)

- **Targets**: [yolo_detector.py](file:///e:/MAJOR_PROJECT/Project/src/detection/yolo_detector.py), [tensorrt_engine.py](file:///e:/MAJOR_PROJECT/Project/src/inference/tensorrt_engine.py), [extractor.py](file:///e:/MAJOR_PROJECT/Project/src/reid/extractor.py)
- **Bottleneck**:
  - Argus currently loads **two heavy GPU runtimes in parallel**: PyTorch + `ultralytics` for `yolo11n.pt` detection (`2.086 s` to import `torch` + `1.807 s` to load YOLO) and `onnxruntime-gpu` (`0.458 s`) for ReID.
  - Maintaining separate PyTorch and ONNX Runtime CUDA allocators increases host memory overhead, cold startup latency (**6.349 s** pipeline init), and VRAM fragmentation on 4 GB GPUs.
- **Recommended Migration**: **C++ TensorRT / ONNX Runtime C++ (`ort::Session`) service or `pybind11` module** executing exported YOLO11 + ReID models inside one runtime, removing `torch` and `ultralytics` from live execution paths.
- **Expected Product Impact**:
  - **Startup Time**: Cuts **~3.5–4.0 s** from cold launch (removing `import torch` + Ultralytics model wrapper overhead).
  - **RAM / VRAM**: Removes PyTorch's duplicate CUDA context and Python wheel footprint (~400–800 MB host RAM, ~300–500 MB VRAM headroom on GTX 1650).
  - **FPS**: Lower kernel launch and Python wrapper overhead for single-stage and batched TensorRT `.engine` execution.

---

## 2. Fused CUDA / C++ ReID Crop Preprocessing (`_preprocess` & `extract_batch`)

- **Target**: [`PyTorchReIDExtractor._preprocess`](file:///e:/MAJOR_PROJECT/Project/src/reid/extractor.py#L310-L346) and [`extract_batch`](file:///e:/MAJOR_PROJECT/Project/src/reid/extractor.py#L433-L474)
- **Bottleneck**:
  - Preprocessing 12 crops takes **10.45 ms** CPU per batch (~15.1% of the **69.06 ms** `extract_batch` call).
  - Each crop runs sequentially in Python through `cv2.copyMakeBorder` -> `cv2.resize` -> `cv2.cvtColor` -> `astype(np.float32) / 255.0` -> `(rgb - mean) / std` -> `np.transpose` -> `np.stack`, generating multiple short-lived CPU buffers and extra PCIe host-to-device uploads.
- **Recommended Migration**: **C++ (`pybind11` / CUDA kernel) or Cython + SIMD extension** executing batched `(crop slice -> replicate letterbox -> bilinear resize -> BGR-to-RGB float32 ImageNet normalize -> NCHW layout)` in a single pass (ideally directly from frame buffer / GPU `GpuMat`).
- **Expected Product Impact**:
  - **FPS / Latency**: Drops 12-crop preprocessing from **~10.5 ms to <1.0 ms**, especially impactful during `LOST`/`UNCERTAIN` recovery or multi-person crowds where `score_all_tracks` runs ReID across multiple tracks/cameras every N frames.

---

## 3. Native C++ / Rust Media Pipeline: Hardware Video Decode, JPEG Encoding & HTTP Streaming

- **Targets**: [`OpenCVCamera`](file:///e:/MAJOR_PROJECT/Project/src/camera/capture.py#L25), [`MultiCameraPipeline.step` Phase 2](file:///e:/MAJOR_PROJECT/Project/src/pipeline/multi_camera_pipeline.py#L1667-L1721), [`MappingAPIHandler`](file:///e:/MAJOR_PROJECT/Project/src/multi_camera/ui_server.py#L427-L507)
- **Bottleneck**:
  - Software JPEG resizing + encoding (`cv2.imencode('.jpg')`) takes **3.78 ms per 720p stream** (**15.11 ms** CPU across 4 streams — nearly 39% of the 4-cam `step()` budget!).
  - [`OpenCVCamera._capture_loop`](file:///e:/MAJOR_PROJECT/Project/src/camera/capture.py#L148) performs software H.264/RTSP frame decode via OpenCV threads while sharing full `1280x720x3` (`2.76 MB`) uncompressed NumPy frames across acquisition, inference, recording, and UI encoding threads under Python's GIL.
  - [`ui_server.py`](file:///e:/MAJOR_PROJECT/Project/src/multi_camera/ui_server.py#L273-L507) uses Python stdlib `ThreadingHTTPServer`, dedicating blocking Python OS threads per MJPEG client connection.
- **Recommended Migration**: **Rust (`tokio`/`axum` + `ffmpeg-next`/GStreamer + `nvjpeg`/`libjpeg-turbo`) or C++ media worker** handling RTSP/NVDEC hardware video decode, hardware `nvjpeg` / SIMD JPEG thumbnail compression, and async MJPEG/WebSocket/WebRTC streaming outside the Python GIL, exposing shared-memory frames (or pre-resized inference buffers) to AI.
- **Expected Product Impact**:
  - **FPS & Multi-Camera Scaling**: Reclaims **15+ ms/frame** at 4 cameras and prevents CPU saturation / GIL lockups when monitoring 8–16+ standby cameras concurrently.
  - **CPU & RAM**: Eliminates per-frame software H.264 decode and multiple Python `ndarray` copies across capture, recording, and web UI threads.

---

## 4. Native C++ / Cython Batched Kalman + GPU Optical-Flow GMC in BoT-SORT

- **Targets**: [`CameraMotionCompensator.estimate_motion`](file:///e:/MAJOR_PROJECT/Project/src/tracking/bot_sort.py#L53-L128), [`BoTSORTKalmanFilter`](file:///e:/MAJOR_PROJECT/Project/src/tracking/bot_sort.py#L135-L246), [`SingleBoTTrack`](file:///e:/MAJOR_PROJECT/Project/src/tracking/bot_sort.py#L252-L355)
- **Bottleneck**:
  - Measured at 25 active tracks (720p): `BoTSORTTracker.update()` takes **12.085 ms/frame** (**48.34 ms** across 4 cameras — exceeding a 4-frame GPU YOLO pass!) with `gmc=sparseOptFlow`, vs **3.250 ms/frame** (`13.00 ms` across 4 cams) with `gmc=none`.
  - Pure-Python loops create dozens of tiny NumPy `8x8`/`4x4` arrays (`np.diag`, `np.dot`, `np.linalg.inv`) per track per frame while holding the Python GIL inside `Phase 3` `ThreadPoolExecutor` workers, limiting CPU parallelism across cameras.
- **Recommended Migration**: Port [`BoTSORTKalmanFilter`](file:///e:/MAJOR_PROJECT/Project/src/tracking/bot_sort.py#L135) state predict/update into vectorized batched C++ (`Eigen` + `pybind11`, or Cython/Rust) releasing the GIL, and offload `sparseOptFlow` GMC to CUDA optical flow (for PTZ/moving cameras) while keeping GMC disabled for fixed static CCTV feeds.
- **Expected Product Impact**:
  - **FPS & Latency**: Cuts dense 4-camera tracking overhead from **~48 ms** down to **<3–5 ms** CPU across multiple threads with true multi-core parallelism.

---

## 5. Native Compressed Packet Ring Buffer for Forensic Clip Recording

- **Targets**: [`ForensicClipRecorder`](file:///e:/MAJOR_PROJECT/Project/src/forensic/clip_recorder.py#L69-L299), [`ForensicConfig.max_memory_frames`](file:///e:/MAJOR_PROJECT/Project/src/core/config.py#L171)
- **Bottleneck**:
  - `ForensicClipRecorder` copies and stores raw uncompressed `1280x720x3` `uint8` `np.ndarray` frames (`~2.76 MB/frame`) in a Python `List` up to `max_memory_frames=9000` (worst-case **~24.88 GB RAM** before `_spill_to_disk()` triggers), then re-encodes all frames synchronously via `cv2.VideoWriter` during `stop_and_export()`.
- **Recommended Migration**: **C++ / Rust FFmpeg (`libavcodec` NVENC) compressed GOP packet ring buffer** encoding active frames once (reusing stream/NVENC packets) instead of retaining thousands of decoded BGR `np.ndarray` buffers in Python memory.
- **Expected Product Impact**:
  - **RAM Usage**: Reduces a 10-minute 15 FPS forensic buffer from **~24.88 GB raw RAM to ~50–150 MB** compressed packets (>99% RAM reduction).
  - **Export Latency**: Converts clip export (`stop_and_export`) from multi-second frame-by-frame software re-encoding into near-instant container remuxing.

---

## Summary Priority Order (Best Effort-to-Gain Ratio)

1. **Unified C++ TensorRT / ONNX Runtime (Drop PyTorch from Live Path)** — largest startup (**-3.5–4.0 s**) and RAM/VRAM win (**-400–800 MB RAM**, **-300–500 MB VRAM**).
2. **Native Media Capture + NVENC Forensic Ring + `nvjpeg`/`turbojpeg` Streaming** — eliminates **~15 ms** CPU/step (at 4 cams) + prevents **~24.9 GB** uncompressed ring RAM growth.
3. **Fused C++/CUDA ReID Crop Preprocessing** — eliminates **~10 ms** CPU Python copy/resize overhead per 12-crop batch.
4. **C++ / Cython GIL-Free BoT-SORT Kalman + GPU GMC** — reclaims **~12–48 ms** CPU under crowded multi-camera load.
