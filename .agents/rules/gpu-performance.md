# GPU and Performance Optimization Rules

## 1. Development Target Baseline
- The target development environment includes limited VRAM hardware (NVIDIA GTX 1650 class with 4GB VRAM).
- Total system VRAM allocation must stay under 3.5GB to prevent CUDA Out-of-Memory (OOM) crashes and Windows Desktop Window Manager paging.

## 2. Shared Inference Engine
- Never load duplicate YOLO or OSNet model instances across separate camera workers.
- Shared models must be executed through the centralized inference management layer (`src/inference/`), which manages provider placement (`CUDAExecutionProvider`, `TensorrtExecutionProvider`, or fallback to `CPUExecutionProvider`).
- Default ReID model for low-VRAM profile is `osnet_x0_25.onnx`. Only switch to `osnet_x1_0.onnx` if VRAM headroom permits.

## 3. Real-Time Latency & Queue Management
- Queues between capture threads and inference workers must be strictly bounded (`maxsize <= 2`).
- If an inference worker falls behind capture rate, obsolete frames must be dropped rather than buffered. High processing freshness is prioritized over stale historical backlog.
- Minimize CPU↔GPU memory copies. Keep frames in numpy arrays until tensor batching is required.
