# GPU and Performance Optimization Rules

## 1. Hardware-Adaptive Baseline
- Never hardcode hardware specs (e.g., 4 GB VRAM limits, fixed batch sizes, or legacy GPU SKUs) in inference, UI, or config defaults.
- Probe the active device at runtime via `detect_hardware_profile()` (`src/inference/device.py`) so compact GPUs keep VRAM well under system ceiling while mid/high-VRAM Tensor-Core GPUs automatically unlock TensorRT/FP16/TF32, larger batch sizes, and higher-capacity models.

## 2. Shared Inference Engine
- Never load duplicate detector or ReID model instances across separate camera workers.
- Shared models must execute through the centralized inference layer (`src/inference/`), dynamically selecting optimal providers (`TensorrtExecutionProvider`, PyTorch CUDA `FP16`/`TF32`, `CUDAExecutionProvider`, or `CPUExecutionProvider`) and memory workspaces for the detected `HardwareProfile`.
- Default ReID resolution flows through the modular ReID registry (`src/reid/extractor.py`) rather than hardcoded legacy names.

## 3. Real-Time Latency & Queue Management
- Queues between capture threads and inference workers must be strictly bounded (`maxsize <= 2`).
- If an inference worker falls behind capture rate, obsolete frames must be dropped rather than buffered. High processing freshness is prioritized over stale historical backlog.
- Minimize CPU↔GPU memory copies. Keep frames in numpy arrays until tensor batching is required.

