---
name: performance-profiler
description: Procedures and benchmarks for profiling FPS, latency, GPU VRAM allocation, and tracking stability to ensure real-time performance on resource-constrained hardware (GTX 1650 4GB).
---

# Performance Profiler Runbook

Use this skill whenever you evaluate, measure, or optimize system throughput, inference latency, or GPU VRAM usage.

## 1. Success Metrics & Thresholds

| Metric | Target (GTX 1650) | Warning Level | Failure Level |
|---|---|---|---|
| End-to-End FPS | >= 24 FPS | < 18 FPS | < 10 FPS |
| Detection Latency | <= 15 ms | > 25 ms | > 40 ms |
| Tracker Latency | <= 5 ms | > 10 ms | > 20 ms |
| ReID Extraction Latency | <= 10 ms | > 18 ms | > 30 ms |
| GPU VRAM Usage | < 3.2 GB | >= 3.5 GB | >= 3.8 GB (OOM Risk) |
| Queue Drop Rate | 0 drops on target camera | < 5% on search cameras | > 20% drops |

## 2. Profiling Procedures

### A. Run Existing Benchmarks
```powershell
.\.venv\Scripts\python -m pytest tests/test_reid_benchmark.py -v -s
.\.venv\Scripts\python -m pytest tests/unit/pipeline/test_pipeline_performance.py -v -s
```

### B. Live Telemetry Inspection
Query the real-time telemetry endpoint while the pipeline is active:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8798/api/status" | ConvertTo-Json -Depth 3
```
Inspect:
- `fps`: Current pipeline throughput.
- `gpu_memory_mb`: Allocated VRAM in MB.
- `candidate_scores`: Real-time cross-camera similarity metrics.

### C. Hardware Profile Assessment
The pipeline includes built-in hardware profiling:
```powershell
.\.venv\Scripts\python -c "from src.pipeline.multi_camera_pipeline import MultiCameraPipeline; print(MultiCameraPipeline.profile_hardware())"
```
Returns recommended ReID model: `osnet_x0_25` (low VRAM) or `osnet_x1_0` (high compute).

## 3. Optimization Rules of Thumb
1. **Never sacrifice tracking accuracy for artificial FPS boosts.**
2. **Batch ReID inferences**: Prefer batch extraction on candidate crops over individual sequential calls.
3. **Target-Aware Processing**: Run expensive ReID only when candidate bounding box meets minimum quality/aspect-ratio criteria (`src/reid/quality.py`).
4. **Measure Before and After**: Always record baseline metrics before applying performance changes.
