# Code Quality and Architecture Standards

## 1. Type Safety and Data Contracts
- Use explicit type hints for all public functions, methods, and classes.
- Use strongly typed dataclasses (e.g. `DetectionResult`, `TrackResult`, `Target`, `TelemetryData`, `CameraNodeConfig`) instead of loosely structured dictionaries when passing data between modules.
- Preserve backward-compatible serialization methods (`to_dict()`, `from_dict()`) where interfaces interact with JSON APIs.

## 2. Resource Management & Cleanup
- Always wrap hardware resources, camera captures (`cv2.VideoCapture`), video writers, and database connections in deterministic cleanup routines (`try ... finally` or context managers).
- Explicitly release camera handles upon pipeline stop or worker termination to prevent Windows DirectShow / MediaFoundation device lockups.
- Periodically clear CUDA caches (`torch.cuda.empty_cache()`) when switching heavy models or stopping pipelines.

## 3. Worker Isolation & Resilience
- Worker threads processing camera feeds must handle frame-level decode or inference exceptions without crashing the entire surveillance pipeline or terminating the main application.
- Log failures with actionable context (camera ID, frame index, error details).

## 4. Preservation of Existing Code
- Surgical changes only: modify only the lines required to satisfy the task.
- Preserve existing comments, architectural docstrings, and invariants unless specifically requested to update them.
- Do not perform broad unsolicited cosmetic reformatting across unrelated files.

