# Testing Rules

## 1. Virtual Environment Execution
- Always invoke pytest through the project virtual environment:
  ```powershell
  .\.venv\Scripts\python -m pytest <path_to_test>
  ```
- Do not run plain `pytest` as it may resolve to an incorrect or global Python environment.

## 2. ReID Testing Constraints
- **Strict Prohibition on Synthetic Footage**: Testing the ReID (Person Re-Identification) appearance extraction or cross-camera matching via synthetic footage is prohibited. Synthetic frames produce artificial color spaces and degenerate feature vectors that do not reflect real-world camera lighting or camera sensors.
- Focus ReID testing and validation strictly on real-world crops, recorded multi-camera benchmark datasets (e.g. EPFL campus), or verified feature vector fixtures.

## 3. Subsystem Test Isolation
- Major subsystems must be independently testable without launching the entire live UI:
  - `tests/unit/core/`: Dataclasses, configuration, and types.
  - `tests/unit/camera/`: Capture interfaces and directory video loading.
  - `tests/unit/detection/`: Detector inference and output contract compliance.
  - `tests/unit/tracking/`: BoT-SORT / ByteTrack tracking state updates with saved detections.
  - `tests/unit/multi_camera/`: Camera graph, search manager, and priority scoring.
  - `tests/unit/identity/`: Gallery and vector store re-identification.
  - `tests/unit/target/`: Target state transitions (UNSELECTED, TRACKING, LOST, UNCERTAIN).

## 4. Network and Sandbox Safety in Tests
- Windows sandbox environments restrict raw socket binding. Tests that verify HTTP or streaming server functionality must either mock socket interactions or handle permission errors gracefully.
- Never write tests that require external internet access or external third-party servers.

## 5. Verification Before Task Completion
- Do not declare a task complete merely because code was edited.
- Always run the relevant unit and integration test suites and verify a clean passing result.
