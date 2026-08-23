---
trigger: always_on
---

# Project Core Rules

## Purpose

This repository is an AI-powered real-time multi-camera surveillance and person-tracking system.

The system must prioritize:

1. Correctness
2. Modularity
3. Readability
4. Debuggability
5. Real-time performance
6. Low GPU memory usage
7. Low CPU overhead
8. Easy team collaboration
9. Replaceable AI components
10. Minimal unnecessary complexity

This ordering is intentional: maintainability and correctness are prioritized over raw throughput, except where real-time performance falls below the minimum viable threshold defined by the Performance Rules.

## Mandatory Development Principles

### 1. Think Before Coding

Before implementing a non-trivial change:

- Inspect only the relevant files.
- Identify assumptions.
- Identify ambiguities.
- State the implementation plan briefly.
- Identify success criteria.
- Ask for clarification when ambiguity materially affects the design.

Do not silently invent requirements.

### 2. Simplicity First

Use the smallest design that solves the actual problem.

Do not add:

- speculative features
- unnecessary abstractions
- unnecessary configuration
- unnecessary dependencies
- frameworks without a clear need
- single-use abstractions without architectural value

Prefer simple, explicit code.

### 3. Surgical Changes

Modify only what is required for the task.

Do not:

- refactor unrelated code
- reformat unrelated files
- rename unrelated files
- delete unrelated code
- "clean up" unrelated code
- replace working components without justification

Every changed line should have a clear relationship to the task.

### 4. Goal-Driven Execution

Define verifiable success criteria.

For a bug:
- reproduce it
- fix it
- verify the fix

For a feature:
- define expected behavior
- implement it
- test it

For performance work:
- measure before
- change one important variable
- measure after
- keep changes only when they improve the desired metric without unacceptable regressions

---

## Architecture Rules

The project is modular.

Major modules include:

- core
- camera
- detection
- tracking
- target
- reid
- identity
- multi_camera
- database
- pipeline
- inference
- visualization
- performance

Each module must have one clear responsibility.

Modules must communicate through explicit data types and interfaces.

Do not pass loosely structured dictionaries throughout the application when a stable typed data structure is appropriate.

Avoid circular dependencies.

---

## Dependency Direction

Preferred direction:

app
→ pipeline
→ domain modules
→ core interfaces/types

Implementation modules must not leak their implementation details into unrelated modules.

Examples:

GOOD:

pipeline → Tracker interface → ByteTrack implementation

pipeline → ReID interface → DINOv2 implementation

identity → VectorStore interface → FAISS implementation

detection/reid → Inference layer → Shared execution/device management

BAD:

pipeline directly depending on FAISS internals

camera importing YOLO

tracking importing visualization

ReID importing application logic

---

## Core Rules

Purpose:
Provide shared types, contracts, and interfaces used across the system.

Examples:
- `Frame`
- `Detection`
- `DetectionResult`
- `Track`
- `TrackResult`
- `Embedding`
- `Identity`
- `TargetState`
- Camera-graph types
- Interfaces such as `Tracker`, `ReID`, and `VectorStore`

Constraints:
- No detection, tracking, or ReID algorithm implementation.
- No business workflow logic.
- No direct dependency on infrastructure implementations.

---

## Camera Rules

Camera modules only handle:

- connection
- frame acquisition
- timestamps
- camera metadata
- source failures
- resource release

Camera modules must not contain:

- detection logic
- tracking logic
- ReID logic
- identity matching
- visualization logic

---

## Detection Rules

Detection must expose a stable interface.

Input:
Frame

Output:
DetectionResult

The rest of the system must not depend directly on YOLO-specific result objects. Detection owns model-specific preprocessing and postprocessing.

---

## Tracking Rules

Tracking must expose a stable interface.

Input:
Frame + DetectionResult

Output:
TrackResult

ByteTrack is an implementation, not the architecture.

A future tracker must be replaceable without rewriting unrelated modules.

---

## Target Rules

Target management is separate from tracking.

Tracking determines:

"Who is currently being tracked?"

Target management determines:

"Which tracked person has the user selected?"

Target logic must support:

- manual selection
- target locking
- temporary target loss
- target state
- re-association

---

## ReID Rules

ReID is responsible for generating appearance embeddings.

Input:
Person crop / observation

Output:
Embedding

The rest of the project must not depend directly on DINOv2.

DINOv2 must remain replaceable.

Do not run expensive ReID inference unnecessarily.

Prefer target-aware, interval-based, cached, or uncertainty-triggered ReID when appropriate.

---

## Identity Rules

Identity management is separate from ReID.

ReID answers:

"How visually similar are these observations?"

Identity management answers:

"Which known identity does this observation belong to?"

Do not put FAISS-specific logic directly into identity logic.

---

## Multi-Camera Search Strategy

The multi-camera subsystem must use a graph-based dynamic search strategy.

Cameras form a graph where:

- nodes represent cameras
- edges represent possible physical/logical transitions

The system must NOT actively run the full AI pipeline on every connected camera.

When a target is confirmed on camera C:

1. Camera C is the current active camera.
2. Adjacent cameras form the initial search set.
3. Non-adjacent cameras are ignored for expensive AI processing.
4. If the target is not found within a configurable timeout, expand the search radius.
5. Search radius may progress from 1-hop neighbors to 2-hop neighbors, then further if necessary.
6. Stop expansion immediately when the target is found or the configured maximum search radius is reached.
7. Already-searched cameras must not be redundantly searched in the same recovery attempt.

The search manager must maintain:

- current camera
- search radius
- search start time
- search timeout
- maximum radius
- active search cameras
- candidate priorities
- target recovery state

Camera connectivity and camera AI activity are separate concepts.

A camera may remain connected while its expensive AI processing is inactive.

Search parameters must be configuration-driven rather than hardcoded constants (`search.initial_radius`, `search.radius_increment`, `search.per_radius_timeout`, `search.max_radius`, `search.total_recovery_timeout`).

The system must support future camera prioritization using:

- graph distance
- target movement direction
- historical transitions
- expected travel time
- camera reliability
- ReID similarity

The initial implementation should use graph distance and configurable timeout/radius expansion.

### Multi-Camera Module Structure

The multi_camera module is organized as:

- `camera_graph.py` — graph structure: nodes, edges, adjacency queries, distance calculations
- `camera_node.py` — per-camera node: camera ID, metadata, connection state, AI activity state
- `transition.py` — transition data: source camera, destination camera, transition metadata
- `search_manager.py` — search orchestration: expand/contract search radius, manage timeouts, activate/deactivate camera AI processing
- `search_state.py` — search state data: current camera, radius, timeout, active cameras, recovery status
- `camera_priority.py` — priority scoring: rank candidate cameras for search (graph distance initially, extensible to direction/history/ReID)

---

## Database Rules

Vector storage must be accessed through an abstraction such as:

`VectorStore`

The application should not depend directly on FAISS internals.

This allows future replacement of FAISS with another storage solution. Domain modules depend on storage interfaces, not concrete database implementations.

---

## Pipeline Rules

Purpose:
Orchestrate the processing flow and coordinate domain modules.

Primary flow:
`camera → detection → tracking → target → reid/identity → visualization`

For multi-camera recovery:
`active camera → target loss → search manager → candidate cameras → detection/ReID → target recovery → active camera handoff`

Constraints:
- Owns queue/thread boundaries and orchestration.
- May coordinate multiple domain modules.
- Must not implement detection, tracking, ReID, or identity algorithms itself.
- Must not contain camera-graph traversal algorithms that belong to the search component.
- Should remain thin and orchestration-focused.

---

## Inference Rules

Purpose:
Provide a shared model-execution layer for executing AI models efficiently on shared compute/GPU resources.

Responsibilities:
- Model loading and lifecycle
- Device placement (GPU / CPU)
- Model caching
- Batching where useful
- Resource management and shared execution configuration

Constraints:
- Must not become a second `detection` or `reid` module.
- Must not contain identity or target-management logic.
- Model-specific preprocessing, postprocessing, and output interpretation remain in their respective domain modules (`detection`, `reid`).

---

## Visualization Rules

Purpose:
Render the current system state for human operators.

Inputs:
- Frame
- Bounding boxes
- Tracking IDs
- Recognized identity
- Target lock state
- Camera ID
- Search/recovery state
- Confidence information

Output:
- Rendered frame
- Display stream
- Optional operator-facing visualization

Constraints:
- Display-only.
- Must not perform detection, tracking, ReID, identity matching, or camera-graph decisions.
- Must not block the processing pipeline.

---

## Performance Rules

Performance is a first-class requirement.

Measure before optimizing.

Important metrics include:

- FPS
- end-to-end latency
- detection latency
- tracking latency
- ReID latency
- identity search latency
- CPU usage
- GPU utilization
- GPU memory
- RAM usage
- frame drops
- queue sizes

Never optimize blindly.

Do not sacrifice tracking or identity quality merely to increase FPS.

---

## GPU Rules

The development target includes limited VRAM (e.g., 4 GB GTX 1650 class).

Avoid:

- duplicate model instances
- unnecessary CPU↔GPU transfers
- unnecessary image copies
- unnecessary tensor copies
- unnecessary image conversions
- keeping tensors alive longer than necessary
- running expensive inference when it is not needed

Prefer shared model instances where safe.

Do not introduce multiprocessing or multiple GPU processes without measurement.

Do not load every model independently in every module.

---

## Real-Time Rules

Avoid unbounded queues.

Prefer bounded queues.

Do not allow stale frames to accumulate indefinitely.

If latency is more important than processing every frame, prefer dropping obsolete frames over allowing latency to grow without bound.

Keep visualization from unnecessarily blocking the processing pipeline.

---

## Debugging Rules

Every major subsystem must be independently testable.

Examples:

Camera can be tested independently.

Detection can be tested using saved frames.

Tracking can be tested using saved DetectionResults.

ReID can be tested using saved person crops.

Identity can be tested with a mock vector store.

A failure in one subsystem should be isolatable without running the entire application.

---

## Documentation Rules

Every major module must have documentation describing:

- purpose
- input
- output
- dependencies
- callers
- state
- GPU usage
- CPU considerations
- error conditions
- testing method

Keep documentation concise.

Do not duplicate the same information in many places.

---

## AI Context Rules

Do not read the entire repository for every task.

For each task:

1. Identify the relevant module.
2. Identify its interface.
3. Identify direct callers.
4. Identify direct dependencies.
5. Inspect relevant tests.
6. Read only the necessary files.
7. Make the smallest change.
8. Run targeted tests.
9. Review the final diff.

Prefer targeted file references and architecture documentation over full-repository scanning.

Do not repeatedly rediscover architecture that is already documented.

Use skills and workflows for specialized tasks instead of putting every instruction into this rule.

---

## Git Rules

Before major changes:

- inspect git status
- inspect relevant diff
- preserve a recoverable checkpoint

Keep commits focused.

Do not combine unrelated features into one commit.

---

## Completion Requirements

Do not declare a task complete merely because code was written.

Before completion:

- verify behavior
- run relevant tests
- inspect the diff
- check for unintended changes
- check performance impact when relevant
- check GPU impact when relevant
- mention known limitations