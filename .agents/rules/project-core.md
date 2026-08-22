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
- database
- pipeline
- cameras
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

BAD:

pipeline directly depending on FAISS internals

camera importing YOLO

tracking importing visualization

ReID importing application logic

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

The rest of the system must not depend directly on YOLO-specific result objects.

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

## Database Rules

Vector storage must be accessed through an abstraction such as:

VectorStore

The application should not depend directly on FAISS internals.

This allows future replacement of FAISS with another storage solution.

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

The development target includes limited VRAM.

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