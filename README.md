# AI-Powered Surveillance and Tracking System

## 1. Project Overview

This project is a modular, real-time AI-powered surveillance and person-tracking system.

The system is intended to process one or more camera feeds, detect and track people, allow the user to manually select a target, maintain that target's identity, and eventually follow the target across multiple cameras.

The long-term system should support:

```text
Camera
  ↓
Person Detection
  ↓
Multi-Object Tracking
  ↓
Manual Target Selection
  ↓
Target Tracking
  ↓
Person Re-Identification
  ↓
Identity Matching
  ↓
Identity Storage
  ↓
Cross-Camera Association
  ↓
Path / Timeline Tracking
  ↓
Relevant Video / Event Output
```

The project is currently being developed incrementally, beginning with a reliable single-camera pipeline before introducing multi-camera functionality.

---

# 2. Primary Goals

The project must prioritize the following:

1. **Correctness**
2. **Accuracy**
3. **Real-time performance**
4. **Low GPU memory usage**
5. **Low CPU overhead**
6. **Low latency**
7. **Modularity**
8. **Readability**
9. **Easy debugging**
10. **Easy collaboration**
11. **Replaceable AI components**
12. **Minimal unnecessary complexity**

The goal is not to create the largest or most sophisticated architecture.

The goal is to create a system where a developer can quickly understand a module, isolate a bug, replace an implementation, and safely contribute code without understanding the entire repository.

---

# 3. Engineering Philosophy

The project follows four major principles:

### Think Before Coding

Understand the existing architecture and requirements before modifying code.

Identify assumptions and important ambiguities rather than silently inventing behavior.

### Simplicity First

Prefer the simplest solution that satisfies the actual requirement.

Avoid speculative abstractions, unnecessary dependencies, and excessive architecture.

### Surgical Changes

Changes should be small and focused.

Do not modify unrelated code, refactor unrelated modules, or rewrite working components without a clear reason.

### Goal-Driven Execution

Every significant change should have a measurable or verifiable success criterion.

A feature is not complete merely because the code compiles or runs.

---

# 4. AI-Assisted Development

AI agents are an important part of the development workflow.

This repository uses Antigravity's project-level Rules, Skills, and Workflows.

The AI-agent configuration lives under:

```text
.agents/
├── rules/
├── skills/
└── workflows/
```

## Rules

Rules contain persistent project requirements that should influence agent behavior.

Current rules include:

```text
.agents/rules/project-core.md
.agents/rules/architecture.md
```

These define project-wide engineering and architecture constraints.

## Skills

Skills contain specialized knowledge that should only be loaded when relevant.

The project includes the Karpathy-inspired engineering skill:

```text
.agents/skills/karpathy-guidelines/SKILL.md
```

Source/reference:

https://github.com/multica-ai/andrej-karpathy-skills

The skill provides guidance around:

* thinking before coding
* simplicity
* surgical changes
* goal-driven execution

Do not duplicate large portions of this skill into every task prompt.

## Workflows

Repeatable development procedures should eventually be placed under:

```text
.agents/workflows/
```

Examples may include:

```text
new-feature
bug-fix
performance-test
code-review
release-check
```

Workflows should be added only when a process becomes sufficiently repeatable to justify one.

---

# 5. Context and Token Efficiency

AI agents must not repeatedly read the entire repository for ordinary tasks.

The project is intentionally designed to make relevant context easy to locate.

For a task, the preferred workflow is:

```text
Identify task
    ↓
Locate relevant module
    ↓
Read its interface / contract
    ↓
Read direct callers
    ↓
Read direct dependencies
    ↓
Read related tests
    ↓
Implement focused change
    ↓
Run targeted tests
    ↓
Inspect diff
```

Prefer targeted file inspection over repository-wide scanning.

Use architecture documentation and module contracts as a source of truth rather than repeatedly rediscovering the architecture.

Do not load specialized Skills unless they are relevant to the current task.

Do not repeat large project descriptions in every AI prompt.

The goal is to reduce:

* unnecessary context
* unnecessary token usage
* repeated repository scanning
* duplicated explanations
* irrelevant file inspection

while still giving the agent enough information to make correct decisions.

---

# 6. Modularity

Every major subsystem must have one clear responsibility.

A module should make it easy to answer:

```text
What does it do?

What does it receive?

What does it return?

Who calls it?

What does it depend on?

Does it use GPU?

What state does it maintain?

How can it be tested independently?

What happens when it fails?
```

If these questions cannot be answered easily, the module is probably too tightly coupled.

---

# 7. High-Level Architecture

The intended architecture is:

```text
                     APPLICATION
                          │
                          ▼
                       PIPELINE
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
        CAMERA        DETECTION         TRACKING
          │               │                │
          └───────────────┼────────────────┘
                          │
                          ▼
                   TARGET MANAGEMENT
                          │
                          ▼
                         REID
                          │
                          ▼
                  IDENTITY MATCHING
                          │
                          ▼
                       DATABASE
                          │
                          ▼
                 MULTI-CAMERA SYSTEM
                          │
                          ▼
                 VISUALIZATION / OUTPUT
```

The architecture should allow individual components to be replaced without rewriting unrelated components.

---

# 8. Module Responsibilities

## Camera

Responsible only for camera/source handling.

Responsibilities:

* connect to source
* read frames
* provide timestamps
* provide frame metadata
* handle source errors
* release resources

Possible sources:

```text
Webcam
Video file
RTSP
IP camera
```

Camera code must not contain detection, tracking, ReID, identity, or visualization logic.

---

## Detection

Responsible for detecting people.

Input:

```text
Frame
```

Output:

```text
DetectionResult
```

The rest of the system should depend on the detector interface rather than a specific implementation.

An initial implementation may use YOLO.

A future detector should be replaceable without rewriting tracking, ReID, or application logic.

---

## Tracking

Responsible for maintaining object tracks across frames.

Input:

```text
Frame
DetectionResult
```

Output:

```text
TrackResult
```

An initial implementation may use ByteTrack.

The rest of the project should depend on the tracking interface rather than ByteTrack-specific internals.

---

## Target Management

Tracking and target selection are separate responsibilities.

Tracking answers:

> Which people are currently being tracked?

Target management answers:

> Which tracked person has the user selected?

Target management must support:

* manual selection
* target locking
* target state
* temporary target loss
* target re-association

---

## Re-Identification

ReID generates appearance embeddings for person observations.

Input:

```text
Person crop / target observation
```

Output:

```text
Embedding
```

An initial implementation may use DINOv2 or another suitable embedding model.

The rest of the application must not depend directly on DINOv2.

ReID should be replaceable without rewriting identity management.

---

## Identity

Identity management determines whether an observation belongs to a known identity.

ReID answers:

> How visually similar are these observations?

Identity management answers:

> Which known identity does this observation belong to?

These responsibilities must remain separate.

---

## Database / Vector Storage

Vector storage is responsible for storing and searching embeddings.

The identity subsystem should depend on a storage interface rather than directly on FAISS.

An initial implementation may use FAISS.

A future storage implementation should be replaceable with minimal changes.

---

## Multi-Camera System

The architecture must eventually support relationships between cameras.

Example:

```text
Camera A ─────► Camera B
    │               │
    └──────► Camera C
```

The camera graph describes camera relationships.

The transition subsystem handles the logic used when a target leaves one camera and may appear in another.

Camera topology must remain separate from camera-source implementation.

---

## Visualization

Visualization should display information produced by the processing pipeline.

It must not contain detection, tracking, ReID, or identity algorithms.

Visualization should not unnecessarily block the real-time processing pipeline.

---

## Performance

Performance monitoring should measure the system rather than guess about bottlenecks.

Important metrics include:

```text
FPS
End-to-end latency
Detection latency
Tracking latency
ReID latency
Identity search latency
CPU utilization
GPU utilization
GPU memory
RAM usage
Frame drops
Queue sizes
```

---

# 9. Data Contracts

Important concepts should use explicit, well-defined data types.

Examples:

```text
Frame
Detection
DetectionResult
Track
TrackResult
Embedding
Identity
Target
CameraInfo
CameraTransition
```

Avoid passing loosely structured dictionaries throughout the project when a stable domain type is appropriate.

Each major interface should clearly document:

```text
INPUT
OUTPUT
ERROR CONDITIONS
STATE
```

This makes the system easier to debug and prevents accidental coupling between modules.

---

# 10. Dependency Direction

Dependencies should remain predictable.

Preferred:

```text
Application
    ↓
Pipeline
    ↓
Domain Modules
    ↓
Core Types / Interfaces
```

Examples of good dependency relationships:

```text
Pipeline
  ↓
Tracker interface
  ↓
ByteTrack implementation
```

```text
Pipeline
  ↓
ReID interface
  ↓
DINOv2 implementation
```

```text
Identity Manager
  ↓
VectorStore interface
  ↓
FAISS implementation
```

Bad examples:

```text
Camera → YOLO
Tracking → Visualization
Pipeline → FAISS internals
ReID → Application logic
```

Lower-level modules must not depend on higher-level application logic.

Circular dependencies must be avoided.

---

# 11. Performance and GPU Efficiency

The project is intended to run on hardware with limited GPU memory.

The development environment may include GPUs such as the NVIDIA GTX 1650 4 GB.

GPU resources must therefore be treated as constrained resources.

Avoid:

* duplicate model instances
* unnecessary CPU ↔ GPU transfers
* unnecessary tensor copies
* unnecessary image copies
* unnecessary image-format conversions
* repeated model initialization
* unnecessary synchronization
* retaining tensors longer than necessary

Prefer:

* shared model instances where safe
* cached results
* bounded queues
* target-aware inference
* configurable inference intervals
* skipping unnecessary expensive operations
* measurement-driven optimization

Do not add GPU processes or duplicate inference workers unless benchmarking demonstrates that the benefit justifies the additional memory usage.

---

# 12. Real-Time Pipeline Optimization

The pipeline should favor low latency and stable throughput.

Important principles:

* avoid unbounded queues
* avoid processing stale frames
* allow obsolete frames to be dropped when appropriate
* keep capture responsive
* avoid blocking visualization
* avoid unnecessary synchronization
* avoid expensive inference when it is unnecessary

Detection does not necessarily need to execute on every frame.

Tracking may operate between detection frames when appropriate.

ReID should not automatically run for every detected person on every frame.

Potential strategies include:

```text
Target-only ReID
ReID interval
Confidence-based ReID
Motion-based gating
Appearance caching
Transition-triggered ReID
Tracker-uncertainty-triggered ReID
```

Any optimization must be measured against accuracy and tracking quality.

---

# 13. Accuracy vs Performance

Higher FPS is not automatically better.

Every important optimization should consider:

```text
Accuracy
Tracking stability
Target persistence
ReID accuracy
Cross-camera matching
FPS
Latency
GPU memory
CPU usage
```

When a performance optimization introduces an accuracy trade-off, document the trade-off and measure it.

---

# 14. Debugging Strategy

The system must be designed so individual stages can be isolated.

Examples:

### Camera

Test camera capture without loading AI models.

### Detection

Test detection using saved frames.

### Tracking

Test tracking using saved detections.

### ReID

Test ReID using saved person crops.

### Identity

Test identity matching using mocked vector storage.

### Pipeline

Test the complete flow using controlled test inputs.

This allows a developer to determine whether a problem originates from:

```text
Camera
Detection
Tracking
Target management
ReID
Identity matching
Storage
Visualization
```

without debugging the entire system at once.

---

# 15. Testing Structure

Tests should mirror the source architecture.

```text
tests/
├── unit/
│   ├── camera/
│   ├── detection/
│   ├── tracking/
│   ├── target/
│   ├── reid/
│   └── identity/
│
├── integration/
│
└── performance/
```

Unit tests should isolate individual modules.

Integration tests should verify interactions between modules.

Performance tests should measure actual performance.

---

# 16. Documentation Structure

The repository should maintain concise technical documentation.

Recommended:

```text
docs/
├── architecture/
│   ├── system_overview.md
│   ├── data_flow.md
│   └── dependency_graph.md
│
├── modules/
│   ├── camera.md
│   ├── detection.md
│   ├── tracking.md
│   ├── target.md
│   ├── reid.md
│   ├── identity.md
│   └── database.md
│
└── development/
    ├── debugging.md
    ├── performance.md
    └── testing.md
```

Documentation should be factual and concise.

Avoid duplicating the same information across several documents.

---

# 17. Team Collaboration

The architecture should allow multiple developers to work independently.

Typical ownership may eventually look like:

```text
Developer A
    camera/

Developer B
    detection/
    tracking/

Developer C
    reid/
    identity/
    database/

Developer D
    visualization/

System / Integration
    app/
    pipeline/
    core/
```

These boundaries are guidelines rather than permanent ownership assignments.

The important requirement is that contributors work through stable interfaces.

Changing a shared interface requires updating:

* documentation
* affected implementations
* affected callers
* tests

---

# 18. Git Workflow

Use Git throughout development.

Before major changes:

```bash
git status
git diff
```

Prefer focused branches such as:

```text
feature/camera
feature/tracking
feature/reid
feature/identity
feature/multi-camera
feature/performance
```

Keep commits focused.

Good:

```text
Add target selection state manager
```

Bad:

```text
fixed everything
```

Always review the diff before committing significant changes.

---

# 19. Incremental Development Strategy

The complete system should not be implemented at once.

Preferred development order:

```text
1. Camera capture
2. Person detection
3. Single-camera tracking
4. Manual target selection
5. Target persistence
6. ReID
7. Identity storage
8. Identity matching
9. Multi-camera support
10. Camera graph
11. Cross-camera identity association
12. Path/timeline tracking
13. Video/event stitching
14. Backend/API
15. Production optimization
```

Each stage should become stable before the next major stage is added.

---

# 20. Initial Vertical Slice

The first working system should be deliberately small:

```text
Camera
   ↓
Frame
   ↓
Person Detection
   ↓
Tracking
   ↓
Visualization
```

The objective is to establish:

* correct frame handling
* stable detection
* stable tracking
* acceptable performance
* clean module boundaries
* working tests
* useful performance measurements

Only after this is stable should the project add the more expensive ReID and identity pipeline.

---

# 21. Repository Organization

The intended project structure is:

```text
Project/
│
├── README.md
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── .agents/
│   ├── rules/
│   ├── skills/
│   └── workflows/
│
├── configs/
│
├── docs/
│
├── src/
│   ├── app/
│   ├── core/
│   ├── camera/
│   ├── detection/
│   ├── tracking/
│   ├── target/
│   ├── reid/
│   ├── identity/
│   ├── database/
│   ├── pipeline/
│   ├── cameras/
│   ├── inference/
│   ├── visualization/
│   ├── performance/
│   └── utils/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── scripts/
├── models/
├── data/
└── logs/
```

This is the target architecture, not a requirement to create every file immediately.

Do not create empty modules just to fill the directory tree.

Create components when they are actually required.

---

# 22. Requirements for Every New Module

When adding a module, document:

```text
Module name
Purpose
Input
Output
Dependencies
Called by
State
CPU usage
GPU usage
Error conditions
Testing strategy
```

A module should be understandable without reading the entire repository.

---

# 23. Adding a New Implementation

A developer should be able to add:

* a new detector
* a new tracker
* a new ReID model
* a new camera source
* a new vector store

without rewriting unrelated modules.

For example:

```text
Detector interface
    ├── YOLODetector
    └── FutureDetector
```

```text
Tracker interface
    ├── ByteTrack
    └── FutureTracker
```

```text
ReID interface
    ├── DINOv2
    └── FutureReIDModel
```

The interfaces should remain stable while implementations evolve.

---

# 24. AI Agent Working Style

For normal development tasks, AI agents should follow this process:

```text
Understand
    ↓
Inspect relevant code
    ↓
Plan
    ↓
Implement
    ↓
Test
    ↓
Measure when relevant
    ↓
Review diff
    ↓
Report
```

The agent should avoid reading unrelated parts of the repository.

For small tasks, the process should remain lightweight.

For architectural or high-risk changes, the agent should provide a plan before implementation.

---

# 25. Definition of a Good System

The architecture is successful when:

* A developer can understand a module quickly.
* A bug can be isolated to a subsystem quickly.
* Individual modules can be tested independently.
* A detector can be replaced without rewriting tracking.
* A tracker can be replaced without rewriting ReID.
* ReID can be replaced without rewriting identity management.
* FAISS can be replaced without rewriting identity logic.
* A new camera source can be added without rewriting the AI pipeline.
* Multi-camera support can be added without destroying the single-camera architecture.
* GPU usage is measurable.
* Performance bottlenecks are measurable.
* AI agents can work without scanning the entire repository for every task.
* Changes remain small and reviewable.
* Multiple teammates can work on separate components safely.
* The system remains readable as it grows.

---

# 26. Current Project Status

The repository is currently being established from an empty project.

Current foundation:

```text
.agents/
├── rules/
│   ├── project-core.md
│   └── architecture.md
│
├── skills/
│   └── karpathy-guidelines/
│       └── SKILL.md
│
└── workflows/

README.md
```

The architecture should be reviewed before substantial application code is generated.

The first source-code milestone is the single-camera detection/tracking vertical slice.

---

# 27. Project Principle

The project should remain:

```text
Simple
Explicit
Modular
Readable
Testable
Measurable
Debuggable
Replaceable
Performance-conscious
Collaboration-friendly
```

Avoid:

```text
Over-engineering
Hidden dependencies
Large monolithic files
Unnecessary abstractions
Duplicated logic
Unmeasured optimization
Unnecessary GPU work
Unbounded queues
Repository-wide context loading
Unrelated changes
```

The ultimate goal is a system where both humans and AI agents can work efficiently without needing to understand everything at once.
