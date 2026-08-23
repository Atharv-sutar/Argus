# ISSUE_FIX.md — Project Core Rules Resolution

## Purpose

This document resolves the issues identified in `ISSUES.md` so that an AI coding agent can update `PROJECT_CORE_RULES.md` consistently before implementing the surveillance system.

The decisions below are specific to the current multi-camera surveillance and target-tracking architecture.

---

## ISSUE-1 — `camera` vs `cameras`

### Decision

Use `camera` as the single per-camera abstraction.

Remove `cameras` from the architecture module list unless the repository already contains a distinct camera-management implementation.

### Rules

- `camera` represents one physical/logical camera source.
- It owns camera connection, frame acquisition, camera state, and camera-specific configuration.
- It must not contain detection, tracking, ReID, identity, or camera-graph search logic.
- Do not introduce `camera_manager` or `camera_registry` merely for naming purposes.
- A camera graph/search component may exist separately when required by multi-camera orchestration.

### Acceptance Criteria

- The architecture module list contains no unexplained `camera`/`cameras` duplication.
- Per-camera responsibilities remain isolated from target-processing logic.

---

## ISSUE-2 — Missing rules for `core`, `pipeline`, `inference`, and `visualization`

Add explicit architecture rules for all four modules.

### `core`

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
- camera-graph types
- interfaces such as `Tracker`, `ReID`, and `VectorStore`

Constraints:

- No detection/tracking/ReID algorithm implementation.
- No business workflow logic.
- No direct dependency on infrastructure implementations.

---

### `pipeline`

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

### `inference`

### Decision

Treat `inference` as a shared model-execution layer.

Purpose:

Provide common execution infrastructure for AI models, especially shared GPU resources.

Responsibilities may include:

- model loading
- model lifecycle
- device selection
- GPU/CPU execution
- model caching
- batching where useful
- inference scheduling/resource management where required
- shared execution configuration

Detection and ReID remain responsible for their own:

- preprocessing
- model-specific input preparation
- postprocessing
- interpretation of model outputs

Constraints:

- `inference` must not become a second `detection` or `reid` module.
- It must not contain identity or target-management logic.
- Model-specific business rules stay in their respective domain modules.

---

### `visualization`

Purpose:

Render the current system state for humans/operators.

Inputs may include:

- frame
- bounding boxes
- tracking IDs
- recognized identity
- target lock state
- camera ID
- search/recovery state
- confidence information

Output:

- rendered frame
- display stream
- optional operator-facing visualization

Constraints:

- Display-only.
- Must not perform detection, tracking, ReID, identity matching, or camera-graph decisions.
- Must not block the processing pipeline.

---

## ISSUE-3 — `inference` responsibility

### Final Decision

Use the shared execution-layer interpretation.

`inference` is infrastructure for executing AI models efficiently.

It is not a generic replacement for `detection` or `reid`.

Architecture:

```text
                 ┌────────────────────┐
                 │     inference      │
                 │                    │
                 │ model execution    │
                 │ device management  │
                 │ model lifecycle    │
                 │ batching/caching   │
                 └─────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                ↓                     ↓
           detection                reid
```

The detection and ReID modules remain the owners of model-specific semantics.

---

## ISSUE-4 — Multi-Camera Search Strategy placement

Move the `Multi-Camera Search Strategy` section into the main architecture rules.

Required conceptual ordering:

`Target Rules → ReID Rules → Identity Rules → Multi-Camera Search Strategy → Database Rules → Performance Rules`

The multi-camera strategy is not an appendix.

It is a core architectural rule because it determines how the system recovers a selected target after the target leaves the currently active camera.

---

# Multi-Camera Search Strategy — Required Architecture

The surveillance system must use a graph-based camera search strategy rather than continuously searching every camera.

## Active Camera

At any time, one camera is considered the current active camera for the selected target.

Normal operation:

```text
Active Camera
      ↓
Detection
      ↓
Tracking
      ↓
Target maintained
```

Only cameras relevant to the current target-search state should be activated for recovery.

---

## Camera Graph

Represent camera relationships as a graph.

Example:

```text
             Camera B
            /        \
      Camera A       Camera C
          |
      Camera D
```

An edge means that the destination camera is a valid next candidate for target recovery.

The graph must not assume that every camera can directly transition to every other camera.

---

## Search Radius

Search cameras by graph distance from the last confirmed target camera.

### Radius 1

Search direct neighbors of the current camera.

```text
Current Camera
      ↓
1-hop neighbors
```

### Radius 2

If the target is not found during the configured radius-1 search period:

```text
Current Camera
      ↓
1-hop neighbors
      ↓
2-hop neighbors
```

### Further Expansion

Additional radii may be searched only when configured and required.

The search must not immediately activate every camera in the deployment.

---

## Search Expansion Behavior

Required behavior:

1. Continue normal tracking in the active camera.
2. Detect target loss or target-confidence failure.
3. Determine the current camera's graph neighbors.
4. Search the configured radius-1 candidate cameras.
5. Attempt target detection and identity verification there.
6. If the target is found, stop expansion immediately.
7. If the target is not found within the configured timeout, expand the search radius.
8. Avoid searching cameras already searched in the current recovery attempt.
9. Stop when the target is recovered, the maximum configured radius is reached, or the recovery attempt expires.

---

## Search Manager Responsibility

A dedicated search/recovery component should own:

- camera graph traversal
- search radius expansion
- candidate-camera selection
- recovery timeout handling
- avoiding duplicate camera searches
- stopping the search when the target is recovered
- reporting recovery status to the pipeline/target system

It must not implement:

- object detection
- person tracking
- ReID embedding generation
- identity matching

Those remain delegated to their respective modules.

---

## Target Recovery State

The architecture should model target recovery explicitly.

Example conceptual states:

```text
TRACKING
    ↓
TARGET_LOST
    ↓
SEARCHING_RADIUS_1
    ↓
TARGET_FOUND
    └────────────→ TRACKING

SEARCHING_RADIUS_1
    ↓ timeout
SEARCHING_RADIUS_2
    ↓
TARGET_FOUND
    └────────────→ TRACKING

SEARCHING_RADIUS_N
    ↓ failure
RECOVERY_FAILED
```

The exact state names may differ in implementation, but the state transitions must preserve this behavior.

---

## Search Configuration

Search values must be configuration-driven.

At minimum provide configuration for:

- initial search radius
- radius increment
- per-radius timeout
- maximum search radius
- overall target-recovery timeout
- maximum number of candidate cameras processed concurrently, if concurrency is implemented

Do not hardcode these values inside graph traversal or tracking logic.

Initial values are implementation/configuration decisions and should be benchmarked against the actual deployment.

---

## ISSUE-5 — `database` scope

### Decision

Keep persistence simple until multiple persistence mechanisms are actually required.

For the current architecture, treat vector storage as a distinct abstraction.

Preferred initial concept:

```text
VectorStore
    ↓
FAISS implementation
```

Do not scatter direct FAISS access through the application.

When relational/session persistence or caching is introduced, create explicit abstractions for those responsibilities rather than making unrelated modules call PostgreSQL or Redis directly.

### Rules

- Domain modules depend on storage interfaces, not storage implementations.
- FAISS-specific code belongs behind `VectorStore`.
- PostgreSQL-specific code must belong behind an appropriate persistence abstraction.
- Redis-specific code must belong behind an appropriate cache/session abstraction.
- Avoid introducing database infrastructure before it is required by the current feature set.

---

## ISSUE-6 — Priority ordering

### Decision

Keep the existing priority ordering, but clarify how real-time performance is treated.

Recommended wording:

> This ordering is intentional: maintainability and correctness are prioritized over raw throughput, except where real-time performance falls below the minimum viable threshold defined by the Performance Rules.

Therefore:

1. Correctness
2. Modularity
3. Readability
4. Debuggability
5. Real-time performance
6. Other secondary concerns

Real-time constraints are still mandatory. The system must meet the minimum performance threshold established for live tracking.

Do not reorder the priority list unless the project maintainer explicitly changes this decision.

---

## ISSUE-7 — Search radius defaults

Search-radius values must be configuration values rather than hardcoded constants.

The initial implementation must expose at least:

```text
search.initial_radius
search.radius_increment
search.per_radius_timeout
search.max_radius
search.total_recovery_timeout
```

Example configuration structure:

```yaml
search:
  initial_radius: 1
  radius_increment: 1
  per_radius_timeout: 2.0
  max_radius: 3
  total_recovery_timeout: 8.0
```

These values are illustrative starting points only.

The implementation must make them easy to benchmark and change without modifying the search algorithm.

---

# Dependency Direction

The architecture should preserve the following dependency direction:

```text
                 ┌──────────────┐
                 │     core     │
                 └──────┬───────┘
                        │
        ┌───────────────┼────────────────┐
        ↓               ↓                ↓
      camera         detection          reid
        │               │                │
        │               └──────┬─────────┘
        │                      ↓
        │                 inference
        │
        └──────────────┐
                       ↓
                    pipeline
                       ↓
             target / identity /
              search manager
                       ↓
                 visualization
```

The exact source-level dependency graph may differ, but higher-level architecture must not bypass the defined abstractions.

---

# Implementation Rules for the AI Coding Agent

Before writing implementation code:

1. Read the updated `PROJECT_CORE_RULES.md`.
2. Inspect the repository to determine which modules already exist.
3. Do not create modules solely because they appear in documentation if their responsibility is unnecessary.
4. Do not silently introduce new infrastructure.
5. Preserve the camera graph as a first-class concept.
6. Keep target recovery separate from local per-camera tracking.
7. Search candidate cameras by graph distance.
8. Stop searching immediately when the target is recovered.
9. Never perform a global all-camera search unless an explicitly configured fallback policy requires it.
10. Keep search radius, timeout, and concurrency values configurable.
11. Keep detection, tracking, ReID, identity, and search responsibilities separate.
12. Prefer interfaces and dependency inversion for infrastructure implementations.
13. Do not optimize prematurely; measure performance before adding complexity.
14. Do not replace working components without evidence that the replacement is necessary.
15. Do not silently change the architecture defined in `PROJECT_CORE_RULES.md`.

---

# Acceptance Criteria

The issue-resolution work is complete when:

- [ ] `camera`/`cameras` duplication is resolved.
- [ ] `core`, `pipeline`, `inference`, and `visualization` have explicit rules.
- [ ] `inference` is defined as a shared execution layer.
- [ ] `Multi-Camera Search Strategy` is part of the main architecture rules.
- [ ] Camera relationships are represented as a graph.
- [ ] Target recovery starts with adjacent cameras.
- [ ] Search expands to further graph hops only after the configured timeout/failure condition.
- [ ] Already-searched cameras are not redundantly searched in the same recovery attempt.
- [ ] Search stops immediately after target recovery.
- [ ] Maximum radius and timeout limits are configurable.
- [ ] Vector storage is accessed through an abstraction.
- [ ] Real-time performance has an explicit minimum viable threshold.
- [ ] The AI coding agent cannot interpret these issues as permission to silently redesign unrelated architecture.

---

# Final Architectural Principle

The surveillance system should prefer a **small, deterministic search space** over brute-force global camera scanning.

The normal case is:

```text
Target tracked
    ↓
Target leaves current camera
    ↓
Search adjacent cameras
    ↓
Target found
    ↓
Switch active camera
    ↓
Continue tracking
```

Only when that fails should the search space expand:

```text
1-hop
  ↓ failure
2-hop
  ↓ failure
3-hop
  ↓
...
```

This preserves the project's real-time behavior while allowing the system to recover from unexpected camera transitions without processing every camera simultaneously.
