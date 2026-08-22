---
trigger: always_on
---

# Architecture Rules

This project must remain modular, readable, testable, and easy to debug.

## Core Principle

Each module must answer:

- What does it do?
- What does it receive?
- What does it return?
- Who calls it?
- What does it depend on?
- Does it use GPU?
- What state does it maintain?
- How can it be tested independently?

## Stable Contracts

Important data should use explicit typed structures:

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

Avoid arbitrary dictionaries for stable domain concepts.

## Replaceable Implementations

Use stable interfaces around components likely to change:

Detector
Tracker
ReIDEngine
VectorStore
CameraSource

Examples:

Detector → YOLODetector

Tracker → ByteTrack

ReIDEngine → DINOv2

VectorStore → FAISSStore

CameraSource → Webcam / VideoFile / RTSP

Changing an implementation should not require rewriting unrelated modules.

## Debuggability

Subsystems must be independently testable.

Tracking should be testable with prerecorded detections.

ReID should be testable with prerecorded person crops.

Identity matching should be testable with a mock vector store.

Camera capture should be testable without loading AI models.

## Performance

Do not duplicate expensive model instances.

Minimize:

- frame copies
- CPU↔GPU transfers
- image conversions
- unnecessary inference
- unnecessary synchronization

Measure before optimizing.

## Dependency Direction

Higher-level orchestration may depend on lower-level modules.

Lower-level modules must not import higher-level application logic.

Avoid circular dependencies.

## Collaboration

Keep module ownership boundaries clear.

A teammate changing one module should not need to modify unrelated modules.

Changes to shared interfaces require:

- documentation update
- affected tests
- affected consumers