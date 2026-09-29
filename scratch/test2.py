from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class DetectionResult:
    detections: List[str] = field(default_factory=list)
    frame_id: int = 0
    timestamp_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

print("TRUTHY" if DetectionResult() else "FALSY")
