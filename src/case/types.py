from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class CaseSummary:
    case_id: str
    status: str
    created_at: str
    mode: str
    operator: str
    last_opened_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Case:
    case_id: str
    created_at: str
    status: str
    mode: str
    footage_path: Optional[str]
    operator: str
    notes: str
    last_opened_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Case":
        return cls(
            case_id=data.get("case_id", ""),
            created_at=data.get("created_at", ""),
            status=data.get("status", "active"),
            mode=data.get("mode", "recorded"),
            footage_path=data.get("footage_path"),
            operator=data.get("operator", ""),
            notes=data.get("notes", ""),
            last_opened_at=data.get("last_opened_at", "")
        )
