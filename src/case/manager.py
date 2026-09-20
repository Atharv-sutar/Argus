import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.case.types import Case, CaseSummary

logger = logging.getLogger(__name__)

class CaseManager:
    """Manages creation, loading, saving, and switching of investigation cases."""

    def __init__(self, cases_dir: str = "cases"):
        self.cases_dir = Path(cases_dir)
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.active_case: Optional[Case] = None

    def create_case(self, case_id: str, mode: str, operator: str = "",
                    notes: str = "", footage_path: Optional[str] = None) -> Case:
        """Create a new case folder with all sub-files initialized."""
        case_dir = self.cases_dir / case_id
        if case_dir.exists():
            raise ValueError(f"Case {case_id} already exists.")

        case_dir.mkdir(parents=True)
        (case_dir / "gallery").mkdir(exist_ok=True)
        (case_dir / "gallery" / "thumbnails").mkdir(exist_ok=True)

        now_str = datetime.now(timezone.utc).isoformat()
        new_case = Case(
            case_id=case_id,
            created_at=now_str,
            status="active",
            mode=mode,
            footage_path=footage_path,
            operator=operator,
            notes=notes,
            last_opened_at=now_str
        )

        # Write metadata
        with open(case_dir / "case.json", "w", encoding="utf-8") as f:
            json.dump(new_case.to_dict(), f, indent=4)

        # Snapshot config and topology if they exist
        if Path("configs/camera_graph.json").exists():
            shutil.copy2("configs/camera_graph.json", case_dir / "topology.json")
        if Path("configs/default.yaml").exists():
            shutil.copy2("configs/default.yaml", case_dir / "config_snapshot.yaml")

        # Create empty sqlite dbs
        open(case_dir / "gallery" / "entries.db", "w").close()
        open(case_dir / "audit.db", "w").close()
        open(case_dir / "annotations.db", "w").close()

        logger.info(f"Created case {case_id}")
        return new_case

    def open_case(self, case_id: str) -> Case:
        """Load an existing case."""
        case_dir = self.cases_dir / case_id
        if not case_dir.exists():
            raise ValueError(f"Case {case_id} does not exist.")

        metadata_path = case_dir / "case.json"
        if not metadata_path.exists():
            raise ValueError(f"case.json missing in case {case_id}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        case = Case.from_dict(data)
        case.last_opened_at = datetime.now(timezone.utc).isoformat()
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, indent=4)

        self.active_case = case
        logger.info(f"Opened case {case_id}")
        return case

    def close_case(self) -> None:
        """Close the active case."""
        if self.active_case:
            logger.info(f"Closed case {self.active_case.case_id}")
            self.active_case = None

    def list_cases(self) -> List[CaseSummary]:
        """List all cases with ID, status, date, mode."""
        summaries = []
        for p in self.cases_dir.iterdir():
            if p.is_dir() and (p / "case.json").exists():
                try:
                    with open(p / "case.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        summaries.append(CaseSummary(
                            case_id=data.get("case_id", ""),
                            status=data.get("status", ""),
                            created_at=data.get("created_at", ""),
                            mode=data.get("mode", ""),
                            operator=data.get("operator", ""),
                            last_opened_at=data.get("last_opened_at", "")
                        ))
                except Exception as e:
                    logger.warning(f"Could not load case {p.name}: {e}")
        return sorted(summaries, key=lambda x: x.last_opened_at, reverse=True)

    def delete_case(self, case_id: str) -> None:
        """Permanently delete a case folder."""
        if self.active_case and self.active_case.case_id == case_id:
            self.close_case()
            
        case_dir = self.cases_dir / case_id
        if case_dir.exists():
            shutil.rmtree(case_dir)
            logger.info(f"Deleted case {case_id}")
            
    def get_active_case_dir(self) -> Optional[Path]:
        if self.active_case:
            return self.cases_dir / self.active_case.case_id
        return None
