import hashlib
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from src.case.types import Case
from src.audit.logger import AuditLogger
from src.playback.annotations import AnnotationStore

logger = logging.getLogger(__name__)

class EvidenceExporter:
    """Generates a forensically complete evidence package from a case."""
    
    def __init__(self, cases_dir: str = "cases", output_dir: str = "exports"):
        self.cases_dir = Path(cases_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_sha256(self, file_path: Path) -> str:
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096 * 1024), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def export(self, case: Case, include_video: bool = True) -> str:
        """
        Package a case into a forensically sound ZIP archive.
        Returns the path to the ZIP archive.
        """
        case_dir = self.cases_dir / case.case_id
        if not case_dir.exists():
            raise ValueError(f"Case directory {case_dir} does not exist")
            
        stage_dir = self.output_dir / f"stage_{case.case_id}"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
        
        # 1. Verify audit log hash chain integrity and export to JSON
        audit_db_path = case_dir / "audit.db"
        audit_chain_valid = False
        if audit_db_path.exists():
            audit = AuditLogger(audit_db_path)
            audit_chain_valid, _ = audit.verify_chain()
            audit.export_as_json(str(stage_dir / "audit_log.json"))
            with open(stage_dir / "audit_chain_verification.txt", "w") as f:
                f.write(f"Audit chain valid: {audit_chain_valid}\\n")
                f.write(f"Verified at: {datetime.now(timezone.utc).isoformat()}\\n")
                
        # 2. Export annotations to JSON
        anno_db_path = case_dir / "annotations.db"
        if anno_db_path.exists():
            anno = AnnotationStore(str(anno_db_path))
            anno.export_as_json(str(stage_dir / "annotations.json"))
            
        # 3. Copy basic metadata files
        if (case_dir / "case.json").exists():
            shutil.copy2(case_dir / "case.json", stage_dir / "case.json")
        if (case_dir / "topology.json").exists():
            shutil.copy2(case_dir / "topology.json", stage_dir / "topology.json")
        if (case_dir / "config_snapshot.yaml").exists():
            shutil.copy2(case_dir / "config_snapshot.yaml", stage_dir / "config_snapshot.yaml")
        if (case_dir / "route.json").exists():
            shutil.copy2(case_dir / "route.json", stage_dir / "route.json")
        if (case_dir / "report.txt").exists():
            shutil.copy2(case_dir / "report.txt", stage_dir / "route_report.txt")
            
        # 4. Copy gallery
        gallery_dir = case_dir / "gallery"
        stage_gallery = stage_dir / "gallery"
        if gallery_dir.exists():
            shutil.copytree(gallery_dir, stage_gallery)
            
        # 5. Include journey video
        if include_video and (case_dir / f"journey_{case.case_id}.mp4").exists():
            shutil.copy2(case_dir / f"journey_{case.case_id}.mp4", stage_dir / "journey_video.mp4")
            
        # 6. Compute SHA-256 hashes of all source footage (if recorded)
        source_hashes = []
        if case.mode == "recorded" and case.footage_path and Path(case.footage_path).exists():
            footage_dir = Path(case.footage_path)
            # Find common video files, e.g. .mp4, .avi, .mkv
            for root, _, files in os.walk(footage_dir):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mkv')):
                        vpath = Path(root) / file
                        h = self._compute_sha256(vpath)
                        # Derive a camera ID from the directory name assuming footage_path/cam_id/video.mp4
                        cam_id = vpath.parent.name
                        source_hashes.append({
                            "camera_id": cam_id,
                            "file": str(vpath),
                            "sha256": h
                        })
            with open(stage_dir / "source_footage_hashes.json", "w") as f:
                json.dump(source_hashes, f, indent=4)
                
        # 7. Generate manifest.json with SHA-256 of every file in the package
        manifest = {
            "package_version": 1,
            "case_id": case.case_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "exported_by": "Argus v2.0.0",
            "audit_chain_valid": audit_chain_valid,
            "files": []
        }
        
        for root, _, files in os.walk(stage_dir):
            for file in files:
                fpath = Path(root) / file
                rel_path = fpath.relative_to(stage_dir)
                h = self._compute_sha256(fpath)
                manifest["files"].append({
                    "path": str(rel_path).replace("\\\\", "/"),
                    "sha256": h
                })
                
        with open(stage_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
            
        # 8. Create ZIP archive
        zip_path = self.output_dir / f"EVIDENCE_{case.case_id}.zip"
        if zip_path.exists():
            zip_path.unlink()
            
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(stage_dir):
                for file in files:
                    fpath = Path(root) / file
                    arcname = fpath.relative_to(stage_dir)
                    zf.write(fpath, arcname)
                    
        # Cleanup stage
        shutil.rmtree(stage_dir)
        
        logger.info(f"Exported evidence package to {zip_path}")
        return str(zip_path)
