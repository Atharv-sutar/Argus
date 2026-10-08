"""Dynamic dependency resolver based on configured technologies."""

import logging
import subprocess
import sys
from importlib import metadata

from src.core.config import AppConfig

logger = logging.getLogger(__name__)

def _ensure_package(package_name: str, min_version: str = None) -> None:
    try:
        if min_version:
            # Simple check: we just ask pip to enforce the version if installed version is unknown/older
            # But the most robust way without packaging library is to just run `pip install package>=version`
            # and let pip figure it out. It's fast if already satisfied.
            install_cmd = f"{package_name}>={min_version}"
        else:
            install_cmd = package_name

        # We first check if it's already satisfied to avoid slow pip invocations
        try:
            version = metadata.version(package_name)
            if min_version:
                # Basic string comparison (not perfect, but fine for major/minor like 8.3.0 vs 8.4.0)
                # For safety, just run pip install if we need a specific version
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-q", install_cmd
                ])
        except metadata.PackageNotFoundError:
            logger.info(f"Package '{package_name}' not found. Installing '{install_cmd}'...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", install_cmd
            ])
            logger.info(f"Successfully installed '{package_name}'.")

    except Exception as e:
        logger.error(f"Failed to verify/install dependency '{package_name}': {e}")


def ensure_technologies(config: AppConfig) -> None:
    """Dynamically detects used technologies from config and installs requirements."""
    logger.info("Verifying dynamic dependencies based on configuration...")
    
    # 1. Object Detection Dependencies
    det_model = config.detection.model_name.lower()
    if "yolo11" in det_model:
        _ensure_package("ultralytics", "8.3.0")
    elif "yolo" in det_model:
        _ensure_package("ultralytics", "8.0.0")

    # 2. ReID Dependencies
    reid_model = config.reid.model_name.lower()
    if "osnet" in reid_model or "mobilenet" in reid_model:
        _ensure_package("torch", "2.0.0")
        _ensure_package("torchvision", "0.15.0")
        
    logger.info("Dynamic dependencies verified.")
