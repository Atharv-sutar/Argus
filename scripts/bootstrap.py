"""Argus AI Self-Healing System Launcher and Dependency Manager.

Pre-flight runner that automatically:
1. Detects and verifies the Python environment (virtualenv creation / re-spawn).
2. Checks for all required core libraries (OpenCV, PyTorch, Ultralytics, SciPy, FAISS, ONNX Runtime, Flask, etc.).
3. Automatically installs any missing packages via pip.
4. Verifies acceleration models.
5. Launches the main Argus surveillance application seamlessly.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Base directory of the repository
REPO_DIR = Path(__file__).resolve().parent.parent

# Core dependencies: mapping of import name -> pip package specifier
REQUIRED_DEPENDENCIES: Dict[str, str] = {
    "cv2": "opencv-python>=4.8.0",
    "numpy": "numpy>=1.24.0",
    "scipy": "scipy>=1.11.0",
    "torch": "torch>=2.0.0",
    "torchvision": "torchvision>=0.15.0",
    "ultralytics": "ultralytics>=8.0.0",
    "yaml": "pyyaml>=6.0",
    "pydantic": "pydantic>=2.0.0",
    "faiss": "faiss-cpu>=1.7.4",
    "onnxruntime": "onnxruntime>=1.16.0",
    "flask": "flask>=2.3.0",
    "requests": "requests>=2.28.0",
}


def print_banner() -> None:
    print("=" * 68)
    print("        ARGUS AI SURVEILLANCE & TARGET-TRACKING SYSTEM        ")
    print("                 SELF-HEALING SYSTEM LAUNCHER                 ")
    print("=" * 68)


def check_and_switch_venv() -> None:
    """Ensures we are running within the project's virtual environment (.venv)."""
    venv_dir = REPO_DIR / ".venv"
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    # If running directly inside virtual environment, proceed
    is_in_venv = sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV") is not None
    if is_in_venv and Path(sys.executable).resolve() == venv_python.resolve():
        return

    # If .venv exists, re-execute using .venv python
    if venv_python.is_file():
        print(f"[ARGUS] Switching to project virtual environment: {venv_python}")
        cmd = [str(venv_python), str(__file__)] + sys.argv[1:]
        proc = subprocess.Popen(cmd)
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
            sys.exit(0)

    # Otherwise, create .venv automatically
    print(f"[ARGUS] Virtual environment not found at '{venv_dir}'.")
    print("[ARGUS] Creating virtual environment (.venv)...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        print("[ARGUS] Virtual environment created successfully.")
        cmd = [str(venv_python), str(__file__)] + sys.argv[1:]
        proc = subprocess.Popen(cmd)
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
            sys.exit(0)
    except Exception as e:
        print(f"[WARNING] Could not create virtual environment automatically ({e}). Continuing with current Python: {sys.executable}")


def detect_missing_packages() -> List[Tuple[str, str]]:
    """Checks all required dependencies and returns a list of (import_name, pip_package)."""
    missing: List[Tuple[str, str]] = []
    for mod_name, pip_pkg in REQUIRED_DEPENDENCIES.items():
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append((mod_name, pip_pkg))
    return missing


def install_packages(packages: List[str]) -> bool:
    """Installs missing packages using pip."""
    print("\n" + "-" * 68)
    print(f"[ARGUS SELF-HEALING] Installing {len(packages)} missing packages:")
    for pkg in packages:
        print(f"  -> {pkg}")
    print("-" * 68)

    try:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
        
        install_cmd = [sys.executable, "-m", "pip", "install"] + packages
        ret = subprocess.call(install_cmd)
        if ret == 0:
            print("[ARGUS SELF-HEALING] All packages installed successfully!\n")
            return True
        else:
            print(f"[ERROR] Pip installation failed with exit code {ret}.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to execute pip installer: {e}")
        return False


def verify_hardware_acceleration() -> None:
    """Logs detected CPU/GPU hardware capabilities."""
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[HARDWARE] GPU Detected: {dev_name} ({vram_gb:.1f} GB VRAM) | CUDA Active")
        else:
            print("[HARDWARE] GPU Not Detected. Running on CPU with accelerated vectorization.")
    except Exception:
        pass


def launch_application() -> None:
    """Launches the Argus main surveillance application with all passed arguments."""
    print("=" * 68)
    print(">>> LAUNCHING ARGUS SURVEILLANCE OPERATIONS CENTER <<<")
    print("=" * 68 + "\n")

    os.chdir(str(REPO_DIR))
    sys.path.insert(0, str(REPO_DIR))

    # Pass all remaining command line arguments directly to main
    from src.app.main import main
    main()


def main_bootstrap() -> None:
    print_banner()
    check_and_switch_venv()

    missing = detect_missing_packages()
    if missing:
        pip_targets = [pkg for _, pkg in missing]
        success = install_packages(pip_targets)
        if not success:
            print("[CRITICAL] Could not install required dependencies. Please check network/permissions.")
            sys.exit(1)

    verify_hardware_acceleration()
    launch_application()


if __name__ == "__main__":
    main_bootstrap()
