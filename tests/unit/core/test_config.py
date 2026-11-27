"""Unit tests for configuration loading and validation."""

from src.core.config import AppConfig


def test_default_config():
    config = AppConfig()
    assert config.camera.name == "camera_0"
    assert config.detection.model_name == "yolo11n.pt"
    assert config.detection.confidence_threshold == 0.4
    assert config.tracking.track_thresh == 0.4
    assert config.visualization.show_window is True


def test_config_from_dict():
    data = {
        "camera": {"source": "test.mp4", "fps": 25},
        "detection": {"confidence_threshold": 0.5},
        "tracking": {"track_buffer": 45},
    }
    config = AppConfig.from_dict(data)
    assert config.camera.source == "test.mp4"
    assert config.camera.fps == 25
    assert config.detection.confidence_threshold == 0.5
    assert config.tracking.track_buffer == 45


def test_ensure_technologies_skips_pip_when_satisfied(monkeypatch):
    """Verify ensure_technologies does not invoke pip subprocesses when dependencies satisfy min versions."""
    from importlib import metadata
    import subprocess
    from src.core.dependencies import ensure_technologies

    monkeypatch.setattr(metadata, "version", lambda pkg: "99.0.0+cu121")

    def fail_subprocess(*args, **kwargs):
        raise AssertionError(f"subprocess.check_call should not run when versions match: {args}")

    monkeypatch.setattr(subprocess, "check_call", fail_subprocess)
    ensure_technologies(AppConfig())


def test_ensure_technologies_installs_cpu_onnxruntime_when_cpu_requested(monkeypatch):
    """Verify CPU device selection checks/installs onnxruntime instead of onnxruntime-gpu when missing."""
    from importlib import metadata
    import subprocess
    from src.core.dependencies import ensure_technologies

    installed = {"ultralytics": "8.4.0", "torch": "2.5.0", "torchvision": "0.20.0", "scipy": "1.12.0", "faiss-cpu": "1.8.0", "onnx": "1.16.0", "nvidia-ml-py": "12.0.0"}

    def fake_version(pkg):
        if pkg not in installed:
            raise metadata.PackageNotFoundError(pkg)
        return installed[pkg]

    calls = []
    monkeypatch.setattr(metadata, "version", fake_version)
    monkeypatch.setattr(subprocess, "check_call", lambda cmd, **kw: calls.append(cmd) or 0)

    cfg = AppConfig()
    cfg.inference.device = "cpu"
    ensure_technologies(cfg)

    assert any(cmd[-1] == "onnxruntime" for cmd in calls)
    assert not any("onnxruntime-gpu" in cmd[-1] for cmd in calls)


def test_main_cli_forwards_device_synthetic_and_target_id(monkeypatch):
    """Verify default web entrypoint forwards --device, --synthetic, --target-id, and --ui-port."""
    import sys
    import src.app.main as app_main

    captured = {}

    def fake_run_multi(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(app_main, "run_multi_camera_app", fake_run_multi)
    monkeypatch.setattr(sys, "argv", ["argus", "--synthetic", "--device", "cpu", "--ui-port", "8791", "--target-id", "42"])

    app_main.main()
    assert captured["synthetic"] is True
    assert captured["device"] == "cpu"
    assert captured["ui_port"] == 8791
    assert captured["target_id"] == 42
