"""Unit tests for hardware capability detection, adaptive scaling, and modular AI registries."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraNodeConfig, SourceType
from src.core.types import BoundingBox, Detection, DetectionResult
from src.detection import DETECTOR_REGISTRY, YOLODetector, create_detector, register_detector_backend
from src.inference.device import (
    build_hardware_profile_from_specs,
    detect_hardware_profile,
    resolve_detector_model_for_profile,
)
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.ui_server import run_ui_server
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline
from src.reid import (
    REID_MODEL_REGISTRY,
    create_reid_extractor,
    get_reid_model_catalog,
    is_reid_model_supported,
)
from src.tracking import TRACKER_REGISTRY, create_tracker


def test_hardware_tiers_adapt_across_cpu_entry_mid_and_high_gpus():
    cpu_p = build_hardware_profile_from_specs(
        device_str="cpu",
        device_name="x86_64 CPU",
        cpu_threads=8,
    )
    assert cpu_p.tier == "cpu"
    assert cpu_p.use_fp16 is False
    assert cpu_p.has_tensor_cores is False
    assert cpu_p.detection_backend_priority == ("onnx", "pt")
    assert cpu_p.reid_extract_interval == 4

    entry_gtx = build_hardware_profile_from_specs(
        device_str="cuda:0",
        device_name="NVIDIA GeForce GTX 1650",
        vram_total_mb=4096.0,
        compute_capability=(7, 5),
        sm_count=14,
        tensorrt_available=False,
    )
    assert entry_gtx.tier == "gpu_entry"
    assert entry_gtx.has_tensor_cores is False
    assert entry_gtx.use_fp16 is True
    assert entry_gtx.use_tf32 is False
    assert entry_gtx.detection_batch_size == 4
    assert entry_gtx.reid_batch_size == 16
    assert entry_gtx.reid_extract_interval == 3
    assert entry_gtx.ort_gpu_mem_limit_bytes == 1024 * 1024 * 1024
    assert entry_gtx.detection_backend_priority[0] in ("engine", "pt")

    mid_rtx = build_hardware_profile_from_specs(
        device_str="cuda:0",
        device_name="NVIDIA GeForce RTX 3070",
        vram_total_mb=8192.0,
        compute_capability=(8, 6),
        sm_count=46,
        tensorrt_available=True,
    )
    assert mid_rtx.tier == "gpu_mid"
    assert mid_rtx.has_tensor_cores is True
    assert mid_rtx.use_fp16 is True
    assert mid_rtx.use_tf32 is True
    assert mid_rtx.detection_batch_size == 8
    assert mid_rtx.reid_batch_size == 32
    assert mid_rtx.reid_extract_interval == 2
    assert mid_rtx.recommended_detector == "yolo11s.pt"
    assert mid_rtx.ort_gpu_mem_limit_bytes > entry_gtx.ort_gpu_mem_limit_bytes

    high_rtx = build_hardware_profile_from_specs(
        device_str="cuda:0",
        device_name="NVIDIA GeForce RTX 4090",
        vram_total_mb=24576.0,
        compute_capability=(8, 9),
        sm_count=128,
        tensorrt_available=True,
    )
    assert high_rtx.tier == "gpu_high"
    assert high_rtx.has_tensor_cores is True
    assert high_rtx.use_fp16 is True
    assert high_rtx.use_tf32 is True
    assert high_rtx.detection_batch_size == 16
    assert high_rtx.reid_batch_size == 64
    assert high_rtx.reid_extract_interval == 1
    assert high_rtx.recommended_detector == "yolo11m.pt"
    assert high_rtx.trt_workspace_bytes == 4 * 1024 * 1024 * 1024


def test_resolve_detector_model_picks_highest_local_tier_and_falls_back(tmp_path):
    high_rtx = build_hardware_profile_from_specs(
        device_str="cuda:0",
        device_name="NVIDIA GeForce RTX 4080",
        vram_total_mb=16384.0,
        compute_capability=(8, 9),
    )
    # Put a dummy yolo11s.pt into tmp_path; expect it to beat yolo11n.pt without needing yolo11m.pt
    s_weights = tmp_path / "yolo11s.pt"
    s_weights.write_bytes(b"stub")
    resolved = resolve_detector_model_for_profile("auto", high_rtx, search_dirs=[tmp_path])
    assert Path(resolved).name == "yolo11s.pt"

    # Explicit model names are preserved verbatim
    assert resolve_detector_model_for_profile("custom.pt", high_rtx, search_dirs=[tmp_path]) == "custom.pt"


def test_modular_registries_and_yaml_auto_resolution():
    cfg = AppConfig.from_yaml("configs/default.yaml")
    assert cfg.detection.model_name == "auto"
    assert cfg.inference.batch_size == "auto"
    assert cfg.reid.model_name == "auto"

    assert "yolo" in DETECTOR_REGISTRY
    assert "botsort" in TRACKER_REGISTRY and "bytetrack" in TRACKER_REGISTRY
    assert "person_reid_youtu" in REID_MODEL_REGISTRY
    assert is_reid_model_supported("person_reid_youtu")
    assert is_reid_model_supported("osnet_x0_25")
    assert is_reid_model_supported("osnet_x1_0")

    cpu_p = detect_hardware_profile("cpu")
    catalog = get_reid_model_catalog(cpu_p)
    ids = {item["id"] for item in catalog}
    assert {"person_reid_youtu", "osnet_x0_25", "osnet_x1_0"}.issubset(ids)

    det = create_detector(cfg, hardware_profile=cpu_p)
    assert isinstance(det, YOLODetector)
    assert det.device == "cpu"
    assert det.batch_size == cpu_p.detection_batch_size

    reid = create_reid_extractor(cfg, hardware_profile=cpu_p)
    assert reid.device_str == "cpu"
    assert reid.batch_size == cpu_p.reid_batch_size

    tracker = create_tracker(cfg, reid_extractor=reid)
    assert tracker is not None

    # Verify custom backend extension point
    sentinel = object()
    register_detector_backend("dummy_test", lambda **kw: sentinel)
    try:
        cfg.detection.backend = "dummy_test"
        assert create_detector(cfg, hardware_profile=cpu_p) is sentinel
    finally:
        DETECTOR_REGISTRY.pop("dummy_test", None)


class _FastDummyDetector:
    active_model_path = "yolo11n.pt"
    batch_size = 4
    _use_fp16 = True

    def detect(self, frame, frame_id=0, timestamp_ms=0.0):
        return DetectionResult(detections=[], frame_id=frame_id, timestamp_ms=timestamp_ms)

    def detect_batch(self, frames, frame_ids, timestamps_ms):
        return [DetectionResult(detections=[], frame_id=fid, timestamp_ms=ts) for fid, ts in zip(frame_ids, timestamps_ms)]


def test_pipeline_set_reid_model_telemetry_and_ui_endpoint():
    cfg = AppConfig.from_yaml("configs/default.yaml")
    cfg.storage.enabled = False
    cfg.recording.enabled = False
    cfg.inference.device = "cpu"

    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_0", name="C0", source="synthetic", source_type=SourceType.SYNTHETIC))

    pipe = MultiCameraPipeline(
        graph=graph,
        config=cfg,
        camera_factory=lambda n: SyntheticCamera(width=320, height=240, fps=30),
        shared_detector=_FastDummyDetector(),
    )
    server = None
    try:
        assert pipe.reid_interval == pipe.hardware_profile.reid_extract_interval
        assert pipe.set_reid_model("person_reid_youtu", auto_rollback=False) is True
        assert pipe.set_reid_model("osnet_x0_25", auto_rollback=False) is True
        assert pipe.set_reid_model("auto", auto_rollback=False) is True
        assert pipe.config.reid.model_name == "auto"

        telem = pipe.get_telemetry()
        assert telem.hardware_info is not None
        assert telem.hardware_info["tier"] == "cpu"

        server = run_ui_server(port=0, pipeline=pipe, block=False)
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/settings/reid", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        option_ids = [o["id"] for o in payload["options"]]
        assert "person_reid_youtu" in option_ids
        assert "hardware" in payload
        assert payload["recommended"] == pipe.profile_hardware()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        pipe.stop()
