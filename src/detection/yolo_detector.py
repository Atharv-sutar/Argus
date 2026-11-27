"""Person detector implementation using YOLO conforming to BaseDetector.

UNUSED:
- register_detector_backend (lines 324-326): Modular detector backend extension hook kept for registering custom BaseDetector plugins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from src.core.interfaces import BaseDetector
from src.core.types import BoundingBox, Detection, DetectionResult
from src.inference.device import (
    HardwareProfile,
    configure_torch_execution,
    detect_hardware_profile,
    resolve_detector_model_for_profile,
)

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """
    Hardware-adaptive YOLO object/person detector.
    Owns model resolution, device/precision configuration, inference invocation,
    confidence filtering, and class postprocessing.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "auto",
        image_size: int = 640,
        batch_size: Union[int, str] = "auto",
        engine_type: str = "auto",
        precision: str = "auto",
        hardware_profile: Optional[HardwareProfile] = None,
    ) -> None:
        self.hardware_profile = hardware_profile or detect_hardware_profile(
            requested_device=device,
            engine_type=engine_type,
            precision=precision,
        )
        self.requested_model_name = model_name
        self.model_name = resolve_detector_model_for_profile(model_name, self.hardware_profile)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes if target_classes is not None else [0]  # 0 is person in COCO
        self.device = self.hardware_profile.device_str
        self.image_size = (
            self.hardware_profile.recommended_detection_imgsz
            if str(image_size).lower() == "auto" or int(image_size) <= 0
            else int(image_size)
        )
        if batch_size is None or str(batch_size).lower() == "auto" or int(batch_size) <= 0:
            self.batch_size = max(1, self.hardware_profile.detection_batch_size)
        else:
            self.batch_size = max(1, int(batch_size))
        self.engine_type = (engine_type or "auto").lower().strip()
        self.precision = (precision or "auto").lower().strip()
        self.active_model_path = str(self.model_name)
        self._use_fp16 = False
        self._supports_quantize_arg = False
        self._model = None
        self._load_model()

    def _select_model_artifact(self, base_model: str) -> str:
        """Select optimal weights/engine artifact matching hardware profile and engine_type."""
        base_path = Path(base_model)
        if self.hardware_profile.device_type == "cpu":
            # Prefer ONNX on CPU when exported alongside weights, else .pt
            onnx_cand = base_path.with_suffix(".onnx")
            if self.engine_type in ("auto", "onnx") and onnx_cand.is_file():
                return str(onnx_cand)
            return str(base_model)

        # CUDA / MPS priority driven by HardwareProfile (avoids slow dynamic ONNX on CUDA unless requested)
        priority = self.hardware_profile.detection_backend_priority
        if self.engine_type == "tensorrt" and not base_path.with_suffix(".engine").is_file():
            try:
                from src.inference.tensorrt_engine import export_yolo_to_tensorrt
                if base_path.is_file():
                    return export_yolo_to_tensorrt(
                        weights_path=str(base_path),
                        imgsz=self.image_size,
                        half=self.hardware_profile.use_fp16,
                        device=self.device,
                        hardware_profile=self.hardware_profile,
                    )
            except Exception as e:
                logger.warning(f"TensorRT export skipped for '{base_model}': {e}")

        for ext_key in priority:
            cand = base_path.with_suffix(f".{ext_key}")
            if cand.is_file():
                return str(cand)

        return str(base_model)

    def _load_model(self) -> None:
        try:
            import os
            os.environ["YOLO_AUTOINSTALL"] = "false"
            import ultralytics.utils as _u_utils
            import ultralytics.utils.checks as _u_checks
            _u_utils.AUTOINSTALL = False
            _u_checks.AUTOINSTALL = False
            from ultralytics import YOLO
            from ultralytics.cfg import DEFAULT_CFG_DICT
            from src.inference.tensorrt_engine import setup_cuda_dll_path

            setup_cuda_dll_path()
            configure_torch_execution(self.hardware_profile)
            self._supports_quantize_arg = "quantize" in DEFAULT_CFG_DICT

            model_to_load = self._select_model_artifact(self.model_name)
            self.active_model_path = str(model_to_load)

            is_cuda = "cuda" in self.device
            is_onnx = str(model_to_load).endswith(".onnx")
            self._use_fp16 = bool(
                is_cuda
                and not is_onnx
                and self.hardware_profile.use_fp16
                and self.precision != "fp32"
            )

            logger.info(
                f"Loading YOLO detector '{model_to_load}' on '{self.device}' "
                f"(tier={self.hardware_profile.tier}, batch={self.batch_size}, fp16={self._use_fp16})..."
            )
            self._model = YOLO(model_to_load, task="detect")
            if hasattr(self._model, "to") and not str(model_to_load).endswith((".engine", ".onnx")):
                self._model.to(self.device)
            logger.info(f"YOLO model '{model_to_load}' loaded successfully.")
        except ImportError as e:
            logger.error("ultralytics package not installed. YOLODetector cannot run.")
            raise RuntimeError("ultralytics package not installed") from e
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.model_name}': {e}")
            raise RuntimeError(f"Failed to load YOLO model '{self.model_name}'") from e

    def _precision_kwargs(self) -> Dict[str, Any]:
        if not self._use_fp16:
            return {}
        if self._supports_quantize_arg:
            return {"quantize": 16}
        return {"half": True}

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0,
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            return DetectionResult(detections=[], frame_id=frame_id, timestamp_ms=timestamp_ms)

        if self._model is None:
            return DetectionResult(detections=[], frame_id=frame_id, timestamp_ms=timestamp_ms)

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
                **self._precision_kwargs(),
            )

            detections: List[Detection] = []
            if results and len(results) > 0:
                boxes_obj = results[0].boxes
                if boxes_obj is not None and len(boxes_obj) > 0:
                    xyxy = boxes_obj.xyxy.cpu().numpy()
                    confs = boxes_obj.conf.cpu().numpy()
                    cls_ids = boxes_obj.cls.cpu().numpy().astype(int)

                    h, w = frame.shape[:2]
                    for box_coords, conf, cid in zip(xyxy, confs, cls_ids):
                        x1, y1, x2, y2 = map(float, box_coords)
                        x1 = max(0.0, min(float(w), x1))
                        y1 = max(0.0, min(float(h), y1))
                        x2 = max(0.0, min(float(w), x2))
                        y2 = max(0.0, min(float(h), y2))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        bbox = BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            confidence=float(conf),
                        )
                        detections.append(
                            Detection(
                                box=bbox,
                                class_id=int(cid),
                                class_name="person" if int(cid) == 0 else f"class_{cid}",
                                confidence=float(conf),
                            )
                        )

            del results
            return DetectionResult(
                detections=detections,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )

        except Exception as e:
            logger.error(f"Error during detection on frame {frame_id}: {e}")
            return DetectionResult(detections=[], frame_id=frame_id, timestamp_ms=timestamp_ms)

    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_ids: Optional[List[int]] = None,
        timestamps_ms: Optional[List[float]] = None,
    ) -> List[DetectionResult]:
        """Executes YOLO object detection on a batch of frames in a single GPU pass."""
        if not frames:
            return []

        if frame_ids is None:
            frame_ids = [0] * len(frames)
        if timestamps_ms is None:
            timestamps_ms = [0.0] * len(frames)

        if self._model is None:
            return [DetectionResult(detections=[], frame_id=fid, timestamp_ms=ts) for fid, ts in zip(frame_ids, timestamps_ms)]

        det_results: List[DetectionResult] = []
        chunk_size = max(1, self.batch_size)
        prec_kwargs = self._precision_kwargs()

        for i in range(0, len(frames), chunk_size):
            chunk_frames = frames[i : i + chunk_size]
            chunk_fids = frame_ids[i : i + chunk_size]
            chunk_ts = timestamps_ms[i : i + chunk_size]

            try:
                results = self._model.predict(
                    source=chunk_frames,
                    batch=len(chunk_frames),
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=self.target_classes,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False,
                    **prec_kwargs,
                )

                for frame, fid, ts, res in zip(chunk_frames, chunk_fids, chunk_ts, results):
                    detections: List[Detection] = []
                    boxes_obj = res.boxes
                    if boxes_obj is not None and len(boxes_obj) > 0:
                        xyxy = boxes_obj.xyxy.cpu().numpy()
                        confs = boxes_obj.conf.cpu().numpy()
                        cls_ids = boxes_obj.cls.cpu().numpy().astype(int)

                        h, w = frame.shape[:2]
                        for box_coords, conf, cid in zip(xyxy, confs, cls_ids):
                            x1, y1, x2, y2 = map(float, box_coords)
                            x1 = max(0.0, min(float(w), x1))
                            y1 = max(0.0, min(float(h), y1))
                            x2 = max(0.0, min(float(w), x2))
                            y2 = max(0.0, min(float(h), y2))
                            if x2 <= x1 or y2 <= y1:
                                continue

                            bbox = BoundingBox(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                confidence=float(conf),
                            )
                            detections.append(
                                Detection(
                                    box=bbox,
                                    class_id=int(cid),
                                    class_name="person" if int(cid) == 0 else f"class_{cid}",
                                    confidence=float(conf),
                                )
                            )

                    det_results.append(
                        DetectionResult(
                            detections=detections,
                            frame_id=fid,
                            timestamp_ms=ts,
                        )
                    )

                del results

            except Exception as e:
                logger.error(f"Error during batched detection for chunk {i // chunk_size}: {e}")
                for fid, ts in zip(chunk_fids, chunk_ts):
                    det_results.append(DetectionResult(detections=[], frame_id=fid, timestamp_ms=ts))

        return det_results


# Modular detector backend registry so new detector families can be plugged in without modifying pipeline orchestration.
DETECTOR_REGISTRY: Dict[str, Callable[..., BaseDetector]] = {
    "yolo": YOLODetector,
}


def register_detector_backend(name: str, builder: Callable[..., BaseDetector]) -> None:
    """Register a custom BaseDetector backend builder under a lowercase identifier."""
    DETECTOR_REGISTRY[name.lower().strip()] = builder


def create_detector(
    config: Any,
    hardware_profile: Optional[HardwareProfile] = None,
    confidence_override: Optional[float] = None,
) -> BaseDetector:
    """Factory creating a configured BaseDetector matched to the detected HardwareProfile."""
    det_cfg = getattr(config, "detection", None)
    inf_cfg = getattr(config, "inference", None)

    backend_key = getattr(det_cfg, "backend", "yolo").lower().strip() if det_cfg else "yolo"
    builder = DETECTOR_REGISTRY.get(backend_key, YOLODetector)

    model_name = getattr(det_cfg, "model_name", "auto") if det_cfg else "auto"
    conf = confidence_override if confidence_override is not None else getattr(det_cfg, "confidence_threshold", 0.4)
    iou = getattr(det_cfg, "iou_threshold", 0.45)
    target_classes = getattr(det_cfg, "target_classes", [0])
    image_size = getattr(det_cfg, "image_size", 640)

    device = getattr(inf_cfg, "device", "auto") if inf_cfg else "auto"
    batch_size = getattr(inf_cfg, "batch_size", "auto") if inf_cfg else "auto"
    engine_type = getattr(inf_cfg, "engine_type", "auto") if inf_cfg else "auto"
    precision = getattr(inf_cfg, "precision", "auto") if inf_cfg else "auto"

    return builder(
        model_name=model_name,
        confidence_threshold=conf,
        iou_threshold=iou,
        target_classes=target_classes,
        device=device,
        image_size=image_size,
        batch_size=batch_size,
        engine_type=engine_type,
        precision=precision,
        hardware_profile=hardware_profile,
    )
