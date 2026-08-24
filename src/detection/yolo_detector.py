"""Person detector implementation using YOLO conforming to BaseDetector."""

from __future__ import annotations

import logging
from typing import List, Optional
import numpy as np

from src.core.interfaces import BaseDetector
from src.core.types import BoundingBox, Detection, DetectionResult
from src.inference.device import resolve_inference_device

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """
    YOLO-based object/person detector.
    Owns model loading, inference invocation, confidence filtering, and class postprocessing.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "auto",
        image_size: int = 640,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes if target_classes is not None else [0]  # 0 is person in COCO
        self.device = resolve_inference_device(device)
        self.image_size = image_size
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model '{self.model_name}' on device '{self.device}'...")
            self._model = YOLO(self.model_name)
            # Move model to resolved device
            if hasattr(self._model, "to"):
                self._model.to(self.device)
            logger.info("YOLO model loaded successfully.")
        except ImportError:
            logger.warning("ultralytics package not installed. YOLODetector running in fallback mode.")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._model = None

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp_ms: float = 0.0
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            return DetectionResult(detections=[], frame_id=frame_id, timestamp_ms=timestamp_ms)

        if self._model is None:
            # If model is unavailable (e.g. headless unit tests without ultralytics), return empty result
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
                        # Clip coordinates strictly within frame boundaries
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

            return DetectionResult(
                detections=detections,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms
            )

        except Exception as e:
            logger.error(f"Error during detection on frame {frame_id}: {e}")
            return DetectionResult(detections=[], frame_id=frame_id, timestamp_ms=timestamp_ms)
