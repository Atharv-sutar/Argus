"""Multi-camera pipeline coordinator orchestrating topology-aware search and handoff."""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

from src.camera.capture import OpenCVCamera, SyntheticCamera
from src.core.config import AppConfig, SearchConfig
from src.core.interfaces import BaseCamera, BaseDetector, BaseTracker
from src.core.multi_camera_types import (
    CameraNodeConfig,
    CameraStatus,
    SearchProgress,
    SearchState,
    SourceType,
)
from src.core.types import BoundingBox, DetectionResult, Target, TargetState, Track, TrackResult
from src.detection.yolo_detector import YOLODetector
from src.identity.manager import IdentityManager
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.camera_node import CameraNode
from src.multi_camera.search_manager import SearchManager
from src.pipeline.single_camera import SingleCameraPipeline
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.quality import CropQualityEvaluator
from src.target.manager import TargetManager
from src.tracking.byte_tracker import ByteTracker
from src.visualization.annotator import FrameAnnotator

logger = logging.getLogger(__name__)

GLOBAL_TARGET_IDENTITY_ID = "target_0"


class MultiCameraPipeline:
    """
    Orchestrates real-time multi-camera person surveillance and tracking.

    Key Invariants:
    1. A single shared IdentityManager maintains the global target identity ('target_0')
       across all cameras. Local tracker IDs remain strictly local.
    2. Only the active camera (and neighbor cameras during active search) run
       expensive AI inference. Standby cameras do not consume AI compute.
    3. Cross-camera handoff occurs only after multi-frame ReID confirmation
       against the immutable global reference gallery.
    """

    def __init__(
        self,
        graph: CameraGraph,
        config: AppConfig,
        identity_manager: Optional[IdentityManager] = None,
        detector_factory: Optional[Callable[[], BaseDetector]] = None,
        tracker_factory: Optional[Callable[[], BaseTracker]] = None,
        camera_factory: Optional[Callable[[CameraNodeConfig], BaseCamera]] = None,
        shared_detector: Optional[BaseDetector] = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.search_config = config.multi_camera.search

        # Shared ReID & Identity Manager across all cameras
        if identity_manager is not None:
            self.identity_manager = identity_manager
        else:
            reid_extractor = PyTorchReIDExtractor(
                model_name=config.reid.model_name,
                device=config.inference.device,
            )
            quality_eval = CropQualityEvaluator(
                min_width=config.reid.min_crop_width,
                min_height=config.reid.min_crop_height,
                min_sharpness=config.reid.min_sharpness,
            )
            self.identity_manager = IdentityManager(
                reid_extractor=reid_extractor,
                similarity_threshold=config.reid.similarity_threshold,
                reference_threshold=config.reid.reference_threshold,
                upper_threshold=config.reid.upper_threshold,
                min_margin=config.reid.min_margin,
                max_reference_samples=config.reid.reference_samples,
                max_gallery_size=config.reid.adaptive_gallery_size,
                redundancy_threshold=config.reid.redundancy_threshold,
                quality_evaluator=quality_eval,
                w_upper=config.reid.w_upper,
                w_color=config.reid.w_color,
                w_deep=config.reid.w_deep,
                w_lower=config.reid.w_lower,
            )



        # Factories for per-camera pipeline construction
        self._shared_detector = shared_detector
        self._detector_factory = detector_factory or self._default_detector_factory
        self._tracker_factory = tracker_factory or self._default_tracker_factory
        self._camera_factory = camera_factory or self._default_camera_factory

        # Node runtime states and single camera pipelines
        self._nodes: Dict[str, CameraNode] = {}
        self._pipelines: Dict[str, SingleCameraPipeline] = {}
        self._annotators: Dict[str, FrameAnnotator] = {}

        # Search manager
        self.search_manager = SearchManager(self.graph, self.search_config)

        # Active camera tracking
        self._active_camera_id: Optional[str] = None
        self._target_lost_timestamp: float = 0.0
        self._handoff_timestamp: float = 0.0
        self._is_running: bool = False
        self._transit_history: List[Dict[str, Any]] = []


        # Initialize camera nodes from graph
        self._sync_nodes_with_graph()

    def _default_detector_factory(self) -> BaseDetector:
        if self._shared_detector is None:
            self._shared_detector = YOLODetector(
                model_name=self.config.detection.model_name,
                confidence_threshold=self.config.detection.confidence_threshold,
                iou_threshold=self.config.detection.iou_threshold,
                target_classes=self.config.detection.target_classes,
                device=self.config.inference.device,
                image_size=self.config.detection.image_size,
            )
        return self._shared_detector

    def _default_tracker_factory(self) -> BaseTracker:
        return ByteTracker(
            track_thresh=self.config.tracking.track_thresh,
            match_thresh=self.config.tracking.match_thresh,
            track_buffer=self.config.tracking.track_buffer,
            min_box_area=self.config.tracking.min_box_area,
        )

    def _default_camera_factory(self, node_cfg: CameraNodeConfig) -> BaseCamera:
        if node_cfg.source_type == SourceType.SYNTHETIC or str(node_cfg.source).lower() == "synthetic":
            return SyntheticCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
        return OpenCVCamera(
            source=node_cfg.source,
            width=self.config.camera.width,
            height=self.config.camera.height,
            fps=self.config.camera.fps,
        )

    def _sync_nodes_with_graph(self) -> None:
        """Instantiate/sync CameraNode objects for all nodes defined in the graph."""
        graph_ids = set(self.graph.all_camera_ids())

        # Remove deleted nodes
        for cid in list(self._nodes.keys()):
            if cid not in graph_ids:
                if cid in self._pipelines:
                    self._pipelines[cid].stop()
                    del self._pipelines[cid]
                del self._nodes[cid]

        # Add new nodes
        for cid in graph_ids:
            if cid not in self._nodes:
                cfg = self.graph.get_node(cid)
                if cfg:
                    self._nodes[cid] = CameraNode(cfg)
                    self._annotators[cid] = FrameAnnotator(
                        draw_fps=self.config.visualization.draw_fps,
                        draw_boxes=self.config.visualization.draw_boxes,
                        draw_ids=self.config.visualization.draw_ids,
                        box_thickness=self.config.visualization.box_thickness,
                        font_scale=self.config.visualization.font_scale,
                    )

    def _get_or_create_pipeline(self, camera_id: str) -> Optional[SingleCameraPipeline]:
        """Lazy-instantiates SingleCameraPipeline for a given camera node."""
        if camera_id in self._pipelines:
            return self._pipelines[camera_id]

        node = self._nodes.get(camera_id)
        if node is None or not node.config.enabled:
            return None

        camera = self._camera_factory(node.config)
        detector = self._detector_factory()
        tracker = self._tracker_factory()
        target_manager = TargetManager(min_margin=self.config.reid.min_margin)
        annotator = self._annotators.get(camera_id) or FrameAnnotator()

        pipeline = SingleCameraPipeline(
            camera=camera,
            detector=detector,
            tracker=tracker,
            target_manager=target_manager,
            identity_manager=self.identity_manager,
            annotator=annotator,
            camera_id=camera_id,
            reid_interval=self.config.reid.extract_interval_frames,
            min_margin=self.config.reid.min_margin,
            identity_key=GLOBAL_TARGET_IDENTITY_ID,
            reference_window_frames=self.config.reid.reference_window_frames,
        )
        self._pipelines[camera_id] = pipeline
        node.mark_online()
        return pipeline


    @property
    def active_camera_id(self) -> Optional[str]:
        return self._active_camera_id

    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        node = self._nodes.get(camera_id)
        return node.status if node else None

    def get_search_progress(self) -> SearchProgress:
        return self.search_manager.get_progress()

    def select_target_on_camera(
        self, camera_id: str, x: float, y: float
    ) -> Optional[int]:
        """Select a target at (x, y) on the specified camera, making it the active camera."""
        self._sync_nodes_with_graph()
        pipeline = self._get_or_create_pipeline(camera_id)
        if pipeline is None:
            logger.warning(f"Cannot select target: camera '{camera_id}' unavailable")
            return None

        # If pipeline has not processed a frame yet, capture and process one to get tracks
        if pipeline._last_track_result is None and pipeline.camera.is_opened():
            success, frame, ts_ms = pipeline.camera.read()
            if success and frame is not None:
                pipeline.process_frame(frame, ts_ms)

        # Deselect old active camera if different (preserving shared global identity)
        if self._active_camera_id and self._active_camera_id != camera_id:
            old_p = self._pipelines.get(self._active_camera_id)
            if old_p:
                old_p.clear_target(clear_identity=False)
            if self._active_camera_id in self._nodes:
                self._nodes[self._active_camera_id].mark_online()

        self._active_camera_id = camera_id
        self.search_manager.reset()

        selected_id = pipeline.select_target_by_point(x, y)
        if selected_id is not None:
            self._nodes[camera_id].mark_active_target()
            self._transit_history.append({
                "camera_id": camera_id,
                "timestamp": time.time(),
                "event": "TARGET_SELECTED",
                "track_id": selected_id,
            })
            logger.info(
                f"[MULTI-CAM] Target selected on active camera '{camera_id}' | Tracker={selected_id}"
            )
            return selected_id
        return None

    def select_target_by_id(self, camera_id: str, track_id: int) -> bool:
        """Select a target by track ID on the specified camera."""
        self._sync_nodes_with_graph()
        pipeline = self._get_or_create_pipeline(camera_id)
        if pipeline is None:
            return False

        if pipeline._last_track_result is None and pipeline.camera.is_opened():
            success, frame, ts_ms = pipeline.camera.read()
            if success and frame is not None:
                pipeline.process_frame(frame, ts_ms)

        if self._active_camera_id and self._active_camera_id != camera_id:
            old_p = self._pipelines.get(self._active_camera_id)
            if old_p:
                old_p.clear_target(clear_identity=False)
            if self._active_camera_id in self._nodes:
                self._nodes[self._active_camera_id].mark_online()


        self._active_camera_id = camera_id
        self.search_manager.reset()

        ok = pipeline.select_target_by_id(track_id)
        if ok and camera_id in self._nodes:
            self._nodes[camera_id].mark_active_target()
            self._transit_history.append({
                "camera_id": camera_id,
                "timestamp": time.time(),
                "event": "TARGET_SELECTED",
                "track_id": track_id,
            })
            logger.info(
                f"[MULTI-CAM] Target selected by ID on camera '{camera_id}' | Tracker={track_id}"
            )
        return ok

    def clear_target(self) -> None:
        """Clear target across all cameras and reset search."""
        for pipeline in self._pipelines.values():
            pipeline.clear_target()
        for node in self._nodes.values():
            if node.is_online:
                node.mark_online()
        self._active_camera_id = None
        self.search_manager.reset()
        self._transit_history.clear()
        logger.info("[MULTI-CAM] Global target cleared.")

    def step(self) -> Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]]:
        """
        Execute one processing cycle across the active camera and any search cameras.

        Returns a dictionary mapping camera_id -> (annotated_frame, track_result, target).
        Only includes cameras that are actively being processed (active + search).
        Monitoring-only frame reads are handled separately by read_monitoring_frames().
        """
        self._sync_nodes_with_graph()
        results: Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]] = {}
        now = time.time()

        # 1. If no active camera, pick the first available enabled camera as default
        if not self._active_camera_id:
            enabled_ids = [cid for cid, n in self._nodes.items() if n.config.enabled]
            if enabled_ids:
                self._active_camera_id = enabled_ids[0]

        # 2. Process Active Camera
        active_pipeline = self._get_or_create_pipeline(self._active_camera_id) if self._active_camera_id else None
        active_target: Optional[Target] = None

        if active_pipeline and active_pipeline.camera.is_opened():
            active_pipeline.target_evaluation_enabled = True
            success, frame, ts_ms = active_pipeline.camera.read()
            if success and frame is not None:
                self._nodes[self._active_camera_id].last_frame = frame
                det_res, track_res, active_target, ann_frame = active_pipeline.process_frame(frame, ts_ms)
                self._nodes[self._active_camera_id].fps = active_pipeline._fps
                results[self._active_camera_id] = (ann_frame, track_res, active_target)
            else:
                self._nodes[self._active_camera_id].mark_offline()
        elif self._active_camera_id:
            self._nodes[self._active_camera_id].mark_offline()

        # 3. Check Target State on Active Camera
        target_is_active = (
            active_target is not None
            and active_target.state in (TargetState.TRACKING, TargetState.OCCLUDED, TargetState.LOCKED, TargetState.ACQUIRING_REFERENCE)
        )
        target_is_lost = (active_target is not None and active_target.state in (TargetState.LOST, TargetState.UNCERTAIN))

        if target_is_active:
            # Target is healthy on active camera
            if self.search_manager.is_searching:
                self.search_manager.reset()
                self._deactivate_search_cameras()
            self._target_lost_timestamp = 0.0
            if self._active_camera_id in self._nodes:
                self._nodes[self._active_camera_id].mark_active_target()

        elif target_is_lost and self._active_camera_id:
            # Target lost on active camera!
            has_adjacent_cams = len(self.graph.get_neighbors(self._active_camera_id, 1)) > 0
            if not has_adjacent_cams:
                # Single camera or isolated node: hold LOST state on current camera without futile search loop
                if self.search_manager.is_searching:
                    self.search_manager.reset()
                    self._deactivate_search_cameras()
            elif not self.search_manager.is_searching:
                self._target_lost_timestamp = now
                search_cams = self.search_manager.on_target_lost(self._active_camera_id)
                if search_cams:
                    logger.info(
                        f"[MULTI-CAM] Target LOST on camera '{self._active_camera_id}'. "
                        f"Initiating graph search on {len(search_cams)} adjacent cameras."
                    )
                    self._activate_search_cameras([cid for cid, _ in search_cams])
                else:
                    logger.info(
                        f"[MULTI-CAM] Target LOST on camera '{self._active_camera_id}'. "
                        f"No reachable search cameras in graph."
                    )
            else:
                # Search is in progress, check for radius expansion or timeout
                elapsed_s = now - self._target_lost_timestamp
                expansion = self.search_manager.tick(elapsed_s)
                if expansion is not None and len(expansion) > 0:
                    self._activate_search_cameras([cid for cid, _ in expansion])
                elif expansion == []:
                    # Search timed out
                    logger.warning("[MULTI-CAM] Multi-camera search timed out without recovery.")
                    self._deactivate_search_cameras()

        # 4. Process Search Cameras
        progress = self.search_manager.get_progress()
        candidate_recovered_camera: Optional[str] = None
        active_search_set = set(progress.active_cameras)

        # Ensure non-search pipelines do not run target evaluation
        for cid, pipe in self._pipelines.items():
            if cid != self._active_camera_id and cid not in active_search_set:
                pipe.target_evaluation_enabled = False

        if self.search_manager.is_searching:
            for search_cam_id in progress.active_cameras:
                if search_cam_id == self._active_camera_id:
                    continue
                s_pipe = self._get_or_create_pipeline(search_cam_id)
                if not s_pipe or not s_pipe.camera.is_opened():
                    continue

                s_pipe.target_evaluation_enabled = True
                success, s_frame, s_ts_ms = s_pipe.camera.read()
                if not success or s_frame is None:
                    continue

                self._nodes[search_cam_id].last_frame = s_frame
                # Process frame headless to detect/track and test candidates against global ReID
                s_det, s_track, s_target = s_pipe.process_frame_headless(s_frame, s_ts_ms)
                s_ann = s_pipe.annotator.annotate(
                    frame=s_frame,
                    track_result=s_track,
                    target=s_target,
                    fps=s_pipe._fps,
                    camera_id=search_cam_id,
                )
                results[search_cam_id] = (s_ann, s_track, s_target)

                # Check if this search camera found the target
                if s_target and s_target.state == TargetState.TRACKING:
                    logger.info(
                        f"[MULTI-CAM CANDIDATE] Candidate verified on '{search_cam_id}' (Tracker: {s_target.track_id})"
                    )
                    self.search_manager.on_candidate_found(search_cam_id, 0.85)
                    candidate_recovered_camera = search_cam_id
                    break
                else:
                    self.search_manager.on_candidate_lost(search_cam_id)

        # 5. Handle Cross-Camera Handoff if confirmed
        if candidate_recovered_camera:
            self._perform_handoff(candidate_recovered_camera)

        return results

    def read_monitoring_frames(self) -> Dict[str, np.ndarray]:
        """
        Grab a single live frame from each enabled camera WITHOUT running AI processing.

        Used by the MONITORING UI state to show live feeds in the camera grid.
        Cameras are lazily initialized on first call.
        """
        self._sync_nodes_with_graph()
        frames: Dict[str, np.ndarray] = {}

        for cid, node in self._nodes.items():
            if not node.config.enabled:
                continue
            pipe = self._get_or_create_pipeline(cid)
            if pipe and pipe.camera.is_opened():
                success, frame, _ = pipe.camera.read()
                if success and frame is not None:
                    node.last_frame = frame
                    frames[cid] = frame
                    if not node.is_online:
                        node.mark_online()

        return frames

    def read_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Read a single live frame from a specific camera without AI processing."""
        pipe = self._get_or_create_pipeline(camera_id)
        if pipe and pipe.camera.is_opened():
            success, frame, _ = pipe.camera.read()
            if success and frame is not None:
                if camera_id in self._nodes:
                    self._nodes[camera_id].last_frame = frame
                return frame
        return None

    def process_single_camera_frame(self, camera_id: str) -> Optional[Tuple[np.ndarray, TrackResult, Target]]:
        """
        Process a single frame from a specific camera through the full AI pipeline.
        Returns (annotated_frame, track_result, target) or None.
        """
        pipe = self._get_or_create_pipeline(camera_id)
        if not pipe or not pipe.camera.is_opened():
            return None

        success, frame, ts_ms = pipe.camera.read()
        if not success or frame is None:
            return None

        if camera_id in self._nodes:
            self._nodes[camera_id].last_frame = frame

        det_res, track_res, target, ann_frame = pipe.process_frame(frame, ts_ms)
        if camera_id in self._nodes:
            self._nodes[camera_id].fps = pipe._fps
        return ann_frame, track_res, target

    @property
    def handoff_timestamp(self) -> float:
        """Timestamp when the last handoff was performed (0.0 if none)."""
        return self._handoff_timestamp

    def _activate_search_cameras(self, camera_ids: List[str]) -> None:
        """Activate AI processing on designated search camera nodes."""
        for cid in camera_ids:
            if cid in self._nodes:
                self._nodes[cid].activate_ai()
            self._get_or_create_pipeline(cid)

    def _deactivate_search_cameras(self) -> None:
        """Deactivate AI processing on search cameras when search finishes."""
        for cid, node in self._nodes.items():
            if cid != self._active_camera_id and node.status == CameraStatus.SEARCHING:
                node.deactivate_ai()

    def _perform_handoff(self, new_camera_id: str) -> None:
        """Executes active camera handoff to the newly recovered camera."""
        old_camera_id = self._active_camera_id
        logger.info(
            f"[MULTI-CAM HANDOFF] Target recovered on '{new_camera_id}'! "
            f"Handing off active camera: '{old_camera_id}' -> '{new_camera_id}'"
        )

        # Deactivate old camera (preserving global identity in shared IdentityManager)
        if old_camera_id and old_camera_id in self._nodes:
            self._nodes[old_camera_id].mark_online()
            old_p = self._pipelines.get(old_camera_id)
            if old_p:
                old_p.target_evaluation_enabled = False
                old_p.clear_target(clear_identity=False)

        # Update active camera
        self._active_camera_id = new_camera_id
        if new_camera_id in self._nodes:
            self._nodes[new_camera_id].mark_active_target()
        new_p = self._pipelines.get(new_camera_id)
        if new_p:
            new_p.target_evaluation_enabled = True

        # Record handoff timestamp for UI confirmation delay
        self._handoff_timestamp = time.time()
        self._transit_history.append({
            "camera_id": new_camera_id,
            "from_camera": old_camera_id,
            "timestamp": self._handoff_timestamp,
            "event": "HANDOFF",
        })

        # Reset search manager and deactivate search cameras
        self.search_manager.reset()
        self._deactivate_search_cameras()
        self._target_lost_timestamp = 0.0

    @property
    def transit_history(self) -> List[Dict[str, Any]]:
        """List of chronological target transition events across cameras."""
        return list(self._transit_history)

    @property
    def target_state(self) -> str:
        """Current target state across the active surveillance camera."""
        if self._active_camera_id:
            p = self._pipelines.get(self._active_camera_id)
            if p and p.target_manager.target:
                return p.target_manager.target.state.value
        return "UNSELECTED"

    def stream(self) -> Generator[Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]], None, None]:
        """Continuous generator yielding multi-camera step results."""
        self._is_running = True
        try:
            while self._is_running:
                results = self.step()
                yield results
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop all camera pipelines and release resources."""
        self._is_running = False
        for pipeline in self._pipelines.values():
            pipeline.stop()
        self._pipelines.clear()
        logger.info("MultiCameraPipeline stopped.")

