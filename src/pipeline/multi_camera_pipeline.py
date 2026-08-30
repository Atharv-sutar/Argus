"""Unified surveillance pipeline coordinator with target-only gallery and graph-aware search."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import cv2
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
from src.core.types import BoundingBox, DetectionResult, Embedding, Target, TargetState, Track, TrackResult
from src.detection.yolo_detector import YOLODetector
from src.multi_camera.camera_graph import CameraGraph
from src.multi_camera.camera_node import CameraNode
from src.multi_camera.search_manager import SearchManager
from src.pipeline.camera_worker import CameraWorker
from src.reid.extractor import PyTorchReIDExtractor
from src.reid.gallery import TargetGallery
from src.reid.quality import ReIDCropQuality
from src.target.manager import TargetManager
from src.tracking.byte_tracker import ByteTracker
from src.visualization.annotator import FrameAnnotator

logger = logging.getLogger(__name__)


class MultiCameraPipeline:
    """
    Unified multi-camera surveillance pipeline.

    Key Invariants:
    1. Target-Only Multi-Image Gallery: Holds up to N appearance samples for the single
       actively tracked target. Manual entries are protected from eviction.
    2. Exact Vectorized Matching: All candidate matching is computed in-memory via max-similarity
       matrix multiplication against the target gallery.
    3. Active-Set Inference Scaling: Only the active camera (and adjacent neighbors during
       target loss) run detection and ReID. Standby cameras do not consume AI compute.
    4. Single & Multi-Camera Unification: A single-camera system is treated as a 1-node graph (0 edges),
       sharing identical tracking, gallery management, and UI logic.
    """

    def __init__(
        self,
        graph: CameraGraph,
        config: AppConfig,
        target_manager: Optional[TargetManager] = None,
        reid_extractor: Optional[PyTorchReIDExtractor] = None,
        detector_factory: Optional[Callable[[], BaseDetector]] = None,
        tracker_factory: Optional[Callable[[], BaseTracker]] = None,
        camera_factory: Optional[Callable[[CameraNodeConfig], BaseCamera]] = None,
        shared_detector: Optional[BaseDetector] = None,
        identity_manager: Optional[Any] = None,
    ) -> None:
        self.graph = graph
        self.config = config
        self.search_config = config.multi_camera.search

        # 1. ReID Extractor
        if reid_extractor is not None:
            self.reid_extractor = reid_extractor
        elif identity_manager is not None and hasattr(identity_manager, "reid_extractor"):
            self.reid_extractor = identity_manager.reid_extractor
        else:
            self.reid_extractor = PyTorchReIDExtractor(
                model_name=config.reid.model_name,
                device=config.inference.device,
            )

        # 2. Target Gallery & Target Manager
        if target_manager is not None:
            self.target_manager = target_manager
        else:
            quality_eval = ReIDCropQuality(
                min_width=config.reid.min_crop_width,
                min_height=config.reid.min_crop_height,
                min_sharpness=config.reid.min_sharpness,
            )
            gallery = TargetGallery(
                reid_extractor=self.reid_extractor,
                quality_evaluator=quality_eval,
                max_size=config.reid.max_gallery_size,
                match_threshold=config.reid.match_threshold,
                auto_add_threshold=config.reid.auto_add_threshold,
                auto_add_min_consecutive=config.reid.auto_add_min_consecutive,
                diversity_threshold=config.reid.diversity_threshold,
            )
            self.target_manager = TargetManager(
                gallery=gallery,
                min_margin=config.reid.min_margin,
            )

        # 3. Factories
        self._shared_detector = shared_detector
        self._detector_factory = detector_factory or self._default_detector_factory
        self._tracker_factory = tracker_factory or self._default_tracker_factory
        self._camera_factory = camera_factory or self._default_camera_factory

        # 4. Runtime state & Camera Workers
        self._nodes: Dict[str, CameraNode] = {}
        self._workers: Dict[str, CameraWorker] = {}
        self._annotators: Dict[str, FrameAnnotator] = {}

        # 5. Search manager
        self.search_manager = SearchManager(self.graph, self.search_config)

        # 6. Active tracking state
        self._active_camera_id: Optional[str] = None
        self._target_lost_timestamp: float = 0.0
        self._handoff_timestamp: float = 0.0
        self._is_running: bool = True
        self._transit_history: List[Dict[str, Any]] = []
        self._frame_count: int = 0
        self.reid_interval: int = config.reid.extract_interval_frames
        self._frame_lock: threading.Lock = threading.Lock()
        self._latest_jpegs: Dict[str, bytes] = {}
        self._last_candidate_scores: Dict[int, float] = {}
        self._switch_consensus: Dict[int, int] = {}
        self._current_track_misses: int = 0

        # Sync nodes from graph
        self._sync_nodes_with_graph()

    @property
    def is_running(self) -> bool:
        """Returns True if the pipeline is active and not stopped."""
        return self._is_running

    @property
    def gallery(self) -> TargetGallery:
        """The single target appearance gallery."""
        return self.target_manager.gallery

    @property
    def last_candidate_scores(self) -> Dict[int, float]:
        """Real-time dictionary of track_id -> similarity score against the target gallery."""
        return dict(self._last_candidate_scores)

    @property
    def _pipelines(self) -> Dict[str, CameraWorker]:
        """Backward-compatible alias for _workers."""
        return self._workers

    @property
    def active_camera_id(self) -> Optional[str]:
        return self._active_camera_id

    @property
    def handoff_timestamp(self) -> float:
        return self._handoff_timestamp

    @property
    def target_state(self) -> str:
        """Current target state across the active camera."""
        if self.target_manager.target:
            return self.target_manager.target.state.value
        return "UNSELECTED"

    @property
    def transit_history(self) -> List[Dict[str, Any]]:
        return list(self._transit_history)

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
        """Instantiate/sync CameraNode and CameraWorker objects for all nodes in the graph."""
        graph_ids = set(self.graph.all_camera_ids())

        # Remove deleted nodes and cleanup caches
        for cid in list(self._nodes.keys()):
            if cid not in graph_ids:
                if cid in self._workers:
                    try:
                        self._workers[cid].stop()
                    except Exception:
                        pass
                    del self._workers[cid]
                del self._nodes[cid]
                self._annotators.pop(cid, None)
                with self._frame_lock:
                    self._latest_jpegs.pop(cid, None)

        # Sync existing and new nodes
        for cid in graph_ids:
            cfg = self.graph.get_node(cid)
            if not cfg:
                continue

            if cid in self._nodes:
                old_cfg = self._nodes[cid].config
                if str(old_cfg.source) != str(cfg.source) or old_cfg.source_type != cfg.source_type:
                    if cid in self._workers:
                        try:
                            self._workers[cid].stop()
                        except Exception:
                            pass
                        del self._workers[cid]
                    with self._frame_lock:
                        self._latest_jpegs.pop(cid, None)
                self._nodes[cid].config = cfg
            else:
                self._nodes[cid] = CameraNode(cfg)
                self._annotators[cid] = FrameAnnotator(
                    draw_fps=self.config.visualization.draw_fps,
                    draw_boxes=self.config.visualization.draw_boxes,
                    draw_ids=self.config.visualization.draw_ids,
                    box_thickness=self.config.visualization.box_thickness,
                    font_scale=self.config.visualization.font_scale,
                )

    def _get_or_create_worker(self, camera_id: str) -> Optional[CameraWorker]:
        """Lazy-instantiates CameraWorker for a given camera node."""
        if camera_id in self._workers:
            return self._workers[camera_id]

        node = self._nodes.get(camera_id)
        if node is None or not node.config.enabled:
            return None

        camera = self._camera_factory(node.config)
        detector = self._detector_factory()
        tracker = self._tracker_factory()
        annotator = self._annotators.get(camera_id) or FrameAnnotator()

        worker = CameraWorker(
            camera=camera,
            detector=detector,
            tracker=tracker,
            annotator=annotator,
            camera_id=camera_id,
        )
        self._workers[camera_id] = worker
        node.mark_online()
        return worker

    def _get_or_create_pipeline(self, camera_id: str) -> Optional[CameraWorker]:
        """Backward-compatible alias for _get_or_create_worker."""
        return self._get_or_create_worker(camera_id)

    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        node = self._nodes.get(camera_id)
        return node.status if node else None

    def get_search_progress(self) -> SearchProgress:
        return self.search_manager.get_progress()

    def select_target_on_camera(
        self, camera_id: str, x: float, y: float
    ) -> Optional[int]:
        """
        Manually select target at pixel coordinates (x, y) on the specified camera.
        Resets gallery and seeds it with the newly clicked person crop.
        """
        self._sync_nodes_with_graph()
        self.set_active_camera(camera_id)
        worker = self._get_or_create_worker(camera_id)
        if worker is None:
            logger.warning(f"Cannot select target: camera '{camera_id}' unavailable")
            return None

        # Ensure we have a fresh processed frame with detections and tracks
        frame_to_process = None
        if worker.camera.is_opened():
            success, frame, ts_ms = worker.read_frame()
            if success and frame is not None:
                frame_to_process = frame
            elif worker._last_frame is not None:
                frame_to_process = worker._last_frame

        if frame_to_process is not None:
            worker.process_frame(frame_to_process, time.time() * 1000.0)

        if worker._last_track_result is None or worker._last_frame is None:
            return None

        selected_id = self.target_manager.select_by_point(
            x=x,
            y=y,
            track_result=worker._last_track_result,
            frame=worker._last_frame,
            camera_id=camera_id,
        )

        if selected_id is not None:
            self._active_camera_id = camera_id
            self.search_manager.reset()
            if camera_id in self._nodes:
                self._nodes[camera_id].mark_active_target()
            self._transit_history.append({
                "camera_id": camera_id,
                "timestamp": time.time(),
                "event": "TARGET_SELECTED",
                "track_id": selected_id,
            })
            logger.info(
                f"[MULTI-CAM] Target selected on camera '{camera_id}' | Tracker={selected_id} | Gallery seeded."
            )
            return selected_id
        return None


    def select_target_by_id(self, camera_id: str, track_id: int) -> bool:
        """
        Manually select target by track ID on the specified camera.
        Resets gallery and seeds it with the newly chosen person crop.
        """
        self._sync_nodes_with_graph()
        worker = self._get_or_create_worker(camera_id)
        if worker is None:
            return False

        if worker._last_track_result is None and worker.camera.is_opened():
            success, frame, ts_ms = worker.read_frame()
            if success and frame is not None:
                worker.process_frame(frame, ts_ms)

        if worker._last_track_result is None or worker._last_frame is None:
            return False

        ok = self.target_manager.select_by_track_id(
            track_id=track_id,
            track_result=worker._last_track_result,
            frame=worker._last_frame,
            camera_id=camera_id,
        )
        if ok:
            self._active_camera_id = camera_id
            self.search_manager.reset()
            if camera_id in self._nodes:
                self._nodes[camera_id].mark_active_target()
            self._transit_history.append({
                "camera_id": camera_id,
                "timestamp": time.time(),
                "event": "TARGET_SELECTED",
                "track_id": track_id,
            })
            logger.info(
                f"[MULTI-CAM] Target selected by ID on camera '{camera_id}' | Tracker={track_id} | Gallery seeded."
            )
        return ok

    def add_manual_target_sample(self, camera_id: Optional[str] = None) -> bool:
        """
        Human-confirmed manual capture: adds the current locked target's appearance crop
        to the target gallery as a protected entry (is_manual=True).
        """
        target = self.target_manager.target
        if not self.target_manager.is_active() or target.track_id is None:
            logger.warning("[MULTI-CAM] Cannot add manual sample: no target currently locked.")
            return False

        cam_id = camera_id or self._active_camera_id
        if not cam_id:
            return False

        worker = self._workers.get(cam_id)
        if worker is None or worker._last_frame is None or worker._last_track_result is None:
            return False

        # Find current target track
        target_track: Optional[Track] = None
        for track in worker._last_track_result.tracks:
            if track.track_id == target.track_id:
                target_track = track
                break

        if target_track is None and target.last_known_box is not None:
            # Fall back to last known box if track dropped for a frame
            box = target.last_known_box
        elif target_track is not None:
            box = target_track.box
        else:
            return False

        crop = worker.extract_crop(worker._last_frame, box)
        if crop is None or crop.size == 0:
            return False

        added = self.target_manager.add_manual_sample(
            crop=crop,
            camera_id=cam_id,
            timestamp_ms=worker._last_track_result.timestamp_ms,
            frame_id=worker._last_track_result.frame_id,
        )
        return added

    def clear_target(self) -> None:
        """Clear the target and purge its appearance gallery."""
        self.target_manager.clear()
        for node in self._nodes.values():
            if node.is_online:
                node.mark_online()
        self._active_camera_id = None
        self.search_manager.reset()
        self._transit_history.clear()
        logger.info("[MULTI-CAM] Target cleared and gallery purged.")

    def update_graph(self, new_graph: CameraGraph) -> None:
        """
        Dynamically updates the running pipeline with a modified CameraGraph topology.
        Safely creates/updates camera nodes, refreshes search manager, and releases removed workers.
        """
        with self._frame_lock:
            self.graph = new_graph
            self.search_manager = SearchManager(self.graph, self.search_config)
            
            # Sync nodes with new graph
            self._sync_nodes_with_graph()

            # Clean up workers for cameras removed from the graph
            current_cams = set(self.graph.all_camera_ids())
            for cid in list(self._workers.keys()):
                if cid not in current_cams:
                    try:
                        self._workers[cid].stop()
                    except Exception:
                        pass
                    del self._workers[cid]
                    self._annotators.pop(cid, None)
                    self._latest_jpegs.pop(cid, None)

            # If active camera was removed or none set, pick first available
            if self._active_camera_id not in current_cams or not self._active_camera_id:
                enabled_cams = [cid for cid, n in self._nodes.items() if n.config.enabled]
                self._active_camera_id = enabled_cams[0] if enabled_cams else None

        logger.info(f"[MULTI-CAM] Topology updated: {len(self._nodes)} cameras, {self.graph.edge_count()} edges.")

    def step(self) -> Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]]:
        """
        Executes a single processing step across the multi-camera network.
        Processes active camera first, and only processes search-radius cameras if target is LOST.
        """
        if not self._is_running:
            return {}

        self._frame_count += 1
        now = time.time()
        results: Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]] = {}
        now = time.time()

        # 1. Default active camera if none selected or if removed
        if not self._active_camera_id or self._active_camera_id not in self._nodes:
            enabled_ids = [cid for cid, n in self._nodes.items() if n.config.enabled]
            self._active_camera_id = enabled_ids[0] if enabled_ids else None

        # 2. Process Active Camera
        active_cam_id = self._active_camera_id
        active_worker = self._get_or_create_worker(active_cam_id) if active_cam_id else None
        active_track_res: Optional[TrackResult] = None
        active_frame: Optional[np.ndarray] = None

        if active_cam_id is not None and active_worker and active_worker.camera.is_opened():
            success, frame, ts_ms = active_worker.read_frame()
            if success and frame is not None:
                active_frame = frame
                det_res, active_track_res = active_worker.process_frame(frame, ts_ms)
                if active_cam_id in self._nodes:
                    self._nodes[active_cam_id].fps = active_worker.fps

                # Target Appearance Evaluation on Active Camera
                self._evaluate_active_camera_target(active_worker, frame, active_track_res, ts_ms)

                ann_frame = active_worker.annotate(
                    frame,
                    active_track_res,
                    self.target_manager.target,
                    candidate_similarities=self._last_candidate_scores,
                )
                results[active_cam_id] = (ann_frame, active_track_res, self.target_manager.target)
            else:
                if active_cam_id in self._nodes:
                    self._nodes[active_cam_id].mark_offline()
        elif active_cam_id is not None and active_cam_id in self._nodes:
            self._nodes[active_cam_id].mark_offline()

        # 3. Check Target State on Active Camera
        target = self.target_manager.target
        target_is_active = (
            self.target_manager.is_active()
            and target.state in (TargetState.TRACKING, TargetState.OCCLUDED, TargetState.LOCKED)
        )
        target_is_lost = (
            self.target_manager.is_active()
            and target.state in (TargetState.LOST, TargetState.UNCERTAIN)
        )

        if target_is_active:
            if self.search_manager.is_searching:
                self.search_manager.reset()
                self._deactivate_search_cameras()
            self._target_lost_timestamp = 0.0
            if self._active_camera_id in self._nodes:
                self._nodes[self._active_camera_id].mark_active_target()

        elif target_is_lost and self._active_camera_id:
            has_adjacent_cams = len(self.graph.get_neighbors(self._active_camera_id, 1)) > 0
            if not has_adjacent_cams:
                # Single camera or isolated node: hold LOST state without futile search
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
                elapsed_s = now - self._target_lost_timestamp
                expansion = self.search_manager.tick(elapsed_s)
                if expansion is not None and len(expansion) > 0:
                    self._activate_search_cameras([cid for cid, _ in expansion])
                elif expansion == []:
                    logger.warning("[MULTI-CAM] Multi-camera search timed out without recovery.")
                    self._deactivate_search_cameras()

        # 4. Process Search Cameras
        candidate_recovered_cam: Optional[str] = None
        candidate_recovered_track: Optional[Track] = None
        candidate_recovered_crop: Optional[np.ndarray] = None
        candidate_recovered_emb: Optional[Embedding] = None

        if self.search_manager.is_searching:
            progress = self.search_manager.get_progress()
            for search_cam_id in progress.active_cameras:
                if search_cam_id == self._active_camera_id:
                    continue
                s_worker = self._get_or_create_worker(search_cam_id)
                if not s_worker or not s_worker.camera.is_opened():
                    continue

                s_success, s_frame, s_ts_ms = s_worker.read_frame()
                if not s_success or s_frame is None:
                    continue

                s_det, s_track = s_worker.process_frame(s_frame, s_ts_ms)
                self._nodes[search_cam_id].fps = s_worker.fps

                # Match candidate detections against target gallery
                if self.target_manager.is_active() and not self.gallery.is_empty and s_track.count > 0:
                    rec_track, rec_crop, rec_emb, rec_sim = self._match_candidates_against_gallery(
                        s_worker, s_frame, s_track
                    )
                    if rec_track is not None and rec_sim >= self.config.reid.match_threshold:
                        confirmed = self.search_manager.on_candidate_found(search_cam_id, rec_sim)
                        if confirmed:
                            logger.info(
                                f"[MULTI-CAM RECOVERY] Target CONFIRMED on '{search_cam_id}' "
                                f"(Track={rec_track.track_id}, sim={rec_sim:.3f})"
                            )
                            candidate_recovered_cam = search_cam_id
                            candidate_recovered_track = rec_track
                            candidate_recovered_crop = rec_crop
                            candidate_recovered_emb = rec_emb
                            break
                        else:
                            logger.debug(
                                f"[MULTI-CAM RECOVERY] Candidate sighting on '{search_cam_id}' "
                                f"(Track={rec_track.track_id}, sim={rec_sim:.3f}), confirming..."
                            )
                    else:
                        self.search_manager.on_candidate_lost(search_cam_id)

                s_ann = s_worker.annotate(s_frame, s_track, None)
                results[search_cam_id] = (s_ann, s_track, None)

        # 5. Cross-Camera Handoff
        if candidate_recovered_cam and candidate_recovered_track:
            self._perform_handoff(
                candidate_recovered_cam,
                candidate_recovered_track,
                candidate_recovered_crop,
                candidate_recovered_emb,
            )

        # 6. Read standby camera frames for operator surveillance UI (No AI overhead)
        active_and_search = {self._active_camera_id} if self._active_camera_id else set()
        if self.search_manager.is_searching:
            active_and_search.update(self.search_manager.get_progress().active_cameras)

        for cid, node in self._nodes.items():
            if cid not in active_and_search and node.config.enabled:
                s_worker = self._get_or_create_worker(cid)
                if s_worker and s_worker.camera.is_opened():
                    s_ok, s_f, _ = s_worker.read_frame()
                    if s_ok and s_f is not None:
                        node.last_frame = s_f
                        if not node.is_online:
                            node.mark_online()
                        if cid not in results:
                            results[cid] = (s_f, None, None)

        # Cache latest annotated/raw frames and pre-encode JPEGs for rock-solid, flicker-free web streaming
        for cid, (f, _, _) in results.items():
            if f is not None and cid in self._nodes:
                self._nodes[cid].last_frame = f
                try:
                    ret, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        with self._frame_lock:
                            self._latest_jpegs[cid] = buf.tobytes()
                except Exception:
                    pass

        return results

    def _is_occluded(self, track: Track, all_tracks: List[Track], iou_threshold: float = 0.25) -> bool:
        """Detects whether a track's bounding box significantly overlaps with any other track in the scene."""
        if not all_tracks or len(all_tracks) <= 1:
            return False
        for other in all_tracks:
            if other.track_id == track.track_id:
                continue
            if track.box.iou(other.box) > iou_threshold:
                return True
        return False

    def _evaluate_active_camera_target(
        self,
        worker: CameraWorker,
        frame: np.ndarray,
        track_res: TrackResult,
        timestamp_ms: float,
    ) -> None:
        """
        Manages target tracking continuity, crowd occlusion awareness, anti-scooping protection,
        temporal consensus before lock-switching, real-time diagnostic telemetry, and gallery growth.
        """
        target = self.target_manager.target
        if not self.target_manager.is_active():
            self._last_candidate_scores.clear()
            self._switch_consensus.clear()
            self._current_track_misses = 0
            return

        # 1. Identify currently locked track
        current_track: Optional[Track] = None
        for track in track_res.tracks:
            if track.track_id == target.track_id:
                current_track = track
                break

        should_reid = (self._frame_count % self.reid_interval == 0) or (target.state == TargetState.LOST)
        match_thresh = self.config.reid.match_threshold
        switch_margin = getattr(self.config.reid, "lock_switch_margin", 0.08)
        auto_thresh = self.config.reid.auto_add_threshold

        # Detect crowd occlusion / overlap on the active target
        is_occluded = self._is_occluded(current_track, track_res.tracks, iou_threshold=0.25) if current_track is not None else False

        # Case A: On frames where ReID is NOT evaluated
        if not should_reid or self.gallery.is_empty:
            if current_track is not None:
                if is_occluded:
                    self.target_manager.mark_tracking(current_track, track_res.frame_id, timestamp_ms)
                    self.target_manager.target.state = TargetState.OCCLUDED
                elif target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
                    self.target_manager.mark_tracking(current_track, track_res.frame_id, timestamp_ms)
            elif current_track is None and target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
                self._current_track_misses += 1
                if self._current_track_misses >= 3:
                    self.target_manager.mark_lost(timestamp_ms)
            return

        # Case B: On frames where ReID IS evaluated (or target is LOST / OCCLUDED)
        candidates_to_extract: List[Tuple[Track, np.ndarray]] = []
        crops_list: List[np.ndarray] = []

        for track in track_res.tracks:
            c_crop = worker.extract_crop(frame, track.box)
            if c_crop is not None and c_crop.size > 0 and c_crop.shape[0] >= 12 and c_crop.shape[1] >= 12:
                candidates_to_extract.append((track, c_crop))
                crops_list.append(c_crop)

        if not crops_list:
            if target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
                self._current_track_misses += 1
                if self._current_track_misses >= 3:
                    self.target_manager.mark_lost(timestamp_ms)
            return

        try:
            embs = self.reid_extractor.extract_batch(crops_list)
            match_details = self.gallery.match_batch_details(embs)
        except Exception as e:
            logger.debug(f"[REID] Candidate batch extraction error: {e}")
            return

        self._last_candidate_scores.clear()
        best_other_track: Optional[Track] = None
        best_other_crop: Optional[np.ndarray] = None
        best_other_emb: Optional[Embedding] = None
        best_other_sim = 0.0

        current_sim = 0.0
        current_crop: Optional[np.ndarray] = None
        current_emb: Optional[Embedding] = None

        cand_telemetry = []

        for (cand_track, cand_crop), cand_emb, (eff_sim, man_sim, auto_sim, _) in zip(candidates_to_extract, embs, match_details):
            eff_sim_f = float(eff_sim)
            self._last_candidate_scores[cand_track.track_id] = eff_sim_f
            cand_telemetry.append(f"#{cand_track.track_id}:{eff_sim_f:.2f}")

            if current_track is not None and cand_track.track_id == current_track.track_id:
                current_sim = eff_sim_f
                current_crop = cand_crop
                current_emb = cand_emb
            else:
                if eff_sim_f > best_other_sim:
                    best_other_sim = eff_sim_f
                    best_other_track = cand_track
                    best_other_crop = cand_crop
                    best_other_emb = cand_emb

        # Diagnostic telemetry
        cur_id_str = f"Track #{target.track_id} (sim={current_sim:.3f})" if (current_track and target.track_id) else "None (LOST)"
        logger.info(
            f"[REID] ActiveCam='{self._active_camera_id}' | Locked={cur_id_str} | "
            f"Candidates=[{', '.join(cand_telemetry)}] | Gallery=(man={self.gallery.manual_count}, auto={self.gallery.auto_count})"
        )

        # Scenario 1: Target was actively tracking, locked, or occluded
        if target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
            if current_track is not None:
                if is_occluded:
                    # Crowd / Occlusion safeguard: maintain spatial lock, freeze gallery updates
                    self.target_manager.mark_tracking(current_track, track_res.frame_id, timestamp_ms)
                    self.target_manager.target.state = TargetState.OCCLUDED
                    self._current_track_misses = 0
                    self._switch_consensus.clear()
                    return

                # Normal non-occluded verification
                if current_sim >= match_thresh:
                    self._current_track_misses = 0
                    self._switch_consensus.clear()
                    self.target_manager.mark_tracking(current_track, track_res.frame_id, timestamp_ms)

                    # Auto-enrollment (strictly when clear and quality passes)
                    if current_crop is not None and current_emb is not None and current_sim >= auto_thresh:
                        self.gallery.add_auto(
                            crop=current_crop,
                            embedding=current_emb,
                            candidate_similarity=current_sim,
                            camera_id=self._active_camera_id or "camera_0",
                            timestamp_ms=timestamp_ms,
                            frame_id=track_res.frame_id,
                            track_id=current_track.track_id,
                        )
                    return

                # Current track similarity dipped below match_thresh: Check if genuine switch or transient dip
                should_switch = False
                if best_other_track is not None and best_other_sim >= match_thresh:
                    margin = best_other_sim - current_sim
                    if margin >= switch_margin:
                        other_tid = best_other_track.track_id
                        self._switch_consensus[other_tid] = self._switch_consensus.get(other_tid, 0) + 1
                        if current_sim < (match_thresh - 0.20) or self._switch_consensus[other_tid] >= 2:
                            should_switch = True
                    else:
                        self._switch_consensus.clear()
                else:
                    self._switch_consensus.clear()

                if should_switch and best_other_track is not None:
                    logger.info(
                        f"[TARGET LOCK SWITCH] Camera '{self._active_camera_id}': Switching target lock from "
                        f"Track #{current_track.track_id} (sim={current_sim:.3f}) to Track #{best_other_track.track_id} "
                        f"(sim={best_other_sim:.3f})"
                    )
                    # Purge any auto-enrolled entries from the deposed track
                    self.gallery.rollback_auto_entries(for_track_id=current_track.track_id)
                    self._switch_consensus.clear()
                    self._current_track_misses = 0

                    self.target_manager.reassociate_target(
                        track=best_other_track,
                        frame_id=track_res.frame_id,
                        timestamp_ms=timestamp_ms,
                        reid_verified=True,
                    )
                    if best_other_crop is not None and best_other_emb is not None and best_other_sim >= auto_thresh:
                        self.gallery.add_auto(
                            crop=best_other_crop,
                            embedding=best_other_emb,
                            candidate_similarity=best_other_sim,
                            camera_id=self._active_camera_id or "camera_0",
                            timestamp_ms=timestamp_ms,
                            frame_id=track_res.frame_id,
                            track_id=best_other_track.track_id,
                        )
                    return

                # Grace period before declaring LOST (to avoid flipping on single noisy frame)
                self._current_track_misses += 1
                if self._current_track_misses < 3:
                    self.target_manager.mark_tracking(current_track, track_res.frame_id, timestamp_ms)
                    self.target_manager.target.state = TargetState.UNCERTAIN
                else:
                    logger.info(
                        f"[TARGET LOST] Track #{current_track.track_id} lost after {self._current_track_misses} "
                        f"misses (sim={current_sim:.3f} < {match_thresh:.3f})"
                    )
                    self.target_manager.mark_lost(timestamp_ms)
            else:
                # Current track dropped out of view
                if best_other_track is not None and best_other_sim >= match_thresh:
                    other_tid = best_other_track.track_id
                    self._switch_consensus[other_tid] = self._switch_consensus.get(other_tid, 0) + 1
                    if self._switch_consensus[other_tid] >= 2:
                        logger.info(
                            f"[TARGET REASSOCIATED] Track switched to #{best_other_track.track_id} (sim={best_other_sim:.3f})"
                        )
                        self._switch_consensus.clear()
                        self._current_track_misses = 0
                        self.target_manager.reassociate_target(
                            track=best_other_track,
                            frame_id=track_res.frame_id,
                            timestamp_ms=timestamp_ms,
                            reid_verified=True,
                        )
                        return
                self._current_track_misses += 1
                if self._current_track_misses >= 2:
                    self.target_manager.mark_lost(timestamp_ms)

        # Scenario 2: Target was LOST or UNCERTAIN - reacquire when candidate passes match threshold
        else:
            best_candidate: Optional[Track] = None
            best_cand_sim = 0.0
            best_cand_crop: Optional[np.ndarray] = None
            best_cand_emb: Optional[Embedding] = None

            if current_track is not None and current_sim >= match_thresh:
                best_candidate = current_track
                best_cand_sim = current_sim
                best_cand_crop = current_crop
                best_cand_emb = current_emb

            if best_other_track is not None and best_other_sim > best_cand_sim and best_other_sim >= match_thresh:
                best_candidate = best_other_track
                best_cand_sim = best_other_sim
                best_cand_crop = best_other_crop
                best_cand_emb = best_other_emb

            if best_candidate is not None:
                cand_tid = best_candidate.track_id
                self._switch_consensus[cand_tid] = self._switch_consensus.get(cand_tid, 0) + 1
                if self._switch_consensus[cand_tid] >= 2:
                    logger.info(
                        f"[TARGET REACQUIRED] Target reacquired on '{self._active_camera_id}' "
                        f"as Track #{best_candidate.track_id} (sim={best_cand_sim:.3f})"
                    )
                    self._switch_consensus.clear()
                    self._current_track_misses = 0
                    self.target_manager.reassociate_target(
                        track=best_candidate,
                        frame_id=track_res.frame_id,
                        timestamp_ms=timestamp_ms,
                        reid_verified=True,
                    )
                    if best_cand_crop is not None and best_cand_emb is not None and best_cand_sim >= auto_thresh:
                        self.gallery.add_auto(
                            crop=best_cand_crop,
                            embedding=best_cand_emb,
                            candidate_similarity=best_cand_sim,
                            camera_id=self._active_camera_id or "camera_0",
                            timestamp_ms=timestamp_ms,
                            frame_id=track_res.frame_id,
                            track_id=best_candidate.track_id,
                        )

    def _match_candidates_against_gallery(
        self,
        worker: CameraWorker,
        frame: np.ndarray,
        track_res: TrackResult,
    ) -> Tuple[Optional[Track], Optional[np.ndarray], Optional[Embedding], float]:
        """
        Extracts crops and embeddings for all candidate tracks in a frame
        and matches them against the target gallery via batch matrix multiply.
        """
        valid_candidates: List[Tuple[Track, np.ndarray, Embedding]] = []
        for track in track_res.tracks:
            crop = worker.extract_crop(frame, track.box)
            if crop is not None and crop.size > 0:
                emb = self.reid_extractor.extract(crop)
                valid_candidates.append((track, crop, emb))

        if not valid_candidates:
            return None, None, None, 0.0

        embs = [c[2] for c in valid_candidates]
        match_results = self.gallery.match_batch(embs)

        # Find best candidate with margin check
        scores = [res[0] for res in match_results]
        ranked_indices = np.argsort(scores)[::-1]
        best_idx = int(ranked_indices[0])
        best_score = scores[best_idx]
        second_best_score = scores[ranked_indices[1]] if len(scores) > 1 else 0.0
        margin = best_score - second_best_score

        if best_score >= self.config.reid.match_threshold and margin >= self.config.reid.min_margin:
            best_track, best_crop, best_emb = valid_candidates[best_idx]
            return best_track, best_crop, best_emb, best_score

        return None, None, None, best_score

    def _activate_search_cameras(self, camera_ids: List[str]) -> None:
        """Activate AI processing on search cameras."""
        for cid in camera_ids:
            if cid in self._nodes:
                self._nodes[cid].activate_ai()
            self._get_or_create_worker(cid)

    def _deactivate_search_cameras(self) -> None:
        """Deactivate AI processing on search cameras."""
        for cid, node in self._nodes.items():
            if cid != self._active_camera_id and node.status == CameraStatus.SEARCHING:
                node.deactivate_ai()

    def _perform_handoff(
        self,
        new_camera_id: str,
        recovered_track: Track,
        crop: Optional[np.ndarray] = None,
        embedding: Optional[Embedding] = None,
    ) -> None:
        """Executes active camera handoff to the newly recovered camera."""
        old_camera_id = self._active_camera_id
        logger.info(
            f"[MULTI-CAM HANDOFF] Target recovered on '{new_camera_id}'! "
            f"Handing off active camera: '{old_camera_id}' -> '{new_camera_id}'"
        )

        if old_camera_id and old_camera_id in self._nodes:
            self._nodes[old_camera_id].mark_online()

        self._active_camera_id = new_camera_id
        if new_camera_id in self._nodes:
            self._nodes[new_camera_id].mark_active_target()

        # Reassociate target to the track on the new camera
        self.target_manager.reassociate_target(
            track=recovered_track,
            frame_id=0,
            timestamp_ms=time.time() * 1000.0,
            reid_verified=True,
        )

        # Auto-enroll cross-camera viewpoint
        if crop is not None and embedding is not None:
            self.gallery.add_auto(
                crop=crop,
                embedding=embedding,
                candidate_similarity=0.95,
                camera_id=new_camera_id,
                timestamp_ms=time.time() * 1000.0,
                track_id=recovered_track.track_id,
            )

        self._handoff_timestamp = time.time()
        self._transit_history.append({
            "camera_id": new_camera_id,
            "from_camera": old_camera_id,
            "timestamp": self._handoff_timestamp,
            "event": "HANDOFF",
            "track_id": recovered_track.track_id,
        })

        self.search_manager.reset()
        self._deactivate_search_cameras()
        self._target_lost_timestamp = 0.0

    def process_single_camera_frame(
        self, camera_id: str
    ) -> Optional[Tuple[np.ndarray, TrackResult, Optional[Target]]]:
        """Runs detection and tracking for a single camera (used in CAMERA_FOCUS state)."""
        self._sync_nodes_with_graph()
        worker = self._get_or_create_worker(camera_id)
        if worker and worker.camera.is_opened():
            success, frame, ts_ms = worker.read_frame()
            if success and frame is not None:
                self._nodes[camera_id].last_frame = frame
                det_res, track_res = worker.process_frame(frame, ts_ms)
                self._nodes[camera_id].fps = worker.fps
                ann_frame = worker.annotate(frame, track_res, self.target_manager.target)
                return ann_frame, track_res, self.target_manager.target
            else:
                self._nodes[camera_id].mark_offline()
        return None

    def read_monitoring_frames(self) -> Dict[str, np.ndarray]:
        """
        Grab a single live frame from each enabled camera WITHOUT running AI processing.
        Used by the MONITORING UI state.
        """
        self._sync_nodes_with_graph()
        frames: Dict[str, np.ndarray] = {}

        for cid, node in self._nodes.items():
            if not node.config.enabled:
                continue
            worker = self._get_or_create_worker(cid)
            if worker and worker.camera.is_opened():
                success, frame, _ = worker.read_frame()
                if success and frame is not None:
                    frames[cid] = frame
                    node.last_frame = frame
                    node.mark_online()
                else:
                    node.mark_offline()
        return frames

    def set_active_camera(self, camera_id: str) -> bool:
        """Sets the active focused camera in the pipeline and ensures worker is initialized."""
        self._sync_nodes_with_graph()
        if camera_id in self._nodes:
            old_cam = self._active_camera_id
            if old_cam and old_cam in self._nodes and old_cam != camera_id:
                if self._nodes[old_cam].is_online:
                    self._nodes[old_cam].mark_online()

            self._active_camera_id = camera_id
            self._nodes[camera_id].mark_active_target()
            worker = self._get_or_create_worker(camera_id)
            if worker and worker.camera.is_opened():
                self._nodes[camera_id].mark_online()
                self._nodes[camera_id].mark_active_target()

            logger.info(f"[MULTI-CAM] Active camera manually switched to '{camera_id}'")
            return True
        return False

    def get_camera_frame_jpeg(self, camera_id: str, quality: int = 75) -> Optional[bytes]:
        """Returns the latest frame-locked JPEG bytes for a camera feed under thread-safe lock."""
        with self._frame_lock:
            if camera_id in self._latest_jpegs:
                return self._latest_jpegs[camera_id]

        # Check node last_frame
        node = self._nodes.get(camera_id)
        frame = node.last_frame if node is not None else None

        # Fallback to worker last_frame
        if frame is None:
            worker = self._workers.get(camera_id)
            if worker is not None and worker._last_frame is not None:
                frame = worker._last_frame

        if frame is not None:
            try:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ret:
                    raw_bytes = buf.tobytes()
                    with self._frame_lock:
                        self._latest_jpegs[camera_id] = raw_bytes
                    return raw_bytes
            except Exception:
                pass
        return None

    def get_all_camera_cards(self) -> List[Dict[str, Any]]:
        """Returns structured metadata for all cameras in the surveillance grid."""
        cards = []
        progress = self.search_manager.get_progress()
        searching_cams = set(progress.active_cameras) if self.search_manager.is_searching else set()
        valid_graph_ids = set(self.graph.all_camera_ids())

        for cid in valid_graph_ids:
            node = self._nodes.get(cid)
            if not node:
                continue
            cfg = node.config
            is_active = (cid == self._active_camera_id)
            is_searching = (cid in searching_cams)
            
            if is_active:
                status_str = "ACTIVE"
            elif is_searching:
                status_str = f"SEARCHING (R={self.search_manager.search_state})"
            elif node.is_online:
                status_str = "STANDBY"
            else:
                status_str = "OFFLINE"

            cards.append({
                "camera_id": cid,
                "name": cfg.name or cid,
                "source": cfg.source,
                "source_type": cfg.source_type.value if hasattr(cfg.source_type, "value") else str(cfg.source_type),
                "enabled": cfg.enabled,
                "is_active": is_active,
                "is_searching": is_searching,
                "status": status_str,
                "fps": round(node.fps, 1),
                "floor": cfg.floor,
                "zone": cfg.zone,
                "has_frame": (node.last_frame is not None),
            })
        return cards

    def stream(
        self,
    ) -> Generator[Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]], None, None]:
        """Continuous generator yielding multi-camera step results."""
        self._is_running = True
        try:
            while self._is_running:
                results = self.step()
                yield results
        finally:
            self.stop()

    def update_graph(self, new_graph: CameraGraph) -> None:
        """
        Dynamically updates the camera topology graph at runtime (Issue 2).
        Synchronizes camera nodes, spawns workers for newly added cameras,
        and cleanly releases workers for removed cameras.
        """
        logger.info(f"[MULTI-CAM] Dynamically updating camera topology graph ({len(new_graph.all_camera_ids())} cameras)...")
        self.graph = new_graph
        self.search_manager._graph = new_graph

        all_new_ids = set(new_graph.all_camera_ids())

        # 1. Stop and remove workers & nodes for cameras no longer in graph
        for old_id in list(self._workers.keys()):
            if old_id not in all_new_ids:
                logger.info(f"[MULTI-CAM] Releasing removed camera worker '{old_id}'")
                try:
                    self._workers[old_id].stop()
                except Exception as e:
                    logger.debug(f"Error stopping worker '{old_id}': {e}")
                del self._workers[old_id]

        for old_id in list(self._nodes.keys()):
            if old_id not in all_new_ids:
                del self._nodes[old_id]
                self._annotators.pop(old_id, None)
                with self._frame_lock:
                    self._latest_jpegs.pop(old_id, None)

        # 2. Sync nodes and update configs for existing or newly added cameras
        for cid in all_new_ids:
            node_cfg = new_graph.get_node(cid)
            if not node_cfg:
                continue

            if cid in self._nodes:
                old_cfg = self._nodes[cid].config
                # Check if camera source changed; if so, recreate worker
                if str(old_cfg.source) != str(node_cfg.source) or old_cfg.source_type != node_cfg.source_type:
                    logger.info(f"[MULTI-CAM] Camera '{cid}' source changed ({old_cfg.source} -> {node_cfg.source}), restarting worker...")
                    if cid in self._workers:
                        try:
                            self._workers[cid].stop()
                        except Exception:
                            pass
                        del self._workers[cid]
                    with self._frame_lock:
                        self._latest_jpegs.pop(cid, None)
                self._nodes[cid].config = node_cfg
            else:
                self._nodes[cid] = CameraNode(node_cfg)
                self._annotators[cid] = FrameAnnotator(
                    draw_fps=self.config.visualization.draw_fps,
                    draw_boxes=self.config.visualization.draw_boxes,
                    draw_ids=self.config.visualization.draw_ids,
                    box_thickness=self.config.visualization.box_thickness,
                    font_scale=self.config.visualization.font_scale,
                )

            if node_cfg.enabled and cid not in self._workers:
                self._get_or_create_worker(cid)

        # 3. Update active camera if previous active camera was deleted
        if self._active_camera_id not in all_new_ids:
            enabled_ids = [cid for cid in all_new_ids if self._nodes.get(cid) and self._nodes[cid].config.enabled]
            self._active_camera_id = sorted(enabled_ids)[0] if enabled_ids else (sorted(list(all_new_ids))[0] if all_new_ids else None)
            if self._active_camera_id:
                logger.info(f"[MULTI-CAM] Active camera reassigned to '{self._active_camera_id}'")
            else:
                logger.info("[MULTI-CAM] No active camera remaining in graph.")

    def stop(self) -> None:
        """Stop all camera workers and release resources."""
        self._is_running = False
        for worker in self._workers.values():
            try:
                worker.stop()
            except Exception:
                pass
        self._workers.clear()
        logger.info("[MULTI-CAM] Pipeline stopped.")

