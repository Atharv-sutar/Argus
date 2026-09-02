"""Unified surveillance pipeline coordinator with target-only gallery and graph-aware search."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import cv2
import numpy as np
import concurrent.futures

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
from src.identity.manager import IdentityManager
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

        # 2. Identity Manager & Target Manager
        if identity_manager is not None:
            self.identity_manager = identity_manager
        else:
            if getattr(config, "storage", None) and config.storage.enabled:
                from src.identity.sqlite_store import SQLiteVectorStore
                vector_store = SQLiteVectorStore(db_path=config.storage.db_path)
            else:
                vector_store = None
                
            self.identity_manager = IdentityManager(
                reid_extractor=self.reid_extractor,
                vector_store=vector_store
            )
            
            # Load identities from database on startup
            if vector_store is not None:
                self.identity_manager.load_from_db(config.storage.db_path)

        if target_manager is not None:
            self.target_manager = target_manager
        else:
            self.target_manager = TargetManager(
                identity_manager=self.identity_manager,
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
        self._is_paused: bool = False
        self._pipeline_lock: threading.RLock = threading.RLock()
        self._transit_history: List[Dict[str, Any]] = []
        self._frame_count: int = 0
        self.reid_interval: int = config.reid.extract_interval_frames
        self._frame_lock: threading.Lock = threading.Lock()
        self._latest_jpegs: Dict[str, bytes] = {}
        self._last_candidate_scores: Dict[int, float] = {}
        self._switch_consensus: Dict[int, int] = {}
        self._current_track_misses: int = 0

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=12, thread_name_prefix="CamWorker")

        # 7. Reacquisition threshold (strictly higher than match_threshold)
        self._reacquisition_threshold: float = getattr(
            config.reid, 'reacquisition_threshold',
            max(0.75, config.reid.match_threshold + 0.10)
        )

        # 8. EvidenceEngine for temporal evidence accumulation (anti-scoop)
        from src.identity.evidence import EvidenceEngine
        self._evidence_engine = EvidenceEngine(
            window_size=5,
            min_similarity_threshold=config.reid.match_threshold,
            reacquisition_threshold=self._reacquisition_threshold,
            reacquisition_min_frames=3,
            min_margin_threshold=config.reid.min_margin,
            min_consistency_ratio=0.70,
        )

        # Sync nodes from graph
        self._sync_nodes_with_graph()

    @property
    def is_running(self) -> bool:
        """Returns True if the pipeline is active and not stopped."""
        return self._is_running

    @property
    def identity(self) -> IdentityManager:
        """The identity manager managing targets."""
        return self.target_manager.identity_manager

    @property
    def gallery(self) -> IdentityManager:
        """Backward-compatible alias: returns the IdentityManager which exposes
        size, max_size, manual_count, auto_count, get_thumbnails, remove_entry."""
        return self.target_manager.identity_manager

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
            reid_extractor=self.reid_extractor,
            reid_weight=0.5,
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
                worker = self._workers.pop(cid, None)
                if worker is not None:
                    try:
                        worker.stop()
                    except Exception as e:
                        logger.debug(f"Error stopping worker '{cid}': {e}")
                self._nodes.pop(cid, None)
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

        try:
            camera = self._camera_factory(node.config)
            tracker = self._tracker_factory()
            annotator = self._annotators.get(camera_id) or FrameAnnotator()

            worker = CameraWorker(
                camera=camera,
                tracker=tracker,
                annotator=annotator,
                camera_id=camera_id,
            )
            self._workers[camera_id] = worker
            if camera.is_opened():
                node.mark_online()
            else:
                node.mark_offline()
            logger.info(f"[MULTI-CAM] Initialized camera worker for '{camera_id}' (source={node.config.source}, opened={camera.is_opened()})")
            return worker
        except Exception as e:
            logger.warning(f"[MULTI-CAM] Failed to initialize worker for camera '{camera_id}': {e}")
            node.mark_offline()
            return None

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
            if hasattr(worker, "detector"):
                det_res = worker.detector.detect(frame_to_process, timestamp_ms=time.time() * 1000.0)
            else:
                det_res = self._default_detector_factory().detect(frame_to_process, timestamp_ms=time.time() * 1000.0)
            worker.process_frame(frame_to_process, time.time() * 1000.0, det_res)

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
                det_res = self._default_detector_factory().detect(frame, timestamp_ms=ts_ms)
                worker.process_frame(frame, ts_ms, det_res)

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

    def step(self) -> Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]]:
        """
        Executes a single processing step across the multi-camera network concurrently.
        """
        if not self._is_running or self._is_paused:
            return {}

        self._frame_count += 1
        now = time.time()
        results: Dict[str, Tuple[Optional[np.ndarray], Optional[TrackResult], Optional[Target]]] = {}

        # 1. Default active camera if none selected or if removed
        if not self._active_camera_id or self._active_camera_id not in self._nodes:
            enabled_ids = [cid for cid in self.graph.all_camera_ids() if cid in self._nodes and self._nodes[cid].config.enabled]
            self._active_camera_id = enabled_ids[0] if enabled_ids else None

        # 2. Gather Phase 1 Tasks: I/O and AI Inference
        tasks = []
        active_cam_id = self._active_camera_id
        active_worker = self._get_or_create_worker(active_cam_id) if active_cam_id else None

        if active_cam_id and active_worker and active_worker.camera.is_opened():
            tasks.append((active_cam_id, active_worker, "active"))

        target_is_lost = (self.target_manager.is_active() and self.target_manager.target.state in (TargetState.LOST, TargetState.UNCERTAIN))
        search_cams_to_process = set()

        if self.search_manager.is_searching:
            search_cams_to_process.update(self.search_manager.get_progress().active_cameras)

        for search_cam_id in search_cams_to_process:
            if search_cam_id == active_cam_id:
                continue
            s_worker = self._get_or_create_worker(search_cam_id)
            if s_worker and s_worker.camera.is_opened():
                tasks.append((search_cam_id, s_worker, "search"))

        active_and_search = {t[0] for t in tasks}

        for cid, node in self._nodes.items():
            if cid not in active_and_search and node.config.enabled:
                worker = self._get_or_create_worker(cid)
                if worker and worker.camera.is_opened():
                    tasks.append((cid, worker, "standby"))

        def _task_acquire(cid: str, worker: CameraWorker, role: str):
            success, frame, ts_ms = worker.read_frame()
            if not success or frame is None:
                return cid, role, None, ts_ms
            return cid, role, frame, ts_ms

        # Execute Phase 1: Parallel Acquisition
        acquired_results = {}
        acq_futures = {self._executor.submit(_task_acquire, cid, w, role): cid for cid, w, role in tasks}
        for future in concurrent.futures.as_completed(acq_futures):
            cid = acq_futures[future]
            try:
                acquired_results[cid] = future.result()
            except Exception as e:
                logger.error(f"[MULTI-CAM] Camera worker error on {cid}: {e}")
                role = next((r for c, _, r in tasks if c == cid), "error")
                acquired_results[cid] = (cid, role, None, 0.0)

        # Execute Phase 2: Batched YOLO Inference (or Mock Detection for Tests)
        yolo_frames = []
        yolo_meta = []
        det_map = {}
        
        for cid, (r_cid, role, frame, ts_ms) in acquired_results.items():
            if frame is not None and role in ("active", "search"):
                worker = self._get_or_create_worker(cid)
                if worker and hasattr(worker, "detector"):
                    # Testing backdoor: if a mock detector is attached to the worker, use it directly
                    det_map[cid] = worker.detector.detect(frame, self._frame_count, ts_ms)
                else:
                    yolo_frames.append(frame)
                    yolo_meta.append((cid, ts_ms))
        
        if yolo_frames:
            batch_det_results = self._default_detector_factory().detect_batch(
                frames=yolo_frames,
                frame_ids=[self._frame_count] * len(yolo_frames),
                timestamps_ms=[ts for _, ts in yolo_meta]
            )
            for meta, det_res in zip(yolo_meta, batch_det_results):
                det_map[meta[0]] = det_res

        # Execute Phase 3: Parallel Tracking
        def _task_track(cid: str, worker: CameraWorker, frame: np.ndarray, ts_ms: float, det_res: DetectionResult):
            track_res = worker.process_frame(frame, ts_ms, det_res)
            return cid, frame, ts_ms, track_res
        
        track_futures = {}
        for cid, (r_cid, role, frame, ts_ms) in acquired_results.items():
            if role in ("active", "search") and frame is not None:
                worker = self._get_or_create_worker(cid)
                det_res = det_map.get(cid)
                if worker and det_res:
                    track_futures[self._executor.submit(_task_track, cid, worker, frame, ts_ms, det_res)] = cid
        
        phase1_results = {}
        for future in concurrent.futures.as_completed(track_futures):
            cid = track_futures[future]
            try:
                _, frame, ts_ms, track_res = future.result()
                role = acquired_results[cid][1]
                phase1_results[cid] = (cid, role, frame, ts_ms, track_res)
            except Exception as e:
                logger.error(f"[MULTI-CAM] Tracking error on {cid}: {e}")
                role = acquired_results[cid][1]
                phase1_results[cid] = (cid, role, None, 0.0, None)

        # Add standby cameras to phase1_results to match old format
        for cid, (r_cid, role, frame, ts_ms) in acquired_results.items():
            if role == "standby" or frame is None:
                phase1_results[cid] = (cid, role, frame, ts_ms, None)

        # Execute Phase 4: Batched ReID Extraction
        target = self.target_manager.target
        should_reid = (self._frame_count % self.reid_interval == 0) or (target.state == TargetState.LOST)
        
        precomputed_reid_candidates: Dict[str, List[Tuple[Track, np.ndarray, Embedding]]] = {}
        
        if should_reid and not self.identity.is_empty:
            crops_to_extract = []
            crop_meta = [] # (cid, Track, crop)
            
            for cid, (r_cid, role, frame, ts_ms, track_res) in phase1_results.items():
                if frame is not None and track_res is not None and role in ("active", "search"):
                    worker = self._get_or_create_worker(cid)
                    if worker:
                        for track in track_res.tracks:
                            c_crop = worker.extract_crop(frame, track.box)
                            if c_crop is not None and c_crop.size > 0 and c_crop.shape[0] >= 12 and c_crop.shape[1] >= 12:
                                crops_to_extract.append(c_crop)
                                crop_meta.append((cid, track, c_crop))
            
            if crops_to_extract:
                try:
                    batch_embs = self.reid_extractor.extract_batch(crops_to_extract)
                    for (cid, track, crop), emb in zip(crop_meta, batch_embs):
                        if cid not in precomputed_reid_candidates:
                            precomputed_reid_candidates[cid] = []
                        precomputed_reid_candidates[cid].append((track, crop, emb))
                except Exception as e:
                    logger.error(f"[MULTI-CAM] Batched ReID extraction error: {e}")

        active_track_res = None
        active_frame = None

        # 3. Process Active Camera Main-Thread Logic
        if active_cam_id and active_cam_id in phase1_results:
            _, role, frame, ts_ms, track_res = phase1_results[active_cam_id]
            if frame is not None and role == "active":
                active_frame = frame
                active_track_res = track_res
                self._nodes[active_cam_id].fps = active_worker.fps
                
                precomp_cands = precomputed_reid_candidates.get(active_cam_id)
                self._evaluate_active_camera_target(
                    worker=active_worker, 
                    frame=frame, 
                    track_res=active_track_res, 
                    timestamp_ms=ts_ms,
                    precomputed_candidates=precomp_cands
                )
            else:
                self._nodes[active_cam_id].mark_offline()
        elif active_cam_id and active_cam_id in self._nodes:
            self._nodes[active_cam_id].mark_offline()

        # 4. Check Target State
        target = self.target_manager.target
        target_is_active = (self.target_manager.is_active() and target.state in (TargetState.TRACKING, TargetState.OCCLUDED, TargetState.LOCKED))
        target_is_lost = (self.target_manager.is_active() and target.state in (TargetState.LOST, TargetState.UNCERTAIN))

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
                if self.search_manager.is_searching:
                    self.search_manager.reset()
                    self._deactivate_search_cameras()
            elif not self.search_manager.is_searching:
                self._target_lost_timestamp = now
                search_cams = self.search_manager.on_target_lost(self._active_camera_id)
                if search_cams:
                    logger.info(f"[MULTI-CAM] Target LOST on camera '{self._active_camera_id}'. Initiating graph search on {len(search_cams)} adjacent cameras.")
                    self._activate_search_cameras([cid for cid, _ in search_cams])
            else:
                elapsed_s = now - self._target_lost_timestamp
                expansion = self.search_manager.tick(elapsed_s)
                if expansion is not None and len(expansion) > 0:
                    self._activate_search_cameras([cid for cid, _ in expansion])
                elif expansion == []:
                    logger.warning("[MULTI-CAM] Multi-camera search timed out without recovery.")
                    self._deactivate_search_cameras()

        # 5. Process Search Cameras Main-Thread Logic
        candidate_recovered_cam = None
        candidate_recovered_track = None
        candidate_recovered_crop = None
        candidate_recovered_emb = None

        for cid, (r_cid, role, frame, ts_ms, track_res) in phase1_results.items():
            if role == "search" and frame is not None and track_res is not None:
                s_worker = self._get_or_create_worker(cid)
                if not s_worker:
                    continue
                self._nodes[cid].fps = s_worker.fps
                if self.target_manager.is_active() and not self.identity.is_empty and track_res.count > 0:
                    precomp_cands = precomputed_reid_candidates.get(cid)
                    rec_track, rec_crop, rec_emb, rec_sim = self._match_candidates_against_gallery(
                        worker=s_worker, 
                        frame=frame, 
                        track_res=track_res,
                        precomputed_candidates=precomp_cands
                    )
                    if rec_track is not None and rec_sim >= self._reacquisition_threshold:
                        confirmed = self.search_manager.on_candidate_found(cid, rec_sim)
                        if confirmed:
                            logger.info(f"[MULTI-CAM RECOVERY] Target CONFIRMED on '{cid}' (Track={rec_track.track_id}, sim={rec_sim:.3f}, reacq_thresh={self._reacquisition_threshold:.3f})")
                            candidate_recovered_cam = cid
                            candidate_recovered_track = rec_track
                            candidate_recovered_crop = rec_crop
                            candidate_recovered_emb = rec_emb
                            break
                        else:
                            logger.debug(f"[MULTI-CAM RECOVERY] Candidate sighting on '{cid}' (Track={rec_track.track_id}, sim={rec_sim:.3f}), confirming...")
                    else:
                        self.search_manager.on_candidate_lost(cid)

        # 6. Cross-Camera Handoff
        if candidate_recovered_cam and candidate_recovered_track:
            self._perform_handoff(candidate_recovered_cam, candidate_recovered_track, candidate_recovered_crop, candidate_recovered_emb)

        # 7. Gather Phase 2 Tasks: Annotation and JPEG Compression
        def _task_annotate_and_encode(cid: str, role: str, worker: CameraWorker, frame: np.ndarray, track_res: Optional[TrackResult], target: Optional[Target], candidate_scores: Dict[int, float]):
            ann_frame = frame
            if role == "active":
                ann_frame = worker.annotate(frame, track_res, target, candidate_similarities=candidate_scores)
            elif role == "search":
                ann_frame = worker.annotate(frame, track_res, None)
            elif role == "standby":
                pass # Standby doesn't need annotation
            
            jpeg_buf = None
            try:
                ret, buf = cv2.imencode(".jpg", ann_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    jpeg_buf = buf.tobytes()
            except Exception:
                pass
            return cid, ann_frame, track_res, target if role == "active" else None, jpeg_buf

        phase2_futures = []
        for cid, (r_cid, role, frame, ts_ms, track_res) in phase1_results.items():
            if frame is not None:
                worker = self._get_or_create_worker(cid)
                if worker:
                    if role == "standby":
                        self._nodes[cid].last_frame = frame
                        if not self._nodes[cid].is_online:
                            self._nodes[cid].mark_online()
                    
                    target_arg = self.target_manager.target if role == "active" else None
                    scores_arg = self._last_candidate_scores if role == "active" else None
                    phase2_futures.append(self._executor.submit(_task_annotate_and_encode, cid, role, worker, frame, track_res, target_arg, scores_arg))

        # 8. Finalize Results
        for future in concurrent.futures.as_completed(phase2_futures):
            try:
                cid, ann_frame, track_res, tgt, jpeg_buf = future.result()
                if cid in self._nodes:
                    self._nodes[cid].last_frame = ann_frame
                results[cid] = (ann_frame, track_res, tgt)
                if jpeg_buf:
                    with self._frame_lock:
                        self._latest_jpegs[cid] = jpeg_buf
            except Exception as e:
                logger.error(f"[MULTI-CAM] Annotation/Encoding error: {e}")

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
        precomputed_candidates: Optional[List[Tuple[Track, np.ndarray, Embedding]]] = None,
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
        if not should_reid or self.identity.is_empty:
            if current_track is not None:
                if is_occluded:
                    self.target_manager.mark_tracking(current_track, track_res.frame_id, timestamp_ms)
                    self.target_manager.target.state = TargetState.OCCLUDED
                elif target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
                    self.target_manager.mark_tracking(current_track, track_res.frame_id, timestamp_ms)
            elif current_track is None and target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
                self.target_manager.mark_lost(timestamp_ms)
            return

        # Case B: On frames where ReID IS evaluated (or target is LOST / OCCLUDED)
        candidates_to_extract: List[Tuple[Track, np.ndarray]] = []
        embs: List[Embedding] = []

        if precomputed_candidates is not None:
            for track, crop, emb in precomputed_candidates:
                candidates_to_extract.append((track, crop))
                embs.append(emb)
        else:
            crops_list: List[np.ndarray] = []
            for track in track_res.tracks:
                c_crop = worker.extract_crop(frame, track.box)
                if c_crop is not None and c_crop.size > 0 and c_crop.shape[0] >= 12 and c_crop.shape[1] >= 12:
                    candidates_to_extract.append((track, c_crop))
                    crops_list.append(c_crop)

            if not crops_list:
                if target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
                    self.target_manager.mark_lost(timestamp_ms)
                return

            try:
                embs = self.reid_extractor.extract_batch(crops_list)
            except Exception as e:
                logger.debug(f"[REID] Candidate batch extraction error: {e}")
                return

        if not candidates_to_extract:
            if target.state in (TargetState.TRACKING, TargetState.LOCKED, TargetState.OCCLUDED):
                self.target_manager.mark_lost(timestamp_ms)
            return

        try:
            match_details = self.identity.match_batch_details(embs)
        except Exception as e:
            logger.debug(f"[REID] Gallery match error: {e}")
            return

        self._last_candidate_scores.clear()
        best_other_track: Optional[Track] = None
        best_other_crop: Optional[np.ndarray] = None
        best_other_emb: Optional[Embedding] = None
        best_other_sim = 0.0
        best_other_man_sim = 0.0

        current_sim = 0.0
        current_man_sim = 0.0
        current_crop: Optional[np.ndarray] = None
        current_emb: Optional[Embedding] = None

        cand_telemetry = []

        for (cand_track, cand_crop), cand_emb, (eff_sim, man_sim, auto_sim, _) in zip(candidates_to_extract, embs, match_details):
            eff_sim_f = float(eff_sim)
            man_sim_f = float(man_sim)
            self._last_candidate_scores[cand_track.track_id] = eff_sim_f
            cand_telemetry.append(f"#{cand_track.track_id}:{eff_sim_f:.2f}(m:{man_sim_f:.2f})")

            if current_track is not None and cand_track.track_id == current_track.track_id:
                current_sim = eff_sim_f
                current_man_sim = man_sim_f
                current_crop = cand_crop
                current_emb = cand_emb
            else:
                if eff_sim_f > best_other_sim:
                    best_other_sim = eff_sim_f
                    best_other_man_sim = man_sim_f
                    best_other_track = cand_track
                    best_other_crop = cand_crop
                    best_other_emb = cand_emb

        # Diagnostic telemetry
        cur_id_str = f"Track #{target.track_id} (sim={current_sim:.3f}, man={current_man_sim:.3f})" if (current_track and target.track_id) else "None (LOST)"
        logger.info(
            f"[REID] ActiveCam='{self._active_camera_id}' | Locked={cur_id_str} | "
            f"Candidates=[{', '.join(cand_telemetry)}] | Gallery=(man={self.identity.manual_count}, auto={self.identity.auto_count})"
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
                        self.identity.add_auto(
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

                        # Anti-scoop: verify switch candidate against manual gallery entries
                        manual_anchor_pass = True
                        if self.identity._manual_matrix is not None and len(self.identity._manual_entries) > 0:
                            manual_anchor_pass = best_other_man_sim >= max(0.55, match_thresh - 0.05)
                            if not manual_anchor_pass:
                                logger.info(
                                    f"[ANTI-SCOOP] Lock-switch to Track #{other_tid} REJECTED: "
                                    f"manual anchor check failed (sim_manual={best_other_man_sim:.3f})"
                                )

                        if manual_anchor_pass:
                            # Immediate switch when margin is decisive or current track is very weak
                            if margin >= 0.15 or current_sim < (match_thresh - 0.20):
                                should_switch = True
                            # Moderate consensus: 2 frames for clear margin
                            elif self._switch_consensus[other_tid] >= 2:
                                should_switch = True
                    else:
                        self._switch_consensus.clear()
                else:
                    self._switch_consensus.clear()

                if should_switch and best_other_track is not None:
                    logger.info(
                        f"[TARGET LOCK SWITCH] Camera '{self._active_camera_id}': Switching target lock from "
                        f"Track #{current_track.track_id} (sim={current_sim:.3f}) to Track #{best_other_track.track_id} "
                        f"(sim={best_other_sim:.3f}, man={best_other_man_sim:.3f})"
                    )
                    # Purge auto-enrolled entries from the deposed track AND any stale entries for the new track
                    self.identity.rollback_auto_entries(for_track_id=current_track.track_id)
                    self.identity.rollback_auto_entries(for_track_id=best_other_track.track_id)
                    self._switch_consensus.clear()
                    self._current_track_misses = 0

                    self.target_manager.reassociate_target(
                        track=best_other_track,
                        frame_id=track_res.frame_id,
                        timestamp_ms=timestamp_ms,
                        reid_verified=True,
                    )
                    if best_other_crop is not None and best_other_emb is not None and best_other_sim >= auto_thresh:
                        self.identity.add_auto(
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
                if self._current_track_misses < 2:
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
                if best_other_track is not None and best_other_sim >= self._reacquisition_threshold:
                    other_tid = best_other_track.track_id
                    self._switch_consensus[other_tid] = self._switch_consensus.get(other_tid, 0) + 1
                    if self._switch_consensus[other_tid] >= 3:
                        logger.info(
                            f"[TARGET REASSOCIATED] Track switched to #{best_other_track.track_id} "
                            f"(sim={best_other_sim:.3f}, reacq_thresh={self._reacquisition_threshold:.3f})"
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
                self.target_manager.mark_lost(timestamp_ms)

        # Scenario 2: Target was LOST or UNCERTAIN — Evidence-gated reacquisition (anti-scoop)
        else:
            # Feed all candidates into the EvidenceEngine
            evidence_records = []
            all_sorted = []
            for (cand_track, cand_crop), cand_emb, (eff_sim, man_sim, auto_sim, _) in zip(
                candidates_to_extract, embs, match_details
            ):
                eff_sim_f = float(eff_sim)
                is_match = eff_sim_f >= self._reacquisition_threshold

                # Lone-bystander anti-scoop: when only 1 person in frame, require higher threshold
                lone_person = len(candidates_to_extract) == 1
                if lone_person:
                    lone_thresh = self._reacquisition_threshold + 0.05
                    is_match = eff_sim_f >= lone_thresh
                    if not is_match:
                        logger.info(
                            f"[ANTI-SCOOP] LONE_PERSON_ANTILOCK: Track #{cand_track.track_id} "
                            f"sim={eff_sim_f:.3f} < lone_thresh={lone_thresh:.3f} | "
                            f"Refusing to lock onto single person in frame while target is LOST"
                        )

                # Manual anchor verification for reacquisition
                if is_match and self.identity._manual_matrix is not None and len(self.identity._manual_entries) > 0:
                    manual_anchor_thresh = max(0.55, self._reacquisition_threshold - 0.15)
                    if man_sim_f < manual_anchor_thresh:
                        logger.info(
                            f"[ANTI-SCOOP] Reacquisition of Track #{cand_track.track_id} REJECTED: "
                            f"manual anchor check failed (sim_manual={man_sim_f:.3f} < {manual_anchor_thresh:.3f})"
                        )
                        is_match = False

                self._evidence_engine.register_observation(
                    track_id=cand_track.track_id,
                    frame_id=self._frame_count,
                    timestamp_ms=timestamp_ms,
                    crop_quality=1.0,
                    similarity=eff_sim_f,
                    margin=0.0,
                    is_match=is_match,
                    box=cand_track.box,
                )
                evidence_records.append((cand_track, eff_sim_f, is_match, 1.0))
                all_sorted.append((cand_track, cand_crop, cand_emb, eff_sim_f))

            # Prune stale tracks
            self._evidence_engine.prune_stale_tracks(
                [t.track_id for t in track_res.tracks],
                current_frame_id=self._frame_count,
                max_stale_frames=30,
            )

            # Evaluate through the EvidenceEngine
            if evidence_records:
                evidence_dec = self._evidence_engine.evaluate_all_candidates(
                    evidence_records,
                    target_identity_id="target_0",
                    current_tracked_id=target.track_id,
                    is_reacquisition=True,
                )

                if evidence_dec.is_confirmed and evidence_dec.best_track_id is not None:
                    # Find the confirmed track's data
                    confirmed_track = None
                    confirmed_crop = None
                    confirmed_emb = None
                    confirmed_sim = evidence_dec.best_score
                    for (ct, cc, ce, cs) in all_sorted:
                        if ct.track_id == evidence_dec.best_track_id:
                            confirmed_track = ct
                            confirmed_crop = cc
                            confirmed_emb = ce
                            confirmed_sim = cs
                            break

                    if confirmed_track is not None:
                        logger.info(
                            f"[TARGET REACQUIRED] Target reacquired on '{self._active_camera_id}' "
                            f"as Track #{confirmed_track.track_id} (sim={confirmed_sim:.3f}, "
                            f"evidence_score={evidence_dec.best_score:.3f}, "
                            f"reason={evidence_dec.decision_reason})"
                        )
                        self._switch_consensus.clear()
                        self._current_track_misses = 0
                        self.target_manager.reassociate_target(
                            track=confirmed_track,
                            frame_id=track_res.frame_id,
                            timestamp_ms=timestamp_ms,
                            decision=evidence_dec.verified_token,
                            reid_verified=True,
                        )
                        if confirmed_crop is not None and confirmed_emb is not None and confirmed_sim >= auto_thresh:
                            self.identity.add_auto(
                                crop=confirmed_crop,
                                embedding=confirmed_emb,
                                candidate_similarity=confirmed_sim,
                                camera_id=self._active_camera_id or "camera_0",
                                timestamp_ms=timestamp_ms,
                                frame_id=track_res.frame_id,
                                track_id=confirmed_track.track_id,
                            )
                elif evidence_dec.diagnostic_log:
                    logger.debug(evidence_dec.diagnostic_log)

    def _match_candidates_against_gallery(
        self,
        worker: CameraWorker,
        frame: np.ndarray,
        track_res: TrackResult,
        precomputed_candidates: Optional[List[Tuple[Track, np.ndarray, Embedding]]] = None,
    ) -> Tuple[Optional[Track], Optional[np.ndarray], Optional[Embedding], float]:
        """
        Extracts crops and embeddings for all candidate tracks in a frame
        and matches them against the target gallery via batch matrix multiply.
        """
        valid_candidates: List[Tuple[Track, np.ndarray, Embedding]] = []
        if precomputed_candidates is not None:
            valid_candidates = precomputed_candidates
        else:
            for track in track_res.tracks:
                crop = worker.extract_crop(frame, track.box)
                if crop is not None and crop.size > 0:
                    emb = self.reid_extractor.extract(crop)
                    valid_candidates.append((track, crop, emb))

        if not valid_candidates:
            return None, None, None, 0.0

        embs = [c[2] for c in valid_candidates]
        match_results = self.identity.match_batch(embs)

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

        # Auto-enroll cross-camera viewpoint (use actual measured similarity, NOT hardcoded)
        if crop is not None and embedding is not None:
            # Compute actual similarity against gallery for honest enrollment
            actual_sim, _ = self.identity.match(embedding)
            self.identity.add_auto(
                crop=crop,
                embedding=embedding,
                candidate_similarity=actual_sim,
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
                det_res = self._default_detector_factory().detect(frame, timestamp_ms=ts_ms)
                track_res = worker.process_frame(frame, ts_ms, det_res)
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
                status_str = f"SEARCHING (R={self.search_manager.current_radius})"
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
        Dynamically updates the camera topology graph at runtime.
        Synchronizes camera nodes, spawns workers for newly added cameras,
        and cleanly releases workers for removed cameras in a thread-safe manner.
        """
        self._is_paused = True
        try:
            with self._pipeline_lock:
                logger.info(f"[MULTI-CAM] Dynamically updating camera topology graph ({len(new_graph.all_camera_ids())} cameras)...")
                self.graph = new_graph
                self.search_manager._graph = new_graph
                self.search_manager.reset()

                all_new_ids = set(new_graph.all_camera_ids())

                # 1. Stop and remove workers & nodes for cameras no longer in graph
                for old_id in list(self._workers.keys()):
                    if old_id not in all_new_ids:
                        logger.info(f"[MULTI-CAM] Releasing removed camera worker '{old_id}'")
                        worker = self._workers.pop(old_id, None)
                        if worker is not None:
                            try:
                                worker.stop()
                            except Exception as e:
                                logger.debug(f"Error stopping worker '{old_id}': {e}")

                for old_id in list(self._nodes.keys()):
                    if old_id not in all_new_ids:
                        self._nodes.pop(old_id, None)
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
                            worker = self._workers.pop(cid, None)
                            if worker is not None:
                                try:
                                    worker.stop()
                                except Exception:
                                    pass
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

                # 3. Update active camera if previous active camera was deleted or disabled
                enabled_ids = [cid for cid in all_new_ids if self._nodes.get(cid) and self._nodes[cid].config.enabled]
                if self._active_camera_id not in enabled_ids:
                    self._active_camera_id = sorted(enabled_ids)[0] if enabled_ids else (sorted(list(all_new_ids))[0] if all_new_ids else None)
                    if self._active_camera_id:
                        logger.info(f"[MULTI-CAM] Active camera reassigned to '{self._active_camera_id}'")
                    else:
                        logger.info("[MULTI-CAM] No active camera remaining in graph.")
        finally:
            self._is_paused = False

    def release_all_cameras(self) -> None:
        """Safely stops and releases all active camera capture handles."""
        logger.info("[MULTI-CAM] Releasing all camera capture handles...")
        with self._frame_lock:
            for cid, worker in list(self._workers.items()):
                try:
                    worker.stop()
                except Exception as e:
                    logger.debug(f"Error stopping worker '{cid}': {e}")
            self._workers.clear()
            self._latest_jpegs.clear()
            for node in self._nodes.values():
                node.last_frame = None

    def restart_cameras(self) -> None:
        """Safely shuts down all camera handles, waits for hardware release, and re-initializes enabled camera workers."""
        logger.info("[MULTI-CAM] Executing full camera shutdown and restart sequence...")
        self._is_paused = True
        try:
            with self._pipeline_lock:
                self.release_all_cameras()
                time.sleep(0.25)  # allow OS/DirectShow to fully release hardware locks
                with self._frame_lock:
                    self._sync_nodes_with_graph()
                    for cid, node in self._nodes.items():
                        if node.config.enabled:
                            self._get_or_create_worker(cid)
            logger.info(f"[MULTI-CAM] All cameras restarted. Re-initialized {len(self._workers)} camera workers.")
        finally:
            self._is_paused = False

    def pause_processing(self) -> None:
        """Safely pauses pipeline step loop and releases all camera handles for hardware probing."""
        logger.info("[MULTI-CAM] Pausing pipeline processing for hardware access...")
        self._is_paused = True
        with self._pipeline_lock:
            self.release_all_cameras()

    def resume_processing(self) -> None:
        """Restores camera handles and resumes pipeline step loop."""
        logger.info("[MULTI-CAM] Resuming pipeline processing...")
        with self._pipeline_lock:
            self.restart_cameras()
            self._is_paused = False

    def stop(self) -> None:
        """Stop all camera workers and release resources."""
        self._is_running = False
        self._is_paused = False
        with self._pipeline_lock:
            for worker in list(self._workers.values()):
                try:
                    worker.stop()
                except Exception:
                    pass
            self._workers.clear()
            
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass
                
            # Persist target gallery on graceful shutdown
            if getattr(self.config, "storage", None) and self.config.storage.enabled:
                self.identity.save_to_db(self.config.storage.db_path)
                
        logger.info("[MULTI-CAM] Pipeline stopped.")

