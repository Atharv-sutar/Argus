
from src.camera.capture import SyntheticCamera
from src.core.config import AppConfig
from src.core.multi_camera_types import CameraEdgeConfig, CameraNodeConfig, CameraStatus, SourceType
from src.multi_camera.camera_graph import CameraGraph
from src.pipeline.multi_camera_pipeline import MultiCameraPipeline


def test_update_graph_removes_deleted_nodes_and_caches():
    """Verify that removing a camera from the graph cleanly stops workers, purges nodes, and clears JPEG cache."""
    cfg = AppConfig()
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_0", name="Entrance", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_node(CameraNodeConfig(camera_id="cam_1", name="Hallway", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_edge(CameraEdgeConfig(source_camera_id="cam_0", target_camera_id="cam_1"))

    pipe = MultiCameraPipeline(
        config=cfg,
        graph=graph,
        camera_factory=lambda node_cfg: SyntheticCamera(width=320, height=240, fps=30),
    )
    
    # Run a step so frames and cards are generated
    res = pipe.step()
    assert "cam_0" in pipe._nodes
    assert "cam_1" in pipe._nodes
    
    cards_before = pipe.get_all_camera_cards()
    assert len(cards_before) == 2

    # Now remove cam_1 from graph and update
    new_graph = CameraGraph()
    new_graph.add_node(CameraNodeConfig(camera_id="cam_0", name="Entrance", source="synthetic", source_type=SourceType.SYNTHETIC))
    
    pipe.update_graph(new_graph)

    assert "cam_1" not in pipe._nodes
    assert "cam_1" not in pipe._workers
    with pipe._frame_lock:
        assert "cam_1" not in pipe._latest_jpegs

    cards_after = pipe.get_all_camera_cards()
    assert len(cards_after) == 1
    assert cards_after[0]["camera_id"] == "cam_0"
    
    pipe.stop()


def test_set_active_camera_and_target_selection_on_standby_camera():
    """Verify that switching active camera and clicking target on a standby camera works seamlessly."""
    cfg = AppConfig()
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_a", name="Cam A", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_node(CameraNodeConfig(camera_id="cam_b", name="Cam B", source="synthetic", source_type=SourceType.SYNTHETIC))
    graph.add_edge(CameraEdgeConfig(source_camera_id="cam_a", target_camera_id="cam_b"))

    pipe = MultiCameraPipeline(
        config=cfg,
        graph=graph,
        camera_factory=lambda node_cfg: SyntheticCamera(width=320, height=240, fps=30),
    )
    
    # Initial active camera is cam_a
    pipe.step()
    assert pipe.active_camera_id == "cam_a"

    # Switch active camera to cam_b
    ok = pipe.set_active_camera("cam_b")
    assert ok is True
    assert pipe.active_camera_id == "cam_b"
    assert pipe._nodes["cam_b"].is_online is True

    # Target selection coordinates attempt on cam_b
    pipe.select_target_on_camera("cam_b", x=100.0, y=100.0)
    assert pipe.active_camera_id == "cam_b"

    pipe.stop()


def test_sync_nodes_marks_disabled_and_deactivate_search_notifies_search_manager():
    """
    Verifies:
    1. Disabling a camera node stops its worker and calls CameraNode.mark_disabled().
    2. Re-enabling transitions DISABLED -> OFFLINE so workers can reconnect.
    3. _deactivate_search_cameras notifies SearchManager.on_search_complete(cid).
    """
    cfg = AppConfig()
    graph = CameraGraph()
    graph.add_node(CameraNodeConfig(camera_id="cam_0", name="Entrance", source="synthetic", source_type=SourceType.SYNTHETIC, enabled=True))
    graph.add_node(CameraNodeConfig(camera_id="cam_1", name="Hallway", source="synthetic", source_type=SourceType.SYNTHETIC, enabled=True))
    graph.add_edge(CameraEdgeConfig(source_camera_id="cam_0", target_camera_id="cam_1"))

    pipe = MultiCameraPipeline(
        config=cfg,
        graph=graph,
        camera_factory=lambda node_cfg: SyntheticCamera(width=320, height=240, fps=30),
    )

    try:
        pipe.step()
        assert pipe._nodes["cam_1"].status != CameraStatus.DISABLED

        # Initiate search so cam_1 is active in search_manager & node status SEARCHING
        pipe._active_camera_id = "cam_0"
        pipe._nodes["cam_1"].mark_online()
        pipe._nodes["cam_1"].activate_ai()
        pipe.search_manager.on_target_lost("cam_0")
        assert "cam_1" in pipe.search_manager.get_progress().active_cameras

        pipe._deactivate_search_cameras()
        assert pipe._nodes["cam_1"].status == CameraStatus.ONLINE
        assert pipe._nodes["cam_1"].is_ai_active is False
        assert "cam_1" not in pipe.search_manager.get_progress().active_cameras

        # Disable cam_1 via graph update
        disabled_graph = CameraGraph()
        disabled_graph.add_node(CameraNodeConfig(camera_id="cam_0", name="Entrance", source="synthetic", source_type=SourceType.SYNTHETIC, enabled=True))
        disabled_graph.add_node(CameraNodeConfig(camera_id="cam_1", name="Hallway", source="synthetic", source_type=SourceType.SYNTHETIC, enabled=False))
        disabled_graph.add_edge(CameraEdgeConfig(source_camera_id="cam_0", target_camera_id="cam_1"))

        pipe.update_graph(disabled_graph)
        assert pipe._nodes["cam_1"].status == CameraStatus.DISABLED
        assert pipe._nodes["cam_1"].is_ai_active is False
        assert "cam_1" not in pipe._workers

        # Re-enable cam_1: status leaves DISABLED so worker init / mark_online works
        enabled_graph = CameraGraph()
        enabled_graph.add_node(CameraNodeConfig(camera_id="cam_0", name="Entrance", source="synthetic", source_type=SourceType.SYNTHETIC, enabled=True))
        enabled_graph.add_node(CameraNodeConfig(camera_id="cam_1", name="Hallway", source="synthetic", source_type=SourceType.SYNTHETIC, enabled=True))
        pipe.update_graph(enabled_graph)
        assert pipe._nodes["cam_1"].status != CameraStatus.DISABLED
    finally:
        pipe.stop()

