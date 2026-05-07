"""Calibration runner (background thread)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .collision import _build_collision_manager, _self_collides, _trajectory_collides
from .constants import (
    _BOARD_TARGET_OFFSETS_LOCAL,
    _BOOTSTRAP_N_CANDIDATES,
    _CALIBRATION_USE_CONTINUOUS,
    _CAM_MOUNT_TILT_DEG,
    _CAM_MOUNT_TRANSLATE_MM,
    _ENABLE_SELF_COLLISION_CHECK,
    _HEMI_DISTANCE_RANGE_M,
    _HEMI_ELEVATION_RANGE_DEG,
    _INTR_CX,
    _INTR_CY,
    _INTR_FX,
    _INTR_FY,
    _INTR_H,
    _INTR_W,
    _MAIN_N_CANDIDATES,
    _MAX_CAM_BOARD_ANGLE_DEG,
    _OCCLUSION_BOARD_SAMPLES_LOCAL,
    _OCCLUSION_MAX_BLOCKED,
    _SETTLE_TIME_REAL_S,
    _SETTLE_TIME_SIM_S,
)
from .frustum import update_frustum  # noqa: F401 - imported for completeness per spec
from .occlusion import _build_occlusion_mesh, _camera_occlusion_count
from .reachability import _greedy_farthest_first
from .state import (
    _T_BOARD2BASE,
    _hemi_azimuth_world_range_deg,
    _hemi_centre_world,
    _state,
)
from .workspace import _ensure_workspace_envelope, envelope_contains

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration runner (background thread)
# ---------------------------------------------------------------------------


def _calibration_thread() -> None:
    """Run the calibration sim end-to-end and push joint updates to the GUI.

    Drives the running parol6-server's controller via UDP — same path
    Waldo-Commander uses — so each ``move_j`` is executed by the controller
    and its joint state is broadcast to all subscribers (including the
    URDF scene's status consumer). The GUI animates naturally; no
    update_urdf_angles thread plumbing required.
    """
    # Lazy import to break panel<->calibration_thread cycle.
    from .panel import _post_status  # noqa: PLC0415
    try:
        # Lazy imports.
        from parol6 import Robot, RobotClient  # noqa: PLC0415

        from parol6_vision.calibration.board import (  # noqa: PLC0415
            BOARD_TABLET_30MM,
            BoardDetector,
        )
        from parol6_vision.calibration.camera_mount import (  # noqa: PLC0415
            CameraMount,
        )
        from parol6_vision.calibration.orchestrator import (  # noqa: PLC0415
            CalibrationOrchestrator,
            OrchestratorConfig,
        )
        from parol6_vision.calibration.pose_generator import (  # noqa: PLC0415
            HemisphereParams,
            PoseGenerator,
        )
        from parol6_vision.camera.intrinsics import Intrinsics  # noqa: PLC0415
        from parol6_vision.sim.virtual_camera import (  # noqa: PLC0415
            VirtualBoard,
            VirtualCamera,
        )
        from parol6_vision.sim.tracing_client import (  # noqa: PLC0415
            _flange_pose_from_client,
        )

        # Mode dispatch: piggyback off Waldo-Commander's existing real/sim
        # toggle (the orange-when-sim "robot" button at the top of the page).
        # robot_state.simulator_active=True means the user is in simulator
        # mode → use VirtualCamera; False means real-hardware mode → use
        # RealSenseCamera. Default to sim if the import fails (defensive).
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
            is_sim_mode = bool(robot_state.simulator_active)
        except Exception:  # noqa: BLE001
            is_sim_mode = True

        T_BOARD2BASE = _T_BOARD2BASE
        if is_sim_mode:
            # Sim ground truth: cold-start tunable + small fixed perturbation
            # (±2 mm / ±2°) so the simulated calibration always has something
            # realistic to converge to.
            ground_truth_mount = CameraMount.from_eyeball_estimate(
                x_mm=_CAM_MOUNT_TRANSLATE_MM[0] + 2.0,
                y_mm=_CAM_MOUNT_TRANSLATE_MM[1] + 2.0,
                z_mm=_CAM_MOUNT_TRANSLATE_MM[2] - 2.0,
                tilt_x_deg=_CAM_MOUNT_TILT_DEG[0] + 2.0,
                tilt_y_deg=_CAM_MOUNT_TILT_DEG[1] - 1.0,
                tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
            )
            intrinsics = Intrinsics(
                fx=_INTR_FX, fy=_INTR_FY, cx=_INTR_CX, cy=_INTR_CY,
                width=_INTR_W, height=_INTR_H,
                dist_coeffs=np.zeros(5, dtype=np.float64),
            )
        else:
            ground_truth_mount = None  # not used in real mode
            intrinsics = None  # set below from the actual RealSense device

        robot = Robot()

        # Drive the running parol6-server (controller in fake-serial mode).
        # Each move_j blocks until motion completes; the controller broadcasts
        # joint state at 50 Hz, which Waldo-Commander's status consumer picks
        # up and feeds into the URDF scene. No thread/queue plumbing needed.
        raw_client = RobotClient(host="127.0.0.1", port=5001)

        # Wrap the client so the STOP button can short-circuit subsequent
        # move_j calls — halt() alone only aborts the in-flight motion;
        # the orchestrator's own loop will happily request the next pose
        # right after. Returning -1 from move_j makes _move_to_joints
        # treat each remaining pose as a MotionError and the orchestrator
        # gives up cleanly with ``insufficient_samples_pass1``.
        class _HaltableClient:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def move_j(self, *args: Any, **kwargs: Any) -> int:
                if _state.get("stop_requested"):
                    return -1
                return self._inner.move_j(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        client = _HaltableClient(raw_client)
        # Stash the raw client so the STOP button can call halt() directly.
        _state["client"] = raw_client

        # Same FK-based flange pose query as the localise thread —
        # client.pose("WRF") returns TCP (with the gripper's 105 mm tool
        # offset baked in), NOT the flange. The mount transform
        # T_cam2flange is defined relative to the FLANGE, so feeding TCP
        # poses into it puts the VirtualCamera 105 mm out of position
        # (verified live: detection succeeded at only 4 / 16 bootstrap
        # poses because the camera was rendering far too close to the
        # board). Run forward kinematics on the live joint angles to
        # bypass the tool offset.
        from scipy.spatial.transform import Rotation as _R_calib_fk  # noqa: PLC0415

        def flange_pose() -> NDArray[np.float64] | None:
            angles_deg = client.angles()
            if angles_deg is None or len(angles_deg) < 6:
                return None
            angles_rad = np.radians(np.asarray(angles_deg, dtype=np.float64))
            fk_pose = np.zeros(6, dtype=np.float64)
            robot.fk(angles_rad, fk_pose)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _R_calib_fk.from_euler("XYZ", fk_pose[3:]).as_matrix()
            T[:3, 3] = fk_pose[:3]
            return T

        if is_sim_mode:
            camera: Any = VirtualCamera(
                intrinsics=intrinsics,
                image_width=_INTR_W,
                image_height=_INTR_H,
                ground_truth_mount=ground_truth_mount,
                flange_pose_provider=flange_pose,
                board=VirtualBoard(config=BOARD_TABLET_30MM, T_board2base=T_BOARD2BASE),
                noise_std=0.0,
            )
            logger.info("calibration camera: VirtualCamera (simulator mode)")
        else:
            from parol6_vision.camera.realsense import RealSenseCamera  # noqa: PLC0415
            camera = RealSenseCamera(
                width=_INTR_W,
                height=_INTR_H,
                fps=30,
                enable_depth=False,  # calibration only needs color frames
                enable_color=True,
            )
            # Stash BEFORE start(): if start() raises mid-pipeline-init the
            # device may have already been claimed; the finally-block must
            # see the camera handle to call .stop() cleanly.
            _state["real_camera"] = camera
            camera.start()
            intrinsics = camera.intrinsics
            logger.info(
                "calibration camera: RealSenseCamera (real-hardware mode), "
                "intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f, dist=%s",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                intrinsics.dist_coeffs.tolist(),
            )
        detector = BoardDetector(BOARD_TABLET_30MM)
        cold_start = CameraMount.from_eyeball_estimate(
            x_mm=_CAM_MOUNT_TRANSLATE_MM[0],
            y_mm=_CAM_MOUNT_TRANSLATE_MM[1],
            z_mm=_CAM_MOUNT_TRANSLATE_MM[2],
            tilt_x_deg=_CAM_MOUNT_TILT_DEG[0],
            tilt_y_deg=_CAM_MOUNT_TILT_DEG[1],
            tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
        )

        # Hemisphere CENTRE — also the calibration's look-at target. Uses the
        # hemisphere override if set (decouples from board rotation), else
        # falls back to the board centre. With no override (sim default), this
        # is the board's geometric centre as before.
        target_world = _hemi_centre_world()

        # Diagnostic: confirm the board CENTER is reachable per waldo-commander's
        # workspace hull. If not, the orchestrator will struggle to find any
        # valid hemisphere candidates.
        _ensure_workspace_envelope()
        max_reach = _state.get("envelope_max_reach")
        if max_reach is not None:
            board_ok = bool(envelope_contains(target_world)[0])
            logger.info(
                "workspace envelope check: board center %s -> %s "
                "(hull max_reach=%.3f m)",
                np.round(target_world, 3).tolist(),
                "INSIDE" if board_ok else "OUTSIDE",
                max_reach,
            )

        # Hull-aware + floor-aware + collision-aware + occlusion-aware +
        # trajectory-aware + farthest-first PoseGenerator.
        # Filters layered on top of the base PoseGenerator:
        #   1. Workspace-hull check on flange position.
        #   2. Floor-clipping check: gripper fingertips above floor.
        #   3. Self-collision: gripper/bracket vs arm links at the destination pose.
        #   4. Camera occlusion: line-of-sight from camera to board NOT blocked
        #      by any robot link. Catches "robot body in front of the lens" poses
        #      that aren't collisions but produce useless calibration captures.
        #   5. Greedy farthest-first selection (spatial spread).
        #   6. Trajectory self-collision: each move from the previous accepted
        #      candidate to this one must not pass through self-collision at
        #      any sampled point along the joint-space line. Plus a check from
        #      HOME → first candidate. Catches the "fingertips clip mid-move"
        #      case the user reported.
        FLOOR_Z_MIN_M = 0.005  # 5 mm safety margin above the workbench
        TCP_OFFSET_FLANGE = np.array([0.0, 0.0, -0.105, 1.0])  # SSG-48 TCP point in flange frame

        # Build collision manager + occlusion meshes once (loads 7 link meshes
        # + gripper for collision; 7 link meshes for occlusion). The tablet
        # primitive is placed at the current _T_BOARD2BASE so the robot won't
        # drive into the physical ChArUco display.
        collision_mgr_pair = (
            _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
            if _ENABLE_SELF_COLLISION_CHECK else None
        )
        occlusion_meshes = (
            _build_occlusion_mesh() if _ENABLE_SELF_COLLISION_CHECK else None
        )

        class HullFilteredPoseGenerator(PoseGenerator):
            def generate(self, *args, **kwargs):  # type: ignore[override]
                max_count = kwargs.get("max_count", args[0] if args else 8)
                # Over-request so we have a pool to thin from.
                kwargs["max_count"] = max_count * 5

                # Multi-target relaxed look-at — DEFERRED_FEATURES.md §6.
                # Build the per-pass aim points in board-local UV ∈ [0, 1]²,
                # convert to world frame, pass to super().generate() as
                # `look_at_targets`. The hemisphere CENTRE stays anchored on
                # self.target_world (the value of _hemi_centre_world() at
                # construction time — possibly the override); only the
                # look-at aim varies per pass.
                #
                # When _HEMI_CENTRE_OVERRIDE_M is set, self.target_world is
                # the override (a fixed workable-space anchor). The aim
                # points STILL come from _T_BOARD2BASE — i.e. cameras aim
                # at the actual board, not the override. Single-target mode
                # uses the centre offset (0.5, 0.5) explicitly so the same
                # invariant holds with or without multi-target.
                cfg = BOARD_TABLET_30MM
                w_m_b = cfg.squares_x * cfg.square_length
                h_m_b = cfg.squares_y * cfg.square_length
                offsets = (
                    _BOARD_TARGET_OFFSETS_LOCAL
                    if len(_BOARD_TARGET_OFFSETS_LOCAL) > 1
                    else ((0.5, 0.5),)
                )
                aim_targets = [
                    (
                        _T_BOARD2BASE
                        @ np.array(
                            [u * w_m_b, v * h_m_b, 0.0, 1.0],
                            dtype=np.float64,
                        )
                    )[:3]
                    for u, v in offsets
                ]
                cands, stats = super().generate(  # type: ignore[arg-type]
                    **kwargs, look_at_targets=aim_targets,
                )
                if len(offsets) > 1:
                    logger.info(
                        "multi-target look-at: %d targets, %d raw candidates "
                        "(pre-filter)",
                        len(offsets), len(cands),
                    )

                # 1. Hull filter.
                if _state.get("envelope_planes_A") is not None:
                    pre = len(cands)
                    cands = [
                        c for c in cands
                        if bool(envelope_contains(
                            np.asarray(c.flange_pose, dtype=np.float64)[:3, 3]
                        )[0])
                    ]
                    if pre - len(cands) > 0:
                        logger.info(
                            "hull filter: %d/%d candidates rejected (flange outside envelope)",
                            pre - len(cands), pre,
                        )

                # 2. Floor-clipping filter — TCP world Z must clear the floor.
                pre = len(cands)
                cands = [
                    c for c in cands
                    if (np.asarray(c.flange_pose, dtype=np.float64)
                        @ TCP_OFFSET_FLANGE)[2] > FLOOR_Z_MIN_M
                ]
                if pre - len(cands) > 0:
                    logger.info(
                        "floor filter: %d/%d candidates rejected (gripper would clip floor)",
                        pre - len(cands), pre,
                    )

                # 3. Self-collision filter — gripper/bracket must not clip arm.
                if collision_mgr_pair is not None:
                    coll_mgr, adjacent_pairs, _ = collision_mgr_pair
                    pre = len(cands)
                    cands = [
                        c for c in cands
                        if not _self_collides(coll_mgr, adjacent_pairs, c.joint_angles_rad)
                    ]
                    if pre - len(cands) > 0:
                        logger.info(
                            "self-collision filter: %d/%d candidates rejected "
                            "(gripper clips into arm)",
                            pre - len(cands), pre,
                        )

                # 4. Camera-occlusion filter — multi-ray sampling across the
                # board face. Cast a ray from the camera position to each of
                # _OCCLUSION_BOARD_SAMPLES_LOCAL (default centre + 4 corners,
                # in board-local UV ∈ [0, 1]²). If MORE than
                # _OCCLUSION_MAX_BLOCKED rays hit a robot link before reaching
                # the target, reject the pose. Catches partial-frame occlusion
                # (e.g. arm body blocks the side of the FOV but not the centre).
                if occlusion_meshes is not None:
                    cold_T = self.mount.T_cam2flange
                    cfg = BOARD_TABLET_30MM
                    w_m_b = cfg.squares_x * cfg.square_length
                    h_m_b = cfg.squares_y * cfg.square_length
                    sample_world = [
                        (
                            _T_BOARD2BASE
                            @ np.array([u * w_m_b, v * h_m_b, 0.0, 1.0],
                                       dtype=np.float64)
                        )[:3]
                        for u, v in _OCCLUSION_BOARD_SAMPLES_LOCAL
                    ]
                    pre = len(cands)
                    survivors = []
                    for c in cands:
                        T_flange2base = np.asarray(c.flange_pose, dtype=np.float64)
                        T_cam2base = T_flange2base @ cold_T
                        cam_pos_world = T_cam2base[:3, 3]
                        n_blocked = _camera_occlusion_count(
                            occlusion_meshes, c.joint_angles_rad,
                            cam_pos_world, sample_world,
                            early_termination_after=_OCCLUSION_MAX_BLOCKED,
                        )
                        if n_blocked <= _OCCLUSION_MAX_BLOCKED:
                            survivors.append(c)
                    cands = survivors
                    if pre - len(cands) > 0:
                        logger.info(
                            "occlusion filter: %d/%d candidates rejected "
                            "(>%d/%d sample rays blocked by robot body)",
                            pre - len(cands), pre,
                            _OCCLUSION_MAX_BLOCKED,
                            len(_OCCLUSION_BOARD_SAMPLES_LOCAL),
                        )

                # 4.5. Viewing-angle filter — reject candidates whose camera
                # optical axis is more than _MAX_CAM_BOARD_ANGLE_DEG off the
                # board surface normal. At extreme angles the board projects
                # as a thin sliver, ChArUco corner detection works but reproj
                # error is high and the calibration's effective resolution
                # drops. Threshold defaults to 65° — admits the bottom of the
                # configured 25°-elevation hemisphere, rejects anything worse.
                if _MAX_CAM_BOARD_ANGLE_DEG < 90.0:
                    cold_T = self.mount.T_cam2flange
                    board_z_world = _T_BOARD2BASE[:3, 2]
                    cos_threshold = float(
                        np.cos(np.radians(_MAX_CAM_BOARD_ANGLE_DEG))
                    )
                    pre = len(cands)
                    survivors = []
                    for c in cands:
                        T_flange2base = np.asarray(c.flange_pose, dtype=np.float64)
                        T_cam2base = T_flange2base @ cold_T
                        cam_z_world = T_cam2base[:3, 2]
                        # Use abs() — board normal could point either way
                        # depending on board placement; we care about the
                        # angle, not the orientation.
                        if abs(float(cam_z_world @ board_z_world)) >= cos_threshold:
                            survivors.append(c)
                    cands = survivors
                    if pre - len(cands) > 0:
                        logger.info(
                            "viewing-angle filter: %d/%d rejected "
                            "(camera axis > %.0f° off board normal)",
                            pre - len(cands), pre,
                            _MAX_CAM_BOARD_ANGLE_DEG,
                        )

                # 5. Farthest-first thinning on flange positions.
                if len(cands) > max_count:
                    cands = _greedy_farthest_first(
                        cands, max_count,
                        key=lambda c: np.asarray(c.flange_pose, dtype=np.float64)[:3, 3],
                    )
                    logger.info("farthest-first selected %d well-spread candidates", len(cands))

                # 6. Trajectory self-collision (pairwise). Each successive
                # move from the previously-selected candidate to the next must
                # not pass through self-collision at any sampled point along
                # the joint-space line. The FIRST candidate is accepted
                # unconditionally — the orchestrator will be coming from the
                # last bootstrap pose at that point, not from HOME, and we
                # don't know where bootstrap landed (it'd over-reject if we
                # assumed HOME).
                if collision_mgr_pair is not None and len(cands) > 0:
                    coll_mgr, adjacent_pairs, _ = collision_mgr_pair
                    safe: list = [cands[0]]
                    n_traj_rejected = 0
                    for c in cands[1:]:
                        prev_q = safe[-1].joint_angles_rad
                        if _trajectory_collides(
                            coll_mgr, adjacent_pairs, prev_q, c.joint_angles_rad,
                        ):
                            n_traj_rejected += 1
                            continue
                        safe.append(c)
                    if n_traj_rejected > 0:
                        logger.info(
                            "trajectory filter: %d/%d candidates rejected "
                            "(consecutive-move joint-space line passes through "
                            "self-collision)", n_traj_rejected, len(cands),
                        )
                    cands = safe

                return cands, stats

        # NB: prior to this commit we monkey-patched
        # ``parol6_vision.calibration.orchestrator.PoseGenerator`` here so
        # the orchestrator's main hemisphere pass would use our filtered
        # subclass. The orchestrator now accepts a ``pose_generator_factory``
        # in its config, so the subclass is injected explicitly via
        # ``OrchestratorConfig(pose_generator_factory=...)`` further down.

        # NB: prior to this commit we monkey-patched
        # ``parol6_vision.calibration.refinement.board_position_from_sample``
        # AND ``parol6_vision.calibration.pose_generator.look_at_pose`` here.
        # Both are now upstreamed — refinement takes a ``board: BoardConfig``
        # kwarg, and the pose generator's ``generate()`` takes
        # ``look_at_targets=[...]`` to drive multi-target relaxed look-at.
        # Zero monkey-patches remain in this thread.

        # Hemisphere params for the bootstrap pass and the main pass.
        #
        # MAIN pass: drives off the user-facing _HEMI_* tunables. Wide range,
        # may include low-elevation / close-distance poses where part of the
        # board can't fit in the FOV — that's fine, the orchestrator only
        # needs PARTIAL board visibility (≥6 corners) to count a sample.
        #
        # BOOTSTRAP pass: uses CONSERVATIVE bounds that are independent of
        # the user-facing tunables. The bootstrap's job is to localise the
        # board reliably (it only takes a handful of seed poses), so we want
        # poses where the entire 210×150 mm board comfortably fits in the
        # 640×480 FOV (≥0.22 m at fx=fy=615 gives ~230 mm horizontal coverage,
        # so the board fits with margin). Mid-range elevations (30–55°) avoid
        # extreme foreshortening that hurts ChArUco corner detection.
        #
        # Both grids are DENSE because tilt_x=180 forces wrist-flip
        # configurations and only ~5% of grid points pass IK.
        d_min, d_max = _HEMI_DISTANCE_RANGE_M
        ev_min, ev_max = _HEMI_ELEVATION_RANGE_DEG
        az_world_range = _hemi_azimuth_world_range_deg()
        if _CALIBRATION_USE_CONTINUOUS:
            # Sobol low-discrepancy sampling — same volume as the discrete
            # grids below, but provably uniform 3D coverage. Avoids the
            # "ring" artefacts where discrete grids put grid corners at
            # specific azimuths and IK feasibility correlates with those
            # corners; survivors come from genuine reachability rather
            # than grid alignment. Bootstrap's elevation range is bumped
            # up (45° lower bound, 88° upper) to match the dense bootstrap
            # config's intent of capturing mostly-overhead views.
            bootstrap_params = HemisphereParams(
                n_candidates=_BOOTSTRAP_N_CANDIDATES,
                distance_range_m=(0.22, 0.34),
                elevation_range_deg=(30.0, 88.0),
                azimuth_range_deg=az_world_range,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
            main_params = HemisphereParams(
                n_candidates=_MAIN_N_CANDIDATES,
                distance_range_m=(d_min, d_max),
                elevation_range_deg=(ev_min, ev_max),
                azimuth_range_deg=az_world_range,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
        else:
            # Legacy discrete grids — kept for bisection if continuous
            # ever regresses. Both grids are dense because tilt_x=180
            # forces wrist-flip configurations and only ~15% of grid
            # points pass IK.
            bootstrap_params = HemisphereParams(
                distances_m=(0.22, 0.26, 0.30, 0.34),
                elevations_deg=(30.0, 45.0, 60.0, 75.0, 88.0),
                azimuth_counts=(16, 14, 12, 10, 8),  # 60 az per distance × 4 = 240
                azimuth_range_deg=az_world_range,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
            main_params = HemisphereParams(
                distances_m=tuple(np.linspace(d_min, d_max, 6).tolist()),
                elevations_deg=tuple(np.linspace(ev_min, ev_max, 6).tolist()),
                azimuth_counts=(20, 16, 12, 9, 6, 4),  # 67 az per distance × 6 = 402
                azimuth_range_deg=az_world_range,
                max_joint_change_deg=180.0,
                workspace_xy_max_m=0.55,
            )


        bg = HullFilteredPoseGenerator(
            robot=robot, mount=cold_start,
            target_world=target_world,
            params=bootstrap_params,
        )
        # 16 bootstrap candidates instead of 8 — at observed ~25-50% detection
        # rate, 8 sometimes gives <4 successful detections (orchestrator's
        # consensus threshold). 16 gives ~4-8 detections, well above threshold.
        # Cost: a few extra seconds of robot motion at calibration start.
        bcands, _ = bg.generate(max_count=16)
        boot_cfg = tuple(
            tuple(np.degrees(c.joint_angles_rad).tolist()) for c in bcands
        )

        # Pose-generator factory: injects our HullFilteredPoseGenerator
        # subclass into the orchestrator's `_collect_hemisphere` so the same
        # six-stage filter pipeline (hull, floor, self-collision, occlusion,
        # farthest-first, trajectory) runs in both the bootstrap and main
        # passes. Replaces the previous monkey-patch on
        # `orchestrator.PoseGenerator`.
        def _hull_filtered_factory(robot, mount, target_world, params):
            return HullFilteredPoseGenerator(
                robot=robot,
                mount=mount,
                target_world=target_world,
                params=params,
            )

        # Sample count trades calibration quality for wallclock time; 12 is
        # enough for a watchable demo. Settle time is short in sim (the
        # controller's motion physics is already discrete) but bumped on
        # real hardware so PAROL6's belt-driven joints + camera bracket
        # flex have time to fully stabilise before each capture.
        settle_s = _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S
        config = OrchestratorConfig(
            bootstrap_joint_configs_deg=boot_cfg,
            target_sample_count=12,
            min_sample_count=6,
            enable_second_pass=False,
            settle_time_s=settle_s,
            hemisphere=main_params,
            pose_generator_factory=_hull_filtered_factory,
        )

        orchestrator = CalibrationOrchestrator(
            robot=robot, client=client,
            camera=camera, detector=detector,
            cold_start_mount=cold_start, config=config,
        )

        t0 = time.perf_counter()
        output = orchestrator.calibrate()
        wall_s = time.perf_counter() - t0

        if output.failed:
            _post_status(f"Calibration FAILED: {output.failure_reason}")
        else:
            gt = ground_truth_mount.T_cam2flange[:3, 3]
            cal = output.mount.T_cam2flange[:3, 3]
            pos_err = float(np.linalg.norm(gt - cal)) * 1000.0
            _post_status(
                f"DONE ({wall_s:.1f}s, {output.n_samples_collected} samples). "
                f"Best={output.best_method}, error={pos_err:.2f}mm"
            )
            _state["calibrated_mount"] = output.mount

            # Post-calibration: drive to a "view board" pose using the
            # freshly-calibrated mount (much more accurate than cold-start).
            # Camera ends up looking straight down at the board centre from
            # ~32 cm — the entire board fits comfortably in the FOV (which
            # is ~33 cm wide at that distance), giving the user a clean
            # visual confirmation of the calibration result. If no overhead
            # pose is reachable for whatever reason, fall back to home.
            try:
                from parol6_vision.calibration.board import (  # noqa: PLC0415
                    BOARD_TABLET_30MM as _view_cfg,
                )
                from parol6_vision.calibration.view_pose import (  # noqa: PLC0415
                    DEFAULT_MARGIN_PX as _VIEW_MARGIN_PX,
                    DEFAULT_TARGET_FILL as _VIEW_TARGET_FILL,
                    board_corners_world,
                    score_view_candidate,
                    view_distance_range,
                )
                from parol6_vision.camera.intrinsics import Intrinsics  # noqa: PLC0415

                view_centre_local = np.array(
                    [
                        _view_cfg.squares_x * _view_cfg.square_length / 2.0,
                        _view_cfg.squares_y * _view_cfg.square_length / 2.0,
                        0.0,
                        1.0,
                    ],
                    dtype=np.float64,
                )
                view_target = (_T_BOARD2BASE @ view_centre_local)[:3]
                view_corners_world = board_corners_world(_T_BOARD2BASE, _view_cfg)
                view_intrinsics = Intrinsics(
                    fx=_INTR_FX, fy=_INTR_FY, cx=_INTR_CX, cy=_INTR_CY,
                    width=_INTR_W, height=_INTR_H,
                    dist_coeffs=np.zeros(5, dtype=np.float64),
                )
                # Hemisphere range derived from intrinsics + board geometry —
                # the score function picks the candidate that best frames the
                # board (entire-board-in-frame is the priority, vertical
                # overhead-ness + target-fill quality break ties). Wide range
                # gives the pose generator IK headroom; PAROL6 typically can
                # only reach the near end of the perfect-fit shell, so the
                # "closest-to-fit" fallback in the scoring is what usually
                # gets selected on this arm.
                d_min, d_max = view_distance_range(view_intrinsics, _view_cfg)
                view_params = HemisphereParams(
                    n_candidates=512,
                    distance_range_m=(d_min, d_max),
                    elevation_range_deg=(50.0, 89.0),
                    azimuth_range_deg=(-180.0, 180.0),
                    workspace_xy_max_m=0.55,
                    max_joint_change_deg=180.0,
                )
                view_gen = PoseGenerator(
                    robot=robot, mount=output.mount,
                    target_world=view_target, params=view_params,
                )
                view_cands, _ = view_gen.generate(max_count=None)

                if view_cands:
                    scored = [
                        (
                            score_view_candidate(
                                c, output.mount, view_target,
                                view_corners_world, view_intrinsics,
                                target_fill=_VIEW_TARGET_FILL,
                                margin_px=_VIEW_MARGIN_PX,
                            ),
                            c,
                        )
                        for c in view_cands
                    ]
                    scored.sort(key=lambda sc: sc[0][0], reverse=True)
                    (best_score, best_info), best_view = scored[0]
                    n_in_frame = sum(
                        1 for (_, info), _ in scored if info["in_frame"]
                    )
                    logger.info(
                        "post-calibration view pose: %d/%d in-frame, picked "
                        "score=%.3f vertical=%.3f in_frame=%s "
                        "max_fill=%.2f distance=%.0fmm "
                        "(d_range=%.0f-%.0fmm, %d reachable)",
                        n_in_frame, len(view_cands),
                        best_score, best_info["vertical"],
                        best_info["in_frame"], best_info["max_fill"],
                        best_info["distance_m"] * 1000,
                        d_min * 1000, d_max * 1000, len(view_cands),
                    )
                    _post_status("Calibrated — moving to view-board pose")
                    try:
                        raw_client.move_j(
                            angles=list(
                                np.degrees(best_view.joint_angles_rad).tolist()
                            ),
                            speed=0.3, accel=0.5, wait=True, timeout=20.0,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "post-calibration: view-pose move_j raised "
                            "%s: %s — going home instead",
                            type(e).__name__, e,
                        )
                        try:
                            raw_client.home(wait=True, timeout=30.0)
                        except Exception as e2:  # noqa: BLE001
                            logger.warning(
                                "post-calibration: home() also raised %s: %s",
                                type(e2).__name__, e2,
                            )
                else:
                    logger.info(
                        "post-calibration: no reachable view-board pose "
                        "(0/%d candidates, d_range=%.0f-%.0f mm), going home",
                        view_params.n_candidates, d_min * 1000, d_max * 1000,
                    )
                    _post_status("Calibrated — view pose unreachable, homing")
                    try:
                        raw_client.home(wait=True, timeout=30.0)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "post-calibration: home() raised %s: %s",
                            type(e).__name__, e,
                        )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "post-calibration view pose failed (%s: %s) — "
                    "robot left at last calibration pose",
                    type(e).__name__, e,
                )

    except Exception as e:  # noqa: BLE001
        # The orchestrator's own `finally` calls set_tcp_offset to restore the
        # previous offset; if the user pressed STOP partway through, the
        # controller is halted and that call raises MotionError. Treat
        # exceptions during a stop request as a clean stop, not a crash.
        if _state.get("stop_requested"):
            logger.info("Calibration stopped by user (caught %s: %s)",
                        type(e).__name__, e)
            _post_status("Calibration stopped by user")
        else:
            logger.exception("Calibration thread crashed")
            _post_status(f"ERROR: {e}")
    finally:
        # Stop the RealSenseCamera if we started one in real-hardware mode.
        real_cam = _state.get("real_camera")
        if real_cam is not None:
            try:
                real_cam.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("RealSenseCamera stop failed: %s", e)
            _state["real_camera"] = None
        _state["is_running"] = False
