"""Calibration runner (background thread)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import settings
from .collision import _build_collision_manager, _self_collides, _trajectory_collides
from .constants import (
    _BOARD_TARGET_OFFSETS_LOCAL,
    _BOOTSTRAP_N_CANDIDATES,
    _CALIBRATION_USE_CONTINUOUS,
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
    _state_lock,
    current_board_config,
    save_recovered_board_pose,
)
from .overlays import refresh_board_dependent_overlays
from .workspace import _ensure_workspace_envelope, envelope_contains

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration runner (background thread)
# ---------------------------------------------------------------------------


def _calibration_thread() -> None:
    """Run calibration end-to-end and animate via the controller's
    50 Hz status broadcasts — no joint-update plumbing here.
    """
    # Lazy import — break the panel<->calibration_thread cycle.
    from .panel import _post_status  # noqa: PLC0415

    # Tracked so the ``finally`` can close the UDP socket + inner
    # asyncio loop and avoid leaks across runs.
    raw_client: Any = None

    try:
        # Lazy imports.
        from parol6 import Robot, RobotClient  # noqa: PLC0415

        from parol6_vision.calibration.board import BoardDetector  # noqa: PLC0415
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

        # Sim vs. real dispatched off the page-level robot toggle.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
            is_sim_mode = bool(robot_state.simulator_active)
        except Exception:  # noqa: BLE001
            is_sim_mode = True

        T_BOARD2BASE = _T_BOARD2BASE
        if is_sim_mode:
            # Sim ground truth — cold start + small fixed perturbation so
            # the simulated calibration has something to converge to.
            cam_translate = settings.cam_mount_translate_mm
            cam_tilt = settings.cam_mount_tilt_deg
            ground_truth_mount = CameraMount.from_eyeball_estimate(
                x_mm=cam_translate[0] + 2.0,
                y_mm=cam_translate[1] + 2.0,
                z_mm=cam_translate[2] - 2.0,
                tilt_x_deg=cam_tilt[0] + 2.0,
                tilt_y_deg=cam_tilt[1] - 1.0,
                tilt_z_deg=cam_tilt[2],
            )
            intrinsics = Intrinsics(
                fx=float(settings.intr_fx), fy=float(settings.intr_fy),
                cx=float(settings.intr_cx), cy=float(settings.intr_cy),
                width=int(settings.intr_width), height=int(settings.intr_height),
                dist_coeffs=np.zeros(5, dtype=np.float64),
            )
        else:
            ground_truth_mount = None  # not used in real mode
            intrinsics = None  # set below from the actual RealSense device

        robot = Robot()

        # Each move_j blocks until motion completes; the controller's
        # 50 Hz status broadcasts drive the URDF scene.
        raw_client = RobotClient(host="127.0.0.1", port=5001)

        # STOP-aware wrapper — halt() only aborts the in-flight motion,
        # so without this the orchestrator queues the next pose anyway.
        # Returning -1 from move_j makes the orchestrator give up cleanly.
        class _HaltableClient:
            """STOP-aware wrapper around motion-bearing calls."""

            _MOTION_METHODS: frozenset[str] = frozenset({
                "move_j", "move_l", "move_p", "move_c", "move_s", "home",
            })

            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def move_j(self, *args: Any, **kwargs: Any) -> int:
                if _state.get("stop_requested"):
                    return -1
                return self._inner.move_j(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                attr = getattr(self._inner, name)
                if name in self._MOTION_METHODS and callable(attr):
                    def _gated(*a: Any, **kw: Any) -> Any:
                        if _state.get("stop_requested"):
                            return -1
                        return attr(*a, **kw)
                    return _gated
                return attr

        # When helper-mode is on, wrap so each significant J0 motion gets a
        # direction warning + pre-move pause. Passthrough when disabled.
        from .helper_mode import maybe_wrap_helper_mode  # noqa: PLC0415
        helper_client = maybe_wrap_helper_mode(raw_client)

        client = _HaltableClient(helper_client)
        # Stash the raw client so STOP can call halt() directly (bypasses
        # both helper-mode and haltable wrappers; halt() must not pause).
        _state["client"] = raw_client

        # HALT latches the controller into DISABLED until a RESUME, so a
        # prior STOP / Localise / Hover would block the first move_j.
        try:
            raw_client.resume()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "calibration start: resume() raised %s: %s "
                "- first move may fail if controller is in disabled state",
                type(e).__name__, e,
            )

        # FK-based flange pose — ``client.pose("WRF")`` returns TCP (tool
        # offset baked in) and ``T_cam2flange`` is flange-relative, so
        # mixing them puts the camera ~105 mm off in sim.
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

        intr_w = int(settings.intr_width)
        intr_h = int(settings.intr_height)
        cfg = current_board_config()
        if is_sim_mode:
            camera: Any = VirtualCamera(
                intrinsics=intrinsics,
                image_width=intr_w,
                image_height=intr_h,
                ground_truth_mount=ground_truth_mount,
                flange_pose_provider=flange_pose,
                board=VirtualBoard(config=cfg, T_board2base=T_BOARD2BASE),
                noise_std=0.0,
            )
            logger.info("calibration camera: VirtualCamera (simulator mode)")
        else:
            from parol6_vision.camera.realsense import RealSenseCamera  # noqa: PLC0415
            camera = RealSenseCamera(
                width=intr_w,
                height=intr_h,
                fps=30,
                enable_depth=False,  # calibration only needs color
                enable_color=True,
            )
            # Stash before start() — if start() partially claims the
            # device, ``finally`` needs the handle to clean up.
            _state["real_camera"] = camera
            camera.start()
            # AE warm-up — D435 auto-exposure takes ~30 frames after start
            # to converge. Without this, the first bootstrap pose's 8-frame
            # average spans the AE-tuning window, mixing over/under-exposed
            # frames into the mean. 1-second dead cost; only on hardware.
            for _ in range(30):
                try:
                    camera.capture_color()
                except Exception as e:  # noqa: BLE001
                    logger.debug("AE warm-up capture failed: %s", e)
                    break
            intrinsics = camera.intrinsics
            logger.info(
                "calibration camera: RealSenseCamera (real-hardware mode), "
                "intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f, dist=%s",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                intrinsics.dist_coeffs.tolist(),
            )
        detector = BoardDetector(cfg)
        cs_translate = settings.cam_mount_translate_mm
        cs_tilt = settings.cam_mount_tilt_deg
        cold_start = CameraMount.from_eyeball_estimate(
            x_mm=cs_translate[0],
            y_mm=cs_translate[1],
            z_mm=cs_translate[2],
            tilt_x_deg=cs_tilt[0],
            tilt_y_deg=cs_tilt[1],
            tilt_z_deg=cs_tilt[2],
        )

        # Hemisphere CENTRE + look-at target. Honours the optional override
        # so board rotation doesn't shift the hemisphere.
        target_world = _hemi_centre_world()

        # Diagnostic — flag when the board centre is outside the hull.
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

        # PoseGenerator filter pipeline (see HullFilteredPoseGenerator
        # below): hull → floor → self-collision → occlusion → viewing
        # angle → farthest-first → trajectory self-collision.
        FLOOR_Z_MIN_M = 0.005  # safety margin above the workbench
        TCP_OFFSET_FLANGE = np.array([0.0, 0.0, -0.105, 1.0])  # SSG-48 TCP

        # Built once. ``_mesh_collision_enabled`` is the Settings master
        # gate; ``enable_self_collision_check`` only applies inside it.
        from .collision import _mesh_collision_enabled  # noqa: PLC0415

        _collision_active = (
            bool(settings.enable_self_collision_check)
            and _mesh_collision_enabled()
        )
        collision_mgr_pair = (
            _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
            if _collision_active else None
        )
        occlusion_meshes = (
            _build_occlusion_mesh() if _collision_active else None
        )

        class HullFilteredPoseGenerator(PoseGenerator):
            def generate(self, *args, **kwargs):  # type: ignore[override]
                max_count = kwargs.get("max_count", args[0] if args else 8)
                # Over-request so we have a pool to thin from.
                kwargs["max_count"] = max_count * 5

                # Multi-target relaxed look-at — DEFERRED_FEATURES.md §6.
                # Hemisphere CENTRE stays at ``self.target_world`` (may be
                # the override); aim points always derive from
                # ``_T_BOARD2BASE``. Single-target uses (0.5, 0.5) so the
                # invariant holds either way.
                cfg = current_board_config()
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

                # 4. Camera-occlusion filter — multi-ray sampling so a
                # robot link blocking a quadrant of the FOV still rejects
                # even when the centre is clear.
                if occlusion_meshes is not None:
                    cold_T = self.mount.T_cam2flange
                    cfg = current_board_config()
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

                # 4.5. Viewing-angle filter — reject when the optical
                # axis is more than the threshold off the board normal.
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
                        # abs() — board normal orientation is incidental;
                        # we care about the angle.
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

                # 6. Pairwise trajectory self-collision. First candidate
                # accepted unconditionally — bootstrap's final pose, not
                # HOME, is the entry config and is unknown here.
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

        # Hemisphere params. The MAIN pass drives the user tunables (may
        # include partial-board views; orchestrator only needs ≥6 corners).
        # The BOOTSTRAP pass uses conservative bounds so the whole board
        # fits comfortably in FOV for reliable initial localisation. Both
        # grids are dense — tilt_x=180 wrist-flip drops IK pass rate to ~5%.
        d_min, d_max = settings.hemi_distance_range_m
        ev_min, ev_max = settings.hemi_elevation_range_deg
        az_world_range = _hemi_azimuth_world_range_deg()
        if _CALIBRATION_USE_CONTINUOUS:
            # Sobol — same volume, uniform 3D coverage. Discrete grids
            # produce "rings" when IK feasibility correlates with axes.
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
            # Legacy discrete grids — bisection fallback.
            bootstrap_params = HemisphereParams(
                distances_m=(0.22, 0.26, 0.30, 0.34),
                elevations_deg=(30.0, 45.0, 60.0, 75.0, 88.0),
                azimuth_counts=(16, 14, 12, 10, 8),
                azimuth_range_deg=az_world_range,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
            main_params = HemisphereParams(
                distances_m=tuple(np.linspace(d_min, d_max, 6).tolist()),
                elevations_deg=tuple(np.linspace(ev_min, ev_max, 6).tolist()),
                azimuth_counts=(20, 16, 12, 9, 6, 4),
                azimuth_range_deg=az_world_range,
                max_joint_change_deg=180.0,
                workspace_xy_max_m=0.55,
            )


        bg = HullFilteredPoseGenerator(
            robot=robot, mount=cold_start,
            target_world=target_world,
            params=bootstrap_params,
        )
        # 16 candidates — at ~25-50% detection rate, 8 sometimes fell
        # below the orchestrator's 4-detection consensus threshold.
        bcands, _ = bg.generate(max_count=16)
        boot_cfg = tuple(
            tuple(np.degrees(c.joint_angles_rad).tolist()) for c in bcands
        )

        # Factory injecting the filter-pipeline subclass into the
        # orchestrator's bootstrap and main passes.
        def _hull_filtered_factory(robot, mount, target_world, params):
            return HullFilteredPoseGenerator(
                robot=robot,
                mount=mount,
                target_world=target_world,
                params=params,
            )

        # 12 samples — enough for a watchable demo without long wallclock.
        # Sim settle is short (discrete-time physics); real hardware needs
        # longer for belt + bracket flex to die out.
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
            # Sim has a known perturbation as ground truth → report
            # error; real hardware reports the calibrated mount.
            if is_sim_mode and ground_truth_mount is not None:
                gt = ground_truth_mount.T_cam2flange[:3, 3]
                cal = output.mount.T_cam2flange[:3, 3]
                pos_err = float(np.linalg.norm(gt - cal)) * 1000.0
                _post_status(
                    f"DONE ({wall_s:.1f}s, {output.n_samples_collected} "
                    f"samples). Best={output.best_method}, "
                    f"error={pos_err:.2f}mm"
                )
            else:
                cal = output.mount.T_cam2flange[:3, 3] * 1000.0
                _post_status(
                    f"DONE ({wall_s:.1f}s, {output.n_samples_collected} "
                    f"samples). Best={output.best_method}, "
                    f"mount=({cal[0]:.1f}, {cal[1]:.1f}, {cal[2]:.1f}) mm"
                )
            _state["calibrated_mount"] = output.mount

            # Persist the calibrated mount onto the active custom tool so
            # the next activation reads it through the per-tool override.
            try:
                from . import custom_tools as _calib_ct  # noqa: PLC0415
                from scipy.spatial.transform import Rotation as _SciR  # noqa: PLC0415

                T_cf = output.mount.T_cam2flange
                translate_mm = tuple(float(v) * 1000.0 for v in T_cf[:3, 3])
                # Use lowercase "xyz" — ``from_eyeball_estimate``
                # reconstructs R = Rz @ Ry @ Rx; uppercase "XYZ" would
                # silently corrupt the persisted tilt across restarts.
                tilt_deg = tuple(
                    float(v) for v in _SciR.from_matrix(T_cf[:3, :3])
                    .as_euler("xyz", degrees=True)
                )
                _calib_ct.update_active_tool_calibrated_mount(
                    cam_mount_translate_mm=translate_mm,
                    cam_mount_tilt_deg=tilt_deg,
                )
            except Exception as _e:  # noqa: BLE001
                logger.debug(
                    "calibrated mount auto-save to custom tool failed: %s",
                    _e,
                )

            # Apply the estimator's refined full pose to ``_T_BOARD2BASE``
            # — tighter than localise can produce because it integrates
            # bootstrap inliers + every hemisphere sample. The estimator
            # tracks the board CENTRE; ``_T_BOARD2BASE`` stores the
            # corner-anchored origin, so we convert via the half-board
            # offset in board-local frame.
            try:
                if output.board_pose_world is not None:
                    new_pose = np.asarray(
                        output.board_pose_world, dtype=np.float64,
                    )
                    new_R = new_pose[:3, :3]
                    new_centre = new_pose[:3, 3]

                    _cfg = current_board_config()
                    center_local = np.array(
                        [
                            _cfg.squares_x * _cfg.square_length / 2.0,
                            _cfg.squares_y * _cfg.square_length / 2.0,
                            0.0,
                        ],
                        dtype=np.float64,
                    )
                    # World origin = centre - R @ centre-in-board-local.
                    new_origin = new_centre - new_R @ center_local

                    old_R = _T_BOARD2BASE[:3, :3].copy()
                    old_origin = _T_BOARD2BASE[:3, 3].copy()
                    old_centre = old_origin + old_R @ center_local

                    pos_shift_mm = float(
                        np.linalg.norm(new_centre - old_centre)
                    ) * 1000.0
                    # Geodesic rotation delta.
                    R_diff = new_R @ old_R.T
                    cos_theta = float(
                        np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
                    )
                    rot_shift_deg = float(np.degrees(np.arccos(cos_theta)))

                    # Apply unconditionally on success — estimator data is
                    # stronger evidence than the localise-set pose.
                    new_T = np.eye(4, dtype=np.float64)
                    new_T[:3, :3] = new_R
                    new_T[:3, 3] = new_origin
                    _T_BOARD2BASE[:] = new_T
                    # Persist (mode-tagged) for refresh + restart restoration.
                    save_recovered_board_pose(new_T)

                    with _state_lock:
                        _state["trajectory_collision_mgr_pair"] = None
                        _state["reach_generation"] = (
                            int(_state.get("reach_generation", 0)) + 1
                        )
                        _state["reachable_candidates"] = []
                    refresh_board_dependent_overlays()

                    log_fn = (
                        logger.info
                        if pos_shift_mm > 0.5 or rot_shift_deg > 0.1
                        else logger.debug
                    )
                    log_fn(
                        "Calibration: applied refined board pose to "
                        "_T_BOARD2BASE — centre (%.3f, %.3f, %.3f) m, "
                        "shift %.1f mm / %.2f deg from previous",
                        float(new_centre[0]),
                        float(new_centre[1]),
                        float(new_centre[2]),
                        pos_shift_mm,
                        rot_shift_deg,
                    )
            except Exception as _e:  # noqa: BLE001
                logger.warning(
                    "Calibration: applying refined board pose to overlay failed: %s",
                    _e,
                )

            # Post-cal courtesy move: drive to an overhead view-board
            # pose with the calibrated mount. Falls back to home() when
            # no candidate is reachable.
            try:
                from parol6_vision.calibration.view_pose import (  # noqa: PLC0415
                    DEFAULT_MARGIN_PX as _VIEW_MARGIN_PX,
                    DEFAULT_TARGET_FILL as _VIEW_TARGET_FILL,
                    board_corners_world,
                    score_view_candidate,
                    view_distance_range,
                )
                from parol6_vision.camera.intrinsics import Intrinsics  # noqa: PLC0415

                _view_cfg = current_board_config()
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
                    fx=float(settings.intr_fx), fy=float(settings.intr_fy),
                    cx=float(settings.intr_cx), cy=float(settings.intr_cy),
                    width=int(settings.intr_width), height=int(settings.intr_height),
                    dist_coeffs=np.zeros(5, dtype=np.float64),
                )
                # Range derived from intrinsics + board geometry. Wide so
                # the pose generator has IK headroom; PAROL6 usually only
                # reaches the near end so the scoring fallback picks
                # "closest-to-fit".
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
                    # Honour STOP between calibration end and view move.
                    if _state.get("stop_requested"):
                        _post_status(
                            "Calibration done; STOP pressed, skipping "
                            "view-board pose."
                        )
                    else:
                        view_angles_deg = list(
                            np.degrees(best_view.joint_angles_rad).tolist()
                        )
                        # The view-pose generator uses the unfiltered
                        # PoseGenerator — re-run the standard check here.
                        view_safe = True
                        try:
                            from .collision import (  # noqa: PLC0415
                                validate_joint_trajectory,
                            )
                            from waldo_commander.state import (  # noqa: PLC0415
                                robot_state as _rs,
                            )

                            current_q_deg = list(_rs.angles.deg[:6])
                            # gripper_only=True — post-IK config; arm
                            # self-collision is the IK solver's job.
                            check = validate_joint_trajectory(
                                current_q_deg, view_angles_deg,
                                gripper_only=True,
                            )
                            if not check.get("safe", True):
                                view_safe = False
                                _post_status(
                                    f"Calibrated; view-board pose "
                                    f"skipped, would collide "
                                    f"({check.get('reason', 'collision')})."
                                )
                        except Exception as e:  # noqa: BLE001
                            logger.debug(
                                "post-cal collision pre-check skipped: %s", e,
                            )
                        if view_safe:
                            _post_status("Calibrated, moving to view-board pose")
                            try:
                                raw_client.move_j(
                                    angles=view_angles_deg,
                                    speed=0.3, accel=0.5, wait=True, timeout=20.0,
                                )
                            except Exception as e:  # noqa: BLE001
                                logger.warning(
                                    "post-calibration: view-pose move_j raised "
                                    "%s: %s; going home instead",
                                    type(e).__name__, e,
                                )
                                # Pre-flight check on the home() recovery.
                                home_safe = True
                                try:
                                    from .collision import (  # noqa: PLC0415
                                        validate_joint_trajectory,
                                    )
                                    from parol6.config import (  # noqa: PLC0415
                                        HOME_ANGLES_DEG,
                                    )
                                    from waldo_commander.state import (  # noqa: PLC0415
                                        robot_state as _rs,
                                    )
                                    current_q_deg = list(_rs.angles.deg[:6])
                                    # gripper_only=True — HOME is fixed-valid.
                                    check = validate_joint_trajectory(
                                        current_q_deg, list(HOME_ANGLES_DEG),
                                        gripper_only=True,
                                    )
                                    if not check.get("safe", True):
                                        home_safe = False
                                        _post_status(
                                            f"Post-calibration: home() skipped, "
                                            f"would collide "
                                            f"({check.get('reason', 'collision')})."
                                        )
                                except Exception as e_check:  # noqa: BLE001
                                    logger.debug(
                                        "post-cal home() pre-check skipped: %s",
                                        e_check,
                                    )
                                if home_safe:
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
                    if _state.get("stop_requested"):
                        _post_status(
                            "Calibrated; STOP pressed, skipping home()."
                        )
                    else:
                        _post_status("Calibrated, view pose unreachable, homing")
                        # Pre-flight check on the home() recovery move.
                        home_safe = True
                        try:
                            from .collision import (  # noqa: PLC0415
                                validate_joint_trajectory,
                            )
                            from parol6.config import (  # noqa: PLC0415
                                HOME_ANGLES_DEG,
                            )
                            from waldo_commander.state import (  # noqa: PLC0415
                                robot_state as _rs,
                            )
                            current_q_deg = list(_rs.angles.deg[:6])
                            # gripper_only=True — HOME is fixed-valid.
                            check = validate_joint_trajectory(
                                current_q_deg, list(HOME_ANGLES_DEG),
                                gripper_only=True,
                            )
                            if not check.get("safe", True):
                                home_safe = False
                                _post_status(
                                    f"Calibrated, view pose unreachable; home() "
                                    f"skipped, would collide "
                                    f"({check.get('reason', 'collision')})."
                                )
                        except Exception as e_check:  # noqa: BLE001
                            logger.debug(
                                "post-cal home() pre-check skipped: %s",
                                e_check,
                            )
                        if home_safe:
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
        # The orchestrator's finally calls set_tcp_offset, which raises
        # MotionError after a STOP — treat that case as a clean stop.
        if _state.get("stop_requested"):
            logger.info("Calibration stopped by user (caught %s: %s)",
                        type(e).__name__, e)
            _post_status("Calibration stopped by user")
        else:
            logger.exception("Calibration thread crashed")
            _post_status(f"ERROR: {e}")
    finally:
        # Real-mode camera cleanup.
        real_cam = _state.get("real_camera")
        if real_cam is not None:
            try:
                real_cam.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("RealSenseCamera stop failed: %s", e)
            _state["real_camera"] = None
        # Re-enable the controller before closing. STOP fires ``halt()``,
        # which sets ``state.enabled=False`` server-side (SIM_GOTCHAS §13);
        # without this resume, jog buttons + scripts fail with "Controller
        # disabled" until the next worker's start-of-run resume.
        if raw_client is not None:
            try:
                raw_client.resume()
            except Exception as e:  # noqa: BLE001
                logger.debug("Calibration finally: resume() raised: %s", e)
        # Close the controller socket + inner loop so they don't leak
        # across runs. None when init failed before construction.
        if raw_client is not None:
            try:
                raw_client.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("Calibration: RobotClient.close raised: %s", e)
        # Drop the panel's handle so STOP can't fire through a dead socket.
        if _state.get("client") is raw_client:
            _state["client"] = None
        _state["is_running"] = False
