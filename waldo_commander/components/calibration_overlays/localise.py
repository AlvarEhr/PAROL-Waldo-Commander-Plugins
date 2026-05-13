"""Board auto-localisation thread."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import settings
from .constants import (
    _LOCALISE_J0_STEPS,
    _LOCALISE_SCAN_AZIMUTH_COUNT,
    _LOCALISE_SCAN_AZIMUTH_RANGE_DEG,
    _LOCALISE_SCAN_DISTANCES_M,
    _LOCALISE_SCAN_ELEVATIONS_DEG,
    _LOCALISE_SCAN_TARGETS_M,
    _LOCALISE_SEED_DISTANCE_RANGE_M,
    _LOCALISE_SEED_ELEVATION_RANGE_DEG,
    _SETTLE_TIME_REAL_S,
    _SETTLE_TIME_SIM_S,
)
from .overlays import refresh_board_dependent_overlays
from .state import (
    _T_BOARD2BASE,
    _hemi_centre_world,
    _state,
    _state_lock,
    clear_recovered_board_pose,
    current_board_config,
    rebuild_T_board2base,
    save_recovered_board_pose,
)


# PAROL6 J0 (joint 1) limits from URDF: ±2.1475731 rad (±123.05°). Other
# joints have asymmetric limits but only J0 is varied during the sweep.
# Hardcoded as a fallback if the runtime import fails (kept in sync via
# the upstream PAROL6.urdf — re-verify if upstream URDF changes).
_PAROL6_J0_LIMIT_RAD: tuple[float, float] = (-2.1475731, 2.1475731)
try:
    from parol6.PAROL6_ROBOT import _joint_limits_radian as _RUNTIME_J_LIMITS
    _PAROL6_J0_LIMIT_RAD = (
        float(_RUNTIME_J_LIMITS[0, 0]),
        float(_RUNTIME_J_LIMITS[0, 1]),
    )
except Exception:  # noqa: BLE001
    pass

# parol6's wire validation rejects ``q == limit`` (not just ``q > limit``).
# Clipping to exactly ±123.05° trips ``ValueError: Joint 1 target out of
# range`` at move_j time. Stay 1° inside both ends.
_PAROL6_J0_SWEEP_SAFETY_RAD: float = float(np.radians(1.0))

# Minimum arc (after J0-limit clipping) we still consider "useful" for a
# sweep. Anything below half the D435 horizontal FOV (~27°) produces
# essentially the same coverage as a single static capture.
_LOCALISE_MIN_USEFUL_SWEEP_DEG: float = 30.0

# Mean per-corner reprojection error gate for samples fed to
# ``solve_localise_joint``. Detections above this threshold are mirror-
# prone (solvePnP returns spurious-but-numerically-consistent poses at
# low corner counts). Falls back to using all samples if too few pass.
_LOCALISE_MAX_REPROJ_PX_FOR_JOINT_SOLVE: float = 1.5
_LOCALISE_MIN_FILTERED_SAMPLES_FOR_JOINT_SOLVE: int = 4


def _filter_pose_pairs_by_reproj(
    pose_pairs: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    qualities: list[tuple[int, float]],
    max_reproj_px: float = _LOCALISE_MAX_REPROJ_PX_FOR_JOINT_SOLVE,
    min_filtered: int = _LOCALISE_MIN_FILTERED_SAMPLES_FOR_JOINT_SOLVE,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Filter pose pairs by per-detection reprojection error.

    High reproj-px correlates with solvePnP mirror flips at low corner
    counts — gating them before the joint solve prevents one bad sample
    from poisoning the cv2 minimisation. Falls back to the full input
    if too few samples pass the gate; RANSAC inside
    ``solve_localise_joint`` then handles whatever outliers remain.
    """
    filtered = [
        pp for pp, (_, rpe) in zip(pose_pairs, qualities)
        if rpe <= max_reproj_px
    ]
    n_filt = len(filtered)
    n_all = len(pose_pairs)
    if n_filt < min_filtered:
        logger.info(
            "Joint solve: only %d/%d samples passed reproj<%.1fpx — "
            "using all (RANSAC handles outliers)",
            n_filt, n_all, max_reproj_px,
        )
        return pose_pairs
    if n_filt < n_all:
        logger.info(
            "Joint solve: filtered %d/%d samples by reproj<%.1fpx",
            n_filt, n_all, max_reproj_px,
        )
    return filtered


def _build_localise_seed_targets_xy() -> list[tuple[float, float]]:
    """Workspace-agnostic XY seed targets covering PAROL6's forward reach.

    Each call returns the SAME five anchor points regardless of any prior
    board pose. The previous (configured-pose-anchored) seed pattern
    silently failed whenever the user placed the board outside the
    ±12 cm bubble around the configured location — defeating the whole
    point of "localise."

    Five polar seeds expressed as XY:
      - r=0.22 m, az=0°    : near-centre
      - r=0.28 m, az=0°    : mid-centre
      - r=0.34 m, az=0°    : far-centre
      - r=0.28 m, az=+50°  : far +Y side
      - r=0.28 m, az=-50°  : far -Y side

    Combined with J0-sweep clipping in the sweep loop, each seed
    contributes a useful arc of workspace coverage even when the centred
    sweep would push past PAROL6's ±123° J0 limit.
    """
    radii_az_deg: list[tuple[float, float]] = [
        (0.22,   0.0),
        (0.28,   0.0),
        (0.34,   0.0),
        (0.28,  50.0),
        (0.28, -50.0),
    ]
    return [
        (
            float(r * np.cos(np.radians(az))),
            float(r * np.sin(np.radians(az))),
        )
        for r, az in radii_az_deg
    ]


def _has_rotation_diversity(
    pose_pairs: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    min_axis_spread_deg: float = 8.6,  # match solver default
) -> bool:
    """True when the accumulated pose pairs have enough rotational
    diversity for the joint AX = YB solve. J0-only sweeps are degenerate
    (all relative rotations about base Z); this check fails on them.
    """
    if len(pose_pairs) < 3:
        return False
    try:
        import cv2  # noqa: PLC0415

        axes: list[np.ndarray] = []
        R_list = [pp[0][:3, :3].astype(np.float64) for pp in pose_pairs]
        for i in range(len(R_list)):
            for j in range(i + 1, len(R_list)):
                R_rel = R_list[i] @ R_list[j].T
                rvec, _ = cv2.Rodrigues(R_rel)
                r = rvec.flatten()
                theta = float(np.linalg.norm(r))
                if theta < 1e-6:
                    continue
                axes.append(r / theta)
        if len(axes) < 2:
            return False
        a = np.asarray(axes, dtype=np.float64)
        dots = np.clip(a @ a.T, -1.0, 1.0)
        np.fill_diagonal(dots, 1.0)
        # collapse antiparallel = parallel
        angles_deg = np.degrees(np.arccos(np.abs(dots)))
        iu = np.triu_indices(len(axes), k=1)
        return float(np.max(angles_deg[iu])) >= min_axis_spread_deg
    except Exception:  # noqa: BLE001
        return False

logger = logging.getLogger(__name__)


def _localise_board_thread() -> None:
    """Workspace scan locating the ChArUco board centre in base frame.

    DEFERRED_FEATURES.md §3. The board can be anywhere in the reachable
    workspace; the scan grid is base-relative. Mode (sim vs real-hardware)
    dispatched via ``robot_state.simulator_active``.
    """
    # Lazy import — break the panel<->localise cycle.
    from .panel import _post_status  # noqa: PLC0415

    # "Every press is fresh" — drop any prior recovered pose so the scan
    # treats the board as if its position is unknown. Reverts
    # ``_T_BOARD2BASE`` to the configured pose so the sim VirtualBoard's
    # ground truth lines up with the workspace defaults and the median-
    # consensus fallback isn't biased by a stale recovery.
    try:
        clear_recovered_board_pose()
        rebuild_T_board2base()
        refresh_board_dependent_overlays()
        logger.info(
            "Localise: cleared prior recovered pose; board ground truth "
            "reverted to configured location for this scan.",
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("Localise: clear-prior-pose skipped (%s: %s)",
                     type(e).__name__, e)

    def _check_collision_or_warn(
        target_q_deg: list[float],
        context: str,
    ) -> bool:
        """Pre-flight collision check before dispatching a localise move.

        Returns True when safe; False to skip (with status + toast).
        Fails open on unexpected errors.
        """
        try:
            from .collision import validate_joint_trajectory  # noqa: PLC0415
            from waldo_commander.state import robot_state  # noqa: PLC0415
            from nicegui import ui as _ui  # noqa: PLC0415

            current_q_deg = list(robot_state.angles.deg[:6])
            # gripper_only=True — post-IK config; arm self-collision is
            # the IK solver's job.
            check = validate_joint_trajectory(
                current_q_deg, list(target_q_deg), gripper_only=True,
            )
            if check.get("safe", True):
                return True
            reason = check.get("reason", "unknown")
            pair = check.get("colliding_pair")
            pair_str = (
                f", {pair[0]} <-> {pair[1]}"
                if isinstance(pair, tuple) and len(pair) == 2
                else ""
            )
            msg = (
                f"Localise: {context}: aborted, would collide "
                f"({reason}{pair_str})."
            )
            _post_status(msg)
            loop = _state.get("main_loop")
            nicegui_client = _state.get("nicegui_client")
            if loop is not None:
                def _scheduled_notify(m: str = msg) -> None:
                    # ``ui.notify`` needs a slot context; ``call_soon_threadsafe``
                    # callbacks run with an empty slot stack. Enter the captured
                    # client first (same pattern as ``show_collision_dialog_threadsafe``).
                    try:
                        if nicegui_client is not None:
                            with nicegui_client:
                                _ui.notify(m, color="warning", position="top")
                        else:
                            _ui.notify(m, color="warning", position="top")
                    except Exception as e:  # noqa: BLE001
                        logger.debug("localise toast fired without slot: %s", e)
                try:
                    loop.call_soon_threadsafe(_scheduled_notify)
                except Exception as e:  # noqa: BLE001
                    logger.debug("localise toast schedule failed: %s", e)
            return False
        except Exception as e:  # noqa: BLE001
            logger.debug("localise trajectory pre-check skipped (%s)", e)
            return True

    # Tracked so ``finally`` closes the UDP socket + inner loop.
    raw_client: Any = None

    try:
        import cv2  # noqa: PLC0415

        from parol6 import Robot, RobotClient  # noqa: PLC0415

        from parol6_vision.calibration.board import (  # noqa: PLC0415
            BoardDetector,
            board_pose_to_matrix,
        )
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
        from parol6_vision.calibration.pose_generator import (  # noqa: PLC0415
            HemisphereParams,
            PoseGenerator,
        )
        from parol6_vision.camera.intrinsics import Intrinsics  # noqa: PLC0415
        from parol6_vision.sim.tracing_client import (  # noqa: PLC0415
            _flange_pose_from_client,
        )
        from parol6_vision.sim.virtual_camera import (  # noqa: PLC0415
            VirtualBoard,
            VirtualCamera,
        )

        # Re-build scan poses fresh on each click. Doesn't rely on
        # ``settings.board_translate_m`` being accurate.
        scan_z = settings.board_translate_m[2]
        if bool(settings.surface_enabled):
            scan_z += float(settings.surface_dimensions_m[2])

        cam_translate = settings.cam_mount_translate_mm
        cam_tilt = settings.cam_mount_tilt_deg
        cold_start = CameraMount.from_eyeball_estimate(
            x_mm=cam_translate[0],
            y_mm=cam_translate[1],
            z_mm=cam_translate[2],
            tilt_x_deg=cam_tilt[0],
            tilt_y_deg=cam_tilt[1],
            tilt_z_deg=cam_tilt[2],
        )
        scan_robot = Robot()
        candidates: list = []

        def _vertical_score_for(c: Any, target_w: NDArray[np.float64]) -> float:
            """cos(angle from straight-down) of the camera-to-target ray.
            1.0 = perfectly overhead, 0.0 = horizontal."""
            cam_pos = cold_start.cam_pose_for_flange_pose(
                np.asarray(c.flange_pose),
            )[:3, 3]
            forward = target_w - cam_pos
            fn = float(np.linalg.norm(forward))
            if fn < 1e-6:
                return 0.0
            return float(-(forward / fn)[2])

        # Seed entries (seed_target_xy, seed_q_rad) for the J0 sweep
        # branches. Seeds cover PAROL6's forward workspace independent
        # of any prior board pose — localise should find the board
        # wherever it's placed, with no assumption about the configured
        # or last-recovered location.
        seed_targets_xy = _build_localise_seed_targets_xy()
        logger.info(
            "localise seed targets (workspace-agnostic): %s",
            [(round(x, 3), round(y, 3)) for x, y in seed_targets_xy],
        )
        seed_q_list: list[tuple[tuple[float, float], NDArray[np.float64]]] = []
        if bool(settings.localise_use_j0_sweep):
            for seed_xy in seed_targets_xy:
                seed_target = np.array(
                    [seed_xy[0], seed_xy[1], scan_z], dtype=np.float64,
                )
                seed_params = HemisphereParams(
                    n_candidates=int(settings.localise_seed_n_candidates),
                    distance_range_m=_LOCALISE_SEED_DISTANCE_RANGE_M,
                    elevation_range_deg=_LOCALISE_SEED_ELEVATION_RANGE_DEG,
                    azimuth_range_deg=(-180.0, 180.0),
                    workspace_xy_max_m=0.55,
                    max_joint_change_deg=180.0,
                )
                seed_gen = PoseGenerator(
                    robot=scan_robot, mount=cold_start,
                    target_world=seed_target, params=seed_params,
                )
                try:
                    seed_cands, seed_stats = seed_gen.generate(max_count=None)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "localise seed-gen at (%.2f, %.2f) raised %s: %s",
                        seed_xy[0], seed_xy[1], type(e).__name__, e,
                    )
                    continue
                if not seed_cands:
                    rej = dict(getattr(seed_stats, "rejection_log", {}) or {})
                    logger.info(
                        "localise seed (%.2f, %.2f): 0/%d reachable, rej=%s",
                        seed_xy[0], seed_xy[1], int(settings.localise_seed_n_candidates), rej,
                    )
                    continue
                seed_cands_sorted = sorted(
                    seed_cands,
                    key=lambda c, t=seed_target: _vertical_score_for(c, t),
                    reverse=True,
                )
                seed_q_rad = np.asarray(
                    seed_cands_sorted[0].joint_angles_rad, dtype=np.float64,
                )
                seed_q_list.append((seed_xy, seed_q_rad))
                logger.info(
                    "localise seed (%.2f, %.2f): %d/%d reachable, vertical-score=%.3f, "
                    "joints (deg)=%s",
                    seed_xy[0], seed_xy[1],
                    len(seed_cands), int(settings.localise_seed_n_candidates),
                    _vertical_score_for(seed_cands_sorted[0], seed_target),
                    ["%.1f" % v for v in np.degrees(seed_q_rad)],
                )
            if not seed_q_list:
                _post_status(
                    "Localise: no reachable IK pose at any of "
                    f"{len(seed_targets_xy)} workspace seed targets. "
                    "Try widening _LOCALISE_SEED_DISTANCE_RANGE_M / "
                    "_LOCALISE_SEED_ELEVATION_RANGE_DEG, or check "
                    "for active-tool / camera-mount mis-config."
                )
                return
            _localise_skip_multitarget = True
        else:
            _localise_skip_multitarget = False

        # Multi-target legacy path — pick the most-vertical IK-feasible
        # candidate per scan target (minimises occlusion + oblique views).
        if _localise_skip_multitarget:
            multitarget_iter: tuple = ()
        else:
            multitarget_iter = _LOCALISE_SCAN_TARGETS_M
        for tx, ty in multitarget_iter:
            target_world = np.array([tx, ty, scan_z], dtype=np.float64)
            params = HemisphereParams(
                distances_m=_LOCALISE_SCAN_DISTANCES_M,
                elevations_deg=_LOCALISE_SCAN_ELEVATIONS_DEG,
                azimuth_counts=tuple(
                    _LOCALISE_SCAN_AZIMUTH_COUNT
                    for _ in _LOCALISE_SCAN_ELEVATIONS_DEG
                ),
                azimuth_range_deg=_LOCALISE_SCAN_AZIMUTH_RANGE_DEG,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
            gen = PoseGenerator(
                robot=scan_robot,
                mount=cold_start,
                target_world=target_world,
                params=params,
            )
            try:
                # All feasible candidates — we want the most-vertical one,
                # not the first IK converged on.
                pose_cands, gen_stats = gen.generate(max_count=None)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise scan target (%.2f, %.2f) — pose generation "
                    "raised %s: %s. Skipping target.",
                    tx, ty, type(e).__name__, e,
                )
                continue
            if pose_cands:
                pose_cands_sorted = sorted(
                    pose_cands,
                    key=lambda c: _vertical_score_for(c, target_world),
                    reverse=True,
                )
                candidates.append(pose_cands_sorted[0])
            else:
                rej = getattr(gen_stats, "rejection_log", None) or {}
                n_total = (
                    len(_LOCALISE_SCAN_DISTANCES_M)
                    * len(_LOCALISE_SCAN_ELEVATIONS_DEG)
                    * _LOCALISE_SCAN_AZIMUTH_COUNT
                )
                logger.info(
                    "localise scan target (%.2f, %.2f) — no reachable pose "
                    "across %d candidates (rejections: %s)",
                    tx, ty, n_total, dict(rej),
                )

        # Empty-candidates check only applies to the multi-target path —
        # J0-sweep mode uses ``seed_q_list`` and reports failures elsewhere.
        if not bool(settings.localise_use_j0_sweep):
            if not candidates:
                _post_status(
                    f"Localise: 0 reachable scan poses out of "
                    f"{len(_LOCALISE_SCAN_TARGETS_M)} targets. Check that "
                    f"_LOCALISE_SCAN_TARGETS_M lies within PAROL6's workspace."
                )
                return
            logger.info(
                "localise (multi-target): %d/%d scan poses generated",
                len(candidates), len(_LOCALISE_SCAN_TARGETS_M),
            )
        else:
            logger.info(
                "localise (J0-sweep): %d/%d seed targets reachable",
                len(seed_q_list), len(seed_targets_xy),
            )

        # Sim vs. real dispatched off the page-level robot toggle.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
            is_sim_mode = bool(robot_state.simulator_active)
        except Exception:  # noqa: BLE001
            is_sim_mode = True

        raw_client = RobotClient(host="127.0.0.1", port=5001)
        # When helper-mode is on, wrap the client to warn + pause before
        # each significant J0 rotation. Passthrough when disabled.
        from .helper_mode import maybe_wrap_helper_mode  # noqa: PLC0415
        raw_client = maybe_wrap_helper_mode(raw_client)
        _state["client"] = raw_client

        # HALT latches the controller into DISABLED until a RESUME, so a
        # prior STOP / early-stop would block the first move_j.
        try:
            raw_client.resume()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "localise start: resume() raised %s: %s "
                "- first move may fail if controller is in disabled state",
                type(e).__name__, e,
            )

        # FK-based flange pose — ``client.pose("WRF")`` returns TCP (tool
        # offset baked in) and ``T_cam2flange`` is flange-relative.
        from scipy.spatial.transform import Rotation as _R_fk  # noqa: PLC0415

        def flange_pose_provider() -> NDArray[np.float64] | None:
            angles_deg = raw_client.angles()
            if angles_deg is None or len(angles_deg) < 6:
                return None
            angles_rad = np.radians(np.asarray(angles_deg, dtype=np.float64))
            fk_pose = np.zeros(6, dtype=np.float64)
            scan_robot.fk(angles_rad, fk_pose)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = _R_fk.from_euler("XYZ", fk_pose[3:]).as_matrix()
            T[:3, 3] = fk_pose[:3]
            return T

        intr_w = int(settings.intr_width)
        intr_h = int(settings.intr_height)
        if is_sim_mode:
            intrinsics = Intrinsics(
                fx=float(settings.intr_fx), fy=float(settings.intr_fy),
                cx=float(settings.intr_cx), cy=float(settings.intr_cy),
                width=intr_w, height=intr_h,
                dist_coeffs=np.zeros(5, dtype=np.float64),
            )
            # Same perturbation scheme as ``_calibration_thread``.
            ground_truth_mount = CameraMount.from_eyeball_estimate(
                x_mm=cam_translate[0] + 2.0,
                y_mm=cam_translate[1] + 2.0,
                z_mm=cam_translate[2] - 2.0,
                tilt_x_deg=cam_tilt[0] + 2.0,
                tilt_y_deg=cam_tilt[1] - 1.0,
                tilt_z_deg=cam_tilt[2],
            )
            # VirtualBoard sees current _T_BOARD2BASE as ground truth.
            camera: Any = VirtualCamera(
                intrinsics=intrinsics,
                image_width=intr_w,
                image_height=intr_h,
                ground_truth_mount=ground_truth_mount,
                flange_pose_provider=flange_pose_provider,
                board=VirtualBoard(
                    config=current_board_config(), T_board2base=_T_BOARD2BASE,
                ),
                noise_std=0.0,
            )
            logger.info("localise camera: VirtualCamera (simulator mode)")
        else:
            from parol6_vision.camera.realsense import RealSenseCamera  # noqa: PLC0415
            camera = RealSenseCamera(
                width=intr_w,
                height=intr_h,
                fps=30,
                enable_depth=False,
                enable_color=True,
            )
            # Stash before start() — see ``_calibration_thread``.
            _state["real_camera"] = camera
            camera.start()
            # AE warm-up — D435 auto-exposure takes ~30 frames to converge.
            # Without this the first sweep's early frames span the AE-tuning
            # window, dropping marker detection on over/under-exposed frames.
            for _ in range(30):
                try:
                    camera.capture_color()
                except Exception as e:  # noqa: BLE001
                    logger.debug("AE warm-up capture failed: %s", e)
                    break
            intrinsics = camera.intrinsics
            logger.info(
                "localise camera: RealSenseCamera (real-hardware mode), "
                "intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            )
        # min_corners=4 — localise only needs a rough position; default 6
        # was rejecting frames with enough constraints for solvePnP. Main
        # calibration keeps the default.
        cfg = current_board_config()
        detector = BoardDetector(cfg, min_corners_for_pose=4)
        center_local = np.array(
            [cfg.squares_x * cfg.square_length / 2.0,
             cfg.squares_y * cfg.square_length / 2.0,
             0.0, 1.0],
            dtype=np.float64,
        )

        # CV2-to-OURS board-frame correction (R_x(180°) + (0, H_m, 0)) is
        # baked into ``board.board_pose_to_matrix(detection, cfg)``, so
        # every consumer of solvePnP-recovered ``T_board2cam`` — including
        # the joint AX=YB solver below — sees OURS-frame poses directly.
        # No per-callsite multiplication needed.

        K = intrinsics.as_camera_matrix()
        D = intrinsics.dist_coeffs

        # Full T_board2base SE(3) per detection so we recover rotation
        # too — real-life placements aren't perfectly aligned to the
        # configured _BOARD_RPY_RAD.
        detected_centres: list[NDArray[np.float64]] = []
        detected_poses: list[NDArray[np.float64]] = []
        detected_qualities: list[int] = []  # corner count → best rotation
        # Pose pairs for AX = YB joint solve — recovers board pose AND
        # camera mount simultaneously when rotation diversity is enough.
        # Median consensus is the fallback when the joint solve declines.
        pose_pair_samples: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        # Parallel to ``pose_pair_samples``: (corner_count, mean_reproj_px)
        # per detection. Used to gate joint-solve inputs to high-quality
        # samples before RANSAC. Low corner counts + high reproj are
        # mirror-flip-prone in solvePnP.
        pose_pair_qualities: list[tuple[int, float]] = []

        # Per-sweep frame-outcome diagnostics — distinguishes blank frames,
        # no-markers, sparse ChArUco, and full detections so the summary
        # log shows where detection is breaking down.
        _diag = {
            "blank": 0, "markers0": 0, "markers_some": 0, "detected": 0,
            "marker_counts": [], "charuco_counts": [],
        }
        # Reset dump dir so saved frames always reflect the latest sweep.
        import tempfile  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        _diag_dump_dir: Path = Path(tempfile.gettempdir()) / "localise_frames"
        if _diag_dump_dir.exists():
            try:
                shutil.rmtree(_diag_dump_dir)
            except Exception:  # noqa: BLE001
                pass
        _diag_dump_dir.mkdir(parents=True, exist_ok=True)
        logger.info("localise: diagnostic frames will be saved to %s", _diag_dump_dir)
        _diag_dumped_blank = 0
        _diag_dumped_content = 0
        _DIAG_DUMP_BLANK_LIMIT = 3
        _DIAG_DUMP_CONTENT_LIMIT = 20

        # Independent ArUco + ChArUco probes — distinguish "no markers
        # visible" from "interpolation produced too few corners".
        _aruco_dict_for_probe = cv2.aruco.getPredefinedDictionary(
            cfg.aruco_dict_id,
        )
        _aruco_probe = cv2.aruco.ArucoDetector(
            _aruco_dict_for_probe, cv2.aruco.DetectorParameters(),
        )
        _charuco_board_for_probe = detector.board
        _charuco_probe = cv2.aruco.CharucoDetector(_charuco_board_for_probe)

        def _capture_and_record(label: str) -> bool:
            """Capture, detect, and on success accumulate ``(centre,
            SE(3), corner_count)``. Returns True on a hit. Per-frame
            diagnostic probe + bounded dumps with marker/corner counts
            in the filename.
            """
            nonlocal _diag_dumped_blank, _diag_dumped_content
            try:
                frame = camera.capture_color()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise %s: capture_color failed (%s: %s); skipping",
                    label, type(e).__name__, e,
                )
                return False
            # Pair the flange pose with the frame BEFORE detection / diagnostic
            # probes. ChArUco + solvePnP take 5-30 ms during which J0 keeps
            # rotating in continuous-sweep mode — querying flange afterwards
            # would record a pose past the frame's capture moment, biasing
            # every back-projection downstream.
            try:
                T_flange2base = flange_pose_provider()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise %s: flange-pose query failed (%s: %s)",
                    label, type(e).__name__, e,
                )
                return False
            if T_flange2base is None:
                return False

            frame_std = float(frame.std())
            n_markers = 0
            n_charuco_corners = 0
            is_blank = frame_std < 2.0
            if is_blank:
                _diag["blank"] += 1
            else:
                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if frame.ndim == 3 else frame
                )
                _, marker_ids, _ = _aruco_probe.detectMarkers(gray)
                n_markers = 0 if marker_ids is None else len(marker_ids)
                if n_markers == 0:
                    _diag["markers0"] += 1
                else:
                    _diag["markers_some"] += 1
                    _diag["marker_counts"].append(n_markers)
                    # Probe interpolation independently of solvePnP.
                    try:
                        ch_corners, ch_ids, _, _ = _charuco_probe.detectBoard(gray)
                        if ch_ids is not None:
                            n_charuco_corners = len(ch_ids)
                            _diag["charuco_counts"].append(n_charuco_corners)
                    except Exception:  # noqa: BLE001
                        pass

            # Dump bounded blanks + content frames with detection counts
            # in the filename for at-a-glance triage.
            should_dump = (
                (is_blank and _diag_dumped_blank < _DIAG_DUMP_BLANK_LIMIT)
                or (not is_blank and _diag_dumped_content < _DIAG_DUMP_CONTENT_LIMIT)
            )
            if should_dump:
                if is_blank:
                    idx = _diag_dumped_blank
                    tag = "blank"
                    _diag_dumped_blank += 1
                else:
                    idx = _diag_dumped_content
                    tag = f"m{n_markers}c{n_charuco_corners}"
                    _diag_dumped_content += 1
                safe_label = (
                    label.replace(" ", "_").replace("(", "").replace(")", "")
                    .replace(",", "")
                )
                fname = _diag_dump_dir / f"{tag}_{idx:02d}_{safe_label}.png"
                try:
                    cv2.imwrite(str(fname), frame)
                except Exception as e:  # noqa: BLE001
                    logger.debug("frame dump failed: %s", e)

            detection = detector.detect(frame, K, D)
            if detection is None:
                return False
            T_board2cam = board_pose_to_matrix(detection, cfg)
            T_board2base_obs = T_flange2base @ cold_start.T_cam2flange @ T_board2cam
            board_centre_obs = (T_board2base_obs @ center_local)[:3]
            detected_centres.append(board_centre_obs)
            detected_poses.append(T_board2base_obs)
            detected_qualities.append(int(detection.num_corners_detected))
            # Copies — solvePnP mutates internals; FK output is reused.
            pose_pair_samples.append(
                (T_flange2base.copy(), T_board2cam.copy())
            )
            pose_pair_qualities.append((
                int(detection.num_corners_detected),
                float(detection.reprojection_error_px),
            ))
            _diag["detected"] += 1
            return True

        attempted = 0
        if bool(settings.localise_use_j0_sweep) and bool(settings.localise_continuous_sweep):
            # Continuous-sweep mode — non-blocking move_j per seed,
            # capture during motion, accumulate detections.
            for seed_xy, seed_q_rad in seed_q_list:
                if _state.get("stop_requested"):
                    raw_client.halt()
                    _post_status("Localise stopped by user")
                    return
                # Skip remaining seeds at the SOFTER post-sweep threshold;
                # the in-sweep early-stop uses the harder one to keep
                # capturing more frames mid-motion.
                if len(detected_centres) >= 1:
                    _det_arr = np.asarray(detected_centres, dtype=np.float64)
                    _med = np.median(_det_arr, axis=0)
                    _resid = np.linalg.norm(_det_arr - _med, axis=1)
                    _n_inliers = int(
                        (_resid < float(settings.localise_inlier_threshold_m)).sum()
                    )
                    if _n_inliers >= int(settings.localise_min_inliers_to_proceed):
                        logger.info(
                            "localise: skipping remaining seeds — %d "
                            "detections with %d inliers within %.0f mm "
                            "(min-to-proceed: %d)",
                            len(detected_centres), _n_inliers,
                            float(settings.localise_inlier_threshold_m) * 1000,
                            int(settings.localise_min_inliers_to_proceed),
                        )
                        break

                # Sweep span centred on the seed's J0, clipped to PAROL6's
                # J0 joint range. The pre-fix behaviour skipped the entire
                # sweep when either endpoint exceeded ±123°; with a side-
                # facing seed (e.g., az=+50°) the centred span would push
                # past +123° on one end. Clipping keeps the reachable arc
                # rather than discarding the whole sweep.
                seed_j0_rad = float(seed_q_rad[0])
                half = np.radians(float(settings.localise_j0_sweep_half_deg))
                j0_lo_lim = float(_PAROL6_J0_LIMIT_RAD[0]) + _PAROL6_J0_SWEEP_SAFETY_RAD
                j0_hi_lim = float(_PAROL6_J0_LIMIT_RAD[1]) - _PAROL6_J0_SWEEP_SAFETY_RAD
                low_j0 = max(seed_j0_rad - half, j0_lo_lim)
                high_j0 = min(seed_j0_rad + half, j0_hi_lim)
                arc_rad = high_j0 - low_j0
                if arc_rad < np.radians(_LOCALISE_MIN_USEFUL_SWEEP_DEG):
                    logger.info(
                        "localise sweep (%.2f, %.2f): clipped arc %.1f° < "
                        "%.1f° minimum (seed J0 %.1f° vs J0 limit [%.1f°, "
                        "%.1f°]), skipping",
                        seed_xy[0], seed_xy[1],
                        float(np.degrees(arc_rad)),
                        _LOCALISE_MIN_USEFUL_SWEEP_DEG,
                        float(np.degrees(seed_j0_rad)),
                        float(np.degrees(j0_lo_lim)),
                        float(np.degrees(j0_hi_lim)),
                    )
                    continue
                try:
                    cur_angles = raw_client.angles()
                    cur_j0_rad = (
                        float(np.radians(cur_angles[0]))
                        if cur_angles is not None and len(cur_angles) > 0
                        else seed_j0_rad
                    )
                except Exception:  # noqa: BLE001
                    cur_j0_rad = seed_j0_rad
                # Start at the end closer to current J0 — saves driving
                # across the workspace before any capture.
                if abs(cur_j0_rad - low_j0) <= abs(cur_j0_rad - high_j0):
                    start_j0 = low_j0
                    end_j0 = high_j0
                else:
                    start_j0 = high_j0
                    end_j0 = low_j0
                start_q = seed_q_rad.copy()
                start_q[0] = start_j0
                end_q = seed_q_rad.copy()
                end_q[0] = end_j0
                # Non-J0 joints inherited from the seed IK — if they're
                # already at a limit, the sweep is unsalvageable here.
                if not (
                    scan_robot.check_limits(start_q)
                    and scan_robot.check_limits(end_q)
                ):
                    logger.info(
                        "localise sweep (%.2f, %.2f): non-J0 limits hit even "
                        "after J0 clip (J0 %.1f° → %.1f°), skipping",
                        seed_xy[0], seed_xy[1],
                        float(np.degrees(start_j0)),
                        float(np.degrees(end_j0)),
                    )
                    continue
                logger.info(
                    "localise sweep (%.2f, %.2f): J0 %.1f° → %.1f° "
                    "(starting end is %.1f° away from current %.1f°)",
                    seed_xy[0], seed_xy[1],
                    float(np.degrees(start_j0)), float(np.degrees(end_j0)),
                    float(np.degrees(abs(start_j0 - cur_j0_rad))),
                    float(np.degrees(cur_j0_rad)),
                )

                _post_status(
                    f"Localise: continuous sweep at seed ({seed_xy[0]:.2f}, "
                    f"{seed_xy[1]:.2f}) — moving to start"
                )
                if not _check_collision_or_warn(
                    list(np.degrees(start_q)),
                    f"sweep start ({seed_xy[0]:.2f}, {seed_xy[1]:.2f})",
                ):
                    continue
                rc = raw_client.move_j(
                    angles=list(np.degrees(start_q)),
                    speed=0.3, accel=0.5, wait=True, timeout=15.0,
                )
                if rc < 0:
                    logger.info(
                        "localise sweep start move returned rc=%d; skipping seed", rc,
                    )
                    continue
                time.sleep(_SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S)

                # Pre-motion capture catches a board already in view.
                _capture_and_record(
                    f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) start"
                )

                _post_status(
                    f"Localise: continuous sweep at seed ({seed_xy[0]:.2f}, "
                    f"{seed_xy[1]:.2f}) — scanning"
                )
                # Chunked sweep — halt() doesn't interrupt the currently
                # executing trajectory, so chunking bounds halt-response.
                start_j0_rad = float(start_q[0])
                end_j0_rad = float(end_q[0])
                total_dj0_rad = end_j0_rad - start_j0_rad
                chunk_step_rad = np.radians(float(settings.localise_j0_chunk_deg)) * np.sign(
                    total_dj0_rad
                )
                # Waypoint list — ends exactly at end_j0_rad even when
                # the last step is shorter than a full chunk.
                chunk_targets: list[float] = []
                cur = start_j0_rad
                while abs(end_j0_rad - cur) > abs(chunk_step_rad) * 1.001:
                    cur += chunk_step_rad
                    chunk_targets.append(cur)
                chunk_targets.append(end_j0_rad)

                # Validate the full sweep once — chunking is for halt-
                # responsiveness, not collision; collision state can't
                # change mid-chunk along a J0-only path.
                sweep_end_q = seed_q_rad.copy()
                sweep_end_q[0] = end_j0_rad
                if not _check_collision_or_warn(
                    list(np.degrees(sweep_end_q)),
                    f"full chunked sweep at ({seed_xy[0]:.2f}, {seed_xy[1]:.2f})",
                ):
                    continue

                stop_outer = False
                last_capture = 0.0
                for chunk_idx, chunk_j0 in enumerate(chunk_targets):
                    chunk_q = seed_q_rad.copy()
                    chunk_q[0] = chunk_j0
                    if not scan_robot.check_limits(chunk_q):
                        logger.info(
                            "localise sweep chunk %d/%d at J0=%.1f° out of "
                            "limits, halting and skipping rest of sweep",
                            chunk_idx + 1, len(chunk_targets),
                            float(np.degrees(chunk_j0)),
                        )
                        break

                    cmd_idx = raw_client.move_j(
                        angles=list(np.degrees(chunk_q)),
                        speed=float(settings.localise_sweep_speed), accel=0.5, wait=False,
                        timeout=10.0,
                    )
                    if cmd_idx < 0:
                        logger.info(
                            "localise sweep chunk %d move_j returned rc=%d; "
                            "skipping rest of sweep", chunk_idx + 1, cmd_idx,
                        )
                        break

                    # wait_command on this chunk in a background thread.
                    done_event = threading.Event()
                    wait_error: list[Exception] = []

                    def _wait_for_completion(idx: int = cmd_idx) -> None:
                        try:
                            raw_client.wait_command(idx, timeout=10.0)
                        except Exception as e:  # noqa: BLE001
                            wait_error.append(e)
                        finally:
                            done_event.set()

                    waiter_thread = threading.Thread(
                        target=_wait_for_completion, daemon=True,
                        name=f"localise-waiter-cmd{cmd_idx}",
                    )
                    waiter_thread.start()

                    chunk_started = time.monotonic()
                    while not done_event.is_set():
                        if _state.get("stop_requested"):
                            raw_client.halt()
                            done_event.wait(2.0)
                            _post_status("Localise stopped by user")
                            # Join the waiter so its 10 s ``wait_command``
                            # doesn't orphan a socket on STOP.
                            waiter_thread.join(timeout=1.0)
                            return
                        # Per-chunk watchdog: ~1 s expected, 5 s caps it.
                        if time.monotonic() - chunk_started > 5.0:
                            logger.warning(
                                "localise sweep chunk %d watchdog tripped, halting",
                                chunk_idx + 1,
                            )
                            raw_client.halt()
                            done_event.wait(2.0)
                            stop_outer = True
                            break
                        now = time.monotonic()
                        if now - last_capture >= float(settings.localise_capture_period_s):
                            last_capture = now
                            attempted += 1
                            if _capture_and_record(
                                f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) "
                                f"chunk {chunk_idx + 1}"
                            ):
                                _post_status(
                                    f"Localise: {len(detected_centres)} "
                                    f"detection"
                                    f"{'s' if len(detected_centres) != 1 else ''} "
                                    f"so far ({attempted} captures attempted)"
                                )
                                # Early-stop only when detections AGREE.
                                # Raw count alone let downstream consensus
                                # fail on low-corner disagreements.
                                if (
                                    len(detected_centres)
                                    >= int(settings.localise_early_stop_detections)
                                ):
                                    _det_arr = np.asarray(
                                        detected_centres, dtype=np.float64,
                                    )
                                    _med = np.median(_det_arr, axis=0)
                                    _resid = np.linalg.norm(
                                        _det_arr - _med, axis=1,
                                    )
                                    _n_inliers = int(
                                        (_resid < float(settings.localise_inlier_threshold_m)).sum()
                                    )
                                    if _n_inliers >= int(settings.localise_early_stop_inliers):
                                        logger.info(
                                            "localise: %d detections, %d "
                                            "inliers within %.0f mm — "
                                            "halting sweep early",
                                            len(detected_centres), _n_inliers,
                                            float(settings.localise_inlier_threshold_m) * 1000,
                                        )
                                        raw_client.halt()
                                        done_event.wait(2.0)
                                        stop_outer = True
                                        break
                                    else:
                                        logger.info(
                                            "localise: %d detections but "
                                            "only %d inliers within %.0f mm "
                                            "of median (need ≥%d) — "
                                            "continuing sweep",
                                            len(detected_centres), _n_inliers,
                                            float(settings.localise_inlier_threshold_m) * 1000,
                                            int(settings.localise_early_stop_inliers),
                                        )
                        time.sleep(0.02)
                    waiter_thread.join(timeout=1.0)
                    if wait_error:
                        logger.warning(
                            "localise wait_command for chunk %d raised %s: %s",
                            chunk_idx + 1, type(wait_error[0]).__name__,
                            wait_error[0],
                        )
                        stop_outer = True
                        break
                    if stop_outer:
                        break

                if stop_outer:
                    # Chunk loop exited; the per-seed loop's top-of-loop
                    # check will terminate naturally.
                    pass

                # End-pose capture mirrors the start capture.
                time.sleep(_SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S)
                attempted += 1
                _capture_and_record(
                    f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) end"
                )

                # Per-sweep summary — surfaces marker and ChArUco-corner
                # distributions to locate the detection bottleneck.
                marker_counts = _diag["marker_counts"]
                charuco_counts = _diag["charuco_counts"]
                marker_summary = (
                    f"min={min(marker_counts)}, max={max(marker_counts)}, "
                    f"mean={sum(marker_counts)/len(marker_counts):.1f}"
                    if marker_counts else "(none)"
                )
                charuco_summary = (
                    f"min={min(charuco_counts)}, max={max(charuco_counts)}, "
                    f"mean={sum(charuco_counts)/len(charuco_counts):.1f}"
                    if charuco_counts else "(none)"
                )
                logger.info(
                    "localise sweep (%.2f, %.2f) diagnostics: "
                    "blank=%d (camera bail); markers=0 in %d frames "
                    "(content but no ArUco); markers≥1 in %d frames "
                    "[count %s]; ChArUco corners interpolated %s; "
                    "%d full detections (need ≥%d corners for pose)",
                    seed_xy[0], seed_xy[1],
                    _diag["blank"], _diag["markers0"],
                    _diag["markers_some"], marker_summary, charuco_summary,
                    _diag["detected"], detector.min_corners_for_pose,
                )
                # Reset per-sweep counters; dump counters stay global.
                _diag["blank"] = 0
                _diag["markers0"] = 0
                _diag["markers_some"] = 0
                _diag["detected"] = 0
                _diag["marker_counts"] = []
                _diag["charuco_counts"] = []
        elif bool(settings.localise_use_j0_sweep):
            # Discrete J0-sweep — stop+capture at each step.
            j0_offsets = np.linspace(
                -float(settings.localise_j0_sweep_half_deg), +float(settings.localise_j0_sweep_half_deg),
                _LOCALISE_J0_STEPS,
            )
            for seed_xy, seed_q_rad in seed_q_list:
                if len(detected_centres) >= int(settings.localise_min_detections):
                    break
                for dj0 in j0_offsets:
                    if _state.get("stop_requested"):
                        _post_status("Localise stopped by user")
                        return
                    q_rad = seed_q_rad.copy()
                    q_rad[0] += np.radians(dj0)
                    if not scan_robot.check_limits(q_rad):
                        continue
                    attempted += 1
                    _post_status(
                        f"Localise: discrete step at seed ({seed_xy[0]:.2f}, "
                        f"{seed_xy[1]:.2f}) — dJ0={dj0:+.0f}°"
                    )
                    if not _check_collision_or_warn(
                        list(np.degrees(q_rad)),
                        f"discrete step ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) dJ0={dj0:+.0f}°",
                    ):
                        continue
                    try:
                        rc = raw_client.move_j(
                            angles=list(np.degrees(q_rad)),
                            speed=0.3, accel=0.5, wait=True, timeout=15.0,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "localise discrete move_j failed: %s", e,
                        )
                        continue
                    if rc < 0:
                        continue
                    time.sleep(
                        _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S,
                    )
                    _capture_and_record(
                        f"discrete ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) dJ0={dj0:+.0f}°"
                    )
        else:
            # Multi-target legacy path — fixed pose per scan target.
            for i, c in enumerate(candidates):
                if _state.get("stop_requested"):
                    _post_status("Localise stopped by user")
                    return
                attempted += 1
                _post_status(f"Localise: scan {i + 1}/{len(candidates)} — moving")
                scan_q_deg = list(np.degrees(c.joint_angles_rad).tolist())
                if not _check_collision_or_warn(
                    scan_q_deg,
                    f"scan {i + 1}/{len(candidates)}",
                ):
                    continue
                try:
                    rc = raw_client.move_j(
                        angles=scan_q_deg,
                        speed=0.3, accel=0.5, wait=True, timeout=15.0,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("localise scan %d move_j failed: %s", i, e)
                    continue
                if rc < 0:
                    continue
                time.sleep(
                    _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S,
                )
                _capture_and_record(f"scan {i + 1}/{len(candidates)}")

        if len(detected_centres) < int(settings.localise_min_detections):
            _post_status(
                f"Localise FAILED: only {len(detected_centres)} detections "
                f"from {attempted} captures across "
                f"{len(seed_q_list) if bool(settings.localise_use_j0_sweep) else len(candidates)} "
                "scan path(s) (need "
                f"≥{int(settings.localise_min_detections)}). The board "
                "may be outside PAROL6's forward workspace, occluded, or "
                "the lighting / contrast may be too low for ChArUco "
                "detection. Move the board into the visible workspace "
                "and retry."
            )
            return

        # Stage 2 — refinement around the recovered board centre. Aim
        # point comes from the intermediate joint solve when rotation
        # diversity is sufficient, else the median of detected centres
        # (biased on bad cold-start). Never aims at a hard-coded location.
        if int(settings.localise_refine_n_poses) > 0 and len(detected_centres) >= 1:
            stage1_count = len(detected_centres)

            refine_target: NDArray[np.float64] | None = None
            stage1_joint_result = None
            if (
                len(pose_pair_samples) >= 4
                and _has_rotation_diversity(pose_pair_samples)
            ):
                try:
                    from parol6_vision.calibration.solver import (  # noqa: PLC0415
                        solve_localise_joint,
                    )

                    stage1_joint_result = solve_localise_joint(
                        _filter_pose_pairs_by_reproj(
                            pose_pair_samples, pose_pair_qualities,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Stage-1 intermediate joint solve raised %s: %s",
                        type(e).__name__, e,
                    )

            if stage1_joint_result is not None:
                # Recovered centre in base frame.
                cx, cy, cz = (
                    stage1_joint_result.T_board2base @ center_local
                )[:3]
                refine_target = np.asarray([cx, cy, cz], dtype=np.float64)
                logger.info(
                    "localise stage 2: aim point from intermediate joint solve = %s "
                    "(residual t=%.1fmm)",
                    refine_target.tolist(),
                    stage1_joint_result.residual_translation_spread_m * 1000.0,
                )
                _post_status(
                    f"Localise: stage 1 found {stage1_count} detections, "
                    f"joint solve recovered board at "
                    f"({refine_target[0]:.2f}, {refine_target[1]:.2f}); refining"
                )
            else:
                rough_centre = np.median(
                    np.asarray(detected_centres, dtype=np.float64), axis=0,
                )
                refine_target = rough_centre.copy()
                logger.info(
                    "localise stage 2: aim point = median of detections %s "
                    "(joint solve not yet viable; rotation diversity insufficient)",
                    refine_target.tolist(),
                )
                _post_status(
                    f"Localise: stage 1 found {stage1_count} detections, "
                    f"refining around median "
                    f"({refine_target[0]:.2f}, {refine_target[1]:.2f})"
                )

            # Helper closure — one refinement pass. Generates N azimuth-
            # diverse poses around ``target_world``, drives each, captures
            # one frame each. Returns the count of new detections.
            def _run_refinement_pass(
                target_world: NDArray[np.float64], label: str,
            ) -> int:
                n_before = len(detected_centres)

                refine_params = HemisphereParams(
                    n_candidates=256,
                    distance_range_m=(
                        max(0.20, float(settings.localise_refine_distance_m) - 0.10),
                        float(settings.localise_refine_distance_m) + 0.06,
                    ),
                    elevation_range_deg=(
                        max(40.0, float(settings.localise_refine_elevation_deg) - 30.0),
                        min(85.0, float(settings.localise_refine_elevation_deg) + 5.0),
                    ),
                    azimuth_range_deg=(-180.0, 180.0),
                    workspace_xy_max_m=0.55,
                    max_joint_change_deg=180.0,
                )
                refine_gen = PoseGenerator(
                    robot=scan_robot, mount=cold_start,
                    target_world=target_world, params=refine_params,
                )
                try:
                    refine_cands, _ = refine_gen.generate(max_count=None)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "localise stage 2 (%s): pose generation raised %s: %s",
                        label, type(e).__name__, e,
                    )
                    return 0

                if not refine_cands:
                    logger.info(
                        "localise stage 2 (%s): no reachable poses at target %s",
                        label, np.round(target_world, 3).tolist(),
                    )
                    return 0

                def _refine_score(c) -> float:
                    cam_pos = cold_start.cam_pose_for_flange_pose(
                        np.asarray(c.flange_pose),
                    )[:3, 3]
                    forward = target_world - cam_pos
                    fn = float(np.linalg.norm(forward))
                    return float(-(forward / fn)[2]) if fn > 1e-6 else 0.0

                refine_cands.sort(key=_refine_score, reverse=True)
                top_half = refine_cands[
                    : max(int(settings.localise_refine_n_poses) * 4, 8)
                ]

                def _cam_azimuth_deg(c) -> float:
                    cam_pos = cold_start.cam_pose_for_flange_pose(
                        np.asarray(c.flange_pose),
                    )[:3, 3]
                    rel = cam_pos[:2] - target_world[:2]
                    return float(np.degrees(np.arctan2(rel[1], rel[0])))

                # Greedy farthest-first by azimuth.
                picked_indices: list[int] = [0]
                while (
                    len(picked_indices) < int(settings.localise_refine_n_poses)
                    and len(picked_indices) < len(top_half)
                ):
                    pick_azs = [
                        _cam_azimuth_deg(top_half[i]) for i in picked_indices
                    ]

                    def _min_az_dist(
                        idx: int, refs: list[float] = pick_azs,
                    ) -> float:
                        a = _cam_azimuth_deg(top_half[idx])
                        return min(
                            min(abs(a - r), 360 - abs(a - r)) for r in refs
                        )

                    remaining_indices = [
                        i for i in range(len(top_half)) if i not in picked_indices
                    ]
                    if not remaining_indices:
                        break
                    picked_indices.append(
                        max(remaining_indices, key=_min_az_dist)
                    )
                picks = [top_half[i] for i in picked_indices]

                logger.info(
                    "localise stage 2 (%s): %d poses at target %s, azimuths %s",
                    label, len(picks), np.round(target_world, 3).tolist(),
                    ["%.0f" % _cam_azimuth_deg(p) for p in picks],
                )

                # RESUME in case a prior halt left controller DISABLED.
                try:
                    raw_client.resume()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "localise refine (%s): resume() raised %s: %s",
                        label, type(e).__name__, e,
                    )

                for ri, c in enumerate(picks):
                    if _state.get("stop_requested"):
                        raw_client.halt()
                        _post_status("Localise stopped by user")
                        return len(detected_centres) - n_before
                    refine_q_deg = list(
                        np.degrees(c.joint_angles_rad).tolist()
                    )
                    _post_status(
                        f"Localise refine ({label}): pose {ri + 1}/{len(picks)} - moving"
                    )
                    if not _check_collision_or_warn(
                        refine_q_deg,
                        f"refine ({label}) pose {ri + 1}/{len(picks)}",
                    ):
                        continue
                    try:
                        rc = raw_client.move_j(
                            angles=refine_q_deg,
                            speed=0.3, accel=0.5, wait=True, timeout=15.0,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "localise refine (%s) pose %d move_j failed: %s",
                            label, ri, e,
                        )
                        continue
                    if rc < 0:
                        continue
                    time.sleep(
                        _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S,
                    )
                    _capture_and_record(
                        f"refine ({label}) pose {ri + 1}/{len(picks)}"
                    )

                added = len(detected_centres) - n_before
                logger.info(
                    "localise stage 2 (%s): %d new detections (total %d)",
                    label, added, len(detected_centres),
                )
                return added

            # Three-pass retry:
            #   A. Primary target (rough median or joint-solve centre).
            #   B. Stage 1 seed locations when A finds nothing.
            #   C. Joint-solved centre when it shifts >30 mm from A.
            # Capped at ``max_passes`` to bound runtime on real hardware.
            max_passes = 5
            passes_done = 0

            # Workspace sanity gate on the primary aim. Stage 1's median is
            # biased by cold_start ≠ true mount; if the bias pushes it
            # outside PAROL6's reachable forward workspace, the primary
            # refinement pass will see nothing and waste ~30 s. Skip
            # straight to the (now sorted) seed retries.
            refine_xy = np.asarray(refine_target[:2], dtype=np.float64)
            refine_r = float(np.linalg.norm(refine_xy))
            primary_in_workspace = (
                0.10 <= refine_r <= 0.42
                and refine_xy[0] >= -0.10  # not behind the base
            )
            if primary_in_workspace:
                added_primary = _run_refinement_pass(refine_target, "primary")
                passes_done += 1
            else:
                logger.info(
                    "localise stage 2: primary aim %s outside reachable "
                    "workspace (r=%.2fm), skipping straight to seed retries",
                    np.round(refine_target[:2], 3).tolist(),
                    refine_r,
                )
                added_primary = 0

            if added_primary == 0 and not _state.get("stop_requested"):
                z_for_alts = float(refine_target[2])
                # Sort seeds by distance from the Stage 1 median (which may
                # be biased by cold_start ≠ true mount, but still preserves
                # rough directional info about where the board lies). On
                # real hardware this typically cuts a 4-retry waste down
                # to 0-1 retries — the closest seed usually wins.
                # refine_xy already computed above for the workspace gate.
                sorted_seeds = sorted(
                    seed_targets_xy,
                    key=lambda s, c=refine_xy: float(
                        np.linalg.norm(np.asarray(s, dtype=np.float64) - c)
                    ),
                )
                logger.info(
                    "localise stage 2 retry order (by distance from "
                    "biased median %s): %s",
                    np.round(refine_xy, 3).tolist(),
                    [(round(sx, 3), round(sy, 3)) for sx, sy in sorted_seeds],
                )
                for seed_xy in sorted_seeds:
                    if (
                        _state.get("stop_requested")
                        or passes_done >= max_passes
                    ):
                        break
                    alt_target = np.array(
                        [seed_xy[0], seed_xy[1], z_for_alts],
                        dtype=np.float64,
                    )
                    # Skip alternatives too close to a target we already swept.
                    if np.linalg.norm(
                        alt_target[:2] - refine_target[:2]
                    ) < 0.05:
                        continue
                    passes_done += 1
                    added_alt = _run_refinement_pass(
                        alt_target,
                        f"retry seed ({seed_xy[0]:.2f}, {seed_xy[1]:.2f})",
                    )
                    if added_alt >= 1:
                        break

            # Pass C: confidence refinement around the joint-solved centre.
            if (
                passes_done < max_passes
                and not _state.get("stop_requested")
                and _has_rotation_diversity(pose_pair_samples)
            ):
                try:
                    from parol6_vision.calibration.solver import (  # noqa: PLC0415
                        solve_localise_joint,
                    )

                    intermediate = solve_localise_joint(
                        _filter_pose_pairs_by_reproj(
                            pose_pair_samples, pose_pair_qualities,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Stage-2 confidence intermediate joint solve raised "
                        "%s: %s", type(e).__name__, e,
                    )
                    intermediate = None
                if intermediate is not None:
                    solved_centre = np.asarray(
                        (intermediate.T_board2base @ center_local)[:3],
                        dtype=np.float64,
                    )
                    shift_mm = float(
                        np.linalg.norm(solved_centre - refine_target)
                    ) * 1000.0
                    if shift_mm > 30.0:
                        logger.info(
                            "localise stage 2: joint-solve recovered "
                            "centre %s (%.0fmm from primary target) — "
                            "running confidence refinement pass",
                            solved_centre.tolist(), shift_mm,
                        )
                        passes_done += 1
                        _run_refinement_pass(
                            solved_centre, "joint-solve refine",
                        )
                    else:
                        logger.info(
                            "localise stage 2: joint-solve centre stable "
                            "(shift %.0fmm) - no extra refinement needed",
                            shift_mm,
                        )

            stage2_total = len(detected_centres) - stage1_count
            logger.info(
                "localise stage 2: %d passes total, %d new detections "
                "(grand total %d)",
                passes_done, stage2_total, len(detected_centres),
            )

        # Don't commit partial localise data after STOP — would silently
        # leak into the next session.
        if _state.get("stop_requested"):
            logger.info(
                "Localise: stop_requested set after Stage 2; skipping "
                "joint solve and state commit",
            )
            _post_status("Localise stopped by user")
            return

        # AX = YB hand-eye joint solve — recovers T_cam2flange AND
        # T_board2base without trusting cold-start. Falls back to median
        # consensus on degenerate pose sets.
        joint_result = None
        if len(pose_pair_samples) >= 4:
            try:
                from parol6_vision.calibration.solver import (  # noqa: PLC0415
                    solve_localise_joint,
                )

                joint_result = solve_localise_joint(
                    _filter_pose_pairs_by_reproj(
                        pose_pair_samples, pose_pair_qualities,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Joint solve raised %s: %s", type(e).__name__, e)
                joint_result = None

        if joint_result is not None:
            # Joint solve succeeded — write the calibrated mount to
            # per-tool storage AND update _T_BOARD2BASE.
            from parol6_vision.calibration.camera_mount import (  # noqa: PLC0415
                CameraMount,
            )
            from scipy.spatial.transform import Rotation as _SciR  # noqa: PLC0415

            T_cam2flange_solved = joint_result.T_cam2flange
            T_board2base_solved = joint_result.T_board2base

            old_origin = _T_BOARD2BASE[:3, 3].copy()
            old_R = _T_BOARD2BASE[:3, :3].copy()
            _T_BOARD2BASE[:] = T_board2base_solved
            # Persist (mode-tagged) for refresh + restart restoration.
            save_recovered_board_pose(T_board2base_solved)

            # ``calibrated_mount`` triggers ``_post_calibration_tick`` to
            # re-draw the frustum at the recovered apex.
            mount_obj = CameraMount(T_cam2flange=T_cam2flange_solved.copy())
            _state["calibrated_mount"] = mount_obj

            # Persist to per-tool storage — see ``_calibration_thread``.
            try:
                from . import custom_tools as _ct  # noqa: PLC0415

                translate_mm = tuple(
                    float(v) * 1000.0 for v in T_cam2flange_solved[:3, 3]
                )
                # Lowercase "xyz" — see ``_calibration_thread`` for rationale.
                tilt_deg = tuple(
                    float(v) for v in _SciR.from_matrix(
                        T_cam2flange_solved[:3, :3]
                    ).as_euler("xyz", degrees=True)
                )
                _ct.update_active_tool_calibrated_mount(
                    cam_mount_translate_mm=translate_mm,
                    cam_mount_tilt_deg=tilt_deg,
                )
                logger.info(
                    "Localise: persisted calibrated mount: translate=%s mm, tilt=%s deg",
                    [round(v, 1) for v in translate_mm],
                    [round(v, 2) for v in tilt_deg],
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Localise: persisting calibrated mount to custom tool failed: %s",
                    e,
                )

            new_centre = (T_board2base_solved @ center_local)[:3]
            delta_mm = float(
                np.linalg.norm(T_board2base_solved[:3, 3] - old_origin)
            ) * 1000.0
            rot_frob = float(
                np.linalg.norm(T_board2base_solved[:3, :3] - old_R, ord="fro")
            )
            rot_delta_deg = float(np.degrees(
                2.0 * np.arcsin(min(1.0, rot_frob / (2.0 * np.sqrt(2))))
            ))
            cam_mm = T_cam2flange_solved[:3, 3] * 1000.0
            _post_status(
                f"Localise OK (joint): {joint_result.n_samples_used} samples, "
                f"board centre ({new_centre[0]:.3f}, {new_centre[1]:.3f}, "
                f"{new_centre[2]:.3f}) m (shift {delta_mm:.1f} mm / "
                f"{rot_delta_deg:.1f} deg), camera mount "
                f"({cam_mm[0]:.1f}, {cam_mm[1]:.1f}, {cam_mm[2]:.1f}) mm"
            )

            with _state_lock:
                _state["trajectory_collision_mgr_pair"] = None
                _state["reach_generation"] = (
                    int(_state.get("reach_generation", 0)) + 1
                )
                _state["reachable_candidates"] = []

            _state["last_localise_ok_at"] = time.time()
            refresh_board_dependent_overlays()
            return

        # Fallback — median consensus against the cold-start mount.
        # Correct only when cold-start is roughly right; fires when the
        # pose set is too small / co-linear for AX = YB.
        logger.info("Localise: falling back to median consensus (joint solve declined)")
        # Median-then-inlier-mean on centres; mirrors the orchestrator's
        # bootstrap RANSAC.
        det_arr = np.asarray(detected_centres, dtype=np.float64)
        median = np.median(det_arr, axis=0)
        residuals = np.linalg.norm(det_arr - median, axis=1)
        inlier_mask = residuals < float(settings.localise_inlier_threshold_m)
        n_inliers = int(inlier_mask.sum())
        if n_inliers < int(settings.localise_min_detections):
            _post_status(
                f"Localise FAILED: {n_inliers} inliers within "
                f"{float(settings.localise_inlier_threshold_m) * 1000:.0f} mm of median "
                f"(need ≥{int(settings.localise_min_detections)}). "
                "Detections too inconsistent — try repositioning the board."
            )
            return

        new_centre = det_arr[inlier_mask].mean(axis=0)

        # Pick the rotation from the best-quality inlier — corner count
        # is a good proxy for solvePnP stability and rotations don't
        # average linearly.
        inlier_indices = np.where(inlier_mask)[0]
        best_inlier_idx = int(max(inlier_indices, key=lambda j: detected_qualities[j]))
        R_detected = np.asarray(detected_poses[best_inlier_idx], dtype=np.float64)[:3, :3]
        old_R = _T_BOARD2BASE[:3, :3].copy()
        # Adopt the detected rotation — user can place the board at any
        # orientation, not just the configured _BOARD_RPY_RAD.
        R_to_use = R_detected
        z_align = float(R_detected[:, 2] @ old_R[:, 2])
        if z_align < 0.5:  # cos(60°) — flag big flips, don't reject
            logger.info(
                "localise: detected board rotation R[:,2] dot "
                "previous R[:,2] = %+.3f (board rotation "
                "significantly differs from previous orientation)",
                z_align,
            )

        # Build the full 4x4 then assign in one `[:] =` — readers under
        # the GIL see either pre or post matrix, never a torn state.
        old_origin = _T_BOARD2BASE[:3, 3].copy()
        center_offset_world = R_to_use @ center_local[:3]
        new_origin = new_centre - center_offset_world

        new_T = np.eye(4, dtype=np.float64)
        new_T[:3, :3] = R_to_use
        new_T[:3, 3] = new_origin
        _T_BOARD2BASE[:] = new_T
        # Persist (mode-tagged) for refresh + restart restoration.
        save_recovered_board_pose(new_T)

        delta_mm = float(np.linalg.norm(new_origin - old_origin)) * 1000.0
        # ‖R_a − R_b‖_F = 2√2 sin(θ/2) — exact identity.
        rot_frob = float(np.linalg.norm(R_to_use - old_R, ord="fro"))
        rot_delta_deg = float(
            np.degrees(2.0 * np.arcsin(min(1.0, rot_frob / (2.0 * np.sqrt(2)))))
        )
        _post_status(
            f"Localise OK: {n_inliers}/{len(detected_centres)} inliers, "
            f"centre ({new_centre[0]:.3f}, {new_centre[1]:.3f}, "
            f"{new_centre[2]:.3f}) m, shift {delta_mm:.1f} mm / {rot_delta_deg:.1f}°"
        )

        # Invalidate caches tied to the previous _T_BOARD2BASE: the
        # collision-mgr pair embeds the old tablet pose and reachable
        # candidates aim at the old centre. Gen bump + candidates clear
        # under ``_state_lock`` keeps the click handler's paired read
        # consistent (sphere names stale-gen rather than wrong-list).
        with _state_lock:
            _state["trajectory_collision_mgr_pair"] = None
            _state["reach_generation"] = (
                int(_state.get("reach_generation", 0)) + 1
            )
            _state["reachable_candidates"] = []

        # Stamp so the Run button skips its "not localised" warning.
        _state["last_localise_ok_at"] = time.time()

        # Refresh visual overlays (asyncio-scheduled).
        refresh_board_dependent_overlays()

    except Exception as e:  # noqa: BLE001
        if _state.get("stop_requested"):
            logger.info("Localise stopped by user (caught %s: %s)",
                        type(e).__name__, e)
            _post_status("Localise stopped by user")
        else:
            logger.exception("Localise thread crashed")
            _post_status(f"Localise ERROR: {e}")
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
        # without an explicit resume, subsequent jogs fail with "Controller
        # disabled" until the next localise/calibration run's start-of-run
        # resume. Mirrors the start-of-run resume so each session leaves the
        # controller in ready state.
        if raw_client is not None:
            try:
                raw_client.resume()
            except Exception as e:  # noqa: BLE001
                logger.debug("Localise finally: resume() raised: %s", e)
        # Close the controller socket + inner loop. None when init
        # failed before construction.
        if raw_client is not None:
            try:
                raw_client.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("Localise: RobotClient.close raised: %s", e)
        # Drop the panel's handle so STOP can't fire through a dead socket.
        if _state.get("client") is raw_client:
            _state["client"] = None
        _state["is_localising"] = False
