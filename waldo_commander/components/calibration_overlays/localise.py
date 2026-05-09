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
    _LOCALISE_SEED_TARGETS_XY,
    _SETTLE_TIME_REAL_S,
    _SETTLE_TIME_SIM_S,
)
from .overlays import refresh_board_dependent_overlays
from .state import _T_BOARD2BASE, _state, current_board_config

logger = logging.getLogger(__name__)


def _localise_board_thread() -> None:
    """Workspace scan that auto-locates the ChArUco board's centre in base frame.

    DEFERRED_FEATURES.md §3 (board auto-localisation). Drives the robot
    through a coarse XY grid of workspace points (`_LOCALISE_SCAN_TARGETS_M`),
    each pose looking down at one point from `_LOCALISE_SCAN_DISTANCE_M`
    above, captures a frame at each, runs ChArUco detection. Successful
    detections (typically 1-2 of N scan poses, depending on where the board
    actually sits) are median-filtered and used to update `_T_BOARD2BASE` in
    place (translation only — rotation stays at the configured
    `_BOARD_RPY_RAD`). The board, hemisphere wireframe, and reachability
    dots all refresh to match.

    The board can be ANYWHERE in the robot's reachable workspace — the
    scan grid is defined relative to the robot base, NOT to the configured
    board location, so it works whether or not the user's existing
    `settings.board_translate_m` matches reality.

    This is a sim-mode runner — uses `VirtualCamera` with the same
    perturbation scheme as `_calibration_thread`. On hardware, swap to
    `RealSenseCamera`.
    """
    # Lazy import to break panel<->localise cycle.
    from .panel import _post_status  # noqa: PLC0415

    def _check_collision_or_warn(
        target_q_deg: list[float],
        context: str,
    ) -> bool:
        """Pre-flight collision check before dispatching a localise move.

        Returns True if the move is safe (caller proceeds); False if
        unsafe (caller skips). Posts both a panel-status update and a
        top-of-page toast on rejection so the user sees the abort even
        when their attention is on the 3D scene.

        Fails open on any unexpected error so a transient glitch in the
        check pipeline doesn't strand a localise run.
        """
        try:
            from .collision import validate_joint_trajectory  # noqa: PLC0415
            from waldo_commander.state import robot_state  # noqa: PLC0415
            from nicegui import ui as _ui  # noqa: PLC0415

            current_q_deg = list(robot_state.angles.deg[:6])
            check = validate_joint_trajectory(
                current_q_deg, list(target_q_deg),
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
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(
                        lambda m=msg: _ui.notify(
                            m, color="warning", position="top",
                        ),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("localise toast schedule failed: %s", e)
            return False
        except Exception as e:  # noqa: BLE001
            logger.debug("localise trajectory pre-check skipped (%s)", e)
            return True

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

        # Build the workspace scan poses fresh each time the button is
        # pressed. Each scan target gets a single overhead pose generated
        # by the standard PoseGenerator (one distance, one elevation, one
        # azimuth — the azimuth is irrelevant at near-overhead elevations).
        # The board can be ANYWHERE in the workspace; we don't depend on
        # the user's `settings.board_translate_m` being accurate.
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

        # Seed-pose list: one entry per (seed_target_xy, seed_q_rad) where
        # the pose generator found a reachable look-down candidate. Used by
        # both continuous-sweep and discrete-J0-sweep branches below; the
        # multi-target legacy path bypasses this entirely.
        seed_q_list: list[tuple[tuple[float, float], NDArray[np.float64]]] = []
        if bool(settings.localise_use_j0_sweep):
            for seed_xy in _LOCALISE_SEED_TARGETS_XY:
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
                    "Localise: no reachable seed pose at any of "
                    f"{len(_LOCALISE_SEED_TARGETS_XY)} configured targets "
                    f"({_LOCALISE_SEED_TARGETS_XY}). Try widening "
                    "_LOCALISE_SEED_DISTANCE_RANGE_M / "
                    "_LOCALISE_SEED_ELEVATION_RANGE_DEG."
                )
                return
            _localise_skip_multitarget = True
        else:
            _localise_skip_multitarget = False

        # Per-target candidate generation with a "look-AT-target" preference:
        # given multiple IK-feasible candidates, pick the one whose camera
        # position is geometrically closest to "directly above the target".
        # That means the camera's optical axis stays nearly vertical and the
        # robot body is on the OPPOSITE side from the workspace — minimising
        # both occlusion and oblique-view ChArUco failures.
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
                # Get ALL feasible candidates, not just the first — we want
                # to choose the BEST one (most-overhead camera position),
                # not the first one IK happened to converge on.
                pose_cands, gen_stats = gen.generate(max_count=None)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise scan target (%.2f, %.2f) — pose generation "
                    "raised %s: %s. Skipping target.",
                    tx, ty, type(e).__name__, e,
                )
                continue
            if pose_cands:
                # Pick the most-vertical candidate (camera optical axis
                # nearest to straight-down). Reuses the helper defined
                # earlier in this thread.
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

        # `candidates` is intentionally empty in J0-sweep mode (we drive the
        # robot from `seed_q_list` instead, building waypoints on-the-fly).
        # The empty-candidates check only applies to the legacy multi-target
        # path; J0-sweep mode reports its own failures earlier (no reachable
        # seed at any target) or later (no detections during the sweep).
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
                len(seed_q_list), len(_LOCALISE_SEED_TARGETS_XY),
            )

        # Mode dispatch: same as _calibration_thread — pick VirtualCamera vs
        # RealSenseCamera based on Waldo-Commander's robot-mode toggle.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
            is_sim_mode = bool(robot_state.simulator_active)
        except Exception:  # noqa: BLE001
            is_sim_mode = True

        raw_client = RobotClient(host="127.0.0.1", port=5001)
        _state["client"] = raw_client

        # client.pose("WRF") returns the TCP pose, which has the tool offset
        # subtracted from the flange — verified empirically: a 105 mm Z
        # offset between WRF and FK-from-joints. The mount transform
        # `cold_start.T_cam2flange` is defined relative to the FLANGE, so
        # feeding TCP poses through it puts the camera 105 mm too close to
        # the workspace and ArUco rejects most markers as the FOV crops the
        # board. Compute the FLANGE pose directly via FK on the joint
        # angles to bypass any tool offset configured on the server.
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
            # Sim ground truth: same perturbation scheme as _calibration_thread.
            ground_truth_mount = CameraMount.from_eyeball_estimate(
                x_mm=cam_translate[0] + 2.0,
                y_mm=cam_translate[1] + 2.0,
                z_mm=cam_translate[2] - 2.0,
                tilt_x_deg=cam_tilt[0] + 2.0,
                tilt_y_deg=cam_tilt[1] - 1.0,
                tilt_z_deg=cam_tilt[2],
            )
            # The VirtualBoard sees the CURRENT _T_BOARD2BASE as ground truth.
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
            # Stash BEFORE start(): see _calibration_thread for rationale.
            _state["real_camera"] = camera
            camera.start()
            intrinsics = camera.intrinsics
            logger.info(
                "localise camera: RealSenseCamera (real-hardware mode), "
                "intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            )
        # Lower min_corners_for_pose for the localise scan — we only need a
        # rough board location estimate (translation, mostly), not a high-
        # accuracy calibration sample. With min_corners=6 (the default) the
        # diagnostic showed many frames where ≥1 ArUco markers were visible
        # but ChArUco interpolation produced fewer than 6 corners and the
        # detector returned None, even though there were enough constraints
        # for a coarse pose. min_corners=4 is the absolute minimum for
        # solvePnP on a planar target; the calibration's main detector
        # keeps the default 6 for accuracy.
        cfg = current_board_config()
        detector = BoardDetector(cfg, min_corners_for_pose=4)
        center_local = np.array(
            [cfg.squares_x * cfg.square_length / 2.0,
             cfg.squares_y * cfg.square_length / 2.0,
             0.0, 1.0],
            dtype=np.float64,
        )

        K = intrinsics.as_camera_matrix()
        D = intrinsics.dist_coeffs

        # Per-detection: keep the full T_board2base SE(3) so we can recover
        # the board's rotation, not just its centre point. This handles
        # tablets that aren't perfectly aligned with the configured
        # _BOARD_RPY_RAD (real-life setups will be slightly tilted / yawed
        # from "perfectly square" no matter how careful the placement).
        detected_centres: list[NDArray[np.float64]] = []
        detected_poses: list[NDArray[np.float64]] = []
        detected_qualities: list[int] = []  # corner count, for picking best rotation

        # Diagnostic state: counts of frame-level outcomes per sweep so the
        # post-sweep summary can tell us WHERE detection is breaking down
        # (blank frames vs no markers vs charuco interpolation vs pose fail).
        # marker_counts / charuco_counts capture distributions to surface
        # the "many markers visible but ChArUco interpolation produces too
        # few corners" failure mode the previous dump couldn't show.
        _diag = {
            "blank": 0, "markers0": 0, "markers_some": 0, "detected": 0,
            "marker_counts": [], "charuco_counts": [],
        }
        # Wipe any previous run's dumps so the saved frames are always
        # the latest sweep's output.
        import tempfile  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        _diag_dump_dir: Path = Path(tempfile.gettempdir()) / "localise_frames"
        if _diag_dump_dir.exists():
            try:
                shutil.rmtree(_diag_dump_dir)
            except Exception:  # noqa: BLE001
                pass  # don't crash localise on a permission glitch
        _diag_dump_dir.mkdir(parents=True, exist_ok=True)
        logger.info("localise: diagnostic frames will be saved to %s", _diag_dump_dir)
        _diag_dumped_blank = 0
        _diag_dumped_content = 0
        # Up to 3 blanks (for reference) + 20 content frames per run.
        _DIAG_DUMP_BLANK_LIMIT = 3
        _DIAG_DUMP_CONTENT_LIMIT = 20

        # Independent ArUco detector for the diagnostic probe — runs the
        # marker layer alone (skips ChArUco interpolation + solvePnP) so we
        # can tell whether the markers were even visible to OpenCV.
        _aruco_dict_for_probe = cv2.aruco.getPredefinedDictionary(
            cfg.aruco_dict_id,
        )
        _aruco_probe = cv2.aruco.ArucoDetector(
            _aruco_dict_for_probe, cv2.aruco.DetectorParameters(),
        )
        # Bare ChArUco detector to probe interpolation independently of
        # solvePnP — some frames have many markers but ChArUco
        # interpolation produces too few corners (e.g. all markers in a
        # single row), and we want to surface that.
        _charuco_board_for_probe = detector.board
        _charuco_probe = cv2.aruco.CharucoDetector(_charuco_board_for_probe)

        def _capture_and_record(label: str) -> bool:
            """Capture one frame, run ChArUco detection, and on success
            append (board centre, full SE(3) pose, corner-count) to the
            detection accumulators. Returns True on a successful detection.

            Per-frame diagnostic probe runs on every capture: frame stats,
            raw ArUco marker count, ChArUco interpolated corner count.
            Saves up to _DIAG_DUMP_CONTENT_LIMIT non-blank frames + a few
            blanks for reference, with marker/corner counts in the
            filename so the user can match a frame to its detection
            outcome at a glance.
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
                    # Probe ChArUco interpolation independently of solvePnP
                    # — distinguishes "markers visible but interpolation
                    # produced too few corners" from "interpolation OK but
                    # solvePnP failed".
                    try:
                        ch_corners, ch_ids, _, _ = _charuco_probe.detectBoard(gray)
                        if ch_ids is not None:
                            n_charuco_corners = len(ch_ids)
                            _diag["charuco_counts"].append(n_charuco_corners)
                    except Exception:  # noqa: BLE001
                        pass

            # Dump strategy: save up to _DIAG_DUMP_BLANK_LIMIT blanks (for
            # reference) and up to _DIAG_DUMP_CONTENT_LIMIT content frames
            # (so we can visually verify what the detector is actually
            # seeing). Filename encodes marker count + ChArUco corner
            # count so a glance at the directory tells the story.
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
            T_board2cam = board_pose_to_matrix(detection)
            T_board2base_obs = T_flange2base @ cold_start.T_cam2flange @ T_board2cam
            board_centre_obs = (T_board2base_obs @ center_local)[:3]
            detected_centres.append(board_centre_obs)
            detected_poses.append(T_board2base_obs)
            detected_qualities.append(int(detection.num_corners_detected))
            _diag["detected"] += 1
            return True

        attempted = 0
        if bool(settings.localise_use_j0_sweep) and bool(settings.localise_continuous_sweep):
            # Continuous-sweep mode — one non-blocking move_j per seed,
            # capture during motion, accumulate detections. Skips to the
            # next seed once enough detections are gathered.
            for seed_xy, seed_q_rad in seed_q_list:
                if _state.get("stop_requested"):
                    raw_client.halt()
                    _post_status("Localise stopped by user")
                    return
                # Post-sweep: skip remaining seeds with the SOFTER threshold
                # (int(settings.localise_min_inliers_to_proceed)). The in-sweep early-stop
                # uses the harder int(settings.localise_early_stop_inliers) to keep
                # capturing more frames mid-motion; once the sweep ENDS, we
                # accept whatever inliers we got rather than restart on a
                # new seed. Avoids re-sweeping when stage 1 already found
                # the board reliably enough — stage 2 will refine further.
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

                # Pick start of sweep based on current J0: whichever end of
                # the [seed_J0 - half, seed_J0 + half] range is CLOSER to the
                # robot's current J0. Eliminates the time-wasting "drive all
                # the way to the far side first" behaviour when the robot
                # starts near home or at the previous sweep's end.
                seed_j0_rad = float(seed_q_rad[0])
                low_j0 = seed_j0_rad - np.radians(float(settings.localise_j0_sweep_half_deg))
                high_j0 = seed_j0_rad + np.radians(float(settings.localise_j0_sweep_half_deg))
                try:
                    cur_angles = raw_client.angles()
                    cur_j0_rad = (
                        float(np.radians(cur_angles[0]))
                        if cur_angles is not None and len(cur_angles) > 0
                        else seed_j0_rad
                    )
                except Exception:  # noqa: BLE001
                    cur_j0_rad = seed_j0_rad
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
                if not (
                    scan_robot.check_limits(start_q)
                    and scan_robot.check_limits(end_q)
                ):
                    logger.info(
                        "localise sweep (%.2f, %.2f): start or end out of joint "
                        "limits (J0 range %.1f° → %.1f°), skipping",
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

                # Capture once at the start before motion so a board
                # already in view at the sweep's starting J0 isn't missed.
                _capture_and_record(
                    f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) start"
                )

                _post_status(
                    f"Localise: continuous sweep at seed ({seed_xy[0]:.2f}, "
                    f"{seed_xy[1]:.2f}) — scanning"
                )
                # CHUNKED SWEEP: break the J0 motion into back-to-back
                # move_j commands of ~float(settings.localise_j0_chunk_deg) each. parol6's
                # halt() clears the command queue but does not interrupt
                # the trajectory currently being executed; chunking gives
                # the user's Stop / E-Stop AND the early-stop-on-enough-
                # detections logic a chance to take effect within ~1 s
                # rather than waiting for the full 180° sweep to finish.
                start_j0_rad = float(start_q[0])
                end_j0_rad = float(end_q[0])
                total_dj0_rad = end_j0_rad - start_j0_rad
                chunk_step_rad = np.radians(float(settings.localise_j0_chunk_deg)) * np.sign(
                    total_dj0_rad
                )
                # Build the list of waypoint J0 angles, ending exactly at
                # end_j0_rad even if the last step is shorter than a full
                # chunk. range/linspace doesn't quite fit here, so step
                # through manually.
                chunk_targets: list[float] = []
                cur = start_j0_rad
                while abs(end_j0_rad - cur) > abs(chunk_step_rad) * 1.001:
                    cur += chunk_step_rad
                    chunk_targets.append(cur)
                chunk_targets.append(end_j0_rad)

                # Whole-sweep collision check: validate the full
                # start_q → end_q trajectory once, rather than each
                # ~20° chunk individually. The chunked dispatch is for
                # halt-responsiveness, not for collision-state
                # transitions — collision state can't change mid-chunk
                # along J0-only motion. Saves ~1 check per chunk.
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
                                # Early stop ONLY if we have enough
                                # detections AND those detections AGREE.
                                # Stopping on raw count alone caused the
                                # downstream consensus to fail when low-
                                # corner detections disagreed by >5 cm.
                                # Requirement: ≥N detections, ≥M of which
                                # are within float(settings.localise_inlier_threshold_m)
                                # of the median.
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
                    # We've already broken out of the chunk loop; let the
                    # outer per-seed loop terminate too via the same
                    # early-stop-detection check at its top.
                    pass

                # Settle briefly after the sweep finishes, then capture once
                # more at the end pose — same rationale as the start
                # capture, with the final J0 still in view of the workspace.
                time.sleep(_SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S)
                attempted += 1
                _capture_and_record(
                    f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) end"
                )

                # Per-sweep diagnostic summary. Surfaces marker count
                # distribution and ChArUco corner-count distribution so
                # we can tell whether interpolation is the bottleneck.
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
                # Reset counters so the next sweep starts clean — we keep
                # the dump counters global to enforce the per-run dump limit.
                _diag["blank"] = 0
                _diag["markers0"] = 0
                _diag["markers_some"] = 0
                _diag["detected"] = 0
                _diag["marker_counts"] = []
                _diag["charuco_counts"] = []
        elif bool(settings.localise_use_j0_sweep):
            # Discrete J0-sweep mode (continuous disabled). Per seed,
            # iterate _LOCALISE_J0_STEPS angles with stop+capture at each.
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
            # Multi-target legacy path — iterate the pre-built `candidates`
            # list as before (one fixed pose per scan target).
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
                f"≥{int(settings.localise_min_detections)}). Board may be outside the "
                f"workspace scan region — check _LOCALISE_SEED_TARGETS_XY "
                "(or _LOCALISE_SCAN_TARGETS_M for legacy mode)."
            )
            return

        # ---- Stage 2: refinement pass around the rough board centre ----
        # Median of Stage-1 detections gives a rough board location. Drive
        # to N high-quality overhead poses centred on that median: gaze
        # straight down at the rough centre, distances/elevations chosen
        # so the board fully fits the FOV. From these poses ChArUco gets
        # the full board in view → 10-20 corners interpolated → much
        # higher-precision detections than the off-axis sweep frames.
        if int(settings.localise_refine_n_poses) > 0 and len(detected_centres) >= 1:
            stage1_count = len(detected_centres)
            rough_centre = np.median(
                np.asarray(detected_centres, dtype=np.float64), axis=0,
            )
            _post_status(
                f"Localise: stage 1 found {stage1_count} detections, "
                f"refining around ({rough_centre[0]:.2f}, {rough_centre[1]:.2f})"
            )
            logger.info(
                "localise stage 2: rough centre = %s, generating %d refinement poses",
                rough_centre.tolist(), int(settings.localise_refine_n_poses),
            )
            refine_target = rough_centre.copy()

            # Sobol-search for IK-feasible refinement candidates around
            # the rough centre. Wider candidate pool than we need so we
            # can pick spatially-diverse picks.
            # Refinement search ranges are wide enough to find SOMETHING
            # reachable for almost any rough_centre in the workspace —
            # narrow ranges fail outright at e.g. (0.16, 0) where
            # PAROL6's wrist-flip kinematics are tight. Wide elevation
            # band (50°-85°) gives the IK enough freedom; wide distance
            # band (0.22-0.36) covers near-base AND far-base targets.
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
                target_world=refine_target, params=refine_params,
            )
            try:
                refine_cands, _ = refine_gen.generate(max_count=None)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise stage 2: pose generation raised %s: %s — "
                    "skipping refinement", type(e).__name__, e,
                )
                refine_cands = []

            if refine_cands:
                # Greedy farthest-first thinning by camera azimuth around
                # rough_centre, so the picks are spread across viewpoints
                # rather than clustered.
                def _refine_score(c) -> float:
                    cam_pos = cold_start.cam_pose_for_flange_pose(
                        np.asarray(c.flange_pose),
                    )[:3, 3]
                    forward = refine_target - cam_pos
                    fn = float(np.linalg.norm(forward))
                    return float(-(forward / fn)[2]) if fn > 1e-6 else 0.0

                # Sort all by vertical-score, take top half, then thin by
                # azimuth diversity to pick int(settings.localise_refine_n_poses).
                refine_cands.sort(key=_refine_score, reverse=True)
                top_half = refine_cands[: max(int(settings.localise_refine_n_poses) * 4, 8)]

                def _cam_azimuth_deg(c) -> float:
                    cam_pos = cold_start.cam_pose_for_flange_pose(
                        np.asarray(c.flange_pose),
                    )[:3, 3]
                    rel = cam_pos[:2] - refine_target[:2]
                    return float(np.degrees(np.arctan2(rel[1], rel[0])))

                # Greedy: pick first, then iteratively pick the one
                # whose azimuth is farthest from the picks-so-far.
                # Track picks by INDEX into top_half — `c in picks` would
                # try element-wise equality on the candidate's numpy
                # fields and raise "truth value of array is ambiguous".
                picked_indices: list[int] = [0]
                while (
                    len(picked_indices) < int(settings.localise_refine_n_poses)
                    and len(picked_indices) < len(top_half)
                ):
                    pick_azs = [
                        _cam_azimuth_deg(top_half[i]) for i in picked_indices
                    ]
                    def _min_az_dist(idx: int, refs: list[float] = pick_azs) -> float:
                        a = _cam_azimuth_deg(top_half[idx])
                        return min(
                            min(abs(a - r), 360 - abs(a - r)) for r in refs
                        )
                    remaining_indices = [
                        i for i in range(len(top_half)) if i not in picked_indices
                    ]
                    if not remaining_indices:
                        break
                    picked_indices.append(max(remaining_indices, key=_min_az_dist))
                picks = [top_half[i] for i in picked_indices]

                logger.info(
                    "localise stage 2: %d refinement poses selected (azimuths %s)",
                    len(picks),
                    ["%.0f" % _cam_azimuth_deg(p) for p in picks],
                )

                # halt() during stage-1 early-stop ALSO disabled the
                # controller; subsequent move_j calls fail with
                # "Controller disabled". resume() re-enables it before
                # we send refinement moves. Without this, all 4
                # refinement move_j calls return error code 50.
                try:
                    raw_client.resume()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "localise refine: resume() raised %s: %s "
                        "— refinement moves may fail",
                        type(e).__name__, e,
                    )

                # Drive to each refinement pose, capture 1 frame.
                for ri, c in enumerate(picks):
                    if _state.get("stop_requested"):
                        raw_client.halt()
                        _post_status("Localise stopped by user")
                        return
                    refine_q_deg = list(np.degrees(c.joint_angles_rad).tolist())
                    _post_status(
                        f"Localise refine: pose {ri + 1}/{len(picks)} — moving"
                    )
                    if not _check_collision_or_warn(
                        refine_q_deg,
                        f"refine pose {ri + 1}/{len(picks)}",
                    ):
                        continue
                    try:
                        rc = raw_client.move_j(
                            angles=refine_q_deg,
                            speed=0.3, accel=0.5, wait=True, timeout=15.0,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "localise refine pose %d move_j failed: %s", ri, e,
                        )
                        continue
                    if rc < 0:
                        continue
                    time.sleep(
                        _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S,
                    )
                    _capture_and_record(f"refine pose {ri + 1}/{len(picks)}")

                stage2_count = len(detected_centres) - stage1_count
                logger.info(
                    "localise stage 2: gathered %d additional detections "
                    "(total %d)",
                    stage2_count, len(detected_centres),
                )
            else:
                logger.info(
                    "localise stage 2: no reachable refinement poses — "
                    "proceeding with %d stage-1 detections",
                    stage1_count,
                )

        # Median-then-inlier-mean on CENTRES: robust to a single outlier
        # without needing full RANSAC. Mirrors the orchestrator's own
        # bootstrap consensus (parol6_vision.calibration.refinement.board_position_ransac).
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

        # Pick rotation from the BEST inlier (most ChArUco corners detected
        # → most stable solvePnP). Averaging rotations across noisy detections
        # is fiddly (rotations don't average linearly); using the single
        # best-quality detection's rotation is robust and simple. With
        # workspace-scan elevations near 90°, the camera-to-board angle is
        # nearly normal so solvePnP rotation accuracy is good.
        inlier_indices = np.where(inlier_mask)[0]
        best_inlier_idx = int(max(inlier_indices, key=lambda j: detected_qualities[j]))
        R_detected = np.asarray(detected_poses[best_inlier_idx], dtype=np.float64)[:3, :3]

        # SIM-MODE workaround for the VirtualCamera Y-flip pose artefact.
        # virtual_camera.py Y-flips the canonical board image before the
        # 4-corner homography warp — that flip is REQUIRED for ArUco to
        # decode the marker bit patterns (verified: removing the flip
        # drops detection from 16 markers to 2, rejected as undecodable).
        # Side effect: the resulting solvePnP pose has the board's local
        # frame Y-mirrored relative to the standard ChArUco convention,
        # which in world frame manifests as the board's +Z axis flipped
        # (pointing world DOWN instead of UP) AND a corresponding shift
        # in the X/Y axes. Detected translation (board CENTRE in base
        # frame) stays accurate because both the canonical and the
        # 4-corner homography are consistent with each other; only the
        # rotation is corrupted by the convention mismatch.
        #
        # On REAL hardware there's no Y-flip (live camera produces a
        # real image of a real board), so the rotation is also correct
        # and we use it. In sim, fall back to the configured RPY which
        # represents the user's known board orientation.
        if is_sim_mode:
            R_to_use = old_R = _T_BOARD2BASE[:3, :3].copy()
            logger.info(
                "localise (sim): using configured board rotation (sim-mode "
                "VirtualCamera Y-flip corrupts detected R); detected R[2, 2] "
                "would have been %+.3f",
                float(R_detected[2, 2]),
            )
        else:
            old_R = _T_BOARD2BASE[:3, :3].copy()
            # Sanity-check the detected rotation against the configured
            # one before adopting it. solvePnP can converge to a
            # mirror solution on a planar target with poor conditioning
            # (low-corner detections at oblique angles); a single bad
            # convergence would otherwise rotate _T_BOARD2BASE wildly
            # and poison every subsequent calibration / hover read. We
            # keep the rotation only when its +Z axis aligns with the
            # configured one's +Z within ~30 deg; otherwise fall back
            # to the configured rotation and warn the user.
            z_align = float(R_detected[:, 2] @ old_R[:, 2])
            if z_align >= 0.866:  # cos(30 deg)
                R_to_use = R_detected
            else:
                R_to_use = old_R
                logger.warning(
                    "localise: detected rotation R[:,2] dot configured "
                    "R[:,2] = %.3f (< 0.866); detected R is wildly "
                    "different from configured. Keeping configured "
                    "rotation; check the physical board orientation.",
                    z_align,
                )
                _post_status(
                    "Localise WARNING: detected board orientation "
                    "doesn't match configured; keeping configured "
                    "rotation. Translation update applied."
                )

        # Build the new full 4x4 transform first, then assign atomically. The
        # previous in-memory rotation/translation are snapshotted for the
        # delta report so it shows the actual change since this update
        # (not the configured-vs-detected delta — that would be misleading
        # on repeat localise calls).
        old_origin = _T_BOARD2BASE[:3, 3].copy()
        center_offset_world = R_to_use @ center_local[:3]
        new_origin = new_centre - center_offset_world

        new_T = np.eye(4, dtype=np.float64)
        new_T[:3, :3] = R_to_use
        new_T[:3, 3] = new_origin
        # Atomic-ish write: a concurrent reader of `_T_BOARD2BASE` either sees
        # the entire pre-update matrix or the entire post-update matrix, never
        # a torn (R_new, t_old) state. NumPy's `[:] = …` invokes element-wise
        # assignment under the GIL; the GIL doesn't make it formally atomic
        # but in practice no Python statement interleaves between the two
        # halves of a 4x4 copy.
        _T_BOARD2BASE[:] = new_T

        delta_mm = float(np.linalg.norm(new_origin - old_origin)) * 1000.0
        # ‖R_a − R_b‖_F = 2√2 sin(θ/2) is the exact identity (not approximate),
        # so this recovers the rotation angle in degrees.
        rot_frob = float(np.linalg.norm(R_to_use - old_R, ord="fro"))
        rot_delta_deg = float(
            np.degrees(2.0 * np.arcsin(min(1.0, rot_frob / (2.0 * np.sqrt(2)))))
        )
        _post_status(
            f"Localise OK: {n_inliers}/{len(detected_centres)} inliers, "
            f"centre ({new_centre[0]:.3f}, {new_centre[1]:.3f}, "
            f"{new_centre[2]:.3f}) m, shift {delta_mm:.1f} mm / {rot_delta_deg:.1f}°"
        )

        # Invalidate caches that depend on the previous _T_BOARD2BASE:
        #   - trajectory_collision_mgr_pair embeds the OLD tablet pose.
        #   - reachable_candidates were generated against the OLD centre, so
        #     their joint configs no longer aim at the new board location.
        # Both rebuild on next use.
        #
        # Bump ``reach_generation`` AT THE SAME TIME as zeroing the
        # candidates list. The renderer pairs gen↔candidates atomically
        # (see _render_reachability_dots), and the click handler checks
        # gen first then dereferences candidates by index — without the
        # bump here, a click on an existing sphere (still tagged with
        # the OLD gen, which still equals current_gen) followed by an
        # already-completed candidates-zero write would index past
        # ``[]`` and silently dismiss the popup. Bumping the gen here
        # invalidates the on-screen sphere names so the click handler
        # sees a stale-gen mismatch (correct) instead of a wrong-list
        # dereference.
        _state["trajectory_collision_mgr_pair"] = None
        _state["reach_generation"] = (
            int(_state.get("reach_generation", 0)) + 1
        )
        _state["reachable_candidates"] = []

        # Stamp the successful-localise time so the Run button knows the
        # board has been localised this session and skips the warning dialog.
        _state["last_localise_ok_at"] = time.time()

        # Refresh visual overlays so board, hemisphere, and reachability dots
        # all reflect the new pose. Schedules on the asyncio loop.
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
        # Stop the RealSenseCamera if we started one in real-hardware mode.
        real_cam = _state.get("real_camera")
        if real_cam is not None:
            try:
                real_cam.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("RealSenseCamera stop failed: %s", e)
            _state["real_camera"] = None
        _state["is_localising"] = False
