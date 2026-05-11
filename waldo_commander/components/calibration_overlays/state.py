"""Module-level state and board/hemisphere geometry helpers."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRotation

from . import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


# Guards paired RMW updates to ``_state`` (e.g. ``reach_generation`` +
# ``reachable_candidates``). Single-key writes are GIL-atomic and don't
# need this. ``RLock`` so a holder can re-acquire via a helper.
_state_lock = threading.RLock()


_state: dict[str, Any] = {
    "merged_stl_path": None,  # set on first add_overlays call
    "board_png_path": None,
    "frustum_objects": [],
    "is_running": False,
    "is_localising": False,  # board-localise thread guard, separate from is_running
    "result_label": None,
    "status_label": None,
    "current_mount": None,  # CameraMount, set by orchestrator
    "calibrated_mount": None,  # set by calibration thread on success
    "main_loop": None,  # asyncio loop captured at setup
    "ssg48_hijacked": False,
    # Cached workspace envelope (loaded once from waldo-commander's hull STL).
    "envelope_planes_A": None,  # (n_faces, 3) face-normal A coefficients
    "envelope_planes_b": None,  # (n_faces,) face-offset constants
    "envelope_max_reach": None, # scalar bounding sphere radius
    "envelope_loaded": False,
    # Scene-overlay handles for refresh-after-auto-localise.
    "scene_root": None,        # urdf_scene.scene root for re-attaching groups
    "board_group": None,       # ChArUco board overlay (deletable scene group)
    "hemisphere_group": None,  # hemisphere wireframe + reachability dots group
    "tablet_group": None,      # translucent tablet collision-box visual
    # Reused by the localise thread as lookout joint configurations.
    "reachable_candidates": [],
    # Built lazily on first ``validate_joint_trajectory`` call.
    "trajectory_collision_mgr_pair": None,
    # Most recent successful Localise Board run. None gates the Run button's
    # warning dialog.
    "last_localise_ok_at": None,
    # True while the localise-before-Run confirmation dialog is open.
    "dialog_open": False,
    # Detection-overlay (live perception viz): group handle + mtime cache
    # for change-detection polling + the ui.timer driving the poll.
    "detection_overlay_group": None,  # NiceGUI group handle
    "detection_last_mtime": 0.0,       # for change detection
    "detection_overlay_timer": None,
    # Visibility toggles for the 3D scene elements.
    "show_board": True,
    "show_hemisphere": True,
    "show_reachability": True,
    "show_near_cone": True,
    "show_centerline": True,
    "show_footprint": True,
    # Off by default — stale perception JSON often outlives its run.
    "show_detections": False,
}


# ---------------------------------------------------------------------------
# ArUco dictionary lookup — string → cv2 constant
# ---------------------------------------------------------------------------


def _aruco_dict_id(name: str) -> int:
    """Translate a dictionary name (e.g. ``"DICT_4X4_50"``) to OpenCV's
    integer constant; falls back to ``DICT_4X4_50``.
    """
    import cv2  # noqa: PLC0415
    attr = getattr(cv2.aruco, name, None)
    if isinstance(attr, int):
        return attr
    logger.warning(
        "Unknown ArUco dictionary %r; falling back to DICT_4X4_50",
        name,
    )
    return cv2.aruco.DICT_4X4_50


def current_board_config() -> Any:
    """Build a fresh ``BoardConfig`` from the current settings. Clamps
    marker_length to ``0.6 * square_length`` when the user is mid-edit and
    has temporarily violated the ``0 < marker < square`` invariant.
    """
    from parol6_vision.calibration.board import BoardConfig  # noqa: PLC0415

    sl = float(settings.board_square_length_m)
    ml = float(settings.board_marker_length_m)
    if ml <= 0.0 or ml >= sl:
        ml_clamped = sl * 0.6
        logger.warning(
            "marker_length %.4f m invalid for square_length %.4f m "
            "(must be 0 < ml < sl); clamping to %.4f m",
            ml, sl, ml_clamped,
        )
        ml = ml_clamped

    return BoardConfig(
        squares_x=int(settings.board_squares_x),
        squares_y=int(settings.board_squares_y),
        square_length=sl,
        marker_length=ml,
        aruco_dict_id=_aruco_dict_id(str(settings.board_dictionary)),
        legacy_pattern=bool(settings.board_legacy_pattern),
    )


# ---------------------------------------------------------------------------
# Hemisphere helpers
# ---------------------------------------------------------------------------


def _hemi_azimuth_center_deg() -> float:
    """Base→hemisphere-anchor azimuth (degrees, CCW from world +X)."""
    cx, cy, _ = _hemi_centre_world().tolist()
    return float(np.degrees(np.arctan2(cy, cx)))


def _hemi_azimuth_world_range_deg() -> tuple[float, float]:
    """World-frame azimuth range centred on the base→board direction."""
    center = _hemi_azimuth_center_deg()
    spread = float(settings.hemi_azimuth_spread_deg)
    return (center - spread, center + spread)


def _build_T_board2base() -> NDArray[np.float64]:
    """Compose the SE(3) board→base matrix from settings. When the mounting
    surface is enabled, the board origin is lifted along board-local +Z by
    the surface thickness so ``board_translate_m`` describes the position
    as if the surface were zero-thickness.
    """
    rpy = settings.board_rpy_rad
    R = np.eye(3, dtype=np.float64)
    if any(abs(a) > 1e-9 for a in rpy):
        R = SciRotation.from_euler("XYZ", rpy).as_matrix()

    translation = np.asarray(settings.board_translate_m, dtype=np.float64).copy()
    if bool(settings.surface_enabled):
        # Lift along board's +Z by surface thickness.
        thickness = float(settings.surface_dimensions_m[2])
        translation += R[:, 2] * thickness

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def _board_center_world() -> NDArray[np.float64]:
    """Board geometric centre in world coords. The ChArUco config places
    the local origin at one corner; the pose-generator look-at target needs
    the centre so the printed pattern stays inside FOV.
    """
    sx = int(settings.board_squares_x)
    sy = int(settings.board_squares_y)
    sl = float(settings.board_square_length_m)
    w_m = sx * sl
    h_m = sy * sl
    center_local = np.array([w_m / 2.0, h_m / 2.0, 0.0, 1.0], dtype=np.float64)
    return (_T_BOARD2BASE @ center_local)[:3]


def _hemi_centre_world() -> NDArray[np.float64]:
    """Hemisphere spatial anchor — ``hemi_centre_override_m`` if set, board
    centre otherwise. Distinct from the look-at target (always the board
    centre regardless of override).
    """
    override = settings.hemi_centre_override_m
    if override is not None:
        return np.asarray(override, dtype=np.float64)
    return _board_center_world()


# Mutated in place via slice assignment so importers see updates without
# rebinding. Defined after ``_build_T_board2base`` so the call resolves.
_T_BOARD2BASE: NDArray[np.float64] = _build_T_board2base()


def rebuild_T_board2base() -> None:
    """Rebuild ``_T_BOARD2BASE`` from settings and write it in place."""
    new_T = _build_T_board2base()
    _T_BOARD2BASE[:] = new_T


# ----------------------------------------------------------------------
# Recovered-board-pose persistence (survives browser refresh + restart)
# ----------------------------------------------------------------------


def save_recovered_board_pose(T_board2base: NDArray[np.float64]) -> None:
    """Persist a recovered board pose to both ``_state`` (in-memory) and
    ``app.storage.general`` (disk). Tagged with sim/real mode so
    ``restore_recovered_board_pose`` can discard stale poses on mode swap.
    """
    import logging  # noqa: PLC0415

    _log = logging.getLogger(__name__)
    arr = np.asarray(T_board2base, dtype=np.float64).reshape(4, 4).copy()
    _state["recovered_board_pose"] = arr

    try:
        import time as _time  # noqa: PLC0415

        from nicegui import app as _ng_app  # noqa: PLC0415
        from waldo_commander.state import robot_state as _rs  # noqa: PLC0415

        _ng_app.storage.general["recovered_board_pose"] = {
            "pose": arr.flatten().tolist(),
            "saved_sim_mode": bool(_rs.simulator_active),
            "saved_at_ts": _time.time(),
        }
        _log.debug(
            "save_recovered_board_pose: wrote storage entry (sim_mode=%s)",
            bool(_rs.simulator_active),
        )
    except Exception as e:  # noqa: BLE001
        _log.debug(
            "save_recovered_board_pose: storage write skipped (%s: %s)",
            type(e).__name__, e,
        )


def restore_recovered_board_pose() -> NDArray[np.float64] | None:
    """Find a previously-recovered board pose: in-memory ``_state`` first,
    ``app.storage.general`` fallback. Storage hits whose sim/real mode tag
    doesn't match the current mode are discarded. On a storage hit the
    in-memory slot is re-populated for the fast path.
    """
    import logging  # noqa: PLC0415

    _log = logging.getLogger(__name__)

    in_memory = _state.get("recovered_board_pose")
    if in_memory is not None:
        return np.asarray(in_memory, dtype=np.float64).reshape(4, 4)

    try:
        from nicegui import app as _ng_app  # noqa: PLC0415
        from waldo_commander.state import robot_state as _rs  # noqa: PLC0415

        stored = _ng_app.storage.general.get("recovered_board_pose")
    except Exception as e:  # noqa: BLE001
        _log.debug(
            "restore_recovered_board_pose: storage read skipped (%s: %s)",
            type(e).__name__, e,
        )
        return None

    if not stored or not isinstance(stored, dict):
        return None

    pose_list = stored.get("pose")
    if not isinstance(pose_list, list) or len(pose_list) != 16:
        _log.warning(
            "restore_recovered_board_pose: storage entry malformed; ignoring",
        )
        return None

    saved_sim = stored.get("saved_sim_mode")
    try:
        current_sim = bool(_rs.simulator_active)
    except Exception:  # noqa: BLE001
        current_sim = True
    if saved_sim is not None and bool(saved_sim) != current_sim:
        _log.info(
            "Board overlay: stored recovered pose was saved in %s mode but "
            "current mode is %s — keeping configured pose. Run Localise to "
            "refresh.",
            "sim" if saved_sim else "real",
            "sim" if current_sim else "real",
        )
        return None

    arr = np.asarray(pose_list, dtype=np.float64).reshape(4, 4)
    _state["recovered_board_pose"] = arr.copy()

    saved_at = stored.get("saved_at_ts", 0.0)
    age_str = ""
    if saved_at:
        import time as _time  # noqa: PLC0415

        age_s = _time.time() - float(saved_at)
        if age_s < 120:
            age_str = f", saved {age_s:.0f}s ago"
        elif age_s < 7200:
            age_str = f", saved {age_s / 60.0:.1f} min ago"
        else:
            age_str = f", saved {age_s / 3600.0:.1f} h ago"
    _log.info(
        "Board overlay: restored pose from storage — origin (%.3f, %.3f, %.3f) m "
        "in %s mode%s",
        float(arr[0, 3]),
        float(arr[1, 3]),
        float(arr[2, 3]),
        "sim" if saved_sim else "real",
        age_str,
    )
    return arr


def clear_recovered_board_pose() -> None:
    """Wipe the recovered-pose state so the next ``add_overlays`` falls
    back to the configured ``_BOARD_TRANSLATE_M`` / ``_BOARD_RPY_RAD``.
    No UI entry point yet — see DEFERRED_FEATURES.md.
    """
    import logging  # noqa: PLC0415

    _log = logging.getLogger(__name__)
    _state["recovered_board_pose"] = None
    try:
        from nicegui import app as _ng_app  # noqa: PLC0415

        _ng_app.storage.general.pop("recovered_board_pose", None)
    except Exception as e:  # noqa: BLE001
        _log.debug(
            "clear_recovered_board_pose: storage clear skipped (%s: %s)",
            type(e).__name__, e,
        )
