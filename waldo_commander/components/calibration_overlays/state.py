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


# Lock guarding paired RMW updates to ``_state``. Acquire when reading or
# writing more than one related key as a unit so other threads can't see
# a half-updated view.
#
# Currently scopes the ``reach_generation`` + ``reachable_candidates`` pair
# (writers in ``localise._update_T_board2base`` and
# ``reachability._render_reachability_dots``; readers in
# ``pose_popup._on_scene_click``). Single-key writes like
# ``_state["is_running"] = True`` don't need the lock — the dict assignment
# is GIL-atomic on its own.
#
# ``RLock`` so a path that already holds the lock can acquire it again
# (e.g. a writer that calls into a helper which itself locks).
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
    # Reachable hemisphere candidates cached by _add_reachability_points; the
    # board-localise thread reuses these as lookout joint configurations so we
    # don't have to re-run pose generation just to pick scan poses.
    "reachable_candidates": [],
    # Cached collision-manager pair for validate_joint_trajectory(); built
    # lazily on first call so users importing this module pay nothing.
    "trajectory_collision_mgr_pair": None,
    # Timestamp of the most recent successful Localise Board run. None
    # means never — the Run button shows a warning dialog in that case
    # because calibration would aim at the configured _BOARD_TRANSLATE_M
    # rather than the actual board pose.
    "last_localise_ok_at": None,
    # True while the localise-before-Run confirmation dialog is open.
    # _busy_warn checks this so the user can't start a parallel Localise
    # while the dialog is awaiting their answer.
    "dialog_open": False,
    # Detection-overlay (live perception viz). The group handle is the
    # NiceGUI scene group containing the wireframe boxes + labels for the
    # most recent set of detections; we delete + rebuild it on each poll
    # tick where the JSON has changed. detection_last_mtime caches the
    # file mtime so we skip unchanged polls cheaply. detection_overlay_timer
    # is the ui.timer handle from add_overlays.
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
    # Off by default — the perception JSON file frequently outlives
    # the pipeline run that wrote it, and the user wouldn't expect a
    # stale wireframe of "the white cube" to greet them on startup.
    "show_detections": False,
}


# ---------------------------------------------------------------------------
# ArUco dictionary lookup — string → cv2 constant
# ---------------------------------------------------------------------------


def _aruco_dict_id(name: str) -> int:
    """Translate a user-facing dictionary name (e.g. ``"DICT_4X4_50"``) to
    OpenCV's integer constant. Falls back to ``DICT_4X4_50`` with a warning
    if the name isn't recognised.
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
    """Build a fresh ``BoardConfig`` from the current settings.

    Imports parol6-vision lazily so this module loads cleanly even when
    the calibration package isn't installed (e.g. in headless tests).

    When the user is mid-edit (e.g. just changed square_length but not yet
    marker_length, or vice-versa), the marker/square invariant
    ``0 < marker < square`` may be temporarily violated. Rather than
    raising and aborting every consumer, we clamp marker_length to
    ``0.6 * square_length`` (the recommended sweet spot per
    Garrido-Jurado 2014) and log a warning. The UI also surfaces the
    invariant in real time when the user types invalid values, but
    this ensures the scene keeps rendering regardless.
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
    """Direction from the robot base origin to the hemisphere anchor, in
    degrees measured from world +X (CCW). Used so the hemisphere naturally
    faces "away from the robot" toward where the workable space (board)
    sits."""
    cx, cy, _ = _hemi_centre_world().tolist()
    return float(np.degrees(np.arctan2(cy, cx)))


def _hemi_azimuth_world_range_deg() -> tuple[float, float]:
    """World-frame azimuth range covering the hemisphere's spread, centered
    on the base→board direction. Used by both the viz and the orchestrator."""
    center = _hemi_azimuth_center_deg()
    spread = float(settings.hemi_azimuth_spread_deg)
    return (center - spread, center + spread)


def _build_T_board2base() -> NDArray[np.float64]:
    """Compose the SE(3) board→base matrix from the board placement settings.

    When the mounting surface is enabled, the board origin is auto-lifted
    along its local +Z by the surface's thickness (``surface_dimensions_m[2]``).
    This matches the physical reality of a tablet lying screen-up on a
    bench: the bench surface is at world z=0, the surface body sits on the
    bench, and the ChArUco face is at z = surface_thickness above the bench.
    The user-facing ``board_translate_m`` continues to describe where the
    board sits IF the surface had zero thickness — ergonomic because it
    preserves the natural mental model "the board's at this XY".
    """
    rpy = settings.board_rpy_rad
    R = np.eye(3, dtype=np.float64)
    if any(abs(a) > 1e-9 for a in rpy):
        R = SciRotation.from_euler("XYZ", rpy).as_matrix()

    translation = np.asarray(settings.board_translate_m, dtype=np.float64).copy()
    if bool(settings.surface_enabled):
        # Lift along board's +Z (out of the screen face) by surface thickness.
        thickness = float(settings.surface_dimensions_m[2])
        translation += R[:, 2] * thickness

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def _board_center_world() -> NDArray[np.float64]:
    """Compute the board's geometric center in world coordinates.

    The ChArUco config places the board's local origin at one corner and the
    board extends to (w_m, h_m, 0) in board-local coordinates. The
    pose-generator look-at target uses the board CENTRE (not the corner);
    aiming at the corner would make the pose generator aim cameras at one
    edge of the board, leaving most of the printed pattern outside FOV.
    """
    sx = int(settings.board_squares_x)
    sy = int(settings.board_squares_y)
    sl = float(settings.board_square_length_m)
    w_m = sx * sl
    h_m = sy * sl
    center_local = np.array([w_m / 2.0, h_m / 2.0, 0.0, 1.0], dtype=np.float64)
    return (_T_BOARD2BASE @ center_local)[:3]


def _hemi_centre_world() -> NDArray[np.float64]:
    """Hemisphere anchor — ``hemi_centre_override_m`` if set, board centre
    otherwise. This is the SPATIAL anchor of the hemisphere (where the dome
    of camera positions sits in world frame). Distinct from the look-at
    target, which is always the actual board centre regardless of override.
    """
    override = settings.hemi_centre_override_m
    if override is not None:
        return np.asarray(override, dtype=np.float64)
    return _board_center_world()


# Module-level computed value. Must come AFTER ``_build_T_board2base`` is
# defined so the call here resolves correctly. Gets mutated in place via
# slice assignment from ``rebuild_T_board2base`` and from auto-localise so
# importers see updates without rebinding.
_T_BOARD2BASE: NDArray[np.float64] = _build_T_board2base()


def rebuild_T_board2base() -> None:
    """Rebuild ``_T_BOARD2BASE`` from the current settings (board placement,
    surface thickness) and write it in place via slice assignment.

    Call this after any board-placement or surface-thickness setting
    changes so all importing modules see the new value.
    """
    new_T = _build_T_board2base()
    _T_BOARD2BASE[:] = new_T
