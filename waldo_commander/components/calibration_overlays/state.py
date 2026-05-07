"""Module-level state and board/hemisphere geometry helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRotation

from .constants import (
    _BOARD_RPY_RAD,
    _BOARD_TRANSLATE_M,
    _HEMI_AZIMUTH_SPREAD_DEG,
    _HEMI_CENTRE_OVERRIDE_M,
    _TABLET_DIMENSIONS_M,
    _TABLET_PRIMITIVE_ENABLED,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


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
}


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
    return (center - _HEMI_AZIMUTH_SPREAD_DEG, center + _HEMI_AZIMUTH_SPREAD_DEG)


def _build_T_board2base() -> NDArray[np.float64]:
    """Compose the SE(3) board→base matrix from the position + RPY tunables.

    When ``_TABLET_PRIMITIVE_ENABLED`` is set, the board origin is
    auto-lifted along its local +Z by ``_TABLET_DIMENSIONS_M[2]``. This
    matches the physical reality of a tablet lying screen-up on a bench:
    the bench surface is at world z=0, the tablet body sits on the bench,
    and the ChArUco screen is at z = tablet_thickness above the bench.
    The user-facing ``_BOARD_TRANSLATE_M`` continues to describe where
    the board sits IF the tablet had zero thickness — ergonomic because
    it preserves the natural mental model "the board's at this XY".
    """
    R = np.eye(3, dtype=np.float64)
    if any(abs(a) > 1e-9 for a in _BOARD_RPY_RAD):
        R = SciRotation.from_euler("XYZ", _BOARD_RPY_RAD).as_matrix()

    translation = np.asarray(_BOARD_TRANSLATE_M, dtype=np.float64).copy()
    if _TABLET_PRIMITIVE_ENABLED:
        # Lift along board's +Z (out of the screen face) by tablet thickness.
        translation += R[:, 2] * _TABLET_DIMENSIONS_M[2]

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
    from parol6_vision.calibration.board import BOARD_TABLET_30MM as _cfg  # noqa: PLC0415
    w_m = _cfg.squares_x * _cfg.square_length
    h_m = _cfg.squares_y * _cfg.square_length
    center_local = np.array([w_m / 2.0, h_m / 2.0, 0.0, 1.0], dtype=np.float64)
    return (_T_BOARD2BASE @ center_local)[:3]


def _hemi_centre_world() -> NDArray[np.float64]:
    """Hemisphere anchor — `_HEMI_CENTRE_OVERRIDE_M` if set, board centre
    otherwise. This is the SPATIAL anchor of the hemisphere (where the dome
    of camera positions sits in world frame). Distinct from the look-at
    target, which is always the actual board centre regardless of override.
    """
    if _HEMI_CENTRE_OVERRIDE_M is not None:
        return np.asarray(_HEMI_CENTRE_OVERRIDE_M, dtype=np.float64)
    return _board_center_world()


# Module-level computed value. Must come AFTER `_build_T_board2base` is
# defined so the call here resolves correctly.
_T_BOARD2BASE: NDArray[np.float64] = _build_T_board2base()
