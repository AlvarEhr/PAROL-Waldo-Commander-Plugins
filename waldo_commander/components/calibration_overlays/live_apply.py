"""Live-apply side effects for settings changes.

Setting reads go through ``settings.get``; these functions cover the cases
where a key change also needs to invalidate a cache or rebuild a derived
structure (``_T_BOARD2BASE``, the collision manager, the live
``CameraMount``). Call :func:`apply_setting_change` after any
``settings.set_value``.
"""

from __future__ import annotations

import logging

from .state import _state, rebuild_T_board2base

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Side-effect categories
# ---------------------------------------------------------------------------


# Rebuild ``_T_BOARD2BASE`` + board-dependent overlays.
_BOARD_PLACEMENT_KEYS: frozenset[str] = frozenset({
    "board_translate_m",
    "board_rpy_rad",
    "board_squares_x",
    "board_squares_y",
    "board_square_length_m",
    "board_marker_length_m",
    "board_dictionary",
    "board_legacy_pattern",
    "surface_enabled",
    "surface_dimensions_m",
    "surface_offset_local_m",
    "surface_show_overlay",
    "hemi_distance_range_m",
    "hemi_elevation_range_deg",
    "hemi_azimuth_spread_deg",
    "hemi_centre_override_m",
})


# Re-render the cached ChArUco PNG (URL gets a cache-bust suffix).
_BOARD_PNG_KEYS: frozenset[str] = frozenset({
    "board_squares_x",
    "board_squares_y",
    "board_square_length_m",
    "board_marker_length_m",
    "board_dictionary",
    "board_legacy_pattern",
})


# Redraw the camera frustum (intrinsics → cone angles + image dimensions).
_FRUSTUM_KEYS: frozenset[str] = frozenset({
    "intr_fx",
    "intr_fy",
    "intr_cx",
    "intr_cy",
    "intr_width",
    "intr_height",
})


# Drop the cached collision manager; rebuilt lazily on next use.
_COLLISION_KEYS: frozenset[str] = frozenset({
    "surface_enabled",
    "surface_dimensions_m",
    "surface_offset_local_m",
    "collision_safety_margin_m",
    "floor_primitive_enabled",
    "tool_jaw_variant",
    "board_translate_m",
    "board_rpy_rad",
    "board_squares_x",
    "board_squares_y",
    "board_square_length_m",
})


# Update the live ``CameraMount`` (used by frustum + hemisphere viz).
_CAM_MOUNT_KEYS: frozenset[str] = frozenset({
    "cam_mount_translate_mm",
    "cam_mount_tilt_deg",
})


# Re-run the reachability IK sweep (changes sweep input itself).
_REACHABILITY_RESWEEP_KEYS: frozenset[str] = frozenset({
    "reachability_n_candidates",
})


# Re-render existing reachability dots at new sphere geometry; no IK sweep.
_REACHABILITY_RERENDER_KEYS: frozenset[str] = frozenset({
    "reachability_dot_radius_m",
})


# ---------------------------------------------------------------------------
# Side-effect runners
# ---------------------------------------------------------------------------


def _rebuild_camera_mount() -> None:
    """Rebuild ``_state['current_mount']`` from settings + redraw frustum."""
    try:
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
    except ImportError:
        return
    from . import settings  # noqa: PLC0415
    from .frustum import update_frustum  # noqa: PLC0415

    cam_translate = settings.cam_mount_translate_mm
    cam_tilt = settings.cam_mount_tilt_deg
    new_mount = CameraMount.from_eyeball_estimate(
        x_mm=cam_translate[0],
        y_mm=cam_translate[1],
        z_mm=cam_translate[2],
        tilt_x_deg=cam_tilt[0],
        tilt_y_deg=cam_tilt[1],
        tilt_z_deg=cam_tilt[2],
    )
    _state["current_mount"] = new_mount
    try:
        update_frustum(new_mount.T_cam2flange)
    except Exception as e:  # noqa: BLE001
        logger.debug("frustum redraw failed (%s) — will refresh on next tick", e)


def _drop_collision_cache() -> None:
    """Invalidate the cached collision manager."""
    _state["trajectory_collision_mgr_pair"] = None


def _refresh_overlays() -> None:
    """Rebuild board-dependent overlays at the current ``_T_BOARD2BASE``."""
    from .overlays import refresh_board_dependent_overlays  # noqa: PLC0415
    refresh_board_dependent_overlays()


def _regenerate_board_png() -> None:
    """Re-render the cached ChArUco PNG."""
    from .overlays import regenerate_board_png  # noqa: PLC0415
    regenerate_board_png()


def _redraw_frustum() -> None:
    """Rebuild the frustum from current cam-mount + intrinsics."""
    from .frustum import update_frustum  # noqa: PLC0415
    mount = _state.get("current_mount")
    if mount is None:
        return
    try:
        update_frustum(mount.T_cam2flange)
    except Exception as e:  # noqa: BLE001
        logger.debug("frustum redraw failed (%s) — will refresh on next tick", e)
    # Invalidate the dynamic-footprint cache so the next tick rebuilds it.
    _state["footprint_last_q"] = None
    _state["footprint_last_mount"] = None


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def apply_setting_change(key: str) -> None:
    """Run side effects for a setting change. Idempotent. Order: T_board2base
    → mount → PNG → frustum → cache drop → overlay refresh.
    """
    needs_t_rebuild = key in _BOARD_PLACEMENT_KEYS
    needs_mount = key in _CAM_MOUNT_KEYS
    needs_collision_drop = key in _COLLISION_KEYS
    needs_png_regen = key in _BOARD_PNG_KEYS
    needs_frustum_redraw = key in _FRUSTUM_KEYS
    needs_overlay_refresh = needs_t_rebuild or needs_png_regen
    needs_reachability_resweep = key in _REACHABILITY_RESWEEP_KEYS
    needs_reachability_rerender = key in _REACHABILITY_RERENDER_KEYS

    if needs_t_rebuild:
        try:
            rebuild_T_board2base()
        except Exception as e:  # noqa: BLE001
            logger.warning("rebuild_T_board2base failed for %s: %s", key, e)
    if needs_mount:
        _rebuild_camera_mount()
    if needs_png_regen:
        _regenerate_board_png()
    if needs_frustum_redraw:
        _redraw_frustum()
    if needs_collision_drop:
        _drop_collision_cache()
    if needs_overlay_refresh:
        _refresh_overlays()
    if needs_reachability_resweep:
        try:
            from .reachability import refresh_reachability_for_active_tool  # noqa: PLC0415

            refresh_reachability_for_active_tool()
        except Exception as e:  # noqa: BLE001
            logger.debug("reachability resweep failed: %s", e)
    if needs_reachability_rerender:
        try:
            from .reachability import re_render_reachability_dots  # noqa: PLC0415

            re_render_reachability_dots()
        except Exception as e:  # noqa: BLE001
            logger.debug("reachability rerender failed: %s", e)
