"""Live-apply side effects for settings changes.

Each setting key maps to a side effect that propagates the new value into
the running scene. Settings reads themselves go through ``settings.get`` /
attribute access so consumers always see the current value — these
functions handle the cases where a setting *also* needs to invalidate a
cache or rebuild a derived structure (e.g. ``_T_BOARD2BASE``, the cached
collision manager, the camera mount used for the live frustum).

The dispatch is intentionally coarse: most settings affect either
``_T_BOARD2BASE`` (board-placement / surface-thickness), the cached
collision manager, or the live ``CameraMount``. Calling
``apply_setting_change(key)`` figures out which side effects are needed
for that key and runs them in the right order.

UI changes that don't need a side effect (every other setting) just call
``settings.set_value`` and read the current value on the next consumer
tick — no wiring required here.
"""

from __future__ import annotations

import logging

from .state import _state, rebuild_T_board2base

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Side-effect categories
# ---------------------------------------------------------------------------


# Settings whose change requires rebuilding ``_T_BOARD2BASE`` and the
# board-dependent overlays (board, tablet, hemisphere wireframe,
# reachability dots).
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


# Settings whose change requires re-rendering the cached board PNG so the
# texture in the 3D scene shows the right marker layout / count. (The
# texture URL gets a cache-bust suffix; same on-disk path.)
_BOARD_PNG_KEYS: frozenset[str] = frozenset({
    "board_squares_x",
    "board_squares_y",
    "board_square_length_m",
    "board_marker_length_m",
    "board_dictionary",
    "board_legacy_pattern",
})


# Settings whose change requires redrawing the camera frustum (intrinsics
# determine the cone angles + image dimensions of the wireframe pyramid).
_FRUSTUM_KEYS: frozenset[str] = frozenset({
    "intr_fx",
    "intr_fy",
    "intr_cx",
    "intr_cy",
    "intr_width",
    "intr_height",
})


# Settings whose change invalidates the cached collision manager
# (``_state['trajectory_collision_mgr_pair']``). Re-built lazily on next
# use, so we only need to drop the cache.
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


# Settings whose change requires updating the live ``CameraMount``
# (used by the frustum + hemisphere visualisations until calibration runs).
_CAM_MOUNT_KEYS: frozenset[str] = frozenset({
    "cam_mount_translate_mm",
    "cam_mount_tilt_deg",
})


# ---------------------------------------------------------------------------
# Side-effect runners
# ---------------------------------------------------------------------------


def _rebuild_camera_mount() -> None:
    """Replace ``_state['current_mount']`` with a fresh CameraMount built
    from the current settings, then redraw the frustum."""
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
    """Invalidate the cached collision manager so the next request rebuilds
    it from the current surface / safety-margin / jaw settings."""
    _state["trajectory_collision_mgr_pair"] = None


def _refresh_overlays() -> None:
    """Rebuild every board-dependent overlay (board, tablet visual,
    hemisphere wireframe, reachability dots) at the current ``_T_BOARD2BASE``."""
    from .overlays import refresh_board_dependent_overlays  # noqa: PLC0415
    refresh_board_dependent_overlays()


def _regenerate_board_png() -> None:
    """Re-render the cached ChArUco PNG when board geometry changes."""
    from .overlays import regenerate_board_png  # noqa: PLC0415
    regenerate_board_png()


def _redraw_frustum() -> None:
    """Force a frustum rebuild from the current cam-mount + intrinsics.

    Used when intrinsics change without a mount change — the existing
    frustum lines are based on stale fx/fy/cx/cy and need replacing.
    """
    from .frustum import update_frustum  # noqa: PLC0415
    mount = _state.get("current_mount")
    if mount is None:
        return
    try:
        update_frustum(mount.T_cam2flange)
    except Exception as e:  # noqa: BLE001
        logger.debug("frustum redraw failed (%s) — will refresh on next tick", e)
    # Also invalidate the dynamic-footprint cache so the next tick
    # rebuilds the centerline + footprint with the new intrinsics.
    _state["footprint_last_q"] = None
    _state["footprint_last_mount"] = None


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def apply_setting_change(key: str) -> None:
    """Run the side effects associated with ``key`` having just changed.

    Idempotent — calling this twice for the same key is harmless. Order
    matters: rebuild ``_T_BOARD2BASE`` first, then the camera mount (so
    the frustum redraw uses the new mount), then regenerate the board
    PNG, then drop caches, then refresh overlays (so they pick up the
    new T_BOARD2BASE + caches + PNG URL).
    """
    needs_t_rebuild = key in _BOARD_PLACEMENT_KEYS
    needs_mount = key in _CAM_MOUNT_KEYS
    needs_collision_drop = key in _COLLISION_KEYS
    needs_png_regen = key in _BOARD_PNG_KEYS
    needs_frustum_redraw = key in _FRUSTUM_KEYS
    needs_overlay_refresh = needs_t_rebuild or needs_png_regen

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
