"""User-tunable calibration settings, persisted via ``app.storage.user``.

Values default to the constants in :mod:`constants`; UI edits override at
runtime via :func:`set_value` and persist across page reloads.

Read APIs::

    settings.board_translate_m            # attribute access
    settings.get("board_translate_m")     # dict-style lookup

Tuples round-trip through JSON as lists and are coerced back on read.
Presets are named snapshots under ``calib_presets``; none ship with the
code.
"""

from __future__ import annotations

import logging
from typing import Any

from . import constants as _c

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage namespace
# ---------------------------------------------------------------------------


# Storage keys are namespaced under ``calib_setting_<key>``.
_STORAGE_PREFIX = "calib_setting_"
_PRESETS_KEY = "calib_presets"  # dict[name, dict[key, value]]
_ACTIVE_PRESET_KEY = "calib_active_preset"  # name | None


# ---------------------------------------------------------------------------
# Defaults — sourced from constants.py
# ---------------------------------------------------------------------------


DEFAULTS: dict[str, Any] = {
    # ---- Board placement -------------------------------------------------
    "board_translate_m": _c._BOARD_TRANSLATE_M,
    "board_rpy_rad": _c._BOARD_RPY_RAD,
    # ---- Board geometry (matches BOARD_TABLET_30MM) ----------------------
    "board_squares_x": 7,
    "board_squares_y": 5,
    "board_square_length_m": 0.030,
    "board_marker_length_m": 0.018,
    "board_dictionary": "DICT_4X4_50",
    "board_legacy_pattern": True,
    # ---- Mounting surface (collision primitive) --------------------------
    # Replaces the old "tablet" naming. Generic rectangular surface that
    # can represent a tablet, a printed-board mounting block, etc.
    "surface_enabled": _c._TABLET_PRIMITIVE_ENABLED,
    "surface_show_overlay": _c._SHOW_TABLET_OVERLAY,
    "surface_dimensions_m": _c._TABLET_DIMENSIONS_M,
    "surface_offset_local_m": _c._TABLET_OFFSET_FROM_CHARUCO_LOCAL_M,
    "collision_safety_margin_m": _c._COLLISION_SAFETY_MARGIN_M,
    "floor_primitive_enabled": _c._FLOOR_PRIMITIVE_ENABLED,
    "enable_self_collision_check": _c._ENABLE_SELF_COLLISION_CHECK,
    # ---- Camera intrinsics ----------------------------------------------
    "intr_fx": _c._INTR_FX,
    "intr_fy": _c._INTR_FY,
    "intr_cx": _c._INTR_CX,
    "intr_cy": _c._INTR_CY,
    "intr_width": _c._INTR_W,
    "intr_height": _c._INTR_H,
    # ---- Camera mount cold-start ----------------------------------------
    "cam_mount_translate_mm": _c._CAM_MOUNT_TRANSLATE_MM,
    "cam_mount_tilt_deg": _c._CAM_MOUNT_TILT_DEG,
    # ---- Hemisphere search ----------------------------------------------
    "hemi_distance_range_m": _c._HEMI_DISTANCE_RANGE_M,
    "hemi_elevation_range_deg": _c._HEMI_ELEVATION_RANGE_DEG,
    "hemi_azimuth_spread_deg": _c._HEMI_AZIMUTH_SPREAD_DEG,
    "hemi_centre_override_m": _c._HEMI_CENTRE_OVERRIDE_M,
    # ---- Reachability dots (visualization-only) -------------------------
    # Sobol candidates for the IK sweep; survivors pass through a
    # non-overlap spread filter, so the drawn count is usually lower.
    "reachability_n_candidates": 64,
    # Doubles as the min-distance threshold for the non-overlap selector.
    "reachability_dot_radius_m": 0.005,
    # ---- Localise sweep -------------------------------------------------
    "localise_use_j0_sweep": _c._LOCALISE_USE_J0_SWEEP,
    "localise_continuous_sweep": _c._LOCALISE_CONTINUOUS_SWEEP,
    "localise_sweep_speed": _c._LOCALISE_SWEEP_SPEED,
    "localise_capture_period_s": _c._LOCALISE_CAPTURE_PERIOD_S,
    "localise_j0_sweep_half_deg": _c._LOCALISE_J0_SWEEP_HALF_DEG,
    "localise_j0_chunk_deg": _c._LOCALISE_J0_CHUNK_DEG,
    "localise_seed_n_candidates": _c._LOCALISE_SEED_N_CANDIDATES,
    "localise_early_stop_detections": _c._LOCALISE_EARLY_STOP_DETECTIONS,
    "localise_early_stop_inliers": _c._LOCALISE_EARLY_STOP_INLIERS,
    "localise_min_inliers_to_proceed": _c._LOCALISE_MIN_INLIERS_TO_PROCEED,
    "localise_inlier_threshold_m": _c._LOCALISE_INLIER_THRESHOLD_M,
    "localise_refine_n_poses": _c._LOCALISE_REFINE_N_POSES,
    "localise_refine_distance_m": _c._LOCALISE_REFINE_DISTANCE_M,
    "localise_refine_elevation_deg": _c._LOCALISE_REFINE_ELEVATION_DEG,
    "localise_min_detections": _c._LOCALISE_MIN_DETECTIONS,
    # ---- Gripper jaw variant --------------------------------------------
    "tool_jaw_variant": _c._SSG48_JAW_VARIANT,
}


# ---------------------------------------------------------------------------
# Type-aware coercion (JSON list ↔ Python tuple, None pass-through)
# ---------------------------------------------------------------------------


# Tuple defaults get flattened to lists in JSON; coerce back on read.
_TUPLE_KEYS: frozenset[str] = frozenset(
    k for k, v in DEFAULTS.items() if isinstance(v, tuple)
)


def _coerce(key: str, raw: Any) -> Any:
    """Restore the original Python type after a round-trip through JSON."""
    if raw is None:
        return None
    if key in _TUPLE_KEYS and isinstance(raw, list):
        return tuple(raw)
    return raw


def _to_storage(key: str, value: Any) -> Any:
    """Convert a Python value to a JSON-serializable form for storage."""
    if value is None:
        return None
    if key in _TUPLE_KEYS and isinstance(value, tuple):
        return list(value)
    return value


# ---------------------------------------------------------------------------
# Runtime overrides
# ---------------------------------------------------------------------------


# Populated by ``load_from_storage()``; absent keys read DEFAULTS.
_runtime: dict[str, Any] = {}


def get(key: str) -> Any:
    """Return ``key``'s value from the highest-priority layer:

    1. Active-tool override (camera intrinsics + cold-start mount only).
    2. Runtime override (set via :func:`set_value`).
    3. Module default.

    Workspace properties (board, hemisphere, localise) skip layer 1.
    """
    if key not in DEFAULTS:
        raise KeyError(f"Unknown calibration setting: {key}")
    # Layer 1: per-tool override. Specific exceptions only so a typo in
    # the override path doesn't get swallowed.
    try:
        from . import custom_tools  # noqa: PLC0415

        override = custom_tools.active_tool_override(key)
        if override is not None:
            return override
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("settings.get: per-tool override layer skipped (%s)", e)
    # Layer 2: runtime (storage-backed) override.
    if key in _runtime:
        return _coerce(key, _runtime[key])
    # Layer 3: shipped default.
    return DEFAULTS[key]


def get_global(key: str) -> Any:
    """Like :func:`get` but skips the per-tool override layer. Use when
    seeding an editor for a NON-active tool, where the global default is
    the right seed.
    """
    if key not in DEFAULTS:
        raise KeyError(f"Unknown calibration setting: {key}")
    if key in _runtime:
        return _coerce(key, _runtime[key])
    return DEFAULTS[key]


def set_value(key: str, value: Any, *, persist: bool = True) -> None:
    """Update ``key``'s runtime value. Persists to ``app.storage.user`` by
    default; pass ``persist=False`` to update transiently (e.g. when applying
    a preset, where the caller persists the bulk write).
    """
    if key not in DEFAULTS:
        raise KeyError(f"Unknown calibration setting: {key}")
    _runtime[key] = value
    if persist:
        _persist_one(key, value)


def reset_to_defaults() -> None:
    """Drop every runtime override + clear ``calib_setting_*`` and
    ``calib_tool_*`` from storage. Also clears the per-tool runtime cache.
    """
    _runtime.clear()
    try:
        from nicegui import app  # noqa: PLC0415
        store = app.storage.user
        # Snapshot keys before mutating.
        keys_to_drop = [
            k for k in list(store.keys())
            if isinstance(k, str) and (
                k.startswith(_STORAGE_PREFIX)
                or k.startswith("calib_tool_")
            )
        ]
        for k in keys_to_drop:
            del store[k]
        store[_ACTIVE_PRESET_KEY] = None
    except Exception as e:  # noqa: BLE001
        logger.debug("settings.reset_to_defaults: storage unavailable (%s)", e)
    # Drop the per-tool runtime cache so worker threads see the reset.
    try:
        from . import custom_tools  # noqa: PLC0415
        custom_tools._per_tool_runtime_cache.clear()
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "settings.reset_to_defaults: per-tool cache clear failed (%s)", e,
        )


def __getattr__(name: str) -> Any:
    """``settings.board_translate_m`` ≡ ``settings.get("board_translate_m")``."""
    if name in DEFAULTS:
        return get(name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


# ---------------------------------------------------------------------------
# Storage I/O
# ---------------------------------------------------------------------------


def load_from_storage() -> None:
    """Populate ``_runtime`` from ``app.storage.user``. Idempotent.
    Existing runtime entries for keys not in storage are kept.
    """
    try:
        from nicegui import app  # noqa: PLC0415
        store = app.storage.user
    except Exception as e:  # noqa: BLE001
        logger.debug("settings.load_from_storage: storage unavailable (%s)", e)
        return

    for key in DEFAULTS:
        storage_key = _STORAGE_PREFIX + key
        if storage_key in store:
            _runtime[key] = _coerce(key, store[storage_key])


def _persist_one(key: str, value: Any) -> None:
    """Write a single setting to storage. No-op outside a request context."""
    try:
        from nicegui import app  # noqa: PLC0415
        app.storage.user[_STORAGE_PREFIX + key] = _to_storage(key, value)
    except Exception as e:  # noqa: BLE001
        logger.debug("settings._persist_one(%s): storage unavailable (%s)", key, e)


def _persist_runtime() -> None:
    """Bulk-write every runtime override to storage."""
    for key, value in _runtime.items():
        _persist_one(key, value)


# ---------------------------------------------------------------------------
# Preset management — named snapshots of every key
# ---------------------------------------------------------------------------


def _load_presets() -> dict[str, dict[str, Any]]:
    """Load the preset dict from storage. Empty dict when none exist."""
    try:
        from nicegui import app  # noqa: PLC0415
        return dict(app.storage.user.get(_PRESETS_KEY) or {})
    except Exception:  # noqa: BLE001
        return {}


def _save_presets(presets: dict[str, dict[str, Any]]) -> None:
    """Write the preset dict to storage."""
    try:
        from nicegui import app  # noqa: PLC0415
        app.storage.user[_PRESETS_KEY] = presets
    except Exception as e:  # noqa: BLE001
        logger.debug("settings._save_presets: storage unavailable (%s)", e)


def list_presets() -> list[str]:
    """Sorted list of preset names; empty when none exist."""
    return sorted(_load_presets().keys())


def get_active_preset() -> str | None:
    """Name of the currently-applied preset, or None for defaults / custom."""
    try:
        from nicegui import app  # noqa: PLC0415
        return app.storage.user.get(_ACTIVE_PRESET_KEY)
    except Exception:  # noqa: BLE001
        return None


def _set_active_preset(name: str | None) -> None:
    try:
        from nicegui import app  # noqa: PLC0415
        app.storage.user[_ACTIVE_PRESET_KEY] = name
    except Exception:  # noqa: BLE001
        pass


def save_preset(name: str) -> None:
    """Snapshot every key (runtime override else default) under ``name``.
    Overwrites if the name exists; complete snapshot survives future
    additions to DEFAULTS.
    """
    if not name:
        raise ValueError("preset name cannot be empty")
    snapshot: dict[str, Any] = {}
    for key in DEFAULTS:
        snapshot[key] = _to_storage(key, get(key))
    presets = _load_presets()
    presets[name] = snapshot
    _save_presets(presets)
    _set_active_preset(name)
    logger.info("calibration preset saved: %s", name)


def load_preset(name: str) -> None:
    """Apply the named preset; missing keys fall back to the default.
    Persists bulk to storage. Raises KeyError when the preset is missing.
    """
    presets = _load_presets()
    if name not in presets:
        raise KeyError(f"preset {name!r} not found")
    snapshot = presets[name]
    _runtime.clear()
    for key, raw in snapshot.items():
        if key in DEFAULTS:
            _runtime[key] = _coerce(key, raw)
        # silently ignore unknown keys from older / future presets
    _persist_runtime()
    _set_active_preset(name)
    logger.info("calibration preset loaded: %s", name)


def delete_preset(name: str) -> None:
    """Remove ``name`` from the preset list. No-op if it doesn't exist."""
    presets = _load_presets()
    if name in presets:
        del presets[name]
        _save_presets(presets)
        if get_active_preset() == name:
            _set_active_preset(None)
        logger.info("calibration preset deleted: %s", name)


def export_preset_json(name: str | None = None) -> str:
    """JSON snapshot of the named preset (or the current runtime state
    when ``name`` is None).
    """
    import json  # noqa: PLC0415
    if name is None:
        snapshot = {key: _to_storage(key, get(key)) for key in DEFAULTS}
    else:
        presets = _load_presets()
        if name not in presets:
            raise KeyError(f"preset {name!r} not found")
        snapshot = presets[name]
    return json.dumps(snapshot, indent=2, sort_keys=True)


def import_preset_json(name: str, payload: str) -> None:
    """Parse JSON ``payload`` and save it as a preset under ``name``.
    Raises ``ValueError`` for unparseable input.
    """
    import json  # noqa: PLC0415
    try:
        snapshot_raw = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(snapshot_raw, dict):
        raise ValueError("preset JSON must be a dict")
    snapshot: dict[str, Any] = {}
    for key, raw in snapshot_raw.items():
        if key in DEFAULTS:
            snapshot[key] = raw
    presets = _load_presets()
    presets[name] = snapshot
    _save_presets(presets)
    logger.info("calibration preset imported: %s", name)
