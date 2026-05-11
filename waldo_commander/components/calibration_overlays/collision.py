"""Public collision-validation API for the Waldo-Commander integration.

Thin wrapper over :mod:`parol6_vision.calibration.collision_core` that wires
in the NiceGUI master toggle, the active-tool lookup via ``robot_state``,
the live ``_T_BOARD2BASE`` board pose, and the user-editable settings layer.

The core module (:mod:`parol6_vision.calibration.collision_core`) holds the
actual FCL + trimesh + parol6 mesh logic and is NiceGUI-free, so the
program-runner subprocess and ``PathPreviewClient`` (both of which can't
import NiceGUI) can use the same code path via direct ``collision_core``
imports.

Backwards-compatible private helpers (``_build_collision_manager``,
``_self_collides``, ``_trajectory_collides``, ``_resolve_active_tool_meshes``,
``_mesh_collision_enabled``) remain for the existing in-package consumers
(``calibration_thread.py``, ``frustum.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from parol6_vision.calibration.collision_core import (
    CollisionEnvironmentConfig,
    build_collision_manager as _core_build,
    parol6_mesh_dir,
    resolve_tool_meshes_from_registry,
    self_collides as _core_self_collides,
    trajectory_collides as _core_trajectory_collides,
    validate_joint_trajectory_core,
)

from . import settings
from .state import _T_BOARD2BASE, _state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Master-toggle gate (NiceGUI-bound)
# ---------------------------------------------------------------------------


def _mesh_collision_enabled() -> bool:
    """Master gate for the mesh-collision machinery. Read from
    ``app.storage.general['mesh_collision_check_enabled']`` (default
    True). Flipped from the bottom-right Settings tab.
    """
    try:
        from nicegui import app  # noqa: PLC0415

        return bool(app.storage.general.get("mesh_collision_check_enabled", True))
    except Exception as e:  # noqa: BLE001
        logger.debug("collision: master-toggle read failed (%s); defaulting on", e)
        return True


# ---------------------------------------------------------------------------
# Active-tool lookup (NiceGUI-bound via robot_state)
# ---------------------------------------------------------------------------


def _resolve_active_tool_meshes(
    mesh_dir: Any,
) -> tuple[str, dict[str, list[Any]]]:
    """Look up the currently-active tool's mesh files, grouped by role.

    Returns ``(tool_key, {"BODY": [Path, ...], "JAW": [Path, ...]})``.

    Resolves the active tool from the GUI's logical selection
    (``app.storage.general["selected_tool"]``) — which includes
    ``custom:<name>`` keys — rather than ``robot_state.tool_key``,
    which only reflects what the controller broadcasts and only ever
    knows BUILT-IN tool keys. For custom tools that have a
    ``proxy_tool_key`` set (so motor commands route through a
    built-in), the controller broadcasts the proxy and
    ``robot_state.tool_key`` would yield e.g. ``"VACUUM"`` even when
    the GUI is actually presenting ``custom:msg_ai_realsense``.
    Using the GUI selection means the collision check loads the
    correct gripper meshes for the user's actual tool.

    Falls back to ``robot_state.tool_key``, then ``"NONE"`` (no
    gripper meshes) when both lookups fail.
    """
    tool_key: str | None = None
    try:
        from .custom_tools import _active_gui_tool_key  # noqa: PLC0415

        tool_key = _active_gui_tool_key()
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "collision: GUI active-tool lookup failed (%s); falling back "
            "to robot_state", e,
        )
    if not tool_key:
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            tool_key = getattr(robot_state, "tool_key", None)
        except Exception as e:  # noqa: BLE001
            logger.debug("collision: robot_state unavailable: %s", e)
    if not tool_key:
        tool_key = "NONE"
    return tool_key, resolve_tool_meshes_from_registry(tool_key, Path(mesh_dir))


# ---------------------------------------------------------------------------
# Config assembly from settings
# ---------------------------------------------------------------------------


def _config_from_settings(
    *,
    gripper_only: bool = False,
    include_tablet: bool = True,
) -> CollisionEnvironmentConfig | None:
    """Build a :class:`CollisionEnvironmentConfig` from the live
    NiceGUI state + settings layer + ``_T_BOARD2BASE``.

    Returns None when the parol6 mesh directory can't be located —
    callers treat that as fail-open.
    """
    mesh_dir = parol6_mesh_dir()
    if mesh_dir is None:
        return None

    tool_key, tool_meshes = _resolve_active_tool_meshes(mesh_dir)
    body_paths = tuple(tool_meshes["BODY"])
    jaw_paths = tuple(tool_meshes["JAW"])

    surface_enabled = bool(settings.surface_enabled) and include_tablet
    surface_dims_value = (
        tuple(settings.surface_dimensions_m) if surface_enabled else None
    )
    tablet_pose = (
        np.asarray(_T_BOARD2BASE, dtype=np.float64) if surface_enabled else None
    )

    board_w = int(settings.board_squares_x) * float(settings.board_square_length_m)
    board_h = int(settings.board_squares_y) * float(settings.board_square_length_m)

    return CollisionEnvironmentConfig.from_tablet_matrix(
        gripper_only=gripper_only,
        tool_key=tool_key,
        body_mesh_paths=body_paths,
        jaw_mesh_paths=jaw_paths,
        tablet_T_board2base=tablet_pose,
        tablet_dimensions_m=surface_dims_value,
        tablet_offset_local_m=tuple(settings.surface_offset_local_m),
        tablet_board_size_m=(board_w, board_h),
        floor_enabled=bool(settings.floor_primitive_enabled),
        safety_margin_m=float(settings.collision_safety_margin_m),
    )


# ---------------------------------------------------------------------------
# Backwards-compatible private helpers
# ---------------------------------------------------------------------------


def _build_collision_manager(
    tablet_T_board2base: NDArray[np.float64] | None = None,
) -> tuple[Any, set[tuple[str, str]], dict[str, Any]] | None:
    """Backwards-compatible shim. ``tablet_T_board2base`` is accepted for
    signature compatibility but the live ``_T_BOARD2BASE`` is always used
    when the surface setting is enabled — which matches the previous
    behaviour because every existing caller already passed
    ``tablet_T_board2base=_T_BOARD2BASE``.
    """
    config = _config_from_settings(gripper_only=False, include_tablet=True)
    if config is None:
        return None
    return _core_build(config)


def _self_collides(
    manager: Any,
    adjacent: set[tuple[str, str]],
    joint_angles_rad: NDArray[np.float64],
) -> bool:
    """Backwards-compatible re-export of the core self-collision query."""
    return _core_self_collides(manager, adjacent, joint_angles_rad)


def _trajectory_collides(
    manager: Any,
    adjacent: set[tuple[str, str]],
    q_from: NDArray[np.float64],
    q_to: NDArray[np.float64],
    n_samples: int = 10,
) -> bool:
    """Backwards-compatible re-export of the core trajectory query."""
    return _core_trajectory_collides(
        manager, adjacent, q_from, q_to, n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# Public workhorse
# ---------------------------------------------------------------------------


def validate_joint_trajectory(
    q_from: NDArray[np.float64] | list[float] | tuple[float, ...],
    q_to: NDArray[np.float64] | list[float] | tuple[float, ...],
    *,
    n_samples: int = 10,
    degrees: bool = True,
    gripper_only: bool = False,
) -> dict[str, Any]:
    """Pre-validate a joint-space move for collision before dispatching.

    Args:
        q_from: 6-vector start joint angles.
        q_to: 6-vector target joint angles.
        n_samples: Number of interior interpolation points to check.
            Endpoints are also checked.
        degrees: True if ``q_from`` / ``q_to`` are degrees.
        gripper_only: When True, skip the arm-link block in the
            collision manager so only gripper-vs-(floor + tablet + jaws)
            is checked. Use for callers where the arm's joint
            configuration is already trusted (user programs whose
            IK has produced a known-good joint vector).

    Returns:
        dict with keys ``safe``, ``start_safe``, ``end_safe``,
        ``interior_safe``, ``manager_ready``, ``reason``.

        ``manager_ready=False`` means the master toggle is off OR
        python-fcl / parol6 link meshes are unavailable; in that case
        ``safe`` defaults to True (fail-open) and the caller should
        treat the move as accepted.
    """
    if not _mesh_collision_enabled():
        return {
            "safe": True,
            "start_safe": True,
            "end_safe": True,
            "interior_safe": True,
            "manager_ready": False,
            "reason": "mesh collision check disabled in Settings",
            "colliding_pair": None,
            "colliding_q": None,
            "start_pair": None,
            "start_q": None,
        }

    config = _config_from_settings(
        gripper_only=gripper_only,
        include_tablet=True,
    )
    if config is None:
        return {
            "safe": True,
            "start_safe": True,
            "end_safe": True,
            "interior_safe": True,
            "manager_ready": False,
            "reason": "collision-manager unavailable",
            "colliding_pair": None,
            "colliding_q": None,
            "start_pair": None,
            "start_q": None,
        }

    return validate_joint_trajectory_core(
        q_from, q_to,
        config=config,
        n_samples=n_samples,
        degrees=degrees,
    )


# ---------------------------------------------------------------------------
# Cache shim: localise.py invalidates _state["trajectory_collision_mgr_pair"]
# on board-pose change. The new core has its own keyed cache so this isn't
# strictly necessary anymore, but a stale entry there harms nothing.
# Frustum still reads it as an opaque tuple of (mgr, adjacent, meshes).
# ---------------------------------------------------------------------------


def _refresh_state_cache_pair() -> None:
    """Rebuild ``_state['trajectory_collision_mgr_pair']`` from the live
    settings + board pose so frustum's per-tick mesh visualisation has a
    consistent reference. The collision-core cache handles its own
    invalidation by config-key; this one is purely for the legacy
    ``_state`` slot.
    """
    pair = _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
    _state["trajectory_collision_mgr_pair"] = pair


def _warm_collision_managers_blocking() -> None:
    """Pre-build both collision managers (full + gripper-only) so the
    first ``_raycast_footprint_tick`` and ``_live_pose_indicator_tick``
    don't block the asyncio loop doing trimesh.load + FCL BVH build.

    Each manager build costs 2-5 s on cold start (10+ STL loads, BVH
    construction for every link + gripper body + jaws + floor + tablet),
    and the two ticks use DIFFERENT configs (``gripper_only=False`` vs
    ``True``), so the collision_core keyed cache doesn't share them —
    both get built independently on first hit.

    Intended to be invoked via ``asyncio.to_thread`` from
    ``add_overlays`` so the heavy work runs on a worker thread while
    the page renders. The full pair is stashed into the legacy
    ``_state`` slot that ``_raycast_footprint_tick`` reads; the
    gripper-only manager is left in the collision_core cache for
    ``validate_joint_trajectory`` to find on its next call.
    """
    from parol6_vision.calibration.collision_core import (  # noqa: PLC0415
        build_collision_manager as _core_build,
    )

    # Full manager for trajectory checks (frustum tick + path-preview).
    full_pair = _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
    if full_pair is not None:
        _state["trajectory_collision_mgr_pair"] = full_pair

    # Gripper-only manager for the live-pose collision indicator.
    # Not stashed in _state; collision_core's keyed cache serves
    # subsequent ``validate_joint_trajectory`` calls.
    gripper_cfg = _config_from_settings(gripper_only=True, include_tablet=True)
    if gripper_cfg is not None:
        _core_build(gripper_cfg)
