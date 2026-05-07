"""Frustum geometry, raycast footprint and per-tick scene updates."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from nicegui import ui
from numpy.typing import NDArray

from . import settings
from .collision import _build_collision_manager
from .constants import (
    _FRUSTUM_DEPTH_M,
    _FRUSTUM_FAR_DEPTH_M,
)
from .state import _T_BOARD2BASE, _state
from .workspace import _ensure_workspace_envelope

logger = logging.getLogger(__name__)


def _frustum_corners_local(
    depth_m: float,
) -> list[tuple[float, float, float]]:
    """5 corner points of the frustum in the camera's optical frame."""
    fx = float(settings.intr_fx)
    fy = float(settings.intr_fy)
    cx = float(settings.intr_cx)
    cy = float(settings.intr_cy)
    w = int(settings.intr_width)
    h = int(settings.intr_height)
    pts: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for px, py in [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]:
        x = (px - cx) * depth_m / fx
        y = (py - cy) * depth_m / fy
        pts.append((float(x), float(y), float(depth_m)))
    return pts


def _raycast_frustum_footprint(
    T_flange2base: NDArray[np.float64],
    T_cam2flange: NDArray[np.float64],
    meshes: dict[str, Any],
    default_depth_m: float,
) -> tuple[list[tuple[float, float, float]], tuple[float, float, float], tuple[float, float, float]]:
    """Cast rays from the camera apex against the static scene primitives
    (``VISUAL_FLOOR`` and ``VISUAL_TABLET`` if present in ``meshes``) and
    return where each ray hits.

    The arm links and gripper meshes are intentionally NOT raycast against:
    the camera is mounted on the gripper looking outward, so it can't see
    its own arm in any normal configuration. If a future mount geometry
    points the camera back at itself, extend ``meshes`` with the relevant
    link entries — they're already populated by ``_build_collision_manager``.

    Returns ``(footprint_hits, camera_world_pos, center_hit)`` — all in
    world frame, all metres. ``footprint_hits`` is the four (or more)
    perimeter samples interpolated along the far-plane edges; ``center_hit``
    is the optical-axis intersection used by the centerline rendering.
    Rays that don't hit anything within ``default_depth_m`` terminate at
    that depth so the visualisation stays bounded."""
    T_cam2base = T_flange2base @ T_cam2flange
    cam_pos_world = T_cam2base[:3, 3]

    # 4 far corners in camera frame, at the default maximum depth
    corners_cam = _frustum_corners_local(default_depth_m)[1:]

    # Generate points along the edges of the far plane to handle hitting multiple surfaces smoothly
    edge_points = []
    num_segments = 10  # 10 segments per edge = 40 rays total for the perimeter
    for i in range(4):
        p_start = np.array(corners_cam[i])
        p_end = np.array(corners_cam[(i + 1) % 4])
        for t in np.linspace(0, 1, num_segments, endpoint=False):
            edge_points.append(p_start * (1 - t) + p_end * t)

    # Add the center of the far plane for the optical axis line
    edge_points.append(np.array([0.0, 0.0, default_depth_m]))

    num_rays = len(edge_points)

    # Rays in world frame
    ray_origins = np.tile(cam_pos_world, (num_rays, 1))
    ray_directions = []
    for c in edge_points:
        c_world = (T_cam2base @ np.array([c[0], c[1], c[2], 1.0]))[:3]
        ray_directions.append(c_world - cam_pos_world)
    ray_directions = np.array(ray_directions)

    # Norms is the distance from apex to the default_depth plane for each corner/point
    norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-6] = 1.0
    ray_directions /= norms

    best_t = np.full(num_rays, np.inf)

    # We check against static scene elements (floor and tablet).
    for name, mesh in meshes.items():
        if name not in ("VISUAL_FLOOR", "VISUAL_TABLET"):
            continue

        # These are already transformed to world space in _build_collision_manager
        T_obj2base = np.eye(4, dtype=np.float64)

        T_base2obj = np.linalg.inv(T_obj2base)

        # Transform rays to object local frame
        origins_local = (T_base2obj[:3, :3] @ ray_origins.T + T_base2obj[:3, 3:4]).T
        dirs_local = (T_base2obj[:3, :3] @ ray_directions.T).T

        try:
            locs, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=origins_local,
                ray_directions=dirs_local,
                multiple_hits=False
            )
            for i, ray_idx in enumerate(index_ray):
                dist = float(np.linalg.norm(locs[i] - origins_local[ray_idx]))
                if dist < best_t[ray_idx]:
                    best_t[ray_idx] = dist
        except Exception:
            pass

    hits_world = []
    for i in range(num_rays):
        t = min(best_t[i], float(norms[i][0]))
        # Pull back slightly to avoid Z-fighting with surfaces,
        # but only if we actually hit something (not at max depth)
        if best_t[i] < np.inf:
            t = max(0.0, t - 0.002)  # 2mm offset towards the camera
        hit_world = ray_origins[i] + ray_directions[i] * t
        hits_world.append(tuple(hit_world.tolist()))

    footprint_hits = hits_world[:-1]
    center_hit = hits_world[-1]

    return footprint_hits, tuple(cam_pos_world.tolist()), center_hit


def _populate_frustum(scene_group: Any, T_cam2flange: NDArray[np.float64]) -> list[Any]:
    """Add frustum lines inside ``scene_group`` (parented to tcp_anchor).

    Two cones are drawn:

    * The **near frustum** at depth ``_FRUSTUM_DEPTH_M`` — bright red, with
      the full apex-to-corner + far-plane-rectangle line set. This is the
      "where the camera is looking right now" indicator.
    * Optionally a **far frustum** at depth ``_FRUSTUM_FAR_DEPTH_M`` (set
      to ``None`` to disable) — drawn fainter, no apex-to-corner lines, just
      the far rectangle and four edge-extension lines from the near corners
      to the far corners, giving a "laser pointer" projection so you can see
      where the FOV lands on the floor / board without dynamic recomputation.

    Frustum corners are transformed by ``T_cam2flange`` so the apex sits at
    the camera optical centre relative to the flange.
    """
    R = T_cam2flange[:3, :3]
    t = T_cam2flange[:3, 3]

    def to_flange(corners: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        return [tuple((R @ np.asarray(c) + t).tolist()) for c in corners]

    # Load workspace envelope (lazy, cached). Used downstream for diagnostic
    # reachability checks of calibration candidates.
    _ensure_workspace_envelope()

    near = to_flange(_frustum_corners_local(_FRUSTUM_DEPTH_M))

    # Diagnostic: log the actual far-plane span in flange frame so we can
    # confirm the tilt rotation is taking effect at the geometry level.
    far_corners = near[1:]
    fx_span = max(c[0] for c in far_corners) - min(c[0] for c in far_corners)
    fy_span = max(c[1] for c in far_corners) - min(c[1] for c in far_corners)
    logger.info(
        "frustum near-plane span (flange frame): X=%.1f mm, Y=%.1f mm "
        "(tilt_deg=%s, depth=%.2f m)",
        fx_span * 1000, fy_span * 1000,
        settings.cam_mount_tilt_deg, _FRUSTUM_DEPTH_M,
    )

    objects: list[Any] = []
    with scene_group:
        # Wrap the near-cone lines in their own sub-group so the panel's
        # "Fixed frustum" toggle can hide/show the whole bundle at runtime
        # via a single `.visible(False)` call instead of iterating eight lines.
        near_cone_group = (
            ui.scene.group()
            .with_name("calib:near_cone")
            .visible(_state.get("show_near_cone", True))
        )
        _state["near_cone_group"] = near_cone_group
        with near_cone_group:
            # Apex-to-corner rays (bright pink) + far-plane rectangle
            # (slightly darker pink).
            for i in range(1, 5):
                objects.append(
                    ui.scene.line(list(near[0]), list(near[i])).material("#ff8080")
                )
            for i in range(1, 5):
                j = 1 + (i % 4)
                objects.append(
                    ui.scene.line(list(near[i]), list(near[j])).material("#ff5050")
                )
    return objects


def update_frustum(T_cam2flange: NDArray[np.float64]) -> None:
    """Replace frustum lines after a mount update."""
    group = _state.get("frustum_group")
    if group is None:
        return
    for obj in _state.get("frustum_objects", []):
        try:
            obj.delete()
        except Exception:  # noqa: BLE001
            pass
    _state["frustum_objects"] = _populate_frustum(group, T_cam2flange)


def _post_calibration_tick() -> None:
    """Apply the calibrated mount to the frustum once calibration finishes.

    Called by a NiceGUI timer at 4 Hz. The robot animates via the existing
    parol6-server status broadcast → Waldo-Commander status consumer →
    URDF scene path, so we don't need to drive joint updates ourselves.
    The only remaining bit of plumbing is updating the frustum once the
    calibration finishes and we have a calibrated ``T_cam2flange``.
    """
    try:
        if not _state.get("is_running") and _state.get("calibrated_mount") is not None:
            update_frustum(_state["calibrated_mount"].T_cam2flange)
            _state["current_mount"] = _state["calibrated_mount"]
            _state["calibrated_mount"] = None
    except RuntimeError as e:
        # Catch "The parent slot of the element has been deleted."
        if "parent slot" in str(e):
            return
        raise


# ``_raycast_footprint_tick`` runs on the asyncio event loop. To keep the
# UI responsive during heavy CPU spikes (calibration thread spawning IK
# sweeps, parol6's 50 Hz status broadcasts updating the URDF scene), it
# uses three early-exit optimisations:
#   1. Skip when both ``show_footprint`` and ``show_centerline`` are off
#      — no point raycasting if neither result is rendered.
#   2. Skip when ``(joint_angles, mount)`` are unchanged from the last
#      successful tick — the projection result is identical, so the
#      delete/recreate websocket churn would be wasted.
#   3. Lower fire rate (5 Hz) — visually fluid for a sanity overlay,
#      halves the websocket pressure compared to the previous 10 Hz.
_FOOTPRINT_TICK_HZ: float = 5.0
_FOOTPRINT_JOINT_DELTA_RAD: float = 1e-4    # ~0.006° per-joint epsilon
_FOOTPRINT_MOUNT_DELTA_M: float = 1e-5      # 10 µm translation epsilon


def _footprint_inputs_changed(
    q: NDArray[np.float64],
    T_cam2flange: NDArray[np.float64],
) -> bool:
    """Return True iff the joint angles or mount differ enough from the
    previous tick to warrant rebuilding the footprint geometry. Also
    returns True when there are no rendered objects yet — covers the
    case where settings changed (or page rebuilt) and the dynamic
    overlays got dropped without being re-rendered.
    """
    if not _state.get("footprint_objects"):
        return True
    last_q = _state.get("footprint_last_q")
    last_mount = _state.get("footprint_last_mount")
    if last_q is None or last_mount is None:
        return True
    if np.max(np.abs(q - last_q)) > _FOOTPRINT_JOINT_DELTA_RAD:
        return True
    if np.max(np.abs(T_cam2flange - last_mount)) > _FOOTPRINT_MOUNT_DELTA_M:
        return True
    return False


def _raycast_footprint_tick() -> None:
    """Update the dynamic ray-projected footprint at ``_FOOTPRINT_TICK_HZ``.

    Projects rays from the camera apex through the far frustum corners and
    draws the resulting polygon (and centerline) in the world scene. Three
    early-exit paths keep the asyncio loop unburdened — see the module
    comment above this function.
    """
    try:
        # Early exit (1): if neither overlay is visible, skip everything —
        # no raycast, no scene churn, no collision-manager rebuild.
        show_footprint = bool(_state.get("show_footprint", True))
        show_centerline = bool(_state.get("show_centerline", True))
        if not show_footprint and not show_centerline:
            # Tear down any stale objects from a previous-frame state
            # change so they don't linger when both flags are off.
            stale = _state.pop("footprint_objects", None)
            if stale:
                for obj in stale:
                    try:
                        obj.delete()
                    except Exception:  # noqa: BLE001
                        pass
                _state["footprint_last_q"] = None
                _state["footprint_last_mount"] = None
            return

        scene_root = _state.get("scene_root")
        if scene_root is None:
            return
        # Scene-liveness guard — avoids "parent slot deleted" exceptions
        # during page-teardown / hot-reload races.
        try:
            if hasattr(scene_root, "id") and scene_root.id is None:
                return
        except Exception:  # noqa: BLE001
            return

        mount = _state.get("current_mount")
        if mount is None:
            return

        pair = _state.get("trajectory_collision_mgr_pair")
        if pair is None:
            pair = _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
            if pair is None:
                return
            _state["trajectory_collision_mgr_pair"] = pair
        # Only the ``meshes`` dict is needed here — the collision manager
        # itself drives self-collision / trajectory checks elsewhere.
        _, _, meshes = pair

        try:
            from waldo_commander.state import robot_state, ui_state  # noqa: PLC0415
            from parol6_vision.sim.robot_kinematics import link_poses  # noqa: PLC0415

            n_joints = 6
            if ui_state.active_robot is not None:
                n_joints = ui_state.active_robot.joints.count
            if len(robot_state.angles.rad) < n_joints:
                return
            q = np.asarray(robot_state.angles.rad[:n_joints], dtype=np.float64)
            poses = link_poses(q)
            T_flange2base = poses.l6
        except Exception as e:  # noqa: BLE001
            logger.warning("footprint tick skipped (kinematics error): %s", e)
            return

        # Early exit (2): if neither joint angles nor mount have changed
        # meaningfully since the last successful tick, the footprint we
        # already drew is still correct — skip the work.
        if not _footprint_inputs_changed(q, mount.T_cam2flange):
            return

        try:
            hits_world, cam_world, center_hit_world = _raycast_frustum_footprint(
                T_flange2base,
                mount.T_cam2flange,
                meshes,
                _FRUSTUM_FAR_DEPTH_M or 1.5,
            )

            objects = []
            # Parent everything to scene_root (world frame) so the footprint
            # stays fixed to the hit surface instead of sticking to the
            # camera locally. Delete-and-recreate inside one ``with``
            # context so the websocket diff is one batched message.
            with scene_root:
                for obj in _state.get("footprint_objects", []):
                    try:
                        obj.delete()
                    except Exception:  # noqa: BLE001
                        pass
                if show_footprint:
                    perimeter_points = [list(p) for p in hits_world]
                    perimeter_points.append(list(hits_world[0]))  # close loop
                    objects.append(
                        ui.scene.polyline(perimeter_points).material("#ff00ff")
                    )
                if show_centerline:
                    objects.append(
                        ui.scene.line(list(cam_world), list(center_hit_world))
                        .material("#ffff00")
                    )
            _state["footprint_objects"] = objects
            # Cache the inputs we just rendered so the next tick can
            # short-circuit if nothing's changed.
            _state["footprint_last_q"] = q.copy()
            _state["footprint_last_mount"] = mount.T_cam2flange.copy()
        except Exception as e:  # noqa: BLE001
            if "parent slot" not in str(e):
                logger.error(
                    "footprint tick failed in raycast or render: %s",
                    e, exc_info=True,
                )

    except RuntimeError as e:
        if "parent slot" in str(e):
            return
        raise
