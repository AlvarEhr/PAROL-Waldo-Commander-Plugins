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
    """Cast rays from the camera apex against static scene primitives
    (``VISUAL_FLOOR`` / ``VISUAL_TABLET`` in ``meshes``) and return hits.

    Arm / gripper meshes are skipped — the camera is mounted on the
    gripper looking outward and can't see its own arm.

    Returns ``(footprint_hits, camera_world_pos, center_hit)`` in world
    metres. Rays that miss terminate at ``default_depth_m`` for bounded viz.
    """
    T_cam2base = T_flange2base @ T_cam2flange
    cam_pos_world = T_cam2base[:3, 3]

    # 4 far corners in camera frame, at the default maximum depth
    corners_cam = _frustum_corners_local(default_depth_m)[1:]

    # Edge points along the far plane — 10 segments × 4 edges = 40 rays,
    # plus one centre ray for the optical axis line.
    edge_points = []
    num_segments = 10
    for i in range(4):
        p_start = np.array(corners_cam[i])
        p_end = np.array(corners_cam[(i + 1) % 4])
        for t in np.linspace(0, 1, num_segments, endpoint=False):
            edge_points.append(p_start * (1 - t) + p_end * t)

    edge_points.append(np.array([0.0, 0.0, default_depth_m]))

    num_rays = len(edge_points)

    ray_origins = np.tile(cam_pos_world, (num_rays, 1))
    ray_directions = []
    for c in edge_points:
        c_world = (T_cam2base @ np.array([c[0], c[1], c[2], 1.0]))[:3]
        ray_directions.append(c_world - cam_pos_world)
    ray_directions = np.array(ray_directions)

    norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    ray_directions /= norms

    best_t = np.full(num_rays, np.inf)

    # The mesh dict already holds world-frame transforms.
    for name, mesh in meshes.items():
        if name not in ("VISUAL_FLOOR", "VISUAL_TABLET"):
            continue

        T_obj2base = np.eye(4, dtype=np.float64)

        T_base2obj = np.linalg.inv(T_obj2base)

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
        # 2 mm pull-back avoids Z-fighting; only when we actually hit.
        if best_t[i] < np.inf:
            t = max(0.0, t - 0.002)
        hit_world = ray_origins[i] + ray_directions[i] * t
        hits_world.append(tuple(hit_world.tolist()))

    footprint_hits = hits_world[:-1]
    center_hit = hits_world[-1]

    return footprint_hits, tuple(cam_pos_world.tolist()), center_hit


def _populate_frustum(scene_group: Any, T_cam2flange: NDArray[np.float64]) -> list[Any]:
    """Add frustum lines inside ``scene_group`` (parented to tcp_anchor).

    Draws the near cone at ``_FRUSTUM_DEPTH_M`` (bright red, full apex +
    far-plane lines). The optional far cone at ``_FRUSTUM_FAR_DEPTH_M``
    extends as a "laser pointer" — fainter, far rectangle + edge extensions
    only. Corners are pre-transformed by ``T_cam2flange``.
    """
    R = T_cam2flange[:3, :3]
    t = T_cam2flange[:3, 3]

    def to_flange(corners: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        return [tuple((R @ np.asarray(c) + t).tolist()) for c in corners]

    # Lazy-cache the workspace envelope for downstream reachability checks.
    _ensure_workspace_envelope()

    near = to_flange(_frustum_corners_local(_FRUSTUM_DEPTH_M))

    # Diagnostic — confirms the tilt is taking effect at geometry level.
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
        # Sub-group so the "Fixed frustum" toggle can hide all eight lines
        # with a single `.visible(False)`.
        near_cone_group = (
            ui.scene.group()
            .with_name("calib:near_cone")
            .visible(_state.get("show_near_cone", True))
        )
        _state["near_cone_group"] = near_cone_group
        with near_cone_group:
            # Apex-to-corner rays + far-plane rectangle.
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
    """Replace frustum lines after a mount update. Deletes the whole
    ``near_cone_group`` (not just line children) so browsers that batch
    websocket diffs lazily drop every line in one scene-tree diff.
    """
    group = _state.get("frustum_group")
    if group is None:
        return
    old_cone = _state.get("near_cone_group")
    if old_cone is not None:
        try:
            old_cone.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["near_cone_group"] = None
    # Belt-and-braces — also delete each tracked line in case the group
    # reference was stale.
    for obj in _state.get("frustum_objects", []) or []:
        try:
            obj.delete()
        except Exception:  # noqa: BLE001
            pass
    _state["frustum_objects"] = _populate_frustum(group, T_cam2flange)


def _post_calibration_tick() -> None:
    """Apply the calibrated mount to the frustum once calibration finishes.
    Robot animation rides on the existing status-broadcast path.
    """
    try:
        if not _state.get("is_running") and _state.get("calibrated_mount") is not None:
            update_frustum(_state["calibrated_mount"].T_cam2flange)
            _state["current_mount"] = _state["calibrated_mount"]
            _state["calibrated_mount"] = None
    except RuntimeError as e:
        # "The parent slot of the element has been deleted."
        if "parent slot" in str(e):
            return
        raise


# ``_raycast_footprint_tick`` runs on the asyncio loop. Three early exits
# keep it cheap: skip when both overlays are hidden, skip when inputs
# haven't moved past the deltas below, and the tick rate itself is low.
_FOOTPRINT_TICK_HZ: float = 5.0
_FOOTPRINT_JOINT_DELTA_RAD: float = 1e-4    # ~0.006° per joint
_FOOTPRINT_MOUNT_DELTA_M: float = 1e-5      # 10 µm translation


def _footprint_inputs_changed(
    q: NDArray[np.float64],
    T_cam2flange: NDArray[np.float64],
) -> bool:
    """True when joint angles or mount differ enough to rebuild, or when
    nothing's rendered (covers settings-change / page-rebuild drops).
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


def _delete_footprint_group() -> None:
    """Drop the dynamic-overlay sub-group AND tracked handles so the next
    render starts clean. One-shot group delete avoids the websocket-batching
    artefact where individually-deleted lines linger client-side.
    """
    grp = _state.get("footprint_group")
    if grp is not None:
        try:
            grp.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["footprint_group"] = None
    # Belt-and-braces — also delete each tracked line in case the group
    # reference was stale.
    for obj in _state.get("footprint_objects", []) or []:
        try:
            obj.delete()
        except Exception:  # noqa: BLE001
            pass
    _state["footprint_objects"] = []


def _raycast_footprint_tick() -> None:
    """Per-tick raycast footprint + centerline update at ``_FOOTPRINT_TICK_HZ``."""
    try:
        # Both overlays hidden — tear down and skip.
        show_footprint = bool(_state.get("show_footprint", True))
        show_centerline = bool(_state.get("show_centerline", True))
        if not show_footprint and not show_centerline:
            if _state.get("footprint_objects") or _state.get("footprint_group"):
                _delete_footprint_group()
                _state["footprint_last_q"] = None
                _state["footprint_last_mount"] = None
            return

        # Three.js drops scene mutations until ``init_objects`` runs; the
        # scene_initialized flag is flipped from its 'init' hook.
        if not _state.get("scene_initialized", False):
            return

        # Wait for the first status broadcast — ``angles`` defaults to
        # zeros otherwise. ``robot_state.connected`` is the hardware-ping
        # flag and stays False in fake-serial mode, so don't gate on it.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
        except ImportError:
            return
        if float(getattr(robot_state, "last_update_ts", 0.0)) <= 0.0:
            if _state.get("footprint_objects") or _state.get("footprint_group"):
                _delete_footprint_group()
                _state["footprint_last_q"] = None
                _state["footprint_last_mount"] = None
            return

        scene_root = _state.get("scene_root")
        if scene_root is None:
            return
        # Liveness guard against page-teardown / hot-reload races.
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
            # Skip while cold-start warmup is running rather than build
            # the FCL manager synchronously on the loop.
            if _state.get("collision_mgr_warming", False):
                return
            pair = _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
            if pair is None:
                return
            _state["trajectory_collision_mgr_pair"] = pair
        # Only the meshes dict matters here.
        _, _, meshes = pair

        try:
            from waldo_commander.state import ui_state  # noqa: PLC0415
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

        # No meaningful input change since the last tick.
        if not _footprint_inputs_changed(q, mount.T_cam2flange):
            return

        try:
            hits_world, cam_world, center_hit_world = _raycast_frustum_footprint(
                T_flange2base,
                mount.T_cam2flange,
                meshes,
                _FRUSTUM_FAR_DEPTH_M or 1.5,
            )

            # One-shot delete so every old line vanishes in one diff.
            _delete_footprint_group()

            objects = []
            with scene_root:
                # Per-tick container — the next ``_delete_footprint_group``
                # tears the whole bundle down in one call.
                footprint_group = (
                    ui.scene.group().with_name("calib:footprint_group")
                )
                _state["footprint_group"] = footprint_group
                with footprint_group:
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
            # Force a client-side flush so the first post-broadcast tick
            # doesn't wait for an unrelated event to show its geometry.
            try:
                scene_root.update()
            except Exception as e:  # noqa: BLE001
                logger.debug("scene_root.update() failed: %s", e)
            # Cache for the next tick's short-circuit check.
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
