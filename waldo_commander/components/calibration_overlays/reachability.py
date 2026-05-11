"""Reachability sampling and farthest-first thinning."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from nicegui import ui
from numpy.typing import NDArray

from . import settings
from .constants import (
    _REACHABILITY_GRID,
    _REACHABILITY_KEEP_COUNT,
    _REACHABILITY_N_CANDIDATES,
    _REACHABILITY_USE_CONTINUOUS,
)
from .state import _hemi_azimuth_world_range_deg, _state, _state_lock
from .workspace import _ensure_workspace_envelope, envelope_contains

logger = logging.getLogger(__name__)


def _compute_reachability_candidates(
    target_world: NDArray[np.float64],
    cold_start: Any,
    params: Any,
    max_count: int,
    collision_config: Any = None,
) -> tuple[list[NDArray[np.float64]], list[Any]] | None:
    """Run the hemisphere IK sweep. Pure compute; safe off the loop.

    Settings must be pre-resolved by the caller because
    ``app.storage.user`` is request-context-bound. ``collision_config``
    enables an end-pose gripper-only filter so the dots match what the
    click-to-go path will accept; pass ``None`` to skip.

    Returns ``(cam_positions_world, candidates)`` paired index-for-index,
    or ``None`` on Robot-instantiation failure.
    """
    try:
        from parol6 import Robot  # noqa: PLC0415
    except ImportError:
        logger.debug("parol6 not importable; skipping reachability viz")
        return None

    _ensure_workspace_envelope()

    try:
        robot = Robot()
    except Exception as e:  # noqa: BLE001
        logger.warning("could not instantiate Robot for reachability viz: %s", e)
        return None

    # Lazy import — keeps parol6_vision off the package-load path.
    try:
        from parol6_vision.calibration.pose_generator import PoseGenerator  # noqa: PLC0415
    except ImportError:
        logger.debug("parol6_vision not importable; skipping reachability viz")
        return None

    gen = PoseGenerator(
        robot=robot, mount=cold_start, target_world=target_world, params=params,
    )
    cands, stats = gen.generate(max_count=max_count)

    # Final flange-hull check (PoseGenerator's bounds are rectangular).
    # ``collision_config`` triggers an end-pose floor/tablet filter so
    # dots stay consistent with click-to-go dispatch.
    validate_core: Any = None
    if collision_config is not None:
        try:
            from parol6_vision.calibration.collision_core import (  # noqa: PLC0415
                validate_joint_trajectory_core,
            )

            validate_core = validate_joint_trajectory_core
        except ImportError:
            validate_core = None

    reachable_cam_world: list[NDArray[np.float64]] = []
    reachable_candidates: list[Any] = []
    n_collision_dropped = 0
    for c in cands:
        T_cam2base = cold_start.cam_pose_for_flange_pose(np.asarray(c.flange_pose))
        cam_pos = T_cam2base[:3, 3]
        if not bool(envelope_contains(np.asarray(c.flange_pose)[:3, 3])[0]):
            continue
        if validate_core is not None and collision_config is not None:
            try:
                # End-pose only — same q for from/to with n_samples=0;
                # dots are positions, not paths.
                q_target_deg = np.degrees(
                    np.asarray(c.joint_angles_rad, dtype=np.float64),
                ).tolist()
                check = validate_core(
                    q_target_deg, q_target_deg,
                    config=collision_config,
                    n_samples=0,
                    degrees=True,
                )
                if check.get("manager_ready", False) and not check.get(
                    "end_safe", True,
                ):
                    n_collision_dropped += 1
                    continue
            except Exception as e:  # noqa: BLE001
                # Fail open — keep the dot on FCL hiccups.
                logger.debug(
                    "reachability end-pose collision check failed: %s", e,
                )
        reachable_cam_world.append(cam_pos)
        reachable_candidates.append(c)
    if n_collision_dropped > 0:
        logger.info(
            "reachability sampling: %d candidates dropped by end-pose "
            "collision filter (would clip floor/tablet at click-to-go)",
            n_collision_dropped,
        )

    logger.info(
        "reachability sampling: %d reachable (considered=%d, IK fails=%d, "
        "joint-jump=%d, joint-limits=%d, workspace=%d, singular=%d)",
        len(reachable_cam_world),
        stats.candidates_considered,
        stats.rejection_log.get("ik_failed", 0),
        stats.rejection_log.get("joint_jump", 0),
        stats.rejection_log.get("joint_limits", 0),
        stats.rejection_log.get("workspace_xy", 0) + stats.rejection_log.get("workspace_z", 0),
        stats.rejection_log.get("singular", 0),
    )

    if reachable_cam_world:
        # Pull shell bounds out of params for the diagnostic log. Covers
        # both continuous and discrete HemisphereParams shapes.
        d_range = getattr(params, "distance_range_m", None) or (
            min(getattr(params, "distances_m", (0.0,))),
            max(getattr(params, "distances_m", (0.0,))),
        )
        ev_range = getattr(params, "elevation_range_deg", None) or (
            min(getattr(params, "elevations_deg", (0.0,))),
            max(getattr(params, "elevations_deg", (0.0,))),
        )
        d_min, d_max = float(d_range[0]), float(d_range[1])
        ev_min, ev_max = float(ev_range[0]), float(ev_range[1])
        cam_arr = np.asarray(reachable_cam_world, dtype=np.float64)
        rel = cam_arr - target_world
        dists = np.linalg.norm(rel, axis=1)
        elevs_deg = np.degrees(
            np.arcsin(np.clip(rel[:, 2] / np.maximum(dists, 1e-9), -1.0, 1.0))
        )
        out_of_shell = ((dists < d_min - 1e-3) | (dists > d_max + 1e-3)).sum()
        logger.info(
            "reachability dots distance range %.3f - %.3f m (shell %.3f - %.3f), "
            "elevation range %.1f - %.1f deg (shell %.1f - %.1f); out-of-shell: %d",
            float(dists.min()), float(dists.max()), d_min, d_max,
            float(elevs_deg.min()), float(elevs_deg.max()), ev_min, ev_max,
            int(out_of_shell),
        )
        if out_of_shell > 0:
            logger.warning(
                "%d reachability dot(s) lie OUTSIDE the wireframe shell; "
                "dots/wireframe disagreeing about the hemisphere centre.",
                int(out_of_shell),
            )

    # Return the FULL set; non-overlap thinning happens at render time
    # via ``_select_visible_dots`` where the user dot radius is in scope.
    return reachable_cam_world, reachable_candidates


def _select_visible_dots(
    points: list[NDArray[np.float64]],
    candidates: list[Any],
    radius_m: float,
    n_target: int,
) -> tuple[list[NDArray[np.float64]], list[Any]]:
    """Pick a spread-out subset via greedy farthest-first.

    ``2 * radius_m`` keeps spheres from overlapping; stops early when no
    remaining candidate sits far enough from every picked one.
    """
    n = len(points)
    if n == 0 or n_target <= 0:
        return [], []
    cap = min(n_target, n)
    # No overlap constraint — take the most spread-out ``cap`` items.
    if radius_m <= 0.0:
        if n <= cap:
            return list(points), list(candidates)
        min_dist = 0.0
    else:
        min_dist = 2.0 * radius_m

    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    first_idx = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
    selected_idx = [first_idx]
    min_distances = np.linalg.norm(pts - pts[first_idx], axis=1)
    while len(selected_idx) < cap:
        next_idx = int(np.argmax(min_distances))
        if min_dist > 0.0 and min_distances[next_idx] < min_dist:
            break
        if min_distances[next_idx] <= 0.0:
            break  # all remaining items coincide
        selected_idx.append(next_idx)
        new_dists = np.linalg.norm(pts - pts[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_dists)
    visible_points = [points[i] for i in selected_idx]
    visible_candidates = [candidates[i] for i in selected_idx]
    return visible_points, visible_candidates


def _render_reachability_dots(
    scene_group: Any,
    target_world: NDArray[np.float64],
    points_to_render: list[NDArray[np.float64]],
    visible_candidates: list[Any] | None = None,
) -> None:
    """Create the green-sphere sub-group inside ``scene_group``. Must
    run on the asyncio loop (NiceGUI scene isn't thread-safe).

    ``scene_group`` is pre-translated to ``target_world``. Spheres tag
    ``calib:reach_dot_<gen>_<i>`` for the click handler. Passing
    ``visible_candidates`` writes ``_state['reachable_candidates']``
    atomically with the gen bump — keeps dot indices in sync.
    """
    if scene_group is None:
        return

    # Three.js drops create RPCs before 'init'. Defer with a single-slot
    # pending payload so cascaded pre-init renders don't multi-bump the
    # gen counter when 'init' lands.
    if not _state.get("scene_initialized", False):
        _state["reachability_pending_payload"] = (
            scene_group, target_world, list(points_to_render),
            list(visible_candidates) if visible_candidates is not None else None,
        )
        if _state.get("reachability_pending_timer") is None:
            try:
                _state["reachability_pending_timer"] = ui.timer(
                    0.1, _flush_pending_render, once=True,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("reachability render defer-timer failed: %s", e)
                _state["reachability_pending_timer"] = None
        return

    radius = float(settings.get("reachability_dot_radius_m"))
    if radius <= 0.0:
        radius = 0.003
    # Cache for dot-radius-only changes so re-render skips the IK sweep.
    _state["reachable_points_world"] = list(points_to_render)
    _state["reachable_target_world"] = target_world
    # Idempotent — drop any prior reach_group (current or orphan from a
    # defer-race) so stacked sphere groups can't mask current dots from
    # the raycaster. See Docs/STATE.md "Open bugs".
    old_reach_group = _state.get("reachability_group")
    if old_reach_group is not None:
        try:
            old_reach_group.delete()
        except Exception as e:  # noqa: BLE001
            logger.debug("reachability prior group cleanup failed: %s", e)
        _state["reachability_group"] = None
    # Bump gen + write candidates atomically — sphere names embed the
    # gen so the click handler can reject stale clicks. The lock pairs
    # the writes against the click handler's paired read.
    with _state_lock:
        generation = int(_state.get("reach_generation", 0)) + 1
        _state["reach_generation"] = generation
        if visible_candidates is not None:
            _state["reachable_candidates"] = list(visible_candidates)
    try:
        with scene_group:
            reach_group = (
                ui.scene.group()
                .with_name("calib:reachability")
                .visible(_state.get("show_reachability", True))
            )
            _state["reachability_group"] = reach_group
            with reach_group:
                for i, cam_pos in enumerate(points_to_render):
                    local = (cam_pos - target_world).tolist()
                    (
                        ui.scene.sphere(radius)
                        .move(*local)
                        .material("#33dd66", opacity=0.7)
                        .with_name(f"calib:reach_dot_{generation}_{i}")
                    )
    except Exception as e:  # noqa: BLE001
        # Page-teardown / parent-slot races — next rebuild repopulates.
        if "parent slot" not in str(e):
            logger.warning("reachability render failed: %s", e)
        # Clear so next cycle's idempotent cleanup starts from None.
        _state["reachability_group"] = None
    # Refresh the "X / Y non-overlapping" label, best-effort.
    _notify_reachability_info_changed()


def _flush_pending_render() -> None:
    """Defer-timer callback — re-attempt the pending render. Renders if
    'init' has landed; otherwise re-render arms a new timer.
    """
    _state["reachability_pending_timer"] = None
    payload = _state.get("reachability_pending_payload")
    if payload is None:
        return
    # ``_render_reachability_dots`` will re-arm + re-store if needed.
    _state["reachability_pending_payload"] = None
    scene_group, target_world, points_to_render, visible_candidates = payload
    _render_reachability_dots(
        scene_group, target_world, points_to_render, visible_candidates,
    )


def re_render_reachability_dots() -> None:
    """Re-run non-overlap selection at the current radius and redraw.

    No IK sweep — uses the cached full reachable set. No-op when the
    cache is missing (previous sweep still in flight).
    """
    grp = _state.get("hemisphere_group")
    all_points = _state.get("reachable_points_all")
    all_candidates = _state.get("reachable_candidates_all")
    target = _state.get("reachable_target_world")
    if grp is None or all_points is None or target is None:
        return
    radius = float(settings.get("reachability_dot_radius_m"))
    if radius <= 0.0:
        radius = 0.003
    n_target = int(settings.get("reachability_n_candidates"))
    if n_target < 1:
        n_target = 8
    visible_points, visible_candidates = _select_visible_dots(
        all_points, all_candidates or [], radius, n_target,
    )
    # Defer the candidates write to the renderer (paired with gen bump).
    old = _state.get("reachability_group")
    if old is not None:
        try:
            old.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["reachability_group"] = None
    _render_reachability_dots(grp, target, visible_points, visible_candidates)


def refresh_reachability_for_active_tool() -> None:
    """Re-run IK + re-render after a tool switch or n-candidates change.
    Drops the cached candidates list before spawning to bar a stale list.
    """
    grp = _state.get("hemisphere_group")
    if grp is None:
        return
    old = _state.get("reachability_group")
    if old is not None:
        try:
            old.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["reachability_group"] = None
    # Bump gen + clear candidates under the lock so a racing click sees
    # old spheres as stale rather than indexing the empty list.
    with _state_lock:
        _state["reach_generation"] = (
            int(_state.get("reach_generation", 0)) + 1
        )
        _state["reachable_candidates"] = []
        _state["reachable_candidates_all"] = []
    _state["reachable_points_all"] = None
    _state["reachable_points_world"] = None
    # Lazy import — breaks the state.py / reachability.py cycle.
    from .state import _hemi_centre_world  # noqa: PLC0415

    target_world = _hemi_centre_world()
    _start_reachability_compute_async(grp, target_world)


def _notify_reachability_info_changed() -> None:
    """Refresh the "X / Y non-overlapping" info label after a render."""
    cb = _state.get("reachability_info_refresh")
    if cb is None:
        return
    try:
        cb()
    except Exception as e:  # noqa: BLE001
        logger.debug("reachability info refresh failed: %s", e)


def _start_reachability_compute_async(
    scene_group: Any,
    target_world: NDArray[np.float64],
) -> None:
    """Dispatch the IK sweep to a thread; render results on the loop.

    The sweep blocks the loop for 1-3 s otherwise — long enough to
    saturate Socket.IO buffers and disconnect the browser. Settings
    must be read on the request-context main thread because
    ``app.storage.user`` is unreachable from workers.
    """
    # Resolve settings on the request-context thread for per-tool overrides.
    try:
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
        from parol6_vision.calibration.pose_generator import HemisphereParams  # noqa: PLC0415
    except ImportError:
        logger.debug("parol6_vision not importable; skipping reachability sweep")
        return

    cam_translate = settings.cam_mount_translate_mm
    cam_tilt = settings.cam_mount_tilt_deg
    cold_start = CameraMount.from_eyeball_estimate(
        x_mm=cam_translate[0], y_mm=cam_translate[1], z_mm=cam_translate[2],
        tilt_x_deg=cam_tilt[0], tilt_y_deg=cam_tilt[1], tilt_z_deg=cam_tilt[2],
    )
    d_min, d_max = settings.hemi_distance_range_m
    ev_min, ev_max = settings.hemi_elevation_range_deg
    az_min, az_max = _hemi_azimuth_world_range_deg()
    n_target = int(settings.get("reachability_n_candidates"))
    if n_target < 1:
        n_target = 8

    if _REACHABILITY_USE_CONTINUOUS:
        # Sobol gives uniform 3D coverage; discrete grids leave ring
        # artefacts when IK feasibility correlates with grid axes.
        # Floor at 256 so the spread filter has a deep enough pool.
        sweep_count = max(n_target, 256)
        params = HemisphereParams(
            n_candidates=sweep_count,
            distance_range_m=(d_min, d_max),
            elevation_range_deg=(ev_min, ev_max),
            azimuth_range_deg=(az_min, az_max),
            workspace_xy_max_m=0.55,
            max_joint_change_deg=180.0,
        )
        max_count = sweep_count
    else:
        n_d, n_ev, n_az = _REACHABILITY_GRID
        distances = tuple(np.linspace(d_min, d_max, n_d).tolist())
        elevations = tuple(np.linspace(ev_min, ev_max, n_ev).tolist())
        azimuth_counts = tuple([n_az] * n_ev)
        params = HemisphereParams(
            distances_m=distances,
            elevations_deg=elevations,
            azimuth_counts=azimuth_counts,
            azimuth_range_deg=(az_min, az_max),
            workspace_xy_max_m=0.55,
            max_joint_change_deg=180.0,
        )
        max_count = n_d * n_ev * n_az

    radius = float(settings.get("reachability_dot_radius_m"))
    if radius <= 0.0:
        radius = 0.003

    # Build the collision config on the main thread (needs request
    # context for settings + ``_T_BOARD2BASE``). Pre-filtering candidates
    # keeps "reachable" dots aligned with what click-to-go will accept.
    collision_config: Any = None
    try:
        from .collision import _config_from_settings  # noqa: PLC0415

        collision_config = _config_from_settings(
            gripper_only=True, include_tablet=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "reachability collision-filter config build skipped: %s", e,
        )
        collision_config = None

    loop = _state.get("main_loop")

    def _worker() -> None:
        result = _compute_reachability_candidates(
            target_world, cold_start, params, max_count,
            collision_config=collision_config,
        )
        if result is None:
            return
        all_points, all_candidates = result
        # Cache the full set so a radius-only change skips IK; the
        # localise sweep reuses it too. Write-once per sweep — no race
        # with the click handler.
        _state["reachable_points_all"] = all_points
        _state["reachable_candidates_all"] = all_candidates
        visible_points, visible_candidates = _select_visible_dots(
            all_points, all_candidates, radius, n_target,
        )
        # ``reachable_candidates`` is set inside the renderer paired
        # with the gen bump — don't write it here.
        logger.info(
            "reachability dots: %d reachable, %d drawn (target %d, %.1f mm radius)",
            len(all_points), len(visible_points), n_target, radius * 1000.0,
        )
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(
                _render_reachability_dots,
                scene_group, target_world, visible_points, visible_candidates,
            )
        except RuntimeError as e:
            logger.info(
                "reachability render skipped (loop unavailable): %s", e,
            )

    threading.Thread(
        target=_worker, name="calib-reachability-sweep", daemon=True,
    ).start()


def _greedy_farthest_first(
    items: list[Any],
    n_target: int,
    key: Callable[[Any], NDArray[np.float64]] = lambda x: x,  # type: ignore[assignment]
) -> list[Any]:
    """Greedy farthest-first selection (k-center heuristic, O(N · n_target)).

    Seeds at the centroid-closest item, then picks the largest-minimum-
    distance candidate each step. ``key`` extracts a 3-vector position so
    this works on raw points or pose candidates.
    """
    if len(items) <= n_target:
        return list(items)
    pts = np.asarray([key(item) for item in items], dtype=np.float64)
    centroid = pts.mean(axis=0)
    first_idx = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
    selected = [first_idx]
    min_dist = np.linalg.norm(pts - pts[first_idx], axis=1)
    while len(selected) < n_target:
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] <= 0:
            break  # remaining items coincide with already-selected
        selected.append(next_idx)
        new_dists = np.linalg.norm(pts - pts[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dists)
    return [items[i] for i in selected]
