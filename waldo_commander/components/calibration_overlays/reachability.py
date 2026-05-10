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
    """Run the hemisphere IK sweep with the supplied pre-resolved
    ``cold_start`` mount + ``params``. Pure compute, safe to run off
    the asyncio loop.

    All settings reads happen on the caller (main thread) so per-tool
    overrides stored in ``app.storage.user`` are picked up correctly.
    NiceGUI's ``app.storage.user`` is request-context-bound and raises
    ``RuntimeError`` from worker threads, so doing the lookups here
    would silently fall back to globals + miss the active tool's
    calibrated values.

    ``collision_config`` (built by the caller on the main thread)
    triggers an end-pose gripper-only collision filter — candidates
    whose target joints would clip the floor / tablet are dropped.
    Passing ``None`` skips the filter entirely (legacy behaviour;
    user could see "reachable" dots that ``_go_to_pose_for_candidate``
    would block at click time).

    Returns ``(cam_positions_world, candidates)`` lined up
    index-for-index, or ``None`` on Robot-instantiation failure.
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

    # Lazy import to avoid pulling parol6_vision into the package-load
    # path when calibration features are off.
    try:
        from parol6_vision.calibration.pose_generator import PoseGenerator  # noqa: PLC0415
    except ImportError:
        logger.debug("parol6_vision not importable; skipping reachability viz")
        return None

    gen = PoseGenerator(
        robot=robot, mount=cold_start, target_world=target_world, params=params,
    )
    cands, stats = gen.generate(max_count=max_count)

    # Extract camera positions; final flange-hull check (PoseGenerator's
    # workspace_xy/z bounds are rectangular; the hull is more accurate).
    # When ``collision_config`` is supplied, ALSO drop candidates whose
    # end pose collides with the floor / tablet — keeps the dots
    # consistent with what the click-to-go path will accept at dispatch.
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
                # End-pose-only check: pass the same q for from/to with
                # n_samples=0 so only the static configuration is
                # tested (no interior interpolation, no live current_q
                # dependence — the dots are POSITIONS, not paths).
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
                # Fail-open per candidate: if FCL hiccups, keep the
                # dot rather than silently dropping it.
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
        # Pull the distance + elevation shell bounds back out of the
        # params object so the diagnostic log can flag dots that fell
        # outside the visualised shell. HemisphereParams supports both
        # continuous (range tuples) and discrete (per-axis tuples)
        # modes; cover both shapes.
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

    # Return the FULL reachable set; non-overlap thinning happens at
    # render time via ``_select_visible_dots`` (where the user-set
    # dot radius is available so the spread threshold matches the
    # actual sphere size).
    return reachable_cam_world, reachable_candidates


def _select_visible_dots(
    points: list[NDArray[np.float64]],
    candidates: list[Any],
    radius_m: float,
    n_target: int,
) -> tuple[list[NDArray[np.float64]], list[Any]]:
    """Pick a spread-out subset of the reachable set.

    Greedy farthest-first selection up to ``n_target`` items. The
    min-distance threshold ``2 * radius_m`` keeps spheres from
    overlapping in the rendered scene. Stops when EITHER the target
    count is reached OR no remaining candidate sits far enough from
    every already-picked one (so we don't draw overlapping dots even
    if the user asked for more than fit at this radius).

    Returns ``(visible_points, visible_candidates)`` paired
    index-for-index.
    """
    n = len(points)
    if n == 0 or n_target <= 0:
        return [], []
    cap = min(n_target, n)
    # Special case: no overlap constraint, just take the most
    # spread-out ``cap`` items.
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
    """Create the green-sphere sub-group inside ``scene_group``. Scene
    mutation only; must run on the asyncio loop (NiceGUI scene is not
    thread-safe).

    ``scene_group`` is already translated to ``target_world``, so each
    sphere's local position is ``cam_pos - target_world``. Each sphere
    is tagged with ``calib:reach_dot_<gen>_<i>`` so the panel-level
    click handler can identify which candidate the user clicked on
    (looked up by index against ``_state['reachable_candidates']``).

    ``visible_candidates`` (when supplied) is written into
    ``_state['reachable_candidates']`` ATOMICALLY with the generation
    bump and the sphere creation. Worker callers should pass it
    through; without that, the worker's separate write of
    ``reachable_candidates`` race-windows past the gen bump leave
    spheres tagged with the OLD gen but the candidates list pointing
    at the NEW set, and a click landing in that window would dispatch
    a candidate that doesn't correspond to the dot the user clicked.
    """
    if scene_group is None:
        return

    # Defer if NiceGUI's three.js scene hasn't reported 'init' yet —
    # otherwise the create RPCs are silently dropped (scene.js:416).
    # Re-schedule via a 0.1s timer; the same check on the next firing
    # will succeed once init has landed.
    #
    # Single-slot defer: multiple pre-init renders all collapse onto
    # the SAME pending payload, so when init lands we don't get N
    # cascading renders each bumping the gen counter and creating
    # stacked sphere groups (the now-fixed cause of "dots render but
    # aren't clickable"). The latest payload wins.
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
    # Cache the last-rendered points so a dot-radius-only change can
    # re-render with the new size without re-running the IK sweep.
    _state["reachable_points_world"] = list(points_to_render)
    _state["reachable_target_world"] = target_world
    # Idempotent: drop any prior reach_group (whether previous render or
    # an orphan from a defer-race where multiple renders queued past the
    # init gate). Without this, stacked sphere groups from different
    # generations end up at identical world positions; the raycaster
    # picks the older one first, the click handler sees a stale-gen
    # mismatch, and dismisses the click. See Docs/STATE.md "Open bugs"
    # entry for the post-9af9e5d "dots render but aren't clickable"
    # symptom.
    old_reach_group = _state.get("reachability_group")
    if old_reach_group is not None:
        try:
            old_reach_group.delete()
        except Exception as e:  # noqa: BLE001
            logger.debug("reachability prior group cleanup failed: %s", e)
        _state["reachability_group"] = None
    # Bump generation + write candidates list ATOMICALLY. Sphere
    # names embed the gen so the click handler can detect a stale
    # click (user clicked a sphere from a prior generation) and
    # ignore it. Visible-candidates write is paired with the gen
    # bump so the indices always reference the SAME list as the
    # spheres were tagged from.
    #
    # Without the lock, a worker thread (localise) bumping gen +
    # clearing candidates could interleave with this block — and
    # the click handler's gen-then-candidates read could see the
    # new gen with the old candidates list. ``_state_lock`` makes
    # the pair appear atomic to readers using the same lock.
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
        # Page-teardown / parent-slot races. Fine to swallow; next
        # rebuild will populate the group.
        if "parent slot" not in str(e):
            logger.warning("reachability render failed: %s", e)
        # Clear the partially-built group ref so the next render
        # cycle's idempotent cleanup doesn't try to .delete() a
        # half-populated handle (the delete itself is wrapped in
        # try/except, but the resulting log noise is confusing —
        # better to start clean).
        _state["reachability_group"] = None
    # Refresh the settings-panel info label ("X / Y non-overlapping")
    # so the user sees the post-render counts. Best-effort.
    _notify_reachability_info_changed()


def _flush_pending_render() -> None:
    """Defer-timer callback: re-attempt the pending render.

    Reads the latest payload from ``_state["reachability_pending_payload"]``
    and either renders (if scene_initialized has flipped) or re-arms
    the timer for another 0.1s. Single pending payload + single
    pending timer keep the gen counter from cascading — late-arriving
    renders always overwrite the prior payload.
    """
    _state["reachability_pending_timer"] = None
    payload = _state.get("reachability_pending_payload")
    if payload is None:
        return
    # Don't drop the payload here — _render_reachability_dots will
    # re-arm + re-store it if init still hasn't landed.
    _state["reachability_pending_payload"] = None
    scene_group, target_world, points_to_render, visible_candidates = payload
    _render_reachability_dots(
        scene_group, target_world, points_to_render, visible_candidates,
    )


def re_render_reachability_dots() -> None:
    """Re-run the non-overlap selection at the current dot radius and
    redraw (no IK sweep).

    Used when the user changes ``reachability_dot_radius_m``: the
    sphere size changes, so the min-distance threshold for the
    spread filter changes too. Operates on the cached FULL reachable
    set so the user can shrink the radius and reveal more dots, or
    grow it and watch dense regions thin out, without re-running IK.

    No-op if the previous sweep hasn't completed yet (cached list
    missing).
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
    # Don't pre-write reachable_candidates here either — pass through
    # to _render_reachability_dots where the gen bump pairs with the
    # candidates write atomically.
    old = _state.get("reachability_group")
    if old is not None:
        try:
            old.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["reachability_group"] = None
    _render_reachability_dots(grp, target, visible_points, visible_candidates)


def refresh_reachability_for_active_tool() -> None:
    """Re-run the IK sweep with the active tool's mount + intrinsics
    and re-render the dots. Called when the user switches grippers
    (the per-tool mount / camera-bearing shape changes which poses
    are reachable) or when ``reachability_n_candidates`` changes.

    Drops the cached candidates list before spawning so a stale list
    can't be served to the click handler in the brief window before
    the new sweep finishes.
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
    _state["reachable_candidates"] = []
    _state["reachable_candidates_all"] = []
    _state["reachable_points_all"] = None
    _state["reachable_points_world"] = None
    # Lazy import; avoids a state.py / reachability.py import cycle.
    from .state import _hemi_centre_world  # noqa: PLC0415

    target_world = _hemi_centre_world()
    _start_reachability_compute_async(grp, target_world)


def _notify_reachability_info_changed() -> None:
    """Fire the settings-UI info label refresh callback so the
    "X / Y non-overlapping" text reflects the latest sweep result.
    Called from ``_render_reachability_dots`` after the dots land.
    """
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
    """Dispatch the (slow) IK sweep to a thread; render results on the
    loop when it finishes.

    Without this, the sweep blocks the asyncio event loop for 1-3 s,
    long enough that Socket.IO's outgoing buffer fills under load
    (50 Hz URDF status broadcasts + 5 Hz frustum tick) and the
    browser disconnects. Doing the IK in a thread keeps the loop
    responsive; only the cheap scene-rendering step lands back on
    the loop.

    Settings reads happen on the calling thread so per-tool overrides
    stored in ``app.storage.user`` (request-context-bound, not
    accessible from worker threads) get picked up correctly. Pre-
    resolved values are passed through to the worker as plain
    arguments.
    """
    # All settings reads happen here (main thread = request context),
    # so app.storage.user-backed per-tool overrides resolve to the
    # active tool's values rather than silently falling back to
    # globals from the worker thread.
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
        # Sobol low-discrepancy sampling: provably uniform 3D coverage
        # of the hemisphere volume. Discrete-grid sampling produces
        # visible "ring" artefacts in the surviving set when IK
        # feasibility correlates with grid axes.
        #
        # Sobol candidate count is bumped to the larger of the user
        # target and a baseline 256 so the spread filter has a rich
        # enough pre-filter pool to actually find n_target spread-out
        # dots. The non-overlap selection in _select_visible_dots
        # caps the rendered count at n_target.
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

    # Build the collision config on the MAIN THREAD where it has
    # request context (settings reads + ``_T_BOARD2BASE`` access).
    # Pass through to the worker so it can filter out candidates
    # whose end pose collides with floor / tablet — without this
    # filter, the user can click a "reachable" dot only to have
    # the click-to-go check (collision.validate_joint_trajectory)
    # reject it with "collides with TABLET" or "collides with
    # FLOOR". The dot rendered as reachable but going to it is
    # blocked, which is the user-reported "obviously not happen"
    # mismatch. End-pose check (q_target → q_target with n_samples=0)
    # is fast (~few ms) per candidate and matches what
    # ``_go_to_pose_for_candidate`` runs at click time, modulo the
    # trajectory-interior portion that depends on the live start
    # pose.
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
        # Cache the FULL reachable set so a dot-radius change can
        # re-run the non-overlap selection without re-running IK.
        # The board-localise sweep also reuses the full candidate set.
        # These two slots are write-once-per-sweep (no race with the
        # click handler, which only reads ``reachable_candidates``),
        # so worker-thread writes are safe.
        _state["reachable_points_all"] = all_points
        _state["reachable_candidates_all"] = all_candidates
        visible_points, visible_candidates = _select_visible_dots(
            all_points, all_candidates, radius, n_target,
        )
        # NOTE: ``reachable_candidates`` is intentionally NOT written
        # here — it must be set ATOMICALLY with the gen bump and
        # sphere creation inside _render_reachability_dots, or the
        # click handler can dereference candidates from one sweep
        # against spheres tagged with the prior sweep's gen.
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
    """Pick ``n_target`` items so their position-space minimum pairwise
    distance is approximately maximised.

    Algorithm: start at the centroid-closest item (deterministic), then
    repeatedly pick the next as the one with the LARGEST minimum distance
    to the already-picked set. Greedy heuristic for the maxmin-distance /
    k-center problem; produces uniform-looking spread without parameter
    tuning. O(N × n_target).

    ``key`` extracts a 3-vector position from each item, so this helper
    works on raw points (key=identity) AND on PoseGenerator candidates
    (key=lambda c: c.flange_pose[:3, 3]).
    """
    if len(items) <= n_target:
        return list(items)
    pts = np.asarray([key(item) for item in items], dtype=np.float64)
    # Start with the item closest to the centroid (deterministic seed).
    centroid = pts.mean(axis=0)
    first_idx = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
    selected = [first_idx]
    # Track min distance from each candidate to the selected set.
    min_dist = np.linalg.norm(pts - pts[first_idx], axis=1)
    while len(selected) < n_target:
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] <= 0:
            break  # all remaining items coincide with already-selected
        selected.append(next_idx)
        new_dists = np.linalg.norm(pts - pts[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dists)
    return [items[i] for i in selected]
