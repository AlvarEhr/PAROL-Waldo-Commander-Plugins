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
from .state import _hemi_azimuth_world_range_deg, _state
from .workspace import _ensure_workspace_envelope, envelope_contains

logger = logging.getLogger(__name__)


def _compute_reachability_candidates(
    target_world: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], list[Any]] | None:
    """Run the hemisphere IK sweep + farthest-first thinning. Pure compute
    — no scene mutation, safe to run off the asyncio loop.

    Returns ``(cam_positions_world, candidates)`` lined up index-for-index,
    or ``None`` on import / Robot-instantiation failure (caller should
    skip rendering and log).

    Why this is a hot path: the IK sweep runs ~1024 candidates through
    pinokin's IKSolver, which takes 1-3 seconds. Doing this on the asyncio
    event loop blocks the websocket pump long enough for Socket.IO to
    drop the browser connection. Run it from a thread instead via
    ``run_in_executor`` and call ``_render_reachability_dots`` on the loop
    once it returns.
    """
    try:
        from parol6 import Robot  # noqa: PLC0415
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
        from parol6_vision.calibration.pose_generator import (  # noqa: PLC0415
            HemisphereParams,
            PoseGenerator,
        )
    except ImportError:
        logger.debug("parol6/parol6_vision not importable; skipping reachability viz")
        return None

    _ensure_workspace_envelope()

    cam_translate = settings.cam_mount_translate_mm
    cam_tilt = settings.cam_mount_tilt_deg
    cold_start = CameraMount.from_eyeball_estimate(
        x_mm=cam_translate[0],
        y_mm=cam_translate[1],
        z_mm=cam_translate[2],
        tilt_x_deg=cam_tilt[0],
        tilt_y_deg=cam_tilt[1],
        tilt_z_deg=cam_tilt[2],
    )

    d_min, d_max = settings.hemi_distance_range_m
    ev_min, ev_max = settings.hemi_elevation_range_deg
    az_min, az_max = _hemi_azimuth_world_range_deg()

    try:
        robot = Robot()
    except Exception as e:  # noqa: BLE001
        logger.warning("could not instantiate Robot for reachability viz: %s", e)
        return None

    if _REACHABILITY_USE_CONTINUOUS:
        # Sobol low-discrepancy sampling — provably uniform 3D coverage of
        # the hemisphere volume. Discrete-grid sampling produces visible
        # "ring" artefacts in the surviving set when IK feasibility
        # correlates with grid axes (which it does on PAROL6 with
        # tilt_x=180 — survivors cluster along specific azimuth bands).
        params = HemisphereParams(
            n_candidates=_REACHABILITY_N_CANDIDATES,
            distance_range_m=(d_min, d_max),
            elevation_range_deg=(ev_min, ev_max),
            azimuth_range_deg=(az_min, az_max),
            workspace_xy_max_m=0.55,
            max_joint_change_deg=180.0,
        )
        max_count = _REACHABILITY_N_CANDIDATES
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

    gen = PoseGenerator(
        robot=robot, mount=cold_start, target_world=target_world, params=params,
    )
    cands, stats = gen.generate(max_count=max_count)

    # Extract camera positions; final flange-hull check (PoseGenerator's
    # workspace_xy/z bounds are rectangular; the hull is more accurate).
    reachable_cam_world: list[NDArray[np.float64]] = []
    reachable_candidates: list[Any] = []
    for c in cands:
        T_cam2base = cold_start.cam_pose_for_flange_pose(np.asarray(c.flange_pose))
        cam_pos = T_cam2base[:3, 3]
        if not bool(envelope_contains(np.asarray(c.flange_pose)[:3, 3])[0]):
            continue
        reachable_cam_world.append(cam_pos)
        reachable_candidates.append(c)

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
        cam_arr = np.asarray(reachable_cam_world, dtype=np.float64)
        rel = cam_arr - target_world
        dists = np.linalg.norm(rel, axis=1)
        elevs_deg = np.degrees(
            np.arcsin(np.clip(rel[:, 2] / np.maximum(dists, 1e-9), -1.0, 1.0))
        )
        out_of_shell = ((dists < d_min - 1e-3) | (dists > d_max + 1e-3)).sum()
        logger.info(
            "reachability dots distance range %.3f - %.3f m (shell %.3f - %.3f), "
            "elevation range %.1f - %.1f° (shell %.1f - %.1f); out-of-shell: %d",
            float(dists.min()), float(dists.max()), d_min, d_max,
            float(elevs_deg.min()), float(elevs_deg.max()), ev_min, ev_max,
            int(out_of_shell),
        )
        if out_of_shell > 0:
            logger.warning(
                "%d reachability dot(s) lie OUTSIDE the wireframe shell — "
                "dots/wireframe disagreeing about the hemisphere centre.",
                int(out_of_shell),
            )

    # Greedy farthest-first thinning for uniform spread.
    selected_candidates = reachable_candidates
    points_to_render = reachable_cam_world
    if (
        _REACHABILITY_KEEP_COUNT is not None
        and len(reachable_candidates) > _REACHABILITY_KEEP_COUNT
    ):
        paired = list(zip(reachable_cam_world, reachable_candidates))
        paired = _greedy_farthest_first(
            paired, _REACHABILITY_KEEP_COUNT, key=lambda p: p[0],
        )
        points_to_render = [p[0] for p in paired]
        selected_candidates = [p[1] for p in paired]
        logger.info(
            "farthest-first thinning: %d -> %d points",
            len(reachable_cam_world), len(points_to_render),
        )

    return points_to_render, selected_candidates


def _render_reachability_dots(
    scene_group: Any,
    target_world: NDArray[np.float64],
    points_to_render: list[NDArray[np.float64]],
) -> None:
    """Create the green-sphere sub-group inside ``scene_group``. Scene
    mutation only — must run on the asyncio loop (NiceGUI scene is not
    thread-safe).

    ``scene_group`` is already translated to ``target_world``, so each
    sphere's local position is ``cam_pos - target_world``. 3 mm radius +
    70 % opacity so dense regions visibly stack instead of fusing.
    """
    if scene_group is None:
        return
    try:
        with scene_group:
            reach_group = (
                ui.scene.group()
                .with_name("calib:reachability")
                .visible(_state.get("show_reachability", True))
            )
            _state["reachability_group"] = reach_group
            with reach_group:
                for cam_pos in points_to_render:
                    local = (cam_pos - target_world).tolist()
                    (
                        ui.scene.sphere(0.003)
                        .move(*local)
                        .material("#33dd66", opacity=0.7)
                    )
    except Exception as e:  # noqa: BLE001
        # Page-teardown / parent-slot races. Fine to swallow — next
        # rebuild will populate the group.
        if "parent slot" not in str(e):
            logger.warning("reachability render failed: %s", e)


def _start_reachability_compute_async(
    scene_group: Any,
    target_world: NDArray[np.float64],
) -> None:
    """Dispatch the (slow) IK sweep to a thread; render results on the loop
    when it finishes.

    Without this, the sweep blocks the asyncio event loop for 1-3 s — long
    enough that Socket.IO's outgoing buffer fills under load (50 Hz URDF
    status broadcasts + 5 Hz frustum tick) and the browser disconnects.
    Doing the IK in a thread keeps the loop responsive; only the cheap
    scene-rendering step lands back on the loop.
    """
    loop = _state.get("main_loop")

    def _worker() -> None:
        result = _compute_reachability_candidates(target_world)
        if result is None:
            return
        points, selected = result
        # Cache candidates so the board-localise sweep can reuse them.
        _state["reachable_candidates"] = selected
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(
                _render_reachability_dots, scene_group, target_world, points,
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
