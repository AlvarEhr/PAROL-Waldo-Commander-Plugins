"""Scene-overlay builders: board, tablet, hemisphere, reachability, frustum."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
from nicegui import app as ng_app, ui
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRotation

from . import settings
from .constants import _DETECTION_POLL_INTERVAL_S
from .detection import _poll_detection_json
from .frustum import (
    _FOOTPRINT_TICK_HZ,
    _populate_frustum,
    _post_calibration_tick,
    _raycast_footprint_tick,
)
from .reachability import _start_reachability_compute_async
from .state import (
    _T_BOARD2BASE,
    _hemi_azimuth_center_deg,
    _hemi_azimuth_world_range_deg,
    _hemi_centre_world,
    _state,
    current_board_config,
)

logger = logging.getLogger(__name__)


def _live_pose_indicator_tick() -> None:
    """Update the live-pose collision-status chip in the panel header.

    Runs a gripper-only check on the live joint angles and updates the
    chip text + tooltip via the handles cached in ``_state``.
    """
    label = _state.get("live_pose_label")
    if label is None:
        return  # panel hasn't built yet

    # Skip while warmup is running — would build the gripper-only FCL
    # manager on the loop and undo the point of the off-loop warmup.
    if _state.get("collision_mgr_warming", False):
        return

    try:
        from waldo_commander.state import robot_state  # noqa: PLC0415

        from .collision import validate_joint_trajectory  # noqa: PLC0415

        cur = list(robot_state.angles.deg[:6])
    except (ImportError, AttributeError):
        return
    try:
        result = validate_joint_trajectory(
            cur, cur, gripper_only=True, n_samples=0,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("live pose tick failed: %s", e)
        return
    if not result.get("manager_ready", False):
        text = "?"
        color = "grey"
        tooltip = result.get("reason", "collision check unavailable")
    elif result.get("safe", True):
        text = "OK"
        color = "positive"
        tooltip = "Live pose is collision-free."
    else:
        reason = result.get("reason", "collision")
        pair = result.get("colliding_pair")
        text = "X"
        color = "negative"
        if pair is not None:
            tooltip = f"Live pose collides: {pair[0]} <-> {pair[1]} ({reason})"
        else:
            tooltip = f"Live pose collides ({reason})"
    try:
        label.text = text
        label.props(f"color={color}")
        # Update the existing tooltip in place — ``label.tooltip(...)``
        # appends a new QTooltip every tick and they stack.
        tooltip_el = _state.get("live_pose_tooltip")
        if tooltip_el is not None:
            try:
                tooltip_el.text = tooltip
            except Exception as e:  # noqa: BLE001
                logger.debug("live pose tooltip text update failed: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.debug("live pose label update failed: %s", e)


# ---------------------------------------------------------------------------
# Scene-overlay builders
# ---------------------------------------------------------------------------


def _ensure_static_mounts(merged_stl_path: Path, board_png_path: Path) -> tuple[str, str]:
    """Mount the merged STL directory + board PNG cache. Returns (stl_url, png_url)."""
    if _state.get("static_mounted"):
        return _state["stl_url"], _state["png_url"]

    ng_app.add_static_files("/calib_meshes", str(merged_stl_path.parent))
    ng_app.add_static_files("/calib_cache", str(board_png_path.parent))

    stl_url = f"/calib_meshes/{merged_stl_path.name}"
    png_url = f"/calib_cache/{board_png_path.name}"
    _state["static_mounted"] = True
    _state["stl_url"] = stl_url
    _state["png_url"] = png_url
    return stl_url, png_url


def add_overlays(urdf_scene: Any) -> None:
    """Bolt the merged STL + board + frustum onto a running ``UrdfScene``.

    Call after ``urdf_scene.show()`` (typically right after the world-axes
    lines in ``main.build_page_content``). The ``overlays_built`` flag
    makes this idempotent across cold-start re-entrants; teardown clears
    it so a toggle-off / toggle-on cycle still rebuilds.
    """
    if _state.get("overlays_built", False):
        logger.debug("add_overlays: already built; skipping duplicate call")
        return
    if urdf_scene is None or urdf_scene.scene is None:
        logger.warning("add_overlays called before UrdfScene is ready; skipping")
        return
    if urdf_scene.tcp_anchor is None:
        logger.warning("add_overlays: UrdfScene has no tcp_anchor; gripper bracket won't follow flange")

    # Hydrate runtime settings before the scene builds so geometry reflects
    # the user's saved values, then rebuild ``_T_BOARD2BASE`` to absorb any
    # shifts in board placement / surface thickness.
    settings.load_from_storage()
    from .state import (  # noqa: PLC0415
        _T_BOARD2BASE,
        rebuild_T_board2base,
        restore_recovered_board_pose,
    )

    rebuild_T_board2base()

    # Apply any previously-recovered board pose (in-memory first, then
    # storage). ``restore_recovered_board_pose`` logs on hit.
    recovered_T = restore_recovered_board_pose()
    if recovered_T is not None:
        _T_BOARD2BASE[:] = recovered_T

    # Lazy import — keep parol6-vision out of the main load path.
    try:
        from parol6_vision.calibration.board import render_board_png  # noqa: PLC0415
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
    except ImportError as e:
        logger.error("parol6-vision not importable; skipping calibration overlays: %s", e)
        return

    import cv2  # noqa: PLC0415

    pkg_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "parol6-vision"
    merged_stl = pkg_root / "parol6_vision" / "sim" / "meshes" / "ssg48_body_realsense.stl"
    if not merged_stl.exists():
        logger.warning("merged STL not found at %s; skipping calibration overlays", merged_stl)
        return

    # Re-render every restart — the PNG always reflects the current
    # BoardConfig without manual cache invalidation.
    cache_dir = Path(__file__).resolve().parent.parent.parent / "_calib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    board_png = cache_dir / "board.png"
    canonical = render_board_png(current_board_config(), pixels_per_metre=4000.0, margin_squares=0.0)
    # No vertical flip — Three.js Texture uses flipY=false and our UV
    # array already maps world (0,0,0) to the PNG's top-left.
    canonical_rgb = cv2.cvtColor(canonical, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(board_png), canonical_rgb)
    logger.info("rendered ChArUco board PNG: %s (%d×%d)", board_png, canonical.shape[1], canonical.shape[0])

    stl_url, png_url = _ensure_static_mounts(merged_stl, board_png)

    # Capture the asyncio loop for worker threads. Don't clobber a
    # previously-captured loop on re-entry without one.
    try:
        _state["main_loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass

    # Capture the NiceGUI client so worker callbacks can enter its slot
    # via ``with client:`` — ``ui.dialog()`` / ``ui.notify()`` need it.
    try:
        from nicegui import context  # noqa: PLC0415

        _state["nicegui_client"] = context.client
    except (ImportError, RuntimeError) as e:
        logger.debug("nicegui client capture failed: %s", e)
        _state["nicegui_client"] = None

    # Cancel timers on browser disconnect — without this, ``ui.timer``
    # ticks after teardown log "parent slot deleted" every tick.
    try:
        client = _state.get("nicegui_client")
        if client is not None:
            from .panel import _teardown_overlays  # noqa: PLC0415

            def _on_disconnect_cleanup() -> None:
                # Reuse the panel's teardown (cancels every tracked timer +
                # group and clears per-tool runtime state).
                try:
                    _teardown_overlays()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "calibration teardown on disconnect failed: %s", exc,
                    )

            client.on_disconnect(_on_disconnect_cleanup)
    except Exception as e:  # noqa: BLE001
        logger.debug("disconnect-handler registration failed: %s", e)

    # Delayed timer for "scene initialized" gating instead of
    # ``scene.on("init", ...)``. The listener path mutates the element
    # and triggers NiceGUI's "event listeners changed" re-mount, which
    # double-renders every mesh under three.js's non-deduping
    # ``parent.add``. 600 ms beats first-mount + ``init_objects``.
    _state["scene_initialized"] = False
    # Cancel a stale prior init-defer timer.
    old_init_timer = _state.get("scene_init_defer_timer")
    if old_init_timer is not None:
        try:
            old_init_timer.cancel()
        except Exception:  # noqa: BLE001
            pass
        _state["scene_init_defer_timer"] = None
    try:
        _state["scene_init_defer_timer"] = ui.timer(
            0.6,
            lambda: _state.update({"scene_initialized": True}),
            once=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("scene init defer-timer failed: %s", e)
        # Optimistic-true so timer-driven creates don't stay gated.
        _state["scene_initialized"] = True

    # Cache mount + paths for the worker thread.
    _state["merged_stl_path"] = merged_stl
    _state["board_png_path"] = board_png
    cam_translate = settings.cam_mount_translate_mm
    cam_tilt = settings.cam_mount_tilt_deg
    _state["current_mount"] = CameraMount.from_eyeball_estimate(
        x_mm=cam_translate[0],
        y_mm=cam_translate[1],
        z_mm=cam_translate[2],
        tilt_x_deg=cam_tilt[0],
        tilt_y_deg=cam_tilt[1],
        tilt_z_deg=cam_tilt[2],
    )

    # Camera frustum group, parented to tcp_anchor. Idempotent — drop
    # any leftover from a prior add_overlays so toggles don't accumulate.
    old_frustum = _state.get("frustum_group")
    if old_frustum is not None:
        try:
            old_frustum.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["frustum_group"] = None
        _state["near_cone_group"] = None
    if urdf_scene.tcp_anchor is not None:
        with urdf_scene.tcp_anchor:
            frustum_group = ui.scene.group().with_name("calib:frustum")
            _state["frustum_group"] = frustum_group
            _state["frustum_objects"] = _populate_frustum(
                frustum_group, _state["current_mount"].T_cam2flange
            )

    # ChArUco board in world frame (1 mm above the floor to avoid the
    # polar-grid z-fight, bordered for visibility if the texture fails).
    scene_root = urdf_scene.scene
    _state["scene_root"] = scene_root

    # ``_state`` survives browser refresh; clear stale handles + cached
    # inputs so the next tick rebuilds into the new scene_root.
    _state["footprint_objects"] = []
    _state["footprint_last_q"] = None
    _state["footprint_last_mount"] = None

    # All board-dependent overlays share a single builder so initial
    # build and post-localise refresh use the same code path.
    _build_board_dependent_overlays(scene_root, png_url)

    # Cancel prior timers so re-add doesn't stack them.
    for old_timer in _state.get("calib_timers", []) or []:
        try:
            old_timer.cancel()
        except Exception:  # noqa: BLE001
            pass
    old_det_timer = _state.get("detection_overlay_timer")
    if old_det_timer is not None:
        try:
            old_det_timer.cancel()
        except Exception:  # noqa: BLE001
            pass

    # Detection-overlay polling — repaints AABBs from
    # ``last_detection.json`` on mtime change.
    _state["detection_overlay_timer"] = ui.timer(
        _DETECTION_POLL_INTERVAL_S, _poll_detection_json,
    )

    # Page-level scene timers. Both early-exit when their preconditions
    # don't hold (see each function's docstring for gates).
    _state["calib_timers"] = [
        ui.timer(0.25, _post_calibration_tick, active=True),
        ui.timer(1.0 / _FOOTPRINT_TICK_HZ, _raycast_footprint_tick, active=True),
        ui.timer(0.5, _live_pose_indicator_tick, active=True),
    ]

    # Click-on-dot popup — idempotent container + handler.
    try:
        from . import pose_popup  # noqa: PLC0415

        pose_popup.init_popup_container()
        pose_popup.register_click_handler(urdf_scene.scene)
    except Exception as e:  # noqa: BLE001
        logger.debug("pose popup setup failed: %s", e)

    # Flag for ``apply_calibration_state`` to detect built-vs-torn-down.
    _state["overlays_built"] = True

    # Pre-build FCL managers off the loop. Synchronous builds (2-5 s
    # each) trip Socket.IO ping_timeout and force a browser reload on
    # cold start; ``_raycast_footprint_tick`` and
    # ``_live_pose_indicator_tick`` skip while ``collision_mgr_warming``.
    _state["collision_mgr_warming"] = True

    async def _warm_managers_task() -> None:
        try:
            from .collision import _warm_collision_managers_blocking  # noqa: PLC0415

            await asyncio.to_thread(_warm_collision_managers_blocking)
        except Exception as e:  # noqa: BLE001
            logger.warning("collision manager warmup failed: %s", e)
        finally:
            _state["collision_mgr_warming"] = False
            logger.debug("collision manager warmup complete")

    asyncio.create_task(_warm_managers_task())


def _build_board_overlay_group(scene_root: Any, png_url: str) -> Any:
    """Build the ChArUco board scene group at ``_T_BOARD2BASE``. Returns
    the group handle.
    """
    cfg = current_board_config()
    w_m = cfg.squares_x * cfg.square_length
    h_m = cfg.squares_y * cfg.square_length

    board_pos = _T_BOARD2BASE[:3, 3].copy()
    board_pos[2] += 0.001  # avoid floor-grid z-fight
    board_group = scene_root.group().move(*board_pos.tolist())
    # NiceGUI's rotate() uses three.js intrinsic XYZ ("xyz" lowercase in
    # scipy); _BOARD_RPY_RAD is scipy-extrinsic so we round-trip the
    # rotation matrix and re-decompose intrinsic.
    rpy = SciRotation.from_matrix(_T_BOARD2BASE[:3, :3]).as_euler("xyz").tolist()
    if any(abs(a) > 1e-6 for a in rpy):
        board_group = board_group.rotate(*rpy)
    with board_group:
        # Gray (not white) backing plane — solid-gray fallback signals
        # texture-load failure clearly. Sits below the texture to avoid
        # z-fighting.
        ui.scene.box(w_m, h_m, 0.001).move(w_m / 2, h_m / 2, -0.003).material(
            "#888888", opacity=1.0
        )
        # Reversed vertex rows orient the triangle winding +Z; the front
        # face is what reads correctly when viewed from the robot side.
        ui.scene.texture(
            png_url,
            [
                [(0, h_m, 0.0005), (w_m, h_m, 0.0005)],
                [(0, 0,    0.0005), (w_m, 0,    0.0005)],
            ],
        )
        # Bright orange outline above the texture, easy to spot at any zoom.
        ui.scene.line([0, 0, 0.0010], [w_m, 0, 0.0010]).material("#ff8800")
        ui.scene.line([w_m, 0, 0.0010], [w_m, h_m, 0.0010]).material("#ff8800")
        ui.scene.line([w_m, h_m, 0.0010], [0, h_m, 0.0010]).material("#ff8800")
        ui.scene.line([0, h_m, 0.0010], [0, 0, 0.0010]).material("#ff8800")

    logger.info("parol6-vision calibration overlay: board at %s", board_pos.tolist())
    return board_group


def _build_tablet_overlay_group(scene_root: Any) -> Any | None:
    """Translucent box overlay matching the TABLET collision primitive.
    Returns the group handle, or None when overlay rendering is off.
    """
    if not (bool(settings.surface_enabled) and bool(settings.surface_show_overlay)):
        return None

    _cfg = current_board_config()

    t_w, t_l, t_h = settings.surface_dimensions_m
    t_off_x, t_off_y = settings.surface_offset_local_m

    # Tablet centre in board-local frame → world via _T_BOARD2BASE. The
    # board pose is pre-lifted by t_h so box centre at z=-t_h/2 lands
    # between bench and screen.
    tablet_centre_local = np.array(
        [
            _cfg.squares_x * _cfg.square_length / 2.0 + t_off_x,
            _cfg.squares_y * _cfg.square_length / 2.0 + t_off_y,
            -t_h / 2.0,
            1.0,
        ],
        dtype=np.float64,
    )
    centre_world = (_T_BOARD2BASE @ tablet_centre_local)[:3]
    # Lowercase "xyz" — see ``_build_board_overlay_group``.
    rpy = SciRotation.from_matrix(_T_BOARD2BASE[:3, :3]).as_euler("xyz").tolist()

    grp = scene_root.group().move(*centre_world.tolist()).with_name("calib:tablet")
    if any(abs(a) > 1e-6 for a in rpy):
        grp = grp.rotate(*rpy)
    with grp:
        # Rusty-orange so it's distinct from the gray board backing.
        ui.scene.box(t_w, t_l, t_h).material("#cc7733", opacity=0.30)

    logger.info(
        "tablet overlay: centre=%s, dims=%s, offset=%s",
        np.round(centre_world, 3).tolist(),
        tuple(settings.surface_dimensions_m),
        tuple(settings.surface_offset_local_m),
    )
    return grp


def regenerate_board_png() -> None:
    """Re-render the cached ChArUco PNG and append a cache-buster to the
    URL. Caller is responsible for calling
    :func:`refresh_board_dependent_overlays` afterwards.
    """
    board_png = _state.get("board_png_path")
    if board_png is None:
        return
    try:
        import time  # noqa: PLC0415

        import cv2  # noqa: PLC0415
        from parol6_vision.calibration.board import render_board_png  # noqa: PLC0415

        cfg = current_board_config()
        canonical = render_board_png(
            cfg, pixels_per_metre=4000.0, margin_squares=0.0,
        )
        canonical_rgb = cv2.cvtColor(canonical, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(board_png), canonical_rgb)
        # Cache-bust by appending the timestamp — same file on disk, new URL.
        cache_buster = int(time.time())
        _state["png_url"] = f"/calib_cache/{board_png.name}?v={cache_buster}"
        logger.info(
            "regenerated board PNG: %dx%d squares, %.1f mm each, dict=%s",
            cfg.squares_x, cfg.squares_y,
            cfg.square_length * 1000, cfg.aruco_dict_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("regenerate_board_png failed: %s", e)


def refresh_board_dependent_overlays() -> None:
    """Rebuild the board, tablet, hemisphere wireframe, and reachability
    dots after ``_T_BOARD2BASE`` mutates. Frustum stays correct (parented
    to ``tcp_anchor``). Safe from any thread — schedules onto the captured
    asyncio loop.
    """
    scene_root = _state.get("scene_root")
    png_url = _state.get("png_url")
    loop = _state.get("main_loop")
    if scene_root is None or png_url is None:
        logger.warning("refresh_board_dependent_overlays: scene not initialised yet")
        return

    def _do_refresh() -> None:
        _build_board_dependent_overlays(scene_root, png_url)

    if loop is None:
        # No loop captured — caller is on the main thread.
        _do_refresh()
    else:
        try:
            loop.call_soon_threadsafe(_do_refresh)
        except RuntimeError as e:
            # Loop closed (page torn down mid-localise).
            logger.info(
                "refresh_board_dependent_overlays: loop unavailable (%s); "
                "skipping scene refresh",
                e,
            )


def _add_hemisphere_wireframe_to_group(scene_group: Any, target_world: NDArray[np.float64]) -> None:
    """Wireframe wedge for the hemisphere search region.

    The wedge is the annular spherical sector
    ``d × elev × [az_center ± spread]``: inner shell, outer shell, four
    corner edges. Wireframe (not arcs) so the user reads a solid volume
    matching what the pose generator samples.
    """
    d_min, d_max = settings.hemi_distance_range_m
    ev_min, ev_max = settings.hemi_elevation_range_deg
    az_min, az_max = _hemi_azimuth_world_range_deg()

    # Grid resolution for the surface meshing.
    n_az_segments = 8     # → 9 longitude lines per shell
    n_ev_segments = 4     # → 5 latitude lines per shell
    azimuths = np.linspace(az_min, az_max, n_az_segments + 1)
    elevations = np.linspace(ev_min, ev_max, n_ev_segments + 1)

    def offset(d: float, elev_deg: float, az_deg: float) -> tuple[float, float, float]:
        elev = np.radians(elev_deg)
        az = np.radians(az_deg)
        return (
            float(d * np.cos(elev) * np.cos(az)),
            float(d * np.cos(elev) * np.sin(az)),
            float(d * np.sin(elev)),
        )

    with scene_group:
        # Sub-group so the wireframe toggles independently of the dots.
        wf_group = (
            ui.scene.group()
            .with_name("calib:hemisphere_wireframe")
            .visible(_state.get("show_hemisphere", True))
        )
        _state["hemisphere_wireframe_group"] = wf_group
        with wf_group:
            # Inner + outer shells (latitude × longitude arcs).
            for d, opacity in [(d_min, 0.55), (d_max, 0.30)]:
                color = "#5599ff" if d == d_min else "#3366cc"
                for ev in elevations:
                    pts = [offset(d, ev, az) for az in azimuths]
                    for i in range(len(pts) - 1):
                        ui.scene.line(list(pts[i]), list(pts[i + 1])).material(
                            color, opacity=opacity
                        )
                for az in azimuths:
                    pts = [offset(d, ev, az) for ev in elevations]
                    for i in range(len(pts) - 1):
                        ui.scene.line(list(pts[i]), list(pts[i + 1])).material(
                            color, opacity=opacity
                        )

            # Corner edges connecting the shells.
            for ev, az in [
                (ev_min, az_min), (ev_min, az_max),
                (ev_max, az_min), (ev_max, az_max),
            ]:
                inner = offset(d_min, ev, az)
                outer = offset(d_max, ev, az)
                ui.scene.line(list(inner), list(outer)).material("#88bbff", opacity=0.8)

    logger.info(
        "hemisphere volume wireframe at %s: d=[%.2f, %.2f] m, "
        "elev=[%.0f, %.0f] deg, az_center=%.0f deg ±%.0f deg",
        np.round(target_world, 3).tolist(),
        d_min, d_max, ev_min, ev_max,
        _hemi_azimuth_center_deg(), float(settings.hemi_azimuth_spread_deg),
    )


def _build_board_dependent_overlays(scene_root: Any, png_url: str) -> None:
    """Build every board-dependent overlay group. Always creates each
    group; ``_state['show_*']`` drives visibility via
    :func:`_set_overlay_visible`. Run on the asyncio loop thread.
    """
    # Tear down anything that exists (post-localise rebuild).
    for key in ("board_group", "tablet_group", "hemisphere_group"):
        old = _state.get(key)
        if old is not None:
            try:
                old.delete()
            except Exception:  # noqa: BLE001
                pass
            _state[key] = None
    # Sub-group handles inside hemisphere_group die with their parent.
    for key in ("hemisphere_wireframe_group", "reachability_group"):
        _state[key] = None

    # Board + tablet — controlled by a single "Board + tablet" toggle.
    show_board = bool(_state.get("show_board", True))
    board_group = _build_board_overlay_group(scene_root, png_url)
    board_group.visible(show_board)
    _state["board_group"] = board_group

    # Tablet overlay is optional; guard against the None return.
    tablet_group = _build_tablet_overlay_group(scene_root)
    if tablet_group is not None:
        tablet_group.visible(show_board)
    _state["tablet_group"] = tablet_group

    # Hemisphere wireframe + reachability dots share a parent at the
    # hemisphere centre; each child has its own sub-group so they toggle
    # independently. The IK sweep runs off the loop (see
    # ``_start_reachability_compute_async``).
    target_world = _hemi_centre_world()
    grp = scene_root.group().move(*target_world.tolist()).with_name("calib:hemisphere")
    _state["hemisphere_group"] = grp
    _add_hemisphere_wireframe_to_group(grp, target_world)
    _start_reachability_compute_async(grp, target_world)


def _set_overlay_visible(name: str, visible: bool) -> None:
    """Toggle a 3D overlay group at runtime.

    ``name``: ``board``, ``hemisphere``, ``reachability``, ``near_cone``,
    ``centerline``, ``footprint``, ``detections``. State persists in
    ``_state['show_<name>']``; dynamic overlays read the flag every tick.
    """
    state_key = f"show_{name}"
    _state[state_key] = bool(visible)

    group_key_map = {
        "board": ("board_group", "tablet_group"),
        "hemisphere": ("hemisphere_wireframe_group",),
        "reachability": ("reachability_group",),
        "near_cone": ("near_cone_group",),
        "detections": ("detection_overlay_group",),
    }
    for grp_key in group_key_map.get(name, ()):
        grp = _state.get(grp_key)
        if grp is None:
            continue
        try:
            grp.visible(bool(visible))
        except Exception as e:  # noqa: BLE001
            # Page-teardown / parent-slot race; next rebuild reads ``_state``.
            logger.debug(
                "_set_overlay_visible(%s): %s: %s",
                name, type(e).__name__, e,
            )


