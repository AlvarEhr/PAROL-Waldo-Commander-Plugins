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
    """2 Hz background poll that updates the live-pose collision-status
    indicator chip in the calibration panel header.

    Reads ``robot_state.angles.deg``, runs a single-config gripper-only
    collision check against the floor + active-tool meshes (no full
    arm-vs-arm; ~7x faster), and updates ``_state["live_pose_status"]``
    with one of:

    * ``"ok"`` — current pose is collision-free.
    * ``"collide:<reason>"`` — current pose collides; reason includes
      the colliding pair when available.
    * ``"unavailable"`` — master toggle off, parol6 not importable, no
      tool selected, etc.

    The indicator chip in panel.py reads this slot and updates its
    color / tooltip on each tick.
    """
    label = _state.get("live_pose_label")
    if label is None:
        return  # panel hasn't built yet

    # Skip while the cold-start collision-manager warmup is in
    # progress. Running ``validate_joint_trajectory`` here would
    # synchronously build the gripper-only FCL manager on the event
    # loop, undoing the entire point of the off-loop warmup. The
    # warmup finishes within ~2 s of page render; this tick fires
    # again 500 ms later and updates the chip with real state.
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
        # Update the existing tooltip element in place (the chip's
        # tooltip is created ONCE in panel.py and stashed in
        # ``_state["live_pose_tooltip"]``). Calling
        # ``label.tooltip(...)`` here used to APPEND a new tooltip
        # element on every tick; the chip ended up with hundreds of
        # stacked QTooltips and hovers fired them all at once.
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

    Call this AFTER ``urdf_scene.show()`` has built the scene — typically right
    after the world-axes lines are drawn in ``main.build_page_content``.

    Idempotent: re-entrant calls during cold start (one from the
    reentrant ``_on_tool_change`` chain via ``apply_calibration_state``,
    one from ``initialize_urdf_scene`` line ~352) used to render every
    overlay group twice — a wasted ~50 ms of WebSocket churn and a
    duplicate set of board/frustum/hemisphere/reachability geometry
    that the next ``_teardown_overlays`` would have to clean up. The
    ``overlays_built`` flag is cleared by teardown, so a legitimate
    toggle-off → toggle-on cycle still rebuilds correctly.
    """
    if _state.get("overlays_built", False):
        logger.debug("add_overlays: already built; skipping duplicate call")
        return
    if urdf_scene is None or urdf_scene.scene is None:
        logger.warning("add_overlays called before UrdfScene is ready; skipping")
        return
    if urdf_scene.tcp_anchor is None:
        logger.warning("add_overlays: UrdfScene has no tcp_anchor; gripper bracket won't follow flange")

    # Pull persisted user settings into the runtime cache BEFORE the scene
    # builds so the initial board / hemisphere / cam-mount geometry uses
    # whatever the user last saved (rather than the module-default values).
    # Any exceptions here are caught inside load_from_storage; safe to call.
    settings.load_from_storage()
    # Settings load may have shifted board placement / surface thickness
    # vs. the initial _T_BOARD2BASE built at module import time. Rebuild
    # before any consumer reads it.
    from .state import (  # noqa: PLC0415
        _T_BOARD2BASE,
        rebuild_T_board2base,
        restore_recovered_board_pose,
    )

    rebuild_T_board2base()

    # Look for a previously-recovered board pose (in-memory first for
    # browser refresh, then storage for waldo-commander restart). If
    # found and the saved sim/real mode matches the current mode, apply
    # it on top of the configured pose. ``restore_recovered_board_pose``
    # logs an info line on hit so the user can see the restore happened.
    recovered_T = restore_recovered_board_pose()
    if recovered_T is not None:
        _T_BOARD2BASE[:] = recovered_T

    # Lazy import — keep parol6-vision out of the main load path so
    # Waldo-Commander still imports cleanly without it.
    try:
        from parol6_vision.calibration.board import render_board_png  # noqa: PLC0415
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
    except ImportError as e:
        logger.error("parol6-vision not importable; skipping calibration overlays: %s", e)
        return

    import cv2  # noqa: PLC0415

    # Resolve paths.
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent.parent / "parol6-vision"
    merged_stl = pkg_root / "parol6_vision" / "sim" / "meshes" / "ssg48_body_realsense.stl"
    if not merged_stl.exists():
        logger.warning("merged STL not found at %s; skipping calibration overlays", merged_stl)
        return

    # Render board texture and cache it. We re-render every restart so the
    # PNG always reflects the current BoardConfig + render settings without
    # needing manual cache invalidation. Render takes ~50 ms.
    cache_dir = Path(__file__).resolve().parent.parent.parent / "_calib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    board_png = cache_dir / "board.png"
    canonical = render_board_png(current_board_config(), pixels_per_metre=4000.0, margin_squares=0.0)
    # No vertical flip: NiceGUI's Three.js Texture material uses flipY=false
    # AND our texture-coord array maps world (0,0,0) -> UV (0,0) -> the PNG's
    # top-left pixel. cv2.flip(...,0) was double-flipping that and producing
    # a mirrored / scrambled-looking board where the markers were unreadable.
    canonical_rgb = cv2.cvtColor(canonical, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(board_png), canonical_rgb)
    logger.info("rendered ChArUco board PNG: %s (%d×%d)", board_png, canonical.shape[1], canonical.shape[0])

    stl_url, png_url = _ensure_static_mounts(merged_stl, board_png)

    # Capture the asyncio loop so the worker thread can post UI updates.
    # Only clobber the existing slot when we have a real loop. A re-add
    # path that runs without a running loop should NOT zero out a
    # previously-captured valid loop ref — worker threads spawned
    # earlier would lose their ability to schedule UI updates. (The
    # state.py module-level seed for this key is None, so on a true
    # first-call without a running loop the slot stays None either way.
    # The protection here is for the multi-add re-entry path.)
    try:
        _state["main_loop"] = asyncio.get_running_loop()
    except RuntimeError:
        # Preserve whatever was there. No-op when slot is None
        # (first call); preserves a real loop on re-add.
        pass

    # Capture the NiceGUI client. Worker-thread callbacks scheduled via
    # ``loop.call_soon_threadsafe`` run OUTSIDE any request slot, so
    # ``ui.dialog()`` / ``ui.notify()`` etc. fail with
    # ``"The current slot cannot be determined because the slot stack
    # for this task is empty."``. The fix: enter the client's content
    # slot via ``with client:`` before creating UI from a worker
    # callback. ``preview_dialog.show_collision_dialog_threadsafe``
    # uses this slot.
    try:
        from nicegui import context  # noqa: PLC0415

        _state["nicegui_client"] = context.client
    except (ImportError, RuntimeError) as e:
        logger.debug("nicegui client capture failed: %s", e)
        _state["nicegui_client"] = None

    # Register a disconnect handler so our timers (footprint tick,
    # post-cal tick, live-pose chip, detection poll, reachability
    # pending defer, scene-init delay) are cancelled when the
    # browser closes its tab. Without this, NiceGUI's ``ui.timer``
    # keeps firing on the asyncio loop after page teardown; its
    # ``_run_once`` enters ``self.parent_slot`` which raises
    # ``RuntimeError: The parent slot of the element has been
    # deleted.`` once for every tick. The error spams the log
    # (~5 / sec for the 5 Hz footprint tick) without affecting
    # functionality, but it's noisy and indicates orphaned tasks
    # consuming CPU.
    try:
        client = _state.get("nicegui_client")
        if client is not None:
            from .panel import _teardown_overlays  # noqa: PLC0415

            def _on_disconnect_cleanup() -> None:
                # Best-effort cancel of every tracked timer + group
                # so NiceGUI doesn't try to render into a dead slot.
                # Reuses the panel's teardown logic (which already
                # cancels calib_timers + detection_overlay_timer +
                # reachability_pending_timer, and clears scene
                # group handles + per-tool runtime state).
                try:
                    _teardown_overlays()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "calibration teardown on disconnect failed: %s", exc,
                    )

            client.on_disconnect(_on_disconnect_cleanup)
    except Exception as e:  # noqa: BLE001
        logger.debug("disconnect-handler registration failed: %s", e)

    # Defer "scene is initialized" gating via a short delayed timer
    # rather than ``urdf_scene.scene.on("init", ...)``. The listener
    # path looks tempting, but ``Element.on()`` calls
    # ``Element.update()`` (element.py:388), which enqueues an update
    # message. ``add_overlays`` runs AFTER ``await client.tools()`` in
    # ``initialize_urdf_scene``, so the outbox already flushed the
    # scene construction message; the new listener arrives in a SECOND
    # update. NiceGUI's frontend (nicegui.js:482-496) compares
    # ``listener_id``s, sees a new one, fires the
    # ``"Event listeners changed after initial definition.
    # Re-rendering affected elements."`` warning, then DELETES the
    # scene element and re-mounts it. Re-mount calls
    # ``init_objects`` AGAIN on the same client-side THREE.scene
    # (scene.js:1138-1172). Each ``create()`` call inside that loop
    # (scene.js:414-608) does
    # ``this.objects.set(id, mesh); parent.add(mesh)`` — ``Map.set``
    # overwrites the dict entry, but THREE.js's ``parent.add(child)``
    # does NOT dedupe by ``object_id``. Result: the OLD mesh stays
    # parented to the old THREE.js group as a phantom while the NEW
    # mesh receives subsequent broadcasts. The user sees TWO arms,
    # one frozen at the page-open pose, one live. ``hard refresh``
    # wipes the THREE.scene and starts clean — confirming the
    # client-side accumulation theory.
    #
    # The delayed-timer alternative achieves the same gate (footprint
    # tick + reachability defer waiting for the browser to be ready)
    # without going through ``Element.on()``. 600ms is comfortably
    # past first-mount + ``init_objects`` for any reasonable machine;
    # even if it lands fractionally early, the worst case is one
    # extra defer-cycle on the reachability render, not a duplicate
    # arm. The timer is fire-once and cheap (no re-render ripple).
    _state["scene_initialized"] = False
    # Cancel a stale prior init-defer timer if a previous add_overlays
    # left one queued; otherwise it'd fire later against this rebuild's
    # state and could clobber a True flag with True (cosmetic, but the
    # parent slot deletion on page-tear-down also surfaces as noisy
    # ``parent_slot deleted`` log lines).
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
        # Fall back to optimistic-true so timer-driven creates
        # aren't stuck behind a permanently-False gate.
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

    # ------------------------------------------------------------------
    # Camera frustum group, parented to tcp_anchor.
    #
    # Idempotent: a leftover frustum_group from a previous add_overlays
    # call (live re-add path after teardown) is deleted before creating
    # a fresh one. Otherwise the old group accumulates as the user
    # toggles features off/on or switches between camera-bearing tools.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ChArUco board → world frame (parented to scene root via instance method).
    # Raised 1mm above the floor so the polar-grid ground plane doesn't
    # z-fight with it, and bordered with a coloured outline so it's easy
    # to spot even if the texture fails to load.
    # ------------------------------------------------------------------
    scene_root = urdf_scene.scene
    _state["scene_root"] = scene_root

    # ``_state`` is a module-level dict that survives a browser refresh
    # (the Python process keeps running). After a refresh, the per-tick
    # raycast cache (``footprint_objects`` / ``footprint_last_q`` /
    # ``footprint_last_mount``) still holds handles parented to the
    # destroyed prior scene_root and joint values matching the current
    # robot pose — so ``_footprint_inputs_changed`` returns False, the
    # 5 Hz tick early-exits, and the dynamic centerline + footprint
    # never render into the new scene. Clearing the cache here is the
    # same pattern ``live_apply._redraw_frustum`` uses after intrinsic
    # changes.
    _state["footprint_objects"] = []
    _state["footprint_last_q"] = None
    _state["footprint_last_mount"] = None

    # All board-dependent overlays (board, tablet, hemisphere wireframe,
    # reachability dots) are built through a single helper so initial build
    # and post-localise refresh share one code path. Each child group's
    # visibility honours the current ``_state['show_*']`` flag, which is
    # initialised from persisted user preferences in ``add_control_panel``
    # before this function is called (see ``add_control_panel``).
    _build_board_dependent_overlays(scene_root, png_url)

    # Cancel any timers from a prior add_overlays call (live re-add path
    # after teardown) so they don't accumulate. Each fresh call installs
    # its own set tracked in ``_state["calib_timers"]``.
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

    # Detection-overlay polling: repaints AABB boxes from the perception
    # pipeline's last_detection.json snapshot when its mtime changes.
    _state["detection_overlay_timer"] = ui.timer(
        _DETECTION_POLL_INTERVAL_S, _poll_detection_json,
    )

    # Calibration-driven scene timers. Installed here (during overlay setup,
    # which happens once per page) instead of in the calibration panel
    # builder so they keep running regardless of which side-tab is active.
    # Both are no-ops while their preconditions don't hold:
    #   - _post_calibration_tick (4 Hz): waits until calibration finishes
    #     and ``_state['calibrated_mount']`` is set.
    #   - _raycast_footprint_tick (5 Hz): waits until a mount is available
    #     and the scene root + collision manager pair are populated; also
    #     short-circuits when both ``show_footprint`` and ``show_centerline``
    #     are off, or when the joint/mount inputs haven't changed since the
    #     last frame (keeps the asyncio loop responsive during heavy CPU
    #     spikes from the calibration thread + 50 Hz status broadcasts).
    _state["calib_timers"] = [
        ui.timer(0.25, _post_calibration_tick, active=True),
        ui.timer(1.0 / _FOOTPRINT_TICK_HZ, _raycast_footprint_tick, active=True),
        ui.timer(0.5, _live_pose_indicator_tick, active=True),
    ]

    # Click-on-dot popup: page-level fixed-position container + a
    # scene click handler that opens the "Go to pose" tooltip when
    # the user clicks one of the green reachability spheres. Both
    # are idempotent: a re-run replaces any prior container +
    # handler so we don't stack them across feature on/off cycles.
    try:
        from . import pose_popup  # noqa: PLC0415

        pose_popup.init_popup_container()
        pose_popup.register_click_handler(urdf_scene.scene)
    except Exception as e:  # noqa: BLE001
        logger.debug("pose popup setup failed: %s", e)

    # Marker for ``apply_calibration_state`` so it can tell whether the
    # overlays are currently built (don't re-add) vs. previously torn
    # down (need to call this function). Initial page-load gets the
    # flag set here; live tear-down clears it.
    _state["overlays_built"] = True

    # Pre-build the FCL collision managers (full + gripper-only) on a
    # worker thread so the first ``_raycast_footprint_tick`` and
    # ``_live_pose_indicator_tick`` don't block the asyncio event loop
    # for 2-5 s doing trimesh.load + FCL BVH builds (~10 link meshes +
    # custom-tool body + 2 jaws + floor + tablet, EACH manager). Both
    # ticks check ``collision_mgr_warming`` and skip until the warmup
    # finishes; the next tick after that draws the footprint and
    # updates the live-pose indicator normally.
    #
    # Without this, the synchronous build was hogging the loop long
    # enough to trip Socket.IO's ping_timeout (~2 s) and NiceGUI's
    # response_timeout (~3 s), surfacing in logs as
    # "binding propagation for N active links took 3.886 s" and
    # causing the browser to drop + auto-reload the page on cold
    # start. Same mitigation precedent as
    # ``_start_reachability_compute_async`` (reachability.py:488) —
    # heavy first-pass work runs off the loop.
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
    """Build the ChArUco board scene group at the current ``_T_BOARD2BASE``.

    Extracted from ``add_overlays`` so ``refresh_board_dependent_overlays``
    can rebuild the group after auto-localise mutates ``_T_BOARD2BASE``.
    Returns the group handle (call ``.delete()`` to remove).
    """
    cfg = current_board_config()
    w_m = cfg.squares_x * cfg.square_length
    h_m = cfg.squares_y * cfg.square_length

    board_pos = _T_BOARD2BASE[:3, 3].copy()
    board_pos[2] += 0.001  # nudge above the floor grid to avoid z-fight
    board_group = scene_root.group().move(*board_pos.tolist())
    # NiceGUI's group.rotate(rx, ry, rz) wraps three.js Object3D.rotation,
    # which uses INTRINSIC XYZ (i.e. "xyz" in scipy convention — lowercase).
    # _BOARD_RPY_RAD is documented as scipy XYZ-extrinsic (uppercase) so
    # we round-trip through the rotation matrix and decompose with the
    # intrinsic convention here. For single-axis rotations (e.g. yaw-only)
    # both conventions give identical Euler angles; the difference only
    # shows up for tilted boards (multiple non-zero axes).
    rpy = SciRotation.from_matrix(_T_BOARD2BASE[:3, :3]).as_euler("xyz").tolist()
    if any(abs(a) > 1e-6 for a in rpy):
        board_group = board_group.rotate(*rpy)
    with board_group:
        # Backing plane — gray (not white) so it's visually distinct from the
        # texture's white squares. If the texture renders, you see distinct
        # black-on-white markers ON TOP of a gray border / fallback. If the
        # texture fails to load, the area shows as solid gray instead of
        # white, which is an obvious diagnostic signal.
        # Sits at local z ∈ [-0.0035, -0.0025], well below the texture, so
        # there's no chance of z-fighting against the texture or the floor.
        ui.scene.box(w_m, h_m, 0.001).move(w_m / 2, h_m / 2, -0.003).material(
            "#888888", opacity=1.0
        )
        # ChArUco texture. Vertex rows are REVERSED (first row Y=h_m, second
        # row Y=0) so the resulting triangle winding produces a +Z-facing
        # surface normal. NiceGUI's Three.js material is MeshLambertMaterial
        # with side=DoubleSide and transparent=true; in this combo the back
        # face can render dim or blank, so we want the FRONT face to be the
        # one users see when looking down at the board from above.
        # UV mapping with this ordering: PNG top-left → world (0, h_m), so
        # the printed PNG's "top" appears at the FAR edge of the board (Y=h_m)
        # and its "bottom" appears at the NEAR edge (Y=0) — i.e. the board
        # reads correctly when viewed from the robot side looking in +Y.
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
    """Build a translucent box overlay matching the TABLET collision primitive.

    Position, orientation, and dimensions match _build_collision_manager's
    tablet exactly, so what you see in the GUI is what's being collision-
    checked. Returns the group handle (``.delete()`` to remove), or None if
    rendering is disabled. Driven by ``settings.surface_enabled`` +
    ``settings.surface_show_overlay``.
    """
    if not (bool(settings.surface_enabled) and bool(settings.surface_show_overlay)):
        return None

    _cfg = current_board_config()

    t_w, t_l, t_h = settings.surface_dimensions_m
    t_off_x, t_off_y = settings.surface_offset_local_m

    # Tablet centre in board-local frame, then push to world via _T_BOARD2BASE.
    # Board pose is auto-lifted by t_h in _build_T_board2base, so placing the
    # box centre at board-local z=-t_h/2 puts it between world z=0 (bench)
    # and world z=+t_h (screen). Matches the collision primitive exactly.
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
    # Lowercase "xyz" = scipy intrinsic, matches three.js Object3D.rotation
    # default. See note in _build_board_overlay_group for context.
    rpy = SciRotation.from_matrix(_T_BOARD2BASE[:3, :3]).as_euler("xyz").tolist()

    grp = scene_root.group().move(*centre_world.tolist()).with_name("calib:tablet")
    if any(abs(a) > 1e-6 for a in rpy):
        grp = grp.rotate(*rpy)
    with grp:
        # Translucent rusty-orange box — distinct from the gray board backing
        # so it's easy to tell where the modelled tablet body extends past
        # the printed ChArUco area.
        ui.scene.box(t_w, t_l, t_h).material("#cc7733", opacity=0.30)

    logger.info(
        "tablet overlay: centre=%s, dims=%s, offset=%s",
        np.round(centre_world, 3).tolist(),
        tuple(settings.surface_dimensions_m),
        tuple(settings.surface_offset_local_m),
    )
    return grp


def regenerate_board_png() -> None:
    """Re-render the cached ChArUco PNG from the current board settings,
    then refresh the URL with a cache-busting suffix so the browser loads
    the new image. Called by ``live_apply`` whenever a board-geometry
    setting changes (squares_x/y, square_length, marker_length, dictionary,
    legacy_pattern). Caller is responsible for ``refresh_board_dependent_overlays``
    afterwards.
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
        # Bust the browser cache by appending the file mtime — the static
        # mount path doesn't change, so the same file on disk now served
        # at a new URL forces a re-fetch.
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
    """Rebuild the board, hemisphere wireframe, reachability dots, and
    tablet visual.

    Call this after ``_T_BOARD2BASE`` is mutated (e.g. by auto-localise) so
    the visualisation reflects the new board pose. Frustum stays correct
    automatically (it's parented to ``tcp_anchor``).

    Safe to call from a background thread: it schedules the actual scene
    surgery on the asyncio loop captured during ``add_overlays``. Returns
    immediately; the redraw happens at the next event-loop tick.
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
        # No event loop captured — caller is on the main thread.
        _do_refresh()
    else:
        try:
            loop.call_soon_threadsafe(_do_refresh)
        except RuntimeError as e:
            # Event loop is closed (page torn down mid-localise). Nothing
            # to refresh.
            logger.info(
                "refresh_board_dependent_overlays: loop unavailable (%s); "
                "skipping scene refresh",
                e,
            )


def _add_hemisphere_wireframe_to_group(scene_group: Any, target_world: NDArray[np.float64]) -> None:
    """Draw a 3D wireframe volume for the hemisphere search region inside ``scene_group``.

    Volume bounds are an annular spherical sector defined by:
        d ∈ [d_min, d_max]                      (radial)
        elev ∈ [elev_min, elev_max]             (latitude)
        az ∈ [az_center ± _HEMI_AZIMUTH_SPREAD]  (longitude)

    We render it as a 6-surface wireframe wedge so the user perceives a
    solid region (not just discrete sampling paths):
        - inner spherical patch at d_min  (latitude × longitude grid)
        - outer spherical patch at d_max  (same grid)
        - 4 corner edges connecting the patches at the bounding corners

    This matches what the pose generator actually does: it samples ANY pose
    within this volume that's also reachable + IK-valid, not just along a
    few discrete arcs.
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
        # Wireframe lines live in their own sub-group so the panel's
        # "Hemisphere wireframe" toggle hides them independently of the
        # reachability dots (which live in a separate sub-group added by
        # ``_add_reachability_points``).
        wf_group = (
            ui.scene.group()
            .with_name("calib:hemisphere_wireframe")
            .visible(_state.get("show_hemisphere", True))
        )
        _state["hemisphere_wireframe_group"] = wf_group
        with wf_group:
            # Two spherical shells (inner d_min, outer d_max). For each shell:
            #   - latitude arcs: constant elev, varying az
            #   - longitude arcs: constant az, varying elev
            for d, opacity in [(d_min, 0.55), (d_max, 0.30)]:
                color = "#5599ff" if d == d_min else "#3366cc"
                # Latitude arcs (one per elevation step).
                for ev in elevations:
                    pts = [offset(d, ev, az) for az in azimuths]
                    for i in range(len(pts) - 1):
                        ui.scene.line(list(pts[i]), list(pts[i + 1])).material(
                            color, opacity=opacity
                        )
                # Longitude arcs (one per azimuth step).
                for az in azimuths:
                    pts = [offset(d, ev, az) for ev in elevations]
                    for i in range(len(pts) - 1):
                        ui.scene.line(list(pts[i]), list(pts[i + 1])).material(
                            color, opacity=opacity
                        )

            # Four corner edges connecting inner shell to outer shell.
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
    """Build (or rebuild) every overlay whose geometry depends on the current
    ``_T_BOARD2BASE`` / hemisphere centre, and cache each group handle in
    ``_state`` so the panel toggles can flip visibility at runtime.

    Always creates every group — visibility is governed by ``_state['show_*']``
    flags applied at creation time and updated thereafter via
    :func:`_set_overlay_visible`. Always-creating keeps initial build and
    post-localise refresh on a single code path; the runtime cost of hidden
    Three.js objects is negligible compared to the multi-second IK sweep
    that populates the reachability dots.

    Call from the asyncio loop thread (NiceGUI scene API is not thread-safe).
    """
    # Tear down anything that already exists (post-localise rebuild path).
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

    # Tablet (mounting-surface) overlay is OPTIONAL — disabled when the
    # surface is turned off OR when the user has hidden the visual. Guard
    # the .visible() call so a None return value (legitimate) doesn't
    # raise and abort the rest of the board-dependent rebuild
    # (hemisphere, reachability dots).
    tablet_group = _build_tablet_overlay_group(scene_root)
    if tablet_group is not None:
        tablet_group.visible(show_board)
    _state["tablet_group"] = tablet_group

    # Hemisphere wireframe + reachability dots, parented to a common group
    # whose origin sits at the hemisphere centre. The two children
    # (wireframe / dots) get their own sub-groups inside their builders so
    # they can be toggled independently. The OUTER ``hemisphere_group``
    # stays visible whenever EITHER child should show — the per-child
    # sub-groups carry the actual show/hide state.
    #
    # The reachability IK sweep is dispatched OFF the asyncio loop — see
    # ``_start_reachability_compute_async``. The wireframe appears
    # immediately; the green dots populate a moment later when the
    # sweep finishes (typically <2 s). This keeps the loop responsive
    # for websocket traffic during the sweep.
    target_world = _hemi_centre_world()
    grp = scene_root.group().move(*target_world.tolist()).with_name("calib:hemisphere")
    _state["hemisphere_group"] = grp
    _add_hemisphere_wireframe_to_group(grp, target_world)
    _start_reachability_compute_async(grp, target_world)


def _set_overlay_visible(name: str, visible: bool) -> None:
    """Toggle one of the 3D overlay groups at runtime.

    ``name`` is one of: ``board`` (board + tablet), ``hemisphere``
    (wireframe), ``reachability`` (green dots), ``near_cone`` (fixed
    frustum lines), ``centerline`` (camera→hit yellow line),
    ``footprint`` (projected magenta polygon).

    Persists the new state in ``_state['show_<name>']`` so the per-tick
    raycast loop and any future rebuild use the right flag. Best-effort
    ``.visible()`` on the cached group handle for the static overlays
    that have one — the dynamic centerline and footprint are recreated
    every 100 ms in ``_raycast_footprint_tick`` and read the flag there.
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
            # Page-teardown / parent-slot races. Not fatal — the next
            # rebuild will pick up the new state from ``_state``.
            logger.debug(
                "_set_overlay_visible(%s): %s: %s",
                name, type(e).__name__, e,
            )


