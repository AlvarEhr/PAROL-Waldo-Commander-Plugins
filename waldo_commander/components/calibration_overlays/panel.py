"""Calibration tab content (UI + thread spawning)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from nicegui import ui

from . import custom_tools, custom_tools_ui, settings, settings_ui
from .calibration_thread import _calibration_thread
from .hover import _drive_hover_pose_thread
from .localise import _localise_board_thread
from .overlays import _set_overlay_visible
from .state import _state, _state_lock, current_board_config

logger = logging.getLogger(__name__)


def _post_status(text: str) -> None:
    """Post a status update to the GUI from a worker thread."""
    label = _state.get("status_label")
    loop = _state.get("main_loop")
    if label is None or loop is None:
        logger.info("Calibration status: %s", text)
        return

    def _update():
        # Re-fetch in case the page tore down between scheduling and
        # dispatch (status_label was deleted, set to None on cleanup).
        live_label = _state.get("status_label")
        if live_label is None:
            return
        try:
            live_label.text = text
        except Exception:  # noqa: BLE001
            # Label exists but is detached / disposed.
            pass

    try:
        loop.call_soon_threadsafe(_update)
    except RuntimeError as e:
        # Loop closed (page tear-down). Log at info — not a real failure.
        logger.info("Status post skipped (loop unavailable): %s", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not post GUI status: %s", e)


# ---------------------------------------------------------------------------
# Calibration tab content
# ---------------------------------------------------------------------------


# Keys are the suffixes used by ``_set_overlay_visible``; values are the
# (label, default_visible) shown in the panel. Order matches the panel layout.
_OVERLAY_TOGGLES: tuple[tuple[str, str, bool], ...] = (
    ("board",        "Board + tablet",          True),
    ("hemisphere",   "Hemisphere wireframe",    True),
    ("reachability", "Reachability dots",       True),
    ("near_cone",    "Fixed frustum (near cone)", True),
    ("centerline",   "Centerline (camera→hit)", True),
    ("footprint",    "Projected footprint",     True),
    # Off by default — perception JSON often outlives its run.
    ("detections",   "Perception detections",   False),
)


def _load_persisted_overlay_prefs() -> None:
    """Read each ``show_*`` flag from ``app.storage.user`` (NiceGUI's
    cookie-backed per-user storage) and seed ``_state`` with them.

    Defaults (from ``_OVERLAY_TOGGLES``) win when the storage key is missing,
    so a brand-new user sees everything on. Failures fall back to the
    defaults — storage isn't available outside a request context, and we
    don't want a one-time read error to lose the user's preference forever.
    """
    try:
        from nicegui import app  # noqa: PLC0415
        store = app.storage.user
    except Exception as e:  # noqa: BLE001
        logger.debug("calibration prefs: app.storage.user unavailable (%s)", e)
        store = None
    for name, _label, default in _OVERLAY_TOGGLES:
        key = f"show_{name}"
        if store is not None:
            value = bool(store.get(key, default))
        else:
            value = default
        _state[key] = value


def _persist_overlay_pref(name: str, visible: bool) -> None:
    """Write one ``show_*`` flag to ``app.storage.user`` so it survives a reload."""
    try:
        from nicegui import app  # noqa: PLC0415
        app.storage.user[f"show_{name}"] = bool(visible)
    except Exception as e:  # noqa: BLE001
        logger.debug("calibration prefs: persist failed for %s (%s)", name, e)


def _features_active() -> bool:
    """Read the soft-toggle state from ``app.storage.general``. Default OFF.

    Mirrors ``main._calibration_features_active`` but lives in this
    module so the panel can flip the storage value + render
    differently without crossing module boundaries.
    """
    try:
        from nicegui import app  # noqa: PLC0415

        return bool(app.storage.general.get("calibration_features_active", False))
    except Exception:  # noqa: BLE001
        return False


def _set_features_active(active: bool) -> None:
    """Persist the soft toggle to ``app.storage.general``."""
    try:
        from nicegui import app  # noqa: PLC0415

        app.storage.general["calibration_features_active"] = bool(active)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not persist calibration_features_active: %s", e)


def _no_camera_override() -> bool:
    """Per-user override that lets the calibration panel + overlays render
    even when the active tool isn't camera-bearing. Persisted via
    ``app.storage.user`` so it survives reloads. Default OFF — without
    this, the panel hides the calibration scaffolding when no camera
    tool is selected.

    When ON, a persistent warning banner reminds the user the values
    they're seeing aren't tied to a real camera.
    """
    try:
        from nicegui import app  # noqa: PLC0415

        return bool(app.storage.user.get("calib_no_camera_override", False))
    except Exception:  # noqa: BLE001
        return False


def _set_no_camera_override(value: bool) -> None:
    try:
        from nicegui import app  # noqa: PLC0415

        app.storage.user["calib_no_camera_override"] = bool(value)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not persist calib_no_camera_override: %s", e)


def _camera_gate_open() -> bool:
    """The camera-related calibration UI renders iff the active tool has
    a camera (built-in MSG, or any custom tool flagged ``has_camera``)
    OR the per-user override is on.
    """
    return custom_tools.active_tool_is_camera_bearing() or _no_camera_override()


# ---------------------------------------------------------------------------
# Live-apply machinery — replaces the previous "reload page to apply" pattern
# ---------------------------------------------------------------------------


def _teardown_overlays() -> None:
    """Delete every scene group + dynamic-overlay handle. Called on
    transitions out of the camera-bearing-render state (master toggle
    flipped off, or active tool became no-camera with no override on),
    AND from the on-disconnect handler registered in ``add_overlays``
    so a browser tab-close also tears the scene down.

    Also signals worker threads (calibration / localise / hover /
    pose-popup go-to-pose) to STOP via ``_state['stop_requested']``
    so they release the controller socket + RealSense camera
    promptly instead of running to natural completion. Without
    this, a tab-close mid-calibration would leave the calibration
    thread holding RealSense (which is exclusively claimed) for
    the remaining sweep duration — re-opened tabs would fail to
    initialise it.
    """
    # Signal worker threads to STOP at their next poll point. Each
    # worker checks ``_state.get("stop_requested")`` between move
    # commands and at the top of its main loop, so an in-flight
    # ``move_j(wait=True)`` blocks until the controller acks the
    # halt — that's followed up below.
    _state["stop_requested"] = True
    # Dispatch a halt() to the controller so an in-flight motion
    # aborts promptly instead of running to its planned end. The
    # raw client is stashed in ``_state["client"]`` by the worker
    # threads; if no thread is active, this is a no-op.
    _client = _state.get("client")
    if _client is not None:
        try:
            _client.halt()
        except Exception as e:  # noqa: BLE001
            logger.debug("teardown halt() raised: %s", e)
    for key in (
        "frustum_group", "board_group", "tablet_group",
        "hemisphere_group", "near_cone_group",
        "hemisphere_wireframe_group", "reachability_group",
        "detection_overlay_group",
        "footprint_group",
    ):
        grp = _state.get(key)
        if grp is not None:
            try:
                grp.delete()
            except Exception:  # noqa: BLE001
                pass
        _state[key] = None
    # Dynamic objects (frustum lines, footprint polyline, centerline).
    for objs_key in ("frustum_objects", "footprint_objects"):
        for obj in _state.get(objs_key, []) or []:
            try:
                obj.delete()
            except Exception:  # noqa: BLE001
                pass
        _state[objs_key] = []
    # Cancel scene timers tracked by add_overlays so they don't keep
    # firing against a torn-down scene + don't accumulate when the
    # user toggles features off/on or switches camera-bearing tools.
    for timer in _state.get("calib_timers", []) or []:
        try:
            timer.cancel()
        except Exception:  # noqa: BLE001
            pass
    _state["calib_timers"] = []
    det_timer = _state.get("detection_overlay_timer")
    if det_timer is not None:
        try:
            det_timer.cancel()
        except Exception:  # noqa: BLE001
            pass
        _state["detection_overlay_timer"] = None
    # Cached state for the per-tick rebuild detection: clear so a
    # future rebuild doesn't get short-circuited.
    _state["footprint_last_q"] = None
    _state["footprint_last_mount"] = None
    # Reachability cache: drop the candidate list + cached points so a
    # stale entry can't be served to the click handler after teardown.
    # Bump ``reach_generation`` alongside the candidates clear under
    # the lock so a click handler racing the teardown sees a clean
    # "stale-gen mismatch" categorisation rather than indexing past
    # an empty list (which bottoms out gracefully but produces
    # confusing log noise).
    with _state_lock:
        _state["reach_generation"] = (
            int(_state.get("reach_generation", 0)) + 1
        )
        _state["reachable_candidates"] = []
    _state["reachable_points_world"] = None
    _state["reachable_target_world"] = None
    # Cancel any pending deferred reachability render scheduled while
    # waiting for the scene 'init' event so it doesn't fire against
    # a torn-down scene.
    pending_timer = _state.get("reachability_pending_timer")
    if pending_timer is not None:
        try:
            pending_timer.cancel()
        except Exception as e:  # noqa: BLE001
            logger.debug("reachability pending-timer cancel failed: %s", e)
        _state["reachability_pending_timer"] = None
    _state["reachability_pending_payload"] = None
    # Cancel the scene-init defer timer too. ``add_overlays`` may
    # leave one pending if teardown happens within 600ms of the
    # rebuild (rare, but matches the rest of the timer cleanup).
    init_timer = _state.get("scene_init_defer_timer")
    if init_timer is not None:
        try:
            init_timer.cancel()
        except Exception as e:  # noqa: BLE001
            logger.debug("scene init defer-timer cancel failed: %s", e)
        _state["scene_init_defer_timer"] = None
    # Tear down the click-on-dot popup container + remove the scene
    # click handler so they don't survive into the next add_overlays
    # call (which re-creates both fresh).
    popup_container = _state.get("pose_popup_container")
    if popup_container is not None:
        try:
            popup_container.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["pose_popup_container"] = None
    old_click_handler = _state.get("dot_click_handler")
    if old_click_handler is not None:
        try:
            from waldo_commander.state import ui_state  # noqa: PLC0415

            scene = getattr(getattr(ui_state, "urdf_scene", None), "scene", None)
            handlers = getattr(scene, "_click_handlers", None)
            if isinstance(handlers, list):
                try:
                    handlers.remove(old_click_handler)
                except ValueError:
                    pass
        except Exception:  # noqa: BLE001
            pass
        _state["dot_click_handler"] = None
    # ``current_mount`` gates the footprint tick; clearing it makes
    # the tick no-op until rebuilt. ``scene_root`` is left intact:
    # it's the URDF scene root, used elsewhere.
    _state["current_mount"] = None
    _state["overlays_built"] = False
    # Live-pose chip handle + tooltip element: clear so the 0.5s
    # indicator tick (if it somehow fires before its ``ui.timer``
    # cancellation completes) is a clean no-op rather than poking a
    # deleted Quasar element.
    _state["live_pose_label"] = None
    _state["live_pose_tooltip"] = None
    # Force-exit any active preview + close any open dialog so a
    # feature-off cycle never strands the URDF in PREVIEW.
    try:
        from .preview_dialog import reset_preview_state  # noqa: PLC0415

        reset_preview_state()
    except (ImportError, AttributeError):
        pass


def _ensure_features_loaded() -> None:
    """Idempotent one-shot init: SSG-48 auto-migrate + custom-tool
    register_all + active_robot tools rebuild. Mirrors what
    ``main.initialize_urdf_scene`` does when
    ``calibration_features_active`` is on at page-build time, so a
    live off→on flip gets the same result as a reload would.

    First flip: ~1–2 s on disk (STL bake for ssg48_realsense + any
    other custom tools). Subsequent flips: fast (sentinel + cached
    bakes).
    """
    try:
        custom_tools.auto_migrate_ssg48_with_bracket()
    except Exception as e:  # noqa: BLE001
        logger.warning("ssg48 auto-migration failed: %s", e)
    try:
        registered = custom_tools.register_all()
        if registered:
            try:
                from parol6.robot import _build_tools as _parol6_build_tools  # noqa: PLC0415
                from waldo_commander.state import ui_state  # noqa: PLC0415

                ui_state.active_robot._tools = _parol6_build_tools()  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001
                logger.debug("active_robot._tools rebuild failed: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("register_all failed: %s", e)


def apply_calibration_state() -> None:
    """Reconcile the rendered scene + panel UI with the current toggles
    and active tool. Idempotent.

    Call after any of these changes:

    * ``app.storage.general["calibration_features_active"]`` flipped
      (the master toggle in the panel header OR the bottom-right
      Settings tab's Calibration features switch).
    * Active tool changed (the gripper-panel dropdown OR the
      "Use this tool" button on a custom-tool card).
    * ``app.storage.user["calib_no_camera_override"]`` flipped (the
      Force-show toggle in the no-camera placeholder OR the
      View overlays expansion).

    Performs the minimum scene mutation needed to bring the rendered
    overlays + panel state into agreement. Replaces the previous
    "reload the page to apply" notify pattern.

    Three coarse transitions:

    * ``should_render`` AND not built — heavy: runs auto-migrate +
      register_all (idempotent) + ``add_overlays``. First off→on
      may take ~1–2 s on disk for the SSG-48 STL bake; subsequent
      flips are fast (sentinel + cached bakes).
    * ``should_render`` AND built — cheap: rebuild ``CameraMount``
      so per-tool intrinsic / mount overrides take effect, redraw
      frustum geometry.
    * Not ``should_render`` AND built — cheap: tear down scene
      groups, clear dynamic caches.

    The panel UI is always refreshed so its three-state branch picks
    up the new ``(features_on, gate_open)`` combination.
    """
    # Prime the thread-safe per-tool override cache from this request
    # context so any worker thread spawned downstream (calibration,
    # localise, hover, reachability) can read the active tool's
    # values via ``settings.get`` without touching app.storage.user.
    try:
        custom_tools.prime_per_tool_overrides_cache()
    except Exception as e:  # noqa: BLE001
        logger.debug("apply_calibration_state: cache prime skipped (%s)", e)

    features_on = _features_active()
    gate_open = _camera_gate_open()
    overlays_built = bool(_state.get("overlays_built", False))
    should_render = features_on and gate_open

    try:
        from waldo_commander.state import ui_state  # noqa: PLC0415

        scene = getattr(ui_state, "urdf_scene", None)
    except Exception:  # noqa: BLE001
        scene = None

    if should_render and not overlays_built:
        if features_on:
            _ensure_features_loaded()
        if scene is not None:
            try:
                from .overlays import add_overlays as _add_overlays  # noqa: PLC0415

                _add_overlays(scene)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "apply_calibration_state: add_overlays failed: %s", e,
                )
    elif should_render and overlays_built:
        # Tool change (or any other state change) where overlays
        # already exist: re-pick up per-tool intrinsics + cam_mount
        # via ``settings.get`` resolution and redraw frustum geometry.
        try:
            from .live_apply import _rebuild_camera_mount  # noqa: PLC0415

            _rebuild_camera_mount()
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "apply_calibration_state: _rebuild_camera_mount failed: %s", e,
            )
        # Force the dynamic projections (centerline + projected
        # footprint) to redraw on the next 5 Hz tick. Without this
        # the cache check in ``_footprint_inputs_changed`` may keep
        # the OLD tool's lines visible until the robot moves enough
        # for the joint-angle epsilon check to trip. Deleting the
        # whole sub-group (rather than each line individually) forces
        # the browser to drop every line at once — same pattern
        # ``update_frustum`` uses for the near-cone.
        old_footprint_group = _state.get("footprint_group")
        if old_footprint_group is not None:
            try:
                old_footprint_group.delete()
            except Exception:  # noqa: BLE001
                pass
            _state["footprint_group"] = None
        for obj in _state.get("footprint_objects", []) or []:
            try:
                obj.delete()
            except Exception:  # noqa: BLE001
                pass
        _state["footprint_objects"] = []
        _state["footprint_last_q"] = None
        _state["footprint_last_mount"] = None
        # Reachable-pose set depends on the active tool's mount + body
        # (different camera position changes which IK solutions are
        # valid; different gripper mesh changes self-collision). Drop
        # the cached candidates and re-run the IK sweep so the green
        # dots reflect the new tool.
        try:
            from .reachability import refresh_reachability_for_active_tool  # noqa: PLC0415

            refresh_reachability_for_active_tool()
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "apply_calibration_state: reachability refresh failed: %s", e,
            )
        # Defensive re-registration of the click handler. The handler
        # itself is a closure over module-level ``_state`` so it
        # survives in-place across teardown / refresh cycles, but
        # ``register_click_handler`` is idempotent (removes any prior
        # registration first) so re-attaching here is cheap and
        # ensures a tool-change path that somehow lost the handler
        # gets it back. Same pattern as the reachability refresh
        # above.
        try:
            from . import pose_popup  # noqa: PLC0415

            scene_ref = getattr(scene, "scene", None)
            if scene_ref is not None:
                pose_popup.register_click_handler(scene_ref)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "apply_calibration_state: click handler re-register failed: %s",
                e,
            )
    elif not should_render and overlays_built:
        _teardown_overlays()
    # else: not should_render and not overlays_built → no-op.

    # Always refresh the panel so its three-state branch picks up the
    # new state. ``panel_refresh`` is set by
    # ``build_calibration_panel_content`` when the panel last rendered.
    refresh = _state.get("panel_refresh")
    if refresh is not None:
        try:
            refresh()
        except Exception as e:  # noqa: BLE001
            logger.debug("panel refresh failed: %s", e)


def _build_inactive_placeholder(close_callback: Callable[[], None] | None) -> None:
    """Render the panel content when features are toggled OFF: just a
    header, an explanatory paragraph, and the master toggle. No
    scene mutations, no timers, no STL bakes — the soft gate stops
    everything heavy at the ``add_overlays`` / ``register_all`` calls
    in ``main.py``.
    """
    with ui.row().classes("w-full items-center"):
        ui.label("Calibration").classes("text-lg font-medium")
        ui.space()
        if close_callback is not None:
            ui.button(icon="close", on_click=close_callback).props(
                "flat round dense color=white"
            )
    ui.label(
        "Calibration features are disabled.",
    ).classes("text-xs opacity-80")
    ui.label(
        "Toggle on to enable.",
    ).classes("text-xs opacity-60 q-mt-xs")

    def _on_toggle(e) -> None:
        # NiceGUI's on_change kwarg passes the new value via e.value —
        # reading it that way avoids the timing race where ``sw.value``
        # might still hold the previous state when ``update:model-value``
        # fires on the underlying Quasar component.
        new_value = bool(getattr(e, "value", False))
        if not new_value:
            return  # only acting on flips to ON here; OFF is a no-op
        _set_features_active(True)
        # Live-apply: register custom tools + add overlays (if a
        # camera-bearing tool is active or override on) + refresh
        # this panel.
        ui.notify(
            "Loading calibration features...", color="info", position="top",
        )
        apply_calibration_state()
        ui.notify(
            "Calibration features active.",
            color="positive", position="top",
        )

    ui.switch(
        "Enable calibration features", value=False, on_change=_on_toggle,
    ).props("dense")


def _build_no_camera_placeholder(close_callback: Callable[[], None] | None) -> None:
    """Render the panel content when features are ON but the active tool
    isn't camera-bearing. The calibration scaffolding (Run / Localise /
    Hover + scene overlays) only makes sense alongside a calibratable
    camera, so we hide it and surface (a) the issue, (b) a force-show
    override, and (c) the custom-tools UI so the user can flag an
    existing custom tool as ``has_camera`` from here.
    """
    # Prefer the GUI's logical tool key (custom: prefix preserved) over
    # the controller's broadcast key. ``robot_state.tool_key`` reflects
    # what the controller broadcasts, which is only ever a BUILT-IN
    # name — for a custom tool with ``proxy_tool_key="SSG-48"`` it
    # would say "SSG-48" while the GUI is presenting
    # ``custom:my_gripper``. Same coverage gap that e6e6ad5 fixed
    # elsewhere; this surface was missed.
    try:
        from .custom_tools import _active_gui_tool_key  # noqa: PLC0415
        active_key = _active_gui_tool_key() or "NONE"
    except Exception:  # noqa: BLE001
        # _active_gui_tool_key reads app.storage.general which raises
        # outside a request context. Fall back to the controller's
        # broadcast key — the placeholder is informational, not
        # safety-critical.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
            active_key = getattr(robot_state, "tool_key", None) or "NONE"
        except Exception:  # noqa: BLE001
            active_key = "(unknown)"

    with ui.row().classes("w-full items-center"):
        ui.label("Calibration").classes("text-lg font-medium")
        ui.space()
        if close_callback is not None:
            ui.button(icon="close", on_click=close_callback).props(
                "flat round dense color=white"
            )

    with ui.card().classes("w-full bg-amber-9 text-white q-mt-xs"):
        ui.label("Active tool has no camera").classes(
            "text-sm font-semibold",
        )
        ui.label(
            f"Active tool: {active_key}. To use calibration:",
        ).classes("text-xs")
        ui.label(
            "1. Switch to a camera-bearing tool in the gripper panel.",
        ).classes("text-xs q-mt-xs")
        ui.label(
            "2. Or flag a custom tool as having a camera below.",
        ).classes("text-xs")
        ui.label(
            "3. Or enable Force-show overlays for UI inspection only.",
        ).classes("text-xs")

    def _on_override(e) -> None:
        new_value = bool(getattr(e, "value", False))
        _set_no_camera_override(new_value)
        # Live-apply: build / tear down overlays + refresh the panel
        # so the three-state branch picks up the new gate state.
        apply_calibration_state()
        if new_value:
            ui.notify(
                "Override enabled. Calibration data is for UI inspection only.",
                color="warning", position="top",
            )
        else:
            ui.notify(
                "Override disabled.", color="info", position="top",
            )

    ui.switch(
        "Force-show overlays (no camera tool active)",
        value=_no_camera_override(),
        on_change=_on_override,
    ).props("dense")

    # Custom tools UI lives down here too so the user can flag a tool as
    # camera-bearing without leaving the panel.
    ui.separator().classes("q-my-sm")
    custom_tools_ui.build_custom_tools_expansion()


def build_calibration_panel_content(close_callback: Callable[[], None] | None = None) -> None:
    """Build the calibration tab's contents.

    Three states:

    * **Features off** (``app.storage.general['calibration_features_active']``
      is False, default): renders only the master toggle + explanation.
      ``main.py`` skips ``add_overlays`` / ``register_all`` so no scene
      timers / STL bakes run.

    * **Features on, no camera-bearing active tool** (and no override):
      renders the no-camera placeholder + override toggle + custom-tools
      UI. ``main.py`` skips ``add_overlays`` so still no scene mutations.

    * **Features on + camera-bearing active tool (or override on)**:
      full panel — Run / Localise / STOP + view-overlay toggles +
      hover-above-board + calibration-settings expansion + custom-tools
      section. Banner at top when override is active.

    Pass ``close_callback`` to wire up the panel's close button.

    The three-state branch is wrapped in :func:`ui.refreshable` so a
    tool change, a master-toggle flip, or a force-show toggle can
    repaint the panel in place without a page reload —
    :func:`apply_calibration_state` calls ``_content.refresh`` after
    its scene mutations.
    """

    @ui.refreshable
    def _content() -> None:
        if not _features_active():
            _build_inactive_placeholder(close_callback)
            return
        if not _camera_gate_open():
            _build_no_camera_placeholder(close_callback)
            return
        _build_full_panel(close_callback)

    _state["panel_refresh"] = _content.refresh
    _content()


def _build_full_panel(close_callback: Callable[[], None] | None = None) -> None:
    """Render the camera-bearing full calibration panel — Run /
    Localise / STOP + view-overlay toggles + hover-above-board +
    calibration settings + custom-tools UI.

    Extracted from :func:`build_calibration_panel_content` so the
    ``@ui.refreshable`` wrapper there covers all three states
    (inactive placeholder / no-camera placeholder / this full panel).
    """

    def _busy_warn(msg: str) -> bool:
        """Reject button press if any motion-bearing thread is running,
        or if the localise-before-Run dialog is open. Returns True if
        a warning was issued."""
        if _state.get("is_running"):
            ui.notify(
                f"{msg}: calibration already running, wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("is_localising"):
            ui.notify(
                f"{msg}: board localise already running, wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("is_hovering"):
            ui.notify(
                f"{msg}: hover move in progress, wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("is_going_to_pose"):
            ui.notify(
                f"{msg}: go-to-pose move in progress, wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("dialog_open"):
            ui.notify(
                f"{msg}: confirmation dialog open, answer it first",
                color="warning",
                position="top",
            )
            return True
        return False

    def _start_calibration() -> None:
        """Spawn the calibration thread (post-confirmation)."""
        _state["is_running"] = True
        _state["stop_requested"] = False
        _state["calibrated_mount"] = None
        # Prime the per-tool override cache from the request context
        # before spawning the worker. The worker can't read
        # app.storage.user (request-context-bound, raises from threads),
        # so calibrating without this prime would silently drop the
        # active tool's calibrated mount + intrinsics and fall back to
        # globals.
        custom_tools.prime_per_tool_overrides_cache()
        # Same for the global runtime cache (settings._runtime).
        settings.load_from_storage()
        _post_status("Running calibration via parol6-server...")
        threading.Thread(target=_calibration_thread, daemon=True).start()

    def _on_run() -> None:
        if _busy_warn("Run"):
            return
        # Localise-before-Run guard. If the user hasn't successfully run
        # the Localise Board sweep this session, the calibration's bootstrap
        # will aim at the configured _BOARD_TRANSLATE_M — which on real
        # hardware is essentially never accurate. A confirmation dialog
        # offers to run anyway (sim mode, or already-trusted setup) or
        # cancel and run Localise first.
        if _state.get("last_localise_ok_at") is None:
            _state["dialog_open"] = True
            with ui.dialog() as dialog, ui.card():
                ui.label("Board hasn't been localised this session").classes(
                    "text-base font-semibold"
                )
                ui.label(
                    "Calibration will aim at the configured board position; "
                    "if the physical tablet isn't there, bootstrap will fail."
                ).classes("text-sm")
                ui.label(
                    "Recommended: Cancel, click Localise Board first, then Run."
                ).classes("text-xs opacity-80")
                with ui.row():
                    def _proceed():
                        _state["dialog_open"] = False
                        dialog.close()
                        _start_calibration()

                    def _cancel():
                        _state["dialog_open"] = False
                        dialog.close()
                    ui.button(
                        "Run anyway", on_click=_proceed, color="warning",
                    ).props("size=sm")
                    ui.button("Cancel", on_click=_cancel).props("size=sm")
            dialog.on("hide", lambda _e=None: _state.update(dialog_open=False))
            dialog.open()
            return
        _start_calibration()

    def _on_localise() -> None:
        """Drive a small lookout sweep + auto-locate the board centre."""
        if _busy_warn("Localise"):
            return
        _state["is_localising"] = True
        _state["stop_requested"] = False
        # Prime the per-tool override cache so the localise worker
        # can read the active tool's intrinsics + cam mount.
        custom_tools.prime_per_tool_overrides_cache()
        settings.load_from_storage()
        _post_status("Localising board, driving lookout sweep...")
        threading.Thread(target=_localise_board_thread, daemon=True).start()

    def _on_stop() -> None:
        """Abort the running calibration OR localise: halt + flag the thread.

        ``RobotClient.halt()`` is sync UDP and refuses to run inside an
        active asyncio event loop, so dispatch the halt to a daemon thread.
        The stop flag is set immediately so subsequent ``move_j`` calls
        short-circuit even before the thread-dispatched halt fires.
        """
        if not (
            _state.get("is_running")
            or _state.get("is_localising")
            or _state.get("is_hovering")
            or _state.get("is_going_to_pose")
        ):
            return
        _state["stop_requested"] = True
        client = _state.get("client")
        if client is not None:
            def _halt_in_thread():
                try:
                    client.halt()
                except Exception as e:  # noqa: BLE001
                    logger.warning("halt() in worker thread failed: %s", e)
            threading.Thread(target=_halt_in_thread, daemon=True).start()
            ui.notify("HALTED, robot motion stopped", color="warning")
        _post_status("Stop requested, wait for current move to finish")

    # Load persisted toggle prefs into _state BEFORE the scene builds (this
    # function runs during page render, after add_overlays). The first-page
    # ordering guarantee is that main.py builds the URDF scene + overlays
    # BEFORE building the side tabs, so initial overlay visibility may use
    # the defaults — but the moment the panel renders, _state is reseeded
    # from storage and the next per-tick read picks up the persisted value.
    # For the static groups (board/tablet/hemisphere/dots/near_cone), we
    # reapply visibility here so any divergence between defaults-at-build
    # and persisted-at-render is corrected.
    _load_persisted_overlay_prefs()
    for name, _label, _default in _OVERLAY_TOGGLES:
        _set_overlay_visible(name, _state.get(f"show_{name}", True))

    # Pinned header — Calibration title + close button stay visible while
    # the body below scrolls. Action row + status label live INSIDE the
    # scroll area so an excess of expansions doesn't push them off-screen,
    # but the user still has Run / Localise / STOP at the top of the
    # scroll area before any expansions.
    def _on_disable_features() -> None:
        with ui.dialog() as confirm, ui.card():
            ui.label("Disable calibration features?").classes(
                "text-base font-semibold",
            )
            ui.label(
                "Tears down overlays. Re-enable any time from the same toggle.",
            ).classes("text-xs opacity-70")
            with ui.row():
                def _confirm() -> None:
                    _set_features_active(False)
                    confirm.close()
                    # Live-apply: tear down scene overlays + refresh
                    # this panel to its inactive placeholder.
                    apply_calibration_state()
                    ui.notify(
                        "Calibration features disabled.",
                        color="info", position="top",
                    )
                ui.button(
                    "Disable", on_click=_confirm, color="warning",
                ).props("size=sm")
                ui.button("Cancel", on_click=confirm.close).props("size=sm")
        confirm.open()

    with ui.row().classes("w-full items-center"):
        ui.label("Calibration").classes("text-lg font-medium")
        # Live-pose collision indicator: 2 Hz background tick in
        # overlays.py updates the text + color + tooltip based on the
        # current robot configuration. Gripper-only check (~7x faster
        # than full); never blocks a move, just a visual signal.
        #
        # The tooltip is created ONCE here and re-used across ticks
        # via ``_state["live_pose_tooltip"]``. Without this, each tick
        # calling ``chip.tooltip(text)`` would APPEND a new
        # ``QTooltip`` child to the chip — ``Element.tooltip()``
        # constructs a fresh Tooltip element each call rather than
        # mutating the existing one. After ~1 minute the chip has 120+
        # stacked tooltips, every hover fires all of them
        # simultaneously, producing the user-reported "almost
        # infinite toasts" cascade.
        live_pose_chip = (
            ui.chip("?", color="grey")
            .props("dense outline size=xs")
        )
        with live_pose_chip:
            live_pose_tooltip = ui.tooltip(
                "Live pose collision check (initialising)",
            )
        _state["live_pose_label"] = live_pose_chip
        _state["live_pose_tooltip"] = live_pose_tooltip
        ui.space()
        ui.button(
            icon="power_settings_new", on_click=_on_disable_features,
        ).props(
            "flat round dense color=warning size=sm",
        ).tooltip("Disable calibration features")
        if close_callback is not None:
            ui.button(icon="close", on_click=close_callback).props(
                "flat round dense color=white"
            )

    # Persistent warning when the override is the only reason we're
    # here — i.e. there's no camera-bearing tool active but overlays
    # render anyway. Reminds the user the calibration data shown is
    # not meaningful.
    if _no_camera_override() and not custom_tools.active_tool_is_camera_bearing():
        with ui.card().classes("w-full bg-amber-9 text-white q-mt-xs"):
            ui.label("Force-show override active").classes(
                "text-sm font-semibold",
            )
            ui.label(
                "No camera-bearing tool is active. "
                "Calibration data is for UI inspection only.",
            ).classes("text-xs")

    # Scrollable body. ``calc(100vh - 80px)`` reserves room for the page
    # chrome above the panel; the column scrolls internally when content
    # exceeds the viewport (which happens once the Calibration settings
    # expansion is opened).
    body = ui.column().classes("w-full")
    body.style("max-height: calc(100vh - 80px); overflow-y: auto;")
    with body:

        _state["status_label"] = ui.label("Idle.").classes(
            "text-xs opacity-80"
        )

        with ui.row().classes("gap-1 q-mt-sm"):
            ui.button("Run", on_click=_on_run, color="primary").props("size=sm")
            ui.button(
                "Localise Board", on_click=_on_localise, color="secondary",
            ).props("size=sm")
            ui.button("STOP", on_click=_on_stop, color="negative").props("size=sm")

        with ui.row().classes("gap-1 q-mt-xs"):
            def _on_check_current_pose() -> None:
                """Single-shot collision check on the live joint
                configuration. Useful as a debug tool: tells the user
                whether the current pose is currently safe per the
                gripper-vs-environment manager.
                """
                try:
                    from waldo_commander.state import (  # noqa: PLC0415
                        robot_state,
                    )

                    from .collision import (  # noqa: PLC0415
                        validate_joint_trajectory,
                    )
                    cur = list(robot_state.angles.deg[:6])
                    # gripper_only=True matches the docstring above
                    # ("per the gripper-vs-environment manager") and
                    # the design decision in commit d609024. With
                    # gripper_only=False, parol6's simplified-mesh
                    # arm self-collisions would falsely flag the
                    # current static pose as unsafe in configurations
                    # not covered by the adjacent-pair whitelist.
                    result = validate_joint_trajectory(
                        cur, cur, gripper_only=True,
                    )
                    if not result.get("manager_ready", False):
                        _post_status(
                            f"Pose check: {result.get('reason', 'unavailable')}.",
                        )
                        return
                    if result.get("safe", True):
                        _post_status(
                            "Pose check: current pose is collision-free.",
                        )
                    else:
                        pair = result.get("colliding_pair")
                        reason = result.get("reason", "collision")
                        if pair is not None:
                            _post_status(
                                f"Pose check: end_safe={result.get('end_safe')}, "
                                f"reason={reason}, pair={pair[0]} <-> {pair[1]}.",
                            )
                        else:
                            _post_status(f"Pose check: {reason}.")
                except (ImportError, AttributeError, RuntimeError) as e:
                    _post_status(f"Pose check failed: {e}")

            ui.button(
                "Check current pose", on_click=_on_check_current_pose,
            ).props("size=sm flat color=info")

        ui.separator().classes("q-my-sm")

        with ui.expansion("View overlays", icon="visibility").classes("w-full"):
            def _make_handler(name: str):
                # Closure-free factory so each checkbox binds to its own name.
                def _on_change(e) -> None:
                    visible = bool(e.value)
                    _set_overlay_visible(name, visible)
                    _persist_overlay_pref(name, visible)
                return _on_change

            for name, label, _default in _OVERLAY_TOGGLES:
                ui.checkbox(
                    label,
                    value=bool(_state.get(f"show_{name}", True)),
                    on_change=_make_handler(name),
                ).props("dense")

            ui.separator().classes("q-my-xs")

            def _on_override_toggle(e) -> None:
                value = bool(getattr(e, "value", False))
                _set_no_camera_override(value)
                # Live-apply: build / tear down overlays + refresh
                # this panel so its three-state branch picks up the
                # new gate state.
                apply_calibration_state()
                if value:
                    ui.notify(
                        "Override on. Calibration data is for UI inspection only.",
                        color="warning", position="top",
                    )
                else:
                    ui.notify(
                        "Override off.", color="info", position="top",
                    )

            ui.checkbox(
                "Force-show overlays without a camera tool",
                value=_no_camera_override(),
                on_change=_on_override_toggle,
            ).props("dense")

        # Hover-above-board verification — drive the camera to a known XY on
        # the board surface at a configurable standoff height, look straight
        # down. Lets the user physically measure with a ruler/caliper and
        # check whether the calibration's mount transform is right. Standoff
        # is measured perpendicular to the board surface (board-local +Z).
        with ui.expansion(
            "Hover above board (verification)", icon="straighten",
        ).classes("w-full"):
            # Persist standoff and mode so they survive a page reload.
            try:
                from nicegui import app as _nicegui_app  # noqa: PLC0415
                _persisted_standoff = float(
                    _nicegui_app.storage.user.get("calib_hover_standoff_mm", 100.0)
                )
                _persisted_hover_mode = str(
                    _nicegui_app.storage.user.get("calib_hover_mode", "camera")
                )
                if _persisted_hover_mode not in ("camera", "tcp"):
                    _persisted_hover_mode = "camera"
            except Exception:  # noqa: BLE001
                _persisted_standoff = 100.0
                _persisted_hover_mode = "camera"

            with ui.row().classes("items-center gap-2 q-mt-xs"):
                standoff_input = (
                    ui.number(
                        label="Standoff (mm)",
                        value=_persisted_standoff,
                        min=10.0, max=300.0, step=5.0, format="%.0f",
                    )
                    .props("dense")
                    .classes("w-32")
                )

                def _persist_standoff(_e) -> None:
                    try:
                        from nicegui import app as _na  # noqa: PLC0415
                        _na.storage.user["calib_hover_standoff_mm"] = float(
                            standoff_input.value
                        )
                    except Exception:  # noqa: BLE001
                        pass

                standoff_input.on("update:model-value", _persist_standoff)

                # Reference-frame toggle: "Camera" hovers the optical centre
                # (validates calibration), "TCP" hovers the gripper fingertips
                # (validates kinematics chain only — independent of calibration).
                hover_mode_input = (
                    ui.toggle(
                        {"camera": "Camera", "tcp": "TCP"},
                        value=_persisted_hover_mode,
                    )
                    .props("dense color=primary unelevated")
                )

                def _persist_hover_mode(_e) -> None:
                    try:
                        from nicegui import app as _na  # noqa: PLC0415
                        _na.storage.user["calib_hover_mode"] = str(
                            hover_mode_input.value or "camera"
                        )
                    except Exception:  # noqa: BLE001
                        pass

                hover_mode_input.on("update:model-value", _persist_hover_mode)

            _hover_cfg = current_board_config()
            bw_mm = _hover_cfg.squares_x * _hover_cfg.square_length * 1000.0
            bh_mm = _hover_cfg.squares_y * _hover_cfg.square_length * 1000.0
            # Board-local frame: origin at one corner, +X along squares_x (long
            # edge, 210 mm), +Y along squares_y (short edge, 150 mm). The
            # buttons label the corners by their (X-low/high, Y-low/high) name
            # — "Origin" is (0, 0), "TR" = top-right = (max_x, max_y), etc.
            hover_presets: list[tuple[str, float, float]] = [
                ("Centre",  bw_mm / 2.0, bh_mm / 2.0),
                ("Origin",  0.0,         0.0),
                ("X+",      bw_mm,       0.0),
                ("Y+",      0.0,         bh_mm),
                ("X+Y+",    bw_mm,       bh_mm),
            ]

            def _make_hover_handler(local_x_mm: float, local_y_mm: float):
                def _click() -> None:
                    if _busy_warn("Hover"):
                        return
                    try:
                        standoff_mm = float(standoff_input.value or 0.0)
                    except (TypeError, ValueError):
                        ui.notify(
                            "Hover: standoff must be a number", color="warning",
                        )
                        return
                    if standoff_mm < 10.0:
                        ui.notify(
                            f"Hover: standoff {standoff_mm:.0f} mm too small "
                            f"(min 10 mm)",
                            color="warning",
                        )
                        return
                    mode = str(hover_mode_input.value or "camera")
                    _state["is_hovering"] = True
                    _state["stop_requested"] = False
                    # Prime the per-tool override cache so the hover
                    # worker can read the active tool's mount.
                    custom_tools.prime_per_tool_overrides_cache()
                    settings.load_from_storage()
                    threading.Thread(
                        target=_drive_hover_pose_thread,
                        args=(
                            local_x_mm / 1000.0,
                            local_y_mm / 1000.0,
                            standoff_mm / 1000.0,
                            mode,
                        ),
                        daemon=True,
                    ).start()
                return _click

            with ui.row().classes("gap-1 q-mt-sm"):
                for label, local_x, local_y in hover_presets:
                    ui.button(
                        label, on_click=_make_hover_handler(local_x, local_y),
                    ).props("size=sm outline")

        # ------------------------------------------------------------------
        # Calibration settings — UI-driven tunables persisted via
        # app.storage.user. Live-applied where possible (board placement,
        # hemisphere, surface, camera mount); the localise / collision / etc.
        # values pick up on next thread invocation.
        # ------------------------------------------------------------------
        settings.load_from_storage()

        @ui.refreshable
        def _settings_panel() -> None:
            # Preset bar at the top of the settings block.
            settings_ui.build_preset_bar(refresh_panel=_settings_panel.refresh)
            ui.separator().classes("q-my-sm")
            settings_ui.build_calibration_settings_expansion()
            custom_tools_ui.build_custom_tools_expansion()

        _settings_panel()
