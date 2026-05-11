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
        # Re-fetch in case the page tore down between scheduling and dispatch.
        live_label = _state.get("status_label")
        if live_label is None:
            return
        try:
            live_label.text = text
        except Exception:  # noqa: BLE001
            # Label detached / disposed.
            pass

    try:
        loop.call_soon_threadsafe(_update)
    except RuntimeError as e:
        # Loop closed (page tear-down) — not a real failure.
        logger.info("Status post skipped (loop unavailable): %s", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not post GUI status: %s", e)


# ---------------------------------------------------------------------------
# Calibration tab content
# ---------------------------------------------------------------------------


# Suffix keys for ``_set_overlay_visible`` + (label, default_visible).
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
    """Seed ``_state`` from ``app.storage.user``; defaults win on missing keys.
    Failures fall back to the default for the key.
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
    """Read the soft-toggle state from ``app.storage.general``. Default OFF."""
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
    """Per-user override that renders overlays without a camera-bearing tool.
    Default OFF; a banner reminds the user the values aren't tied to a real
    camera when ON.
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
    """True iff the active tool is camera-bearing OR the user override is on."""
    return custom_tools.active_tool_is_camera_bearing() or _no_camera_override()


# ---------------------------------------------------------------------------
# Live-apply machinery — replaces the previous "reload page to apply" pattern
# ---------------------------------------------------------------------------


def _teardown_overlays() -> None:
    """Delete every scene group + dynamic-overlay handle and signal worker
    threads to STOP. Called when leaving the camera-bearing-render state
    (toggle off, tool change) and on browser disconnect.
    """
    # Workers poll ``stop_requested`` between move commands.
    _state["stop_requested"] = True
    # Dispatch a halt to abort any in-flight motion. ``_state["client"]``
    # is set by the worker threads; no client means no-op.
    _client = _state.get("client")
    if _client is not None:
        # Halt on a fresh thread: ``client.halt`` is the sync wrapper
        # around an async halt whose inbox is bound to the sync client's
        # own loop. Scheduling on the NiceGUI loop trips a queue-loop
        # binding mismatch; a fresh thread has no running loop so the
        # wrapper's own thread-bound loop is used.
        import threading  # noqa: PLC0415

        def _halt_in_thread() -> None:
            try:
                _client.halt()
            except Exception as e:  # noqa: BLE001
                logger.debug("teardown halt() in thread raised: %s", e)

        threading.Thread(
            target=_halt_in_thread,
            daemon=True,
            name="calib-teardown-halt",
        ).start()
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
    # Cancel scene timers so they don't fire against a torn-down scene
    # or accumulate across features-on/off cycles.
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
    # Clear per-tick rebuild cache so a future rebuild isn't short-circuited.
    _state["footprint_last_q"] = None
    _state["footprint_last_mount"] = None
    # Bump ``reach_generation`` under the lock so a racing click handler
    # sees a stale-gen mismatch rather than indexing an empty list.
    with _state_lock:
        _state["reach_generation"] = (
            int(_state.get("reach_generation", 0)) + 1
        )
        _state["reachable_candidates"] = []
    _state["reachable_points_world"] = None
    _state["reachable_target_world"] = None
    # Cancel any deferred reachability render waiting on scene 'init'.
    pending_timer = _state.get("reachability_pending_timer")
    if pending_timer is not None:
        try:
            pending_timer.cancel()
        except Exception as e:  # noqa: BLE001
            logger.debug("reachability pending-timer cancel failed: %s", e)
        _state["reachability_pending_timer"] = None
    _state["reachability_pending_payload"] = None
    # Cancel the scene-init defer timer.
    init_timer = _state.get("scene_init_defer_timer")
    if init_timer is not None:
        try:
            init_timer.cancel()
        except Exception as e:  # noqa: BLE001
            logger.debug("scene init defer-timer cancel failed: %s", e)
        _state["scene_init_defer_timer"] = None
    # Tear down the click-on-dot popup + scene click handler; ``add_overlays``
    # will recreate them.
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
    # Clearing ``current_mount`` makes the footprint tick a no-op until
    # rebuilt. ``scene_root`` is left intact.
    _state["current_mount"] = None
    _state["overlays_built"] = False
    # Clear so a late indicator tick can't poke a deleted Quasar element.
    _state["live_pose_label"] = None
    _state["live_pose_tooltip"] = None
    # Force-exit any active preview so a feature-off cycle can't strand
    # the URDF in PREVIEW.
    try:
        from .preview_dialog import reset_preview_state  # noqa: PLC0415

        reset_preview_state()
    except (ImportError, AttributeError):
        pass


def _ensure_features_loaded() -> None:
    """One-shot init: SSG-48 auto-migrate + register_all + active_robot
    tools rebuild. Mirrors ``main.initialize_urdf_scene``'s behaviour so a
    live off→on flip matches a reload. Idempotent.
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
    and active tool. Idempotent. Call after master-toggle flip, active-
    tool change, or force-show override toggle.

    Three coarse transitions:

    * ``should_render`` AND not built — runs auto-migrate + register_all
      + ``add_overlays``.
    * ``should_render`` AND built — rebuilds ``CameraMount`` so per-tool
      overrides take effect, redraws the frustum.
    * Not ``should_render`` AND built — tears down scene groups.

    Always refreshes the panel so its three-state branch picks up the
    new ``(features_on, gate_open)`` combination.
    """
    # Prime the per-tool override cache from this request context so
    # any worker thread spawned downstream can read the active tool's
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
        # Tool change: re-pick up per-tool intrinsics + cam_mount and
        # redraw the frustum.
        try:
            from .live_apply import _rebuild_camera_mount  # noqa: PLC0415

            _rebuild_camera_mount()
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "apply_calibration_state: _rebuild_camera_mount failed: %s", e,
            )
        # Force centerline + footprint to redraw on the next tick;
        # without this the cache may keep the OLD tool's lines visible
        # until joint movement trips the epsilon check.
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
        # Drop reachability cache and re-run the IK sweep — mount + body
        # changes flip which solutions are valid.
        try:
            from .reachability import refresh_reachability_for_active_tool  # noqa: PLC0415

            refresh_reachability_for_active_tool()
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "apply_calibration_state: reachability refresh failed: %s", e,
            )
        # Defensive re-registration of the click handler.
        # ``register_click_handler`` is idempotent.
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

    # Refresh the panel so its three-state branch picks up the new state.
    refresh = _state.get("panel_refresh")
    if refresh is not None:
        try:
            refresh()
        except Exception as e:  # noqa: BLE001
            logger.debug("panel refresh failed: %s", e)


def _build_inactive_placeholder(close_callback: Callable[[], None] | None) -> None:
    """Header + explanation + master toggle. No scene mutations."""
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
        # Read e.value, not sw.value — Quasar may not have propagated yet.
        new_value = bool(getattr(e, "value", False))
        if not new_value:
            return  # only acting on flips to ON here
        _set_features_active(True)
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
    """Render the placeholder shown when features are ON but the active
    tool isn't camera-bearing: issue + force-show toggle + custom-tools UI.
    """
    # GUI's logical tool key wins over the controller's broadcast — the
    # controller only knows built-ins, so a custom tool with a proxy
    # would report under the proxy's name.
    try:
        from .custom_tools import _active_gui_tool_key  # noqa: PLC0415
        active_key = _active_gui_tool_key() or "NONE"
    except Exception:  # noqa: BLE001
        # Fall back to the controller's broadcast key when storage
        # raises outside a request context.
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

    # Custom-tools UI so the user can flag a tool as camera-bearing here.
    ui.separator().classes("q-my-sm")
    custom_tools_ui.build_custom_tools_expansion()


def build_calibration_panel_content(close_callback: Callable[[], None] | None = None) -> None:
    """Build the calibration tab.

    Three states selected at render:

    * Features off — master toggle + explanation.
    * Features on, no camera-bearing tool, no override — no-camera
      placeholder + force-show toggle + custom-tools UI.
    * Features on + camera-bearing tool (or override on) — full panel.

    The branch is wrapped in :func:`ui.refreshable` so
    :func:`apply_calibration_state` can repaint in place after scene
    mutations.
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
    """Camera-bearing full panel: Run / Localise / STOP + view-overlay
    toggles + hover-above-board + calibration settings + custom-tools UI.
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
        # Prime caches from the request context — the worker thread
        # can't read app.storage.user directly.
        custom_tools.prime_per_tool_overrides_cache()
        settings.load_from_storage()
        _post_status("Running calibration via parol6-server...")
        threading.Thread(target=_calibration_thread, daemon=True).start()

    def _on_run() -> None:
        if _busy_warn("Run"):
            return
        # Without a successful Localise this session, calibration aims at
        # the configured ``_BOARD_TRANSLATE_M`` — rarely accurate on real
        # hardware. Confirm before running.
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
        # Prime caches so the worker sees per-tool intrinsics + mount.
        custom_tools.prime_per_tool_overrides_cache()
        settings.load_from_storage()
        _post_status("Localising board, driving lookout sweep...")
        threading.Thread(target=_localise_board_thread, daemon=True).start()

    def _on_stop() -> None:
        """Abort the running calibration or localise. Dispatches halt on
        a daemon thread because ``RobotClient.halt()`` is sync UDP and
        refuses to run inside an active asyncio loop.
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

    # Reseed _state from storage and reapply visibility for the static
    # groups; initial overlay build runs before this with defaults.
    _load_persisted_overlay_prefs()
    for name, _label, _default in _OVERLAY_TOGGLES:
        _set_overlay_visible(name, _state.get(f"show_{name}", True))

    # Pinned header; action row + status label live inside the scroll area
    # so expansions can't push them off-screen.
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
        # Live-pose collision indicator updated by a 2 Hz tick.
        # Tooltip element is created ONCE and reused; ``chip.tooltip(text)``
        # constructs a fresh Tooltip each call, so per-tick recreation
        # would stack hundreds and fire them all on hover.
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

    # Warning when overlays render only because of the force-show override.
    if _no_camera_override() and not custom_tools.active_tool_is_camera_bearing():
        with ui.card().classes("w-full bg-amber-9 text-white q-mt-xs"):
            ui.label("Force-show override active").classes(
                "text-sm font-semibold",
            )
            ui.label(
                "No camera-bearing tool is active. "
                "Calibration data is for UI inspection only.",
            ).classes("text-xs")

    # Scrollable body. ``calc(100vh - 80px)`` reserves room for page chrome.
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
                """Single-shot collision check on the live joint config."""
                try:
                    from waldo_commander.state import (  # noqa: PLC0415
                        robot_state,
                    )

                    from .collision import (  # noqa: PLC0415
                        validate_joint_trajectory,
                    )
                    cur = list(robot_state.angles.deg[:6])
                    # gripper_only=True so parol6's simplified-mesh arm
                    # doesn't falsely flag adjacent-link contact.
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
                # Factory binds each checkbox to its own ``name``.
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

        # Hover-above-board verification: drives the camera to a known
        # board-local XY at the configured standoff (board-local +Z) so
        # the user can physically measure against the calibration.
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

                # Reference frame: Camera hovers optical centre (validates
                # calibration); TCP hovers fingertips (validates kinematics
                # only, independent of calibration).
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
            # Board-local frame: origin at one corner, +X along squares_x,
            # +Y along squares_y. Buttons label corners by (X, Y) extreme.
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
                    # Prime caches so the worker sees the active tool's mount.
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

        # Calibration settings — tunables persisted via app.storage.user.
        # Live-applied for board/hemisphere/surface/mount; localise +
        # collision values pick up on the next thread invocation.
        settings.load_from_storage()

        @ui.refreshable
        def _settings_panel() -> None:
            # Preset bar at the top of the settings block.
            settings_ui.build_preset_bar(refresh_panel=_settings_panel.refresh)
            ui.separator().classes("q-my-sm")
            settings_ui.build_calibration_settings_expansion()
            custom_tools_ui.build_custom_tools_expansion()

        _settings_panel()
