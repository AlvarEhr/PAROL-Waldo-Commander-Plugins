"""Settings component for serial port, theme, and visualization preferences."""

import logging
from collections.abc import Callable
from contextlib import contextmanager

from nicegui import app as ng_app
from nicegui import ui

from waldoctl import RobotClient

from waldo_commander.services.camera_service import (
    camera_service,
    enumerate_video_devices,
)
from waldo_commander.state import EnvelopeMode, robot_state, simulation_state, ui_state

logger = logging.getLogger(__name__)


def get_available_serial_ports() -> list[str]:
    """Detect available serial ports on the system."""
    try:
        import serial.tools.list_ports

        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    except ImportError:
        logger.warning("pyserial not installed - cannot detect serial ports")
        return []
    except OSError as e:
        logger.error("Error detecting serial ports: %s", e)
        return []


@contextmanager
def _setting_row(title: str, description: str):
    """Standard layout for a settings row: label column + yielded control widget."""
    with ui.row().classes("items-center justify-between w-full overflow-hidden"):
        with ui.column().classes("gap-0 overflow-hidden flex-shrink"):
            ui.label(title).classes("text-sm font-medium truncate")
            ui.label(description).classes(
                "text-xs text-gray-500 dark:text-gray-400 truncate"
            )
        yield


def _live_apply_calibration_state() -> None:
    """Best-effort hook into the parol6-vision calibration overlays so
    a tool change / master-toggle flip / force-show flip live-applies
    without a page reload.

    Lazy-imports the package because ``WALDO_CALIBRATION_ENABLED=0``
    keeps it out of the main load path entirely. Failures are
    swallowed — the calibration package is optional for the rest of
    the app.
    """
    try:
        from waldo_commander.components.calibration_overlays import (  # noqa: PLC0415
            apply_calibration_state,
        )

        apply_calibration_state()
    except Exception as e:  # noqa: BLE001
        logger.debug("calibration overlays live-apply skipped: %s", e)


class SettingsContent:
    """Settings content that can be embedded in the control panel."""

    def __init__(self, client: RobotClient) -> None:
        self.client = client
        self._port_select: ui.select | None = None
        self._refresh_timer: ui.timer | None = None
        self._cam_select: ui.select | None = None
        self._cam_refresh_timer: ui.timer | None = None
        self._variant_container: ui.column | None = None
        self._tcp_offset_container: ui.column | None = None
        # Tool dropdown reference + the parol6-vision callback we
        # subscribe to so newly-added / deleted custom tools refresh
        # the dropdown without a page reload.
        self._tool_select: ui.select | None = None
        self._tool_registry_cb: Callable[[], None] | None = None

    def _load_preferences(self) -> dict:
        """Load persisted preferences from storage."""
        valid_profiles = ui_state.active_robot.motion_profiles
        stored_profile = ng_app.storage.general.get("motion_profile", "TOPPRA")
        if stored_profile not in valid_profiles and valid_profiles:
            stored_profile = valid_profiles[0]
        return {
            "com_port": ng_app.storage.general.get("com_port", ""),
            "show_route": ng_app.storage.general.get("show_route", True),
            "envelope_mode": EnvelopeMode(
                ng_app.storage.general.get("envelope_mode", "auto")
            ),
            "theme_mode": ng_app.storage.general.get("theme_mode", "system"),
            "motion_profile": stored_profile,
        }

    def _refresh_serial_ports(self) -> None:
        """Refresh the available serial ports in the dropdown."""
        if self._port_select:
            ports = get_available_serial_ports()
            self._port_select.options = ports
            self._port_select.update()

    def cleanup(self) -> None:
        """Cancel background timers during shutdown."""
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()
        if self._cam_refresh_timer is not None:
            self._cam_refresh_timer.cancel()
        # Unsubscribe the tool-registry-changed callback so a stale
        # SettingsContent instance can't keep firing dropdown updates
        # after the panel is torn down.
        if self._tool_registry_cb is not None:
            try:
                from waldo_commander.components.calibration_overlays import (  # noqa: PLC0415
                    custom_tools as _ct,
                )

                _ct.unsubscribe_tool_registry_changed(self._tool_registry_cb)
            except Exception:  # noqa: BLE001
                pass
            self._tool_registry_cb = None

    def _refresh_tool_options(self) -> None:
        """Re-read parol6's tool registry, update the gripper dropdown's
        options, and sync the visible value to whatever is currently
        persisted in ``app.storage.general["selected_tool"]``.

        Called after a custom tool is registered or deleted via the
        calibration_overlays callback. Reading the persisted selection
        ensures that the "Use this tool" button on a custom-tool card
        (which writes ``selected_tool`` but doesn't directly touch the
        dropdown) updates the dropdown's visible value.
        """
        if self._tool_select is None:
            return
        try:
            tool_options = {
                t.key: t.display_name
                for t in ui_state.active_robot.tools.available
            }
        except Exception:  # noqa: BLE001
            return
        self._tool_select.options = tool_options
        # Sync the displayed selection to the persisted active tool so
        # a programmatic tool change updates the dropdown. Falls back
        # to the first option when the persisted value is gone (e.g.
        # the user just deleted the active custom tool).
        stored = ng_app.storage.general.get("selected_tool")
        if stored and stored in tool_options:
            self._tool_select.value = stored
        elif self._tool_select.value not in tool_options:
            first = next(iter(tool_options), None)
            if first is not None:
                self._tool_select.value = first
        self._tool_select.update()

    # ── Tool helpers (promoted from closures) ────────────────────────

    def _get_variant_key(self, tool_key: str) -> str | None:
        """Get stored variant key for a tool, or first variant key."""
        try:
            tool_spec = ui_state.active_robot.tools[tool_key]
        except (KeyError, AttributeError):
            return None
        variants = tool_spec.variants
        if not variants:
            return None
        stored = ng_app.storage.general.get(f"tool_variant_{tool_key}")
        if stored and any(v.key == stored for v in variants):
            return stored
        return variants[0].key

    def _get_tcp_offset(self, tool_key: str) -> dict:
        """Get stored TCP offset for a tool (mm)."""
        return ng_app.storage.general.get(
            f"tcp_offset_{tool_key}", {"x": 0, "y": 0, "z": 0}
        )

    def _tcp_offset_m(self, tool_key: str) -> tuple[float, float, float] | None:
        """Get stored TCP offset in meters, or None if zero."""
        o = self._get_tcp_offset(tool_key)
        x, y, z = o.get("x", 0), o.get("y", 0), o.get("z", 0)
        if x == 0 and y == 0 and z == 0:
            return None
        return (x / 1000, y / 1000, z / 1000)

    def _notify_and_resimulate(self) -> None:
        """Notify simulation state changed and trigger debounced re-simulation."""
        simulation_state.notify_changed()
        try:
            ui_state.editor_panel.schedule_debounced_simulation()
        except RuntimeError:
            pass

    def _apply_tool_scene(self, tool_key: str, variant_key: str | None = None) -> None:
        """Apply tool to local FK/IK model and 3D scene."""
        ui_state.active_robot.set_active_tool(
            tool_key,
            tcp_offset_m=self._tcp_offset_m(tool_key),
            variant_key=variant_key,
        )
        if ui_state.urdf_scene:
            ui_state.urdf_scene.apply_tool(tool_key, variant_key=variant_key)

    def _rebuild_variant_selector(self, tool_key: str) -> None:
        """Rebuild variant sub-selector for the current tool."""
        assert self._variant_container is not None
        self._variant_container.clear()
        try:
            tool_spec = ui_state.active_robot.tools[tool_key]
        except (KeyError, AttributeError):
            tool_spec = None
        variants = tool_spec.variants if tool_spec else ()
        is_none = tool_key == "NONE"
        if not variants and not is_none:
            return

        variant_options = (
            {v.key: v.display_name for v in variants} if variants else {"": "—"}
        )
        current_vk = self._get_variant_key(tool_key) or (
            next(iter(variant_options), "") if variants else ""
        )

        async def _on_variant_change(e):
            vk = e.value
            ng_app.storage.general[f"tool_variant_{tool_key}"] = vk
            robot_state.tool_variant_key = vk or ""
            self._apply_tool_scene(tool_key, variant_key=vk)
            self._notify_and_resimulate()

        with self._variant_container:
            with _setting_row("Variant", "Tool configuration variant"):
                sel = (
                    ui.select(
                        options=variant_options,
                        value=current_vk,
                        on_change=_on_variant_change,
                    )
                    .classes("w-32")
                    .props("dense")
                    .mark("select-tool-variant")
                )
                if is_none or not variants:
                    sel.props("disable")

    def _rebuild_tcp_offset(self, tool_key: str) -> None:
        """Rebuild per-tool TCP offset inputs."""
        assert self._tcp_offset_container is not None
        self._tcp_offset_container.clear()
        is_none = tool_key == "NONE"
        offset = (
            self._get_tcp_offset(tool_key) if not is_none else {"x": 0, "y": 0, "z": 0}
        )

        async def _on_offset_change(_e=None):
            vals = {
                "x": x_input.value or 0,
                "y": y_input.value or 0,
                "z": z_input.value or 0,
            }
            ng_app.storage.general[f"tcp_offset_{tool_key}"] = vals
            vk = self._get_variant_key(tool_key)
            self._apply_tool_scene(tool_key, variant_key=vk)
            self._notify_and_resimulate()

        with self._tcp_offset_container:
            with _setting_row("TCP Offset", "Offset from default TCP (mm)"):
                with ui.row().classes("gap-1"):
                    x_input = (
                        ui.number(label="X", value=offset.get("x", 0), step=0.5)
                        .style("width: 48px;")
                        .props("dense borderless" + (" disable" if is_none else ""))
                        .on("update:model-value", _on_offset_change)
                    )
                    y_input = (
                        ui.number(label="Y", value=offset.get("y", 0), step=0.5)
                        .style("width: 48px;")
                        .props("dense borderless" + (" disable" if is_none else ""))
                        .on("update:model-value", _on_offset_change)
                    )
                    z_input = (
                        ui.number(label="Z", value=offset.get("z", 0), step=0.5)
                        .style("width: 48px;")
                        .props("dense borderless" + (" disable" if is_none else ""))
                        .on("update:model-value", _on_offset_change)
                    )

    # ── Section builders ─────────────────────────────────────────────

    def _build_serial_port(self, prefs: dict) -> None:
        available_ports = get_available_serial_ports()
        stored_port = prefs["com_port"]

        with _setting_row("Serial Port", "Select robot communication port"):
            self._port_select = (
                ui.select(
                    options=available_ports,
                    value=stored_port if stored_port in available_ports else None,
                    label="Port",
                    new_value_mode="add-unique",
                    clearable=True,
                )
                .classes("w-32")
                .props("dense")
            )

        if stored_port and stored_port not in available_ports:
            self._port_select.value = stored_port

        port_select_ref = self._port_select

        async def _apply_port():
            port_val = port_select_ref.value or ""
            try:
                await self.client.connect_hardware(port_val)
            except Exception as exc:
                logger.warning("connect_hardware(%s) failed: %s", port_val, exc)
                ui.notify(f"Port change failed: {exc}", color="negative")
                return
            ng_app.storage.general["com_port"] = port_val
            ui.notify(f"SET_PORT {port_val}", color="primary")

        port_select_ref.on("update:model-value", lambda e: _apply_port())
        self._refresh_timer = ui.timer(10.0, self._refresh_serial_ports)

    def _build_show_route(self, prefs: dict) -> None:
        async def _on_show_route_change(e):
            val = bool(e.value)
            simulation_state.paths_visible = val
            ng_app.storage.general["show_route"] = val
            simulation_state.notify_changed()

        with _setting_row("Show Route", "Display path visualization in 3D view"):
            ui.switch(
                value=prefs["show_route"],
                on_change=_on_show_route_change,
            ).props("dense").mark("switch-show-route")

        simulation_state.paths_visible = prefs["show_route"]

    def _build_envelope(self, prefs: dict) -> None:
        async def _on_envelope_mode_change(e):
            mode = EnvelopeMode(e.value)
            simulation_state.envelope_mode = mode
            ng_app.storage.general["envelope_mode"] = mode.value
            simulation_state.notify_changed()

        with _setting_row("Workspace Envelope", "Show reachable workspace boundary"):
            ui.select(
                options={m.value: m.value.capitalize() for m in EnvelopeMode},
                value=prefs["envelope_mode"].value,
                on_change=_on_envelope_mode_change,
            ).classes("w-24").props("dense").mark("select-envelope-mode")

        simulation_state.envelope_mode = prefs["envelope_mode"]

    def _build_tool_section(self) -> None:
        async def _on_tool_change(e):
            tool = e.value
            vk = self._get_variant_key(tool)
            # Custom tools (``custom:<name>``) live only in the GUI
            # process — the controller's _TOOL_REGISTRY doesn't know
            # them, so ``client.select_tool("custom:...")`` raises with
            # "Unknown tool". Instead, route motor commands through the
            # custom tool's ``proxy_tool_key`` (a built-in the
            # controller does know), and apply the local scene + state
            # using the custom key.
            if isinstance(tool, str) and tool.startswith("custom:"):
                proxy = ""
                try:
                    from waldo_commander.components.calibration_overlays import (  # noqa: PLC0415
                        custom_tools as _ct,
                    )

                    cfg = _ct.load_config(tool[len("custom:"):])
                    if cfg is not None:
                        proxy = str(cfg.proxy_tool_key or "")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("custom-tool config lookup: %s", exc)
                if proxy:
                    try:
                        await self.client.select_tool(proxy, variant_key=vk or "")
                    except Exception as exc:  # noqa: BLE001
                        logger.info(
                            "proxy select_tool(%s) for %s failed: %s",
                            proxy, tool, exc,
                        )
                        # Don't bail — local apply still works; tell
                        # the user motor commands won't be live.
                        ui.notify(
                            f"Custom tool {tool} active locally; motor "
                            f"commands proxied to {proxy} failed "
                            f"({exc}). Visualisation + IK still correct.",
                            color="warning", position="top",
                        )
                ng_app.storage.general["selected_tool"] = tool
                robot_state.tool_variant_key = vk or ""
                self._apply_tool_scene(tool, variant_key=vk)
                self._rebuild_variant_selector(tool)
                self._rebuild_tcp_offset(tool)
                self._notify_and_resimulate()
                # Live-apply calibration overlays + panel for the new
                # tool: re-evaluates camera-bearing gating, picks up
                # per-tool intrinsic / mount overrides, builds or
                # tears down scene overlays as needed. No page reload
                # required.
                _live_apply_calibration_state()
                ui.notify(
                    f"Switched to {tool}.",
                    color="info", position="top",
                )
                return

            try:
                await self.client.select_tool(tool, variant_key=vk or "")
            except Exception as exc:
                logger.warning("select_tool(%s) failed: %s", tool, exc)
                ui.notify(f"Tool change failed: {exc}", color="negative")
                return

            ng_app.storage.general["selected_tool"] = tool
            robot_state.tool_variant_key = vk or ""
            self._apply_tool_scene(tool, variant_key=vk)
            self._rebuild_variant_selector(tool)
            self._rebuild_tcp_offset(tool)
            self._notify_and_resimulate()
            # Live-apply calibration overlays + panel for the new
            # tool — see custom-tool branch above.
            _live_apply_calibration_state()

        # Defensive rebuild — Robot._tools is a snapshot taken at
        # construction time, so any tools registered afterwards (the
        # parol6-vision custom tools system pushes ``custom:<name>``
        # entries into the registry post-init) wouldn't show up here
        # without this. Cheap call (~ms) so safe to run on every
        # settings-panel open.
        try:
            from parol6.robot import _build_tools as _parol6_build_tools  # noqa: PLC0415

            ui_state.active_robot._tools = _parol6_build_tools()  # type: ignore[attr-defined]
        except Exception as _e:  # noqa: BLE001
            logger.debug(
                "settings: parol6 tool refresh skipped (%s)", _e,
            )

        tool_options = {}
        for tool in ui_state.active_robot.tools.available:
            tool_options[tool.key] = tool.display_name

        default_tool = next(iter(tool_options), "NONE")
        stored_tool = ng_app.storage.general.get("selected_tool", default_tool)
        if stored_tool not in tool_options:
            stored_tool = default_tool

        with _setting_row("Tool", "Select end effector tool"):
            self._tool_select = (
                ui.select(
                    options=tool_options,
                    value=stored_tool,
                    on_change=_on_tool_change,
                )
                .classes("w-32")
                .props("dense")
                .mark("select-tool")
            )

        # Subscribe to parol6-vision's tool-registry-changed events so
        # newly-added / deleted custom tools update the dropdown
        # without a page reload. Idempotent — re-builds of the section
        # (panel refresh, etc.) replace the prior subscription with a
        # fresh one bound to the new dropdown ref.
        if self._tool_registry_cb is not None:
            try:
                from waldo_commander.components.calibration_overlays import (  # noqa: PLC0415
                    custom_tools as _ct_unsub,
                )

                _ct_unsub.unsubscribe_tool_registry_changed(
                    self._tool_registry_cb,
                )
            except Exception:  # noqa: BLE001
                pass
        self._tool_registry_cb = self._refresh_tool_options
        try:
            from waldo_commander.components.calibration_overlays import (  # noqa: PLC0415
                custom_tools as _ct_sub,
            )

            _ct_sub.subscribe_tool_registry_changed(self._tool_registry_cb)
        except Exception as _e:  # noqa: BLE001
            logger.debug(
                "settings: tool-registry subscribe skipped (%s)", _e,
            )

        self._variant_container = ui.column().classes("w-full gap-1")
        self._rebuild_variant_selector(stored_tool)

        self._tcp_offset_container = ui.column().classes("w-full gap-1")
        self._rebuild_tcp_offset(stored_tool)

        vk_initial = self._get_variant_key(stored_tool)
        robot_state.tool_variant_key = vk_initial or ""
        if stored_tool:
            self._apply_tool_scene(stored_tool, variant_key=vk_initial)

    def _build_camera(self) -> None:
        stored_cam = ng_app.storage.general.get("camera_device", -1)
        cam_devices = enumerate_video_devices()
        cam_options: dict[int | str, str] = {-1: "Disabled"}
        for dev in cam_devices:
            cam_options[dev["index"]] = str(dev["label"])
        # Validate stored camera still exists (index may change across reboots)
        if stored_cam not in cam_options:
            stored_cam = -1
            ng_app.storage.general["camera_device"] = -1
        ui_state.camera_device = stored_cam

        def _on_camera_change(e):
            val = e.value
            ng_app.storage.general["camera_device"] = val
            ui_state.camera_device = val
            if val is None or val == -1:
                camera_service.stop()
            else:
                camera_service.start(val)

        with _setting_row("Camera", "Video device for gripper panel"):
            self._cam_select = (
                ui.select(
                    options=cam_options,
                    value=stored_cam,
                    on_change=_on_camera_change,
                    new_value_mode="add-unique",
                    clearable=True,
                )
                .classes("w-32")
                .props("dense")
                .mark("select-camera")
            )

        with ui.column().classes("w-full gap-0 px-2"):
            ui.label(
                "AI annotations: webcam \u2192 your script \u2192 pyvirtualcam \u2192 select virtual device"
            ).classes("text-xs text-gray-500 dark:text-gray-400")
            ui.label("Linux: sudo apt install v4l2loopback-dkms").classes(
                "text-xs text-gray-500 dark:text-gray-400"
            )

        def _refresh_camera_devices() -> None:
            if self._cam_select:
                new_devices = enumerate_video_devices()
                new_options: dict[int | str, str] = {-1: "Disabled"}
                for dev in new_devices:
                    new_options[dev["index"]] = str(dev["label"])
                # Keep any custom entries the user typed
                for k, v in self._cam_select.options.items():  # ty: ignore[unresolved-attribute]
                    if k not in new_options and k != -1:
                        new_options[k] = v
                self._cam_select.options = new_options
                self._cam_select.update()

        # Auto-refresh disabled — on Windows, ``cv2.VideoCapture(i)`` actually
        # opens each camera device for a moment, which causes user-visible
        # flicker on USB cameras (especially Intel RealSense). The user can
        # click the Settings panel to trigger a one-shot refresh instead.
        self._cam_refresh_timer = ui.timer(
            10.0, _refresh_camera_devices, active=False
        )

        if stored_cam is not None and stored_cam != -1:
            camera_service.start(stored_cam)

    def _build_motion_profile(self, prefs: dict) -> None:
        async def _on_motion_profile_change(e):
            profile = e.value
            try:
                await self.client.select_profile(profile)
            except Exception as exc:
                logger.warning("select_profile(%s) failed: %s", profile, exc)
                ui.notify(f"Profile change failed: {exc}", color="negative")
                return
            ng_app.storage.general["motion_profile"] = profile

        motion_profile_options = {}
        for p in ui_state.active_robot.motion_profiles:
            motion_profile_options[p] = p.replace("_", " ").title()

        with _setting_row("Motion Profile", "Trajectory generation algorithm"):
            ui.select(
                options=motion_profile_options,
                value=prefs["motion_profile"],
                on_change=_on_motion_profile_change,
            ).classes("w-32").props("dense").mark("select-motion-profile")

    def _build_theme(self, prefs: dict) -> None:
        with _setting_row("Theme", "Application color scheme"):
            with ui.element("span").tooltip(
                "Light mode will be available in a future update"
            ):
                ui.select(
                    options={"dark": "Dark"},
                    value="dark",
                ).classes("w-24").props("dense disable")

    def _build_reference_frames(self) -> None:
        with _setting_row("Translation RF", "Reference frame for translation moves"):
            with ui.element("span").tooltip(
                "Mode is currently locked but will be configurable in a future update"
            ):
                ui.select(
                    options={"WRF": "World", "TRF": "Tool"},
                    value="WRF",
                ).classes("w-24").props("dense disable")

        ui.separator().classes("my-1")

        with _setting_row("Rotation RF", "Reference frame for rotation moves"):
            with ui.element("span").tooltip(
                "Mode is currently locked but will be configurable in a future update"
            ):
                ui.select(
                    options={"WRF": "World", "TRF": "Tool"},
                    value="TRF",
                ).classes("w-24").props("dense disable")

    # ── Calibration & motion-safety extras ──────────────────────────
    #
    # Toggles for the parol6-vision additions: the calibration features
    # master switch (mirrored here for discoverability — same value as
    # the toggle in the calibration panel) and the mesh collision check
    # gate. The mesh collision check is independent of the calibration
    # features — it powers ``validate_joint_trajectory`` (a generic
    # motion-safety primitive) and the calibration-time pose filter,
    # but is useful even for users who don't run calibration.

    def _build_calibration_extras(self) -> None:
        ui.label("Calibration & motion safety").classes(
            "text-sm font-medium opacity-80",
        )

        # Master soft-toggle for calibration features. Same storage key
        # the calibration panel writes to.
        cal_initial = bool(
            ng_app.storage.general.get("calibration_features_active", False)
        )

        def _on_calibration_toggle(e) -> None:
            new_value = bool(getattr(e, "value", False))
            ng_app.storage.general["calibration_features_active"] = new_value
            # Live-apply: build or tear down overlays + refresh the
            # calibration panel so its three-state branch picks up
            # the new state. First off→on may take ~1–2 s on disk
            # (custom-tool STL bake); subsequent flips are fast.
            if new_value:
                ui.notify(
                    "Loading calibration features…",
                    color="info", position="top",
                )
            _live_apply_calibration_state()
            if new_value:
                ui.notify(
                    "Calibration features active.",
                    color="positive", position="top",
                )
            else:
                ui.notify(
                    "Calibration features disabled.",
                    color="info", position="top",
                )

        with _setting_row(
            "Calibration features",
            "Calibration overlays + Run / Localise / Hover panel + custom-tools UI",
        ):
            ui.switch(value=cal_initial, on_change=_on_calibration_toggle).props(
                "dense",
            )

        # Mesh collision check gate. Default ON because collision
        # checking is generally useful for motion safety (it powers
        # ``validate_joint_trajectory`` plus the calibration-time
        # pose filter). Users can disable on lower-spec machines or
        # when iterating quickly without safety guards.
        coll_initial = bool(
            ng_app.storage.general.get("mesh_collision_check_enabled", True)
        )

        def _on_collision_toggle(e) -> None:
            new_value = bool(getattr(e, "value", False))
            ng_app.storage.general["mesh_collision_check_enabled"] = new_value
            # Drop the cached collision manager so the next call
            # picks up the new gate state.
            try:
                from waldo_commander.components.calibration_overlays import (  # noqa: PLC0415
                    state as _calib_state,
                )

                _calib_state._state["trajectory_collision_mgr_pair"] = None
            except Exception:  # noqa: BLE001
                pass
            if new_value:
                ui.notify(
                    "Mesh collision check enabled — validate_joint_trajectory "
                    "will run full checks on next call.",
                    color="positive", position="top",
                )
            else:
                ui.notify(
                    "Mesh collision check DISABLED. validate_joint_trajectory "
                    "now returns safe=True without checking. Calibration's "
                    "self-collision filter is bypassed too.",
                    color="warning", position="top",
                )

        with _setting_row(
            "Mesh collision check",
            "Self-collision + workspace check (validate_joint_trajectory)",
        ):
            ui.switch(value=coll_initial, on_change=_on_collision_toggle).props(
                "dense",
            )

    # ── Main entry point ─────────────────────────────────────────────

    def build_embedded(self) -> None:
        """Build the settings content for embedding in control panel."""
        prefs = self._load_preferences()

        sections = [
            lambda: self._build_serial_port(prefs),
            lambda: self._build_show_route(prefs),
            lambda: self._build_envelope(prefs),
            self._build_tool_section,
            self._build_camera,
            lambda: self._build_motion_profile(prefs),
            lambda: self._build_theme(prefs),
            self._build_reference_frames,
            self._build_calibration_extras,
        ]

        for i, section in enumerate(sections):
            section()
            if i < len(sections) - 1:
                ui.separator().classes("my-1")

        simulation_state.notify_changed()
