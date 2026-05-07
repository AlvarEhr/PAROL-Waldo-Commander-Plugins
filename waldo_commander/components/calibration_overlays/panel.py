"""Calibration tab content (UI + thread spawning)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from nicegui import ui

from .calibration_thread import _calibration_thread
from .hover import _drive_hover_pose_thread
from .localise import _localise_board_thread
from .overlays import _set_overlay_visible
from .state import _state

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


def build_calibration_panel_content(close_callback: Callable[[], None] | None = None) -> None:
    """Build the calibration tab's contents.

    The panel hosts both the action buttons (Run / Localise / STOP) that
    drive the calibration + localise threads, and the view-overlay
    checkboxes that toggle visibility of each scene element. Designed to
    live inside a ``ui.tab_panel`` in ``main.py``'s side-tab system, so the
    calibration tooling shares a common UI pattern with Program / I/O /
    Gripper. Pass ``close_callback`` to wire up the panel's close button.
    """

    def _busy_warn(msg: str) -> bool:
        """Reject button press if calibration / localise / hover is running,
        or if the localise-before-Run dialog is open. Returns True if a
        warning was issued."""
        if _state.get("is_running"):
            ui.notify(
                f"{msg}: calibration already running — wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("is_localising"):
            ui.notify(
                f"{msg}: board localise already running — wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("is_hovering"):
            ui.notify(
                f"{msg}: hover move in progress — wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("dialog_open"):
            ui.notify(
                f"{msg}: confirmation dialog open — answer it first",
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
                    "Calibration will use the configured _BOARD_TRANSLATE_M as "
                    "the bootstrap target. If your physical tablet isn't there, "
                    "bootstrap will fail to find the board."
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
        _post_status("Localising board — driving lookout sweep...")
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
            ui.notify("HALTED — robot motion stopped", color="warning")
        _post_status("Stop requested — wait for current move to finish")

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

    with ui.row().classes("w-full items-center"):
        ui.label("Calibration").classes("text-lg font-medium")
        ui.space()
        if close_callback is not None:
            ui.button(icon="close", on_click=close_callback).props(
                "flat round dense color=white"
            )

    _state["status_label"] = ui.label("Idle.").classes(
        "text-xs opacity-80"
    )

    with ui.row().classes("gap-1 q-mt-sm"):
        ui.button("Run", on_click=_on_run, color="primary").props("size=sm")
        ui.button(
            "Localise Board", on_click=_on_localise, color="secondary",
        ).props("size=sm")
        ui.button("STOP", on_click=_on_stop, color="negative").props("size=sm")

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

        from parol6_vision.calibration.board import (  # noqa: PLC0415
            BOARD_TABLET_30MM as _hover_cfg,
        )
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
