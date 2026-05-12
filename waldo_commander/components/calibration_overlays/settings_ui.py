"""UI builders for the calibration settings panel.

Each ``build_*_section`` renders one sub-expansion. Inputs persist via
``settings.set_value`` and trigger live-apply side effects.

Settings store SI internally; inputs convert to mm + degrees for display
via a ``scale`` factor (display = storage × scale).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from nicegui import ui

from . import live_apply, settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


# ArUco dictionaries available in OpenCV ≥4.10, listed by decreasing
# usefulness for ChArUco; 4x4_50 is the tablet-board default.
ARUCO_DICTIONARIES: tuple[str, ...] = (
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
)


JAW_VARIANTS: tuple[str, ...] = ("finger", "pinch")


def _on_setting_change(key: str, value: Any) -> None:
    """Persist + live-apply a single setting. Surfaces a warning when the
    marker/square invariant is violated (``current_board_config`` clamps
    at consumer-read time so the scene keeps rendering).
    """
    try:
        settings.set_value(key, value)
    except Exception as e:  # noqa: BLE001
        logger.warning("settings.set_value(%s, %r) failed: %s", key, value, e)
        return

    # Marker/square invariant warning.
    if key in ("board_square_length_m", "board_marker_length_m"):
        sl = float(settings.get("board_square_length_m"))
        ml = float(settings.get("board_marker_length_m"))
        if ml <= 0.0 or ml >= sl:
            ui.notify(
                f"Marker length ({ml * 1000:.1f} mm) must be smaller than "
                f"square length ({sl * 1000:.1f} mm). Using 60% of square "
                f"({sl * 600:.1f} mm) until you fix it.",
                color="warning", position="top",
            )
        else:
            ratio = ml / sl
            if ratio < 0.4 or ratio > 0.85:
                ui.notify(
                    f"Marker/square ratio {ratio:.2f} outside 0.40-0.85; "
                    f"detection accuracy may suffer.",
                    color="info", position="top",
                )

    try:
        live_apply.apply_setting_change(key)
    except Exception as e:  # noqa: BLE001
        logger.warning("live_apply for %s failed: %s", key, e)


def _number_input(
    key: str,
    label: str,
    *,
    scale: float = 1.0,  # storage * scale = display
    fmt: str = "%.3f",
    step: float = 0.01,
    min_val: float | None = None,
    max_val: float | None = None,
    cast: Callable[[Any], Any] = float,
    width: str = "w-32",
    debounce_ms: int = 500,
) -> ui.number:
    """Single-number input bound to ``settings[key]``. ``debounce_ms`` holds
    back v-model emit so rapid typing doesn't flood live-apply.
    """
    current = float(settings.get(key))
    inp = (
        ui.number(
            label=label, value=current * scale,
            format=fmt, step=step, min=min_val, max=max_val,
        )
        .props(f"dense debounce={debounce_ms}")
        .classes(width)
    )

    def _on_change(_e: Any = None) -> None:
        try:
            display = float(inp.value if inp.value is not None else current * scale)
        except (TypeError, ValueError):
            return
        new_storage = cast(display / scale)
        _on_setting_change(key, new_storage)

    inp.on("update:model-value", _on_change)
    return inp


def _int_input(
    key: str,
    label: str,
    *,
    step: int = 1,
    min_val: int | None = None,
    max_val: int | None = None,
    width: str = "w-24",
) -> ui.number:
    return _number_input(
        key, label, fmt="%d", step=step, min_val=min_val, max_val=max_val,
        cast=int, width=width,
    )


def _tuple_input(
    key: str,
    labels: tuple[str, ...],
    *,
    scale: float = 1.0,
    fmt: str = "%.3f",
    step: float = 0.1,
    min_val: float | None = None,
    max_val: float | None = None,
    width: str = "w-24",
    debounce_ms: int = 500,
) -> list[ui.number]:
    """N number inputs in a row; all persist+apply ``key`` on any change."""
    current = settings.get(key)
    inputs: list[ui.number] = []

    def _on_change(_e: Any = None) -> None:
        try:
            new_display = tuple(
                float(inp.value if inp.value is not None else 0.0)
                for inp in inputs
            )
        except (TypeError, ValueError):
            return
        new_storage = tuple(v / scale for v in new_display)
        _on_setting_change(key, new_storage)

    with ui.row().classes("items-center gap-1 q-gutter-x-sm"):
        for i, label in enumerate(labels):
            initial = (
                float(current[i]) * scale
                if current is not None and i < len(current)
                else 0.0
            )
            inp = (
                ui.number(
                    label=label, value=initial,
                    format=fmt, step=step, min=min_val, max=max_val,
                )
                .props(f"dense debounce={debounce_ms}")
                .classes(width)
            )
            inp.on("update:model-value", _on_change)
            inputs.append(inp)
    return inputs


def _switch_input(key: str, label: str) -> ui.switch:
    """Bool setting as a NiceGUI switch."""
    sw = ui.switch(label, value=bool(settings.get(key))).props("dense")

    def _on_change(_e: Any = None) -> None:
        # Read e.value, not sw.value — Quasar may not have propagated.
        _on_setting_change(key, bool(getattr(_e, "value", False)))

    sw.on("update:model-value", _on_change)
    return sw


def _select_input(
    key: str,
    label: str,
    options: tuple[str, ...],
    *,
    width: str = "w-48",
) -> ui.select:
    """String setting as a dropdown."""
    current = str(settings.get(key))
    sel = ui.select(
        list(options), value=current if current in options else options[0],
        label=label,
    ).props("dense").classes(width)

    def _on_change(_e: Any = None) -> None:
        _on_setting_change(key, str(sel.value))

    sel.on("update:model-value", _on_change)
    return sel


def _optional_xyz_input(
    key: str,
    label_prefix: str,
    *,
    scale: float = 1000.0,
    fmt: str = "%.1f",
) -> None:
    """Optional XYZ setting: toggle + X/Y/Z fields. Used for
    ``hemi_centre_override_m``.
    """
    current = settings.get(key)
    enabled_initial = current is not None
    sw = ui.switch("Override (use custom centre)", value=enabled_initial).props(
        "dense"
    )

    inputs: list[ui.number] = []

    def _on_xyz_change(_e: Any = None, *, force_enabled: bool = False) -> None:
        # ``force_enabled`` bypasses the ``sw.value`` gate during the
        # OFF->ON transition (Quasar may not have propagated yet).
        if not force_enabled and not bool(sw.value):
            return
        try:
            new_display = tuple(
                float(inp.value if inp.value is not None else 0.0)
                for inp in inputs
            )
        except (TypeError, ValueError):
            return
        new_storage = tuple(v / scale for v in new_display)
        _on_setting_change(key, new_storage)

    def _on_switch(_e: Any = None) -> None:
        # Read e.value — see ``_switch_input`` for the rationale.
        new_value = bool(getattr(_e, "value", False))
        if new_value:
            # Persist whatever's in the inputs as the initial value;
            # ``force_enabled`` bypasses the not-yet-propagated sw.value.
            _on_xyz_change(force_enabled=True)
            for inp in inputs:
                inp.set_enabled(True)
        else:
            _on_setting_change(key, None)
            for inp in inputs:
                inp.set_enabled(False)

    sw.on("update:model-value", _on_switch)

    with ui.row().classes("items-center gap-1 q-mt-xs"):
        seed = current if current is not None else (0.30, 0.0, 0.014)
        for i, axis in enumerate("XYZ"):
            inp = (
                ui.number(
                    label=f"{label_prefix} {axis} (mm)",
                    value=float(seed[i]) * scale,
                    format=fmt, step=1.0,
                )
                .props("dense debounce=500")
                .classes("w-24")
            )
            inp.set_enabled(enabled_initial)
            inp.on("update:model-value", _on_xyz_change)
            inputs.append(inp)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _render_charuco_png_dialog() -> None:
    """Render the current board to PNG at user-chosen DPI, then download.
    Converts DPI → pixels-per-metre via ``ppi / 0.0254``.
    """
    try:
        from .state import current_board_config  # noqa: PLC0415
        from parol6_vision.calibration.board import render_board_png  # noqa: PLC0415
    except ImportError as e:
        ui.notify(f"ChArUco render unavailable: {e}", color="warning")
        return

    cfg = current_board_config()
    physical_w_mm, physical_h_mm = cfg.physical_size_mm

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
        ui.label("Generate ChArUco board PNG").classes(
            "text-base font-semibold",
        )
        ui.label(
            f"Board: {cfg.squares_x}×{cfg.squares_y} squares, "
            f"{cfg.square_length * 1000:.1f} mm each. "
            f"Physical size: {physical_w_mm:.0f} × {physical_h_mm:.0f} mm.",
        ).classes("text-xs opacity-70")
        ppi_input = ui.number(
            label="Display / printer DPI", value=240.0,
            step=1.0, min=50.0, max=2400.0, format="%.0f",
        ).props("dense")
        margin_input = ui.number(
            label="Margin (squares)", value=0.5,
            step=0.1, min=0.0, max=2.0, format="%.1f",
        ).props("dense")
        info_label = ui.label("").classes("text-xs opacity-70")

        def _update_info(_e: Any = None) -> None:
            try:
                ppi = float(ppi_input.value or 240.0)
                margin_sq = float(margin_input.value or 0.5)
            except (TypeError, ValueError):
                return
            ppm = ppi / 0.0254
            margin_m = margin_sq * cfg.square_length
            full_w = (physical_w_mm / 1000.0 + 2 * margin_m) * ppm
            full_h = (physical_h_mm / 1000.0 + 2 * margin_m) * ppm
            info_label.text = (
                f"Image: {int(round(full_w))} × {int(round(full_h))} px "
                f"(physical {physical_w_mm + 2 * margin_m * 1000:.0f} × "
                f"{physical_h_mm + 2 * margin_m * 1000:.0f} mm at {ppi:.0f} DPI)."
            )
        ppi_input.on("update:model-value", _update_info)
        margin_input.on("update:model-value", _update_info)
        _update_info()

        def _on_render() -> None:
            try:
                ppi = float(ppi_input.value or 240.0)
                margin_sq = float(margin_input.value or 0.5)
            except (TypeError, ValueError):
                ui.notify("Invalid DPI / margin", color="warning")
                return
            try:
                import cv2  # noqa: PLC0415
                ppm = ppi / 0.0254
                img = render_board_png(
                    cfg, pixels_per_metre=ppm, margin_squares=margin_sq,
                )
                ok, encoded = cv2.imencode(".png", img)
                if not ok:
                    ui.notify("PNG encode failed", color="warning")
                    return
                payload = bytes(encoded)
            except Exception as e:  # noqa: BLE001
                ui.notify(f"Render failed: {e}", color="warning")
                return
            filename = (
                f"charuco_{cfg.squares_x}x{cfg.squares_y}_"
                f"{int(cfg.square_length * 1000)}mm_{int(ppi)}dpi.png"
            )
            ui.download(payload, filename)
            ui.notify(f"Generated {filename}", color="positive")
            dialog.close()

        with ui.row():
            ui.button(
                "Render & download", on_click=_on_render, color="primary",
            ).props("size=sm")
            ui.button("Cancel", on_click=dialog.close).props("size=sm")
    dialog.open()


def build_board_section() -> None:
    """Calibration board: placement + geometry + dictionary."""
    ui.label("Placement (board → base)").classes("text-xs opacity-70")
    _tuple_input(
        "board_translate_m",
        ("X (mm)", "Y (mm)", "Z (mm)"),
        scale=1000.0, fmt="%.1f", step=1.0,
    )
    rpy_inputs = _tuple_input(
        "board_rpy_rad",
        ("Rx (deg)", "Ry (deg)", "Rz (deg)"),
        scale=180.0 / math.pi, fmt="%.2f", step=0.5,
    )
    ui.label("Rotation order: scipy XYZ-extrinsic.").classes(
        "text-xs opacity-60",
    )
    del rpy_inputs

    ui.separator().classes("q-my-sm")
    ui.label("Geometry").classes("text-xs opacity-70")
    with ui.row().classes("items-center gap-1"):
        _int_input("board_squares_x", "Squares X", step=1, min_val=3)
        _int_input("board_squares_y", "Squares Y", step=1, min_val=3)
    with ui.row().classes("items-center gap-1"):
        _number_input(
            "board_square_length_m", "Square length (mm)",
            scale=1000.0, fmt="%.2f", step=0.5, min_val=1.0,
        )
        _number_input(
            "board_marker_length_m", "Marker length (mm)",
            scale=1000.0, fmt="%.2f", step=0.5, min_val=1.0,
        )
    _select_input("board_dictionary", "ArUco dictionary", ARUCO_DICTIONARIES)
    _switch_input("board_legacy_pattern", "Legacy ChArUco corner ordering")
    ui.separator().classes("q-my-sm")
    ui.button(
        "Generate board PNG...", icon="download",
        on_click=_render_charuco_png_dialog,
    ).props("size=sm outline")
    ui.label(
        "Renders the current geometry at chosen DPI for print or tablet display.",
    ).classes("text-xs opacity-60")


def build_surface_section() -> None:
    """Mounting surface: collision primitive + safety margin."""
    _switch_input(
        "surface_enabled",
        "Enable mounting surface (collision primitive)",
    )
    _switch_input(
        "surface_show_overlay",
        "Show translucent surface overlay in scene",
    )
    ui.label("Dimensions (board-local W × H × thickness)").classes(
        "text-xs opacity-70 q-mt-sm",
    )
    _tuple_input(
        "surface_dimensions_m",
        ("W (mm)", "H (mm)", "Thickness (mm)"),
        scale=1000.0, fmt="%.1f", step=1.0, min_val=1.0,
    )
    ui.label("Offset of surface centre from ChArUco centre (board-local)").classes(
        "text-xs opacity-70 q-mt-sm",
    )
    _tuple_input(
        "surface_offset_local_m",
        ("X offset (mm)", "Y offset (mm)"),
        scale=1000.0, fmt="%.1f", step=1.0,
    )
    ui.separator().classes("q-my-sm")
    ui.label("Collision behaviour").classes("text-xs opacity-70")
    _switch_input("floor_primitive_enabled", "Floor collision box (z<0)")
    _switch_input("enable_self_collision_check", "Self-collision check")
    _number_input(
        "collision_safety_margin_m", "Safety margin (mm)",
        scale=1000.0, fmt="%.1f", step=0.5, min_val=0.0,
    )
    ui.label(
        "Lower margin lets the robot approach closer, with higher collision risk.",
    ).classes("text-xs opacity-60")


def build_intrinsics_section() -> None:
    """Camera intrinsics."""
    ui.label("Pinhole intrinsics (px)").classes("text-xs opacity-70")
    with ui.row().classes("items-center gap-1"):
        _number_input("intr_fx", "fx", fmt="%.1f", step=1.0, min_val=1.0)
        _number_input("intr_fy", "fy", fmt="%.1f", step=1.0, min_val=1.0)
    with ui.row().classes("items-center gap-1"):
        _number_input("intr_cx", "cx", fmt="%.1f", step=1.0, min_val=0.0)
        _number_input("intr_cy", "cy", fmt="%.1f", step=1.0, min_val=0.0)
    ui.label("Image size").classes("text-xs opacity-70 q-mt-sm")
    with ui.row().classes("items-center gap-1"):
        _int_input("intr_width", "Width (px)", step=1, min_val=1)
        _int_input("intr_height", "Height (px)", step=1, min_val=1)


def build_cam_mount_section() -> None:
    """Cold-start camera mount transform."""
    ui.label("Translation (flange frame)").classes("text-xs opacity-70")
    _tuple_input(
        "cam_mount_translate_mm",
        ("X (mm)", "Y (mm)", "Z (mm)"),
        fmt="%.1f", step=0.5,
    )
    ui.label("Tilt (XYZ-extrinsic)").classes("text-xs opacity-70 q-mt-sm")
    _tuple_input(
        "cam_mount_tilt_deg",
        ("Rx (deg)", "Ry (deg)", "Rz (deg)"),
        fmt="%.2f", step=1.0,
    )
    ui.label(
        "Seed values for IK warm-starts and the live frustum.",
    ).classes("text-xs opacity-60")


# ---------------------------------------------------------------------------
# Per-tool camera config (active tool override)
# ---------------------------------------------------------------------------


def _per_tool_number_input(
    tool_key: str,
    setting_key: str,
    label: str,
    *,
    scale: float = 1.0,
    fmt: str = "%.3f",
    step: float = 0.01,
    cast: Callable[[Any], Any] = float,
    width: str = "w-32",
) -> ui.number:
    """Number input bound to the per-tool override for ``tool_key``.
    Initial value: override if set, else the global. Edits auto-write.
    """
    from . import custom_tools  # noqa: PLC0415

    override = custom_tools.get_per_tool_override(tool_key, setting_key)
    if override is None:
        current = float(settings.get(setting_key))
    else:
        current = float(override)
    def _on_change(e: Any = None) -> None:
        # Read e.value over inp.value to avoid the post-debounce race.
        raw = getattr(e, "value", None)
        if raw is None:
            raw = inp.value if inp is not None else current * scale
        try:
            display = float(raw if raw is not None else current * scale)
        except (TypeError, ValueError):
            return
        new_storage = cast(display / scale)
        custom_tools.set_per_tool_override(tool_key, setting_key, new_storage)
        # Live-apply: rebuild mount / redraw frustum.
        try:
            from . import live_apply  # noqa: PLC0415

            live_apply.apply_setting_change(setting_key)
        except Exception:  # noqa: BLE001
            pass

    inp = (
        ui.number(
            label=label, value=current * scale,
            format=fmt, step=step, on_change=_on_change,
        )
        .props("dense debounce=500")
        .classes(width)
    )
    return inp


def _per_tool_tuple_input(
    tool_key: str,
    setting_key: str,
    labels: tuple[str, ...],
    *,
    scale: float = 1.0,
    fmt: str = "%.3f",
    step: float = 0.1,
    width: str = "w-24",
) -> list[ui.number]:
    from . import custom_tools  # noqa: PLC0415

    override = custom_tools.get_per_tool_override(tool_key, setting_key)
    if override is None:
        current = settings.get(setting_key)
    else:
        current = override
    inputs: list[ui.number] = []

    def _on_change(_e: Any = None) -> None:
        # Bound via ``on_change=`` kwarg to avoid the post-debounce race.
        try:
            new_display = tuple(
                float(inp.value if inp.value is not None else 0.0)
                for inp in inputs
            )
        except (TypeError, ValueError):
            return
        new_storage = tuple(v / scale for v in new_display)
        custom_tools.set_per_tool_override(tool_key, setting_key, new_storage)
        # Live-apply: rebuild mount / redraw frustum.
        try:
            from . import live_apply  # noqa: PLC0415

            live_apply.apply_setting_change(setting_key)
        except Exception:  # noqa: BLE001
            pass

    with ui.row().classes("items-center gap-1 q-gutter-x-sm"):
        for i, label in enumerate(labels):
            initial = (
                float(current[i]) * scale
                if current is not None and i < len(current)
                else 0.0
            )
            inp = (
                ui.number(
                    label=label, value=initial,
                    format=fmt, step=step,
                    on_change=_on_change,
                )
                .props("dense debounce=500")
                .classes(width)
            )
            inputs.append(inp)
    return inputs


def build_per_tool_camera_section() -> None:
    """Per-tool camera intrinsics + cold-start mount override for the
    active tool. Writes to ``app.storage.general[calib_tool_<key>_*]`` so
    each camera-bearing tool carries its own values (installation-global
    scope — see ``custom_tools._per_tool_storage_dict`` for rationale).
    """
    from . import custom_tools  # noqa: PLC0415

    @ui.refreshable
    def _content() -> None:
        active = custom_tools._active_gui_tool_key()
        if not active or active == "NONE":
            ui.label(
                "No tool active. Pick one in the gripper panel.",
            ).classes("text-xs opacity-70")
            return
        if not custom_tools.is_camera_bearing(active):
            ui.label(
                f"Active tool {active!r} isn't flagged as camera-bearing.",
            ).classes("text-xs opacity-70")
            return

        ui.label(f"Active tool: {active}").classes("text-xs opacity-70")
        has_any = custom_tools.has_per_tool_override(active)
        ui.label(
            "Per-tool override active."
            if has_any
            else "No per-tool override yet. Editing any field saves one.",
        ).classes(
            "text-xs " + ("text-green-7" if has_any else "opacity-60"),
        )

        ui.separator().classes("q-my-sm")
        ui.label("Camera intrinsics (px)").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-1"):
            _per_tool_number_input(
                active, "intr_fx", "fx", fmt="%.1f", step=1.0,
            )
            _per_tool_number_input(
                active, "intr_fy", "fy", fmt="%.1f", step=1.0,
            )
        with ui.row().classes("items-center gap-1"):
            _per_tool_number_input(
                active, "intr_cx", "cx", fmt="%.1f", step=1.0,
            )
            _per_tool_number_input(
                active, "intr_cy", "cy", fmt="%.1f", step=1.0,
            )
        with ui.row().classes("items-center gap-1"):
            _per_tool_number_input(
                active, "intr_width", "Width (px)", fmt="%d",
                step=1, cast=int, width="w-28",
            )
            _per_tool_number_input(
                active, "intr_height", "Height (px)", fmt="%d",
                step=1, cast=int, width="w-28",
            )

        ui.separator().classes("q-my-sm")
        ui.label("Camera mount (cold-start)").classes("text-xs opacity-70")
        ui.label("Translation (flange frame)").classes(
            "text-xs opacity-70 q-mt-xs",
        )
        _per_tool_tuple_input(
            active, "cam_mount_translate_mm",
            ("X (mm)", "Y (mm)", "Z (mm)"),
            fmt="%.1f", step=0.5,
        )
        ui.label("Tilt (XYZ-extrinsic)").classes(
            "text-xs opacity-70 q-mt-xs",
        )
        _per_tool_tuple_input(
            active, "cam_mount_tilt_deg",
            ("Rx (deg)", "Ry (deg)", "Rz (deg)"),
            fmt="%.2f", step=1.0,
        )

        ui.separator().classes("q-my-sm")

        def _reset_overrides() -> None:
            custom_tools.clear_per_tool_overrides(active)
            ui.notify(
                f"Cleared all per-tool overrides for {active}.",
                color="info", position="top",
            )
            _content.refresh()

        ui.button(
            "Reset all to globals", on_click=_reset_overrides, icon="restart_alt",
        ).props("size=sm outline color=warning")

    _content()


def build_hemisphere_section() -> None:
    """Hemisphere search region."""
    ui.label("Distance shell (m)").classes("text-xs opacity-70")
    _tuple_input(
        "hemi_distance_range_m",
        ("Min (m)", "Max (m)"),
        fmt="%.3f", step=0.01, min_val=0.05,
    )
    ui.label("Elevation range (deg)").classes("text-xs opacity-70 q-mt-sm")
    _tuple_input(
        "hemi_elevation_range_deg",
        ("Min (deg)", "Max (deg)"),
        fmt="%.1f", step=1.0, min_val=0.0, max_val=89.0,
    )
    _number_input(
        "hemi_azimuth_spread_deg", "Azimuth half-spread (deg)",
        fmt="%.1f", step=5.0, min_val=0.0, max_val=180.0,
    )
    ui.separator().classes("q-my-sm")
    ui.label("Hemisphere centre override").classes("text-xs opacity-70")
    _optional_xyz_input("hemi_centre_override_m", "Centre")


def build_reachability_section() -> None:
    """Reachability dot count + size. Count changes re-run the IK sweep;
    radius changes only re-run the non-overlap selection.
    """
    ui.label("Sphere radius (mm)").classes("text-xs opacity-70")
    _number_input(
        "reachability_dot_radius_m", "Radius",
        scale=1000.0, fmt="%.1f", step=0.5, min_val=0.5, max_val=20.0,
    )
    ui.label("Sweep candidate count").classes("text-xs opacity-70 q-mt-sm")
    _int_input(
        "reachability_n_candidates", "Candidates",
        step=8, min_val=8, max_val=4096,
    )
    ui.label(
        "More candidates means a denser pre-filter pool; the non-overlap "
        "selector keeps only spread-out dots so the rendered count is "
        "usually lower.",
    ).classes("text-xs opacity-60 q-mt-xs")

    # Refreshed via ``_state['reachability_info_refresh']`` after each sweep.
    @ui.refreshable
    def _info_label() -> None:
        from .state import _state as _s  # noqa: PLC0415

        all_pts = _s.get("reachable_points_all")
        candidates_visible = _s.get("reachable_candidates") or []
        drawn = len(candidates_visible)
        try:
            target = int(settings.get("reachability_n_candidates"))
        except Exception:  # noqa: BLE001
            target = drawn
        if all_pts is None:
            ui.label("Sweep pending...").classes(
                "text-xs opacity-60 q-mt-xs",
            )
            return
        total = len(all_pts)
        if total == 0:
            ui.label(
                "No reachable poses found. Try lowering the sweep "
                "elevation range or expanding the workspace.",
            ).classes("text-xs text-amber-7 q-mt-xs")
            return
        # We hit the user's target count. No need to nag about overlap.
        if drawn >= target:
            ui.label(
                f"Drawing {drawn} reachable poses.",
            ).classes("text-xs opacity-70 q-mt-xs")
            return
        # Drew fewer than the target: explain whether the bottleneck
        # was reachability (not enough valid IK solutions) or overlap
        # (radius too large to fit the rest without intersecting).
        if drawn >= total:
            ui.label(
                f"Only {total} reachable poses found "
                f"(target was {target}).",
            ).classes("text-xs opacity-70 q-mt-xs")
        else:
            ui.label(
                f"Drawing {drawn} non-overlapping of {total} reachable "
                f"poses. Shrink the sphere radius to see more.",
            ).classes("text-xs opacity-70 q-mt-xs")

    _info_label()
    # Register the refresh callback for reachability.py to invoke after
    # each sweep. Idempotent; stale callbacks don't crash the render.
    from .state import _state as _s_module  # noqa: PLC0415

    _s_module["reachability_info_refresh"] = _info_label.refresh


def build_localise_section() -> None:
    """Localise sweep tunables."""
    ui.label("Sweep mode").classes("text-xs opacity-70")
    _switch_input("localise_use_j0_sweep", "J0-sweep mode (recommended)")
    _switch_input("localise_continuous_sweep", "Continuous (vs discrete steps)")

    ui.label("Sweep parameters").classes("text-xs opacity-70 q-mt-sm")
    with ui.row().classes("items-center gap-1"):
        _number_input(
            "localise_sweep_speed", "Speed (frac)",
            fmt="%.2f", step=0.05, min_val=0.01, max_val=1.0,
        )
        _number_input(
            "localise_capture_period_s", "Capture period (s)",
            fmt="%.2f", step=0.05, min_val=0.05,
        )
    with ui.row().classes("items-center gap-1"):
        _number_input(
            "localise_j0_sweep_half_deg", "J0 half-sweep (deg)",
            fmt="%.1f", step=5.0, min_val=10.0,
        )
        _number_input(
            "localise_j0_chunk_deg", "J0 chunk (deg)",
            fmt="%.1f", step=1.0, min_val=1.0,
        )
    _int_input(
        "localise_seed_n_candidates", "Seed-pose Sobol candidates",
        step=32, min_val=16,
    )

    ui.label("Detection thresholds").classes("text-xs opacity-70 q-mt-sm")
    with ui.row().classes("items-center gap-1"):
        _int_input(
            "localise_early_stop_detections", "Early-stop detections", min_val=1,
        )
        _int_input(
            "localise_early_stop_inliers", "Early-stop inliers", min_val=1,
        )
    with ui.row().classes("items-center gap-1"):
        _int_input(
            "localise_min_inliers_to_proceed", "Min inliers to proceed", min_val=1,
        )
        _int_input(
            "localise_min_detections", "Min detections", min_val=1,
        )
    _number_input(
        "localise_inlier_threshold_m", "Inlier threshold (mm)",
        scale=1000.0, fmt="%.1f", step=1.0, min_val=1.0,
    )

    ui.label("Stage-2 refinement").classes("text-xs opacity-70 q-mt-sm")
    with ui.row().classes("items-center gap-1"):
        _int_input(
            "localise_refine_n_poses", "Refine poses", step=1, min_val=0,
        )
        _number_input(
            "localise_refine_distance_m", "Refine distance (m)",
            fmt="%.2f", step=0.02, min_val=0.10,
        )
        _number_input(
            "localise_refine_elevation_deg", "Refine elevation (deg)",
            fmt="%.1f", step=1.0, min_val=20.0, max_val=89.0,
        )


def build_gripper_section() -> None:
    """Gripper notice; tool selection lives in the bottom-right panel."""
    ui.label(
        "Tool and jaw variant are configured via the bottom-right gripper panel.",
    ).classes("text-xs opacity-70")


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_calibration_settings_expansion() -> None:
    """Render the "Calibration settings" expansion with all sub-sections."""
    with ui.expansion("Calibration settings", icon="tune").classes("w-full"):
        with ui.expansion("Calibration board", icon="grid_4x4").classes("w-full"):
            build_board_section()
        with ui.expansion("Mounting surface", icon="layers").classes("w-full"):
            build_surface_section()
        with ui.expansion(
            "Camera config (active tool override)",
            icon="precision_manufacturing",
        ).classes("w-full"):
            build_per_tool_camera_section()
        with ui.expansion("Camera intrinsics (global default)", icon="camera").classes("w-full"):
            build_intrinsics_section()
        with ui.expansion(
            "Camera mount (global default)", icon="open_with",
        ).classes("w-full"):
            build_cam_mount_section()
        with ui.expansion("Hemisphere search", icon="motion_photos_on").classes(
            "w-full",
        ):
            build_hemisphere_section()
        with ui.expansion("Reachability dots", icon="grain").classes("w-full"):
            build_reachability_section()
        with ui.expansion("Localise sweep", icon="search").classes("w-full"):
            build_localise_section()
        with ui.expansion("Gripper", icon="precision_manufacturing").classes(
            "w-full",
        ):
            build_gripper_section()


# ---------------------------------------------------------------------------
# Preset bar
# ---------------------------------------------------------------------------


def build_preset_bar(refresh_panel: Callable[[], None]) -> None:
    """Preset dropdown + Save / Save as / Delete / Reset / Export.
    ``refresh_panel`` rebuilds inputs after load / reset.
    """
    presets = settings.list_presets()
    active = settings.get_active_preset()

    options = ["(Defaults)"] + list(presets)
    initial = active if active in presets else "(Defaults)"

    with ui.row().classes("items-center gap-1 q-mt-xs"):
        sel = ui.select(options, value=initial, label="Setup preset").props(
            "dense",
        ).classes("w-56")

        def _on_select(_e: Any = None) -> None:
            choice = str(sel.value or "(Defaults)")
            if choice == "(Defaults)":
                settings.reset_to_defaults()
                # Re-run side effects so the scene rebuilds.
                for key in settings.DEFAULTS:
                    try:
                        live_apply.apply_setting_change(key)
                    except Exception:  # noqa: BLE001
                        pass
                refresh_panel()
                ui.notify("Settings reset to defaults", color="info")
                return
            try:
                settings.load_preset(choice)
            except KeyError:
                ui.notify(f"Preset {choice!r} not found", color="warning")
                return
            for key in settings.DEFAULTS:
                try:
                    live_apply.apply_setting_change(key)
                except Exception:  # noqa: BLE001
                    pass
            refresh_panel()
            ui.notify(f"Loaded preset: {choice}", color="positive")

        sel.on("update:model-value", _on_select)

        def _on_save_as() -> None:
            with ui.dialog() as dialog, ui.card():
                ui.label("Save current settings as preset").classes(
                    "text-base font-semibold",
                )
                name_input = ui.input("Preset name").props("dense autofocus")
                with ui.row():
                    def _confirm() -> None:
                        name = str(name_input.value or "").strip()
                        if not name:
                            ui.notify("Name cannot be empty", color="warning")
                            return
                        settings.save_preset(name)
                        ui.notify(f"Saved preset: {name}", color="positive")
                        dialog.close()
                        refresh_panel()
                    ui.button("Save", on_click=_confirm, color="primary").props(
                        "size=sm",
                    )
                    ui.button("Cancel", on_click=dialog.close).props("size=sm")
            dialog.open()

        def _on_delete() -> None:
            choice = str(sel.value or "(Defaults)")
            if choice == "(Defaults)":
                ui.notify("Cannot delete defaults", color="warning")
                return
            settings.delete_preset(choice)
            ui.notify(f"Deleted preset: {choice}", color="info")
            refresh_panel()

        def _on_export() -> None:
            choice = str(sel.value or "(Defaults)")
            try:
                payload = settings.export_preset_json(
                    None if choice == "(Defaults)" else choice,
                )
            except KeyError:
                ui.notify(f"Preset {choice!r} not found", color="warning")
                return
            # Show in a dialog for copy-paste.
            with ui.dialog() as dialog, ui.card().classes("w-full max-w-2xl"):
                ui.label(f"Preset JSON ({choice})").classes(
                    "text-base font-semibold",
                )
                ui.textarea(value=payload).props(
                    "outlined readonly dense",
                ).classes("w-full font-mono text-xs").style("min-height: 300px")
                ui.button("Close", on_click=dialog.close).props("size=sm")
            dialog.open()

        ui.button(
            "Save as...", on_click=_on_save_as, color="primary",
        ).props("size=sm outline")
        ui.button("Delete", on_click=_on_delete, color="negative").props(
            "size=sm outline",
        )
        ui.button("Export JSON", on_click=_on_export).props("size=sm outline")
