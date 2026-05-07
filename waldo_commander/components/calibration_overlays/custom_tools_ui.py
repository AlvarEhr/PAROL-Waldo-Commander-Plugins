"""UI for the calibration panel's "Custom tools" expansion.

Lists every tool found under ``~/.waldo-commander/custom_tools/`` and
provides per-tool transform inputs (translate, RPY, scale), STL upload
slots (body + optional jaws), TCP transform inputs, and snap helpers
(bbox centre, flange-face plane align, hole-detection).

Each transform edit:
1. updates the on-disk ``config.json``,
2. rebakes the STL into parol6's mesh dir with the new transform,
3. notifies the user that switching the active tool in the gripper
   panel (or restarting waldo-commander) will pick up the new mesh.

Live-updating the URDF scene's already-loaded mesh in place is non-
trivial — it requires touching waldo-commander's UrdfScene internals,
which is out of scope for Phase 1B. For iteration, the workflow is
"edit → save → switch the gripper dropdown back-and-forth → see the
update". We surface this in a notice on every save.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import events, ui

from . import custom_tools

logger = logging.getLogger(__name__)


def _open_folder_in_os(path: Path) -> bool:
    """Open ``path`` in the OS's native file browser. NiceGUI server-side
    runs on the user's machine (waldo-commander is a local app, not
    hosted), so this opens the folder on the user's desktop. Returns
    True on success.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606 — local app, not server
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])  # noqa: S603, S607
        else:
            subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("could not open folder %s: %s", path, e)
        return False


# ---------------------------------------------------------------------------
# Per-tool editor
# ---------------------------------------------------------------------------


def _tuple3_inputs(
    label_prefix: str,
    initial: tuple[float, float, float],
    *,
    scale: float = 1.0,
    fmt: str = "%.3f",
    step: float = 0.1,
    on_change: Callable[[tuple[float, float, float]], None] | None = None,
) -> list[ui.number]:
    """Three numeric inputs in a row, scaled for display (storage × scale)."""
    inputs: list[ui.number] = []

    def _emit(_e: Any = None) -> None:
        if on_change is None:
            return
        try:
            new_display = tuple(
                float(inp.value if inp.value is not None else 0.0)
                for inp in inputs
            )
        except (TypeError, ValueError):
            return
        new_storage = tuple(v / scale for v in new_display)
        on_change(new_storage)  # type: ignore[arg-type]

    with ui.row().classes("items-center gap-1 q-gutter-x-sm"):
        for i, axis in enumerate("XYZ"):
            inp = (
                ui.number(
                    label=f"{label_prefix} {axis}",
                    value=float(initial[i]) * scale,
                    format=fmt, step=step,
                )
                .props("dense debounce=500")
                .classes("w-24")
            )
            inp.on("update:model-value", _emit)
            inputs.append(inp)
    return inputs


def _build_tool_card(cfg: custom_tools.CustomToolConfig, refresh: Callable[[], None]) -> None:
    """Render the editor for one custom tool. ``refresh`` rebuilds the
    list view after a structural change (delete, rename, etc.).
    """
    is_active = custom_tools.is_active_tool(cfg.name)

    with ui.card().classes("w-full q-mt-sm"):
        with ui.row().classes("w-full items-center"):
            ui.label(cfg.display_name or cfg.name).classes("text-base font-medium")
            ui.label(f"({cfg.name})").classes("text-xs opacity-60")
            if is_active:
                ui.badge("ACTIVE", color="positive").props("outline")
            ui.space()
            ui.label(
                "body ✓" if cfg.has_body else "body ✗"
            ).classes(
                "text-xs " + ("text-green-7" if cfg.has_body else "text-red-7"),
            )
            ui.label(
                "jaws ✓" if cfg.has_jaws else "no jaws"
            ).classes(
                "text-xs " + ("text-green-7" if cfg.has_jaws else "opacity-60"),
            )

            async def _on_use_this(_e: Any = None) -> None:
                if not cfg.has_body:
                    ui.notify("Upload a body STL first", color="warning")
                    return
                # Bake first — registry mutation should be in place before
                # the controller asks for this tool's mesh path.
                custom_tools.register_one(cfg)
                ok = await custom_tools.select_as_active(cfg.name)
                if ok:
                    ui.notify(
                        f"Switched active tool to custom:{cfg.name}",
                        color="positive",
                    )
                    refresh()
                else:
                    ui.notify(
                        f"Could not switch tools — controller may be "
                        f"disconnected. Use the gripper panel manually.",
                        color="warning",
                    )

            if not is_active:
                ui.button(
                    "Use this tool", on_click=_on_use_this, icon="check_circle",
                ).props("size=sm outline color=primary")

            def _on_delete() -> None:
                with ui.dialog() as confirm, ui.card():
                    ui.label(f"Delete custom tool {cfg.name!r}?").classes(
                        "text-base font-semibold",
                    )
                    ui.label(
                        "Removes the folder, config, and all baked STLs. "
                        "Restart waldo-commander to fully unregister.",
                    ).classes("text-xs opacity-70")
                    with ui.row():
                        def _confirm() -> None:
                            custom_tools.delete_tool(cfg.name)
                            ui.notify(f"Deleted {cfg.name}", color="info")
                            confirm.close()
                            refresh()
                        ui.button("Delete", on_click=_confirm, color="negative").props(
                            "size=sm",
                        )
                        ui.button("Cancel", on_click=confirm.close).props("size=sm")
                confirm.open()

            ui.button(icon="delete", on_click=_on_delete).props(
                "flat round dense color=negative size=sm",
            )

        if cfg.description:
            ui.label(cfg.description).classes("text-xs opacity-70")

        # ----- STL slots (upload / re-upload body + jaws) -----
        ui.label("STL files").classes("text-xs opacity-70 q-mt-sm")

        def _make_upload_handler(role: str):
            def _handle(e: events.UploadEventArguments) -> None:
                try:
                    suffix = Path(e.name).suffix or ".stl"
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False,
                    ) as tmp:
                        tmp.write(e.content.read())
                        tmp_path = Path(tmp.name)
                    custom_tools.import_stl(cfg.name, role, tmp_path)
                    tmp_path.unlink(missing_ok=True)
                    ui.notify(
                        f"Uploaded {role} STL ({e.name})",
                        color="positive",
                    )
                    # Auto-detect units on first body upload — just
                    # report; user decides whether to apply.
                    if role == "body":
                        scale, label = custom_tools.detect_mesh_unit_scale(
                            cfg.body_path,
                        )
                        ui.notify(
                            f"Body STL: detected units = {label}. "
                            f"Suggested mesh_scale = {scale}.",
                            color="info", position="top",
                        )
                    refresh()
                except Exception as ex:  # noqa: BLE001
                    ui.notify(f"Upload failed: {ex}", color="warning")
            return _handle

        with ui.row().classes("items-center gap-2"):
            ui.upload(
                label="Body STL",
                on_upload=_make_upload_handler("body"),
                auto_upload=True,
                max_files=1,
            ).props("dense accept=.stl").classes("w-48")
            ui.upload(
                label="Jaw left",
                on_upload=_make_upload_handler("jaw_left"),
                auto_upload=True,
                max_files=1,
            ).props("dense accept=.stl").classes("w-48")
            ui.upload(
                label="Jaw right",
                on_upload=_make_upload_handler("jaw_right"),
                auto_upload=True,
                max_files=1,
            ).props("dense accept=.stl").classes("w-48")

        # ----- Mesh placement transform -----
        ui.separator().classes("q-my-sm")
        ui.label("Mesh placement (flange frame)").classes("text-xs opacity-70")

        def _save_config() -> None:
            custom_tools.save_config(cfg)

        def _save_and_rebake() -> None:
            """Save config + rebake STL + reregister in tool registry +
            (when this tool is active) live-refresh the URDF scene so
            the user sees the change immediately. No-op refresh when
            this isn't the active tool — the user sees the update once
            they switch to this tool via the gripper panel or the
            "Use this tool" button.
            """
            custom_tools.save_config(cfg)
            ok = custom_tools.register_one(cfg)
            if not ok:
                ui.notify(
                    "Re-bake failed — check logs (probably no body.stl yet).",
                    color="warning",
                )
                return
            refreshed = custom_tools.live_refresh_active_tool(cfg.name)
            if refreshed:
                # Active tool — silent live update; no notify needed.
                # Each transform-input keystroke shouldn't pop a toast.
                pass
            else:
                # User is editing a non-active tool. Quiet info-level
                # toast so they remember to switch when they're done.
                ui.notify(
                    f"Re-baked custom:{cfg.name}. Click \"Use this tool\" "
                    f"or pick it in the gripper panel to view the change.",
                    color="info", position="top", timeout=2000,
                )

        def _on_translate(value: tuple[float, float, float]) -> None:
            cfg.mesh_translate_m = value
            _save_and_rebake()

        def _on_rpy(value: tuple[float, float, float]) -> None:
            cfg.mesh_rpy_rad = value
            _save_and_rebake()

        translate_inputs = _tuple3_inputs(
            "Translate (mm)",
            cfg.mesh_translate_m, scale=1000.0, fmt="%.1f", step=1.0,
            on_change=_on_translate,
        )
        rpy_inputs = _tuple3_inputs(
            "Rotate (deg)",
            cfg.mesh_rpy_rad, scale=180.0 / math.pi, fmt="%.2f", step=1.0,
            on_change=_on_rpy,
        )

        # Scale row
        with ui.row().classes("items-center gap-2"):
            scale_input = (
                ui.number(
                    label="Scale (× input units → metres)",
                    value=float(cfg.mesh_scale),
                    format="%.6f",
                    step=0.001,
                    min=0.0,
                )
                .props("dense debounce=500")
                .classes("w-48")
            )

            def _on_scale_change(_e: Any = None) -> None:
                try:
                    cfg.mesh_scale = float(scale_input.value or 1.0)
                except (TypeError, ValueError):
                    return
                _save_and_rebake()
            scale_input.on("update:model-value", _on_scale_change)

            def _auto_detect_scale() -> None:
                if not cfg.body_path.exists():
                    ui.notify("Upload a body STL first", color="warning")
                    return
                detected, label = custom_tools.detect_mesh_unit_scale(
                    cfg.body_path,
                )
                scale_input.value = detected
                cfg.mesh_scale = detected
                _save_and_rebake()
                ui.notify(
                    f"Auto-detected: {label} → scale = {detected}",
                    color="info",
                )
            ui.button(
                "Auto-detect units", on_click=_auto_detect_scale, icon="straighten",
            ).props("size=sm outline")

        # ----- Snap helpers -----
        ui.label("Snap helpers").classes("text-xs opacity-70 q-mt-sm")

        def _apply_translate(t: tuple[float, float, float]) -> None:
            # Update both the inputs (visually) AND the stored config.
            for i, inp in enumerate(translate_inputs):
                inp.value = float(t[i]) * 1000.0
            cfg.mesh_translate_m = t
            _save_and_rebake()

        def _apply_rpy(r: tuple[float, float, float]) -> None:
            for i, inp in enumerate(rpy_inputs):
                inp.value = float(r[i]) * (180.0 / math.pi)
            cfg.mesh_rpy_rad = r
            _save_and_rebake()

        def _snap_bbox() -> None:
            if not cfg.body_path.exists():
                ui.notify("Upload a body STL first", color="warning")
                return
            t = custom_tools.snap_to_bbox_centre(cfg.body_path)
            if t is None:
                ui.notify("Bbox-centre snap failed", color="warning")
                return
            # The mesh transform is applied as scale * mesh + translate;
            # the bbox centroid is in raw STL units, so scale it before
            # using as a translate.
            t_scaled = tuple(v * cfg.mesh_scale for v in t)
            _apply_translate(t_scaled)
            ui.notify("Snapped to bbox centre", color="positive")

        def _snap_flange_face() -> None:
            if not cfg.body_path.exists():
                ui.notify("Upload a body STL first", color="warning")
                return
            res = custom_tools.snap_flange_face_to_origin(cfg.body_path)
            if res is None:
                ui.notify("Flange-face snap failed", color="warning")
                return
            t, r = res
            t_scaled = tuple(v * cfg.mesh_scale for v in t)
            _apply_rpy(r)
            _apply_translate(t_scaled)
            ui.notify("Snapped largest planar face to z=0", color="positive")

        def _snap_hole() -> None:
            if not cfg.body_path.exists():
                ui.notify("Upload a body STL first", color="warning")
                return
            t = custom_tools.snap_to_largest_circular_hole(cfg.body_path)
            if t is None:
                ui.notify(
                    "No circular hole found in the largest planar facet — "
                    "try the flange-face snap instead.",
                    color="warning", position="top",
                )
                return
            t_scaled = tuple(v * cfg.mesh_scale for v in t)
            _apply_translate(t_scaled)
            ui.notify("Snapped largest detected hole to origin", color="positive")

        with ui.row().classes("gap-1"):
            ui.button(
                "Snap bbox centre", on_click=_snap_bbox,
            ).props("size=sm outline")
            ui.button(
                "Snap flange face", on_click=_snap_flange_face,
            ).props("size=sm outline")
            ui.button(
                "Snap mount hole", on_click=_snap_hole,
            ).props("size=sm outline")

        # ----- TCP transform -----
        ui.separator().classes("q-my-sm")
        ui.label("TCP transform (flange → tool tip)").classes("text-xs opacity-70")

        def _on_tcp_origin(value: tuple[float, float, float]) -> None:
            cfg.tcp_origin_m = value
            _save_and_rebake()

        def _on_tcp_rpy(value: tuple[float, float, float]) -> None:
            cfg.tcp_rpy_rad = value
            _save_and_rebake()

        _tuple3_inputs(
            "TCP origin (mm)",
            cfg.tcp_origin_m, scale=1000.0, fmt="%.1f", step=0.5,
            on_change=_on_tcp_origin,
        )
        _tuple3_inputs(
            "TCP rotate (deg)",
            cfg.tcp_rpy_rad, scale=180.0 / math.pi, fmt="%.2f", step=1.0,
            on_change=_on_tcp_rpy,
        )

        # ----- Jaw motion (only if jaws are present) -----
        if cfg.has_jaws:
            ui.separator().classes("q-my-sm")
            ui.label("Jaw motion").classes("text-xs opacity-70")

            with ui.row().classes("items-center gap-2"):
                travel_input = (
                    ui.number(
                        label="Travel (mm)",
                        value=float(cfg.jaw_travel_m) * 1000.0,
                        format="%.2f", step=0.5, min=0.0,
                    )
                    .props("dense debounce=500")
                    .classes("w-32")
                )

                def _on_travel(_e: Any = None) -> None:
                    try:
                        cfg.jaw_travel_m = float(travel_input.value or 0.0) / 1000.0
                    except (TypeError, ValueError):
                        return
                    _save_and_rebake()
                travel_input.on("update:model-value", _on_travel)

                sym_switch = ui.switch(
                    "Symmetric", value=bool(cfg.jaw_symmetric),
                ).props("dense")

                def _on_symmetric(_e: Any = None) -> None:
                    cfg.jaw_symmetric = bool(sym_switch.value)
                    _save_and_rebake()
                sym_switch.on("update:model-value", _on_symmetric)

            ui.label("Axis").classes("text-xs opacity-70 q-mt-xs")

            def _on_jaw_axis(value: tuple[float, float, float]) -> None:
                cfg.jaw_axis = value
                _save_and_rebake()

            _tuple3_inputs(
                "Axis",
                cfg.jaw_axis, scale=1.0, fmt="%.2f", step=0.1,
                on_change=_on_jaw_axis,
            )


# ---------------------------------------------------------------------------
# Add-tool wizard
# ---------------------------------------------------------------------------


def _import_existing_tool_dialog(refresh: Callable[[], None]) -> None:
    """Dialog to fork an existing registered tool into a new custom tool.

    The picker lists every entry in ``parol6.tools._TOOL_REGISTRY`` —
    that includes built-ins (SSG-48, MSG, PNEUMATIC, VACUUM) AND any
    other custom tools already registered. Picking SSG-48 after the
    SSG-48 mesh hijack has run gives you a custom tool whose body is
    the merged camera-bracket STL — exactly the test target for this
    workflow.
    """
    available = custom_tools.list_registered_tools()
    if not available:
        ui.notify(
            "No source tools available — parol6 registry is empty.",
            color="warning",
        )
        return
    options = {key: f"{display}  ({key})" for key, display in available}

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
        ui.label("Import existing tool as custom").classes(
            "text-base font-semibold",
        )
        ui.label(
            "Forks any tool from parol6's registry — including the "
            "SSG-48 entry mutated by the mesh hijack — into a custom "
            "tool you can iterate on. Mesh files are copied; the new "
            "custom tool starts with an identity placement transform "
            "since the source meshes are already in flange coordinates.",
        ).classes("text-xs opacity-70")
        first_key = next(iter(options))
        source_select = ui.select(
            options=options, value=first_key, label="Source tool",
        ).props("dense").classes("w-full")
        target_input = ui.input(
            label="New custom-tool name",
            placeholder="ssg48_my_setup",
        ).props("dense autofocus")

        def _on_import() -> None:
            source_key = str(source_select.value or "")
            target = str(target_input.value or "").strip()
            if not source_key:
                ui.notify("Pick a source tool", color="warning")
                return
            if not target or not all(c.isalnum() or c == "_" for c in target):
                ui.notify(
                    "Target name must be non-empty letters/digits/underscore",
                    color="warning",
                )
                return
            cfg = custom_tools.import_from_registered(source_key, target)
            if cfg is None:
                ui.notify(
                    f"Import failed — check logs (target may already "
                    f"exist or source has no body mesh).",
                    color="warning",
                )
                return
            # Bake + register so it's picked up by the gripper dropdown
            # without a restart.
            custom_tools.register_one(cfg)
            ui.notify(
                f"Imported {source_key} → custom:{target}. "
                f"Use it from the gripper panel or the card's "
                f"'Use this tool' button.",
                color="positive", position="top",
            )
            dialog.close()
            refresh()

        with ui.row():
            ui.button("Import", on_click=_on_import, color="primary").props(
                "size=sm",
            )
            ui.button("Cancel", on_click=dialog.close).props("size=sm")
    dialog.open()


def _add_tool_dialog(refresh: Callable[[], None]) -> None:
    """Open a dialog to create a new custom tool: name + description +
    optional immediate body STL upload.
    """
    with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
        ui.label("Add custom tool").classes("text-base font-semibold")
        ui.label(
            "Creates a new entry under ~/.waldo-commander/custom_tools/. "
            "STLs and transforms can be edited after creation.",
        ).classes("text-xs opacity-70")
        name_input = ui.input(
            label="Name (folder + registry key)",
            placeholder="my_gripper",
        ).props("dense autofocus")
        ui.label(
            "Letters/digits/underscore only. Becomes 'custom:<name>' in "
            "the tool dropdown.",
        ).classes("text-xs opacity-60")
        display_input = ui.input(
            label="Display name (optional)",
            placeholder="My Custom Gripper",
        ).props("dense")
        description_input = ui.input(
            label="Description (optional)",
        ).props("dense")

        def _on_create() -> None:
            name = str(name_input.value or "").strip()
            if not name or not all(c.isalnum() or c == "_" for c in name):
                ui.notify(
                    "Name must be non-empty and use only "
                    "letters/digits/underscore.",
                    color="warning",
                )
                return
            if name in custom_tools.list_tool_names():
                ui.notify(
                    f"Tool {name!r} already exists.", color="warning",
                )
                return
            cfg = custom_tools.CustomToolConfig(
                name=name,
                display_name=str(display_input.value or "").strip() or name,
                description=str(description_input.value or "").strip(),
            )
            custom_tools.save_config(cfg)
            ui.notify(
                f"Created {name}. Upload STLs in the tool's card.",
                color="positive",
            )
            dialog.close()
            refresh()

        with ui.row():
            ui.button("Create", on_click=_on_create, color="primary").props(
                "size=sm",
            )
            ui.button("Cancel", on_click=dialog.close).props("size=sm")
    dialog.open()


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_custom_tools_expansion() -> None:
    """Render the "Custom tools" expansion: list + add button."""
    custom_tools.ensure_root()

    @ui.refreshable
    def _list_view() -> None:
        configs = custom_tools.list_configs()
        if not configs:
            ui.label(
                "No custom tools yet. Click \"Add custom tool\" to create one — "
                "drop your gripper / bracket STL files into the tool's card "
                "afterwards.",
            ).classes("text-xs opacity-70")
        for cfg in configs:
            _build_tool_card(cfg, refresh=_list_view.refresh)

    with ui.expansion("Custom tools", icon="extension").classes("w-full"):
        ui.label(
            f"Drop STLs into ~/.waldo-commander/custom_tools/<name>/ "
            f"or use the wizard. Registered tools appear in the gripper "
            f"dropdown as ‘custom:<name>’ after the next tool refresh.",
        ).classes("text-xs opacity-70")
        with ui.row().classes("q-mt-xs"):
            ui.button(
                "Add custom tool", icon="add",
                on_click=lambda: _add_tool_dialog(_list_view.refresh),
            ).props("size=sm outline")
            def _open_root() -> None:
                if not _open_folder_in_os(custom_tools.CUSTOM_TOOLS_ROOT):
                    ui.notify(
                        f"Could not open. Path: {custom_tools.CUSTOM_TOOLS_ROOT}",
                        color="warning",
                    )

            ui.button(
                "Open folder", on_click=_open_root, icon="folder_open",
            ).props("size=sm outline")
            ui.button(
                "Import existing tool", icon="content_copy",
                on_click=lambda: _import_existing_tool_dialog(_list_view.refresh),
            ).props("size=sm outline")
        _list_view()
