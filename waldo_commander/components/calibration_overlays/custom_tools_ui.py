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


def _build_intrinsics_override(
    cfg: custom_tools.CustomToolConfig, refresh: Callable[[], None],
) -> None:
    """Per-tool intrinsics override section. Toggle-on to override the
    globals; off resets every per-tool intrinsics field to None
    (inherit). When on, the six fields are pre-populated with the
    current global values so the user has a starting point.
    """
    from . import settings  # noqa: PLC0415

    has_override = (
        cfg.intr_fx is not None or cfg.intr_fy is not None
        or cfg.intr_cx is not None or cfg.intr_cy is not None
        or cfg.intr_width is not None or cfg.intr_height is not None
    )

    inputs_box = ui.column().classes("w-full q-mt-xs").style(
        f"display: {'block' if has_override else 'none'};",
    )

    def _seed_value(attr: str, fallback_key: str) -> float:
        v = getattr(cfg, attr)
        if v is not None:
            return float(v)
        # Read the GLOBAL setting (runtime override or shipped default),
        # not ``settings.get()`` — that one returns the ACTIVE tool's
        # per-tool override first, which is wrong here: when the user
        # opens this editor for a non-active tool, the seed would
        # show the currently-active tool's intrinsics instead of the
        # global default. Misleading UX.
        return float(settings.get_global(fallback_key))

    fx = _seed_value("intr_fx", "intr_fx")
    fy = _seed_value("intr_fy", "intr_fy")
    cx = _seed_value("intr_cx", "intr_cx")
    cy = _seed_value("intr_cy", "intr_cy")
    w = int(_seed_value("intr_width", "intr_width"))
    h = int(_seed_value("intr_height", "intr_height"))

    inputs: dict[str, Any] = {}
    with inputs_box:
        with ui.row().classes("items-center gap-1"):
            inputs["fx"] = ui.number(
                label="fx", value=fx, format="%.1f", step=1.0,
            ).props("dense debounce=500").classes("w-24")
            inputs["fy"] = ui.number(
                label="fy", value=fy, format="%.1f", step=1.0,
            ).props("dense debounce=500").classes("w-24")
        with ui.row().classes("items-center gap-1"):
            inputs["cx"] = ui.number(
                label="cx", value=cx, format="%.1f", step=1.0,
            ).props("dense debounce=500").classes("w-24")
            inputs["cy"] = ui.number(
                label="cy", value=cy, format="%.1f", step=1.0,
            ).props("dense debounce=500").classes("w-24")
        with ui.row().classes("items-center gap-1"):
            inputs["w"] = ui.number(
                label="Width (px)", value=w, format="%d", step=1, min=1,
            ).props("dense debounce=500").classes("w-28")
            inputs["h"] = ui.number(
                label="Height (px)", value=h, format="%d", step=1, min=1,
            ).props("dense debounce=500").classes("w-28")

    def _save_overrides(_e: Any = None) -> None:
        try:
            cfg.intr_fx = float(inputs["fx"].value)
            cfg.intr_fy = float(inputs["fy"].value)
            cfg.intr_cx = float(inputs["cx"].value)
            cfg.intr_cy = float(inputs["cy"].value)
            cfg.intr_width = int(inputs["w"].value)
            cfg.intr_height = int(inputs["h"].value)
        except (TypeError, ValueError):
            return
        custom_tools.save_config(cfg)

    for inp in inputs.values():
        inp.on("update:model-value", _save_overrides)

    def _on_toggle(e) -> None:
        new_value = bool(getattr(e, "value", False))
        if new_value:
            inputs_box.style("display: block;")
            _save_overrides()
            ui.notify(
                f"Per-tool intrinsics enabled for custom:{cfg.name}.",
                color="info",
            )
        else:
            inputs_box.style("display: none;")
            cfg.intr_fx = None
            cfg.intr_fy = None
            cfg.intr_cx = None
            cfg.intr_cy = None
            cfg.intr_width = None
            cfg.intr_height = None
            custom_tools.save_config(cfg)
            ui.notify(
                f"Per-tool intrinsics reset for custom:{cfg.name}.",
                color="info",
            )

    ui.switch(
        "Override globals for this tool", value=has_override,
        on_change=_on_toggle,
    ).props("dense")


def _build_cam_mount_override(
    cfg: custom_tools.CustomToolConfig, refresh: Callable[[], None],
) -> None:
    """Per-tool camera-mount override (translate + tilt). Same pattern
    as the intrinsics override — toggle-on populates from globals,
    toggle-off resets to None. Also displays a "Save calibrated mount
    here" button — when calibration writes a result via the calibration
    panel, this is where it lands automatically.
    """
    import math  # noqa: PLC0415

    from . import settings  # noqa: PLC0415

    has_override = (
        cfg.cam_mount_translate_mm is not None
        or cfg.cam_mount_tilt_deg is not None
    )

    inputs_box = ui.column().classes("w-full q-mt-xs").style(
        f"display: {'block' if has_override else 'none'};",
    )

    # Same global-vs-active-tool concern as _build_intrinsics_override
    # — when this editor opens for a NON-active tool, ``settings.get``
    # returns the currently-active tool's per-tool override, which
    # silently leaks the wrong cam-mount values into the seed.
    # ``settings.get_global`` skips the active-tool override layer.
    seed_translate = (
        cfg.cam_mount_translate_mm
        if cfg.cam_mount_translate_mm is not None
        else tuple(settings.get_global("cam_mount_translate_mm"))
    )
    seed_tilt = (
        cfg.cam_mount_tilt_deg
        if cfg.cam_mount_tilt_deg is not None
        else tuple(settings.get_global("cam_mount_tilt_deg"))
    )

    translate_inputs: list[ui.number] = []
    tilt_inputs: list[ui.number] = []

    with inputs_box:
        ui.label("Translate (flange frame, mm)").classes(
            "text-xs opacity-70",
        )
        with ui.row().classes("items-center gap-1"):
            for i, axis in enumerate("XYZ"):
                inp = (
                    ui.number(
                        label=axis,
                        value=float(seed_translate[i]),
                        format="%.1f", step=0.5,
                    )
                    .props("dense debounce=500")
                    .classes("w-24")
                )
                translate_inputs.append(inp)
        ui.label("Tilt (XYZ-extrinsic, deg)").classes(
            "text-xs opacity-70 q-mt-xs",
        )
        with ui.row().classes("items-center gap-1"):
            for i, axis in enumerate(("Rx", "Ry", "Rz")):
                inp = (
                    ui.number(
                        label=axis,
                        value=float(seed_tilt[i]),
                        format="%.2f", step=1.0,
                    )
                    .props("dense debounce=500")
                    .classes("w-24")
                )
                tilt_inputs.append(inp)

    def _save_overrides(_e: Any = None) -> None:
        try:
            cfg.cam_mount_translate_mm = tuple(
                float(inp.value or 0.0) for inp in translate_inputs
            )
            cfg.cam_mount_tilt_deg = tuple(
                float(inp.value or 0.0) for inp in tilt_inputs
            )
        except (TypeError, ValueError):
            return
        custom_tools.save_config(cfg)

    for inp in translate_inputs + tilt_inputs:
        inp.on("update:model-value", _save_overrides)

    def _on_toggle(e) -> None:
        new_value = bool(getattr(e, "value", False))
        if new_value:
            inputs_box.style("display: block;")
            _save_overrides()
            ui.notify(
                f"Per-tool cam mount enabled for custom:{cfg.name}.",
                color="info",
            )
        else:
            inputs_box.style("display: none;")
            cfg.cam_mount_translate_mm = None
            cfg.cam_mount_tilt_deg = None
            custom_tools.save_config(cfg)
            ui.notify(
                f"Per-tool cam mount reset for custom:{cfg.name}.",
                color="info",
            )

    ui.switch(
        "Override globals for this tool", value=has_override,
        on_change=_on_toggle,
    ).props("dense")


def _build_variants_section(
    cfg: custom_tools.CustomToolConfig, refresh: Callable[[], None],
) -> None:
    """Per-tool variants editor. Each variant ships its own jaw STL pair
    + jaw motion and gets a ``ToolVariant`` entry in the parol6 registry
    so the gripper panel's variant dropdown can swap among them.

    Default tool jaws (jaw_left.stl / jaw_right.stl) stay around — they
    apply when the user picks the tool with no variant selected. The
    per-variant jaws override at variant-pick time.
    """
    ui.label(
        "Each variant has its own jaw STL pair and jaw motion.",
    ).classes("text-xs opacity-70")

    def _on_add_variant() -> None:
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Add variant to custom:{cfg.name}").classes(
                "text-base font-semibold",
            )
            key_input = ui.input(
                label="Variant key (alnum + underscore)",
                placeholder="e.g. finger, pinch, custom",
            ).props("dense autofocus")
            display_input = ui.input(
                label="Display name (optional)",
                placeholder="e.g. Finger",
            ).props("dense")

            def _on_create() -> None:
                vkey = str(key_input.value or "").strip()
                if not vkey or not all(c.isalnum() or c == "_" for c in vkey):
                    ui.notify(
                        "Key must be non-empty, letters/digits/underscore only",
                        color="warning",
                    )
                    return
                if any(v.key == vkey for v in cfg.variants):
                    ui.notify(
                        f"Variant {vkey!r} already exists", color="warning",
                    )
                    return
                cfg.variants.append(
                    custom_tools.CustomToolVariant(
                        key=vkey,
                        display_name=str(display_input.value or "").strip() or vkey,
                    )
                )
                custom_tools.save_config(cfg)
                dialog.close()
                refresh()
                ui.notify(
                    f"Added variant {vkey}. Upload its jaw STLs in the card.",
                    color="positive",
                )

            with ui.row():
                ui.button("Create", on_click=_on_create, color="primary").props(
                    "size=sm",
                )
                ui.button("Cancel", on_click=dialog.close).props("size=sm")
        dialog.open()

    ui.button(
        "Add variant", on_click=_on_add_variant, icon="add",
    ).props("size=sm outline")

    if not cfg.variants:
        return

    for variant in cfg.variants:
        _build_variant_card(cfg, variant, refresh)


def _build_variant_card(
    cfg: custom_tools.CustomToolConfig,
    variant: custom_tools.CustomToolVariant,
    refresh: Callable[[], None],
) -> None:
    """Editor for one variant — STL upload slots + jaw motion params."""
    with ui.card().classes("w-full q-mt-xs bg-blue-grey-9"):
        with ui.row().classes("w-full items-center"):
            ui.label(variant.display_name or variant.key).classes(
                "text-sm font-medium",
            )
            ui.label(f"({variant.key})").classes("text-xs opacity-60")
            ui.space()
            ui.label(
                "jaws ✓" if variant.has_jaws else "jaws ✗",
            ).classes(
                "text-xs " + (
                    "text-green-7" if variant.has_jaws else "text-red-7"
                ),
            )

            def _on_delete_variant(v=variant) -> None:
                cfg.variants = [x for x in cfg.variants if x.key != v.key]
                # Best-effort delete the on-disk variant STLs too —
                # both the user-uploaded source under ``CUSTOM_TOOLS_ROOT``
                # and the baked transformed copy in parol6's mesh dir.
                # Without the baked-copy unlink, deleting a variant left
                # an orphan ``custom_<name>_variant_<key>_jaw_<side>.stl``
                # in parol6's mesh dir for the lifetime of the install.
                for side in ("left", "right"):
                    p = cfg.variant_jaw_path(v.key, side)
                    try:
                        if p.exists():
                            p.unlink()
                    except OSError as e:
                        logger.debug(
                            "couldn't delete variant STL %s: %s", p, e,
                        )
                # Sweep baked variant copies out of parol6's mesh dir.
                mesh_dir = custom_tools._parol6_mesh_dir()
                if mesh_dir is not None:
                    for side in ("left", "right"):
                        baked = mesh_dir / custom_tools._baked_filename(
                            cfg.name, f"variant_{v.key}_jaw_{side}",
                        )
                        try:
                            if baked.exists():
                                baked.unlink()
                        except OSError as e:
                            logger.debug(
                                "couldn't delete baked variant %s: %s",
                                baked, e,
                            )
                custom_tools.save_config(cfg)
                ui.notify(f"Deleted variant {v.key}", color="info")
                refresh()

            ui.button(
                icon="delete", on_click=_on_delete_variant,
            ).props("flat round dense color=negative size=sm")

        # STL upload slots
        ui.label("STL files").classes("text-xs opacity-70 q-mt-xs")

        def _make_variant_upload_handler(side: str, v=variant):
            def _handle(e: events.UploadEventArguments) -> None:
                try:
                    suffix = Path(e.name).suffix or ".stl"
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False,
                    ) as tmp:
                        tmp.write(e.content.read())
                        tmp_path = Path(tmp.name)
                    custom_tools.import_variant_stl(
                        cfg.name, v.key, side, tmp_path,
                    )
                    tmp_path.unlink(missing_ok=True)
                    ui.notify(
                        f"Uploaded variant {v.key}/{side} STL",
                        color="positive",
                    )
                    refresh()
                except Exception as ex:  # noqa: BLE001
                    ui.notify(f"Upload failed: {ex}", color="warning")
            return _handle

        with ui.row().classes("items-center gap-2"):
            ui.upload(
                label="Jaw left",
                on_upload=_make_variant_upload_handler("left"),
                auto_upload=True, max_files=1,
            ).props("dense accept=.stl").classes("w-48")
            ui.upload(
                label="Jaw right",
                on_upload=_make_variant_upload_handler("right"),
                auto_upload=True, max_files=1,
            ).props("dense accept=.stl").classes("w-48")

        # Jaw motion
        ui.label("Jaw motion").classes("text-xs opacity-70 q-mt-xs")
        with ui.row().classes("items-center gap-1"):
            travel_input = (
                ui.number(
                    label="Travel (mm)",
                    value=float(variant.jaw_travel_m) * 1000.0,
                    format="%.2f", step=0.5, min=0.0,
                )
                .props("dense debounce=500")
                .classes("w-32")
            )

            def _save_travel(_e: Any = None, v=variant) -> None:
                try:
                    v.jaw_travel_m = float(travel_input.value or 0.0) / 1000.0
                except (TypeError, ValueError):
                    return
                custom_tools.save_config(cfg)

            def _save_symmetric(e, v=variant) -> None:
                v.jaw_symmetric = bool(getattr(e, "value", False))
                custom_tools.save_config(cfg)

            travel_input.on("update:model-value", _save_travel)
            ui.switch(
                "Symmetric", value=bool(variant.jaw_symmetric),
                on_change=_save_symmetric,
            ).props("dense")


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
                # Bake first — registry mutation should be in place
                # before the local apply queries meshes. ``register_one``
                # returns False when the bake itself fails (corrupt STL,
                # trimesh import error, etc.); without checking the
                # return, we'd fall through to ``select_as_active`` and
                # parol6's apply_tool would fail because the canonical
                # mesh entries don't exist in the registry. Surface the
                # bake failure to the user and bail.
                if not custom_tools.register_one(cfg):
                    ui.notify(
                        f"Bake failed for custom:{cfg.name}; check the "
                        "logs for the failing mesh, fix the source STL, "
                        "and try again.",
                        color="warning", position="top",
                    )
                    return
                ok = await custom_tools.select_as_active(
                    cfg.name, proxy_tool_key=str(cfg.proxy_tool_key or ""),
                )
                if ok:
                    msg = f"Switched active tool to custom:{cfg.name}"
                    if cfg.proxy_tool_key:
                        msg += f" (motor proxied to {cfg.proxy_tool_key})"
                    ui.notify(msg, color="positive", position="top")
                    # Live-apply calibration overlays + panel for the
                    # new tool: re-evaluates camera-bearing gating,
                    # picks up per-tool intrinsic / mount overrides,
                    # builds or tears down scene overlays as needed.
                    # No page reload required.
                    try:
                        from .panel import apply_calibration_state  # noqa: PLC0415

                        apply_calibration_state()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "apply_calibration_state failed: %s", exc,
                        )
                    refresh()
                else:
                    ui.notify(
                        f"Local apply failed for custom:{cfg.name}. "
                        f"Check the logs for details.",
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
                        "Removes the folder, config, and all baked STLs.",
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
                    "Re-bake failed. Check logs (probably no body.stl yet).",
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
                    f"Re-baked custom:{cfg.name}. Click \"Use this tool\" to view.",
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
                    "No circular hole found. Try the flange-face snap instead.",
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

        # ----- Camera flag + per-tool calibration overrides -----
        ui.separator().classes("q-my-sm")
        ui.label("Calibration").classes("text-xs opacity-70")
        def _on_camera_toggle(e) -> None:
            cfg.has_camera = bool(getattr(e, "value", False))
            custom_tools.save_config(cfg)
            ui.notify(
                f"has_camera = {cfg.has_camera}.",
                color="info", position="top",
            )
        ui.switch(
            "Tool has a calibratable camera",
            value=bool(cfg.has_camera),
            on_change=_on_camera_toggle,
        ).props("dense")

        # ----- Per-tool intrinsics override -----
        with ui.expansion(
            "Camera intrinsics (override globals for this tool)",
            icon="camera",
        ).classes("w-full q-mt-xs"):
            _build_intrinsics_override(cfg, refresh)

        # ----- Per-tool cam mount override -----
        with ui.expansion(
            "Camera mount (override globals for this tool)",
            icon="open_with",
        ).classes("w-full"):
            _build_cam_mount_override(cfg, refresh)

        # ----- Controller proxy -----
        ui.separator().classes("q-my-sm")
        ui.label(
            "Controller proxy (motor commands)",
        ).classes("text-xs opacity-70")
        ui.label(
            "Pick a built-in tool the controller should act as for jaw "
            "motion. Leave empty for visualisation only.",
        ).classes("text-xs opacity-60")
        proxy_options: dict[str, str] = {"": "(none, visualisation only)"}
        for key, display in custom_tools.list_registered_tools():
            if key.startswith("custom:"):
                continue  # avoid proxying through other custom tools
            proxy_options[key] = f"{display}  ({key})"
        proxy_select = ui.select(
            options=proxy_options,
            value=str(cfg.proxy_tool_key or ""),
            label="Proxy tool",
        ).props("dense").classes("w-64")

        def _on_proxy_change(_e: Any = None) -> None:
            cfg.proxy_tool_key = str(proxy_select.value or "")
            _save_and_rebake()
        proxy_select.on("update:model-value", _on_proxy_change)

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

                def _on_symmetric(e) -> None:
                    cfg.jaw_symmetric = bool(getattr(e, "value", False))
                    _save_and_rebake()
                ui.switch(
                    "Symmetric", value=bool(cfg.jaw_symmetric),
                    on_change=_on_symmetric,
                ).props("dense")

            ui.label("Axis").classes("text-xs opacity-70 q-mt-xs")

            def _on_jaw_axis(value: tuple[float, float, float]) -> None:
                cfg.jaw_axis = value
                _save_and_rebake()

            _tuple3_inputs(
                "Axis",
                cfg.jaw_axis, scale=1.0, fmt="%.2f", step=0.1,
                on_change=_on_jaw_axis,
            )

        # ----- Variants -----
        ui.separator().classes("q-my-sm")
        with ui.expansion("Variants", icon="layers").classes("w-full"):
            _build_variants_section(cfg, refresh)


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
            "No source tools available. parol6 registry is empty.",
            color="warning",
        )
        return
    options = {key: f"{display}  ({key})" for key, display in available}

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
        ui.label("Import existing tool as custom").classes(
            "text-base font-semibold",
        )
        ui.label(
            "Forks any tool from parol6's registry into a custom tool "
            "you can iterate on. Mesh files are copied with an identity transform.",
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
                    f"Import failed. Check logs (target may already exist "
                    f"or source has no body mesh).",
                    color="warning",
                )
                return
            # Bake + register so it's picked up by the gripper dropdown
            # without a restart.
            custom_tools.register_one(cfg)
            ui.notify(
                f"Imported {source_key} as custom:{target}.",
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
            "STLs and transforms can be edited after creation.",
        ).classes("text-xs opacity-70")
        name_input = ui.input(
            label="Name (folder + registry key)",
            placeholder="my_gripper",
        ).props("dense autofocus")
        ui.label(
            "Letters, digits, underscore only.",
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
                "No custom tools yet. Click \"Add custom tool\" to create one.",
            ).classes("text-xs opacity-70")
        for cfg in configs:
            _build_tool_card(cfg, refresh=_list_view.refresh)

    with ui.expansion("Custom tools", icon="extension").classes("w-full"):
        ui.label(
            "Registered tools appear in the gripper dropdown as 'custom:<name>'.",
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
