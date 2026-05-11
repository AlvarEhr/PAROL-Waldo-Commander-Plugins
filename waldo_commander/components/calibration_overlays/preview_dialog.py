"""Shared collision-rejected-move dialog with preview-in-sim option.

Offers Cancel / Preview-in-sim (pose-jump the URDF scene to the rejected
target) / Send-anyway (invoke caller's un-pre-checked dispatch). Per-move
bypass — flip ``mesh_collision_check_enabled`` in Settings for a
session-wide off.

Worker threads use :func:`show_collision_dialog_threadsafe`; request-context
code can call :func:`show_collision_dialog` directly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from .state import _state

logger = logging.getLogger(__name__)


# Module state: at most one active preview AND one open dialog. A second
# rejection replaces both rather than stacking, avoiding stranded state.
_active_preview: dict[str, Any] | None = None
_active_dialog: Any | None = None


def reset_preview_state() -> None:
    """Exit any active preview AND close any open dialog. Called from
    ``_teardown_overlays`` so a feature toggle can't strand PREVIEW.
    """
    global _active_preview, _active_dialog
    _exit_active_preview()
    if _active_dialog is not None:
        try:
            _active_dialog.close()
        except (RuntimeError, AttributeError):
            pass
        _active_dialog = None


def _exit_active_preview() -> None:
    """Restore the URDF scene from any active preview. No-op when none."""
    global _active_preview
    if _active_preview is None:
        return
    try:
        from waldo_commander.state import ui_state  # noqa: PLC0415

        scene = ui_state.urdf_scene
        if scene is not None and hasattr(scene, "exit_preview"):
            scene.exit_preview()
    except (ImportError, AttributeError) as e:
        logger.debug("preview_dialog: exit_preview lookup failed: %s", e)
    _active_preview = None


def _enter_preview(target_q_deg: list[float]) -> bool:
    """Pose-jump the URDF scene to ``target_q_deg`` for preview.

    Idempotent: exits any existing preview first so the scene's
    ``_preview_previous_mode`` slot isn't clobbered with PREVIEW.
    """
    global _active_preview
    if _active_preview is not None:
        _exit_active_preview()
    try:
        from waldo_commander.state import ui_state  # noqa: PLC0415
    except ImportError:
        return False
    scene = getattr(ui_state, "urdf_scene", None)
    if scene is None or not hasattr(scene, "apply_preview_pose"):
        return False
    angles_rad = np.deg2rad(np.asarray(target_q_deg, dtype=np.float64)).tolist()
    try:
        scene.apply_preview_pose(angles_rad)
    except Exception as e:  # noqa: BLE001
        logger.debug("preview_dialog: apply_preview_pose failed: %s", e)
        return False
    _active_preview = {"target_q_deg": list(target_q_deg)}
    return True


def show_collision_dialog(
    *,
    message: str,
    target_q_deg: list[float],
    on_send_anyway: Callable[[], None],
    on_cancel: Callable[[], None] | None = None,
) -> None:
    """Open the collision-rejected-move dialog.

    Must run inside a NiceGUI request context; worker threads should call
    :func:`show_collision_dialog_threadsafe`. A second invocation closes
    any prior dialog so persistent modals never stack.
    """
    global _active_dialog
    from nicegui import ui  # noqa: PLC0415

    if _active_dialog is not None:
        try:
            _active_dialog.close()
        except (RuntimeError, AttributeError):
            pass
        _active_dialog = None
    _exit_active_preview()

    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("min-w-[28rem]"):
        ui.label("Move would collide").classes("text-h6 text-warning")
        ui.label(message).classes("text-body2")
        with ui.expansion("Target joint angles (deg)", icon="info").classes("w-full"):
            ui.label(", ".join(f"{v:.2f}" for v in target_q_deg)).classes(
                "font-mono text-caption",
            )

        # Revealed once Preview is entered.
        preview_panel = ui.row().classes("w-full justify-end gap-2")
        preview_panel.set_visibility(False)

        # Captured so handlers can disable Preview after first click.
        preview_button: Any = None

        def _close_dialog() -> None:
            global _active_dialog
            try:
                dialog.close()
            except (RuntimeError, AttributeError):
                pass
            if _active_dialog is dialog:
                _active_dialog = None

        def _do_cancel() -> None:
            _exit_active_preview()
            _close_dialog()
            if on_cancel is not None:
                try:
                    on_cancel()
                except Exception as e:  # noqa: BLE001
                    logger.debug("on_cancel raised: %s", e)

        def _do_preview() -> None:
            ok = _enter_preview(list(target_q_deg))
            if not ok:
                ui.notify(
                    "Preview unavailable: URDF scene not ready.",
                    color="warning", position="top",
                )
                return
            preview_panel.set_visibility(True)
            # Block a re-click that would clobber previous-mode tracking.
            if preview_button is not None:
                preview_button.disable()

        def _do_exit_preview() -> None:
            _exit_active_preview()
            preview_panel.set_visibility(False)
            if preview_button is not None:
                preview_button.enable()

        def _do_send_anyway() -> None:
            # Snap the scene back to LIVE before the actual move starts.
            _exit_active_preview()
            _close_dialog()
            try:
                on_send_anyway()
            except Exception as e:  # noqa: BLE001
                logger.warning("on_send_anyway raised: %s", e)

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            cancel_button = ui.button("Cancel", on_click=_do_cancel).props(
                "flat autofocus",
            )
            preview_button = ui.button(
                "Preview in sim", on_click=_do_preview,
            ).props("flat color=info")
            ui.button(
                "Send anyway", on_click=_do_send_anyway,
            ).props("color=negative").classes("text-white")

        with preview_panel:
            ui.label("Preview mode active - robot frozen, scene shows rejected pose.").classes(
                "text-caption text-info self-center",
            )
            ui.button(
                "Exit preview", on_click=_do_exit_preview,
            ).props("flat color=info")

        # Escape cancels (Quasar persistent dialogs ignore it by default).
        # The listener is scoped to dialog lifetime via visibility binding.
        def _handle_keydown(e: Any) -> None:
            key = getattr(e, "key", None) or getattr(e, "args", {}).get("key", "")
            if key == "Escape":
                _do_cancel()

        try:
            ui.keyboard(on_key=_handle_keydown).bind_visibility_from(
                dialog, "value",
            )
        except Exception as exc:  # noqa: BLE001
            # ui.keyboard needs a UI context; without it Cancel still works.
            logger.debug("preview_dialog: keyboard hook failed: %s", exc)

    _active_dialog = dialog
    dialog.open()


def show_collision_dialog_threadsafe(
    *,
    message: str,
    target_q_deg: list[float],
    on_send_anyway: Callable[[], None],
    on_cancel: Callable[[], None] | None = None,
) -> bool:
    """Schedule the dialog on the main asyncio loop from a worker thread.

    Returns False when the loop is unavailable (caller should fall back to
    status line). Enters the captured NiceGUI client's slot before calling
    :func:`show_collision_dialog`, without which ``ui.dialog()`` raises
    "slot stack is empty" on every worker rejection.
    """
    loop = _state.get("main_loop")
    if loop is None:
        return False

    def _scheduled() -> None:
        client = _state.get("nicegui_client")
        try:
            if client is not None:
                with client:
                    show_collision_dialog(
                        message=message,
                        target_q_deg=list(target_q_deg),
                        on_send_anyway=on_send_anyway,
                        on_cancel=on_cancel,
                    )
            else:
                # No client captured — try anyway; the dialog will fail
                # loudly if the current slot stack is invalid.
                show_collision_dialog(
                    message=message,
                    target_q_deg=list(target_q_deg),
                    on_send_anyway=on_send_anyway,
                    on_cancel=on_cancel,
                )
        except RuntimeError as e:
            # Client likely deleted between schedule and fire (page reload).
            logger.debug(
                "preview_dialog: scheduled show failed (%s)", e,
            )

    try:
        loop.call_soon_threadsafe(_scheduled)
        return True
    except RuntimeError as e:
        logger.debug("preview_dialog: schedule failed: %s", e)
        return False
