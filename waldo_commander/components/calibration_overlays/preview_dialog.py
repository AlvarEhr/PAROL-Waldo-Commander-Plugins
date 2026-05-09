"""Shared collision-rejected-move dialog with preview-in-sim option.

Called from any worker thread that pre-flight-rejected a move (hover,
pose-popup, localise, post-cal home, etc.). Surfaces a modal with three
choices:

* **Cancel** — close the dialog, do nothing. Default.
* **Preview in sim** — pose-jump the URDF scene to the rejected target
  pose, painted red translucent, with the live broadcast frozen until
  the user explicitly exits or dispatches a real move. The user can
  rotate / zoom the scene to inspect exactly which mesh would collide.
* **Send anyway** — invoke ``on_send_anyway()`` (the caller's
  un-pre-checked dispatch). Useful when the user has reviewed the
  preview and decided the FCL margin is overcautious.

This is a per-move bypass, NOT a master-toggle replacement. Flipping
the master ``mesh_collision_check_enabled`` toggle off in Settings
remains the way to disable checks session-wide.

Threading: workers should call :func:`show_collision_dialog_threadsafe`
which schedules dialog construction on the main asyncio loop via
``loop.call_soon_threadsafe``. Direct calls (from request-context code
in the panel UI) can use :func:`show_collision_dialog` directly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from .state import _state

logger = logging.getLogger(__name__)


# Module-level state: at most one active preview AND at most one open
# collision dialog at a time. A second collision-rejection while a
# preview is active replaces the old preview rather than stacking; a
# second rejection while a dialog is open closes the old dialog and
# replaces it. This avoids the stranding-and-stacking failure modes
# where a stale dialog dismisses a different preview's state.
_active_preview: dict[str, Any] | None = None
_active_dialog: Any | None = None


def reset_preview_state() -> None:
    """Forcefully exit any active preview AND close any open dialog.

    Called from `_teardown_overlays` so a feature on/off cycle never
    leaves the URDF scene stranded in PREVIEW.
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
    """Restore the URDF scene from any active preview. Safe no-op when
    no preview is active.
    """
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

    Returns True on success, False if the scene isn't available or the
    apply call fails. Idempotent: a second `_enter_preview` while a
    preview is already active first exits the old one so the
    `_preview_previous_mode` slot on the scene doesn't get clobbered
    with PREVIEW (which would break exit).
    """
    global _active_preview
    if _active_preview is not None:
        # Already in a preview — exit cleanly first so the previous-mode
        # slot is restored to LIVE before we re-enter.
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

    Must be called from within a NiceGUI request context (i.e. from
    the main asyncio loop). For worker-thread callers use
    :func:`show_collision_dialog_threadsafe`.

    Serialises with any prior open dialog: a second invocation while
    one is already showing closes the old one first so the user is
    never staring at two stacked persistent modals.
    """
    global _active_dialog
    from nicegui import ui  # noqa: PLC0415

    # Close any prior dialog to avoid stacking persistent modals.
    if _active_dialog is not None:
        try:
            _active_dialog.close()
        except (RuntimeError, AttributeError):
            pass
        _active_dialog = None
    # Also exit any prior preview before opening the new dialog so the
    # scene mode is clean on entry.
    _exit_active_preview()

    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("min-w-[28rem]"):
        ui.label("Move would collide").classes("text-h6 text-warning")
        ui.label(message).classes("text-body2")
        with ui.expansion("Target joint angles (deg)", icon="info").classes("w-full"):
            ui.label(", ".join(f"{v:.2f}" for v in target_q_deg)).classes(
                "font-mono text-caption",
            )

        # Preview-mode controls reveal when the user enters preview.
        preview_panel = ui.row().classes("w-full justify-end gap-2")
        preview_panel.set_visibility(False)

        # Captured here so handlers can disable Preview after first click.
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
            # Disable the Preview button after first click so a fast
            # re-click can't re-enter and confuse the previous-mode
            # slot tracking on the scene.
            if preview_button is not None:
                preview_button.disable()

        def _do_exit_preview() -> None:
            _exit_active_preview()
            preview_panel.set_visibility(False)
            if preview_button is not None:
                preview_button.enable()

        def _do_send_anyway() -> None:
            # Auto-exit any active preview before dispatch so the scene
            # snaps back to LIVE before the actual move begins.
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

        # Keyboard accessibility: Escape cancels (Quasar persistent
        # dialogs ignore Escape by default), Enter confirms Cancel
        # (the autofocused button). The keyboard listener is scoped
        # to the dialog's lifetime via the cancel handler that closes
        # it, so it doesn't leak across dialogs.
        def _handle_keydown(e: Any) -> None:
            key = getattr(e, "key", None) or getattr(e, "args", {}).get("key", "")
            if key == "Escape":
                _do_cancel()

        try:
            ui.keyboard(on_key=_handle_keydown).bind_visibility_from(
                dialog, "value",
            )
        except Exception as exc:  # noqa: BLE001
            # ui.keyboard requires a UI context; if the dialog is being
            # built from a non-page context (e.g. a teardown race), we
            # accept the lack of Escape support — Cancel button still
            # works.
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
    """Schedule the dialog on the main asyncio loop. Safe to call from
    a worker thread.

    Returns True if the schedule was queued, False if the main loop
    isn't available (no preview / dialog will appear; caller should
    fall back to a status-line update).
    """
    loop = _state.get("main_loop")
    if loop is None:
        return False
    try:
        loop.call_soon_threadsafe(
            lambda: show_collision_dialog(
                message=message,
                target_q_deg=list(target_q_deg),
                on_send_anyway=on_send_anyway,
                on_cancel=on_cancel,
            ),
        )
        return True
    except RuntimeError as e:
        logger.debug("preview_dialog: schedule failed: %s", e)
        return False
