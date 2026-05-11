"""Click-on-dot pose popup for the reachability visualization.

Clicking a green dot opens a card above the click with a "Go to pose"
button that dispatches ``client.move_j`` on a daemon thread.

* ``init_popup_container()`` — idempotent fixed-position DOM container.
* ``register_click_handler(scene)`` — replaces any prior handler so
  re-runs don't stack listeners.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from nicegui import ui

from .state import _state, _state_lock

logger = logging.getLogger(__name__)


# Card + arrow share the same colour so the triangle blends into the
# card's bottom edge.
_POPUP_BG: str = "#2c3038"
_POPUP_FG: str = "#ffffff"


def _close_popup() -> None:
    """Empty the popup container. No-op when uninitialised."""
    container = _state.get("pose_popup_container")
    if container is not None:
        try:
            container.clear()
        except Exception:  # noqa: BLE001
            pass


def _go_to_pose_for_candidate(candidate: Any, dot_idx: int) -> None:
    """Dispatch ``client.move_j`` for the captured candidate.

    Candidate is bound at popup-open time so a re-render between open
    and click can't dispatch the wrong pose. Refuses if another motion
    thread (calibration / localise / hover / pose) is already active —
    rapid clicks otherwise clobber ``_state["client"]`` and break STOP.
    """
    if candidate is None:
        _close_popup()
        return
    # One motion thread at a time.
    if any(
        _state.get(flag)
        for flag in ("is_running", "is_localising", "is_hovering", "is_going_to_pose")
    ):
        _close_popup()
        try:
            from nicegui import ui as _ui  # noqa: PLC0415

            _ui.notify(
                "Another move is in progress; wait for it to finish.",
                color="warning", position="top",
            )
        except Exception:  # noqa: BLE001
            pass
        return
    _state["is_going_to_pose"] = True
    _state["stop_requested"] = False
    _close_popup()

    def _spawn_dispatch(angles_deg: list[float]) -> None:
        """Spawn a daemon thread to dispatch the move. Used by the safe
        path and the dialog's "Send anyway" (which runs on the main loop
        and needs an off-loop thread for the synchronous client).
        """
        if _state.get("is_going_to_pose"):
            return
        _state["is_going_to_pose"] = True
        _state["stop_requested"] = False

        def _dispatch_thread() -> None:
            from .panel import _post_status as _ps  # noqa: PLC0415

            client: Any = None
            try:
                from parol6 import RobotClient  # noqa: PLC0415

                client = RobotClient(host="127.0.0.1", port=5001)
                _state["client"] = client
                _ps(f"Driving to reachable pose #{dot_idx}...")
                rc = client.move_j(
                    angles=list(angles_deg), speed=0.3, accel=0.5,
                    wait=True, timeout=20.0,
                )
                if rc < 0:
                    _ps(f"Move halted at pose #{dot_idx}.")
                else:
                    _ps(f"At reachable pose #{dot_idx}.")
            except Exception as e:  # noqa: BLE001
                try:
                    _ps(f"Move failed: {e}")
                except Exception:  # noqa: BLE001
                    logger.exception("go-to-pose dispatch crashed")
            finally:
                # HALT latches the controller into DISABLED until a RESUME,
                # so a user-pressed STOP would block the next move.
                if _state.get("stop_requested") and client is not None:
                    try:
                        client.resume()
                    except Exception as e:  # noqa: BLE001
                        logger.debug("go-to-pose post-halt resume raised: %s", e)
                # Drop our handle so ``_teardown_overlays`` doesn't halt a
                # finished move. ``is`` guards against a newer worker's client.
                if _state.get("client") is client:
                    _state["client"] = None
                _state["is_going_to_pose"] = False
                _state["stop_requested"] = False

        threading.Thread(
            target=_dispatch_thread, daemon=True,
            name=f"go-to-pose-dispatch-{dot_idx}",
        ).start()

    def _worker() -> None:
        from .panel import _post_status  # noqa: PLC0415

        # Ownership token; the collision-blocked path releases so "Send
        # anyway" can re-acquire. ``finally`` must honour the token.
        worker_holds_flag = True

        try:
            angles_rad = np.asarray(candidate.joint_angles_rad, dtype=np.float64)
            angles_deg = np.degrees(angles_rad).tolist()

            # Collision pre-check — gated on ``mesh_collision_check_enabled``.
            try:
                from waldo_commander.state import robot_state  # noqa: PLC0415

                from .collision import validate_joint_trajectory  # noqa: PLC0415
                from .preview_dialog import (  # noqa: PLC0415
                    show_collision_dialog_threadsafe,
                )

                current_q_deg = list(robot_state.angles.deg[:6])
                # gripper_only=True — post-IK config; arm self-collision
                # is the IK solver's job.
                check = validate_joint_trajectory(
                    current_q_deg, list(angles_deg), gripper_only=True,
                )
                if not check.get("safe", True):
                    reason = check.get("reason", "collision")
                    pair = check.get("colliding_pair")
                    pair_str = (
                        f", {pair[0]} <-> {pair[1]}"
                        if isinstance(pair, tuple) and len(pair) == 2
                        else ""
                    )
                    msg = (
                        f"Pose #{dot_idx}: aborted, would collide "
                        f"({reason}{pair_str})."
                    )
                    _post_status(msg)
                    # Release ownership before scheduling so "Send anyway"
                    # can re-acquire. ``finally`` honours the token.
                    worker_holds_flag = False
                    _state["is_going_to_pose"] = False
                    # Drop the handle so STOP between dialog open and Send
                    # anyway is a clean no-op.
                    _state["client"] = None
                    scheduled = show_collision_dialog_threadsafe(
                        message=msg,
                        target_q_deg=list(angles_deg),
                        on_send_anyway=lambda a=list(angles_deg): _spawn_dispatch(a),
                    )
                    if not scheduled:
                        # No main loop (likely teardown in progress).
                        loop = _state.get("main_loop")
                        if loop is not None:
                            from nicegui import ui as _ui  # noqa: PLC0415
                            try:
                                loop.call_soon_threadsafe(
                                    lambda m=msg: _ui.notify(
                                        m, color="warning", position="top",
                                    ),
                                )
                            except RuntimeError:
                                pass
                    return
            except Exception as e:  # noqa: BLE001
                logger.debug("go-to-pose pre-check skipped (%s)", e)

            # Safe path — dispatch inline; this worker already holds the
            # busy flag so ``_spawn_dispatch`` would block on re-acquire.
            from parol6 import RobotClient  # noqa: PLC0415

            client = RobotClient(host="127.0.0.1", port=5001)
            _state["client"] = client
            _post_status(f"Driving to reachable pose #{dot_idx}...")
            rc = client.move_j(
                angles=angles_deg, speed=0.3, accel=0.5,
                wait=True, timeout=20.0,
            )
            if rc < 0:
                _post_status(f"Move halted at pose #{dot_idx}.")
            else:
                _post_status(f"At reachable pose #{dot_idx}.")
        except Exception as e:  # noqa: BLE001
            try:
                _post_status(f"Move failed: {e}")
            except Exception:  # noqa: BLE001
                logger.exception("go-to-pose worker crashed")
        finally:
            # HALT latches DISABLED until a RESUME — a user STOP would
            # otherwise block the next move.
            if _state.get("stop_requested") and client is not None:
                try:
                    client.resume()
                except Exception as e:  # noqa: BLE001
                    logger.debug("go-to-pose post-halt resume raised: %s", e)
            # Drop our handle so ``_teardown_overlays`` doesn't halt a
            # finished move. ``is`` guards a newer worker's client.
            if _state.get("client") is client:
                _state["client"] = None
            # Honour the ownership token — a re-acquiring dispatch must
            # not have its flag cleared by us.
            if worker_holds_flag:
                _state["is_going_to_pose"] = False
            _state["stop_requested"] = False

    threading.Thread(
        target=_worker, name="reach-go-to-pose", daemon=True,
    ).start()


def _go_to_pose(dot_idx: int) -> None:
    """Backwards-compatible wrapper. Subject to re-render races; prefer
    binding the candidate at popup-open time.
    """
    candidates = _state.get("reachable_candidates") or []
    if dot_idx < 0 or dot_idx >= len(candidates):
        _close_popup()
        return
    _go_to_pose_for_candidate(candidates[dot_idx], dot_idx)


def _open_popup(
    client_x: int, client_y: int, dot_idx: int, candidate: Any,
) -> None:
    """Render the popup at ``(client_x, client_y)`` (viewport pixels).
    Card sits above the click; the bottom-centre arrow points at the dot.
    """
    container = _state.get("pose_popup_container")
    if container is None:
        return
    container.clear()
    try:
        pos_m = np.asarray(candidate.flange_pose, dtype=np.float64)[:3, 3]
        pos_mm = (pos_m * 1000.0).tolist()
    except Exception:  # noqa: BLE001
        pos_mm = [0.0, 0.0, 0.0]

    with container:
        # Translate -50%/-100% so the wrapper's bottom-centre sits on the
        # click; the arrow then points at the dot.
        with ui.element("div").style(
            f"position: fixed; "
            f"left: {client_x}px; top: {max(client_y - 6, 6)}px; "
            f"transform: translate(-50%, -100%); "
            f"pointer-events: auto; "
            f"display: flex; flex-direction: column; align-items: center; "
            f"font-size: 12px;",
        ):
            with ui.element("div").style(
                f"background: {_POPUP_BG}; color: {_POPUP_FG}; "
                f"padding: 8px 10px; border-radius: 6px; "
                f"box-shadow: 0 2px 8px rgba(0,0,0,0.4); "
                f"min-width: 160px;",
            ):
                ui.label(f"Reachable pose #{dot_idx}").style(
                    "font-weight: 500;",
                )
                ui.label(
                    f"Flange: ({pos_mm[0]:.0f}, {pos_mm[1]:.0f}, "
                    f"{pos_mm[2]:.0f}) mm",
                ).style("opacity: 0.75; margin-top: 2px;")
                with ui.row().classes("gap-1 q-mt-xs"):
                    ui.button(
                        "Go to pose", icon="navigation",
                        on_click=lambda c=candidate, i=dot_idx:
                            _go_to_pose_for_candidate(c, i),
                    ).props("size=sm dense color=primary unelevated")
                    ui.button(
                        "Cancel", on_click=_close_popup,
                    ).props("size=sm dense flat color=white")
            # Downward-pointing arrow (CSS triangle).
            ui.element("div").style(
                "width: 0; height: 0; "
                "border-left: 8px solid transparent; "
                "border-right: 8px solid transparent; "
                f"border-top: 10px solid {_POPUP_BG};",
            )


def init_popup_container() -> None:
    """Create / replace the page-level popup container.

    Zero-size fixed div with ``pointer-events: none`` so scene clicks pass
    through; popup contents flip to ``pointer-events: auto``.
    """
    old = _state.get("pose_popup_container")
    if old is not None:
        try:
            old.delete()
        except Exception:  # noqa: BLE001
            pass
        _state["pose_popup_container"] = None
    container = ui.element("div").style(
        "position: fixed; top: 0; left: 0; "
        "width: 0; height: 0; "
        "pointer-events: none; z-index: 9999;",
    )
    _state["pose_popup_container"] = container


def register_click_handler(scene: Any) -> None:
    """Add a click handler that opens the popup on dot hits. Replaces any
    prior registration. Triggers on left-button mouseup (canonical
    "click released over the dot").
    """
    if scene is None:
        return
    # NiceGUI's public scene API only appends; mutate ``_click_handlers``
    # directly to remove our prior handler.
    old_handler = _state.get("dot_click_handler")
    if old_handler is not None:
        handlers = getattr(scene, "_click_handlers", None)
        if isinstance(handlers, list):
            try:
                handlers.remove(old_handler)
            except ValueError:
                pass

    def _on_scene_click(e: Any) -> None:
        click_type = getattr(e, "click_type", "")
        button = getattr(e, "button", 0)
        if click_type != "mouseup" or button != 0:
            return
        hits = getattr(e, "hits", None) or []
        # Sphere names are ``calib:reach_dot_<gen>_<idx>``; gen is bumped
        # on every render so we can reject hits from a stale group. Iterate
        # ALL hits — an orphan from a prior gen mustn't suppress a current
        # one stacked at the same coords.
        dot_idx: int | None = None
        had_dot_hit = False
        had_stale_dot_hit = False
        # Snapshot gen + candidates atomically; writer pairs them under
        # ``_state_lock``.
        with _state_lock:
            current_gen = int(_state.get("reach_generation", 0))
            candidates: list = list(_state.get("reachable_candidates") or [])
        for hit in hits:
            name = getattr(hit, "object_name", "") or ""
            if not name.startswith("calib:reach_dot_"):
                continue
            had_dot_hit = True
            tail = name[len("calib:reach_dot_"):]
            parts = tail.split("_")
            if len(parts) != 2:
                continue
            try:
                gen = int(parts[0])
                idx = int(parts[1])
            except ValueError:
                continue
            if gen != current_gen:
                had_stale_dot_hit = True
                continue
            dot_idx = idx
            break
        if dot_idx is None:
            if had_stale_dot_hit and not had_dot_hit:
                # Defensive — should be unreachable, but kept explicit.
                _close_popup()
                return
            if had_dot_hit:
                # All hits stale (orphan-group race) — leave any prior
                # valid popup alone.
                logger.debug(
                    "reach-dot click: only stale-gen hits "
                    "(current_gen=%d); skipping",
                    current_gen,
                )
                return
            # Click off-dot — dismiss any open popup.
            _close_popup()
            return
        # Use the snapshot; re-reading would defeat its atomicity.
        if dot_idx >= len(candidates):
            logger.debug(
                "reach-dot click: dot_idx=%d but only %d candidates "
                "(stale render?)",
                dot_idx, len(candidates),
            )
            _close_popup()
            return
        cand = candidates[dot_idx]
        client_x = getattr(e, "client_x", None)
        client_y = getattr(e, "client_y", None)
        if client_x is None or client_y is None:
            return
        _open_popup(int(client_x), int(client_y), dot_idx, cand)

    scene.on_click(_on_scene_click)
    _state["dot_click_handler"] = _on_scene_click
