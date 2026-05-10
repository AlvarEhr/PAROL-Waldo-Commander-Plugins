"""Click-on-dot pose popup for the reachability visualization.

When the user clicks a green reachability dot in the 3D scene, a small
popup card with a downward-pointing arrow appears anchored above the
click position. The card shows the candidate's flange position and
offers a "Go to pose" button that dispatches a daemon thread driving
``client.move_j`` to the candidate's joint configuration.

Architecture:

* ``init_popup_container()`` creates a fixed-position container element
  on the page DOM. Stashes the handle in ``_state`` so subsequent calls
  replace the previous container (idempotent across page rebuilds).
* ``register_click_handler(scene)`` adds a click handler to the URDF
  scene's handler list. Filters for left-mouseup events that hit a
  ``calib:reach_dot_<i>`` named sphere. Replaces any prior handler so
  multiple ``add_overlays`` calls don't stack.
* The handler looks up ``_state["reachable_candidates"]`` by index and
  opens the popup at the click coordinates.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from nicegui import ui

from .state import _state

logger = logging.getLogger(__name__)


# Background colour for the card + arrow. Pick a dark mid-grey that
# reads well over the scene's dark background and on either theme. The
# arrow uses the SAME colour so the triangle blends seamlessly with
# the card's bottom edge.
_POPUP_BG: str = "#2c3038"
_POPUP_FG: str = "#ffffff"


def _close_popup() -> None:
    """Tear down whatever the popup container currently holds. No-op
    when the container hasn't been initialised yet (e.g. clicked
    before ``add_overlays`` finished its first run)."""
    container = _state.get("pose_popup_container")
    if container is not None:
        try:
            container.clear()
        except Exception:  # noqa: BLE001
            pass


def _go_to_pose_for_candidate(candidate: Any, dot_idx: int) -> None:
    """Drive the robot to ``candidate.joint_angles_rad`` via
    ``client.move_j``. The candidate is captured at popup-open time
    (not re-looked-up from ``_state`` here) so a re-render or tool
    switch between popup-open and the user's button click can't cause
    the wrong-pose dispatch.

    Closes the popup immediately, dispatches the move on a daemon
    thread, and posts status to the calibration panel. The collision
    pre-check raises a notify (not just a status update) on abort so
    the user immediately sees why nothing moved.

    Refuses to dispatch if another motion-bearing thread is active
    (calibration, localise, hover, or another go-to-pose). Without
    this guard, rapid clicks spawn multiple workers that clobber
    ``_state["client"]`` and break STOP for all but the latest.
    """
    if candidate is None:
        _close_popup()
        return
    # Concurrency guard: only one motion thread at a time.
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
        """Spawn a fresh daemon thread to dispatch the move. Used by the
        normal safe path AND by the dialog's 'Send anyway' button (which
        runs on the main loop and needs an off-loop thread for the
        synchronous parol6 client).
        """
        if _state.get("is_going_to_pose"):
            return
        _state["is_going_to_pose"] = True
        _state["stop_requested"] = False

        def _dispatch_thread() -> None:
            from .panel import _post_status as _ps  # noqa: PLC0415
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
                _state["is_going_to_pose"] = False
                _state["stop_requested"] = False

        threading.Thread(
            target=_dispatch_thread, daemon=True,
            name=f"go-to-pose-dispatch-{dot_idx}",
        ).start()

    def _worker() -> None:
        from .panel import _post_status  # noqa: PLC0415

        # Per-thread ownership token. The caller set
        # ``_state["is_going_to_pose"] = True`` before spawning this
        # worker; the worker initially "owns" the flag. If the
        # collision-blocked path releases ownership (so the dialog's
        # `Send anyway` can re-acquire), the worker's `finally` must
        # NOT clear the flag — otherwise it races with the new
        # dispatch thread that just set it back to True.
        worker_holds_flag = True

        try:
            angles_rad = np.asarray(candidate.joint_angles_rad, dtype=np.float64)
            angles_deg = np.degrees(angles_rad).tolist()

            # Collision pre-check. Same check the calibration filter
            # pipeline + hover-above-board use; gated on the master
            # ``mesh_collision_check_enabled`` toggle.
            try:
                from waldo_commander.state import robot_state  # noqa: PLC0415

                from .collision import validate_joint_trajectory  # noqa: PLC0415
                from .preview_dialog import (  # noqa: PLC0415
                    show_collision_dialog_threadsafe,
                )

                current_q_deg = list(robot_state.angles.deg[:6])
                # gripper_only=True: angles_deg is a post-IK joint
                # config from PoseGenerator's IK pass on the
                # reachability candidate. Arm self-collision is the
                # IK solver's job (d609024).
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
                    # Release ownership BEFORE scheduling the dialog so
                    # the dialog's `Send anyway` callback can
                    # re-acquire. Worker's `finally` won't touch the
                    # flag below (token tracks ownership separately).
                    worker_holds_flag = False
                    _state["is_going_to_pose"] = False
                    # Clear the stale client handle so STOP between
                    # dialog open and Send anyway is a clean no-op.
                    _state["client"] = None
                    scheduled = show_collision_dialog_threadsafe(
                        message=msg,
                        target_q_deg=list(angles_deg),
                        on_send_anyway=lambda a=list(angles_deg): _spawn_dispatch(a),
                    )
                    if not scheduled:
                        # No main loop available (rare; likely
                        # teardown in progress). Surface a fallback
                        # toast via best-effort scheduling.
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

            # Safe path - dispatch directly. This worker thread already
            # holds the busy flag (set by the caller when it scheduled
            # _worker). We don't want _spawn_dispatch's busy-flag
            # acquisition to block, so just dispatch inline here.
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
            # Only clear the flag if THIS worker still owns it. The
            # collision-blocked path already released ownership above
            # (worker_holds_flag = False); a later `_spawn_dispatch`
            # may have re-acquired the flag and set it True - we MUST
            # NOT clobber it.
            if worker_holds_flag:
                _state["is_going_to_pose"] = False
            _state["stop_requested"] = False

    threading.Thread(
        target=_worker, name="reach-go-to-pose", daemon=True,
    ).start()


def _go_to_pose(dot_idx: int) -> None:
    """Backwards-compatible wrapper. Resolves the candidate from
    ``_state['reachable_candidates']`` (subject to re-render races)
    and dispatches via :func:`_go_to_pose_for_candidate`. Prefer
    binding the candidate directly at popup-open time.
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
    The card's BOTTOM-CENTRE arrow points at the click position; the
    card itself sits ABOVE the click so the user can see the dot.
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
        # Wrapper anchored at the click. Translate -50%/-100% so the
        # bottom-centre of the wrapper sits AT the click coords; the
        # arrow at the wrapper's bottom-centre then visually points
        # to the dot itself.
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
            # Downward-pointing arrow (CSS triangle). Same colour as
            # the card's background so it blends into a single shape.
            ui.element("div").style(
                "width: 0; height: 0; "
                "border-left: 8px solid transparent; "
                "border-right: 8px solid transparent; "
                f"border-top: 10px solid {_POPUP_BG};",
            )


def init_popup_container() -> None:
    """Create / replace the page-level popup container element. Called
    by ``add_overlays`` so the container lives in the same UI context
    as the calibration overlays.

    The container is a zero-size fixed-position div; child elements
    we add at click time use absolute positioning to sit at the click
    coordinates. ``pointer-events: none`` on the container lets clicks
    pass through to the scene below; the popup itself flips to
    ``pointer-events: auto`` so the user can interact with the
    buttons.
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
    """Add a click handler that opens the popup on dot hits. Replaces
    any previously-registered handler so multiple ``add_overlays``
    calls don't stack listeners.

    The URDF scene's ``click_events`` is set to
    ``["mousedown", "mouseup", "mouseleave", "contextmenu"]`` (see
    ``urdf_scene.py``), so we trigger on left-button mouseup as the
    canonical "click released over the dot" event.
    """
    if scene is None:
        return
    # Remove any prior registration of our handler so re-runs don't
    # stack callbacks. NiceGUI's scene exposes _click_handlers; we
    # mutate it directly because the public API only appends.
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
        # Sphere names are ``calib:reach_dot_<generation>_<idx>``. The
        # generation counter (bumped on every render) lets us detect
        # a stale click: user clicked a sphere that's since been
        # redrawn at a different index by an intervening sweep or
        # radius change. Without this guard, the click handler can
        # dispatch the WRONG candidate when re-render races with the
        # click event.
        #
        # IMPORTANT: when the raycaster delivers multiple dot hits
        # (e.g. two sphere groups stacked at the same position from a
        # defer-race that the renderer's idempotency fix should now
        # prevent — but cheap defence in depth), we iterate ALL of
        # them looking for a valid-gen match before bailing. The old
        # behaviour of returning on first stale-gen hit could
        # silently swallow a click on a CURRENT sphere just because
        # an orphan from a prior gen happened to be in the hit list.
        dot_idx: int | None = None
        had_dot_hit = False
        had_stale_dot_hit = False
        current_gen = int(_state.get("reach_generation", 0))
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
                # Defensive: should be unreachable since had_stale_dot_hit
                # implies had_dot_hit was set. Kept as an explicit branch
                # so future edits don't accidentally drop the close.
                _close_popup()
                return
            if had_dot_hit:
                # All dot hits were stale (orphan-group race). Don't
                # close any open popup — the user may have an active
                # popup from an earlier valid click and the stale hit
                # shouldn't dismiss it.
                logger.debug(
                    "reach-dot click: only stale-gen hits "
                    "(current_gen=%d); skipping",
                    current_gen,
                )
                return
            # Click landed elsewhere; dismiss any open popup so the
            # user can clear it by clicking off.
            _close_popup()
            return
        candidates = _state.get("reachable_candidates") or []
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
