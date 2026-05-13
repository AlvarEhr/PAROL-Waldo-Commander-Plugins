"""Hover-above-board verification mode."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import settings
from .state import _T_BOARD2BASE, _state

logger = logging.getLogger(__name__)


# Controller endpoint — shared with calibration_thread, localise, and the
# pose_popup go-to-pose dispatcher.
_CONTROLLER_HOST: str = "127.0.0.1"
_CONTROLLER_PORT: int = 5001


# ---------------------------------------------------------------------------
# Hover-above-board verification mode
# ---------------------------------------------------------------------------


def _build_R_with_z_axis(z_axis_world: NDArray[np.float64]) -> NDArray[np.float64]:
    """3x3 rotation whose Z column equals ``z_axis_world``. The roll about
    Z is arbitrary-but-deterministic — adequate for hover-above-board where
    only position and downward orientation matter.
    """
    z = np.asarray(z_axis_world, dtype=np.float64).reshape(3)
    z = z / float(np.linalg.norm(z))
    candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(candidate, z))) > 0.99:
        candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x = candidate - float(np.dot(candidate, z)) * z
    x /= float(np.linalg.norm(x))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _resolve_active_tool_transform(client: Any) -> tuple[np.ndarray, str]:
    """Look up the active tool's flange→TCP transform.

    Prefers the client's bound tool; falls back to the configured SSG-48
    default when nothing is bound (fresh connection). Returns
    ``(T_flange2tcp, tool_label)`` where label is for the status line.
    """
    from parol6 import tools as parol6_tools  # noqa: PLC0415
    from scipy.spatial.transform import Rotation as SciRot  # noqa: PLC0415

    try:
        spec = client.tool
        tcp_origin = np.asarray(spec.tcp_origin, dtype=np.float64).reshape(3)
        tcp_rpy = np.asarray(spec.tcp_rpy, dtype=np.float64).reshape(3)
        # XYZ-extrinsic Euler — see Docs/HANDOFF.md "Conventions".
        R = SciRot.from_euler("XYZ", tcp_rpy, degrees=False).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tcp_origin
        return T, str(spec.display_name)
    except RuntimeError:
        # No tool bound — use the configured default.
        T = np.asarray(
            parol6_tools.get_tool_transform("SSG-48", str(settings.tool_jaw_variant)),
            dtype=np.float64,
        )
        return T, "SSG-48 (fallback)"


def _drive_hover_pose_thread(
    board_local_x_m: float,
    board_local_y_m: float,
    standoff_m: float,
    mode: str = "camera",
) -> None:
    """Drive a reference frame to a standoff above the board for physical
    verification. Runs in a daemon thread spawned by the panel's hover
    buttons.

    Modes:
    * ``"camera"`` — drive the camera optical centre, axis pointing AT
      the target. Validates ``T_cam2flange`` (the calibration result).
    * ``"tcp"`` — drive the gripper TCP perpendicular to the board.
      Validates ``T_tcp2flange`` independent of calibration.

    Standoff is in board-local +Z so a tilted board still gets a
    perpendicular standoff.
    """
    # Lazy import — break the panel<->hover cycle.
    from .panel import _post_status  # noqa: PLC0415

    # Ownership token mirroring _state["is_hovering"]. The collision-blocked
    # path releases it so "Send anyway" can re-acquire; the `finally` below
    # must not clobber a re-acquired flag.
    worker_holds_flag = True

    # Tracked so ``finally`` can resume the controller before exit (STOP →
    # halt leaves state.enabled=False, blocking subsequent jogs).
    client: Any = None

    try:
        from parol6 import Robot, RobotClient  # noqa: PLC0415
        from parol6_vision.calibration.camera_mount import (  # noqa: PLC0415
            look_at_pose,
        )
        from scipy.spatial.transform import Rotation as SciRot  # noqa: PLC0415

        # Common world-frame quantities used by both modes.
        target_local = np.array(
            [board_local_x_m, board_local_y_m, 0.0, 1.0], dtype=np.float64,
        )
        target_world = (_T_BOARD2BASE @ target_local)[:3]
        # Board-local +Z in world frame.
        direction_up_world = _T_BOARD2BASE[:3, :3] @ np.array(
            [0.0, 0.0, 1.0], dtype=np.float64,
        )
        direction_up_world /= float(np.linalg.norm(direction_up_world))

        # Open the client up front so STOP can grab it via ``_state["client"]``
        # and TCP mode can read ``client.tool``.
        client = RobotClient(host=_CONTROLLER_HOST, port=_CONTROLLER_PORT)
        _state["client"] = client

        # HALT latches the controller into DISABLED until a RESUME, so a
        # prior STOP would block this move without this.
        try:
            client.resume()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "hover start: resume() raised %s: %s "
                "- first move may fail if controller is in disabled state",
                type(e).__name__, e,
            )

        if mode == "camera":
            mount = _state.get("current_mount")
            if mount is None:
                _post_status(
                    "Hover (camera): no camera mount available — "
                    "cannot compute pose.",
                )
                return
            cam_world = target_world + standoff_m * direction_up_world
            T_cam2base = look_at_pose(cam_world, target_world)
            T_flange2base = mount.flange_pose_for_cam_pose(T_cam2base)
            ref_label = "camera"
        elif mode == "tcp":
            # Compose TCP-in-base with TCP +Z aligned to board-local +Z,
            # then back out T_flange2base via the active tool transform.
            T_flange2tcp, tool_label = _resolve_active_tool_transform(client)
            tcp_origin_world = target_world + standoff_m * direction_up_world
            R_tcp = _build_R_with_z_axis(direction_up_world)
            T_tcp2base = np.eye(4, dtype=np.float64)
            T_tcp2base[:3, :3] = R_tcp
            T_tcp2base[:3, 3] = tcp_origin_world
            T_flange2base = T_tcp2base @ np.linalg.inv(T_flange2tcp)
            ref_label = f"TCP[{tool_label}]"
        else:
            _post_status(f"Hover: unknown mode '{mode}'")
            return

        pos = T_flange2base[:3, 3]
        rpy = SciRot.from_matrix(T_flange2base[:3, :3]).as_euler(
            "XYZ", degrees=False,
        )
        flange_xyz_rpy = np.concatenate([pos, rpy])

        robot = Robot()
        seed = np.radians(np.array([0.0, -90.0, 180.0, 0.0, 0.0, 180.0]))
        try:
            ik_result = robot.ik(flange_xyz_rpy, seed)
        except Exception as e:  # noqa: BLE001
            _post_status(f"Hover ({ref_label}): IK raised {type(e).__name__}: {e}")
            return
        q_rad = np.asarray(ik_result.q, dtype=np.float64)

        # FK-verify — parol6's IK reports success=False even on fine q.
        fk_pose = np.zeros(6, dtype=np.float64)
        robot.fk(q_rad, fk_pose)
        pos_err_mm = float(np.linalg.norm(fk_pose[:3] - flange_xyz_rpy[:3])) * 1000.0
        if pos_err_mm > 1.0:
            _post_status(
                f"Hover ({ref_label}): pose unreachable "
                f"(IK pos_err={pos_err_mm:.2f} mm). "
                f"Try a different XY or smaller standoff.",
            )
            return
        if not robot.check_limits(q_rad):
            _post_status(f"Hover ({ref_label}): pose violates joint limits.")
            return

        if _state.get("stop_requested"):
            _post_status(f"Hover ({ref_label}): stopped before starting motion.")
            return

        angles_deg = np.degrees(q_rad).tolist()

        # Collision pre-check against tablet + workspace + self. Skipped
        # by the master toggle; uses live joints as q_from for valid sweep.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            from .collision import validate_joint_trajectory  # noqa: PLC0415
            from .preview_dialog import (  # noqa: PLC0415
                show_collision_dialog_threadsafe,
            )

            current_q_deg = list(robot_state.angles.deg[:6])
            # gripper_only=True — post-IK joint configs shouldn't re-check
            # arm self-collision (simplified meshes false-positive there).
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
                    f"Hover ({ref_label}): aborted, would collide "
                    f"({reason}{pair_str})."
                )
                _post_status(msg)
                # Release ownership before scheduling so "Send anyway" can
                # re-acquire. ``finally`` honours worker_holds_flag.
                worker_holds_flag = False
                _state["is_hovering"] = False
                # Drop the handle so a STOP between open and Send-anyway no-ops.
                _state["client"] = None

                hover_angles = list(angles_deg)
                hover_label = ref_label
                hover_lx_m = board_local_x_m
                hover_ly_m = board_local_y_m
                hover_so_m = standoff_m

                def _send_anyway() -> None:
                    if _state.get("is_hovering"):
                        return
                    _state["is_hovering"] = True
                    _state["stop_requested"] = False

                    def _thread() -> None:
                        from .panel import _post_status as _ps  # noqa: PLC0415
                        c: Any = None
                        try:
                            from parol6 import RobotClient as _RC  # noqa: PLC0415

                            c = _RC(host=_CONTROLLER_HOST, port=_CONTROLLER_PORT)
                            _state["client"] = c
                            _ps(
                                f"Hover ({hover_label}): moving to "
                                f"({hover_lx_m * 1000:.0f}, {hover_ly_m * 1000:.0f}) mm "
                                f"@ standoff {hover_so_m * 1000:.0f} mm (override)",
                            )
                            rc2 = c.move_j(
                                angles=hover_angles, speed=0.3, accel=0.5,
                                wait=True, timeout=20.0,
                            )
                            if rc2 < 0:
                                _ps(f"Hover ({hover_label}): move halted.")
                            else:
                                _ps(
                                    f"Hover OK ({hover_label}, override): "
                                    f"({hover_lx_m * 1000:.0f}, {hover_ly_m * 1000:.0f}) mm "
                                    f"@ {hover_so_m * 1000:.0f} mm - measure now.",
                                )
                        except Exception as e:  # noqa: BLE001
                            try:
                                _ps(f"Hover override failed: {e}")
                            except Exception:  # noqa: BLE001
                                logger.exception("hover-override thread crashed")
                        finally:
                            # Resume so subsequent jogs aren't blocked by a
                            # halt-disabled controller (SIM_GOTCHAS §13).
                            if c is not None:
                                try:
                                    c.resume()
                                except Exception as e:  # noqa: BLE001
                                    logger.debug(
                                        "hover-override finally: resume() raised: %s",
                                        e,
                                    )
                            _state["is_hovering"] = False
                            _state["stop_requested"] = False

                    threading.Thread(
                        target=_thread, daemon=True,
                        name="hover-send-anyway",
                    ).start()

                show_collision_dialog_threadsafe(
                    message=msg,
                    target_q_deg=hover_angles,
                    on_send_anyway=_send_anyway,
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.debug("hover trajectory pre-check skipped (%s)", e)

        _post_status(
            f"Hover ({ref_label}): moving to "
            f"({board_local_x_m * 1000:.0f}, {board_local_y_m * 1000:.0f}) mm "
            f"@ standoff {standoff_m * 1000:.0f} mm",
        )
        rc = client.move_j(
            angles=angles_deg, speed=0.3, accel=0.5, wait=True, timeout=20.0,
        )
        if rc < 0:
            _post_status(f"Hover ({ref_label}): move halted.")
        else:
            _post_status(
                f"Hover OK ({ref_label}): "
                f"({board_local_x_m * 1000:.0f}, {board_local_y_m * 1000:.0f}) mm "
                f"@ {standoff_m * 1000:.0f} mm — measure now.",
            )
    except Exception as e:  # noqa: BLE001
        if _state.get("stop_requested"):
            _post_status("Hover: stopped by user.")
        else:
            logger.exception("hover thread crashed")
            _post_status(f"Hover ERROR: {e}")
    finally:
        # Re-enable the controller before exit. STOP fires ``halt()``, which
        # sets ``state.enabled=False`` server-side (SIM_GOTCHAS §13); without
        # this resume, subsequent jogs fail with "Controller disabled" until
        # the next worker's start-of-run resume. Only this worker's client
        # is resumed; the "Send anyway" inner thread has its own (and its
        # own start-of-run resume).
        if client is not None and worker_holds_flag:
            try:
                client.resume()
            except Exception as e:  # noqa: BLE001
                logger.debug("hover finally: resume() raised: %s", e)
        # Only clear is_hovering if this worker still owns it; a re-acquired
        # flag (via "Send anyway") belongs to a different worker.
        if worker_holds_flag:
            _state["is_hovering"] = False
        _state["stop_requested"] = False
