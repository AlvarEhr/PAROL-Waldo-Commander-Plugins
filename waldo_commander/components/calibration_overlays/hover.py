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


# Controller endpoint — same loopback host:port the rest of the
# calibration package uses (calibration_thread + localise + the
# pose_popup go-to-pose dispatcher). Centralised here so a future
# multi-host setup only has to flip one constant.
_CONTROLLER_HOST: str = "127.0.0.1"
_CONTROLLER_PORT: int = 5001


# ---------------------------------------------------------------------------
# Hover-above-board verification mode
# ---------------------------------------------------------------------------


def _build_R_with_z_axis(z_axis_world: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build a 3x3 rotation matrix whose Z column equals ``z_axis_world``.

    The X axis is world +X projected perpendicular to ``z_axis_world`` (or
    world +Y when world +X is nearly parallel). Y is then ``Z × X``. The
    rotation about Z is therefore arbitrary-but-deterministic — fine for
    hover-above-board where the gripper roll doesn't matter for the
    physical measurement, only the position + downward orientation do.
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
    """Look up the currently-active tool's flange→TCP transform.

    Tries the live client's bound tool first (so the lookup tracks
    whatever the user has selected — pneumatic gripper, swapped jaw
    variant, etc.). Falls back to ``parol6.tools.get_tool_transform("SSG-48",
    settings.tool_jaw_variant)`` if no tool is bound on this client yet
    (which can happen if the GUI hasn't called ``select_tool`` on this
    fresh connection).

    Returns ``(T_flange2tcp, tool_label)``. ``tool_label`` is the tool's
    display name (or "SSG-48" on fallback) for the status line.
    """
    from parol6 import tools as parol6_tools  # noqa: PLC0415
    from scipy.spatial.transform import Rotation as SciRot  # noqa: PLC0415

    try:
        spec = client.tool
        tcp_origin = np.asarray(spec.tcp_origin, dtype=np.float64).reshape(3)
        tcp_rpy = np.asarray(spec.tcp_rpy, dtype=np.float64).reshape(3)
        # XYZ extrinsic Euler — see Docs/HANDOFF.md "Conventions".
        R = SciRot.from_euler("XYZ", tcp_rpy, degrees=False).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tcp_origin
        return T, str(spec.display_name)
    except RuntimeError:
        # No tool bound — fall back to the configured default.
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
    """Drive a chosen reference frame to ``(board_local_x, board_local_y,
    standoff_m)`` above the board surface, looking / pointing straight down.
    Run in a daemon thread spawned by the panel's hover buttons.

    Use case: post-calibration physical verification on real hardware. The
    user picks a known XY on the board (corners or centre), commands a
    standoff, and physically measures the reference-to-board distance with
    a ruler/caliper. If the chain is correct, the reference lands at the
    commanded standoff above the commanded XY (within a few mm); if it's
    off, the discrepancy reveals the error magnitude and direction.

    Modes:

    * ``"camera"`` — drive the camera optical centre to ``target +
      standoff × up``, with optical axis pointing AT the target (camera
      looks straight down at the marked point). Validates ``T_cam2flange``
      i.e. the calibration result directly.

    * ``"tcp"`` — drive the gripper TCP (fingertips for the SSG-48
      "finger" variant) to ``target + standoff × up``, with the gripper
      pointing perpendicular to the board surface (flange ``+Z`` aligned
      with the board's local ``+Z``). Validates ``T_tcp2flange`` i.e. the
      kinematic chain from joints to fingertip — independent of
      calibration. Useful as a baseline check that the rest of the chain
      is right before blaming calibration for a calibration error.

    The standoff is measured in BOARD-LOCAL +Z (perpendicular to the board
    surface) — not world +Z — so a tilted board still gets a perpendicular
    standoff.
    """
    # Lazy import to break panel<->hover cycle.
    from .panel import _post_status  # noqa: PLC0415

    # Per-thread ownership token mirrors _state["is_hovering"]. The
    # caller already set is_hovering=True before spawning this thread.
    # If the collision-blocked path releases ownership so the dialog's
    # "Send anyway" callback can re-acquire, the worker's `finally`
    # below MUST NOT clobber the flag - otherwise it races with the
    # new dispatch thread that just set it back to True.
    worker_holds_flag = True

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
        # Board-local +Z in world frame — direction perpendicular to the
        # board surface, pointing AWAY from the ChArUco face.
        direction_up_world = _T_BOARD2BASE[:3, :3] @ np.array(
            [0.0, 0.0, 1.0], dtype=np.float64,
        )
        direction_up_world /= float(np.linalg.norm(direction_up_world))

        # Open the controller client up front so we can:
        #   - look up the active tool's TCP transform via ``client.tool``
        #     (TCP mode only, but cheap regardless),
        #   - hand the same client to STOP via ``_state["client"]``.
        client = RobotClient(host=_CONTROLLER_HOST, port=_CONTROLLER_PORT)
        _state["client"] = client

        # A prior run's STOP / halt() left the controller in disabled state;
        # subsequent move_j calls would fail with "Controller disabled".
        # Resume at the start of every run so consecutive Hover clicks work
        # without the user manually re-enabling. Same fix as localise.py
        # and calibration_thread.py.
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
            # Pull the flange→TCP transform from the active tool (queries
            # ``client.tool``, falls back to SSG-48 default if no tool
            # has been bound on this client). Compose the desired
            # TCP-in-base pose with TCP +Z aligned to the board's
            # local +Z (gripper pointing perpendicular into the board),
            # then back out T_flange2base.
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

        # FK-verify (parol6's IK reports success=False even when q is
        # numerically fine; trust FK + tight tolerance instead).
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

        # Collision pre-check against the calibration board / tablet
        # primitive + workspace + self. Skips when the master toggle
        # ``mesh_collision_check_enabled`` is off (validate returns
        # safe=True without checking). The check uses the live joint
        # broadcast as q_from so the swept trajectory is valid.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            from .collision import validate_joint_trajectory  # noqa: PLC0415
            from .preview_dialog import (  # noqa: PLC0415
                show_collision_dialog_threadsafe,
            )

            current_q_deg = list(robot_state.angles.deg[:6])
            # gripper_only=True: angles_deg is a post-IK joint config
            # (from Robot().ik above). Per the design decision in
            # commit d609024 ("arm self-collision is the IK solver's
            # job"), post-IK targets should not re-check arm-link
            # self-collision against parol6's simplified meshes —
            # those produce known false-positive overlaps that the
            # adjacent-pair whitelist only partially suppresses.
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
                # Release ownership BEFORE scheduling the dialog so the
                # dialog's `Send anyway` callback can re-acquire the
                # busy flag. Worker's `finally` (below) tracks
                # ownership via worker_holds_flag and won't clobber.
                worker_holds_flag = False
                _state["is_hovering"] = False
                # Clear the stale client handle so STOP between dialog
                # open and Send anyway is a clean no-op.
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
        # Only clear is_hovering if THIS worker still owns it. The
        # collision-blocked path releases ownership before scheduling
        # the dialog (worker_holds_flag = False); a subsequent
        # `Send anyway` may have re-acquired the flag - we MUST NOT
        # clobber it.
        if worker_holds_flag:
            _state["is_hovering"] = False
        _state["stop_requested"] = False
