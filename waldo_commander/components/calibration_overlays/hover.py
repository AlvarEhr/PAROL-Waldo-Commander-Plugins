"""Hover-above-board verification mode."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .constants import _SSG48_JAW_VARIANT
from .state import _T_BOARD2BASE, _state

logger = logging.getLogger(__name__)


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
    try:
        from parol6 import Robot, RobotClient  # noqa: PLC0415
        from parol6 import tools as parol6_tools  # noqa: PLC0415
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
            # SSG-48's TCP transform: ``T_flange2tcp`` has translation
            # ``(0, 0, -tcp_offset_z)`` for the active jaw variant. Compose
            # the desired TCP-in-base pose, then back out T_flange2base.
            try:
                T_flange2tcp = parol6_tools.get_tool_transform(
                    "SSG-48", _SSG48_JAW_VARIANT,
                )
            except ValueError as e:
                _post_status(f"Hover (TCP): {e}")
                return
            tcp_origin_world = target_world + standoff_m * direction_up_world
            R_tcp = _build_R_with_z_axis(direction_up_world)
            T_tcp2base = np.eye(4, dtype=np.float64)
            T_tcp2base[:3, :3] = R_tcp
            T_tcp2base[:3, 3] = tcp_origin_world
            T_flange2base = T_tcp2base @ np.linalg.inv(T_flange2tcp)
            ref_label = "TCP"
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

        client = RobotClient(host="127.0.0.1", port=5001)
        _state["client"] = client  # so STOP can halt the hover move
        angles_deg = np.degrees(q_rad).tolist()
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
        _state["is_hovering"] = False
        _state["stop_requested"] = False
