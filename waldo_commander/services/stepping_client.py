"""
Stepping client wrapper for GUI-controlled script execution.

Provides a wrapper around RobotClient that:
1. Emits events for each motion command (start/complete)
2. Optionally pauses after each command for stepping through scripts
3. Communicates with GUI via file-based IPC

Cross-platform compatible (Windows, macOS, Linux).
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

# One-shot warning state for the no-tool-selected fail-open path.
_NO_TOOL_WARNED = False
# One-shot warning state for the blended-motion bypass path. Emitted
# the first time a blended Cartesian / joint motion is intercepted so
# users know the FCL pre-flight is skipped on those.
_BLENDED_BYPASS_WARNED = False


def _resolve_tool_params_for_ik(
    wrapped_client: Any,
) -> tuple[str, str | None, tuple[float, float, float] | None]:
    """Resolve ``(tool_key, variant_key, tcp_offset_m)`` for the
    subprocess's local IK + collision-check.

    ``tool_key`` resolution prefers the GUI's forwarded env var when
    it carries a ``custom:`` key (the controller broadcast carries
    only the proxy ``built-in`` so ``wrapped_client.tool.key`` would
    return e.g. ``"VACUUM"`` even when the GUI is presenting
    ``custom:my_gripper``). For built-in tools we prefer
    ``wrapped_client.tool.key`` so a script that calls
    ``client.set_active_tool(...)`` mid-execution updates the IK
    config. This is a heuristic — a script switching FROM a custom
    tool TO a built-in mid-execution will keep the stale custom: key
    until process restart, but that's an uncommon flow.

    ``variant_key`` and ``tcp_offset_m`` are read from companion env
    vars set by ``script_runner``. Empty / unset means "no override".
    """
    env_key = os.environ.get("WALDO_GUI_ACTIVE_TOOL_KEY", "").strip()
    try:
        client_key = getattr(wrapped_client.tool, "key", None) or ""
    except (RuntimeError, AttributeError):
        # parol6 RobotClient.tool raises RuntimeError when no tool is
        # bound. Treat the same as "no client side tool".
        client_key = ""

    # Read env's variant + tcp_offset early — they're paired with
    # ``env_key`` semantically; carrying them forward to a different
    # tool_key would silently corrupt the local IK's helper Robot
    # cache (e.g. ``Robot.set_active_tool("VACUUM",
    # variant_key="ssg48_tape_jaws")`` falls back to VACUUM's
    # default variant after parol6 logs a warning, but the cache
    # still keys the result against the wrong variant_key).
    env_variant = (
        os.environ.get("WALDO_GUI_ACTIVE_TOOL_VARIANT", "").strip() or None
    )
    env_tcp_offset_m: tuple[float, float, float] | None = None
    tcp_offset_str = os.environ.get("WALDO_GUI_ACTIVE_TCP_OFFSET_M", "").strip()
    if tcp_offset_str:
        try:
            parsed = json.loads(tcp_offset_str)
            if isinstance(parsed, list) and len(parsed) == 3:
                env_tcp_offset_m = (
                    float(parsed[0]), float(parsed[1]), float(parsed[2]),
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            env_tcp_offset_m = None

    if env_key.startswith("custom:"):
        # Controller broadcast doesn't carry custom: keys; the env
        # var is the only source of truth. Env's variant + tcp_offset
        # belong to this tool.
        tool_key = env_key
        variant_key = env_variant
        tcp_offset_m = env_tcp_offset_m
    elif client_key:
        # Built-in tool. Trust the client's runtime state so a
        # mid-script set_active_tool propagates. The env's
        # variant + tcp_offset are paired with ``env_key``; if the
        # client's tool differs, those env values are stale and
        # must NOT be carried forward — fall back to whatever the
        # client exposes (variant_key best-effort; tcp_offset_m
        # isn't on the parol6 client API so falls back to None).
        tool_key = client_key
        if env_key == client_key:
            # No mid-script change happened; env values still
            # apply to this tool.
            variant_key = env_variant
            tcp_offset_m = env_tcp_offset_m
        else:
            try:
                variant_key = (
                    getattr(wrapped_client.tool, "variant_key", None) or None
                )
            except (RuntimeError, AttributeError):
                variant_key = None
            tcp_offset_m = None
    else:
        # No client side tool bound. Fall back to env or NONE.
        tool_key = env_key or "NONE"
        variant_key = env_variant
        tcp_offset_m = env_tcp_offset_m

    return tool_key, variant_key, tcp_offset_m

from .path_preview_client import MOTION_METHODS

# Methods that trigger wait_command and stepping behavior.
# Includes all motion methods plus non-motion commands that queue on the controller.
STEPPABLE_METHODS = frozenset(MOTION_METHODS) | frozenset(
    {"home", "tool_action", "delay"}
)


def _atomic_write(path: Path, data: dict) -> None:
    """Write data to file atomically using temp file + move."""
    temp_path = path.with_suffix(".tmp")
    try:
        temp_path.write_text(json.dumps(data, indent=2))
        shutil.move(str(temp_path), str(path))
    except Exception:
        # Clean up temp file if move failed
        if temp_path.exists():
            temp_path.unlink()
        raise


def _read_control(control_file: Path) -> dict:
    """Read control file, return defaults if not exists or parse error."""
    try:
        return json.loads(control_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {"paused": True, "step_signal": 0, "step_acked": 0}


class StepIO:
    """
    File-based IPC for stepping control between script subprocess and GUI.

    Uses two files:
    - Control file (GUI -> Script): Contains paused flag and step signals
    - Event file (Script -> GUI): Contains command start/complete events
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._temp_dir = Path(tempfile.gettempdir())
        self._control_file = self._temp_dir / f".parol_control_{session_id}"
        self._event_file = self._temp_dir / f".parol_events_{session_id}"
        self._step_count = 0
        self._last_step_acked = 0

    @classmethod
    def from_env(cls) -> "StepIO | None":
        """
        Create StepIO from environment variables.
        Returns None if WALDO_STEP_SESSION is not set.
        """
        session_id = os.environ.get("WALDO_STEP_SESSION")
        if not session_id:
            return None
        return cls(session_id)

    def _read_events(self) -> list[dict]:
        """Read events from event file."""
        try:
            data = json.loads(self._event_file.read_text())
            return data.get("events", [])
        except (json.JSONDecodeError, OSError):
            return []

    def emit_event(self, event_type: str, method: str, **extra: Any) -> None:
        """
        Emit an event to the event file.

        Args:
            event_type: "start" or "complete"
            method: Name of the motion method
            **extra: Additional event data
        """
        events = self._read_events()
        events.append(
            {
                "event": event_type,
                "method": method,
                "step": self._step_count,
                "ts": time.time(),
                **extra,
            }
        )
        _atomic_write(self._event_file, {"events": events})

    def check_should_pause(self) -> bool:
        """Check if the script should pause (paused flag is true)."""
        control = _read_control(self._control_file)
        return control.get("paused", True)

    def wait_for_step_or_play(
        self, timeout: float = 300.0, poll_interval: float = 0.05
    ) -> bool:
        """
        Wait until either:
        - step_signal > step_acked (step forward requested)
        - paused becomes False (play mode activated)

        Returns True if should continue, False on timeout.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            control = _read_control(self._control_file)

            # If paused is False, we're in play mode - continue immediately
            if not control.get("paused", True):
                return True

            # Check if a step signal was sent
            step_signal = control.get("step_signal", 0)
            step_acked = control.get("step_acked", 0)

            if step_signal > step_acked:
                # Step requested - acknowledge it
                self._ack_step(control, step_signal)
                return True

            time.sleep(poll_interval)

        return False  # Timeout

    def _ack_step(self, control: dict, step_signal: int) -> None:
        """Acknowledge a step by incrementing step_acked."""
        control["step_acked"] = step_signal
        _atomic_write(self._control_file, control)

    def increment_step_count(self) -> None:
        """Increment the internal step counter."""
        self._step_count += 1


_STEPPABLE_TOOL_METHODS = frozenset({"set_position", "open", "close", "calibrate"})


# Methods whose joint-space target we can extract for pre-flight.
# move_j / home land directly in joint space; move_l / move_p / move_c /
# move_s take Cartesian targets and we run local IK (parol6.Robot.ik)
# to map them. Cartesian moves with frame!="WRF" are skipped (TRF
# needs current TCP composition; deferred). The FCL pre-flight is
# best-effort and is not a substitute for the controller-side soft-
# stop / current-limit safeties.
_PREFLIGHT_CHECKABLE_METHODS = frozenset(
    {"move_j", "home", "move_l", "move_p", "move_c", "move_s"},
)
_CARTESIAN_METHODS = frozenset({"move_l", "move_p", "move_c", "move_s"})

# Joint-delta sanity threshold: when the local IK result wanders more
# than this from the seed (current pose), it likely landed in a
# different kinematic branch than the controller's continuity-seeded
# IK will pick. We reject the IK result and fall through unchecked
# (the controller IK is the source of truth and will reject if truly
# unreachable). 90 deg per joint is permissive enough for typical
# Cartesian moves while catching the obvious elbow-flip / wrist-flip
# failures.
import numpy as _np  # noqa: E402, PLC0415

_IK_MAX_JOINT_DELTA_RAD: float = float(_np.deg2rad(90.0))

# Cached parol6.Robot instances for local IK in the subprocess. Each
# distinct (tool_key, variant_key, tcp_offset_m) combination gets its
# own Robot built lazily — constructing the Robot loads the URDF +
# initialises pinokin (~0.1-0.5s) and applies set_active_tool so its
# IK targets TCP poses (matching the controller). Pure-move_j programs
# never pay this cost.
_LOCAL_ROBOT_CACHE: dict[
    tuple[str, str | None, tuple[float, float, float] | None],
    Any,
] = {}


def _get_local_robot(
    tool_key: str = "NONE",
    variant_key: str | None = None,
    tcp_offset_m: tuple[float, float, float] | None = None,
):  # noqa: ANN202
    """Lazily build a parol6.Robot for local IK, configured with the
    same tool transform the controller has applied.

    Without ``set_active_tool``, ``Robot.ik`` solves "joints such that
    flange = pose"; the controller (with its tool transform applied)
    solves "joints such that TCP = pose". For Cartesian moves with
    frame="WRF" (and rel=False), the user's pose is interpreted as TCP
    by the controller — pre-flight that runs flange-frame IK would
    validate a different joint trajectory than the controller will
    execute, off by the tool's TCP offset. Passing the tool config
    here keeps both ends consistent.

    Cached process-wide by ``(tool_key, variant_key, tcp_offset_m)``.

    Returns the Robot instance on success or None when parol6 isn't
    importable or construction fails (subprocess fails-open: skip the
    Cartesian pre-flight rather than block the user's script).
    """
    cache_key = (tool_key, variant_key, tcp_offset_m)
    cached = _LOCAL_ROBOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from parol6 import Robot  # noqa: PLC0415
    except ImportError:
        return None
    try:
        robot = Robot()
        # Apply the same tool transform the controller has so local
        # IK targets TCP poses, not flange poses. set_active_tool
        # internally clears the transform when key=="NONE" or the
        # resolved transform reduces to identity.
        if tool_key:
            robot.set_active_tool(
                tool_key,
                tcp_offset_m=tcp_offset_m,
                variant_key=variant_key,
            )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(
            f"[collision pre-flight] failed to construct local "
            f"parol6.Robot for IK (tool_key={tool_key!r}, "
            f"variant_key={variant_key!r}, tcp_offset_m={tcp_offset_m!r}): "
            f"{type(e).__name__}: {e}; "
            "skipping Cartesian collision pre-flight.\n",
        )
        sys.stderr.flush()
        return None
    _LOCAL_ROBOT_CACHE[cache_key] = robot
    return robot


def _ik_cartesian_to_joints_deg(
    pose_mm_deg: list[float],
    seed_deg: list[float],
    tool_key: str = "NONE",
    variant_key: str | None = None,
    tcp_offset_m: tuple[float, float, float] | None = None,
) -> list[float] | None:
    """Solve IK for a Cartesian pose, returning joint angles in degrees.

    The pose is interpreted in the same frame as the helper Robot's
    IK convention: when ``tool_key`` resolves to a real tool,
    ``Robot.set_active_tool`` is applied and ``ik()`` solves "joints
    such that TCP = pose" (matching the controller). When
    ``tool_key="NONE"`` the helper Robot has no tool transform and
    ``ik()`` solves for flange.

    Args:
        pose_mm_deg: 6-vector ``[x, y, z, rx, ry, rz]`` in mm + deg
            (parol6's standard external-API convention). For absolute
            WRF / TCP-targeted moves this is a TCP pose; for the
            ``trf_to_world_pose_mm_deg`` output it's already a TCP
            pose composed from the live tool transform.
        seed_deg: Current joint angles in degrees, used as the IK seed.
        tool_key: forwarded to ``_get_local_robot`` — must match what
            the controller has applied so the IK frame agrees.
        variant_key: variant of the active tool, forwarded to
            ``_get_local_robot``.
        tcp_offset_m: user TCP offset (m) the controller has composed
            on top of the tool's base TCP. Forwarded so cache key
            matches and the helper Robot's TCP includes it.

    Returns:
        Joint angles in degrees if IK converged AND FK-verified to
        within 1 mm / 1 deg of the requested pose AND the joint
        configuration didn't wander more than
        ``_IK_MAX_JOINT_DELTA_RAD`` from the seed (which would
        indicate a wrong-branch solution unlikely to match the
        controller's continuity-seeded IK). None otherwise.

        FK-verify is necessary because parol6.Robot.ik's ``success``
        flag is unreliable on near-singular configurations (per the
        existing hover.py precedent). The joint-delta sanity check
        catches elbow-flip / wrist-flip cases where FK matches but
        the joint trajectory the controller will execute is different
        from the one we'd validate.
    """
    import numpy as np  # noqa: PLC0415
    from scipy.spatial.transform import Rotation as _R  # noqa: PLC0415

    robot = _get_local_robot(
        tool_key=tool_key,
        variant_key=variant_key,
        tcp_offset_m=tcp_offset_m,
    )
    if robot is None:
        return None

    pose = np.asarray(pose_mm_deg, dtype=np.float64)
    if pose.shape != (6,):
        return None
    pose_m_rad = np.array(
        [
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            np.radians(pose[3]),
            np.radians(pose[4]),
            np.radians(pose[5]),
        ],
        dtype=np.float64,
    )
    seed = np.radians(np.asarray(seed_deg, dtype=np.float64))

    try:
        result = robot.ik(pose_m_rad, seed)
    except Exception:  # noqa: BLE001
        return None

    q = getattr(result, "q", None)
    if q is None:
        return None
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.shape != (6,):
        return None

    # FK-verify: parol6.Robot.ik reports success=False on poses that
    # actually converged within tolerance.
    fk_pose = np.zeros(6, dtype=np.float64)
    try:
        robot.fk(q_arr, fk_pose)
    except Exception:  # noqa: BLE001
        return None
    pos_err = float(np.linalg.norm(fk_pose[:3] - pose_m_rad[:3]))
    R_target = _R.from_euler("XYZ", pose_m_rad[3:]).as_matrix()
    R_actual = _R.from_euler("XYZ", fk_pose[3:]).as_matrix()
    rot_err_rad = float(
        np.arccos(np.clip((np.trace(R_target.T @ R_actual) - 1) / 2, -1, 1)),
    )
    if pos_err > 0.001 or rot_err_rad > np.radians(1.0):
        return None

    # Joint-delta sanity. parol6's controller-side IK is continuity-
    # seeded (picks the branch closest to the current joints); a local
    # IK seeded with the same current_q should agree on the branch and
    # produce a small joint delta. A large delta indicates we picked a
    # different branch (elbow-up vs elbow-down, wrist-flip) that FK-
    # verifies but won't match the controller's actual execution. The
    # safer behavior is to skip the pre-flight than to validate a
    # trajectory the robot won't take.
    joint_delta = float(np.max(np.abs(q_arr - seed)))
    if joint_delta > _IK_MAX_JOINT_DELTA_RAD:
        return None

    return np.degrees(q_arr).tolist()


class CollisionPreFlightError(RuntimeError):
    """Raised by the program runner's pre-flight check when a script's
    move would clip a static obstacle (floor, gripper-vs-arm, etc.).

    The exception unwinds through the user's script naturally and the
    GUI's `_monitor_script_completion` reset path streams the message
    to the program log. Users can override by flipping the
    "Mesh collision check" toggle off in the bottom-right Settings tab.
    """


def _maybe_check_collision(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    wrapped_client: Any,
) -> None:
    """Pre-flight gripper-vs-environment collision check for a single
    motion call inside the program-runner subprocess.

    The check is gripper-only (skips arm-vs-arm self-collision; IK has
    already constrained the joints to a non-self-intersecting config).
    Master gate is the ``WALDO_MESH_COLLISION_ENABLED`` env var, set by
    ``script_runner.py`` at subprocess launch from the GUI's storage.

    No-ops cleanly when:

    * The master toggle is off.
    * ``method_name`` isn't one of move_j / home (move_l / move_p need
      IK we don't run here).
    * parol6-vision isn't importable (collision_core unavailable).
    * The active tool's mesh dir can't be located.

    Raises :class:`CollisionPreFlightError` on a real collision so the
    user's script aborts with a visible traceback.
    """
    if os.environ.get("WALDO_MESH_COLLISION_ENABLED", "1") != "1":
        return
    if method_name not in _PREFLIGHT_CHECKABLE_METHODS:
        return

    try:
        from parol6_vision.calibration.collision_core import (  # noqa: PLC0415
            CollisionEnvironmentConfig,
            parol6_mesh_dir,
            resolve_tool_meshes_from_registry,
            validate_joint_trajectory_core,
        )
    except ImportError:
        return

    mesh_dir = parol6_mesh_dir()
    if mesh_dir is None:
        return

    # Resolve the tool configuration ONCE for both the local IK
    # (apply set_active_tool so kinematics target TCP poses, matching
    # the controller) and the FCL mesh lookup below. Without
    # set_active_tool the local Robot would interpret WRF Cartesian
    # poses as flange-frame while the controller treats them as TCP-
    # frame, validating a different joint trajectory than executes.
    ik_tool_key, ik_variant_key, ik_tcp_offset_m = (
        _resolve_tool_params_for_ik(wrapped_client)
    )

    # Resolve target joint configs. For Cartesian methods the target
    # is a list (one per waypoint); the trajectory check then runs
    # current -> wp1 -> wp2 ... -> wpN segment-by-segment so middle
    # waypoints can't slip through unchecked. For joint-space methods
    # there's a single target.
    target_q_deg_list: list[list[float]] = []

    if method_name == "move_j":
        target = kwargs.get("angles")
        if target is None and args:
            target = args[0]
        if target is None:
            return
        try:
            target_q_deg_list = [list(target)]
        except TypeError:
            return
    elif method_name == "home":
        try:
            from parol6.config import HOME_ANGLES_DEG  # noqa: PLC0415
        except ImportError:
            return
        target_q_deg_list = [list(HOME_ANGLES_DEG)]
    elif method_name in _CARTESIAN_METHODS:
        # Cartesian moves: matches parol6's controller-side semantics.
        # - move_l: respects ``rel`` arg. ``rel=False`` (default)
        #   treats pose as absolute world target regardless of frame
        #   (matches cartesian_commands._compute_target_pose).
        #   ``rel=True + frame=TRF`` post-multiplies a delta on the
        #   current TCP. ``rel=True + frame=WRF`` pre-multiplies a
        #   delta on the initial pose; we don't plumb that and skip.
        # - move_p / move_c / move_s: no rel flag in the client API.
        #   TRF means "all waypoints relative to START TCP" (matches
        #   curved_commands._transform_waypoints_trf_to_wrf using
        #   the same start tool_pose for every waypoint).
        # Unknown frames fall through silently (controller IK is the
        # source of truth).
        frame = kwargs.get("frame", "WRF")
        if frame not in ("WRF", "TRF"):
            return
        rel = bool(kwargs.get("rel", False))
        if method_name == "move_l" and rel and frame == "WRF":
            # delta @ initial_pose semantics not plumbed; let the
            # controller validate.
            return

        # Build the list of Cartesian waypoints. Each move method
        # takes a different signature:
        #
        # * ``move_l(pose, *, frame=...)`` — single pose.
        # * ``move_c(via, end, *, frame=...)`` — TWO separate positional
        #   args (NOT a single waypoints list), via point and end point.
        # * ``move_p(waypoints, *, frame=...)`` — list of poses.
        # * ``move_s(waypoints, *, frame=...)`` — list of poses.
        #
        # NOTE: move_c interpolates an arc through (current, via, end);
        # move_s interpolates a smooth spline through ``waypoints``.
        # We endpoint-IK and joint-space interpolate between sampled
        # waypoints, so a curve passing through obstacles BETWEEN
        # waypoints isn't caught — known limitation, document at the
        # call site.
        poses_cartesian: list[list[float]] = []
        if method_name == "move_l":
            pose = kwargs.get("pose")
            if pose is None and args:
                pose = args[0]
            if pose is None:
                return
            try:
                poses_cartesian = [list(pose)]
            except TypeError:
                return
        elif method_name == "move_c":
            via = kwargs.get("via")
            end = kwargs.get("end")
            if via is None and len(args) >= 1:
                via = args[0]
            if end is None and len(args) >= 2:
                end = args[1]
            if via is None or end is None:
                return
            try:
                poses_cartesian = [list(via), list(end)]
            except TypeError:
                return
        else:
            waypoints = kwargs.get("waypoints")
            if waypoints is None and args:
                waypoints = args[0]
            if not waypoints:
                return
            try:
                poses_cartesian = [list(wp) for wp in waypoints]
            except TypeError:
                return
        # Need the live joints snapshot for the start-TCP compose AND
        # the seed for the first IK.
        try:
            current = wrapped_client.angles()
            if current is None:
                return
            current_q_deg = list(current)[:6]
        except (OSError, RuntimeError, ValueError):
            return
        # TRF needs the active tool's TCP transform. We pull it
        # lazily here so WRF programs don't pay the lookup cost.
        # The "TRF means compose" decision is per-method:
        # - move_l: only when rel=True (rel=False uses pose verbatim
        #   as world).
        # - move_p / move_c / move_s: always (no rel flag).
        compose_trf = frame == "TRF" and (
            method_name != "move_l" or rel
        )
        tcp_origin: tuple[float, float, float] | None = None
        tcp_rpy: tuple[float, float, float] | None = None
        if compose_trf:
            try:
                tool = wrapped_client.tool
                tcp_origin = tuple(tool.tcp_origin)
                tcp_rpy = tuple(tool.tcp_rpy)
            except (RuntimeError, AttributeError):
                return  # no tool bound; can't compose TRF -> skip
            try:
                from parol6_vision.runtime.safe_motion import (  # noqa: PLC0415
                    trf_to_world_pose_mm_deg,
                )
            except ImportError:
                return
        # Seed for IK is the current-or-prior solution; the TRF
        # compose itself always uses ``current_q_deg`` (start TCP)
        # so all waypoints share the same world-frame transform.
        seed_for_ik = current_q_deg
        for pose_list in poses_cartesian:
            if compose_trf:
                world_pose = trf_to_world_pose_mm_deg(
                    pose_list, current_q_deg, tcp_origin, tcp_rpy,
                    tool_key=ik_tool_key,
                    variant_key=ik_variant_key,
                    tcp_offset_m_for_ik=ik_tcp_offset_m,
                )
                if world_pose is None:
                    return
                wp_q_deg = _ik_cartesian_to_joints_deg(
                    world_pose, seed_for_ik,
                    tool_key=ik_tool_key,
                    variant_key=ik_variant_key,
                    tcp_offset_m=ik_tcp_offset_m,
                )
            else:
                # WRF, OR move_l with TRF + rel=False (pose verbatim
                # as TCP-in-world). Either way, the helper Robot has
                # set_active_tool applied so its IK targets TCP.
                wp_q_deg = _ik_cartesian_to_joints_deg(
                    pose_list, seed_for_ik,
                    tool_key=ik_tool_key,
                    variant_key=ik_variant_key,
                    tcp_offset_m=ik_tcp_offset_m,
                )
            if wp_q_deg is None:
                # Cannot validate this segment; skip the whole check
                # rather than checking only the prefix.
                return
            target_q_deg_list.append(wp_q_deg)
            seed_for_ik = wp_q_deg
    else:
        return

    # For Cartesian paths the angles() snapshot was already taken
    # above as the IK seed. For move_j / home, take it here.
    if method_name in ("move_j", "home"):
        try:
            current = wrapped_client.angles()
            if current is None:
                return
            current_q_deg = list(current)[:6]
        except (OSError, RuntimeError, ValueError) as e:
            # Controller is unreachable / non-responsive / malformed
            # response. Fail-open so the user's script isn't stranded
            # by a transient network glitch.
            sys.stderr.write(
                f"[collision pre-flight] angles() unavailable ({type(e).__name__}: {e}); "
                "skipping check for this move.\n",
            )
            sys.stderr.flush()
            return

    # Reuse the tool-key resolved up top for both the IK transform
    # and the FCL mesh lookup. The heuristic in
    # ``_resolve_tool_params_for_ik`` prefers env-var ``custom:``
    # keys (controller broadcast can't carry them) and otherwise
    # trusts the client's runtime tool state so mid-script tool
    # changes propagate.
    tool_key = ik_tool_key
    tool_meshes = resolve_tool_meshes_from_registry(tool_key, mesh_dir)

    # No-tool detection: a manager built with an empty body_paths +
    # empty jaw_paths and gripper_only=True contains only the FLOOR
    # primitive — there's no movable object to collide it with, so
    # every check would falsely report safe. Surface this as a
    # one-shot stderr warning and skip the check (rather than running
    # an expensive but useless FCL pass). Users can correct by
    # selecting a tool in the GUI before running the program.
    body_paths = tuple(tool_meshes["BODY"])
    jaw_paths = tuple(tool_meshes["JAW"])
    if tool_key == "NONE" or (not body_paths and not jaw_paths):
        global _NO_TOOL_WARNED
        if not _NO_TOOL_WARNED:
            sys.stderr.write(
                "[collision pre-flight] no gripper tool selected "
                f"(tool_key={tool_key!r}); skipping checks for this run. "
                "Select a tool in the GUI to enable collision pre-flight.\n",
            )
            sys.stderr.flush()
            _NO_TOOL_WARNED = True
        return

    config = CollisionEnvironmentConfig(
        gripper_only=True,
        tool_key=tool_key,
        body_mesh_paths=body_paths,
        jaw_mesh_paths=jaw_paths,
        tablet_T_board2base=None,
        tablet_dimensions_m=None,
        floor_enabled=True,
        safety_margin_m=0.008,
    )
    # Walk the segment list: current -> waypoint[0] -> waypoint[1]
    # -> ... -> waypoint[-1]. Each segment runs the same FCL check.
    # First failure raises; subsequent waypoints aren't validated
    # (the user's script will abort here).
    seg_from = current_q_deg
    for seg_idx, seg_to in enumerate(target_q_deg_list):
        result = validate_joint_trajectory_core(
            seg_from, seg_to, config=config,
        )
        if not result.get("safe", True):
            reason = result.get("reason", "collision")
            seg_label = (
                f"{method_name}()"
                if len(target_q_deg_list) == 1
                else f"{method_name}() segment {seg_idx + 1}/{len(target_q_deg_list)}"
            )
            raise CollisionPreFlightError(
                f"{seg_label} pre-flight aborted, would collide "
                f"({reason}). q_from={seg_from}, q_to={seg_to}. "
                "Flip 'Mesh collision check' off in Settings to override."
            )
        seg_from = seg_to


class _SteppingToolProxy:
    """Proxy that wraps a sync tool's action methods with stepping behavior."""

    def __init__(self, sync_tool: Any, step_io: StepIO) -> None:
        self._tool = sync_tool
        self._step_io = step_io

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._tool, name)
        if not callable(attr) or name not in _STEPPABLE_TOOL_METHODS:
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._step_io.emit_event("start", "tool_action")
            result = attr(*args, **kwargs)
            self._step_io.emit_event("complete", "tool_action")
            self._step_io.increment_step_count()
            if self._step_io.check_should_pause():
                self._step_io.wait_for_step_or_play()
            return result

        return wrapper


class SteppingClientWrapper:
    """
    Wrapper around RobotClient that adds stepping behavior.

    - Intercepts motion methods
    - Emits start/complete events for GUI visualization
    - Calls wait_command() after each motion command
    - Optionally pauses for stepping based on control file
    - Blended commands (r > 0) are grouped as a single step
    """

    def __init__(self, wrapped_client: Any, step_io: StepIO) -> None:
        """
        Initialize the wrapper.

        Args:
            wrapped_client: The RobotClient instance to wrap
            step_io: StepIO instance for IPC
        """
        self._wrapped = wrapped_client
        self._step_io = step_io
        self._in_blend = False
        self._last_blend_index: int = -1

    def _flush_blend(self) -> None:
        """Flush any pending blend group, emit events, and pause if stepping."""
        if not self._in_blend:
            return
        if self._last_blend_index >= 0:
            self._wrapped.wait_command(self._last_blend_index)
        self._in_blend = False
        self._last_blend_index = -1
        self._step_io.emit_event("complete", "blend_group")
        self._step_io.increment_step_count()
        if self._step_io.check_should_pause():
            self._step_io.wait_for_step_or_play()

    @property
    def tool(self):
        """Return the sync tool with stepping behavior on action methods."""
        self._flush_blend()
        return _SteppingToolProxy(self._wrapped.tool, self._step_io)

    def __enter__(self) -> "SteppingClientWrapper":
        self._wrapped.__enter__()
        return self

    def __exit__(self, *args: Any) -> bool | None:
        # Flush any pending blend before closing
        if self._in_blend:
            # Only wait/complete if not exiting due to an exception
            if args[0] is None:
                if self._last_blend_index >= 0:
                    self._wrapped.wait_command(self._last_blend_index)
                self._step_io.emit_event("complete", "blend_group")
                self._step_io.increment_step_count()
            self._in_blend = False
            self._last_blend_index = -1
        return self._wrapped.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to wrapped client.
        Intercept motion methods to add stepping behavior.
        """
        attr = getattr(self._wrapped, name)

        if name in STEPPABLE_METHODS and callable(attr):
            return self._wrap_motion_method(name, attr)

        self._flush_blend()
        return attr

    @staticmethod
    def _is_blended(kwargs: dict) -> bool:
        """Check if motion kwargs specify a blend radius."""
        return float(kwargs.get("r", 0)) > 0

    def _wrap_motion_method(self, name: str, method: Callable) -> Callable:
        """Create a wrapper function for a motion method."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            is_blended = self._is_blended(kwargs)

            if is_blended:
                # Blended command — emit start event on first blend command,
                # then execute without waiting or stepping.
                #
                # NOTE: blended motions BYPASS the FCL pre-flight check
                # below. The controller's blended trajectory is
                # interpolated server-side between waypoints; we don't
                # have the swept curve locally, so we can't validate
                # it the way we validate non-blended endpoints. The
                # next non-blended endpoint (or blend-group flush) is
                # still validated. Emit a one-shot stderr breadcrumb
                # the first time a blended motion is intercepted so
                # users running with ``WALDO_MESH_COLLISION_ENABLED=1``
                # know the gap exists.
                global _BLENDED_BYPASS_WARNED
                if (
                    not _BLENDED_BYPASS_WARNED
                    and name in _PREFLIGHT_CHECKABLE_METHODS
                    and os.environ.get("WALDO_MESH_COLLISION_ENABLED", "1") == "1"
                ):
                    sys.stderr.write(
                        f"[collision pre-flight] blended motion (r>0 on "
                        f"{name}()) bypasses the FCL pre-flight. The "
                        "controller's blended trajectory interpolates "
                        "between waypoints server-side, so a fresh check "
                        "isn't run for the blended segment. The next "
                        "non-blended endpoint will be validated. "
                        "(One-shot warning per subprocess.)\n",
                    )
                    sys.stderr.flush()
                    _BLENDED_BYPASS_WARNED = True
                if not self._in_blend:
                    self._step_io.emit_event("start", name, blend=True)
                    self._in_blend = True
                result = method(*args, **kwargs)
                if isinstance(result, int) and result >= 0:
                    self._last_blend_index = result
                return result

            # Non-blended command — flush any pending blend group first
            if self._in_blend:
                if self._last_blend_index >= 0:
                    self._wrapped.wait_command(self._last_blend_index)
                self._in_blend = False
                self._last_blend_index = -1
                self._step_io.emit_event("complete", "blend_group")
                self._step_io.increment_step_count()

            self._step_io.emit_event("start", name)

            # Pre-flight gripper-vs-environment collision check. Raises
            # CollisionPreFlightError on a real collision, which unwinds
            # through the user's script. No-op when the master toggle is
            # off or when the method isn't a checkable joint-space move.
            _maybe_check_collision(name, args, kwargs, self._wrapped)

            # Call the actual method
            result = method(*args, **kwargs)

            # Wait for command to complete
            if isinstance(result, int) and result >= 0:
                self._wrapped.wait_command(result)

            # Emit complete event
            self._step_io.emit_event("complete", name)

            # Increment step counter for next command
            self._step_io.increment_step_count()

            # Check if we should pause for stepping
            if self._step_io.check_should_pause():
                self._step_io.wait_for_step_or_play()

            return result

        return wrapper


class GUIStepController:
    """
    GUI-side controller for stepping.
    Used by the GUI to control script execution via IPC files.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._temp_dir = Path(tempfile.gettempdir())
        self._control_file = self._temp_dir / f".parol_control_{session_id}"
        self._event_file = self._temp_dir / f".parol_events_{session_id}"
        self._last_event_count = 0

    def initialize(self) -> None:
        """Initialize control file with default state (paused=True)."""
        _atomic_write(
            self._control_file,
            {
                "paused": True,
                "step_signal": 0,
                "step_acked": 0,
            },
        )
        # Clear any existing events
        _atomic_write(self._event_file, {"events": []})

    def signal_step(self) -> None:
        """Signal the script to execute one command then pause."""
        control = _read_control(self._control_file)
        control["paused"] = True
        control["step_signal"] = control.get("step_signal", 0) + 1
        _atomic_write(self._control_file, control)

    def signal_play(self) -> None:
        """Signal the script to continue without pausing (play mode)."""
        control = _read_control(self._control_file)
        control["paused"] = False
        _atomic_write(self._control_file, control)

    def signal_pause(self) -> None:
        """Signal the script to pause after the current command."""
        control = _read_control(self._control_file)
        control["paused"] = True
        _atomic_write(self._control_file, control)

    def poll_events(self) -> list[dict]:
        """
        Poll for new events from the script.
        Returns list of new events since last poll.
        """
        try:
            data = json.loads(self._event_file.read_text())
            events = data.get("events", [])
            new_events = events[self._last_event_count :]
            self._last_event_count = len(events)
            return new_events
        except (json.JSONDecodeError, OSError):
            return []

    def get_step_count(self) -> int:
        """Get the current step count from events."""
        try:
            data = json.loads(self._event_file.read_text())
            events = data.get("events", [])
            return sum(1 for e in events if e.get("event") == "complete")
        except (json.JSONDecodeError, OSError):
            return 0

    def cleanup(self) -> None:
        """Remove IPC files."""
        try:
            if self._control_file.exists():
                self._control_file.unlink()
        except OSError:
            pass
        try:
            if self._event_file.exists():
                self._event_file.unlink()
        except OSError:
            pass
