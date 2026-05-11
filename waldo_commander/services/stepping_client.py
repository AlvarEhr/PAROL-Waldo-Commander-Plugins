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

# One-shot warning flags for the no-tool-selected and blended-bypass paths.
_NO_TOOL_WARNED = False
_BLENDED_BYPASS_WARNED = False


def _resolve_tool_params_for_ik(
    wrapped_client: Any,
) -> tuple[str, str | None, tuple[float, float, float] | None]:
    """Resolve ``(tool_key, variant_key, tcp_offset_m)`` for the subprocess's
    local IK + collision check.

    Precedence: an env ``custom:`` key wins (the controller broadcast
    can't carry it); otherwise trust the client's runtime tool so a
    mid-script ``set_active_tool`` propagates. Variant and TCP offset
    are paired with ``env_key`` and never carried forward to a
    different ``tool_key`` — doing so corrupts the helper-Robot cache.
    """
    env_key = os.environ.get("WALDO_GUI_ACTIVE_TOOL_KEY", "").strip()
    try:
        client_key = getattr(wrapped_client.tool, "key", None) or ""
    except (RuntimeError, AttributeError):
        # parol6 RobotClient.tool raises when no tool is bound.
        client_key = ""

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
        # Env is the only source of truth for custom: keys.
        tool_key = env_key
        variant_key = env_variant
        tcp_offset_m = env_tcp_offset_m
    elif client_key:
        # Trust the client; env values only apply when keys match.
        tool_key = client_key
        if env_key == client_key:
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
        # No client side tool bound.
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


# Methods whose target we can extract for pre-flight. Cartesian methods
# go through local IK; non-WRF frames are skipped. Best-effort — not a
# substitute for controller-side safeties.
_PREFLIGHT_CHECKABLE_METHODS = frozenset(
    {"move_j", "home", "move_l", "move_p", "move_c", "move_s"},
)
_CARTESIAN_METHODS = frozenset({"move_l", "move_p", "move_c", "move_s"})

# Reject IK results that wander > this from the seed — they likely
# picked a different kinematic branch than the controller's continuity-
# seeded IK will. 90° catches elbow-flip / wrist-flip cases.
import numpy as _np  # noqa: E402, PLC0415

_IK_MAX_JOINT_DELTA_RAD: float = float(_np.deg2rad(90.0))

# Lazy per-(tool_key, variant_key, tcp_offset_m) Robot cache. Building a
# Robot loads the URDF and applies set_active_tool so IK targets TCP.
_LOCAL_ROBOT_CACHE: dict[
    tuple[str, str | None, tuple[float, float, float] | None],
    Any,
] = {}


def _get_local_robot(
    tool_key: str = "NONE",
    variant_key: str | None = None,
    tcp_offset_m: tuple[float, float, float] | None = None,
):  # noqa: ANN202
    """Build a parol6.Robot for local IK matching the controller's tool
    transform. Without ``set_active_tool``, Robot.ik would solve for
    flange while the controller solves for TCP. Cached process-wide;
    returns None on import / construction failure (fail-open).
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
        # set_active_tool internally clears the transform on
        # key=="NONE" or identity offsets.
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
    """Solve IK for a Cartesian pose ``[x, y, z, rx, ry, rz]`` (mm + deg).

    Returns degrees if IK converged, FK matches within 1 mm / 1°, and
    the joint delta from the seed stays under ``_IK_MAX_JOINT_DELTA_RAD``.
    FK-verify is necessary because Robot.ik's success flag is unreliable
    near singularities (see hover.py precedent).
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

    # parol6.Robot.ik's success flag is unreliable; verify with FK.
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

    # Skip the pre-flight rather than validate a trajectory the robot
    # won't take: a large delta means a different IK branch than the
    # controller's continuity-seeded solver will pick.
    joint_delta = float(np.max(np.abs(q_arr - seed)))
    if joint_delta > _IK_MAX_JOINT_DELTA_RAD:
        return None

    return np.degrees(q_arr).tolist()


class CollisionPreFlightError(RuntimeError):
    """Raised by the pre-flight check when a script's move would clip a
    static obstacle. Unwinds through the user's script; can be disabled
    via the Settings panel's "Mesh collision check" toggle.
    """


def _maybe_check_collision(
    method_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    wrapped_client: Any,
) -> None:
    """Pre-flight gripper-vs-environment check for one motion call.

    Gripper-only (IK has already constrained joints away from self-
    collision). Master gate is ``WALDO_MESH_COLLISION_ENABLED``. Raises
    :class:`CollisionPreFlightError` on collision; no-ops cleanly when
    the toggle is off, parol6_vision isn't installed, or the method
    isn't checkable.
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

    # Resolve once for both the local IK and the FCL mesh lookup.
    ik_tool_key, ik_variant_key, ik_tcp_offset_m = (
        _resolve_tool_params_for_ik(wrapped_client)
    )

    # Cartesian methods produce one entry per waypoint; the check then
    # walks current → wp1 → ... → wpN so middle waypoints don't slip by.
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
        # Matches parol6 controller semantics: move_l respects rel; the
        # rel=True + frame=WRF case isn't plumbed and is skipped. Other
        # methods have no rel flag and TRF means waypoints relative to
        # start TCP. Unknown frames fall through silently.
        frame = kwargs.get("frame", "WRF")
        if frame not in ("WRF", "TRF"):
            return
        rel = bool(kwargs.get("rel", False))
        if method_name == "move_l" and rel and frame == "WRF":
            return

        # Known limitation: we endpoint-IK between sampled waypoints, so
        # an arc / spline that passes through obstacles BETWEEN waypoints
        # isn't caught. Method signatures differ — move_c takes two
        # positional args (via, end) rather than a waypoints list.
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
        # move_l TRF only composes when rel=True; other methods always do.
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
        # TRF compose always uses ``current_q_deg`` (start TCP) so all
        # waypoints share the same world-frame transform.
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
                # WRF, or move_l TRF + rel=False (pose verbatim as
                # TCP-in-world). Helper Robot's IK targets TCP either way.
                wp_q_deg = _ik_cartesian_to_joints_deg(
                    pose_list, seed_for_ik,
                    tool_key=ik_tool_key,
                    variant_key=ik_variant_key,
                    tcp_offset_m=ik_tcp_offset_m,
                )
            if wp_q_deg is None:
                # Skip the whole check rather than validate only the prefix.
                return
            target_q_deg_list.append(wp_q_deg)
            seed_for_ik = wp_q_deg
    else:
        return

    # Cartesian paths already snapshotted angles() above.
    if method_name in ("move_j", "home"):
        try:
            current = wrapped_client.angles()
            if current is None:
                return
            current_q_deg = list(current)[:6]
        except (OSError, RuntimeError, ValueError) as e:
            # Fail-open: don't strand the script on a transient glitch.
            sys.stderr.write(
                f"[collision pre-flight] angles() unavailable ({type(e).__name__}: {e}); "
                "skipping check for this move.\n",
            )
            sys.stderr.flush()
            return

    tool_key = ik_tool_key
    tool_meshes = resolve_tool_meshes_from_registry(tool_key, mesh_dir)

    # An empty-paths gripper_only manager contains only the FLOOR, so
    # every check would falsely report safe. Skip with a one-shot warn.
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
    # Walk current → waypoint[0] → ... → waypoint[-1]. First failure
    # raises; the user's script aborts before later segments run.
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
                # then execute without waiting or stepping. Bypasses the
                # FCL pre-flight (the server-side blended trajectory isn't
                # available locally); the next non-blended endpoint is.
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

            # Pre-flight gripper-vs-environment check; raises
            # CollisionPreFlightError on collision.
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
