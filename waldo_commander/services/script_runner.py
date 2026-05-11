from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, TypedDict

logger = logging.getLogger(__name__)


class ScriptRunConfig(TypedDict):
    """Configuration for running a Python script."""

    filename: str  # absolute path to saved script
    python_exe: str  # sys.executable path
    env: dict[str, str]  # extra environment variables; optional
    cwd: str  # working directory for the script; default project root


class ScriptProcessHandle(TypedDict):
    """Handle for a running script process."""

    proc: asyncio.subprocess.Process
    stdout_task: asyncio.Task
    stderr_task: asyncio.Task
    start_ts: float


async def _stream_output(
    stream: asyncio.StreamReader, callback: Callable[[str], None], prefix: str = ""
) -> None:
    """Read lines from stream and forward to callback with optional prefix."""
    try:
        while True:
            line_bytes = await stream.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="ignore").rstrip()
            if line:
                callback(f"{prefix}{line}")
    except Exception as e:
        logger.error("Stream reader error: %s", e)


async def run_script(
    cfg: ScriptRunConfig,
    on_stdout: Callable[[str], None],
    on_stderr: Callable[[str], None],
    session_id: str | None = None,
) -> ScriptProcessHandle:
    """
    Start a Python script as a subprocess and stream output to callbacks.

    Args:
        cfg: Configuration for the script run
        on_stdout: Callback for stdout lines
        on_stderr: Callback for stderr lines
        session_id: Optional stepping session ID for GUI-controlled execution

    Returns:
        Handle for managing the process

    Raises:
        FileNotFoundError: If script file doesn't exist
        PermissionError: If Python executable not found/executable
        OSError: If process creation fails
    """
    # Validate configuration
    script_path = Path(cfg["filename"])
    if not script_path.exists():
        raise FileNotFoundError(f"Script file not found: {cfg['filename']}")

    if not script_path.suffix == ".py":
        raise ValueError(f"Script must be a .py file: {cfg['filename']}")

    python_exe = cfg["python_exe"]
    if not Path(python_exe).exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")

    # Build environment variables - inherit from parent process and add config overrides
    env = {**os.environ, **cfg.get("env", {})}
    if session_id:
        env["WALDO_STEP_SESSION"] = session_id
    # Pass backend package to subprocess so stepping_bootstrap can patch the right module
    from waldo_commander.state import ui_state

    env["WALDO_BACKEND_PACKAGE"] = ui_state.active_robot.backend_package

    # Forward the mesh-collision master toggle so the subprocess pre-
    # flight matches the GUI's gating. Default "1" when storage is unset.
    try:
        from nicegui import app as _ng_app  # noqa: PLC0415

        env["WALDO_MESH_COLLISION_ENABLED"] = (
            "1"
            if bool(_ng_app.storage.general.get("mesh_collision_check_enabled", True))
            else "0"
        )
    except (ImportError, RuntimeError, AttributeError):
        env["WALDO_MESH_COLLISION_ENABLED"] = "1"

    # Forward the GUI's active tool key. The controller broadcast only
    # carries built-in keys; without this the subprocess would load the
    # proxy's meshes instead of the custom tool's actual meshes.
    selected_tool: str = ""
    try:
        from nicegui import app as _ng_app  # noqa: PLC0415

        selected_tool = str(
            _ng_app.storage.general.get("selected_tool", "") or "",
        )
    except (ImportError, RuntimeError, AttributeError):
        selected_tool = ""
    env["WALDO_GUI_ACTIVE_TOOL_KEY"] = selected_tool

    # Forward the active tool's variant so the subprocess's local IK
    # targets the same TCP the controller uses.
    variant_for_active: str = ""
    if selected_tool:
        try:
            from nicegui import app as _ng_app  # noqa: PLC0415

            variant_for_active = str(
                _ng_app.storage.general.get(
                    f"tool_variant_{selected_tool}", "",
                ) or "",
            )
        except (ImportError, RuntimeError, AttributeError):
            variant_for_active = ""
    env["WALDO_GUI_ACTIVE_TOOL_VARIANT"] = variant_for_active

    # Forward the user TCP offset, converting GUI mm → metres + JSON.
    # Empty string means no override (tool's registered TCP only).
    tcp_offset_for_active: str = ""
    if selected_tool:
        try:
            from nicegui import app as _ng_app  # noqa: PLC0415

            tcp_offset_mm = _ng_app.storage.general.get(
                f"tcp_offset_{selected_tool}",
            )
            if isinstance(tcp_offset_mm, dict):
                x_mm = float(tcp_offset_mm.get("x", 0) or 0)
                y_mm = float(tcp_offset_mm.get("y", 0) or 0)
                z_mm = float(tcp_offset_mm.get("z", 0) or 0)
                if x_mm != 0 or y_mm != 0 or z_mm != 0:
                    import json as _json  # noqa: PLC0415
                    tcp_offset_for_active = _json.dumps(
                        [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0],
                    )
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            tcp_offset_for_active = ""
    env["WALDO_GUI_ACTIVE_TCP_OFFSET_M"] = tcp_offset_for_active

    # Determine which script to run
    if session_id:
        # Use bootstrap script to inject stepping wrapper
        bootstrap_path = Path(__file__).parent / "stepping_bootstrap.py"
        if not bootstrap_path.exists():
            raise FileNotFoundError(f"Bootstrap script not found: {bootstrap_path}")
        exec_args = [python_exe, "-u", str(bootstrap_path), str(script_path)]
    else:
        # Run script directly
        exec_args = [python_exe, "-u", str(script_path)]

    # Create the subprocess
    # On Unix, create a new process group so we can kill the entire tree
    kwargs: dict = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cfg["cwd"],
        "env": env,
    }
    if sys.platform != "win32":
        # Create new process group on Unix
        kwargs["start_new_session"] = True

    proc = await asyncio.create_subprocess_exec(*exec_args, **kwargs)

    # Start streaming tasks
    if proc.stdout:
        stdout_task = asyncio.create_task(_stream_output(proc.stdout, on_stdout))
    else:
        stdout_task = asyncio.create_task(asyncio.sleep(0))  # no-op task

    if proc.stderr:
        stderr_task = asyncio.create_task(
            _stream_output(proc.stderr, on_stderr, "[ERR] ")
        )
    else:
        stderr_task = asyncio.create_task(asyncio.sleep(0))  # no-op task

    handle: ScriptProcessHandle = {
        "proc": proc,
        "stdout_task": stdout_task,
        "stderr_task": stderr_task,
        "start_ts": time.time(),
    }

    logger.info("Started script process: %s (PID: %s)", cfg["filename"], proc.pid)
    return handle


async def stop_script(handle: ScriptProcessHandle, timeout: float = 2.0) -> None:
    """
    Stop a running script process gracefully.

    Args:
        handle: Process handle from run_script
        timeout: Seconds to wait for graceful termination before force kill
    """
    proc = handle["proc"]

    if proc.returncode is not None:
        logger.debug("Script process already terminated (code: %s)", proc.returncode)
        return

    try:
        # On Unix, terminate the entire process group
        if sys.platform != "win32" and proc.pid:
            try:
                # Send SIGTERM to the process group
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, 15)  # SIGTERM
                logger.debug("Sent SIGTERM to process group %s", pgid)
            except (ProcessLookupError, OSError):
                # Fall back to regular terminate if process group doesn't exist
                proc.terminate()
        else:
            # Graceful termination
            proc.terminate()

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            logger.debug("Script process terminated gracefully")
        except asyncio.TimeoutError:
            # Force kill if graceful termination failed
            if sys.platform != "win32" and proc.pid:
                try:
                    # Send SIGKILL to the process group
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, 9)  # SIGKILL
                    logger.debug("Sent SIGKILL to process group %s", pgid)
                except (ProcessLookupError, OSError):
                    # Fall back to regular kill
                    proc.kill()
            else:
                proc.kill()
            await proc.wait()
            logger.warning("Script process force-killed after timeout")

    except ProcessLookupError:
        # Process already dead
        pass
    except Exception as e:
        logger.error("Error stopping script process: %s", e)

    # Cancel streaming tasks
    for task in [handle["stdout_task"], handle["stderr_task"]]:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("Error canceling stream task: %s", e)


def create_default_config(filename: str, cwd: str | None = None) -> ScriptRunConfig:
    """Create a default script configuration."""
    return {
        "filename": filename,
        "python_exe": sys.executable,
        "env": {},
        "cwd": cwd or str(Path.cwd()),
    }
