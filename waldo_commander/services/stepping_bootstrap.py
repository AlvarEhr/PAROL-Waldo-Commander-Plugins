#!/usr/bin/env python3
"""
Bootstrap script for running user scripts with stepping wrapper.

This script is run as the main entry point when GUI-controlled stepping is enabled.
It patches parol6.RobotClient to wrap it with SteppingClientWrapper, then executes
the user's script.

Usage:
    python stepping_bootstrap.py <script_path>

Environment:
    WALDO_STEP_SESSION: Required. Session ID for IPC with GUI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    """Bootstrap and run user script with stepping wrapper."""
    if len(sys.argv) < 2:
        print("Usage: stepping_bootstrap.py <script_path>", file=sys.stderr)
        sys.exit(1)

    script_path = Path(sys.argv[1])
    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    session_id = os.environ.get("WALDO_STEP_SESSION")
    if not session_id:
        print("WALDO_STEP_SESSION environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Import and set up the stepping wrapper
    import importlib

    from waldo_commander.services.stepping_client import (
        SteppingClientWrapper,
        StepIO,
    )

    step_io = StepIO(session_id)

    # Read backend package from environment (set by the GUI process)
    backend_package = os.environ.get("WALDO_BACKEND_PACKAGE", "parol6")

    # Import the backend and patch RobotClient
    try:
        backend = importlib.import_module(backend_package)
        OriginalRobotClient = backend.RobotClient

        # Store original for reference
        _original_robot_client = OriginalRobotClient

        # Create a factory that wraps the client
        class WrappedRobotClient:
            """RobotClient replacement that wraps with SteppingClientWrapper."""

            def __new__(cls, *args, **kwargs):
                # Create the original client
                original = _original_robot_client(*args, **kwargs)
                # Wrap it with stepping wrapper
                return SteppingClientWrapper(original, step_io)

        # Patch backend module
        setattr(backend, "RobotClient", WrappedRobotClient)
        if hasattr(backend, "client"):
            setattr(backend.client, "RobotClient", WrappedRobotClient)

        # Also patch sys.modules entries
        if backend_package in sys.modules:
            setattr(sys.modules[backend_package], "RobotClient", WrappedRobotClient)
        client_mod_name = f"{backend_package}.client"
        if client_mod_name in sys.modules:
            setattr(sys.modules[client_mod_name], "RobotClient", WrappedRobotClient)

    except ImportError as e:
        print(f"Failed to import {backend_package}: {e}", file=sys.stderr)
        sys.exit(1)

    # Register custom tools in the subprocess's parol6 ``_TOOL_REGISTRY``.
    # The GUI process registers them at startup (main.py: register_all)
    # but the subprocess gets a fresh registry that only contains the
    # built-ins. Without this, ``Robot.set_active_tool("custom:foo")``
    # raises ``ValueError: Unknown tool 'custom:foo'`` and the
    # collision pre-flight in stepping_client._get_local_robot silently
    # falls through — exactly the user population that needed the IK
    # fix the most (custom tools with non-default tcp_offset_m).
    #
    # ``register_all`` re-bakes meshes (idempotent overwrite to
    # parol6_mesh_dir) AND mutates ``_TOOL_REGISTRY`` AND tries to
    # refresh ``ui_state.active_robot._tools`` — the last step
    # defensively skips when ui_state isn't bound (subprocess case),
    # so the call is safe here. Cost: ~0.1-0.5s per custom tool (the
    # bake), paid once per subprocess spawn.
    try:
        from waldo_commander.components.calibration_overlays import (
            custom_tools as _custom_tools,
        )

        _custom_tools.register_all()
    except Exception as e:  # noqa: BLE001
        print(
            f"[stepping_bootstrap] custom-tool registration in "
            f"subprocess failed ({type(e).__name__}: {e}); "
            "Cartesian pre-flight on custom tools will be skipped.",
            file=sys.stderr,
        )

    # Prepare execution environment for the user script
    # Remove our bootstrap script from argv so the user script sees correct args
    sys.argv = [str(script_path)] + sys.argv[2:]

    # Set up globals for exec
    script_globals = {
        "__name__": "__main__",
        "__file__": str(script_path),
        "__builtins__": __builtins__,
    }

    # Read and execute the user's script
    script_code = script_path.read_text(encoding="utf-8")

    try:
        # Compile with the script's filename for proper tracebacks
        code = compile(script_code, str(script_path), "exec")
        exec(code, script_globals)
    except SystemExit:
        # Let SystemExit propagate (normal script termination)
        raise
    except Exception:
        # Re-raise to show traceback in user script
        raise


if __name__ == "__main__":
    main()
