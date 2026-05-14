"""Standalone bootstrap for the parol6_mcp plugin.

Spins up the FastAPI app (NiceGUI's, since the plugin mounts on it) with
just the ``parol6_mcp`` plugin loaded — no full Waldo-Commander UI
required. Reads via ``host.state.*`` return empty / null because no
broadcast feeds the cache in standalone mode.

By default the demo runs with no RobotClient, so motion tools return
``"no robot client available"`` rejections. Pass ``--connect-parol6`` to
attach a client at ``127.0.0.1:5001``; motion tools then dispatch
through parol6-server in whatever mode it's running (sim per the WC GUI
toggle, real otherwise). The WC GUI shares the same parol6-server, so
toggling sim mode in the GUI affects MCP-driven motion too.

Required venv setup:

    pip install fastmcp uvicorn pydantic nicegui
    pip install -e .[parol6]   # only if using --connect-parol6

Launch (read-only, no robot motion):

    python examples/run_mcp_demo.py

Launch (motion-enabled, connects to running parol6-server):

    python examples/run_mcp_demo.py --connect-parol6

Expected stdout: a paste-ready test-instructions block plus uvicorn's
startup log. Press Ctrl+C to stop; ``unload_all()`` runs in the cleanup
path so the FastMCP session manager exits cleanly and the RobotClient
(if any) is closed.

Troubleshooting:

- ``ModuleNotFoundError: fastmcp`` → ``pip install fastmcp``.
- ``ModuleNotFoundError: parol6`` with ``--connect-parol6`` →
  ``pip install -e .[parol6]``.
- ``Address already in use`` → another process is on port 8080. Set
  ``MCP_DEMO_PORT=8081`` (or any free port) before launching.
- ``curl`` returns 406 / 415 → the request needs both Accept types;
  use ``Accept: application/json, text/event-stream``.
- ``tools/list`` returns "session not initialized" → call ``initialize``
  first (see the printed instructions).
- Motion calls timeout → parol6-server isn't running at
  ``127.0.0.1:5001``. Start it (or WC, which spawns it) before passing
  ``--connect-parol6``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

import uvicorn
from nicegui import app as nicegui_app

from waldo_commander.plugin_host import set_demo_robot_client
from waldo_commander.plugins import load_all, unload_all

PORT = int(os.environ.get("MCP_DEMO_PORT", "8080"))
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}/mcp"
PAROL6_HOST = "127.0.0.1"
PAROL6_PORT = 5001

logger = logging.getLogger(__name__)


def _maybe_connect_robot_client(connect: bool) -> Any:
    """Construct + register an AsyncRobotClient when ``--connect-parol6``
    is set. Returns the client (caller is responsible for ``close()``)
    or None.
    """
    if not connect:
        print(
            f"[disconnected] no RobotClient — motion tools return "
            f"'no robot client' rejections; pass --connect-parol6 to "
            f"dispatch against parol6-server at "
            f"{PAROL6_HOST}:{PAROL6_PORT}",
        )
        return None

    try:
        from parol6.client.async_client import AsyncRobotClient  # noqa: PLC0415
    except ImportError as e:
        print(
            "ERROR: --connect-parol6 requires the parol6 package "
            f"({e}). Install via 'pip install -e .[parol6]' or relaunch "
            "without the flag.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = AsyncRobotClient(
        host=PAROL6_HOST, port=PAROL6_PORT, timeout=5.0,
    )
    set_demo_robot_client(client)
    print(
        f"[connected] RobotClient bound to parol6-server at "
        f"{PAROL6_HOST}:{PAROL6_PORT} — sim/real mode follows the WC GUI "
        f"toggle (start WC + toggle sim mode in the GUI before driving)",
    )
    return client


def _print_instructions() -> None:
    """Paste-ready test commands. Covers curl, mcp-inspector, Claude Code,
    Claude Desktop, and a generic Python client."""

    init_body = (
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
        '{"protocolVersion":"2025-06-18","capabilities":{},'
        '"clientInfo":{"name":"demo","version":"0"}}}'
    )

    print(
        f"""
=============================================================
  parol6_mcp demo server — listening at {URL}
=============================================================

Tools (8):
  parol6_get_joints        parol6_get_pose           parol6_get_tool_state
  parol6_check_collision   parol6_move_j             parol6_force_move_j
  parol6_halt              parol6_resume

Verify the server is alive (minimum path — initialize handshake):

  curl -sN -X POST {URL} \\
    -H "Content-Type: application/json" \\
    -H "Accept: application/json, text/event-stream" \\
    -d '{init_body}'

MCP Inspector (interactive UI; configures the session for you):

  npx @modelcontextprotocol/inspector
  # In the UI: transport=streamable-http, URL={URL}

Claude Code CLI:

  claude mcp add --scope user parol6 --transport http --url {URL}
  claude mcp list   # parol6 should show "Connected"

Claude Desktop — paste into claude_desktop_config.json under "mcpServers":

  "parol6": {{
    "transport": "http",
    "url": "{URL}"
  }}

Generic Python (any LLM host that speaks MCP JSON-RPC):

  python -c "import httpx, json; r = httpx.post('{URL}', json=json.loads('{init_body}'), headers={{'Accept': 'application/json, text/event-stream'}}); print(r.status_code, r.text[:500])"

Behaviour today:
  - Read-only state tools (get_joints, get_pose, get_tool_state) return
    null in standalone mode because no broadcast feeds the per-process
    state cache. The WC GUI process has its own cache.
  - parol6_check_collision dispatches if parol6_vision is installed; else
    returns manager_ready=false.
  - parol6_move_j and parol6_force_move_j both dispatch against
    parol6-server when --connect-parol6 is set (the safety gate is a
    forward to the unchecked path in the proto; see TBD #17 in
    PLUGIN_CONTRACT.md §9). Without --connect-parol6 both return
    'no robot client' rejections.
  - parol6_halt / parol6_resume dispatch immediately when a client is
    bound.

Press Ctrl+C to stop.
=============================================================
"""
    )


async def _bootstrap_and_serve(connect_parol6: bool) -> None:
    """Load plugins, optionally bind a RobotClient, then serve uvicorn.

    Order matters: ``load_all()`` runs each plugin's ``on_load``, which
    registers ``app.on_startup`` hooks (FastMCP's session-manager
    lifespan). Uvicorn's lifespan startup then fires those hooks. Calling
    ``load_all()`` AFTER ``serve()`` would register the hooks too late —
    startup has already completed.

    The RobotClient is set BEFORE ``load_all()`` so any hook that probes
    ``host.robot_client()`` during startup sees the bound client.
    """
    client = _maybe_connect_robot_client(connect_parol6)

    try:
        state = await load_all()
        loaded_ids = [p.plugin_id for p in state.loaded]
        if "parol6_mcp" not in loaded_ids:
            print(
                "ERROR: parol6_mcp plugin failed to load. Check stderr for "
                "loader errors (most likely: fastmcp not installed).",
                file=sys.stderr,
            )
            sys.exit(1)

        _print_instructions()

        config = uvicorn.Config(
            nicegui_app,
            host=HOST,
            port=PORT,
            log_level="info",
            lifespan="on",
            access_log=False,
        )
        server = uvicorn.Server(config)
        try:
            await server.serve()
        finally:
            await unload_all()
    finally:
        set_demo_robot_client(None)
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    await close()
                except (OSError, RuntimeError) as e:
                    logger.debug("client.close raised: %s", e)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone bootstrap for the parol6_mcp plugin. Serves the "
            "MCP route on /mcp without booting full Waldo-Commander."
        ),
    )
    parser.add_argument(
        "--connect-parol6",
        action="store_true",
        help=(
            f"Connect an AsyncRobotClient to parol6-server at "
            f"{PAROL6_HOST}:{PAROL6_PORT}. Required for motion tools to "
            f"dispatch; without this flag they return 'no robot client' "
            f"rejections."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    try:
        asyncio.run(_bootstrap_and_serve(connect_parol6=args.connect_parol6))
    except KeyboardInterrupt:
        # uvicorn already handles SIGINT internally; this branch covers
        # the rare case where Ctrl+C lands during pre-serve setup.
        print("\nshutdown requested.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
