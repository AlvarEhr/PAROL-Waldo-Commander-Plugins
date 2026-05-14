"""Standalone bootstrap for the parol6_mcp plugin.

Spins up the FastAPI app (NiceGUI's, since the plugin mounts on it) with
just the ``parol6_mcp`` plugin loaded — no full Waldo-Commander UI, no
RobotClient required. Reads via ``host.state.*`` return empty / null
because no broadcast feeds the cache; ``host.motion.move_j`` returns the
``"host.motion not yet wired"`` stub rejection until the consolidated
dispatch path lands. This is enough to verify the MCP transport, tool
registration, and rejection payloads end-to-end from any LLM host.

Required venv setup:

    pip install fastmcp uvicorn pydantic nicegui

Launch:

    python examples/run_mcp_demo.py

Expected stdout: a paste-ready test-instructions block plus uvicorn's
startup log. Press Ctrl+C to stop; ``unload_all()`` runs in the cleanup
path so the FastMCP session manager exits cleanly.

Troubleshooting:

- ``ModuleNotFoundError: fastmcp`` → ``pip install fastmcp``.
- ``Address already in use`` → another process is on port 8080. Set
  ``MCP_DEMO_PORT=8081`` (or any free port) before launching.
- ``curl`` returns 406 / 415 → the request needs both Accept types;
  use ``Accept: application/json, text/event-stream``.
- ``tools/list`` returns "session not initialized" → call ``initialize``
  first (see the printed instructions).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import uvicorn
from nicegui import app as nicegui_app

from waldo_commander.plugins import load_all, unload_all

PORT = int(os.environ.get("MCP_DEMO_PORT", "8080"))
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}/mcp"

logger = logging.getLogger(__name__)


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
  - Read-only tools work (return null until WC's broadcast feeds the cache).
  - parol6_check_collision dispatches if parol6_vision is installed; else
    returns manager_ready=false.
  - parol6_move_j returns a structured rejection ('host.motion not yet
    wired') until the consolidated motion path lands.
  - parol6_force_move_j works end-to-end if a RobotClient is connected.

Press Ctrl+C to stop.
=============================================================
"""
    )


async def _bootstrap_and_serve() -> None:
    """Load plugins, then serve uvicorn against the FastAPI app.

    Order matters: ``load_all()`` runs each plugin's ``on_load``, which
    registers ``app.on_startup`` hooks (FastMCP's session-manager
    lifespan). Uvicorn's lifespan startup then fires those hooks. Calling
    ``load_all()`` AFTER ``serve()`` would register the hooks too late —
    startup has already completed.
    """
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_bootstrap_and_serve())
    except KeyboardInterrupt:
        # uvicorn already handles SIGINT internally; this branch covers
        # the rare case where Ctrl+C lands during pre-serve setup.
        print("\nshutdown requested.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
