# parol6_mcp

Waldo-Commander plugin that exposes the active PAROL6 robot as a Model
Context Protocol server. Any LLM host with MCP client support can drive
the robot through the same tool surface — no per-LLM integration needed.

## What it exposes (v0)

Eight tools, prefixed `parol6_` so they coexist with other MCP servers:

- `parol6_get_joints` — current joint angles (degrees).
- `parol6_get_pose` — TCP pose `[x, y, z, rx, ry, rz]` in mm + deg; frame
  is WRF (default) or TRF. With a gripper selected, WRF returns the TCP
  pose, not the flange.
- `parol6_get_tool_state` — active tool key plus live status.
- `parol6_check_collision` — pre-flight a joint move against the live
  collision environment without dispatching it.
- `parol6_move_j` — gated joint move. Routes through Waldo-Commander's
  existing `validate_joint_trajectory` and returns a structured rejection
  on gate failure.
- `parol6_force_move_j` — bypasses the gate. Requires
  `acknowledge_unsafe: true`. Use only after explicit user authorisation.
- `parol6_halt` — immediate stop. Always allowed; works mid-move.
- `parol6_resume` — re-enable after halt. Required before any further
  motion.

Future scope (not in v0): vision tools mapped onto Gemini Robotics-ER 1.6,
program-file editing, async task store for long-running runs, calibration
triggers. See `docs/MCP_SERVER_DESIGN.md` §8.

## Testing locally

You do not need to boot full Waldo-Commander to verify the plugin. The
repo ships a standalone demo bootstrap that loads only `parol6_mcp`,
serves uvicorn on `127.0.0.1:8080`, and prints paste-ready test
commands.

Install the runtime deps once:

```
pip install fastmcp uvicorn pydantic nicegui
```

Launch the demo:

```
python examples/run_mcp_demo.py
```

Expected stdout: a one-screen instructions block listing the 8 tools,
curl / mcp-inspector / Claude Code / Claude Desktop test recipes, then
uvicorn's startup log ending with `Uvicorn running on http://127.0.0.1:8080`.

Minimum "is the server alive" check in a second terminal — the MCP
initialize handshake:

```
curl -sN -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

A 200 response with a `result.serverInfo.name = "parol6_mcp"` field
proves transport + tool registration are alive. For `tools/list` and
beyond, use the MCP Inspector (handles the session for you) or any
MCP-aware LLM client (configs below).

Press Ctrl+C in the demo terminal to shut down — `unload_all()` runs in
the cleanup path so the FastMCP session manager exits cleanly.

### Claude Desktop

Paste into `claude_desktop_config.json` (Windows:
`%APPDATA%\Claude\claude_desktop_config.json`) under `mcpServers`:

```
"parol6": {
  "transport": "http",
  "url": "http://127.0.0.1:8080/mcp"
}
```

Restart Desktop. The 8 tools surface under the parol6 server name.

### Claude Code CLI

```
claude mcp add --scope user parol6 --transport http \
  --url http://127.0.0.1:8080/mcp
claude mcp list   # parol6 should show "Connected"
```

In any Claude Code session you'll see `mcp__parol6__*` tools available.

### Integration test

```
pip install httpx asgi_lifespan pytest-asyncio
pytest tests/integration/test_mcp_server.py -v
```

Three checks: `tools/list` returns the 8-tool surface with correct
annotations, `parol6_get_joints` handles an empty state cache cleanly,
`parol6_move_j` round-trips the `host.motion not yet wired` stub
rejection into a structured MCP `isError` payload.

### Expected behaviour today

- **Read-only tools work.** `parol6_get_joints`, `parol6_get_pose`,
  `parol6_get_tool_state` read from `host.state.*`. They return `null`
  in the demo because no robot broadcast feeds the cache.
- **`parol6_check_collision`** dispatches if `parol6_vision` is
  installed; otherwise returns `manager_ready=false`.
- **`parol6_move_j`** returns a structured rejection
  (`"host.motion not yet wired"`) until the consolidated motion
  dispatch path lands. The rejection payload is the same shape it'll
  have once real `validate_joint_trajectory_core` rejections flow
  through — proves the pipeline.
- **`parol6_force_move_j`** works end-to-end if a `RobotClient` is
  connected (the unchecked path is wired today).
- **`parol6_halt` / `parol6_resume`** work end-to-end if a `RobotClient`
  is connected.

## How it works

The plugin mounts a FastMCP Starlette sub-app on Waldo-Commander's NiceGUI
FastAPI server at `/mcp`. Transport is stateless streamable HTTP, which
the MCP spec calls out as the recommended remote transport. Motion-
mutating tools route through the plugin host's motion API so they
inherit `validate_joint_trajectory`, collision checks, joint-limit
guards, simulator-mode awareness, and the dispatch lock for free.

When Waldo-Commander runs locally, the endpoint is
`http://127.0.0.1:<wc-port>/mcp`. v0 binds to localhost only and has no
auth. LAN bind plus bearer-token auth is v1.5 scope.

## Connecting an LLM client

The endpoint is the same for every client. Configuration differs by host.

### Claude Code (CLI)

```bash
claude mcp add --scope user parol6 --transport http \
  --url http://127.0.0.1:8080/mcp
claude mcp list   # parol6 should show "Connected"
```

In any Claude Code session you'll see `mcp__parol6__*` tools available.

### Claude Desktop

Add an HTTP MCP server entry in Claude Desktop's settings (Extensions →
Add MCP Server → HTTP). Set the URL to
`http://127.0.0.1:8080/mcp`. Restart Desktop.

### Continue / Cursor / Goose

Continue and similar clients accept the same URL through their MCP
configuration block. Consult the client's docs for the exact JSON key
(typically `mcpServers` with a `url` field for HTTP transport).

### Custom Python / hand-rolled clients

The endpoint speaks plain JSON-RPC over HTTP. Any HTTP client that can
post JSON-RPC envelopes works:

```python
import httpx
r = httpx.post(
    "http://127.0.0.1:8080/mcp",
    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
)
print(r.json())
```

Or use the MCP Inspector for interactive testing:

```bash
npx @modelcontextprotocol/inspector
# Configure: streamable-http, http://127.0.0.1:8080/mcp
```

## Universal-LLM contract

The plugin stays inside the MCP spec proper and avoids any Anthropic-
specific extension. Specifically:

- Stateless streamable HTTP transport.
- No `Context.elicit()` — tools return structured errors instead of
  mid-tool prompts.
- Single-content tool responses (some clients pick only the first block
  of a multi-content result).
- Plaintext tool descriptions — no Markdown tables, no
  indentation-significant lists.
- Flat Pydantic input schemas — no `$ref` cycles.

Any MCP-spec-compliant client should work. If you find one that doesn't,
the discrepancy is either a client bug or a spec ambiguity; please flag.

## Safety

The gated motion tools call Waldo-Commander's existing
`validate_joint_trajectory` against the live collision environment
(active tool meshes, calibrated tablet pose, floor). Gate rejection
returns a structured payload — `reason`, `colliding_pair`, joint config
at the colliding sample, plus a `next_steps` hint to the LLM. The
parol6_force_move_j tool bypasses this gate, requires an explicit
`acknowledge_unsafe: true` flag, and is annotated `destructiveHint: true`
so MCP clients can surface it distinctly to the user.

`parol6_halt` always works, even while another motion command holds the
dispatch lock. After a halt, call `parol6_resume` before sending further
motion commands — without it, the controller stays disabled.

Waldo-Commander's standard safety warnings (no software safety
guarantees, hardware E-stop is not optional, etc.) apply unchanged.

## Files

- `__init__.py` — Plugin class, FastMCP server, 8 tool functions.
- `safety.py` — structured rejection payload.

Design rationale: `Docs/MCP_SERVER_DESIGN.md` in the repo root.
Host-API contract: `Docs/PLUGIN_CONTRACT.md` (owned by the plugin-arch
work; the plugin's `# TODO` markers reference it).
