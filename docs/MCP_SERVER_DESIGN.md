# parol6_mcp — MCP server design

A Waldo-Commander plugin that exposes the active PAROL6 robot as a
Model-Context-Protocol server. Any LLM host with MCP client support (Claude
Desktop, Claude Code, OpenAI's MCP client, Gemini Code Assist, Continue,
Goose, OpenClaw, hand-rolled scripts speaking JSON-RPC) can drive the robot
through the same tool surface. The plugin lives in-process inside
Waldo-Commander and mounts a Starlette sub-app on the existing NiceGUI
FastAPI server at `/mcp`. Motion-mutating tools route through the plugin
host's motion API so they inherit `validate_joint_trajectory` and the rest
of WC's safety machinery without re-implementing it.

This document is the canonical design reference. The user-facing intro and
client-side configuration snippets live in
`waldo_commander/plugins/parol6_mcp/README.md`. Cross-plugin contracts (host
API signatures, lifecycle hooks, panel registration) are owned by
`Docs/PLUGIN_CONTRACT.md` and the open dependencies are listed in §9.

## 1. Architecture overview

```
                                ┌──────────────────────────────────────────┐
                                │  Waldo-Commander process                  │
                                │                                           │
   LLM host                     │  NiceGUI app (FastAPI subclass)           │
   ┌────────────┐               │   ├── /                (existing UI)      │
   │  Claude    │               │   └── /mcp             (mounted by plug)  │
   │  Desktop / │   HTTP        │                                           │
   │  Code      ├──── POST ───►─┤  FastMCP Starlette sub-app                │
   │  Gemini    │   JSON-RPC    │   └── routes tool/call to @mcp.tool fn    │
   │  OpenAI    │               │                                           │
   │  Continue  │               │  parol6_mcp.Plugin                        │
   │  Goose     │               │   ├── on_load(host) mounts /mcp           │
   │  custom    │               │   ├── tool fns: parol6_get_joints, ...    │
   └────────────┘               │   └── each tool ──► host.motion / robot   │
                                │                                           │
                                │  plugin host (PLUGIN_CONTRACT.md)         │
                                │   ├── host.robot_client()                 │
                                │   ├── host.motion.{move_j, check, ...}    │
                                │   ├── host.state.active_tool_key()        │
                                │   └── host.fastapi_app() / on_startup     │
                                │                                           │
                                │  WC existing infrastructure               │
                                │   ├── safe_motion / collision_core        │
                                │   ├── waldoctl.RobotClient                │
                                │   └── app.storage / robot_state           │
                                └──────────────────┬───────────────────────┘
                                                   │ UDP :5001 (existing)
                                                   ▼
                                          parol6-server (controller)
                                                   │ serial
                                                   ▼
                                                PAROL6
```

The MCP plugin is one Python module. The FastMCP `mcp` server instance is a
module-level singleton; tools register against it via `@mcp.tool`
decorators at import time. `Plugin.on_load(host)` is the only side-effect
entry point — it stashes the host reference into a module-private slot the
tool functions read, mounts the FastMCP Starlette app onto the host's
FastAPI app, and wires FastMCP's session-manager lifespan into the host's
startup/shutdown hooks. There is no subprocess and no second event loop.

## 2. Transport: stateless streamable HTTP

Streamable HTTP is the spec-blessed remote MCP transport. Stateless mode
(no session affinity, no server-pushed events held across requests) is the
recommended default per the official MCP best-practices guide — it
sidesteps the per-host quirks of stateful streaming and matches our domain
shape, which is request/response over a stateless control plane (all
real state lives in the parol6 controller, not in the MCP layer).

Mount pattern (verified against FastMCP 2.x source):

- `mcp.http_app(transport="streamable-http", path="/", stateless_http=True)`
  returns a `StarletteWithLifespan` ASGI app.
- `nicegui.app` is a `FastAPI` subclass, so `nicegui.app.mount("/mcp", ...)`
  works directly.
- The sub-app carries its own lifespan (FastMCP's session manager). The
  parent NiceGUI app does not automatically invoke sub-app lifespans, so
  the plugin enters `mcp_app.router.lifespan_context(mcp_app).__aenter__()`
  on host startup and `__aexit__()` on host shutdown. FastMCP's source
  prints an explicit error if this isn't wired ("This commonly occurs when
  the FastMCP application's lifespan is not …") — easy to diagnose.

Bind: v0 is `127.0.0.1`-only (NiceGUI's default). LAN bind plus bearer-token
auth is v1.5 scope (§8). For `127.0.0.1` operation, DNS-rebinding
protection and `Origin` validation are still spec-recommended; FastMCP's
HTTP layer handles this when configured. The plugin does not lower these
defaults.

Stdio compatibility for older clients ships as a thin shim later (§8).

## 3. Tool surface (v0 — 8 tools)

Tool names use the `parol6_` prefix per MCP best practice. Annotations
follow the spec four-flag schema. Inputs are Pydantic models or no-arg.
Outputs are flat JSON-serialisable dicts. Errors are returned inside the
tool result (`isError: true`), never raised as JSON-RPC protocol errors.

| Tool | Purpose | Annotations | Lock |
|---|---|---|---|
| `parol6_get_joints` | Read live joint angles (degrees). | readOnly, idempotent | none |
| `parol6_get_pose` | Read TCP pose `[x,y,z,rx,ry,rz]` mm+deg, frame WRF or TRF. With a gripper selected, WRF returns TCP not flange. | readOnly, idempotent | none |
| `parol6_get_tool_state` | Active tool key plus live ToolStatus (state, engaged, positions, channels). | readOnly, idempotent | none |
| `parol6_check_collision` | Pre-flight `validate_joint_trajectory_core` for `(q_from, q_to)` without dispatching. Returns safe/unsafe plus the colliding pair. | readOnly, idempotent | none |
| `parol6_move_j` | Gated joint move. Routes through `host.motion.move_j`; returns structured rejection on gate failure. | not readOnly, NOT destructive (gate prevents the destructive case), not idempotent | host dispatch lock |
| `parol6_halt` | Immediate stop. Bypasses the dispatch lock so it works mid-move. | not readOnly, NOT destructive, idempotent | none (bypass) |
| `parol6_resume` | Re-enable controller after a halt. Required before any motion after `parol6_halt`. | not readOnly, NOT destructive, idempotent | none |
| `parol6_force_move_j` | Bypasses the safety gate. Requires `acknowledge_unsafe=true`. For recovering from false-positive gate rejections after user authorisation. | not readOnly, **destructive**, not idempotent | host dispatch lock |

Input models live in `__init__.py`. Each list-typed parameter (`angles_deg`,
`q_from_deg`, `q_to_deg`) routes through a `field_validator(mode="before")`
that calls `_coerce_to_list` — a lifted version of claude-pair-mcp's
`_coerce_to_str_list` that handles JSON-array strings, semicolon-split, and
newline-split forms. This is defensive against MCP hosts that JSON-encode
list parameters into strings before sending (empirically: Claude Desktop
does this for some types).

Tool docstrings are plaintext — no Markdown tables, no nested lists with
significant indentation. Some MCP clients render Markdown; others show
plaintext. Writing for plaintext means it reads correctly everywhere.

## 4. Safety semantics

The gate is `validate_joint_trajectory_core` from
`parol6_vision.calibration.collision_core` (NiceGUI-free workhorse). It
returns a dict with the keys `safe`, `start_safe`, `end_safe`,
`interior_safe`, `manager_ready`, `reason`, `colliding_pair`, `colliding_q`.
The plugin host wraps that into `host.motion.move_j` which returns
`(ok, reason, detail)` — the same three-state contract `safe_motion`
already uses (`(True, "")` checked-passed, `(True, "skipped: …")` bypassed,
`(False, reason)` rejected).

On rejection the MCP tool returns:

```
{
  "ok": false,
  "isError": true,
  "rejection": {
    "reason": "end config self-collides",
    "colliding_pair": ["L4", "TABLET"],
    "colliding_q_deg": [...],
    "manager_ready": true,
    "next_steps": "Adjust target joints away from the colliding object, or, after explicit user authorisation, call parol6_force_move_j with acknowledge_unsafe=true."
  }
}
```

The structured payload mirrors the gate's internal shape so the LLM sees
the same fields whether the gate fired in the GUI or via MCP. Type
definitions live in `safety.py`.

`parol6_force_move_j` is a separate tool, not a `force=true` flag on
`parol6_move_j`. The reason is auditability: a force call appears as a
distinct tool name in the conversation log. The schema requires
`acknowledge_unsafe: true` literally — Pydantic rejects `false` at the
input boundary, so the LLM cannot accidentally bypass by setting a flag.
The docstring spells out "ONLY call after the user has explicitly
authorised the unsafe motion". `destructiveHint: true` on this tool also
matters: MCP clients are expected to surface destructive operations
distinctly.

`parol6_halt` is annotated `destructiveHint: false` despite stopping
motion, because stopping is always-safe-to-call by definition. It also
bypasses the dispatch lock — the lock is held by an in-flight `move_j` and
halt must work *while* that lock is held.

## 5. Cross-LLM compatibility

The MCP spec is the contract. The plugin stays inside the spec proper —
tools, prompts, resources, plus the four spec annotations — and avoids
Anthropic-specific extensions so any conforming client works.

Constraints to maintain:

- Stateless streamable HTTP transport. Avoid stateful streaming and SSE.
- No `Context.elicit()` — Claude Desktop renders elicitation; other hosts
  may silently drop it. If a tool needs more input, return a structured
  error and let the LLM retry with the missing parameter.
- No multi-content tool responses. Some clients pick only the first block.
  When a tool conceptually returns image + text, wrap both in a single
  JSON-text block (or return a URL/path the client can fetch separately).
- Plaintext-friendly tool descriptions. No tables, no indentation-
  significant Markdown.
- Flat Pydantic input schemas. Deeply nested schemas with `$ref` cycles
  break some clients' tool-discovery rendering.
- Pluggable auth surface. v0 has no auth (127.0.0.1 only). v1.5 wires
  bearer-token verification through a `host.auth.verify_request(...)`
  hook so the same plugin works whether the host is local-only or
  network-exposed. Bearer tokens are the lowest-common-denominator across
  MCP clients; OAuth 2.1 stays optional.

Anthropic-specific constructs we deliberately do NOT use:

- `cache_control` markers — Anthropic API feature, not MCP transport.
- `mcpb` packaging (Claude Desktop extension format) — fine to also ship
  later but not the primary distribution channel.
- Claude Desktop's permission-prompt JSON in tool results.
- Markdown-heavy descriptions that only render nicely in Claude Desktop.

## 6. Translates / doesn't translate from `claude-pair-mcp`

`claude-pair-mcp` is Alvar's other FastMCP server (the one this very pair
conversation runs through). Patterns lift cleanly where they don't.

Translates:

- FastMCP 2.x as the SDK choice (`from fastmcp import Context, FastMCP`).
- `@mcp.tool` with explicit `annotations={...}` dict matching the spec.
- Pydantic input models with `ConfigDict(extra="forbid")` and field
  constraints.
- `_coerce_to_str_list` defensive pre-validator for list-typed parameters,
  generalised to `_coerce_to_list` for any sequence input (we apply it to
  `list[float]` angle inputs).
- `Context.info(...)` for progress events during multi-step operations —
  used here for the `move_j` rejection log and `force_move_j` audit line.
- Atomic state writes (`os.replace(tmp, final)`) for any disk-persisted
  state. v0 has none; v1 calibration triggers will.
- Per-handle FIFO lock with both in-process `threading.Lock` AND on-disk
  `filelock`. Our dispatch lock is keyed on "the robot" not "the pair";
  same shape. The filelock matters because WC's existing
  `script_runner.py` subprocesses also send motion commands.

Doesn't translate:

- `PairRuntime` / subprocess management / idle eviction — our tools call
  functions, not spawn subprocesses.
- Stream-json parsing — Claude-CLI artefact, irrelevant here.
- `.mcpb` packaging — extension format for Claude Desktop only. We bind
  to WC's HTTP server instead; users configure their LLM host to point at
  `http://127.0.0.1:<port>/mcp`. MCPB-packaged stdio shim is an optional
  later distribution channel (§8).
- `pair_create`/`pair_forget`/`pair_compact` lifecycle tools — manage the
  agent itself. The robot is a single managed singleton; no parallel.
- Auto-mode permission classifier — claude-pair-mcp gates Claude Code's
  tool surface; we gate motion via `validate_joint_trajectory`. Different
  problem.

## 7. Validation checklist

Quality is measured by whether an LLM with no other context can drive the
robot through the tools. Before declaring v0 done:

1. Server self-checks:
   - `python -m py_compile waldo_commander/plugins/parol6_mcp/__init__.py`
     and `safety.py` both succeed.
   - With WC running, `curl -sS -X POST http://127.0.0.1:8080/mcp -H
     'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,
     "method":"tools/list"}'` returns 8 tools with the right names.
   - `mcp-inspector` (`npx @modelcontextprotocol/inspector` against
     `http://127.0.0.1:8080/mcp`) loads the tool list and lets you call
     each tool interactively without errors.
2. mcp-builder evaluation harness — see
   `~/AppData/Roaming/Claude/.../skills/mcp-builder/scripts/evaluation.py`.
   Phase 4 writes ≥10 read-only QA pairs and runs the harness against the
   live `/mcp` endpoint. Read-only means questions answerable via
   `get_joints` / `get_pose` / `get_tool_state` / `check_collision`; no
   `move_j` in eval (the gate accepts/rejects but actual motion in eval
   would be unsafe). Phase 4 is outside v0 but the scaffolding is here.
3. Manual cross-client validation:
   - Claude Desktop with the URL configured as an HTTP MCP server.
   - Claude Code CLI: `claude mcp add --scope user parol6 --transport
     http --url http://127.0.0.1:8080/mcp` and verify
     `claude mcp list` shows it connected.
   - At minimum one non-Anthropic client (Continue / Goose / a hand-rolled
     `httpx` POST that speaks the MCP JSON-RPC envelope manually). The
     hand-rolled path is the strongest universality test because it
     guarantees no hidden host-specific assumption.

## 8. Future scope

Out of v0 but pre-designed so we don't paint into corners:

- **Stdio shim** (option 3 from the Phase 2 architecture question). A
  thin entrypoint (~50 LOC) that reads MCP JSON-RPC framing on stdin,
  POSTs each request to `http://127.0.0.1:<port>/mcp`, streams responses
  back. Lets stdio-only MCP clients (some older configurations) work
  against a WC that's running. Ship as `waldo_commander.plugins.parol6_mcp
  .__main__` for a `python -m` invocation, plus a one-line wrapper script
  for `claude_desktop_config.json`.
- **LAN bind + bearer-token auth**. NiceGUI bound to `0.0.0.0`, plus a
  `host.auth.verify_request(req) -> principal | None` hook that the
  FastMCP middleware calls before tool dispatch. Bearer token read from
  a config file the user controls. DNS-rebinding protection and `Origin`
  validation already on by default.
- **Vision tools** mapped onto Gemini Robotics-ER 1.6 modes already
  verified in parol6-vision: `parol6_capture_frame`,
  `parol6_detect_objects` (→ `detect_boxes`), `parol6_find_grasp` (→
  `detect_grasp_pose`), `parol6_detect_trajectory`, `parol6_perceive_scene`
  (detect_boxes + back-project to base frame via `runtime.detection_3d`).
  Mask mode is broken in ER 1.6 + Gemini 2.5 — skip until upstream fixes.
- **Program editing** via `host.editor.write_file()` — `parol6_read_program`,
  `parol6_write_program`, `parol6_append_to_program`,
  `parol6_insert_at_line`, `parol6_run_program`, `parol6_stop_program`,
  `parol6_get_program_status`. Plugin-arch will confirm whether WC's
  editor auto-detects external writes (file watcher / refresh-on-tab-open
  / manual reopen) — see B1 in Alvar's notes; the route depends on that.
- **Async task store** — pattern from `claude_pair_mcp.async_tasks`.
  `parol6_run_program_async(filename) → {task_id}`, plus
  `parol6_poll(task_id)`. Required when programs run longer than a typical
  MCP RPC timeout (~60s on Claude Desktop).
- **Calibration triggers** — `parol6_trigger_localise`,
  `parol6_trigger_calibration`, plus read-only
  `parol6_is_calibrated`, `parol6_get_camera_mount`,
  `parol6_get_obstacles`. Triggers need `Context.report_progress` for
  long-running orchestration and `destructiveHint: true` because they
  move the robot through a hemisphere sweep.
- **Future "custom commands inside Waldo-Commander"** — Alvar's wish-list
  item; on the radar but not yet designed. Plugin-arch's `host.commands`
  surface (if any) will dictate the shape.

## 9. Open dependencies on `PLUGIN_CONTRACT.md`

These are unresolved at the time of writing this doc. The plugin source
flags each with `# TODO: see PLUGIN_CONTRACT.md` at the call site so the
two pieces fit cleanly when plugin-arch publishes the contract.

- **`PluginManifest` field list.** This doc and the plugin's local
  `PluginManifest` define the minimum (`name`, `version`, `display_name`,
  `description`, `mount_path`). Plugin-arch will add others (load order,
  required capabilities, etc.). The local model is marked `frozen=True`
  so swapping it for the canonical import is mechanical.
- **Host motion API signature.** Assumed shape: `await host.motion.move_j(
  angles_deg, *, speed, accel, wait) -> (ok: bool, reason: str, detail:
  dict | None)` mirroring `safe_motion.safe_move_j`. Plus
  `host.motion.check(q_from, q_to) -> dict` and
  `host.motion.move_j_unchecked(...)` for the force path.
- **Host robot-client access.** Assumed: `host.robot_client() -> RobotClient`
  returning the live async client. Equivalent to
  `ui_state.active_robot.create_async_client()` in vanilla WC but plugins
  must not reach into `ui_state` directly.
- **Host state access.** Assumed: `host.state.active_tool_key() -> str`
  matching the GUI's canonical key (custom-tool-aware, not the proxy).
  Equivalent to the existing `_active_gui_tool_key()` pattern but exposed
  via the host so `app.storage` access from MCP request context isn't
  required.
- **Dispatch lock.** Assumed: `host.motion.move_j` acquires the lock
  internally; readonly tools skip it; halt/resume bypass. Filelock layer
  also needed against script-runner subprocesses (per Phase 2 §4).
- **NiceGUI FastAPI handle.** Assumed: `host.fastapi_app() -> FastAPI`
  returns the NiceGUI app instance. Equivalent to `from nicegui import
  app` but routed through host so the plugin doesn't import NiceGUI
  directly except in panel-building code.
- **Lifespan hooks.** Assumed: `host.on_startup(coro)` /
  `host.on_shutdown(coro)` register awaitables to run at uvicorn
  startup/shutdown. Equivalent to `nicegui.app.on_startup` /
  `on_shutdown`.
- **Panel registration.** Assumed: `panels.add(category, title, builder)`
  where `category` is "calibration" / "tools" / "diagnostics" or similar
  and `builder` is a zero-arg function building NiceGUI elements inside
  the panel's request scope. If plugin-arch chooses a different shape
  (each plugin owns its own `ui.expansion`, etc.) the plugin's
  `register_panels` adapts.

Plugin-arch publishes; this plugin's TODOs resolve in a single pass.
