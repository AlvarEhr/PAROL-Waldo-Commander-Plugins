# Plugin contract

Status: proto-framework, pre-upstream. The contract Jepson eventually ships
in main Waldo-Commander is the source of truth; this doc describes a
provisional shape that lets plugin authors start building today against a
host class that lives in `waldo_commander/plugin_host.py`. Items marked
**TBD upstream** are intentionally left for Jepson's framework to set.

The contract is built around two concrete first users: `parol6_mcp` (the
MCP server plugin, in-repo) and, later, an extracted form of the
calibration_overlays package that currently lives on Alvar's
`parol6-vision-calibration` branch. The shape below is the smallest
surface that lets both compose cleanly without each plugin knowing the
other exists.


## 1. Discovery

Plugins live in `waldo_commander/plugins/<plugin-id>/`. At Waldo-Commander
startup the framework walks that directory, treats every child folder
containing an `__init__.py` as a plugin, imports it, and looks for a
top-level `Plugin` class. The folder name is the plugin id; ids must be
valid Python identifiers and must be unique within the process.

Convention over config: no `plugin.json`, no `pyproject.toml`
entry-points (for v0). Discovery is purely a directory scan because
Alvar's mental model is "drop a folder in, framework picks it up." Pip
entry-points are a future addition for installed plugins; they will share
the same `Plugin` protocol so the loader can fall through to them.

Plugin id rules:

- Lowercase identifier, underscores allowed, no leading underscore.
- Matches the storage prefix the host applies on the plugin's behalf
  (`plugin:<id>:<key>`).
- Matches the MCP tool prefix the plugin's tools should use (`<id>_…`).
- Folder name and Python package name must agree.


## 2. The `Plugin` class

```python
class Plugin:
    manifest: PluginManifest = PluginManifest(...)

    def __init__(self) -> None: ...

    def enabled(self, host: Host) -> bool: ...                       # optional
    async def on_load(self, host: Host) -> None: ...                 # optional
    def register_panels(self, host: Host, panels: PanelRegistry) -> None: ...
    def register_mcp_tools(self, host: Host, mcp: MCPRegistry) -> None: ...
    async def on_unload(self, host: Host) -> None: ...               # optional
```

Required: the `manifest` class attribute. Everything else is optional —
the loader checks for each method by name and skips silently if missing.
A plugin that only mounts an HTTP route can be ~30 LOC.

Lifecycle order:

1. Discovery imports the module.
2. `Plugin()` is instantiated.
3. `enabled(host)` is called if defined. False → plugin stops here.
4. `on_load(host)` runs. This is the side-effect entry point: mount
   routes, register lifespan hooks, install custom tools, subscribe to
   events.
5. `register_panels(host, panels)` and `register_mcp_tools(host, mcp)`
   run after `on_load` so the plugin can decide registrations based on
   what `on_load` discovered.
6. The plugin runs for the lifetime of the NiceGUI process. The host
   does not periodically poll or restart plugins.
7. On WC shutdown, `on_unload(host)` runs in reverse load order.

The host catches and logs exceptions from each hook. A failing plugin is
isolated: load failure logs an error and skips the plugin, it does not
abort WC startup.


## 3. `PluginManifest`

Pydantic v2 model, frozen. Fields the host needs to know about the
plugin without instantiating it:

```python
class PluginManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str                # plugin id; must match the folder name
    version: str             # semver
    display_name: str        # human-facing name
    description: str         # one paragraph
    mount_path: str = ""     # optional FastAPI mount; "" means none
```

Fields a later iteration likely adds (**TBD upstream**): `min_host_version`,
`requires_capabilities`, `load_priority`, `default_panel_category`. The
plugin should set `frozen=True` so swapping for the canonical import is a
type-level change rather than behavioural.


## 4. `Host` capabilities

The host is a single object passed to every plugin hook. It exposes
typed accessors for things plugins need; plugins do not reach into
`nicegui.app`, `waldo_commander.state`, or backend modules directly. Each
namespace below is a sub-object on `host`.

### 4.1 `host.motion` — gated dispatch

```python
async def move_j(angles_deg, *, speed=0.3, accel=0.5, wait=True) -> (ok, reason, detail)
async def check(q_from_deg, q_to_deg) -> dict
async def move_j_unchecked(angles_deg, *, speed, accel, wait) -> (ok, reason, detail)
```

Return shape is a 3-tuple `(ok: bool, reason: str, detail: dict | None)`.
This generalises `safe_motion.safe_move_j`'s `(ok, reason)` to keep the
structured detail accessible at the call site — needed by `parol6_mcp`
so the gate rejection round-trips cleanly into an MCP `isError` payload
without a separate accessor.

Semantics of the three states:

- `(True, "", {"command_index": N})` — gate passed, command dispatched, returned index.
- `(True, "skipped: <why>", None)` — gate deliberately bypassed (no tool, no
  manager, simulator mode, etc.). Caller decides whether to proceed.
- `(False, reason, validate_joint_trajectory_core_dict)` — gate rejected;
  detail carries the full result dict (`safe`, `start_safe`, `end_safe`,
  `interior_safe`, `manager_ready`, `colliding_pair`, `colliding_q`).

`check` is the read-only equivalent: runs the same machinery, returns
the raw `validate_joint_trajectory_core` dict, never dispatches.

`move_j_unchecked` skips the gate entirely. Auditable: lives at a
distinct name in the host API surface, distinct method name on the
plugin, distinct tool name on MCP.

Dispatch lock: `host.motion.move_j` and `move_j_unchecked` acquire an
internal robot-scoped FIFO lock so concurrent callers serialise.
Read-only callers (`check`) skip the lock. `client.halt()` does not go
through the host — it is intentionally lock-bypassing and plugins call
`host.robot_client().halt()` directly. The lock has two layers:
in-process `asyncio.Lock` for plugin-to-plugin serialisation, plus
on-disk `filelock` keyed on the controller endpoint so the editor's
`script_runner.py` subprocesses serialise against MCP traffic too.

**TBD upstream**: the exact filelock path. v0 uses
`~/.waldo-commander/locks/motion-<host>-<port>.lock`. Jepson may move
this into the controller process or the WC home dir spec.

### 4.2 `host.robot_client() -> RobotClient`

Returns the live async client. Same instance the GUI binds. Plugins do
not call `create_async_client()` directly — the host hands out the
already-connected one so a second client doesn't compete for the UDP
port.

May return `None` before the first connect. Plugins should check, log,
and skip rather than crash. A future iteration may expose
`await host.wait_for_connect()` for plugins that need to block.

### 4.3 `host.state` — typed state accessors

Cache-mirrored read-only views. Safe to call from any thread or
asyncio context — they do not enter NiceGUI's request scope. The
underlying implementation mirrors `app.storage.general` and
`robot_state` snapshots into a runtime cache, matching the existing
`custom_tools._per_tool_runtime_cache` pattern that solved the same
problem for calibration worker threads.

```python
def active_tool_key() -> str          # GUI's canonical key (custom-tool-aware)
def active_tool_variant() -> str
def simulator_active() -> bool
def joint_angles_deg() -> tuple[float, ...]   # last broadcast snapshot
def tcp_pose() -> tuple[float, ...]           # [x,y,z,rx,ry,rz] mm+deg
def tool_status() -> ToolStatus | None
def connected() -> bool
```

No raw `app.storage` access for plugins. The mcp-server spike on storage
context was inconclusive last turn; the cache-mirror keeps plugins
working regardless of how that lands. If the spike eventually proves
request scope works for plugin contexts, additional accessors land here
without changing the public shape.

Writers: plugins do not write controller state directly. To change tool
selection or settings, plugins call the same paths the GUI does, which
will eventually be exposed as `host.actions.select_tool(...)` etc.
**TBD upstream** for v0.

### 4.4 `host.fastapi_app() -> FastAPI`

Returns the NiceGUI `app` instance (a `FastAPI` subclass). Plugins use
this to mount sub-apps — `parol6_mcp` mounts FastMCP's Starlette app at
`/mcp` this way. Plugins do not `from nicegui import app` directly; the
host is the only path that survives if Jepson eventually wraps NiceGUI
in a different container.

### 4.5 `host.on_startup(coro)` / `host.on_shutdown(coro)`

Register awaitables for the uvicorn startup and shutdown lifecycle.
Equivalent to `nicegui.app.on_startup` / `on_shutdown`. The host
serialises these (FIFO on startup, LIFO on shutdown) so a plugin that
sets up state in startup can rely on its own teardown running before
plugins it depends on tear down.

`parol6_mcp` uses these to enter and exit FastMCP's session-manager
lifespan context — the parent FastAPI app does not invoke sub-app
lifespans automatically, so the plugin manually pumps it via the host's
hooks.

### 4.6 `host.storage` — namespaced persistence

```python
host.storage.general[key]   # dict-like proxy, prefixes with "plugin:<id>:"
host.storage.user[key]      # same shape for app.storage.user
```

Reads go through the cache mirror so they work from any context. Writes
go through `app.storage` directly, which means they require a NiceGUI
request scope when writing `app.storage.user`. Worker threads that need
to write should write to `host.storage.general` (process-wide,
context-free) or schedule the write back onto the asyncio loop.

The prefix is automatic; plugins read and write `key`, the host stores
`plugin:<id>:key`. Cross-plugin reads (uncommon) go through an explicit
`host.foreign_storage(other_id).general[key]` call — **TBD upstream**
whether shared keys require both plugins to declare the share.

### 4.7 `host.scene` — scene-overlay registration

```python
def add_group(name, *, parent="root") -> ui.scene.group
def add_timer(interval_s, callback) -> ui.timer
```

Wraps `ui.scene.group()` and `ui.timer(...)`, tagging each handle with
the plugin id and caching them in a per-plugin list. On `on_unload`
(and on browser disconnect), the host deletes the groups and cancels
the timers automatically. Replaces the 130-line manual teardown shape
in `calibration_overlays/panel._teardown_overlays`.

`parent` is one of `"root"` (the URDF scene root), `"tcp_anchor"`
(follows the flange), or `"world_axes"` (fixed).

### 4.8 `host.editor` — editor coherence

```python
async def write_file(filename, text) -> bool
```

Writes to `PROGRAM_DIR/filename`, then updates any open `EditorTab`
whose `file_path` matches so the in-memory copy stays coherent with
disk. Notifies the user via `ui.notify` when an open tab is updated.
Resolves the B1 finding (editor has no file watcher) without requiring
each plugin to reach into `editor_tabs_state` directly.

### 4.9 Event bus

```python
def subscribe(event: str, callback) -> None
def publish(event: str, payload: dict) -> None
```

Built-in events the host publishes: `tool_change` (old, new, variant),
`simulator_mode_change` (active: bool), `robot_connect`,
`robot_disconnect`, `browser_disconnect`. Plugins can subscribe and
publish custom events under their namespace (`<plugin_id>.<event>`).

Wraps a thin in-memory list of `(event_name, callback)` tuples; no
async dispatch, no priority, no ordering guarantees beyond
registration order. **TBD upstream** if Jepson wants an async event
bus.


## 5. MCP-tool registration

For plugins that expose MCP tools via the `parol6_mcp` plugin or any
future replacement:

```python
host.mcp.register(
    name="<plugin_id>_<verb>",
    description="...",                          # plaintext
    input_schema=PydanticModel,
    output_schema=...,
    handler=callable,
    side_effects=["motion" | "storage" | "tool_state" | "scene" | "none"],
    requires=["camera_tool" | "robot_connected" | ...],
    annotations={                                # MCP spec four flags
        "readOnlyHint": bool,
        "destructiveHint": bool,
        "idempotentHint": bool,
        "openWorldHint": bool,
    },
)
```

`side_effects` and `requires` drive host-side gating. The `parol6_mcp`
server reads the registry at request time and dispatches via the
declared handler.

For v0, `parol6_mcp` does not consume this registry — its eight tools
are decorated directly on the module-level `FastMCP` instance with
`@mcp.tool`. The registry is the **forward path**: when calibration is
extracted as a plugin, it registers its tools through `host.mcp` and
they appear in the MCP server's tool list without `parol6_mcp` knowing
calibration exists. Until then `host.mcp.register` is a no-op stub on
the host.


## 6. Cross-plugin coupling

Plugins do not import each other. All composition goes through the
host:

- A plugin needing motion calls `host.motion.move_j`, which inherits
  every safety gate every other plugin (and the GUI) has installed.
- A plugin exposing capabilities to other plugins registers them on the
  host (custom MCP tools via `host.mcp.register`, scene overlays via
  `host.scene`, etc.).
- A plugin reacting to another plugin's actions subscribes via
  `host.subscribe(...)` to events the host publishes.

The one exception is the host itself: `parol6_mcp` is "the MCP server"
plugin and other plugins' MCP tools appear through it. Concretely, the
`host.mcp` namespace IS `parol6_mcp`'s registry, made accessible
through the host for separation. If `parol6_mcp` is not loaded,
`host.mcp.register(...)` records the registration but no transport
serves it.


## 7. Open dependencies — resolution map

mcp-server flagged 9 dependencies in `MCP_SERVER_DESIGN.md` §9. Each
resolves to a section above:

| mcp-server flag | Resolved by |
|---|---|
| `PluginManifest` field list | §3 |
| Host motion API signature | §4.1 (3-tuple `(ok, reason, detail)`) |
| `host.robot_client()` | §4.2 |
| `host.state.active_tool_key()` | §4.3 (cache-mirrored) |
| Dispatch lock | §4.1 (internal to motion API) |
| `host.fastapi_app()` | §4.4 |
| `host.on_startup` / `on_shutdown` | §4.5 |
| Panel registration | §4 / §8 — `PanelRegistry` shape below |
| Storage from MCP context | §4.3 / §4.6 (cache-mirrored, no raw `app.storage`) |


## 8. Panel registration

```python
panels.add(category: str, title: str, builder: Callable[[], None]) -> None
```

`category` is one of `"calibration"`, `"tools"`, `"diagnostics"`,
`"plugins"`, or a custom string. The host's Settings tab renders a
plugin-section for each category that has registrations.

`builder` is a zero-arg callable invoked inside `ui.column()` in the
plugin's panel slot. It builds NiceGUI elements directly; the host
provides the column container, the plugin owns its contents.

Panels render in registration order within a category. Categories
render in insertion order.

For v0, mcp-server registers its status panel under `"calibration"`
because that category is what calibration_overlays' settings panel
also targets — both plugins surface alongside calibration-related
controls. **TBD upstream** if Jepson reorganises the Settings tab.


## 9. TBDs pinned for Jepson's framework

These are intentionally left for the canonical contract to set. Code in
this repo uses provisional values; flipping each is a small, mechanical
change once Jepson lands.

1. Discovery mechanism (folder scan vs entry points vs hybrid).
2. Manifest schema (Pydantic, JSON, dataclass).
3. Lifecycle hook names + signatures (`on_load` vs `setup` vs `__call__`).
4. Panel registration shape (categories+builders vs declarative spec).
5. MCP-tool registration shape (declarative metadata vs decorator).
6. Side-effect category vocabulary.
7. `requires` precondition vocabulary.
8. Event bus model (sync list vs async pubsub vs callbacks at registration).
9. Per-plugin dependency declarations.
10. Hard kill-switch convention (env var, storage, manifest field).
11. Camera-bearing render gate generalisation.
12. Backend-capability query surface (`host.backend.has(name)`).
13. Cross-plugin storage sharing model.
14. MCP tool input/output schema shape (JSON-schema vs Pydantic vs callable).
15. Side-effect taxonomy (TBD #6 finer split — `motion`, `storage`, …).
16. `requires` taxonomy (TBD #7 finer split — `robot_connected`, …).
17. Host motion-dispatch consolidation (the host owns one motion-dispatch
    path; today's WC has motion calls scattered across control,
    pose_popup, hover, editor — the consolidation is its own refactor).


## 10. Versioning

The contract version (this document) is the source of compatibility.
Plugins should pin a minimum contract version once Jepson's framework
publishes one. Pre-Jepson, the contract is "what's in this doc on the
`proto-plugin-framework` branch."


## 11. Wiring into `main.py`

This section is **provisional**. Jepson's upstream framework will own
the actual wiring shape; the snippets below describe what's needed today
to bootstrap the loader against the existing `main.py` without
modifying it. Treat them as a hookup recipe an integrator can apply
locally, not a prescribed structure.

Three call sites; each is small and additive.

### 11.1 Load plugins on app startup

The existing `_on_startup` hook in `main.py` (decorated `@ng_app.on_startup`)
already runs after `client` has been constructed in `main()` and after
`_restore_settings()` populates the user's tool selection. That's the
right moment: plugins can rely on `host.robot_client()` returning a live
client and `host.state.active_tool_key()` returning the persisted choice.

```python
# waldo_commander/main.py — top-of-file imports
from waldo_commander.plugins import load_all as _load_plugins

# waldo_commander/main.py — inside _on_startup, after _restore_settings()
# and before the "ready on http://..." log line.
try:
    await _load_plugins()
except (RuntimeError, ImportError) as e:
    logger.error("plugin load failed: %s", e)
    # Deliberately do not re-raise — a failing plugin must not block WC
    # startup. The loader already isolates per-plugin failures; this
    # outer guard catches catastrophic discovery failures.
```

Why here, not earlier: plugins that mount FastAPI sub-apps (parol6_mcp
does this for `/mcp`) need the FastAPI app instance to exist and be
attached to the running event loop. `_on_startup` is invoked by
uvicorn after the app is constructed but before it begins serving
requests, which is exactly the window FastMCP's
`http_app(...).router.lifespan_context(...)` is designed for.

Why after `_restore_settings`: settings populate `app.storage.general`,
which `host.state.active_tool_key()` falls through to when the runtime
cache hasn't been primed yet. Loading plugins before settings would
yield stale tool-key reads during `on_load`.

### 11.2 Unload plugins on app shutdown

```python
# waldo_commander/main.py — top-of-file imports
from waldo_commander.plugins import unload_all as _unload_plugins

# waldo_commander/main.py — inside _on_shutdown, at the top of the
# function body (before the camera_service.stop() / script-runner
# teardown blocks).
try:
    await _unload_plugins()
except (RuntimeError, AttributeError) as e:
    logger.warning("plugin unload failed: %s", e)
```

Why at the top of `_on_shutdown`: plugins may hold references to
WC-owned resources (the editor's tabs, the FastAPI app's routes,
running asyncio tasks). Giving them their `on_unload` window before
WC tears down its own services means a plugin can clean up its
side-effects against still-live infrastructure. Mirrors the FIFO/LIFO
pairing the contract documents in §2.

### 11.3 Render plugin panels in the Settings tab

`PanelRegistry.entries` collects category-keyed builders during
`register_panels`. The host renders them; for v0 the most natural
home is the existing Settings tab's `build_embedded` method, where the
calibration extras already live (see `components/settings.py`).

```python
# waldo_commander/components/settings.py — top-of-file imports
from waldo_commander.plugins import loader_state

# waldo_commander/components/settings.py — inside Settings.build_embedded,
# after the existing sections loop, before the trailing
# simulation_state.notify_changed().
plugin_entries = loader_state().panels.entries
if plugin_entries:
    ui.separator().classes("my-1")
    for category in ("calibration", "tools", "diagnostics", "plugins"):
        matching = [e for e in plugin_entries if e["category"] == category]
        if not matching:
            continue
        for entry in matching:
            ui.label(entry["title"]).classes("text-sm font-medium opacity-80")
            with ui.column().classes("w-full"):
                entry["builder"]()
            ui.separator().classes("my-1")
```

Why in `build_embedded`: `parol6_mcp` registers under
`category="calibration"`, which signals "this belongs alongside
calibration-adjacent settings" — exactly where the existing Settings
tab routes user-facing plugin controls today. Rendering inline keeps
the change footprint minimal and avoids preempting a Jepson decision on
whether plugins eventually warrant a dedicated tab.

Why iterate canonical categories rather than insertion order: each
plugin's `register_panels` runs at load time in directory-walk order,
which is alphabetical and not semantically meaningful. Iterating a
fixed category list groups related plugins together regardless of load
order. A future iteration may surface a dedicated "Plugins" tab in
`_build_left_panels` for plugins that earn first-class real estate
(calibration when extracted is the obvious candidate).

### 11.4 Things deliberately not wired here

- **Per-page-build hooks.** Today's contract does not define a hook
  that fires inside each `@ui.page("/")` invocation. Plugins that need
  per-browser-session setup (calibration's scene-overlay rebuild on
  reload) have to register their own NiceGUI page handlers via
  `host.fastapi_app()` until the contract grows an `on_page_init` hook.
  See TBD list item #11.
- **Browser disconnect.** `host.scene.cleanup_plugin()` exists for
  the framework to call on `client.on_disconnect`, but there's no
  current call site. Once a per-page hook lands, the symmetric
  disconnect hook follows.
- **Live reload.** No filesystem watcher; adding a plugin folder
  requires restarting WC. Acceptable for v0; trivially upgradeable.
