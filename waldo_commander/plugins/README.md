# Waldo-Commander plugins

Drop a folder here, restart WC, and the framework picks it up. This page
is the smallest-possible orientation; the canonical contract lives in
[`docs/PLUGIN_CONTRACT.md`](../../docs/PLUGIN_CONTRACT.md).

## Status

Proto-framework, pre-upstream. Jepson's eventual framework supersedes
this; the shapes here track what we've agreed will likely fit cleanly
into that. Two plugins live or are landing under this tree:

- `parol6_mcp/` — exposes PAROL6 as an MCP server. In-tree, v0 shipping.
- `calibration/` — extraction of the calibration_overlays package from
  Alvar's `parol6-vision-calibration` branch. Not migrated yet; will
  become the second user of the contract.

## Anatomy of a plugin

```
waldo_commander/plugins/
  my_plugin/
    __init__.py        # exports a top-level `Plugin` class + `PluginManifest`
    safety.py          # implementation modules, your structure
    README.md          # optional, user-facing
```

The folder name (`my_plugin`) is the plugin id. It must be a valid
lowercase Python identifier; the framework uses it as the namespace
prefix for storage keys, MCP tool names, scene-group ownership, etc.

`__init__.py` exports two things:

```python
from pydantic import BaseModel, ConfigDict

class PluginManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = "my_plugin"
    version: str = "0.1.0"
    display_name: str = "My Plugin"
    description: str = "..."

class Plugin:
    manifest = PluginManifest()

    async def on_load(self, host) -> None:
        # mount routes, register panels, install custom tools, etc.
        ...

    async def on_unload(self, host) -> None:
        ...
```

Only `manifest` is required. Every method below is optional:

- `enabled(host) -> bool` — gate the whole plugin
- `on_load(host)` — side-effect entry point (async or sync)
- `register_panels(host, panels)` — `panels.add(category, title, builder)`
- `register_mcp_tools(host, mcp)` — declarative MCP-tool registration
- `on_unload(host)` — teardown in reverse load order

If a method isn't present, the framework skips it silently.

## The host

The single `host` argument is your interface to Waldo-Commander. Plugins
do not import `nicegui.app`, `waldo_commander.state`, or backend modules
directly; everything goes through `host`. See PLUGIN_CONTRACT.md §4 for
the full surface. Quick reference:

| Call | What |
|---|---|
| `host.motion.move_j(angles, …)` | Gated joint move, returns `(ok, reason, detail)` |
| `host.motion.check(q_from, q_to)` | Pre-flight collision check, no dispatch |
| `host.motion.move_j_unchecked(…)` | Bypass the gate |
| `host.robot_client()` | Live async RobotClient (or None pre-connect) |
| `host.state.active_tool_key()` | GUI's canonical tool key, custom-tool-aware |
| `host.state.joint_angles_deg()` | Last broadcast joint snapshot |
| `host.fastapi_app()` | NiceGUI app instance (FastAPI subclass) |
| `host.on_startup(coro)` / `host.on_shutdown(coro)` | uvicorn lifecycle hooks |
| `host.storage.general[key]` | Per-plugin persistent storage |
| `host.scene.add_group(name)` | Plugin-owned scene overlay, auto-cleaned |
| `host.editor.write_file(name, text)` | Write a program and sync the open tab |
| `host.subscribe(event, cb)` | Listen for `tool_change`, `robot_connect`, etc. |
| `host.mcp.register(name, …)` | Expose a tool over MCP |

## Loading order

1. WC scans `plugins/` and imports each child package.
2. `Plugin()` is instantiated.
3. `enabled(host)` runs if defined; False means stop.
4. `on_load(host)` runs (async or sync, framework awaits if needed).
5. `register_panels(host, panels)` and `register_mcp_tools(host, mcp)`
   run after `on_load`.
6. The plugin runs for the lifetime of the WC process.
7. On WC shutdown, `on_unload(host)` runs in reverse load order.

A plugin that fails to load is logged and skipped — it never aborts
WC startup. Errors land in the WC log at error level.

## What changes when Jepson ships

The shape above is provisional. When Jepson's framework lands, the
mechanical migration for each plugin is small:

- The `PluginManifest` Pydantic model swaps for the canonical import.
- Method names may rename (`on_load` → `setup`, etc.).
- Some host capabilities expand (a real motion-dispatch path lands
  behind `host.motion.move_j`, the scene-group teardown stops being a
  stub).

The TBDs are catalogued in PLUGIN_CONTRACT.md §9.

## Writing your first plugin

Read `parol6_mcp/__init__.py` first — it's a complete, working plugin.
That plus PLUGIN_CONTRACT.md gives you the full shape.
