"""Plugin host — the surface plugins talk to instead of reaching into WC.

This is a provisional implementation that backs the contract documented in
``docs/PLUGIN_CONTRACT.md``. Methods that need deeper hooks into existing
WC infrastructure are stubbed with a logged warning and a safe default;
the shape matches the contract so plugins can build against it now and
the implementation fills in as Jepson's framework lands or the proto
plugins (parol6_mcp first, calibration second) drive the requirements.

One ``PluginRuntime`` exists per WC process. Each loaded plugin gets its
own ``Host`` instance, namespaced by ``plugin_id`` for storage,
scene-group ownership, and event subscriptions.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from nicegui import ui
    from waldoctl import RobotClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared process-level runtime
# ---------------------------------------------------------------------------


@dataclass
class _SceneHandle:
    """Tracks a scene element (group or timer) owned by a plugin."""

    kind: str        # "group" | "timer"
    obj: Any
    plugin_id: str


@dataclass
class _Subscription:
    """One event-bus subscriber."""

    event: str
    callback: Callable[..., None]
    plugin_id: str


@dataclass
class PluginRuntime:
    """Process-wide plugin infrastructure. One instance, shared by all hosts."""

    motion_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    subscriptions: list[_Subscription] = field(default_factory=list)
    scene_handles: list[_SceneHandle] = field(default_factory=list)
    startup_hooks: list[Callable[[], Any]] = field(default_factory=list)
    shutdown_hooks: list[Callable[[], Any]] = field(default_factory=list)
    mcp_registrations: list[dict[str, Any]] = field(default_factory=list)
    _state_cache_lock: threading.RLock = field(default_factory=threading.RLock)
    _state_cache: dict[str, Any] = field(default_factory=dict)

    def cache_read(self, key: str, default: Any = None) -> Any:
        with self._state_cache_lock:
            return self._state_cache.get(key, default)

    def cache_write(self, key: str, value: Any) -> None:
        with self._state_cache_lock:
            self._state_cache[key] = value


_runtime: PluginRuntime | None = None


def get_runtime() -> PluginRuntime:
    """Return the process-wide runtime, creating it on first access."""
    global _runtime
    if _runtime is None:
        _runtime = PluginRuntime()
    return _runtime


def reset_runtime_for_tests() -> None:
    """Drop the runtime singleton. Test fixtures only."""
    global _runtime
    _runtime = None


# ---------------------------------------------------------------------------
# Host facade — per-plugin
# ---------------------------------------------------------------------------


class _StorageProxy:
    """Dict-like proxy that prefixes every key with ``plugin:<id>:``."""

    def __init__(self, plugin_id: str, scope: str) -> None:
        # scope is "general" or "user"; lookup is lazy so import order
        # is forgiving.
        self._plugin_id = plugin_id
        self._scope = scope

    def _backing(self) -> Any:
        from nicegui import app  # noqa: PLC0415

        return getattr(app.storage, self._scope)

    def _full_key(self, key: str) -> str:
        return f"plugin:{self._plugin_id}:{key}"

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._backing().get(self._full_key(key), default)
        except RuntimeError as e:
            logger.debug(
                "storage.%s read outside request context (%s); returning default",
                self._scope, e,
            )
            return default

    def __getitem__(self, key: str) -> Any:
        return self._backing()[self._full_key(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._backing()[self._full_key(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._backing()[self._full_key(key)]

    def __contains__(self, key: str) -> bool:
        try:
            return self._full_key(key) in self._backing()
        except RuntimeError:
            return False


class _StorageNS:
    """Wraps ``host.storage.general`` and ``host.storage.user``."""

    def __init__(self, plugin_id: str) -> None:
        self.general = _StorageProxy(plugin_id, "general")
        self.user = _StorageProxy(plugin_id, "user")


class _StateAPI:
    """Typed read-only accessors, safe from any thread or context.

    Backed by the runtime's state cache. The cache is populated by the
    main app's state-broadcast loop; until that wiring lands the
    accessors return conservative defaults rather than raising.
    """

    def __init__(self, runtime: PluginRuntime) -> None:
        self._rt = runtime

    def active_tool_key(self) -> str:
        # Storage write happens in main; mirror cached value.
        cached = self._rt.cache_read("active_tool_key")
        if cached is not None:
            return str(cached)
        # Fallback: try the storage path directly. Returns "" if no
        # request scope (which is the right answer for "no UI yet").
        try:
            from nicegui import app  # noqa: PLC0415

            return str(app.storage.general.get("selected_tool", "") or "")
        except RuntimeError:
            return ""

    def active_tool_variant(self) -> str:
        cached = self._rt.cache_read("active_tool_variant")
        if cached is not None:
            return str(cached)
        try:
            from nicegui import app  # noqa: PLC0415

            tool = self.active_tool_key()
            if not tool:
                return ""
            return str(app.storage.general.get(f"tool_variant_{tool}", "") or "")
        except RuntimeError:
            return ""

    def simulator_active(self) -> bool:
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            return bool(robot_state.simulator_active)
        except ImportError:
            return False

    def joint_angles_deg(self) -> tuple[float, ...]:
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            return tuple(float(v) for v in robot_state.angles.deg)
        except ImportError:
            return ()

    def tcp_pose(self) -> tuple[float, ...]:
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            return (
                float(robot_state.x),
                float(robot_state.y),
                float(robot_state.z),
                float(robot_state.rx),
                float(robot_state.ry),
                float(robot_state.rz),
            )
        except ImportError:
            return ()

    def tool_status(self) -> Any:
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            return robot_state.tool_status
        except ImportError:
            return None

    def connected(self) -> bool:
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415

            return bool(robot_state.connected)
        except ImportError:
            return False


class _MotionAPI:
    """Gated motion dispatch.

    The contract's 3-tuple ``(ok, reason, detail)`` matches the shape
    ``parol6_mcp`` expects so the gate rejection round-trips into MCP
    ``isError`` payloads. Real wiring into ``safe_motion`` /
    ``validate_joint_trajectory_core`` lands when the consolidated
    motion-dispatch path (TBD #17) ships; until then move_j returns
    ``(False, "host.motion not yet wired", None)`` so callers see a
    clean rejection instead of a silent dispatch.
    """

    def __init__(self, runtime: PluginRuntime, host: Host) -> None:
        self._rt = runtime
        self._host = host

    async def move_j(
        self,
        angles_deg: list[float],
        *,
        speed: float = 0.3,
        accel: float = 0.5,
        wait: bool = True,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        async with self._rt.motion_lock:
            return await self._dispatch_gated(
                angles_deg, speed=speed, accel=accel, wait=wait,
            )

    async def check(
        self, q_from_deg: list[float], q_to_deg: list[float],
    ) -> dict[str, Any]:
        # Read-only — does not acquire the dispatch lock.
        try:
            from parol6_vision.calibration.collision_core import (  # noqa: PLC0415
                validate_joint_trajectory_core,
            )

            return await asyncio.to_thread(
                validate_joint_trajectory_core, q_from_deg, q_to_deg,
            )
        except ImportError:
            return {
                "safe": False,
                "manager_ready": False,
                "reason": "collision-core not available",
            }

    async def move_j_unchecked(
        self,
        angles_deg: list[float],
        *,
        speed: float = 0.3,
        accel: float = 0.5,
        wait: bool = True,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        async with self._rt.motion_lock:
            return await self._dispatch_unchecked(
                angles_deg, speed=speed, accel=accel, wait=wait,
            )

    async def _dispatch_gated(
        self,
        angles_deg: list[float],
        *,
        speed: float,
        accel: float,
        wait: bool,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        # Stub — flips to the real safe_motion dispatcher when the
        # consolidated path is wired. Logs once so the integrator sees
        # this in practice rather than silently no-op'ing.
        logger.warning(
            "host.motion.move_j called but consolidated dispatch is not yet "
            "wired; returning structured rejection (caller: gated path).",
        )
        return (False, "host.motion not yet wired", None)

    async def _dispatch_unchecked(
        self,
        angles_deg: list[float],
        *,
        speed: float,
        accel: float,
        wait: bool,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        # Same as above for the unchecked path. The unchecked variant
        # exists to keep the API shape stable; the implementation just
        # bypasses the gate.
        client = self._host.robot_client()
        if client is None:
            return (False, "no robot client available", None)
        try:
            rc = await client.move_j(
                angles_deg, speed=speed, accel=accel, wait=wait,
            )
        except (RuntimeError, ConnectionError) as e:
            return (False, f"dispatch error: {type(e).__name__}: {e}", None)
        if rc < 0:
            return (False, f"controller rejected (rc={rc})", None)
        return (True, "unchecked-dispatch", {"command_index": int(rc)})


class _SceneAPI:
    """Scene-overlay registration scoped to the calling plugin.

    Groups and timers tracked here are auto-deleted/cancelled when the
    plugin unloads or the browser disconnects. Mirrors today's manual
    teardown bookkeeping in calibration_overlays/panel.py.
    """

    def __init__(self, runtime: PluginRuntime, plugin_id: str) -> None:
        self._rt = runtime
        self._plugin_id = plugin_id

    def add_group(self, name: str, *, parent: str = "root") -> Any:
        # Real impl resolves ``parent`` to a NiceGUI parent element
        # (urdf_scene.scene root, tcp_anchor, world_axes). For now the
        # stub returns None and logs; callers should defensively handle
        # ``None`` until wiring lands.
        logger.debug(
            "scene.add_group(name=%s, parent=%s) for plugin=%s — not yet wired",
            name, parent, self._plugin_id,
        )
        return None

    def add_timer(self, interval_s: float, callback: Callable[[], None]) -> Any:
        try:
            from nicegui import ui  # noqa: PLC0415

            timer = ui.timer(interval_s, callback)
        except (ImportError, RuntimeError) as e:
            logger.debug("scene.add_timer skipped (%s)", e)
            return None
        self._rt.scene_handles.append(
            _SceneHandle(kind="timer", obj=timer, plugin_id=self._plugin_id),
        )
        return timer

    def cleanup_plugin(self) -> None:
        # Called by the framework on plugin teardown. Iterates the
        # plugin's tracked handles and disposes each.
        remaining: list[_SceneHandle] = []
        for h in self._rt.scene_handles:
            if h.plugin_id != self._plugin_id:
                remaining.append(h)
                continue
            try:
                if h.kind == "timer":
                    h.obj.cancel()
                elif h.kind == "group":
                    h.obj.delete()
            except (AttributeError, RuntimeError) as e:
                logger.debug("scene handle teardown error (%s)", e)
        self._rt.scene_handles = remaining


class _EditorAPI:
    """Editor-coherent file writes per the B1 finding.

    Writes to ``PROGRAM_DIR/filename`` and syncs any open EditorTab on
    that file so the in-memory copy doesn't go stale against disk.
    """

    def __init__(self, plugin_id: str) -> None:
        self._plugin_id = plugin_id

    async def write_file(self, filename: str, text: str) -> bool:
        # Resolve PROGRAM_DIR lazily — keeps plugin_host importable
        # before the editor panel has built.
        try:
            from pathlib import Path  # noqa: PLC0415

            from waldo_commander.components.editor import EditorPanel  # noqa: PLC0415
            from waldo_commander.state import editor_tabs_state  # noqa: PLC0415
        except ImportError as e:
            logger.warning("editor.write_file skipped (%s)", e)
            return False

        program_dir: Path = getattr(EditorPanel, "PROGRAM_DIR", None)
        if program_dir is None:
            logger.warning("editor.write_file: PROGRAM_DIR unavailable")
            return False

        target = program_dir / filename
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as e:
            logger.warning("editor.write_file: disk write failed (%s)", e)
            return False

        # Sync any open tab so the editor doesn't show stale content.
        full_path = str(target)
        tab = editor_tabs_state.find_tab_by_path(full_path)
        if tab is not None:
            tab.content = text
            tab.saved_content = text
            try:
                from nicegui import ui  # noqa: PLC0415

                ui.notify(
                    f"Plugin {self._plugin_id!r} updated {filename}",
                    color="info", position="top",
                )
            except (ImportError, RuntimeError) as e:
                logger.debug("editor.write_file: notify skipped (%s)", e)
        return True


class _MCPRegistry:
    """Cross-plugin MCP-tool registry.

    Receives declarative tool registrations from any plugin. The
    parol6_mcp plugin (or any future replacement) reads this list to
    expose tools over MCP. For v0, parol6_mcp registers its eight tools
    directly on its module-level FastMCP instance and does not consume
    this registry — entries here are forward-looking.
    """

    def __init__(self, runtime: PluginRuntime, plugin_id: str) -> None:
        self._rt = runtime
        self._plugin_id = plugin_id

    def register(
        self,
        name: str,
        *,
        description: str,
        handler: Callable[..., Any],
        input_schema: Any = None,
        output_schema: Any = None,
        side_effects: list[str] | None = None,
        requires: list[str] | None = None,
        annotations: dict[str, bool] | None = None,
    ) -> None:
        entry = {
            "name": name,
            "description": description,
            "handler": handler,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "side_effects": list(side_effects or []),
            "requires": list(requires or []),
            "annotations": dict(annotations or {}),
            "plugin_id": self._plugin_id,
        }
        self._rt.mcp_registrations.append(entry)
        logger.debug(
            "mcp.register: %s (plugin=%s, side_effects=%s)",
            name, self._plugin_id, entry["side_effects"],
        )


class Host:
    """Per-plugin host facade.

    A new ``Host`` is constructed for each loaded plugin and passed to
    every lifecycle hook. Sub-namespaces are scoped to the plugin id
    where it matters (storage, scene-group ownership, MCP registrations).
    Process-wide singletons (motion lock, FastAPI app, event bus) live
    on the shared ``PluginRuntime`` accessed via :func:`get_runtime`.
    """

    def __init__(self, plugin_id: str, *, runtime: PluginRuntime | None = None) -> None:
        self.plugin_id = plugin_id
        self._rt = runtime or get_runtime()
        self.storage = _StorageNS(plugin_id)
        self.state = _StateAPI(self._rt)
        self.motion = _MotionAPI(self._rt, self)
        self.scene = _SceneAPI(self._rt, plugin_id)
        self.editor = _EditorAPI(plugin_id)
        self.mcp = _MCPRegistry(self._rt, plugin_id)

    # ---- Robot client -----------------------------------------------------

    def robot_client(self) -> RobotClient | None:
        """Return the live async RobotClient, or None pre-connect.

        Shared with the GUI — plugins do not call ``create_async_client``
        themselves. A future iteration may add ``wait_for_connect()``.
        """
        try:
            from waldo_commander import main as wc_main  # noqa: PLC0415
        except ImportError as e:
            logger.debug("robot_client: WC main not importable (%s)", e)
            return None
        return getattr(wc_main, "client", None)

    # ---- FastAPI handle ---------------------------------------------------

    def fastapi_app(self) -> FastAPI:
        """Return the NiceGUI app (a FastAPI subclass).

        Plugins mount sub-apps on this; parol6_mcp mounts FastMCP's
        Starlette app at ``/mcp`` via this handle.
        """
        from nicegui import app  # noqa: PLC0415

        return app  # type: ignore[return-value]

    # ---- Lifecycle hooks --------------------------------------------------

    def on_startup(self, coro: Callable[[], Any]) -> None:
        from nicegui import app  # noqa: PLC0415

        self._rt.startup_hooks.append(coro)
        app.on_startup(coro)

    def on_shutdown(self, coro: Callable[[], Any]) -> None:
        from nicegui import app  # noqa: PLC0415

        self._rt.shutdown_hooks.append(coro)
        app.on_shutdown(coro)

    # ---- Event bus --------------------------------------------------------

    def subscribe(self, event: str, callback: Callable[..., None]) -> None:
        self._rt.subscriptions.append(
            _Subscription(event=event, callback=callback, plugin_id=self.plugin_id),
        )

    def publish(self, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        for sub in list(self._rt.subscriptions):
            if sub.event != event:
                continue
            try:
                sub.callback(payload)
            except (RuntimeError, TypeError, ValueError) as e:
                logger.warning(
                    "event %s subscriber (plugin=%s) raised: %s",
                    event, sub.plugin_id, e,
                )

    # ---- Teardown helper --------------------------------------------------

    def cleanup(self) -> None:
        """Drop all per-plugin runtime state. Called by the loader on unload."""
        self.scene.cleanup_plugin()
        self._rt.subscriptions = [
            s for s in self._rt.subscriptions if s.plugin_id != self.plugin_id
        ]
        self._rt.mcp_registrations = [
            m for m in self._rt.mcp_registrations if m["plugin_id"] != self.plugin_id
        ]


__all__ = [
    "Host",
    "PluginRuntime",
    "get_runtime",
    "reset_runtime_for_tests",
]
