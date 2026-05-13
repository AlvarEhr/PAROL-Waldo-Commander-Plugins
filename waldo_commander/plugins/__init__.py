"""Plugin discovery + loader for Waldo-Commander.

Walks ``waldo_commander/plugins/<plugin-id>/``, imports each child
package, looks for a top-level ``Plugin`` class, instantiates it, and
runs the lifecycle hooks documented in ``docs/PLUGIN_CONTRACT.md``.

The loader is conservative by design: a single plugin's failure (import
error, bad manifest, hook exception) logs and is skipped — it never
aborts WC startup.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from waldo_commander.plugin_host import Host

logger = logging.getLogger(__name__)


_PLUGINS_DIR = Path(__file__).resolve().parent
_PACKAGE_PREFIX = "waldo_commander.plugins"


@dataclass
class LoadedPlugin:
    """Bookkeeping for one loaded plugin instance."""

    plugin_id: str
    instance: Any
    host: Host
    manifest: Any
    panels_registered: bool = False
    mcp_registered: bool = False


@dataclass
class PanelRegistry:
    """Provisional ``panels`` object passed to ``Plugin.register_panels``.

    Final shape is TBD upstream — see PLUGIN_CONTRACT.md §8. For now,
    every plugin's panel builder is collected here so the host can
    render them in a single pass after all plugins have loaded.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        category: str,
        title: str,
        builder: Callable[[], None],
    ) -> None:
        self.entries.append(
            {"category": category, "title": title, "builder": builder},
        )


@dataclass
class LoaderState:
    """Process-wide loader state, primarily for tests + introspection."""

    loaded: list[LoadedPlugin] = field(default_factory=list)
    panels: PanelRegistry = field(default_factory=PanelRegistry)


_state: LoaderState | None = None


def loader_state() -> LoaderState:
    """Return the singleton ``LoaderState``, creating it on first access."""
    global _state
    if _state is None:
        _state = LoaderState()
    return _state


def discover_plugin_ids() -> list[str]:
    """Return the list of plugin folder names under ``plugins/``.

    A folder qualifies as a plugin if it contains an ``__init__.py`` and
    its name is a valid lowercase Python identifier (no leading underscore).
    """
    ids: list[str] = []
    for child in sorted(_PLUGINS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "__init__.py").exists():
            continue
        name = child.name
        if name.startswith("_") or not name.isidentifier() or name != name.lower():
            logger.debug("plugins: skipping %r — not a valid plugin id", name)
            continue
        ids.append(name)
    return ids


def _import_plugin_class(plugin_id: str) -> type | None:
    """Import the plugin package and return its ``Plugin`` class, or None."""
    module_name = f"{_PACKAGE_PREFIX}.{plugin_id}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        logger.error("plugin %s: import failed: %s", plugin_id, e)
        return None
    cls = getattr(module, "Plugin", None)
    if cls is None or not isinstance(cls, type):
        logger.error(
            "plugin %s: no top-level ``Plugin`` class found in %s",
            plugin_id, module_name,
        )
        return None
    return cls


def _manifest_id(manifest: Any, fallback: str) -> str:
    """Read ``manifest.name`` if present; else use the folder name."""
    name = getattr(manifest, "name", None)
    if isinstance(name, str) and name:
        return name
    return fallback


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it's awaitable, else return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


async def load_all() -> LoaderState:
    """Discover, instantiate, and load every plugin.

    Idempotent: returns the existing ``LoaderState`` if any plugins are
    already loaded. Tests can reset via :func:`reset_loader_for_tests`.
    """
    state = loader_state()
    if state.loaded:
        logger.debug("plugins: load_all called but %d already loaded", len(state.loaded))
        return state

    for plugin_id in discover_plugin_ids():
        cls = _import_plugin_class(plugin_id)
        if cls is None:
            continue

        try:
            instance = cls()
        except (TypeError, RuntimeError) as e:
            logger.error("plugin %s: instantiation failed: %s", plugin_id, e)
            continue

        manifest = getattr(instance, "manifest", None)
        if manifest is None:
            logger.error("plugin %s: missing ``manifest`` attribute; skipping", plugin_id)
            continue

        # The folder name wins as the plugin id; the manifest can claim
        # a different name but the loader uses the folder for namespacing
        # so the two must agree. Warn loudly when they don't.
        declared = _manifest_id(manifest, plugin_id)
        if declared != plugin_id:
            logger.warning(
                "plugin %s: manifest.name=%r disagrees with folder name; "
                "using folder name for namespacing",
                plugin_id, declared,
            )

        host = Host(plugin_id=plugin_id)

        enabled_fn = getattr(instance, "enabled", None)
        if callable(enabled_fn):
            try:
                if not enabled_fn(host):
                    logger.info("plugin %s: disabled by enabled()", plugin_id)
                    continue
            except (RuntimeError, AttributeError) as e:
                logger.error("plugin %s: enabled() raised %s; skipping", plugin_id, e)
                continue

        on_load = getattr(instance, "on_load", None)
        if callable(on_load):
            try:
                await _maybe_await(on_load(host))
            except (RuntimeError, ImportError, AttributeError) as e:
                logger.error("plugin %s: on_load failed: %s", plugin_id, e)
                continue

        loaded = LoadedPlugin(
            plugin_id=plugin_id,
            instance=instance,
            host=host,
            manifest=manifest,
        )

        register_panels = getattr(instance, "register_panels", None)
        if callable(register_panels):
            try:
                register_panels(host, state.panels)
                loaded.panels_registered = True
            except (RuntimeError, AttributeError) as e:
                logger.warning(
                    "plugin %s: register_panels raised %s", plugin_id, e,
                )

        register_mcp = getattr(instance, "register_mcp_tools", None)
        if callable(register_mcp):
            try:
                register_mcp(host, host.mcp)
                loaded.mcp_registered = True
            except (RuntimeError, AttributeError) as e:
                logger.warning(
                    "plugin %s: register_mcp_tools raised %s", plugin_id, e,
                )

        state.loaded.append(loaded)
        logger.info(
            "plugin %s loaded (version=%s, panels=%s, mcp=%s)",
            plugin_id,
            getattr(manifest, "version", "?"),
            loaded.panels_registered,
            loaded.mcp_registered,
        )

    return state


async def unload_all() -> None:
    """Run ``on_unload`` for each loaded plugin in reverse load order."""
    state = loader_state()
    for loaded in reversed(state.loaded):
        on_unload = getattr(loaded.instance, "on_unload", None)
        if callable(on_unload):
            try:
                await _maybe_await(on_unload(loaded.host))
            except (RuntimeError, AttributeError) as e:
                logger.warning(
                    "plugin %s: on_unload raised %s", loaded.plugin_id, e,
                )
        try:
            loaded.host.cleanup()
        except (RuntimeError, AttributeError) as e:
            logger.debug("plugin %s: host cleanup raised %s", loaded.plugin_id, e)
    state.loaded.clear()


def reset_loader_for_tests() -> None:
    """Drop loader state. Test fixtures only."""
    global _state
    _state = None


__all__ = [
    "LoadedPlugin",
    "LoaderState",
    "PanelRegistry",
    "discover_plugin_ids",
    "load_all",
    "loader_state",
    "reset_loader_for_tests",
    "unload_all",
]
