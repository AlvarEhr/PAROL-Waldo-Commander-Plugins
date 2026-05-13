"""Smoke test for the plugin discovery + load lifecycle.

Backend-only — does not boot a NiceGUI request scope. Verifies the loader
walks ``waldo_commander/plugins/``, imports each plugin package,
instantiates its ``Plugin`` class, runs ``on_load(host)`` without
raising, and tears the same set down on ``unload_all()``.

Skips when ``fastmcp`` is unavailable: the bundled ``parol6_mcp`` plugin
imports FastMCP at module level, so a missing dep would surface as a
loader skip and the assertions below would fail with a misleading
"parol6_mcp missing" message. Surfacing as a skip is the right signal.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastmcp")

from waldo_commander.plugin_host import reset_runtime_for_tests
from waldo_commander.plugins import (
    discover_plugin_ids,
    load_all,
    reset_loader_for_tests,
    unload_all,
)


@pytest.fixture(autouse=True)
def _reset_loader():
    """Drop loader + runtime singletons either side of each test."""
    reset_loader_for_tests()
    reset_runtime_for_tests()
    yield
    reset_loader_for_tests()
    reset_runtime_for_tests()


async def test_loader_lifecycle() -> None:
    """End-to-end: discover, load parol6_mcp, idempotency, unload."""
    ids = discover_plugin_ids()
    assert "parol6_mcp" in ids, f"parol6_mcp missing from discovered ids: {ids}"

    state = await load_all()
    loaded_ids = [p.plugin_id for p in state.loaded]
    assert "parol6_mcp" in loaded_ids

    loaded = next(p for p in state.loaded if p.plugin_id == "parol6_mcp")
    assert loaded.manifest.name == "parol6_mcp"
    assert loaded.host.plugin_id == "parol6_mcp"

    # on_load actually ran — the plugin module's private host slot was set.
    import waldo_commander.plugins.parol6_mcp as mcp_mod
    assert mcp_mod._host is loaded.host

    # Idempotent — second load_all returns the same LoaderState, no double-load.
    again = await load_all()
    assert again is state
    assert len(again.loaded) == len(state.loaded)

    await unload_all()
    assert state.loaded == []
    # parol6_mcp.Plugin.on_unload clears the module-private host slot.
    assert mcp_mod._host is None
