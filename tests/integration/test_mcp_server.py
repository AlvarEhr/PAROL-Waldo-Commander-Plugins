"""Integration test for the parol6_mcp HTTP transport.

Drives the mounted ``/mcp`` route via httpx + ASGI transport — no network
socket, no uvicorn. Exercises three things:

1. ``tools/list`` returns all 8 v0 tools with the expected annotations.
2. ``parol6_get_joints`` handles a missing RobotClient without raising —
   should return ``angles_deg=None`` (no broadcast cached) cleanly.
3. ``parol6_move_j`` round-trips the host.motion stub rejection
   (``"host.motion not yet wired"``) into a structured MCP isError
   payload via ``format_rejection``.

Skips when ``fastmcp``, ``httpx``, or ``asgi_lifespan`` are missing — same
pattern as ``tests/test_plugin_loader.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("httpx")
asgi_lifespan = pytest.importorskip("asgi_lifespan")

import httpx
from asgi_lifespan import LifespanManager
from nicegui import app as nicegui_app

from waldo_commander.plugin_host import reset_runtime_for_tests
from waldo_commander.plugins import (
    load_all,
    reset_loader_for_tests,
    unload_all,
)


# MCP streamable HTTP requires both Accept types per spec; the server
# picks one based on capability. JSON or SSE may come back, so callers
# parse defensively below.
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

EXPECTED_TOOLS = {
    "parol6_get_joints",
    "parol6_get_pose",
    "parol6_get_tool_state",
    "parol6_check_collision",
    "parol6_move_j",
    "parol6_force_move_j",
    "parol6_halt",
    "parol6_resume",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Drop loader + runtime state either side of each test."""
    reset_loader_for_tests()
    reset_runtime_for_tests()
    yield
    reset_loader_for_tests()
    reset_runtime_for_tests()


@pytest.fixture
async def mcp_client():
    """Load parol6_mcp into the live NiceGUI FastAPI app, enter the app
    lifespan (which fires the plugin's FastMCP session-manager startup
    hook), and yield an httpx AsyncClient pointed at the in-process app.
    """
    state = await load_all()
    assert any(p.plugin_id == "parol6_mcp" for p in state.loaded), (
        "parol6_mcp failed to load — check fastmcp is installed."
    )

    async with LifespanManager(nicegui_app):
        transport = httpx.ASGITransport(app=nicegui_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver",
        ) as client:
            session_id = await _initialize(client)
            yield _BoundClient(client, session_id)

    await unload_all()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _BoundClient:
    """httpx AsyncClient bound to an MCP session id (or none in stateless mode)."""

    def __init__(self, http: httpx.AsyncClient, session_id: str | None) -> None:
        self.http = http
        self.session_id = session_id

    def _headers(self) -> dict[str, str]:
        h = dict(MCP_HEADERS)
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    async def rpc(self, method: str, params: dict | None = None, id_: int = 1) -> dict:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": id_, "method": method}
        if params is not None:
            body["params"] = params
        r = await self.http.post("/mcp/", json=body, headers=self._headers())
        assert r.status_code == 200, (
            f"{method} returned {r.status_code}: {r.text!r}"
        )
        return _parse_response(r)


def _parse_response(r: httpx.Response) -> dict:
    """Parse a streamable-HTTP response. The body is either JSON or SSE
    framed with ``data:`` lines depending on the server's negotiation."""
    ct = r.headers.get("content-type", "")
    if "event-stream" in ct:
        for line in r.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                payload = stripped[5:].strip()
                if payload:
                    return json.loads(payload)
        raise AssertionError(f"no data: line in SSE response: {r.text!r}")
    return r.json()


async def _initialize(http: httpx.AsyncClient) -> str | None:
    """MCP session handshake. Returns the session id from the response
    header (stateful) or None (stateless)."""
    body = {
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "parol6-mcp-test", "version": "0"},
        },
    }
    r = await http.post("/mcp/", json=body, headers=MCP_HEADERS)
    assert r.status_code == 200, f"initialize returned {r.status_code}: {r.text!r}"
    sid = r.headers.get("Mcp-Session-Id")
    # Notifications/initialized is required by spec after initialize.
    notify_headers = dict(MCP_HEADERS)
    if sid:
        notify_headers["Mcp-Session-Id"] = sid
    await http.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers=notify_headers,
    )
    return sid


def _tool_call_text(rpc_response: dict) -> dict:
    """Extract the tool's return value from an MCP tools/call response.

    FastMCP wraps tool returns into ``result.content`` as a list of
    content blocks; for our JSON-serialisable dicts the first block is
    ``{type: "text", text: "<json>"}``.
    """
    assert "result" in rpc_response, f"no result in {rpc_response!r}"
    content = rpc_response["result"]["content"]
    assert content, f"empty content in {rpc_response!r}"
    return json.loads(content[0]["text"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_tools_list_returns_eight_v0_tools(mcp_client: _BoundClient) -> None:
    body = await mcp_client.rpc("tools/list")
    tools = {t["name"]: t for t in body["result"]["tools"]}
    assert set(tools.keys()) == EXPECTED_TOOLS, (
        f"tool set drift: {set(tools.keys()) ^ EXPECTED_TOOLS}"
    )

    # Spot-check the load-bearing annotations.
    assert tools["parol6_get_joints"]["annotations"]["readOnlyHint"] is True
    assert tools["parol6_get_joints"]["annotations"]["idempotentHint"] is True

    assert tools["parol6_move_j"]["annotations"]["readOnlyHint"] is False
    assert tools["parol6_move_j"]["annotations"]["destructiveHint"] is False

    assert tools["parol6_force_move_j"]["annotations"]["destructiveHint"] is True

    assert tools["parol6_halt"]["annotations"]["idempotentHint"] is True


async def test_get_joints_handles_empty_state_cache(mcp_client: _BoundClient) -> None:
    """No broadcast feeds the cache in this test, so joint_angles_deg()
    returns an empty tuple. The tool maps that to ``angles_deg=None``
    without raising."""
    body = await mcp_client.rpc(
        "tools/call",
        {"name": "parol6_get_joints", "arguments": {}},
    )
    result = _tool_call_text(body)
    assert result == {"angles_deg": None}


async def test_move_j_returns_host_motion_stub_rejection(
    mcp_client: _BoundClient,
) -> None:
    """The stubbed host.motion.move_j returns (False, 'host.motion not yet
    wired', None); the tool wraps that into a structured rejection."""
    body = await mcp_client.rpc(
        "tools/call",
        {
            "name": "parol6_move_j",
            "arguments": {
                "angles_deg": [0.0, -90.0, 0.0, 0.0, 0.0, 0.0],
                "speed": 0.3,
                "accel": 0.5,
                "wait": True,
            },
        },
    )
    result = _tool_call_text(body)
    assert result["ok"] is False
    assert result["isError"] is True
    rejection = result["rejection"]
    assert "host.motion not yet wired" in rejection["reason"]
    # detail=None from the stub → manager_ready defaults to True in the
    # Pydantic model; the next_steps text is the generic remediation.
    assert "parol6_check_collision" in rejection["next_steps"]
