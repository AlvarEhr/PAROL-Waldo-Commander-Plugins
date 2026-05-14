"""parol6_mcp — Waldo-Commander plugin exposing PAROL6 over MCP.

Mounts a FastMCP Starlette sub-app on the NiceGUI FastAPI server at
``/mcp``. v0 surface is eight tools — three queries, one collision check,
one gated joint move, one force bypass, plus halt/resume. Motion-mutating
tools route through ``host.motion`` so the existing safe_motion +
validate_joint_trajectory gate applies uniformly.

Transport is stateless streamable HTTP, bound to ``127.0.0.1`` (NiceGUI's
default) with no auth in v0. Any LLM host with MCP client support
connects at ``http://127.0.0.1:<wc-port>/mcp``. See
``docs/MCP_SERVER_DESIGN.md`` for the design rationale and
``docs/PLUGIN_CONTRACT.md`` for the host API this plugin calls into.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from fastmcp import Context, FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

from waldo_commander.plugins.parol6_mcp.safety import (
    CollisionRejection,
    format_rejection,
)

if TYPE_CHECKING:
    from waldo_commander.plugin_host import Host

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin manifest
# ---------------------------------------------------------------------------


class PluginManifest(BaseModel):
    """Plugin identity consumed by the loader.

    Matches the v0 shape in ``docs/PLUGIN_CONTRACT.md`` §3. Defined locally
    rather than imported because plugin-arch's contract leaves the class to
    each plugin for now; the loader only requires the attribute name and
    field set, not a shared type.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "parol6_mcp"
    version: str = "0.1.0"
    display_name: str = "PAROL6 MCP"
    description: str = (
        "Exposes the active PAROL6 robot as an MCP server over streamable "
        "HTTP. Any MCP-spec-compliant LLM client (Claude, Gemini, OpenAI, "
        "Continue, Goose, OpenClaw, custom scripts) can connect at /mcp."
    )
    mount_path: str = "/mcp"


# ---------------------------------------------------------------------------
# Defensive coercion at the MCP boundary
# ---------------------------------------------------------------------------


def _coerce_to_list(value: Any) -> Any:
    """Accept list, JSON-array string, or delimiter-split string; return list.

    Some MCP hosts JSON-encode list parameters into a string before sending,
    which then fails Pydantic type validation. This pre-validator unwraps
    the stringified form. Returns ``value`` unchanged if it can't be coerced
    so the downstream Pydantic error surfaces the user's actual input.

    Lifted from claude-pair-mcp's ``_coerce_to_str_list`` and generalised.
    Comma-splitting intentionally not supported (paths or labels containing
    commas would silently break).
    """
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.startswith("["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return value
        if "\n" in s:
            return [p.strip() for p in s.splitlines() if p.strip()]
        if ";" in s:
            return [p.strip() for p in s.split(";") if p.strip()]
        return [s]
    return value


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class PoseQueryInput(BaseModel):
    """Frame selector for parol6_get_pose."""

    model_config = ConfigDict(extra="forbid")

    frame: Literal["WRF", "TRF"] = Field(
        default="WRF",
        description="WRF = world frame, TRF = tool-relative frame.",
    )


class MoveJInput(BaseModel):
    """Joint target plus motion profile for parol6_move_j."""

    model_config = ConfigDict(extra="forbid")

    angles_deg: list[float] = Field(
        ...,
        description="Six joint angles in degrees, base to wrist.",
        min_length=6,
        max_length=6,
    )
    speed: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Joint speed scale (0..1).",
    )
    accel: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Joint acceleration scale (0..1).",
    )
    wait: bool = Field(
        default=True,
        description="Block until the move completes.",
    )

    @field_validator("angles_deg", mode="before")
    @classmethod
    def _coerce_angles(cls, v: Any) -> Any:
        return _coerce_to_list(v)


class ForceMoveJInput(MoveJInput):
    """parol6_force_move_j input — adds the explicit unsafe acknowledgement."""

    acknowledge_unsafe: Literal[True] = Field(
        ...,
        description=(
            "Must be true. Confirms the LLM has received explicit user "
            "authorisation to bypass the safety gate."
        ),
    )


class CheckCollisionInput(BaseModel):
    """Joint-pair endpoints for parol6_check_collision."""

    model_config = ConfigDict(extra="forbid")

    q_from_deg: list[float] = Field(
        ...,
        description="Start joint config, degrees, length 6.",
        min_length=6, max_length=6,
    )
    q_to_deg: list[float] = Field(
        ...,
        description="Target joint config, degrees, length 6.",
        min_length=6, max_length=6,
    )

    @field_validator("q_from_deg", "q_to_deg", mode="before")
    @classmethod
    def _coerce_q(cls, v: Any) -> Any:
        return _coerce_to_list(v)


# ---------------------------------------------------------------------------
# Host binding
# ---------------------------------------------------------------------------


# Module-private host reference, set by Plugin.on_load. Tools read it via
# _require_host(); never write to it from a tool body.
_host: "Host | None" = None


def _require_host() -> "Host":
    """Return the bound host or raise. Tool bodies call this first."""
    if _host is None:
        raise RuntimeError(
            "parol6_mcp plugin not loaded — Plugin.on_load(host) was not called."
        )
    return _host


def _serialise_tool_status(status: Any) -> dict[str, Any]:
    """Flatten a waldoctl ToolStatus to a JSON-safe dict for MCP responses."""
    return {
        "key": status.key,
        "state": int(status.state),
        "engaged": bool(status.engaged),
        "part_detected": bool(status.part_detected),
        "positions": list(status.positions),
        "channels": list(status.channels),
    }


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------


mcp = FastMCP("parol6_mcp")


# --- Queries ---------------------------------------------------------------


@mcp.tool(
    annotations={
        "title": "Get joint angles",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parol6_get_joints() -> dict[str, list[float] | None]:
    """Return the PAROL6's current joint angles in degrees.

    Six values, base joint first to wrist last. Reads the cached
    broadcast snapshot first (~50 Hz refresh inside WC); falls back to a
    live ``client.angles()`` query when the cache is empty (standalone
    demo with --connect-parol6, or pre-broadcast startup). Returns null
    only when neither path produces a value.
    """
    host = _require_host()
    angles = host.state.joint_angles_deg()
    if angles:
        return {"angles_deg": list(angles)}
    client = host.robot_client()
    if client is None:
        return {"angles_deg": None}
    live = await client.angles()
    return {"angles_deg": list(live) if live is not None else None}


@mcp.tool(
    annotations={
        "title": "Get TCP pose",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parol6_get_pose(params: PoseQueryInput) -> dict[str, Any]:
    """Return the current TCP pose as [x, y, z, rx, ry, rz] in mm + degrees.

    WRF reads come from the cached broadcast snapshot when available;
    TRF and any cache miss fall through to a live ``client.pose(frame)``
    query. With a gripper tool selected, WRF is the TCP pose, not the
    flange — for flange position run forward kinematics on the joint
    angles.
    """
    host = _require_host()
    if params.frame == "WRF":
        cached = host.state.tcp_pose()
        if cached:
            return {"pose_mm_deg": list(cached), "frame": "WRF"}
    client = host.robot_client()
    if client is None:
        return {"pose_mm_deg": None, "frame": params.frame,
                "reason": "no robot client"}
    pose = await client.pose(params.frame)
    return {
        "pose_mm_deg": list(pose) if pose is not None else None,
        "frame": params.frame,
    }


@mcp.tool(
    annotations={
        "title": "Get tool state",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parol6_get_tool_state() -> dict[str, Any]:
    """Return the active end-effector tool key plus live status.

    Status includes operational state, engagement, normalised positions of
    each motion DOF, and tool-specific channels (current draw for electric
    grippers, etc.). Reads the cached broadcast snapshot first; on cache
    miss falls through to a live status query against the bound
    RobotClient when one is present.
    """
    host = _require_host()
    tool_key = host.state.active_tool_key()
    status = host.state.tool_status()
    if status is not None:
        return {
            "active_tool_key": tool_key,
            "status": _serialise_tool_status(status),
        }
    # Cache miss — fall back to the client's private status primitive.
    # parol6.Robot.create_async_client() binds this through
    # ToolSpec._get_status; calling it directly avoids the tool-binding
    # requirement that the higher-level client.tool API has, so it works
    # even with the standalone AsyncRobotClient the demo runner builds.
    client = host.robot_client()
    if client is None:
        return {"active_tool_key": tool_key, "status": None}
    fetch = getattr(client, "_tool_status", None)
    if fetch is None:
        return {"active_tool_key": tool_key, "status": None}
    try:
        live_status = await fetch()
    except (RuntimeError, OSError, NotImplementedError) as e:
        logger.debug("parol6_get_tool_state: live status query failed (%s)", e)
        return {"active_tool_key": tool_key, "status": None}
    if live_status is None:
        return {"active_tool_key": tool_key, "status": None}
    if not tool_key:
        tool_key = getattr(live_status, "key", "") or ""
    return {
        "active_tool_key": tool_key,
        "status": _serialise_tool_status(live_status),
    }


@mcp.tool(
    annotations={
        "title": "Check joint-move collision",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parol6_check_collision(params: CheckCollisionInput) -> dict[str, Any]:
    """Pre-flight a joint-space move WITHOUT dispatching it.

    Runs the same validate_joint_trajectory check the gated motion tools
    use. Returns whether the move is safe plus the colliding pair on
    rejection. Use this to plan multi-step sequences before committing.
    """
    host = _require_host()
    result = await host.motion.check(params.q_from_deg, params.q_to_deg)
    safe = bool(result.get("safe", False))
    return {
        "safe": safe,
        "manager_ready": bool(result.get("manager_ready", False)),
        "rejection": None if safe else format_rejection(result).model_dump(),
    }


# --- Gated motion ---------------------------------------------------------


@mcp.tool(
    annotations={
        "title": "Move to joint angles (gated)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def parol6_move_j(
    params: MoveJInput,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Move to six-joint target. Safety gate runs first.

    Returns the command index on success or a structured rejection payload
    on gate failure. The gate runs validate_joint_trajectory against the
    live collision environment (active tool meshes, calibrated tablet,
    floor). If the gate is unavailable, the move dispatches anyway and
    manager_ready=false in the response.

    On rejection, the LLM's options are: pick a different target, call
    parol6_check_collision on intermediate configs to plan a route, or —
    after explicit user authorisation — call parol6_force_move_j.
    """
    host = _require_host()
    ok, reason, detail = await host.motion.move_j(
        params.angles_deg,
        speed=params.speed,
        accel=params.accel,
        wait=params.wait,
    )
    if not ok:
        if ctx is not None:
            await ctx.info(f"parol6_move_j rejected: {reason}")
        rejection = format_rejection(
            detail if isinstance(detail, dict) else {"reason": reason}
        )
        return {"ok": False, "isError": True, "rejection": rejection.model_dump()}
    detail = detail or {}
    return {
        "ok": True,
        "reason": reason,
        "command_index": detail.get("command_index") if isinstance(detail, dict) else None,
    }


@mcp.tool(
    annotations={
        "title": "Force-move to joint angles (UNGATED)",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def parol6_force_move_j(
    params: ForceMoveJInput,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Bypass the safety gate and dispatch a joint move.

    ONLY call after the user explicitly authorises the unsafe motion. The
    acknowledge_unsafe flag is required by the input schema; the LLM
    cannot accidentally bypass by setting a flag to a default. This tool
    exists to recover from a parol6_move_j rejection the user has
    determined is a false positive.
    """
    host = _require_host()
    if ctx is not None:
        await ctx.info(
            f"parol6_force_move_j dispatched "
            f"(acknowledge_unsafe={params.acknowledge_unsafe})"
        )
    ok, reason, detail = await host.motion.move_j_unchecked(
        params.angles_deg,
        speed=params.speed,
        accel=params.accel,
        wait=params.wait,
    )
    if not ok:
        return {"ok": False, "isError": True, "reason": reason}
    detail = detail or {}
    return {
        "ok": True,
        "reason": reason or "unchecked-dispatch",
        "command_index": detail.get("command_index") if isinstance(detail, dict) else None,
    }


# --- Always-allowed control ------------------------------------------------


@mcp.tool(
    annotations={
        "title": "Halt all motion",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parol6_halt() -> dict[str, Any]:
    """Immediate stop. Always allowed; bypasses the dispatch lock so it
    works while another motion command holds the lock.

    Leaves the controller disabled — call parol6_resume before sending
    further motion commands.
    """
    host = _require_host()
    client = host.robot_client()
    if client is None:
        return {"ok": False, "command_index": -1, "reason": "no robot client"}
    rc = await client.halt()
    return {"ok": rc >= 0, "command_index": int(rc)}


@mcp.tool(
    annotations={
        "title": "Resume after halt",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def parol6_resume() -> dict[str, Any]:
    """Re-enable the controller after a halt or e-stop.

    Required before any motion command if the previous operation ended
    with parol6_halt or the controller went into an error state.
    """
    host = _require_host()
    client = host.robot_client()
    if client is None:
        return {"ok": False, "command_index": -1, "reason": "no robot client"}
    rc = await client.resume()
    return {"ok": rc >= 0, "command_index": int(rc)}


# ---------------------------------------------------------------------------
# Plugin entry
# ---------------------------------------------------------------------------


class Plugin:
    """Waldo-Commander plugin exposing PAROL6 over MCP.

    Lifecycle hooks per ``docs/PLUGIN_CONTRACT.md`` §2:
      - ``on_load(host)``: mount FastMCP at ``/mcp``, register lifespan.
      - ``register_panels(host, panels)``: status panel under "calibration".
      - ``on_unload(host)``: tear down the lifespan + clear host reference.

    ``register_mcp_tools`` is deliberately not implemented — v0 serves its
    tools from this plugin's own ``mcp`` instance with ``@mcp.tool``, not
    via ``host.mcp.register`` (see contract §5; that path is forward-
    looking for when calibration extracts as its own plugin).
    """

    manifest = PluginManifest()

    def __init__(self) -> None:
        self._mcp_app: Any = None
        self._lifespan_cm: Any = None
        self._mounted: bool = False

    async def on_load(self, host: "Host") -> None:
        """Bind the host, mount /mcp, register lifespan hooks."""
        global _host
        _host = host

        app = host.fastapi_app()
        self._mcp_app = mcp.http_app(
            transport="streamable-http",
            path="/",
            stateless_http=True,
        )
        app.mount(self.manifest.mount_path, self._mcp_app)
        self._mounted = True

        # FastMCP's session manager owns its own lifespan; the parent
        # FastAPI app does not invoke sub-app lifespans automatically.
        host.on_startup(self._on_app_startup)
        host.on_shutdown(self._on_app_shutdown)

        logger.info(
            "parol6_mcp v%s mounted at %s (stateless streamable HTTP)",
            self.manifest.version, self.manifest.mount_path,
        )

    async def _on_app_startup(self) -> None:
        """Enter the FastMCP session manager's lifespan context."""
        if self._mcp_app is None:
            return
        # Starlette's router exposes lifespan_context for exactly this case.
        self._lifespan_cm = self._mcp_app.router.lifespan_context(self._mcp_app)
        await self._lifespan_cm.__aenter__()

    async def _on_app_shutdown(self) -> None:
        cm = self._lifespan_cm
        if cm is None:
            return
        await cm.__aexit__(None, None, None)
        self._lifespan_cm = None

    def register_panels(self, host: "Host", panels: Any) -> None:
        """Add a status panel to the host's "calibration" category.

        ``panels`` is the loader's ``PanelRegistry``. The builder runs
        inside a host-provided ``ui.column``, so it adds elements
        directly rather than wrapping its own container.
        """
        from nicegui import ui

        def _build() -> None:
            ui.label(self.manifest.display_name).classes("text-sm font-medium")
            ui.label(
                f"endpoint http://127.0.0.1:<wc-port>{self.manifest.mount_path}"
            ).classes("text-xs text-gray-500")
            ui.chip(
                "mounted" if self._mounted else "not mounted",
                color="green" if self._mounted else "grey",
            ).props("dense")
            ui.label(f"8 tools  ·  manifest v{self.manifest.version}").classes(
                "text-xs text-gray-500"
            )

        panels.add("calibration", self.manifest.display_name, _build)

    async def on_unload(self, host: "Host") -> None:
        """Tear down the lifespan and clear the host reference."""
        global _host
        await self._on_app_shutdown()
        _host = None
        self._mounted = False


__all__ = ["Plugin", "PluginManifest", "CollisionRejection", "mcp"]
