"""Structured rejection payload for gated motion tools.

The shape mirrors ``validate_joint_trajectory_core``'s return dict so the
LLM sees the same fields whether the gate fired in the WC GUI or via MCP.
``parol6_force_move_j`` exists as a separate tool, not a ``force=true``
flag — that pattern is auditable and the literal ``acknowledge_unsafe:
True`` schema prevents accidental bypass.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CollisionRejection(BaseModel):
    """Body of an ``isError: true`` tool result when the gate rejects a move.

    ``manager_ready=False`` means the collision manager (FCL + mesh load)
    was unavailable, so the move was dispatched WITHOUT verification. The
    LLM is told this explicitly via ``next_steps``.
    """

    model_config = ConfigDict(frozen=True)

    reason: str = Field(..., description="Why the move was rejected.")
    colliding_pair: tuple[str, str] | None = Field(
        default=None,
        description='Two object names that collide, e.g. ["L4", "TABLET"].',
    )
    colliding_q_deg: list[float] | None = Field(
        default=None,
        description="Joint config (degrees) at which the collision occurs.",
    )
    manager_ready: bool = Field(
        default=True,
        description="False = collision manager unavailable; check was skipped.",
    )
    next_steps: str = Field(
        default="",
        description="Suggested remediation for the LLM.",
    )


def format_rejection(result: dict[str, Any] | None) -> CollisionRejection:
    """Convert a host.motion detail dict into a structured rejection payload.

    Tolerant of the two natural shapes — ``validate_joint_trajectory_core``'s
    raw dict (``colliding_pair``, ``colliding_q``) and a normalised one
    (``colliding_q_deg``) — so this works whether plugin-arch surfaces the
    raw gate dict or a translated one.
    """
    result = result or {}
    pair_raw = result.get("colliding_pair")
    pair: tuple[str, str] | None = None
    if isinstance(pair_raw, (list, tuple)) and len(pair_raw) == 2:
        pair = (str(pair_raw[0]), str(pair_raw[1]))

    q_raw = result.get("colliding_q_deg") or result.get("colliding_q")
    q: list[float] | None = None
    if isinstance(q_raw, (list, tuple)):
        try:
            q = [float(v) for v in q_raw]
        except (TypeError, ValueError):
            q = None

    manager_ready = bool(result.get("manager_ready", True))
    reason = str(result.get("reason") or "collision")
    return CollisionRejection(
        reason=reason,
        colliding_pair=pair,
        colliding_q_deg=q,
        manager_ready=manager_ready,
        next_steps=_default_next_steps(manager_ready),
    )


def _default_next_steps(manager_ready: bool) -> str:
    """Boilerplate remediation text. Stays plaintext for cross-client rendering."""
    if not manager_ready:
        return (
            "Collision manager unavailable. The move dispatched without "
            "verification; confirm the robot environment manually before "
            "relying on this command."
        )
    return (
        "Adjust target joints away from the colliding object, run "
        "parol6_check_collision on intermediate configs to plan a route, or, "
        "after explicit user authorisation, call parol6_force_move_j with "
        "acknowledge_unsafe=true to bypass the gate."
    )


__all__ = ["CollisionRejection", "format_rejection"]
