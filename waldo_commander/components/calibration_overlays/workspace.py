"""Workspace envelope (loads waldo-commander's cached convex hull STL)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .state import _state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workspace envelope (loads waldo-commander's cached convex hull STL)
# ---------------------------------------------------------------------------


def _ensure_workspace_envelope() -> bool:
    """Load + cache the workspace hull STL into a half-space representation
    (``Ax + b <= 0`` per face) for fast point-in-hull tests. Returns True
    when the envelope is available.
    """
    if _state["envelope_loaded"]:
        return _state["envelope_planes_A"] is not None

    _state["envelope_loaded"] = True  # don't re-attempt this turn
    try:
        from pathlib import Path  # noqa: PLC0415
        import trimesh  # noqa: PLC0415
        from scipy.spatial import ConvexHull  # noqa: PLC0415

        stl_path = Path.home() / ".waldo-commander" / "workspace_hull.stl"
        if not stl_path.exists():
            logger.info(
                "workspace envelope: %s not found — calibration reachability "
                "checks disabled. Open waldo-commander once to generate it.",
                stl_path,
            )
            return False
        mesh = trimesh.load(stl_path, force="mesh")
        hull = ConvexHull(np.asarray(mesh.vertices, dtype=np.float64))
        _state["envelope_planes_A"] = hull.equations[:, :-1].copy()
        _state["envelope_planes_b"] = hull.equations[:, -1].copy()
        _state["envelope_max_reach"] = float(
            np.linalg.norm(np.asarray(mesh.vertices), axis=1).max()
        )
        logger.info(
            "workspace envelope loaded: %d hull faces, max_reach=%.3f m",
            len(hull.simplices), _state["envelope_max_reach"],
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("workspace envelope load failed: %s", e)
        return False


def envelope_contains(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Vectorised point-in-hull test. ``points`` is (N, 3) or (3,); returns
    an (N,) bool array. All-True when the envelope isn't loaded.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    A = _state.get("envelope_planes_A")
    b = _state.get("envelope_planes_b")
    if A is None or b is None:
        return np.ones(len(pts), dtype=bool)
    return np.all(pts @ A.T + b <= 1e-9, axis=1)
