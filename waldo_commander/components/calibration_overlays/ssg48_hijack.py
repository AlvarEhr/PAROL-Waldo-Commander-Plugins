"""DEPRECATED — superseded by ``custom_tools.auto_migrate_ssg48_with_bracket``.

Tool-registry hijack — replaces SSG-48 body mesh with user's merged STL.

This module's :func:`hijack_ssg48_body_mesh` was the original integration
path for the calibration package. Commit ee8674e migrated the use case
into a custom-tool registration (see ``custom_tools.py`` +
``main.py:initialize_urdf_scene``), which is more flexible (supports
any tool, not just SSG-48) and survives upstream tool-registry edits
without colliding with the stock entry.

The hijack helper is kept here, behaviourally unchanged, for any
out-of-tree script that imports it directly. New integrations should
use ``custom_tools.register_one`` / ``custom_tools.register_all``
instead. The module is no longer imported from ``main.py`` and the
``hijack_ssg48_body_mesh`` name is no longer re-exported from the
package's ``__init__.py`` — it remains accessible only via direct
``from ...ssg48_hijack import hijack_ssg48_body_mesh``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from .constants import (
    _MERGED_STL_FIT_SCALE,
    _MERGED_STL_FIT_TRANSLATE_M,
    _MERGED_STL_RPY_RAD,
    _MERGED_STL_TRANSLATE_M,
)
from .state import _state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-registry hijack — replaces SSG-48 body mesh with user's merged STL
# ---------------------------------------------------------------------------


def _bake_merged_stl_to_parol6_mesh_dir() -> Path | None:
    """Apply the fit transform to the user's merged STL and write it into
    parol6's mesh dir as ``ssg48_body_realsense.stl``.

    Returns the path to the baked STL, or None if anything failed.
    """
    try:
        import trimesh  # noqa: PLC0415
        from importlib.resources import files as pkg_files  # noqa: PLC0415

        # Source: parol6-vision's merged STL.
        pv_root = (
            Path(__file__).resolve().parent.parent.parent.parent.parent / "parol6-vision"
        )
        src = pv_root / "parol6_vision" / "sim" / "meshes" / "ssg48_body_realsense.stl"
        if not src.exists():
            logger.info("merged STL not at %s; SSG-48 hijack skipped", src)
            return None

        parol6_root = Path(str(pkg_files("parol6")))
        mesh_dir = parol6_root / "urdf_model" / "meshes"
        if not mesh_dir.exists():
            logger.warning("parol6 mesh dir not found at %s", mesh_dir)
            return None

        dst = mesh_dir / "ssg48_body_realsense.stl"

        # ALWAYS rebake. Earlier versions skipped when dst.mtime >= src.mtime,
        # but that ignored edits to the placement constants below — when the
        # python constants change the source STL's mtime stays the same, so
        # the cached bake silently kept the OLD transform. Bake takes <1 s.
        mesh = trimesh.load(src, force="mesh")
        # Step 1: fit transform — scale + translate to align the merged mesh
        # with the flange. These are the calibrated values; do not edit.
        T_fit = np.eye(4, dtype=np.float64)
        T_fit[:3, :3] = np.eye(3) * _MERGED_STL_FIT_SCALE
        T_fit[:3, 3] = _MERGED_STL_FIT_TRANSLATE_M
        mesh.apply_transform(T_fit)

        # Step 2: user-tunable extra rotation + translation in flange frame.
        if any(abs(v) > 1e-9 for v in _MERGED_STL_TRANSLATE_M) or any(
            abs(v) > 1e-9 for v in _MERGED_STL_RPY_RAD
        ):
            T_extra = np.eye(4, dtype=np.float64)
            if any(abs(v) > 1e-9 for v in _MERGED_STL_RPY_RAD):
                T_extra[:3, :3] = SciRotation.from_euler(
                    "XYZ", _MERGED_STL_RPY_RAD
                ).as_matrix()
            T_extra[:3, 3] = _MERGED_STL_TRANSLATE_M
            mesh.apply_transform(T_extra)

        mesh.export(dst, file_type="stl")
        logger.info(
            "baked SSG-48 body STL to %s (%d tris, user translate=%s, rpy=%s)",
            dst, len(mesh.faces),
            _MERGED_STL_TRANSLATE_M, _MERGED_STL_RPY_RAD,
        )
        return dst

    except Exception as e:  # noqa: BLE001
        logger.warning("failed to bake SSG-48 body STL: %s", e)
        return None


def hijack_ssg48_body_mesh(active_robot: Any | None = None) -> bool:
    """Replace the SSG-48 tool's BODY mesh with the merged camera-bracket STL.

    Idempotent. Call this BEFORE ``Robot.__init__`` runs ``_build_tools()``
    if you want the running ``active_robot.tools`` collection to reflect the
    new mesh. If the robot is already constructed, pass ``active_robot`` and
    its ``_tools`` collection will be rebuilt from the (mutated) registry.

    Returns True if the swap succeeded, False if anything was missing.
    """
    if _state["ssg48_hijacked"] and active_robot is None:
        return True

    baked = _bake_merged_stl_to_parol6_mesh_dir()
    if baked is None:
        return False

    try:
        from parol6 import tools as parol6_tools  # noqa: PLC0415
        from waldoctl import MeshRole, MeshSpec  # noqa: PLC0415

        registry = parol6_tools.get_registry()
        if "SSG-48" not in registry:
            logger.warning("SSG-48 tool not in parol6 registry; nothing to hijack")
            return False

        # Cache-busting query string: ?v=<mtime>. The static-file server ignores
        # query strings, but Three.js' STLLoader treats different URLs as
        # different resources, so this guarantees the browser refetches whenever
        # the bake produces new bytes (which is on EVERY waldo-commander
        # restart since the bake-skip logic was removed).
        baked_versioned = f"{baked.name}?v={int(baked.stat().st_mtime)}"

        if not _state["ssg48_hijacked"]:
            cfg = registry["SSG-48"]

            def _swap_body(meshes: tuple) -> tuple:
                """Replace the BODY MeshSpec, keep JAW/SPINDLE entries intact."""
                out = []
                for m in meshes:
                    if m.role == MeshRole.BODY:
                        out.append(
                            MeshSpec(
                                file=baked_versioned,
                                origin=m.origin,
                                rpy=m.rpy,
                                role=MeshRole.BODY,
                            )
                        )
                    else:
                        out.append(m)
                return tuple(out)

            # Construct a new ToolConfig with replaced top-level meshes + variant meshes.
            # ToolConfig is a frozen dataclass; ``dataclasses.replace`` builds a copy.
            import dataclasses  # noqa: PLC0415

            new_variants = tuple(
                dataclasses.replace(v, meshes=_swap_body(v.meshes))
                for v in cfg.variants
            )
            new_cfg = dataclasses.replace(
                cfg,
                meshes=_swap_body(cfg.meshes),
                variants=new_variants,
            )
            parol6_tools.register_tool("SSG-48", new_cfg)
            _state["ssg48_hijacked"] = True
            logger.info(
                "SSG-48 body mesh hijacked -> %s (jaws + TCP transform unchanged)",
                baked.name,
            )

        # If the robot is already constructed, its ``_tools`` collection is a
        # stale snapshot — rebuild it from the now-mutated registry so
        # ``apply_tool("SSG-48")`` sees the new BODY mesh.
        if active_robot is not None:
            try:
                from parol6.robot import _build_tools  # noqa: PLC0415

                active_robot._tools = _build_tools()
                logger.info("active_robot._tools rebuilt from mutated registry")
            except Exception as e:  # noqa: BLE001
                logger.warning("could not rebuild active_robot._tools: %s", e)

        return True

    except Exception as e:  # noqa: BLE001
        logger.warning("SSG-48 hijack failed: %s", e)
        return False
