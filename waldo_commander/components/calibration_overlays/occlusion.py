"""Occlusion mesh + camera-line-of-sight checks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _build_occlusion_mesh() -> Any | None:
    """Load PAROL6 link meshes for line-of-sight occlusion queries. Gripper
    is excluded — it carries the camera and can't occlude its own view.

    Returns None (with a loud warning) when trimesh's ray BVH is unavailable
    so the occlusion filter fails open with notice rather than silently.
    """
    try:
        import trimesh  # noqa: PLC0415
        from importlib.resources import files as pkg_files  # noqa: PLC0415
        parol6_root = Path(str(pkg_files("parol6")))
    except Exception as e:  # noqa: BLE001
        logger.info("occlusion check disabled: %s", e)
        return None
    mesh_dir = parol6_root / "urdf_model" / "meshes"
    meshes_with_link: list[tuple[str, Any]] = []
    for name in ("base_link", "L1", "L2", "L3", "L4", "L5", "L6"):
        path = mesh_dir / f"{name}_simplified.stl"
        if not path.exists():
            path = mesh_dir / f"{name}.STL"
        try:
            meshes_with_link.append((name, trimesh.load(path, force="mesh")))
        except Exception as e:  # noqa: BLE001
            logger.warning("occlusion mesh load failed for %s: %s", path, e)

    # One-shot ray trace to surface ``rtree`` missing before per-candidate
    # failures silently no-op the whole filter.
    if meshes_with_link:
        try:
            first_mesh = meshes_with_link[0][1]
            bb_centre = np.asarray(first_mesh.bounding_box.centroid)
            origin = (bb_centre + np.array([1.0, 0.0, 0.0])).reshape(1, 3)
            direction = np.array([[-1.0, 0.0, 0.0]])
            first_mesh.ray.intersects_location(
                ray_origins=origin, ray_directions=direction,
                multiple_hits=False,
            )
        except ModuleNotFoundError as e:
            logger.warning(
                "OCCLUSION CHECK DISABLED — trimesh ray-casting requires "
                "the 'rtree' package, which is not installed (%s). Install "
                "with: uv pip install rtree (or pip install rtree). The "
                "calibration pipeline will run without occlusion filtering "
                "and may accept poses where a robot link blocks the camera "
                "view of the board.", e,
            )
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "occlusion sanity-check raised %s: %s — disabling filter "
                "to avoid silent per-candidate failures.",
                type(e).__name__, e,
            )
            return None
    return meshes_with_link


def _camera_occlusion_count(
    link_meshes: list[tuple[str, Any]],
    joint_angles_rad: NDArray[np.float64],
    camera_pos_world: NDArray[np.float64],
    target_points_world: list[NDArray[np.float64]] | NDArray[np.float64],
    *,
    early_termination_after: int | None = None,
) -> int:
    """Count rays from the camera to each target point blocked by a link mesh.

    Ray tests run in each link's local frame (cheaper than transforming
    triangles). Per-link transforms and their inverses are computed once.

    Args:
        link_meshes: ``(link_name, trimesh.Trimesh)`` for arm links only.
        joint_angles_rad: Current joint configuration.
        camera_pos_world: (3,) world-frame camera optical centre.
        target_points_world: World points to test (e.g. board centre + corners).
        early_termination_after: Return as soon as blocked count exceeds this.

    Returns:
        Number of rays blocked. ``0`` means full line-of-sight.
    """
    from parol6_vision.sim.robot_kinematics import link_poses  # noqa: PLC0415
    poses = link_poses(np.asarray(joint_angles_rad, dtype=np.float64))
    link_transforms = {
        "base_link": poses.base_link,
        "L1": poses.l1,
        "L2": poses.l2,
        "L3": poses.l3,
        "L4": poses.l4,
        "L5": poses.l5,
        "L6": poses.l6_visual,
    }
    inv_transforms = {
        k: np.linalg.inv(v) for k, v in link_transforms.items()
    }

    blocked = 0
    safety_margin_m = 0.01  # ignore intersections within 1cm of target
    for target in target_points_world:
        target_arr = np.asarray(target, dtype=np.float64).reshape(3)
        direction = target_arr - camera_pos_world
        target_distance = float(np.linalg.norm(direction))
        if target_distance < 1e-6:
            continue
        direction_unit = direction / target_distance

        ray_blocked = False
        for link_name, mesh in link_meshes:
            T_base2link = inv_transforms[link_name]
            ray_origin_local = (T_base2link @ np.append(camera_pos_world, 1.0))[:3]
            ray_direction_local = T_base2link[:3, :3] @ direction_unit
            try:
                locations, _, _ = mesh.ray.intersects_location(
                    ray_origins=ray_origin_local.reshape(1, 3),
                    ray_directions=ray_direction_local.reshape(1, 3),
                    multiple_hits=False,
                )
            except (ValueError, RuntimeError, IndexError) as e:
                # Numerical / degenerate-ray case — skip this link only;
                # package-level failures are caught upstream.
                logger.debug(
                    "ray.intersects_location skipped for %s: %s: %s",
                    link_name, type(e).__name__, e,
                )
                continue
            if len(locations) == 0:
                continue
            # Local-frame distance equals world-frame distance (rigid transform).
            intersection_dist = float(np.linalg.norm(locations[0] - ray_origin_local))
            if intersection_dist < target_distance - safety_margin_m:
                ray_blocked = True
                break
        if ray_blocked:
            blocked += 1
            if (
                early_termination_after is not None
                and blocked > early_termination_after
            ):
                return blocked
    return blocked
