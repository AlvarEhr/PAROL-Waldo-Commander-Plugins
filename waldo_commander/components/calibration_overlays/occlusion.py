"""Occlusion mesh + camera-line-of-sight checks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _build_occlusion_mesh() -> Any | None:
    """Load all PAROL6 LINK meshes (NOT the gripper) into a single combined
    trimesh for line-of-sight occlusion queries. The gripper is excluded
    because it sits between the camera and... well, IS where the camera is
    mounted, so it can't occlude the camera's own view of the scene.

    Verifies trimesh's ray-casting is functional before returning. Trimesh
    requires the ``rtree`` package for the BVH used by
    ``ray.intersects_location``; without rtree, every ray query raises
    ``ModuleNotFoundError`` and the occlusion filter silently no-ops.
    Returns None and logs a loud warning when rtree is missing — better
    to fail open with notice than fail silently.
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

    # Sanity-check: a single ray trace through the first mesh's bounding
    # volume. If this raises ModuleNotFoundError on rtree, the rest of
    # the occlusion filter would silently no-op for every candidate, so
    # bail loudly here instead.
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
    """Count how many rays from the camera to each board sample point pass
    through a robot link mesh.

    For each target point, we cast a ray from ``camera_pos_world`` toward
    that point and check intersection against each robot link mesh in its
    local frame (cheaper than transforming triangles into world frame).
    Per-link transforms + their inverses are computed ONCE per pose and
    reused across all target rays.

    Args:
        link_meshes: List of (link_name, trimesh.Trimesh) for the robot
            arm links. The gripper is intentionally NOT in this list — it
            holds the camera, so it can't occlude the camera's view of the
            scene.
        joint_angles_rad: Current joint configuration.
        camera_pos_world: (3,) world-frame camera optical centre.
        target_points_world: Iterable of (3,) world points to test
            line-of-sight against. Typical use: the board centre + 4 corners.
        early_termination_after: If set, return as soon as the blocked
            count strictly EXCEEDS this number (saves work when the caller
            only cares whether it's "too many").

    Returns:
        Count of rays blocked by any robot link. ``0`` means full
        line-of-sight to every target point.
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
    # Pre-invert each link transform once — saves a 4×4 inverse per ray.
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
                # Numerical / degenerate-ray cases — skip this link only.
                # ModuleNotFoundError (rtree missing) and other "this is
                # broken at the package level" errors are caught by the
                # one-shot sanity check in _build_occlusion_mesh; if we
                # reach here for one, it's a per-call quirk worth a debug
                # log but not silent skip.
                logger.debug(
                    "ray.intersects_location skipped for %s: %s: %s",
                    link_name, type(e).__name__, e,
                )
                continue
            if len(locations) == 0:
                continue
            # Distance from ray origin (camera) to first intersection in LOCAL frame
            # is the same as in world frame (rigid transforms preserve distances).
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
