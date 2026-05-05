"""parol6-vision hand-eye calibration overlays for Waldo-Commander.

Two integration points:

1. ``hijack_ssg48_body_mesh()`` — at startup, replace the BODY mesh entry
   of the ``SSG-48`` tool in parol6's ``_TOOL_REGISTRY`` with the user's
   merged ssg48_body_realsense.stl. This is non-destructive: jaws, TCP
   transform, and motion descriptors stay the same, so gripper open/close
   simulation still works. The merged STL is pre-baked with the load-time
   fit transform applied (scale 0.1 + translate +13.955, -0.007, 0 m) into
   parol6's mesh dir as ``ssg48_body_realsense.stl`` so it sits beside the
   stock body file.

2. ``add_overlays()`` and ``add_control_panel()`` — at page-build time,
   add a ChArUco board (textured plane) at a known base-frame pose, a
   wireframe camera frustum parented to ``tcp_anchor``, and a small
   floating control panel with a "Run Calibration" button.

The calibration loop runs in a background daemon thread. A re-entry guard
refuses double-clicks on Run while a run is in-flight. Joint angles from
each move are pushed into the URDF scene via ``update_urdf_angles`` so the
user sees the robot animate live.

To remove this integration: delete the imports + calls in ``main.py``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from nicegui import app as ng_app, ui
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants matching parol6-vision's sim config
# ---------------------------------------------------------------------------


# =============================================================================
# === USER-FACING TUNABLES — EDIT THESE ====================================
# =============================================================================
#
# All edits take effect on next waldo-commander restart (no cache invalidation
# step needed; the STL bake re-runs unconditionally on every startup).
#
# Flange-frame convention: X = out toward bracket extension (+ side away from
# gripper centerline), Y = sideways across fingers, Z = down toward fingertip
# (Jepson's gripper hangs in -Z from the flange origin).

# Translation in the flange frame, in METRES, applied on top of the fit
# transform that aligns the merged STL with the flange. (0, 0, 0) = "use the
# fitted position as-is". Use small (cm-scale) values to nudge the bracket.
_MERGED_STL_TRANSLATE_M: tuple[float, float, float] = (-0.02788, 0.0, 0.0)
# Rotation in the flange frame as XYZ-extrinsic Euler angles, in RADIANS.
# Use np.deg2rad(angle_deg) if you prefer to think in degrees.
_MERGED_STL_RPY_RAD: tuple[float, float, float] = (0.0, 0.0, 0.0)


# =============================================================================
# === FIT TRANSFORM — DO NOT EDIT ===========================================
# =============================================================================
#
# These constants align the user's merged ssg48_body_realsense.stl with the
# flange origin. The user's CAD export was at ~10× Jepson's body STL scale and
# offset by approximately -139.55 m on the X axis from the flange origin.
# The values below were derived by the bbox-fit step in parol6-vision and pull
# the merged mesh to where Jepson's ssg48_body.stl sits.
#
# Editing these throws the gripper way off-screen. To shift the bracket
# visually, edit _MERGED_STL_TRANSLATE_M / _MERGED_STL_RPY_RAD above.
_MERGED_STL_FIT_SCALE: float = 0.1
_MERGED_STL_FIT_TRANSLATE_M: tuple[float, float, float] = (13.95538, -0.00737, 0.0)

# ChArUco board pose in the robot base frame.
#
# Edit these two tunables and restart waldo-commander to reposition / reorient
# the board. _BOARD_TRANSLATE_M is the position in metres (X, Y, Z in the robot
# base frame). _BOARD_RPY_RAD is the orientation as XYZ-extrinsic Euler angles
# in radians — same convention as scipy.spatial.transform.Rotation.from_euler("XYZ", ...).
#
# Examples:
#   90° rotation about base Z (lay board "sideways"): _BOARD_RPY_RAD = (0.0, 0.0, np.pi/2)
#   Tilt board 30° upward toward the robot: _BOARD_RPY_RAD = (np.deg2rad(30), 0.0, 0.0)
#   Mount on a vertical wall facing -X: _BOARD_RPY_RAD = (0.0, np.pi/2, 0.0)
#   Along X: (0.25, -0.1, 0.0), (0.0, 0.0, np.deg2rad(90))
#   Diagnoal: (0, 0.15, 0.0), (0.0, 0.0, np.deg2rad(-45))
_BOARD_TRANSLATE_M: tuple[float, float, float] = (0.25, -0.1, 0.0)
_BOARD_RPY_RAD: tuple[float, float, float] = (0.0, 0.0, np.deg2rad(90))

# Hemisphere search region — single source of truth for BOTH the visual
# wireframe AND the orchestrator's HemisphereParams. The volume defined here
# is where the pose generator may place camera candidates around the board.
#
# Distance: radial distance from the board center.
# Elevation: angle above the horizontal plane through the board (0° = level
#            with the board, 90° = directly overhead).
# Azimuth range: half-angle SPREAD around the natural facing direction
#            (auto-computed: from the robot base origin toward the board
#            center, projected onto the floor). e.g. spread=60 means the
#            hemisphere covers ±60° around that base→board ray.
# Set _SHOW_HEMISPHERE_WIREFRAME=False to hide the visualisation.
_SHOW_HEMISPHERE_WIREFRAME: bool = True
_HEMI_DISTANCE_RANGE_M: tuple[float, float] = (0.22, 0.35)  # (min, max) radial distance
# 25° → 90°. ev_min=25° avoids heavily-foreshortened low-angle poses where
# the ChArUco detector struggles to find ≥6 corners; ev_max=90° closes the
# dome at the top. Earlier we tried (10, 90) but the orchestrator generated
# 24 candidates and got 0 useful samples — too many extreme poses where
# board detection failed at runtime.
_HEMI_ELEVATION_RANGE_DEG: tuple[float, float] = (25.0, 90.0)
# Distance lower bound 0.22 m: at fx=fy=615 the camera covers ~230 mm of
# horizontal scene at this distance, just enough for the 210 mm-wide board
# to fit with safety margin. Closer than this and the board falls outside
# the FOV → no detection.
# 150° spread = 300° azimuth coverage. Pushes the hemisphere toward
# wrapping the board entirely (full 360° = spread 180°), giving the
# orchestrator more "from-behind-the-board" pose options. The
# back-of-hemisphere (between board and robot base) tends to be unreachable
# anyway — the arm has to fold over itself — but the workspace-hull filter
# silently rejects those, so widening here is safe.
_HEMI_AZIMUTH_SPREAD_DEG: float = 150.0

# Reachability sampling: at scene-build time we densely sample the hemisphere
# volume, run IK + workspace-hull check on each candidate, and render the
# actually-reachable subset as small green spheres. Lets you see at a glance
# how much of the theoretical hemisphere is usable BEFORE running calibration.
# Turn off if it makes startup feel slow (the IK loop is the main cost ~1-3 s
# after numba warmup; first run can be slower while numba JITs).
# Self-collision rejection: builds a trimesh.CollisionManager from PAROL6's
# simplified link meshes + the merged SSG-48 body (with camera bracket fused in)
# and rejects pose-generator candidates whose joint config would have the
# gripper/bracket clipping into the robot's own arm. Adjacent links (which
# always touch at joints) are whitelisted. ~12 ms per candidate on real
# hemisphere poses, ~0.4 s total for a 30-candidate pose-generator pass.
# Set False to disable (e.g. if python-fcl misbehaves).
_ENABLE_SELF_COLLISION_CHECK: bool = True

_SHOW_REACHABILITY_POINTS: bool = True
# (n_distances, n_elevations, n_azimuths). With tilt_x=180 the wrist must
# flip to satisfy the look-at constraint, so only ~4% of grid samples are
# reachable. We compensate with a denser grid (~840 samples → ~33 reachable)
# and the farthest-first thinning below picks the best spread from those.
_REACHABILITY_GRID = (7, 6, 20)
# Greedy farthest-first selection: thin the dense reachable set down to a
# spatially well-spread subset, so visualisation (and, optionally,
# calibration) gets points that are far enough apart instead of clustered
# along grid lines. None = no thinning, show every reachable point.
_REACHABILITY_KEEP_COUNT: int | None = 30


def _hemi_azimuth_center_deg() -> float:
    """Direction from the robot base origin to the board center, in degrees
    measured from world +X (CCW). Used so the hemisphere naturally faces
    "away from the robot" toward where the board sits."""
    cx, cy, _ = _board_center_world().tolist()
    return float(np.degrees(np.arctan2(cy, cx)))


def _hemi_azimuth_world_range_deg() -> tuple[float, float]:
    """World-frame azimuth range covering the hemisphere's spread, centered
    on the base→board direction. Used by both the viz and the orchestrator."""
    center = _hemi_azimuth_center_deg()
    return (center - _HEMI_AZIMUTH_SPREAD_DEG, center + _HEMI_AZIMUTH_SPREAD_DEG)


def _build_T_board2base() -> NDArray[np.float64]:
    """Compose the SE(3) board→base matrix from the position + RPY tunables."""
    T = np.eye(4, dtype=np.float64)
    if any(abs(a) > 1e-9 for a in _BOARD_RPY_RAD):
        T[:3, :3] = SciRotation.from_euler("XYZ", _BOARD_RPY_RAD).as_matrix()
    T[:3, 3] = _BOARD_TRANSLATE_M
    return T


_T_BOARD2BASE: NDArray[np.float64] = _build_T_board2base()


def _board_center_world() -> NDArray[np.float64]:
    """Compute the board's geometric center in world coordinates.

    The ChArUco config places the board's local origin at one corner and the
    board extends to (w_m, h_m, 0) in board-local coordinates. The hemisphere
    + pose-generator targets need the board CENTER, not the corner — using
    the corner makes the pose generator aim its candidates at the corner
    rather than at the middle of the board, which is visible in the GUI as a
    hemisphere wireframe that's offset to one side of the visible board.
    """
    from parol6_vision.calibration.board import BOARD_TABLET_30MM as _cfg  # noqa: PLC0415
    w_m = _cfg.squares_x * _cfg.square_length
    h_m = _cfg.squares_y * _cfg.square_length
    center_local = np.array([w_m / 2.0, h_m / 2.0, 0.0, 1.0], dtype=np.float64)
    return (_T_BOARD2BASE @ center_local)[:3]

# Camera intrinsics for the frustum size (matches sim).
_INTR_FX, _INTR_FY = 615.0, 615.0
_INTR_CX, _INTR_CY = 320.0, 240.0
_INTR_W, _INTR_H = 640, 480

# Camera mount in the flange frame — defines the cone APEX position and
# (via tilt) its optical-axis orientation. Edit these to move/orient the
# camera frustum. Used for BOTH the visual cone AND the calibration cold
# start AND the simulated ground truth (the sim's ground truth is set to
# the cold start plus a small fixed perturbation so the calibration has
# something to converge to).
#
# Translation in MILLIMETRES, in the flange frame.
#   X: out toward bracket extension
#   Y: sideways across fingers
#   Z: along flange axis (positive = toward base, negative = toward fingertip)
_CAM_MOUNT_TRANSLATE_MM: tuple[float, float, float] = (-52.0, 0, -50.0)
# Tilt in DEGREES, XYZ-extrinsic Euler.
#   tilt_x = 180°: encodes that the user's physical camera lens points in
#                  the flange's -Z direction (toward the gripper fingertips
#                  / "down" relative to the bracket), not flange +Z. Without
#                  this, look_at_pose would orient flange +Z at the target
#                  and the physical camera (looking the OTHER way) would
#                  point AWAY from the board during calibration. PAROL6 has
#                  to wrist-flip to satisfy this constraint, so reachable
#                  pose count drops from ~16% to ~4% — still plenty given
#                  the dense grid.
#   tilt_z = 90°: rolls the camera body 90° about its optical axis so the
#                  image-row direction aligns with the robot's side-to-side
#                  axis (matches the physical camera body orientation).
_CAM_MOUNT_TILT_DEG: tuple[float, float, float] = (180.0, 0.0, 90.0)

# How far in front of the camera to draw the frustum far plane, in CAMERA
# optical-axis units. With tilt_x=180 in the mount, camera +Z = flange -Z,
# so a POSITIVE depth here makes the cone extend along flange -Z = the
# physical lens direction (toward the gripper fingertips at home, toward
# the board during calibration). The cone, the yellow centerline, and the
# physical camera lens should all point the same way as a result.
_FRUSTUM_DEPTH_M: float = 0.20

# "Laser pointer" extension — a second, fainter frustum drawn from the same
# apex with the same FOV but extending much further. Useful for seeing where
# the camera's view actually lands on the floor / board during calibration
# without redrawing dynamically as the robot moves.
#
# Same sign convention as _FRUSTUM_DEPTH_M (must match for the long cone to
# extend in the same direction as the short one). Set to None to disable.
_FRUSTUM_FAR_DEPTH_M: float | None = 0.50


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


_state: dict[str, Any] = {
    "merged_stl_path": None,  # set on first add_overlays call
    "board_png_path": None,
    "frustum_objects": [],
    "is_running": False,
    "result_label": None,
    "status_label": None,
    "current_mount": None,  # CameraMount, set by orchestrator
    "calibrated_mount": None,  # set by calibration thread on success
    "main_loop": None,  # asyncio loop captured at setup
    "ssg48_hijacked": False,
    # Cached workspace envelope (loaded once from waldo-commander's hull STL).
    "envelope_planes_A": None,  # (n_faces, 3) face-normal A coefficients
    "envelope_planes_b": None,  # (n_faces,) face-offset constants
    "envelope_max_reach": None, # scalar bounding sphere radius
    "envelope_loaded": False,
}


# ---------------------------------------------------------------------------
# Workspace envelope (loads waldo-commander's cached convex hull STL)
# ---------------------------------------------------------------------------


def _ensure_workspace_envelope() -> bool:
    """Load and cache waldo-commander's workspace hull STL.

    The hull is a scipy.spatial.ConvexHull computed by waldo-commander from
    sampled FK over joint limits, then exported to STL at
    ``~/.waldo-commander/workspace_hull.stl``. We reload the half-space
    representation (``Ax + b <= 0`` for each face) so we can do fast
    point-in-hull tests for calibration candidates.

    Returns True if the envelope is available, False if anything went wrong.
    Logs once on success / failure.
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


def _build_collision_manager() -> tuple[Any, set[tuple[str, str]]] | None:
    """Build a trimesh CollisionManager populated with PAROL6's link meshes
    + the merged SSG-48 gripper body (which has the camera bracket fused in).

    Returns (manager, adjacent_pairs) on success, None if python-fcl or any
    of the meshes is missing. The adjacent_pairs set lists (link_a, link_b)
    pairs that are EXPECTED to touch (e.g. base_link↔L1 at the joint) and
    should NOT be flagged as self-collisions.
    """
    try:
        import trimesh  # noqa: PLC0415
        import fcl  # noqa: PLC0415, F401
        from importlib.resources import files as pkg_files  # noqa: PLC0415
    except ImportError as e:
        logger.info("self-collision check disabled: %s", e)
        return None
    try:
        parol6_root = Path(str(pkg_files("parol6")))
    except Exception as e:  # noqa: BLE001
        logger.warning("could not locate parol6 package: %s", e)
        return None
    mesh_dir = parol6_root / "urdf_model" / "meshes"

    mgr = trimesh.collision.CollisionManager()
    link_names = ["base_link", "L1", "L2", "L3", "L4", "L5", "L6"]
    for name in link_names:
        path = mesh_dir / f"{name}_simplified.stl"
        if not path.exists():
            path = mesh_dir / f"{name}.STL"
        try:
            mesh = trimesh.load(path, force="mesh")
            mgr.add_object(name, mesh, transform=np.eye(4))
        except Exception as e:  # noqa: BLE001
            logger.warning("collision mesh load failed for %s: %s", path, e)
            return None
    # Merged gripper body (camera bracket fused in via the hijack).
    grip_path = mesh_dir / "ssg48_body_realsense.stl"
    if grip_path.exists():
        try:
            mgr.add_object("gripper", trimesh.load(grip_path, force="mesh"), transform=np.eye(4))
        except Exception as e:  # noqa: BLE001
            logger.warning("collision mesh load failed for gripper: %s", e)

    adjacent: set[tuple[str, str]] = {
        ("base_link", "L1"), ("L1", "L2"), ("L2", "L3"),
        ("L3", "L4"), ("L4", "L5"), ("L5", "L6"),
        ("L6", "gripper"),
    }
    # Add reverse pairs for symmetric lookup.
    adjacent |= {(b, a) for a, b in adjacent}
    logger.info("self-collision manager loaded: 7 links + gripper")
    return mgr, adjacent


def _self_collides(
    manager: Any,
    adjacent: set[tuple[str, str]],
    joint_angles_rad: NDArray[np.float64],
) -> bool:
    """Returns True if the robot at this joint config has any non-adjacent
    self-collision (i.e. the gripper / camera bracket clips into its own arm)."""
    from parol6_vision.sim.robot_kinematics import link_poses  # noqa: PLC0415
    poses = link_poses(np.asarray(joint_angles_rad, dtype=np.float64))
    manager.set_transform("base_link", poses.base_link)
    manager.set_transform("L1", poses.l1)
    manager.set_transform("L2", poses.l2)
    manager.set_transform("L3", poses.l3)
    manager.set_transform("L4", poses.l4)
    manager.set_transform("L5", poses.l5)
    manager.set_transform("L6", poses.l6_visual)
    # Gripper rides on the flange (its mesh is in flange-frame coordinates).
    manager.set_transform("gripper", poses.l6_visual)
    in_coll, names = manager.in_collision_internal(return_names=True)
    if not in_coll:
        return False
    # Filter out adjacent-pair collisions (joints always touch their neighbours).
    return any(p not in adjacent for p in names)


def _trajectory_collides(
    manager: Any,
    adjacent: set[tuple[str, str]],
    q_from: NDArray[np.float64],
    q_to: NDArray[np.float64],
    n_samples: int = 10,
) -> bool:
    """Check whether straight-line joint-space interpolation from ``q_from``
    to ``q_to`` ever passes through a self-collision state. Real robot moves
    are joint-space linear-interpolated, so this approximates the actual
    trajectory the controller will follow.

    We sample 10 intermediate configs (skip endpoints — they're checked by
    the regular self-collision filter elsewhere). If any sample collides,
    the move would clip mid-trajectory.
    """
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    # Skip t=0 and t=1 (endpoints handled by single-pose collision check).
    for t in np.linspace(0.0, 1.0, n_samples + 2)[1:-1]:
        q = q_from + t * (q_to - q_from)
        if _self_collides(manager, adjacent, q):
            return True
    return False


def _build_occlusion_mesh() -> Any | None:
    """Load all PAROL6 LINK meshes (NOT the gripper) into a single combined
    trimesh for line-of-sight occlusion queries. The gripper is excluded
    because it sits between the camera and... well, IS where the camera is
    mounted, so it can't occlude the camera's own view of the scene.
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
    return meshes_with_link


def _camera_occluded(
    link_meshes: list[tuple[str, Any]],
    joint_angles_rad: NDArray[np.float64],
    camera_pos_world: NDArray[np.float64],
    target_world: NDArray[np.float64],
) -> bool:
    """Cast a ray from camera position to target (board centre); return True
    if any robot link mesh blocks the line of sight before the target.

    Computes per-link FK to get each link's world-frame transform, then
    transforms the ray into each link's local frame for the intersect query
    (cheaper than transforming every triangle of every mesh into world frame).
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
    direction = target_world - camera_pos_world
    target_distance = float(np.linalg.norm(direction))
    if target_distance < 1e-6:
        return False
    direction_unit = direction / target_distance

    safety_margin_m = 0.01  # ignore intersections within 1cm of target
    for link_name, mesh in link_meshes:
        T_link2base = link_transforms[link_name]
        # Inverse-transform the ray into the link's local frame.
        T_base2link = np.linalg.inv(T_link2base)
        ray_origin_local = (T_base2link @ np.append(camera_pos_world, 1.0))[:3]
        ray_direction_local = T_base2link[:3, :3] @ direction_unit
        try:
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=ray_origin_local.reshape(1, 3),
                ray_directions=ray_direction_local.reshape(1, 3),
                multiple_hits=False,
            )
        except Exception:  # noqa: BLE001
            continue
        if len(locations) == 0:
            continue
        # Distance from ray origin (camera) to first intersection in LOCAL frame
        # is the same as in world frame (rigid transforms preserve distances).
        intersection_dist = float(np.linalg.norm(locations[0] - ray_origin_local))
        if intersection_dist < target_distance - safety_margin_m:
            return True
    return False


def envelope_contains(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Vectorised point-in-hull test against waldo-commander's envelope.

    Args:
        points: (N, 3) world-frame positions, or a single (3,).

    Returns:
        (N,) bool array, True = inside the workspace hull. Returns all-True
        if the envelope hasn't been loaded (e.g. no waldo-commander cache yet).
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    A = _state.get("envelope_planes_A")
    b = _state.get("envelope_planes_b")
    if A is None or b is None:
        return np.ones(len(pts), dtype=bool)
    # A point is inside iff it satisfies every face's half-space inequality.
    return np.all(pts @ A.T + b <= 1e-9, axis=1)


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
            Path(__file__).resolve().parent.parent.parent.parent / "parol6-vision"
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


# ---------------------------------------------------------------------------
# Scene-overlay builders
# ---------------------------------------------------------------------------


def _ensure_static_mounts(merged_stl_path: Path, board_png_path: Path) -> tuple[str, str]:
    """Mount the merged STL directory + board PNG cache. Returns (stl_url, png_url)."""
    if _state.get("static_mounted"):
        return _state["stl_url"], _state["png_url"]

    ng_app.add_static_files("/calib_meshes", str(merged_stl_path.parent))
    ng_app.add_static_files("/calib_cache", str(board_png_path.parent))

    stl_url = f"/calib_meshes/{merged_stl_path.name}"
    png_url = f"/calib_cache/{board_png_path.name}"
    _state["static_mounted"] = True
    _state["stl_url"] = stl_url
    _state["png_url"] = png_url
    return stl_url, png_url


def _frustum_corners_local(
    depth_m: float,
) -> list[tuple[float, float, float]]:
    """5 corner points of the frustum in the camera's optical frame."""
    fx, fy = _INTR_FX, _INTR_FY
    cx, cy = _INTR_CX, _INTR_CY
    w, h = _INTR_W, _INTR_H
    pts: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for px, py in [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]:
        x = (px - cx) * depth_m / fx
        y = (py - cy) * depth_m / fy
        pts.append((float(x), float(y), float(depth_m)))
    return pts


def _populate_frustum(scene_group: Any, T_cam2flange: NDArray[np.float64]) -> list[Any]:
    """Add frustum lines inside ``scene_group`` (parented to tcp_anchor).

    Two cones are drawn:

    * The **near frustum** at depth ``_FRUSTUM_DEPTH_M`` — bright red, with
      the full apex-to-corner + far-plane-rectangle line set. This is the
      "where the camera is looking right now" indicator.
    * Optionally a **far frustum** at depth ``_FRUSTUM_FAR_DEPTH_M`` (set
      to ``None`` to disable) — drawn fainter, no apex-to-corner lines, just
      the far rectangle and four edge-extension lines from the near corners
      to the far corners, giving a "laser pointer" projection so you can see
      where the FOV lands on the floor / board without dynamic recomputation.

    Frustum corners are transformed by ``T_cam2flange`` so the apex sits at
    the camera optical centre relative to the flange.
    """
    R = T_cam2flange[:3, :3]
    t = T_cam2flange[:3, 3]

    def to_flange(corners: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        return [tuple((R @ np.asarray(c) + t).tolist()) for c in corners]

    # Load workspace envelope (lazy, cached). Used downstream for diagnostic
    # reachability checks of calibration candidates.
    _ensure_workspace_envelope()

    near = to_flange(_frustum_corners_local(_FRUSTUM_DEPTH_M))

    # Optical-axis center line: from camera apex straight along the optical
    # axis to the far-plane center. Lets the user see at a glance what's at
    # the center of the camera image, which is harder to read off four
    # corner-edges alone.
    apex_local = (0.0, 0.0, 0.0)
    far_center_local = (0.0, 0.0, _FRUSTUM_DEPTH_M)
    R = T_cam2flange[:3, :3]
    t = T_cam2flange[:3, 3]
    apex_flange = tuple((R @ np.asarray(apex_local) + t).tolist())
    far_flange = tuple((R @ np.asarray(far_center_local) + t).tolist())
    if _FRUSTUM_FAR_DEPTH_M is not None:
        # Extend the centerline through the far cone too.
        far_long_local = (0.0, 0.0, _FRUSTUM_FAR_DEPTH_M)
        far_long_flange = tuple((R @ np.asarray(far_long_local) + t).tolist())

    # Diagnostic: log the actual far-plane span in flange frame so we can
    # confirm the tilt rotation is taking effect at the geometry level.
    far_corners = near[1:]
    fx_span = max(c[0] for c in far_corners) - min(c[0] for c in far_corners)
    fy_span = max(c[1] for c in far_corners) - min(c[1] for c in far_corners)
    logger.info(
        "frustum near-plane span (flange frame): X=%.1f mm, Y=%.1f mm "
        "(tilt_deg=%s, depth=%.2f m)",
        fx_span * 1000, fy_span * 1000,
        _CAM_MOUNT_TILT_DEG, _FRUSTUM_DEPTH_M,
    )

    objects: list[Any] = []
    with scene_group:
        # Centerline from apex along the optical axis.
        cline_end = far_long_flange if _FRUSTUM_FAR_DEPTH_M is not None else far_flange
        objects.append(
            ui.scene.line(list(apex_flange), list(cline_end)).material("#ffff00")
        )

        # --- Near cone: bright apex-to-corner + far-plane rectangle. ---
        for i in range(1, 5):
            objects.append(
                ui.scene.line(list(near[0]), list(near[i])).material("#ff8080")
            )
        for i in range(1, 5):
            j = 1 + (i % 4)
            objects.append(
                ui.scene.line(list(near[i]), list(near[j])).material("#ff5050")
            )

        # --- Far "laser pointer" cone, fainter, edge extensions only. ---
        if _FRUSTUM_FAR_DEPTH_M is not None:
            far = to_flange(_frustum_corners_local(_FRUSTUM_FAR_DEPTH_M))
            # Extension lines: each near far-plane corner -> matching far corner.
            for i in range(1, 5):
                objects.append(
                    ui.scene.line(list(near[i]), list(far[i])).material("#ff8080")
                )
            # Far-plane rectangle (the "footprint" projected onto your scene).
            for i in range(1, 5):
                j = 1 + (i % 4)
                objects.append(
                    ui.scene.line(list(far[i]), list(far[j])).material("#ffaaaa")
                )
    return objects


def add_overlays(urdf_scene: Any) -> None:
    """Bolt the merged STL + board + frustum onto a running ``UrdfScene``.

    Call this AFTER ``urdf_scene.show()`` has built the scene — typically right
    after the world-axes lines are drawn in ``main.build_page_content``.
    """
    if urdf_scene is None or urdf_scene.scene is None:
        logger.warning("add_overlays called before UrdfScene is ready; skipping")
        return
    if urdf_scene.tcp_anchor is None:
        logger.warning("add_overlays: UrdfScene has no tcp_anchor; gripper bracket won't follow flange")

    # Lazy import — keep parol6-vision out of the main load path so
    # Waldo-Commander still imports cleanly without it.
    try:
        from parol6_vision.calibration.board import (  # noqa: PLC0415
            BOARD_TABLET_30MM,
            render_board_png,
        )
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
    except ImportError as e:
        logger.error("parol6-vision not importable; skipping calibration overlays: %s", e)
        return

    import cv2  # noqa: PLC0415

    # Resolve paths.
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent / "parol6-vision"
    merged_stl = pkg_root / "parol6_vision" / "sim" / "meshes" / "ssg48_body_realsense.stl"
    if not merged_stl.exists():
        logger.warning("merged STL not found at %s; skipping calibration overlays", merged_stl)
        return

    # Render board texture and cache it. We re-render every restart so the
    # PNG always reflects the current BoardConfig + render settings without
    # needing manual cache invalidation. Render takes ~50 ms.
    cache_dir = Path(__file__).resolve().parent.parent / "_calib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    board_png = cache_dir / "board.png"
    canonical = render_board_png(BOARD_TABLET_30MM, pixels_per_metre=4000.0, margin_squares=0.0)
    # No vertical flip: NiceGUI's Three.js Texture material uses flipY=false
    # AND our texture-coord array maps world (0,0,0) -> UV (0,0) -> the PNG's
    # top-left pixel. cv2.flip(...,0) was double-flipping that and producing
    # a mirrored / scrambled-looking board where the markers were unreadable.
    canonical_rgb = cv2.cvtColor(canonical, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(board_png), canonical_rgb)
    logger.info("rendered ChArUco board PNG: %s (%d×%d)", board_png, canonical.shape[1], canonical.shape[0])

    stl_url, png_url = _ensure_static_mounts(merged_stl, board_png)

    # Capture the asyncio loop so the worker thread can post UI updates.
    try:
        _state["main_loop"] = asyncio.get_running_loop()
    except RuntimeError:
        _state["main_loop"] = None

    # Cache mount + paths for the worker thread.
    _state["merged_stl_path"] = merged_stl
    _state["board_png_path"] = board_png
    _state["current_mount"] = CameraMount.from_eyeball_estimate(
        x_mm=_CAM_MOUNT_TRANSLATE_MM[0],
        y_mm=_CAM_MOUNT_TRANSLATE_MM[1],
        z_mm=_CAM_MOUNT_TRANSLATE_MM[2],
        tilt_x_deg=_CAM_MOUNT_TILT_DEG[0],
        tilt_y_deg=_CAM_MOUNT_TILT_DEG[1],
        tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
    )

    # ------------------------------------------------------------------
    # Camera frustum group, parented to tcp_anchor. The merged STL with
    # the camera bracket is mounted automatically because we hijacked the
    # SSG-48 BODY mesh in the parol6 tool registry at startup.
    # ------------------------------------------------------------------
    if urdf_scene.tcp_anchor is not None:
        with urdf_scene.tcp_anchor:
            frustum_group = ui.scene.group().with_name("calib:frustum")
            _state["frustum_group"] = frustum_group
            _state["frustum_objects"] = _populate_frustum(
                frustum_group, _state["current_mount"].T_cam2flange
            )

    # ------------------------------------------------------------------
    # ChArUco board → world frame (parented to scene root via instance method).
    # Raised 1mm above the floor so the polar-grid ground plane doesn't
    # z-fight with it, and bordered with a coloured outline so it's easy
    # to spot even if the texture fails to load.
    # ------------------------------------------------------------------
    cfg = BOARD_TABLET_30MM
    w_m = cfg.squares_x * cfg.square_length
    h_m = cfg.squares_y * cfg.square_length
    scene_root = urdf_scene.scene

    board_pos = _T_BOARD2BASE[:3, 3].copy()
    board_pos[2] += 0.001  # nudge above the floor grid to avoid z-fight
    board_group = scene_root.group().move(*board_pos.tolist())
    rpy = SciRotation.from_matrix(_T_BOARD2BASE[:3, :3]).as_euler("XYZ").tolist()
    if any(abs(a) > 1e-6 for a in rpy):
        board_group = board_group.rotate(*rpy)
    with board_group:
        # Backing plane — gray (not white) so it's visually distinct from the
        # texture's white squares. If the texture renders, you see distinct
        # black-on-white markers ON TOP of a gray border / fallback. If the
        # texture fails to load, the area shows as solid gray instead of
        # white, which is an obvious diagnostic signal.
        # Sits at local z ∈ [-0.0035, -0.0025], well below the texture, so
        # there's no chance of z-fighting against the texture or the floor.
        ui.scene.box(w_m, h_m, 0.001).move(w_m / 2, h_m / 2, -0.003).material(
            "#888888", opacity=1.0
        )
        # ChArUco texture. Vertex rows are REVERSED (first row Y=h_m, second
        # row Y=0) so the resulting triangle winding produces a +Z-facing
        # surface normal. NiceGUI's Three.js material is MeshLambertMaterial
        # with side=DoubleSide and transparent=true; in this combo the back
        # face can render dim or blank, so we want the FRONT face to be the
        # one users see when looking down at the board from above.
        # UV mapping with this ordering: PNG top-left → world (0, h_m), so
        # the printed PNG's "top" appears at the FAR edge of the board (Y=h_m)
        # and its "bottom" appears at the NEAR edge (Y=0) — i.e. the board
        # reads correctly when viewed from the robot side looking in +Y.
        ui.scene.texture(
            png_url,
            [
                [(0, h_m, 0.0005), (w_m, h_m, 0.0005)],
                [(0, 0,    0.0005), (w_m, 0,    0.0005)],
            ],
        )
        # Bright orange outline above the texture, easy to spot at any zoom.
        ui.scene.line([0, 0, 0.0010], [w_m, 0, 0.0010]).material("#ff8800")
        ui.scene.line([w_m, 0, 0.0010], [w_m, h_m, 0.0010]).material("#ff8800")
        ui.scene.line([w_m, h_m, 0.0010], [0, h_m, 0.0010]).material("#ff8800")
        ui.scene.line([0, h_m, 0.0010], [0, 0, 0.0010]).material("#ff8800")

    logger.info("parol6-vision calibration overlays added (board at %s)", board_pos.tolist())

    # ------------------------------------------------------------------
    # Hemisphere wireframe — shows the (distance × elevation × azimuth)
    # region around the board where the pose generator places camera
    # candidates. Drawn in WORLD frame because hemisphere_camera_position()
    # uses world Z-up regardless of board rotation. Centered on the board
    # CENTRE (not the corner-anchored origin in _T_BOARD2BASE).
    # ------------------------------------------------------------------
    if _SHOW_HEMISPHERE_WIREFRAME:
        _add_hemisphere_wireframe(scene_root, _board_center_world())


def _add_hemisphere_wireframe(scene_root: Any, target_world: NDArray[np.float64]) -> None:
    """Draw a 3D wireframe volume for the hemisphere search region.

    Volume bounds are an annular spherical sector defined by:
        d ∈ [d_min, d_max]                      (radial)
        elev ∈ [elev_min, elev_max]             (latitude)
        az ∈ [az_center ± _HEMI_AZIMUTH_SPREAD]  (longitude)

    We render it as a 6-surface wireframe wedge so the user perceives a
    solid region (not just discrete sampling paths):
        - inner spherical patch at d_min  (latitude × longitude grid)
        - outer spherical patch at d_max  (same grid)
        - 4 corner edges connecting the patches at the bounding corners

    This matches what the pose generator actually does: it samples ANY pose
    within this volume that's also reachable + IK-valid, not just along a
    few discrete arcs.
    """
    d_min, d_max = _HEMI_DISTANCE_RANGE_M
    ev_min, ev_max = _HEMI_ELEVATION_RANGE_DEG
    az_min, az_max = _hemi_azimuth_world_range_deg()

    # Grid resolution for the surface meshing.
    n_az_segments = 8     # → 9 longitude lines per shell
    n_ev_segments = 4     # → 5 latitude lines per shell
    azimuths = np.linspace(az_min, az_max, n_az_segments + 1)
    elevations = np.linspace(ev_min, ev_max, n_ev_segments + 1)

    grp = scene_root.group().move(*target_world.tolist()).with_name("calib:hemisphere")

    def offset(d: float, elev_deg: float, az_deg: float) -> tuple[float, float, float]:
        elev = np.radians(elev_deg)
        az = np.radians(az_deg)
        return (
            float(d * np.cos(elev) * np.cos(az)),
            float(d * np.cos(elev) * np.sin(az)),
            float(d * np.sin(elev)),
        )

    with grp:
        # Two spherical shells (inner d_min, outer d_max). For each shell:
        #   - latitude arcs: constant elev, varying az
        #   - longitude arcs: constant az, varying elev
        for d, opacity in [(d_min, 0.55), (d_max, 0.30)]:
            color = "#5599ff" if d == d_min else "#3366cc"
            # Latitude arcs (one per elevation step).
            for ev in elevations:
                pts = [offset(d, ev, az) for az in azimuths]
                for i in range(len(pts) - 1):
                    ui.scene.line(list(pts[i]), list(pts[i + 1])).material(
                        color, opacity=opacity
                    )
            # Longitude arcs (one per azimuth step).
            for az in azimuths:
                pts = [offset(d, ev, az) for ev in elevations]
                for i in range(len(pts) - 1):
                    ui.scene.line(list(pts[i]), list(pts[i + 1])).material(
                        color, opacity=opacity
                    )

        # Four corner edges connecting inner shell to outer shell.
        for ev, az in [
            (ev_min, az_min), (ev_min, az_max),
            (ev_max, az_min), (ev_max, az_max),
        ]:
            inner = offset(d_min, ev, az)
            outer = offset(d_max, ev, az)
            ui.scene.line(list(inner), list(outer)).material("#88bbff", opacity=0.8)

    logger.info(
        "hemisphere volume wireframe at %s: d=[%.2f, %.2f] m, "
        "elev=[%.0f, %.0f] deg, az_center=%.0f deg ±%.0f deg",
        np.round(target_world, 3).tolist(),
        d_min, d_max, ev_min, ev_max,
        _hemi_azimuth_center_deg(), _HEMI_AZIMUTH_SPREAD_DEG,
    )

    # Reachability sampling — overlay green dots at hemisphere positions where
    # the pose generator's IK + workspace check actually succeeds. Gives a
    # PRE-CALIBRATION view of which parts of the volume are usable.
    if _SHOW_REACHABILITY_POINTS:
        _add_reachability_points(grp, target_world)


def _add_reachability_points(scene_group: Any, target_world: NDArray[np.float64]) -> None:
    """Sample the hemisphere volume via the orchestrator's PoseGenerator,
    render reachable camera positions as green spheres inside ``scene_group``.

    Why the PoseGenerator and not naive IK: the PoseGenerator uses continuity
    seeds (each successful candidate's joint angles seed the next IK call),
    which dramatically improves IK convergence. With a flat zero-seed we get
    near-zero reachable poses; with continuity seeds we get the same set the
    orchestrator will actually use during calibration. The viz here is then
    a true preview of "where will calibration go" before you click Run.

    ``scene_group`` is already translated to ``target_world``, so we offset
    each camera-world position by -target_world before drawing.
    """
    try:
        from parol6 import Robot  # noqa: PLC0415
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
        from parol6_vision.calibration.pose_generator import (  # noqa: PLC0415
            HemisphereParams,
            PoseGenerator,
        )
    except ImportError:
        logger.debug("parol6/parol6_vision not importable; skipping reachability viz")
        return

    _ensure_workspace_envelope()

    cold_start = CameraMount.from_eyeball_estimate(
        x_mm=_CAM_MOUNT_TRANSLATE_MM[0],
        y_mm=_CAM_MOUNT_TRANSLATE_MM[1],
        z_mm=_CAM_MOUNT_TRANSLATE_MM[2],
        tilt_x_deg=_CAM_MOUNT_TILT_DEG[0],
        tilt_y_deg=_CAM_MOUNT_TILT_DEG[1],
        tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
    )

    n_d, n_ev, n_az = _REACHABILITY_GRID
    d_min, d_max = _HEMI_DISTANCE_RANGE_M
    ev_min, ev_max = _HEMI_ELEVATION_RANGE_DEG
    az_min, az_max = _hemi_azimuth_world_range_deg()

    distances = tuple(np.linspace(d_min, d_max, n_d).tolist())
    elevations = tuple(np.linspace(ev_min, ev_max, n_ev).tolist())
    azimuth_counts = tuple([n_az] * n_ev)  # same number of azimuths at every elevation

    try:
        robot = Robot()
    except Exception as e:  # noqa: BLE001
        logger.warning("could not instantiate Robot for reachability viz: %s", e)
        return

    params = HemisphereParams(
        distances_m=distances,
        elevations_deg=elevations,
        azimuth_counts=azimuth_counts,
        azimuth_range_deg=(az_min, az_max),
        workspace_xy_max_m=0.55,
        max_joint_change_deg=180.0,  # relax for viz — we just want reachability, not smoothness
    )
    gen = PoseGenerator(
        robot=robot, mount=cold_start, target_world=target_world, params=params,
    )
    cands, stats = gen.generate(max_count=n_d * n_ev * n_az)

    # Extract the camera position implied by each accepted candidate's flange
    # pose (T_cam2base = T_flange2base @ T_cam2flange).
    reachable_cam_world: list[NDArray[np.float64]] = []
    for c in cands:
        T_cam2base = cold_start.cam_pose_for_flange_pose(np.asarray(c.flange_pose))
        cam_pos = T_cam2base[:3, 3]
        # Optional final hull check on flange position (the PoseGenerator's
        # workspace_xy/z bounds are rectangular; the hull is more accurate).
        if not bool(envelope_contains(np.asarray(c.flange_pose)[:3, 3])[0]):
            continue
        reachable_cam_world.append(cam_pos)

    logger.info(
        "reachability sampling: %d reachable (considered=%d, IK fails=%d, "
        "joint-jump=%d, joint-limits=%d, workspace=%d, singular=%d)",
        len(reachable_cam_world),
        stats.candidates_considered,
        stats.rejection_log.get("ik_failed", 0),
        stats.rejection_log.get("joint_jump", 0),
        stats.rejection_log.get("joint_limits", 0),
        stats.rejection_log.get("workspace_xy", 0) + stats.rejection_log.get("workspace_z", 0),
        stats.rejection_log.get("singular", 0),
    )

    # Greedy farthest-first thinning so the rendered points form a uniform
    # spread across the reachable region instead of clustered along grid
    # lines. Approximates Poisson-disc sampling without an explicit minimum-
    # distance threshold — instead we pick a target count and let the
    # algorithm maximise minimum pairwise distance for that count.
    points_to_render = reachable_cam_world
    if _REACHABILITY_KEEP_COUNT is not None and len(points_to_render) > _REACHABILITY_KEEP_COUNT:
        points_to_render = _greedy_farthest_first(
            points_to_render, _REACHABILITY_KEEP_COUNT
        )
        logger.info(
            "farthest-first thinning: %d -> %d points",
            len(reachable_cam_world), len(points_to_render),
        )

    # Render points as small green spheres, in scene_group's local frame
    # (which is already translated to target_world).
    with scene_group:
        for cam_pos in points_to_render:
            local = (cam_pos - target_world).tolist()
            ui.scene.sphere(0.006).move(*local).material("#33dd66", opacity=0.85)


def _greedy_farthest_first(
    items: list[Any],
    n_target: int,
    key: Callable[[Any], NDArray[np.float64]] = lambda x: x,  # type: ignore[assignment]
) -> list[Any]:
    """Pick ``n_target`` items so their position-space minimum pairwise
    distance is approximately maximised.

    Algorithm: start at the centroid-closest item (deterministic), then
    repeatedly pick the next as the one with the LARGEST minimum distance
    to the already-picked set. Greedy heuristic for the maxmin-distance /
    k-center problem; produces uniform-looking spread without parameter
    tuning. O(N × n_target).

    ``key`` extracts a 3-vector position from each item, so this helper
    works on raw points (key=identity) AND on PoseGenerator candidates
    (key=lambda c: c.flange_pose[:3, 3]).
    """
    if len(items) <= n_target:
        return list(items)
    pts = np.asarray([key(item) for item in items], dtype=np.float64)
    # Start with the item closest to the centroid (deterministic seed).
    centroid = pts.mean(axis=0)
    first_idx = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
    selected = [first_idx]
    # Track min distance from each candidate to the selected set.
    min_dist = np.linalg.norm(pts - pts[first_idx], axis=1)
    while len(selected) < n_target:
        next_idx = int(np.argmax(min_dist))
        if min_dist[next_idx] <= 0:
            break  # all remaining items coincide with already-selected
        selected.append(next_idx)
        new_dists = np.linalg.norm(pts - pts[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dists)
    return [items[i] for i in selected]


def update_frustum(T_cam2flange: NDArray[np.float64]) -> None:
    """Replace frustum lines after a mount update."""
    group = _state.get("frustum_group")
    if group is None:
        return
    for obj in _state.get("frustum_objects", []):
        try:
            obj.delete()
        except Exception:  # noqa: BLE001
            pass
    _state["frustum_objects"] = _populate_frustum(group, T_cam2flange)


# ---------------------------------------------------------------------------
# Calibration runner (background thread)
# ---------------------------------------------------------------------------


def _calibration_thread() -> None:
    """Run the calibration sim end-to-end and push joint updates to the GUI.

    Drives the running parol6-server's controller via UDP — same path
    Waldo-Commander uses — so each ``move_j`` is executed by the controller
    and its joint state is broadcast to all subscribers (including the
    URDF scene's status consumer). The GUI animates naturally; no
    update_urdf_angles thread plumbing required.
    """
    try:
        # Lazy imports.
        from parol6 import Robot, RobotClient  # noqa: PLC0415

        from parol6_vision.calibration.board import (  # noqa: PLC0415
            BOARD_TABLET_30MM,
            BoardDetector,
        )
        from parol6_vision.calibration.camera_mount import (  # noqa: PLC0415
            CameraMount,
        )
        from parol6_vision.calibration.orchestrator import (  # noqa: PLC0415
            CalibrationOrchestrator,
            OrchestratorConfig,
        )
        from parol6_vision.calibration.pose_generator import (  # noqa: PLC0415
            HemisphereParams,
            PoseGenerator,
        )
        from parol6_vision.camera.intrinsics import Intrinsics  # noqa: PLC0415
        from parol6_vision.sim.virtual_camera import (  # noqa: PLC0415
            VirtualBoard,
            VirtualCamera,
        )
        from parol6_vision.sim.tracing_client import (  # noqa: PLC0415
            _flange_pose_from_client,
        )

        # Ground truth = cold-start tunable + a small fixed perturbation, so
        # the simulated calibration always has something realistic to converge
        # to regardless of how the user sets _CAM_MOUNT_TRANSLATE_MM. The
        # perturbation is small (±2 mm, ±2°) so convergence is reliable.
        ground_truth_mount = CameraMount.from_eyeball_estimate(
            x_mm=_CAM_MOUNT_TRANSLATE_MM[0] + 2.0,
            y_mm=_CAM_MOUNT_TRANSLATE_MM[1] + 2.0,
            z_mm=_CAM_MOUNT_TRANSLATE_MM[2] - 2.0,
            tilt_x_deg=_CAM_MOUNT_TILT_DEG[0] + 2.0,
            tilt_y_deg=_CAM_MOUNT_TILT_DEG[1] - 1.0,
            tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
        )
        T_BOARD2BASE = _T_BOARD2BASE
        intrinsics = Intrinsics(
            fx=_INTR_FX, fy=_INTR_FY, cx=_INTR_CX, cy=_INTR_CY,
            width=_INTR_W, height=_INTR_H,
            dist_coeffs=np.zeros(5, dtype=np.float64),
        )

        robot = Robot()

        # Drive the running parol6-server (controller in fake-serial mode).
        # Each move_j blocks until motion completes; the controller broadcasts
        # joint state at 50 Hz, which Waldo-Commander's status consumer picks
        # up and feeds into the URDF scene. No thread/queue plumbing needed.
        raw_client = RobotClient(host="127.0.0.1", port=5001)

        # Wrap the client so the STOP button can short-circuit subsequent
        # move_j calls — halt() alone only aborts the in-flight motion;
        # the orchestrator's own loop will happily request the next pose
        # right after. Returning -1 from move_j makes _move_to_joints
        # treat each remaining pose as a MotionError and the orchestrator
        # gives up cleanly with ``insufficient_samples_pass1``.
        class _HaltableClient:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def move_j(self, *args: Any, **kwargs: Any) -> int:
                if _state.get("stop_requested"):
                    return -1
                return self._inner.move_j(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        client = _HaltableClient(raw_client)
        # Stash the raw client so the STOP button can call halt() directly.
        _state["client"] = raw_client

        def flange_pose():
            return _flange_pose_from_client(client)

        virtual_camera = VirtualCamera(
            intrinsics=intrinsics,
            image_width=_INTR_W,
            image_height=_INTR_H,
            ground_truth_mount=ground_truth_mount,
            flange_pose_provider=flange_pose,
            board=VirtualBoard(config=BOARD_TABLET_30MM, T_board2base=T_BOARD2BASE),
            noise_std=0.0,
        )
        detector = BoardDetector(BOARD_TABLET_30MM)
        cold_start = CameraMount.from_eyeball_estimate(
            x_mm=_CAM_MOUNT_TRANSLATE_MM[0],
            y_mm=_CAM_MOUNT_TRANSLATE_MM[1],
            z_mm=_CAM_MOUNT_TRANSLATE_MM[2],
            tilt_x_deg=_CAM_MOUNT_TILT_DEG[0],
            tilt_y_deg=_CAM_MOUNT_TILT_DEG[1],
            tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
        )

        # The hemisphere + pose generator must aim at the board CENTER, not
        # the board origin (corner) stored in T_BOARD2BASE[:3, 3]. The viz
        # uses the same center via _board_center_world() so what's drawn in
        # the GUI matches what the calibration math actually targets.
        target_world = _board_center_world()

        # Diagnostic: confirm the board CENTER is reachable per waldo-commander's
        # workspace hull. If not, the orchestrator will struggle to find any
        # valid hemisphere candidates.
        _ensure_workspace_envelope()
        max_reach = _state.get("envelope_max_reach")
        if max_reach is not None:
            board_ok = bool(envelope_contains(target_world)[0])
            logger.info(
                "workspace envelope check: board center %s -> %s "
                "(hull max_reach=%.3f m)",
                np.round(target_world, 3).tolist(),
                "INSIDE" if board_ok else "OUTSIDE",
                max_reach,
            )

        # Hull-aware + floor-aware + collision-aware + occlusion-aware +
        # trajectory-aware + farthest-first PoseGenerator.
        # Filters layered on top of the base PoseGenerator:
        #   1. Workspace-hull check on flange position.
        #   2. Floor-clipping check: gripper fingertips above floor.
        #   3. Self-collision: gripper/bracket vs arm links at the destination pose.
        #   4. Camera occlusion: line-of-sight from camera to board NOT blocked
        #      by any robot link. Catches "robot body in front of the lens" poses
        #      that aren't collisions but produce useless calibration captures.
        #   5. Greedy farthest-first selection (spatial spread).
        #   6. Trajectory self-collision: each move from the previous accepted
        #      candidate to this one must not pass through self-collision at
        #      any sampled point along the joint-space line. Plus a check from
        #      HOME → first candidate. Catches the "fingertips clip mid-move"
        #      case the user reported.
        FLOOR_Z_MIN_M = 0.005  # 5 mm safety margin above the workbench
        TCP_OFFSET_FLANGE = np.array([0.0, 0.0, -0.105, 1.0])  # SSG-48 TCP point in flange frame

        # Build collision manager + occlusion meshes once (loads 7 link meshes
        # + gripper for collision; 7 link meshes for occlusion).
        collision_mgr_pair = (
            _build_collision_manager() if _ENABLE_SELF_COLLISION_CHECK else None
        )
        occlusion_meshes = (
            _build_occlusion_mesh() if _ENABLE_SELF_COLLISION_CHECK else None
        )

        class HullFilteredPoseGenerator(PoseGenerator):
            def generate(self, *args, **kwargs):  # type: ignore[override]
                max_count = kwargs.get("max_count", args[0] if args else 8)
                # Over-request so we have a pool to thin from.
                kwargs["max_count"] = max_count * 5
                cands, stats = super().generate(**kwargs)  # type: ignore[arg-type]

                # 1. Hull filter.
                if _state.get("envelope_planes_A") is not None:
                    pre = len(cands)
                    cands = [
                        c for c in cands
                        if bool(envelope_contains(
                            np.asarray(c.flange_pose, dtype=np.float64)[:3, 3]
                        )[0])
                    ]
                    if pre - len(cands) > 0:
                        logger.info(
                            "hull filter: %d/%d candidates rejected (flange outside envelope)",
                            pre - len(cands), pre,
                        )

                # 2. Floor-clipping filter — TCP world Z must clear the floor.
                pre = len(cands)
                cands = [
                    c for c in cands
                    if (np.asarray(c.flange_pose, dtype=np.float64)
                        @ TCP_OFFSET_FLANGE)[2] > FLOOR_Z_MIN_M
                ]
                if pre - len(cands) > 0:
                    logger.info(
                        "floor filter: %d/%d candidates rejected (gripper would clip floor)",
                        pre - len(cands), pre,
                    )

                # 3. Self-collision filter — gripper/bracket must not clip arm.
                if collision_mgr_pair is not None:
                    coll_mgr, adjacent_pairs = collision_mgr_pair
                    pre = len(cands)
                    cands = [
                        c for c in cands
                        if not _self_collides(coll_mgr, adjacent_pairs, c.joint_angles_rad)
                    ]
                    if pre - len(cands) > 0:
                        logger.info(
                            "self-collision filter: %d/%d candidates rejected "
                            "(gripper clips into arm)",
                            pre - len(cands), pre,
                        )

                # 4. Camera-occlusion filter — line of sight from the camera
                # to the board centre must not pass through any robot link.
                if occlusion_meshes is not None:
                    cold_T = self.mount.T_cam2flange
                    pre = len(cands)
                    survivors = []
                    for c in cands:
                        T_flange2base = np.asarray(c.flange_pose, dtype=np.float64)
                        T_cam2base = T_flange2base @ cold_T
                        cam_pos_world = T_cam2base[:3, 3]
                        if not _camera_occluded(
                            occlusion_meshes, c.joint_angles_rad,
                            cam_pos_world, np.asarray(self.target_world),
                        ):
                            survivors.append(c)
                    cands = survivors
                    if pre - len(cands) > 0:
                        logger.info(
                            "occlusion filter: %d/%d candidates rejected "
                            "(robot body blocks camera view of board)",
                            pre - len(cands), pre,
                        )

                # 5. Farthest-first thinning on flange positions.
                if len(cands) > max_count:
                    cands = _greedy_farthest_first(
                        cands, max_count,
                        key=lambda c: np.asarray(c.flange_pose, dtype=np.float64)[:3, 3],
                    )
                    logger.info("farthest-first selected %d well-spread candidates", len(cands))

                # 6. Trajectory self-collision (pairwise). Each successive
                # move from the previously-selected candidate to the next must
                # not pass through self-collision at any sampled point along
                # the joint-space line. The FIRST candidate is accepted
                # unconditionally — the orchestrator will be coming from the
                # last bootstrap pose at that point, not from HOME, and we
                # don't know where bootstrap landed (it'd over-reject if we
                # assumed HOME).
                if collision_mgr_pair is not None and len(cands) > 0:
                    coll_mgr, adjacent_pairs = collision_mgr_pair
                    safe: list = [cands[0]]
                    n_traj_rejected = 0
                    for c in cands[1:]:
                        prev_q = safe[-1].joint_angles_rad
                        if _trajectory_collides(
                            coll_mgr, adjacent_pairs, prev_q, c.joint_angles_rad,
                        ):
                            n_traj_rejected += 1
                            continue
                        safe.append(c)
                    if n_traj_rejected > 0:
                        logger.info(
                            "trajectory filter: %d/%d candidates rejected "
                            "(consecutive-move joint-space line passes through "
                            "self-collision)", n_traj_rejected, len(cands),
                        )
                    cands = safe

                return cands, stats

        # Monkey-patch the orchestrator's PoseGenerator binding so the
        # subclass is used during the main hemisphere pass too. The original
        # is stashed in _state so the finally block can restore even if we
        # crash mid-run.
        import parol6_vision.calibration.orchestrator as _orch_mod  # noqa: PLC0415
        _state["orig_pose_generator"] = _orch_mod.PoseGenerator
        _orch_mod.PoseGenerator = HullFilteredPoseGenerator

        # Monkey-patch `board_position_from_sample` to return the BOARD CENTER
        # rather than the BOARD ORIGIN (corner). The orchestrator uses the
        # bootstrap result as the look-at target for the main hemisphere pass;
        # without this fix the camera ends up aimed at one corner of the
        # board, not the geometric centre, so off-axis poses see a heavily
        # cropped view (or miss the board entirely).
        import parol6_vision.calibration.refinement as _refinement_mod  # noqa: PLC0415
        from parol6_vision.calibration.board import board_pose_to_matrix as _board_pose_to_matrix  # noqa: PLC0415
        _state["orig_board_position_from_sample"] = _refinement_mod.board_position_from_sample
        _bcfg = BOARD_TABLET_30MM
        _board_center_offset_local = np.array(
            [_bcfg.squares_x * _bcfg.square_length / 2.0,
             _bcfg.squares_y * _bcfg.square_length / 2.0,
             0.0, 1.0]
        )

        def _board_center_from_sample(T_flange2base, detection, mount):
            """Return the board CENTER (not the origin/corner) in base frame."""
            T_board2cam = _board_pose_to_matrix(detection)
            T_board2base = (
                np.asarray(T_flange2base, dtype=np.float64)
                @ mount.T_cam2flange
                @ T_board2cam
            )
            return (T_board2base @ _board_center_offset_local)[:3]

        _refinement_mod.board_position_from_sample = _board_center_from_sample

        # Hemisphere params for the bootstrap pass and the main pass.
        #
        # MAIN pass: drives off the user-facing _HEMI_* tunables. Wide range,
        # may include low-elevation / close-distance poses where part of the
        # board can't fit in the FOV — that's fine, the orchestrator only
        # needs PARTIAL board visibility (≥6 corners) to count a sample.
        #
        # BOOTSTRAP pass: uses CONSERVATIVE bounds that are independent of
        # the user-facing tunables. The bootstrap's job is to localise the
        # board reliably (it only takes a handful of seed poses), so we want
        # poses where the entire 210×150 mm board comfortably fits in the
        # 640×480 FOV (≥0.22 m at fx=fy=615 gives ~230 mm horizontal coverage,
        # so the board fits with margin). Mid-range elevations (30–55°) avoid
        # extreme foreshortening that hurts ChArUco corner detection.
        #
        # Both grids are DENSE because tilt_x=180 forces wrist-flip
        # configurations and only ~5% of grid points pass IK.
        d_min, d_max = _HEMI_DISTANCE_RANGE_M
        ev_min, ev_max = _HEMI_ELEVATION_RANGE_DEG
        az_world_range = _hemi_azimuth_world_range_deg()
        bootstrap_params = HemisphereParams(
            distances_m=(0.22, 0.26, 0.30, 0.34),
            elevations_deg=(30.0, 45.0, 60.0, 75.0, 88.0),
            azimuth_counts=(16, 14, 12, 10, 8),  # 60 az per distance × 4 = 240
            azimuth_range_deg=az_world_range,
            workspace_xy_max_m=0.55,
            max_joint_change_deg=180.0,
        )
        main_params = HemisphereParams(
            distances_m=tuple(np.linspace(d_min, d_max, 6).tolist()),
            elevations_deg=tuple(np.linspace(ev_min, ev_max, 6).tolist()),
            azimuth_counts=(20, 16, 12, 9, 6, 4),  # 67 azimuths per distance × 6 = 402
            azimuth_range_deg=az_world_range,
            max_joint_change_deg=180.0,
            workspace_xy_max_m=0.55,
        )


        bg = HullFilteredPoseGenerator(
            robot=robot, mount=cold_start,
            target_world=target_world,
            params=bootstrap_params,
        )
        # 16 bootstrap candidates instead of 8 — at observed ~25-50% detection
        # rate, 8 sometimes gives <4 successful detections (orchestrator's
        # consensus threshold). 16 gives ~4-8 detections, well above threshold.
        # Cost: a few extra seconds of robot motion at calibration start.
        bcands, _ = bg.generate(max_count=16)
        boot_cfg = tuple(
            tuple(np.degrees(c.joint_angles_rad).tolist()) for c in bcands
        )

        # Reduced sample count + zero settle time — the controller's motion
        # physics already enforces realistic timing, so we don't need
        # additional settling. Sample count trades calibration quality for
        # wallclock time; 12 is enough for a watchable demo.
        config = OrchestratorConfig(
            bootstrap_joint_configs_deg=boot_cfg,
            target_sample_count=12,
            min_sample_count=6,
            enable_second_pass=False,
            settle_time_s=0.2,
            hemisphere=main_params,
        )

        orchestrator = CalibrationOrchestrator(
            robot=robot, client=client,
            camera=virtual_camera, detector=detector,
            cold_start_mount=cold_start, config=config,
        )

        t0 = time.perf_counter()
        output = orchestrator.calibrate()
        wall_s = time.perf_counter() - t0

        if output.failed:
            _post_status(f"Calibration FAILED: {output.failure_reason}")
        else:
            gt = ground_truth_mount.T_cam2flange[:3, 3]
            cal = output.mount.T_cam2flange[:3, 3]
            pos_err = float(np.linalg.norm(gt - cal)) * 1000.0
            _post_status(
                f"DONE ({wall_s:.1f}s, {output.n_samples_collected} samples). "
                f"Best={output.best_method}, error={pos_err:.2f}mm"
            )
            _state["calibrated_mount"] = output.mount

    except Exception as e:  # noqa: BLE001
        # The orchestrator's own `finally` calls set_tcp_offset to restore the
        # previous offset; if the user pressed STOP partway through, the
        # controller is halted and that call raises MotionError. Treat
        # exceptions during a stop request as a clean stop, not a crash.
        if _state.get("stop_requested"):
            logger.info("Calibration stopped by user (caught %s: %s)",
                        type(e).__name__, e)
            _post_status("Calibration stopped by user")
        else:
            logger.exception("Calibration thread crashed")
            _post_status(f"ERROR: {e}")
    finally:
        # Restore the orchestrator's PoseGenerator binding if we patched it.
        if _state.get("orig_pose_generator") is not None:
            try:
                import parol6_vision.calibration.orchestrator as _orch_mod  # noqa: PLC0415
                _orch_mod.PoseGenerator = _state["orig_pose_generator"]
            except Exception:  # noqa: BLE001
                pass
            _state["orig_pose_generator"] = None
        # Restore the board-position function if we patched it.
        if _state.get("orig_board_position_from_sample") is not None:
            try:
                import parol6_vision.calibration.refinement as _refinement_mod  # noqa: PLC0415
                _refinement_mod.board_position_from_sample = _state["orig_board_position_from_sample"]
            except Exception:  # noqa: BLE001
                pass
            _state["orig_board_position_from_sample"] = None
        _state["is_running"] = False


def _post_status(text: str) -> None:
    """Post a status update to the GUI from a worker thread."""
    label = _state.get("status_label")
    loop = _state.get("main_loop")
    if label is None or loop is None:
        logger.info("Calibration status: %s", text)
        return

    def _update():
        label.text = text

    try:
        loop.call_soon_threadsafe(_update)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not post GUI status: %s", e)


def _post_calibration_tick() -> None:
    """Apply the calibrated mount to the frustum once calibration finishes.

    Called by a NiceGUI timer at 4 Hz. The robot animates via the existing
    parol6-server status broadcast → Waldo-Commander status consumer →
    URDF scene path, so we don't need to drive joint updates ourselves.
    The only remaining bit of plumbing is updating the frustum once the
    calibration finishes and we have a calibrated ``T_cam2flange``.
    """
    if not _state.get("is_running") and _state.get("calibrated_mount") is not None:
        update_frustum(_state["calibrated_mount"].T_cam2flange)
        _state["current_mount"] = _state["calibrated_mount"]
        _state["calibrated_mount"] = None


# ---------------------------------------------------------------------------
# Floating control panel
# ---------------------------------------------------------------------------


def add_control_panel() -> None:
    """Add a small floating panel with a 'Run Calibration' button + status.

    Call this from inside the page context after the scene is built.
    """

    def _on_click() -> None:
        if _state.get("is_running"):
            ui.notify(
                "Calibration is already running — wait for it to finish",
                color="warning",
                position="top",
            )
            return
        _state["is_running"] = True
        _state["stop_requested"] = False
        _state["calibrated_mount"] = None
        _post_status("Running calibration via parol6-server...")
        threading.Thread(target=_calibration_thread, daemon=True).start()

    def _on_stop() -> None:
        """Abort the running calibration: halt the controller + flag the thread.

        ``RobotClient.halt()`` is a sync UDP call that refuses to run inside
        an active asyncio event loop. The NiceGUI callback runs in the main
        event loop, so calling halt() directly from here raises
        ``RobotClient was used while an event loop is running``. Workaround:
        dispatch halt() to a daemon thread which has no event loop attached.
        The flag (_state["stop_requested"]) is set immediately so subsequent
        move_j calls in the calibration thread short-circuit even if the
        thread-dispatched halt hasn't fired yet.
        """
        if not _state.get("is_running"):
            return
        _state["stop_requested"] = True
        client = _state.get("client")
        if client is not None:
            def _halt_in_thread():
                try:
                    client.halt()
                except Exception as e:  # noqa: BLE001
                    logger.warning("halt() in worker thread failed: %s", e)
            threading.Thread(target=_halt_in_thread, daemon=True).start()
            ui.notify("Calibration HALTED — robot motion stopped", color="warning")
        _post_status("Stop requested — wait for current move to finish")

    with ui.element("div").style(
        "position: absolute; top: 78px; left: 80px; z-index: 30; "
        "background: rgba(20, 22, 28, 0.85); padding: 8px 12px; "
        "border-radius: 8px; color: #ddd; font-size: 12px; "
        "max-width: 280px;"
    ):
        ui.label("parol6-vision calibration").classes("font-semibold text-sm")
        _state["status_label"] = ui.label("Idle.").classes("text-xs opacity-80")
        with ui.row().classes("gap-1"):
            ui.button("Run", on_click=_on_click, color="primary").props("size=sm")
            ui.button("STOP", on_click=_on_stop, color="negative").props("size=sm")

    # 4 Hz tick to apply the calibrated mount once calibration finishes.
    # No joint-update plumbing needed — the parol6-server status broadcast
    # drives the URDF scene naturally.
    ui.timer(0.25, _post_calibration_tick, active=True)
