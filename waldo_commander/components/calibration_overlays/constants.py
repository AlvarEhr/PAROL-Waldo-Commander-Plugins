"""User-facing tunables and fit constants for calibration overlays.

Edits take effect on next waldo-commander restart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants matching parol6-vision's sim config
# ---------------------------------------------------------------------------


# =============================================================================
# === USER-FACING TUNABLES — EDIT THESE ====================================
# =============================================================================
#
# Flange-frame convention: X = out along bracket, Y = across fingers,
# Z = down toward fingertip.

# Translation in metres, applied on top of the bbox-fit transform.
_MERGED_STL_TRANSLATE_M: tuple[float, float, float] = (-0.02788, 0.0, 0.0)
# Rotation as XYZ-extrinsic Euler in radians.
_MERGED_STL_RPY_RAD: tuple[float, float, float] = (0.0, 0.0, 0.0)


# =============================================================================
# === FIT TRANSFORM — DO NOT EDIT ===========================================
# =============================================================================
#
# Derived from the bbox-fit step in parol6-vision; pulls the merged STL onto
# the upstream ssg48_body. To shift the bracket visually use the tunables
# above instead.
_MERGED_STL_FIT_SCALE: float = 0.1
_MERGED_STL_FIT_TRANSLATE_M: tuple[float, float, float] = (13.95538, -0.00737, 0.0)

# ChArUco board pose in the base frame. Translate is the corner-anchored
# origin (OpenCV ChArUco convention); RPY is XYZ-extrinsic Euler in radians.
_BOARD_TRANSLATE_M: tuple[float, float, float] = (0.25, -0.1, 0.0)
_BOARD_RPY_RAD: tuple[float, float, float] = (0.0, 0.0, np.deg2rad(90))

# Hemisphere search region — shared between the wireframe overlay and the
# orchestrator's HemisphereParams. Elevation is angle above the board plane
# (0 = level, 90 = overhead); azimuth spread is half-angle around the
# auto-computed base→board direction. Overlay visibility is driven by
# calibration-panel checkboxes; see ``_set_overlay_visible``.
_HEMI_DISTANCE_RANGE_M: tuple[float, float] = (0.14, 0.28)
# Upper bound 89° is the practical near-overhead cap (exactly 90° is
# degenerate for look_at). The pose generator appends an explicit cap ring
# at ev_hi so the boundary is sampled despite Sobol never reaching u=1.
_HEMI_ELEVATION_RANGE_DEG: tuple[float, float] = (15.0, 89.0)
# 150° spread = 300° azimuth coverage. Going wider than the reachable
# workspace is safe; the hull filter drops unreachable back-side candidates.
_HEMI_AZIMUTH_SPREAD_DEG: float = 150.0

# Hemisphere centre override — fixed world-frame anchor for the wireframe,
# reachability sampler, and pose-generator hemisphere. Look-at target stays
# the board centre. Decouples the hemisphere position from board yaw. Set
# to None to use the board centre.
_HEMI_CENTRE_OVERRIDE_M: tuple[float, float, float] | None = None

# Self-collision rejection via trimesh.CollisionManager; adjacent links
# are whitelisted. Disable if python-fcl misbehaves.
_ENABLE_SELF_COLLISION_CHECK: bool = True

# (n_distances, n_elevations, n_azimuths). Dense to compensate for the
# ~4% reachability rate caused by tilt_x=180 wrist-flip.
_REACHABILITY_GRID = (7, 6, 20)
# Sobol vs discrete sampling for the green-dot viz. Continuous gives
# uniform 3D coverage; discrete forms visible rings after IK filtering.
_REACHABILITY_USE_CONTINUOUS: bool = True
_REACHABILITY_N_CANDIDATES: int = 1024

# Calibration hemisphere sampling toggle — same as the viz toggle above,
# applied via HullFilteredPoseGenerator on both bootstrap and main passes.
_CALIBRATION_USE_CONTINUOUS: bool = True
_BOOTSTRAP_N_CANDIDATES: int = 512
_MAIN_N_CANDIDATES: int = 1024
# Farthest-first thinning to a spatially well-spread subset. None = no
# thinning, show every reachable point.
_REACHABILITY_KEEP_COUNT: int | None = None

# Multi-target relaxed look-at — DEFERRED_FEATURES.md §6. Each entry is a
# (u, v) in board-local UV ∈ [0, 1]² where (0.5, 0.5) is the centre. Pose
# generator runs IK against EACH target and accumulates all feasible
# candidates. ``((0.5, 0.5),)`` is single-centre behaviour.
_BOARD_TARGET_OFFSETS_LOCAL: tuple[tuple[float, float], ...] = (
    (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5),
)

# Multi-ray occlusion sampling — partial-frame robot-body occlusion. 9
# samples per board (centre + corners + mid-edges) in UV ∈ [0, 1]² catches
# quadrant occlusion that a single centre ray misses.
_OCCLUSION_BOARD_SAMPLES_LOCAL: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),                                       # centre
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),   # corners
    (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),   # mid-edges
)
# Reject when more than this many rays are blocked. 0 = any link occlusion
# fails; ChArUco needs ≥6 corners so this is justified-strict.
_OCCLUSION_MAX_BLOCKED: int = 0

# Per-pose settle time before frame capture (vibration + auto-exposure).
# Sim's discrete-time physics doesn't ring; real hardware needs longer.
_SETTLE_TIME_SIM_S: float = 0.2
_SETTLE_TIME_REAL_S: float = 2.5

# Viewing-angle filter — reject when the camera axis is more than this off
# the board normal. Picked just above ``ELEVATION_RANGE`` lower bound so
# the lowest hemisphere elevations still survive.
_MAX_CAM_BOARD_ANGLE_DEG: float = 78.0

# Floor-collision primitive — flat box at z<0 caught by the same mesh
# check as self-collision. More accurate than a TCP-point check.
_FLOOR_PRIMITIVE_ENABLED: bool = True

# Safety margin (metres) inflating FLOOR and TABLET primitives. Cushions
# residual mount error so the planner rejects rather than visualises
# actual penetration.
_COLLISION_SAFETY_MARGIN_M: float = 0.008

# SSG-48 gripper jaw variant for collision checking. Only one variant is
# physically mounted; loading both would over-reject.
_SSG48_JAW_VARIANT: str = "finger"  # "finger" or "pinch"

# Tablet collision primitive — static box at the current _T_BOARD2BASE
# pose. Top face is at z=0 (screen surface); extends into board-local -Z.
_TABLET_PRIMITIVE_ENABLED: bool = True
# Board-local axes: +X = ChArUco long edge, +Y = short edge, +Z = out of
# the board surface. Tab S9 Ultra (with case) in landscape orientation.
_TABLET_DIMENSIONS_M: tuple[float, float, float] = (
    0.340,  # along board-local +X (ChArUco long edge / 7-square direction)
    0.215,  # along board-local +Y (ChArUco short edge / 5-square direction)
    0.014,  # along board-local +Z (thickness — Tab S9 Ultra with case)
)
# Offset of the tablet centre from the ChArUco centre in board-local
# (+X, +Y). Non-zero when the ChArUco doesn't fill the tablet screen.
_TABLET_OFFSET_FROM_CHARUCO_LOCAL_M: tuple[float, float] = (0.0, 0.0)
# Translucent overlay for visual verification of the modelled box.
_SHOW_TABLET_OVERLAY: bool = True

# Auto-localise board (DEFERRED_FEATURES.md §3) — workspace scan for the
# "Localise Board" button. Drives a grid of (x, y) scan points, each
# looking down; the first FOV that detects ChArUco updates _T_BOARD2BASE
# (with median filtering). Z is auto-lifted by tablet thickness.
_LOCALISE_SCAN_TARGETS_M: tuple[tuple[float, float], ...] = (
    (0.20, -0.15), (0.20,  0.00), (0.20, +0.15),
    (0.30, -0.15), (0.30,  0.00), (0.30, +0.15),
    (0.40, -0.15), (0.40,  0.00), (0.40, +0.15),
)
# Camera standoff per scan target. Tried in order — first reachable wins.
_LOCALISE_SCAN_DISTANCES_M: tuple[float, ...] = (0.28, 0.32)
# Elevation sweep per target (90° = pure overhead). Lower bounds avoid
# the IK failures pure overhead causes with tilt_x=180 wrist-flip.
_LOCALISE_SCAN_ELEVATIONS_DEG: tuple[float, ...] = (70.0, 60.0, 50.0, 45.0)
# Azimuth range and sample count per (distance, elevation).
_LOCALISE_SCAN_AZIMUTH_RANGE_DEG: tuple[float, float] = (-100.0, 100.0)
_LOCALISE_SCAN_AZIMUTH_COUNT: int = 4

# J0-sweep localise mode (DEFAULT) — sweep the base joint while J1-J5 stay
# fixed; the camera gaze traces an arc around the base. Multiple seed
# targets cover different annular bands. Fall back to multi-target scan
# above by setting False.
_LOCALISE_USE_J0_SWEEP: bool = True
# Continuous capture during motion vs. discrete stop-and-capture.
_LOCALISE_CONTINUOUS_SWEEP: bool = True
# Seed look-at targets (close, centre, far), tried in order.
_LOCALISE_SEED_TARGETS_XY: tuple[tuple[float, float], ...] = (
    (0.22, 0.0),  # close — workspaces near the base
    (0.30, 0.0),  # workspace centre
    (0.38, 0.0),  # farther — outward-placed boards
)
# Backwards-compat alias.
_LOCALISE_SEED_TARGET_M: tuple[float, float] = _LOCALISE_SEED_TARGETS_XY[1]
# Continuous-sweep speed as a fraction of max joint speed.
_LOCALISE_SWEEP_SPEED: float = 0.10
# Halt mid-sweep once this many detections are gathered. Large value
# (e.g. 999) disables early stop.
_LOCALISE_EARLY_STOP_DETECTIONS: int = 10
# Chunk size for the J0 sweep. ``halt()`` doesn't interrupt the currently
# executing trajectory, so chunking bounds the halt-response time.
_LOCALISE_J0_CHUNK_DEG: float = 20.0
# Stage-2 refinement: after the sweep finds the board, drive N overhead
# poses above the rough median centre and capture cleaner frames. 0 = off.
_LOCALISE_REFINE_N_POSES: int = 4
_LOCALISE_REFINE_DISTANCE_M: float = 0.30
_LOCALISE_REFINE_ELEVATION_DEG: float = 80.0
# Capture period during a continuous sweep.
_LOCALISE_CAPTURE_PERIOD_S: float = 0.15
# Seed-pose search ranges. Azimuth covers FULL 360° because the
# tilt_x=180 mount's reachable cluster sits behind the base.
_LOCALISE_SEED_DISTANCE_RANGE_M: tuple[float, float] = (0.22, 0.34)
_LOCALISE_SEED_ELEVATION_RANGE_DEG: tuple[float, float] = (60.0, 85.0)
_LOCALISE_SEED_N_CANDIDATES: int = 128
# J0 sweep half-range (degrees relative to seed) and step count. Step
# spacing stays under the camera horizontal FOV at typical standoff.
_LOCALISE_J0_SWEEP_HALF_DEG: float = 90.0
_LOCALISE_J0_STEPS: int = 13
# Min detections to accept a localise. 1 trusts a single confident hit.
_LOCALISE_MIN_DETECTIONS: int = 1
# Two-tier stop logic. EARLY_STOP halts mid-sweep; MIN_TO_PROCEED skips
# remaining seeds when a sweep finishes with enough inliers.
_LOCALISE_EARLY_STOP_INLIERS: int = 7
_LOCALISE_MIN_INLIERS_TO_PROCEED: int = 4
# RANSAC inlier threshold (metres).
_LOCALISE_INLIER_THRESHOLD_M: float = 0.05

# Camera intrinsics for the frustum size (matches sim).
_INTR_FX, _INTR_FY = 615.0, 615.0
_INTR_CX, _INTR_CY = 320.0, 240.0
_INTR_W, _INTR_H = 640, 480

# Camera mount in the flange frame — cone apex + optical-axis orientation
# for the visual cone, the calibration cold start, and the sim ground
# truth. Translation in millimetres (X out along bracket, Y across
# fingers, Z along flange axis, +Z toward base).
_CAM_MOUNT_TRANSLATE_MM: tuple[float, float, float] = (-52.0, 0, -50.0)
# Tilt in degrees, XYZ-extrinsic Euler. tilt_x=180 makes the physical lens
# point in flange -Z; PAROL6 wrist-flips to satisfy look_at. tilt_z=90
# aligns image rows with the robot's side-to-side axis.
_CAM_MOUNT_TILT_DEG: tuple[float, float, float] = (180.0, 0.0, 90.0)

# Frustum far-plane distance in camera optical-axis units. Positive
# extends along the physical lens direction (flange -Z with tilt_x=180).
_FRUSTUM_DEPTH_M: float = 0.20

# "Laser pointer" extension — faint cone from the same apex with the same
# FOV but longer reach. None disables.
_FRUSTUM_FAR_DEPTH_M: float | None = 0.50


# === Detection overlay (live perception viz) ===
# JSON written by parol6-vision's find_object.py. Five parents up reaches
# the sibling parol6-vision repo under "Project Files/".
_DETECTION_JSON_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "parol6-vision" / "Results" / "perception" / "last_detection.json"
)
_DETECTION_POLL_INTERVAL_S: float = 0.5
_DETECTION_OVERLAY_COLOR: str = "#ff8800"  # orange — distinct from board and frustum
