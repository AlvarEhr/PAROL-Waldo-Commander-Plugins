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
import json
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
# base frame) of the board's CORNER-anchored origin (OpenCV ChArUco convention).
# _BOARD_RPY_RAD is the orientation as XYZ-extrinsic Euler angles in radians —
# same convention as scipy.spatial.transform.Rotation.from_euler("XYZ", ...).
#
# Examples:
#   90° rotation about base Z (lay board "sideways"): _BOARD_RPY_RAD = (0.0, 0.0, np.pi/2)
#   Tilt board 30° upward toward the robot: _BOARD_RPY_RAD = (np.deg2rad(30), 0.0, 0.0)
#   Mount on a vertical wall facing -X: _BOARD_RPY_RAD = (0.0, np.pi/2, 0.0)
#   Along X: (0.25, -0.1, 0.0), (0.0, 0.0, np.deg2rad(90))
#   Diagonal: (0, 0.15, 0.0), (0.0, 0.0, np.deg2rad(-45))
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
_HEMI_DISTANCE_RANGE_M: tuple[float, float] = (0.14, 0.28)  # (min, max) radial distance
# 15° → 70°. Lower bound is genuinely useful — low-elevation oblique
# views see the board's edges in heavy foreshortening, but ChArUco
# tolerates that down to ~75° off-normal (= ev=15° if board lies flat).
# Upper bound trimmed to 70° because elevations >80° essentially never
# IK-solve on the wrist-flip mount, and 75°-80° is a thin sliver that
# isn't worth the wireframe shell extending into.
_HEMI_ELEVATION_RANGE_DEG: tuple[float, float] = (15.0, 70.0)
# Distance lower bound 0.14 m: at fx=fy=615 the camera covers ~146 mm
# horizontally at this distance — narrower than the 210 mm board, so a
# straight-on overhead pose at d=d_min would crop the board. That's fine
# in practice because (a) low-elevation oblique views see the board
# foreshortened to a much smaller projected width, and (b) ChArUco only
# needs 4-6 corners detected, not the whole board, so partial-board
# views still calibrate. If you want the inner shell smaller still, you
# can drop to ~0.12 m before the high-elevation candidates start losing
# enough corners to fail detection. Going below 0.10 m starts depth-of-
# field issues on the D435i (min focus ~0.10 m) and is not recommended.
# 150° spread = 300° azimuth coverage. Pushes the hemisphere toward
# wrapping the board entirely (full 360° = spread 180°), giving the
# orchestrator more "from-behind-the-board" pose options. The
# back-of-hemisphere (between board and robot base) tends to be unreachable
# anyway — the arm has to fold over itself — but the workspace-hull filter
# silently rejects those, so widening here is safe.
_HEMI_AZIMUTH_SPREAD_DEG: float = 150.0

# Hemisphere centre override — when set, the hemisphere wireframe + reachability
# sampler + pose-generator hemisphere CENTRE all use this fixed world-frame
# point instead of the board centre. The look-at target (where cameras aim)
# stays the actual board centre regardless. This decouples the hemisphere's
# spatial anchor from the board pose, so changing _BOARD_RPY_RAD doesn't
# shift the hemisphere (the centre offset = R · (w/2, h/2, 0) depends on
# rotation, so without an override the hemisphere shifts ~10 cm when yaw
# changes).
#
# Set to e.g. (0.30, 0.0, 0.014) to pin the hemisphere forward of the robot
# at typical workable-space position. Set to None to use the board centre
# (legacy / sim-default behaviour).
_HEMI_CENTRE_OVERRIDE_M: tuple[float, float, float] | None = None

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
# Reachability sampling for the green-dot viz. Two modes:
#   _REACHABILITY_USE_CONTINUOUS = False  -> use _REACHABILITY_GRID above
#       (legacy discrete sampling, 7×6×20 = 840 candidates considered)
#   _REACHABILITY_USE_CONTINUOUS = True   -> Sobol low-discrepancy sampling
#       within the hemisphere ranges, _REACHABILITY_N_CANDIDATES total.
# Continuous gives provably uniform 3D coverage; discrete clumps at grid
# corners and tends to produce visible "rings" of survivors after IK
# filtering. Default to continuous; the discrete path is kept for
# bisecting if anything regresses.
_REACHABILITY_USE_CONTINUOUS: bool = True
_REACHABILITY_N_CANDIDATES: int = 1024

# Calibration hemisphere sampling — same Sobol vs discrete toggle as the
# reachability viz, applied to the actual calibration run. The bootstrap
# pass and main pass both feed through HullFilteredPoseGenerator, which
# inherits the base PoseGenerator's continuous-mode plumbing. Default to
# continuous so the calibration samples come from the same provably-
# uniform distribution the viz shows.
_CALIBRATION_USE_CONTINUOUS: bool = True
# Bootstrap pass — needs only enough survivors to seat consensus (~6-8
# detections); 512 Sobol candidates × ~15 % IK pass rate on tilt_x=180 ≈
# 75 reachable, plenty.
_BOOTSTRAP_N_CANDIDATES: int = 512
# Main pass — denser than bootstrap, used for the actual calibration
# samples; ~1024 Sobol × ~15 % ≈ 150 reachable, then thinned to
# target_sample_count by farthest-first and trajectory filters.
_MAIN_N_CANDIDATES: int = 1024
# Greedy farthest-first selection: thin the dense reachable set down to a
# spatially well-spread subset, so visualisation (and, optionally,
# calibration) gets points that are far enough apart instead of clustered
# along grid lines. None = no thinning, show every reachable point.
_REACHABILITY_KEEP_COUNT: int | None = None

# Multi-target relaxed look-at — DEFERRED_FEATURES.md §6.
#
# Each entry is a (u, v) pair in board-local UV coordinates ∈ [0, 1]² where
# (0.5, 0.5) is the geometric centre of the board. For every hemisphere
# camera position, the pose generator runs IK against EACH of these targets
# and accumulates all IK-feasible candidates. Multi-target raises bootstrap
# detection rate (single-centre observed at 25-50% on tilt_x=180; expected
# 60-80% with the 5-point grid below) at the cost of a 5× longer pose-
# generation pass (~5 s extra at startup).
#
# The default ((0.5, 0.5),) is single-centre and identical to the original
# behaviour. To enable multi-target, replace with e.g.:
#     ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5))
# which targets the four off-centre quadrants plus centre.
_BOARD_TARGET_OFFSETS_LOCAL: tuple[tuple[float, float], ...] = (
    (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.5),
)

# Multi-ray occlusion sampling — partial-frame robot-body occlusion.
# Single-ray (camera → board centre) misses cases where part of the FOV is
# blocked by a robot link but the centre line is clear. We cast rays to
# multiple sample points spread across the board face and reject the pose
# if more than _OCCLUSION_MAX_BLOCKED rays are blocked. Each (u, v) is in
# board-local UV ∈ [0, 1]². Default 9-point sampling: centre + 4 corners +
# 4 mid-edges. Denser sampling catches "robot link clipping one quadrant"
# cases that the 5-point grid let through.
_OCCLUSION_BOARD_SAMPLES_LOCAL: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),                                       # centre
    (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),   # corners
    (0.5, 0.0), (0.5, 1.0), (0.0, 0.5), (1.0, 0.5),   # mid-edges
)
# Reject if more than this many sampled rays are blocked. With 9 samples
# and threshold=0, ANY robot-link occlusion of the board area rejects the
# pose — strict but justified: ChArUco needs ≥6 visible corners for a
# reliable pose, and any robot-body occlusion of even one mid-edge usually
# means a corner is also occluded. Bump to 1 if you find this rejecting
# poses that look fine in the GUI; bump further only if hardware testing
# shows ChArUco coping with heavier partial-occlusion than expected.
_OCCLUSION_MAX_BLOCKED: int = 0

# Per-pose settle time. The orchestrator pauses for this long after each
# move_j completes before capturing a frame, letting vibration die out
# (mechanical, camera-mount flex, gripper sway) and the camera's auto-
# exposure settle. Sim mode can be aggressive (the controller's motion
# physics is already discrete-time so there's no real ringing); on real
# hardware PAROL6's belt-driven joints + the camera bracket flex want a
# couple of seconds to fully stabilise.
_SETTLE_TIME_SIM_S: float = 0.2
_SETTLE_TIME_REAL_S: float = 2.5

# Viewing-angle filter — reject candidates where the camera optical axis is
# more than this many degrees off the board surface normal. At extreme
# angles the board projection is heavily foreshortened, ChArUco corners are
# detected at low count + high reproj error, and the calibration's effective
# spatial resolution drops. With _HEMI_ELEVATION_RANGE_DEG = (15°, 70°),
# elevation 15° puts the camera at 75° off the board normal; we set the
# limit slightly above that (78°) so the lowest hemisphere elevations
# survive the filter, with multi-target look-at having a few degrees of
# headroom. Tune downward (more aggressive filtering) if low-angle poses
# look like they're hurting calibration accuracy.
_MAX_CAM_BOARD_ANGLE_DEG: float = 78.0

# Floor-collision primitive — add a flat collision box at z<0 to the
# CollisionManager. Any gripper / arm link that dips below the workbench
# surface is then caught by the same mesh-collision check that handles
# self-collision. More accurate than the TCP-point check (which can miss
# gripper-finger-clipping when the wrist is rolled). Keep True unless
# python-fcl breaks against thin box geometry.
_FLOOR_PRIMITIVE_ENABLED: bool = True

# Tablet collision primitive — the physical ChArUco display (Galaxy Tab S9
# Ultra: 208.6 × 326.4 × 5.5 mm bare; ~11 mm with the case) is added as a
# static collision box at the current _T_BOARD2BASE pose. Anything (gripper,
# camera bracket, arm link) clipping into the tablet body during a
# calibration move is rejected. Dimensions below include a small margin
# over the case-on tablet so brushing contacts get caught too. The box is
# centred on the ChArUco geometric centre with its top face at z=0 (the
# screen surface), extending in board-local -Z by the tablet thickness.
_TABLET_PRIMITIVE_ENABLED: bool = True
# Board-local axes: +X = ChArUco LONG edge (squares_x direction, 7×30mm = 210mm
# for the default Tab S9 Ultra board), +Y = ChArUco SHORT edge (squares_y, 150mm),
# +Z = out of board surface. Tablet in landscape mode → tablet's LONG edge
# (326.4mm) lines up with the board's +X axis, tablet's SHORT edge (208.6mm)
# along +Y. Thickness extends in +Z (above the bench, where the body sits).
_TABLET_DIMENSIONS_M: tuple[float, float, float] = (
    0.340,  # along board-local +X (ChArUco long edge / 7-square direction)
    0.215,  # along board-local +Y (ChArUco short edge / 5-square direction)
    0.014,  # along board-local +Z (thickness — Tab S9 Ultra with case)
)
# Offset of the tablet's geometric centre from the ChArUco geometric centre,
# in board-local (+X, +Y) coordinates. Default (0, 0) places the tablet
# centred on the printed ChArUco, which is right when the printed pattern
# fills the tablet screen. If the ChArUco is rendered with extra screen real
# estate on one side (e.g. the long Tab S9 Ultra has ~190 mm of "leftover"
# tablet length past the 150 mm ChArUco height), the tablet body extends
# further in that direction:
#   • Set Y > 0 (e.g. +0.090) if leftover tablet is on the +Y side of the ChArUco.
#   • Set Y < 0 (e.g. -0.090) if leftover tablet is on the -Y side.
# Wrong offsets only matter for collision filtering precision, not for
# detection — the calibration will still converge if the tablet model is
# slightly off. Set _SHOW_TABLET_OVERLAY=True to render the modelled box
# in the GUI as a translucent overlay so you can dial this in visually.
_TABLET_OFFSET_FROM_CHARUCO_LOCAL_M: tuple[float, float] = (0.0, 0.0)
# Render the tablet collision box as a translucent overlay in the GUI for
# visual debugging. The drawn box matches the collision primitive's pose
# and dimensions exactly, so once it lines up with where you've placed
# your physical tablet, the collision check is using the right model.
_SHOW_TABLET_OVERLAY: bool = True

# Auto-localise board (DEFERRED_FEATURES.md §3) — the workspace scan used
# by the "Localise Board" button. The button drives the robot through a
# coarse XY grid of scan target points, each pose looking down at one
# point from _LOCALISE_SCAN_DISTANCE_M above. The board can be ANYWHERE
# in the reachable workspace — some pose's FOV will overlap it and detect
# the ChArUco. Successful detections are median-filtered and used to
# update _T_BOARD2BASE. The board, hemisphere wireframe, and reachability
# dots all refresh to match the new pose.
#
# Scan targets are (x, y) world-frame points. Z is auto-computed from the
# configured board height (auto-lifted by tablet thickness). Default grid
# covers PAROL6's typical reachable workspace: 2 distances × 3 angles =
# 6 points covering radii 0.20-0.40 m and azimuths ±25°. Add or remove
# entries to change the search pattern.
# Default 9-cell grid — 3 X positions × 3 Y positions covering 0.20-0.40 m
# radii × ±0.15 m sideways. Add or remove entries to change the search
# pattern.
_LOCALISE_SCAN_TARGETS_M: tuple[tuple[float, float], ...] = (
    (0.20, -0.15), (0.20,  0.00), (0.20, +0.15),
    (0.30, -0.15), (0.30,  0.00), (0.30, +0.15),
    (0.40, -0.15), (0.40,  0.00), (0.40, +0.15),
)
# Distance from each scan target to the camera (camera sits this far above
# the target, looking down). 0.30 m gives ~33 cm × 25 cm FOV at the floor
# with the default fx=fy=615 intrinsics — wide enough that consecutive
# scan points overlap, so a board near the boundary between two grid
# points gets seen from both. Tuple form: try each in order until one
# yields a reachable pose.
_LOCALISE_SCAN_DISTANCES_M: tuple[float, ...] = (0.28, 0.32)
# Elevation angle for each scan pose. 90° = pure overhead (camera looks
# straight down); lower values tilt the camera forward. Pure overhead can
# cause IK failures with tilt_x=180 wrist-flip. The sweep covers 70° down
# to 45° — the pose generator tries each elevation in turn per scan target
# and accepts the first that yields a reachable, occlusion-free pose.
# Wider sweep than before because some workspace targets only have a
# reachable pose at the lower elevations.
_LOCALISE_SCAN_ELEVATIONS_DEG: tuple[float, ...] = (70.0, 60.0, 50.0, 45.0)
# Azimuth range for the per-target search. Wider than (-90, 90) because the
# robot's reach is asymmetric in azimuth (the cold-start mount's offset
# means some viewpoints work from the side but not the front). 4 azimuth
# samples within this range per (distance, elevation), so a single target
# gets up to 4 dist × 4 elev × 4 az = 64 candidates evaluated. Pinokin's
# IK is fast enough (~5 ms per call) that this is well under a second.
_LOCALISE_SCAN_AZIMUTH_RANGE_DEG: tuple[float, float] = (-100.0, 100.0)
_LOCALISE_SCAN_AZIMUTH_COUNT: int = 4

# J0-sweep localise mode (DEFAULT) — much simpler and more reliable than
# the multi-target scan above. Conceptually: drive the robot to ONE
# overhead pose, then sweep just the base joint (J0) while keeping J1-J5
# fixed. The camera gaze rotates in a circular arc around the base,
# scanning a band of the forward workspace in one continuous motion.
# Multiple SEED TARGETS at increasing distances let the camera scan
# different annular bands without any per-pose start/stop cycle.
# Set _LOCALISE_USE_J0_SWEEP=False to fall back to the multi-target scan.
_LOCALISE_USE_J0_SWEEP: bool = True
# Continuous-sweep mode: instead of stopping at each J0 step, drive the
# robot non-blocking through the full sweep range and capture frames
# every _LOCALISE_CAPTURE_PERIOD_S during motion. Successful detections
# are recorded with the flange pose at capture time. Much faster than
# discrete steps and lets ChArUco "find" the board mid-motion. Set False
# to fall back to discrete J0 steps with stop+capture at each.
_LOCALISE_CONTINUOUS_SWEEP: bool = True
# Seed-pose look-at targets, tried in order (closer first, then centre,
# then farther). The camera position emerges from the seed search but
# the LOOK-AT-TARGET determines which annular band of the workspace the
# camera covers during the J0 sweep. With the wrist-flip mount, the
# closer seed target also tends to put the camera's gaze sweeping a
# tighter circle around the base — covers more of the workspace per
# unit of J0 rotation.
_LOCALISE_SEED_TARGETS_XY: tuple[tuple[float, float], ...] = (
    (0.22, 0.0),  # closer to base — try first (catches workspaces "more inward")
    (0.30, 0.0),  # centre — original default
    (0.38, 0.0),  # farther — fallback (catches workspaces "more outward")
)
# Backwards-compat alias; consumers that wired _LOCALISE_SEED_TARGET_M
# directly still work.
_LOCALISE_SEED_TARGET_M: tuple[float, float] = _LOCALISE_SEED_TARGETS_XY[1]
# Continuous-sweep speed (fraction of max joint speed). 0.10 = 10% of
# max — gentle enough that 150 ms capture period covers <2° of J0 per
# frame, so consecutive frames at the same detection event give similar
# pose measurements that median-consensus well together.
_LOCALISE_SWEEP_SPEED: float = 0.10
# Time between captures during a continuous sweep (seconds). 150 ms →
# ~7 captures per second; with sweep speed 10% of max joint, that's
# roughly one capture every 1-2° of J0. Tune up (longer period) if the
# capture loop is overrunning, down (shorter) if frames are too sparse
# to catch the board mid-pass.
_LOCALISE_CAPTURE_PERIOD_S: float = 0.15
# Seed-pose search ranges. The seed is generated by Sobol low-discrepancy
# sampling within these ranges (distance × elevation × full 360° azimuth)
# — same continuous-sampling approach as the calibration hemisphere and
# reachability viz. Picks the highest-vertical-score IK-feasible
# candidate from N samples.
#
# Why FULL 360° azimuth: PAROL6's reachable cluster with the tilt_x=180
# mount sits at azimuth ±150°, NOT in the front hemisphere — a narrower
# search range misses every reachable seed. Empirically: same target,
# same (d, ev), 0 reachable in (−90°, 90°) vs ~6 reachable in
# (−180°, 180°).
_LOCALISE_SEED_DISTANCE_RANGE_M: tuple[float, float] = (0.22, 0.34)
_LOCALISE_SEED_ELEVATION_RANGE_DEG: tuple[float, float] = (60.0, 85.0)
_LOCALISE_SEED_N_CANDIDATES: int = 128
# J0 sweep range (relative to the seed pose's J0 angle, in degrees) and
# step count. ±90° covers the entire forward hemisphere; 13 steps at
# ~15° spacing gives a step smaller than the camera's horizontal FOV
# (~60° at 30 cm distance), so consecutive steps see overlapping
# regions and the board can't slip between cracks.
_LOCALISE_J0_SWEEP_HALF_DEG: float = 90.0
_LOCALISE_J0_STEPS: int = 13
# Minimum number of successful detections to accept the localise. With
# workspace scan, only the 1-3 scan poses whose FOV overlap the actual
# board succeed; lowering this to 1 lets us accept a single confident
# detection. Increase for more robustness against false positives.
_LOCALISE_MIN_DETECTIONS: int = 1
# RANSAC inlier threshold (metres) — detections within this distance of the
# inlier-set median count as agreeing on the board location. 1 cm is loose
# enough that the cold-start mount's ~2 mm / 2° error doesn't reject good
# detections, tight enough that an outlier 5 cm off doesn't fool us.
_LOCALISE_INLIER_THRESHOLD_M: float = 0.01


def _hemi_azimuth_center_deg() -> float:
    """Direction from the robot base origin to the hemisphere anchor, in
    degrees measured from world +X (CCW). Used so the hemisphere naturally
    faces "away from the robot" toward where the workable space (board)
    sits."""
    cx, cy, _ = _hemi_centre_world().tolist()
    return float(np.degrees(np.arctan2(cy, cx)))


def _hemi_azimuth_world_range_deg() -> tuple[float, float]:
    """World-frame azimuth range covering the hemisphere's spread, centered
    on the base→board direction. Used by both the viz and the orchestrator."""
    center = _hemi_azimuth_center_deg()
    return (center - _HEMI_AZIMUTH_SPREAD_DEG, center + _HEMI_AZIMUTH_SPREAD_DEG)


def _build_T_board2base() -> NDArray[np.float64]:
    """Compose the SE(3) board→base matrix from the position + RPY tunables.

    When ``_TABLET_PRIMITIVE_ENABLED`` is set, the board origin is
    auto-lifted along its local +Z by ``_TABLET_DIMENSIONS_M[2]``. This
    matches the physical reality of a tablet lying screen-up on a bench:
    the bench surface is at world z=0, the tablet body sits on the bench,
    and the ChArUco screen is at z = tablet_thickness above the bench.
    The user-facing ``_BOARD_TRANSLATE_M`` continues to describe where
    the board sits IF the tablet had zero thickness — ergonomic because
    it preserves the natural mental model "the board's at this XY".
    """
    R = np.eye(3, dtype=np.float64)
    if any(abs(a) > 1e-9 for a in _BOARD_RPY_RAD):
        R = SciRotation.from_euler("XYZ", _BOARD_RPY_RAD).as_matrix()

    translation = np.asarray(_BOARD_TRANSLATE_M, dtype=np.float64).copy()
    if _TABLET_PRIMITIVE_ENABLED:
        # Lift along board's +Z (out of the screen face) by tablet thickness.
        translation += R[:, 2] * _TABLET_DIMENSIONS_M[2]

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


_T_BOARD2BASE: NDArray[np.float64] = _build_T_board2base()


def _board_center_world() -> NDArray[np.float64]:
    """Compute the board's geometric center in world coordinates.

    The ChArUco config places the board's local origin at one corner and the
    board extends to (w_m, h_m, 0) in board-local coordinates. The
    pose-generator look-at target uses the board CENTRE (not the corner);
    aiming at the corner would make the pose generator aim cameras at one
    edge of the board, leaving most of the printed pattern outside FOV.
    """
    from parol6_vision.calibration.board import BOARD_TABLET_30MM as _cfg  # noqa: PLC0415
    w_m = _cfg.squares_x * _cfg.square_length
    h_m = _cfg.squares_y * _cfg.square_length
    center_local = np.array([w_m / 2.0, h_m / 2.0, 0.0, 1.0], dtype=np.float64)
    return (_T_BOARD2BASE @ center_local)[:3]


def _hemi_centre_world() -> NDArray[np.float64]:
    """Hemisphere anchor — `_HEMI_CENTRE_OVERRIDE_M` if set, board centre
    otherwise. This is the SPATIAL anchor of the hemisphere (where the dome
    of camera positions sits in world frame). Distinct from the look-at
    target, which is always the actual board centre regardless of override.
    """
    if _HEMI_CENTRE_OVERRIDE_M is not None:
        return np.asarray(_HEMI_CENTRE_OVERRIDE_M, dtype=np.float64)
    return _board_center_world()

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


# === Detection overlay (live perception viz) ===
# JSON snapshot written by parol6-vision's find_object.py. From this file at
# Waldo-Commander/waldo_commander/components/calibration_overlays.py, four
# parents up = "Project Files/" (the directory that contains both
# Waldo-Commander/ and parol6-vision/), then sibling parol6-vision/Results/...
_DETECTION_JSON_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "parol6-vision" / "Results" / "perception" / "last_detection.json"
)
_DETECTION_POLL_INTERVAL_S: float = 0.5
_DETECTION_OVERLAY_COLOR: str = "#ff8800"  # orange — distinct from board (textured) and frustum (cyan)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


_state: dict[str, Any] = {
    "merged_stl_path": None,  # set on first add_overlays call
    "board_png_path": None,
    "frustum_objects": [],
    "is_running": False,
    "is_localising": False,  # board-localise thread guard, separate from is_running
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
    # Scene-overlay handles for refresh-after-auto-localise.
    "scene_root": None,        # urdf_scene.scene root for re-attaching groups
    "board_group": None,       # ChArUco board overlay (deletable scene group)
    "hemisphere_group": None,  # hemisphere wireframe + reachability dots group
    "tablet_group": None,      # translucent tablet collision-box visual
    # Reachable hemisphere candidates cached by _add_reachability_points; the
    # board-localise thread reuses these as lookout joint configurations so we
    # don't have to re-run pose generation just to pick scan poses.
    "reachable_candidates": [],
    # Cached collision-manager pair for validate_joint_trajectory(); built
    # lazily on first call so users importing this module pay nothing.
    "trajectory_collision_mgr_pair": None,
    # Timestamp of the most recent successful Localise Board run. None
    # means never — the Run button shows a warning dialog in that case
    # because calibration would aim at the configured _BOARD_TRANSLATE_M
    # rather than the actual board pose.
    "last_localise_ok_at": None,
    # True while the localise-before-Run confirmation dialog is open.
    # _busy_warn checks this so the user can't start a parallel Localise
    # while the dialog is awaiting their answer.
    "dialog_open": False,
    # Detection-overlay (live perception viz). The group handle is the
    # NiceGUI scene group containing the wireframe boxes + labels for the
    # most recent set of detections; we delete + rebuild it on each poll
    # tick where the JSON has changed. detection_last_mtime caches the
    # file mtime so we skip unchanged polls cheaply. detection_overlay_timer
    # is the ui.timer handle from add_overlays.
    "detection_overlay_group": None,  # NiceGUI group handle
    "detection_last_mtime": 0.0,       # for change detection
    "detection_overlay_timer": None,
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


def _build_collision_manager(
    tablet_T_board2base: NDArray[np.float64] | None = None,
) -> tuple[Any, set[tuple[str, str]]] | None:
    """Build a trimesh CollisionManager populated with PAROL6's link meshes
    + the merged SSG-48 gripper body (which has the camera bracket fused in)
    + optional FLOOR and TABLET static collision primitives.

    Args:
        tablet_T_board2base: Optional 4×4 board→base transform. When
            provided AND ``_TABLET_PRIMITIVE_ENABLED`` is True, a box of
            ``_TABLET_DIMENSIONS_M`` is placed at the ChArUco centre, with
            its top face on the board's z=0 surface and the body extending
            in board-local -Z. When None, no tablet primitive is added.

    Returns:
        (manager, adjacent_pairs) on success, None if python-fcl or any
        link mesh is missing. ``adjacent_pairs`` whitelists (link_a, link_b)
        pairs whose collisions should NOT count as self-collisions — joint
        neighbours that always touch, plus the FLOOR-vs-base / FLOOR-vs-
        TABLET background pairs.
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

    # FLOOR collision primitive — wide flat box at z ∈ [-0.05, 0]. Anything
    # dipping below z=0 collides with FLOOR. Catches gripper-finger and
    # camera-bracket clipping that the per-pose TCP-point check misses.
    floor_added = False
    if _FLOOR_PRIMITIVE_ENABLED:
        try:
            floor_box = trimesh.creation.box(extents=(10.0, 10.0, 0.05))
            floor_pose = np.eye(4, dtype=np.float64)
            floor_pose[2, 3] = -0.025  # box top at z=0
            mgr.add_object("FLOOR", floor_box, transform=floor_pose)
            floor_added = True
        except Exception as e:  # noqa: BLE001
            logger.warning("FLOOR primitive add failed: %s", e)

    # TABLET collision primitive — the physical ChArUco display, sized per
    # _TABLET_DIMENSIONS_M and placed at the supplied board pose. Centred on
    # (ChArUco centre + _TABLET_OFFSET_FROM_CHARUCO_LOCAL_M). Box extends in
    # board-local -Z by the tablet thickness (the body sits BELOW the screen).
    # The board pose is auto-lifted by tablet thickness in _build_T_board2base,
    # so in WORLD frame the box ends up between z=0 (bench) and z=+t_h (screen).
    tablet_added = False
    if _TABLET_PRIMITIVE_ENABLED and tablet_T_board2base is not None:
        try:
            from parol6_vision.calibration.board import BOARD_TABLET_30MM as _cfg  # noqa: PLC0415
            t_w, t_l, t_h = _TABLET_DIMENSIONS_M
            t_off_x, t_off_y = _TABLET_OFFSET_FROM_CHARUCO_LOCAL_M
            tablet_box = trimesh.creation.box(extents=(t_w, t_l, t_h))
            tablet_centre_local = np.array(
                [
                    _cfg.squares_x * _cfg.square_length / 2.0 + t_off_x,
                    _cfg.squares_y * _cfg.square_length / 2.0 + t_off_y,
                    -t_h / 2.0,  # box centre below the screen plane
                    1.0,
                ],
                dtype=np.float64,
            )
            T_b2b = np.asarray(tablet_T_board2base, dtype=np.float64)
            centre_world = T_b2b @ tablet_centre_local
            tablet_pose = np.eye(4, dtype=np.float64)
            tablet_pose[:3, :3] = T_b2b[:3, :3]
            tablet_pose[:3, 3] = centre_world[:3]
            mgr.add_object("TABLET", tablet_box, transform=tablet_pose)
            tablet_added = True
        except Exception as e:  # noqa: BLE001
            logger.warning("TABLET primitive add failed: %s", e)

    adjacent: set[tuple[str, str]] = {
        ("base_link", "L1"), ("L1", "L2"), ("L2", "L3"),
        ("L3", "L4"), ("L4", "L5"), ("L5", "L6"),
        ("L6", "gripper"),
    }
    # The robot base sits at z=0 by definition, so base_link / FLOOR "collide"
    # at the contact patch. The tablet sits ON the floor too (back of tablet
    # box clips the floor box).
    if floor_added:
        adjacent |= {("base_link", "FLOOR")}
        if tablet_added:
            adjacent |= {("FLOOR", "TABLET")}
    # Tablet collision is GRIPPER-ONLY by user request. The realistic danger
    # is the gripper (with its camera bracket) hitting the tablet during a
    # calibration move; arm-link-vs-tablet collisions are mostly false
    # positives caused by tablet model imprecision (asymmetric ChArUco
    # placement on the screen, case thickness uncertainty, etc.). Whitelist
    # every arm link vs TABLET so only (gripper, TABLET) fires as a real
    # rejection.
    if tablet_added:
        for _link in ("base_link", "L1", "L2", "L3", "L4", "L5", "L6"):
            adjacent |= {(_link, "TABLET")}
    # Add reverse pairs for symmetric lookup.
    adjacent |= {(b, a) for a, b in adjacent}
    logger.info(
        "self-collision manager loaded: 7 links + gripper%s%s",
        " + FLOOR" if floor_added else "",
        " + TABLET" if tablet_added else "",
    )
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


def validate_joint_trajectory(
    q_from: NDArray[np.float64] | list[float] | tuple[float, ...],
    q_to: NDArray[np.float64] | list[float] | tuple[float, ...],
    *,
    n_samples: int = 10,
    degrees: bool = True,
) -> dict[str, Any]:
    """Pre-validate a joint-space move for self-collision before dispatching it.

    This is the public face of the same self-collision machinery the
    calibration filter pipeline uses internally (filters 3 + 6). The intended
    use is to check any move you're about to send to the controller:

        result = validate_joint_trajectory(current_q_deg, target_q_deg)
        if not result["safe"]:
            print("Move would collide:", result["reason"])
            return
        client.move_j(target_q_deg, ...)

    Args:
        q_from: 6-vector start joint angles. Either degrees (default) or
            radians depending on ``degrees=``.
        q_to: 6-vector target joint angles.
        n_samples: Number of interior interpolation points to check between
            ``q_from`` and ``q_to``. The endpoints themselves are also checked.
            Default 10 matches the calibration-time pairwise-trajectory filter.
        degrees: If True, ``q_from`` / ``q_to`` are interpreted as degrees.
            If False, radians.

    Returns:
        dict with keys:
            ``safe`` (bool): True iff every checked sample (start, end, and
                ``n_samples`` interior) is collision-free.
            ``start_safe`` (bool): start config is collision-free.
            ``end_safe`` (bool): end config is collision-free.
            ``interior_safe`` (bool): every interior sample is collision-free.
            ``manager_ready`` (bool): False if python-fcl or the link meshes
                are unavailable — in that case nothing was actually checked
                and ``safe`` defaults to True (fail-open, with a warning logged).
            ``reason`` (str): empty on success, short explanation on failure.

    Mesh fidelity caveat: the simplified link STLs ship with parol6 and are
    designed for collision queries (~80k tris total), so the verdict is
    accurate to within a few millimetres of mesh outline. Approximate; not a
    substitute for soft-stop / current-limit hardware safeties.
    """
    q_from_arr = np.asarray(q_from, dtype=np.float64).reshape(-1)
    q_to_arr = np.asarray(q_to, dtype=np.float64).reshape(-1)
    if degrees:
        q_from_arr = np.deg2rad(q_from_arr)
        q_to_arr = np.deg2rad(q_to_arr)

    pair = _state.get("trajectory_collision_mgr_pair")
    if pair is None:
        # Include the tablet primitive at the current _T_BOARD2BASE so any
        # move that would clip the physical ChArUco display gets caught.
        # The cache is invalidated by _localise_board_thread after it
        # mutates _T_BOARD2BASE so subsequent calls see the new pose.
        pair = _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
        if pair is None:
            logger.warning(
                "validate_joint_trajectory: collision manager unavailable; "
                "returning safe=True (fail-open). Install python-fcl + "
                "ensure parol6 link meshes are present to enable real checks."
            )
            return {
                "safe": True,
                "start_safe": True,
                "end_safe": True,
                "interior_safe": True,
                "manager_ready": False,
                "reason": "collision-manager unavailable",
            }
        _state["trajectory_collision_mgr_pair"] = pair

    mgr, adjacent = pair
    start_safe = not _self_collides(mgr, adjacent, q_from_arr)
    end_safe = not _self_collides(mgr, adjacent, q_to_arr)
    interior_safe = not _trajectory_collides(
        mgr, adjacent, q_from_arr, q_to_arr, n_samples=n_samples,
    )
    safe = start_safe and end_safe and interior_safe
    if safe:
        reason = ""
    elif not start_safe:
        reason = "start config self-collides"
    elif not end_safe:
        reason = "end config self-collides"
    else:
        reason = "trajectory interior self-collides"
    return {
        "safe": safe,
        "start_safe": start_safe,
        "end_safe": end_safe,
        "interior_safe": interior_safe,
        "manager_ready": True,
        "reason": reason,
    }


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
    scene_root = urdf_scene.scene
    _state["scene_root"] = scene_root

    _state["board_group"] = _build_board_overlay_group(scene_root, png_url)
    _state["tablet_group"] = _build_tablet_overlay_group(scene_root)

    # ------------------------------------------------------------------
    # Hemisphere wireframe — shows the (distance × elevation × azimuth)
    # region around the board where the pose generator places camera
    # candidates. Drawn in WORLD frame because hemisphere_camera_position()
    # uses world Z-up regardless of board rotation. Anchor is the hemisphere
    # centre (override-aware via _HEMI_CENTRE_OVERRIDE_M; falls back to the
    # board centre when no override is set).
    # ------------------------------------------------------------------
    if _SHOW_HEMISPHERE_WIREFRAME:
        _add_hemisphere_wireframe(scene_root, _hemi_centre_world())

    # Detection-overlay polling — repaints AABB boxes from the perception
    # pipeline's last_detection.json snapshot when its mtime changes.
    _state["detection_overlay_timer"] = ui.timer(
        _DETECTION_POLL_INTERVAL_S, _poll_detection_json,
    )


def _build_board_overlay_group(scene_root: Any, png_url: str) -> Any:
    """Build the ChArUco board scene group at the current ``_T_BOARD2BASE``.

    Extracted from ``add_overlays`` so ``refresh_board_dependent_overlays``
    can rebuild the group after auto-localise mutates ``_T_BOARD2BASE``.
    Returns the group handle (call ``.delete()`` to remove).
    """
    from parol6_vision.calibration.board import BOARD_TABLET_30MM  # noqa: PLC0415

    cfg = BOARD_TABLET_30MM
    w_m = cfg.squares_x * cfg.square_length
    h_m = cfg.squares_y * cfg.square_length

    board_pos = _T_BOARD2BASE[:3, 3].copy()
    board_pos[2] += 0.001  # nudge above the floor grid to avoid z-fight
    board_group = scene_root.group().move(*board_pos.tolist())
    # NiceGUI's group.rotate(rx, ry, rz) wraps three.js Object3D.rotation,
    # which uses INTRINSIC XYZ (i.e. "xyz" in scipy convention — lowercase).
    # _BOARD_RPY_RAD is documented as scipy XYZ-extrinsic (uppercase) so
    # we round-trip through the rotation matrix and decompose with the
    # intrinsic convention here. For single-axis rotations (e.g. yaw-only)
    # both conventions give identical Euler angles; the difference only
    # shows up for tilted boards (multiple non-zero axes).
    rpy = SciRotation.from_matrix(_T_BOARD2BASE[:3, :3]).as_euler("xyz").tolist()
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

    logger.info("parol6-vision calibration overlay: board at %s", board_pos.tolist())
    return board_group


def _build_tablet_overlay_group(scene_root: Any) -> Any | None:
    """Build a translucent box overlay matching the TABLET collision primitive.

    Position, orientation, and dimensions match _build_collision_manager's
    tablet exactly, so what you see in the GUI is what's being collision-
    checked. Returns the group handle (``.delete()`` to remove), or None if
    rendering is disabled. Driven by _TABLET_PRIMITIVE_ENABLED + _SHOW_TABLET_OVERLAY.
    """
    if not (_TABLET_PRIMITIVE_ENABLED and _SHOW_TABLET_OVERLAY):
        return None

    from parol6_vision.calibration.board import BOARD_TABLET_30MM as _cfg  # noqa: PLC0415

    t_w, t_l, t_h = _TABLET_DIMENSIONS_M
    t_off_x, t_off_y = _TABLET_OFFSET_FROM_CHARUCO_LOCAL_M

    # Tablet centre in board-local frame, then push to world via _T_BOARD2BASE.
    # Board pose is auto-lifted by t_h in _build_T_board2base, so placing the
    # box centre at board-local z=-t_h/2 puts it between world z=0 (bench)
    # and world z=+t_h (screen). Matches the collision primitive exactly.
    tablet_centre_local = np.array(
        [
            _cfg.squares_x * _cfg.square_length / 2.0 + t_off_x,
            _cfg.squares_y * _cfg.square_length / 2.0 + t_off_y,
            -t_h / 2.0,
            1.0,
        ],
        dtype=np.float64,
    )
    centre_world = (_T_BOARD2BASE @ tablet_centre_local)[:3]
    # Lowercase "xyz" = scipy intrinsic, matches three.js Object3D.rotation
    # default. See note in _build_board_overlay_group for context.
    rpy = SciRotation.from_matrix(_T_BOARD2BASE[:3, :3]).as_euler("xyz").tolist()

    grp = scene_root.group().move(*centre_world.tolist()).with_name("calib:tablet")
    if any(abs(a) > 1e-6 for a in rpy):
        grp = grp.rotate(*rpy)
    with grp:
        # Translucent rusty-orange box — distinct from the gray board backing
        # so it's easy to tell where the modelled tablet body extends past
        # the printed ChArUco area.
        ui.scene.box(t_w, t_l, t_h).material("#cc7733", opacity=0.30)

    logger.info(
        "tablet overlay: centre=%s, dims=%s, offset=%s",
        np.round(centre_world, 3).tolist(),
        _TABLET_DIMENSIONS_M,
        _TABLET_OFFSET_FROM_CHARUCO_LOCAL_M,
    )
    return grp


def refresh_board_dependent_overlays() -> None:
    """Rebuild the board, hemisphere wireframe, reachability dots, and
    tablet visual.

    Call this after ``_T_BOARD2BASE`` is mutated (e.g. by auto-localise) so
    the visualisation reflects the new board pose. Frustum stays correct
    automatically (it's parented to ``tcp_anchor``).

    Safe to call from a background thread: it schedules the actual scene
    surgery on the asyncio loop captured during ``add_overlays``. Returns
    immediately; the redraw happens at the next event-loop tick.
    """
    scene_root = _state.get("scene_root")
    png_url = _state.get("png_url")
    loop = _state.get("main_loop")
    if scene_root is None or png_url is None:
        logger.warning("refresh_board_dependent_overlays: scene not initialised yet")
        return

    def _do_refresh() -> None:
        for key in ("board_group", "hemisphere_group", "tablet_group"):
            old = _state.get(key)
            if old is not None:
                try:
                    old.delete()
                except Exception:  # noqa: BLE001
                    pass
                _state[key] = None
        _state["board_group"] = _build_board_overlay_group(scene_root, png_url)
        _state["tablet_group"] = _build_tablet_overlay_group(scene_root)
        if _SHOW_HEMISPHERE_WIREFRAME:
            _add_hemisphere_wireframe(scene_root, _hemi_centre_world())

    if loop is None:
        # No event loop captured — caller is on the main thread.
        _do_refresh()
    else:
        try:
            loop.call_soon_threadsafe(_do_refresh)
        except RuntimeError as e:
            # Event loop is closed (page torn down mid-localise). Nothing
            # to refresh.
            logger.info(
                "refresh_board_dependent_overlays: loop unavailable (%s); "
                "skipping scene refresh",
                e,
            )


def _render_detection_overlay(detections_payload: dict[str, Any]) -> None:
    """Rebuild the perception-detection overlay group from a parsed JSON payload.

    Deletes the previous group (if any), then for each detection in the payload
    draws a wireframe AABB + a 2D text label at the box top centre. Skips
    rendering entirely when the payload's frame is not "base" (camera-frame
    detections don't belong in the base-frame URDF scene).

    Schedules the actual scene mutation on the asyncio loop captured during
    ``add_overlays``, mirroring ``refresh_board_dependent_overlays``.
    """
    scene_root = _state.get("scene_root")
    loop = _state.get("main_loop")
    if scene_root is None:
        return  # add_overlays hasn't run yet

    frame = detections_payload.get("frame")
    detections = detections_payload.get("detections") or []
    refinement = detections_payload.get("refinement")
    refining = bool(refinement and refinement.get("in_progress"))
    refine_done = (refinement or {}).get("frames_done")
    refine_target = (refinement or {}).get("frames_target")

    def _do_render() -> None:
        # Always tear down the previous group before deciding whether to
        # rebuild — that way a frame switch from "base" to "camera" still
        # clears stale boxes.
        old = _state.get("detection_overlay_group")
        if old is not None:
            try:
                old.delete()
            except Exception as e:  # noqa: BLE001 - NiceGUI raises various types on torn-down scenes
                logger.debug("detection overlay delete failed: %s", e)
            _state["detection_overlay_group"] = None

        if frame != "base":
            return  # only render base-frame detections in the URDF scene
        if not detections:
            return

        grp = scene_root.group().with_name("calib:detections")
        _state["detection_overlay_group"] = grp
        with grp:
            for det in detections:
                mins = det.get("aabb_mins_mm")
                maxs = det.get("aabb_maxs_mm")
                if mins is None or maxs is None or len(mins) != 3 or len(maxs) != 3:
                    continue
                mins_m = np.asarray(mins, dtype=np.float64) / 1000.0
                maxs_m = np.asarray(maxs, dtype=np.float64) / 1000.0
                centre_m = (mins_m + maxs_m) / 2.0
                size_m = maxs_m - mins_m
                # Guard against degenerate bboxes (zero or negative extent).
                if not np.all(size_m > 0):
                    continue

                confidence = float(det.get("confidence") or 0.0)
                opacity = float(np.clip(0.3 + 0.7 * confidence, 0.0, 1.0))

                # Wireframe AABB. NiceGUI's Box has wireframe=True support
                # (Jepson2k fork), which renders the 12 edges as line segments.
                ui.scene.box(
                    width=float(size_m[0]),
                    height=float(size_m[1]),
                    depth=float(size_m[2]),
                    wireframe=True,
                ).move(*centre_m.tolist()).material(
                    _DETECTION_OVERLAY_COLOR, opacity=opacity
                )

                label = str(det.get("label") or f"obj{det.get('index', '?')}")
                if len(label) > 32:
                    label = label[:29] + "..."
                pct = int(round(confidence * 100))
                text_lines = [f"{label} ({pct}%)"]
                if refining and refine_done is not None and refine_target is not None:
                    text_lines.insert(0, f"[refining {refine_done}/{refine_target}]")
                # Text element always faces the camera; place it slightly
                # above the box top face so it doesn't z-fight the wireframe.
                text_pos = (
                    float(centre_m[0]),
                    float(centre_m[1]),
                    float(centre_m[2] + size_m[2] / 2.0 + 0.02),
                )
                ui.scene.text(
                    " ".join(text_lines),
                    style=f"color: {_DETECTION_OVERLAY_COLOR}; font-size: 12px;",
                ).move(*text_pos)

    if loop is None:
        # No event loop captured — caller is on the main thread.
        _do_render()
    else:
        try:
            loop.call_soon_threadsafe(_do_render)
        except RuntimeError as e:
            logger.debug(
                "_render_detection_overlay: loop unavailable (%s); skipping",
                e,
            )


def _poll_detection_json() -> None:
    """Timer tick: re-read the detection JSON if its mtime has changed.

    Designed to be cheap on the common case where the file is missing
    (perception not running) or unchanged since the last tick. Logs at
    DEBUG level on any error so we don't spam the log when no perception
    pipeline has run yet.
    """
    path = _DETECTION_JSON_PATH
    try:
        if not path.exists():
            return
        mtime = path.stat().st_mtime
    except OSError as e:
        logger.debug("_poll_detection_json: stat failed: %s", e)
        return

    if mtime == _state.get("detection_last_mtime"):
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug("_poll_detection_json: read/parse failed: %s", e)
        return

    _state["detection_last_mtime"] = mtime
    try:
        _render_detection_overlay(payload)
    except Exception as e:  # noqa: BLE001
        logger.debug("_poll_detection_json: render failed: %s", e)


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
    _state["hemisphere_group"] = grp

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

    d_min, d_max = _HEMI_DISTANCE_RANGE_M
    ev_min, ev_max = _HEMI_ELEVATION_RANGE_DEG
    az_min, az_max = _hemi_azimuth_world_range_deg()

    try:
        robot = Robot()
    except Exception as e:  # noqa: BLE001
        logger.warning("could not instantiate Robot for reachability viz: %s", e)
        return

    if _REACHABILITY_USE_CONTINUOUS:
        # Sobol low-discrepancy sampling — provably uniform 3D coverage of
        # the hemisphere volume. Discrete-grid sampling produces visible
        # "ring" artefacts in the surviving set when IK feasibility
        # correlates with grid axes (which it does on PAROL6 with
        # tilt_x=180 — survivors cluster along specific azimuth bands).
        params = HemisphereParams(
            n_candidates=_REACHABILITY_N_CANDIDATES,
            distance_range_m=(d_min, d_max),
            elevation_range_deg=(ev_min, ev_max),
            azimuth_range_deg=(az_min, az_max),
            workspace_xy_max_m=0.55,
            max_joint_change_deg=180.0,
        )
        max_count = _REACHABILITY_N_CANDIDATES
    else:
        n_d, n_ev, n_az = _REACHABILITY_GRID
        distances = tuple(np.linspace(d_min, d_max, n_d).tolist())
        elevations = tuple(np.linspace(ev_min, ev_max, n_ev).tolist())
        azimuth_counts = tuple([n_az] * n_ev)
        params = HemisphereParams(
            distances_m=distances,
            elevations_deg=elevations,
            azimuth_counts=azimuth_counts,
            azimuth_range_deg=(az_min, az_max),
            workspace_xy_max_m=0.55,
            max_joint_change_deg=180.0,
        )
        max_count = n_d * n_ev * n_az

    gen = PoseGenerator(
        robot=robot, mount=cold_start, target_world=target_world, params=params,
    )
    cands, stats = gen.generate(max_count=max_count)

    # Extract the camera position implied by each accepted candidate's flange
    # pose (T_cam2base = T_flange2base @ T_cam2flange). Track the candidate
    # alongside its camera position so the board-localise sweep can reuse the
    # joint angles directly (skips a redundant pose-generation pass).
    reachable_cam_world: list[NDArray[np.float64]] = []
    reachable_candidates: list = []
    for c in cands:
        T_cam2base = cold_start.cam_pose_for_flange_pose(np.asarray(c.flange_pose))
        cam_pos = T_cam2base[:3, 3]
        # Optional final hull check on flange position (the PoseGenerator's
        # workspace_xy/z bounds are rectangular; the hull is more accurate).
        if not bool(envelope_contains(np.asarray(c.flange_pose)[:3, 3])[0]):
            continue
        reachable_cam_world.append(cam_pos)
        reachable_candidates.append(c)

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

    # Sanity check: every surviving dot's distance from target_world should
    # lie inside [d_min, d_max], the same range the wireframe is drawn at.
    # If any survivors are outside the shell, the dots will visually appear
    # "above" or "below" the dome — surface that mismatch loudly so we can
    # debug rather than silently rendering inconsistent geometry.
    if reachable_cam_world:
        cam_arr = np.asarray(reachable_cam_world, dtype=np.float64)
        rel = cam_arr - target_world
        dists = np.linalg.norm(rel, axis=1)
        elevs_deg = np.degrees(np.arcsin(np.clip(rel[:, 2] / np.maximum(dists, 1e-9), -1.0, 1.0)))
        out_of_shell = ((dists < d_min - 1e-3) | (dists > d_max + 1e-3)).sum()
        logger.info(
            "reachability dots distance range %.3f - %.3f m (shell %.3f - %.3f), "
            "elevation range %.1f - %.1f° (shell %.1f - %.1f); out-of-shell: %d",
            float(dists.min()), float(dists.max()), d_min, d_max,
            float(elevs_deg.min()), float(elevs_deg.max()), ev_min, ev_max,
            int(out_of_shell),
        )
        if out_of_shell > 0:
            logger.warning(
                "%d reachability dot(s) lie OUTSIDE the wireframe shell — "
                "this means the dots and wireframe disagree about the "
                "hemisphere. Likely cause: target_world drifted between "
                "wireframe and dot rendering, or the cold-start mount's "
                "T_cam2flange differs from the one used by the pose "
                "generator. Tight numerical mismatches (sub-mm) are "
                "expected from FK-verify tolerance.",
                int(out_of_shell),
            )

    # Greedy farthest-first thinning so the rendered points form a uniform
    # spread across the reachable region instead of clustered along grid
    # lines. Approximates Poisson-disc sampling without an explicit minimum-
    # distance threshold — instead we pick a target count and let the
    # algorithm maximise minimum pairwise distance for that count. Run on the
    # candidate list (key=cam_pos) so the surviving CANDIDATES stay in lockstep
    # with the surviving points.
    selected_candidates = reachable_candidates
    if (
        _REACHABILITY_KEEP_COUNT is not None
        and len(reachable_candidates) > _REACHABILITY_KEEP_COUNT
    ):
        # Build (point, candidate) tuples for thinning, then unzip.
        paired = list(zip(reachable_cam_world, reachable_candidates))
        paired = _greedy_farthest_first(
            paired, _REACHABILITY_KEEP_COUNT, key=lambda p: p[0],
        )
        points_to_render = [p[0] for p in paired]
        selected_candidates = [p[1] for p in paired]
        logger.info(
            "farthest-first thinning: %d -> %d points",
            len(reachable_cam_world), len(points_to_render),
        )
    else:
        points_to_render = reachable_cam_world

    # Cache the selected candidates so the board-localise thread can reuse
    # their joint angles as scan poses.
    _state["reachable_candidates"] = selected_candidates

    # Render points as small green spheres, in scene_group's local frame
    # (which is already translated to target_world). Smaller radius
    # (3 mm instead of 6 mm) so dense reachable regions don't fuse into
    # a single blob; semi-transparent so overlapping dots visibly stack.
    with scene_group:
        for cam_pos in points_to_render:
            local = (cam_pos - target_world).tolist()
            ui.scene.sphere(0.003).move(*local).material("#33dd66", opacity=0.7)


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

        # Mode dispatch: piggyback off Waldo-Commander's existing real/sim
        # toggle (the orange-when-sim "robot" button at the top of the page).
        # robot_state.simulator_active=True means the user is in simulator
        # mode → use VirtualCamera; False means real-hardware mode → use
        # RealSenseCamera. Default to sim if the import fails (defensive).
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
            is_sim_mode = bool(robot_state.simulator_active)
        except Exception:  # noqa: BLE001
            is_sim_mode = True

        T_BOARD2BASE = _T_BOARD2BASE
        if is_sim_mode:
            # Sim ground truth: cold-start tunable + small fixed perturbation
            # (±2 mm / ±2°) so the simulated calibration always has something
            # realistic to converge to.
            ground_truth_mount = CameraMount.from_eyeball_estimate(
                x_mm=_CAM_MOUNT_TRANSLATE_MM[0] + 2.0,
                y_mm=_CAM_MOUNT_TRANSLATE_MM[1] + 2.0,
                z_mm=_CAM_MOUNT_TRANSLATE_MM[2] - 2.0,
                tilt_x_deg=_CAM_MOUNT_TILT_DEG[0] + 2.0,
                tilt_y_deg=_CAM_MOUNT_TILT_DEG[1] - 1.0,
                tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
            )
            intrinsics = Intrinsics(
                fx=_INTR_FX, fy=_INTR_FY, cx=_INTR_CX, cy=_INTR_CY,
                width=_INTR_W, height=_INTR_H,
                dist_coeffs=np.zeros(5, dtype=np.float64),
            )
        else:
            ground_truth_mount = None  # not used in real mode
            intrinsics = None  # set below from the actual RealSense device

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

        if is_sim_mode:
            camera: Any = VirtualCamera(
                intrinsics=intrinsics,
                image_width=_INTR_W,
                image_height=_INTR_H,
                ground_truth_mount=ground_truth_mount,
                flange_pose_provider=flange_pose,
                board=VirtualBoard(config=BOARD_TABLET_30MM, T_board2base=T_BOARD2BASE),
                noise_std=0.0,
            )
            logger.info("calibration camera: VirtualCamera (simulator mode)")
        else:
            from parol6_vision.camera.realsense import RealSenseCamera  # noqa: PLC0415
            camera = RealSenseCamera(
                width=_INTR_W,
                height=_INTR_H,
                fps=30,
                enable_depth=False,  # calibration only needs color frames
                enable_color=True,
            )
            # Stash BEFORE start(): if start() raises mid-pipeline-init the
            # device may have already been claimed; the finally-block must
            # see the camera handle to call .stop() cleanly.
            _state["real_camera"] = camera
            camera.start()
            intrinsics = camera.intrinsics
            logger.info(
                "calibration camera: RealSenseCamera (real-hardware mode), "
                "intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f, dist=%s",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                intrinsics.dist_coeffs.tolist(),
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

        # Hemisphere CENTRE — also the calibration's look-at target. Uses the
        # hemisphere override if set (decouples from board rotation), else
        # falls back to the board centre. With no override (sim default), this
        # is the board's geometric centre as before.
        target_world = _hemi_centre_world()

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
        # + gripper for collision; 7 link meshes for occlusion). The tablet
        # primitive is placed at the current _T_BOARD2BASE so the robot won't
        # drive into the physical ChArUco display.
        collision_mgr_pair = (
            _build_collision_manager(tablet_T_board2base=_T_BOARD2BASE)
            if _ENABLE_SELF_COLLISION_CHECK else None
        )
        occlusion_meshes = (
            _build_occlusion_mesh() if _ENABLE_SELF_COLLISION_CHECK else None
        )

        class HullFilteredPoseGenerator(PoseGenerator):
            def generate(self, *args, **kwargs):  # type: ignore[override]
                max_count = kwargs.get("max_count", args[0] if args else 8)
                # Over-request so we have a pool to thin from.
                kwargs["max_count"] = max_count * 5

                # Multi-target relaxed look-at — DEFERRED_FEATURES.md §6.
                # Build the per-pass aim points in board-local UV ∈ [0, 1]²,
                # convert to world frame, pass to super().generate() as
                # `look_at_targets`. The hemisphere CENTRE stays anchored on
                # self.target_world (the value of _hemi_centre_world() at
                # construction time — possibly the override); only the
                # look-at aim varies per pass.
                #
                # When _HEMI_CENTRE_OVERRIDE_M is set, self.target_world is
                # the override (a fixed workable-space anchor). The aim
                # points STILL come from _T_BOARD2BASE — i.e. cameras aim
                # at the actual board, not the override. Single-target mode
                # uses the centre offset (0.5, 0.5) explicitly so the same
                # invariant holds with or without multi-target.
                cfg = BOARD_TABLET_30MM
                w_m_b = cfg.squares_x * cfg.square_length
                h_m_b = cfg.squares_y * cfg.square_length
                offsets = (
                    _BOARD_TARGET_OFFSETS_LOCAL
                    if len(_BOARD_TARGET_OFFSETS_LOCAL) > 1
                    else ((0.5, 0.5),)
                )
                aim_targets = [
                    (
                        _T_BOARD2BASE
                        @ np.array(
                            [u * w_m_b, v * h_m_b, 0.0, 1.0],
                            dtype=np.float64,
                        )
                    )[:3]
                    for u, v in offsets
                ]
                cands, stats = super().generate(  # type: ignore[arg-type]
                    **kwargs, look_at_targets=aim_targets,
                )
                if len(offsets) > 1:
                    logger.info(
                        "multi-target look-at: %d targets, %d raw candidates "
                        "(pre-filter)",
                        len(offsets), len(cands),
                    )

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

                # 4. Camera-occlusion filter — multi-ray sampling across the
                # board face. Cast a ray from the camera position to each of
                # _OCCLUSION_BOARD_SAMPLES_LOCAL (default centre + 4 corners,
                # in board-local UV ∈ [0, 1]²). If MORE than
                # _OCCLUSION_MAX_BLOCKED rays hit a robot link before reaching
                # the target, reject the pose. Catches partial-frame occlusion
                # (e.g. arm body blocks the side of the FOV but not the centre).
                if occlusion_meshes is not None:
                    cold_T = self.mount.T_cam2flange
                    cfg = BOARD_TABLET_30MM
                    w_m_b = cfg.squares_x * cfg.square_length
                    h_m_b = cfg.squares_y * cfg.square_length
                    sample_world = [
                        (
                            _T_BOARD2BASE
                            @ np.array([u * w_m_b, v * h_m_b, 0.0, 1.0],
                                       dtype=np.float64)
                        )[:3]
                        for u, v in _OCCLUSION_BOARD_SAMPLES_LOCAL
                    ]
                    pre = len(cands)
                    survivors = []
                    for c in cands:
                        T_flange2base = np.asarray(c.flange_pose, dtype=np.float64)
                        T_cam2base = T_flange2base @ cold_T
                        cam_pos_world = T_cam2base[:3, 3]
                        n_blocked = _camera_occlusion_count(
                            occlusion_meshes, c.joint_angles_rad,
                            cam_pos_world, sample_world,
                            early_termination_after=_OCCLUSION_MAX_BLOCKED,
                        )
                        if n_blocked <= _OCCLUSION_MAX_BLOCKED:
                            survivors.append(c)
                    cands = survivors
                    if pre - len(cands) > 0:
                        logger.info(
                            "occlusion filter: %d/%d candidates rejected "
                            "(>%d/%d sample rays blocked by robot body)",
                            pre - len(cands), pre,
                            _OCCLUSION_MAX_BLOCKED,
                            len(_OCCLUSION_BOARD_SAMPLES_LOCAL),
                        )

                # 4.5. Viewing-angle filter — reject candidates whose camera
                # optical axis is more than _MAX_CAM_BOARD_ANGLE_DEG off the
                # board surface normal. At extreme angles the board projects
                # as a thin sliver, ChArUco corner detection works but reproj
                # error is high and the calibration's effective resolution
                # drops. Threshold defaults to 65° — admits the bottom of the
                # configured 25°-elevation hemisphere, rejects anything worse.
                if _MAX_CAM_BOARD_ANGLE_DEG < 90.0:
                    cold_T = self.mount.T_cam2flange
                    board_z_world = _T_BOARD2BASE[:3, 2]
                    cos_threshold = float(
                        np.cos(np.radians(_MAX_CAM_BOARD_ANGLE_DEG))
                    )
                    pre = len(cands)
                    survivors = []
                    for c in cands:
                        T_flange2base = np.asarray(c.flange_pose, dtype=np.float64)
                        T_cam2base = T_flange2base @ cold_T
                        cam_z_world = T_cam2base[:3, 2]
                        # Use abs() — board normal could point either way
                        # depending on board placement; we care about the
                        # angle, not the orientation.
                        if abs(float(cam_z_world @ board_z_world)) >= cos_threshold:
                            survivors.append(c)
                    cands = survivors
                    if pre - len(cands) > 0:
                        logger.info(
                            "viewing-angle filter: %d/%d rejected "
                            "(camera axis > %.0f° off board normal)",
                            pre - len(cands), pre,
                            _MAX_CAM_BOARD_ANGLE_DEG,
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

        # NB: prior to this commit we monkey-patched
        # ``parol6_vision.calibration.orchestrator.PoseGenerator`` here so
        # the orchestrator's main hemisphere pass would use our filtered
        # subclass. The orchestrator now accepts a ``pose_generator_factory``
        # in its config, so the subclass is injected explicitly via
        # ``OrchestratorConfig(pose_generator_factory=...)`` further down.

        # NB: prior to this commit we monkey-patched
        # ``parol6_vision.calibration.refinement.board_position_from_sample``
        # AND ``parol6_vision.calibration.pose_generator.look_at_pose`` here.
        # Both are now upstreamed — refinement takes a ``board: BoardConfig``
        # kwarg, and the pose generator's ``generate()`` takes
        # ``look_at_targets=[...]`` to drive multi-target relaxed look-at.
        # Zero monkey-patches remain in this thread.

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
        if _CALIBRATION_USE_CONTINUOUS:
            # Sobol low-discrepancy sampling — same volume as the discrete
            # grids below, but provably uniform 3D coverage. Avoids the
            # "ring" artefacts where discrete grids put grid corners at
            # specific azimuths and IK feasibility correlates with those
            # corners; survivors come from genuine reachability rather
            # than grid alignment. Bootstrap's elevation range is bumped
            # up (45° lower bound, 88° upper) to match the dense bootstrap
            # config's intent of capturing mostly-overhead views.
            bootstrap_params = HemisphereParams(
                n_candidates=_BOOTSTRAP_N_CANDIDATES,
                distance_range_m=(0.22, 0.34),
                elevation_range_deg=(30.0, 88.0),
                azimuth_range_deg=az_world_range,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
            main_params = HemisphereParams(
                n_candidates=_MAIN_N_CANDIDATES,
                distance_range_m=(d_min, d_max),
                elevation_range_deg=(ev_min, ev_max),
                azimuth_range_deg=az_world_range,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
        else:
            # Legacy discrete grids — kept for bisection if continuous
            # ever regresses. Both grids are dense because tilt_x=180
            # forces wrist-flip configurations and only ~15% of grid
            # points pass IK.
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
                azimuth_counts=(20, 16, 12, 9, 6, 4),  # 67 az per distance × 6 = 402
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

        # Pose-generator factory: injects our HullFilteredPoseGenerator
        # subclass into the orchestrator's `_collect_hemisphere` so the same
        # six-stage filter pipeline (hull, floor, self-collision, occlusion,
        # farthest-first, trajectory) runs in both the bootstrap and main
        # passes. Replaces the previous monkey-patch on
        # `orchestrator.PoseGenerator`.
        def _hull_filtered_factory(robot, mount, target_world, params):
            return HullFilteredPoseGenerator(
                robot=robot,
                mount=mount,
                target_world=target_world,
                params=params,
            )

        # Sample count trades calibration quality for wallclock time; 12 is
        # enough for a watchable demo. Settle time is short in sim (the
        # controller's motion physics is already discrete) but bumped on
        # real hardware so PAROL6's belt-driven joints + camera bracket
        # flex have time to fully stabilise before each capture.
        settle_s = _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S
        config = OrchestratorConfig(
            bootstrap_joint_configs_deg=boot_cfg,
            target_sample_count=12,
            min_sample_count=6,
            enable_second_pass=False,
            settle_time_s=settle_s,
            hemisphere=main_params,
            pose_generator_factory=_hull_filtered_factory,
        )

        orchestrator = CalibrationOrchestrator(
            robot=robot, client=client,
            camera=camera, detector=detector,
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
        # Stop the RealSenseCamera if we started one in real-hardware mode.
        real_cam = _state.get("real_camera")
        if real_cam is not None:
            try:
                real_cam.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("RealSenseCamera stop failed: %s", e)
            _state["real_camera"] = None
        _state["is_running"] = False


def _localise_board_thread() -> None:
    """Workspace scan that auto-locates the ChArUco board's centre in base frame.

    DEFERRED_FEATURES.md §3 (board auto-localisation). Drives the robot
    through a coarse XY grid of workspace points (`_LOCALISE_SCAN_TARGETS_M`),
    each pose looking down at one point from `_LOCALISE_SCAN_DISTANCE_M`
    above, captures a frame at each, runs ChArUco detection. Successful
    detections (typically 1-2 of N scan poses, depending on where the board
    actually sits) are median-filtered and used to update `_T_BOARD2BASE` in
    place (translation only — rotation stays at the configured
    `_BOARD_RPY_RAD`). The board, hemisphere wireframe, and reachability
    dots all refresh to match.

    The board can be ANYWHERE in the robot's reachable workspace — the
    scan grid is defined relative to the robot base, NOT to the configured
    board location, so it works whether or not the user's existing
    `_BOARD_TRANSLATE_M` matches reality.

    This is a sim-mode runner — uses `VirtualCamera` with the same
    perturbation scheme as `_calibration_thread`. On hardware, swap to
    `RealSenseCamera`.
    """
    try:
        import cv2  # noqa: PLC0415

        from parol6 import Robot, RobotClient  # noqa: PLC0415

        from parol6_vision.calibration.board import (  # noqa: PLC0415
            BOARD_TABLET_30MM,
            BoardDetector,
            board_pose_to_matrix,
        )
        from parol6_vision.calibration.camera_mount import CameraMount  # noqa: PLC0415
        from parol6_vision.calibration.pose_generator import (  # noqa: PLC0415
            HemisphereParams,
            PoseGenerator,
        )
        from parol6_vision.camera.intrinsics import Intrinsics  # noqa: PLC0415
        from parol6_vision.sim.tracing_client import (  # noqa: PLC0415
            _flange_pose_from_client,
        )
        from parol6_vision.sim.virtual_camera import (  # noqa: PLC0415
            VirtualBoard,
            VirtualCamera,
        )

        # Build the workspace scan poses fresh each time the button is
        # pressed. Each scan target gets a single overhead pose generated
        # by the standard PoseGenerator (one distance, one elevation, one
        # azimuth — the azimuth is irrelevant at near-overhead elevations).
        # The board can be ANYWHERE in the workspace; we don't depend on
        # the user's `_BOARD_TRANSLATE_M` being accurate.
        scan_z = _BOARD_TRANSLATE_M[2]
        if _TABLET_PRIMITIVE_ENABLED:
            scan_z += _TABLET_DIMENSIONS_M[2]

        cold_start = CameraMount.from_eyeball_estimate(
            x_mm=_CAM_MOUNT_TRANSLATE_MM[0],
            y_mm=_CAM_MOUNT_TRANSLATE_MM[1],
            z_mm=_CAM_MOUNT_TRANSLATE_MM[2],
            tilt_x_deg=_CAM_MOUNT_TILT_DEG[0],
            tilt_y_deg=_CAM_MOUNT_TILT_DEG[1],
            tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
        )
        scan_robot = Robot()
        candidates: list = []

        def _vertical_score_for(c: Any, target_w: NDArray[np.float64]) -> float:
            """cos(angle from straight-down) of the camera-to-target ray.
            1.0 = perfectly overhead, 0.0 = horizontal."""
            cam_pos = cold_start.cam_pose_for_flange_pose(
                np.asarray(c.flange_pose),
            )[:3, 3]
            forward = target_w - cam_pos
            fn = float(np.linalg.norm(forward))
            if fn < 1e-6:
                return 0.0
            return float(-(forward / fn)[2])

        # Seed-pose list: one entry per (seed_target_xy, seed_q_rad) where
        # the pose generator found a reachable look-down candidate. Used by
        # both continuous-sweep and discrete-J0-sweep branches below; the
        # multi-target legacy path bypasses this entirely.
        seed_q_list: list[tuple[tuple[float, float], NDArray[np.float64]]] = []
        if _LOCALISE_USE_J0_SWEEP:
            for seed_xy in _LOCALISE_SEED_TARGETS_XY:
                seed_target = np.array(
                    [seed_xy[0], seed_xy[1], scan_z], dtype=np.float64,
                )
                seed_params = HemisphereParams(
                    n_candidates=_LOCALISE_SEED_N_CANDIDATES,
                    distance_range_m=_LOCALISE_SEED_DISTANCE_RANGE_M,
                    elevation_range_deg=_LOCALISE_SEED_ELEVATION_RANGE_DEG,
                    azimuth_range_deg=(-180.0, 180.0),
                    workspace_xy_max_m=0.55,
                    max_joint_change_deg=180.0,
                )
                seed_gen = PoseGenerator(
                    robot=scan_robot, mount=cold_start,
                    target_world=seed_target, params=seed_params,
                )
                try:
                    seed_cands, seed_stats = seed_gen.generate(max_count=None)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "localise seed-gen at (%.2f, %.2f) raised %s: %s",
                        seed_xy[0], seed_xy[1], type(e).__name__, e,
                    )
                    continue
                if not seed_cands:
                    rej = dict(getattr(seed_stats, "rejection_log", {}) or {})
                    logger.info(
                        "localise seed (%.2f, %.2f): 0/%d reachable, rej=%s",
                        seed_xy[0], seed_xy[1], _LOCALISE_SEED_N_CANDIDATES, rej,
                    )
                    continue
                seed_cands_sorted = sorted(
                    seed_cands,
                    key=lambda c, t=seed_target: _vertical_score_for(c, t),
                    reverse=True,
                )
                seed_q_rad = np.asarray(
                    seed_cands_sorted[0].joint_angles_rad, dtype=np.float64,
                )
                seed_q_list.append((seed_xy, seed_q_rad))
                logger.info(
                    "localise seed (%.2f, %.2f): %d/%d reachable, vertical-score=%.3f, "
                    "joints (deg)=%s",
                    seed_xy[0], seed_xy[1],
                    len(seed_cands), _LOCALISE_SEED_N_CANDIDATES,
                    _vertical_score_for(seed_cands_sorted[0], seed_target),
                    ["%.1f" % v for v in np.degrees(seed_q_rad)],
                )
            if not seed_q_list:
                _post_status(
                    "Localise: no reachable seed pose at any of "
                    f"{len(_LOCALISE_SEED_TARGETS_XY)} configured targets "
                    f"({_LOCALISE_SEED_TARGETS_XY}). Try widening "
                    "_LOCALISE_SEED_DISTANCE_RANGE_M / "
                    "_LOCALISE_SEED_ELEVATION_RANGE_DEG."
                )
                return
            _localise_skip_multitarget = True
        else:
            _localise_skip_multitarget = False

        # Per-target candidate generation with a "look-AT-target" preference:
        # given multiple IK-feasible candidates, pick the one whose camera
        # position is geometrically closest to "directly above the target".
        # That means the camera's optical axis stays nearly vertical and the
        # robot body is on the OPPOSITE side from the workspace — minimising
        # both occlusion and oblique-view ChArUco failures.
        if _localise_skip_multitarget:
            multitarget_iter: tuple = ()
        else:
            multitarget_iter = _LOCALISE_SCAN_TARGETS_M
        for tx, ty in multitarget_iter:
            target_world = np.array([tx, ty, scan_z], dtype=np.float64)
            params = HemisphereParams(
                distances_m=_LOCALISE_SCAN_DISTANCES_M,
                elevations_deg=_LOCALISE_SCAN_ELEVATIONS_DEG,
                azimuth_counts=tuple(
                    _LOCALISE_SCAN_AZIMUTH_COUNT
                    for _ in _LOCALISE_SCAN_ELEVATIONS_DEG
                ),
                azimuth_range_deg=_LOCALISE_SCAN_AZIMUTH_RANGE_DEG,
                workspace_xy_max_m=0.55,
                max_joint_change_deg=180.0,
            )
            gen = PoseGenerator(
                robot=scan_robot,
                mount=cold_start,
                target_world=target_world,
                params=params,
            )
            try:
                # Get ALL feasible candidates, not just the first — we want
                # to choose the BEST one (most-overhead camera position),
                # not the first one IK happened to converge on.
                pose_cands, gen_stats = gen.generate(max_count=None)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise scan target (%.2f, %.2f) — pose generation "
                    "raised %s: %s. Skipping target.",
                    tx, ty, type(e).__name__, e,
                )
                continue
            if pose_cands:
                # Pick the most-vertical candidate (camera optical axis
                # nearest to straight-down). Reuses the helper defined
                # earlier in this thread.
                pose_cands_sorted = sorted(
                    pose_cands,
                    key=lambda c: _vertical_score_for(c, target_world),
                    reverse=True,
                )
                candidates.append(pose_cands_sorted[0])
            else:
                rej = getattr(gen_stats, "rejection_log", None) or {}
                n_total = (
                    len(_LOCALISE_SCAN_DISTANCES_M)
                    * len(_LOCALISE_SCAN_ELEVATIONS_DEG)
                    * _LOCALISE_SCAN_AZIMUTH_COUNT
                )
                logger.info(
                    "localise scan target (%.2f, %.2f) — no reachable pose "
                    "across %d candidates (rejections: %s)",
                    tx, ty, n_total, dict(rej),
                )

        # `candidates` is intentionally empty in J0-sweep mode (we drive the
        # robot from `seed_q_list` instead, building waypoints on-the-fly).
        # The empty-candidates check only applies to the legacy multi-target
        # path; J0-sweep mode reports its own failures earlier (no reachable
        # seed at any target) or later (no detections during the sweep).
        if not _LOCALISE_USE_J0_SWEEP:
            if not candidates:
                _post_status(
                    f"Localise: 0 reachable scan poses out of "
                    f"{len(_LOCALISE_SCAN_TARGETS_M)} targets. Check that "
                    f"_LOCALISE_SCAN_TARGETS_M lies within PAROL6's workspace."
                )
                return
            logger.info(
                "localise (multi-target): %d/%d scan poses generated",
                len(candidates), len(_LOCALISE_SCAN_TARGETS_M),
            )
        else:
            logger.info(
                "localise (J0-sweep): %d/%d seed targets reachable",
                len(seed_q_list), len(_LOCALISE_SEED_TARGETS_XY),
            )

        # Mode dispatch: same as _calibration_thread — pick VirtualCamera vs
        # RealSenseCamera based on Waldo-Commander's robot-mode toggle.
        try:
            from waldo_commander.state import robot_state  # noqa: PLC0415
            is_sim_mode = bool(robot_state.simulator_active)
        except Exception:  # noqa: BLE001
            is_sim_mode = True

        raw_client = RobotClient(host="127.0.0.1", port=5001)
        _state["client"] = raw_client

        def flange_pose_provider():
            return _flange_pose_from_client(raw_client)

        if is_sim_mode:
            intrinsics = Intrinsics(
                fx=_INTR_FX, fy=_INTR_FY, cx=_INTR_CX, cy=_INTR_CY,
                width=_INTR_W, height=_INTR_H,
                dist_coeffs=np.zeros(5, dtype=np.float64),
            )
            # Sim ground truth: same perturbation scheme as _calibration_thread.
            ground_truth_mount = CameraMount.from_eyeball_estimate(
                x_mm=_CAM_MOUNT_TRANSLATE_MM[0] + 2.0,
                y_mm=_CAM_MOUNT_TRANSLATE_MM[1] + 2.0,
                z_mm=_CAM_MOUNT_TRANSLATE_MM[2] - 2.0,
                tilt_x_deg=_CAM_MOUNT_TILT_DEG[0] + 2.0,
                tilt_y_deg=_CAM_MOUNT_TILT_DEG[1] - 1.0,
                tilt_z_deg=_CAM_MOUNT_TILT_DEG[2],
            )
            # The VirtualBoard sees the CURRENT _T_BOARD2BASE as ground truth.
            camera: Any = VirtualCamera(
                intrinsics=intrinsics,
                image_width=_INTR_W,
                image_height=_INTR_H,
                ground_truth_mount=ground_truth_mount,
                flange_pose_provider=flange_pose_provider,
                board=VirtualBoard(config=BOARD_TABLET_30MM, T_board2base=_T_BOARD2BASE),
                noise_std=0.0,
            )
            logger.info("localise camera: VirtualCamera (simulator mode)")
        else:
            from parol6_vision.camera.realsense import RealSenseCamera  # noqa: PLC0415
            camera = RealSenseCamera(
                width=_INTR_W,
                height=_INTR_H,
                fps=30,
                enable_depth=False,
                enable_color=True,
            )
            # Stash BEFORE start(): see _calibration_thread for rationale.
            _state["real_camera"] = camera
            camera.start()
            intrinsics = camera.intrinsics
            logger.info(
                "localise camera: RealSenseCamera (real-hardware mode), "
                "intrinsics fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            )
        # Lower min_corners_for_pose for the localise scan — we only need a
        # rough board location estimate (translation, mostly), not a high-
        # accuracy calibration sample. With min_corners=6 (the default) the
        # diagnostic showed many frames where ≥1 ArUco markers were visible
        # but ChArUco interpolation produced fewer than 6 corners and the
        # detector returned None, even though there were enough constraints
        # for a coarse pose. min_corners=4 is the absolute minimum for
        # solvePnP on a planar target; the calibration's main detector
        # keeps the default 6 for accuracy.
        detector = BoardDetector(BOARD_TABLET_30MM, min_corners_for_pose=4)

        cfg = BOARD_TABLET_30MM
        center_local = np.array(
            [cfg.squares_x * cfg.square_length / 2.0,
             cfg.squares_y * cfg.square_length / 2.0,
             0.0, 1.0],
            dtype=np.float64,
        )

        K = intrinsics.as_camera_matrix()
        D = intrinsics.dist_coeffs

        # Per-detection: keep the full T_board2base SE(3) so we can recover
        # the board's rotation, not just its centre point. This handles
        # tablets that aren't perfectly aligned with the configured
        # _BOARD_RPY_RAD (real-life setups will be slightly tilted / yawed
        # from "perfectly square" no matter how careful the placement).
        detected_centres: list[NDArray[np.float64]] = []
        detected_poses: list[NDArray[np.float64]] = []
        detected_qualities: list[int] = []  # corner count, for picking best rotation

        # Diagnostic state: counts of frame-level outcomes per sweep so the
        # post-sweep summary can tell us WHERE detection is breaking down
        # (blank frames vs no markers vs charuco interpolation vs pose fail).
        _diag = {"blank": 0, "markers0": 0, "markers_some": 0, "detected": 0}
        _diag_dump_dir: Path | None = None
        _diag_dumped = 0
        _DIAG_DUMP_LIMIT = 6  # save up to 6 sample frames total per localise run

        # Independent ArUco detector for the diagnostic probe — runs the
        # marker layer alone (skips ChArUco interpolation + solvePnP) so we
        # can tell whether the markers were even visible to OpenCV.
        _aruco_dict_for_probe = cv2.aruco.getPredefinedDictionary(
            BOARD_TABLET_30MM.aruco_dict_id,
        )
        _aruco_probe = cv2.aruco.ArucoDetector(
            _aruco_dict_for_probe, cv2.aruco.DetectorParameters(),
        )

        def _capture_and_record(label: str) -> bool:
            """Capture one frame, run ChArUco detection, and on success
            append (board centre, full SE(3) pose, corner-count) to the
            detection accumulators. Queries the live flange pose AFTER
            capture so the joint angles match the captured frame within
            ~10 ms — negligible drift at the configured sweep speeds.
            Returns True on a successful detection.

            Also runs a low-level diagnostic probe on each frame (frame
            stats + raw marker count) and saves up to _DIAG_DUMP_LIMIT
            sample frames to a temp directory for visual inspection.
            """
            nonlocal _diag_dumped, _diag_dump_dir
            try:
                frame = camera.capture_color()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise %s: capture_color failed (%s: %s); skipping",
                    label, type(e).__name__, e,
                )
                return False

            # Frame stats — distinguishes blank (mean ≈ background) from
            # filled frames where detection still fails.
            frame_mean = float(frame.mean())
            frame_std = float(frame.std())
            if frame_std < 2.0:
                # Solid-grey blank — VirtualCamera bail path returns these
                # when flange pose is None or board is out of view.
                _diag["blank"] += 1
            else:
                # Probe raw marker count without going through full board
                # detection — tells us if ArUco itself can find markers.
                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if frame.ndim == 3 else frame
                )
                _, marker_ids, _ = _aruco_probe.detectMarkers(gray)
                n_markers = 0 if marker_ids is None else len(marker_ids)
                if n_markers == 0:
                    _diag["markers0"] += 1
                else:
                    _diag["markers_some"] += 1

            # Save up to _DIAG_DUMP_LIMIT sample frames for visual inspection.
            if _diag_dumped < _DIAG_DUMP_LIMIT:
                if _diag_dump_dir is None:
                    import tempfile  # noqa: PLC0415
                    _diag_dump_dir = Path(tempfile.gettempdir()) / "localise_frames"
                    _diag_dump_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        "localise: saving diagnostic frames to %s",
                        _diag_dump_dir,
                    )
                fname = _diag_dump_dir / (
                    f"frame_{_diag_dumped:02d}_{label.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
                )
                try:
                    cv2.imwrite(str(fname), frame)
                    _diag_dumped += 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("frame dump failed: %s", e)

            detection = detector.detect(frame, K, D)
            if detection is None:
                return False
            try:
                T_flange2base = _flange_pose_from_client(raw_client)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "localise %s: flange-pose query failed (%s: %s)",
                    label, type(e).__name__, e,
                )
                return False
            if T_flange2base is None:
                return False
            T_board2cam = board_pose_to_matrix(detection)
            T_board2base_obs = T_flange2base @ cold_start.T_cam2flange @ T_board2cam
            board_centre_obs = (T_board2base_obs @ center_local)[:3]
            detected_centres.append(board_centre_obs)
            detected_poses.append(T_board2base_obs)
            detected_qualities.append(int(detection.num_corners_detected))
            _diag["detected"] += 1
            return True

        attempted = 0
        if _LOCALISE_USE_J0_SWEEP and _LOCALISE_CONTINUOUS_SWEEP:
            # Continuous-sweep mode — one non-blocking move_j per seed,
            # capture during motion, accumulate detections. Skips to the
            # next seed once enough detections are gathered.
            for seed_xy, seed_q_rad in seed_q_list:
                if _state.get("stop_requested"):
                    raw_client.halt()
                    _post_status("Localise stopped by user")
                    return
                if len(detected_centres) >= _LOCALISE_MIN_DETECTIONS:
                    break  # already enough

                # Pick start of sweep based on current J0: whichever end of
                # the [seed_J0 - half, seed_J0 + half] range is CLOSER to the
                # robot's current J0. Eliminates the time-wasting "drive all
                # the way to the far side first" behaviour when the robot
                # starts near home or at the previous sweep's end.
                seed_j0_rad = float(seed_q_rad[0])
                low_j0 = seed_j0_rad - np.radians(_LOCALISE_J0_SWEEP_HALF_DEG)
                high_j0 = seed_j0_rad + np.radians(_LOCALISE_J0_SWEEP_HALF_DEG)
                try:
                    cur_angles = raw_client.angles()
                    cur_j0_rad = (
                        float(np.radians(cur_angles[0]))
                        if cur_angles is not None and len(cur_angles) > 0
                        else seed_j0_rad
                    )
                except Exception:  # noqa: BLE001
                    cur_j0_rad = seed_j0_rad
                if abs(cur_j0_rad - low_j0) <= abs(cur_j0_rad - high_j0):
                    start_j0 = low_j0
                    end_j0 = high_j0
                else:
                    start_j0 = high_j0
                    end_j0 = low_j0
                start_q = seed_q_rad.copy()
                start_q[0] = start_j0
                end_q = seed_q_rad.copy()
                end_q[0] = end_j0
                if not (
                    scan_robot.check_limits(start_q)
                    and scan_robot.check_limits(end_q)
                ):
                    logger.info(
                        "localise sweep (%.2f, %.2f): start or end out of joint "
                        "limits (J0 range %.1f° → %.1f°), skipping",
                        seed_xy[0], seed_xy[1],
                        float(np.degrees(start_j0)),
                        float(np.degrees(end_j0)),
                    )
                    continue
                logger.info(
                    "localise sweep (%.2f, %.2f): J0 %.1f° → %.1f° "
                    "(starting end is %.1f° away from current %.1f°)",
                    seed_xy[0], seed_xy[1],
                    float(np.degrees(start_j0)), float(np.degrees(end_j0)),
                    float(np.degrees(abs(start_j0 - cur_j0_rad))),
                    float(np.degrees(cur_j0_rad)),
                )

                _post_status(
                    f"Localise: continuous sweep at seed ({seed_xy[0]:.2f}, "
                    f"{seed_xy[1]:.2f}) — moving to start"
                )
                rc = raw_client.move_j(
                    angles=list(np.degrees(start_q)),
                    speed=0.3, accel=0.5, wait=True, timeout=15.0,
                )
                if rc < 0:
                    logger.info(
                        "localise sweep start move returned rc=%d; skipping seed", rc,
                    )
                    continue
                time.sleep(_SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S)

                # Capture once at the start before motion so a board
                # already in view at the sweep's starting J0 isn't missed.
                _capture_and_record(
                    f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) start"
                )

                _post_status(
                    f"Localise: continuous sweep at seed ({seed_xy[0]:.2f}, "
                    f"{seed_xy[1]:.2f}) — scanning"
                )
                # Issue non-blocking move to end of sweep.
                rc = raw_client.move_j(
                    angles=list(np.degrees(end_q)),
                    speed=_LOCALISE_SWEEP_SPEED, accel=0.5, wait=False,
                    timeout=30.0,
                )
                if rc < 0:
                    logger.info(
                        "localise sweep continuous move returned rc=%d; skipping",
                        rc,
                    )
                    continue
                # Capture loop while motion executes.
                #
                # COMPLETION DETECTION: previously used is_robot_stopped()
                # (joint speed below 2 °/s threshold). Live diagnostic
                # showed that lied — parol6's status broadcaster reports
                # speeds at ~50 Hz and stale-status windows make
                # is_robot_stopped() flip True while joints are still
                # moving. Loop exited mid-motion, captured ~2 frames, and
                # the user saw the "completed" log line appear well
                # before the robot actually reached the target.
                #
                # New approach: check joint ANGLES against the target.
                # That's deterministic — current angles update from the
                # same status stream but their VALUE is what we care
                # about, not their derivative. When max-per-joint
                # angle error drops below 1°, motion is genuinely done.
                # A stuck-progress watchdog (no movement >3 s while not
                # at target) catches the rare case of move_j being
                # rejected silently.
                target_q_deg = np.degrees(end_q)
                last_capture = 0.0
                sweep_started = time.monotonic()
                sweep_max_duration_s = 30.0
                last_progress_t = sweep_started
                last_q_deg: NDArray[np.float64] | None = None
                while True:
                    if _state.get("stop_requested"):
                        raw_client.halt()
                        _post_status("Localise stopped by user")
                        return
                    now = time.monotonic()
                    if now - sweep_started > sweep_max_duration_s:
                        logger.warning(
                            "localise sweep timed out after %.1f s, halting",
                            sweep_max_duration_s,
                        )
                        raw_client.halt()
                        break

                    try:
                        cur_angles = raw_client.angles()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "localise: angles() raised %s: %s; halting",
                            type(e).__name__, e,
                        )
                        raw_client.halt()
                        break
                    if cur_angles is None or len(cur_angles) < 6:
                        time.sleep(0.05)
                        continue
                    cur_q_deg = np.asarray(cur_angles, dtype=np.float64)

                    # Track progress: any joint moved by >0.3° since
                    # last sample → robot is actively traversing the
                    # trajectory, reset the stuck-progress timer.
                    if (
                        last_q_deg is not None
                        and float(np.max(np.abs(cur_q_deg - last_q_deg))) > 0.3
                    ):
                        last_progress_t = now
                    last_q_deg = cur_q_deg

                    # Motion done — at the target within 1° per joint.
                    angle_err_deg = float(
                        np.max(np.abs(cur_q_deg - target_q_deg)),
                    )
                    if angle_err_deg < 1.0:
                        break

                    # Stuck-progress watchdog: no joint movement for
                    # >3 s WHILE we're not at target. Means move_j was
                    # rejected or parol6-server is genuinely hung;
                    # halt + bail rather than waiting for the 30 s timeout.
                    if now - last_progress_t > 3.0:
                        logger.warning(
                            "localise sweep stuck at angle err=%.1f° with "
                            "no progress for >3 s; halting",
                            angle_err_deg,
                        )
                        raw_client.halt()
                        break

                    if now - last_capture >= _LOCALISE_CAPTURE_PERIOD_S:
                        last_capture = now
                        attempted += 1
                        if _capture_and_record(
                            f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f})"
                        ):
                            _post_status(
                                f"Localise: {len(detected_centres)} detection"
                                f"{'s' if len(detected_centres) != 1 else ''} "
                                f"so far ({attempted} captures attempted)"
                            )
                    # Short sleep so we don't pin the CPU between captures.
                    time.sleep(0.02)

                # Settle briefly after the sweep finishes, then capture once
                # more at the end pose — same rationale as the start
                # capture, with the final J0 still in view of the workspace.
                time.sleep(_SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S)
                attempted += 1
                _capture_and_record(
                    f"sweep ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) end"
                )

                # Per-sweep diagnostic summary. Goal: distinguish
                # "camera never saw anything" from "camera saw things but
                # detector kept failing", so the user can fix the right
                # thing instead of guessing.
                logger.info(
                    "localise sweep (%.2f, %.2f) diagnostics: "
                    "blank=%d (frame too uniform → camera likely returned "
                    "_blank_frame, board out of view or flange-pose query "
                    "failed); markers=0 in %d frames (frame had content but "
                    "ArUco found no markers); markers≥1 in %d frames (markers "
                    "visible, may need more for ChArUco interpolation); "
                    "%d full detections succeeded",
                    seed_xy[0], seed_xy[1],
                    _diag["blank"], _diag["markers0"],
                    _diag["markers_some"], _diag["detected"],
                )
                # Reset counters so the next sweep starts clean — we keep
                # _diag_dumped global to enforce the per-run dump limit.
                _diag["blank"] = 0
                _diag["markers0"] = 0
                _diag["markers_some"] = 0
                _diag["detected"] = 0
        elif _LOCALISE_USE_J0_SWEEP:
            # Discrete J0-sweep mode (continuous disabled). Per seed,
            # iterate _LOCALISE_J0_STEPS angles with stop+capture at each.
            j0_offsets = np.linspace(
                -_LOCALISE_J0_SWEEP_HALF_DEG, +_LOCALISE_J0_SWEEP_HALF_DEG,
                _LOCALISE_J0_STEPS,
            )
            for seed_xy, seed_q_rad in seed_q_list:
                if len(detected_centres) >= _LOCALISE_MIN_DETECTIONS:
                    break
                for dj0 in j0_offsets:
                    if _state.get("stop_requested"):
                        _post_status("Localise stopped by user")
                        return
                    q_rad = seed_q_rad.copy()
                    q_rad[0] += np.radians(dj0)
                    if not scan_robot.check_limits(q_rad):
                        continue
                    attempted += 1
                    _post_status(
                        f"Localise: discrete step at seed ({seed_xy[0]:.2f}, "
                        f"{seed_xy[1]:.2f}) — dJ0={dj0:+.0f}°"
                    )
                    try:
                        rc = raw_client.move_j(
                            angles=list(np.degrees(q_rad)),
                            speed=0.3, accel=0.5, wait=True, timeout=15.0,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "localise discrete move_j failed: %s", e,
                        )
                        continue
                    if rc < 0:
                        continue
                    time.sleep(
                        _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S,
                    )
                    _capture_and_record(
                        f"discrete ({seed_xy[0]:.2f}, {seed_xy[1]:.2f}) dJ0={dj0:+.0f}°"
                    )
        else:
            # Multi-target legacy path — iterate the pre-built `candidates`
            # list as before (one fixed pose per scan target).
            for i, c in enumerate(candidates):
                if _state.get("stop_requested"):
                    _post_status("Localise stopped by user")
                    return
                attempted += 1
                _post_status(f"Localise: scan {i + 1}/{len(candidates)} — moving")
                try:
                    rc = raw_client.move_j(
                        angles=list(np.degrees(c.joint_angles_rad).tolist()),
                        speed=0.3, accel=0.5, wait=True, timeout=15.0,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("localise scan %d move_j failed: %s", i, e)
                    continue
                if rc < 0:
                    continue
                time.sleep(
                    _SETTLE_TIME_SIM_S if is_sim_mode else _SETTLE_TIME_REAL_S,
                )
                _capture_and_record(f"scan {i + 1}/{len(candidates)}")

        if len(detected_centres) < _LOCALISE_MIN_DETECTIONS:
            _post_status(
                f"Localise FAILED: only {len(detected_centres)} detections "
                f"from {attempted} captures across "
                f"{len(seed_q_list) if _LOCALISE_USE_J0_SWEEP else len(candidates)} "
                "scan path(s) (need "
                f"≥{_LOCALISE_MIN_DETECTIONS}). Board may be outside the "
                f"workspace scan region — check _LOCALISE_SEED_TARGETS_XY "
                "(or _LOCALISE_SCAN_TARGETS_M for legacy mode)."
            )
            return

        # Median-then-inlier-mean on CENTRES: robust to a single outlier
        # without needing full RANSAC. Mirrors the orchestrator's own
        # bootstrap consensus (parol6_vision.calibration.refinement.board_position_ransac).
        det_arr = np.asarray(detected_centres, dtype=np.float64)
        median = np.median(det_arr, axis=0)
        residuals = np.linalg.norm(det_arr - median, axis=1)
        inlier_mask = residuals < _LOCALISE_INLIER_THRESHOLD_M
        n_inliers = int(inlier_mask.sum())
        if n_inliers < _LOCALISE_MIN_DETECTIONS:
            _post_status(
                f"Localise FAILED: {n_inliers} inliers within "
                f"{_LOCALISE_INLIER_THRESHOLD_M * 1000:.0f} mm of median "
                f"(need ≥{_LOCALISE_MIN_DETECTIONS}). "
                "Detections too inconsistent — try repositioning the board."
            )
            return

        new_centre = det_arr[inlier_mask].mean(axis=0)

        # Pick rotation from the BEST inlier (most ChArUco corners detected
        # → most stable solvePnP). Averaging rotations across noisy detections
        # is fiddly (rotations don't average linearly); using the single
        # best-quality detection's rotation is robust and simple. With
        # workspace-scan elevations near 90°, the camera-to-board angle is
        # nearly normal so solvePnP rotation accuracy is good.
        inlier_indices = np.where(inlier_mask)[0]
        best_inlier_idx = int(max(inlier_indices, key=lambda j: detected_qualities[j]))
        R_detected = np.asarray(detected_poses[best_inlier_idx], dtype=np.float64)[:3, :3]

        # Build the new full 4x4 transform first, then assign atomically. The
        # previous in-memory rotation/translation are snapshotted for the
        # delta report so it shows the actual change since this update
        # (not the configured-vs-detected delta — that would be misleading
        # on repeat localise calls).
        old_R = _T_BOARD2BASE[:3, :3].copy()
        old_origin = _T_BOARD2BASE[:3, 3].copy()
        center_offset_world = R_detected @ center_local[:3]
        new_origin = new_centre - center_offset_world

        new_T = np.eye(4, dtype=np.float64)
        new_T[:3, :3] = R_detected
        new_T[:3, 3] = new_origin
        # Atomic-ish write: a concurrent reader of `_T_BOARD2BASE` either sees
        # the entire pre-update matrix or the entire post-update matrix, never
        # a torn (R_new, t_old) state. NumPy's `[:] = …` invokes element-wise
        # assignment under the GIL; the GIL doesn't make it formally atomic
        # but in practice no Python statement interleaves between the two
        # halves of a 4x4 copy.
        _T_BOARD2BASE[:] = new_T

        delta_mm = float(np.linalg.norm(new_origin - old_origin)) * 1000.0
        # ‖R_a − R_b‖_F = 2√2 sin(θ/2) is the exact identity (not approximate),
        # so this recovers the rotation angle in degrees.
        rot_frob = float(np.linalg.norm(R_detected - old_R, ord="fro"))
        rot_delta_deg = float(
            np.degrees(2.0 * np.arcsin(min(1.0, rot_frob / (2.0 * np.sqrt(2)))))
        )
        _post_status(
            f"Localise OK: {n_inliers}/{len(detected_centres)} inliers, "
            f"centre ({new_centre[0]:.3f}, {new_centre[1]:.3f}, "
            f"{new_centre[2]:.3f}) m, shift {delta_mm:.1f} mm / {rot_delta_deg:.1f}°"
        )

        # Invalidate caches that depend on the previous _T_BOARD2BASE:
        #   - trajectory_collision_mgr_pair embeds the OLD tablet pose.
        #   - reachable_candidates were generated against the OLD centre, so
        #     their joint configs no longer aim at the new board location.
        # Both rebuild on next use.
        _state["trajectory_collision_mgr_pair"] = None
        _state["reachable_candidates"] = []

        # Stamp the successful-localise time so the Run button knows the
        # board has been localised this session and skips the warning dialog.
        _state["last_localise_ok_at"] = time.time()

        # Refresh visual overlays so board, hemisphere, and reachability dots
        # all reflect the new pose. Schedules on the asyncio loop.
        refresh_board_dependent_overlays()

    except Exception as e:  # noqa: BLE001
        if _state.get("stop_requested"):
            logger.info("Localise stopped by user (caught %s: %s)",
                        type(e).__name__, e)
            _post_status("Localise stopped by user")
        else:
            logger.exception("Localise thread crashed")
            _post_status(f"Localise ERROR: {e}")
    finally:
        # Stop the RealSenseCamera if we started one in real-hardware mode.
        real_cam = _state.get("real_camera")
        if real_cam is not None:
            try:
                real_cam.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("RealSenseCamera stop failed: %s", e)
            _state["real_camera"] = None
        _state["is_localising"] = False


def _post_status(text: str) -> None:
    """Post a status update to the GUI from a worker thread."""
    label = _state.get("status_label")
    loop = _state.get("main_loop")
    if label is None or loop is None:
        logger.info("Calibration status: %s", text)
        return

    def _update():
        # Re-fetch in case the page tore down between scheduling and
        # dispatch (status_label was deleted, set to None on cleanup).
        live_label = _state.get("status_label")
        if live_label is None:
            return
        try:
            live_label.text = text
        except Exception:  # noqa: BLE001
            # Label exists but is detached / disposed.
            pass

    try:
        loop.call_soon_threadsafe(_update)
    except RuntimeError as e:
        # Loop closed (page tear-down). Log at info — not a real failure.
        logger.info("Status post skipped (loop unavailable): %s", e)
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

    def _busy_warn(msg: str) -> bool:
        """Reject button press if either the calibration or the localise thread
        is already running, OR if the localise-before-Run dialog is open.
        Returns True if a warning was issued."""
        if _state.get("is_running"):
            ui.notify(
                f"{msg}: calibration already running — wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("is_localising"):
            ui.notify(
                f"{msg}: board localise already running — wait for it to finish",
                color="warning",
                position="top",
            )
            return True
        if _state.get("dialog_open"):
            ui.notify(
                f"{msg}: confirmation dialog open — answer it first",
                color="warning",
                position="top",
            )
            return True
        return False

    def _start_calibration() -> None:
        """Spawn the calibration thread (post-confirmation)."""
        _state["is_running"] = True
        _state["stop_requested"] = False
        _state["calibrated_mount"] = None
        _post_status("Running calibration via parol6-server...")
        threading.Thread(target=_calibration_thread, daemon=True).start()

    def _on_click() -> None:
        if _busy_warn("Run"):
            return
        # Localise-before-Run guard. If the user hasn't successfully run
        # the Localise Board sweep this session, the calibration's bootstrap
        # will aim at the configured _BOARD_TRANSLATE_M — which on real
        # hardware is essentially never accurate. A confirmation dialog
        # offers to run anyway (sim mode, or already-trusted setup) or
        # cancel and run Localise first.
        if _state.get("last_localise_ok_at") is None:
            # Block any other Run / Localise click while the dialog is open.
            # Without this, the user can click Localise during the open
            # dialog → both threads end up running in parallel.
            _state["dialog_open"] = True
            with ui.dialog() as dialog, ui.card():
                ui.label("Board hasn't been localised this session").classes(
                    "text-base font-semibold"
                )
                ui.label(
                    "Calibration will use the configured _BOARD_TRANSLATE_M as "
                    "the bootstrap target. If your physical tablet isn't there, "
                    "bootstrap will fail to find the board."
                ).classes("text-sm")
                ui.label(
                    "Recommended: Cancel, click Localise Board first, then Run."
                ).classes("text-xs opacity-80")
                with ui.row():
                    def _proceed():
                        _state["dialog_open"] = False
                        dialog.close()
                        _start_calibration()

                    def _cancel():
                        _state["dialog_open"] = False
                        dialog.close()
                    ui.button(
                        "Run anyway", on_click=_proceed, color="warning",
                    ).props("size=sm")
                    ui.button("Cancel", on_click=_cancel).props("size=sm")
            # Backdrop / Esc dismisses the dialog without invoking either
            # button — clear the flag in that case too.
            dialog.on("hide", lambda _e=None: _state.update(dialog_open=False))
            dialog.open()
            return
        _start_calibration()

    def _on_localise() -> None:
        """Drive a small lookout sweep + auto-locate the board centre."""
        if _busy_warn("Localise"):
            return
        _state["is_localising"] = True
        _state["stop_requested"] = False
        _post_status("Localising board — driving lookout sweep...")
        threading.Thread(target=_localise_board_thread, daemon=True).start()

    def _on_stop() -> None:
        """Abort the running calibration OR localise: halt + flag the thread.

        ``RobotClient.halt()`` is a sync UDP call that refuses to run inside
        an active asyncio event loop. The NiceGUI callback runs in the main
        event loop, so calling halt() directly from here raises
        ``RobotClient was used while an event loop is running``. Workaround:
        dispatch halt() to a daemon thread which has no event loop attached.
        The flag (_state["stop_requested"]) is set immediately so subsequent
        move_j calls in the running thread short-circuit even if the
        thread-dispatched halt hasn't fired yet.
        """
        if not (_state.get("is_running") or _state.get("is_localising")):
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
            ui.notify("HALTED — robot motion stopped", color="warning")
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
            ui.button(
                "Localise Board", on_click=_on_localise, color="secondary",
            ).props("size=sm")
            ui.button("STOP", on_click=_on_stop, color="negative").props("size=sm")

    # 4 Hz tick to apply the calibrated mount once calibration finishes.
    # No joint-update plumbing needed — the parol6-server status broadcast
    # drives the URDF scene naturally.
    ui.timer(0.25, _post_calibration_tick, active=True)
