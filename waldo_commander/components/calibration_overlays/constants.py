"""User-facing tunables and fit constants for calibration overlays.

All edits take effect on next waldo-commander restart (no cache invalidation
step needed; the STL bake re-runs unconditionally on every startup).
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
# Visibility of the various 3D overlays (board+tablet, hemisphere, dots,
# fixed frustum near-cone, projected footprint, centerline) is now driven
# at runtime by the calibration panel's checkboxes; per-user state is
# persisted via NiceGUI's ``app.storage.user`` and surfaces through
# ``_state['show_*']``. See ``_set_overlay_visible``.
_HEMI_DISTANCE_RANGE_M: tuple[float, float] = (0.14, 0.28)  # (min, max) radial distance
# 15° → 89°. Lower bound is genuinely useful — low-elevation oblique
# views see the board's edges in heavy foreshortening, but ChArUco
# tolerates that down to ~75° off-normal (= ev=15° if board lies flat).
# Upper bound is now near-overhead (89° — practical limit; exactly 90°
# is degenerate for look_at). The pose-generator's continuous-Sobol path
# now appends an explicit cap ring at ev=ev_hi (8 azimuths × 3 distances
# = 24 extra candidates, see ``parol6_vision.calibration.pose_generator``)
# so the boundary IS sampled even though Sobol's unit cube never lands
# on u=1. Most cap candidates fail IK on PAROL6's wrist-flip mount, but
# the few that succeed are exactly the perfectly-overhead poses the
# view-pose selector wants — and the wireframe now visually closes the
# dome instead of cutting off at 70°.
_HEMI_ELEVATION_RANGE_DEG: tuple[float, float] = (15.0, 89.0)
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

# Safety margin (metres) inflating the FLOOR and TABLET collision
# primitives. Calibration / localise have residual mount error of
# ~1-5 mm even after convergence; on real hardware sensor noise +
# joint-encoder error adds another mm or two on top. Inflating the
# floor/tablet collision boxes by this margin makes the planner
# REJECT a pose if the gripper would come within margin_m of contact,
# rather than waiting until the visualisation shows actual penetration.
# 8 mm is comfortable cushion without rejecting too many useful poses
# (the calibration's diversity benefits are above ~2 cm clearance).
_COLLISION_SAFETY_MARGIN_M: float = 0.008

# SSG-48 gripper jaw variant for collision checking. The user has two
# interchangeable jaw STLs in parol6's mesh dir — "finger" (the standard
# Spectral SSG-48 finger) and "pinch" (narrower pinch tips). Only ONE
# variant is physically mounted at a time; loading both into the
# collision manager would over-reject (the union of finger+pinch
# extents). Picks the variant that matches the physically-mounted jaws;
# default "finger" matches Jepson's default SSG-48 config.
_SSG48_JAW_VARIANT: str = "finger"  # "finger" or "pinch"

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
    (0.22, 0.0),  # close — workspaces near the base
    (0.30, 0.0),  # workspace centre
    (0.38, 0.0),  # farther — outward-placed boards
)
# Backwards-compat alias; consumers that wired _LOCALISE_SEED_TARGET_M
# directly still work.
_LOCALISE_SEED_TARGET_M: tuple[float, float] = _LOCALISE_SEED_TARGETS_XY[1]
# Continuous-sweep speed (fraction of max joint speed). 0.10 = 10% of
# max — gentle enough that 150 ms capture period covers <2° of J0 per
# frame, so consecutive frames at the same detection event give similar
# pose measurements that median-consensus well together.
_LOCALISE_SWEEP_SPEED: float = 0.10
# Early-stop threshold: once this many detections are gathered, halt the
# in-progress sweep and skip remaining seeds. 3 gives the median
# consensus enough samples to reject a single bad detection while
# avoiding the full 3-seed × 18-second motion when the board is found
# quickly. Set to a large number (e.g. 999) to disable early stop and
# always run all seeds for the densest possible consensus sample set.
_LOCALISE_EARLY_STOP_DETECTIONS: int = 10
# Chunk size for the J0 sweep in degrees. The full sweep is broken into
# back-to-back move_j commands of this size so that halt() (used both by
# early-stop AND the user's Stop / E-Stop button) interrupts within at
# most one chunk's worth of motion. parol6's halt() clears the command
# queue but does NOT interrupt the trajectory currently being executed
# by the controller — so a single 180° sweep was uninterruptible until
# the controller naturally finished it. With 20° chunks at speed 0.10
# (~1 s per chunk), halt response time is ≤1 s.
_LOCALISE_J0_CHUNK_DEG: float = 20.0
# Stage-2 refinement: after the J0 sweep finds the board, drive to N
# overhead poses ABOVE the rough median centre and capture additional
# frames. With the gaze pointed directly at the board (no offset), the
# whole board fits comfortably in FOV, the markers are well-resolved,
# and ChArUco interpolation gets ≥6 corners — much higher-quality
# detections than the sweep frames where the board was off-axis. The
# refined detections feed the same consensus pool as the sweep ones,
# pulling the median toward the truth even when sweep detections were
# noisy. Set to 0 to disable refinement.
_LOCALISE_REFINE_N_POSES: int = 4
_LOCALISE_REFINE_DISTANCE_M: float = 0.30
_LOCALISE_REFINE_ELEVATION_DEG: float = 80.0
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
# Two-tier stop logic:
#
#   IN-SWEEP TARGET (_LOCALISE_EARLY_STOP_DETECTIONS / _INLIERS):
#     Halt the sweep mid-motion when we hit this. Higher = more captures
#     before stopping, more samples for the consensus → tighter median.
#
#   POST-SWEEP MIN (_LOCALISE_MIN_INLIERS_TO_PROCEED):
#     After a sweep finishes naturally without hitting the in-sweep
#     target, skip remaining seeds anyway IF we have at least this many
#     inliers. Avoids the user-noted case where stage 1 found enough
#     detections to localise the board but we kept driving through more
#     seeds chasing a stricter target.
#
# Combined: best case the first sweep hits the TARGET quickly and we
# halt early. Medium case the sweep ends with ≥ MIN inliers and we move
# on to stage 2. Worst case (<MIN inliers) we try the next seed.
_LOCALISE_EARLY_STOP_INLIERS: int = 7
_LOCALISE_MIN_INLIERS_TO_PROCEED: int = 4
# RANSAC inlier threshold (metres) — detections within this distance of the
# inlier-set median count as agreeing on the board location. 5 cm is generous
# enough that the cold-start mount's ~2 mm / 2° error doesn't reject good
# detections, tight enough that an outlier 5 cm off doesn't fool us.
_LOCALISE_INLIER_THRESHOLD_M: float = 0.05

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
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "parol6-vision" / "Results" / "perception" / "last_detection.json"
)
_DETECTION_POLL_INTERVAL_S: float = 0.5
_DETECTION_OVERLAY_COLOR: str = "#ff8800"  # orange — distinct from board (textured) and frustum (cyan)
