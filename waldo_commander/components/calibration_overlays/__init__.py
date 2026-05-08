"""parol6-vision hand-eye calibration overlays for Waldo-Commander.

Two integration points exposed at the package root:

1. :func:`hijack_ssg48_body_mesh` — at startup (before
   ``initialize_urdf_scene``), replace the BODY mesh entry of the
   ``SSG-48`` tool in parol6's ``_TOOL_REGISTRY`` with the user's merged
   ``ssg48_body_realsense.stl``. Non-destructive: jaws, TCP transform,
   and motion descriptors stay the same. The merged STL is pre-baked
   with the load-time fit transform applied (scale 0.1 + translate
   +13.955, -0.007, 0 m) into parol6's mesh dir so it sits beside the
   stock body file.

2. :func:`add_overlays` and :func:`build_calibration_panel_content` — at
   page-build time. ``add_overlays`` decorates the existing URDF scene
   with the ChArUco board (textured plane), tablet primitive, hemisphere
   wireframe + reachability dots, camera frustum (parented to
   ``tcp_anchor``), and the dynamic ray-projected footprint. It also
   installs the 4 Hz post-calibration-mount tick and the 5 Hz frustum-
   raycast tick. ``build_calibration_panel_content`` builds the
   contents of the "calibration" side-tab: Run / Localise / STOP plus
   the view-overlay toggles plus the hover-above-board verification
   controls.

Layout of the package:

* ``constants`` — every tunable (intrinsics, hemisphere params, board
  placement, mount, frustum depths, localise sweep config, etc).
* ``state`` — the shared ``_state`` dict + ``_T_BOARD2BASE`` and the
  geometry helpers that compose them.
* ``workspace``, ``collision``, ``occlusion`` — three pieces of the
  pose-generator filter pipeline; each builds its own CollisionManager
  / occlusion mesh and exposes the per-pose query.
* ``ssg48_hijack`` — startup-time tool-registry mutation.
* ``overlays`` — scene-build entry point + per-overlay group builders +
  visibility toggles + the localise-driven refresh.
* ``frustum`` — the dynamic ray-projected footprint logic plus the two
  scene timers.
* ``reachability`` — the green-dot IK sweep, dispatched off the asyncio
  loop so the websocket pump stays responsive.
* ``detection`` — perception-pipeline overlay (AABB rendering from
  ``last_detection.json``).
* ``localise``, ``calibration_thread``, ``hover`` — the three background
  threads that drive the controller via UDP.
* ``panel`` — the side-tab panel UI + the ``_post_status`` helper that
  the threads use to push status updates back to the GUI.
"""

from . import custom_tools
from .collision import validate_joint_trajectory
from .overlays import add_overlays, refresh_board_dependent_overlays
from .panel import apply_calibration_state, build_calibration_panel_content
from .ssg48_hijack import hijack_ssg48_body_mesh

__all__ = [
    "add_overlays",
    "apply_calibration_state",
    "build_calibration_panel_content",
    "custom_tools",
    "hijack_ssg48_body_mesh",
    "refresh_board_dependent_overlays",
    "validate_joint_trajectory",
]
