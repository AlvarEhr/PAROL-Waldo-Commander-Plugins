"""parol6-vision hand-eye calibration overlays for Waldo-Commander.

Two integration points exposed at the package root:

1. **Startup-time custom-tool registration**. ``main.py`` calls into
   ``custom_tools.auto_migrate_ssg48_with_bracket()`` and
   ``custom_tools.register_all()`` to migrate any legacy SSG-48 +
   camera-bracket setup into a regular custom tool, then bakes + binds
   every entry under ``~/.waldo-commander/custom_tools/`` so the
   gripper dropdown shows them. Gated by the
   ``WALDO_CALIBRATION_ENABLED`` env flag plus the in-app
   ``calibration_features_active`` storage toggle (OFF by default for
   fresh users — the calibration tab is visible but no disk-write
   side effects fire until the user opts in).

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

Historical note: an earlier integration point #1
(:func:`hijack_ssg48_body_mesh`) replaced the SSG-48 BODY mesh entry
in parol6's ``_TOOL_REGISTRY``. Commit ee8674e replaced that hijack
with the custom-tool migration path; the helper still ships in
``ssg48_hijack.py`` for any out-of-tree code that imports it directly,
but it's no longer wired in. New integrations should use the custom-
tool path instead.

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

# ``hijack_ssg48_body_mesh`` is intentionally NOT re-exported here —
# the custom-tool migration path supersedes it. Direct importers
# (``from ...ssg48_hijack import hijack_ssg48_body_mesh``) still work,
# but the package's public surface only advertises the active path.
__all__ = [
    "add_overlays",
    "apply_calibration_state",
    "build_calibration_panel_content",
    "custom_tools",
    "refresh_board_dependent_overlays",
    "validate_joint_trajectory",
]
