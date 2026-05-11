"""parol6-vision hand-eye calibration overlays for Waldo-Commander.

Two integration points at the package root:

1. Startup-time custom-tool registration via ``custom_tools.register_all``
   (called from ``main.py``). Gated by ``WALDO_CALIBRATION_ENABLED`` and
   the ``calibration_features_active`` storage toggle.
2. :func:`add_overlays` and :func:`build_calibration_panel_content` —
   page-build-time scene decoration and side-tab UI.

Package layout:

* ``constants`` — tunables (intrinsics, hemisphere, board, mount, etc).
* ``state`` — shared ``_state`` + ``_T_BOARD2BASE`` + geometry helpers.
* ``workspace``, ``collision``, ``occlusion`` — pose-generator filters.
* ``ssg48_hijack`` — deprecated; superseded by ``custom_tools``.
* ``overlays``, ``frustum``, ``reachability``, ``detection``.
* ``localise``, ``calibration_thread``, ``hover`` — worker threads.
* ``panel`` — side-tab UI + ``_post_status`` helper.
"""

from . import custom_tools
from .collision import validate_joint_trajectory
from .overlays import add_overlays, refresh_board_dependent_overlays
from .panel import apply_calibration_state, build_calibration_panel_content

# ``hijack_ssg48_body_mesh`` is not re-exported; superseded by custom_tools.
__all__ = [
    "add_overlays",
    "apply_calibration_state",
    "build_calibration_panel_content",
    "custom_tools",
    "refresh_board_dependent_overlays",
    "validate_joint_trajectory",
]
