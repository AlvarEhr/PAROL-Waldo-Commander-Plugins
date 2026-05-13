"""J0 helper-mode wrapper for the parol6 sync client.

PAROL6's base joint (J0) can stick or skip under torque on some setups.
Small/slow moves are often the most problematic since they don't build
enough torque to overcome static friction. When the user enables helper
mode in the calibration panel, EVERY ``move_j`` call that touches J0
gets:

  1. A log line announcing the direction (LEFT/RIGHT from above) and
     magnitude.
  2. An optional pause (configurable seconds) so the user can be near
     the robot and physically assist the joint over its sticky range.

Composed via attribute-delegation around any RobotClient (sync or
``_HaltableClient`` wrapper); enable by wrapping the client in
``HelperModeJ0Client(inner, pause_seconds=...)``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Floor on what counts as a "J0 move" — below this many degrees the
# delta is effectively zero (rounding noise, identity moves). Anything
# above triggers the helper warning, including the small/slow moves
# that struggle most with base-joint stiction.
_J0_DELTA_FLOOR_DEG: float = 0.1


class HelperModeJ0Client:
    """Wrap a parol6 RobotClient so each ``move_j`` is preceded by a
    direction warning and a configurable pause when J0 motion exceeds
    ``j0_threshold_deg``. All other client methods pass through via
    ``__getattr__``.
    """

    def __init__(
        self,
        inner: Any,
        pause_seconds: float,
        j0_floor_deg: float = _J0_DELTA_FLOOR_DEG,
    ) -> None:
        self._inner = inner
        self._pause_seconds = max(0.0, float(pause_seconds))
        self._j0_floor_deg = float(j0_floor_deg)

    def __getattr__(self, name: str) -> Any:
        # Delegate every non-overridden attribute access to the wrapped
        # client. Required to keep parol6's halt/resume/angles/etc.
        # behaviour intact.
        return getattr(self._inner, name)

    def move_j(self, angles: Any = None, **kwargs: Any) -> Any:
        if angles is None:
            return self._inner.move_j(**kwargs)
        self._maybe_warn_and_pause(angles)
        return self._inner.move_j(angles=angles, **kwargs)

    def _maybe_warn_and_pause(self, target_angles: Any) -> None:
        """Log direction + sleep when the planned J0 delta exceeds the
        threshold. Best-effort: any errors in the warning path fall
        through silently — the move itself must not be blocked.
        """
        try:
            target_seq = list(target_angles)
            if not target_seq:
                return
            target_j0 = float(target_seq[0])
        except Exception:  # noqa: BLE001
            return
        try:
            cur_angles = self._inner.angles()
            if cur_angles is None or len(cur_angles) < 1:
                return
            cur_j0 = float(cur_angles[0])
        except Exception:  # noqa: BLE001
            return

        delta = target_j0 - cur_j0
        if abs(delta) < self._j0_floor_deg:
            return
        # Looking at the robot from above (camera-down convention), a
        # positive J0 delta rotates the base CCW which appears as LEFT
        # from the operator standing in front of the robot.
        direction = "LEFT (CCW from above)" if delta > 0 else "RIGHT (CW from above)"
        logger.warning(
            "HELPER MODE: J0 about to rotate %s by %.1f deg "
            "(current=%.1f deg -> target=%.1f deg). Push gently in the "
            "indicated direction. Move begins in %.1fs.",
            direction,
            abs(delta),
            cur_j0,
            target_j0,
            self._pause_seconds,
        )
        if self._pause_seconds > 0.0:
            # Tick-poll ``stop_requested`` so STOP doesn't have to wait out
            # the full pause before the worker thread can react.
            self._sleep_with_stop_check(self._pause_seconds)

    def _sleep_with_stop_check(self, seconds: float) -> None:
        """Sleep up to ``seconds`` total, breaking early if the panel's
        STOP button was pressed. Best-effort: state-read failures fall
        through to a plain ``time.sleep``.
        """
        from .state import _state  # noqa: PLC0415
        deadline = time.monotonic() + seconds
        tick = 0.1
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                if _state.get("stop_requested"):
                    logger.info(
                        "HELPER MODE: pause interrupted by STOP after %.2fs",
                        seconds - remaining,
                    )
                    return
            except Exception:  # noqa: BLE001
                time.sleep(remaining)
                return
            time.sleep(min(tick, remaining))


def maybe_wrap_helper_mode(client: Any) -> Any:
    """Wrap ``client`` in :class:`HelperModeJ0Client` when the user has
    enabled the helper-mode setting; pass through unchanged otherwise.
    """
    try:
        from . import settings as _s  # noqa: PLC0415
        enabled = bool(_s.get("helper_mode_j0_enabled"))
        if not enabled:
            return client
        pause_s = float(_s.get("helper_mode_j0_pause_s"))
    except Exception as e:  # noqa: BLE001
        logger.debug("helper_mode setting read failed (%s); not wrapping", e)
        return client
    logger.info(
        "Helper mode J0 ENABLED: %.1fs pause before EVERY J0 rotation "
        "(> %.2f deg)",
        pause_s, _J0_DELTA_FLOOR_DEG,
    )
    return HelperModeJ0Client(client, pause_seconds=pause_s)
