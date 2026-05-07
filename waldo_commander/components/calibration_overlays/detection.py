"""Detection overlay (live perception viz) — polling + scene rendering."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
from nicegui import ui

from .constants import (
    _DETECTION_JSON_PATH,
    _DETECTION_OVERLAY_COLOR,
)
from .state import _state

logger = logging.getLogger(__name__)


# Stale detections — anything older than this in seconds is ignored. The
# JSON file gets written once per ``find_object.py`` invocation and stays
# on disk indefinitely; without a freshness check the overlay would
# show whatever the last test run wrote, possibly weeks ago, on every
# waldo-commander startup. Tunable here if a long-running pipeline
# needs more headroom.
_DETECTION_FRESHNESS_S: float = 60.0


def _clear_detection_group() -> None:
    """Tear down the rendered detection wireframe group, if any."""
    grp = _state.get("detection_overlay_group")
    if grp is None:
        return
    try:
        grp.delete()
    except Exception as e:  # noqa: BLE001
        logger.debug("detection overlay delete failed: %s", e)
    _state["detection_overlay_group"] = None


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
        _clear_detection_group()

        # Visibility toggle (panel checkbox). Off by default — the JSON
        # snapshot frequently outlives the perception run that wrote it.
        if not bool(_state.get("show_detections", False)):
            return
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

    Files older than ``_DETECTION_FRESHNESS_S`` are ignored — without
    this, a stale ``last_detection.json`` from a prior ``find_object.py``
    test would show its wireframe forever after every restart. Any
    rendered group from a previous tick is cleared when staleness or
    the visibility toggle says we shouldn't be rendering.
    """
    path = _DETECTION_JSON_PATH
    show = bool(_state.get("show_detections", False))
    if not show:
        # User turned the overlay off — make sure any stale group from
        # a previous tick is gone, then bail without polling the file.
        if _state.get("detection_overlay_group") is not None:
            loop = _state.get("main_loop")
            if loop is None:
                _clear_detection_group()
            else:
                try:
                    loop.call_soon_threadsafe(_clear_detection_group)
                except RuntimeError:
                    pass
        return

    try:
        if not path.exists():
            return
        mtime = path.stat().st_mtime
    except OSError as e:
        logger.debug("_poll_detection_json: stat failed: %s", e)
        return

    # Freshness guard — drop stale detections so old pipeline runs don't
    # ghost across restarts.
    if (time.time() - mtime) > _DETECTION_FRESHNESS_S:
        if _state.get("detection_overlay_group") is not None:
            loop = _state.get("main_loop")
            if loop is None:
                _clear_detection_group()
            else:
                try:
                    loop.call_soon_threadsafe(_clear_detection_group)
                except RuntimeError:
                    pass
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
