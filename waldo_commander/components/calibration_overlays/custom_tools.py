"""Custom tool ingestion — drop-in STLs become user-defined tools.

Drop STLs + ``config.json`` into ``~/.waldo-commander/custom_tools/<name>/``;
startup scans the directory, applies the placement transform per STL,
bakes into parol6's mesh dir, and registers a new entry in
``parol6.tools._TOOL_REGISTRY``.

Folder layout::

    ~/.waldo-commander/custom_tools/<name>/
      config.json          # placement + TCP + jaw motion + meta
      body.stl             # required
      jaw_left.stl         # optional
      jaw_right.stl        # optional

config.json schema (every numeric field optional, defaults shown)::

    {
      "display_name": "My Custom Gripper",
      "description": "...",
      "mesh_translate_m": [0, 0, 0],
      "mesh_rpy_rad":     [0, 0, 0],
      "mesh_scale":       1.0,
      "tcp_origin_m":     [0, 0, -0.10],
      "tcp_rpy_rad":      [0, 0, 0],
      "jaw_travel_m":     0.0,
      "jaw_axis":         [0, 1, 0],
      "jaw_symmetric":    true
    }
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRot

logger = logging.getLogger(__name__)


# Path-traversal defence: pathlib's ``/`` resets on absolute RHS and
# doesn't block ``..``, so a crafted name could escape ``CUSTOM_TOOLS_ROOT``.
_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _validate_safe_name(value: str, kind: str = "name") -> None:
    """Reject anything that isn't ASCII alphanumerics + underscore."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"empty / non-string {kind}: {value!r}")
    if not _VALID_NAME_RE.match(value):
        raise ValueError(
            f"invalid {kind}: {value!r}; must match [A-Za-z0-9_]+ "
            "(no path separators, dots, spaces, or unicode)"
        )


# Caps trimesh.load against OOM on arbitrarily large user uploads.
_STL_MAX_SIZE_MB: int = 200


def _stl_size_check(stl_path: Path) -> bool:
    """True iff ``stl_path`` exists and is within ``_STL_MAX_SIZE_MB``."""
    try:
        size_bytes = stl_path.stat().st_size
    except OSError as e:
        logger.warning("STL stat failed for %s: %s", stl_path, e)
        return False
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > _STL_MAX_SIZE_MB:
        logger.warning(
            "STL %s exceeds size cap (%.1f MB > %d MB); rejecting load",
            stl_path, size_mb, _STL_MAX_SIZE_MB,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Tool-registry callback registry — drives gripper-dropdown live refresh.
# ---------------------------------------------------------------------------


_registry_changed_callbacks: list[Callable[[], None]] = []


def subscribe_tool_registry_changed(cb: Callable[[], None]) -> None:
    """Register a callback fired after registry mutations. Idempotent.
    Failures dispatched best-effort so a buggy subscriber can't break it."""
    if cb not in _registry_changed_callbacks:
        _registry_changed_callbacks.append(cb)


def unsubscribe_tool_registry_changed(cb: Callable[[], None]) -> None:
    """Remove ``cb`` from the callback list (no-op if not present)."""
    try:
        _registry_changed_callbacks.remove(cb)
    except ValueError:
        pass


def _notify_tool_registry_changed() -> None:
    """Fire every registered callback. Best-effort dispatch."""
    for cb in list(_registry_changed_callbacks):
        try:
            cb()
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "custom_tools: registry-changed callback %r failed: %s",
                getattr(cb, "__qualname__", cb), e,
            )


def _refresh_active_robot_tools() -> None:
    """Rebuild ``ui_state.active_robot._tools`` from parol6's registry so
    newly-registered custom tools are visible to ``Robot.tools`` consumers.
    """
    try:
        from parol6.robot import _build_tools as _parol6_build_tools  # noqa: PLC0415
        from waldo_commander.state import ui_state  # noqa: PLC0415

        ui_state.active_robot._tools = _parol6_build_tools()  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        logger.debug("active_robot._tools rebuild failed: %s", e)


# ---------------------------------------------------------------------------
# Discovery / paths
# ---------------------------------------------------------------------------


CUSTOM_TOOLS_ROOT: Path = Path.home() / ".waldo-commander" / "custom_tools"
CONFIG_FILENAME: str = "config.json"
BODY_STL_NAME: str = "body.stl"
JAW_LEFT_STL_NAME: str = "jaw_left.stl"
JAW_RIGHT_STL_NAME: str = "jaw_right.stl"


def ensure_root() -> Path:
    """Make sure the custom-tools root exists, return it."""
    CUSTOM_TOOLS_ROOT.mkdir(parents=True, exist_ok=True)
    return CUSTOM_TOOLS_ROOT


def list_tool_names() -> list[str]:
    """Return the list of custom-tool directory names that have a config."""
    if not CUSTOM_TOOLS_ROOT.exists():
        return []
    out: list[str] = []
    for entry in sorted(CUSTOM_TOOLS_ROOT.iterdir()):
        if entry.is_dir() and (entry / CONFIG_FILENAME).exists():
            out.append(entry.name)
    return out


# ---------------------------------------------------------------------------
# Config dataclass + (de)serialisation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CustomToolVariant:
    """One variant on a custom tool — same body, different jaws + motion.
    Mirrors parol6's ``ToolVariant``. Variant jaws live at
    ``<root>/<name>/variant_<key>_jaw_<side>.stl``.
    """
    key: str  # registry key (used in client.select_tool variant_key)
    display_name: str = ""
    jaw_travel_m: float = 0.0
    jaw_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    jaw_symmetric: bool = True
    has_jaws: bool = False  # derived — True when both variant jaw STLs exist


@dataclass(slots=True)
class CustomToolConfig:
    """User-editable parameters for one custom tool."""

    name: str  # used as registry key + folder name
    display_name: str = ""
    description: str = ""
    mesh_translate_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_scale: float = 1.0
    tcp_origin_m: tuple[float, float, float] = (0.0, 0.0, -0.10)
    tcp_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jaw_travel_m: float = 0.0
    jaw_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    jaw_symmetric: bool = True
    # Built-in tool key the CONTROLLER should act as when this custom
    # tool is selected (e.g. "SSG-48"). Custom tools live only in the
    # GUI process; an empty string disables proxying.
    proxy_tool_key: str = ""
    # True when the tool has a calibratable camera; gates the
    # calibration overlays + Run/Localise/Hover.
    has_camera: bool = False
    # Per-tool camera overrides. Non-None REPLACES the global at runtime
    # while this tool is active; None inherits the global.
    intr_fx: float | None = None
    intr_fy: float | None = None
    intr_cx: float | None = None
    intr_cy: float | None = None
    intr_width: int | None = None
    intr_height: int | None = None
    cam_mount_translate_mm: tuple[float, float, float] | None = None
    cam_mount_tilt_deg: tuple[float, float, float] | None = None
    # Empty = single-variant tool; non-empty registers ToolVariant entries
    # so the gripper panel's variant dropdown can pick among them.
    variants: list[CustomToolVariant] = field(default_factory=list)
    has_jaws: bool = False  # True when jaw_left.stl + jaw_right.stl exist
    has_body: bool = False  # True when body.stl exists

    @property
    def folder(self) -> Path:
        return CUSTOM_TOOLS_ROOT / self.name

    @property
    def config_path(self) -> Path:
        return self.folder / CONFIG_FILENAME

    @property
    def body_path(self) -> Path:
        return self.folder / BODY_STL_NAME

    @property
    def jaw_left_path(self) -> Path:
        return self.folder / JAW_LEFT_STL_NAME

    @property
    def jaw_right_path(self) -> Path:
        return self.folder / JAW_RIGHT_STL_NAME

    def variant_jaw_path(self, variant_key: str, side: str) -> Path:
        """One variant's jaw STL path; ``side`` is ``"left"`` or ``"right"``."""
        return self.folder / f"variant_{variant_key}_jaw_{side}.stl"

    def to_json(self) -> str:
        # has_body / has_jaws are derived (don't persist them).
        payload: dict[str, Any] = {
            "display_name": self.display_name,
            "description": self.description,
            "mesh_translate_m": list(self.mesh_translate_m),
            "mesh_rpy_rad": list(self.mesh_rpy_rad),
            "mesh_scale": float(self.mesh_scale),
            "tcp_origin_m": list(self.tcp_origin_m),
            "tcp_rpy_rad": list(self.tcp_rpy_rad),
            "jaw_travel_m": float(self.jaw_travel_m),
            "jaw_axis": list(self.jaw_axis),
            "jaw_symmetric": bool(self.jaw_symmetric),
            "proxy_tool_key": str(self.proxy_tool_key or ""),
            "has_camera": bool(self.has_camera),
        }
        # Persist only non-None overrides. Explicit per-key float/int
        # membership keeps a future ``intr_*_max`` field from being
        # silently coerced via a substring check.
        _INTR_FLOAT_KEYS = ("intr_fx", "intr_fy", "intr_cx", "intr_cy")
        _INTR_INT_KEYS = ("intr_width", "intr_height")
        for k in (*_INTR_FLOAT_KEYS, *_INTR_INT_KEYS):
            v = getattr(self, k)
            if v is None:
                continue
            payload[k] = float(v) if k in _INTR_FLOAT_KEYS else int(v)
        if self.cam_mount_translate_mm is not None:
            payload["cam_mount_translate_mm"] = list(self.cam_mount_translate_mm)
        if self.cam_mount_tilt_deg is not None:
            payload["cam_mount_tilt_deg"] = list(self.cam_mount_tilt_deg)
        if self.variants:
            payload["variants"] = [
                {
                    "key": v.key,
                    "display_name": v.display_name,
                    "jaw_travel_m": float(v.jaw_travel_m),
                    "jaw_axis": list(v.jaw_axis),
                    "jaw_symmetric": bool(v.jaw_symmetric),
                }
                for v in self.variants
            ]
        return json.dumps(payload, indent=2, sort_keys=True)


def _coerce_tuple3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    try:
        x, y, z = [float(v) for v in value]
    except (TypeError, ValueError):
        logger.warning("custom_tools: bad 3-tuple %r — using %s", value, default)
        return default
    return (x, y, z)


def load_config(name: str) -> CustomToolConfig | None:
    """Load one custom tool's config from disk; None if missing/malformed."""
    folder = CUSTOM_TOOLS_ROOT / name
    config_path = folder / CONFIG_FILENAME
    if not config_path.exists():
        return None
    try:
        raw = json.loads(config_path.read_text())
    except Exception as e:  # noqa: BLE001
        logger.warning("custom_tools: %s — invalid config JSON: %s", config_path, e)
        return None

    # Backfill ``has_camera`` for pre-Phase-1D SSG-48 migration configs;
    # the field defaults to False otherwise so the camera gate would close.
    backfilled = False
    if name == _SSG48_MIGRATION_NAME and "has_camera" not in raw:
        raw["has_camera"] = True
        backfilled = True
    if backfilled:
        try:
            config_path.write_text(json.dumps(raw, indent=2, sort_keys=True))
            logger.info(
                "custom_tools: backfilled has_camera=True on %s/config.json",
                name,
            )
        except OSError as e:
            logger.debug("backfill rewrite failed: %s", e)

    def _opt_float(k: str) -> float | None:
        v = raw.get(k)
        return None if v is None else float(v)

    def _opt_int(k: str) -> int | None:
        v = raw.get(k)
        return None if v is None else int(v)

    def _opt_tuple3(k: str) -> tuple[float, float, float] | None:
        v = raw.get(k)
        if v is None:
            return None
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except (TypeError, ValueError, IndexError):
            return None

    cfg = CustomToolConfig(
        name=name,
        display_name=str(raw.get("display_name") or name),
        description=str(raw.get("description") or ""),
        mesh_translate_m=_coerce_tuple3(raw.get("mesh_translate_m"), (0.0, 0.0, 0.0)),
        mesh_rpy_rad=_coerce_tuple3(raw.get("mesh_rpy_rad"), (0.0, 0.0, 0.0)),
        mesh_scale=float(raw.get("mesh_scale", 1.0)),
        tcp_origin_m=_coerce_tuple3(raw.get("tcp_origin_m"), (0.0, 0.0, -0.10)),
        tcp_rpy_rad=_coerce_tuple3(raw.get("tcp_rpy_rad"), (0.0, 0.0, 0.0)),
        jaw_travel_m=float(raw.get("jaw_travel_m", 0.0)),
        jaw_axis=_coerce_tuple3(raw.get("jaw_axis"), (0.0, 1.0, 0.0)),
        jaw_symmetric=bool(raw.get("jaw_symmetric", True)),
        proxy_tool_key=str(raw.get("proxy_tool_key") or ""),
        has_camera=bool(raw.get("has_camera", False)),
        intr_fx=_opt_float("intr_fx"),
        intr_fy=_opt_float("intr_fy"),
        intr_cx=_opt_float("intr_cx"),
        intr_cy=_opt_float("intr_cy"),
        intr_width=_opt_int("intr_width"),
        intr_height=_opt_int("intr_height"),
        cam_mount_translate_mm=_opt_tuple3("cam_mount_translate_mm"),
        cam_mount_tilt_deg=_opt_tuple3("cam_mount_tilt_deg"),
    )
    # Variants (optional)
    raw_variants = raw.get("variants") or []
    if isinstance(raw_variants, list):
        for vraw in raw_variants:
            if not isinstance(vraw, dict):
                continue
            vkey = str(vraw.get("key") or "").strip()
            if not vkey:
                continue
            v = CustomToolVariant(
                key=vkey,
                display_name=str(vraw.get("display_name") or vkey),
                jaw_travel_m=float(vraw.get("jaw_travel_m", 0.0)),
                jaw_axis=_coerce_tuple3(vraw.get("jaw_axis"), (0.0, 1.0, 0.0)),
                jaw_symmetric=bool(vraw.get("jaw_symmetric", True)),
            )
            v.has_jaws = (
                cfg.variant_jaw_path(vkey, "left").exists()
                and cfg.variant_jaw_path(vkey, "right").exists()
            )
            cfg.variants.append(v)

    cfg.has_body = (folder / BODY_STL_NAME).exists()
    cfg.has_jaws = (
        (folder / JAW_LEFT_STL_NAME).exists()
        and (folder / JAW_RIGHT_STL_NAME).exists()
    )
    return cfg


def save_config(cfg: CustomToolConfig) -> None:
    """Write the config.json (creates the folder if needed)."""
    cfg.folder.mkdir(parents=True, exist_ok=True)
    cfg.config_path.write_text(cfg.to_json())


def list_configs() -> list[CustomToolConfig]:
    """Return one config per custom tool found on disk."""
    out: list[CustomToolConfig] = []
    for name in list_tool_names():
        cfg = load_config(name)
        if cfg is not None:
            out.append(cfg)
    return out


def delete_tool(name: str) -> None:
    """Wipe a tool from disk + registry + per-tool storage. Clearing the
    storage means a same-named recreate doesn't inherit stale overrides.
    """
    folder = CUSTOM_TOOLS_ROOT / name
    if folder.exists():
        shutil.rmtree(folder)
    # Sweep baked copies so the parol6 mesh dir doesn't accumulate orphans.
    mesh_dir = _parol6_mesh_dir()
    if mesh_dir is not None:
        try:
            for stl in mesh_dir.glob(f"custom_{name}_*.stl"):
                try:
                    stl.unlink()
                except OSError as e:
                    logger.debug(
                        "delete_tool: couldn't unlink baked %s: %s", stl, e,
                    )
        except OSError as e:
            logger.debug("delete_tool: glob on mesh_dir failed (%s)", e)
    # Best-effort unregister; pops directly since parol6's mutation API
    # is minimal. Falls back to "tool gone after restart" on shape change.
    key = f"custom:{name}"
    try:
        from parol6 import tools as parol6_tools  # noqa: PLC0415

        registry = getattr(parol6_tools, "_TOOL_REGISTRY", None)
        if registry is not None and key in registry:
            del registry[key]
    except Exception as e:  # noqa: BLE001
        logger.debug("custom_tools: delete_tool registry unregister failed: %s", e)
    # Clear persisted per-tool overrides so a same-name recreate is clean.
    try:
        clear_per_tool_overrides(key)
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "custom_tools: delete_tool clear_per_tool_overrides failed: %s", e,
        )
    _refresh_active_robot_tools()
    _notify_tool_registry_changed()


def import_stl(name: str, role: str, src: Path) -> None:
    """Copy a user-provided STL into the tool's folder under the
    canonical name. ``role`` is one of ``"body"``, ``"jaw_left"``,
    ``"jaw_right"``.
    """
    _validate_safe_name(name, "tool name")
    folder = CUSTOM_TOOLS_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    dst_name = {
        "body": BODY_STL_NAME,
        "jaw_left": JAW_LEFT_STL_NAME,
        "jaw_right": JAW_RIGHT_STL_NAME,
    }.get(role)
    if dst_name is None:
        raise ValueError(f"unknown role: {role!r}")
    shutil.copyfile(str(src), str(folder / dst_name))


def import_variant_stl(name: str, variant_key: str, side: str, src: Path) -> None:
    """Copy a user-provided STL into the tool's folder as a variant
    jaw mesh. ``side`` is ``"left"`` or ``"right"``.
    """
    _validate_safe_name(name, "tool name")
    _validate_safe_name(variant_key, "variant key")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    folder = CUSTOM_TOOLS_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"variant_{variant_key}_jaw_{side}.stl"
    shutil.copyfile(str(src), str(dst))


# ---------------------------------------------------------------------------
# Mesh-aware helpers — auto unit detection + snap to bbox / flange face
# ---------------------------------------------------------------------------


def _load_trimesh(stl_path: Path) -> Any | None:
    """Best-effort load with trimesh; None on failure or size-cap reject."""
    if not _stl_size_check(stl_path):
        return None
    try:
        import trimesh  # noqa: PLC0415

        mesh = trimesh.load(str(stl_path), force="mesh")
        if mesh is None or not hasattr(mesh, "extents"):
            return None
        return mesh
    except Exception as e:  # noqa: BLE001
        logger.warning("trimesh load failed for %s: %s", stl_path, e)
        return None


def detect_mesh_unit_scale(stl_path: Path) -> tuple[float, str]:
    """Heuristic bbox-based unit detection; returns ``(scale, label)`` to
    convert STL units to metres:

    * bbox.max > 1 m   → mm   → 0.001
    * bbox.max < 1 mm  → sub  → 1000.0
    * bbox.max < 1 in  → ambiguous → 1.0 (user picks)
    * else             → m    → 1.0
    """
    mesh = _load_trimesh(stl_path)
    if mesh is None:
        return (1.0, "unknown (load failed)")
    bbox_max = float(np.max(mesh.extents))
    if bbox_max > 1.0:
        return (0.001, f"mm (bbox max {bbox_max:.0f} units)")
    if bbox_max < 0.001:
        return (1000.0, f"sub-mm scaled (bbox max {bbox_max:.6f} units)")
    if bbox_max < 0.0254:
        # Sub-inch is ambiguous (m or cm); let the user override.
        return (1.0, f"ambiguous (bbox max {bbox_max:.4f} units)")
    return (1.0, f"m (bbox max {bbox_max:.3f} units)")


def snap_to_bbox_centre(stl_path: Path) -> tuple[float, float, float] | None:
    """Translate (in mesh units) that would move the bbox centre to the
    origin. Returns None when the mesh can't be loaded.
    """
    mesh = _load_trimesh(stl_path)
    if mesh is None:
        return None
    centre = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    return (-float(centre[0]), -float(centre[1]), -float(centre[2]))


def snap_flange_face_to_origin(
    stl_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Snap the largest planar facet's centroid to origin with its normal
    aligned to flange +Z. Returns ``(translate, rpy_rad)`` or None if the
    mesh has no clear planar facet.
    """
    mesh = _load_trimesh(stl_path)
    if mesh is None:
        return None
    try:
        facets = mesh.facets
        facet_areas = mesh.facets_area
        if len(facets) == 0:
            return None
        biggest = int(np.argmax(facet_areas))
        # Average normal of the facet group, weighted by face area.
        face_idx = facets[biggest]
        face_areas = mesh.area_faces[face_idx]
        face_normals = mesh.face_normals[face_idx]
        avg_normal = (face_normals * face_areas[:, None]).sum(axis=0)
        avg_normal /= float(np.linalg.norm(avg_normal))
        # Centroid: area-weighted face centroid.
        face_centres = mesh.triangles_center[face_idx]
        centroid = (face_centres * face_areas[:, None]).sum(axis=0) / float(
            face_areas.sum()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("snap_flange_face_to_origin: %s", e)
        return None

    # Rotation that maps avg_normal to +Z.
    target = np.array([0.0, 0.0, 1.0])
    rot = _rotation_between(avg_normal, target)
    rpy = SciRot.from_matrix(rot).as_euler("XYZ").tolist()
    rotated_centroid = rot @ centroid
    translate = (-float(rotated_centroid[0]), -float(rotated_centroid[1]),
                 -float(rotated_centroid[2]))
    return (translate, (float(rpy[0]), float(rpy[1]), float(rpy[2])))


def _rotation_between(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """3x3 rotation matrix that maps unit vector ``a`` to unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cos_ang = float(np.dot(a, b))
    if cos_ang > 0.999999:
        return np.eye(3)
    if cos_ang < -0.999999:
        # Antiparallel — pick any axis perpendicular to a.
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        return SciRot.from_rotvec(np.pi * axis).as_matrix()
    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    angle = float(np.arccos(cos_ang))
    return SciRot.from_rotvec(angle * axis).as_matrix()


def snap_to_largest_circular_hole(
    stl_path: Path,
) -> tuple[float, float, float] | None:
    """Snap the largest circular hole in the largest planar facet (the
    flange-face mount bore) to origin. Returns None when no clear hole
    is detected; callers fall back to ``snap_flange_face_to_origin``.
    """
    mesh = _load_trimesh(stl_path)
    if mesh is None:
        return None
    try:
        facets = mesh.facets
        facet_areas = mesh.facets_area
        if len(facets) == 0:
            return None
        biggest_idx = int(np.argmax(facet_areas))
        face_idx_arr = facets[biggest_idx]
        # Submesh of just this facet group.
        submesh = mesh.submesh([face_idx_arr], append=True)
        # Boundary loops via the trimesh "outline" path.
        outline = submesh.outline()
        if outline is None or not hasattr(outline, "discrete"):
            return None
        # outline.discrete is a list of Nx3 arrays — the boundary polylines.
        loops = list(outline.discrete)
    except Exception as e:  # noqa: BLE001
        logger.debug("snap_to_largest_circular_hole: outline failed (%s)", e)
        return None
    if len(loops) < 2:
        # No interior holes (only the outer perimeter).
        return None

    # Pick a coordinate frame on the facet plane: average normal as Z,
    # arbitrary in-plane X/Y. Project each loop into 2D and fit a circle.
    face_idx = facets[biggest_idx]
    face_normals = mesh.face_normals[face_idx]
    face_areas = mesh.area_faces[face_idx]
    avg_normal = (face_normals * face_areas[:, None]).sum(axis=0)
    avg_normal /= float(np.linalg.norm(avg_normal))
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(avg_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    in_plane_x = np.cross(avg_normal, helper)
    in_plane_x /= float(np.linalg.norm(in_plane_x))
    in_plane_y = np.cross(avg_normal, in_plane_x)

    best: tuple[float, NDArray[np.float64]] | None = None  # (radius, centre_3d)
    for loop in loops:
        loop_arr = np.asarray(loop, dtype=np.float64)
        if loop_arr.shape[0] < 6:
            continue
        # Project into 2D.
        u = loop_arr @ in_plane_x
        v = loop_arr @ in_plane_y
        try:
            cx, cy, r = _fit_circle_2d(u, v)
        except Exception:  # noqa: BLE001
            continue
        # Reject loops whose residual std exceeds 10 % of radius.
        residuals = np.hypot(u - cx, v - cy) - r
        if r <= 0 or float(np.std(residuals)) > 0.1 * r:
            continue
        # 3D centre: in-plane (cx, cy) + the loop's mean depth along normal.
        loop_centroid_3d = loop_arr.mean(axis=0)
        depth = float(loop_centroid_3d @ avg_normal)
        centre_3d = cx * in_plane_x + cy * in_plane_y + depth * avg_normal
        if best is None or r > best[0]:
            best = (r, centre_3d)

    if best is None:
        return None
    centre = best[1]
    return (-float(centre[0]), -float(centre[1]), -float(centre[2]))


def _fit_circle_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Fit a circle to 2D points via the linear least-squares method.
    Returns ``(cx, cy, r)`` or raises ``ValueError`` on degenerate input.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if u.size < 3:
        raise ValueError("need at least 3 points for circle fit")
    A = np.column_stack([2.0 * u, 2.0 * v, np.ones_like(u)])
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        raise ValueError("circle fit produced negative r²")
    return float(cx), float(cy), float(np.sqrt(r2))


# ---------------------------------------------------------------------------
# Bake — apply placement transform to STL and write into parol6's mesh dir
# ---------------------------------------------------------------------------


def _parol6_mesh_dir() -> Path | None:
    try:
        from importlib.resources import files as pkg_files  # noqa: PLC0415
        return Path(str(pkg_files("parol6"))) / "urdf_model" / "meshes"
    except Exception as e:  # noqa: BLE001
        logger.warning("could not locate parol6 mesh dir: %s", e)
        return None


def _baked_filename(name: str, role: str) -> str:
    """Filename used INSIDE parol6's mesh dir for the given custom tool."""
    return f"custom_{name}_{role}.stl"


def bake_one(cfg: CustomToolConfig) -> dict[str, Path] | None:
    """Apply the placement transform to each STL and write to parol6's
    mesh dir. Returns ``{role: dst_path}`` for the roles that had a
    source STL, or None on hard failure. Idempotent.
    """
    mesh_dir = _parol6_mesh_dir()
    if mesh_dir is None:
        return None
    if not cfg.body_path.exists():
        logger.info("custom_tools: %s has no body.stl; skipping bake", cfg.name)
        return None
    try:
        import trimesh  # noqa: PLC0415
    except ImportError:
        logger.warning("custom_tools: trimesh unavailable; bake skipped")
        return None

    sca = float(cfg.mesh_scale)
    if sca <= 0:
        logger.warning("custom_tools: %s scale %s invalid; bake skipped", cfg.name, sca)
        return None
    R = SciRot.from_euler("XYZ", cfg.mesh_rpy_rad).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R * sca
    T[:3, 3] = np.asarray(cfg.mesh_translate_m, dtype=np.float64)

    out: dict[str, Path] = {}
    bake_jobs: list[tuple[str, Path]] = [
        ("body", cfg.body_path),
        ("jaw_left", cfg.jaw_left_path),
        ("jaw_right", cfg.jaw_right_path),
    ]
    for v in cfg.variants:
        bake_jobs.append(
            (f"variant_{v.key}_jaw_left", cfg.variant_jaw_path(v.key, "left")),
        )
        bake_jobs.append(
            (f"variant_{v.key}_jaw_right", cfg.variant_jaw_path(v.key, "right")),
        )
    # Captured once so repeated stat() across roles doesn't add up.
    try:
        cfg_mtime = (
            cfg.config_path.stat().st_mtime if cfg.config_path.exists() else 0.0
        )
    except OSError:
        cfg_mtime = 0.0

    for role, src_path in bake_jobs:
        if not src_path.exists():
            continue
        # Size-cap rejects mirror the missing-file branch.
        if not _stl_size_check(src_path):
            continue
        dst = mesh_dir / _baked_filename(cfg.name, role)
        # Skip the trimesh round-trip when the cached output is newer
        # than both inputs (source STL + config.json holds the params).
        try:
            if dst.exists():
                dst_mtime = dst.stat().st_mtime
                src_mtime = src_path.stat().st_mtime
                if dst_mtime >= max(src_mtime, cfg_mtime):
                    out[role] = dst
                    continue
        except OSError as e:
            logger.debug(
                "custom_tools: mtime check failed for %s/%s: %s; "
                "will re-bake defensively", cfg.name, role, e,
            )
        try:
            mesh = trimesh.load(str(src_path), force="mesh")
            mesh.apply_transform(T)
            mesh.export(str(dst))
            out[role] = dst
        except Exception as e:  # noqa: BLE001
            logger.warning("custom_tools: bake %s/%s failed: %s", cfg.name, role, e)
    return out


# ---------------------------------------------------------------------------
# Registry mutation
# ---------------------------------------------------------------------------


def register_one(cfg: CustomToolConfig) -> bool:
    """Bake + register one custom tool. Returns True iff registered.
    Idempotent; replaces the prior entry for the same key.
    """
    baked = bake_one(cfg)
    if not baked or "body" not in baked:
        return False

    # Surface partial-jaw bake failures (uploaded but failed to bake);
    # otherwise the tool silently registers as has_jaws=False.
    cfg_has_left = cfg.jaw_left_path.exists()
    cfg_has_right = cfg.jaw_right_path.exists()
    baked_has_left = "jaw_left" in baked
    baked_has_right = "jaw_right" in baked
    if (cfg_has_left and not baked_has_left) or (
        cfg_has_right and not baked_has_right
    ):
        logger.warning(
            "custom_tools: tool %r baked with PARTIAL jaws — "
            "uploaded jaws exist on disk but bake failed "
            "(left: uploaded=%s baked=%s; right: uploaded=%s baked=%s); "
            "tool will register WITHOUT jaws. Check earlier log lines "
            "for the per-mesh ``bake ... failed`` message.",
            cfg.name,
            cfg_has_left, baked_has_left,
            cfg_has_right, baked_has_right,
        )
    try:
        from parol6 import tools as parol6_tools  # noqa: PLC0415
    except ImportError:
        logger.warning("custom_tools: parol6 unavailable; skipping registration")
        return False

    MeshSpec = parol6_tools.MeshSpec  # noqa: N806
    MeshRole = parol6_tools.MeshRole  # noqa: N806

    # Build the body + jaw mesh specs pointing at the baked filenames.
    meshes = [
        MeshSpec(file=_baked_filename(cfg.name, "body"), role=MeshRole.BODY),
    ]
    has_jaws = "jaw_left" in baked and "jaw_right" in baked
    if has_jaws:
        meshes.extend([
            MeshSpec(file=_baked_filename(cfg.name, "jaw_left"), role=MeshRole.JAW),
            MeshSpec(file=_baked_filename(cfg.name, "jaw_right"), role=MeshRole.JAW),
        ])

    # Build TCP transform 4×4.
    R_tcp = SciRot.from_euler("XYZ", cfg.tcp_rpy_rad).as_matrix()
    T_tcp = np.eye(4, dtype=np.float64)
    T_tcp[:3, :3] = R_tcp
    T_tcp[:3, 3] = np.asarray(cfg.tcp_origin_m, dtype=np.float64)

    motions: tuple = ()
    LinearMotion = parol6_tools.LinearMotion  # noqa: N806
    if has_jaws and cfg.jaw_travel_m > 0.0:
        motions = (
            LinearMotion(
                role=MeshRole.JAW,
                axis=cfg.jaw_axis,
                travel_m=float(cfg.jaw_travel_m),
                symmetric=bool(cfg.jaw_symmetric),
            ),
        )

    # Each variant's meshes list must include the body since parol6's
    # swap_tool_mesh substitutes the full list at variant-pick time.
    ToolVariant = parol6_tools.ToolVariant  # noqa: N806
    variant_specs: list = []
    for v in cfg.variants:
        v_left_path = baked.get(f"variant_{v.key}_jaw_left")
        v_right_path = baked.get(f"variant_{v.key}_jaw_right")
        if v_left_path is None or v_right_path is None:
            logger.info(
                "custom_tools: variant %s/%s has no jaws baked; skipping",
                cfg.name, v.key,
            )
            continue
        v_meshes = (
            MeshSpec(file=_baked_filename(cfg.name, "body"), role=MeshRole.BODY),
            MeshSpec(
                file=_baked_filename(cfg.name, f"variant_{v.key}_jaw_left"),
                role=MeshRole.JAW,
            ),
            MeshSpec(
                file=_baked_filename(cfg.name, f"variant_{v.key}_jaw_right"),
                role=MeshRole.JAW,
            ),
        )
        v_motions: tuple = ()
        if v.jaw_travel_m > 0.0:
            v_motions = (
                LinearMotion(
                    role=MeshRole.JAW,
                    axis=v.jaw_axis,
                    travel_m=float(v.jaw_travel_m),
                    symmetric=bool(v.jaw_symmetric),
                ),
            )
        variant_specs.append(
            ToolVariant(
                key=v.key,
                display_name=v.display_name or v.key,
                meshes=v_meshes,
                motions=v_motions,
                tcp_origin=tuple(cfg.tcp_origin_m),
                tcp_rpy=tuple(cfg.tcp_rpy_rad),
            )
        )

    # ``ToolConfig`` directly — gripper-specific configs add motor/encoder
    # fields we don't have. Works for static brackets + jaw grippers alike.
    ToolConfig = parol6_tools.ToolConfig  # noqa: N806
    config = ToolConfig(
        name=cfg.display_name or cfg.name,
        description=cfg.description or f"User-defined tool {cfg.name}",
        transform=T_tcp,
        meshes=tuple(meshes),
        motions=motions,
        variants=tuple(variant_specs),
    )

    key = f"custom:{cfg.name}"
    parol6_tools.register_tool(key, config)
    logger.info(
        "custom_tools: registered %s (%s) — body + %d jaw(s) + %d variant(s)",
        key, cfg.display_name or cfg.name,
        2 if has_jaws else 0, len(variant_specs),
    )
    # Refresh Robot.tools + notify subscribers so the gripper dropdown
    # picks up the new entry without a page reload.
    _refresh_active_robot_tools()
    _notify_tool_registry_changed()
    return True


def register_all() -> list[str]:
    """Discover + register every custom tool. Returns the list of
    successfully-registered registry keys (``custom:<name>``).
    """
    # Diagnostic: log caller stack to attribute multi-fire on cold start.
    import traceback  # noqa: PLC0415
    stack_frames = traceback.format_stack()
    caller_summary = "".join(stack_frames[-7:-1]).strip()
    logger.info("custom_tools: register_all called — caller stack:\n%s", caller_summary)
    ensure_root()
    registered: list[str] = []
    for cfg in list_configs():
        if register_one(cfg):
            registered.append(f"custom:{cfg.name}")
    if registered:
        logger.info(
            "custom_tools: %d registered: %s",
            len(registered), ", ".join(registered),
        )
    return registered


# ---------------------------------------------------------------------------
# One-shot SSG-48 + camera-bracket migration
# ---------------------------------------------------------------------------


_SSG48_MIGRATION_NAME: str = "ssg48_realsense"
_SSG48_SENTINEL_FILENAME: str = ".ssg48_realsense_migrated"


def _ssg48_merged_stl_path() -> Path | None:
    """Locate the merged SSG-48 + camera-bracket STL in the parol6-vision
    sibling clone. None when not present.
    """
    pv_root = (
        Path(__file__).resolve().parent.parent.parent.parent.parent / "parol6-vision"
    )
    candidate = pv_root / "parol6_vision" / "sim" / "meshes" / "ssg48_body_realsense.stl"
    return candidate if candidate.exists() else None


def auto_migrate_ssg48_with_bracket() -> bool:
    """Convert the legacy SSG-48 hijack into a ``custom:ssg48_realsense``
    tool, pre-baked from the merged SSG-48 + camera-bracket STL.
    Idempotent via a sentinel at ``<root>/.ssg48_realsense_migrated``.

    Pre-conditions: sentinel missing, merged STL present in the
    parol6-vision sibling clone, name not in use.

    Side effects on success: writes the new tool folder, registers
    ``custom:ssg48_realsense``, writes the sentinel, and also writes
    the legacy ``ssg48_body_realsense.stl`` into parol6's mesh dir
    (collision.py still resolves the body by that filename).

    Returns True when a new tool is created.
    """
    ensure_root()
    sentinel = CUSTOM_TOOLS_ROOT / _SSG48_SENTINEL_FILENAME
    if sentinel.exists():
        return False
    if _SSG48_MIGRATION_NAME in list_tool_names():
        # Tool already exists; keep it and write the sentinel.
        sentinel.touch(exist_ok=True)
        return False

    src_body = _ssg48_merged_stl_path()
    if src_body is None:
        # No merged STL — skip without writing the sentinel so a later
        # checkout that brings the file in still runs the migration.
        logger.info(
            "ssg48 migration: merged STL not found; skipping",
        )
        return False

    try:
        import trimesh  # noqa: PLC0415
        from importlib.resources import files as pkg_files  # noqa: PLC0415
        from scipy.spatial.transform import Rotation as SciRot  # noqa: PLC0415
    except ImportError as e:
        logger.info("ssg48 migration: deps unavailable (%s); skipping", e)
        return False

    # Fit constants live in ``constants.py`` so this matches the legacy
    # hijack's body bytes-for-bytes.
    from . import constants as _c  # noqa: PLC0415

    fit_scale = float(_c._MERGED_STL_FIT_SCALE)
    fit_translate = np.asarray(_c._MERGED_STL_FIT_TRANSLATE_M, dtype=np.float64)
    user_translate = np.asarray(_c._MERGED_STL_TRANSLATE_M, dtype=np.float64)
    user_rpy = np.asarray(_c._MERGED_STL_RPY_RAD, dtype=np.float64)

    try:
        mesh = trimesh.load(str(src_body), force="mesh")
    except Exception as e:  # noqa: BLE001
        logger.warning("ssg48 migration: trimesh load failed: %s", e)
        return False

    # Step 1: fit transform (scale + translate).
    T_fit = np.eye(4, dtype=np.float64)
    T_fit[:3, :3] = np.eye(3) * fit_scale
    T_fit[:3, 3] = fit_translate
    mesh.apply_transform(T_fit)

    # Step 2: user RPY + translate on top of the fit.
    if any(abs(a) > 1e-9 for a in user_rpy):
        T_extra = np.eye(4, dtype=np.float64)
        T_extra[:3, :3] = SciRot.from_euler("XYZ", user_rpy).as_matrix()
        T_extra[:3, 3] = user_translate
        mesh.apply_transform(T_extra)
    elif np.linalg.norm(user_translate) > 1e-9:
        T_extra = np.eye(4, dtype=np.float64)
        T_extra[:3, 3] = user_translate
        mesh.apply_transform(T_extra)

    # Write the body into the new custom-tool folder.
    target_folder = CUSTOM_TOOLS_ROOT / _SSG48_MIGRATION_NAME
    target_folder.mkdir(parents=True, exist_ok=True)
    mesh.export(str(target_folder / BODY_STL_NAME), file_type="stl")

    # Find parol6's mesh dir for the stock SSG-48 finger STLs.
    try:
        parol6_root = Path(str(pkg_files("parol6")))
        mesh_dir = parol6_root / "urdf_model" / "meshes"
    except Exception as e:  # noqa: BLE001
        logger.warning("ssg48 migration: parol6 mesh dir unreachable: %s", e)
        return False

    # Stock jaw STLs are already in flange metres. Default slots get the
    # finger pair; per-variant slots get both finger and pinch.
    for stock_name, dst_name in (
        ("ssg48_finger_left.stl", JAW_LEFT_STL_NAME),
        ("ssg48_finger_right.stl", JAW_RIGHT_STL_NAME),
    ):
        src = mesh_dir / stock_name
        if src.exists():
            shutil.copyfile(str(src), str(target_folder / dst_name))
    for variant_key, stock_left, stock_right in (
        ("finger", "ssg48_finger_left.stl", "ssg48_finger_right.stl"),
        ("pinch", "ssg48_pinch_left.stl", "ssg48_pinch_right.stl"),
    ):
        for stock_name, side in (
            (stock_left, "left"),
            (stock_right, "right"),
        ):
            src = mesh_dir / stock_name
            if src.exists():
                shutil.copyfile(
                    str(src),
                    str(target_folder / f"variant_{variant_key}_jaw_{side}.stl"),
                )

    # collision.py still resolves the body via the legacy filename;
    # write a copy under that name to avoid a parallel refactor.
    try:
        legacy_path = mesh_dir / "ssg48_body_realsense.stl"
        mesh.export(str(legacy_path), file_type="stl")
        logger.info("ssg48 migration: wrote legacy %s", legacy_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("ssg48 migration: legacy bake failed: %s", e)

    cfg = CustomToolConfig(
        name=_SSG48_MIGRATION_NAME,
        display_name="SSG-48 + RealSense bracket",
        description="SSG-48 body with the RealSense camera bracket fused in.",
        mesh_translate_m=(0.0, 0.0, 0.0),
        mesh_rpy_rad=(0.0, 0.0, 0.0),
        mesh_scale=1.0,
        # Same flange→TCP and jaw motion as parol6's built-in SSG-48.
        tcp_origin_m=(0.0, 0.0, -0.105),
        tcp_rpy_rad=(0.0, 0.0, 0.0),
        jaw_travel_m=0.024,
        jaw_axis=(0.0, 1.0, 0.0),
        jaw_symmetric=True,
        # Controller doesn't know ``custom:`` keys; proxy through SSG-48.
        proxy_tool_key="SSG-48",
        has_camera=True,
        # Match parol6's SSG-48 ToolVariant entries (finger / pinch).
        variants=[
            CustomToolVariant(
                key="finger", display_name="Finger",
                jaw_travel_m=0.024,
                jaw_axis=(0.0, 1.0, 0.0),
                jaw_symmetric=True,
            ),
            CustomToolVariant(
                key="pinch", display_name="Pinch",
                jaw_travel_m=0.024,
                jaw_axis=(0.0, 1.0, 0.0),
                jaw_symmetric=True,
            ),
        ],
    )
    save_config(cfg)
    cfg.has_body = cfg.body_path.exists()
    cfg.has_jaws = (
        cfg.jaw_left_path.exists() and cfg.jaw_right_path.exists()
    )

    # Register so the dropdown picks up the new tool without a restart.
    register_one(cfg)

    sentinel.touch()
    logger.info(
        "ssg48 migration complete — custom:%s created, sentinel %s written",
        _SSG48_MIGRATION_NAME, sentinel,
    )
    return True


# ---------------------------------------------------------------------------
# Live-refresh hooks — re-load tool meshes in the URDF scene without a
# round-trip through the gripper-panel dropdown.
# ---------------------------------------------------------------------------


def _active_gui_tool_key() -> str | None:
    """GUI's logical active tool key (custom: prefix preserved). Reads
    ``app.storage.general['selected_tool']`` first so the user's choice
    wins over the controller's built-in-only broadcast.
    """
    try:
        from nicegui import app  # noqa: PLC0415

        stored = app.storage.general.get("selected_tool")
        if stored:
            return str(stored)
    except Exception:  # noqa: BLE001
        pass
    try:
        from waldo_commander.state import robot_state  # noqa: PLC0415

        return getattr(robot_state, "tool_key", None)
    except Exception:  # noqa: BLE001
        return None


def is_active_tool(name: str) -> bool:
    """True when ``custom:<name>`` is the GUI's active tool."""
    return _active_gui_tool_key() == f"custom:{name}"


# Built-in tool keys with a calibratable camera. parol6's ``ToolConfig``
# carries no ``has_camera`` field, hence the whitelist.
BUILTIN_CAMERA_BEARING_TOOLS: frozenset[str] = frozenset({"MSG"})


def is_camera_bearing(tool_key: str | None) -> bool:
    """True when ``tool_key`` is camera-bearing (custom tool with
    ``has_camera``, or a built-in in ``BUILTIN_CAMERA_BEARING_TOOLS``).
    """
    if not tool_key:
        return False
    if tool_key in BUILTIN_CAMERA_BEARING_TOOLS:
        return True
    if tool_key.startswith("custom:"):
        cfg = load_config(tool_key[len("custom:"):])
        return bool(cfg and cfg.has_camera)
    return False


def active_tool_is_camera_bearing() -> bool:
    """``is_camera_bearing(_active_gui_tool_key())``."""
    return is_camera_bearing(_active_gui_tool_key())


# ---------------------------------------------------------------------------
# Per-tool camera-config override
# ---------------------------------------------------------------------------


# Keys overridable per-tool; the rest are workspace properties.
_PER_TOOL_OVERRIDABLE_KEYS: frozenset[str] = frozenset({
    "intr_fx", "intr_fy", "intr_cx", "intr_cy",
    "intr_width", "intr_height",
    "cam_mount_translate_mm", "cam_mount_tilt_deg",
})


# ---------------------------------------------------------------------------
# Per-tool override storage layer.
#
# Persistent: ``app.storage.user[f"calib_tool_{tool_key}_{setting_key}"]``.
# A thread-safe runtime cache mirrors it; worker threads can't read
# app.storage.user directly. Call ``prime_per_tool_overrides_cache()`` from
# a request thread before spawning workers.
#
# Resolution priority for overridable keys:
#   1. Per-tool storage override (active tool)
#   2. CustomToolConfig field (legacy fallback)
#   3. Built-in tool defaults
#   4. Global runtime override
#   5. Module default
# ---------------------------------------------------------------------------


_per_tool_runtime_cache: dict[str, dict[str, Any]] = {}


def _per_tool_storage_key(tool_key: str, setting_key: str) -> str:
    return f"calib_tool_{tool_key}_{setting_key}"


def _coerce_per_tool_value(setting_key: str, raw: Any) -> Any:
    """Restore tuple shape on cam_mount keys after a JSON round-trip."""
    if raw is None:
        return None
    if setting_key in ("cam_mount_translate_mm", "cam_mount_tilt_deg"):
        if isinstance(raw, list | tuple):
            try:
                return (float(raw[0]), float(raw[1]), float(raw[2]))
            except (TypeError, ValueError, IndexError):
                return None
    return raw


def _per_tool_storage_dict():
    """Return ``app.storage.general`` (NOT ``.user``) — works from any
    context, including worker threads. Returns None if NiceGUI isn't
    available or storage isn't ready.

    Per-tool overrides used to live in ``app.storage.user``, but that
    requires a UI request context which worker threads (localise /
    calibration / hover) don't have. Writes from those threads failed
    silently with "app.storage.user can only be used within a UI
    context", losing every calibrated-mount auto-save. ``.general`` is
    installation-global, which is the correct scope for per-tool
    calibration data anyway (the calibrated mount of a physical
    gripper isn't tied to a browser session).
    """
    try:
        from nicegui import app  # noqa: PLC0415

        return app.storage.general
    except Exception:  # noqa: BLE001
        return None


def prime_per_tool_overrides_cache() -> None:
    """Refresh ``_per_tool_runtime_cache`` from ``app.storage.general``.
    Idempotent. Safe to call from any context.
    """
    store = _per_tool_storage_dict()
    if store is None:
        logger.debug("prime_per_tool_overrides_cache: storage unavailable")
        return
    new_cache: dict[str, dict[str, Any]] = {}
    prefix = "calib_tool_"
    suffixes = tuple(
        ("_" + s, s) for s in _PER_TOOL_OVERRIDABLE_KEYS
    )
    try:
        items = list(store.items())
    except Exception as e:  # noqa: BLE001
        logger.debug("prime_per_tool_overrides_cache: items iteration failed (%s)", e)
        return
    for storage_key, raw in items:
        if not isinstance(storage_key, str) or not storage_key.startswith(prefix):
            continue
        rest = storage_key[len(prefix):]
        for suffix, setting_key in suffixes:
            if rest.endswith(suffix):
                tool_key = rest[: -len(suffix)]
                value = _coerce_per_tool_value(setting_key, raw)
                if value is None:
                    continue
                new_cache.setdefault(tool_key, {})[setting_key] = value
                break
    _per_tool_runtime_cache.clear()
    _per_tool_runtime_cache.update(new_cache)


def get_per_tool_override(tool_key: str, setting_key: str) -> Any:
    """Per-tool override for ``setting_key`` on ``tool_key`` (None unset).
    Reads the cache first, falls back to persistent storage.
    """
    if not tool_key or setting_key not in _PER_TOOL_OVERRIDABLE_KEYS:
        return None
    cached = _per_tool_runtime_cache.get(tool_key, {}).get(setting_key)
    if cached is not None:
        return cached
    store = _per_tool_storage_dict()
    if store is None:
        return None
    try:
        raw = store.get(_per_tool_storage_key(tool_key, setting_key))
    except Exception:  # noqa: BLE001
        return None
    return _coerce_per_tool_value(setting_key, raw)


def set_per_tool_override(
    tool_key: str, setting_key: str, value: Any,
) -> bool:
    """Persist a per-tool override + mirror into the runtime cache. Pass
    ``value=None`` to clear. Returns True on storage write success.
    """
    if not tool_key or setting_key not in _PER_TOOL_OVERRIDABLE_KEYS:
        return False
    # Update the cache first so subsequent reads see the new value even
    # if the persistent write fails.
    if value is None:
        bucket = _per_tool_runtime_cache.get(tool_key)
        if bucket is not None:
            bucket.pop(setting_key, None)
            if not bucket:
                _per_tool_runtime_cache.pop(tool_key, None)
    else:
        normalised = _coerce_per_tool_value(setting_key, value)
        _per_tool_runtime_cache.setdefault(tool_key, {})[setting_key] = normalised

    store = _per_tool_storage_dict()
    if store is None:
        return False
    skey = _per_tool_storage_key(tool_key, setting_key)
    try:
        if value is None:
            store.pop(skey, None)
        else:
            if isinstance(value, tuple):
                store[skey] = list(value)
            else:
                store[skey] = value
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "custom_tools: set_per_tool_override(%s, %s) persistent write failed: %s",
            tool_key, setting_key, e,
        )
        return False


def has_per_tool_override(tool_key: str) -> bool:
    """True iff any overridable key has a stored value for ``tool_key``."""
    return any(
        get_per_tool_override(tool_key, k) is not None
        for k in _PER_TOOL_OVERRIDABLE_KEYS
    )


def clear_per_tool_overrides(tool_key: str) -> None:
    """Remove every stored per-tool override for ``tool_key``."""
    for k in _PER_TOOL_OVERRIDABLE_KEYS:
        set_per_tool_override(tool_key, k, None)


def active_tool_override(key: str) -> Any:
    """Active tool's per-tool override for ``key``, or None. Layers:
    storage → CustomToolConfig field → built-in defaults.
    """
    if key not in _PER_TOOL_OVERRIDABLE_KEYS:
        return None
    tool_key = _active_gui_tool_key()
    if not tool_key:
        return None

    # Layer 1: storage.
    storage_value = get_per_tool_override(tool_key, key)
    if storage_value is not None:
        return storage_value

    # Layer 2: CustomToolConfig field (legacy fallback).
    if tool_key.startswith("custom:"):
        cfg = load_config(tool_key[len("custom:"):])
        if cfg is None or not cfg.has_camera:
            return None
        return getattr(cfg, key, None)

    # Layer 3: built-in tool defaults so stock tools work out of the box.
    builtin = _BUILTIN_TOOL_DEFAULTS.get(tool_key)
    if builtin is not None and key in builtin:
        return builtin[key]

    return None


# Built-in camera-bearing tool defaults. MSG AI gripper: OV9732 sensor,
# 1280x720, f=2.84 mm, 3.0 um pitch → fx=fy≈947 px; mount geometry from
# CAD inspection of the MSG body STL.
_BUILTIN_TOOL_DEFAULTS: dict[str, dict[str, Any]] = {
    "MSG": {
        "intr_fx": 947.0,
        "intr_fy": 947.0,
        "intr_cx": 640.0,
        "intr_cy": 360.0,
        "intr_width": 1280,
        "intr_height": 720,
        "cam_mount_translate_mm": (50.0, 0.0, -45.0),
        "cam_mount_tilt_deg": (225.0, 0.0, 90.0),
    },
}


def update_active_tool_calibrated_mount(
    cam_mount_translate_mm: tuple[float, float, float],
    cam_mount_tilt_deg: tuple[float, float, float],
) -> bool:
    """Persist a calibrated mount as a per-tool override on the active
    tool. Returns True iff a write happened.
    """
    tool_key = _active_gui_tool_key()
    if not tool_key:
        return False
    # Only meaningful for camera-bearing tools.
    if not is_camera_bearing(tool_key):
        logger.debug(
            "calibrated mount not persisted — active tool %r isn't camera-bearing",
            tool_key,
        )
        return False
    translate = (
        float(cam_mount_translate_mm[0]),
        float(cam_mount_translate_mm[1]),
        float(cam_mount_translate_mm[2]),
    )
    tilt = (
        float(cam_mount_tilt_deg[0]),
        float(cam_mount_tilt_deg[1]),
        float(cam_mount_tilt_deg[2]),
    )
    ok_t = set_per_tool_override(tool_key, "cam_mount_translate_mm", translate)
    ok_r = set_per_tool_override(tool_key, "cam_mount_tilt_deg", tilt)
    if ok_t and ok_r:
        logger.info(
            "saved calibrated mount as per-tool override on %s "
            "(translate=%s mm, tilt=%s deg)",
            tool_key, translate, tilt,
        )
        return True
    return False


def live_refresh_active_tool(name: str) -> bool:
    """If ``custom:<name>`` is active, re-call ``apply_tool`` so a fresh
    bake appears immediately. Returns True when a refresh was issued.
    """
    if not is_active_tool(name):
        return False
    try:
        from waldo_commander.state import ui_state  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    scene = getattr(ui_state, "urdf_scene", None)
    if scene is None:
        return False
    full_key = f"custom:{name}"
    try:
        scene.apply_tool(full_key, variant_key=None)
        logger.info("custom_tools: live-refreshed %s in URDF scene", full_key)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("custom_tools: live refresh failed for %s: %s", full_key, e)
        return False


# ---------------------------------------------------------------------------
# Import-from-existing-tool — fork any registered tool into a custom one
# ---------------------------------------------------------------------------


def list_registered_tools() -> list[tuple[str, str]]:
    """``[(key, display_name), ...]`` for every tool in
    ``parol6.tools._TOOL_REGISTRY``, excluding ``"NONE"``.
    """
    try:
        from parol6 import tools as parol6_tools  # noqa: PLC0415
    except ImportError:
        return []
    out: list[tuple[str, str]] = []
    for key, cfg in parol6_tools._TOOL_REGISTRY.items():
        if key == "NONE":
            continue
        out.append((key, getattr(cfg, "name", key)))
    return sorted(out, key=lambda p: p[1])


def import_from_registered(source_key: str, target_name: str) -> CustomToolConfig | None:
    """Fork ``source_key`` into a new custom tool. Copies the source's
    body + jaw STLs from parol6's mesh dir, extracts TCP + jaw motion,
    and writes an identity-placement ``config.json``. Returns None when
    the source is missing, has no body, the target exists, or a copy
    fails.
    """
    if target_name in list_tool_names():
        logger.warning("import: target %r already exists", target_name)
        return None
    try:
        from parol6 import tools as parol6_tools  # noqa: PLC0415
    except ImportError:
        return None
    src = parol6_tools._TOOL_REGISTRY.get(source_key)
    if src is None:
        logger.warning("import: no tool with key %r", source_key)
        return None
    try:
        from importlib.resources import files as pkg_files  # noqa: PLC0415
        parol6_root = Path(str(pkg_files("parol6")))
        mesh_dir = parol6_root / "urdf_model" / "meshes"
    except Exception as e:  # noqa: BLE001
        logger.warning("import: could not locate parol6 mesh dir: %s", e)
        return None

    # Locate the meshes — pick the first BODY + first 2 JAWS by role.
    MeshRole = parol6_tools.MeshRole  # noqa: N806
    body_spec = None
    jaw_specs: list = []
    for spec in src.meshes:
        if spec.role == MeshRole.BODY and body_spec is None:
            body_spec = spec
        elif spec.role == MeshRole.JAW:
            jaw_specs.append(spec)
    if body_spec is None:
        logger.warning("import: source %r has no body mesh", source_key)
        return None

    def _resolve(filename: str) -> Path:
        """Strip the ``?v=<mtime>`` cache-bust suffix added by the
        SSG-48 hijack so the URL becomes a usable filesystem path."""
        plain = filename.split("?", 1)[0]
        return mesh_dir / plain

    # Copy STLs into the new custom-tool folder.
    target_folder = CUSTOM_TOOLS_ROOT / target_name
    target_folder.mkdir(parents=True, exist_ok=True)
    body_src = _resolve(body_spec.file)
    if not body_src.exists():
        logger.warning("import: body STL missing: %s", body_src)
        return None
    shutil.copyfile(str(body_src), str(target_folder / BODY_STL_NAME))
    # jaw_specs[0]/[1] map to right/left by parol6's convention.
    if len(jaw_specs) >= 2:
        for jaw_spec, dst_name in (
            (jaw_specs[0], JAW_RIGHT_STL_NAME),
            (jaw_specs[1], JAW_LEFT_STL_NAME),
        ):
            jaw_src = _resolve(jaw_spec.file)
            if jaw_src.exists():
                shutil.copyfile(str(jaw_src), str(target_folder / dst_name))

    # Hoist the LinearMotion lookup so the variants loop + tool-level
    # extraction below share it.
    LinearMotion = getattr(parol6_tools, "LinearMotion", None)

    # Extract variant-specific jaws + motion (body is shared, already copied).
    variants_out: list[CustomToolVariant] = []
    for src_variant in getattr(src, "variants", ()) or ():
        vkey = str(getattr(src_variant, "key", ""))
        if not vkey:
            continue
        v_jaw_specs: list = []
        for spec in getattr(src_variant, "meshes", ()):
            if spec.role == MeshRole.JAW:
                v_jaw_specs.append(spec)
        if len(v_jaw_specs) < 2:
            logger.info(
                "import: variant %r has %d jaws (need 2); skipping",
                vkey, len(v_jaw_specs),
            )
            continue
        for spec, side in (
            (v_jaw_specs[0], "right"),
            (v_jaw_specs[1], "left"),
        ):
            jaw_src = _resolve(spec.file)
            if jaw_src.exists():
                dst = target_folder / f"variant_{vkey}_jaw_{side}.stl"
                shutil.copyfile(str(jaw_src), str(dst))
        # Variant jaw motion (first LinearMotion if any).
        v_travel = 0.0
        v_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
        v_symmetric = True
        for motion in getattr(src_variant, "motions", ()) or ():
            if LinearMotion is not None and isinstance(motion, LinearMotion):
                v_travel = float(getattr(motion, "travel_m", 0.0))
                axis_raw = getattr(motion, "axis", (0.0, 1.0, 0.0))
                v_axis = (float(axis_raw[0]), float(axis_raw[1]), float(axis_raw[2]))
                v_symmetric = bool(getattr(motion, "symmetric", True))
                break
        variants_out.append(
            CustomToolVariant(
                key=vkey,
                display_name=str(getattr(src_variant, "display_name", "") or vkey),
                jaw_travel_m=v_travel,
                jaw_axis=v_axis,
                jaw_symmetric=v_symmetric,
            )
        )

    # Extract flange→TCP transform.
    transform = np.asarray(src.transform, dtype=np.float64)
    tcp_origin = tuple(float(v) for v in transform[:3, 3])
    tcp_rpy = tuple(
        float(v) for v in SciRot.from_matrix(transform[:3, :3]).as_euler("XYZ")
    )

    # Take the first LinearMotion as jaw motion (custom_tools models one).
    jaw_travel_m = 0.0
    jaw_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    jaw_symmetric = True
    for motion in getattr(src, "motions", ()) or ():
        if LinearMotion is not None and isinstance(motion, LinearMotion):
            jaw_travel_m = float(getattr(motion, "travel_m", 0.0))
            axis_raw = getattr(motion, "axis", (0.0, 1.0, 0.0))
            jaw_axis = (
                float(axis_raw[0]), float(axis_raw[1]), float(axis_raw[2]),
            )
            jaw_symmetric = bool(getattr(motion, "symmetric", True))
            break

    cfg = CustomToolConfig(
        name=target_name,
        display_name=getattr(src, "name", target_name),
        description=(
            f"Imported from {source_key}. "
            f"{getattr(src, 'description', '') or ''}".strip()
        ),
        # Source STLs are already in flange coords.
        mesh_translate_m=(0.0, 0.0, 0.0),
        mesh_rpy_rad=(0.0, 0.0, 0.0),
        mesh_scale=1.0,
        tcp_origin_m=(tcp_origin[0], tcp_origin[1], tcp_origin[2]),
        tcp_rpy_rad=(tcp_rpy[0], tcp_rpy[1], tcp_rpy[2]),
        jaw_travel_m=jaw_travel_m,
        jaw_axis=jaw_axis,
        jaw_symmetric=jaw_symmetric,
        # Proxy motor commands through the source built-in tool.
        proxy_tool_key=str(source_key),
        has_camera=is_camera_bearing(source_key),
        variants=variants_out,
    )
    save_config(cfg)
    # Refresh derived flags now that the STLs exist on disk.
    cfg.has_body = cfg.body_path.exists()
    cfg.has_jaws = cfg.jaw_left_path.exists() and cfg.jaw_right_path.exists()
    for v in cfg.variants:
        v.has_jaws = (
            cfg.variant_jaw_path(v.key, "left").exists()
            and cfg.variant_jaw_path(v.key, "right").exists()
        )
    logger.info(
        "import: %r forked from %s (body=%s, jaws=%s, variants=%d, tcp=%s)",
        target_name, source_key, cfg.has_body, cfg.has_jaws,
        len(cfg.variants), tcp_origin,
    )
    return cfg


async def select_as_active(name: str, proxy_tool_key: str = "") -> bool:
    """Make ``custom:<name>`` the active tool.

    Always: local apply via ``active_robot.set_active_tool`` +
    ``urdf_scene.apply_tool`` (FK/IK + 3D scene use the custom TCP
    and meshes).

    Optional: with ``proxy_tool_key`` non-empty, send
    ``client.select_tool(proxy_tool_key)`` so the controller acts as
    that built-in (needed for jaw motion / motor commands).

    Returns True iff the local apply succeeded; the proxy call is
    best-effort.
    """
    full_key = f"custom:{name}"
    try:
        from waldo_commander.state import ui_state  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False

    # ---- LOCAL APPLY (always) ------------------------------------------
    local_ok = False
    try:
        ui_state.active_robot.set_active_tool(full_key, variant_key=None)
        local_ok = True
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "custom_tools: active_robot.set_active_tool(%s) failed: %s",
            full_key, e,
        )
    scene = getattr(ui_state, "urdf_scene", None)
    if scene is not None:
        try:
            scene.apply_tool(full_key, variant_key=None)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "custom_tools: urdf_scene.apply_tool(%s) failed: %s",
                full_key, e,
            )

    # Persist so a page reload restores this tool; otherwise the
    # controller's broadcast (the proxy built-in) wins on reload.
    try:
        from nicegui import app  # noqa: PLC0415

        app.storage.general["selected_tool"] = full_key
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "custom_tools: could not persist selected_tool=%s (%s)",
            full_key, e,
        )

    # ---- CONTROLLER PROXY (optional) -----------------------------------
    if proxy_tool_key:
        # ``ui_state.control_panel`` raises RuntimeError when uninitialised.
        client = None
        try:
            panel = ui_state.control_panel
            client = getattr(panel, "client", None)
        except Exception as e:  # noqa: BLE001
            logger.info(
                "custom_tools: control_panel unavailable (%s); skipping "
                "proxy select_tool(%s)",
                e, proxy_tool_key,
            )
        if client is None:
            logger.info(
                "custom_tools: no client to send proxy select_tool(%s); "
                "controller stays at previous tool",
                proxy_tool_key,
            )
        else:
            try:
                await client.select_tool(proxy_tool_key, variant_key="")
                logger.info(
                    "custom_tools: %s active locally; controller proxied to %s",
                    full_key, proxy_tool_key,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "custom_tools: proxy select_tool(%s) failed: %s — "
                    "visualisation/IK fine, motor control unchanged",
                    proxy_tool_key, e,
                )

    return local_ok
