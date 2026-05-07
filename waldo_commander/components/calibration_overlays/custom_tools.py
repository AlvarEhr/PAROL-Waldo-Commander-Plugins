"""Custom tool ingestion — drop-in STLs become user-defined tools.

A user with a custom gripper / camera bracket / end-effector drops their
STLs into ``~/.waldo-commander/custom_tools/<name>/`` along with a
``config.json`` describing the placement transform + TCP transform +
optional jaw motion. On waldo-commander startup we:

1. Scan the directory for ``config.json`` files.
2. For each, apply the placement transform to the user's STL and bake
   the result into parol6's mesh dir under a unique filename
   (``custom_<name>_body.stl``, ``custom_<name>_jaw_left.stl``, ...).
3. Register a new entry in ``parol6.tools._TOOL_REGISTRY`` so the
   gripper dropdown picks it up alongside the built-in tools.

Folder layout:

    ~/.waldo-commander/custom_tools/<name>/
      config.json          # placement + TCP + jaw motion + meta
      body.stl             # required — gripper body / bracket
      jaw_left.stl         # optional — left jaw if the tool has fingers
      jaw_right.stl        # optional — right jaw

config.json schema (every numeric field optional, defaults shown):

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

The placement transform fields (translate / rpy / scale) are what the
calibration panel's "Custom tools" UI iterates on — they shift the STL
relative to the flange origin until the mesh visually aligns with the
URDF arm.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRot

logger = logging.getLogger(__name__)


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

    def to_json(self) -> str:
        # has_body / has_jaws are derived (don't persist them).
        payload = {
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
        }
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
    )
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
    """Remove a custom tool's folder + config + STLs from disk."""
    folder = CUSTOM_TOOLS_ROOT / name
    if folder.exists():
        shutil.rmtree(folder)


def import_stl(name: str, role: str, src: Path) -> None:
    """Copy a user-provided STL into the tool's folder under the
    canonical name. ``role`` is one of ``"body"``, ``"jaw_left"``,
    ``"jaw_right"``.
    """
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


# ---------------------------------------------------------------------------
# Mesh-aware helpers — auto unit detection + snap to bbox / flange face
# ---------------------------------------------------------------------------


def _load_trimesh(stl_path: Path) -> Any | None:
    """Best-effort load with trimesh; None on failure."""
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
    """Heuristic: examine the STL's bbox and suggest a scale factor that
    converts its units to metres. Returns ``(scale, label)``.

    * bbox.max > 1.0 m → likely millimetres → scale = 0.001
    * bbox.max > 0.0254 m AND < 1.0 m → likely already metres → scale = 1.0
    * bbox.max < 0.001 m → likely 1000× over-scaled → scale = 1000.0
    * bbox.max in (0.001, 0.0254) → ambiguous (very small in m, plausible
      in cm) → default to 1.0 with an "unknown" label so the user picks.

    Real STL units are not embedded in the file; this heuristic matches
    the most common cases (CAD exports in mm, modeling tools in m).
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
        # Smaller than an inch — could be metres-of-tiny-thing or
        # mm-of-mid-thing. Mark as ambiguous; user override expected.
        return (1.0, f"ambiguous (bbox max {bbox_max:.4f} units)")
    return (1.0, f"m (bbox max {bbox_max:.3f} units)")


def snap_to_bbox_centre(stl_path: Path) -> tuple[float, float, float] | None:
    """Compute the translate-in-metres that would move the mesh's bbox
    centre to the origin. Returns ``-bbox.centre × scale``; caller
    multiplies by their desired scale or hands raw to the placement
    transform.

    Returns None if the mesh can't be loaded.
    """
    mesh = _load_trimesh(stl_path)
    if mesh is None:
        return None
    centre = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    return (-float(centre[0]), -float(centre[1]), -float(centre[2]))


def snap_flange_face_to_origin(
    stl_path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Find the largest planar facet whose normal is closest to ±Z, then
    return ``(translate, rpy_rad)`` that would put that face's centroid
    at the origin with its outward normal aligned to flange +Z.

    Approach (trimesh):

    1. ``mesh.facets`` groups co-planar faces — pick the largest facet
       group by total area.
    2. From that group's average normal, compute a rotation that maps the
       normal to flange +Z.
    3. Apply rotation to the group's centroid; the negative of that
       gives the translate to bring the rotated centroid to origin.

    The "flange face" is the planar surface where the gripper bolts onto
    the robot — usually the largest flat disc/square facing one of the
    cardinal axes. This snap is the right move when the user dropped in
    a mesh whose flange-face is in some random orientation.

    Returns None if the mesh has no clear planar facet (rare for
    machined parts) or fails to load.
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
    """Tier-3 snap: find the largest circular hole in the largest planar
    facet (typically the flange face's mount bore) and return the
    translate that would put its centre at the origin.

    Approach: for the largest facet (assumed flange face), get its
    boundary loops. For each interior loop, fit a circle in 2D (after
    projecting to the facet's plane). Pick the loop with the LARGEST
    fitted-circle radius; return the negative of its centre as the snap
    translate.

    Returns None if the mesh has no clear circular hole, or trimesh
    can't extract the facet boundary. This is a stretch goal — a fair
    chunk of the time it returns None and the user falls back to the
    flange-face snap.
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
        # Reject loops whose points don't actually look circular —
        # residual std > 10% of radius is too noisy to call a circle.
        residuals = np.hypot(u - cx, v - cy) - r
        if r <= 0 or float(np.std(residuals)) > 0.1 * r:
            continue
        # Reconstruct 3D centre.
        loop_centroid_3d = loop_arr.mean(axis=0)
        # The fitted circle centre in 3D = nearest-plane projection of
        # the in-plane (cx, cy) anchored on the loop's mean depth.
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
    """Apply the placement transform to each STL (body + jaws) and write
    the results into parol6's mesh dir under unique names.

    Returns ``{role: dst_path}`` on success (where role is "body",
    "jaw_left", "jaw_right" — only roles that actually have a source
    STL appear in the dict), or None on hard failure (parol6 mesh dir
    unreachable, body STL missing, etc.).

    Idempotent — re-baking with the same cfg produces the same on-disk
    bytes (modulo trimesh's internal float rounding).
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
    for role, src_path in (
        ("body", cfg.body_path),
        ("jaw_left", cfg.jaw_left_path),
        ("jaw_right", cfg.jaw_right_path),
    ):
        if not src_path.exists():
            continue
        try:
            mesh = trimesh.load(str(src_path), force="mesh")
            mesh.apply_transform(T)
            dst = mesh_dir / _baked_filename(cfg.name, role)
            mesh.export(str(dst))
            out[role] = dst
        except Exception as e:  # noqa: BLE001
            logger.warning("custom_tools: bake %s/%s failed: %s", cfg.name, role, e)
    return out


# ---------------------------------------------------------------------------
# Registry mutation
# ---------------------------------------------------------------------------


def register_one(cfg: CustomToolConfig) -> bool:
    """Bake + register one custom tool in ``parol6.tools._TOOL_REGISTRY``.

    Returns True iff the tool ended up registered. Idempotent — calling
    this for an already-registered key replaces the prior entry.
    """
    baked = bake_one(cfg)
    if not baked or "body" not in baked:
        return False
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
    if has_jaws and cfg.jaw_travel_m > 0.0:
        LinearMotion = parol6_tools.LinearMotion  # noqa: N806
        motions = (
            LinearMotion(
                role=MeshRole.JAW,
                axis=cfg.jaw_axis,
                travel_m=float(cfg.jaw_travel_m),
                symmetric=bool(cfg.jaw_symmetric),
            ),
        )

    # Register against ``ToolConfig`` directly — it's a concrete
    # dataclass in parol6.tools (the gripper-specific configs add motor
    # / encoder fields we don't have for a generic custom tool). Motions
    # attach when jaws + travel are configured. This keeps custom-tool
    # support generic — works for static brackets AND for grippers
    # with jaws.
    ToolConfig = parol6_tools.ToolConfig  # noqa: N806
    config = ToolConfig(
        name=cfg.display_name or cfg.name,
        description=cfg.description or f"User-defined tool {cfg.name}",
        transform=T_tcp,
        meshes=tuple(meshes),
        motions=motions,
    )

    key = f"custom:{cfg.name}"
    parol6_tools.register_tool(key, config)
    logger.info(
        "custom_tools: registered %s (%s) — body + %d jaw(s)",
        key, cfg.display_name or cfg.name, 2 if has_jaws else 0,
    )
    return True


def register_all() -> list[str]:
    """Discover + register every custom tool. Returns the list of
    successfully-registered registry keys (``custom:<name>``).
    """
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
    """Locate parol6-vision's merged SSG-48 + camera bracket STL relative
    to the package install. Returns None when the sibling clone isn't
    where we expect (e.g. fresh checkout without parol6-vision next door).
    """
    pv_root = (
        Path(__file__).resolve().parent.parent.parent.parent.parent / "parol6-vision"
    )
    candidate = pv_root / "parol6_vision" / "sim" / "meshes" / "ssg48_body_realsense.stl"
    return candidate if candidate.exists() else None


def auto_migrate_ssg48_with_bracket() -> bool:
    """One-shot migration: convert the historical SSG-48 hijack into a
    user-defined ``custom:ssg48_realsense`` tool, pre-baked from the merged
    SSG-48 + camera-bracket STL with the same fit constants the hijack
    used. Idempotent via a sentinel file at
    ``~/.waldo-commander/custom_tools/.ssg48_realsense_migrated``.

    Why a migration instead of a permanent hijack: the hijack hardcodes
    one user's CAD setup (Alvar's merged STL) into the SSG-48 entry,
    which is wrong for upstream code. After this migration the regular
    "SSG-48" entry stays as Jepson's stock body, and the camera-bracket
    setup lives as a normal custom tool the user can iterate via the UI.

    Pre-conditions for the migration to fire:

    * Sentinel file does NOT exist (first run after this code lands).
    * ``parol6-vision/parol6_vision/sim/meshes/ssg48_body_realsense.stl``
      exists at the sibling-clone location (the merged STL is Alvar's
      personal CAD export — for users without the file, this no-ops
      silently).
    * The custom tool name ``ssg48_realsense`` is not already in use.

    Side effects on success:

    * ``~/.waldo-commander/custom_tools/ssg48_realsense/{body.stl,
      jaw_left.stl, jaw_right.stl, config.json}`` written. The body
      STL has the SSG-48 fit transform pre-applied so the placement
      transform stays at identity — user iteration still works via
      ``mesh_translate_m`` etc.
    * ``custom:ssg48_realsense`` registered in ``parol6.tools._TOOL_REGISTRY``.
    * Sentinel file written so this migration never runs again.
    * The legacy ``ssg48_body_realsense.stl`` is also written into
      parol6's mesh dir so the collision check (which still hardcodes
      that filename) keeps working without further changes.

    Returns True when the migration created a new custom tool, False
    when skipped (sentinel present, source missing, name conflict, etc.).
    """
    ensure_root()
    sentinel = CUSTOM_TOOLS_ROOT / _SSG48_SENTINEL_FILENAME
    if sentinel.exists():
        return False
    if _SSG48_MIGRATION_NAME in list_tool_names():
        # Custom tool already exists — write the sentinel so we don't
        # try again, and keep what's there. User-customised state wins.
        sentinel.touch(exist_ok=True)
        return False

    src_body = _ssg48_merged_stl_path()
    if src_body is None:
        # No merged STL on disk — common case for users without Alvar's
        # personal CAD. Skip without fanfare; don't write the sentinel
        # so a later checkout that brings the file in does run the
        # migration.
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

    # Same fit constants the SSG-48 hijack used historically — kept as
    # module-level defaults in ``constants.py`` so this migration
    # produces an identical body to the legacy hijack output.
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

    # Step 2: user RPY + translate (Alvar's tuning on top of the fit).
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

    # Copy stock jaw STLs (already in flange-metres coords).
    for stock_name, dst_name in (
        ("ssg48_finger_left.stl", JAW_LEFT_STL_NAME),
        ("ssg48_finger_right.stl", JAW_RIGHT_STL_NAME),
    ):
        src = mesh_dir / stock_name
        if src.exists():
            shutil.copyfile(str(src), str(target_folder / dst_name))

    # Also write the baked body to the legacy ``ssg48_body_realsense.stl``
    # filename in parol6's mesh dir — collision.py and other consumers
    # still resolve the gripper body via that filename. Keeps existing
    # collision checks working without a parallel refactor in this turn.
    try:
        legacy_path = mesh_dir / "ssg48_body_realsense.stl"
        mesh.export(str(legacy_path), file_type="stl")
        logger.info("ssg48 migration: wrote legacy %s", legacy_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("ssg48 migration: legacy bake failed: %s", e)

    cfg = CustomToolConfig(
        name=_SSG48_MIGRATION_NAME,
        display_name="SSG-48 + RealSense bracket",
        description=(
            "Migrated from the legacy SSG-48 hijack. Body has the merged "
            "camera bracket fused in; jaws are parol6's stock finger STLs. "
            "Fit transform pre-baked into body.stl so placement starts at "
            "identity — iterate via the placement inputs."
        ),
        mesh_translate_m=(0.0, 0.0, 0.0),
        mesh_rpy_rad=(0.0, 0.0, 0.0),
        mesh_scale=1.0,
        # SSG-48 TCP — same flange→TCP transform as the built-in entry.
        tcp_origin_m=(0.0, 0.0, -0.105),
        tcp_rpy_rad=(0.0, 0.0, 0.0),
        # Jaw motion from parol6's _SSG48_JAW_MOTION (24 mm symmetric, +Y).
        jaw_travel_m=0.024,
        jaw_axis=(0.0, 1.0, 0.0),
        jaw_symmetric=True,
    )
    save_config(cfg)
    cfg.has_body = cfg.body_path.exists()
    cfg.has_jaws = (
        cfg.jaw_left_path.exists() and cfg.jaw_right_path.exists()
    )

    # Register the new custom tool so it shows up in the dropdown
    # without a restart.
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


def is_active_tool(name: str) -> bool:
    """Return True when ``custom:<name>`` is the controller's currently-
    active tool. Used by the UI to show an ACTIVE badge and to decide
    whether a transform edit gets a live in-scene refresh.
    """
    try:
        from waldo_commander.state import robot_state  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    return robot_state.tool_key == f"custom:{name}"


def live_refresh_active_tool(name: str) -> bool:
    """If ``custom:<name>`` is the active tool, call the URDF scene's
    ``apply_tool`` to re-load tool meshes from disk so a fresh re-bake
    appears immediately. Returns True when a refresh was issued.

    No-op when:
    * the tool isn't currently active (the user needs to switch to it
      via the gripper panel before edits become visible),
    * waldo-commander's UI state isn't initialised yet (e.g. called
      during page-teardown).
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
    """Return ``[(registry_key, display_name), ...]`` for every tool
    currently in ``parol6.tools._TOOL_REGISTRY``. Used by the import
    dialog so the user picks a source by name. Excludes ``"NONE"`` (it
    has no meshes worth copying).
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
    """Fork the tool at ``parol6.tools._TOOL_REGISTRY[source_key]`` into a
    new custom tool under ``~/.waldo-commander/custom_tools/<target_name>/``.

    Reads the source's meshes (BODY + JAWs), copies the underlying STL
    files from parol6's mesh dir to the new custom-tool folder, extracts
    the TCP transform + jaw motion, and writes a ``config.json`` with a
    placement transform of identity (the source meshes are already in
    flange coordinates so no extra placement is needed).

    Returns the new ``CustomToolConfig`` on success, or None when:
    * the source key isn't in the registry,
    * the source has no body mesh,
    * the target name already exists,
    * file copies fail.

    Use case: migrate a built-in tool (or one that was modified by a
    startup hijack like the SSG-48 merged-bracket bake) into the custom
    system so the user can iterate transforms / TCP / motion via the UI
    instead of editing module constants.
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
        """Strip any browser cache-bust ``?v=<mtime>`` suffix the SSG-48
        hijack adds for Three.js URL invalidation — it's URL syntax,
        invalid as a filesystem path."""
        plain = filename.split("?", 1)[0]
        return mesh_dir / plain

    # Copy STLs into the new custom-tool folder under canonical names.
    target_folder = CUSTOM_TOOLS_ROOT / target_name
    target_folder.mkdir(parents=True, exist_ok=True)
    body_src = _resolve(body_spec.file)
    if not body_src.exists():
        logger.warning("import: body STL missing: %s", body_src)
        return None
    shutil.copyfile(str(body_src), str(target_folder / BODY_STL_NAME))
    # Up to two jaws: jaw_specs[0] -> jaw_left, jaw_specs[1] -> jaw_right.
    # parol6's tools list "right" before "left" by convention; both
    # orderings work since the only thing that matters is one-per-side.
    if len(jaw_specs) >= 2:
        for jaw_spec, dst_name in (
            (jaw_specs[0], JAW_RIGHT_STL_NAME),
            (jaw_specs[1], JAW_LEFT_STL_NAME),
        ):
            jaw_src = _resolve(jaw_spec.file)
            if jaw_src.exists():
                shutil.copyfile(str(jaw_src), str(target_folder / dst_name))

    # Extract TCP transform: source.transform is the 4×4 flange→TCP.
    transform = np.asarray(src.transform, dtype=np.float64)
    tcp_origin = tuple(float(v) for v in transform[:3, 3])
    tcp_rpy = tuple(
        float(v) for v in SciRot.from_matrix(transform[:3, :3]).as_euler("XYZ")
    )

    # Jaw motion: pick the first LinearMotion (custom_tools only models
    # one). parol6 ships LinearMotion with axis + travel_m + symmetric.
    jaw_travel_m = 0.0
    jaw_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    jaw_symmetric = True
    LinearMotion = getattr(parol6_tools, "LinearMotion", None)
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
        # Identity placement — source STLs are already in flange coords.
        mesh_translate_m=(0.0, 0.0, 0.0),
        mesh_rpy_rad=(0.0, 0.0, 0.0),
        mesh_scale=1.0,
        tcp_origin_m=(tcp_origin[0], tcp_origin[1], tcp_origin[2]),
        tcp_rpy_rad=(tcp_rpy[0], tcp_rpy[1], tcp_rpy[2]),
        jaw_travel_m=jaw_travel_m,
        jaw_axis=jaw_axis,
        jaw_symmetric=jaw_symmetric,
    )
    save_config(cfg)
    # Refresh the derived flags now that body / jaws exist on disk.
    cfg.has_body = cfg.body_path.exists()
    cfg.has_jaws = cfg.jaw_left_path.exists() and cfg.jaw_right_path.exists()
    logger.info(
        "import: %r forked from %s (body=%s, jaws=%s, tcp=%s)",
        target_name, source_key, cfg.has_body, cfg.has_jaws, tcp_origin,
    )
    return cfg


async def select_as_active(name: str) -> bool:
    """Send a ``select_tool`` to the controller so ``custom:<name>``
    becomes the live active tool. Returns True on success.

    Requires a running parol6-server connection — fails cleanly when the
    controller isn't reachable. Mirrors what the gripper-panel dropdown
    does when the user picks a tool there, so the behaviour is identical
    after this call returns.
    """
    full_key = f"custom:{name}"
    try:
        from waldo_commander.state import ui_state  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    panel = getattr(ui_state, "control_panel", None)
    client = getattr(panel, "client", None) if panel else None
    if client is None:
        logger.warning("custom_tools: no client; can't select %s", full_key)
        return False
    try:
        await client.select_tool(full_key, variant_key="")
    except Exception as e:  # noqa: BLE001
        logger.warning("custom_tools: select_tool(%s) failed: %s", full_key, e)
        return False
    # Apply locally too (active_robot + scene). The select_tool RPC
    # eventually triggers the broadcast that flips robot_state.tool_key,
    # but applying immediately keeps the UI feeling responsive.
    try:
        ui_state.active_robot.set_active_tool(full_key, variant_key=None)
    except Exception as e:  # noqa: BLE001
        logger.debug("custom_tools: set_active_tool local apply failed: %s", e)
    scene = getattr(ui_state, "urdf_scene", None)
    if scene is not None:
        try:
            scene.apply_tool(full_key, variant_key=None)
        except Exception as e:  # noqa: BLE001
            logger.debug("custom_tools: scene apply_tool after select failed: %s", e)
    return True
