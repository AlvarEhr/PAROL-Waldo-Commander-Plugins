"""Collision and trajectory validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import settings
from .state import _T_BOARD2BASE, _state, current_board_config

logger = logging.getLogger(__name__)


def _build_collision_manager(
    tablet_T_board2base: NDArray[np.float64] | None = None,
) -> tuple[Any, set[tuple[str, str]], dict[str, Any]] | None:
    """Build a trimesh CollisionManager populated with PAROL6's link meshes
    + the merged SSG-48 gripper body (which has the camera bracket fused in)
    + optional FLOOR and TABLET static collision primitives.

    Args:
        tablet_T_board2base: Optional 4×4 board→base transform. When
            provided AND the mounting surface is enabled, a box of
            ``settings.surface_dimensions_m`` is placed at the ChArUco
            centre, with its top face on the board's z=0 surface and the
            body extending in board-local -Z. When None, no surface
            primitive is added.

    Returns:
        (manager, adjacent_pairs, meshes) on success, None if python-fcl or any
        link mesh is missing. ``adjacent_pairs`` whitelists (link_a, link_b)
        pairs whose collisions should NOT count as self-collisions — joint
        neighbours that always touch, plus the FLOOR-vs-base / FLOOR-vs-
        TABLET background pairs.
        ``meshes`` is a dict mapping object name to its ``trimesh.Trimesh`` instance.
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
    meshes: dict[str, Any] = {}
    link_names = ["base_link", "L1", "L2", "L3", "L4", "L5", "L6"]
    for name in link_names:
        path = mesh_dir / f"{name}_simplified.stl"
        if not path.exists():
            path = mesh_dir / f"{name}.STL"
        try:
            mesh = trimesh.load(path, force="mesh")
            mgr.add_object(name, mesh, transform=np.eye(4))
            meshes[name] = mesh
        except Exception as e:  # noqa: BLE001
            logger.warning("collision mesh load failed for %s: %s", path, e)
            return None
    # Active gripper body + jaws — sourced from whatever tool is currently
    # selected in the parol6 registry (built-ins like SSG-48, MSG, or any
    # ``custom:<name>`` registered by the custom-tools system). No
    # hardcoded filenames — switching to a new gripper picks up its
    # meshes automatically. Cache-bust ``?v=<mtime>`` suffixes that the
    # legacy SSG-48 hijack added for browser invalidation are stripped
    # before opening the file.
    jaw_loaded: list[str] = []
    active_tool_key, tool_meshes_by_role = _resolve_active_tool_meshes(mesh_dir)
    body_mesh_files = tool_meshes_by_role.get("BODY", ())
    jaw_mesh_files = tool_meshes_by_role.get("JAW", ())

    for body_path in body_mesh_files:
        if not body_path.exists():
            logger.info(
                "collision: gripper body STL missing at %s; "
                "body collision skipped", body_path,
            )
            continue
        try:
            grip_mesh = trimesh.load(body_path, force="mesh")
            mgr.add_object("gripper", grip_mesh, transform=np.eye(4))
            meshes["gripper"] = grip_mesh
        except Exception as e:  # noqa: BLE001
            logger.warning("collision mesh load failed for gripper: %s", e)
        # Only load the FIRST body mesh — there's typically just one,
        # and the collision manager keys by name ("gripper").
        break

    for idx, jaw_path in enumerate(jaw_mesh_files):
        if not jaw_path.exists():
            logger.warning(
                "collision: jaw mesh missing at %s — fingertip collision "
                "not active for this jaw", jaw_path,
            )
            continue
        jaw_name = f"jaw_{idx}"
        try:
            jaw_mesh = trimesh.load(jaw_path, force="mesh")
            mgr.add_object(jaw_name, jaw_mesh, transform=np.eye(4))
            jaw_loaded.append(jaw_name)
            meshes[jaw_name] = jaw_mesh
        except Exception as e:  # noqa: BLE001
            logger.warning("collision mesh load failed for %s: %s", jaw_name, e)

    # FLOOR collision primitive — wide flat box at z ∈ [-0.05,
    # +safety_margin]. Top is RAISED above z=0 by the safety margin so
    # any gripper/jaw approach below that altitude triggers a collision
    # before actual contact. base_link is on the ("base_link", "FLOOR")
    # adjacent whitelist so the robot's own base sitting on the floor
    # doesn't fire a false positive — only links that AREN'T expected
    # to touch the floor (everything except base_link) get rejected.
    floor_added = False
    safety_margin = float(settings.collision_safety_margin_m)
    if bool(settings.floor_primitive_enabled):
        try:
            box_thickness = 0.05 + safety_margin
            floor_box = trimesh.creation.box(extents=(10.0, 10.0, box_thickness))
            floor_pose = np.eye(4, dtype=np.float64)
            # Position so box top is at +safety_margin: centre = top - h/2.
            floor_pose[2, 3] = safety_margin - box_thickness / 2.0
            mgr.add_object("FLOOR", floor_box, transform=floor_pose)
            meshes["FLOOR"] = floor_box

            visual_floor_box = trimesh.creation.box(extents=(10.0, 10.0, 0.05))
            visual_floor_pose = np.eye(4, dtype=np.float64)
            visual_floor_pose[2, 3] = -0.025
            visual_floor_box.apply_transform(visual_floor_pose)
            meshes["VISUAL_FLOOR"] = visual_floor_box

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
    if bool(settings.surface_enabled) and tablet_T_board2base is not None:
        try:
            _cfg = current_board_config()
            t_w, t_l, t_h = settings.surface_dimensions_m
            t_off_x, t_off_y = settings.surface_offset_local_m
            # Inflate by 2 × safety margin in each axis (margin on each side).
            # Centre of the inflated box stays at the same point as the original
            # tablet centre, so the inflation is symmetric.
            margin = safety_margin
            tablet_box = trimesh.creation.box(
                extents=(t_w + 2 * margin, t_l + 2 * margin, t_h + 2 * margin),
            )
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
            meshes["TABLET"] = tablet_box

            visual_tablet_box = trimesh.creation.box(extents=(t_w, t_l, t_h))
            visual_tablet_box.apply_transform(tablet_pose)
            meshes["VISUAL_TABLET"] = visual_tablet_box

            tablet_added = True
        except Exception as e:  # noqa: BLE001
            logger.warning("TABLET primitive add failed: %s", e)

    adjacent: set[tuple[str, str]] = {
        ("base_link", "L1"), ("L1", "L2"), ("L2", "L3"),
        ("L3", "L4"), ("L4", "L5"), ("L5", "L6"),
        ("L6", "gripper"),
    }
    # Jaws are rigidly attached to the gripper body (we move them with
    # the same flange transform). Whitelist body↔jaw and every jaw-pair
    # so the always-overlapping-at-the-base roots don't fire false
    # self-collision rejections. Generic across any jaw count — cooperates
    # with custom tools that ship more than 2 jaws (rare but possible).
    for jaw in jaw_loaded:
        adjacent |= {("L6", jaw), ("gripper", jaw)}
    for i in range(len(jaw_loaded)):
        for j in range(i + 1, len(jaw_loaded)):
            adjacent |= {(jaw_loaded[i], jaw_loaded[j])}
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
        # The user-noted bug: gripper FINGERS were the actual things
        # clipping the tablet in low-elevation poses. We DO want
        # finger-vs-tablet collisions to fire as REAL rejections, so
        # do NOT whitelist them here. (The body-vs-tablet pair is also
        # NOT whitelisted, by the same reasoning.)
    # Add reverse pairs for symmetric lookup.
    adjacent |= {(b, a) for a, b in adjacent}
    jaws_str = (
        f" + {len(jaw_loaded)} jaws" if jaw_loaded else ""
    )
    logger.info(
        "self-collision manager loaded for %s: 7 links + gripper%s%s%s",
        active_tool_key, jaws_str,
        " + FLOOR" if floor_added else "",
        " + TABLET" if tablet_added else "",
    )
    # Stash the loaded jaw names on the manager so _self_collides can
    # apply the gripper transform to them too.
    mgr._loaded_jaw_names = list(jaw_loaded)  # type: ignore[attr-defined]
    # Stash the tool key the cache was built for so callers can detect
    # tool changes and invalidate the manager appropriately.
    mgr._built_for_tool_key = active_tool_key  # type: ignore[attr-defined]

    return mgr, adjacent, meshes


def _resolve_active_tool_meshes(
    mesh_dir: Any,
) -> tuple[str, dict[str, list[Any]]]:
    """Look up the currently-active tool's mesh files from parol6's
    ``_TOOL_REGISTRY``, grouped by role. Returns ``(tool_key,
    {"BODY": [Path, ...], "JAW": [Path, ...]})``.

    The active tool is sourced from ``robot_state.tool_key`` (the
    controller's broadcast value) when populated, otherwise falls back
    to ``"NONE"`` (which has no meshes and produces an empty result —
    the collision manager builds with arm links only).

    URL cache-bust suffixes (``?v=<mtime>``) on filenames are stripped
    before resolving the filesystem path — they're valid as Three.js
    URLs but invalid as ``open()`` arguments.
    """
    try:
        from parol6 import tools as parol6_tools  # noqa: PLC0415
        from waldo_commander.state import robot_state  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        logger.debug("collision: tool/state lookup unavailable: %s", e)
        return ("NONE", {"BODY": [], "JAW": []})

    tool_key = getattr(robot_state, "tool_key", None) or "NONE"
    cfg = parol6_tools._TOOL_REGISTRY.get(tool_key)
    if cfg is None:
        logger.debug(
            "collision: tool key %r not in registry; collision built "
            "with arm links only", tool_key,
        )
        return (tool_key, {"BODY": [], "JAW": []})

    body_paths: list[Any] = []
    jaw_paths: list[Any] = []
    for spec in getattr(cfg, "meshes", ()):
        # Strip the cache-bust suffix the legacy SSG-48 hijack added.
        plain = str(spec.file).split("?", 1)[0]
        path = mesh_dir / plain
        # MeshRole is an enum; compare via .name to stay forward-
        # compatible if parol6 introduces new roles.
        role_name = getattr(getattr(spec, "role", None), "name", "")
        if role_name == "BODY":
            body_paths.append(path)
        elif role_name == "JAW":
            jaw_paths.append(path)
    return (tool_key, {"BODY": body_paths, "JAW": jaw_paths})


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
    # Gripper body rides on the flange (its mesh is in flange-frame
    # coordinates). The jaws (ssg48_finger_left/right) are also in
    # flange-frame and rigidly attached to the body — they ride on
    # exactly the same transform. Without this update the jaws would
    # stay at the world origin and never collide with anything except
    # base_link, missing the actual fingertip-clip-tablet case.
    manager.set_transform("gripper", poses.l6_visual)
    for jaw_name in getattr(manager, "_loaded_jaw_names", ()):
        manager.set_transform(jaw_name, poses.l6_visual)
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
    if pair is not None:
        # Drop the cache when the active tool has changed since the
        # manager was built — its loaded gripper / jaw meshes correspond
        # to a different tool and would falsely accept collisions for
        # the new gripper.
        try:
            from waldo_commander.state import robot_state as _rs  # noqa: PLC0415

            cached_key = getattr(pair[0], "_built_for_tool_key", None)
            current_key = getattr(_rs, "tool_key", None)
            if cached_key is not None and cached_key != current_key:
                logger.info(
                    "collision: tool change %s -> %s; rebuilding manager",
                    cached_key, current_key,
                )
                pair = None
                _state["trajectory_collision_mgr_pair"] = None
        except Exception:  # noqa: BLE001
            # If state lookup fails, keep the cached manager — better
            # to use the old one than to fail open.
            pass
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

    mgr, adjacent, _ = pair
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
