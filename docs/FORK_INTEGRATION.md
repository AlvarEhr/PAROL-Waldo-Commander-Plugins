# parol6-vision-calibration fork: integration overview

This document describes how the `parol6-vision-calibration` branch of
Waldo-Commander tacks onto Jepson's upstream code. It exists for two
reasons:

1. **For future contributors** (the user, future Claude sessions,
   anyone reviewing the fork) who need to understand where the new
   code lives, where it touches upstream, and how to keep the two
   layers cleanly separable.
2. **As a checklist for upstream-PR readiness** — what gates the new
   features, how to disable them without editing code, and what we
   intentionally chose to NOT push upstream.

The companion `parol6-vision` repo (sibling to `Waldo-Commander/`)
holds the actual calibration toolkit; this fork is the GUI integration
layer plus a thin scripting-side pre-flight in
`services/stepping_client.py`.

## High-level shape

The integration is a **package + touchpoints** model:

```
waldo_commander/
├── components/
│   └── calibration_overlays/         ← NEW. Self-contained package.
│       ├── __init__.py               (~22 files, ~12.8k LOC)
│       ├── calibration_thread.py     orchestrator worker thread
│       ├── collision.py              FCL gate + tool/mesh resolution
│       ├── constants.py              tunables (intrinsics, hemisphere, ...)
│       ├── custom_tools.py           drop-in custom-tool ingestion
│       ├── custom_tools_ui.py        custom-tool editor panel
│       ├── detection.py              perception AABB overlay
│       ├── frustum.py                live camera frustum + footprint
│       ├── hover.py                  hover-above-board verification thread
│       ├── live_apply.py             live settings re-apply hooks
│       ├── localise.py               board-localise sweep thread
│       ├── overlays.py               scene-build entry point
│       ├── panel.py                  side-tab panel + status helper
│       ├── pose_popup.py             click-to-pose card
│       ├── preview_dialog.py         collision-rejection dialog
│       ├── reachability.py           green-dot IK sweep
│       ├── settings.py               calibration-only settings layer
│       ├── settings_ui.py            calibration-settings editor
│       ├── ssg48_hijack.py           DEPRECATED legacy entry point
│       ├── state.py                  shared `_state` dict + lock
│       └── workspace.py              workspace-hull pose-gen filter
│
├── services/
│   ├── stepping_client.py            UPSTREAM + ~520 lines added
│   ├── path_preview_client.py        UPSTREAM + ~210 lines added
│   ├── path_visualizer.py            UPSTREAM + ~40 lines added
│   ├── script_runner.py              UPSTREAM + ~30 lines added
│   ├── urdf_scene/urdf_scene.py      UPSTREAM + ~85 lines added
│   └── urdf_scene/config.py          UPSTREAM + ~10 lines added
│
├── components/
│   ├── control.py                    UPSTREAM + ~160 lines added
│   └── settings.py                   UPSTREAM + ~290 lines added
│
└── main.py                            UPSTREAM + ~270 lines added
```

The `calibration_overlays/` package is **fully self-contained** — it
imports parol6_vision, parol6, and its own internal modules but is
never imported from outside the package except through the integration
points listed below. It can be deleted from a fork and the only thing
that breaks is the import statements in the touchpoints.

## How upstream code is touched

Each upstream file has a small set of additive edits. None modify
existing public API; all are extensions or new branches gated on the
calibration toggles.

### `main.py` (~270 lines added)

Three additions:

1. **`_calibration_enabled()`** (env-var hard gate) and
   **`_calibration_features_active()`** (storage soft gate) helpers.
2. **`initialize_urdf_scene`** body extended to call
   `custom_tools.auto_migrate_ssg48_with_bracket()` +
   `custom_tools.register_all()` when the soft gate is True.
3. **Page-build path** extended to register the calibration tab and
   add `calibration_overlays.add_overlays(scene)` after the URDF scene
   is constructed. Both register-points are inside `if
   _calibration_enabled()` blocks so the entire integration is
   inert when the env var is off.

### `services/stepping_client.py` (~520 lines added)

Adds a gripper-vs-environment FCL pre-flight that runs in the
program-runner subprocess before each motion command. The pre-flight
imports parol6_vision lazily (`try: import parol6_vision`; falls
through to `return` on ImportError) so the file is safe to load on
machines without parol6_vision installed.

The new code is wrapped in `_maybe_check_collision(method_name, args,
kwargs, wrapped_client)` which is called from `_wrap_motion_method`'s
non-blended branch. Blended motions (`r > 0`) bypass the pre-flight
and emit a one-shot stderr breadcrumb (audit fix #3).

A `_resolve_tool_params_for_ik(wrapped_client)` helper resolves the
tool key + variant + tcp_offset_m from `WALDO_GUI_ACTIVE_TOOL_*` env
vars (set by `script_runner`) and the live `wrapped_client.tool.key`,
prioritising env-custom: > client > env-noncustom: per audit fix #4.

A new `_LOCAL_ROBOT_CACHE` (per-tool-config keyed) holds parol6.Robot
instances built with `set_active_tool(...)` applied so the local IK
runs in TCP frame matching the controller (audit fix #1).

### `services/script_runner.py` (~30 lines added)

Three new env-var forwardings before subprocess spawn:

* `WALDO_MESH_COLLISION_ENABLED` ← `app.storage.general["mesh_collision_check_enabled"]`
* `WALDO_GUI_ACTIVE_TOOL_KEY` ← `app.storage.general["selected_tool"]`
* `WALDO_GUI_ACTIVE_TOOL_VARIANT` ← `app.storage.general[f"tool_variant_{selected_tool}"]`
* `WALDO_GUI_ACTIVE_TCP_OFFSET_M` ← `app.storage.general[f"tcp_offset_{selected_tool}"]` (mm dict, JSON-encoded as `[m, m, m]` tuple)

The first three exist so the subprocess's collision check uses the
GUI's canonical tool key (custom tools' `proxy_tool_key` would
otherwise mask the real tool); the fourth is for tool-aware local IK
(see audit fix #1).

### `services/path_preview_client.py` (~210 lines added)

Adds edit-time collision pre-flight: each motion command in the user's
script is collision-checked while the editor is open, with results fed
into `accumulated_errors` to draw lint squiggles in the code editor.
Backed by an LRU cache keyed by quantised joints + config.

### `services/path_visualizer.py` (~40 lines added)

Forwards the GUI's selected tool key into the cpu_bound worker
process's environment so the worker's `path_preview_client` instance
resolves to the correct meshes for custom tools.

### `services/urdf_scene/urdf_scene.py` (~85 lines added)

Adds a new `PREVIEW` appearance mode for the click-to-pose preview
feature. The PREVIEW mode is entered when the user clicks a
reachability sphere; the URDF is briefly painted with the candidate
joint configuration so they can see where the move would go before
confirming. Existing `LIVE` / `SIMULATOR` / `EDITING` modes are
untouched; the PREVIEW addition is purely additive.

### `components/control.py` (~160 lines added)

Adds a collision-rejection dialog to the Home, joint-limit, and
go-to-angle buttons. Pre-flight runs in `_collision_check_with_dialog`
which gracefully falls back to "allow move" on any unavailability of
parol6_vision. A startup-zeros bypass (live broadcast hasn't
populated yet) prevents false-rejections at initial page load.

### `components/settings.py` (~290 lines added)

Adds the **Calibration & motion safety** section to the bottom-right
Settings panel, with two switches:

* **Calibration features** — flips
  `app.storage.general["calibration_features_active"]`.
* **Mesh collision check** — flips
  `app.storage.general["mesh_collision_check_enabled"]`.

Both switches are visible regardless of `WALDO_CALIBRATION_ENABLED`
(when the env var is off, the switches are still there but flipping
them has no effect because the package isn't imported).

### `.github/workflows/tests.yml` (commit `13da8e2`)

**Fork-only.** Disables the auto-trigger on push/PR because the
parol6-vision-calibration branch installs parol6-vision-specific deps
that the upstream test matrix doesn't expect. Tests can still be run
manually via `workflow_dispatch`. **This commit MUST be excluded when
preparing an upstream PR** — cherry-pick / rebase the audit-fix
commits onto the upstream main while skipping `13da8e2`.

## Three-tier gating — disable without code edits

The whole integration is layered behind three gates so a user can
turn it off at any granularity without touching code:

### Tier 1: hard gate (env var)

```bash
WALDO_CALIBRATION_ENABLED=0 waldo-commander
```

When set to `0`:

* `calibration_overlays/` is never imported.
* No calibration tab is added to the left strip.
* `initialize_urdf_scene` does nothing extra.
* `add_overlays` is never called.
* `path_preview_client` and `stepping_client`'s pre-flight code paths
  are present but never run (they early-return on
  `os.environ.get("WALDO_MESH_COLLISION_ENABLED", "1") != "1"` —
  which `script_runner` defaults to "1" but is irrelevant when the
  package itself isn't imported).

Resource impact when `=0`: **identical to upstream**. The package's
parol6_vision import + 22-file load doesn't happen; no per-tick
timers, no FCL machinery initialised.

### Tier 2: soft gate (in-app toggle)

`app.storage.general["calibration_features_active"]` (default
`False`).

When False:

* The tab is visible but shows a placeholder ("Calibration features
  are off") with a Turn-on switch.
* `add_overlays` returns before scheduling any timers.
* Custom tools are NOT auto-registered (the user has to flip on).
* `register_all()` is not called, so no STL bakes hit
  `parol6_mesh_dir`.

Resource impact when False: **the package is loaded but idle**. The
import cost (~50-100 ms) is paid at startup; no per-frame work runs.

### Tier 3: collision check toggle

`app.storage.general["mesh_collision_check_enabled"]` (default
`True`).

When False:

* `validate_joint_trajectory` returns
  `{"safe": True, "manager_ready": False, ...}` without invoking FCL.
* Control-panel buttons no longer show the override dialog; moves
  dispatch directly.
* `path_preview_client` skips the cache lookup and FCL call.
* `stepping_client._maybe_check_collision` returns at the first
  early-return.

Resource impact when False: **scene overlays still render**, but no
FCL collision queries run. The 5 Hz frustum tick + 0.5 Hz live-pose
chip continue but the live-pose chip's per-tick `validate_joint_trajectory`
call returns immediately.

## Audit + fix-batch summary

After integration landed, ran a multi-agent code review (8 Opus 1M
sub-agents reading the changed files end-to-end). Found ~50 issues
across 12 ship-blocker findings, ~25 yellow-priority items, and
~13 minor / cosmetic. Applied fixes in 7 batches:

| Batch | Commit | Findings | Lines |
|---|---|---|---|
| 1 — Safety pre-flight | `edb3010` | #1 IK frame, #2 gripper_only, #3 blended bypass, #4 env precedence | +301 / -45 |
| (twin) | `fb769d5` (parol6-vision) | #1 in safe_motion.py | +162 / -35 |
| 2 — Threads + state | `3985896` | #5 socket leak, #6 orphan waiter, #8 RMW race | +104 / -20 |
| 3a — UI correctness | `861931a` | bake-failure check, no-camera tool name, sw.value race | +49 / -7 |
| 3b — Settings persistence | `8b0a4fe` | delete_tool / variant delete / reset / get_global | +110 / -7 |
| 4 — Robustness | `4113ea3` | cache poisoning, exception cleanup, platform gating | +66 / -22 |
| 5 — Input hardening | `af050a0` | path traversal, STL size cap, partial-jaw warning | +94 / -1 |
| 6 — Dead-code | `e9de29f` | #12 hijack docs cleanup | +53 / -18 |
| 7 — Cleanup + tests | `b88e690` | cosmetic + `tests/test_audit_fixes.py` (34 cases) | +355 / -12 |
| 8 — 2nd bug-hunt | `e1ee0bd` | regression in #1 (subprocess registry), incomplete #2 (sw.value race), variant/tcp_offset desync, partial lock coverage, main_loop semantic, deferred #7 was wrong | +201 / -41 |

After Batch 7 landed, ran a second multi-agent bug-hunt over all
the fix commits to catch regressions and anything the first audit
missed. Found 6 issues — 2 ship-blocker regressions in my own fixes
(custom-tool IK broken in subprocess; sw.value race fix incomplete),
3 warnings, and 1 case where my deferral reasoning for #7 was
wrong on review (`_teardown_overlays` cancels timers but doesn't
set `stop_requested=True`, so worker threads don't actually
release the controller socket on tab-close). Batch 8 addresses all
six. Test suite grew to 37 cases.

One ship-blocker finding remains deferred with explicit reasoning:

* **#9 — `_T_BOARD2BASE[:] = new_T` torn-read.** The author's
  comment in `localise.py` explicitly accepts the race ("the GIL
  doesn't make it formally atomic but in practice no Python statement
  interleaves between the two halves of a 4×4 copy"). The contention
  window is tiny (single write per board-localise vs frustum tick at
  5 Hz reading), and the worst case is a one-frame visual glitch.
  The second bug-hunt verified — NumPy in-place slice-assign lowers
  to ``PyArray_CopyInto`` which holds the GIL throughout, and reader-
  side ``_T_BOARD2BASE @ vec`` operations also hold the GIL through
  their C calls.

(Originally deferred #7 — daemon-thread lifecycle — but the second
bug-hunt found my reasoning was wrong: `_teardown_overlays` does NOT
set `stop_requested=True`, so the on_disconnect path didn't actually
stop worker threads on tab-close. Batch 8 fixed it: teardown now sets
`stop_requested = True` AND dispatches `client.halt()` so an in-flight
motion aborts promptly.)

The remaining yellow / minor items not addressed are:

* `workspace.py` hull-cache invalidation — only matters if the user
  regenerates the workspace_hull.stl mid-session.
* `collision.py` `_config_from_settings` memoisation — performance,
  not correctness.
* `urdf_scene.py` PREVIEW re-color on rapid re-entry was fixed in
  Batch 7.
* Various log-message / formatting / phrasing nits.

## Testing strategy

### What's covered by automated tests

`tests/test_audit_fixes.py` (added in Batch 7) — 34 cases, ~10 s
runtime. Covers:

* `_validate_safe_name` — accepts alphanumeric+underscore; rejects
  path traversal, slashes, dots, spaces, dashes, unicode, absolute
  paths, empties, non-strings.
* `_stl_size_check` — accepts small files; rejects oversize via
  stat-stub; rejects missing files; sanity-check on the cap constant.
* `_resolve_tool_params_for_ik` — empty env / custom env / built-in
  conflict / no client; variant + JSON tcp_offset parsing including
  malformed input.
* `settings.get_global` — confirms it skips Layer 1; unknown-key
  KeyError; runtime-or-default fallback.
* `_state_lock` paired-write consistency — concurrent reader+writer
  over `reach_generation` / `reachable_candidates`.

### What still needs in-environment testing

* **Cartesian moves with a real tool bound** (SSG-48 + non-zero
  tcp_offset_m) — confirm the local IK fix produces joint
  trajectories matching what the controller actually executes.
* **Multi-step user workflows** — delete a tool → re-create with the
  same name → verify it doesn't inherit stale per-tool overrides.
* **Linux/macOS camera dropdown auto-refresh** — Batch 4's platform
  gate switches the timer back on for non-Windows; verify on a real
  Linux/macOS install.
* **Browser timing edge cases** — concurrent click on a reachability
  dot during a board-localise update; rapid tab switching with the
  calibration tab open; etc.
* **Real-controller STOP latency** — verify the orphan-waiter-thread
  fix in Batch 2 reduces STOP-to-stop time as expected on real
  hardware.
* **End-to-end FCL pre-flight** — drive a programme that intentionally
  collides with the floor / tablet, confirm the dialog shows up
  before the move dispatches.

## Resource-usage testing strategy

To compare upstream behaviour vs the integration enabled, run two
identical sessions and compare. The three modes worth measuring:

1. **Upstream baseline** — `WALDO_CALIBRATION_ENABLED=0`
2. **Integration loaded but features off** —
   `WALDO_CALIBRATION_ENABLED=1`,
   `calibration_features_active = False` (the in-app toggle),
   `mesh_collision_check_enabled = True` (the default — we're
   measuring the package-imported-but-idle case).
3. **Integration fully active** — all three switches on.

### What to measure

For each mode, record:

* **Process memory** — `Get-Process waldo-commander | Select-Object WorkingSet64`
  on Windows, or `ps -o rss,vsz` on Linux.
  Take the reading at 30 s after first page load (lets the lazy
  imports settle).
* **Idle CPU** — `Get-Counter '\Process(*waldo*)\% Processor Time'`
  on Windows, or `top -p $PID` on Linux. Watch for 30 s with the
  page open and idle, take the median.
* **Page-load wall time** — time from `waldo-commander` process
  spawn to first browser page-ready.
* **Per-tick CPU during scripted move** — run a 30-second
  sweep program with `path_visualizer` doing edit-time pre-flight,
  measure CPU% during the run.

### Expected deltas (educated guess, would need empirical confirm)

* Upstream → Mode 2 (loaded, idle): ~50-100 MB extra RSS for the
  parol6_vision import + numpy + scipy + opencv-headless + trimesh.
  CPU should be unchanged at idle.
* Mode 2 → Mode 3 (active, idle): negligible memory delta. ~1-3 % CPU
  for the 5 Hz frustum tick + 0.5 Hz live-pose chip + 4 Hz post-cal
  tick on a modern x86_64.
* Mode 3 active during a 30 s scripted run with edit-time pre-flight
  on: per-tick FCL queries are ~5-50 ms each depending on mesh
  complexity; with the LRU cache (256 entries, gated on
  `manager_ready`), most queries should hit cache after the first
  pass.

### Quick benchmark: code-path timing

For specific helpers, `pytest-benchmark` (already a dev dep) can
measure tight code paths:

```python
# tests/test_audit_perf.py
import pytest

def test_validate_joint_trajectory_perf(benchmark):
    from waldo_commander.components.calibration_overlays.collision import (
        validate_joint_trajectory,
    )
    q = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]
    benchmark(validate_joint_trajectory, q, q, gripper_only=True)
```

This lets us catch perf regressions to the FCL pre-flight if a
future refactor breaks the cache or rebuilds the manager on every
call. Would also detect the audit's `_config_from_settings`
memoisation gap if it becomes a real bottleneck.

### Manual A/B procedure

For a one-shot eyeball comparison:

1. Open Task Manager / `htop` alongside two waldo-commander
   instances on different ports.
2. Open browser tabs to both. Watch for 30 s with no input.
3. Compare RSS, CPU%, threads.
4. In the integration-active session, open the calibration tab
   and watch for an additional ~1-3 % CPU on the 5 Hz tick group.
5. Run a scripted 30-second sweep in both. Compare wall-clock
   completion time and average CPU% during the run.
