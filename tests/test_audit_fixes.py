"""Unit tests for the helpers introduced by the audit fix-batches.

Each finding referenced as ``audit #N`` corresponds to the multi-agent
code review's numbering from the synthesis report (1-12 in the original
audit). Tests here cover the pure / deterministic helpers; cross-cutting
GUI flows (browser interaction, real-controller IK with a tool bound,
multi-step user workflows for tool delete + re-create) need
in-environment testing — see commit messages for what specifically was
deferred.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Path-traversal validation (audit batch 5 — _validate_safe_name)
# ---------------------------------------------------------------------------


class TestValidateSafeName:
    """Reject any string that could escape ``CUSTOM_TOOLS_ROOT`` /
    ``parol6_mesh_dir`` via path-traversal or control chars."""

    def _validator(self):
        from waldo_commander.components.calibration_overlays.custom_tools import (
            _validate_safe_name,
        )
        return _validate_safe_name

    def test_accepts_alphanumeric_and_underscore(self) -> None:
        validator = self._validator()
        for name in ("my_tool", "tool_v1", "A123", "x", "Z"):
            validator(name)  # no raise

    @pytest.mark.parametrize("bad_name", [
        "../escape",       # parent traversal
        "..",              # explicit parent
        ".",               # current dir
        "",                # empty
        "foo/bar",         # forward slash
        "foo\\bar",        # backslash
        "foo bar",         # space
        "foo.bar",         # dot
        "foo-bar",         # dash
        "fooé",            # unicode
        "../foo",
        "/etc/passwd",     # absolute
    ])
    def test_rejects_unsafe_strings(self, bad_name: str) -> None:
        validator = self._validator()
        with pytest.raises(ValueError):
            validator(bad_name)

    @pytest.mark.parametrize("bad_value", [None, 123, 1.5, [], {}])
    def test_rejects_non_strings(self, bad_value: Any) -> None:
        validator = self._validator()
        with pytest.raises((ValueError, TypeError)):
            validator(bad_value)


# ---------------------------------------------------------------------------
# STL size cap (audit batch 5 — _stl_size_check)
# ---------------------------------------------------------------------------


class TestStlSizeCheck:
    """Reject STLs above the configured size cap so trimesh.load can't
    OOM the server on a malicious / accidental large upload."""

    def test_accepts_small_file(self, tmp_path: Path) -> None:
        from waldo_commander.components.calibration_overlays.custom_tools import (
            _stl_size_check,
        )
        small = tmp_path / "tiny.stl"
        small.write_bytes(b"\x00" * 1024)  # 1 KB
        assert _stl_size_check(small) is True

    def test_rejects_oversized_file(self, tmp_path: Path) -> None:
        from waldo_commander.components.calibration_overlays.custom_tools import (
            _stl_size_check, _STL_MAX_SIZE_MB,
        )
        big = tmp_path / "huge.stl"
        # Stub stat instead of writing 200 MB to disk
        oversize_bytes = (_STL_MAX_SIZE_MB + 1) * 1024 * 1024
        with patch.object(Path, "stat") as stat_mock:
            stat_mock.return_value.st_size = oversize_bytes
            assert _stl_size_check(big) is False

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        from waldo_commander.components.calibration_overlays.custom_tools import (
            _stl_size_check,
        )
        missing = tmp_path / "does_not_exist.stl"
        assert _stl_size_check(missing) is False

    def test_max_size_constant_is_reasonable(self) -> None:
        from waldo_commander.components.calibration_overlays.custom_tools import (
            _STL_MAX_SIZE_MB,
        )
        # Far above any realistic CAD-export end-effector (typical
        # gripper STLs are 1-20 MB) but small enough to keep memory
        # bounded on shared hardware.
        assert 50 <= _STL_MAX_SIZE_MB <= 500


# ---------------------------------------------------------------------------
# Tool-params resolution heuristic (audit batch 1 — #4 env precedence)
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, key: str) -> None:
        self.key = key


class _FakeClient:
    def __init__(self, tool_key: str) -> None:
        self.tool = _FakeTool(tool_key)


@pytest.fixture
def clean_env() -> None:
    """Drop the three env vars the resolver reads, so each test starts
    from a known baseline. Restores prior values after the test."""
    keys = (
        "WALDO_GUI_ACTIVE_TOOL_KEY",
        "WALDO_GUI_ACTIVE_TOOL_VARIANT",
        "WALDO_GUI_ACTIVE_TCP_OFFSET_M",
    )
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestResolveToolParamsForIk:
    """The heuristic: env-custom: > client > env-noncustom:."""

    def _resolver(self):
        from waldo_commander.services.stepping_client import (
            _resolve_tool_params_for_ik,
        )
        return _resolve_tool_params_for_ik

    def test_empty_env_falls_back_to_client(self, clean_env) -> None:
        resolver = self._resolver()
        tool_key, variant, tcp_offset = resolver(_FakeClient("VACUUM"))
        assert tool_key == "VACUUM"
        assert variant is None
        assert tcp_offset is None

    def test_custom_env_wins_over_client_built_in(self, clean_env) -> None:
        # Controller broadcast carries only the proxy built-in for
        # custom tools, so the env var is the only source of truth
        # for ``custom:`` keys.
        os.environ["WALDO_GUI_ACTIVE_TOOL_KEY"] = "custom:my_grip"
        resolver = self._resolver()
        tool_key, _, _ = resolver(_FakeClient("VACUUM"))
        assert tool_key == "custom:my_grip"

    def test_client_wins_over_built_in_env(self, clean_env) -> None:
        # Mid-script ``client.set_active_tool(...)`` change updates
        # client.tool.key but the env var is stale; trust the client
        # for built-in conflicts.
        os.environ["WALDO_GUI_ACTIVE_TOOL_KEY"] = "SSG-48"
        resolver = self._resolver()
        tool_key, _, _ = resolver(_FakeClient("VACUUM"))
        assert tool_key == "VACUUM"

    def test_no_client_no_env_returns_none(self, clean_env) -> None:
        resolver = self._resolver()
        tool_key, _, _ = resolver(_FakeClient(""))
        assert tool_key == "NONE"

    def test_variant_parsed_from_env(self, clean_env) -> None:
        os.environ["WALDO_GUI_ACTIVE_TOOL_VARIANT"] = "v1"
        resolver = self._resolver()
        _, variant, _ = resolver(_FakeClient("SSG-48"))
        assert variant == "v1"

    def test_empty_variant_env_yields_none(self, clean_env) -> None:
        os.environ["WALDO_GUI_ACTIVE_TOOL_VARIANT"] = ""
        resolver = self._resolver()
        _, variant, _ = resolver(_FakeClient("SSG-48"))
        assert variant is None

    def test_tcp_offset_parsed_from_json(self, clean_env) -> None:
        os.environ["WALDO_GUI_ACTIVE_TCP_OFFSET_M"] = json.dumps(
            [0.001, -0.002, 0.003],
        )
        resolver = self._resolver()
        _, _, tcp_offset = resolver(_FakeClient("SSG-48"))
        assert tcp_offset == (0.001, -0.002, 0.003)

    def test_malformed_tcp_offset_yields_none(self, clean_env) -> None:
        # Robustness: a corrupt env var must not crash the subprocess
        # — the pre-flight just falls through to "no tcp offset".
        os.environ["WALDO_GUI_ACTIVE_TCP_OFFSET_M"] = "not json"
        resolver = self._resolver()
        _, _, tcp_offset = resolver(_FakeClient("SSG-48"))
        assert tcp_offset is None


# ---------------------------------------------------------------------------
# settings.get_global vs settings.get layer skip
# (audit batch 3b — settings.get_global helper)
# ---------------------------------------------------------------------------


class TestSettingsGetGlobal:
    """``get_global`` skips Layer 1 (active-tool override) so an editor
    seeded for a non-active tool sees the GLOBAL setting, not whatever
    the currently-active tool happens to override the global with."""

    def test_get_global_skips_active_tool_override(self) -> None:
        from waldo_commander.components.calibration_overlays import settings
        from waldo_commander.components.calibration_overlays import custom_tools

        # Pin ``active_tool_override`` to return a sentinel so we can
        # confirm get_global ignores it while get returns it.
        sentinel = 999.0
        with patch.object(
            custom_tools, "active_tool_override", return_value=sentinel,
        ):
            assert settings.get("intr_fx") == sentinel
            assert settings.get_global("intr_fx") != sentinel

    def test_get_global_unknown_key_raises(self) -> None:
        from waldo_commander.components.calibration_overlays import settings

        with pytest.raises(KeyError):
            settings.get_global("__not_a_real_setting_key__")

    def test_get_global_returns_runtime_or_default(self) -> None:
        from waldo_commander.components.calibration_overlays import settings

        # No override patched → returns shipped default.
        val = settings.get_global("intr_width")
        assert val is not None
        assert isinstance(val, int | float)


# ---------------------------------------------------------------------------
# _state_lock consistency (audit batch 2 — #8)
# ---------------------------------------------------------------------------


class TestStateLockConsistency:
    """The lock guards the ``reach_generation`` + ``reachable_candidates``
    pair so a concurrent reader can never see a half-updated view."""

    def test_paired_writes_visible_atomically(self) -> None:
        from waldo_commander.components.calibration_overlays.state import (
            _state, _state_lock,
        )

        # Seed a known-good initial state.
        with _state_lock:
            _state["reach_generation"] = 0
            _state["reachable_candidates"] = []

        snapshots: list[tuple[int, list]] = []
        stop_reading = threading.Event()
        reader_iterations = [0]

        def reader() -> None:
            while not stop_reading.is_set():
                with _state_lock:
                    gen = int(_state.get("reach_generation", 0))
                    cands = list(_state.get("reachable_candidates") or [])
                snapshots.append((gen, cands))
                reader_iterations[0] += 1

        def writer() -> None:
            for i in range(50):
                gen = 100 + i
                cands = [f"item_{i}_{j}" for j in range(i % 3 + 1)]
                with _state_lock:
                    _state["reach_generation"] = gen
                    _state["reachable_candidates"] = cands
                time.sleep(0.0001)  # Yield to reader

        t_reader = threading.Thread(target=reader, daemon=True)
        t_writer = threading.Thread(target=writer, daemon=True)
        t_reader.start()
        t_writer.start()
        t_writer.join()
        stop_reading.set()
        t_reader.join(timeout=2.0)

        # Sanity: the test actually exercised concurrency.
        assert reader_iterations[0] > 100, (
            f"reader only ran {reader_iterations[0]} times — too few "
            "to meaningfully test concurrency"
        )

        # Every snapshot's gen must pair with the writer-side cands.
        for gen, cands in snapshots:
            if gen == 0:
                expected: list = []
            elif 100 <= gen < 150:
                i = gen - 100
                expected = [f"item_{i}_{j}" for j in range(i % 3 + 1)]
            else:
                pytest.fail(f"unexpected gen value {gen}")
            assert cands == expected, (
                f"mismatched snapshot at gen={gen}: cands={cands}, "
                f"expected={expected}"
            )
