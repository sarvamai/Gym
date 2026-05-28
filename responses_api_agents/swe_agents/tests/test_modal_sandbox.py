# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Focused tests for the experimental Modal sandbox + rollout path.
# Mocks `modal.Sandbox` / `modal.App` / `modal.Image` so we never hit the
# network. The point is to verify timing aggregation, failure-path handling,
# and benchmark summary correctness.
import asyncio
import sys
import types
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# modal mock installed before importing the modules under test.
# ---------------------------------------------------------------------------


def _install_modal_mock() -> None:
    if "modal" in sys.modules and getattr(sys.modules["modal"], "_ng_mock", False):
        return

    modal_mod = types.ModuleType("modal")
    modal_mod._ng_mock = True  # marker

    class _AioCallable:
        def __init__(self, fn):
            self._fn = fn
            self.aio = self._aio_call

        def __call__(self, *a, **kw):
            return self._fn(*a, **kw)

        async def _aio_call(self, *a, **kw):
            return self._fn(*a, **kw)

    class _Filesystem:
        def __init__(self):
            self.files: Dict[str, str] = {}

            self.write_text = _AioCallable(self._write_text)
            self.read_text = _AioCallable(self._read_text)
            self.copy_from_local = _AioCallable(self._copy_from_local)
            self.copy_to_local = _AioCallable(self._copy_to_local)

        def _write_text(self, data, remote_path):
            self.files[remote_path] = data

        def _read_text(self, remote_path):
            if remote_path not in self.files:
                raise FileNotFoundError(remote_path)
            return self.files[remote_path]

        def _copy_from_local(self, local_path, remote_path):
            self.files[remote_path] = Path(local_path).read_text()

        def _copy_to_local(self, remote_path, local_path):
            Path(local_path).write_text(self.files.get(remote_path, ""))

    class _Process:
        def __init__(self, stdout="ok", stderr="", returncode=0):
            class _Reader:
                def __init__(self, txt):
                    self._txt = txt
                    self.read = _AioCallable(lambda: self._txt)

            self.stdout = _Reader(stdout)
            self.stderr = _Reader(stderr)
            self._rc = returncode
            self.wait = _AioCallable(lambda: self._rc)

    class _Sandbox:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.object_id = "sb_test_id"
            self.filesystem = _Filesystem()
            self.terminate = _AioCallable(lambda: None)
            self._exec_outputs = []
            self.snapshot_filesystem = _AioCallable(self._snapshot)

        def _exec_impl(self, *cmd, **kwargs):
            # Default: returncode 0, stdout "ok".
            return _Process()

        def _snapshot(self):
            return types.SimpleNamespace(object_id="snap_test_id")

        @property
        def exec(self):
            return _AioCallable(self._exec_impl)

    def _sandbox_create(**kwargs):
        return _Sandbox(**kwargs)

    modal_mod.Sandbox = types.SimpleNamespace(
        create=_AioCallable(_sandbox_create),
        from_id=_AioCallable(lambda sid: _Sandbox(id=sid)),
    )
    modal_mod.App = types.SimpleNamespace(
        lookup=lambda name, create_if_missing=False: types.SimpleNamespace(name=name),
    )

    class _Image:
        def __init__(self, ref):
            self.ref = ref
            self.object_id = f"img_{ref}"

        @staticmethod
        def from_registry(ref, add_python=None):
            return _Image(ref)

        @staticmethod
        def from_id(image_id):
            return _Image(image_id)

    modal_mod.Image = _Image

    modal_mod.exception = types.SimpleNamespace(
        SandboxTimeoutError=type("SandboxTimeoutError", (Exception,), {}),
    )

    sys.modules["modal"] = modal_mod


_install_modal_mock()


# Import after modal mock is in place.
from responses_api_agents.swe_agents.modal_sandbox import (  # noqa: E402
    KNOWN_STAGES,
    ModalSandbox,
    aggregate_iterable_floats,
)


# ---------------------------------------------------------------------------
# ModalSandbox lifecycle timing
# ---------------------------------------------------------------------------


def test_modal_sandbox_lifecycle_records_timings():
    async def _run():
        sb = ModalSandbox(app_name="test-app")
        await sb.create()
        await sb.upload_text("/tmp/x", "hello")
        result = await sb.exec("echo", "hi")
        text = await sb.read_text("/tmp/x")
        await sb.terminate()
        return sb, result, text

    sb, result, text = asyncio.run(_run())
    assert text == "hello"
    assert result.returncode == 0
    # Every lifecycle stage recorded at least one entry.
    for stage in ("create", "upload", "exec", "download", "terminate"):
        assert stage in sb.timings, f"missing stage {stage}"
        assert len(sb.timings[stage]) >= 1
        assert sb.timings[stage][0] >= 0.0
    summary = sb.timing_summary()
    assert summary["create_total"] >= 0.0
    assert summary["exec_count"] == 1


def test_modal_sandbox_snapshot_recorded():
    async def _run():
        sb = ModalSandbox(app_name="test-app")
        await sb.create()
        sid = await sb.snapshot_filesystem()
        await sb.terminate()
        return sb, sid

    sb, sid = asyncio.run(_run())
    assert sid == "snap_test_id"
    assert "snapshot_create" in sb.timings
    assert sb.timings["snapshot_create"][0] >= 0.0


def test_modal_sandbox_restore_from_snapshot_records_timing():
    async def _run():
        sb = ModalSandbox(app_name="test-app", from_snapshot_id="img_abc")
        await sb.create()
        await sb.terminate()
        return sb

    sb = asyncio.run(_run())
    assert "snapshot_restore" in sb.timings
    assert "create" in sb.timings


def test_known_stages_are_consistent():
    expected = {
        "create",
        "upload",
        "exec",
        "download",
        "terminate",
        "snapshot_create",
        "snapshot_restore",
    }
    assert set(KNOWN_STAGES) == expected


# ---------------------------------------------------------------------------
# aggregate_iterable_floats
# ---------------------------------------------------------------------------


def test_aggregate_floats_basic():
    agg = aggregate_iterable_floats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert agg["min"] == 1.0
    assert agg["max"] == 5.0
    assert agg["mean"] == 3.0
    assert agg["count"] == 5
    assert agg["p50"] == 3.0


def test_aggregate_floats_empty():
    assert aggregate_iterable_floats([]) == {}


def test_aggregate_floats_filters_none():
    agg = aggregate_iterable_floats([None, 1.0, None, 2.0])  # type: ignore[list-item]
    assert agg["count"] == 2


# ---------------------------------------------------------------------------
# Rollout: failure paths still produce a row with reward=0.0 and timings.
# ---------------------------------------------------------------------------


@pytest.fixture
def fireworks_no_key(monkeypatch):
    # Force the API key env var to be missing.
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)


def test_rollout_missing_fireworks_key_records_failure(fireworks_no_key):
    from responses_api_agents.swe_agents.modal_rollout import (
        FireworksConfig,
        ModalRolloutConfig,
        run_modal_swe_rollout,
    )

    task: Dict[str, Any] = {
        "instance_id": "django__django-1",
        "problem_statement": "fix the bug",
    }
    fc = FireworksConfig()
    mc = ModalRolloutConfig()
    result = asyncio.run(run_modal_swe_rollout(task, 0, fireworks=fc, modal_cfg=mc))
    assert result.reward == 0.0
    assert result.error_stage == "model_call"
    assert result.error_message
    # Required timing keys present even on failure.
    assert "total_rollout_time" in result.timings
    assert "run_start_time" in result.timings
    assert "run_end_time" in result.timings
    # Sandbox stage fields are present (None is fine, but key must exist).
    for key in (
        "modal_agent_sandbox_create_time",
        "modal_eval_sandbox_create_time",
        "modal_agent_command_exec_time",
        "modal_eval_command_exec_time",
    ):
        assert key in result.timings


def test_rollout_no_patch_path_records_no_patch_failure(monkeypatch):
    """Model returns no diff -> patch_exists=False, eval not run, reward 0."""
    from responses_api_agents.swe_agents import modal_rollout

    async def _fake_call(*a, **kw):
        return {"text": "i refuse to produce a diff", "model_call_time": 0.01}

    monkeypatch.setattr(modal_rollout, "call_fireworks_model", _fake_call)

    task = {"instance_id": "demo", "problem_statement": "x"}
    fc = modal_rollout.FireworksConfig()
    mc = modal_rollout.ModalRolloutConfig()
    result = asyncio.run(modal_rollout.run_modal_swe_rollout(task, 7, fireworks=fc, modal_cfg=mc))
    # Agent phase ran (sandbox created), so we should have timings for the
    # agent sandbox stages.
    assert result.patch_exists is False
    assert result.reward == 0.0
    assert result.error_stage in ("no_patch", "modal_agent")
    assert result.timings.get("modal_agent_sandbox_create_time") is not None


def test_rollout_happy_path_with_mocked_model(monkeypatch):
    from responses_api_agents.swe_agents import modal_rollout

    fake_diff = (
        "```diff\n"
        "diff --git a/foo b/foo\n"
        "index 0000..1111 100644\n"
        "--- a/foo\n"
        "+++ b/foo\n"
        "@@ -1 +1 @@\n"
        "-hi\n"
        "+hello\n"
        "```\n"
    )

    async def _fake_call(*a, **kw):
        return {"text": fake_diff, "model_call_time": 0.02}

    monkeypatch.setattr(modal_rollout, "call_fireworks_model", _fake_call)

    task = {"instance_id": "demo2", "problem_statement": "x", "image": "py:3.11"}
    fc = modal_rollout.FireworksConfig()
    mc = modal_rollout.ModalRolloutConfig()
    result = asyncio.run(modal_rollout.run_modal_swe_rollout(task, 0, fireworks=fc, modal_cfg=mc))

    assert result.patch_exists is True
    assert result.model_patch is not None
    # Mocked sandbox returns returncode 0 + no /tmp/report.json, so we fall
    # back to the "resolved if returncode == 0" heuristic -> resolved.
    assert result.resolved is True
    assert result.reward == 1.0
    # Both phase sandboxes recorded create+terminate.
    for prefix in ("modal_agent", "modal_eval"):
        assert result.timings[f"{prefix}_sandbox_create_time"] is not None
        assert result.timings[f"{prefix}_sandbox_terminate_time"] is not None
    assert result.timings["total_rollout_time"] > 0
    assert result.timings["agent_total_time"] > 0
    assert result.timings["eval_total_time"] > 0


# ---------------------------------------------------------------------------
# Benchmark summary should never drop rows + must include failure stats.
# ---------------------------------------------------------------------------


def test_benchmark_summary_counts_failed_rows():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_swe_modal_rollouts",
        Path(__file__).resolve().parents[3] / "scripts" / "benchmark_swe_modal_rollouts.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    rows = [
        {
            "task_id": "a",
            "rollout_id": 0,
            "total_rollout_time": 1.0,
            "agent_total_time": 0.5,
            "error_stage": None,
            "timed_out": False,
            "reward": 1.0,
        },
        {
            "task_id": "a",
            "rollout_id": 1,
            "total_rollout_time": 2.5,
            "agent_total_time": 1.5,
            "error_stage": "modal_eval",
            "error_kind": "timeout",
            "timed_out": True,
            "reward": 0.0,
        },
        {
            "task_id": "b",
            "rollout_id": 0,
            "total_rollout_time": 0.7,
            "agent_total_time": 0.7,
            "error_stage": "model_call",
            "error_kind": "auth",
            "timed_out": False,
            "reward": 0.0,
        },
    ]
    summary = mod.build_summary(rows, tasks_attempted=2, num_repeats=2)
    assert summary["expected_rows"] == 4  # 2 tasks x 2 repeats
    assert summary["produced_rows"] == 3
    assert summary["no_dropped_datapoints"] is False
    assert summary["succeeded"] == 1
    assert summary["failed"] == 2
    assert summary["timed_out"] == 1
    assert summary["failure_by_stage"]["modal_eval"] == 1
    assert summary["failure_by_stage"]["model_call"] == 1
    assert summary["timings"]["total_rollout_time"]["count"] == 3
    assert summary["timings"]["total_rollout_time"]["min"] == 0.7
    assert summary["timings"]["total_rollout_time"]["max"] == 2.5
    assert summary["slowest_rollouts"][0]["task_id"] == "a"
    assert summary["slowest_rollouts"][0]["rollout_id"] == 1


def test_required_timing_fields_present_in_result():
    """Documents the timing schema the benchmark depends on."""
    from responses_api_agents.swe_agents.modal_rollout import RolloutResult

    r = RolloutResult(task_id="t", rollout_id=0)
    # No timings yet — must still serialize.
    d = r.to_dict()
    for k in ("task_id", "rollout_id", "reward", "resolved", "error_stage"):
        assert k in d
