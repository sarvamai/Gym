# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Experimental SWE-bench style rollout that exercises the Modal sandbox path
# and Fireworks model API. NOT production ready. This is the smallest possible
# end-to-end loop that touches every interesting timing bucket so we can later
# hill-climb on slow stages.
#
# TODO(production): integrate this path into SWEBenchWrapper via a backend
# selector (apptainer vs modal). For now the regular Gym SWE rollout is
# untouched; only `scripts/benchmark_swe_modal_rollouts.py` uses this module.
# TODO(production): replace the trivial single-turn agent prompt with the
# full OpenHands / SWE-agent loop, but inside the Modal sandbox.
# TODO(production): cache snapshots of the seeded sandbox per instance_id so
# subsequent rollouts can restore instead of re-bootstrapping.
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from openai import AsyncOpenAI

from responses_api_agents.swe_agents.modal_sandbox import (
    KNOWN_STAGES,
    ModalSandbox,
    build_image_from_docker,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

# TODO(production): move all of this into a Pydantic config tied to
# SWEBenchWrapperConfig. Right now we accept a plain dict / dataclass for
# experimental wiring.


@dataclass
class FireworksConfig:
    """Fireworks API config. Fireworks exposes an OpenAI-compatible endpoint."""

    base_url: str = "https://api.fireworks.ai/inference/v1"
    api_key_env: str = "FIREWORKS_API_KEY"
    model: str = "accounts/fireworks/models/gpt-oss-120b"
    # Reasoning models like gpt-oss-120b spend most tokens in reasoning_content
    # before producing the visible answer. Default high enough to leave room.
    max_output_tokens: int = 16384
    temperature: float = 0.7
    request_timeout: float = 600.0

    def get_api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


@dataclass
class ModalRolloutConfig:
    """Experimental Modal sandbox config for one rollout."""

    app_name: str = "ng-swe-modal-bench"
    # Image used when no instance-specific image is supplied. We try to pull a
    # swebench eval image if the task gives us one.
    fallback_image: str = "python:3.11-slim"
    sandbox_timeout: int = 30 * 60
    agent_command_timeout: int = 5 * 60
    eval_command_timeout: int = 20 * 60
    cpu: Optional[float] = None
    memory: Optional[int] = None
    snapshot_mode: str = "none"  # one of: none, per_rollout
    # If True we attempt one ModalSandbox per phase (agent & eval). If False we
    # reuse the same sandbox. Default True so create/terminate timings reflect
    # the worst-case path.
    separate_sandboxes: bool = True

    # TODO(production): plumb via YAML.


@dataclass
class RolloutResult:
    task_id: str
    rollout_id: int
    reward: float = 0.0
    resolved: bool = False
    patch_exists: bool = False
    model_patch: Optional[str] = None
    timed_out: bool = False
    error_stage: Optional[str] = None
    error_kind: Optional[str] = None
    error_message: Optional[str] = None
    timings: Dict[str, Any] = field(default_factory=dict)
    raw_model_output: Optional[str] = None
    raw_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "reward": self.reward,
            "resolved": self.resolved,
            "patch_exists": self.patch_exists,
            "model_patch": self.model_patch,
            "timed_out": self.timed_out,
            "error_stage": self.error_stage,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "raw_model_output": self.raw_model_output,
            "raw_report": self.raw_report,
            **self.timings,
        }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def extract_patch(text: str) -> Optional[str]:
    """Pull a unified diff out of a model response.

    Looks for a fenced block first, then falls back to anything starting with
    `diff --git` or `--- a/`.
    """
    if not text:
        return None
    # Fenced block heuristic.
    fence_markers = ("```diff", "```patch", "```")
    for marker in fence_markers:
        if marker in text:
            after = text.split(marker, 1)[1]
            inner = after.split("```", 1)[0]
            if "diff --git" in inner or inner.lstrip().startswith("--- "):
                return inner.strip() + "\n"
    if "diff --git" in text:
        idx = text.index("diff --git")
        return text[idx:].strip() + "\n"
    return None


def classify_error(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "timeout" in name.lower() or "timeout" in msg:
        return "timeout"
    if "connection" in msg or "network" in msg:
        return "network"
    if "auth" in msg or "401" in msg or "403" in msg:
        return "auth"
    return name


def _flatten_sandbox_timings(prefix: str, sandbox: ModalSandbox) -> Dict[str, Any]:
    """Convert ModalSandbox.timings into rollout-row fields.

    Produces:
        f"{prefix}_sandbox_create_time"
        f"{prefix}_sandbox_file_upload_time"
        f"{prefix}_command_exec_time"
        f"{prefix}_output_download_time"
        f"{prefix}_sandbox_terminate_time"
        f"{prefix}_snapshot_create_time"
        f"{prefix}_snapshot_restore_time"
    Each is total across calls in that stage.
    """
    label_map = {
        "create": "sandbox_create_time",
        "upload": "sandbox_file_upload_time",
        "exec": "command_exec_time",
        "download": "output_download_time",
        "terminate": "sandbox_terminate_time",
        "snapshot_create": "snapshot_create_time",
        "snapshot_restore": "snapshot_restore_time",
    }
    out: Dict[str, Any] = {}
    for stage in KNOWN_STAGES:
        label = label_map[stage]
        vals = sandbox.timings.get(stage, [])
        out[f"{prefix}_{label}"] = sum(vals) if vals else None
    return out


# ----------------------------------------------------------------------------
# Fireworks model call
# ----------------------------------------------------------------------------


async def call_fireworks_model(
    problem_statement: str,
    fireworks: FireworksConfig,
) -> Dict[str, Any]:
    """Single-turn model call. Returns dict with text + timings."""
    t0 = time.perf_counter()
    api_key = fireworks.get_api_key()
    if not api_key:
        raise RuntimeError(
            f"Fireworks API key not set. Set env var {fireworks.api_key_env}."
        )

    client = AsyncOpenAI(
        base_url=fireworks.base_url,
        api_key=api_key,
        timeout=fireworks.request_timeout,
    )
    # Single-turn prompt asking for a unified diff patch.
    system = (
        "You are a software engineer. Given a problem description for a code "
        "repository, produce a unified diff patch (output as a ```diff fenced "
        "block) that fixes the issue. Do not include explanation outside the "
        "diff."
    )
    user = problem_statement
    completion = await client.chat.completions.create(
        model=fireworks.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=fireworks.max_output_tokens,
        temperature=fireworks.temperature,
    )
    elapsed = time.perf_counter() - t0
    msg = completion.choices[0].message
    # Reasoning models (gpt-oss, deepseek-r1, etc.) often emit the diff in
    # `reasoning_content` when message.content is empty or contains the chain
    # of thought rather than the final answer. Concatenate both so the patch
    # extractor can find the diff either way.
    parts: List[str] = []
    content = getattr(msg, "content", None)
    reasoning = getattr(msg, "reasoning_content", None)
    if content:
        parts.append(content)
    if reasoning and reasoning not in parts:
        parts.append(reasoning)
    text = "\n".join(parts) if parts else ""
    finish = completion.choices[0].finish_reason
    return {
        "text": text,
        "model_call_time": elapsed,
        "finish_reason": finish,
        "had_reasoning_content": bool(reasoning),
    }


# ----------------------------------------------------------------------------
# Main rollout
# ----------------------------------------------------------------------------


async def run_modal_swe_rollout(
    task: Dict[str, Any],
    rollout_id: int,
    *,
    fireworks: FireworksConfig,
    modal_cfg: ModalRolloutConfig,
) -> RolloutResult:
    """Run one experimental SWE-bench style rollout against Modal + Fireworks.

    `task` schema (minimal):
        {
          "instance_id": str,
          "problem_statement": str,
          "image": str | None,   # docker image ref for eval/agent sandbox
          "test_command": str | None,  # shell to invoke after applying patch
        }

    Always returns a RolloutResult. Failures populate error_* + reward=0.0;
    we never raise out of this function (the benchmark relies on that).
    """
    task_id = task.get("instance_id") or task.get("task_id") or "unknown"
    result = RolloutResult(task_id=task_id, rollout_id=rollout_id)
    rollout_t0 = time.perf_counter()
    run_start_time = time.time()
    timings: Dict[str, Any] = {
        "run_start_time": run_start_time,
        "agent_total_time": None,
        "model_call_time": None,
        "patch_extraction_time": None,
        "eval_total_time": None,
        "test_execution_time": None,
        "report_parse_time": None,
        "modal_snapshot_mode": modal_cfg.snapshot_mode,
        "modal_snapshot_id": None,
    }
    # Pre-populate sandbox stage fields with None so failed rows still have
    # every key.
    _STAGE_LABELS = (
        "sandbox_create_time",
        "sandbox_file_upload_time",
        "command_exec_time",
        "output_download_time",
        "sandbox_terminate_time",
        "snapshot_create_time",
        "snapshot_restore_time",
    )
    for prefix in ("modal_agent", "modal_eval"):
        for label in _STAGE_LABELS:
            timings[f"{prefix}_{label}"] = None

    image_ref = task.get("image") or modal_cfg.fallback_image
    # ------------------------------------------------------------ agent phase
    agent_t0 = time.perf_counter()
    patch_text: Optional[str] = None
    model_output_text: Optional[str] = None
    agent_sandbox: Optional[ModalSandbox] = None
    try:
        # 1) Model call (Fireworks)
        try:
            model_out = await call_fireworks_model(
                task.get("problem_statement", ""), fireworks
            )
        except Exception as e:
            result.error_stage = "model_call"
            result.error_kind = classify_error(e)
            result.error_message = str(e)[:1000]
            timings["agent_total_time"] = time.perf_counter() - agent_t0
            timings["total_rollout_time"] = time.perf_counter() - rollout_t0
            timings["run_end_time"] = time.time()
            result.timings = timings
            return result

        model_output_text = model_out["text"]
        timings["model_call_time"] = model_out["model_call_time"]
        timings["model_finish_reason"] = model_out.get("finish_reason")
        timings["model_had_reasoning_content"] = model_out.get("had_reasoning_content")
        result.raw_model_output = model_output_text

        # 2) Patch extraction
        t_extract = time.perf_counter()
        patch_text = extract_patch(model_output_text)
        timings["patch_extraction_time"] = time.perf_counter() - t_extract
        result.model_patch = patch_text
        result.patch_exists = bool(patch_text)

        # 3) Modal agent sandbox: create + sanity exec. Even when we have a
        # patch we exercise the sandbox so the bench captures full lifecycle
        # timings.
        # TODO(production): in a real multi-turn flow the agent_sandbox would
        # host the actual edit/test/iterate loop. Here we only verify the
        # sandbox is reachable and capture timing.
        try:
            image = build_image_from_docker(image_ref)
            agent_sandbox = ModalSandbox(
                app_name=modal_cfg.app_name,
                image=image,
                timeout=modal_cfg.sandbox_timeout,
                cpu=modal_cfg.cpu,
                memory=modal_cfg.memory,
                label=f"agent-{task_id}-{rollout_id}",
            )
            await agent_sandbox.create()
            await agent_sandbox.exec(
                "bash", "-lc", "echo hello && uname -a",
                timeout=modal_cfg.agent_command_timeout,
            )
            if modal_cfg.snapshot_mode == "per_rollout":
                snap_id = await agent_sandbox.snapshot_filesystem()
                timings["modal_snapshot_id"] = snap_id
        except Exception as e:
            result.error_stage = "modal_agent"
            result.error_kind = classify_error(e)
            result.error_message = str(e)[:1000]
        finally:
            if agent_sandbox is not None:
                try:
                    await agent_sandbox.terminate()
                except Exception:
                    pass
                timings.update(_flatten_sandbox_timings("modal_agent", agent_sandbox))

    finally:
        timings["agent_total_time"] = time.perf_counter() - agent_t0

    # If the agent phase already failed, bail early but still write a row.
    if result.error_stage is not None and result.error_stage != "modal_agent":
        # model_call failed already returned earlier; this branch is for safety.
        timings["total_rollout_time"] = time.perf_counter() - rollout_t0
        timings["run_end_time"] = time.time()
        result.timings = timings
        return result

    # ------------------------------------------------------------- eval phase
    eval_sandbox: Optional[ModalSandbox] = None
    eval_t0 = time.perf_counter()
    if not patch_text:
        # No patch -> reward 0, but still record the row.
        result.error_stage = result.error_stage or "no_patch"
        result.error_kind = result.error_kind or "no_patch"
        result.error_message = result.error_message or "model returned no patch"
        timings["eval_total_time"] = 0.0
        timings["total_rollout_time"] = time.perf_counter() - rollout_t0
        timings["run_end_time"] = time.time()
        result.timings = timings
        return result

    try:
        image = build_image_from_docker(image_ref)
        eval_sandbox = ModalSandbox(
            app_name=modal_cfg.app_name,
            image=image,
            timeout=modal_cfg.sandbox_timeout,
            cpu=modal_cfg.cpu,
            memory=modal_cfg.memory,
            label=f"eval-{task_id}-{rollout_id}",
        )
        await eval_sandbox.create()
        await eval_sandbox.upload_text("/tmp/patch.diff", patch_text)

        # Apply patch + run tests. Mirrors the SWE-bench eval contract loosely.
        test_command = task.get("test_command") or (
            "cd /testbed 2>/dev/null || cd / && "
            "git apply /tmp/patch.diff 2>&1 || patch -p1 < /tmp/patch.diff 2>&1 || true && "
            "echo '__NG_TESTS_BEGIN__' && "
            "(pytest -x --tb=short 2>&1 || true) && "
            "echo '__NG_TESTS_END__'"
        )
        t_tests = time.perf_counter()
        exec_result = await eval_sandbox.exec(
            "bash", "-lc", test_command,
            timeout=modal_cfg.eval_command_timeout,
        )
        timings["test_execution_time"] = time.perf_counter() - t_tests
        if exec_result.timed_out:
            result.timed_out = True
            result.error_stage = "modal_eval"
            result.error_kind = "timeout"
            result.error_message = "eval command timed out"

        # Parse a report file if the eval command produced one. SWE-bench eval
        # images typically write /tmp/report.json. If not, fall back to exit
        # code heuristic.
        t_parse = time.perf_counter()
        report: Dict[str, Any] = {}
        try:
            report_text = await eval_sandbox.read_text("/tmp/report.json")
            report = json.loads(report_text)
        except Exception:
            # Fall back to "tests passed if returncode == 0".
            report = {
                task_id: {
                    "resolved": (exec_result.returncode == 0 and not exec_result.timed_out),
                    "raw_stdout_tail": exec_result.stdout[-2000:],
                }
            }
        timings["report_parse_time"] = time.perf_counter() - t_parse
        result.raw_report = report
        instance_report = report.get(task_id) or next(iter(report.values()), {})
        result.resolved = bool(instance_report.get("resolved", False))
        result.reward = 1.0 if result.resolved else 0.0
    except Exception as e:
        result.error_stage = "modal_eval"
        result.error_kind = classify_error(e)
        result.error_message = str(e)[:1000]
    finally:
        if eval_sandbox is not None:
            try:
                await eval_sandbox.terminate()
            except Exception:
                pass
            timings.update(_flatten_sandbox_timings("modal_eval", eval_sandbox))
        timings["eval_total_time"] = time.perf_counter() - eval_t0

    timings["total_rollout_time"] = time.perf_counter() - rollout_t0
    timings["run_end_time"] = time.time()
    result.timings = timings
    return result


async def run_modal_swe_rollouts_concurrent(
    tasks: List[Dict[str, Any]],
    *,
    num_repeats: int,
    concurrency: int,
    fireworks: FireworksConfig,
    modal_cfg: ModalRolloutConfig,
    on_complete: Optional[Callable[["RolloutResult"], None]] = None,
) -> List[RolloutResult]:
    """Fan out `tasks x num_repeats` rollouts with bounded concurrency.

    Every (task, rollout_id) produces exactly one RolloutResult — even on
    failure. The benchmark relies on this no-drop guarantee.
    """
    sem = asyncio.Semaphore(concurrency)
    results: List[RolloutResult] = []
    lock = asyncio.Lock()

    async def _one(task: Dict[str, Any], rid: int) -> None:
        async with sem:
            try:
                r = await run_modal_swe_rollout(
                    task, rid, fireworks=fireworks, modal_cfg=modal_cfg
                )
            except Exception as e:
                # Last-resort net: should never trigger because run_modal_swe_rollout
                # is supposed to swallow internal errors. Still, never drop a row.
                r = RolloutResult(
                    task_id=task.get("instance_id", "unknown"),
                    rollout_id=rid,
                    reward=0.0,
                    error_stage="rollout_uncaught",
                    error_kind=classify_error(e),
                    error_message=str(e)[:1000],
                    timings={"total_rollout_time": None},
                )
            async with lock:
                results.append(r)
                if on_complete is not None:
                    on_complete(r)

    coros = []
    for task in tasks:
        for rid in range(num_repeats):
            coros.append(_one(task, rid))
    await asyncio.gather(*coros)
    return results
