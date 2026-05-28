#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Experimental benchmark runner for SWE-bench-style rollouts that exercises
# the Modal sandbox path (responses_api_agents.swe_agents.modal_rollout) and
# Fireworks model API.
#
# This script is intentionally minimal and NOT production ready. It exists to
# measure full rollout lifecycle timings so we can hill-climb on slow stages.
#
# Usage:
#   python scripts/benchmark_swe_modal_rollouts.py \
#       --input-jsonl path/to/tasks.jsonl \
#       --output-dir results/modal_bench \
#       --limit 4 \
#       --num-repeats 16 \
#       --concurrency 4
#
# Required env: FIREWORKS_API_KEY (and Modal must already be authenticated via
# `modal token new`).
#
# TODO(production): wire through Gym ng_collect_rollouts CLI by registering a
# config-driven backend selector instead of a standalone script.
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

# Make `responses_api_agents` importable when run from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from responses_api_agents.swe_agents.modal_rollout import (  # noqa: E402
    FireworksConfig,
    ModalRolloutConfig,
    RolloutResult,
    run_modal_swe_rollouts_concurrent,
)
from responses_api_agents.swe_agents.modal_sandbox import (  # noqa: E402
    aggregate_iterable_floats,
)

logger = logging.getLogger("benchmark_swe_modal_rollouts")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_tasks_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    """Load tasks. Accepts either NeMo-Gym schema (responses_create_params +
    verifier_metadata) or a flat schema with `instance_id`/`problem_statement`.
    """
    tasks: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(_normalize_task(row))
            if len(tasks) >= limit:
                break
    return tasks


def _normalize_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """Try to map a Gym JSONL row to our minimal task schema."""
    if "instance_id" in row and "problem_statement" in row:
        return row
    md = row.get("verifier_metadata") or {}
    instance_id = (
        md.get("instance_id")
        or md.get("task_id")
        or row.get("task_id")
        or "unknown"
    )
    problem_statement = (
        md.get("problem_statement")
        or md.get("text")
        or _extract_user_text(row)
        or ""
    )
    image = (
        md.get("image")
        or md.get("container")
        or md.get("docker_image")
    )
    test_command = md.get("test_command") or md.get("eval_command")
    return {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "image": image,
        "test_command": test_command,
        "_raw": row,
    }


def _extract_user_text(row: Dict[str, Any]) -> Optional[str]:
    params = row.get("responses_create_params") or {}
    msgs = params.get("input") or []
    for m in reversed(msgs):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
    return None


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


TIMING_FIELDS = [
    "total_rollout_time",
    "agent_total_time",
    "model_call_time",
    "patch_extraction_time",
    "eval_total_time",
    "test_execution_time",
    "report_parse_time",
    "modal_agent_sandbox_create_time",
    "modal_agent_sandbox_file_upload_time",
    "modal_agent_command_exec_time",
    "modal_agent_output_download_time",
    "modal_agent_sandbox_terminate_time",
    "modal_agent_snapshot_create_time",
    "modal_agent_snapshot_restore_time",
    "modal_eval_sandbox_create_time",
    "modal_eval_sandbox_file_upload_time",
    "modal_eval_command_exec_time",
    "modal_eval_output_download_time",
    "modal_eval_sandbox_terminate_time",
    "modal_eval_snapshot_create_time",
    "modal_eval_snapshot_restore_time",
]


def build_summary(
    rows: List[Dict[str, Any]],
    *,
    tasks_attempted: int,
    num_repeats: int,
) -> Dict[str, Any]:
    expected = tasks_attempted * num_repeats
    succeeded = [r for r in rows if r.get("error_stage") is None]
    failed = [r for r in rows if r.get("error_stage") is not None]
    timed_out = [r for r in rows if r.get("timed_out")]

    timings: Dict[str, Dict[str, float]] = {}
    for f in TIMING_FIELDS:
        vals = [r.get(f) for r in rows if isinstance(r.get(f), (int, float))]
        agg = aggregate_iterable_floats(vals) if vals else {}
        timings[f] = agg

    by_stage: Dict[str, int] = {}
    for r in failed:
        stage = r.get("error_stage") or "unknown"
        by_stage[stage] = by_stage.get(stage, 0) + 1

    slowest = sorted(
        rows,
        key=lambda r: r.get("total_rollout_time") or 0.0,
        reverse=True,
    )[:10]
    slowest_view = [
        {
            "task_id": r.get("task_id"),
            "rollout_id": r.get("rollout_id"),
            "total_rollout_time": r.get("total_rollout_time"),
            "error_stage": r.get("error_stage"),
            "error_kind": r.get("error_kind"),
        }
        for r in slowest
    ]

    return {
        "expected_rows": expected,
        "produced_rows": len(rows),
        "no_dropped_datapoints": len(rows) == expected,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "timed_out": len(timed_out),
        "failure_by_stage": by_stage,
        "timings": timings,
        "slowest_rollouts": slowest_view,
    }


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["field,min,p50,p90,p95,max,mean,count"]
    for f, agg in summary["timings"].items():
        if not agg:
            lines.append(f"{f},,,,,,,0")
            continue
        row_values = ",".join(
            str(agg.get(k, "")) for k in ("min", "p50", "p90", "p95", "max", "mean", "count")
        )
        lines.append(f"{f},{row_values}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark SWE Modal rollouts.")
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=4, help="max tasks from input jsonl")
    ap.add_argument("--num-repeats", type=int, default=16, help="rollouts per task")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--fireworks-model",
        type=str,
        default="accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
    )
    ap.add_argument("--fireworks-base-url", default="https://api.fireworks.ai/inference/v1")
    ap.add_argument("--modal-app-name", default="ng-swe-modal-bench")
    ap.add_argument("--fallback-image", default="python:3.11-slim")
    ap.add_argument("--sandbox-timeout", type=int, default=30 * 60)
    ap.add_argument("--snapshot-mode", default="none", choices=["none", "per_rollout"])
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def check_credentials(fireworks: FireworksConfig) -> List[str]:
    """Return a list of blocker strings if creds are missing."""
    blockers: List[str] = []
    if not fireworks.get_api_key():
        blockers.append(
            f"FIREWORKS_API_KEY not set (env var {fireworks.api_key_env})."
        )
    # Modal: rely on its own config file existing.
    modal_cfg_path = Path.home() / ".modal.toml"
    if not modal_cfg_path.exists():
        blockers.append(
            "Modal credentials not configured. Run `modal token new` first "
            f"(expected config at {modal_cfg_path})."
        )
    return blockers


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fireworks = FireworksConfig(
        base_url=args.fireworks_base_url,
        model=args.fireworks_model,
    )
    modal_cfg = ModalRolloutConfig(
        app_name=args.modal_app_name,
        fallback_image=args.fallback_image,
        sandbox_timeout=args.sandbox_timeout,
        snapshot_mode=args.snapshot_mode,
    )

    blockers = check_credentials(fireworks)
    if blockers:
        logger.error("Cannot start benchmark — missing credentials:")
        for b in blockers:
            logger.error("  - %s", b)
        return 2

    tasks = load_tasks_jsonl(args.input_jsonl, args.limit)
    logger.info(
        "Loaded %d tasks (limit=%d) x num_repeats=%d -> %d rollouts",
        len(tasks),
        args.limit,
        args.num_repeats,
        len(tasks) * args.num_repeats,
    )
    if not tasks:
        logger.error("No tasks loaded from %s", args.input_jsonl)
        return 2

    total = len(tasks) * args.num_repeats
    pbar = tqdm(total=total, desc="rollouts")

    def _on_done(_r: RolloutResult) -> None:
        pbar.update(1)

    started = time.time()
    results = await run_modal_swe_rollouts_concurrent(
        tasks,
        num_repeats=args.num_repeats,
        concurrency=args.concurrency,
        fireworks=fireworks,
        modal_cfg=modal_cfg,
        on_complete=_on_done,
    )
    pbar.close()
    elapsed = time.time() - started

    rows = [r.to_dict() for r in results]
    raw_path = args.output_dir / "rollouts.jsonl"
    write_jsonl(raw_path, rows)
    logger.info("Wrote %d raw rollouts -> %s", len(rows), raw_path)

    summary = build_summary(rows, tasks_attempted=len(tasks), num_repeats=args.num_repeats)
    summary["wall_clock_seconds"] = elapsed
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote summary -> %s", summary_path)

    csv_path = args.output_dir / "summary.csv"
    write_summary_csv(csv_path, summary)
    logger.info("Wrote summary csv -> %s", csv_path)

    logger.info(
        "Done. produced=%d expected=%d succeeded=%d failed=%d wall=%.1fs",
        summary["produced_rows"],
        summary["expected_rows"],
        summary["succeeded"],
        summary["failed"],
        elapsed,
    )
    if not summary["no_dropped_datapoints"]:
        logger.error("Row count mismatch — some rollouts were dropped!")
        return 3
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
