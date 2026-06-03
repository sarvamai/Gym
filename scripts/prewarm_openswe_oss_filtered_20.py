"""Pre-warm e2b templates for the sampled openswe_oss tasks.

Why this is necessary:
  e2b's API enforces singleton-per-template builds — `CheckAndCancelConcurrentBuilds`
  in `packages/api/internal/handlers/deprecated_template_start_build.go` actively
  cancels any in-progress build for the SAME template when a new one is registered.
  So 10 concurrent rollouts that each force-build the same template produce 9
  cancellations. The cancellation is per-template (per-alias), NOT a cluster
  capacity limit — see the `e2b-build-was-cancelled` note.

  We dodge this by pre-warming each template once, up front, so that on the
  production run the alias is already present + working. E2BLocalContextEnvironment
  then skips rebuild for templates already in its in-process `_built_templates`
  set, and falls back to a single rebuild + retry if it hits the stale-alias 404.

Concurrency:
  `--concurrency N` builds up to N templates at once. This is safe BECAUSE the
  template alias embeds the (unique) task dir name —
  `f"{task_dir.name}__{dirhash(environment)[:8]}"` — so distinct tasks always
  map to distinct aliases and the singleton-per-template cancellation can never
  fire across different tasks. (Two tasks would only collide if their names plus
  env hashes flattened to the same alias, which does not happen for these
  datasets.) `--concurrency 1` (the default) preserves the old sequential
  behaviour for direct callers.

Manifest merge / retry:
  If `--manifest` already exists, tasks recorded as `status == "ok"` are skipped
  and their records carried forward; only not-ok / new tasks are (re)built and
  merged in. This is what makes the run_openswe_e2b.sh two-pass flow viable: a
  fast parallel pass, then a sequential `--concurrency 1` retry that rebuilds
  ONLY the failures instead of all N templates again.

This script must run inside the harbor_agent venv because it imports
`harbor.environments.e2b` and the e2b SDK.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dirhash import dirhash
from e2b import AsyncTemplate, Template

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS_DIR = (
    REPO_ROOT
    / "responses_api_agents"
    / "harbor_agent"
    / "data"
    / "openswe_harbor_filtered_20"
    / "patched_tasks"
    / "openswe_oss"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tasks-dir",
        type=Path,
        default=DEFAULT_TASKS_DIR,
        help="Directory containing per-task subdirs (each with environment/Dockerfile).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_TASKS_DIR.parent.parent / "prewarm_manifest.json",
        help="Where to record which template aliases were built.",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=2,
        help="cpu_count passed to AsyncTemplate.build (matches harbor defaults).",
    )
    p.add_argument(
        "--memory-mb",
        type=int,
        default=8192,
        help="memory_mb passed to AsyncTemplate.build (matches openswe task.toml).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max templates to build in parallel. Safe because each task maps to a "
        "unique alias (singleton-per-template cancellation cannot fire across "
        "distinct tasks). Default 1 = sequential.",
    )
    return p.parse_args()


def template_name_for(task_dir: Path) -> str:
    # Mirrors harbor.environments.e2b.E2BEnvironment.__init__: alias =
    # f"{environment_name}__{dirhash(environment_dir, 'sha256')[:8]}".replace(".", "-")
    environment_name = task_dir.name
    environment_dir = task_dir / "environment"
    return f"{environment_name}__{dirhash(str(environment_dir), 'sha256')[:8]}".replace(".", "-")


async def build_one(task_dir: Path, cpus: int, memory_mb: int) -> dict:
    environment_dir = task_dir / "environment"
    dockerfile = environment_dir / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"No Dockerfile under {environment_dir}")

    alias = template_name_for(task_dir)
    template = Template(file_context_path=environment_dir).from_dockerfile(
        dockerfile_content_or_path=str(dockerfile),
    )
    started = time.time()
    await AsyncTemplate.build(
        template=template,
        alias=alias,
        cpu_count=cpus,
        memory_mb=memory_mb,
    )
    elapsed = time.time() - started
    return {"task": task_dir.name, "alias": alias, "elapsed_sec": elapsed}


def _load_ok_tasks(manifest: Path) -> dict[str, dict]:
    """Return {task_name: record} for tasks already built ok in an existing manifest."""
    if not manifest.is_file():
        return {}
    try:
        prior = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {r["task"]: r for r in prior.get("records", []) if r.get("status") == "ok"}


async def _build_with_sema(task_dir: Path, args: argparse.Namespace, sema: asyncio.Semaphore, label: str) -> dict:
    async with sema:
        print(f"{label} {task_dir.name} ...", flush=True)
        try:
            rec = await build_one(task_dir, args.cpus, args.memory_mb)
            rec["status"] = "ok"
            print(f"  ok in {rec['elapsed_sec']:.1f}s  alias={rec['alias']}", flush=True)
        except Exception as exc:
            rec = {
                "task": task_dir.name,
                "alias": template_name_for(task_dir),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {task_dir.name}: {rec['error']}", flush=True)
        return rec


async def main_async(args: argparse.Namespace) -> int:
    task_dirs = sorted(p for p in args.tasks_dir.iterdir() if p.is_dir())
    if not task_dirs:
        raise SystemExit(f"No task dirs under {args.tasks_dir}")

    # Carry forward already-ok records from a prior pass; only (re)build the rest.
    # This makes the run_openswe_e2b.sh sequential retry pass rebuild ONLY failures.
    already_ok = _load_ok_tasks(args.manifest)
    todo = [d for d in task_dirs if d.name not in already_ok]
    skipped = len(task_dirs) - len(todo)

    concurrency = max(1, args.concurrency)
    print(
        f"Pre-warming {len(todo)}/{len(task_dirs)} templates "
        f"(concurrency={concurrency}, {skipped} already ok carried forward).",
        flush=True,
    )

    sema = asyncio.Semaphore(concurrency)
    n = len(todo)
    built = await asyncio.gather(
        *(_build_with_sema(d, args, sema, f"[{i}/{n}]") for i, d in enumerate(todo, 1))
    )

    by_task = {**already_ok, **{r["task"]: r for r in built}}
    # Preserve the original task-dir ordering in the manifest.
    records = [by_task[d.name] for d in task_dirs]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"records": records}, indent=2))
    ok = sum(1 for r in records if r.get("status") == "ok")
    print(f"Done. ok={ok}/{len(records)}  manifest={args.manifest}")
    return 0 if ok == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
