"""Pre-warm e2b templates for the 20 sampled openswe_oss tasks.

Why this is necessary:
  e2b's API enforces singleton-per-template builds — `CheckAndCancelConcurrentBuilds`
  in `packages/api/internal/handlers/deprecated_template_start_build.go` actively
  cancels any in-progress build for the same template when a new one is registered.
  So 10 concurrent rollouts that each force-build will produce 9 cancellations.

  We dodge this by pre-warming once, sequentially, so that:
    - Each template build runs alone (no concurrent same-template cancellations)
    - On the production run, the alias is already present + working

  E2BLocalContextEnvironment then skips rebuild for templates already in its
  in-process `_built_templates` set, and falls back to a single rebuild + retry
  if it hits the stale-alias 404.

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


async def main_async(args: argparse.Namespace) -> int:
    task_dirs = sorted(p for p in args.tasks_dir.iterdir() if p.is_dir())
    if not task_dirs:
        raise SystemExit(f"No task dirs under {args.tasks_dir}")
    print(f"Pre-warming {len(task_dirs)} templates sequentially.")

    records: list[dict] = []
    for i, task_dir in enumerate(task_dirs, 1):
        print(f"[{i}/{len(task_dirs)}] {task_dir.name} ...", flush=True)
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
            print(f"  ERROR: {rec['error']}", flush=True)
        records.append(rec)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"records": records}, indent=2))
    ok = sum(1 for r in records if r.get("status") == "ok")
    print(f"Done. ok={ok}/{len(records)}  manifest={args.manifest}")
    return 0 if ok == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
