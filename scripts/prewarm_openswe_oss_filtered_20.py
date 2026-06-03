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

Skip already-built / manifest merge / retry:
  Before building, the script lists the e2b server's templates once and SKIPS any
  task whose alias is already `buildStatus == "ready"` (the alias is content-
  addressed, so a ready alias is the same environment) — pass `--force-rebuild` to
  rebuild anyway. Separately, if `--manifest` already exists, tasks recorded as
  `status == "ok"` are carried forward. Together these make the run_openswe_e2b.sh
  two-pass flow cheap: the parallel pass reuses ready templates and builds only the
  missing ones; the reduced-concurrency retry pass then rebuilds ONLY the failures.

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
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild every template even if an identical alias is already 'ready' on "
        "the e2b server. Default: skip already-ready aliases (the alias is content-"
        "addressed, so a ready alias is the same environment already built).",
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


def fetch_ready_aliases() -> set[str]:
    """Lowercased aliases whose latest build is 'ready' on the e2b server.

    One `get_templates` call. Aliases are content-addressed, so a ready alias means
    that exact environment is already built and can be reused without rebuilding.
    e2b stores aliases lowercased, so callers must compare case-insensitively.
    """
    from e2b.api.client.api.templates import get_templates
    from e2b.api.client_sync import get_api_client
    from e2b.connection_config import ConnectionConfig

    client = get_api_client(ConnectionConfig(), require_api_key=True, require_access_token=False)
    items = get_templates.sync_detailed(client=client).parsed or []
    ready: set[str] = set()
    for t in items:
        d = t.to_dict()
        if str(d.get("buildStatus")).lower() == "ready":
            for alias in d.get("aliases") or []:
                ready.add(alias.lower())
    return ready


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

    # Carry forward already-ok records from a prior pass (local manifest); only
    # (re)build the rest. This makes the run_openswe_e2b.sh retry pass redo failures.
    already_ok = _load_ok_tasks(args.manifest)
    todo = [d for d in task_dirs if d.name not in already_ok]
    carried = len(task_dirs) - len(todo)

    # Skip tasks whose template alias is already 'ready' on the e2b server. The alias
    # is content-addressed (env dir hash), so a ready alias is the SAME environment
    # already built — reuse it instead of rebuilding every run.
    skipped_ready: list[dict] = []
    if not args.force_rebuild and todo:
        try:
            ready = fetch_ready_aliases()
        except Exception as exc:
            print(f"[warn] could not list server templates ({type(exc).__name__}: {exc}); building all", flush=True)
            ready = set()
        remaining = []
        for d in todo:
            alias = template_name_for(d)
            if alias.lower() in ready:
                skipped_ready.append({"task": d.name, "alias": alias, "status": "ok", "reused": "already_ready"})
            else:
                remaining.append(d)
        if skipped_ready:
            print(f"[skip] {len(skipped_ready)} templates already ready on server; reusing.", flush=True)
        todo = remaining

    concurrency = max(1, args.concurrency)
    print(
        f"Pre-warming {len(todo)}/{len(task_dirs)} templates (concurrency={concurrency}; "
        f"{carried} carried from manifest, {len(skipped_ready)} reused from server).",
        flush=True,
    )

    sema = asyncio.Semaphore(concurrency)
    n = len(todo)
    built = await asyncio.gather(*(_build_with_sema(d, args, sema, f"[{i}/{n}]") for i, d in enumerate(todo, 1)))

    by_task = {**already_ok, **{r["task"]: r for r in skipped_ready}, **{r["task"]: r for r in built}}
    # Preserve the original task-dir ordering in the manifest.
    records = [by_task[d.name] for d in task_dirs]

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"records": records}, indent=2))
    ok = sum(1 for r in records if r.get("status") == "ok")
    print(f"Done. ok={ok}/{len(records)}  manifest={args.manifest}")
    return 0 if ok == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
