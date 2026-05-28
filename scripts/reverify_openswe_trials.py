"""Re-verify openswe trials by replaying the agent's bash commands into a fresh
e2b sandbox and running the (now-patched) verifier.

Why: the original 96 rollouts all reported reward=0 because /tests/test.sh
invoked /tests/openswe_eval.sh directly, but e2b strips the +x bit on file
upload — so the verifier never ran. We've fixed test.sh locally to do
`chmod +x` + `bash …`. To get the real rewards without re-running the LLM,
we replay each trial's bash command sequence into a fresh sandbox from the
prewarmed template alias, then re-run the verifier.

Outputs per trial:
  <trial_dir>/reverify/reward.txt        — "1.0" or "0.0"
  <trial_dir>/reverify/test-stdout.txt   — full verifier stdout
  <trial_dir>/reverify/patch.diff        — `git diff <base_commit>..HEAD` from the replayed sandbox
  <trial_dir>/reverify/replay.log        — per-command exit codes + timings
And a top-level JSONL summary at the path passed via --output-jsonl.

Run from the harbor_agent venv (it has e2b SDK + dirhash + tomllib):
  responses_api_agents/harbor_agent/.venv/bin/python scripts/reverify_openswe_trials.py \
      --jobs-root responses_api_agents/harbor_agent/jobs/20260526/openswe_oss/kimi-k2p6 \
      --tasks-root responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/patched_tasks/openswe_oss \
      --rollouts-jsonl responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/rollouts_20260526_092017.jsonl \
      --output-jsonl responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/reverify_<ts>.jsonl \
      --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from dirhash import dirhash
from e2b import AsyncSandbox

BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)


@dataclass
class TrialJob:
    trial_dir: Path
    task_name: str
    base_commit: str
    template_alias: str
    local_tests_dir: Path
    commands: list[str]


def template_alias_for(task_dir: Path) -> str:
    """Match harbor.environments.e2b.E2BEnvironment.__init__'s alias formula."""
    environment_name = task_dir.name
    environment_dir = task_dir / "environment"
    return f"{environment_name}__{dirhash(str(environment_dir), 'sha256')[:8]}".replace(".", "-")


def extract_commands(trajectory_path: Path) -> list[str]:
    """Pull the bash command from each assistant turn in the mini-swe trajectory."""
    obj = json.loads(trajectory_path.read_text())
    msgs = obj.get("messages") or []
    cmds: list[str] = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        match = BASH_BLOCK_RE.search(m.get("content") or "")
        if match:
            cmd = match.group(1).strip()
            if cmd:
                cmds.append(cmd)
    return cmds


def discover_trials(jobs_root: Path, tasks_root: Path, want_task_names: set[str] | None) -> list[TrialJob]:
    jobs: list[TrialJob] = []
    # Layout: <jobs_root>/<HHMMSS_xxx>/<task>__<rand>/agent/mini-swe-agent.trajectory.json
    # WARN: trial_dir.name often has the task component truncated (harbor caps the
    # combined trial_name length), e.g. `Clueless-Community__scrape-up-1014__xxxxx`
    # becomes `Clueless-Community__scrape-up-10__xxxxx` on disk. So we read the
    # real task path from each trial's config.json rather than parsing the dir name.
    repo_root = Path(__file__).resolve().parents[1]
    for trial_dir in sorted(jobs_root.glob("*/*")):
        if not trial_dir.is_dir():
            continue
        traj = trial_dir / "agent" / "mini-swe-agent.trajectory.json"
        cfg_path = trial_dir / "config.json"
        if not traj.is_file() or not cfg_path.is_file():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            task_path_str = cfg["task"]["path"]
        except Exception:
            continue
        task_dir = Path(task_path_str)
        if not task_dir.is_absolute():
            task_dir = repo_root / task_dir
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if want_task_names and task_name not in want_task_names:
            continue
        toml_meta = tomllib.loads((task_dir / "task.toml").read_text())["metadata"]
        cmds = extract_commands(traj)
        if not cmds:
            continue
        jobs.append(
            TrialJob(
                trial_dir=trial_dir,
                task_name=task_name,
                base_commit=toml_meta["base_commit"],
                template_alias=template_alias_for(task_dir),
                local_tests_dir=task_dir / "tests",
                commands=cmds,
            )
        )
    return jobs


async def upload_dir(sandbox: AsyncSandbox, local_dir: Path, remote_dir: str) -> None:
    """Recursively upload local_dir into remote_dir. Writes as root because
    /tests and similar harbor paths are root-owned in the prewarmed templates."""
    await sandbox.commands.run(f"mkdir -p {remote_dir}", user="root")
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            remote = f"{remote_dir}/{rel}"
            await sandbox.commands.run(f"mkdir -p $(dirname {remote})", user="root")
            await sandbox.files.write(remote, p.read_text(), user="root")


async def replay_one(job: TrialJob, sem: asyncio.Semaphore, command_timeout: int) -> dict:
    out_dir = job.trial_dir / "reverify"
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_path = out_dir / "reward.txt"
    stdout_path = out_dir / "test-stdout.txt"
    patch_path = out_dir / "patch.diff"
    log_path = out_dir / "replay.log"

    result: dict = {
        "task_name": job.task_name,
        "trial_dir": str(job.trial_dir),
        "n_commands": len(job.commands),
        "reward": None,
        "exception": None,
        "elapsed_sec": None,
    }

    async with sem:
        started = time.time()
        log_lines: list[str] = [f"replay start  template={job.template_alias} cmds={len(job.commands)}"]
        sandbox: AsyncSandbox | None = None
        try:
            sandbox = await AsyncSandbox.create(
                template=job.template_alias,
                timeout=3500,
            )
            log_lines.append(f"sandbox id={sandbox.sandbox_id}")

            # Replay: chain all commands into ONE bash script so cwd / env state
            # carries across commands (mini-swe-agent runs in a persistent shell;
            # sandbox.commands.run uses a fresh shell each call, which breaks
            # `cd subdir` followed by later `ls` in that subdir).
            script_path = "/tmp/__replay.sh"
            replay_script = "#!/bin/bash\ncd /testbed\n" + "\n".join(
                # Sentinel echo per command so we can post-mortem which one failed.
                f"echo '__REPLAY_CMD__ {i}'\n{cmd}\n" for i, cmd in enumerate(job.commands)
            ) + "\necho '__REPLAY_DONE__'\n"
            await sandbox.files.write(script_path, replay_script, user="root")
            replay_t0 = time.time()
            try:
                replay_proc = await sandbox.commands.run(
                    f"bash {script_path}",
                    cwd="/testbed",
                    timeout=command_timeout,
                    user="root",
                )
                log_lines.append(
                    f"replay exit={replay_proc.exit_code} t={(time.time() - replay_t0):.1f}s "
                    f"tail={(replay_proc.stdout or '')[-200:]!r}"
                )
            except Exception as exc:
                log_lines.append(f"replay EXC {type(exc).__name__}: {exc}")

            # Capture the resulting diff for archival.
            try:
                diff_proc = await sandbox.commands.run(
                    f"git -C /testbed diff {job.base_commit}..HEAD",
                    timeout=60,
                    user="root",
                )
                patch_path.write_text(diff_proc.stdout or "")
            except Exception as exc:
                log_lines.append(f"git diff failed: {exc!r}")

            # Upload the (patched) tests/ tree and run the verifier.
            try:
                await sandbox.commands.run("rm -rf /tests", user="root")
            except Exception:
                pass
            await upload_dir(sandbox, job.local_tests_dir, "/tests")
            await sandbox.commands.run("chmod -R +x /tests", user="root")

            verifier = await sandbox.commands.run(
                "bash /tests/test.sh 2>&1",
                timeout=900,
                user="root",
            )
            stdout_path.write_text(verifier.stdout or "")

            # Read /logs/verifier/reward.txt
            try:
                reward_text = (await sandbox.files.read("/logs/verifier/reward.txt", user="root")).strip()
                reward_path.write_text(reward_text + "\n")
                result["reward"] = float(reward_text)
            except Exception as exc:
                log_lines.append(f"read reward failed: {exc!r}")
                # Fall back to test.sh exit code: 0 means tests passed
                result["reward"] = 1.0 if verifier.exit_code == 0 else 0.0
                reward_path.write_text(str(result["reward"]) + "\n")

        except Exception as exc:
            result["exception"] = f"{type(exc).__name__}: {exc}"
            log_lines.append("TOPLEVEL EXC: " + traceback.format_exc())
        finally:
            if sandbox is not None:
                try:
                    await sandbox.kill()
                except Exception:
                    pass
            elapsed = time.time() - started
            result["elapsed_sec"] = elapsed
            log_lines.append(f"replay done elapsed={elapsed:.1f}s reward={result['reward']}")
            log_path.write_text("\n".join(log_lines) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs-root", type=Path, required=True)
    p.add_argument("--tasks-root", type=Path, required=True)
    p.add_argument("--rollouts-jsonl", type=Path, default=None,
                   help="If set, only re-verify trials whose task is in this JSONL.")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--command-timeout", type=int, default=120)
    p.add_argument("--only-task", default=None,
                   help="Only re-verify this task_name (debug)")
    p.add_argument("--skip-done", type=Path, default=None,
                   help="Path to an existing reverify JSONL; trials whose trial_dir is "
                        "already there AND have reward != None will be skipped.")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    want_tasks: set[str] | None = None
    if args.rollouts_jsonl:
        want_tasks = set()
        with args.rollouts_jsonl.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iid = row.get("instance_id") or ""
                _, _, t = iid.partition("::")
                if t:
                    want_tasks.add(t)
    if args.only_task:
        want_tasks = {args.only_task}

    jobs = discover_trials(args.jobs_root, args.tasks_root, want_tasks)

    if args.skip_done and args.skip_done.is_file():
        done_dirs: set[str] = set()
        for line in args.skip_done.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("reward") is not None:
                done_dirs.add(r["trial_dir"])
        before = len(jobs)
        jobs = [j for j in jobs if str(j.trial_dir) not in done_dirs]
        print(f"Skipping {before - len(jobs)} already-done trials from {args.skip_done}")

    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"Discovered {len(jobs)} trials to re-verify "
          f"(concurrency={args.concurrency}, command_timeout={args.command_timeout}s)")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.touch(exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    progress = {"done": 0, "total": len(jobs)}

    async def run_and_record(job: TrialJob) -> None:
        result = await replay_one(job, sem, args.command_timeout)
        async with lock:
            with args.output_jsonl.open("a") as fh:
                fh.write(json.dumps(result) + "\n")
            progress["done"] += 1
            print(
                f"[{progress['done']}/{progress['total']}] "
                f"{job.task_name} reward={result['reward']} "
                f"elapsed={(result['elapsed_sec'] or 0):.0f}s "
                f"exc={result['exception']}",
                flush=True,
            )

    await asyncio.gather(*(run_and_record(j) for j in jobs))

    summary = {"total": len(jobs)}
    for line in args.output_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = "ok" if r.get("reward") == 1.0 else "fail" if r.get("reward") == 0.0 else "error"
        summary[key] = summary.get(key, 0) + 1
    print("Final tally:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
