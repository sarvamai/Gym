"""Driver: fire 20 tasks * N repeats POSTs at harbor_agent /run for openswe_oss + e2b.

Use as:
  HARBOR_PORT=12345 python scripts/run_openswe_oss_filtered_20_rollouts.py \
      --input-jsonl responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/openswe_harbor_filtered_20_input.jsonl \
      --output-jsonl responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/rollouts_<ts>.jsonl \
      --num-repeats 8 \
      --port 12345

Notes:
- harbor_agent has its own asyncio.Semaphore(concurrency) (= 10 from YAML), so no
  client-side semaphore needed; fire all rollouts as asyncio.gather tasks.
- Each completion is appended to output JSONL immediately to survive Ctrl-C.
- `repeat_idx` / `task_idx` are tagged in the local record, NOT into the request
  body (HarborRunRequest is strict Pydantic).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--num-repeats", type=int, default=8)
    p.add_argument("--port", type=int, required=True, help="harbor_agent port")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fire the first N (task, repeat) jobs. For smoke runs.",
    )
    return p.parse_args()


async def fire_one(
    session: aiohttp.ClientSession,
    url: str,
    row: dict,
    task_idx: int,
    repeat_idx: int,
    output_path: Path,
    lock: asyncio.Lock,
    progress: dict,
) -> None:
    started = time.time()
    error_str: str | None = None
    data: dict | None = None
    try:
        async with session.post(url, json=row, timeout=aiohttp.ClientTimeout(total=None)) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    record = {
        "task_idx": task_idx,
        "repeat_idx": repeat_idx,
        "instance_id": row["instance_id"],
        "elapsed_sec": elapsed,
    }
    if error_str is not None:
        record["driver_error"] = error_str
        record["reward"] = None
    else:
        record["reward"] = data.get("reward") if data else None
        record["response"] = data

    async with lock:
        with output_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        progress["done"] += 1
        if error_str is not None:
            progress["errors"] += 1
        status = "ERR" if error_str else f"r={record['reward']}"
        print(
            f"[{progress['done']}/{progress['total']}] task={task_idx} "
            f"repeat={repeat_idx} {status} {elapsed:.1f}s {row['instance_id']}",
            flush=True,
        )


async def main_async(args: argparse.Namespace) -> int:
    tasks_rows: list[dict] = []
    with args.input_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tasks_rows.append(json.loads(line))

    jobs: list[tuple[int, int, dict]] = []
    for task_idx, row in enumerate(tasks_rows):
        for repeat_idx in range(args.num_repeats):
            jobs.append((task_idx, repeat_idx, row))

    if args.limit is not None:
        jobs = jobs[: args.limit]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.touch(exist_ok=True)

    url = f"http://{args.host}:{args.port}/run"
    print(f"Posting {len(jobs)} jobs to {url}", flush=True)

    lock = asyncio.Lock()
    progress = {"done": 0, "total": len(jobs), "errors": 0}

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        coros = [
            fire_one(session, url, row, task_idx, repeat_idx, args.output_jsonl, lock, progress)
            for task_idx, repeat_idx, row in jobs
        ]
        await asyncio.gather(*coros)

    print(f"Done. errors={progress['errors']}/{progress['total']}", flush=True)
    return 0 if progress["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
