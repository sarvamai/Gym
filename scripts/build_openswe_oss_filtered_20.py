"""Sample 20 random openswe_oss filtered tasks (seed=42), patch their Dockerfiles
for e2b (apt-get install git + git clone <repo> + git reset --hard <commit>),
and emit a NeMo-Gym harbor_agent input JSONL.

Sampling rules (per `run-harbor-modal-benchmarks` skill + advisor notes):
  - Only include rows from data/openswe/routing/openswe_oss_filtered.jsonl
  - Only include tasks where task.toml metadata.base_image_published == true
    (otherwise the FROM line still points at the unpublished local
    openswe-python-X.Y image and the build will fail on e2b)
  - Only include tasks whose raw task dir exists under data/openswe/tasks/

Outputs go under
  responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
FILTERED_JSONL = REPO_ROOT / "data" / "openswe" / "routing" / "openswe_oss_filtered.jsonl"
RAW_TASKS_DIR = REPO_ROOT / "data" / "openswe" / "tasks" / "openswe_oss"
OUT_BASE = REPO_ROOT / "responses_api_agents" / "harbor_agent" / "data" / "openswe_harbor_filtered_20"
PATCHED_TASKS_DIR = OUT_BASE / "patched_tasks" / "openswe_oss"
INPUT_JSONL = OUT_BASE / "openswe_harbor_filtered_20_input.jsonl"
MANIFEST_JSON = OUT_BASE / "openswe_harbor_filtered_20_manifest.json"

SEED = 42
NUM_TASKS = 20


def patch_dockerfile(dockerfile: str, repo: str, commit: str) -> str:
    needle = "COPY repo /testbed"
    if needle not in dockerfile:
        raise RuntimeError(f"Dockerfile missing `{needle}` line; cannot patch.")
    replacement = (
        "# Rewritten for e2b/Modal: clone the source repository remotely instead of\n"
        "# expecting a local repo/ build context.\n"
        "RUN apt-get update -qq && apt-get install -qq -y --no-install-recommends "
        "git ca-certificates && rm -rf /var/lib/apt/lists/*\n"
        f"RUN git clone https://github.com/{repo}.git /testbed && \\\n"
        f"    cd /testbed && \\\n"
        f"    git reset --hard {commit}"
    )
    return dockerfile.replace(needle, replacement, 1)


def load_filtered_ids() -> list[str]:
    ids: list[str] = []
    with FILTERED_JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = row["instance_id"]
            alias, _, task_name = instance_id.partition("::")
            if alias != "openswe_oss":
                continue
            ids.append(task_name)
    return ids


def read_task_meta(task_dir: Path) -> dict:
    return tomllib.loads((task_dir / "task.toml").read_text())["metadata"]


def main() -> None:
    all_ids = load_filtered_ids()
    print(f"Loaded {len(all_ids)} filtered openswe_oss ids from {FILTERED_JSONL.relative_to(REPO_ROOT)}")

    rng = random.Random(SEED)
    rng.shuffle(all_ids)

    picked: list[dict] = []
    skipped: list[dict] = []
    for task_name in all_ids:
        if len(picked) >= NUM_TASKS:
            break
        task_dir = RAW_TASKS_DIR / task_name
        if not task_dir.is_dir():
            skipped.append({"task_name": task_name, "reason": "raw task dir missing"})
            continue
        try:
            meta = read_task_meta(task_dir)
        except Exception as exc:
            skipped.append({"task_name": task_name, "reason": f"task.toml unreadable: {exc!r}"})
            continue
        if not meta.get("base_image_published"):
            skipped.append({"task_name": task_name, "reason": "base_image_published=false"})
            continue
        repo = meta.get("repo")
        commit = meta.get("base_commit")
        if not repo or not commit:
            skipped.append({"task_name": task_name, "reason": "missing repo/base_commit"})
            continue
        picked.append({"task_name": task_name, "repo": repo, "base_commit": commit})

    if len(picked) < NUM_TASKS:
        raise RuntimeError(f"Only found {len(picked)} usable tasks; need {NUM_TASKS}.")

    print(f"Picked {len(picked)} tasks; skipped {len(skipped)} earlier candidates.")

    if PATCHED_TASKS_DIR.exists():
        shutil.rmtree(PATCHED_TASKS_DIR)
    PATCHED_TASKS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in picked:
        src = RAW_TASKS_DIR / entry["task_name"]
        dst = PATCHED_TASKS_DIR / entry["task_name"]
        shutil.copytree(src, dst)
        dockerfile_path = dst / "environment" / "Dockerfile"
        original = dockerfile_path.read_text()
        patched = patch_dockerfile(original, entry["repo"], entry["base_commit"])
        dockerfile_path.write_text(patched)

    with INPUT_JSONL.open("w") as fh:
        for entry in picked:
            row = {
                "agent_ref": {"name": "harbor_agent"},
                "instance_id": f"openswe_oss::{entry['task_name']}",
                "responses_create_params": {"input": []},
            }
            fh.write(json.dumps(row) + "\n")

    manifest = {
        "seed": SEED,
        "num_tasks": NUM_TASKS,
        "filter": "openswe_oss filtered + base_image_published=true",
        "picked": picked,
        "skipped_earlier_candidates": skipped[:50],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote {len(picked)} patched task dirs to {PATCHED_TASKS_DIR.relative_to(REPO_ROOT)}")
    print(f"Wrote input jsonl: {INPUT_JSONL.relative_to(REPO_ROOT)}")
    print(f"Wrote manifest:    {MANIFEST_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
