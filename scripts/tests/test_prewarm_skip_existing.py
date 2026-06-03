"""Unit tests for prewarm server-side skip logic (no real e2b calls).

Run with the harbor_agent venv (the prewarm module imports the e2b SDK):
  responses_api_agents/harbor_agent/.venv/bin/python -m pytest \
    scripts/tests/test_prewarm_skip_existing.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "prewarm_openswe_oss_filtered_20.py"


def load_mod():
    spec = importlib.util.spec_from_file_location("prewarm_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tasks(tmp_path, names):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for nm in names:
        env = tasks_dir / nm / "environment"
        env.mkdir(parents=True)
        (env / "Dockerfile").write_text(f"FROM scratch\n# {nm}\n")
    return tasks_dir


def run_main(mod, tasks_dir, manifest, *, force_rebuild=False, concurrency=4):
    args = argparse.Namespace(
        tasks_dir=tasks_dir,
        manifest=manifest,
        cpus=2,
        memory_mb=8192,
        concurrency=concurrency,
        force_rebuild=force_rebuild,
    )
    return asyncio.run(mod.main_async(args))


def test_skips_ready_aliases(tmp_path, monkeypatch):
    mod = load_mod()
    tasks_dir = make_tasks(tmp_path, ["task_a", "task_b", "task_c"])
    manifest = tmp_path / "m.json"

    ready_alias = mod.template_name_for(tasks_dir / "task_a").lower()
    monkeypatch.setattr(mod, "fetch_ready_aliases", lambda: {ready_alias}, raising=False)

    built = []

    async def fake_build_one(task_dir, cpus, memory_mb):
        built.append(task_dir.name)
        return {"task": task_dir.name, "alias": mod.template_name_for(task_dir), "elapsed_sec": 0.0}

    monkeypatch.setattr(mod, "build_one", fake_build_one)

    rc = run_main(mod, tasks_dir, manifest)

    assert rc == 0
    assert "task_a" not in built                 # reused from server, NOT rebuilt
    assert set(built) == {"task_b", "task_c"}
    recs = {r["task"]: r for r in json.loads(manifest.read_text())["records"]}
    assert recs["task_a"]["status"] == "ok"
    assert recs["task_a"]["reused"] == "already_ready"
    assert len(recs) == 3


def test_force_rebuild_ignores_server(tmp_path, monkeypatch):
    mod = load_mod()
    tasks_dir = make_tasks(tmp_path, ["task_a", "task_b"])
    manifest = tmp_path / "m.json"

    called = {"n": 0}

    def fake_fetch():
        called["n"] += 1
        return {mod.template_name_for(tasks_dir / "task_a").lower()}

    monkeypatch.setattr(mod, "fetch_ready_aliases", fake_fetch, raising=False)

    built = []

    async def fake_build_one(task_dir, cpus, memory_mb):
        built.append(task_dir.name)
        return {"task": task_dir.name, "alias": mod.template_name_for(task_dir), "elapsed_sec": 0.0}

    monkeypatch.setattr(mod, "build_one", fake_build_one)

    rc = run_main(mod, tasks_dir, manifest, force_rebuild=True)

    assert rc == 0
    assert called["n"] == 0                       # --force-rebuild must NOT query the server
    assert set(built) == {"task_a", "task_b"}
