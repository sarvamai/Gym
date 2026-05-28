---
name: run-harbor-modal-benchmarks
description: Use when running or debugging NeMo Gym Harbor benchmarks with Modal, OpenHands, Fireworks or OpenAI-compatible providers, SWE-bench, FastAPI localhost routes, or harbor_agent rollout collection.
---

# Run Harbor Modal Benchmarks

## Purpose

Use this when running NeMo Gym rollouts through `harbor_agent`, especially with Harbor built-in agents such as OpenHands and Modal environments. This captures the operational gotchas discovered while running a SWE-bench Verified smoke task with Fireworks.

## Critical Rule

Do not skip datapoints. If a rollout fails, preserve an output row or artifact reference and report the exact blocker. If a datapoint cannot be run because a dataset, image, provider, or credential is missing, ask the user what to do.

## Preflight Checklist

Before starting servers:

- Confirm Modal auth exists: `~/.modal.toml`.
- Confirm provider key is present, for example `FIREWORKS_API_KEY`.
- Set localhost proxy bypass:

```bash
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
```

- For Harbor OpenHands with unknown provider model names, also set:

```bash
export LLM_API_KEY="${FIREWORKS_API_KEY}"
```

- Use `Gym/.venv/bin/ng_run`, not `uv run ng_run`, when Ray + uv runtime hooks cause working-directory errors.

## Known Localhost Proxy Failure

If Python clients produce FastAPI access logs like:

```text
POST http%3A//127.0.0.1%3A12060/run HTTP/1.1
```

then Python is using the macOS system proxy for localhost. Verify:

```bash
scutil --proxy
```

and:

```python
import requests
print(requests.utils.get_environ_proxies("http://127.0.0.1:12060/run"))
```

Expected fix:

```bash
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
```

Then Python should return origin-form routes correctly, for example `GET /run` should return `405`, not `404`.

## Harbor Dataset Names

Do not guess Harbor registry names. Query them:

```bash
responses_api_agents/harbor_agent/.venv/bin/harbor datasets list
```

For SWE-bench Verified in the pinned Harbor package, the dataset is:

```yaml
dataset_name: "swebench-verified"
```

not `swe-bench/swe-bench-verified`.

## Start Server Pattern

Use a config that includes both `harbor_agent` and `policy_model`:

```bash
set -a
source .env
set +a
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
export LLM_API_KEY="${FIREWORKS_API_KEY}"

config_paths="responses_api_agents/harbor_agent/configs/harbor_agent_openhands_modal_fireworks.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

"$PWD/.venv/bin/ng_run" "+config_paths=[${config_paths}]" \
  +policy_base_url="https://api.fireworks.ai/inference/v1" \
  +policy_api_key="${FIREWORKS_API_KEY}" \
  +policy_model_name="accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"
```

Wait for:

```text
All 2 / 2 servers ready
```

Record the printed `harbor_agent` port.

## One-Row Smoke Run

For a direct smoke test, call `/run` directly and save one JSONL row. This avoids `ng_collect_rollouts` failing later on `/aggregate_metrics` when an agent server does not implement it.

```bash
"$PWD/.venv/bin/python" - <<'PY'
import json
from pathlib import Path
import requests

input_path = Path("responses_api_agents/harbor_agent/data/swe_bench_verified_smoke/swe_verified_openhands_modal_input.jsonl")
output_path = Path("responses_api_agents/harbor_agent/data/swe_bench_verified_smoke/swe_verified_openhands_modal_output.jsonl")
harbor_port = 17295  # replace with printed harbor_agent port

row = json.loads(input_path.read_text().strip().splitlines()[0])
response = requests.post(f"http://127.0.0.1:{harbor_port}/run", json=row, timeout=None)
response.raise_for_status()
data = response.json()
output_path.write_text(json.dumps(data) + "\n")
print(json.dumps({"reward": data.get("reward"), "instance_id": data.get("instance_id")}, indent=2))
PY
```

## Artifact Checks

Harbor writes trial artifacts under:

```text
responses_api_agents/harbor_agent/jobs/<date>/<dataset_alias>/<model>/<job>/<trial>/
```

Inspect:

- `result.json`
- `exception.txt`
- `agent/openhands.txt`
- `verifier/report.json`
- `verifier/reward.txt`

## Common Outcomes

- `ValueError: Dataset ... not found`: wrong Harbor dataset registry name. Run `harbor datasets list`.
- `Unknown model ... Please set LLM_API_KEY`: export `LLM_API_KEY` for provider fallback.
- `/aggregate_metrics` 404: direct `/run` succeeded, but `ng_collect_rollouts` tried an endpoint `harbor_agent` does not expose.
- `No module named openhands.core.main`: Harbor/OpenHands install or command mismatch; inspect `agent/setup/*` and `agent/openhands.txt`.

For detailed notes, read `reference.md`.

## E2B Rollouts — Mandatory Steps and Pitfalls

When running rollouts against the **E2B** environment (not Modal), there are additional gotchas that have produced cascading failures in past sessions. Apply all of these before starting `ng_collect_rollouts`.

### 1. Pre-warm E2B templates sequentially (mandatory)

**Always** run the prewarm script before collecting rollouts:

```bash
python scripts/prewarm_openswe_oss_filtered_20.py
# (copy/generalize the script for larger filtered sets)
```

Why: E2B's `CheckAndCancelConcurrentBuilds` (in `packages/api/internal/handlers/deprecated_template_start_build.go`) actively cancels concurrent builds of the same template alias. With N concurrent rollouts each triggering a build, E2B cancels N−1. Even across distinct aliases, total concurrent builds are capped near 9–10. The prewarm script builds all required templates **sequentially**, so the rollout phase only ever reads pre-existing templates via `_does_template_exist()`.

Skipping the prewarm produces a confusing failure cascade: "access denied" / build-cancelled errors that look like infra bugs but are concurrency artifacts.

A useful side effect: prewarm surfaces per-task base-image and Dockerfile errors **sequentially with clear attribution**, before any model/sandbox cost is spent. Treat the prewarm exit log as the canonical "which tasks are actually runnable" filter.

### 2. Both Fireworks env-var names must be exported

mini-swe-agent uses LiteLLM internally. LiteLLM derives env-var names from the model-name prefix: `fireworks_ai/...` → `FIREWORKS_AI_API_KEY` (note the `_AI_`). NeMo Gym's own OpenAI client reads bare `FIREWORKS_API_KEY`. Both must be set:

```bash
export FIREWORKS_API_KEY="fw_..."             # for nemo_gym's own clients
export FIREWORKS_AI_API_KEY="$FIREWORKS_API_KEY"  # for LiteLLM inside the sandbox
```

Symptom when only `FIREWORKS_API_KEY` is exported: rollouts crash inside the sandbox with `Unable to determine API key for model fireworks_ai/accounts/fireworks/models/<name>`. Harbor-orchestrator-level model calls still work, so the failure only surfaces in agent rollouts, not in `ng_status` or smoke pings.

### 3. Harbor editable-install path is machine-local

`responses_api_agents/harbor_agent/requirements.txt` references the harbor library via an absolute `-e <path>` install. The path was originally `/Users/rvk7895/Projects/sarvam_personal_projects/e2b/harbor` (laptop layout). On any new machine (pod, teammate), fix this line to point at the local harbor clone before running `uv pip install -r requirements.txt`. Prefer a relative or `~`-prefixed path so this isn't a per-machine edit forever.

### 4. Base images must exist where E2B can pull them

Patched task Dockerfiles use `FROM openswe-python-<X.Y>` (bare image, no namespace). The base images are published on Docker Hub at `rvk7895/openswe-python-<X.Y>` (versions 3.7, 3.9, 3.10, 3.11, 3.12, 3.13 — not 3.6, not 3.8). Two requirements:

- The `patch_dockerfile()` function in `scripts/build_openswe_oss_filtered_*.py` must rewrite the `FROM` line to `FROM docker.io/rvk7895/openswe-python-<X.Y>` (in addition to its existing `COPY repo /testbed` → `git clone` rewrite).
- The Docker Hub repos must be **public** (or E2B sandbox builders must be configured with pull credentials).
- Tasks needing Python 3.6 or 3.8 will fail until those base images are also pushed; pre-filter them out of the input JSONL or build/push them first.

### Correct E2B rollout sequence

```bash
cd ~/Gym
# 1. Build input JSONL + patched task dirs
python scripts/build_openswe_oss_filtered_20.py

# 2. Pre-warm E2B templates sequentially (REQUIRED — do not skip)
python scripts/prewarm_openswe_oss_filtered_20.py

# 3. Export both env-var names
export FIREWORKS_API_KEY="fw_..."
export FIREWORKS_AI_API_KEY="$FIREWORKS_API_KEY"
export E2B_API_KEY="e2b_..."

# 4. Start harbor (in tmux)
ng_run "+config_paths=[responses_api_agents/harbor_agent/configs/harbor_agent_openswe_e2b.yaml]"

# 5. Collect rollouts (in another tmux pane)
ng_collect_rollouts +agent_name=harbor_agent \
    +input_jsonl_fpath=responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/openswe_harbor_filtered_20_input.jsonl \
    +output_jsonl_fpath=results/openswe20_$(date +%s).jsonl \
    +num_repeats=1 \
    +max_workers=20 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```
