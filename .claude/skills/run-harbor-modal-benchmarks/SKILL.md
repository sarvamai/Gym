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
