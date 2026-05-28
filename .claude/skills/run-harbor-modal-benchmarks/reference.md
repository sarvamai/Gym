# Run Harbor Modal Benchmarks Reference

## What Was Learned

This run established a working path for NeMo Gym to call `harbor_agent`, Harbor to resolve SWE-bench Verified from its registry, and Modal to execute a task. It also uncovered several environment and integration gotchas that future runs should account for.

## Working Components

- NeMo Gym server launch works with direct `.venv/bin/ng_run`.
- `harbor_agent` can be used as the Gym agent server.
- Harbor built-in Modal environment is selected with:

```yaml
harbor_environment_type: "modal"
harbor_environment_import_path: null
```

- Harbor built-in OpenHands agent is selected with:

```yaml
harbor_agent_name: "openhands"
harbor_agent_import_path: null
```

- Fireworks can be exposed to Gym through `responses_api_models/openai_model`.
- The pinned Harbor registry contains SWE-bench Verified as `swebench-verified`.
- Harbor job artifacts include verifier results and per-trial logs even for reward `0.0`.

## Files Created During The Smoke Setup

Config:

```text
responses_api_agents/harbor_agent/configs/harbor_agent_openhands_modal_fireworks.yaml
```

Input routing JSONL:

```text
responses_api_agents/harbor_agent/data/swe_bench_verified_smoke/swe_verified_openhands_modal_input.jsonl
```

Output JSONL:

```text
responses_api_agents/harbor_agent/data/swe_bench_verified_smoke/swe_verified_openhands_modal_output.jsonl
```

Example routing row:

```json
{"instance_id":"swe_verified::astropy__astropy-12907","responses_create_params":{"input":[]},"agent_ref":{"name":"harbor_agent"}}
```

## Correct Harbor Config Values

For SWE-bench Verified + OpenHands + Modal + Fireworks:

```yaml
harbor_datasets:
  swe_verified:
    dataset_name: "swebench-verified"
    dataset_version: null
    workdir: "/testbed"

harbor_agent_name: "openhands"
harbor_agent_import_path: null

harbor_environment_type: "modal"
harbor_environment_import_path: null
harbor_environment_kwargs: {}

model_server:
  type: responses_api_models
  name: policy_model
```

Avoid stale Singularity config:

```yaml
harbor_environment_kwargs:
  singularity_image_cache_dir: ...
```

## Fireworks Provider Setup

Set:

```bash
export FIREWORKS_API_KEY="..."
export LLM_API_KEY="${FIREWORKS_API_KEY}"
```

`LLM_API_KEY` is required because Harbor OpenHands tries to infer provider-specific key names from the model string. Fireworks model names such as:

```text
accounts/fireworks/models/qwen3-coder-480b-a35b-instruct
```

are not recognized by Harbor's provider-key inference, so OpenHands raises:

```text
Unable to determine API key for model ... Please set LLM_API_KEY environment variable as fallback
```

## Localhost Proxy Root Cause

Symptom:

```text
POST http%3A//127.0.0.1%3A12060/run HTTP/1.1
```

Python clients returned 404 while `curl` returned the expected route behavior.

Root cause: macOS system proxy was enabled:

```text
HTTPProxy : 127.0.0.1
HTTPPort : 58080
HTTPSProxy : 127.0.0.1
HTTPSPort : 58080
```

Python `requests`/`urllib` discovered the proxy through system proxy APIs even when shell proxy variables were empty. Local requests were sent in proxy absolute-form and then reached FastAPI as a literal path.

Diagnosis:

```bash
scutil --proxy
```

```python
import requests
print(requests.utils.get_environ_proxies("http://127.0.0.1:12060/run"))
```

Fix:

```bash
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"
```

Permanent macOS fix: add `localhost`, `127.0.0.1`, and `::1` to the active network service's proxy bypass list.

## Command Pattern

Start servers from `Gym/`:

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

Run one `/run` smoke after reading the printed `harbor_agent` port:

```bash
"$PWD/.venv/bin/python" - <<'PY'
import json
from pathlib import Path
import requests

harbor_port = 17295  # replace
input_path = Path("responses_api_agents/harbor_agent/data/swe_bench_verified_smoke/swe_verified_openhands_modal_input.jsonl")
output_path = Path("responses_api_agents/harbor_agent/data/swe_bench_verified_smoke/swe_verified_openhands_modal_output.jsonl")
row = json.loads(input_path.read_text().strip().splitlines()[0])
response = requests.post(f"http://127.0.0.1:{harbor_port}/run", json=row, timeout=None)
response.raise_for_status()
data = response.json()
output_path.write_text(json.dumps(data) + "\n")
print(json.dumps({"reward": data.get("reward"), "instance_id": data.get("instance_id")}, indent=2))
PY
```

## Why Not Use `uv run ng_run`

Starting with:

```bash
uv run --directory Gym ng_run ...
```

triggered a Ray uv runtime hook error in the `harbor_agent` worker:

```text
pyproject.toml is not in the working_dir ... responses_api_agents/harbor_agent
```

Use:

```bash
Gym/.venv/bin/ng_run
```

instead.

## `ng_collect_rollouts` Gotcha

`ng_collect_rollouts` can successfully collect the `/run` result, but then it calls:

```text
POST /aggregate_metrics
```

on the agent server. `harbor_agent` currently does not expose that route, so the command can fail after writing the output row.

For smoke tests, call `/run` directly. For full `ng_collect_rollouts`, add or implement aggregate metrics support in `harbor_agent`, or handle the post-collection aggregate step separately.

## Artifact Interpretation

Main per-trial result:

```text
.../astropy__astropy-12907__*/result.json
```

Useful files:

```text
exception.txt
agent/openhands.txt
agent/setup/stdout.txt
agent/setup/stderr.txt
verifier/report.json
verifier/reward.txt
```

Observed reward `0.0` case:

```json
"patch_exists": true,
"patch_successfully_applied": true,
"resolved": false
```

The verifier report showed the `FAIL_TO_PASS` tests still failing, so reward stayed `0`.

## Debugging Sequence

When a run fails:

1. Check Python proxy bypass first.
2. Check Harbor dataset name with `harbor datasets list`.
3. Check server launch method; avoid `uv run` if Ray worker env fails.
4. Check `LLM_API_KEY` for unsupported provider model names.
5. Inspect Harbor job artifacts.
6. Preserve output rows and ask before excluding any datapoint.
