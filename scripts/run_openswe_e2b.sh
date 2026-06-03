#!/usr/bin/env bash
# Configurable Harbor rollout pipeline for NeMo Gym harbor_agent.
#
# Point it at any Harbor-format dataset directory (a dir of per-task subdirs,
# each containing task.toml + environment/Dockerfile) and it runs rollouts on
# it. Everything is configured by editing the CONFIG block below — no CLI flags.
#
#   SANDBOX=modal   Harbor built-in Modal env; builds images on demand.  (default)
#   SANDBOX=e2b     self-hosted E2B (e2b.sarvam.ai); prewarms templates first.
#
# It selects NUM_TASKS tasks from DATASET_PATH, (for e2b) prewarms them, brings
# up ng_run with the matching sandbox config + the configured PROVIDER's
# policy_model pointed at the dataset, and collects NUM_REPEATS rollouts per task.
#
# For OpenSWE: build the patched task tree first with
#   .venv/bin/python scripts/build_openswe_oss_filtered_20.py
# then set DATASET_PATH to
#   responses_api_agents/harbor_agent/data/openswe_harbor_filtered_20/patched_tasks/openswe_oss
#
# Credentials are read from the process environment (export them, or put them in
# Gym/.env which this script sources). Nothing secret is hardcoded here.
#
# Run from repo root (or anywhere — the script cd's to the repo root):
#   bash scripts/run_openswe_e2b.sh

set -uo pipefail

# ============================================================================
# CONFIG — edit these.
# ============================================================================
SANDBOX="e2b"                                          # modal | e2b

# Dataset: a Harbor-format directory of per-task subdirs (each has task.toml).
# openswe-harbor-rlsafe-pool = sarvam/OpenSWE-Harbor @ rl-safe-high-confidence,
# extracted + filtered to base_image_published=true + COPY-repo rewritten to
# git-clone (see scripts/prep_openswe_harbor_pool.py).
DATASET_PATH="data/openswe-harbor-rlsafe-pool"
DATASET_ALIAS="openswe"                                  # alias used in instance_ids + harbor_datasets key
DATASET_WORKDIR="/testbed"                               # container workdir for this dataset

NUM_TASKS=0                                             # tasks to sample (0 = all)
NUM_REPEATS=8                                            # rollouts per task

# Some datasets ship Dockerfiles with `FROM local/...` base images that live
# only in the local Docker daemon — remote sandboxes (modal/e2b) can't pull
# them. When REMAP_LOCAL_IMAGES=true, the staged Dockerfiles' `FROM local/...`
# lines are rewritten to ${IMAGE_REGISTRY}/<flattened-path>:<tag>. PUSH_LOCAL_IMAGES
# additionally `docker tag`+`push`es those images first (needs push auth in THIS
# shell); set it false if you've already pushed them from an authenticated shell.
REMAP_LOCAL_IMAGES=true
PUSH_LOCAL_IMAGES=false
IMAGE_REGISTRY="docker.io/rvk7895"

# Policy LLM provider. mini-swe-agent calls it DIRECTLY via litellm (not through
# the policy_model server), so the provider determines model id + key + base url.
#   fireworks : accounts/fireworks/models/gpt-oss-120b  (was timing out at 600s)
#   ark       : BytePlus Ark gpt-oss-120b-250805        (OpenAI-compatible, fast)
PROVIDER="ark"                                           # fireworks | ark
FIREWORKS_MODEL="accounts/fireworks/models/gpt-oss-120b"
ARK_MODEL="deepseek-v4-flash-260425"
ARK_BASE_URL="https://ark.ap-southeast.bytepluses.com/api/v3"

MAX_OUTPUT_TOKENS=32556
TEMPERATURE=1.0
SEED=42                                                  # task-sampling seed

MAX_WORKERS_CEILING=256                                   # rollout concurrency (drives BOTH the
                                                         # client max_workers AND the harbor_agent
                                                         # server's concurrency, overridden at ng_run)

# Whole-trial retries on failure. Harbor reruns a failed trial (fresh env + agent
# setup) up to this many extra times with exponential backoff — clears transient
# setup/build/sandbox flakes (e.g. the uv-installer setup_failed bucket). Timeouts
# and reward-file errors are never retried (harbor RetryConfig.exclude_exceptions).
# 0 = no retry (harbor default).
RETRY_ATTEMPTS=2

OUTPUT_DIR="results"

# Env vars to forward into the verifier process inside the sandbox. Each NAME
# listed here is injected into every staged task's [verifier].env as
# NAME = "${NAME}", so harbor's resolve_env_vars pulls the value from THIS
# shell's environment at verify time (templated — the secret is never written
# into task.toml). OpenSWE-Harbor verifiers grade via pytest and don't need a
# key, but forwarding ANTHROPIC_API_KEY is a harmless hedge for the few tasks
# whose own suites call an LLM (it must be in the shell / .env). Empty = ().
VERIFIER_ENV=(ANTHROPIC_API_KEY)

# E2B prewarm knobs (only used when SANDBOX=e2b).
PREWARM_CPUS=2
PREWARM_MEMORY_MB=8192
PREWARM_CONCURRENCY=20

# ============================================================================
# Repo root + env.
# ============================================================================
case "$SANDBOX" in
  modal|e2b) ;;
  *) echo "ERROR: SANDBOX must be 'modal' or 'e2b' (got '$SANDBOX')" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load credentials from .env if present (FIREWORKS_API_KEY, E2B_*, ...).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

# Localhost proxy bypass — without this macOS routes 127.0.0.1 through the
# system proxy and FastAPI sees mangled absolute-form paths.
export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="127.0.0.1,localhost,::1"

# ============================================================================
# Provider-specific policy LLM wiring + sandbox config + preflight.
# ============================================================================
MODEL_CONFIG="responses_api_models/openai_model/configs/openai_model.yaml"
case "$PROVIDER" in
  fireworks)
    POLICY_BASE_URL="https://api.fireworks.ai/inference/v1"
    POLICY_API_KEY="${FIREWORKS_API_KEY:-}"
    POLICY_MODEL_NAME="$FIREWORKS_MODEL"
    # In-sandbox mini-swe-agent / LiteLLM key aliases + Harbor key inference.
    export FIREWORKS_AI_API_KEY="${FIREWORKS_AI_API_KEY:-${FIREWORKS_API_KEY:-}}"
    export MSWEA_API_KEY="${FIREWORKS_API_KEY:-}"
    export LLM_API_KEY="${FIREWORKS_API_KEY:-}"
    if [[ "$SANDBOX" == "modal" ]]; then
      HARBOR_CONFIG="responses_api_agents/harbor_agent/configs/harbor_agent_openswe_modal.yaml"
    else
      HARBOR_CONFIG="responses_api_agents/harbor_agent/configs/harbor_agent_openswe_e2b.yaml"
    fi
    ;;
  ark)
    POLICY_BASE_URL="$ARK_BASE_URL"
    POLICY_API_KEY="${ARK_API_KEY:-}"
    POLICY_MODEL_NAME="$ARK_MODEL"
    # mini-swe-agent (OpenAI-compat wrapper) reads these for the litellm openai/ provider.
    export OPENAI_API_KEY="${ARK_API_KEY:-}"
    export OPENAI_API_BASE="$ARK_BASE_URL"
    export MSWEA_API_KEY="${ARK_API_KEY:-}"
    export LLM_API_KEY="${ARK_API_KEY:-}"
    if [[ "$SANDBOX" == "modal" ]]; then
      HARBOR_CONFIG="responses_api_agents/harbor_agent/configs/harbor_agent_openswe_modal_ark.yaml"
    else
      HARBOR_CONFIG="responses_api_agents/harbor_agent/configs/harbor_agent_openswe_e2b_ark.yaml"
    fi
    ;;
  *) echo "ERROR: PROVIDER must be 'fireworks' or 'ark' (got '$PROVIDER')" >&2; exit 2 ;;
esac

GYM_PY=".venv/bin/python"
GYM_NG_RUN=".venv/bin/ng_run"
GYM_NG_COLLECT=".venv/bin/ng_collect_rollouts"
HARBOR_PY="responses_api_agents/harbor_agent/.venv/bin/python"

# Hydra paths into the harbor_agent server config.
HD_AGENT="harbor_agent.responses_api_agents.harbor_agent"
HD_BASE="${HD_AGENT}.harbor_datasets"

[[ -n "$POLICY_API_KEY" ]]        || { echo "ERROR: API key for PROVIDER=$PROVIDER is empty (set it in .env)" >&2; exit 1; }
[[ -x "$GYM_NG_RUN" ]]            || { echo "ERROR: $GYM_NG_RUN missing; run uv sync first" >&2; exit 1; }
[[ -f "$HARBOR_CONFIG" ]]         || { echo "ERROR: harbor config not found: $HARBOR_CONFIG" >&2; exit 1; }
[[ -d "$DATASET_PATH" ]]          || { echo "ERROR: DATASET_PATH not a directory: $DATASET_PATH" >&2; exit 1; }

if [[ "$SANDBOX" == "modal" ]]; then
  [[ -f "$HOME/.modal.toml" ]] || { echo "ERROR: ~/.modal.toml missing; run 'modal token new'" >&2; exit 1; }
else
  [[ -n "${E2B_API_KEY:-}" ]]  || { echo "ERROR: E2B_API_KEY empty" >&2; exit 1; }
  [[ -x "$HARBOR_PY" ]]        || { echo "ERROR: $HARBOR_PY missing; harbor_agent venv needs setup" >&2; exit 1; }
fi

# Every var forwarded to the verifier must exist in THIS shell — harbor's
# resolve_env_vars raises at verify time otherwise (after the agent has run).
# Fail fast here instead.
for _v in "${VERIFIER_ENV[@]:-}"; do
  [[ -z "$_v" ]] && continue
  [[ -n "${!_v:-}" ]] || { echo "ERROR: VERIFIER_ENV needs '$_v' but it is unset (add it to .env)" >&2; exit 1; }
done

mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════════════"
echo "Harbor rollouts:  sandbox=$SANDBOX  tasks=$NUM_TASKS  repeats=$NUM_REPEATS"
echo "  dataset=$DATASET_PATH  (alias=$DATASET_ALIAS  workdir=$DATASET_WORKDIR)"
echo "  model=$POLICY_MODEL_NAME"
echo "  config=$HARBOR_CONFIG"
echo "════════════════════════════════════════════════════════════════════════"

# ============================================================================
# Select tasks from DATASET_PATH (seeded shuffle, first NUM_TASKS) and stage
# them into a per-run dir. The staging dir is what gets used as the harbor
# local_dataset_path and (for e2b) the prewarm tasks-dir, so the selected set
# is the single source of truth for both steps.
# ============================================================================
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${DATASET_ALIAS}_${SANDBOX}_${NUM_TASKS}x${NUM_REPEATS}_${RUN_STAMP}"
STAGE_DIR="${OUTPUT_DIR}/${RUN_LABEL}_tasks"
RUN_INPUT_JSONL="${OUTPUT_DIR}/${RUN_LABEL}_input.jsonl"

echo "[select] sampling up to $NUM_TASKS tasks from $DATASET_PATH (seed=$SEED)"
DATASET_PATH="$DATASET_PATH" DATASET_ALIAS="$DATASET_ALIAS" NUM_TASKS="$NUM_TASKS" \
SEED="$SEED" STAGE_DIR="$STAGE_DIR" RUN_INPUT_JSONL="$RUN_INPUT_JSONL" "$GYM_PY" - <<'PY'
import json, os, random, shutil
from pathlib import Path

ds = Path(os.environ["DATASET_PATH"])
alias = os.environ["DATASET_ALIAS"]
num = int(os.environ["NUM_TASKS"])
seed = int(os.environ["SEED"])
stage = Path(os.environ["STAGE_DIR"])
out = Path(os.environ["RUN_INPUT_JSONL"])

tasks = sorted(p.name for p in ds.iterdir() if (p / "task.toml").is_file())
if not tasks:
    raise SystemExit(f"No Harbor tasks (subdirs with task.toml) under {ds}")
rng = random.Random(seed)
rng.shuffle(tasks)
if num > 0:
    tasks = tasks[:num]

if stage.exists():
    shutil.rmtree(stage)
stage.mkdir(parents=True)
with out.open("w") as fh:
    for t in tasks:
        shutil.copytree(ds / t, stage / t)
        fh.write(json.dumps({
            "agent_ref": {"name": "harbor_agent"},
            "instance_id": f"{alias}::{t}",
            "responses_create_params": {"input": []},
        }) + "\n")
print(f"[select] staged {len(tasks)} tasks -> {stage}")
PY
[[ -s "$RUN_INPUT_JSONL" ]] || { echo "ERROR: no tasks selected." >&2; exit 1; }

# ============================================================================
# Inject [verifier].env into each staged task.toml so harbor forwards the listed
# vars to the verifier process in the sandbox. Values are templated as
# "${NAME}" — harbor's resolve_env_vars substitutes them from the host env at
# verify time, so no secret is written into task.toml. Merges into an existing
# [verifier] table (or appends one); idempotent.
# ============================================================================
if [[ "${#VERIFIER_ENV[@]}" -gt 0 ]]; then
  echo "[verifier-env] forwarding to verifier: ${VERIFIER_ENV[*]}"
  STAGE_DIR="$STAGE_DIR" VERIFIER_ENV="${VERIFIER_ENV[*]}" "$GYM_PY" - <<'PY'
import os, re, tomllib
from pathlib import Path

stage = Path(os.environ["STAGE_DIR"])
names = [n for n in os.environ["VERIFIER_ENV"].split() if n]
# TOML inline table: env = { A = "${A}", B = "${B}" }
inline = "{ " + ", ".join(f'{n} = "${{{n}}}"' for n in names) + " }"
env_line = f"env = {inline}"

patched = 0
for toml_path in sorted(stage.glob("*/task.toml")):
    text = toml_path.read_text()
    cfg = tomllib.loads(text)
    existing = (cfg.get("verifier") or {}).get("env") or {}
    if all(n in existing for n in names):
        continue  # already has all forwarded names

    if re.search(r"^\[verifier\]\s*$", text, flags=re.MULTILINE):
        # Replace an existing env= line in the table, else insert after header.
        if re.search(r"^\s*env\s*=", text, flags=re.MULTILINE):
            text = re.sub(r"^\s*env\s*=.*$", env_line, text, count=1, flags=re.MULTILINE)
        else:
            text = re.sub(r"(^\[verifier\]\s*$)", r"\1\n" + env_line, text, count=1, flags=re.MULTILINE)
    else:
        text = text.rstrip() + f"\n\n[verifier]\n{env_line}\n"
    toml_path.write_text(text)
    patched += 1
print(f"[verifier-env] patched {patched} task.toml(s)")
PY
  [[ $? -eq 0 ]] || { echo "ERROR: verifier-env injection failed" >&2; exit 1; }
fi

# ============================================================================
# Normalize staged Dockerfiles for Modal's stricter parser: rewrite a bare
# `ARG NAME=` (empty default — legal in Docker/BuildKit but REJECTED by Modal's
# parser, "expected string or arg_value") to `ARG NAME`. Repo2RLEnv-generated
# Dockerfiles ship `ARG GITHUB_TOKEN=`, which fails every Modal build at parse
# time. Harmless for e2b, so run unconditionally; no-op when the pattern absent.
# ============================================================================
echo "[normalize] fixing bare 'ARG NAME=' -> 'ARG NAME' in staged Dockerfiles (Modal parser compat)"
STAGE_DIR="$STAGE_DIR" "$GYM_PY" - <<'PY'
import os, re
from pathlib import Path

stage = Path(os.environ["STAGE_DIR"])
pat = re.compile(r"^(\s*ARG\s+[A-Za-z_][A-Za-z0-9_]*)=[ \t]*$", flags=re.MULTILINE)
fixed = 0
for df in sorted(stage.glob("*/environment/Dockerfile")):
    text = df.read_text()
    new = pat.sub(r"\1", text)
    if new != text:
        df.write_text(new)
        fixed += 1
print(f"[normalize] fixed {fixed} Dockerfile(s)")
PY
[[ $? -eq 0 ]] || { echo "ERROR: Dockerfile normalize step failed" >&2; exit 1; }

# ============================================================================
# Remap local-only base images so remote sandboxes can pull them: rewrite the
# staged Dockerfiles' `FROM local/...` to ${IMAGE_REGISTRY}/<flattened>:<tag>,
# optionally tagging+pushing first. Idempotent.
# ============================================================================
if [[ "${REMAP_LOCAL_IMAGES}" == "true" ]]; then
  echo "[remap] scanning staged Dockerfiles for 'FROM local/...' base images (push=${PUSH_LOCAL_IMAGES})"
  STAGE_DIR="$STAGE_DIR" IMAGE_REGISTRY="$IMAGE_REGISTRY" PUSH_LOCAL_IMAGES="$PUSH_LOCAL_IMAGES" "$GYM_PY" - <<'PY'
import os, re, subprocess, sys
from pathlib import Path

stage = Path(os.environ["STAGE_DIR"])
registry = os.environ["IMAGE_REGISTRY"].rstrip("/")
do_push = os.environ.get("PUSH_LOCAL_IMAGES", "false") == "true"

def published_ref(local_ref: str) -> str:
    # local/r2e-bootstrap/fastapi__fastapi:TAG -> <registry>/r2e-bootstrap-fastapi-fastapi:TAG
    # Docker Hub rejects repeated special chars (e.g. "__"), so flatten "/" to "-"
    # and collapse any run of separators to a single "-".
    body = local_ref[len("local/"):]
    repo, _, tag = body.partition(":")
    flat = repo.replace("/", "-")
    flat = re.sub(r"[-_.]{2,}", "-", flat).strip("-_.").lower()
    return f"{registry}/{flat}" + (f":{tag}" if tag else "")

dockerfiles = sorted(stage.glob("*/environment/Dockerfile"))
local_refs = set()
for df in dockerfiles:
    for line in df.read_text().splitlines():
        m = re.match(r"\s*FROM\s+(local/\S+)", line)
        if m:
            local_refs.add(m.group(1))

if not local_refs:
    print("[remap] no local/ base images; nothing to do")
    sys.exit(0)

mapping = {ref: published_ref(ref) for ref in sorted(local_refs)}
for ref, pub in mapping.items():
    print(f"[remap] {ref} -> {pub}")
    if do_push:
        subprocess.run(["docker", "tag", ref, pub], check=True)
        subprocess.run(["docker", "push", pub], check=True)

rewritten = 0
for df in dockerfiles:
    text = df.read_text()
    new = text
    for ref, pub in mapping.items():
        new = re.sub(rf"(^\s*FROM\s+){re.escape(ref)}(\s|$)", rf"\1{pub}\2", new, flags=re.MULTILINE)
    if new != text:
        df.write_text(new)
        rewritten += 1
print(f"[remap] rewrote FROM in {rewritten} staged Dockerfile(s)")
PY
  [[ $? -eq 0 ]] || { echo "ERROR: remap step failed" >&2; exit 1; }
fi

# ============================================================================
# E2B prewarm: build each selected template once, sequentially, so concurrent
# rollouts don't trigger e2b's CheckAndCancelConcurrentBuilds. Then subset the
# input to successfully-warmed tasks. (Modal builds on demand — no prewarm.)
# ============================================================================
if [[ "$SANDBOX" == "e2b" ]]; then
  PREWARM_MANIFEST="${OUTPUT_DIR}/${RUN_LABEL}_prewarm_manifest.json"
  echo "[prewarm] parallel pass concurrency=$PREWARM_CONCURRENCY ..."
  "$HARBOR_PY" scripts/prewarm_openswe_oss_filtered_20.py \
    --tasks-dir "$STAGE_DIR" --manifest "$PREWARM_MANIFEST" \
    --cpus "$PREWARM_CPUS" --memory-mb "$PREWARM_MEMORY_MB" \
    --concurrency "$PREWARM_CONCURRENCY" || true

  # The prewarm script writes the manifest even when every build fails (records
  # carry status="error"). A MISSING manifest therefore means prewarm itself
  # crashed (bad args, import error, ...). Abort loudly — silently continuing
  # here is what once launched a 50-task rollout with zero warmed templates.
  [[ -s "$PREWARM_MANIFEST" ]] || {
    echo "ERROR: prewarm produced no manifest ($PREWARM_MANIFEST)." >&2
    echo "       Refusing to roll out on un-prewarmed templates. Check the prewarm output above." >&2
    exit 1; }

  retry_needed=$("$HARBOR_PY" - <<PY
import json
m = json.load(open("$PREWARM_MANIFEST"))
print(len([r for r in m["records"] if r.get("status") != "ok"]))
PY
)
  if [[ "$retry_needed" -gt 0 ]]; then
    echo "[prewarm] sequential retry for $retry_needed failed templates ..."
    "$HARBOR_PY" scripts/prewarm_openswe_oss_filtered_20.py \
      --tasks-dir "$STAGE_DIR" --manifest "$PREWARM_MANIFEST" \
      --cpus "$PREWARM_CPUS" --memory-mb "$PREWARM_MEMORY_MB" \
      --concurrency 1 || true
  fi

  RUN_INPUT_JSONL="$RUN_INPUT_JSONL" PREWARM_MANIFEST="$PREWARM_MANIFEST" "$HARBOR_PY" - <<'PY'
import json, os
inp = os.environ["RUN_INPUT_JSONL"]
recs = json.load(open(os.environ["PREWARM_MANIFEST"]))["records"]
ok = {r["task"] for r in recs if r.get("status") == "ok"}
print(f"[prewarm] {len(ok)} / {len(recs)} templates ok")
kept = []
n_in = 0
with open(inp) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        n_in += 1
        if json.loads(line)["instance_id"].partition("::")[2] in ok:
            kept.append(line)
with open(inp, "w") as fh:
    fh.write("\n".join(kept) + ("\n" if kept else ""))
print(f"[rollout] using {len(kept)}/{n_in} warmed tasks")
PY
fi

ROLLOUT_N=$(grep -c . "$RUN_INPUT_JSONL" 2>/dev/null || echo 0)
[[ "$ROLLOUT_N" -gt 0 ]] || { echo "ERROR: no tasks to roll out." >&2; exit 1; }

# ============================================================================
# Bring up ng_run, pointing harbor_datasets[<alias>] at the staged tasks.
# ============================================================================
HARBOR_LOG="${OUTPUT_DIR}/harbor_${RUN_LABEL}.log"
echo "[serve] starting ng_run, log: $HARBOR_LOG"
"$GYM_NG_RUN" "+config_paths=[${HARBOR_CONFIG},${MODEL_CONFIG}]" \
  +policy_base_url="${POLICY_BASE_URL}" \
  +policy_api_key="${POLICY_API_KEY}" \
  +policy_model_name="${POLICY_MODEL_NAME}" \
  "++${HD_BASE}.${DATASET_ALIAS}.local_dataset_path=${STAGE_DIR}" \
  "++${HD_BASE}.${DATASET_ALIAS}.workdir=${DATASET_WORKDIR}" \
  "++${HD_AGENT}.concurrency=${MAX_WORKERS_CEILING}" \
  "++${HD_AGENT}.harbor_orchestrator_max_retries=${RETRY_ATTEMPTS}" > "$HARBOR_LOG" 2>&1 &
NG_RUN_PID=$!
cleanup() { echo "[serve] stopping ng_run (pid $NG_RUN_PID)"; kill $NG_RUN_PID 2>/dev/null || true; }
trap cleanup EXIT

echo "[serve] waiting for 'servers ready' ..."
for _ in $(seq 1 180); do
  if grep -q "servers ready! Polling" "$HARBOR_LOG" 2>/dev/null; then
    echo "[serve] servers ready."; break
  fi
  if ! kill -0 $NG_RUN_PID 2>/dev/null; then
    echo "ERROR: ng_run exited early. Tail:" >&2; tail -50 "$HARBOR_LOG" >&2; exit 1
  fi
  sleep 2
done
grep -q "servers ready! Polling" "$HARBOR_LOG" 2>/dev/null || {
  echo "ERROR: timed out waiting for servers." >&2; tail -50 "$HARBOR_LOG" >&2; exit 1; }

# ============================================================================
# Rollout.
# ============================================================================
DESIRED_WORKERS=$(( ROLLOUT_N * NUM_REPEATS ))
MAX_WORKERS=$(( DESIRED_WORKERS < MAX_WORKERS_CEILING ? DESIRED_WORKERS : MAX_WORKERS_CEILING ))
OUTPUT_JSONL="${OUTPUT_DIR}/${RUN_LABEL}.jsonl"
echo "[rollout] ${ROLLOUT_N} tasks × ${NUM_REPEATS} repeats, max_workers=${MAX_WORKERS}"
echo "          → $OUTPUT_JSONL"

START_EPOCH=$(date +%s)
if "$GYM_NG_COLLECT" \
     +agent_name=harbor_agent \
     +input_jsonl_fpath="$RUN_INPUT_JSONL" \
     +output_jsonl_fpath="$OUTPUT_JSONL" \
     +num_repeats="$NUM_REPEATS" \
     +max_workers="$MAX_WORKERS" \
     "+responses_create_params={max_output_tokens: $MAX_OUTPUT_TOKENS, temperature: $TEMPERATURE}"; then
  STATUS="ok"
else
  STATUS="ng_collect_rollouts_failed"
fi
ELAPSED=$(( $(date +%s) - START_EPOCH ))

# harbor_agent's server venv may not expose /aggregate_metrics, so
# ng_collect_rollouts can exit non-zero AFTER writing every rollout row. If the
# output already has all expected rows, treat the run as successful.
EXPECTED_ROWS=$(( ROLLOUT_N * NUM_REPEATS ))
ACTUAL_ROWS=$(grep -c . "$OUTPUT_JSONL" 2>/dev/null || echo 0)
if [[ "$STATUS" != "ok" && "$ACTUAL_ROWS" -ge "$EXPECTED_ROWS" ]]; then
  echo "[note] ng_collect_rollouts exited non-zero but all $ACTUAL_ROWS/$EXPECTED_ROWS rows were written"
  echo "       (likely the harmless /aggregate_metrics 404); treating run as ok."
  STATUS="ok_no_aggregate_metrics"
fi

# ============================================================================
# Summary.
# ============================================================================
echo "[done] $STATUS in ${ELAPSED}s"
"$GYM_PY" - <<PY
import json
from collections import Counter
counts, statuses, excs = Counter(), Counter(), Counter()
n = 0
try:
    with open("$OUTPUT_JSONL") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            row = json.loads(line)
            counts[row.get("reward")] += 1
            if "status" in row: statuses[row["status"]] += 1
            if row.get("exception_type"): excs[row["exception_type"]] += 1
except FileNotFoundError:
    pass
print(f"  rows={n}")
for r, c in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0])):
    print(f"    reward={r}: {c}")
if statuses: print(f"  statuses: {dict(statuses)}")
if excs:     print(f"  exceptions: {dict(excs)}")
PY

echo
echo "Output JSONL: $OUTPUT_JSONL"
echo "Harbor log:   $HARBOR_LOG"
[[ "$STATUS" == ok* ]] || { tail -50 "$HARBOR_LOG" >&2; exit 1; }
