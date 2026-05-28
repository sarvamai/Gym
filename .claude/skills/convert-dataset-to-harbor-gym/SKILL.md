---
name: convert-dataset-to-harbor-gym
description: Use when converting SWE-bench-style, software-engineering, or custom agent datasets for NeMo Gym, Harbor, Claude Code harness, Modal, task-wise sandbox images, or harbor_agent routing.
---

# Convert Dataset To Harbor + NeMo Gym

## Core Principle

Create two artifacts and keep their responsibilities separate:

1. **NeMo Gym routing JSONL**: tells Gym which agent server to call and which Harbor task to run.
2. **Harbor task dataset**: contains task definitions, verification metadata, setup files, and task-wise sandbox image information.

Do not skip any datapoint. If a row cannot be converted because required fields are missing, ambiguous, duplicated, invalid, or unsupported, stop and ask the user what to do.

## Required Context To Gather

Before converting, identify:

- Source dataset path or repo, split, and format.
- Desired dataset alias, for example `swe_verified` or `my_dataset`.
- Target Harbor agent, for example `claude-code`, `terminus-2`, or custom import path.
- Target Harbor environment, for example `modal`, `daytona`, `docker`, or custom import path.
- Whether task-wise images are already known, need to be generated, or should be derived from `instance_id`.
- Whether results should be saved or only printed. If not explicit, ask.

## NeMo Gym Routing JSONL

For each source datapoint, create one routing row:

```json
{"instance_id":"<dataset_alias>::<task_name>","responses_create_params":{"input":[]},"agent_ref":{"name":"harbor_agent"}}
```

Rules:

- `instance_id` must be `<dataset_alias>::<task_name>`.
- `dataset_alias` must match a key in `harbor_datasets`.
- `task_name` must match the Harbor task name.
- `agent_ref.name` must be `harbor_agent` for the Harbor integration.
- Keep full task content out of the routing row unless the local code is explicitly extended to build Harbor tasks from row metadata.

## Harbor Dataset Contract

Harbor receives a dataset config from `HarborAgent._build_job_config()`:

- Local mode: `LocalDatasetConfig(path=<local_dataset_path>, task_names=[task_name])`
- Registry mode: `RegistryDatasetConfig(name=<dataset_name>, version=<dataset_version>, task_names=[task_name])`

Therefore each routing row must point to a task that Harbor can resolve through the configured alias.

Task-wise images belong in the Harbor task definition, not in the NeMo Gym routing row. Use the task's `[environment]` section or the field expected by Harbor for that dataset/environment, for example:

```toml
[environment]
docker_image = "swebench/sweb.eval.x86_64.<instance_id>:latest"
```

## Harbor Agent Config Pattern

For built-in Claude Code on built-in Modal:

```yaml
harbor_agent_name: "claude-code"
harbor_agent_import_path: null
harbor_environment_type: "modal"
harbor_environment_import_path: null
harbor_environment_kwargs: {}
```

Important:

- `harbor_agent_import_path` overrides `harbor_agent_name`.
- `harbor_environment_import_path` overrides `harbor_environment_type`.
- When switching from Singularity to Modal, remove Singularity kwargs such as `singularity_image_cache_dir`.

## SWE-bench Verified Mapping

Canonical public source: `SWE-bench/SWE-bench_Verified` on Hugging Face.

Useful fields:

- `instance_id`: task name, for example `astropy__astropy-12907`.
- `repo`, `base_commit`, `problem_statement`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `version`, `environment_setup_commit`, `difficulty`.

Routing row example:

```json
{"instance_id":"swe_verified::astropy__astropy-12907","responses_create_params":{"input":[]},"agent_ref":{"name":"harbor_agent"}}
```

## Validation Checklist

Before finishing:

- Every source datapoint has either a converted output or a user decision.
- No datapoint was silently skipped.
- Routing JSONL line count equals the intended source row count.
- Every routing `instance_id` has exactly one `::`.
- Every dataset alias exists in the Harbor config.
- Every task name exists in the local Harbor dataset or intended registry dataset.
- Task-wise image information is present where Harbor expects it.
- Modal config has `harbor_environment_type: "modal"` and `harbor_environment_import_path: null`.
- The output location is explicit and was approved or requested by the user.

## Additional Reference

For detailed templates and examples, read `reference.md` in this skill directory.
