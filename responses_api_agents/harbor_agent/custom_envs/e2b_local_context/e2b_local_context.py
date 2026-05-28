# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2B environment override tailored to the self-hosted e2b.sarvam.ai backend.

Three stock-Harbor problems this fixes:

1. ``file_context_path`` not set. Stock ``E2BEnvironment._create_template`` calls
   ``Template().from_dockerfile(...)`` without ``file_context_path``. The e2b SDK
   falls back to ``get_caller_directory()`` which resolves to
   ``site-packages/harbor/environments/`` — so ``COPY repo /testbed`` tries to read
   ``site-packages/harbor/environments/repo`` and fails with
   ``ValueError: No files found``.

2. Stale alias / no ``default`` tag. On the self-hosted e2b backend, a failed
   build can leave an alias behind that ``AsyncTemplate.alias_exists()`` reports as
   present, but sandbox creation then fails with
   ``404: tag 'default' does not exist for template '<alias>'``.

3. Concurrent builds of the same template get cancelled API-side. e2b's
   ``CheckAndCancelConcurrentBuilds`` (packages/api/internal/handlers/
   deprecated_template_start_build.go) actively cancels any in-progress build
   for the same template when a new one is registered. Stock harbor's
   force-build-every-call pattern means 8 concurrent rollouts of the same task
   produce 7 ``BuildException: build was cancelled`` errors, regardless of
   cluster capacity.

   Fix: build the template at most once per process lifetime (track in
   ``_built_templates``), serialize first-build per ``_template_name`` with
   ``_build_locks``, and only call ``_create_sandbox`` on subsequent rollouts.
   Sandbox creation is API-side safe to fan out.

   If ``_create_sandbox`` hits the stale-alias 404 (problem 2), evict the
   template from ``_built_templates`` and force one rebuild + retry.

   Note: in-process coordination only covers a single Ray worker process. For
   true cross-process safety, pre-warm templates sequentially before firing
   high-concurrency rollouts — see scripts/prewarm_openswe_oss_filtered_20.py.
"""

import asyncio
from collections import defaultdict

from e2b import AsyncTemplate, Template
from tenacity import retry, stop_after_attempt, wait_exponential

from harbor.environments.e2b import E2BEnvironment


class E2BLocalContextEnvironment(E2BEnvironment):
    _build_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    _built_templates: set[str] = set()

    async def start(self, force_build: bool):
        # Skip the build on the happy path: assume the template was pre-warmed
        # (scripts/prewarm_openswe_oss_filtered_20.py) or already built by an
        # earlier call. Sandbox creation is the only safe-to-fan-out path on
        # e2b.sarvam.ai. Only rebuild when explicitly forced or when sandbox
        # creation surfaces the stale-alias 404.
        if force_build:
            await self._ensure_template_built(force_build=True)

        try:
            await self._create_sandbox()
        except Exception as exc:
            if not self._is_stale_alias_error(exc):
                raise
            self._built_templates.discard(self._template_name)
            await self._ensure_template_built(force_build=True)
            await self._create_sandbox()

        if not self._sandbox:
            raise RuntimeError(
                "Sandbox not found but was just created. This should never happen."
            )

        from harbor.models.trial.paths import EnvironmentPaths

        await self._sandbox.files.make_dir(str(EnvironmentPaths.agent_dir))
        await self._sandbox.files.make_dir(str(EnvironmentPaths.verifier_dir))

    async def _ensure_template_built(self, *, force_build: bool) -> None:
        name = self._template_name
        async with self._build_locks[name]:
            if not force_build and name in self._built_templates:
                return
            await self._build_template()
            self._built_templates.add(name)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _build_template(self) -> None:
        if self.task_env_config.docker_image:
            template = Template(file_context_path=self.environment_dir).from_image(
                image=self.task_env_config.docker_image,
            )
        else:
            template = Template(
                file_context_path=self.environment_dir
            ).from_dockerfile(
                dockerfile_content_or_path=str(self._environment_definition_path),
            )

        await AsyncTemplate.build(
            template=template,
            alias=self._template_name,
            cpu_count=self.task_env_config.cpus,
            memory_mb=self.task_env_config.memory_mb,
        )

    @staticmethod
    def _is_stale_alias_error(exc: BaseException) -> bool:
        msg = str(exc)
        return "tag 'default' does not exist" in msg or (
            "404" in msg and "template" in msg.lower()
        )
