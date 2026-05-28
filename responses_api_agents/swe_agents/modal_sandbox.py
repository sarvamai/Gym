# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Experimental Modal-backed sandbox for SWE-style rollouts.
#
# This module is intentionally minimal and NOT production ready. It exists so we
# can benchmark Modal sandbox lifecycle timings (create/upload/exec/download/
# terminate/snapshot) for SWE-bench style rollouts (Fireworks model + Modal
# sandbox).
#
# `modal` is imported directly. If it is missing, this import will fail and the
# user must install modal (`pip install modal`) and configure credentials
# (`modal token new`). We do NOT wrap the import in try/except.
#
# TODO(production): expose backend selection (apptainer vs modal vs local) via
# SWEBenchWrapperConfig and select at runtime instead of having this be a
# separate experimental module.
# TODO(production): wire Modal resource sizing (cpu, memory, gpu, timeout,
# idle_timeout) via SWE agent YAML config.
# TODO(production): wire snapshot mode (none, per_instance, shared_registry) via
# YAML config + a persistent snapshot registry keyed on container/image.
# TODO(production): add retry policy, cleanup policy, and stronger storage
# management for orphaned sandboxes/snapshots.
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import modal

logger = logging.getLogger(__name__)


# Stages we time inside ModalSandbox. Surfaced by the rollout layer so the
# benchmark aggregator can compute distributions per stage.
SandboxStage = Literal[
    "create",
    "upload",
    "exec",
    "download",
    "terminate",
    "snapshot_create",
    "snapshot_restore",
]


class ExecResult:
    """Result of a single sandbox exec call."""

    def __init__(
        self,
        returncode: Optional[int],
        stdout: str,
        stderr: str,
        elapsed: float,
        timed_out: bool,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed
        self.timed_out = timed_out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed": self.elapsed,
            "timed_out": self.timed_out,
        }


class ModalSandbox:
    """Thin async wrapper around `modal.Sandbox` that records per-stage timings.

    Designed for experimental SWE rollout benchmarking. Each method records the
    wall-clock duration in `self.timings` keyed by stage.

    Lifecycle:
        sb = ModalSandbox(app_name="ng-swe-bench", image=image)
        await sb.create()
        await sb.upload_text("/root/patch.diff", patch_text)
        result = await sb.exec("bash", "-lc", "cd /testbed && pytest -x")
        local = await sb.download_file("/root/report.json", local_path)
        snap = await sb.snapshot_filesystem()
        await sb.terminate()

    The `timings` dict accumulates a list of floats per stage. Multiple exec
    calls add multiple entries. The rollout layer can sum or pick max as needed.
    """

    def __init__(
        self,
        *,
        app_name: str,
        image: Optional[modal.Image] = None,
        timeout: int = 60 * 30,
        cpu: Optional[float] = None,
        memory: Optional[int] = None,
        gpu: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        workdir: Optional[str] = None,
        from_snapshot_id: Optional[str] = None,
        label: str = "sandbox",
    ) -> None:
        # TODO(production): plumb cpu/memory/gpu/timeout via SWEBenchWrapperConfig.
        # TODO(production): support secrets/volumes for shared dependency caches.
        self.app_name = app_name
        self.image = image
        self.timeout = timeout
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu
        self.env = env or {}
        self.workdir = workdir
        self.from_snapshot_id = from_snapshot_id
        self.label = label

        self._sandbox: Optional[modal.Sandbox] = None
        self._app: Optional[modal.App] = None
        self.sandbox_id: Optional[str] = None
        self.snapshot_id: Optional[str] = None

        # timings[stage] -> list[float]. Use list (not single value) because
        # exec/upload/download may be called many times per rollout.
        self.timings: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ helpers
    def _record(self, stage: str, elapsed: float) -> None:
        self.timings.setdefault(stage, []).append(elapsed)

    def timing_summary(self) -> Dict[str, Any]:
        """Aggregate timings: per-stage total, count, and max."""
        out: Dict[str, Any] = {}
        for stage, vals in self.timings.items():
            out[f"{stage}_total"] = sum(vals)
            out[f"{stage}_count"] = len(vals)
            out[f"{stage}_max"] = max(vals) if vals else None
        return out

    # ------------------------------------------------------------------ create
    async def create(self) -> None:
        """Create (or restore from snapshot) the underlying Modal sandbox.

        Records `create` timing even on failure (we want time-to-failure for
        benchmarking).
        """
        t0 = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            self._app = await loop.run_in_executor(
                None, lambda: modal.App.lookup(self.app_name, create_if_missing=True)
            )

            if self.from_snapshot_id is not None:
                # TODO(production): hide snapshot Image lookup behind a registry
                # keyed by (dataset_name, instance_id) instead of opaque IDs.
                t_snap = time.perf_counter()
                image = await loop.run_in_executor(
                    None, lambda: modal.Image.from_id(self.from_snapshot_id)
                )
                self._record("snapshot_restore", time.perf_counter() - t_snap)
            else:
                image = self.image

            create_kwargs: Dict[str, Any] = {
                "app": self._app,
                "timeout": self.timeout,
            }
            if image is not None:
                create_kwargs["image"] = image
            if self.cpu is not None:
                create_kwargs["cpu"] = self.cpu
            if self.memory is not None:
                create_kwargs["memory"] = self.memory
            if self.gpu is not None:
                create_kwargs["gpu"] = self.gpu
            if self.workdir is not None:
                create_kwargs["workdir"] = self.workdir
            if self.env:
                create_kwargs["env"] = self.env

            # Modal exposes .aio for async creation.
            self._sandbox = await modal.Sandbox.create.aio(**create_kwargs)
            self.sandbox_id = getattr(self._sandbox, "object_id", None)
        finally:
            self._record("create", time.perf_counter() - t0)

    # ------------------------------------------------------------------ upload
    async def upload_text(self, remote_path: str, text: str) -> None:
        assert self._sandbox is not None, "create() must be called first"
        t0 = time.perf_counter()
        # filesystem.write_text supports .aio in current modal API.
        await self._sandbox.filesystem.write_text.aio(text, remote_path)
        self._record("upload", time.perf_counter() - t0)

    async def upload_file(self, local_path: Path | str, remote_path: str) -> None:
        assert self._sandbox is not None
        t0 = time.perf_counter()
        await self._sandbox.filesystem.copy_from_local.aio(str(local_path), remote_path)
        self._record("upload", time.perf_counter() - t0)

    # ------------------------------------------------------------------ exec
    async def exec(
        self,
        *cmd: str,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecResult:
        """Run a command in the sandbox; capture stdout/stderr as text."""
        assert self._sandbox is not None
        t0 = time.perf_counter()
        timed_out = False
        kwargs: Dict[str, Any] = {"text": True}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if workdir is not None:
            kwargs["workdir"] = workdir
        if env:
            kwargs["env"] = env

        try:
            process = await self._sandbox.exec.aio(*cmd, **kwargs)
            # process.stdout/stderr are StreamReaders; .read.aio returns full text.
            stdout_text = await process.stdout.read.aio()
            stderr_text = await process.stderr.read.aio()
            # wait for exit code.
            returncode = await process.wait.aio()
        except modal.exception.SandboxTimeoutError:  # type: ignore[attr-defined]
            timed_out = True
            returncode = None
            stdout_text = ""
            stderr_text = "modal SandboxTimeoutError"
        except Exception as e:
            # Surface — let caller catch and tag stage. Still record timing.
            self._record("exec", time.perf_counter() - t0)
            raise e

        elapsed = time.perf_counter() - t0
        self._record("exec", elapsed)
        return ExecResult(
            returncode=returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            elapsed=elapsed,
            timed_out=timed_out,
        )

    # ------------------------------------------------------------------ download
    async def download_file(self, remote_path: str, local_path: Path | str) -> Path:
        assert self._sandbox is not None
        t0 = time.perf_counter()
        await self._sandbox.filesystem.copy_to_local.aio(remote_path, str(local_path))
        self._record("download", time.perf_counter() - t0)
        return Path(local_path)

    async def read_text(self, remote_path: str) -> str:
        assert self._sandbox is not None
        t0 = time.perf_counter()
        text = await self._sandbox.filesystem.read_text.aio(remote_path)
        self._record("download", time.perf_counter() - t0)
        return text

    # ------------------------------------------------------------------ snapshot
    async def snapshot_filesystem(self) -> Optional[str]:
        """Capture current FS as a Modal Image. Returns the image id (if available).

        TODO(production): persist the snapshot id in a registry keyed by
        (dataset_name, instance_id) so subsequent rollouts can restore instead
        of re-bootstrapping.
        """
        assert self._sandbox is not None
        t0 = time.perf_counter()
        image = await self._sandbox.snapshot_filesystem.aio()
        self._record("snapshot_create", time.perf_counter() - t0)
        self.snapshot_id = getattr(image, "object_id", None)
        return self.snapshot_id

    # ------------------------------------------------------------------ terminate
    async def terminate(self) -> None:
        if self._sandbox is None:
            return
        t0 = time.perf_counter()
        try:
            await self._sandbox.terminate.aio()
        except Exception as e:
            logger.warning("modal sandbox terminate failed (%s): %s", self.label, e)
        finally:
            self._record("terminate", time.perf_counter() - t0)
            self._sandbox = None

    # ------------------------------------------------------------------ context
    async def __aenter__(self) -> "ModalSandbox":
        await self.create()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.terminate()


def build_image_from_docker(
    image_ref: str, add_python: Optional[str] = None
) -> modal.Image:
    """Build a Modal Image from a docker registry reference.

    `add_python` is only passed when the upstream image does NOT already ship
    a python interpreter (Modal requires python in the image). Passing
    `add_python` for an image like `python:3.11-slim` causes a build failure.

    TODO(production): support private registries via modal.Secret and pinning
    by digest. For experimental use we accept a tag like
    `swebench/sweb.eval.x86_64.<instance_id>:latest`.
    """
    ref_l = image_ref.lower()
    if add_python is None and not (
        ref_l.startswith("python:") or "/python:" in ref_l or "python3" in ref_l
    ):
        # Heuristic: most SWE-bench eval images derive from ubuntu without a
        # ready python; add one so Modal can manage the sandbox.
        add_python = "3.11"
    kwargs: Dict[str, Any] = {}
    if add_python:
        kwargs["add_python"] = add_python
    return modal.Image.from_registry(image_ref, **kwargs)


def split_stage_timings(timings: Dict[str, List[float]]) -> Dict[str, float]:
    """Convenience: flatten timings dict to a `<stage>_total_time` dict."""
    return {f"{stage}_total_time": sum(vals) for stage, vals in timings.items()}


# What stage labels are recognized. Useful for benchmark schema validation.
KNOWN_STAGES: Tuple[str, ...] = (
    "create",
    "upload",
    "exec",
    "download",
    "terminate",
    "snapshot_create",
    "snapshot_restore",
)


def empty_stage_dict() -> Dict[str, Optional[float]]:
    """All stage timing keys initialised to None — used when a rollout fails
    early so the row still has every field."""
    out: Dict[str, Optional[float]] = {}
    for stage in KNOWN_STAGES:
        out[f"{stage}_total_time"] = None
        out[f"{stage}_max_time"] = None
        out[f"{stage}_count"] = None
    return out


def aggregate_iterable_floats(values: Iterable[float]) -> Dict[str, float]:
    """Compute min/p50/p90/p95/max/mean for the given numeric iterable."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {}
    n = len(vals)

    def _pct(p: float) -> float:
        if n == 1:
            return vals[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return vals[f]
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    return {
        "min": vals[0],
        "p50": _pct(0.5),
        "p90": _pct(0.9),
        "p95": _pct(0.95),
        "max": vals[-1],
        "mean": sum(vals) / n,
        "count": n,
    }
