# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenHands compatibility for local OpenAI-compatible NeMo Gym model servers."""

from pathlib import Path

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.openhands import OpenHands


_INSTALL_TEMPLATE_PATH = (
    Path(__file__).parent / "install-openhands-idempotent.sh.j2"
)


class OpenHandsLiteLLMCompat(OpenHands):
    """Route OpenHands through LiteLLM's OpenAI-compatible provider path.

    The Harbor policy model server exposes an OpenAI-compatible endpoint at
    ``LLM_BASE_URL`` and forwards the original Fireworks model name internally.
    LiteLLM still needs the provider prefix on ``LLM_MODEL`` to select its
    OpenAI-compatible client, so this wrapper only changes the environment seen
    by OpenHands. The Harbor job and policy_model config keep the original model
    name.

    Also swaps the install template for an idempotent variant that uses
    ``uv venv --clear`` — the OpenSWE base images
    (docker.io/rvk7895/openswe-python-*) ship a pre-baked /opt/openhands-venv
    and the stock script aborts on "A virtual environment already exists".
    """

    @property
    def _install_agent_template_path(self) -> Path:
        return _INSTALL_TEMPLATE_PATH

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        commands = super().create_run_agent_commands(instruction)
        if not self.model_name:
            return commands

        if self._api_base and "fireworks.ai" in self._api_base:
            litellm_model = (
                self.model_name
                if self.model_name.startswith("fireworks_ai/")
                else f"fireworks_ai/{self.model_name}"
            )
        else:
            litellm_model = (
                self.model_name
                if self.model_name.startswith("openai/")
                else f"openai/{self.model_name}"
            )

        patched_commands: list[ExecInput] = []
        for command in commands:
            env = dict(command.env or {})
            env["LLM_MODEL"] = litellm_model
            if litellm_model.startswith("fireworks_ai/") and "LLM_API_KEY" in env:
                env.setdefault("FIREWORKS_AI_API_KEY", env["LLM_API_KEY"])
            patched_commands.append(command.model_copy(update={"env": env}))

        return patched_commands
