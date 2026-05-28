# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mini SWE-agent compatibility fixes for Fireworks/LiteLLM rollouts."""

import os
from typing import Any

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.mini_swe_agent import MiniSweAgent


class MiniSweAgentFireworksCompat(MiniSweAgent):
    """Run mini-swe-agent against Fireworks via litellm.

    Two fixes layered on top of stock harbor:

    1. The local harbor fork's install-mini-swe-agent.sh.j2 already patches
       minisweagent's litellm_model.py to strip the ``extra`` key from outgoing
       messages, so no additional sandbox-side patching is done here.
    2. The NeMo Gym side passes raw Fireworks model names like
       ``accounts/fireworks/models/kimi-k2p6``. litellm needs the
       ``fireworks_ai/`` provider prefix to route correctly, so we prepend it
       at construction time unless it's already present.
    """

    def __init__(self, *args: Any, model_name: str | None = None, **kwargs: Any) -> None:
        if model_name and not model_name.startswith("fireworks_ai/"):
            # Stock mini-swe-agent expects model_name as "<provider>/<model>"
            # for litellm provider routing. Our policy_model server exposes
            # the raw Fireworks model name, so prepend the litellm provider.
            model_name = f"fireworks_ai/{model_name}"
        super().__init__(*args, model_name=model_name, **kwargs)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        commands = super().create_run_agent_commands(instruction)
        patched_commands: list[ExecInput] = []
        for command in commands:
            env = dict(command.env or {})
            api_key = (
                os.environ.get("FIREWORKS_AI_API_KEY")
                or os.environ.get("FIREWORKS_API_KEY")
                or os.environ.get("MSWEA_API_KEY")
                or os.environ.get("LLM_API_KEY")
            )
            if api_key:
                env.setdefault("FIREWORKS_AI_API_KEY", api_key)
                env.setdefault("FIREWORKS_API_KEY", api_key)
                env.setdefault("MSWEA_API_KEY", api_key)
            patched_commands.append(command.model_copy(update={"env": env}))
        return patched_commands
