"""Runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str | None = None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            system_prompt=os.environ.get("NEXUS_SYSTEM_PROMPT") or None,
        )
