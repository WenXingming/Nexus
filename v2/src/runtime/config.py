"""Runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str | None = None
