"""Model configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for selecting the model provider."""

    provider: str = "fake"
    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            provider=os.environ.get("NEXUS_MODEL_PROVIDER", "fake"),
            api_key=os.environ.get("OPENAI_API_KEY") or None,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )
