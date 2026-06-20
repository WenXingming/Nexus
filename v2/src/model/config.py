"""Model configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for selecting the model provider."""

    provider: str = "fake"

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            provider=os.environ.get("NEXUS_MODEL_PROVIDER", "fake"),
        )
