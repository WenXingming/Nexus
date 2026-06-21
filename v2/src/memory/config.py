"""Memory configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for selecting the session store."""

    store: str = "file"
    root: Path = Path(".nexus-v2/sessions")

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            store=os.environ.get("NEXUS_MEMORY_STORE", "file"),
            root=Path(os.environ.get("NEXUS_SESSION_ROOT", ".nexus-v2/sessions")),
        )
