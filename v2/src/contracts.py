"""Core data contracts for Nexus v2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """A single chat message."""

    role: str
    content: str
