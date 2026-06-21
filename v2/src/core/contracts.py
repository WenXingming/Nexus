"""Core data contracts for Nexus v2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """A single chat message."""

    role: str
    content: str


@dataclass(frozen=True)
class AgentRequest:
    """Input for one agent run."""

    input: str
    session_id: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """Output from one agent run."""

    output: str
    session_id: str


@dataclass(frozen=True)
class AgentStreamChunk:
    """One text chunk from a streaming agent run."""

    text: str
    session_id: str
