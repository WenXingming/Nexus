"""Deterministic fake model for early runtime tests."""

from __future__ import annotations

from src.core.contracts import Message


class FakeModel:
    """A tiny model double that echoes the latest user message."""

    def complete(self, messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return f"Echo: {message.content}"
        return "Echo:"
