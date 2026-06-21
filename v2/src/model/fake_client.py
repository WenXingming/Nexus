"""Deterministic fake model client for early runtime tests."""

from __future__ import annotations

from collections.abc import Iterable

from src.core.contracts import Message


class FakeClient:
    """A tiny model client double that echoes the latest user message."""

    def complete(self, messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return f"Echo: {message.content}"
        return "Echo:"

    def stream(self, messages: list[Message]) -> Iterable[str]:
        for message in reversed(messages):
            if message.role == "user":
                return ["Echo: ", message.content]
        return ["Echo:"]
